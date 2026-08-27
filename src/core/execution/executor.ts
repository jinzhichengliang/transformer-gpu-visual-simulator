/**
 * Execution Executor — OperatorGraph → TVIR 事件流（Sprint 5 D2 / Sprint 9 / Sprint 10）。
 *
 * 任务书要求：
 *   - Prefill：完整前向（Embedding → Layers → LM Head），phase=PREFILL；
 *   - Decode：独立 Planner，禁止简单退化为 "prefill with seq=1"；
 *     必须体现 decode step / previous state / cache access / new token 语义；
 *   - prefill_decode：先完整 prefill，再逐步 decode（Decode Step 1..N 可见）。
 *
 * 架构铁律：
 *   - 输出只能是合法 TVIRTrace（12 种事件类型，零新增）；
 *   - Decode 的 KV cache 语义通过 metadata.model.decode / metadata.kvCache 表达；
 *   - 本模块不 import React、不输出 UI 数据。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder, type EventBuilder } from '../simulation/eventBuilder';
import { emitGemmEvents } from '../simulation/gemmPrimitives';
import { emitElementwiseEvents } from '../simulation/elementwisePrimitives';
import type { ModelProfile } from '../modelprofile/types';
import { validateModelProfile } from '../modelprofile/validation';
import type { InferenceTask } from './task';
import { validateInferenceTask, hasPrefill, hasDecode } from './task';
import { buildOperatorGraph, validateOperatorGraph, type OperatorNode } from './planner';
import { getOperatorTemplate } from './templates';

/** 执行计划生成结果（失败安全：不抛异常，返回明确错误） */
export type ExecutionPlanResult =
  | { ok: true; trace: TVIRTrace }
  | { ok: false; error: string };

/** 事件携带的模型上下文（写入 metadata.model，供 Model Overview / Semantic Zoom 投影） */
interface ModelContextMeta {
  modelId: string;
  modelDisplayName: string;
  layerIndex: number;
  layerType: string;
  operatorType: string;
  phase: string;
  decodeStep?: number;
  decodeTotal?: number;
}

/** 给 builder 中 [startCounter, 末尾] 区间的事件补挂模型上下文 */
function tagModelContext(
  builder: EventBuilder,
  startCounter: number,
  ctx: ModelContextMeta,
): void {
  for (let i = startCounter; i < builder.events.length; i++) {
    const event = builder.events[i];
    if (!event) continue;
    event.metadata = { ...(event.metadata ?? {}), model: { ...(event.metadata?.model as object ?? {}), ...ctx } };
  }
}

function hiddenSizeOf(profile: ModelProfile): number {
  return profile.architecture.hiddenSize?.value ?? 128;
}

/** 查找某层号对应的 attention 层（取 profile 中的 attention 细节） */
function attentionOf(profile: ModelProfile) {
  return profile.layers.find((l) => l.type === 'attention')?.attention;
}

/**
 * 生成模型执行计划（TVIRTrace）。
 */
export function planExecution(profile: ModelProfile, task: InferenceTask): ExecutionPlanResult {
  // 1. 输入校验（失败安全）
  const profileCheck = validateModelProfile(profile);
  if (!profileCheck.valid) {
    return { ok: false, error: `ModelProfile 校验失败：${profileCheck.errors.join('；')}` };
  }
  const taskCheck = validateInferenceTask(task);
  if (!taskCheck.valid) {
    return { ok: false, error: `InferenceTask 校验失败：${taskCheck.errors.join('；')}` };
  }

  // 2. OperatorGraph（Prefill 与 Decode 共用同一结构图，展开语义不同）
  const graph = buildOperatorGraph(profile, task);
  const graphCheck = validateOperatorGraph(graph);
  if (!graphCheck.valid) {
    return { ok: false, error: `OperatorGraph 校验失败：${graphCheck.errors.join('；')}` };
  }

  const builder = createEventBuilder();
  const { numSM, warpsPerBlock, tileSize } = task.hardwareProfile;
  const prefillRows = task.promptTokens * task.batchSize;

  // ───────────── 教学抽样与层折叠（Task H3，大型模型性能守护） ─────────────
  // 完整大模型可能产生百万级事件：大型模型（>6 层）只详细展开前 2 层，
  // 其余重复结构的层折叠为"重复结构 ×N"语义步（collapsed execution）。
  const maxLayerIndex = graph.nodes.reduce((m, n) => Math.max(m, n.layerIndex), -1);
  const isLargeModel = maxLayerIndex > 5;
  const sample = isLargeModel ? { blocks: 2, kIterations: 2 } : undefined;
  const detailedLayerLimit = isLargeModel ? 1 : Number.POSITIVE_INFINITY;

  /** 折叠状态：把连续被折叠的层合并为一条"重复结构 ×N"事件（按层数统计） */
  let pendingCollapse: { from: number; to: number; layers: Set<number> } | null = null;

  const flushCollapse = (): void => {
    if (!pendingCollapse) return;
    const { from, to } = pendingCollapse;
    const count = pendingCollapse.layers.size;
    pendingCollapse = null;
    builder.push({
      type: 'SYNC',
      title: `Layer ${from}～${to}：重复结构 ×${count}（与 Layer 0 结构相同）`,
      what: `Layer ${from} 到 Layer ${to} 共 ${count} 层，与已详细展示的 Layer 0 结构完全相同（Attention + ${graph.nodes.some((n) => n.operatorType === 'moe') ? 'MoE' : 'FFN'} + 残差），此处折叠为一个语义步，不再逐一展开。`,
      why: '真实大模型有几十上百层，结构逐层重复。教学仿真把重复结构折叠：展开的 Layer 0 就是它们的精确写照。逐层全量展开会产生百万级事件，既超出教学需要，也无法流畅播放。（点击 Timeline 上该步可定位，Expand 展开是后续交互增强。）',
      operator: 'Collapsed Layers',
      metadata: { collapsed: { fromLayer: from, toLayer: to, count } },
    });
    tagModelContext(builder, builder.events.length - 1, {
      modelId: profile.id,
      modelDisplayName: profile.displayName,
      layerIndex: from,
      layerType: 'collapsed',
      operatorType: 'collapsed',
      phase: 'prefill',
    });
  };

  const emitNode = (node: OperatorNode, rows: number, phaseLabel: string, phaseName: string): void => {
    // 层折叠判定：超出详细展示范围的层（非结构层节点）进入折叠
    if (node.layerIndex > detailedLayerLimit && node.operatorType !== 'embedding' && node.operatorType !== 'lm_head') {
      if (pendingCollapse) {
        pendingCollapse.to = node.layerIndex;
        pendingCollapse.layers.add(node.layerIndex);
      } else {
        pendingCollapse = { from: node.layerIndex, to: node.layerIndex, layers: new Set([node.layerIndex]) };
      }
      return;
    }
    flushCollapse();
    const template = getOperatorTemplate(node.operatorType);
    if (!template) {
      // 未注册模板：生成一条占位事件（不中断计划，明确暴露缺口）
      builder.push({
        type: 'SYNC',
        title: `${phaseLabel}${node.operatorType}（未注册展开模板）`,
        what: `算子 ${node.operatorType} 暂无展开模板。`,
        why: '模板注册表缺失该算子，请补充注册（OperatorTemplateRegistry）。',
        operator: node.operatorType,
      });
      return;
    }
    const start = builder.counter;
    template({
      builder,
      profile,
      task,
      node,
      rows,
      phaseLabel,
      tileSize,
      numSM,
      warpsPerBlock,
      ...(sample ? { sample } : {}),
    });
    tagModelContext(builder, start, {
      modelId: profile.id,
      modelDisplayName: profile.displayName,
      layerIndex: node.layerIndex,
      layerType: node.layerType,
      operatorType: node.operatorType,
      phase: phaseName,
    });
  };

  // ───────────── 3. Prefill 阶段 ─────────────
  if (hasPrefill(task)) {
    builder.push({
      type: 'GEMM_START',
      title: `Prefill 开始：${profile.displayName}`,
      what: `开始 Prefill：把 ${task.promptTokens} 个输入 token（batch=${task.batchSize}，共 ${prefillRows} 行）一次性送入 ${profile.displayName} 做完整前向计算。`,
      why: 'Prefill 阶段并行处理整个输入序列：所有 token 同时参与计算，是计算密集型（compute-bound）阶段——与逐 token 生成的 Decode 形成鲜明对比。',
      operator: 'Prefill',
      metadata: { model: { modelId: profile.id, modelDisplayName: profile.displayName, phase: 'prefill', layerIndex: -1, layerType: 'prefill', operatorType: 'prefill' }, promptTokens: task.promptTokens, batchSize: task.batchSize },
    });

    for (const node of graph.nodes) {
      emitNode(node, prefillRows, '[Prefill] ', 'prefill');
    }
    flushCollapse();

    builder.push({
      type: 'GEMM_END',
      title: 'Prefill 完成：KV Cache 已建立',
      what: `Prefill 完成：${task.promptTokens} 个 token 的 Key/Value 已写入 KV Cache，模型已准备好逐 token 生成。`,
      why: 'Prefill 的副产物是 KV Cache——后续 Decode 每步只需算新 token 的 Q，K/V 复用缓存，避免重复计算整个历史序列。',
      operator: 'Prefill',
      metadata: { model: { modelId: profile.id, modelDisplayName: profile.displayName, phase: 'prefill', layerIndex: -1, layerType: 'prefill', operatorType: 'prefill' }, cacheRows: task.promptTokens },
    });
  }

  // ───────────── 4. Decode 阶段（独立语义，非退化的 prefill） ─────────────
  if (hasDecode(task)) {
    const startCacheLen = hasPrefill(task) ? task.promptTokens : 0;
    const decodeNodesAll = graph.nodes.filter((n) => n.operatorType !== 'embedding' && n.operatorType !== 'lm_head');
    // 层折叠（Task H3）：decode 同样只详细展开 Layer 0~detailedLayerLimit，
    // 其余重复结构的层在每步内折叠为一条"重复结构 ×N"事件。
    const decodeNodesDetailed = decodeNodesAll.filter((n) => n.layerIndex <= detailedLayerLimit);
    const collapsedDecodeLayers = decodeNodesAll.filter((n) => n.layerIndex > detailedLayerLimit);
    let decodeCollapseInfo: { from: number; to: number } | null = null;
    if (collapsedDecodeLayers.length > 0) {
      decodeCollapseInfo = {
        from: collapsedDecodeLayers[0]!.layerIndex,
        to: collapsedDecodeLayers[collapsedDecodeLayers.length - 1]!.layerIndex,
      };
    }

    for (let step = 1; step <= task.outputTokens; step++) {
      const cacheLen = startCacheLen + step - 1;
      const phaseLabel = `[Decode ${step}/${task.outputTokens}] `;
      const decodeMeta = { step, total: task.outputTokens, cacheLen };

      builder.push({
        type: 'GEMM_START',
        title: `Decode Step ${step}：生成第 ${step} 个新 token`,
        what: `Decode Step ${step}：只处理 1 个新 token（batch=${task.batchSize}），读取已有 ${cacheLen} 个 token 的 KV Cache，计算注意力后生成下一个词的分布。`,
        why: 'Decode 是自回归生成：每次只前进一步，新 token 能看到之前所有 token（通过 KV Cache）。每步计算量小但要反复执行，是访存密集型（memory-bound）阶段。',
        operator: 'Decode',
        metadata: { model: { modelId: profile.id, modelDisplayName: profile.displayName, phase: 'decode', decodeStep: step, decodeTotal: task.outputTokens, layerIndex: -1, layerType: 'decode', operatorType: 'decode' }, decode: decodeMeta },
      });

      for (const node of decodeNodesDetailed) {
        const start = builder.counter;
        if (node.operatorType === 'attention') {
          emitDecodeAttention(builder, {
            profile,
            cacheLen,
            step,
            total: task.outputTokens,
            batchSize: task.batchSize,
            tileSize,
            numSM,
            warpsPerBlock,
            phaseLabel,
            ...(sample ? { sample } : {}),
          });
        } else {
          const template = getOperatorTemplate(node.operatorType);
          if (template) {
            template({
              builder,
              profile,
              task,
              node,
              rows: task.batchSize, // decode 每步只有 1 个新 token / 序列
              phaseLabel,
              tileSize,
              numSM,
              warpsPerBlock,
              ...(sample ? { sample } : {}),
            });
          }
        }
        tagModelContext(builder, start, {
          modelId: profile.id,
          modelDisplayName: profile.displayName,
          layerIndex: node.layerIndex,
          layerType: node.layerType,
          operatorType: node.operatorType,
          phase: 'decode',
          decodeStep: step,
          decodeTotal: task.outputTokens,
        });
      }

      // 折叠的重复层（每步一条语义事件，与 Prefill 的折叠语义一致）
      if (decodeCollapseInfo) {
        const { from, to } = decodeCollapseInfo;
        const count = to - from + 1;
        builder.push({
          type: 'SYNC',
          title: `${phaseLabel}Layer ${from}～${to}：重复结构 ×${count}（折叠）`,
          what: `Layer ${from} 到 Layer ${to} 共 ${count} 层与已详细展示的层结构相同，本步折叠为一个语义事件。`,
          why: 'Decode 每步都要流过全部层，逐层全量展开会产生海量事件。教学仿真折叠重复结构：已展开层即是它们的精确写照。',
          operator: 'Collapsed Layers',
          metadata: { collapsed: { fromLayer: from, toLayer: to, count }, decode: decodeMeta },
        });
        tagModelContext(builder, builder.events.length - 1, {
          modelId: profile.id,
          modelDisplayName: profile.displayName,
          layerIndex: from,
          layerType: 'collapsed',
          operatorType: 'collapsed',
          phase: 'decode',
          decodeStep: step,
          decodeTotal: task.outputTokens,
        });
      }

      builder.push({
        type: 'GEMM_END',
        title: `Decode Step ${step} 完成：新 token 加入序列`,
        what: `第 ${step} 个新 token 已生成，其 K/V 写入 KV Cache（缓存长度 ${cacheLen} → ${cacheLen + 1}）。`,
        why: '新 token 的 K/V 追加进缓存，下一步生成时它就能被"看到"——这就是自回归记忆的物理载体。',
        operator: 'Decode',
        metadata: { model: { modelId: profile.id, modelDisplayName: profile.displayName, phase: 'decode', decodeStep: step, decodeTotal: task.outputTokens, layerIndex: -1, layerType: 'decode', operatorType: 'decode' }, decode: { ...decodeMeta, cacheLen: cacheLen + 1 } },
      });
    }

    // LM Head 只在最后一步展示（教学抽样：每步都算但只详细展示最后一步）
    const lmHeadNode = graph.nodes.find((n) => n.operatorType === 'lm_head');
    if (lmHeadNode) {
      emitNode(lmHeadNode, task.batchSize, '[Decode Final] ', 'decode');
    }
  }

  const trace: TVIRTrace = {
    description: buildDescription(profile, task),
    events: builder.events,
    provenance: 'simulation',
  };
  return { ok: true, trace };
}

/** Decode 专用 Attention 展开（独立语义：new token + cache access + previous state） */
function emitDecodeAttention(
  builder: EventBuilder,
  opts: {
    profile: ModelProfile;
    cacheLen: number;
    step: number;
    total: number;
    batchSize: number;
    tileSize: number;
    numSM: number;
    warpsPerBlock: number;
    phaseLabel: string;
    /** 教学抽样（可选）：大型模型下限制 GEMM 详细展开规模 */
    sample?: { blocks: number; kIterations: number };
  },
): void {
  const h = hiddenSizeOf(opts.profile);
  const attn = attentionOf(opts.profile);
  const headDim = attn?.headDim?.value ?? Math.min(64, h);
  const rows = opts.batchSize;
  const fullLen = opts.cacheLen + 1; // 含新 token
  const gemmBase = {
    tileM: opts.tileSize,
    tileN: opts.tileSize,
    tileK: opts.tileSize,
    numSM: opts.numSM,
    warpsPerBlock: opts.warpsPerBlock,
    ...(opts.sample
      ? { sampledBlocks: opts.sample.blocks, sampledKIterations: opts.sample.kIterations }
      : {}),
  };
  const label = (name: string) => `${opts.phaseLabel}${name}`;

  builder.push({
    type: 'GEMM_START',
    title: label(`Attention（Decode）：1 个新 token × ${fullLen} 长度上下文`),
    what: `Decode 注意力：新 token 的 Q 与"缓存中 ${opts.cacheLen} 个旧 K/V + 新 token 自身"计算注意力。`,
    why: '与 Prefill 不同：Q 只有 1 行（新 token），但 K/V 覆盖整个历史——计算量从 O(n²) 降到 O(n)，代价是每步都要读取整个 KV Cache。',
    operator: 'Attention (Decode)',
    metadata: { seqLen: fullLen, dModel: h, headDim, decodeStep: opts.step, cacheLen: opts.cacheLen },
  });

  // Q Projection（只算新 token）
  emitGemmEvents(builder, {
    ...gemmBase,
    M: rows,
    N: headDim,
    K: h,
    left: 'x_new',
    right: 'Wq',
    out: 'Q_new',
    operator: 'Q Projection (Decode)',
    kernel: 'proj_q_kernel',
    label: label('Q Projection'),
    startWhat: `计算新 token 的 Query：Q_new[${rows}×${headDim}] = x_new[${rows}×${h}] × Wq[${h}×${headDim}]。`,
    startWhy: 'Decode 每步只产生 1 个新 token，所以 Q 只有 1 行——这是 Decode 计算量小的根本原因。',
  });

  // KV Cache 读取（cache access：Decode 的标志性访存行为）
  builder.push({
    type: 'MEMORY_LOAD',
    title: label(`读取 KV Cache（${opts.cacheLen} 个历史 token）`),
    what: `从 HBM 读取该层缓存的 K/V：共 ${opts.cacheLen} 个历史 token 的 Key 与 Value（previous state）。`,
    why: 'KV Cache 存放了之前所有 token 的 K/V——Decode 不需要重算历史，但每步都要把它们读进来与新 Q 做注意力。序列越长，这步的带宽压力越大（Decode 是 memory-bound 的根源）。',
    operator: 'KV Cache Read',
    source: 'HBM',
    destination: 'L2',
    tensor: 'KVCache',
    metadata: { kvCache: { cacheLen: opts.cacheLen, decodeStep: opts.step }, seqLen: fullLen },
  });

  // K/V Projection（新 token 的 K/V，写入缓存）
  for (const name of ['K', 'V'] as const) {
    emitGemmEvents(builder, {
      ...gemmBase,
      M: rows,
      N: headDim,
      K: h,
      left: 'x_new',
      right: `W${name.toLowerCase()}`,
      out: `${name}_new`,
      operator: `${name} Projection (Decode)`,
      kernel: `proj_${name.toLowerCase()}_kernel`,
      label: label(`${name} Projection`),
      startWhat: `计算新 token 的 ${name}：${name}_new[${rows}×${headDim}] = x_new[${rows}×${h}] × W${name.toLowerCase()}。`,
    });
    builder.push({
      type: 'MEMORY_STORE',
      title: label(`新 ${name} 追加写入 KV Cache（位置 ${opts.cacheLen}）`),
      what: `把新 token 的 ${name} 向量追加到 KV Cache 第 ${opts.cacheLen} 个位置。`,
      why: '新 token 的 K/V 必须入缓存，后续步骤才能关注到它。KV Cache 因此每步增长 1 行。',
      operator: 'KV Cache Write',
      source: 'REGISTER',
      destination: 'HBM',
      tensor: 'KVCache',
      metadata: { kvCache: { cacheLen: opts.cacheLen + 1, decodeStep: opts.step } },
    });
  }

  // QK^T：新 Q 与全部 K（缓存+新）
  emitGemmEvents(builder, {
    ...gemmBase,
    M: rows,
    N: fullLen,
    K: headDim,
    left: 'Q_new',
    right: 'K_allᵀ',
    out: 'S',
    operator: 'QK MatMul (Decode)',
    kernel: 'qk_matmul_kernel',
    label: label('QK MatMul'),
    startWhat: `计算注意力分数：S[${rows}×${fullLen}] = Q_new[${rows}×${headDim}] × K_allᵀ[${headDim}×${fullLen}]，新 token 对全部 ${fullLen} 个位置打分。`,
  });

  // Softmax（行归约）
  emitElementwiseEvents(builder, {
    rows,
    cols: fullLen,
    numSM: opts.numSM,
    warpsPerBlock: opts.warpsPerBlock,
    operator: 'Softmax (Decode)',
    kernel: 'softmax_kernel',
    label: label('Softmax'),
    tensor: 'S',
    launchWhat: '对分数行做 Scale + Softmax，得到新 token 对每个历史位置的注意力权重。',
    launchWhy: 'Softmax 把原始分数归一化为概率分布：权重之和为 1，大的权重表示"更关注那个位置"。',
    computeWhat: '行内求最大值与指数和（归约），再逐元素归一化。',
    computeWhy: '行归约需要线程协作，属于访存密集 kernel。',
  });

  // AV：权重 × 全部 V
  emitGemmEvents(builder, {
    ...gemmBase,
    M: rows,
    N: headDim,
    K: fullLen,
    left: 'S',
    right: 'V_all',
    out: 'O',
    operator: 'AV MatMul (Decode)',
    kernel: 'av_matmul_kernel',
    label: label('AV MatMul'),
    startWhat: `加权求和：O[${rows}×${headDim}] = S[${rows}×${fullLen}] × V_all[${fullLen}×${headDim}]，聚合所有位置的 Value。`,
  });

  // Output Projection
  emitGemmEvents(builder, {
    ...gemmBase,
    M: rows,
    N: h,
    K: headDim,
    left: 'O',
    right: 'Wo',
    out: 'Y',
    operator: 'Output Projection (Decode)',
    kernel: 'proj_o_kernel',
    label: label('Output Projection'),
    startWhat: `计算 Y[${rows}×${h}] = O[${rows}×${headDim}] × Wo[${headDim}×${h}]，投影回模型主干维度。`,
  });

  builder.push({
    type: 'GEMM_END',
    title: label('Attention（Decode）完成'),
    what: '新 token 的注意力输出 Y 已就绪，进入残差相加与 FFN/MoE。',
    why: 'Decode 的 Attention 复用 V0.2 的完整计算图，只是把"Q 的行数"从序列长度变为 1——这正是 KV Cache 带来的结构差异。',
    operator: 'Attention (Decode)',
    metadata: { seqLen: fullLen, dModel: h, headDim, decodeStep: opts.step },
  });
}

function buildDescription(profile: ModelProfile, task: InferenceTask): string {
  const phases: string[] = [];
  if (hasPrefill(task)) phases.push(`Prefill(${task.promptTokens} tokens)`);
  if (hasDecode(task)) phases.push(`Decode(${task.outputTokens} steps)`);
  return `${profile.displayName}（${profile.family} ${profile.version}）${phases.join(' + ')}, batch=${task.batchSize}, ${task.hardwareProfile.numSM} SM（Educational simulation · ${profile.fidelity} · not cycle-accurate）`;
}
