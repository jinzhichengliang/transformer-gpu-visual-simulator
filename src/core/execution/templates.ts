/**
 * Operator Template Registry — 算子展开模板注册表（Sprint 5, Task D2）。
 *
 * 任务书 PASS 标准：
 *   - Operator expansion 使用注册机制（OperatorTemplateRegistry），
 *     而不是 switch(modelName)；
 *   - Attention 展开为 Q/K/V Projection → Attention Compute → Output Projection；
 *   - MoE 展开为 Router → Top-K → Dispatch → Expert Compute → Combine。
 *
 * 所有模板复用既有 TVIR 原语（emitGemmEvents / emitElementwiseEvents /
 * emitAttentionEvents），不新增事件类型。
 * 本模块是纯逻辑层，不 import React。
 */

import type { EventBuilder } from '../simulation/eventBuilder';
import { emitGemmEvents } from '../simulation/gemmPrimitives';
import { emitElementwiseEvents } from '../simulation/elementwisePrimitives';
import { emitAttentionEvents } from '../simulation/attentionEngine';
import type { OperatorNode } from './planner';
import type { ModelProfile } from '../modelprofile/types';
import type { InferenceTask } from './task';

/** 模板展开上下文 */
export interface EmitContext {
  builder: EventBuilder;
  profile: ModelProfile;
  task: InferenceTask;
  node: OperatorNode;
  /** 当前展开的有效行数（prefill=M，decode=batch） */
  rows: number;
  /** 当前阶段标签（写入教学文案） */
  phaseLabel: string;
  /** GEMM tiling / 硬件参数 */
  tileSize: number;
  numSM: number;
  warpsPerBlock: number;
  /**
   * 教学抽样（可选，Task H3）：大型模型下限制每个 GEMM 详细展示的
   * Block 数与 K 段数，其余用汇总事件概括。缺省 = 全部展开。
   */
  sample?: { blocks: number; kIterations: number };
}

export type OperatorEmitter = (ctx: EmitContext) => void;

/** 注册表：算子类型 → 展开模板（注册机制，非 switch(modelName)） */
const registry = new Map<string, OperatorEmitter>();

export function registerOperatorTemplate(operatorType: string, emitter: OperatorEmitter): void {
  registry.set(operatorType, emitter);
}

export function getOperatorTemplate(operatorType: string): OperatorEmitter | undefined {
  return registry.get(operatorType);
}

/** 模型隐藏维度（带回退） */
function hiddenSizeOf(profile: ModelProfile): number {
  return profile.architecture.hiddenSize?.value ?? 128;
}

/** FFN 中间层维度：优先取 profile 中 ffn 层声明，回退 4×hidden */
function intermediateSizeOf(profile: ModelProfile): number {
  const ffnLayer = profile.layers.find((l) => l.type === 'ffn');
  return ffnLayer?.ffn?.intermediateSize?.value ?? hiddenSizeOf(profile) * 4;
}

/** 层标签（用于教学文案） */
function layerTag(node: OperatorNode): string {
  return node.layerIndex >= 0 ? `Layer ${node.layerIndex} · ` : '';
}

// ─────────────────────────────────────────────
// 模板：Embedding（查表，访存密集）
// ─────────────────────────────────────────────
registerOperatorTemplate('embedding', (ctx) => {
  const h = hiddenSizeOf(ctx.profile);
  emitElementwiseEvents(ctx.builder, {
    rows: ctx.rows,
    cols: h,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    operator: 'Embedding',
    kernel: 'embedding_lookup_kernel',
    label: `${ctx.phaseLabel}Embedding`,
    tensor: 'EmbeddingTable',
    launchWhat: `启动 Embedding 查表 kernel：把 ${ctx.rows} 个 token id 各自映射为一个 ${h} 维向量（从词嵌入表中取出对应行）。`,
    launchWhy: '模型的输入是离散的 token id，必须先查嵌入表变成稠密向量才能参与矩阵运算。这是一次纯访存操作（gather），没有乘加计算。',
    computeWhat: '每个线程按 token id 计算行偏移，从嵌入表中读出对应的嵌入向量。',
    computeWhy: 'Embedding 是典型的带宽瓶颈 kernel：计算量≈0，时间几乎全花在 HBM 读取上。',
  });
});

// ─────────────────────────────────────────────
// 模板：Attention（复用 V0.2/V0.3 子图原语，禁止复制逻辑）
// ─────────────────────────────────────────────
registerOperatorTemplate('attention', (ctx) => {
  const h = hiddenSizeOf(ctx.profile);
  const attn = ctx.profile.layers.find((l) => l.type === 'attention')?.attention;
  const headDim = attn?.headDim?.value ?? Math.min(64, h);
  emitAttentionEvents(ctx.builder, {
    seqLen: ctx.rows,
    dModel: h,
    headDim,
    tileM: ctx.tileSize,
    tileN: ctx.tileSize,
    tileK: ctx.tileSize,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    ...(ctx.sample ? { sample: ctx.sample } : {}),
  });
});

// ─────────────────────────────────────────────
// 模板：FFN（Up → SiLU → Down，复用 block engine 的形状约定）
// ─────────────────────────────────────────────
registerOperatorTemplate('ffn', (ctx) => {
  const h = hiddenSizeOf(ctx.profile);
  const ffnDim = intermediateSizeOf(ctx.profile);
  const tag = layerTag(ctx.node);
  const gemmBase = {
    tileM: ctx.tileSize,
    tileN: ctx.tileSize,
    tileK: ctx.tileSize,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    ...(ctx.sample
      ? { sampledBlocks: ctx.sample.blocks, sampledKIterations: ctx.sample.kIterations }
      : {}),
  };

  emitGemmEvents(ctx.builder, {
    ...gemmBase,
    M: ctx.rows,
    N: ffnDim,
    K: h,
    left: 'Xn',
    right: 'Wup',
    out: 'H',
    operator: 'FFN Up Projection',
    kernel: 'ffn_up_kernel',
    label: `${ctx.phaseLabel}${tag}FFN Up Projection`,
    startWhat: `计算 H[${ctx.rows}×${ffnDim}] = Xn[${ctx.rows}×${h}] × Wup[${h}×${ffnDim}]，把每个 token 升维到 ${ffnDim}。`,
  });

  emitElementwiseEvents(ctx.builder, {
    rows: ctx.rows,
    cols: ffnDim,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    operator: 'FFN SiLU',
    kernel: 'silu_kernel',
    label: `${ctx.phaseLabel}${tag}SiLU`,
    tensor: 'H',
    launchWhat: `启动 SiLU 激活 kernel：对 H 的每个元素计算 SiLU(x) = x·σ(x)。`,
    launchWhy: '非线性激活让两次 GEMM 的组合能表达复杂函数。SiLU 是现代 LLM FFN 的常见选择。',
    computeWhat: '每个线程对寄存器中的元素计算 x·σ(x)。',
    computeWhy: '激活是逐元素操作，访存密集，瓶颈在 HBM 带宽。',
  });

  emitGemmEvents(ctx.builder, {
    ...gemmBase,
    M: ctx.rows,
    N: h,
    K: ffnDim,
    left: 'H',
    right: 'Wdown',
    out: 'F',
    operator: 'FFN Down Projection',
    kernel: 'ffn_down_kernel',
    label: `${ctx.phaseLabel}${tag}FFN Down Projection`,
    startWhat: `计算 F[${ctx.rows}×${h}] = H[${ctx.rows}×${ffnDim}] × Wdown[${ffnDim}×${h}]，把维度降回 ${h}。`,
  });
});

// ─────────────────────────────────────────────
// 模板：MoE（Router → Top-K → Dispatch → Expert GEMM → Combine）
// ─────────────────────────────────────────────
registerOperatorTemplate('moe', (ctx) => {
  const h = hiddenSizeOf(ctx.profile);
  const moeLayer = ctx.profile.layers.find((l) => l.type === 'moe');
  const numExperts = moeLayer?.moe?.numExperts?.value ?? 8;
  const topK = moeLayer?.moe?.expertsPerToken?.value ?? 2;
  const ffnDim = intermediateSizeOf(ctx.profile);
  const tag = layerTag(ctx.node);
  const gemmBase = {
    tileM: ctx.tileSize,
    tileN: ctx.tileSize,
    tileK: ctx.tileSize,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    ...(ctx.sample
      ? { sampledBlocks: ctx.sample.blocks, sampledKIterations: ctx.sample.kIterations }
      : {}),
  };

  // 1. Router：每个 token 对全部专家打分 [rows×h] × [h×numExperts]
  emitGemmEvents(ctx.builder, {
    ...gemmBase,
    M: ctx.rows,
    N: numExperts,
    K: h,
    left: 'Xn',
    right: 'Wrouter',
    out: 'RouterScores',
    operator: 'MoE Router',
    kernel: 'moe_router_kernel',
    label: `${ctx.phaseLabel}${tag}MoE Router`,
    startWhat: `计算 Router 分数 [${ctx.rows}×${numExperts}] = Xn[${ctx.rows}×${h}] × Wrouter[${h}×${numExperts}]：每个 token 对 ${numExperts} 个专家各得到一个匹配分数。`,
    startWhy: '并不是所有专家都会处理当前 Token。Router 根据当前 Token 表征计算专家选择分数，选择部分专家参与计算，从而降低激活计算量——这是 MoE 的核心思想。',
  });

  // 2. Top-K：选出每 token 的 top-K 个专家
  emitElementwiseEvents(ctx.builder, {
    rows: ctx.rows,
    cols: numExperts,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    operator: 'MoE Top-K',
    kernel: 'moe_topk_kernel',
    label: `${ctx.phaseLabel}${tag}MoE Top-K`,
    tensor: 'RouterScores',
    launchWhat: `启动 Top-K 选择 kernel：对每个 token 的 ${numExperts} 个分数排序，选出分数最高的 ${topK} 个专家并做 softmax 归一化权重。`,
    launchWhy: `只激活 ${topK}/${numExperts} 的专家，让模型拥有巨大总参数量的同时，每个 token 的激活计算量只相当于一小部分专家——"大参数、低激活成本"。`,
    computeWhat: '行内排序取前 K，再对选中的分数做 softmax 得到专家权重。',
    computeWhy: 'Top-K 是行归约类操作，与 Softmax 同类，由 CUDA Core 执行。',
  });

  // 3. Dispatch：把 token 按专家分组（教学示意：每激活专家一条搬运事件）
  const shownExperts = Math.min(topK, numExperts);
  ctx.builder.push({
    type: 'MEMORY_MOVE',
    title: `${ctx.phaseLabel}${tag}Token Dispatch → ${shownExperts} 个专家`,
    what: `根据 Top-K 结果，把每个 token 的隐藏向量分发到被选中的 ${shownExperts} 个专家的输入缓冲区（每个 token 去 ${topK} 个专家）。`,
    why: '专家之间相互独立，必须先把 token 按"归属专家"重排分组，每个专家才能连续处理属于它的 token 批。这一步是纯数据搬运（gather/scatter），不做乘加。',
    operator: 'MoE Dispatch',
    source: 'HBM',
    destination: 'SHARED_MEMORY',
    tensor: 'Xn',
    metadata: { moe: { numExperts, topK, rows: ctx.rows } },
  });

  // 4. Expert Compute：展示被激活的专家（教学抽样：前 topK 个），其余专家空闲
  const tokensPerExpert = Math.max(1, Math.ceil((ctx.rows * topK) / numExperts));
  for (let e = 0; e < shownExperts; e++) {
    emitGemmEvents(ctx.builder, {
      ...gemmBase,
      M: tokensPerExpert,
      N: h,
      K: ffnDim,
      left: `Expert${e}_in`,
      right: `Expert${e}_W`,
      out: `Expert${e}_out`,
      operator: `MoE Expert GEMM (Expert ${e})`,
      kernel: `moe_expert${e}_kernel`,
      label: `${ctx.phaseLabel}${tag}Expert ${e} FFN`,
      startWhat: `专家 ${e} 对属于它的 ${tokensPerExpert} 个 token 做 FFN 计算（Up→SiLU→Down 的教学合并视图，等效形状 [${tokensPerExpert}×${h}]）。`,
      startWhy: '每个专家本质上是一个小型 FFN。被选中的专家并行计算各自的 token 子集——专家间无依赖，天然适合 GPU 并行。',
    });
  }
  if (numExperts > shownExperts) {
    ctx.builder.push({
      type: 'SYNC',
      title: `${ctx.phaseLabel}${tag}其余 ${numExperts - shownExperts} 个专家本轮空闲`,
      what: `Top-K 路由只选中 ${topK} 个专家，其余 ${numExperts - shownExperts} 个专家在本步不参与计算（权重为 0）。`,
      why: '这正是 MoE 稀疏激活的意义：总参数量 = 全部专家之和，但每步的计算量只与少数被激活专家有关。',
      operator: 'MoE Dispatch',
      metadata: { moe: { numExperts, topK, idleExperts: numExperts - shownExperts } },
    });
  }

  // 5. Combine：按路由权重加权求和
  emitElementwiseEvents(ctx.builder, {
    rows: ctx.rows,
    cols: h,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    operator: 'MoE Combine',
    kernel: 'moe_combine_kernel',
    label: `${ctx.phaseLabel}${tag}MoE Combine`,
    tensor: 'Expert_out',
    launchWhat: `启动 Combine kernel：对每个 token，把 ${topK} 个专家的输出按路由权重加权求和，得到该层 MoE 的最终输出。`,
    launchWhy: '多专家的结果必须合并回每个 token 一条的隐藏向量，才能进入残差相加与下一层。加权系数来自 Top-K 阶段的 softmax 权重。',
    computeWhat: '每个线程对对应位置累加 topK 个专家输出 × 权重。',
    computeWhy: 'Combine 是访存密集的归约操作，瓶颈在 HBM 读写。',
  });
});

// ─────────────────────────────────────────────
// 模板：LM Head（hidden → vocab logits）
// ─────────────────────────────────────────────
registerOperatorTemplate('lm_head', (ctx) => {
  const h = hiddenSizeOf(ctx.profile);
  const vocab = ctx.profile.architecture.vocabSize?.value ?? 1024;
  emitGemmEvents(ctx.builder, {
    tileM: ctx.tileSize,
    tileN: ctx.tileSize,
    tileK: ctx.tileSize,
    numSM: ctx.numSM,
    warpsPerBlock: ctx.warpsPerBlock,
    // 教学抽样：vocab 很大（如 128000）时必须折叠，否则事件爆炸
    ...(ctx.sample
      ? { sampledBlocks: ctx.sample.blocks, sampledKIterations: ctx.sample.kIterations }
      : {}),
    M: ctx.rows,
    N: vocab,
    K: h,
    left: 'X_final',
    right: 'Wlm',
    out: 'logits',
    operator: 'LM Head',
    kernel: 'lm_head_kernel',
    label: `${ctx.phaseLabel}LM Head`,
    startWhat: `计算 logits[${ctx.rows}×${vocab}] = X_final[${ctx.rows}×${h}] × Wlm[${h}×${vocab}]：把每个 token 的最终表征投影到词表空间，得到下一个 token 的候选分数。`,
    startWhy: '语言模型的输出是"下一个词的概率分布"。LM Head 把隐藏向量与词表嵌入相乘，得到每个词的得分（logits），之后 softmax 采样出下一个 token。',
  });
});
