/**
 * 教学型 Multi-GPU Simulation Engine（实施手册 §27，V0.9）。
 *
 * 展示 Transformer 在 4 张 GPU 上以三种并行策略执行：
 *   - DP（Data Parallelism）：每 GPU 持有完整模型副本，处理不同 micro-batch，
 *     反向传播后对梯度做 AllReduce。
 *   - TP（Tensor Parallelism）：每个 GEMM 沿张量维度切分到各 GPU，
 *     每个线性层后需 AllReduce（列并行）/ AllGather（行并行）。
 *   - PP（Pipeline Parallelism）：不同层放在不同 GPU，micro-batch 沿流水线
 *     前向/反向流动，阶段间用 P2P 传激活与梯度。
 *
 * 借鉴 Chakra + ASTRA-sim 的解析式通信建模（手册 §27）：
 *   不模拟网络细节，只给出教学上合理的通信量与耗时估算（Simulated）。
 *
 * 架构约束：
 *   - 不修改 TVIR 基础架构（12 种事件类型保持不变）；
 *   - 计算阶段复用 emitGemmEvents / emitElementwiseEvents 原语；
 *   - 通信阶段用 commPrimitives（MEMORY_MOVE + metadata.comm）；
 *   - 本模块不 import React。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder } from '../simulation/eventBuilder';
import { emitGemmEvents } from '../simulation/gemmPrimitives';
import {
  emitAllReduce,
  emitReduceScatter,
  emitAllGather,
  emitP2P,
  formatBytes,
  DEFAULT_COMM_LINK,
  type CommLinkSpec,
} from './commPrimitives';

/** 并行策略 */
export type ParallelStrategy =
  | 'dp'
  | 'tp'
  | 'pp'
  | 'comm-allreduce'
  | 'comm-allgather'
  | 'comm-reducescatter';

/** 集合通信独立演示策略（不属于 TP/PP/DP 计算编排，只展示通信原语本身） */
export type CollectiveDemoStrategy = Extract<ParallelStrategy, `comm-${string}`>;

/** 判断是否为集合通信演示策略 */
export function isCollectiveDemo(strategy: ParallelStrategy): strategy is CollectiveDemoStrategy {
  return strategy.startsWith('comm-');
}

export interface MultiGpuConfig {
  /** 并行策略 */
  strategy: ParallelStrategy;
  /** GPU 数量（V0.9 固定 4） */
  numGpus: number;
  /** 序列长度 */
  seqLen: number;
  /** 隐藏维度 */
  dModel: number;
  /** FFN 中间维度 */
  ffnDim: number;
  /** GEMM tile 尺寸 */
  tileM: number;
  tileN: number;
  tileK: number;
  /** 每 GPU 的 SM 数 */
  numSM: number;
  warpsPerBlock: number;
  /** 通信链路（可选，默认 NVLink 量级教学常量） */
  link?: CommLinkSpec;
  /** DP 的 micro-batch 数（每 GPU 处理的 batch 数，教学用） */
  dpMicroBatches?: number;
  /** PP 的 micro-batch 数（流水线并发度） */
  ppMicroBatches?: number;
}

export const DEFAULT_MULTI_GPU_CONFIG: MultiGpuConfig = {
  strategy: 'dp',
  numGpus: 4,
  seqLen: 64,
  dModel: 64,
  ffnDim: 256,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  numSM: 4,
  warpsPerBlock: 4,
  dpMicroBatches: 4,
  ppMicroBatches: 4,
};

/**
 * 生成 Multi-GPU trace。
 */
export function simulateMultiGpu(config: MultiGpuConfig): TVIRTrace {
  const builder = createEventBuilder();
  const link = config.link ?? DEFAULT_COMM_LINK;

  builder.push({
    type: 'GEMM_START',
    title: `Multi-GPU 执行开始（${strategyLabel(config.strategy)}）`,
    what: `${config.numGpus} 张 GPU 以${strategyLabel(config.strategy)}并行执行一个 Transformer Block（seq=${config.seqLen}, d_model=${config.dModel}）。通信链路：${link.bandwidthGBps} GB/s、延迟 ${link.latencyUs} µs。`,
    why: `单卡放不下或算不动大模型时，需要把计算/数据拆分到多张 GPU。${strategyWhy(config.strategy)} 本 trace 用解析式模型估算通信量与耗时（Simulated，借鉴 ASTRA-sim），不模拟网络细节。`,
    operator: `Multi-GPU (${strategyLabel(config.strategy)})`,
    metadata: { multigpu: { strategy: config.strategy, numGpus: config.numGpus } },
  });

  switch (config.strategy) {
    case 'dp':
      emitDataParallel(builder, config, link);
      break;
    case 'tp':
      emitTensorParallel(builder, config, link);
      break;
    case 'pp':
      emitPipelineParallel(builder, config, link);
      break;
    case 'comm-allreduce':
      emitAllReduceDemo(builder, config, link);
      break;
    case 'comm-allgather':
      emitAllGatherDemo(builder, config, link);
      break;
    case 'comm-reducescatter':
      emitReduceScatterDemo(builder, config, link);
      break;
  }

  builder.push({
    type: 'GEMM_END',
    title: 'Multi-GPU 执行完成',
    what: `${config.numGpus} 张 GPU 协同完成一个 Transformer Block 的${strategyLabel(config.strategy)}执行。`,
    why: '理解三种并行策略的通信模式，是理解大规模分布式训练的基础。真实系统（Megatron、DeepSpeed）通常组合使用 TP+PP+DP（3D 并行）。',
    operator: `Multi-GPU (${strategyLabel(config.strategy)})`,
    metadata: { multigpu: { strategy: config.strategy, numGpus: config.numGpus } },
  });

  return {
    description: `Multi-GPU (${strategyLabel(config.strategy)}): ${config.numGpus} GPUs · seq=${config.seqLen} · d_model=${config.dModel} · link ${link.bandwidthGBps}GB/s · Simulated`,
    events: builder.events,
    provenance: 'simulation',
  };
}

function strategyLabel(strategy: ParallelStrategy): string {
  switch (strategy) {
    case 'dp':
      return '数据并行';
    case 'tp':
      return '张量并行';
    case 'pp':
      return '流水线并行';
    case 'comm-allreduce':
      return 'AllReduce 原语';
    case 'comm-allgather':
      return 'AllGather 原语';
    case 'comm-reducescatter':
      return 'ReduceScatter 原语';
  }
}

function strategyWhy(strategy: ParallelStrategy): string {
  switch (strategy) {
    case 'dp':
      return '数据并行（DP）让每张 GPU 持有完整模型副本，各自处理不同的 micro-batch，反向传播后对梯度做 AllReduce 求平均——通信集中在反向阶段。';
    case 'tp':
      return '张量并行（TP）把每个 GEMM 沿张量维度切分到各 GPU，让单个矩阵乘的计算本身并行化——通信穿插在每一层的计算中，频率高但单次数据量较小。';
    case 'pp':
      return '流水线并行（PP）把模型的不同层分配到不同 GPU，micro-batch 像流水线一样依次流过各阶段——通信是相邻阶段间的点对点（P2P）激活与梯度传递。';
    case 'comm-allreduce':
      return 'AllReduce 让每个 GPU 都拿到"所有 GPU 数据的归约结果"。Ring 实现把它分解为 ReduceScatter + AllGather 两段，总通信量 2(N-1)/N × S 与 GPU 数几乎无关——这是 NCCL 大规模训练的标准做法。';
    case 'comm-allgather':
      return 'AllGather 让每个 GPU 最终持有所有 GPU 数据的拼接结果（各 GPU 先各持 1/N）。Ring 实现下每个 GPU 沿环逐步广播自己完整的 chunk，N-1 步后人人持有全量。';
    case 'comm-reducescatter':
      return 'ReduceScatter 把"全量归约"拆成 N 份并行归约：N-1 步后，每个 GPU 各自持有 1/N 的完整归约结果。它是 Ring AllReduce 的前半段，也常用于 ZeRO 优化器的梯度分片。';
  }
}

// ---------------------------------------------------------------------------
// 数据并行（DP）
// ---------------------------------------------------------------------------

function emitDataParallel(builder: ReturnType<typeof createEventBuilder>, config: MultiGpuConfig, link: CommLinkSpec): void {
  const { numGpus, seqLen, dModel, tileM, tileN, tileK, numSM, warpsPerBlock } = config;
  const operator = 'Multi-GPU (数据并行)';
  const gemmBase = { tileM, tileN, tileK, numSM, warpsPerBlock };

  builder.push({
    type: 'TILE_CREATE',
    title: 'DP：每 GPU 持有完整模型副本',
    what: `${numGpus} 张 GPU 各自持有完整的模型权重副本。全局 batch 被切成 ${numGpus} 份，GPU ${Array.from({ length: numGpus }, (_, i) => i).join('/')} 分别处理 micro-batch 0/${numGpus - 1}。`,
    why: '数据并行的核心思想：模型副本相同，数据不同。每张 GPU 独立完成一次完整的前向+反向，只在梯度同步时通信。',
    operator,
    metadata: { multigpu: { strategy: 'dp', numGpus } },
  });

  // 各 GPU 并行前向 + 反向（用一个代表性 GEMM 示意每 GPU 的计算）
  for (let gpu = 0; gpu < numGpus; gpu++) {
    builder.push({
      type: 'KERNEL_LAUNCH',
      title: `GPU ${gpu}：前向 + 反向（micro-batch ${gpu}）`,
      what: `GPU ${gpu} 在本地 micro-batch ${gpu} 上独立完成完整的前向传播与反向传播，计算出本地梯度。`,
      why: '数据并行下，每张 GPU 的计算相互独立、完全并行——这是 DP 扩展性好的原因。此时各 GPU 的梯度还只是"本地"的，尚未汇总。',
      operator,
      sm: gpu,
      metadata: { multigpu: { strategy: 'dp', numGpus, gpu } },
    });

    // 用一个小 GEMM 示意该 GPU 上的代表性计算
    emitGemmEvents(builder, {
      ...gemmBase,
      M: Math.ceil(seqLen / numGpus),
      N: dModel,
      K: dModel,
      left: `X_gpu${gpu}`,
      right: 'Wq',
      out: `Q_gpu${gpu}`,
      operator: `DP GPU${gpu} 本地计算`,
      kernel: `dp_local_fwd_bwd_gpu${gpu}`,
      label: `GPU ${gpu} 本地 Q 投影（示意）`,
      startWhat: `GPU ${gpu} 在 micro-batch ${gpu} 上计算本地 Q 投影（前向的代表性一步；实际还包含完整 Attention、FFN 与反向传播）。`,
      startWhy: '为控制演示长度，此处只展示每 GPU 计算的一个代表性 GEMM。真实 DP 中每 GPU 会跑完整的 Transformer Block 前向+反向。',
    });
  }

  // 梯度 AllReduce
  const gradBytes = 2 * dModel * dModel; // 示意：一个权重矩阵的梯度（fp16）
  emitAllReduce(builder, {
    collective: 'allreduce',
    label: '梯度 AllReduce',
    numGpus,
    totalBytes: gradBytes,
    link,
    operator,
  });

  builder.push({
    type: 'SYNC',
    title: 'DP：梯度同步完成，各 GPU 更新权重',
    what: `AllReduce 完成后，${numGpus} 张 GPU 都持有相同的平均梯度，各自用优化器更新本地权重副本。更新后所有副本保持一致。`,
    why: '数据并行的正确性保证：所有 GPU 用相同的平均梯度更新相同的初始权重，因此副本保持同步。通信量与模型参数量成正比，是 DP 的主要开销。',
    operator,
    metadata: { multigpu: { strategy: 'dp', numGpus } },
  });
}

// ---------------------------------------------------------------------------
// 张量并行（TP）
// ---------------------------------------------------------------------------

function emitTensorParallel(builder: ReturnType<typeof createEventBuilder>, config: MultiGpuConfig, link: CommLinkSpec): void {
  const { numGpus, seqLen, dModel, tileM, tileN, tileK, numSM, warpsPerBlock } = config;
  const operator = 'Multi-GPU (张量并行)';
  const gemmBase = { tileM, tileN, tileK, numSM, warpsPerBlock };

  builder.push({
    type: 'TILE_CREATE',
    title: 'TP：权重沿列/行切分到各 GPU',
    what: `把线性层的权重矩阵沿输出维（列并行）或输入维（行并行）切成 ${numGpus} 份，GPU 0..${numGpus - 1} 各持有一份分片。每个 GPU 只需计算结果的一部分。`,
    why: '张量并行让单个 GEMM 的计算本身被多 GPU 分摊——适合单层计算量极大的情况（如大 FFN）。代价是每层计算后都要通信汇总。',
    operator,
    metadata: { multigpu: { strategy: 'tp', numGpus } },
  });

  // 列并行 GEMM（如 QKV 投影）：每 GPU 算一部分列，结果直接拼接（无通信）
  builder.push({
    type: 'KERNEL_LAUNCH',
    title: 'TP 列并行：QKV 投影（各 GPU 算一部分列）',
    what: `QKV 投影的权重按列切成 ${numGpus} 份，每个 GPU 用完整的输入 X 与自己的权重分片相乘，得到输出的 ${1 / numGpus} 列。`,
    why: '列并行的优点：每个 GPU 的输出列彼此独立，无需通信即可各自计算——输入 X 在每个 GPU 上已有完整副本。',
    operator,
    metadata: { multigpu: { strategy: 'tp', numGpus, phase: 'column' } },
  });

  for (let gpu = 0; gpu < numGpus; gpu++) {
    emitGemmEvents(builder, {
      ...gemmBase,
      M: seqLen,
      N: Math.ceil(dModel / numGpus),
      K: dModel,
      left: 'X',
      right: `W_shard${gpu}`,
      out: `Y_shard${gpu}`,
      operator: `TP GPU${gpu} 列并行分片`,
      kernel: `tp_column_gemm_gpu${gpu}`,
      label: `GPU ${gpu} 列并行 GEMM 分片`,
      startWhat: `GPU ${gpu} 计算 QKV 投影的第 ${gpu} 个列分片：Y_shard${gpu}[${seqLen}×${Math.ceil(dModel / numGpus)}] = X × W_shard${gpu}。`,
      startWhy: `每个 GPU 只算输出的 ${1 / numGpus}，计算量被 ${numGpus} 均分。列并行阶段各分片独立，无需通信。`,
    });
  }

  // 行并行 GEMM（如 Output Projection / FFN Down）：每 GPU 算部分和，需 AllReduce 汇总
  builder.push({
    type: 'KERNEL_LAUNCH',
    title: 'TP 行并行：Output Projection（各 GPU 算部分和）',
    what: `行并行把权重按输入维切分，每个 GPU 用自己的输入分片与权重分片相乘，得到一个"部分和"。${numGpus} 个部分和相加才是完整结果。`,
    why: '行并行的输出是各 GPU 部分和之和，因此必须通信汇总——这正是列并行+行并行配对使用、把通信压缩到每层一次的原因。',
    operator,
    metadata: { multigpu: { strategy: 'tp', numGpus, phase: 'row' } },
  });

  for (let gpu = 0; gpu < numGpus; gpu++) {
    emitGemmEvents(builder, {
      ...gemmBase,
      M: seqLen,
      N: dModel,
      K: Math.ceil(dModel / numGpus),
      left: `H_shard${gpu}`,
      right: `Wo_shard${gpu}`,
      out: `PartialSum_gpu${gpu}`,
      operator: `TP GPU${gpu} 行并行分片`,
      kernel: `tp_row_gemm_gpu${gpu}`,
      label: `GPU ${gpu} 行并行 GEMM 分片`,
      startWhat: `GPU ${gpu} 计算 Output Projection 的第 ${gpu} 个部分和：PartialSum_gpu${gpu} = H_shard${gpu} × Wo_shard${gpu}。`,
      startWhy: `每个 GPU 用 ${1 / numGpus} 的输入分片计算部分和，计算量被 ${numGpus} 均分。接下来需要 AllReduce 把 ${numGpus} 个部分和相加。`,
    });
  }

  // 行并行的部分和 AllReduce
  const partialSumBytes = 2 * seqLen * dModel; // fp16
  emitAllReduce(builder, {
    collective: 'allreduce',
    label: 'TP 行并行部分和 AllReduce',
    numGpus,
    totalBytes: partialSumBytes,
    link,
    operator,
  });

  builder.push({
    type: 'SYNC',
    title: 'TP：本层张量并行完成',
    what: `AllReduce 汇总后，每个 GPU 都得到完整的 Output Projection 结果，可继续下一层。一个 Transformer 层的 TP 包含"列并行 → 行并行 + AllReduce"。`,
    why: '张量并行把通信压缩到每层两次集合通信（Attention 输出一次、FFN 输出一次）。通信频繁但单次数据量较小，适合 NVLink 等高带宽互连。',
    operator,
    metadata: { multigpu: { strategy: 'tp', numGpus } },
  });
}

// ---------------------------------------------------------------------------
// 流水线并行（PP）
// ---------------------------------------------------------------------------

function emitPipelineParallel(builder: ReturnType<typeof createEventBuilder>, config: MultiGpuConfig, link: CommLinkSpec): void {
  const { numGpus, seqLen, dModel, tileM, tileN, tileK, numSM, warpsPerBlock, ppMicroBatches } = config;
  const operator = 'Multi-GPU (流水线并行)';
  const gemmBase = { tileM, tileN, tileK, numSM, warpsPerBlock };
  const microBatches = ppMicroBatches ?? 4;
  const activationBytes = 2 * seqLen * dModel; // fp16 激活

  builder.push({
    type: 'TILE_CREATE',
    title: 'PP：模型层按阶段分配到各 GPU',
    what: `把 Transformer 的层切成 ${numGpus} 个连续阶段（stage），GPU 0..${numGpus - 1} 各负责一段。输入 batch 被切成 ${microBatches} 个 micro-batch，依次流过各阶段。`,
    why: '流水线并行让每张 GPU 只持有模型的一部分层，降低单卡显存压力。micro-batch 让多个 GPU 能同时处理不同 micro-batch，填满流水线、减少空闲（bubble）。',
    operator,
    metadata: { multigpu: { strategy: 'pp', numGpus, microBatches } },
  });

  // 前向：micro-batch 依次流过各阶段（展示前 2 个 micro-batch 的完整流动，其余概括）
  const shownMicroBatches = Math.min(2, microBatches);
  for (let mb = 0; mb < shownMicroBatches; mb++) {
    for (let stage = 0; stage < numGpus; stage++) {
      // 阶段间 P2P（第一个阶段接收输入，不算 P2P）
      if (stage > 0) {
        emitP2P(builder, {
          from: stage - 1,
          to: stage,
          bytes: activationBytes,
          link,
          operator,
          label: `PP 前向：GPU ${stage - 1} → GPU ${stage}（micro-batch ${mb}）`,
          what: `GPU ${stage - 1} 把 micro-batch ${mb} 在阶段 ${stage - 1} 的输出激活（${formatBytes(activationBytes)}）点对点发送给 GPU ${stage}。`,
          why: '流水线并行的通信是相邻阶段间的 P2P：每个阶段只需把激活传给下一阶段。通信量小（只传激活，不传全部梯度/参数），但要求阶段间严格同步。',
        });
      }

      // 该阶段的计算
      emitGemmEvents(builder, {
        ...gemmBase,
        M: seqLen,
        N: dModel,
        K: dModel,
        left: `Act_stage${stage}_mb${mb}`,
        right: `W_stage${stage}`,
        out: `Act_stage${stage + 1}_mb${mb}`,
        operator: `PP GPU${stage} 阶段计算`,
        kernel: `pp_stage${stage}_mb${mb}`,
        label: `GPU ${stage} 处理 micro-batch ${mb}`,
        startWhat: `GPU ${stage} 对 micro-batch ${mb} 执行它负责的层（阶段 ${stage}），产出传给下一阶段的激活。`,
        startWhy: `流水线的每个阶段只做自己那几层的计算。多个 micro-batch 错开进入，让不同 GPU 能同时工作。`,
      });
    }
  }

  if (microBatches > shownMicroBatches) {
    builder.push({
      type: 'SYNC',
      title: `PP：其余 ${microBatches - shownMicroBatches} 个 micro-batch 继续流过流水线`,
      what: `micro-batch ${shownMicroBatches}..${microBatches - 1} 以相同方式依次流过 ${numGpus} 个阶段（为控制演示长度不再逐一展开）。`,
      why: 'micro-batch 越多，流水线填得越满，GPU 空闲（bubble）越少——这是 GPipe 等流水线调度的核心权衡。',
      operator,
      metadata: { multigpu: { strategy: 'pp', numGpus, microBatches } },
    });
  }

  builder.push({
    type: 'SYNC',
    title: 'PP：前向完成，进入反向（梯度沿流水线回流）',
    what: `所有 micro-batch 前向完成后，反向传播沿相反方向流动：梯度从最后一个阶段逐级 P2P 传回，每个阶段用收到的梯度更新自己那几层的权重。`,
    why: '流水线的反向是前向的镜像：梯度沿阶段反向传递。前向+反向共同构成一次完整的流水线迭代，bubble 比例约为 (阶段数-1)/micro-batch 数。',
    operator,
    metadata: { multigpu: { strategy: 'pp', numGpus, microBatches } },
  });
}

// ---------------------------------------------------------------------------
// 集合通信原语独立演示（AllReduce / AllGather / ReduceScatter）
// 手册 §27：Multi-GPU 视图需展示 AllReduce、AllGather、ReduceScatter。
// 这三个演示不编排模型计算，只聚焦通信原语本身，帮助理解 DP/TP 背后的通信基元。
// ---------------------------------------------------------------------------

/** 集合通信演示的数据量：以一个权重矩阵（dModel×dModel，fp16）为示意负载 */
function collectiveDemoBytes(config: MultiGpuConfig): number {
  return 2 * config.dModel * config.dModel;
}

function emitAllReduceDemo(
  builder: ReturnType<typeof createEventBuilder>,
  config: MultiGpuConfig,
  link: CommLinkSpec,
): void {
  const { numGpus } = config;
  const operator = 'Multi-GPU (AllReduce 原语)';
  const totalBytes = collectiveDemoBytes(config);

  builder.push({
    type: 'TILE_CREATE',
    title: 'AllReduce：每 GPU 持有一份待归约数据',
    what: `${numGpus} 张 GPU 各持有一份本地数据（示意：一个 ${config.dModel}×${config.dModel} 的梯度矩阵，共 ${formatBytes(totalBytes)}）。目标：让每张 GPU 都拿到所有数据的归约和。`,
    why: 'AllReduce 是数据并行梯度同步的核心原语。Ring 实现分 ReduceScatter（前半段，各 GPU 归约出 1/N 完整 chunk）+ AllGather（后半段，把完整 chunk 广播给所有人）两段，总通信量 2(N-1)/N × S。',
    operator,
    metadata: { multigpu: { strategy: 'comm-allreduce', numGpus } },
  });

  emitAllReduce(builder, {
    collective: 'allreduce',
    label: 'AllReduce 演示',
    numGpus,
    totalBytes,
    link,
    operator,
  });

  builder.push({
    type: 'SYNC',
    title: 'AllReduce 完成：每 GPU 都持有归约和',
    what: `经过 2(N-1) = ${2 * (numGpus - 1)} 步环通信，${numGpus} 张 GPU 都持有完整的归约结果。`,
    why: 'Ring AllReduce 的通信量 2(N-1)/N × S 与 GPU 数几乎无关，因此能扩展到成百上千张 GPU——这是它取代星型/树型归约的原因。',
    operator,
    metadata: { multigpu: { strategy: 'comm-allreduce', numGpus } },
  });
}

function emitAllGatherDemo(
  builder: ReturnType<typeof createEventBuilder>,
  config: MultiGpuConfig,
  link: CommLinkSpec,
): void {
  const { numGpus } = config;
  const operator = 'Multi-GPU (AllGather 原语)';
  const totalBytes = collectiveDemoBytes(config);

  builder.push({
    type: 'TILE_CREATE',
    title: 'AllGather：每 GPU 持有 1/N 数据',
    what: `${numGpus} 张 GPU 各持有完整数据的 1/${numGpus}（示意：一个 ${config.dModel}×${config.dModel} 矩阵被切成 ${numGpus} 份，总计 ${formatBytes(totalBytes)}）。目标：让每张 GPU 都持有拼接后的全量数据。`,
    why: 'AllGather 是"把分片重新拼成整体"的通信原语，用于 ZeRO-3 参数收集、TP 行并行输出拼接等场景。Ring 实现下每个 GPU 沿环逐步广播自己完整的 chunk，N-1 步后人人持有全量。',
    operator,
    metadata: { multigpu: { strategy: 'comm-allgather', numGpus } },
  });

  emitAllGather(builder, {
    collective: 'allgather',
    label: 'AllGather 演示',
    numGpus,
    totalBytes,
    link,
    operator,
  });

  builder.push({
    type: 'SYNC',
    title: 'AllGather 完成：每 GPU 都持有全量数据',
    what: `经过 N-1 = ${numGpus - 1} 步环通信，${numGpus} 张 GPU 都持有拼接后的完整数据。`,
    why: 'AllGather 的通信量为 (N-1)/N × S：每个 GPU 发送自己的 1/N 共 N-1 次，同时接收他人发来的 N-1 个 1/N。',
    operator,
    metadata: { multigpu: { strategy: 'comm-allgather', numGpus } },
  });
}

function emitReduceScatterDemo(
  builder: ReturnType<typeof createEventBuilder>,
  config: MultiGpuConfig,
  link: CommLinkSpec,
): void {
  const { numGpus } = config;
  const operator = 'Multi-GPU (ReduceScatter 原语)';
  const totalBytes = collectiveDemoBytes(config);

  builder.push({
    type: 'TILE_CREATE',
    title: 'ReduceScatter：每 GPU 持有一份完整待归约数据',
    what: `${numGpus} 张 GPU 各持有一份完整数据（示意：一个 ${config.dModel}×${config.dModel} 矩阵，共 ${formatBytes(totalBytes)}）。目标：把数据切成 ${numGpus} 份，每份被所有 GPU 归约，最终每张 GPU 各自持有 1/${numGpus} 的完整归约结果。`,
    why: 'ReduceScatter 是"归约后分散持有"的通信原语——它是 Ring AllReduce 的前半段，也用于 ZeRO 优化器把梯度分片到各 GPU。Ring 实现下每步每个 GPU 把自己负责 chunk 的局部归约结果发给下一个邻居，N-1 步完成。',
    operator,
    metadata: { multigpu: { strategy: 'comm-reducescatter', numGpus } },
  });

  emitReduceScatter(builder, {
    collective: 'reduce_scatter',
    label: 'ReduceScatter 演示',
    numGpus,
    totalBytes,
    link,
    operator,
  });

  builder.push({
    type: 'SYNC',
    title: 'ReduceScatter 完成：每 GPU 各持有 1/N 归约结果',
    what: `经过 N-1 = ${numGpus - 1} 步环通信，${numGpus} 张 GPU 各自持有完整归约结果的 1/${numGpus}（不同 GPU 持有不同分片）。`,
    why: '与 AllReduce 不同，ReduceScatter 结束后没有 GPU 持有全量——这正是它通信量只有 AllReduce 一半、并适合 ZeRO 梯度分片的原因。再接一段 AllGather 即构成完整 Ring AllReduce。',
    operator,
    metadata: { multigpu: { strategy: 'comm-reducescatter', numGpus } },
  });
}
