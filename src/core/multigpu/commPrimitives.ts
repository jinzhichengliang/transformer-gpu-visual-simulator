/**
 * 集合通信原语（V0.9，实施手册 §27）。
 *
 * 借鉴 Chakra + ASTRA-sim 的解析式通信建模（alpha-beta 模型），
 * **不模拟网络细节**（不做路由、拥塞、逐跳仿真）——只给出教学上合理的
 * 通信量与耗时估算（全部标注 Simulated，见 CONCEPTS.md V0.9）。
 *
 * 架构约束：
 *   - 复用 TVIR 既有 12 种事件类型，不新增词汇表（手册 §31 的延续）；
 *   - 通信步骤用 MEMORY_MOVE 事件表达，跨 GPU 信息写入 metadata.comm；
 *   - 计算/同步用 KERNEL_LAUNCH / SYNC 事件表达；
 *   - 本模块不 import React、不 import 其他 Simulation 业务逻辑
 *     （仅复用 eventBuilder 编号基础设施）。
 *
 * 环（Ring）算法（NCCL 的标准实现，教学呈现）：
 *   - ReduceScatter：N-1 步，每步每个 GPU 把一个 chunk 的局部归约结果
 *     发给下一个邻居，最终每个 GPU 持有 1/N 的完整归约结果。
 *   - AllGather：N-1 步，每步每个 GPU 把一个已完整的 chunk 发给下一个邻居，
 *     最终每个 GPU 持有完整数据。
 *   - AllReduce = ReduceScatter + AllGather（通信量 2(N-1)/N × S）。
 */

import type { EventBuilder } from '../simulation/eventBuilder';

/** 集合通信类型 */
export type CollectiveType =
  | 'allreduce'
  | 'allgather'
  | 'reduce_scatter'
  | 'p2p'
  | 'broadcast';

/** 环算法单步内的一次 GPU 间传输 */
export interface CommTransfer {
  from: number;
  to: number;
  /** 传输的 chunk 编号（教学示意） */
  chunk: number;
}

/** 写入事件 metadata.comm 的通信元信息（供 MultiGpuView 消费） */
export interface CommMeta {
  collective: CollectiveType;
  /** AllReduce 分解出的子阶段；其他集合通信为 'single' */
  phase: 'reduce_scatter' | 'allgather' | 'single';
  /** 环算法步序号（0-based）；非环算法为 0 */
  ringStep: number;
  totalRingSteps: number;
  /** 本步内的全部传输（环上所有 GPU 同时发送） */
  transfers: CommTransfer[];
  /** 每次传输的字节数 */
  bytesPerTransfer: number;
  /** 本步耗时估算（µs，Simulated） */
  durationUs: number;
  numGpus: number;
  /** 所属逻辑阶段名（如 'Gradient AllReduce'） */
  label: string;
}

/** 通信链路的解析式耗时参数（alpha-beta 模型，教学常量） */
export interface CommLinkSpec {
  /** 链路带宽（GB/s），如 NVLink */
  bandwidthGBps: number;
  /** 单跳延迟（µs，alpha 项） */
  latencyUs: number;
}

/** 默认通信链路参数（教学示意，不代表真实互连） */
export const DEFAULT_COMM_LINK: CommLinkSpec = {
  bandwidthGBps: 300, // 近似 NVLink 量级
  latencyUs: 5,
};

/**
 * 环算法单步耗时估算（alpha-beta 模型）：
 *   time = latency + bytesPerTransfer / bandwidth
 * 标注为 Simulated 教学估算。
 */
export function ringStepDurationUs(
  bytesPerTransfer: number,
  link: CommLinkSpec,
): number {
  const transferUs = (bytesPerTransfer / (link.bandwidthGBps * 1e9)) * 1e6;
  return link.latencyUs + transferUs;
}

/**
 * Ring ReduceScatter 的 chunk 传输模式。
 * N-1 步；第 k 步，GPU i 把 chunk (i - k) mod N 发给 GPU (i+1) mod N。
 */
export function reduceScatterTransfers(
  numGpus: number,
  step: number,
): CommTransfer[] {
  const transfers: CommTransfer[] = [];
  for (let i = 0; i < numGpus; i++) {
    const chunk = ((i - step) % numGpus + numGpus) % numGpus;
    transfers.push({ from: i, to: (i + 1) % numGpus, chunk });
  }
  return transfers;
}

/**
 * Ring AllGather 的 chunk 传输模式。
 * N-1 步；第 k 步，GPU i 把 chunk (i - k + 1) mod N 发给 GPU (i+1) mod N。
 */
export function allGatherTransfers(numGpus: number, step: number): CommTransfer[] {
  const transfers: CommTransfer[] = [];
  for (let i = 0; i < numGpus; i++) {
    const chunk = ((i - step + 1) % numGpus + numGpus) % numGpus;
    transfers.push({ from: i, to: (i + 1) % numGpus, chunk });
  }
  return transfers;
}

export interface EmitCollectiveConfig {
  collective: CollectiveType;
  /** 逻辑阶段名，如 'Gradient AllReduce' */
  label: string;
  /** 参与通信的 GPU 数 */
  numGpus: number;
  /** 待通信的总数据量（字节） */
  totalBytes: number;
  /** 通信链路参数 */
  link: CommLinkSpec;
  /** 所属算子名（写入 operator 字段） */
  operator: string;
}

/**
 * 向 builder 追加一段 Ring ReduceScatter 事件流（N-1 个 MEMORY_MOVE 事件）。
 */
export function emitReduceScatter(builder: EventBuilder, config: EmitCollectiveConfig): number {
  const { numGpus, totalBytes, link, operator, label } = config;
  const steps = numGpus - 1;
  const bytesPerTransfer = totalBytes / numGpus;
  const durationUs = ringStepDurationUs(bytesPerTransfer, link);
  let totalUs = 0;

  for (let step = 0; step < steps; step++) {
    const transfers = reduceScatterTransfers(numGpus, step);
    const meta: CommMeta = {
      collective: 'reduce_scatter',
      phase: 'reduce_scatter',
      ringStep: step,
      totalRingSteps: steps,
      transfers,
      bytesPerTransfer,
      durationUs,
      numGpus,
      label,
    };
    builder.push({
      type: 'MEMORY_MOVE',
      title: `${label} · ReduceScatter 第 ${step + 1}/${steps} 步`,
      what: `环上 ${numGpus} 个 GPU 同时把各自负责 chunk 的局部归约结果发给下一个邻居（每 GPU 发送 ${formatBytes(bytesPerTransfer)}）。第 ${step + 1} 步完成后，每个 GPU 多持有一个完整归约的 chunk。`,
      why: `ReduceScatter 把"全量归约"拆成 ${numGpus} 份并行归约：每个 GPU 只需处理 1/${numGpus} 的数据，通信与计算都能并行。这是 Ring AllReduce 的前半段，通信量 (N-1)/N × S。`,
      operator,
      source: 'HBM',
      destination: 'HBM',
      metadata: { comm: meta },
    });
    totalUs += durationUs;
  }
  return totalUs;
}

/**
 * 向 builder 追加一段 Ring AllGather 事件流（N-1 个 MEMORY_MOVE 事件）。
 */
export function emitAllGather(builder: EventBuilder, config: EmitCollectiveConfig): number {
  const { numGpus, totalBytes, link, operator, label } = config;
  const steps = numGpus - 1;
  const bytesPerTransfer = totalBytes / numGpus;
  const durationUs = ringStepDurationUs(bytesPerTransfer, link);
  let totalUs = 0;

  for (let step = 0; step < steps; step++) {
    const transfers = allGatherTransfers(numGpus, step);
    const meta: CommMeta = {
      collective: 'allgather',
      phase: 'allgather',
      ringStep: step,
      totalRingSteps: steps,
      transfers,
      bytesPerTransfer,
      durationUs,
      numGpus,
      label,
    };
    builder.push({
      type: 'MEMORY_MOVE',
      title: `${label} · AllGather 第 ${step + 1}/${steps} 步`,
      what: `环上 ${numGpus} 个 GPU 同时把各自已完整归约的 chunk 发给下一个邻居（每 GPU 发送 ${formatBytes(bytesPerTransfer)}）。第 ${step + 1} 步完成后，更多 GPU 拿到了完整数据。`,
      why: `AllGather 是 Ring AllReduce 的后半段：把 ReduceScatter 分散在各 GPU 上的 1/${numGpus} 归约结果广播给所有人，最终每个 GPU 都持有完整归约结果。通信量 (N-1)/N × S。`,
      operator,
      source: 'HBM',
      destination: 'HBM',
      metadata: { comm: meta },
    });
    totalUs += durationUs;
  }
  return totalUs;
}

/**
 * 向 builder 追加一段 AllReduce 事件流。
 * AllReduce = ReduceScatter + AllGather（NCCL Ring 实现）。
 * 返回总耗时估算（µs）。
 */
export function emitAllReduce(builder: EventBuilder, config: EmitCollectiveConfig): number {
  const { numGpus, totalBytes, operator, label } = config;
  // Ring AllReduce 总通信量 2(N-1)/N × S，分两个子阶段
  builder.push({
    type: 'SYNC',
    title: `${label} · AllReduce 开始（Ring）`,
    what: `${numGpus} 个 GPU 发起 AllReduce，总数据量 ${formatBytes(totalBytes)}。Ring 实现把它分解为 ReduceScatter + AllGather 两个阶段，共 ${2 * (numGpus - 1)} 步。`,
    why: `AllReduce 让每个 GPU 都拿到到"所有 GPU 数据的归约结果"（如梯度求和）。Ring 算法的通信量 2(N-1)/N × S 与 GPU 数几乎无关，是大规模训练的标准做法（NCCL）。`,
    operator,
    metadata: {
      comm: {
        collective: 'allreduce',
        phase: 'single',
        ringStep: 0,
        totalRingSteps: 2 * (numGpus - 1),
        transfers: [],
        bytesPerTransfer: totalBytes / numGpus,
        durationUs: 0,
        numGpus,
        label,
      } satisfies CommMeta,
    },
  });

  const rsUs = emitReduceScatter(builder, config);
  const agUs = emitAllGather(builder, config);
  return rsUs + agUs;
}

/**
 * 向 builder 追加一次点对点（P2P）传输事件（流水线并行的激活/梯度传递）。
 * 返回耗时估算（µs）。
 */
export function emitP2P(
  builder: EventBuilder,
  config: {
    from: number;
    to: number;
    bytes: number;
    link: CommLinkSpec;
    operator: string;
    label: string;
    what: string;
    why: string;
  },
): number {
  const durationUs = ringStepDurationUs(config.bytes, config.link);
  const meta: CommMeta = {
    collective: 'p2p',
    phase: 'single',
    ringStep: 0,
    totalRingSteps: 0,
    transfers: [{ from: config.from, to: config.to, chunk: 0 }],
    bytesPerTransfer: config.bytes,
    durationUs,
    numGpus: 2,
    label: config.label,
  };
  builder.push({
    type: 'MEMORY_MOVE',
    title: config.label,
    what: config.what,
    why: config.why,
    operator: config.operator,
    source: 'HBM',
    destination: 'HBM',
    metadata: { comm: meta },
  });
  return durationUs;
}

/** 字节数格式化（教学展示用） */
export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes.toFixed(0)} B`;
}
