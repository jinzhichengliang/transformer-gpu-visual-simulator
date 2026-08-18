/**
 * 性能分析层（V0.6，实施手册 §24）。
 *
 * 六项指标：Kernel duration / Tensor Core utilization / Memory bandwidth /
 * L2 hit rate / Occupancy / Arithmetic intensity。
 *
 * 数据可信度铁律（手册 §24 + CONCEPTS.md 规则 8/26/30）：
 * 每个指标必须携带 source 标注（measured / estimated / simulated / unavailable），
 * UI 绝不能把三者混为一谈；示例 trace 的任何数值都不得标成 Measured。
 *
 * 本模块是对 TVIR 的纯消费（投影 + 教学屋顶线模型）：
 * 不 import React，不 import Simulation 引擎内部实现，只读事件公开字段。
 */

import type { TVIREvent, TVIRTrace } from '../tvir/types';
import type { NsightKernelMetrics } from '../realtrace';

/** 指标数据来源（手册 §24 三分法 + 不可用） */
export type MetricSource = 'measured' | 'estimated' | 'simulated' | 'unavailable';

export const METRIC_KEYS = ['duration', 'tcUtil', 'bandwidth', 'l2Hit', 'occupancy', 'ai'] as const;
export type MetricKey = (typeof METRIC_KEYS)[number];

export interface PerfMetric {
  key: MetricKey;
  /** 指标名（英文，与手册一致） */
  label: string;
  value: number | null;
  unit: string;
  /** 数据可信度标注（UI 必须原样展示，不得改写） */
  source: MetricSource;
  /** 教学说明：数值如何得来 / 为什么不可用 */
  note: string;
}

/** trace 级数据类别（驱动面板顶部徽标） */
export type DataClass = 'simulated' | 'measured' | 'sample';

export interface OperatorTimeShare {
  operator: string;
  totalUs: number;
  kernelCount: number;
}

export interface PerfReport {
  /** 当前指标的作用域描述（如某算子、某 kernel、整条 trace） */
  scopeLabel: string;
  dataClass: DataClass;
  /** 固定顺序的六项指标 */
  metrics: PerfMetric[];
  /** 按算子聚合的耗时分布（仅真实 trace 有数据） */
  breakdown: OperatorTimeShare[];
  /** trace 级汇总（仅真实 trace 有数据） */
  totals: { kernelCount: number; totalUs: number } | null;
  /** 面板级教学说明 */
  contextNote: string;
}

/**
 * 教学用假设硬件规格（CONCEPTS.md 已声明为教学简化）。
 * V0.7 起该结构是 Architecture Playground 的可配置参数：
 * 所有屋顶线计算接收 HardwareSpec，不再硬编码常量。
 * 这些数值不代表任何真实 GPU。
 */
export interface HardwareSpec {
  /** SM 数量 */
  smCount: number;
  /** Tensor Core fp16 峰值算力（TFLOPS） */
  tensorCoreTflops: number;
  /** CUDA Core fp32 峰值算力（TFLOPS） */
  cudaCoreTflops: number;
  /** HBM 峰值带宽（GB/s） */
  hbmBandwidthGBps: number;
  /** L2 缓存容量（MB） */
  l2SizeMB: number;
  /** 每 SM 共享内存容量（KB） */
  smemPerSMKB: number;
  /** 每 SM 最大驻留 warp 数 */
  maxWarpsPerSM: number;
  /** 每元素字节数：fp16/bf16 假设（算术强度依赖此假设，不开放给用户修改） */
  bytesPerElement: number;
}

/** 默认教学硬件规格（V0.6 的 ASSUMED_HARDWARE 数值，保持向后兼容） */
export const DEFAULT_HARDWARE_SPEC: HardwareSpec = {
  smCount: 4,
  tensorCoreTflops: 100,
  cudaCoreTflops: 10,
  hbmBandwidthGBps: 2000,
  l2SizeMB: 40,
  smemPerSMKB: 100,
  maxWarpsPerSM: 64,
  bytesPerElement: 2,
};

/** @deprecated 兼容别名：请改用 DEFAULT_HARDWARE_SPEC */
export const ASSUMED_HARDWARE = DEFAULT_HARDWARE_SPEC;

/**
 * SM 数量基准（V0.7）：每 SM 峰值算力 = 滑块算力 ÷ BASE_SM_COUNT。
 * 锚定在默认 SM 数上，保证"滑块算力"在默认配置下等于机器总算力；
 * 用户增加 SM 数时，总算力按比例增长（大负载加速），小负载因 Block
 * 数不足而收益饱和——对齐真实硬件的并行扩展语义。
 */
const BASE_SM_COUNT = DEFAULT_HARDWARE_SPEC.smCount;

const METRIC_LABELS: Record<MetricKey, { label: string; unit: string }> = {
  duration: { label: 'Kernel Duration', unit: 'µs' },
  tcUtil: { label: 'Tensor Core Utilization', unit: '%' },
  bandwidth: { label: 'Memory Bandwidth', unit: 'GB/s' },
  l2Hit: { label: 'L2 Hit Rate', unit: '%' },
  occupancy: { label: 'Occupancy', unit: '%' },
  ai: { label: 'Arithmetic Intensity', unit: 'FLOP/byte' },
};

function metric(
  key: MetricKey,
  value: number | null,
  source: MetricSource,
  note: string,
): PerfMetric {
  return { key, label: METRIC_LABELS[key].label, value, unit: METRIC_LABELS[key].unit, source, note };
}

function unavailable(key: MetricKey, note: string): PerfMetric {
  return metric(key, null, 'unavailable', note);
}

// ---------------------------------------------------------------------------
// 事件元数据读取（只读公开字段）
// ---------------------------------------------------------------------------

interface KernelInfoMeta {
  durationUs?: number;
  metrics?: NsightKernelMetrics;
}

interface GemmMeta {
  M?: number;
  N?: number;
  K?: number;
  tileM?: number;
  tileN?: number;
  tileK?: number;
  warpsPerBlock?: number;
}

function readKernelInfo(event: TVIREvent | null): (KernelInfoMeta & { kernelName?: string | undefined }) | null {
  if (!event) return null;
  const meta = event.metadata as { kernelInfo?: KernelInfoMeta } | undefined;
  const info = meta?.kernelInfo;
  if (!info || info.durationUs === undefined) return null;
  return { ...info, kernelName: event.kernel };
}

function readGemmMeta(event: TVIREvent | null): Required<Omit<GemmMeta, 'warpsPerBlock'>> & { warpsPerBlock: number } | null {
  const meta = event?.metadata as { gemm?: GemmMeta } | undefined;
  const gemm = meta?.gemm;
  if (
    !gemm ||
    typeof gemm.M !== 'number' ||
    typeof gemm.N !== 'number' ||
    typeof gemm.K !== 'number' ||
    typeof gemm.tileM !== 'number' ||
    typeof gemm.tileN !== 'number' ||
    typeof gemm.tileK !== 'number'
  ) {
    return null;
  }
  return {
    M: gemm.M,
    N: gemm.N,
    K: gemm.K,
    tileM: gemm.tileM,
    tileN: gemm.tileN,
    tileK: gemm.tileK,
    warpsPerBlock: typeof gemm.warpsPerBlock === 'number' ? gemm.warpsPerBlock : 4,
  };
}

function readRowsCols(event: TVIREvent | null): { rows: number; cols: number } | null {
  const meta = event?.metadata as { rows?: unknown; cols?: unknown } | undefined;
  if (typeof meta?.rows === 'number' && typeof meta.cols === 'number') {
    return { rows: meta.rows, cols: meta.cols };
  }
  return null;
}

// ---------------------------------------------------------------------------
// 仿真模式：教学屋顶线模型（全部标注 Simulated）
//
// V0.7：所有计算接收 HardwareSpec，不再硬编码常量。
// SM 数量语义（对齐真实硬件）：每 SM 峰值算力固定（= 滑块算力 ÷ 基线 SM 数），
// 实际算力 = 忙碌 SM 数 × 每 SM 峰值；忙碌 SM 数 = min(SM 数, Block 数)。
// Block 数不足时部分 SM 空闲——加 SM 只会让大负载更快，不会让小负载变慢。
// L2 容量通过"复用收益折扣"影响有效访存量（L2 太小则 Tile 复用收益下降）。
// 这些均为教学简化模型（见 CONCEPTS.md），不代表真实硬件行为。
// ---------------------------------------------------------------------------

/** GEMM 屋顶线建模结果（数值，供 PerfPanel 与 Playground 共用） */
export interface GemmModelResult {
  /** 估算时长（µs） */
  durationUs: number;
  /** 瓶颈判定 */
  bound: 'compute' | 'memory';
  /** 算术强度（FLOP/byte，基于最少访存量） */
  ai: number;
  /** Tensor Core 利用率（%） */
  tcUtil: number;
  /** HBM 实际吞吐（GB/s） */
  bandwidth: number;
  /** L2 命中率（%，含 L2 容量折扣） */
  l2Hit: number;
  /** Occupancy（%，warp 槽位与 SMEM 双约束） */
  occupancy: number;
  /** SM 利用率（0-1）：忙碌 SM 数 ÷ 总 SM 数，即 min(1, Block 数 / SM 数) */
  smEfficiency: number;
  /** 有效访存量（字节，含 L2 复用折扣） */
  effectiveBytes: number;
  flops: number;
}

/** 逐元素算子屋顶线建模结果 */
export interface ElementwiseModelResult {
  durationUs: number;
  bound: 'compute' | 'memory';
  ai: number;
  bandwidth: number;
  occupancy: number;
  flops: number;
  bytes: number;
}

type GemmShape = ReturnType<typeof readGemmMeta> & object;

/** GEMM 教学屋顶线建模（纯函数，可被 Playground 复用） */
export function modelGemm(g: GemmShape, hw: HardwareSpec): GemmModelResult {
  const { M, N, K, tileM, tileN, warpsPerBlock } = g;

  const flops = 2 * M * N * K;
  // 最少访存量：A、B 各读一次 + C 写一次（fp16 假设）
  const uniqueBytes = (M * K + K * N + M * N) * hw.bytesPerElement;
  const ai = flops / uniqueBytes;

  // --- L2 复用模型：tiling 带来的重复读，受 L2 容量折扣 ---
  const tilesM = Math.ceil(M / tileM);
  const tilesN = Math.ceil(N / tileN);
  const readElems = tilesN * M * K + tilesM * K * N; // 无 L2 复用时的总读元素
  const hitElems = M * K * (tilesN - 1) + K * N * (tilesM - 1); // 可被复用的元素
  // 在飞 Tile 工作集（近似）：每个并发 Block 持有一组 A/B Tile
  const blocksPerSMSmem = Math.max(
    1,
    Math.floor(
      (hw.smemPerSMKB * 1024) /
        Math.max(1, (tileM * g.tileK + g.tileK * tileN) * hw.bytesPerElement),
    ),
  );
  const blocksByWarps = Math.max(1, Math.floor(hw.maxWarpsPerSM / warpsPerBlock));
  const blocksPerSM = Math.min(blocksPerSMSmem, blocksByWarps);
  const totalBlocks = tilesM * tilesN;
  const concurrentBlocks = Math.min(totalBlocks, hw.smCount * blocksPerSM);
  const workingSetElems = (tileM * g.tileK + g.tileK * tileN) * concurrentBlocks;
  const l2CapacityElems = (hw.l2SizeMB * 1e6) / hw.bytesPerElement;
  const l2Coverage = workingSetElems > 0 ? Math.min(1, l2CapacityElems / workingSetElems) : 1;
  const effectiveHitElems = hitElems * l2Coverage;
  const effectiveReadElems = readElems - effectiveHitElems;
  const effectiveBytes = effectiveReadElems * hw.bytesPerElement + M * N * hw.bytesPerElement;
  const l2Hit = readElems > 0 ? Math.max(0, effectiveHitElems / readElems) * 100 : 0;

  // --- SM 利用率：忙碌 SM 数 ÷ 总 SM 数（Block 不足时部分 SM 空闲） ---
  // 每 SM 峰值算力固定（滑块算力 ÷ 基线 SM 数）；总算力 = 忙碌 SM 数 × 每 SM 峰值。
  // 加 SM 只会让 Block 充足的大负载更快，不会让小负载变慢。
  const busySMs = Math.min(hw.smCount, Math.max(1, Math.ceil(totalBlocks / blocksPerSM)));
  const smEfficiency = hw.smCount > 0 ? busySMs / hw.smCount : 1;
  const perSmTflops = hw.tensorCoreTflops / BASE_SM_COUNT;
  const effectiveTflops = busySMs * perSmTflops;
  const machinePeakTflops = hw.smCount * perSmTflops;

  // --- 屋顶线：计算时间（随忙碌 SM 数扩展）vs 访存时间（HBM 为机器级，固定） ---
  const computeSec = flops / (effectiveTflops * 1e12);
  const memSec = effectiveBytes / (hw.hbmBandwidthGBps * 1e9);
  const durSec = Math.max(computeSec, memSec);
  const durationUs = durSec * 1e6;

  const achievedTflops = flops / durSec / 1e12;
  const tcUtil = Math.min(100, (achievedTflops / machinePeakTflops) * 100);
  const bandwidth = effectiveBytes / durSec / 1e9;

  // Occupancy：warp 槽位与共享内存容量双约束（未计寄存器限制）
  const occupancy = ((blocksPerSM * warpsPerBlock) / hw.maxWarpsPerSM) * 100;

  return {
    durationUs,
    bound: computeSec >= memSec ? 'compute' : 'memory',
    ai,
    tcUtil,
    bandwidth,
    l2Hit,
    occupancy,
    smEfficiency,
    effectiveBytes,
    flops,
  };
}

/** 逐元素算子教学屋顶线建模（纯函数，可被 Playground 复用） */
export function modelElementwise(rows: number, cols: number, hw: HardwareSpec): ElementwiseModelResult {
  // 逐元素算子教学模型：每元素约 2 FLOP，读一次写一次
  const flops = rows * cols * 2;
  const bytes = rows * cols * hw.bytesPerElement * 2;
  const ai = flops / bytes;

  // SM 利用率：行数（Block 数）能填满多少 SM（与 GEMM 同语义）
  const warpsPerBlock = Math.max(1, Math.ceil(cols / 32));
  const blocksPerSM = Math.max(1, Math.floor(hw.maxWarpsPerSM / warpsPerBlock));
  const busySMs = Math.min(hw.smCount, Math.max(1, Math.ceil(rows / blocksPerSM)));
  const perSmTflops = hw.cudaCoreTflops / BASE_SM_COUNT;
  const effectiveTflops = busySMs * perSmTflops;

  const computeSec = flops / (effectiveTflops * 1e12);
  const memSec = bytes / (hw.hbmBandwidthGBps * 1e9);
  const durSec = Math.max(computeSec, memSec);
  const durationUs = durSec * 1e6;
  const bandwidth = bytes / durSec / 1e9;

  const occupancy = Math.min(100, ((blocksPerSM * warpsPerBlock) / hw.maxWarpsPerSM) * 100);

  return { durationUs, bound: computeSec >= memSec ? 'compute' : 'memory', ai, bandwidth, occupancy, flops, bytes };
}

function gemmSimulatedMetrics(g: GemmShape, hw: HardwareSpec): PerfMetric[] {
  const r = modelGemm(g, hw);
  const { M, N, tileM, tileN } = g;
  const tilesM = Math.ceil(M / tileM);
  const tilesN = Math.ceil(N / tileN);
  const boundLabel = r.bound === 'compute' ? '计算密集（compute-bound）' : '访存密集（memory-bound）';

  return [
    metric(
      'duration',
      r.durationUs,
      'simulated',
      `屋顶线模型：max(FLOPs/峰值算力, 有效访存量/HBM带宽)÷并行效率，判定为${boundLabel}。峰值 ${hw.tensorCoreTflops} TFLOPS（fp16）、带宽 ${hw.hbmBandwidthGBps} GB/s、${hw.smCount} SM、L2 ${hw.l2SizeMB} MB。非真实时序。`,
    ),
    metric(
      'tcUtil',
      r.tcUtil,
      'simulated',
      '实际算力 ÷ 峰值算力。计算密集时接近上限，访存密集时偏低。',
    ),
    metric(
      'bandwidth',
      r.bandwidth,
      'simulated',
      `HBM 实际吞吐 = 有效访存量 ÷ 时长；峰值 ${hw.hbmBandwidthGBps} GB/s。`,
    ),
    metric(
      'l2Hit',
      r.l2Hit,
      'simulated',
      `Tile 复用模型：A Tile 被同行 ${tilesN} 个 Block 复用、B Tile 被同列 ${tilesM} 个 Block 复用；命中率受 L2 容量（${hw.l2SizeMB} MB）折扣。`,
    ),
    metric(
      'occupancy',
      r.occupancy,
      'simulated',
      '教学模型：取 warp 槽位与共享内存容量两个约束的较小值；未计寄存器限制。',
    ),
    metric(
      'ai',
      r.ai,
      'estimated',
      `2·M·N·K FLOPs ÷ 最少访存字节（fp16 假设）。AI 越高越偏向计算密集；由形状推导，属估算。`,
    ),
  ];
}

function elementwiseSimulatedMetrics(rows: number, cols: number, hw: HardwareSpec): PerfMetric[] {
  const r = modelElementwise(rows, cols, hw);

  return [
    metric(
      'duration',
      r.durationUs,
      'simulated',
      `屋顶线模型：逐元素算子几乎必为访存密集（memory-bound），时长≈访存量÷带宽÷并行效率。带宽 ${hw.hbmBandwidthGBps} GB/s。非真实时序。`,
    ),
    metric(
      'tcUtil',
      0,
      'simulated',
      '该算子不使用 Tensor Core（逐元素计算在 CUDA Core 完成），利用率为 0。',
    ),
    metric(
      'bandwidth',
      r.bandwidth,
      'simulated',
      `逐元素算子逼近带宽上限：读+写各一次。峰值 ${hw.hbmBandwidthGBps} GB/s。`,
    ),
    metric(
      'l2Hit',
      0,
      'simulated',
      '逐元素算子没有数据复用：每个元素读一次、写一次，L2 命中率为 0（理想模型）。',
    ),
    metric(
      'occupancy',
      r.occupancy,
      'simulated',
      '教学模型：逐元素 kernel 通常不受共享内存限制，occupancy 主要由 warp 槽位决定。',
    ),
    metric(
      'ai',
      r.ai,
      'estimated',
      '每元素约 2 FLOP ÷ 4 字节读写（fp16 假设）≈ 0.5 FLOP/byte，远低于 GEMM——这就是"访存密集"的含义。',
    ),
  ];
}

function simulationReport(event: TVIREvent | null, hw: HardwareSpec): PerfReport {
  const gemm = readGemmMeta(event);
  const rowsCols = gemm ? null : readRowsCols(event);

  let scopeLabel: string;
  let metrics: PerfMetric[];

  if (gemm) {
    scopeLabel = `${event?.operator ?? 'GEMM'}（GEMM · 教学屋顶线模型）`;
    metrics = gemmSimulatedMetrics(gemm, hw);
  } else if (rowsCols) {
    scopeLabel = `${event?.operator ?? '逐元素算子'}（逐元素/归约 · 教学屋顶线模型）`;
    metrics = elementwiseSimulatedMetrics(rowsCols.rows, rowsCols.cols, hw);
  } else {
    scopeLabel = '未选中具体算子';
    metrics = [
      unavailable('duration', '导航到具体算子事件（如 MMA、MEMORY_LOAD、KERNEL_LAUNCH）后显示教学估算值。'),
      unavailable('tcUtil', '导航到具体算子事件后显示教学估算值。'),
      unavailable('bandwidth', '导航到具体算子事件后显示教学估算值。'),
      unavailable('l2Hit', '导航到具体算子事件后显示教学估算值。'),
      unavailable('occupancy', '导航到具体算子事件后显示教学估算值。'),
      unavailable('ai', '导航到具体算子事件后显示教学估算值。'),
    ];
  }

  return {
    scopeLabel,
    dataClass: 'simulated',
    metrics,
    breakdown: [],
    totals: null,
    contextNote:
      '仿真模式没有真实时序：以下数值由教学屋顶线模型 + 假设硬件参数估算（见各项说明），仅用于理解"计算密集 vs 访存密集"，不代表真实性能。',
  };
}

// ---------------------------------------------------------------------------
// 真实 trace 模式：Measured（非示例）/ 示例数据绝不标 Measured
// ---------------------------------------------------------------------------

function realKernelMetrics(info: KernelInfoMeta, isSample: boolean): PerfMetric[] {
  const source: MetricSource = isSample ? 'simulated' : 'measured';
  const notePrefix = isSample ? '示例数据（教学示意值，非实测）。' : '来自 trace 的实测值。';
  const na = '需 Nsight Compute 采集，本 trace 未提供该指标。';

  return [
    metric('duration', info.durationUs ?? null, source, `${notePrefix}kernel 执行时长（相对 trace 起点的时间轴见 Kernel Timeline）。`),
    info.metrics?.tensorCoreUtilization !== undefined
      ? metric('tcUtil', info.metrics.tensorCoreUtilization, source, `${notePrefix}Tensor Core 活跃周期占比。`)
      : unavailable('tcUtil', na),
    info.metrics?.memoryBandwidthGBps !== undefined
      ? metric('bandwidth', info.metrics.memoryBandwidthGBps, source, `${notePrefix}实测显存带宽。`)
      : unavailable('bandwidth', na),
    info.metrics?.l2HitRate !== undefined
      ? metric('l2Hit', info.metrics.l2HitRate, source, `${notePrefix}L2 缓存命中率。`)
      : unavailable('l2Hit', na),
    info.metrics?.occupancy !== undefined
      ? metric('occupancy', info.metrics.occupancy, source, `${notePrefix}实测 occupancy（驻留 warp 占比）。`)
      : unavailable('occupancy', na),
    info.metrics?.arithmeticIntensity !== undefined
      ? metric('ai', info.metrics.arithmeticIntensity, source, `${notePrefix}实测 FLOPs ÷ 访存字节。`)
      : unavailable('ai', na),
  ];
}

function realTraceReport(trace: TVIRTrace, event: TVIREvent | null): PerfReport {
  const isSample = trace.isSample === true;
  const dataClass: DataClass = isSample ? 'sample' : 'measured';

  // trace 级汇总与算子耗时分布
  const totals = { kernelCount: 0, totalUs: 0 };
  const byOperator = new Map<string, OperatorTimeShare>();
  for (const e of trace.events) {
    if (e.type !== 'KERNEL_LAUNCH') continue;
    const meta = e.metadata as { kernelInfo?: KernelInfoMeta } | undefined;
    const dur = meta?.kernelInfo?.durationUs;
    if (dur === undefined) continue;
    totals.kernelCount += 1;
    totals.totalUs += dur;
    const key = e.operator ?? e.kernel ?? 'unknown';
    const share = byOperator.get(key);
    if (share) {
      share.totalUs += dur;
      share.kernelCount += 1;
    } else {
      byOperator.set(key, { operator: key, totalUs: dur, kernelCount: 1 });
    }
  }
  const breakdown = [...byOperator.values()].sort((a, b) => b.totalUs - a.totalUs);

  // 当前事件是否为某个 kernel 作用域
  const info = readKernelInfo(event);
  if (info) {
    return {
      scopeLabel: `${info.kernelName ?? 'kernel'}（单 kernel）`,
      dataClass,
      metrics: realKernelMetrics(info, isSample),
      breakdown,
      totals,
      contextNote: isSample
        ? '当前为内置示例 trace：所有数值是教学示意值，不是任何真实 GPU 的实测结果。'
        : '当前为真实 GPU trace：kernel 时长等指标为 Measured 实测数据。',
    };
  }

  // trace 级作用域
  const measuredSource: MetricSource = isSample ? 'simulated' : 'measured';
  const prefix = isSample ? '示例数据（教学示意值，非实测）。' : '实测值。';
  const drillNote = '在 Kernel Timeline 或时间轴上点击某个 kernel，可查看其单项指标。';

  return {
    scopeLabel: '整条 trace（kernel 级汇总）',
    dataClass,
    metrics: [
      metric('duration', totals.totalUs, measuredSource, `${prefix}全部 ${totals.kernelCount} 个 kernel 的时长总和。${drillNote}`),
      unavailable('tcUtil', `${drillNote} trace 级 Tensor Core 利用率需 Nsight Compute 汇总。`),
      unavailable('bandwidth', `${drillNote} trace 级带宽需 Nsight Compute 汇总。`),
      unavailable('l2Hit', `${drillNote} trace 级 L2 命中率需 Nsight Compute 汇总。`),
      unavailable('occupancy', `${drillNote} trace 级 occupancy 需 Nsight Compute 汇总。`),
      unavailable('ai', `${drillNote} trace 级算术强度需 FLOPs 与访存量（Nsight Compute）。`),
    ],
    breakdown,
    totals,
    contextNote: isSample
      ? '当前为内置示例 trace：所有数值是教学示意值，不是任何真实 GPU 的实测结果。'
      : '当前为真实 GPU trace：时间轴与时长为 Measured 实测数据；未提供的指标如实显示 N/A，不做臆测填充。',
  };
}

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

/**
 * 计算性能报告（纯投影函数，供 UI 通过 useMemo 消费）。
 * 真实 trace → Measured（示例数据除外）；仿真 → Simulated（AI 为 Estimated）。
 *
 * V0.7：`hardware` 参数允许 Architecture Playground 传入用户修改的硬件规格；
 * 省略时使用 DEFAULT_HARDWARE_SPEC（向后兼容）。该参数只影响仿真模式的
 * 屋顶线估算，对真实 trace 的 Measured 数据无任何影响。
 */
export function computePerfReport(
  trace: TVIRTrace,
  event: TVIREvent | null,
  hardware: HardwareSpec = DEFAULT_HARDWARE_SPEC,
): PerfReport {
  if (trace.provenance === 'real-trace') {
    return realTraceReport(trace, event);
  }
  return simulationReport(event, hardware);
}
