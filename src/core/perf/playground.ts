/**
 * Architecture Playground 分析层（V0.7，实施手册 §25）。
 *
 * 让用户修改硬件规格（SM count / Tensor Core throughput / Shared Memory size /
 * L2 size / HBM bandwidth），观察"Transformer 快多少、哪些算子几乎不变"，
 * 进入 Roofline / bottleneck thinking。
 *
 * 本层是对仿真 trace 的纯投影：
 *   1. 从 trace 提取工作负载清单（每个算子的形状与出现次数）；
 *   2. 用基线/修改后的 HardwareSpec 分别跑屋顶线模型（modelGemm/modelElementwise）；
 *   3. 输出每个算子的耗时对比、加速比与瓶颈变化。
 * 不 import React，不生成事件。所有数值为 Simulated（教学估算）。
 */

import type { TVIRTrace } from '../tvir/types';
import type { HardwareSpec } from './metrics';
import { modelGemm, modelElementwise } from './metrics';

/** 工作负载项：仿真 trace 中的一个算子（可能多次出现） */
export interface PlaygroundWorkload {
  operator: string;
  /** 出现次数（如 Block trace 中 RMSNorm 出现 2 次） */
  count: number;
  /** GEMM 形状（与 elementwise 二选一） */
  gemm?: {
    M: number;
    N: number;
    K: number;
    tileM: number;
    tileN: number;
    tileK: number;
    warpsPerBlock: number;
  };
  /** 逐元素/归约形状 */
  elementwise?: { rows: number; cols: number };
}

/** 单个算子在基线 vs 修改后硬件下的对比 */
export interface OperatorImpact {
  operator: string;
  count: number;
  /** 基线总耗时（µs，含 count 次） */
  baselineUs: number;
  /** 修改后总耗时（µs，含 count 次） */
  modifiedUs: number;
  /** 加速比（baseline / modified） */
  speedup: number;
  /** 基线瓶颈 */
  baselineBound: 'compute' | 'memory';
  /** 修改后瓶颈 */
  modifiedBound: 'compute' | 'memory';
  /** 瓶颈是否翻转 */
  boundFlipped: boolean;
}

export interface PlaygroundAnalysis {
  workloads: PlaygroundWorkload[];
  impacts: OperatorImpact[];
  /** 总耗时对比 */
  totals: { baselineUs: number; modifiedUs: number; speedup: number };
  /** 教学解读（回答手册 §25 的三个问题） */
  takeaways: string[];
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

/**
 * 从仿真 trace 提取工作负载清单。
 * - GEMM：取每个算子段的 GEMM_START（携带 metadata.gemm）
 * - 逐元素/归约：取每个算子段的 KERNEL_LAUNCH（携带 metadata.rows/cols）
 */
export function extractWorkloads(trace: TVIRTrace): PlaygroundWorkload[] {
  const workloads: PlaygroundWorkload[] = [];
  const seen = new Map<string, PlaygroundWorkload>();

  for (const event of trace.events) {
    if (!event.operator) continue;

    if (event.type === 'GEMM_START') {
      const gemm = (event.metadata as { gemm?: GemmMeta } | undefined)?.gemm;
      if (
        !gemm ||
        typeof gemm.M !== 'number' ||
        typeof gemm.N !== 'number' ||
        typeof gemm.K !== 'number' ||
        typeof gemm.tileM !== 'number' ||
        typeof gemm.tileN !== 'number' ||
        typeof gemm.tileK !== 'number'
      ) {
        continue;
      }
      const existing = seen.get(event.operator);
      if (existing) {
        existing.count += 1;
      } else {
        const item: PlaygroundWorkload = {
          operator: event.operator,
          count: 1,
          gemm: {
            M: gemm.M,
            N: gemm.N,
            K: gemm.K,
            tileM: gemm.tileM,
            tileN: gemm.tileN,
            tileK: gemm.tileK,
            warpsPerBlock: typeof gemm.warpsPerBlock === 'number' ? gemm.warpsPerBlock : 4,
          },
        };
        workloads.push(item);
        seen.set(event.operator, item);
      }
    } else if (event.type === 'KERNEL_LAUNCH') {
      const meta = event.metadata as { rows?: unknown; cols?: unknown } | undefined;
      if (typeof meta?.rows !== 'number' || typeof meta.cols !== 'number') continue;
      const existing = seen.get(event.operator);
      if (existing) {
        existing.count += 1;
      } else {
        const item: PlaygroundWorkload = {
          operator: event.operator,
          count: 1,
          elementwise: { rows: meta.rows, cols: meta.cols },
        };
        workloads.push(item);
        seen.set(event.operator, item);
      }
    }
  }

  return workloads;
}

/** Playground 五个可调参数的滑块范围（教学用） */
export const PLAYGROUND_SLIDER_RANGES = {
  smCount: { min: 1, max: 32, step: 1, label: 'SM Count', unit: '个' },
  tensorCoreTflops: { min: 25, max: 400, step: 25, label: 'Tensor Core Throughput', unit: 'TFLOPS' },
  smemPerSMKB: { min: 16, max: 228, step: 4, label: 'Shared Memory / SM', unit: 'KB' },
  l2SizeMB: { min: 4, max: 128, step: 4, label: 'L2 Size', unit: 'MB' },
  hbmBandwidthGBps: { min: 500, max: 8000, step: 250, label: 'HBM Bandwidth', unit: 'GB/s' },
} as const;

/** 判断两组硬件规格是否有差异（决定是否需要展示对比） */
export function specsDiffer(a: HardwareSpec, b: HardwareSpec): boolean {
  return (
    a.smCount !== b.smCount ||
    a.tensorCoreTflops !== b.tensorCoreTflops ||
    a.smemPerSMKB !== b.smemPerSMKB ||
    a.l2SizeMB !== b.l2SizeMB ||
    a.hbmBandwidthGBps !== b.hbmBandwidthGBps
  );
}

function modelWorkload(
  workload: PlaygroundWorkload,
  hw: HardwareSpec,
): { durationUs: number; bound: 'compute' | 'memory' } {
  if (workload.gemm) {
    const r = modelGemm(workload.gemm, hw);
    return { durationUs: r.durationUs, bound: r.bound };
  }
  if (workload.elementwise) {
    const r = modelElementwise(workload.elementwise.rows, workload.elementwise.cols, hw);
    return { durationUs: r.durationUs, bound: r.bound };
  }
  return { durationUs: 0, bound: 'memory' };
}

/**
 * 对比分析：基线硬件 vs 修改后硬件，逐算子重算屋顶线耗时。
 * 所有数值为 Simulated 教学估算。
 */
export function analyzePlayground(
  trace: TVIRTrace,
  baseline: HardwareSpec,
  modified: HardwareSpec,
): PlaygroundAnalysis {
  const workloads = extractWorkloads(trace);

  const impacts: OperatorImpact[] = [];
  let baselineTotal = 0;
  let modifiedTotal = 0;

  for (const workload of workloads) {
    const base = modelWorkload(workload, baseline);
    const mod = modelWorkload(workload, modified);
    const baselineUs = base.durationUs * workload.count;
    const modifiedUs = mod.durationUs * workload.count;
    baselineTotal += baselineUs;
    modifiedTotal += modifiedUs;
    impacts.push({
      operator: workload.operator,
      count: workload.count,
      baselineUs,
      modifiedUs,
      speedup: modifiedUs > 0 ? baselineUs / modifiedUs : 1,
      baselineBound: base.bound,
      modifiedBound: mod.bound,
      boundFlipped: base.bound !== mod.bound,
    });
  }

  impacts.sort((a, b) => b.baselineUs - a.baselineUs);

  return {
    workloads,
    impacts,
    totals: {
      baselineUs: baselineTotal,
      modifiedUs: modifiedTotal,
      speedup: modifiedTotal > 0 ? baselineTotal / modifiedTotal : 1,
    },
    takeaways: buildTakeaways(baseline, modified, impacts),
  };
}

/** 生成教学解读：回答"谁快了、谁几乎没变、为什么"（Roofline thinking） */
function buildTakeaways(
  baseline: HardwareSpec,
  modified: HardwareSpec,
  impacts: OperatorImpact[],
): string[] {
  const notes: string[] = [];

  const bwRatio = modified.hbmBandwidthGBps / baseline.hbmBandwidthGBps;
  const tcRatio = modified.tensorCoreTflops / baseline.tensorCoreTflops;
  const smRatio = modified.smCount / baseline.smCount;

  if (bwRatio !== 1) {
    const dir = bwRatio > 1 ? '提升' : '降低';
    const memBound = impacts.filter((i) => i.baselineBound === 'memory');
    const computeBound = impacts.filter((i) => i.baselineBound === 'compute');
    if (memBound.length > 0) {
      notes.push(
        `HBM 带宽${dir} ${Math.abs((bwRatio - 1) * 100).toFixed(0)}%：访存密集算子（${memBound
          .slice(0, 3)
          .map((i) => i.operator)
          .join('、')}${memBound.length > 3 ? ' 等' : ''}）时长近似按带宽比例变化——它们受"搬数据的速度"限制，与算力无关。`,
      );
    }
    if (computeBound.length > 0) {
      notes.push(
        `计算密集算子（${computeBound
          .slice(0, 3)
          .map((i) => i.operator)
          .join('、')}${computeBound.length > 3 ? ' 等' : ''}）对带宽变化几乎不敏感：它们的瓶颈在 Tensor Core 算力，数据搬运不是瓶颈。这就是"为什么有些 Operator 几乎没变"。`,
      );
    }
  }

  if (tcRatio !== 1) {
    const dir = tcRatio > 1 ? '提升' : '降低';
    const computeBound = impacts.filter((i) => i.baselineBound === 'compute');
    if (computeBound.length > 0) {
      notes.push(
        `Tensor Core 算力${dir} ${Math.abs((tcRatio - 1) * 100).toFixed(0)}%：计算密集算子（大 GEMM）近似按算力比例加速；访存密集算子几乎不变——它们的瓶颈不在算力。`,
      );
    } else {
      notes.push(
        'Tensor Core 算力变化对当前工作负载影响很小：没有算子处于计算密集状态，瓶颈在别处（带宽或并行度）。',
      );
    }
  }

  if (smRatio !== 1) {
    const dir = smRatio > 1 ? '增多' : '减少';
    notes.push(
      `SM 数量${dir}：提高/降低并行度。当工作负载的 Block 数足以填满更多 SM 时整体加速；Block 数不足时（小矩阵），加 SM 收益递减——这是"并行效率"的作用。`,
    );
  }

  if (modified.l2SizeMB !== baseline.l2SizeMB || modified.smemPerSMKB !== baseline.smemPerSMKB) {
    notes.push(
      'L2 / Shared Memory 变化影响的是"数据复用"与"可驻留 Block 数"：L2 更大使 Tile 复用收益更充分（有效访存量下降），SMEM 更大允许更高 occupancy；对已经复用充分的 GEMM 影响较小。',
    );
  }

  if (notes.length === 0) {
    notes.push('调整左侧任意硬件参数，观察各算子的加速比与瓶颈变化——这就是 Roofline 思维：优化必须对准真正的瓶颈资源。');
  }

  return notes;
}
