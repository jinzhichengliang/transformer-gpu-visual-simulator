/**
 * Nsight 风格 GPU trace 解析器（V0.5，实施手册 §23）。
 *
 * 架构路径：PyTorch/CUDA → Nsight Systems → Trace Parser（本模块）→ TVIR → Visualizer
 *
 * 这是 TVIR 架构验收点（手册 §9）的实现：把 GEMM simulator 换成 trace parser，
 * Playback 与 UI 零改动——因为 parser 的输出与仿真引擎一样，都是 TVIRTrace。
 *
 * 本模块不 import React、不 import Simulation 逻辑（仅复用事件编号基础设施）。
 */

import type { TVIRTrace, TVIREvent } from '../tvir/types';
import { createEventBuilder } from '../simulation/eventBuilder';

/**
 * kernel 级实测性能指标（V0.6，全部可选）。
 *
 * 数据来源约定：Nsight Systems 主要提供时间轴；这些指标通常来自
 * Nsight Compute 或 profiler 的附加采集。未提供的字段解析后保持缺失，
 * PerfPanel 如实显示 N/A，绝不臆测填充（CONCEPTS.md 规则 27）。
 */
export interface NsightKernelMetrics {
  /** Tensor Core 活跃周期占比（0-100，%） */
  tensorCoreUtilization?: number | undefined;
  /** L2 缓存命中率（0-100，%） */
  l2HitRate?: number | undefined;
  /** 实测显存带宽（GB/s） */
  memoryBandwidthGBps?: number | undefined;
  /** Occupancy：驻留 warp 数占 SM 上限的比例（0-100，%） */
  occupancy?: number | undefined;
  /** 算术强度（FLOP/byte） */
  arithmeticIntensity?: number | undefined;
}

const METRIC_FIELDS = [
  'tensorCoreUtilization',
  'l2HitRate',
  'memoryBandwidthGBps',
  'occupancy',
  'arithmeticIntensity',
] as const;

/** 单个 kernel 记录（Nsight Systems 导出格式的教学子集） */
export interface NsightKernelRecord {
  /** kernel 名（如 "ampere_sgemm_128x64_nn"） */
  name: string;
  /** 相对 trace 起点的开始时间（纳秒） */
  startNs: number;
  /** 执行时长（纳秒） */
  durationNs: number;
  /** Grid 维度 [x, y, z]（可选） */
  grid?: [number, number, number] | undefined;
  /** Block 维度 [x, y, z]（可选） */
  block?: [number, number, number] | undefined;
  /** 可选：该 kernel 对应的算子名（与编译知识表登记名一致） */
  operator?: string | undefined;
  /** 可选：kernel 级实测性能指标（V0.6；缺失时 UI 显示 N/A） */
  metrics?: NsightKernelMetrics | undefined;
}

/** Nsight 风格 trace 文件的顶层结构 */
export interface NsightTraceFile {
  meta?: {
    tool?: string | undefined;
    gpu?: string | undefined;
    capturedAt?: string | undefined;
    command?: string | undefined;
    /** GPU 的 SM 数量（用于 GPU View 渲染） */
    smCount?: number | undefined;
  } | undefined;
  /** 是否为示例数据（数值为教学示意，非真实测量） */
  sample?: boolean | undefined;
  kernels: NsightKernelRecord[];
}

export type ParseResult =
  | { ok: true; trace: TVIRTrace }
  | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function formatUs(ns: number): string {
  return (ns / 1000).toFixed(1);
}

/** 校验并解析未知输入为 NsightTraceFile；失败时返回中文错误说明 */
function validateTraceFile(input: unknown): { file: NsightTraceFile } | { error: string } {
  if (!isRecord(input)) {
    return { error: 'JSON 顶层必须是对象（参考格式见 samples/ 目录）' };
  }

  const kernels = input.kernels;
  if (!Array.isArray(kernels)) {
    return { error: '缺少 kernels 数组：真实 trace 必须包含 kernel 执行记录列表' };
  }
  if (kernels.length === 0) {
    return { error: 'kernels 数组为空：trace 中没有任何 kernel 执行记录' };
  }

  for (let i = 0; i < kernels.length; i++) {
    const kernel = kernels[i];
    if (!isRecord(kernel)) {
      return { error: `kernels[${i}] 必须是对象` };
    }
    if (typeof kernel.name !== 'string' || kernel.name.length === 0) {
      return { error: `kernels[${i}].name 必须是非空字符串（kernel 名）` };
    }
    if (typeof kernel.startNs !== 'number' || !Number.isFinite(kernel.startNs) || kernel.startNs < 0) {
      return { error: `kernels[${i}].startNs 必须是非负数字（开始时间，纳秒）` };
    }
    if (typeof kernel.durationNs !== 'number' || !Number.isFinite(kernel.durationNs) || kernel.durationNs <= 0) {
      return { error: `kernels[${i}].durationNs 必须是正数（执行时长，纳秒）` };
    }
    for (const dimName of ['grid', 'block'] as const) {
      const dim = kernel[dimName];
      if (dim !== undefined) {
        if (!Array.isArray(dim) || dim.length !== 3 || dim.some((d) => typeof d !== 'number' || d <= 0)) {
          return { error: `kernels[${i}].${dimName} 必须是三个正数的数组 [x, y, z]` };
        }
      }
    }
    // V0.6：kernel 级实测指标（全部可选；提供时必须是非负有限数字）
    if (kernel.metrics !== undefined) {
      if (!isRecord(kernel.metrics)) {
        return { error: `kernels[${i}].metrics 若提供必须是对象` };
      }
      for (const field of METRIC_FIELDS) {
        const value = kernel.metrics[field];
        if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0)) {
          return { error: `kernels[${i}].metrics.${field} 若提供必须是非负数字` };
        }
      }
    }
  }

  if (input.meta !== undefined && !isRecord(input.meta)) {
    return { error: 'meta 若提供必须是对象' };
  }
  const meta = input.meta as NsightTraceFile['meta'];
  if (meta?.smCount !== undefined && (typeof meta.smCount !== 'number' || meta.smCount <= 0)) {
    return { error: 'meta.smCount 若提供必须是正数' };
  }

  return {
    file: {
      ...(meta !== undefined ? { meta } : {}),
      ...(typeof input.sample === 'boolean' ? { sample: input.sample } : {}),
      kernels: kernels as NsightKernelRecord[],
    },
  };
}

/**
 * 把 Nsight 风格 trace 解析为 TVIRTrace。
 *
 * 每个真实 kernel 生成两类事件：
 *   - KERNEL_LAUNCH：kernel 开始执行（携带实测时长、grid/block 配置）
 *   - BLOCK_SCHEDULE × N：grid 中的 Block 分发到各 SM（每个 SM 一条）
 *
 * 真实 trace 中没有 Block 内部的访存/MMA 细节（profiler 不采集），
 * 因此不生成 MEMORY/MMA 事件——这是诚实的表达，不做臆测。
 */
export function parseNsightTrace(input: unknown): ParseResult {
  const validated = validateTraceFile(input);
  if ('error' in validated) {
    return { ok: false, error: validated.error };
  }
  const { file } = validated;

  const isSample = file.sample === true;
  const smCount = Math.max(1, Math.floor(file.meta?.smCount ?? 4));
  const builder = createEventBuilder();

  // 按开始时间排序（profiler 输出通常有序，防御性处理）
  const kernels = [...file.kernels].sort((a, b) => a.startNs - b.startNs);

  const dataSource = isSample
    ? '内置示例 trace（数值为教学示意值，非真实测量）'
    : '真实 GPU profiler trace（Nsight Systems 格式）';
  const measureNote = isSample
    ? '注意：这是示例数据，时长仅用于演示时间轴视图，不代表任何真实硬件。'
    : '时间轴与时长为 Measured（实测）数据，来自真实 GPU。';
  const durationLabel = isSample ? '示意时长' : '实测时长';

  for (const kernel of kernels) {
    const grid = kernel.grid ?? [1, 1, 1];
    const block = kernel.block ?? [128, 1, 1];
    const totalBlocks = grid[0] * grid[1] * grid[2];
    const threadsPerBlock = block[0] * block[1] * block[2];

    const kernelInfo = {
      startNs: kernel.startNs,
      durationNs: kernel.durationNs,
      startUs: kernel.startNs / 1000,
      durationUs: kernel.durationNs / 1000,
      gridX: grid[0],
      gridY: grid[1],
      gridZ: grid[2],
      blockX: block[0],
      blockY: block[1],
      blockZ: block[2],
      totalBlocks,
      threadsPerBlock,
      ...(kernel.metrics !== undefined ? { metrics: kernel.metrics } : {}),
    };

    builder.push({
      type: 'KERNEL_LAUNCH',
      title: `Kernel: ${kernel.name}`,
      what: `真实 GPU 执行 kernel「${kernel.name}」：grid ${grid[0]}×${grid[1]}×${grid[2]}（共 ${totalBlocks} 个 Block），block ${block[0]}×${block[1]}×${block[2]}（${threadsPerBlock} 线程/Block），${durationLabel} ${formatUs(kernel.durationNs)} µs（起始偏移 ${formatUs(kernel.startNs)} µs）。`,
      why: `本事件由 ${dataSource} 转换而来：kernel 名、时间轴与 grid 配置均为原始记录。${measureNote} 与仿真模式共用同一套 TVIR → Playback → UI 管线——这正是"数据源可替换、前端不改动"的架构设计。`,
      ...(kernel.operator !== undefined ? { operator: kernel.operator } : {}),
      kernel: kernel.name,
      metadata: { provenance: 'real-trace', kernelInfo },
    });

    // Block 到 SM 的分发：每个接收到 Block 的 SM 生成一条事件
    const blocksPerSm = Math.ceil(totalBlocks / smCount);
    for (let sm = 0; sm < smCount; sm++) {
      const firstBlock = sm * blocksPerSm;
      if (firstBlock >= totalBlocks) break;
      const count = Math.min(blocksPerSm, totalBlocks - firstBlock);

      builder.push({
        type: 'BLOCK_SCHEDULE',
        title: `「${kernel.name}」分发 Block → SM ${sm}`,
        what: `kernel「${kernel.name}」的 ${totalBlocks} 个 Thread Block 由 GPU 调度器分发：SM ${sm} 接收 ${count} 个（首个为 Block ${firstBlock}）。`,
        why: '真实硬件遵循与教学仿真相同的调度规则：Thread Block 是调度基本单位，一个 Block 只会被调度到一个 SM 上运行。',
        ...(kernel.operator !== undefined ? { operator: kernel.operator } : {}),
        kernel: kernel.name,
        block: firstBlock,
        sm,
        metadata: { provenance: 'real-trace', kernelInfo, blockCount: count },
      });
    }
  }

  const gpuLabel = file.meta?.gpu ?? '未知 GPU';
  const toolLabel = file.meta?.tool ?? 'profiler';

  return {
    ok: true,
    trace: {
      description: `Real Trace: ${kernels.length} kernels on ${gpuLabel}（来源：${toolLabel}${file.meta?.capturedAt ? `，采集于 ${file.meta.capturedAt}` : ''}）${isSample ? ' · 示例数据（教学示意值，非实测）' : ' · Measured 数据'}`,
      events: builder.events,
      provenance: 'real-trace',
      ...(isSample ? { isSample: true } : {}),
    },
  };
}

/**
 * 从真实 trace 推导 GPU View 需要的硬件参数。
 * - numSM：BLOCK_SCHEDULE 事件中出现的最大 SM 编号 + 1（回退 4）
 * - warpsPerBlock：首个 kernel 的 Block 线程数 / 32（回退 4）
 */
export function inferRealTraceHardware(trace: TVIRTrace): {
  numSM: number;
  warpsPerBlock: number;
} {
  let maxSm = -1;
  let warpsPerBlock = 4;
  let warpsResolved = false;

  for (const event of trace.events) {
    if (event.type === 'BLOCK_SCHEDULE' && event.sm !== undefined) {
      maxSm = Math.max(maxSm, event.sm);
    }
    if (!warpsResolved && event.type === 'KERNEL_LAUNCH') {
      const info = (event.metadata as { kernelInfo?: { threadsPerBlock?: number } } | undefined)
        ?.kernelInfo;
      if (info?.threadsPerBlock !== undefined && info.threadsPerBlock > 0) {
        warpsPerBlock = Math.max(1, Math.round(info.threadsPerBlock / 32));
        warpsResolved = true;
      }
    }
  }

  return { numSM: maxSm >= 0 ? maxSm + 1 : 4, warpsPerBlock };
}

/** 判断事件是否来自真实 trace（供 UI 做数据可信度标注） */
export function isRealTraceEvent(event: TVIREvent | null): boolean {
  const meta = event?.metadata as { provenance?: string } | undefined;
  return meta?.provenance === 'real-trace';
}
