/**
 * TVIR 投影工具：把事件序列投影为"展示用"的派生状态。
 *
 * 这是对 TVIR 的纯消费（不理解任何 Operator 语义），
 * 不属于 Simulation，也不属于 Playback。
 */

import type { TVIREvent } from './types';

export interface SmDisplayState {
  /** 截至目前已调度到该 SM 的 Block 列表 */
  blocks: number[];
  /** 最近一次被调度的 Warp（相对 Block 内编号） */
  lastWarp: number | null;
  /** 最近一次调度的 Block */
  lastBlock: number | null;
}

export interface OperatorSegment {
  /** 算子名（event.operator 字段；无该字段的事件归入空字符串分组，通常不出现） */
  operator: string;
  /** 该算子在 trace 中的第一个事件索引 */
  startIndex: number;
  /** 该算子的最后一个事件索引（含） */
  endIndex: number;
  /** 事件数量 */
  count: number;
}

/**
 * 把 trace 按 event.operator 字段切分为连续段。
 * 纯消费 TVIR：不理解任何算子语义，只按 operator 值分组。
 */
export function projectOperatorSegments(events: TVIREvent[]): OperatorSegment[] {
  const segments: OperatorSegment[] = [];
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (!event) continue;
    const operator = event.operator ?? '';
    const last = segments[segments.length - 1];
    if (last && last.operator === operator) {
      last.endIndex = i;
      last.count += 1;
    } else {
      segments.push({ operator, startIndex: i, endIndex: i, count: 1 });
    }
  }
  return segments;
}

/**
 * 从当前事件提取 MatrixView 需要的矩阵场景。
 *
 * 纯消费 TVIR：只读取 metadata.gemm（Simulation 写入的标准字段），
 * 不理解任何 Operator 语义。没有 gemm 元数据时返回 null
 * （如 Scale/Mask/Softmax 等逐元素阶段，此时矩阵视图保持上一次场景）。
 */
export interface MatrixScene {
  left: string;
  right: string;
  out: string;
  M: number;
  N: number;
  K: number;
  tileM: number;
  tileN: number;
  tileK: number;
}

export function projectMatrixScene(event: TVIREvent | null): MatrixScene | null {
  const meta = event?.metadata as
    | {
        gemm?: {
          left: string;
          right: string;
          out: string;
          M: number;
          N: number;
          K: number;
          tileM: number;
          tileN: number;
          tileK: number;
        };
      }
    | undefined;
  const gemm = meta?.gemm;
  if (!gemm) return null;
  return {
    left: gemm.left,
    right: gemm.right,
    out: gemm.out,
    M: gemm.M,
    N: gemm.N,
    K: gemm.K,
    tileM: gemm.tileM,
    tileN: gemm.tileN,
    tileK: gemm.tileK,
  };
}

/**
 * Kernel 时间轴投影（V0.5）：从真实 trace 事件提取 kernel 执行时间段。
 *
 * 纯消费 TVIR：只读取 KERNEL_LAUNCH 事件 metadata 中的 kernelInfo
 * （由 realtrace parser 写入的标准字段），不理解任何算子语义。
 */
export interface KernelTimelineSegment {
  /** 该 KERNEL_LAUNCH 事件在 trace 中的索引 */
  eventIndex: number;
  /** kernel 名 */
  kernel: string;
  /** 算子名（可选） */
  operator: string | null;
  /** 开始时间（微秒，相对 trace 起点） */
  startUs: number;
  /** 执行时长（微秒） */
  durationUs: number;
}

interface KernelInfoMeta {
  kernelInfo?: {
    startUs?: number;
    durationUs?: number;
  };
}

export function projectKernelTimeline(events: TVIREvent[]): KernelTimelineSegment[] {
  const segments: KernelTimelineSegment[] = [];
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (!event || event.type !== 'KERNEL_LAUNCH') continue;
    const info = (event.metadata as KernelInfoMeta | undefined)?.kernelInfo;
    if (!info || info.startUs === undefined || info.durationUs === undefined) continue;
    segments.push({
      eventIndex: i,
      kernel: event.kernel ?? 'unknown',
      operator: event.operator ?? null,
      startUs: info.startUs,
      durationUs: info.durationUs,
    });
  }
  return segments;
}

/**
 * 扫描 events[0..currentIndex]，汇总每个 SM 的展示状态。
 */
export function projectSmStates(
  events: TVIREvent[],
  currentIndex: number,
  numSM: number,
): SmDisplayState[] {
  const states: SmDisplayState[] = Array.from({ length: numSM }, () => ({
    blocks: [],
    lastWarp: null,
    lastBlock: null,
  }));

  const limit = Math.min(currentIndex + 1, events.length);
  for (let i = 0; i < limit; i++) {
    const event = events[i];
    if (!event) continue;
    if (event.sm === undefined || event.sm < 0 || event.sm >= numSM) continue;
    const smState = states[event.sm];
    if (!smState) continue;

    if (event.type === 'BLOCK_SCHEDULE' && event.block !== undefined) {
      if (!smState.blocks.includes(event.block)) {
        smState.blocks.push(event.block);
      }
      smState.lastBlock = event.block;
    }
    if (event.type === 'WARP_SCHEDULE' && event.warp !== undefined) {
      smState.lastWarp = event.warp;
      if (event.block !== undefined) {
        smState.lastBlock = event.block;
      }
    }
  }

  return states;
}

/**
 * SASS 指令视图投影（V0.8）。
 *
 * 纯消费 TVIR：只读取 metadata.sass（sass-trace Adapter 写入的标准字段），
 * 不理解任何指令语义。非 SASS 事件不产生投影行。
 */
export interface SassInstructionRow {
  /** 该指令事件在 trace 中的索引 */
  eventIndex: number;
  /** 指令地址 */
  pc: string;
  /** 操作码（含修饰符，如 LDG.E.128） */
  opcode: string;
  /** 操作数（可选） */
  operands: string | null;
  /** 指令类别（Adapter 写入的教学分类） */
  category: string;
  /** Warp 编号 */
  warp: number;
}

export function projectSassInstructions(events: TVIREvent[]): SassInstructionRow[] {
  const rows: SassInstructionRow[] = [];
  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    if (!event) continue;
    const sass = (event.metadata as { sass?: SassInstructionRow & { instructionIndex?: number } } | undefined)
      ?.sass;
    if (!sass || typeof sass.pc !== 'string' || typeof sass.opcode !== 'string') continue;
    rows.push({
      eventIndex: i,
      pc: sass.pc,
      opcode: sass.opcode,
      operands: typeof sass.operands === 'string' ? sass.operands : null,
      category: typeof sass.category === 'string' ? sass.category : 'address-calc',
      warp: typeof sass.warp === 'number' ? sass.warp : 0,
    });
  }
  return rows;
}

/** 判断当前事件是否为 SASS 指令事件（供 UI 决定 InstructionView 是否高亮） */
export function isSassInstructionEvent(event: TVIREvent | null): boolean {
  const meta = event?.metadata as { sass?: { opcode?: string } } | undefined;
  return typeof meta?.sass?.opcode === 'string';
}
