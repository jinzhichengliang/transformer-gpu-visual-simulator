/**
 * TVIR — Transformer Visual Intermediate Representation
 *
 * TVIR 是 Simulation 与 Visualization 之间唯一的数据接口（见 ARCHITECTURE.md）。
 * 本模块不得包含 UI 样式、不得 import React。
 */

/** 全部事件类型（V0.1 共 12 种，来自实施手册 §4） */
export const TVIR_EVENT_TYPES = [
  'GEMM_START',
  'TILE_CREATE',
  'KERNEL_LAUNCH',
  'BLOCK_SCHEDULE',
  'WARP_SCHEDULE',
  'MEMORY_LOAD',
  'MEMORY_MOVE',
  'SYNC',
  'MMA',
  'ACCUMULATE',
  'MEMORY_STORE',
  'GEMM_END',
] as const;

export type TVIREventType = (typeof TVIR_EVENT_TYPES)[number];

/** GPU 内存层级（教学视图，见 CONCEPTS.md 中的简化声明） */
export const MEMORY_LEVELS = [
  'HBM',
  'L2',
  'L1',
  'SHARED_MEMORY',
  'REGISTER',
] as const;

export type MemoryLevel = (typeof MEMORY_LEVELS)[number];

/** 计算单元类型 */
export const COMPUTE_UNITS = ['SM', 'WARP', 'TENSOR_CORE'] as const;

export type ComputeUnit = (typeof COMPUTE_UNITS)[number];

/** 张量引用（A / B / C 等逻辑张量） */
export interface TensorRef {
  /** 张量名，如 "A"、"B"、"C" */
  name: string;
  rows: number;
  cols: number;
}

/** Tile 引用，如 "A[0,1]" 表示 A 矩阵第 0 行第 1 列的 Tile */
export interface TileRef {
  tensor: string;
  tileRow: number;
  tileCol: number;
  /** 可读标识，如 "A[0,1]" */
  label: string;
}

/** TVIR 事件（实施手册 §4 定义） */
export interface TVIREvent {
  id: string;
  step: number;
  type: TVIREventType;
  /** 简短标题，如 "Load A Tile" */
  title: string;
  /** 当前发生了什么（教学解释） */
  what: string;
  /** 为什么这么做（教学解释） */
  why: string;

  /** 可选：算子名（V0.2 起 Attention 会用到） */
  operator?: string;
  /** 可选：Kernel 名 */
  kernel?: string;
  /** 可选：Block 编号 */
  block?: number;
  /** 可选：Warp 编号 */
  warp?: number;
  /** 可选：SM 编号 */
  sm?: number;
  /** 可选：数据来源内存层级 */
  source?: MemoryLevel;
  /** 可选：数据目标内存层级 */
  destination?: MemoryLevel;
  /** 可选：涉及的张量名 */
  tensor?: string;
  /** 可选：涉及的 Tile */
  tile?: TileRef;
  /** 可选：附加元数据 */
  metadata?: Record<string, unknown>;
}

/**
 * trace 的数据来源（V0.5）。
 * - 'simulation'：教学仿真引擎生成（Simulated 数据）
 * - 'real-trace'：来自真实 GPU profiler（如 Nsight Systems）的 trace（Measured 数据）
 * 该字段是可选的：不提供时按 V0.1-V0.4 的既有行为处理（向后兼容）。
 */
export type TVIRProvenance = 'simulation' | 'real-trace';

/** 一条完整的 TVIR trace */
export interface TVIRTrace {
  /** 生成该 trace 的配置描述（供 UI 展示，非模拟状态） */
  description: string;
  events: TVIREvent[];
  /** 数据来源（V0.5，可选）：决定 UI 的数据可信度标注（Simulated / Measured） */
  provenance?: TVIRProvenance | undefined;
  /**
   * 是否为示例数据（V0.5，可选）。
   * 内置示例 trace 的数值是教学示意值（非真实测量），必须显式标注，
   * 绝不能标成 Measured（CONCEPTS.md 规则 8）。
   */
  isSample?: boolean | undefined;
}
