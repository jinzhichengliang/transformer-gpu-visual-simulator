/**
 * Semantic Zoom 投影层（Sprint 8, Task G1-G5）。
 *
 * 任务书要求：完整大模型不能把每一个 warp 都直接展开，
 * 必须引入 Semantic Zoom —— 同一个 TVIR Event 在不同 zoom level
 * 显示不同粒度，但 event id / model context / execution position 保持一致。
 *
 * 五级：
 *   L1 Model    → Embedding / Layers / LM Head（阶段粒度）
 *   L2 Layer    → Attention / MoE / Norm / Residual（层内组件粒度）
 *   L3 Operator → QKV / QK / Softmax / AV / Router / Expert GEMM（算子粒度）
 *   L4 Kernel   → Kernel / Grid / Block（内核粒度）
 *   L5 GPU      → SM / Warp / Memory / Tensor Core（硬件粒度）
 *
 * 架构约束：
 *   - 纯投影，不 import React，不新增事件类型；
 *   - 旧事件（无 metadata.model）在 L1 降级为 null，其余级别仍可投影。
 */

import type { TVIREvent } from '../tvir/types';
import { projectModelContext, type ModelExecutionContext } from '../model/modelContext';

/** 缩放级别（L1 最粗 → L5 最细） */
export type ZoomLevel = 'model' | 'layer' | 'operator' | 'kernel' | 'gpu';

/** 缩放级别的有序列表（用于 UI 步进切换） */
export const ZOOM_LEVELS: ZoomLevel[] = ['model', 'layer', 'operator', 'kernel', 'gpu'];

/** 每个级别的显示标签 */
export const ZOOM_LEVEL_LABELS: Record<ZoomLevel, string> = {
  model: 'Model',
  layer: 'Layer',
  operator: 'Operator',
  kernel: 'Kernel',
  gpu: 'GPU',
};

/** 缩放下当前事件应聚焦的内容（不同级别粒度不同，锚点一致） */
export interface ZoomFocus {
  level: ZoomLevel;
  /** 该级别的主显示内容 */
  primary: string;
  /** 该级别的补充细节（粒度随级别加深） */
  details: string[];
  /** 执行位置锚点（所有级别保持一致，证明锚点不随缩放漂移） */
  anchor: {
    step: number;
    type: string;
    modelContext: ModelExecutionContext | null;
  };
}

/** GEMM 元数据（与 gemmPrimitives 写入的结构对齐） */
interface GemmMeta {
  M?: number;
  N?: number;
  K?: number;
  tileM?: number;
  tileN?: number;
  tilesM?: number;
  tilesN?: number;
  tilesK?: number;
  numBlocks?: number;
  warpsPerBlock?: number;
  left?: string;
  right?: string;
  out?: string;
}

function readGemmMeta(event: TVIREvent): GemmMeta | null {
  const g = event.metadata?.gemm as GemmMeta | undefined;
  return g ?? null;
}

/** L1 Model：Embedding / Layers / LM Head */
function focusModel(event: TVIREvent, ctx: ModelExecutionContext | null): ZoomFocus {
  const details: string[] = [];
  let primary = 'Layers';
  if (ctx) {
    if (ctx.layerType === 'embedding') primary = 'Embedding';
    else if (ctx.layerType === 'lm_head') primary = 'LM Head';
    else primary = `Layers (${ctx.phase === 'decode' ? 'Decode' : 'Prefill'})`;
    details.push(`阶段：${ctx.phase}${ctx.decodeStep !== undefined ? ` Step ${ctx.decodeStep}` : ''}`);
  }
  return {
    level: 'model',
    primary,
    details,
    anchor: { step: event.step, type: event.type, modelContext: ctx },
  };
}

/** L2 Layer：Attention / MoE / Norm / Residual */
function focusLayer(event: TVIREvent, ctx: ModelExecutionContext | null): ZoomFocus {
  const details: string[] = [];
  let primary = ctx ? layerComponentDisplay(ctx.layerType) : 'Layers';
  if (ctx && ctx.layerIndex >= 0) details.push(`Layer ${ctx.layerIndex}`);
  const gemm = readGemmMeta(event);
  if (gemm?.M !== undefined) details.push(`激活形状：${gemm.M}×${gemm.N ?? '?'}${gemm.K !== undefined ? ` × ${gemm.K}` : ''}`);
  return {
    level: 'layer',
    primary,
    details,
    anchor: { step: event.step, type: event.type, modelContext: ctx },
  };
}

/** L3 Operator：QKV / QK / Softmax / AV / Router / Expert GEMM */
function focusOperator(event: TVIREvent, ctx: ModelExecutionContext | null): ZoomFocus {
  const details: string[] = [];
  const primary = operatorZoomDisplay(event, ctx);
  const gemm = readGemmMeta(event);
  if (gemm?.left) details.push(`${gemm.out ?? 'Out'} = ${gemm.left} × ${gemm.right ?? '?'}`);
  if (gemm?.M !== undefined) details.push(`规模：${gemm.M}×${gemm.N ?? '?'}${gemm.K !== undefined ? ` × ${gemm.K}` : ''}`);
  return {
    level: 'operator',
    primary,
    details,
    anchor: { step: event.step, type: event.type, modelContext: ctx },
  };
}

/** 读取 Kernel 级可用的 Block/Warp 规模（兼容 GEMM 元数据与逐元素顶层元数据） */
function readLaunchScale(event: TVIREvent): { numBlocks?: number; warpsPerBlock?: number } {
  const gemm = readGemmMeta(event);
  const md = event.metadata as Record<string, unknown> | undefined;
  const topNumBlocks = typeof md?.numBlocks === 'number' ? (md.numBlocks as number) : undefined;
  const topWarps = typeof md?.warpsPerBlock === 'number' ? (md.warpsPerBlock as number) : undefined;
  const result: { numBlocks?: number; warpsPerBlock?: number } = {};
  const numBlocks = gemm?.numBlocks ?? topNumBlocks;
  const warpsPerBlock = gemm?.warpsPerBlock ?? topWarps;
  if (numBlocks !== undefined) result.numBlocks = numBlocks;
  if (warpsPerBlock !== undefined) result.warpsPerBlock = warpsPerBlock;
  return result;
}

/** L4 Kernel：Kernel / Grid / Block */
function focusKernel(event: TVIREvent, ctx: ModelExecutionContext | null): ZoomFocus {
  const details: string[] = [];
  const gemm = readGemmMeta(event);
  const scale = readLaunchScale(event);
  let primary = event.kernel ?? 'Kernel';

  if (event.type === 'KERNEL_LAUNCH') {
    if (scale.numBlocks !== undefined) details.push(`Grid：${scale.numBlocks} 个 Thread Block`);
    if (scale.warpsPerBlock !== undefined) details.push(`每个 Block：${scale.warpsPerBlock} 个 Warp`);
  } else if (
    event.type === 'BLOCK_SCHEDULE' ||
    event.type === 'WARP_SCHEDULE' ||
    event.type === 'MMA' ||
    event.type === 'ACCUMULATE'
  ) {
    if (typeof event.block === 'number') {
      primary = `Block ${event.block}`;
      details.push(`Grid 中的第 ${event.block + 1} 个 Thread Block`);
    }
    if (gemm?.tilesM !== undefined && gemm.tilesN !== undefined) {
      details.push(`Grid 布局：${gemm.tilesM}×${gemm.tilesN}`);
    }
  }
  return {
    level: 'kernel',
    primary,
    details,
    anchor: { step: event.step, type: event.type, modelContext: ctx },
  };
}

/** L5 GPU：SM / Warp / Memory / Tensor Core */
function focusGpu(event: TVIREvent, ctx: ModelExecutionContext | null): ZoomFocus {
  const details: string[] = [];
  let primary = 'SM / Warp';

  if (typeof event.sm === 'number') {
    primary = `SM ${event.sm}`;
    if (typeof event.warp === 'number') details.push(`Warp ${event.warp}（32 线程）`);
  }

  switch (event.type) {
    case 'MEMORY_MOVE':
      primary = 'Memory';
      details.push('数据在 HBM / L2 / Shared Memory / Register 之间搬运');
      break;
    case 'MEMORY_LOAD':
    case 'MEMORY_STORE':
      primary = 'Memory';
      details.push('内存层级间数据读写');
      break;
    case 'MMA':
    case 'ACCUMULATE':
      details.push('Tensor Core / CUDA Core 执行算术运算');
      break;
    case 'BLOCK_SCHEDULE':
    case 'WARP_SCHEDULE':
      details.push('Block / Warp 调度与并发执行');
      break;
    case 'SYNC':
      details.push('线程间同步（Barrier）');
      break;
    default:
      break;
  }
  return {
    level: 'gpu',
    primary,
    details,
    anchor: { step: event.step, type: event.type, modelContext: ctx },
  };
}

/**
 * 核心投影：同一事件在不同缩放级别产生不同粒度的焦点，
 * 但 anchor（step + type + modelContext）在所有级别保持一致。
 */
export function projectZoomFocus(event: TVIREvent | null, level: ZoomLevel): ZoomFocus | null {
  if (!event) return null;
  const ctx = projectModelContext(event);
  switch (level) {
    case 'model':
      return focusModel(event, ctx);
    case 'layer':
      return focusLayer(event, ctx);
    case 'operator':
      return focusOperator(event, ctx);
    case 'kernel':
      return focusKernel(event, ctx);
    case 'gpu':
      return focusGpu(event, ctx);
  }
}

/** 相邻级别切换（用于"下一步缩放"按钮；返回 null 表示已到边界） */
export function nextZoomLevel(current: ZoomLevel, direction: 'in' | 'out'): ZoomLevel | null {
  const idx = ZOOM_LEVELS.indexOf(current);
  if (direction === 'in') return idx < ZOOM_LEVELS.length - 1 ? ZOOM_LEVELS[idx + 1] ?? null : null;
  return idx > 0 ? ZOOM_LEVELS[idx - 1] ?? null : null;
}

/** L2 层内组件的可读名 */
function layerComponentDisplay(layerType: string): string {
  const map: Record<string, string> = {
    attention: 'Attention',
    moe: 'MoE',
    ffn: 'FFN',
    norm: 'Norm',
    residual: 'Residual',
    embedding: 'Embedding',
    lm_head: 'LM Head',
  };
  return map[layerType] ?? layerType;
}

/** L3 算子的可读名（从事件 operator / operatorType 推导） */
function operatorZoomDisplay(event: TVIREvent, ctx: ModelExecutionContext | null): string {
  const op = event.operator ?? '';
  // 优先用事件自身的算子名（已含 Router / Expert GEMM / QKV 等语义）
  if (op) {
    if (/router/i.test(op)) return 'Router';
    if (/expert/i.test(op)) return 'Expert GEMM';
    if (/qkv|q.proj|k.proj|v.proj/i.test(op)) return 'QKV';
    if (/qk|attention/i.test(op)) return 'QK / Attention';
    if (/softmax/i.test(op)) return 'Softmax';
    if (/av|output.proj/i.test(op)) return 'AV';
    return op;
  }
  // 回退到模型上下文的算子类型
  if (ctx) {
    const map: Record<string, string> = {
      attention: 'Attention',
      moe: 'MoE',
      norm: 'Norm',
      residual: 'Residual',
      embedding: 'Embedding',
      lm_head: 'LM Head',
    };
    return map[ctx.operatorType] ?? ctx.operatorType;
  }
  return event.type;
}
