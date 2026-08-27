/**
 * Model Context 投影层（Sprint 6, Task E1）。
 *
 * 任务书要求：一个 Event 最终可以定位
 *   模型 → 阶段 → Layer → 算子 → GPU 位置。
 *
 * 架构约束（回归要求）：
 *   - TVIR 12 种事件类型保持唯一词汇表，零新增事件类型；
 *   - 模型上下文通过既有 `metadata.model` 可选字段承载；
 *   - 旧 trace（无 metadata.model）投影返回 null，行为不变；
 *   - 本模块是纯投影，不 import React。
 */

import type { TVIREvent } from '../tvir/types';

/** 事件携带的模型执行上下文（由 Executor 写入 metadata.model） */
export interface ModelExecutionContext {
  modelId: string;
  modelDisplayName: string;
  /** 层序号（-1 表示非层结构：embedding / lm_head / prefill / decode 包裹） */
  layerIndex: number;
  layerType: string;
  operatorType: string;
  /** prefill | decode */
  phase: string;
  /** Decode 步序号（仅 decode 阶段） */
  decodeStep?: number;
  /** Decode 总步数（仅 decode 阶段） */
  decodeTotal?: number;
}

/** 位置面包屑：模型 → 阶段 → 层 → 算子（回答"WHERE"） */
export interface ModelBreadcrumb {
  parts: string[];
}

/**
 * 从当前事件投影模型上下文（纯投影，事件无上下文时返回 null）。
 */
export function projectModelContext(event: TVIREvent | null): ModelExecutionContext | null {
  if (!event?.metadata?.model) return null;
  const m = event.metadata.model as Record<string, unknown>;
  if (typeof m.modelId !== 'string' || typeof m.modelDisplayName !== 'string') return null;
  const ctx: ModelExecutionContext = {
    modelId: m.modelId,
    modelDisplayName: m.modelDisplayName,
    layerIndex: typeof m.layerIndex === 'number' ? m.layerIndex : -1,
    layerType: typeof m.layerType === 'string' ? m.layerType : 'unknown',
    operatorType: typeof m.operatorType === 'string' ? m.operatorType : 'unknown',
    phase: typeof m.phase === 'string' ? m.phase : 'unknown',
  };
  if (typeof m.decodeStep === 'number') ctx.decodeStep = m.decodeStep;
  if (typeof m.decodeTotal === 'number') ctx.decodeTotal = m.decodeTotal;
  return ctx;
}

/** 该事件是否属于模型感知模式（区分旧数据源与新的模型执行计划） */
export function isModelAwareEvent(event: TVIREvent | null): boolean {
  return projectModelContext(event) !== null;
}

/** 阶段的可读标签 */
export function phaseLabel(ctx: ModelExecutionContext): string {
  if (ctx.phase === 'prefill') return 'Prefill';
  if (ctx.phase === 'decode') {
    return ctx.decodeStep !== undefined
      ? `Decode Step ${ctx.decodeStep}/${ctx.decodeTotal ?? '?'}`
      : 'Decode';
  }
  return ctx.phase;
}

/**
 * 位置面包屑（Semantic Zoom 与 Model Overview 共用）：
 *   模型名 → 阶段 → Layer N（或 Embedding/LM Head）→ 算子
 */
export function projectModelBreadcrumb(event: TVIREvent | null): ModelBreadcrumb | null {
  const ctx = projectModelContext(event);
  if (!ctx) return null;
  const parts: string[] = [ctx.modelDisplayName, phaseLabel(ctx)];
  if (ctx.layerIndex >= 0) {
    parts.push(`Layer ${ctx.layerIndex}`);
  } else if (ctx.layerType === 'embedding') {
    parts.push('Embedding');
  } else if (ctx.layerType === 'lm_head') {
    parts.push('LM Head');
  }
  // 包裹事件（prefill/decode 的 START/END）不追加算子名
  if (ctx.operatorType !== 'prefill' && ctx.operatorType !== 'decode') {
    parts.push(operatorDisplay(ctx.operatorType));
  }
  return { parts };
}

/** 算子类型的可读名 */
export function operatorDisplay(operatorType: string): string {
  const map: Record<string, string> = {
    embedding: 'Embedding',
    attention: 'Attention',
    ffn: 'FFN',
    moe: 'MoE',
    lm_head: 'LM Head',
    norm: 'Norm',
    residual: 'Residual',
  };
  return map[operatorType] ?? operatorType;
}
