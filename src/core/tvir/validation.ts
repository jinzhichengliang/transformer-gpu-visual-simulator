/**
 * TVIR schema validation（实施手册 §8 要求：编写 schema validation）
 *
 * 纯函数实现，不依赖外部库。UI 与 Simulation 都可以安全调用。
 */

import type { TVIREvent } from './types';
import {
  COMPUTE_UNITS,
  MEMORY_LEVELS,
  TVIR_EVENT_TYPES,
} from './types';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isOptional(
  value: unknown,
  check: (v: unknown) => boolean,
): boolean {
  return value === undefined || check(value);
}

/** 校验单个 TVIR 事件 */
export function validateTVIREvent(event: unknown): ValidationResult {
  const errors: string[] = [];

  if (!isRecord(event)) {
    return { valid: false, errors: ['event 必须是对象'] };
  }

  if (!isString(event.id)) errors.push('id 必须是非空字符串');
  if (!isFiniteNumber(event.step) || event.step < 0) {
    errors.push('step 必须是非负数字');
  }
  if (!isString(event.type) || !TVIR_EVENT_TYPES.includes(event.type as never)) {
    errors.push(`type 必须是 TVIREventType 之一，收到: ${String(event.type)}`);
  }
  if (!isString(event.title)) errors.push('title 必须是非空字符串');
  if (!isString(event.what)) errors.push('what 必须是非空字符串（教学解释）');
  if (!isString(event.why)) errors.push('why 必须是非空字符串（教学解释）');

  if (!isOptional(event.operator, isString)) errors.push('operator 若提供必须是非空字符串');
  if (!isOptional(event.kernel, isString)) errors.push('kernel 若提供必须是非空字符串');
  if (!isOptional(event.block, isFiniteNumber)) errors.push('block 若提供必须是数字');
  if (!isOptional(event.warp, isFiniteNumber)) errors.push('warp 若提供必须是数字');
  if (!isOptional(event.sm, isFiniteNumber)) errors.push('sm 若提供必须是数字');
  if (
    !isOptional(event.source, (v) =>
      isString(v) && MEMORY_LEVELS.includes(v as never),
    )
  ) {
    errors.push('source 若提供必须是 MemoryLevel');
  }
  if (
    !isOptional(event.destination, (v) =>
      isString(v) && MEMORY_LEVELS.includes(v as never),
    )
  ) {
    errors.push('destination 若提供必须是 MemoryLevel');
  }
  if (!isOptional(event.tensor, isString)) errors.push('tensor 若提供必须是非空字符串');

  if (event.tile !== undefined) {
    if (!isRecord(event.tile)) {
      errors.push('tile 若提供必须是对象');
    } else {
      if (!isString(event.tile.tensor)) errors.push('tile.tensor 必须是非空字符串');
      if (!isFiniteNumber(event.tile.tileRow)) errors.push('tile.tileRow 必须是数字');
      if (!isFiniteNumber(event.tile.tileCol)) errors.push('tile.tileCol 必须是数字');
      if (!isString(event.tile.label)) errors.push('tile.label 必须是非空字符串');
    }
  }

  if (event.metadata !== undefined && !isRecord(event.metadata)) {
    errors.push('metadata 若提供必须是对象');
  }

  return { valid: errors.length === 0, errors };
}

/** 校验整条 trace：逐事件校验 + step 单调递增 + id 唯一 */
export function validateTVIRTrace(trace: unknown): ValidationResult {
  const errors: string[] = [];

  if (!isRecord(trace)) {
    return { valid: false, errors: ['trace 必须是对象'] };
  }
  if (!isString(trace.description)) {
    errors.push('trace.description 必须是非空字符串');
  }
  if (!Array.isArray(trace.events)) {
    errors.push('trace.events 必须是数组');
    return { valid: false, errors };
  }
  if (trace.events.length === 0) {
    errors.push('trace.events 不能为空');
    return { valid: false, errors };
  }

  const seenIds = new Set<string>();
  let previousStep = -1;

  trace.events.forEach((event: unknown, index: number) => {
    const result = validateTVIREvent(event);
    if (!result.valid) {
      errors.push(...result.errors.map((e) => `events[${index}]: ${e}`));
      return;
    }
    const e = event as TVIREvent;
    if (seenIds.has(e.id)) {
      errors.push(`events[${index}]: id 重复 "${e.id}"`);
    }
    seenIds.add(e.id);
    if (e.step <= previousStep) {
      errors.push(`events[${index}]: step 必须严格递增（前一个为 ${previousStep}，当前为 ${e.step}）`);
    }
    previousStep = e.step;
  });

  // 教学完整性检查：仅对仿真 trace（或未标注来源的既有 trace）要求
  // 以 GEMM_START 开始、GEMM_END 结束；真实 trace（provenance === 'real-trace'）
  // 来自 profiler，不遵循该教学结构，豁免此检查（V0.5）。
  const isRealTrace = trace.provenance === 'real-trace';
  const first = trace.events[0] as TVIREvent | undefined;
  const last = trace.events[trace.events.length - 1] as TVIREvent | undefined;
  if (!isRealTrace && first && first.type !== 'GEMM_START') {
    errors.push('首个事件类型应为 GEMM_START（教学完整性）');
  }
  if (!isRealTrace && last && last.type !== 'GEMM_END') {
    errors.push('末个事件类型应为 GEMM_END（教学完整性）');
  }

  // provenance / isSample 字段校验（V0.5）
  if (
    trace.provenance !== undefined &&
    trace.provenance !== 'simulation' &&
    trace.provenance !== 'real-trace'
  ) {
    errors.push('trace.provenance 若提供必须是 "simulation" 或 "real-trace"');
  }
  if (trace.isSample !== undefined && typeof trace.isSample !== 'boolean') {
    errors.push('trace.isSample 若提供必须是布尔值');
  }

  return { valid: errors.length === 0, errors };
}

/** 类型守卫：事件引用了某个计算单元 */
export function referencesComputeUnit(
  event: TVIREvent,
  unit: (typeof COMPUTE_UNITS)[number],
): boolean {
  switch (unit) {
    case 'SM':
      return event.sm !== undefined;
    case 'WARP':
      return event.warp !== undefined;
    case 'TENSOR_CORE':
      return event.type === 'MMA';
  }
}
