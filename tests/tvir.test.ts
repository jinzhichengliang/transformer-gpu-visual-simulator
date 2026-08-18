/**
 * TVIR schema validation 单元测试（实施手册 §8 要求）。
 */

import { describe, expect, it } from 'vitest';
import { validateTVIREvent, validateTVIRTrace } from '../src/core/tvir/validation';
import { EXAMPLE_TVIR_TRACE } from '../src/core/tvir/exampleTrace';
import type { TVIREvent } from '../src/core/tvir/types';

const validEvent: TVIREvent = {
  id: 'event-0',
  step: 0,
  type: 'GEMM_START',
  title: 'GEMM 开始',
  what: '准备计算 C = A × B',
  why: '教学示例',
};

describe('validateTVIREvent', () => {
  it('接受结构完整的事件', () => {
    expect(validateTVIREvent(validEvent).valid).toBe(true);
  });

  it('接受带可选字段的内存搬运事件', () => {
    const event: TVIREvent = {
      ...validEvent,
      id: 'event-1',
      type: 'MEMORY_MOVE',
      block: 0,
      sm: 1,
      warp: 2,
      source: 'L2',
      destination: 'SHARED_MEMORY',
      tensor: 'A',
      tile: { tensor: 'A', tileRow: 0, tileCol: 1, label: 'A[0,1]' },
      metadata: { note: 'ok' },
    };
    expect(validateTVIREvent(event).valid).toBe(true);
  });

  it('拒绝缺少 what/why 的事件（教学解释是强制的）', () => {
    const event = { ...validEvent } as Record<string, unknown>;
    delete event.what;
    delete event.why;
    const result = validateTVIREvent(event);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes('what'))).toBe(true);
    expect(result.errors.some((e) => e.includes('why'))).toBe(true);
  });

  it('拒绝非法事件类型', () => {
    const result = validateTVIREvent({ ...validEvent, type: 'SOFTMAX' });
    expect(result.valid).toBe(false);
  });

  it('拒绝非法 MemoryLevel', () => {
    const result = validateTVIREvent({ ...validEvent, type: 'MEMORY_MOVE', source: 'DISK' });
    expect(result.valid).toBe(false);
  });

  it('拒绝非对象输入', () => {
    expect(validateTVIREvent(null).valid).toBe(false);
    expect(validateTVIREvent('string').valid).toBe(false);
  });
});

describe('validateTVIRTrace', () => {
  it('接受手写示例 trace', () => {
    const result = validateTVIRTrace(EXAMPLE_TVIR_TRACE);
    expect(result.valid, result.errors.join('; ')).toBe(true);
  });

  it('拒绝 step 不递增的 trace', () => {
    const trace = {
      description: 'bad',
      events: [
        { ...validEvent, id: 'a', step: 0 },
        { ...validEvent, id: 'b', step: 0 },
        { ...validEvent, id: 'c', type: 'GEMM_END', step: 0 },
      ],
    };
    expect(validateTVIRTrace(trace).valid).toBe(false);
  });

  it('拒绝 id 重复的 trace', () => {
    const trace = {
      description: 'bad',
      events: [
        { ...validEvent, id: 'same', step: 0 },
        { ...validEvent, id: 'same', type: 'GEMM_END', step: 1 },
      ],
    };
    expect(validateTVIRTrace(trace).valid).toBe(false);
  });

  it('拒绝空事件列表', () => {
    expect(validateTVIRTrace({ description: 'empty', events: [] }).valid).toBe(false);
  });
});
