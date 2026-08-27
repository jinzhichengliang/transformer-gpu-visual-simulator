/**
 * TVIR 模型上下文扩展回归测试（Sprint 6, Task E1）。
 *
 * 任务书回归要求：
 *   - 旧 TVIR 测试 + 新 TVIR 测试全部通过；
 *   - 不新增事件类型（12 种唯一词汇表）；
 *   - 旧 trace（无 metadata.model）投影返回 null，行为不变。
 */

import { describe, expect, it } from 'vitest';
import { EXAMPLE_TVIR_TRACE, TVIR_EVENT_TYPES, validateTVIRTrace } from '../src/core/tvir';
import {
  projectModelContext,
  isModelAwareEvent,
  projectModelBreadcrumb,
  phaseLabel,
} from '../src/core/model';
import { planExecution } from '../src/core/execution/executor';
import { makeGenericDenseProfile, makeGenericMoEProfile } from '../src/core/modelprofile';
import { DEFAULT_INFERENCE_TASK, type InferenceTask } from '../src/core/execution/task';

describe('TVIR 事件类型词汇表（回归守护）', () => {
  it('仍然只有 12 种事件类型（零新增）', () => {
    expect(TVIR_EVENT_TYPES).toHaveLength(12);
  });
});

describe('旧 trace 回归（V0.1 GEMM 必须仍然能播放）', () => {
  it('示例 GEMM trace 校验仍通过', () => {
    const result = validateTVIRTrace(EXAMPLE_TVIR_TRACE);
    expect(result.valid).toBe(true);
  });

  it('无 metadata.model 的旧事件投影返回 null（行为不变）', () => {
    const event = EXAMPLE_TVIR_TRACE.events[0];
    expect(projectModelContext(event)).toBeNull();
    expect(isModelAwareEvent(event)).toBe(false);
    expect(projectModelBreadcrumb(event)).toBeNull();
  });
});

describe('新 trace 模型上下文（Task E1）', () => {
  const profile = makeGenericDenseProfile({ numLayers: 2 });
  const task: InferenceTask = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill_decode', promptTokens: 16, outputTokens: 2 };
  const result = planExecution(profile, task);

  it('模型执行计划生成合法 TVIR（旧校验器无需修改即通过）', () => {
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const check = validateTVIRTrace(result.trace);
    expect(check.valid).toBe(true);
  });

  it('事件可定位：模型 → 阶段 → Layer → 算子', () => {
    if (!result.ok) throw new Error(result.error);
    // 找一个带模型上下文且带层号的事件
    const layerEvent = result.trace.events.find((e) => {
      const ctx = projectModelContext(e);
      return ctx && ctx.layerIndex >= 0;
    });
    expect(layerEvent).toBeDefined();
    const ctx = projectModelContext(layerEvent!);
    expect(ctx).not.toBeNull();
    expect(ctx!.modelId).toBe('generic-dense');
    expect(ctx!.layerIndex).toBeGreaterThanOrEqual(0);
    expect(['prefill', 'decode']).toContain(ctx!.phase);
  });

  it('面包屑定位：Kimi 式完整路径（模型/阶段/层/算子）', () => {
    if (!result.ok) throw new Error(result.error);
    const layerEvent = result.trace.events.find((e) => {
      const ctx = projectModelContext(e);
      return ctx && ctx.layerIndex >= 0 && ctx.phase === 'decode';
    });
    expect(layerEvent).toBeDefined();
    const crumb = projectModelBreadcrumb(layerEvent!);
    expect(crumb).not.toBeNull();
    // 形如 [Generic Dense Transformer, Decode Step 1/2, Layer 0, Attention]
    expect(crumb!.parts.length).toBeGreaterThanOrEqual(3);
    expect(crumb!.parts[0]).toBe('Generic Dense Transformer');
    expect(crumb!.parts[1]).toMatch(/^Decode Step \d+\/\d+$/);
    expect(crumb!.parts[2]).toMatch(/^Layer \d+$/);
  });

  it('Decode 事件携带 decodeStep/decodeTotal', () => {
    if (!result.ok) throw new Error(result.error);
    const decodeEvents = result.trace.events
      .map(projectModelContext)
      .filter((c): c is NonNullable<typeof c> => c !== null && c.phase === 'decode');
    expect(decodeEvents.length).toBeGreaterThan(0);
    const withStep = decodeEvents.filter((c) => c.decodeStep !== undefined);
    expect(withStep.length).toBeGreaterThan(0);
    expect(withStep[0].decodeTotal).toBe(2);
  });

  it('prefill 与 decode 阶段标签正确', () => {
    if (!result.ok) throw new Error(result.error);
    const prefillCtx = result.trace.events.map(projectModelContext).find((c) => c?.phase === 'prefill');
    const decodeCtx = result.trace.events.map(projectModelContext).find((c) => c?.phase === 'decode' && c.decodeStep !== undefined);
    expect(phaseLabel(prefillCtx!)).toBe('Prefill');
    expect(phaseLabel(decodeCtx!)).toMatch(/^Decode Step \d+\/\d+$/);
  });
});

describe('MoE 模型上下文（跨模型一致性）', () => {
  it('MoE 事件的模型上下文同样可定位', () => {
    const profile = makeGenericMoEProfile({ numLayers: 1, numExperts: 4, expertsPerToken: 2 });
    const result = planExecution(profile, { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const moeEvent = result.trace.events.find((e) => {
      const ctx = projectModelContext(e);
      return ctx && ctx.operatorType === 'moe';
    });
    expect(moeEvent).toBeDefined();
    const ctx = projectModelContext(moeEvent!);
    expect(ctx!.modelId).toBe('generic-moe');
    expect(ctx!.layerType).toBe('moe');
  });
});
