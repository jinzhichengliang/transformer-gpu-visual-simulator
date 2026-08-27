/**
 * Execution Planner / Executor 单元测试（Sprint 4-5, Task C1-D2）。
 *
 * 验证：
 *   - InferenceTask 校验（非法输入拒绝）
 *   - OperatorGraph 构建与校验
 *   - planExecution 生成合法 TVIR
 *   - Prefill 与 Decode 语义差异
 *   - MoE 展开（Router → Top-K → Dispatch → Expert → Combine）
 */

import { describe, expect, it } from 'vitest';
import {
  validateInferenceTask,
  DEFAULT_INFERENCE_TASK,
  type InferenceTask,
} from '../src/core/execution/task';
import {
  buildOperatorGraph,
  validateOperatorGraph,
} from '../src/core/execution/planner';
import { planExecution } from '../src/core/execution/executor';
import {
  makeGenericDenseProfile,
  makeGenericMoEProfile,
} from '../src/core/modelprofile';
import { validateTVIRTrace } from '../src/core/tvir';

describe('validateInferenceTask（Task C1）', () => {
  it('合法任务通过', () => {
    expect(validateInferenceTask(DEFAULT_INFERENCE_TASK).valid).toBe(true);
  });

  it('promptTokens = -10 被拒绝', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, promptTokens: -10 };
    const result = validateInferenceTask(task);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('promptTokens');
  });

  it('batchSize = 0 被拒绝', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, batchSize: 0 };
    expect(validateInferenceTask(task).valid).toBe(false);
  });

  it('outputTokens = -1 被拒绝', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, outputTokens: -1 };
    expect(validateInferenceTask(task).valid).toBe(false);
  });

  it('decode 阶段 outputTokens = 0 被拒绝', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'decode' as const, outputTokens: 0 };
    expect(validateInferenceTask(task).valid).toBe(false);
  });

  it('prefill 阶段 outputTokens = 0 允许（不做 decode）', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const, outputTokens: 0 };
    expect(validateInferenceTask(task).valid).toBe(true);
  });

  it('NaN 输入被拒绝', () => {
    const task = { ...DEFAULT_INFERENCE_TASK, promptTokens: NaN };
    expect(validateInferenceTask(task).valid).toBe(false);
  });
});

describe('buildOperatorGraph（Task D1）', () => {
  it('Dense Profile 生成合法 OperatorGraph', () => {
    const profile = makeGenericDenseProfile({ numLayers: 2 });
    const graph = buildOperatorGraph(profile, DEFAULT_INFERENCE_TASK);
    const result = validateOperatorGraph(graph);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('Graph 包含 embedding → attention → ffn → lm_head 节点', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const graph = buildOperatorGraph(profile, DEFAULT_INFERENCE_TASK);
    const types = graph.nodes.map((n) => n.operatorType);
    expect(types).toContain('embedding');
    expect(types).toContain('attention');
    expect(types).toContain('ffn');
    expect(types).toContain('lm_head');
  });

  it('MoE Profile 的 Graph 包含 moe 节点而非 ffn', () => {
    const profile = makeGenericMoEProfile({ numLayers: 1 });
    const graph = buildOperatorGraph(profile, DEFAULT_INFERENCE_TASK);
    const types = graph.nodes.map((n) => n.operatorType);
    expect(types).toContain('moe');
    expect(types).not.toContain('ffn');
  });

  it('依赖链为线性（无孤立节点、无环）', () => {
    const profile = makeGenericDenseProfile({ numLayers: 3 });
    const graph = buildOperatorGraph(profile, DEFAULT_INFERENCE_TASK);
    // 首节点无依赖
    expect(graph.nodes[0].dependsOn).toHaveLength(0);
    // 后续节点恰好依赖前一个
    for (let i = 1; i < graph.nodes.length; i++) {
      expect(graph.nodes[i].dependsOn).toEqual([graph.nodes[i - 1].id]);
    }
  });

  it('不同 promptTokens 产生不同 shape（Task C2 验收）', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task128 = { ...DEFAULT_INFERENCE_TASK, promptTokens: 128 };
    const task4096 = { ...DEFAULT_INFERENCE_TASK, promptTokens: 4096 };
    const g128 = buildOperatorGraph(profile, task128);
    const g4096 = buildOperatorGraph(profile, task4096);
    // embedding 输出行数应不同
    const emb128 = g128.nodes.find((n) => n.operatorType === 'embedding');
    const emb4096 = g4096.nodes.find((n) => n.operatorType === 'embedding');
    expect(emb128?.outputShape.rows).toBe(128);
    expect(emb4096?.outputShape.rows).toBe(4096);
  });
});

describe('planExecution — Prefill（Task C2）', () => {
  it('Generic Dense + Prefill 生成合法 TVIR', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const };
    const result = planExecution(profile, task);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const tvirCheck = validateTVIRTrace(result.trace);
    expect(tvirCheck.valid).toBe(true);
    expect(result.trace.events.length).toBeGreaterThan(10);
  });

  it('Prefill 事件携带 model context metadata', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const };
    const result = planExecution(profile, task);
    if (!result.ok) throw new Error(result.error);
    const withModel = result.trace.events.filter(
      (e) => e.metadata?.model && (e.metadata.model as Record<string, unknown>).modelId,
    );
    expect(withModel.length).toBeGreaterThan(0);
    const first = withModel[0].metadata?.model as Record<string, unknown>;
    expect(first.modelId).toBe('generic-dense');
    expect(first.phase).toBe('prefill');
  });

  it('Prompt 长度变化改变事件中的 shape（非仅 UI 数字变化）', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const r128 = planExecution(profile, { ...DEFAULT_INFERENCE_TASK, promptTokens: 32 });
    const r512 = planExecution(profile, { ...DEFAULT_INFERENCE_TASK, promptTokens: 64 });
    if (!r128.ok || !r512.ok) throw new Error('plan failed');
    // 找 GEMM_START 事件的 metadata.gemm.M
    const gemm128 = r128.trace.events.find((e) => e.metadata?.gemm && (e.metadata.gemm as Record<string, unknown>).M);
    const gemm512 = r512.trace.events.find((e) => e.metadata?.gemm && (e.metadata.gemm as Record<string, unknown>).M);
    expect(gemm128).toBeDefined();
    expect(gemm512).toBeDefined();
    expect((gemm128!.metadata!.gemm as Record<string, unknown>).M).toBe(32);
    expect((gemm512!.metadata!.gemm as Record<string, unknown>).M).toBe(64);
  });
});

describe('planExecution — Decode（Task C3）', () => {
  it('Decode 生成独立事件流（含 KV Cache 读取语义）', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task: InferenceTask = { ...DEFAULT_INFERENCE_TASK, phase: 'decode', promptTokens: 32, outputTokens: 2 };
    const result = planExecution(profile, task);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const titles = result.trace.events.map((e) => e.title);
    // 必须看到 Decode Step 1 和 Decode Step 2
    expect(titles.some((t) => t.includes('Decode Step 1'))).toBe(true);
    expect(titles.some((t) => t.includes('Decode Step 2'))).toBe(true);
    // 必须有 KV Cache 读取事件
    expect(titles.some((t) => t.includes('KV Cache'))).toBe(true);
  });

  it('Decode 不是简单退化为 seq=1 prefill（有 cache access / previous state 语义）', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task: InferenceTask = { ...DEFAULT_INFERENCE_TASK, phase: 'decode', promptTokens: 16, outputTokens: 1 };
    const result = planExecution(profile, task);
    if (!result.ok) throw new Error(result.error);
    // 检查存在 kvCache metadata
    const cacheEvents = result.trace.events.filter(
      (e) => e.metadata?.kvCache,
    );
    expect(cacheEvents.length).toBeGreaterThan(0);
    // 检查 decode metadata
    const decodeEvents = result.trace.events.filter(
      (e) => e.metadata?.decode,
    );
    expect(decodeEvents.length).toBeGreaterThan(0);
  });

  it('prefill_decode 模式先完整 prefill 再逐步 decode', () => {
    const profile = makeGenericDenseProfile({ numLayers: 1 });
    const task: InferenceTask = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill_decode', promptTokens: 16, outputTokens: 3 };
    const result = planExecution(profile, task);
    if (!result.ok) throw new Error(result.error);
    const titles = result.trace.events.map((e) => e.title);
    // Prefill 在前
    const prefillIdx = titles.findIndex((t) => t.includes('Prefill 开始'));
    expect(prefillIdx).toBeGreaterThanOrEqual(0);
    // Decode 在后
    const decodeIdx = titles.findIndex((t) => t.includes('Decode Step 1'));
    expect(decodeIdx).toBeGreaterThan(prefillIdx);
    // 3 个 decode step（每步有"开始"+"完成"两条事件）
    const stepStarts = titles.filter((t) => /^Decode Step \d+：/.test(t));
    expect(stepStarts.length).toBe(3);
  });
});

describe('planExecution — MoE 展开（Task D2）', () => {
  it('MoE 层展开为 Router → Top-K → Dispatch → Expert → Combine', () => {
    const profile = makeGenericMoEProfile({ numLayers: 1, numExperts: 4, expertsPerToken: 2 });
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const };
    const result = planExecution(profile, task);
    if (!result.ok) throw new Error(result.error);
    const operators = result.trace.events.map((e) => e.operator).filter(Boolean);
    expect(operators).toContain('MoE Router');
    expect(operators).toContain('MoE Top-K');
    expect(operators).toContain('MoE Dispatch');
    expect(operators).toContain('MoE Combine');
    // Expert GEMM 存在
    expect(operators.some((op) => op?.includes('MoE Expert GEMM'))).toBe(true);
  });

  it('修改 numExperts 无需改 Planner（schema 可扩展性）', () => {
    const p4 = makeGenericMoEProfile({ numLayers: 1, numExperts: 4, expertsPerToken: 1 });
    const p64 = makeGenericMoEProfile({ numLayers: 1, numExperts: 64, expertsPerToken: 8 });
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const };
    const r4 = planExecution(p4, task);
    const r64 = planExecution(p64, task);
    expect(r4.ok).toBe(true);
    expect(r64.ok).toBe(true);
    if (!r4.ok || !r64.ok) return;
    // 64 专家模型的 trace 应更长（更多 expert GEMM 段）
    expect(r64.trace.events.length).toBeGreaterThan(r4.trace.events.length);
  });
});

describe('planExecution — 失败安全（Task 27 Failure Injection）', () => {
  it('非法 Profile 返回明确错误而非 crash', () => {
    const badProfile = makeGenericDenseProfile({ id: '' });
    const result = planExecution(badProfile, DEFAULT_INFERENCE_TASK);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('ModelProfile 校验失败');
  });

  it('非法 Task 返回明确错误而非产生 NaN', () => {
    const profile = makeGenericDenseProfile();
    const badTask = { ...DEFAULT_INFERENCE_TASK, promptTokens: -5 };
    const result = planExecution(profile, badTask);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('InferenceTask 校验失败');
  });
});
