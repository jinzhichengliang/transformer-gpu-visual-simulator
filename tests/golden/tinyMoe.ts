/**
 * TinyMoETransformer — Golden Trace 固定测试模型（Sprint 18, Task 18）。
 *
 * 任务书要求：固定一个最小模型作为回归基准，
 * 任何架构修改后重新生成 TVIR，自动比较 expected vs actual，
 * 防止 AI 在后续开发中悄悄修改 execution semantics。
 *
 * 固定配置：
 *   - 2 Layers
 *   - 2 Heads
 *   - 4 Experts / Top-2
 *   - Prefill 4 tokens + Decode 2 tokens
 */

import type { ModelProfile } from '../../src/core/modelprofile/types';
import { traced, officialSource } from '../../src/core/modelprofile/helpers';
import type { InferenceTask } from '../../src/core/execution/task';
import { DEFAULT_HARDWARE_PROFILE } from '../../src/core/execution/task';

/** TinyMoETransformer 的固定 ModelProfile（合成教学模型，非真实模型） */
export function makeTinyMoEProfile(): ModelProfile {
  const src = officialSource('TinyMoETransformer (golden test fixture)', '2026-08-26', 'high');

  return {
    id: 'tiny-moe',
    displayName: 'TinyMoE Transformer',
    family: 'TestFixture',
    version: '1.0',
    architecture: {
      type: 'moe',
      hiddenSize: traced(64, [src]),
      vocabSize: traced(256, [src]),
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers: [
      { type: 'embedding' },
      // Layer 0
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(2, [src]),
          numKVHeads: traced(2, [src]),
          headDim: traced(32, [src]),
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: {
          numExperts: traced(4, [src]),
          expertsPerToken: traced(2, [src]),
          hasSharedExperts: traced(false, [src]),
        },
      },
      { type: 'residual' },
      // Layer 1
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(2, [src]),
          numKVHeads: traced(2, [src]),
          headDim: traced(32, [src]),
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: {
          numExperts: traced(4, [src]),
          expertsPerToken: traced(2, [src]),
          hasSharedExperts: traced(false, [src]),
        },
      },
      { type: 'residual' },
      { type: 'lm_head' },
    ],
    source: [src],
    fidelity: 'L1',
  };
}

/** Golden Trace 的固定推理任务：Prefill 4 tokens + Decode 2 tokens */
export const GOLDEN_TASK: InferenceTask = {
  phase: 'prefill_decode',
  batchSize: 1,
  promptTokens: 4,
  outputTokens: 2,
  hardwareProfile: DEFAULT_HARDWARE_PROFILE,
};

/** 执行语义的结构指纹（不含教学文案——文案微调不应触发回归失败） */
export interface StructuralEvent {
  step: number;
  type: string;
  operator?: string;
  layerIndex?: number;
  phase?: string;
  decodeStep?: number;
  /** KV Cache 长度（Decode 语义的关键状态） */
  cacheLen?: number;
}

/** 从 trace 提取结构指纹 */
export function extractStructuralFingerprint(events: Array<{
  step: number;
  type: string;
  operator?: string;
  metadata?: Record<string, unknown>;
}>): StructuralEvent[] {
  return events.map((e) => {
    const model = e.metadata?.model as Record<string, unknown> | undefined;
    const kvCache = e.metadata?.kvCache as Record<string, unknown> | undefined;
    const entry: StructuralEvent = { step: e.step, type: e.type };
    if (e.operator) entry.operator = e.operator;
    if (model && typeof model.layerIndex === 'number') entry.layerIndex = model.layerIndex;
    if (model && typeof model.phase === 'string') entry.phase = model.phase;
    if (model && typeof model.decodeStep === 'number') entry.decodeStep = model.decodeStep;
    if (kvCache && typeof kvCache.cacheLen === 'number') entry.cacheLen = kvCache.cacheLen;
    return entry;
  });
}
