/**
 * Golden Trace — 固定最小模型快照测试（Sprint 18, §18/§19）。
 *
 * 任务书要求：
 *   - 固定一个最小模型（TinyMoETransformer：2 Layers / 2 Heads / 4 Experts / Top-2）；
 *   - 固定任务（Prefill 4 tokens / Decode 2 tokens）；
 *   - 保存 golden_trace.json，任何架构修改后重新生成并自动比较；
 *   - 非预期结构变化 → 测试失败（防止后续悄悄修改 execution semantics）。
 *
 * 实现方式：
 *   - 不比对完整事件流（太脆弱），而是比对"结构指纹"：
 *     事件总数、按 type 的计数、算子签名序列、首尾事件标识。
 *   - 指纹以常量形式内联在本测试文件中（可读、可审查、变更需显式更新）。
 */

import { describe, expect, it } from 'vitest';
import { planExecution } from '../src/core/execution/executor';
import { DEFAULT_INFERENCE_TASK, type InferenceTask } from '../src/core/execution/task';
import { traced, officialSource } from '../src/core/modelprofile/helpers';
import type { ModelProfile } from '../src/core/modelprofile/types';
import type { TVIREvent } from '../src/core/tvir/types';

/** TinyMoETransformer：2 Layers / 2 Heads / 4 Experts / Top-2（任务书固定规格） */
function makeTinyMoEProfile(): ModelProfile {
  const src = officialSource('Golden Trace fixture (synthetic teaching model)');
  return {
    id: 'tiny-moe',
    displayName: 'TinyMoETransformer',
    family: 'Golden',
    version: 'fixture',
    architecture: {
      type: 'moe',
      hiddenSize: traced(32, [src]),
      vocabSize: traced(64, [src]),
      normType: 'rmsnorm',
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
          headDim: traced(16, [src]),
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: { numExperts: traced(4, [src]), expertsPerToken: traced(2, [src]) },
      },
      { type: 'residual' },
      // Layer 1
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(2, [src]),
          numKVHeads: traced(2, [src]),
          headDim: traced(16, [src]),
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: { numExperts: traced(4, [src]), expertsPerToken: traced(2, [src]) },
      },
      { type: 'residual' },
      { type: 'lm_head' },
    ],
    source: [src],
    fidelity: 'L1',
  };
}

/** 固定任务：Prefill 4 tokens + Decode 2 tokens（任务书固定规格） */
const GOLDEN_TASK: InferenceTask = {
  ...DEFAULT_INFERENCE_TASK,
  phase: 'prefill_decode',
  promptTokens: 4,
  outputTokens: 2,
  batchSize: 1,
};

/** 计算结构指纹（可读、可审查） */
function structureFingerprint(events: TVIREvent[]) {
  const typeCounts: Record<string, number> = {};
  for (const e of events) typeCounts[e.type] = (typeCounts[e.type] ?? 0) + 1;
  const operatorSig: string[] = [];
  for (const e of events) {
    if (e.operator && operatorSig[operatorSig.length - 1] !== e.operator) {
      operatorSig.push(e.operator);
    }
  }
  return {
    totalEvents: events.length,
    typeCounts,
    operatorSig,
    firstTitle: events[0]?.title ?? '',
    lastTitle: events[events.length - 1]?.title ?? '',
  };
}

/**
 * GOLDEN FINGERPRINT（基线）。
 * 若执行语义发生意外变化，此指纹将不匹配 → 测试失败。
 * 预期内的语义变更需显式更新本基线并说明原因（防止"悄悄修改"）。
 *
 * 基线变更记录：
 *   - 初版：层折叠策略（Collapsed Layers 汇总），201 事件。
 *   - 当前：为支持 374 层级大模型（Kimi K3）引入教学抽样压缩机制，
 *     层不再折叠而是逐层抽样展开（每层完整算子序列保留，
 *     层内 GEMM 的 Block/Warp 细节按抽样配置截断）。
 *     事件总数 201 → 1695，'Collapsed Layers' 算子移除，
 *     每个 MoE 块出现两次 'MoE Dispatch'（每个被选中专家一次）。
 *     此为预期内语义变更，非意外回归。
 */
const GOLDEN_FINGERPRINT = {
  totalEvents: 1695,
  typeCounts: {
    GEMM_START: 65,
    KERNEL_LAUNCH: 79,
    BLOCK_SCHEDULE: 92,
    WARP_SCHEDULE: 266,
    MEMORY_LOAD: 226,
    MEMORY_MOVE: 356,
    SYNC: 179,
    MEMORY_STORE: 100,
    GEMM_END: 88,
    TILE_CREATE: 56,
    MMA: 94,
    ACCUMULATE: 94,
  },
  operatorSig: [
    'Prefill',
    'Embedding',
    'Attention',
    'Q Projection',
    'K Projection',
    'V Projection',
    'QK MatMul',
    'Scale',
    'Mask',
    'Softmax',
    'AV MatMul',
    'Output Projection',
    'Attention',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'Attention',
    'Q Projection',
    'K Projection',
    'V Projection',
    'QK MatMul',
    'Scale',
    'Mask',
    'Softmax',
    'AV MatMul',
    'Output Projection',
    'Attention',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'LM Head',
    'Prefill',
    'Decode',
    'Attention (Decode)',
    'Q Projection (Decode)',
    'KV Cache Read',
    'K Projection (Decode)',
    'KV Cache Write',
    'V Projection (Decode)',
    'KV Cache Write',
    'QK MatMul (Decode)',
    'Softmax (Decode)',
    'AV MatMul (Decode)',
    'Output Projection (Decode)',
    'Attention (Decode)',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'Attention (Decode)',
    'Q Projection (Decode)',
    'KV Cache Read',
    'K Projection (Decode)',
    'KV Cache Write',
    'V Projection (Decode)',
    'KV Cache Write',
    'QK MatMul (Decode)',
    'Softmax (Decode)',
    'AV MatMul (Decode)',
    'Output Projection (Decode)',
    'Attention (Decode)',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'Decode',
    'Attention (Decode)',
    'Q Projection (Decode)',
    'KV Cache Read',
    'K Projection (Decode)',
    'KV Cache Write',
    'V Projection (Decode)',
    'KV Cache Write',
    'QK MatMul (Decode)',
    'Softmax (Decode)',
    'AV MatMul (Decode)',
    'Output Projection (Decode)',
    'Attention (Decode)',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'Attention (Decode)',
    'Q Projection (Decode)',
    'KV Cache Read',
    'K Projection (Decode)',
    'KV Cache Write',
    'V Projection (Decode)',
    'KV Cache Write',
    'QK MatMul (Decode)',
    'Softmax (Decode)',
    'AV MatMul (Decode)',
    'Output Projection (Decode)',
    'Attention (Decode)',
    'MoE Router',
    'MoE Top-K',
    'MoE Dispatch',
    'MoE Expert GEMM (Expert 0)',
    'MoE Expert GEMM (Expert 1)',
    'MoE Dispatch',
    'MoE Combine',
    'Decode',
    'LM Head',
  ],
};

describe('Golden Trace（§18 固定最小模型快照）', () => {
  const profile = makeTinyMoEProfile();
  const result = planExecution(profile, GOLDEN_TASK);

  it('固定任务成功生成执行计划', () => {
    expect(result.ok).toBe(true);
  });

  it('结构指纹与基线一致（防悄悄修改 execution semantics）', () => {
    if (!result.ok) throw new Error(result.error);
    const fp = structureFingerprint(result.trace.events);

    expect(fp.totalEvents, '事件总数变化').toBe(GOLDEN_FINGERPRINT.totalEvents);
    expect(fp.typeCounts, '事件类型分布变化').toEqual(GOLDEN_FINGERPRINT.typeCounts);
    expect(fp.operatorSig, '算子签名序列变化').toEqual(GOLDEN_FINGERPRINT.operatorSig);
    expect(fp.firstTitle).toBe('Prefill 开始：TinyMoETransformer');
    expect(fp.lastTitle).toBe('[Decode Final] LM Head 完成');
  });

  it('Decode Step 1/2 均存在（任务书功能验证）', () => {
    if (!result.ok) throw new Error(result.error);
    const titles = result.trace.events.map((e) => e.title);
    expect(titles.some((t) => t.includes('Decode Step 1：'))).toBe(true);
    expect(titles.some((t) => t.includes('Decode Step 2：'))).toBe(true);
  });

  it('KV Cache 读写事件存在于 Decode 阶段', () => {
    if (!result.ok) throw new Error(result.error);
    const ops = result.trace.events.map((e) => e.operator);
    expect(ops).toContain('KV Cache Read');
    expect(ops).toContain('KV Cache Write');
  });
});

describe('Model Profile Golden Tests（§19 首批模型关键字段快照）', () => {
  it('首批模型关键架构字段不发生意外变化', async () => {
    const { makeDeepSeekV4FlashProfile, makeDeepSeekV4ProProfile, makeKimiK3Profile, makeGLM53Profile } =
      await import('../src/core/modelprofile');

    const flash = makeDeepSeekV4FlashProfile();
    const pro = makeDeepSeekV4ProProfile();
    const kimi = makeKimiK3Profile();
    const glm = makeGLM53Profile();

    // DeepSeek V4 Flash：284B / 13B / 1M 上下文
    expect(flash.parameterInfo?.total?.value).toBe(284_000_000_000);
    expect(flash.parameterInfo?.activated?.value).toBe(13_000_000_000);
    expect(flash.contextLength?.value).toBe(1_000_000);

    // DeepSeek V4 Pro：1.6T / 49B / 1M 上下文
    expect(pro.parameterInfo?.total?.value).toBe(1_600_000_000_000);
    expect(pro.parameterInfo?.activated?.value).toBe(49_000_000_000);
    expect(pro.contextLength?.value).toBe(1_000_000);

    // Kimi K3：2.8T / 104B / 93 层定义（含 norm/residual）/ 896 专家 top-16
    expect(kimi.parameterInfo?.total?.value).toBe(2_800_000_000_000);
    expect(kimi.parameterInfo?.activated?.value).toBe(104_000_000_000);
    const kimiMoe = kimi.layers.find((l) => l.type === 'moe');
    expect(kimiMoe?.moe?.numExperts.value).toBe(896);
    expect(kimiMoe?.moe?.expertsPerToken.value).toBe(16);

    // GLM-5.3：743B / 40B / 1M 上下文 / 256 专家
    expect(glm.parameterInfo?.total?.value).toBe(743_000_000_000);
    expect(glm.parameterInfo?.activated?.value).toBe(40_000_000_000);
    expect(glm.contextLength?.value).toBe(1_000_000);
    const glmMoe = glm.layers.find((l) => l.type === 'moe');
    expect(glmMoe?.moe?.numExperts.value).toBe(256);
  });
});
