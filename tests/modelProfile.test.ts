/**
 * ModelProfile Framework 单元测试（Sprint 1, Task A1/A2）。
 *
 * 任务书自动验证要求：
 *   - Dense ModelProfile
 *   - MoE ModelProfile
 *   - Hybrid Layer ModelProfile
 *   - Invalid Profile
 *   - Missing Required Fields
 *   - Unknown Operator Type
 *   - 来源区分（公开/推断/估算）
 *   - estimated 不得展示为 measured
 */

import { describe, expect, it } from 'vitest';
import type { ModelProfile } from '../src/core/modelprofile';
import {
  validateModelProfile,
  hasEstimatedMarkedAsMeasured,
  traced,
  officialSource,
  inferredSource,
  estimatedSource,
} from '../src/core/modelprofile';

/** 构造一个合法的 Dense Transformer Profile（教学用小模型） */
function makeDenseProfile(overrides: Partial<ModelProfile> = {}): ModelProfile {
  const src = officialSource('Test Official Doc');
  return {
    id: 'test-dense',
    displayName: 'Test Dense',
    family: 'TestFamily',
    version: '1.0',
    architecture: {
      type: 'dense',
      hiddenSize: traced(128, [src]),
      vocabSize: traced(1000, [src]),
      normType: 'rmsnorm',
    },
    layers: [
      { type: 'embedding' },
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(4, [src]),
          numKVHeads: traced(4, [src]),
          headDim: traced(32, [src]),
        },
      },
      { type: 'norm' },
      { type: 'ffn', ffn: { intermediateSize: traced(512, [src]), activation: 'silu' } },
      { type: 'residual' },
      { type: 'lm_head' },
    ],
    contextLength: traced(2048, [src]),
    parameterInfo: { total: traced(1_000_000, [src]) },
    source: [src],
    fidelity: 'L1',
    ...overrides,
  };
}

/** 构造一个合法的 MoE Transformer Profile */
function makeMoEProfile(overrides: Partial<ModelProfile> = {}): ModelProfile {
  const src = officialSource('Test MoE Doc');
  return {
    id: 'test-moe',
    displayName: 'Test MoE',
    family: 'TestFamily',
    version: '1.0',
    architecture: { type: 'moe', hiddenSize: traced(256, [src]) },
    layers: [
      { type: 'embedding' },
      {
        type: 'attention',
        attention: {
          attentionType: 'gqa',
          numHeads: traced(8, [src]),
          numKVHeads: traced(2, [src]),
          headDim: traced(32, [src]),
        },
      },
      {
        type: 'moe',
        moe: {
          numExperts: traced(8, [src]),
          expertsPerToken: traced(2, [src]),
          hasSharedExperts: traced(true, [src]),
        },
      },
      { type: 'lm_head' },
    ],
    source: [src],
    fidelity: 'L1',
    ...overrides,
  };
}

describe('validateModelProfile — Dense', () => {
  it('合法 Dense Profile 通过校验', () => {
    const result = validateModelProfile(makeDenseProfile());
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('Dense Profile 的 hiddenSize 携带来源与保真度（可溯源）', () => {
    const profile = makeDenseProfile();
    expect(profile.architecture.hiddenSize?.value).toBe(128);
    expect(profile.architecture.hiddenSize?.sources[0].sourceType).toBe('official');
    expect(profile.architecture.hiddenSize?.fidelity).toBe('L1');
  });
});

describe('validateModelProfile — MoE', () => {
  it('合法 MoE Profile 通过校验（8 experts / top-2）', () => {
    const result = validateModelProfile(makeMoEProfile());
    expect(result.valid).toBe(true);
  });

  it('expertsPerToken 超过 numExperts 被拒绝', () => {
    const profile = makeMoEProfile();
    const moeLayer = profile.layers.find((l) => l.type === 'moe');
    if (moeLayer?.moe) moeLayer.moe.expertsPerToken = traced(64, [officialSource('x')]);
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('不能超过 numExperts');
  });
});

describe('validateModelProfile — Hybrid Layer', () => {
  it('Dense 层 + MoE 层混合的 Hybrid Profile 通过校验', () => {
    const src = officialSource('Hybrid Doc');
    const profile: ModelProfile = {
      id: 'test-hybrid',
      displayName: 'Test Hybrid',
      family: 'TestFamily',
      version: '1.0',
      architecture: { type: 'hybrid' },
      layers: [
        { type: 'embedding' },
        // 前几层是 Dense FFN
        {
          type: 'attention',
          attention: { attentionType: 'mha', numHeads: traced(4, [src]) },
        },
        { type: 'ffn', ffn: { intermediateSize: traced(512, [src]) } },
        // 后续层切换为 MoE
        {
          type: 'attention',
          attention: { attentionType: 'mha', numHeads: traced(4, [src]) },
        },
        {
          type: 'moe',
          moe: { numExperts: traced(4, [src]), expertsPerToken: traced(1, [src]) },
        },
        { type: 'lm_head' },
      ],
      source: [src],
      fidelity: 'L2',
    };
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(true);
  });
});

describe('validateModelProfile — Invalid Profile', () => {
  it('Missing Required Fields：空 id / displayName 被拒绝', () => {
    const profile = makeDenseProfile({ id: '', displayName: '' });
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('id 不能为空');
    expect(result.errors.join(' ')).toContain('displayName 不能为空');
  });

  it('layers 为空数组被拒绝', () => {
    const result = validateModelProfile(makeDenseProfile({ layers: [] }));
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('至少需要 1 层');
  });

  it('Unknown Layer Type 被拒绝', () => {
    const profile = makeDenseProfile();
    // 强制注入非法类型（模拟恶意输入）
    profile.layers = [{ type: 'convolution' as never }];
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('type 非法');
  });

  it('非法 architecture.type 被拒绝', () => {
    const profile = makeDenseProfile();
    profile.architecture = { type: 'cnn' as never };
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('architecture.type 非法');
  });

  it('非法 fidelity 被拒绝', () => {
    const profile = makeDenseProfile();
    profile.fidelity = 'L9' as never;
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('fidelity 非法');
  });

  it('attention 层缺少 attention 描述被拒绝', () => {
    const profile = makeDenseProfile();
    profile.layers[1] = { type: 'attention' };
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('缺少 attention 描述');
  });

  it('moe 层缺少 moe 描述被拒绝', () => {
    const profile = makeMoEProfile();
    const idx = profile.layers.findIndex((l) => l.type === 'moe');
    profile.layers[idx] = { type: 'moe' };
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('缺少 moe 描述');
  });

  it('Traced 字段缺少 sources 被拒绝', () => {
    const profile = makeDenseProfile();
    if (profile.architecture.hiddenSize) profile.architecture.hiddenSize.sources = [];
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('sources 缺失');
  });

  it('来源缺少 verifiedAt 合法日期被拒绝', () => {
    const profile = makeDenseProfile();
    profile.source = [{ sourceType: 'official', reference: 'x', verifiedAt: '2026/08/26', confidence: 'high' }];
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(false);
    expect(result.errors.join(' ')).toContain('verifiedAt 必须为 YYYY-MM-DD');
  });
});

describe('Task A2 — 来源区分与防虚假精度', () => {
  it('同一 Profile 中公开/推断/估算参数可区分', () => {
    const profile = makeDenseProfile();
    // hiddenSize=公开，vocabSize=推断，contextLength=估算
    profile.architecture.hiddenSize = traced(128, [officialSource('Official Doc')], 'L1');
    profile.architecture.vocabSize = traced(1000, [inferredSource('Inferred from tokenizer')], 'L1');
    profile.contextLength = traced(2048, [estimatedSource('Estimated from product page')], 'L1');

    expect(profile.architecture.hiddenSize?.sources[0].sourceType).toBe('official');
    expect(profile.architecture.vocabSize?.sources[0].sourceType).toBe('inferred');
    expect(profile.contextLength?.sources[0].sourceType).toBe('estimated');

    // 校验仍通过（三种来源都是合法来源类型）
    expect(validateModelProfile(profile).valid).toBe(true);
  });

  it('estimated 来源与 L4 实测保真度矛盾会被检测出', () => {
    const profile = makeDenseProfile();
    profile.source = [estimatedSource('pure guess')];
    profile.fidelity = 'L4';
    expect(hasEstimatedMarkedAsMeasured(profile)).toBe(true);
  });

  it('正常 Profile 不存在 estimated 冒充 measured 的情况', () => {
    expect(hasEstimatedMarkedAsMeasured(makeDenseProfile())).toBe(false);
    expect(hasEstimatedMarkedAsMeasured(makeMoEProfile())).toBe(false);
  });
});

describe('Schema 可扩展性（任务书 §4：不为具体模型硬编码）', () => {
  it('schema 中不出现具体模型名称判断逻辑', () => {
    // 通过构造任意 family 的 profile 都能通过校验来间接验证：
    // schema 对 family 字段无白名单限制
    for (const family of ['DeepSeek-V4', 'Kimi', 'GLM', 'Qwen', 'AnyFutureModel']) {
      const result = validateModelProfile(makeDenseProfile({ family }));
      expect(result.valid).toBe(true);
    }
  });

  it('不同 Attention 类型（mha/mqa/gqa/mla）均可表达', () => {
    for (const attentionType of ['mha', 'mqa', 'gqa', 'mla'] as const) {
      const profile = makeDenseProfile();
      profile.layers[1] = {
        type: 'attention',
        attention: {
          attentionType,
          numHeads: traced(4, [officialSource('doc')]),
          numKVHeads: traced(attentionType === 'mqa' ? 1 : 2, [officialSource('doc')]),
          headDim: traced(32, [officialSource('doc')]),
        },
      };
      expect(validateModelProfile(profile).valid).toBe(true);
    }
  });
});
