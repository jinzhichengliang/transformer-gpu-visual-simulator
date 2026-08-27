/**
 * Generic Transformer Profile 单元测试（Sprint 2 Task B1 / Sprint 3 Task B2）。
 *
 * 任务书验证目标：
 *   - ModelProfile 不是为四个模型硬编码的（Generic 能进 Planner 生成 TVIR）；
 *   - MoE schema 可扩展：修改 numExperts / expertsPerToken 无需改 Planner 或 UI。
 */

import { describe, expect, it } from 'vitest';
import {
  makeGenericDenseProfile,
  makeGenericMoEProfile,
  validateModelProfile,
} from '../src/core/modelprofile';

describe('Generic Dense Transformer Profile（Task B1）', () => {
  it('默认配置生成合法 Dense Profile', () => {
    const profile = makeGenericDenseProfile();
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('Profile 不含任何特定模型名称（family=Generic）', () => {
    const profile = makeGenericDenseProfile();
    expect(profile.family).toBe('Generic');
    expect(profile.id).not.toMatch(/deepseek|kimi|glm|qwen/i);
  });

  it('层序列遵循 Embedding → Layer×N → LM Head 结构', () => {
    const profile = makeGenericDenseProfile({ numLayers: 3 });
    expect(profile.layers[0].type).toBe('embedding');
    expect(profile.layers[profile.layers.length - 1].type).toBe('lm_head');
    // 中间应有 3 层 × (attention+norm+ffn+residual) = 12 层
    const middle = profile.layers.slice(1, -1);
    expect(middle).toHaveLength(12);
    const attentionLayers = middle.filter((l) => l.type === 'attention');
    expect(attentionLayers).toHaveLength(3);
  });

  it('architecture.type 为 dense', () => {
    const profile = makeGenericDenseProfile();
    expect(profile.architecture.type).toBe('dense');
  });

  it('自定义维度参数被正确传入且可溯源', () => {
    const profile = makeGenericDenseProfile({ hiddenSize: 256, numHeads: 8 });
    expect(profile.architecture.hiddenSize?.value).toBe(256);
    const attn = profile.layers.find((l) => l.type === 'attention');
    expect(attn?.attention?.numHeads?.value).toBe(8);
    // 可溯源：每个值都有 sources
    expect(profile.architecture.hiddenSize?.sources.length).toBeGreaterThan(0);
  });
});

describe('Generic MoE Transformer Profile（Task B2）', () => {
  it('默认配置生成合法 MoE Profile', () => {
    const profile = makeGenericMoEProfile();
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(true);
  });

  it('architecture.type 为 moe', () => {
    const profile = makeGenericMoEProfile();
    expect(profile.architecture.type).toBe('moe');
  });

  // schema 可扩展性：不同专家规模无需改 Planner / UI
  const moeConfigs = [
    { numExperts: 4, expertsPerToken: 1, name: '4 experts / top-1' },
    { numExperts: 8, expertsPerToken: 2, name: '8 experts / top-2' },
    { numExperts: 64, expertsPerToken: 8, name: '64 experts / top-8' },
  ];
  for (const cfg of moeConfigs) {
    it(`schema 支持 ${cfg.name}（修改参数不改 Planner）`, () => {
      const profile = makeGenericMoEProfile({
        numExperts: cfg.numExperts,
        expertsPerToken: cfg.expertsPerToken,
      });
      const result = validateModelProfile(profile);
      expect(result.valid).toBe(true);
      const moeLayer = profile.layers.find((l) => l.type === 'moe');
      expect(moeLayer?.moe?.numExperts.value).toBe(cfg.numExperts);
      expect(moeLayer?.moe?.expertsPerToken.value).toBe(cfg.expertsPerToken);
    });
  }

  it('MoE 层与 Attention 层在同一模型内共存', () => {
    const profile = makeGenericMoEProfile({ numLayers: 2 });
    const types = profile.layers.map((l) => l.type);
    expect(types).toContain('attention');
    expect(types).toContain('moe');
    expect(types).toContain('norm');
    expect(types).toContain('residual');
  });
});

describe('ModelProfile 与具体模型解耦', () => {
  it('Dense 与 MoE Profile 均通过同一 validateModelProfile（无模型特判）', () => {
    const dense = validateModelProfile(makeGenericDenseProfile());
    const moe = validateModelProfile(makeGenericMoEProfile());
    expect(dense.valid).toBe(true);
    expect(moe.valid).toBe(true);
  });
});
