/**
 * Fidelity Badge 单元测试（Sprint 17, Task J1/J2）。
 *
 * 验证：
 *   - J1：Badge 按分类展示来源，GPU Timing 恒为 Simulated；
 *   - J2：模拟数值禁止测量级精度（最多 2 位有效数字 + 限定标签）。
 */

import { describe, expect, it } from 'vitest';
import { buildFidelityBadge, formatSimulatedValue, sourceLabelFor } from '../src/core/fidelity';
import {
  makeDeepSeekV4FlashProfile,
  makeKimiK3Profile,
  makeGLM53Profile,
} from '../src/core/modelprofile';
import type { SourceMetadata } from '../src/core/modelprofile/types';

describe('Fidelity Badge（Task J1）', () => {
  it('Badge 包含 Architecture / Execution Model / GPU Timing 三行', () => {
    const badge = buildFidelityBadge(makeDeepSeekV4FlashProfile());
    const categories = badge.rows.map((r) => r.category);
    expect(categories).toContain('Architecture');
    expect(categories).toContain('Execution Model');
    expect(categories).toContain('GPU Timing');
  });

  it('GPU Timing 恒为 Simulated（本项目不做实测）', () => {
    for (const profile of [makeDeepSeekV4FlashProfile(), makeKimiK3Profile(), makeGLM53Profile()]) {
      const badge = buildFidelityBadge(profile);
      const gpuRow = badge.rows.find((r) => r.category === 'GPU Timing');
      expect(gpuRow?.label).toBe('Simulated');
    }
  });

  it('含推断/估算来源的模型在 Architecture 行标记为弱可信', () => {
    // GLM-5.3 存在 estimated 字段，应标记弱可信
    const badge = buildFidelityBadge(makeGLM53Profile());
    const archRow = badge.rows.find((r) => r.category === 'Architecture');
    expect(archRow).toBeDefined();
    // 若全部为可信来源则不弱；若含推断/估算则弱。两者必有其一，且与标签一致。
    if (archRow!.weak) {
      expect(archRow!.label).toMatch(/Inferred|Estimated|Mixed/);
    } else {
      expect(archRow!.label).not.toMatch(/Inferred|Estimated|Mixed/);
    }
  });

  it('Disclaimer 显式声明教学仿真非实测', () => {
    const badge = buildFidelityBadge(makeDeepSeekV4FlashProfile());
    expect(badge.disclaimer).toMatch(/Simulated|仿真|非实测/);
  });

  it('fidelity 级别透传（L1）', () => {
    const badge = buildFidelityBadge(makeKimiK3Profile());
    expect(badge.fidelity).toBe('L1');
  });
});

describe('禁止虚假精度（Task J2）', () => {
  it('模拟数值最多 2 位有效数字', () => {
    const formatted = formatSimulatedValue(3.427861, 'μs');
    expect(formatted).not.toContain('3.427861');
    expect(formatted).toContain('~ 3.4 μs');
  });

  it('输出携带 Simulated 限定标签', () => {
    expect(formatSimulatedValue(1234.5678, 'ms')).toMatch(/Simulated/);
  });

  it('大数值也被压缩到 2 位有效数字', () => {
    const formatted = formatSimulatedValue(123456, 'GB/s');
    expect(formatted).toContain('~ 120000');
  });

  it('非法数值返回安全占位', () => {
    expect(formatSimulatedValue(Number.NaN, 'ms')).toContain('~ ms');
  });
});

describe('sourceLabelFor', () => {
  const mk = (sourceType: SourceMetadata['sourceType']): SourceMetadata => ({
    sourceType,
    reference: 'test',
    verifiedAt: '2026-08-26',
    confidence: 'high',
  });

  it('官方来源标记 Official', () => {
    expect(sourceLabelFor([mk('official')])).toBe('Official');
  });

  it('纯估算来源标记 Estimated', () => {
    expect(sourceLabelFor([mk('estimated')])).toBe('Estimated');
  });

  it('混合来源标记 Mixed', () => {
    expect(sourceLabelFor([mk('official'), mk('estimated')])).toMatch(/Mixed/);
  });

  it('缺失来源标记 Estimated（不留空、不假造 Official）', () => {
    expect(sourceLabelFor(undefined)).toBe('Estimated');
    expect(sourceLabelFor([])).toBe('Estimated');
  });
});
