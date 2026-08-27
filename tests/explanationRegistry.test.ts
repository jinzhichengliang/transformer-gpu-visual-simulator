/**
 * ExplanationRegistry 单元测试（Sprint 16, Task I1）。
 *
 * 验收（任务书）：
 *   - 同一 Operator 不同模型可以重用 Explanation（跨模型一致性）；
 *   - 解释不写死在 React component（本模块为纯逻辑层）。
 */

import { describe, expect, it } from 'vitest';
import {
  EXPLANATION_KEYS,
  getExplanation,
  explanationKeyForOperator,
} from '../src/core/explanation';

describe('ExplanationRegistry', () => {
  it('全部登记的解释键都有非空 what/why', () => {
    for (const key of EXPLANATION_KEYS) {
      const explanation = getExplanation(key);
      expect(explanation, `键 ${key} 缺少解释`).not.toBeNull();
      expect(explanation!.what.length).toBeGreaterThan(0);
      expect(explanation!.why.length).toBeGreaterThan(0);
    }
  });

  it('MoE 算子映射到对应解释键', () => {
    expect(explanationKeyForOperator('MoE Router')).toBe('MOE_ROUTE');
    expect(explanationKeyForOperator('MoE Top-K')).toBe('MOE_TOPK');
    expect(explanationKeyForOperator('MoE Dispatch')).toBe('MOE_DISPATCH');
    expect(explanationKeyForOperator('MoE Expert GEMM (Expert 0)')).toBe('MOE_EXPERT');
    expect(explanationKeyForOperator('MoE Combine')).toBe('MOE_COMBINE');
  });

  it('Decode 缓存算子映射正确', () => {
    expect(explanationKeyForOperator('KV Cache Read')).toBe('CACHE_READ');
    expect(explanationKeyForOperator('KV Cache Write')).toBe('CACHE_WRITE');
  });

  it('同一算子在任意模型下重用同一解释（跨模型一致性）', () => {
    // MoE Router 的解释对所有模型相同——注册表是模型无关的
    const explanation = getExplanation('MOE_ROUTE');
    expect(explanation?.why).toContain('Router 根据当前 Token 表征计算专家选择分数');
  });

  it('未知算子返回 null（不臆测）', () => {
    expect(explanationKeyForOperator('Some Unknown Op')).toBeNull();
    expect(explanationKeyForOperator(undefined)).toBeNull();
  });
});
