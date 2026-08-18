/**
 * Attention Simulation Engine 单元测试（实施手册 §19/§20，V0.2）。
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_ATTENTION_CONFIG, simulateAttention } from '../src/core/simulation';
import { validateTVIRTrace } from '../src/core/tvir/validation';

describe('simulateAttention', () => {
  it('生成的 trace 通过 TVIR schema 校验', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const result = validateTVIRTrace(trace);
    expect(result.valid, result.errors.join('; ')).toBe(true);
  });

  it('包含全部 9 个 Operator，且每个事件都带 operator 与 what/why', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const operators = new Set(trace.events.map((e) => e.operator).filter(Boolean));
    const expected = [
      'Q Projection',
      'K Projection',
      'V Projection',
      'QK MatMul',
      'Scale',
      'Mask',
      'Softmax',
      'AV MatMul',
      'Output Projection',
    ];
    for (const op of expected) {
      expect(operators.has(op), `缺少 Operator: ${op}`).toBe(true);
    }
    for (const event of trace.events) {
      expect(event.what.length).toBeGreaterThan(0);
      expect(event.why.length).toBeGreaterThan(0);
    }
  });

  it('Operator 出现顺序符合 Attention 计算图', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const seen: string[] = [];
    for (const event of trace.events) {
      const op = event.operator ?? '';
      if (op && seen[seen.length - 1] !== op) seen.push(op);
    }
    expect(seen).toEqual([
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
    ]);
  });

  it('QK MatMul 输出为 seqLen×seqLen 的注意力分数矩阵 S', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const qkStart = trace.events.find(
      (e) => e.operator === 'QK MatMul' && e.type === 'GEMM_START',
    );
    expect(qkStart).toBeDefined();
    const meta = qkStart?.metadata as { gemm: { out: string; M: number; N: number } };
    expect(meta.gemm.out).toBe('S');
    expect(meta.gemm.M).toBe(DEFAULT_ATTENTION_CONFIG.seqLen);
    expect(meta.gemm.N).toBe(DEFAULT_ATTENTION_CONFIG.seqLen);
  });

  it('Scale/Mask/Softmax 不使用 Tensor Core（无 MMA 事件）', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const nonGemmOps = new Set(['Scale', 'Mask', 'Softmax']);
    const mmaInElementwise = trace.events.filter(
      (e) => e.type === 'MMA' && nonGemmOps.has(e.operator ?? ''),
    );
    expect(mmaInElementwise.length).toBe(0);
  });

  it('GEMM 类 Operator 均含 MMA 事件（复用 Tensor Core 路径）', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const gemmOps = ['Q Projection', 'K Projection', 'V Projection', 'QK MatMul', 'AV MatMul', 'Output Projection'];
    for (const op of gemmOps) {
      const hasMma = trace.events.some((e) => e.type === 'MMA' && e.operator === op);
      expect(hasMma, `${op} 应包含 MMA 事件`).toBe(true);
    }
  });

  it('内存搬运遵循层级：无 HBM 直连 Register', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const direct = trace.events.filter(
      (e) => e.type === 'MEMORY_MOVE' && e.source === 'HBM' && e.destination === 'REGISTER',
    );
    expect(direct.length).toBe(0);
  });
});
