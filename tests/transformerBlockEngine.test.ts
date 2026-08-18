/**
 * Transformer Block Simulation Engine 单元测试（实施手册 §21/§22，V0.3）。
 *
 * V0.3 架构考点：Attention 必须作为可嵌入子图复用（emitAttentionEvents），
 * 且 Transformer Block 中 Attention 的输入是 RMSNorm 之后的 Xn（Pre-Norm）。
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_TRANSFORMER_BLOCK_CONFIG, simulateTransformerBlock } from '../src/core/simulation';
import { validateTVIRTrace } from '../src/core/tvir/validation';

describe('simulateTransformerBlock', () => {
  it('生成的 trace 通过 TVIR schema 校验', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const result = validateTVIRTrace(trace);
    expect(result.valid, result.errors.join('; ')).toBe(true);
  });

  it('Operator 出现顺序符合 Pre-Norm Transformer Block 数据流', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const seen: string[] = [];
    for (const event of trace.events) {
      const op = event.operator ?? '';
      if (op && seen[seen.length - 1] !== op) seen.push(op);
    }
    expect(seen).toEqual([
      'Transformer Block',
      'RMSNorm (pre-Attention)',
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
      'Residual 1 (+ Attention)',
      'RMSNorm (pre-FFN)',
      'FFN',
      'FFN Up Projection',
      'FFN SiLU',
      'FFN Down Projection',
      'Residual 2 (+ FFN)',
      'Transformer Block',
    ]);
  });

  it('Attention 子图输入为 RMSNorm 之后的 Xn（Pre-Norm 概念正确）', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const qStart = trace.events.find(
      (e) => e.operator === 'Q Projection' && e.type === 'GEMM_START',
    );
    expect(qStart).toBeDefined();
    const meta = qStart?.metadata as { gemm: { left: string } };
    expect(meta.gemm.left).toBe('Xn');
  });

  it('FFN 维度正确：Up 升维到 ffnDim，Down 从 ffnDim 降回 d_model', () => {
    const cfg = DEFAULT_TRANSFORMER_BLOCK_CONFIG;
    const trace = simulateTransformerBlock(cfg);
    const upStart = trace.events.find(
      (e) => e.operator === 'FFN Up Projection' && e.type === 'GEMM_START',
    );
    const downStart = trace.events.find(
      (e) => e.operator === 'FFN Down Projection' && e.type === 'GEMM_START',
    );
    expect(upStart).toBeDefined();
    expect(downStart).toBeDefined();
    const upMeta = upStart?.metadata as { gemm: { N: number; K: number } };
    const downMeta = downStart?.metadata as { gemm: { N: number; K: number } };
    expect(upMeta.gemm.N).toBe(cfg.ffnDim);
    expect(upMeta.gemm.K).toBe(cfg.dModel);
    expect(downMeta.gemm.K).toBe(cfg.ffnDim);
    expect(downMeta.gemm.N).toBe(cfg.dModel);
  });

  it('RMSNorm/Residual/SiLU 不使用 Tensor Core（无 MMA 事件）', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const nonGemmOps = new Set([
      'RMSNorm (pre-Attention)',
      'RMSNorm (pre-FFN)',
      'Residual 1 (+ Attention)',
      'Residual 2 (+ FFN)',
      'FFN SiLU',
    ]);
    const mmaInElementwise = trace.events.filter(
      (e) => e.type === 'MMA' && nonGemmOps.has(e.operator ?? ''),
    );
    expect(mmaInElementwise.length).toBe(0);
  });

  it('FFN 的两次投影是真正的 GEMM（含 MMA 事件）', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    for (const op of ['FFN Up Projection', 'FFN Down Projection']) {
      const hasMma = trace.events.some((e) => e.type === 'MMA' && e.operator === op);
      expect(hasMma, `${op} 应包含 MMA 事件`).toBe(true);
    }
  });

  it('内存搬运遵循层级：无 HBM 直连 Register', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const direct = trace.events.filter(
      (e) => e.type === 'MEMORY_MOVE' && e.source === 'HBM' && e.destination === 'REGISTER',
    );
    expect(direct.length).toBe(0);
  });

  it('所有事件带 operator 与 what/why 教学解释', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    for (const event of trace.events) {
      expect(event.operator, `事件 ${event.id} 缺少 operator`).toBeTruthy();
      expect(event.what.length).toBeGreaterThan(0);
      expect(event.why.length).toBeGreaterThan(0);
    }
  });
});
