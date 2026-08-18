/**
 * Compiler View 知识层单元测试（实施手册 §22，V0.4）。
 *
 * V0.4 核心要求：每个 Operator 都能展示 Math → Operator → IR → Kernel → GPU
 * 五层下钻链，且 TVIR 基础架构零改动（知识层只消费事件的公开字段）。
 */

import { describe, expect, it } from 'vitest';
import {
  hasCompileRecipe,
  projectCompileChain,
} from '../src/core/compiler';
import {
  DEFAULT_ATTENTION_CONFIG,
  DEFAULT_GEMM_CONFIG,
  DEFAULT_TRANSFORMER_BLOCK_CONFIG,
  simulateAttention,
  simulateGemm,
  simulateTransformerBlock,
} from '../src/core/simulation';
import { EXAMPLE_TVIR_TRACE } from '../src/core/tvir';

describe('projectCompileChain', () => {
  it('无 operator 字段的事件返回 null（如手写示例 trace）', () => {
    const first = EXAMPLE_TVIR_TRACE.events[0];
    expect(first?.operator).toBeUndefined();
    expect(projectCompileChain(first ?? null)).toBeNull();
    expect(projectCompileChain(null)).toBeNull();
  });

  it('GEMM 场景的数学公式由 metadata.gemm 动态生成', () => {
    const trace = simulateGemm(DEFAULT_GEMM_CONFIG);
    const gemmStart = trace.events.find((e) => e.type === 'GEMM_START');
    const chain = projectCompileChain(gemmStart ?? null);
    expect(chain).not.toBeNull();
    expect(chain?.math).toBe('C = A × B');
    expect(chain?.operatorType).toBe('MatMul');
    expect(chain?.gpuUnit).toBe('Tensor Core');
  });

  it('Attention 场景的 Q Projection 展示正确的下钻链', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const qStart = trace.events.find(
      (e) => e.operator === 'Q Projection' && e.type === 'GEMM_START',
    );
    const chain = projectCompileChain(qStart ?? null);
    expect(chain).not.toBeNull();
    expect(chain?.math).toBe('Q = X × Wq');
    expect(chain?.operatorType).toBe('MatMul');
    expect(chain?.kernelType).toBe('Tiled GEMM Kernel');
    expect(chain?.gpuUnit).toBe('Tensor Core');
    expect(chain?.ir.some((inst) => inst.op === 'MUL' || inst.op === 'ACC')).toBe(true);
  });

  it('Softmax 的 IR 包含行归约指令，且执行单元不是 Tensor Core', () => {
    const trace = simulateAttention(DEFAULT_ATTENTION_CONFIG);
    const softmax = trace.events.find((e) => e.operator === 'Softmax');
    const chain = projectCompileChain(softmax ?? null);
    expect(chain).not.toBeNull();
    expect(chain?.operatorType).toBe('RowReduce');
    expect(chain?.ir.some((inst) => inst.op === 'REDUCE')).toBe(true);
    expect(chain?.gpuUnit).not.toBe('Tensor Core');
  });

  it('RMSNorm 是行归约 kernel，由 CUDA Core 执行', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const rmsnorm = trace.events.find((e) => e.operator === 'RMSNorm (pre-Attention)');
    const chain = projectCompileChain(rmsnorm ?? null);
    expect(chain).not.toBeNull();
    expect(chain?.operatorType).toBe('RowReduce');
    expect(chain?.gpuUnit).toContain('CUDA Core');
  });

  it('每条 IR 指令都带教学解释', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    for (const event of trace.events) {
      const chain = projectCompileChain(event);
      if (chain === null) continue;
      for (const inst of chain.ir) {
        expect(inst.op.length).toBeGreaterThan(0);
        expect(inst.meaning.length).toBeGreaterThan(0);
      }
    }
  });
});

describe('hasCompileRecipe — 知识表覆盖度', () => {
  it('GEMM / Attention / Transformer Block 所有算子均已登记编译知识', () => {
    const traces = [
      simulateGemm(DEFAULT_GEMM_CONFIG),
      simulateAttention(DEFAULT_ATTENTION_CONFIG),
      simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG),
    ];
    const missing: string[] = [];
    for (const trace of traces) {
      for (const event of trace.events) {
        if (event.operator && !hasCompileRecipe(event.operator)) {
          missing.push(event.operator);
        }
      }
    }
    expect(missing, `未登记编译知识的算子: ${[...new Set(missing)].join(', ')}`).toEqual([]);
  });
});
