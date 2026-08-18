/**
 * Model 层结构模块单元测试（实施手册 §28 六层联动，V1.0）。
 *
 * V1.0 核心要求：Model 层展示 Transformer Block 结构树，
 * 根据当前 TVIR 事件的 operator 投影"激活路径"（根 → 激活子层）。
 * Model 层不理解仿真细节，只读 event.operator（纯查找逻辑）。
 */

import { describe, expect, it } from 'vitest';
import {
  TRANSFORMER_BLOCK_MODEL,
  findActivePath,
  isBlockModelEvent,
  projectActiveModelPath,
} from '../src/core/model';
import {
  DEFAULT_TRANSFORMER_BLOCK_CONFIG,
  simulateTransformerBlock,
} from '../src/core/simulation';
import type { TVIREvent } from '../src/core/tvir/types';

describe('TRANSFORMER_BLOCK_MODEL', () => {
  it('根节点是 Transformer Block，包含六个直接子层', () => {
    expect(TRANSFORMER_BLOCK_MODEL.id).toBe('block');
    expect(TRANSFORMER_BLOCK_MODEL.children.map((c) => c.id)).toEqual([
      'norm1',
      'attention',
      'residual1',
      'norm2',
      'ffn',
      'residual2',
    ]);
  });

  it('Attention 有 9 个子算子，FFN 有 3 个子算子', () => {
    const attention = TRANSFORMER_BLOCK_MODEL.children.find((c) => c.id === 'attention');
    const ffn = TRANSFORMER_BLOCK_MODEL.children.find((c) => c.id === 'ffn');
    expect(attention?.children).toHaveLength(9);
    expect(ffn?.children).toHaveLength(3);
  });
});

describe('findActivePath', () => {
  it('Block 级算子命中根节点', () => {
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'Transformer Block')).toEqual(['block']);
  });

  it('顶层子层命中时路径包含根与自身', () => {
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'RMSNorm (pre-Attention)')).toEqual([
      'block',
      'norm1',
    ]);
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'Residual 2 (+ FFN)')).toEqual([
      'block',
      'residual2',
    ]);
  });

  it('Attention 子算子命中时路径经过 attention 父节点', () => {
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'QK MatMul')).toEqual([
      'block',
      'attention',
      'qk',
    ]);
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'Softmax')).toEqual([
      'block',
      'attention',
      'softmax',
    ]);
  });

  it('FFN 子算子命中时路径经过 ffn 父节点', () => {
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'FFN SiLU')).toEqual([
      'block',
      'ffn',
      'ffn-silu',
    ]);
  });

  it('未知算子或未提供算子返回空路径', () => {
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, 'NotAnOperator')).toEqual([]);
    expect(findActivePath(TRANSFORMER_BLOCK_MODEL, undefined)).toEqual([]);
  });
});

describe('projectActiveModelPath / isBlockModelEvent', () => {
  it('null 事件返回空路径且非 Block 事件', () => {
    expect(projectActiveModelPath(null)).toEqual([]);
    expect(isBlockModelEvent(null)).toBe(false);
  });

  it('无 operator 的事件不是 Block 事件', () => {
    const event: TVIREvent = { type: 'TILE_DISPATCH' } as unknown as TVIREvent;
    expect(isBlockModelEvent(event)).toBe(false);
    expect(projectActiveModelPath(event)).toEqual([]);
  });

  it('Block 仿真的每个事件都能投影出非空激活路径', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    expect(trace.events.length).toBeGreaterThan(0);
    for (const event of trace.events) {
      if (!event.operator) continue;
      expect(isBlockModelEvent(event)).toBe(true);
      const path = projectActiveModelPath(event);
      expect(path.length).toBeGreaterThanOrEqual(1);
      expect(path[0]).toBe('block');
      // 非块级算子必须命中具体子层（路径长度 ≥ 2）
      if (event.operator !== 'Transformer Block') {
        expect(path.length).toBeGreaterThanOrEqual(2);
      }
    }
  });

  it('Block 仿真的激活路径覆盖 Pre-Norm 六子层', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const hit = new Set<string>();
    for (const event of trace.events) {
      for (const id of projectActiveModelPath(event)) hit.add(id);
    }
    for (const id of ['block', 'norm1', 'attention', 'residual1', 'norm2', 'ffn', 'residual2']) {
      expect(hit.has(id)).toBe(true);
    }
  });
});
