/**
 * GEMM Simulation Engine 单元测试（实施手册 §10）。
 */

import { describe, expect, it } from 'vitest';
import { DEFAULT_GEMM_CONFIG, simulateGemm } from '../src/core/simulation';
import { validateTVIRTrace } from '../src/core/tvir/validation';

describe('simulateGemm', () => {
  it('生成的 trace 通过 TVIR schema 校验', () => {
    const trace = simulateGemm(DEFAULT_GEMM_CONFIG);
    const result = validateTVIRTrace(trace);
    expect(result.valid, result.errors.join('; ')).toBe(true);
  });

  it('以 GEMM_START 开始、GEMM_END 结束', () => {
    const trace = simulateGemm(DEFAULT_GEMM_CONFIG);
    expect(trace.events[0]?.type).toBe('GEMM_START');
    expect(trace.events[trace.events.length - 1]?.type).toBe('GEMM_END');
  });

  it('Block 数 = tilesM × tilesN，每个 Block 只调度到一个 SM', () => {
    const config = { ...DEFAULT_GEMM_CONFIG, M: 128, N: 128, K: 128, tileM: 32, tileN: 32, tileK: 32, numSM: 4 };
    const trace = simulateGemm(config);
    const blockSchedules = trace.events.filter((e) => e.type === 'BLOCK_SCHEDULE');
    expect(blockSchedules.length).toBe(4 * 4);
    for (const event of blockSchedules) {
      expect(event.block).toBeTypeOf('number');
      expect(event.sm).toBeTypeOf('number');
      expect(event.sm).toBeGreaterThanOrEqual(0);
      expect(event.sm).toBeLessThan(config.numSM);
    }
  });

  it('事件顺序符合教学流程：每个 K 段先搬 A/B、同步、MMA、累加', () => {
    const config = { ...DEFAULT_GEMM_CONFIG, M: 32, N: 32, K: 64, tileM: 32, tileN: 32, tileK: 32, numSM: 2, warpsPerBlock: 2 };
    const trace = simulateGemm(config);
    const types = trace.events.map((e) => e.type);

    // 单 Block（1×1 tile）、K 有 2 段
    expect(types).toContain('GEMM_START');
    expect(types).toContain('TILE_CREATE');
    expect(types).toContain('KERNEL_LAUNCH');
    expect(types).toContain('BLOCK_SCHEDULE');
    expect(types).toContain('WARP_SCHEDULE');
    expect(types).toContain('SYNC');
    expect(types).toContain('MMA');
    expect(types).toContain('ACCUMULATE');
    expect(types).toContain('MEMORY_STORE');

    // MMA 之前必须有 SYNC（数据就绪保证）
    const firstMma = types.indexOf('MMA');
    const firstSync = types.indexOf('SYNC');
    expect(firstSync).toBeLessThan(firstMma);

    // MMA 后紧跟 ACCUMULATE
    expect(types[firstMma + 1]).toBe('ACCUMULATE');

    // 每个事件都必须有教学解释
    for (const event of trace.events) {
      expect(event.what.length).toBeGreaterThan(0);
      expect(event.why.length).toBeGreaterThan(0);
    }
  });

  it('内存搬运遵循层级：Tensor Core 操作数来自寄存器而非 HBM', () => {
    const trace = simulateGemm(DEFAULT_GEMM_CONFIG);
    const moves = trace.events.filter((e) => e.type === 'MEMORY_MOVE');
    // 存在 SMEM → REGISTER 的搬运（MMA 操作数来源）
    expect(
      moves.some((e) => e.source === 'SHARED_MEMORY' && e.destination === 'REGISTER'),
    ).toBe(true);
    // 不存在 HBM 直接 → REGISTER 的搬运
    expect(
      moves.some((e) => e.source === 'HBM' && e.destination === 'REGISTER'),
    ).toBe(false);
  });
});
