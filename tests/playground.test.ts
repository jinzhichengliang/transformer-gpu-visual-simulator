/**
 * Architecture Playground 单元测试（实施手册 §25，V0.7）。
 *
 * 验证：硬件参数化后的屋顶线模型、加速比计算、瓶颈判定、
 * 手册三个问题的可回答性（带宽翻倍/算力翻倍/为何有的算子不变）。
 */

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_HARDWARE_SPEC,
  modelGemm,
  modelElementwise,
  type HardwareSpec,
} from '../src/core/perf/metrics';
import {
  analyzePlayground,
  extractWorkloads,
  specsDiffer,
} from '../src/core/perf/playground';
import { simulateTransformerBlock, DEFAULT_TRANSFORMER_BLOCK_CONFIG } from '../src/core/simulation';

const GEMM_SHAPE = {
  M: 256,
  N: 256,
  K: 256,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  warpsPerBlock: 4,
};

describe('HardwareSpec 参数化', () => {
  it('HBM 带宽翻倍使访存密集算子近似减半', () => {
    const base = modelElementwise(64, 64, DEFAULT_HARDWARE_SPEC);
    const doubled: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, hbmBandwidthGBps: DEFAULT_HARDWARE_SPEC.hbmBandwidthGBps * 2 };
    const mod = modelElementwise(64, 64, doubled);
    // 访存密集：时长 ≈ 访存量/带宽 → 带宽翻倍时长近似减半
    expect(base.bound).toBe('memory');
    expect(mod.bound).toBe('memory');
    const ratio = base.durationUs / mod.durationUs;
    expect(ratio).toBeGreaterThan(1.8);
    expect(ratio).toBeLessThan(2.2);
  });

  it('Tensor Core 算力翻倍加速计算密集 GEMM，但访存密集算子不变', () => {
    const doubled: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, tensorCoreTflops: DEFAULT_HARDWARE_SPEC.tensorCoreTflops * 2 };
    const gemmBase = modelGemm(GEMM_SHAPE, DEFAULT_HARDWARE_SPEC);
    const gemmMod = modelGemm(GEMM_SHAPE, doubled);
    expect(gemmBase.bound).toBe('compute');
    expect(gemmMod.durationUs).toBeLessThan(gemmBase.durationUs);

    // 逐元素算子不受 Tensor Core 算力影响
    const ewBase = modelElementwise(64, 64, DEFAULT_HARDWARE_SPEC);
    const ewMod = modelElementwise(64, 64, doubled);
    expect(ewMod.durationUs).toBeCloseTo(ewBase.durationUs, 5);
  });

  it('L2 容量影响 GEMM 有效访存量与命中率', () => {
    const tinyL2: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, l2SizeMB: 1 };
    const hugeL2: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, l2SizeMB: 128 };
    const rTiny = modelGemm(GEMM_SHAPE, tinyL2);
    const rHuge = modelGemm(GEMM_SHAPE, hugeL2);
    // L2 更大 → 命中率更高、有效访存量更低
    expect(rHuge.l2Hit).toBeGreaterThanOrEqual(rTiny.l2Hit);
    expect(rHuge.effectiveBytes).toBeLessThanOrEqual(rTiny.effectiveBytes);
  });

  it('SM 数量通过并行效率影响时长（Block 不足时收益递减）', () => {
    // 小 GEMM：只有少量 Block，加 SM 收益受限
    const smallGemm = { ...GEMM_SHAPE, M: 32, N: 32, K: 32 };
    const oneSm: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, smCount: 1 };
    const manySm: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, smCount: 16 };
    const r1 = modelGemm(smallGemm, oneSm);
    const r16 = modelGemm(smallGemm, manySm);
    expect(r16.durationUs).toBeLessThanOrEqual(r1.durationUs);
    // 效率字段应反映 Block 数相对 SM 槽位的填充
    expect(r16.smEfficiency).toBeLessThanOrEqual(r1.smEfficiency + 1e-9);
  });
});

describe('specsDiffer', () => {
  it('相同规格返回 false，任一字段不同返回 true', () => {
    expect(specsDiffer(DEFAULT_HARDWARE_SPEC, { ...DEFAULT_HARDWARE_SPEC })).toBe(false);
    expect(specsDiffer(DEFAULT_HARDWARE_SPEC, { ...DEFAULT_HARDWARE_SPEC, hbmBandwidthGBps: 4000 })).toBe(true);
    expect(specsDiffer(DEFAULT_HARDWARE_SPEC, { ...DEFAULT_HARDWARE_SPEC, smCount: 8 })).toBe(true);
  });
});

describe('analyzePlayground — 工作负载提取与对比', () => {
  it('从 Transformer Block trace 提取 GEMM 与逐元素算子', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const workloads = extractWorkloads(trace);
    expect(workloads.length).toBeGreaterThan(0);
    const gemmCount = workloads.filter((w) => w.gemm).length;
    const ewCount = workloads.filter((w) => w.elementwise).length;
    expect(gemmCount).toBeGreaterThan(0); // Q/K/V/QK/AV/O/FFN Up/Down
    expect(ewCount).toBeGreaterThan(0); // RMSNorm/Residual/SiLU
  });

  it('带宽翻倍：访存密集算子加速，计算密集算子几乎不变', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const doubled: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, hbmBandwidthGBps: DEFAULT_HARDWARE_SPEC.hbmBandwidthGBps * 2 };
    const analysis = analyzePlayground(trace, DEFAULT_HARDWARE_SPEC, doubled);

    const memBound = analysis.impacts.filter((i) => i.baselineBound === 'memory');
    const computeBound = analysis.impacts.filter((i) => i.baselineBound === 'compute');
    expect(memBound.length).toBeGreaterThan(0);
    for (const impact of memBound) {
      expect(impact.speedup).toBeGreaterThan(1.5); // 近似按带宽比例加速
    }
    for (const impact of computeBound) {
      expect(impact.speedup).toBeCloseTo(1, 1); // 几乎不变
    }
  });

  it('总加速比等于各算子加权和的比值', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const doubled: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, hbmBandwidthGBps: DEFAULT_HARDWARE_SPEC.hbmBandwidthGBps * 2 };
    const analysis = analyzePlayground(trace, DEFAULT_HARDWARE_SPEC, doubled);
    const sumBase = analysis.impacts.reduce((acc, i) => acc + i.baselineUs, 0);
    const sumMod = analysis.impacts.reduce((acc, i) => acc + i.modifiedUs, 0);
    expect(analysis.totals.baselineUs).toBeCloseTo(sumBase, 5);
    expect(analysis.totals.modifiedUs).toBeCloseTo(sumMod, 5);
    expect(analysis.totals.speedup).toBeCloseTo(sumBase / sumMod, 5);
  });

  it('硬件未变化时加速比为 1 且无瓶颈翻转', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const analysis = analyzePlayground(trace, DEFAULT_HARDWARE_SPEC, { ...DEFAULT_HARDWARE_SPEC });
    expect(analysis.totals.speedup).toBeCloseTo(1, 5);
    for (const impact of analysis.impacts) {
      expect(impact.speedup).toBeCloseTo(1, 5);
      expect(impact.boundFlipped).toBe(false);
    }
  });

  it('教学解读非空（回答手册 §25 的问题）', () => {
    const trace = simulateTransformerBlock(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
    const doubled: HardwareSpec = { ...DEFAULT_HARDWARE_SPEC, hbmBandwidthGBps: DEFAULT_HARDWARE_SPEC.hbmBandwidthGBps * 2 };
    const analysis = analyzePlayground(trace, DEFAULT_HARDWARE_SPEC, doubled);
    expect(analysis.takeaways.length).toBeGreaterThan(0);
    expect(analysis.takeaways.some((t) => t.includes('访存密集'))).toBe(true);
  });
});
