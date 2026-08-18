/**
 * V0.6 Performance Analysis — metrics module unit tests.
 *
 * Validates the roofline-style simulated metrics and the Measured/Estimated/
 * Simulated data-source labeling required by §24.
 */

import { describe, expect, it } from 'vitest';
import { computePerfReport, modelGemm, ASSUMED_HARDWARE, DEFAULT_HARDWARE_SPEC } from '../src/core/perf/metrics';
import type { TVIRTrace, TVIREvent } from '../src/core/tvir/types';
import { simulateGemm, DEFAULT_GEMM_CONFIG } from '../src/core/simulation';
import { parseNsightTrace, SAMPLE_REAL_TRACE } from '../src/core/realtrace';

function makeSimTrace(): TVIRTrace {
  return simulateGemm(DEFAULT_GEMM_CONFIG);
}

function findGemmEvent(trace: TVIRTrace): TVIREvent {
  const ev = trace.events.find((e) => e.metadata && 'gemm' in (e.metadata as object));
  if (!ev) throw new Error('no gemm event found');
  return ev;
}

describe('computePerfReport — simulation mode', () => {
  it('returns six metrics in fixed order for a GEMM event', () => {
    const trace = makeSimTrace();
    const event = findGemmEvent(trace);
    const report = computePerfReport(trace, event);
    expect(report.metrics.map((m) => m.key)).toEqual([
      'duration',
      'tcUtil',
      'bandwidth',
      'l2Hit',
      'occupancy',
      'ai',
    ]);
  });

  it('labels simulated GEMM metrics as simulated and arithmetic intensity as estimated', () => {
    const trace = makeSimTrace();
    const event = findGemmEvent(trace);
    const report = computePerfReport(trace, event);
    const byKey = Object.fromEntries(report.metrics.map((m) => [m.key, m.source]));
    expect(byKey.duration).toBe('simulated');
    expect(byKey.tcUtil).toBe('simulated');
    expect(byKey.bandwidth).toBe('simulated');
    expect(byKey.l2Hit).toBe('simulated');
    expect(byKey.occupancy).toBe('simulated');
    expect(byKey.ai).toBe('estimated');
    expect(report.dataClass).toBe('simulated');
  });

  it('delegates GEMM duration to modelGemm (V0.7 roofline with SM efficiency & L2 discount)', () => {
    const trace = makeSimTrace();
    const event = findGemmEvent(trace);
    const report = computePerfReport(trace, event);
    const gemm = (event.metadata as { gemm: object }).gemm as {
      M: number; N: number; K: number; tileM: number; tileN: number; tileK: number; warpsPerBlock?: number;
    };
    const modeled = modelGemm(
      { ...gemm, warpsPerBlock: gemm.warpsPerBlock ?? 4 },
      DEFAULT_HARDWARE_SPEC,
    );
    const duration = report.metrics.find((m) => m.key === 'duration');
    expect(duration?.value).toBeCloseTo(modeled.durationUs, 5);
    // duration must be positive and finite
    expect(modeled.durationUs).toBeGreaterThan(0);
    expect(Number.isFinite(modeled.durationUs)).toBe(true);
  });

  it('marks all metrics unavailable when no operator is selected', () => {
    const trace = makeSimTrace();
    const report = computePerfReport(trace, null);
    for (const metric of report.metrics) {
      expect(metric.source).toBe('unavailable');
      expect(metric.value).toBeNull();
    }
  });

  it('computes AI as 2*M*N*K / unique-bytes for GEMM', () => {
    const trace = makeSimTrace();
    const event = findGemmEvent(trace);
    const report = computePerfReport(trace, event);
    const meta = (event.metadata as { gemm: { M: number; N: number; K: number } }).gemm;
    const expectedAi =
      (2 * meta.M * meta.N * meta.K) /
      ((meta.M * meta.K + meta.K * meta.N + meta.M * meta.N) * ASSUMED_HARDWARE.bytesPerElement);
    const ai = report.metrics.find((m) => m.key === 'ai');
    expect(ai?.value).toBeCloseTo(expectedAi, 5);
  });
});

describe('computePerfReport — real-trace mode', () => {
  it('labels sample trace as sample dataClass, never measured', () => {
    const parsed = parseNsightTrace(SAMPLE_REAL_TRACE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const report = computePerfReport(parsed.trace, null);
    expect(report.dataClass).toBe('sample');
    // trace-level duration metric must NOT be measured for sample data
    const dur = report.metrics.find((m) => m.key === 'duration');
    expect(dur?.source).not.toBe('measured');
  });

  it('labels a non-sample trace as measured', () => {
    const parsed = parseNsightTrace({ ...SAMPLE_REAL_TRACE, sample: false });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    const report = computePerfReport(parsed.trace, null);
    expect(report.dataClass).toBe('measured');
    const dur = report.metrics.find((m) => m.key === 'duration');
    expect(dur?.source).toBe('measured');
  });

  it('aggregates per-operator breakdown sorted by total time', () => {
    const parsed = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!parsed.ok) throw new Error(parsed.error);
    const report = computePerfReport(parsed.trace, null);
    expect(report.breakdown.length).toBeGreaterThan(0);
    for (let i = 1; i < report.breakdown.length; i++) {
      expect(report.breakdown[i].totalUs).toBeLessThanOrEqual(report.breakdown[i - 1].totalUs);
    }
  });

  it('marks Nsight-Compute-only metrics unavailable at trace level', () => {
    const parsed = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!parsed.ok) throw new Error(parsed.error);
    const report = computePerfReport(parsed.trace, null);
    for (const key of ['tcUtil', 'bandwidth', 'l2Hit', 'occupancy', 'ai'] as const) {
      const m = report.metrics.find((x) => x.key === key);
      expect(m?.source).toBe('unavailable');
    }
  });

  it('uses kernel-level metrics when a kernel event is selected', () => {
    const parsed = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!parsed.ok) throw new Error(parsed.error);
    const kernelEvent = parsed.trace.events.find(
      (e) => e.type === 'KERNEL_LAUNCH' && (e.metadata as { kernelInfo?: { metrics?: object } })?.kernelInfo?.metrics,
    );
    expect(kernelEvent).toBeDefined();
    const report = computePerfReport(parsed.trace, kernelEvent ?? null);
    const tc = report.metrics.find((m) => m.key === 'tcUtil');
    expect(tc?.value).toBeGreaterThan(0);
    expect(report.scopeLabel).toContain('kernel');
  });
});
