/**
 * Real Trace 模式单元测试（实施手册 §23，V0.5）。
 *
 * V0.5 架构验收点（手册 §9）：把 GEMM simulator 换成 trace parser，
 * Playback 与 UI 零改动——parser 的输出与仿真引擎一样都是合法 TVIRTrace。
 * 数据可信度规则（手册 §24）：示例数据绝不能被标成 Measured。
 */

import { describe, expect, it } from 'vitest';
import {
  parseNsightTrace,
  inferRealTraceHardware,
  isRealTraceEvent,
  SAMPLE_REAL_TRACE,
} from '../src/core/realtrace';
import { validateTVIRTrace } from '../src/core/tvir/validation';
import { projectKernelTimeline } from '../src/core/tvir/projection';
import { PlaybackEngine } from '../src/core/playback';

describe('parseNsightTrace', () => {
  it('内置示例 trace 解析成功，输出通过 TVIR schema 校验', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const validation = validateTVIRTrace(result.trace);
    expect(validation.valid, validation.errors.join('; ')).toBe(true);
  });

  it('输出的 trace 标记为 real-trace 来源，示例数据带 isSample', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.trace.provenance).toBe('real-trace');
    expect(result.trace.isSample).toBe(true);
    // 数据可信度规则：示例数据的描述绝不能声称 Measured
    expect(result.trace.description).toContain('示例数据');
    expect(result.trace.description).not.toContain('Measured 数据');
  });

  it('非示例 trace 标记为 Measured 数据', () => {
    const realFile = { ...SAMPLE_REAL_TRACE, sample: false };
    const result = parseNsightTrace(realFile);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.trace.isSample).toBeUndefined();
    expect(result.trace.description).toContain('Measured 数据');
  });

  it('每个 kernel 生成 KERNEL_LAUNCH + BLOCK_SCHEDULE 事件，携带实测时长', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!result.ok) throw new Error(result.error);
    const launches = result.trace.events.filter((e) => e.type === 'KERNEL_LAUNCH');
    expect(launches.length).toBe(SAMPLE_REAL_TRACE.kernels.length);
    for (const launch of launches) {
      expect(launch.kernel).toBeTruthy();
      expect(launch.what).toContain('µs');
      // 真实 trace 不臆测 Block 内部细节：没有 MMA / MEMORY 事件
    }
    expect(result.trace.events.some((e) => e.type === 'MMA')).toBe(false);
    expect(result.trace.events.some((e) => e.type === 'MEMORY_LOAD')).toBe(false);
    expect(result.trace.events.some((e) => e.type === 'BLOCK_SCHEDULE')).toBe(true);
  });

  it('拒绝非法输入并给出中文错误说明', () => {
    expect(parseNsightTrace(null).ok).toBe(false);
    expect(parseNsightTrace({}).ok).toBe(false);
    expect(parseNsightTrace({ kernels: [] }).ok).toBe(false);

    const badKernel = { kernels: [{ name: '', startNs: 0, durationNs: 10 }] };
    const r1 = parseNsightTrace(badKernel);
    expect(r1.ok).toBe(false);
    if (!r1.ok) expect(r1.error).toContain('name');

    const badDuration = { kernels: [{ name: 'k', startNs: 0, durationNs: -5 }] };
    const r2 = parseNsightTrace(badDuration);
    expect(r2.ok).toBe(false);
    if (!r2.ok) expect(r2.error).toContain('durationNs');

    const badGrid = { kernels: [{ name: 'k', startNs: 0, durationNs: 10, grid: [1, 2] }] };
    const r3 = parseNsightTrace(badGrid);
    expect(r3.ok).toBe(false);
    if (!r3.ok) expect(r3.error).toContain('grid');
  });

  it('kernels 按开始时间排序，事件 step 连续递增', () => {
    const unordered = {
      ...SAMPLE_REAL_TRACE,
      kernels: [...SAMPLE_REAL_TRACE.kernels].reverse(),
    };
    const result = parseNsightTrace(unordered);
    if (!result.ok) throw new Error(result.error);
    const launches = result.trace.events.filter((e) => e.type === 'KERNEL_LAUNCH');
    for (let i = 1; i < launches.length; i++) {
      const prev = launches[i - 1]?.metadata as { kernelInfo: { startNs: number } };
      const cur = launches[i]?.metadata as { kernelInfo: { startNs: number } };
      expect(cur.kernelInfo.startNs).toBeGreaterThanOrEqual(prev.kernelInfo.startNs);
    }
    // step 全局连续
    result.trace.events.forEach((event, index) => {
      expect(event.step).toBe(index);
    });
  });
});

describe('架构验收：parser 输出可直接驱动 Playback（UI 零改动）', () => {
  it('PlaybackEngine 无需任何修改即可消费 real-trace', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!result.ok) throw new Error(result.error);
    const engine = new PlaybackEngine(result.trace);
    expect(engine.totalEvents).toBe(result.trace.events.length);
    expect(engine.currentEvent?.type).toBe('KERNEL_LAUNCH');
    engine.next();
    expect(engine.getState().currentIndex).toBe(1);
    engine.seek(engine.totalEvents - 1);
    expect(engine.isLast).toBe(true);
    engine.dispose();
  });

  it('isRealTraceEvent 正确识别真实 trace 事件', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!result.ok) throw new Error(result.error);
    expect(isRealTraceEvent(result.trace.events[0] ?? null)).toBe(true);
    expect(isRealTraceEvent(null)).toBe(false);
  });
});

describe('projectKernelTimeline', () => {
  it('从 real-trace 提取 kernel 时间段，与源数据一致', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!result.ok) throw new Error(result.error);
    const segments = projectKernelTimeline(result.trace.events);
    expect(segments.length).toBe(SAMPLE_REAL_TRACE.kernels.length);
    const sorted = [...SAMPLE_REAL_TRACE.kernels].sort((a, b) => a.startNs - b.startNs);
    segments.forEach((segment, index) => {
      const source = sorted[index];
      expect(segment.kernel).toBe(source?.name);
      expect(segment.startUs).toBeCloseTo((source?.startNs ?? 0) / 1000, 5);
      expect(segment.durationUs).toBeCloseTo((source?.durationNs ?? 0) / 1000, 5);
    });
  });

  it('仿真 trace（无 kernelInfo）返回空时间轴', () => {
    expect(
      projectKernelTimeline([
        { id: 'a', step: 0, type: 'KERNEL_LAUNCH', title: 't', what: 'w', why: 'y' },
      ]),
    ).toEqual([]);
  });
});

describe('inferRealTraceHardware', () => {
  it('从事件推导 SM 数量与 warps/Block', () => {
    const result = parseNsightTrace(SAMPLE_REAL_TRACE);
    if (!result.ok) throw new Error(result.error);
    const hardware = inferRealTraceHardware(result.trace);
    expect(hardware.numSM).toBe(4); // smCount=4
    expect(hardware.warpsPerBlock).toBe(2); // 64 线程/Block ÷ 32
  });
});
