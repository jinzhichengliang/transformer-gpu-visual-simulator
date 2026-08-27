/**
 * Semantic Zoom 投影单元测试（Sprint 8, Task G1-G5）。
 *
 * 任务书验证要求：
 *   同一个 TVIR Event 在不同 zoom level 必须显示不同粒度，
 *   但 event id / model context / execution position 保持一致。
 */

import { describe, expect, it } from 'vitest';
import { ZOOM_LEVELS, projectZoomFocus, nextZoomLevel } from '../src/core/zoom';
import { makeTinyMoEProfile, GOLDEN_TASK } from './golden/tinyMoe';
import { planExecution } from '../src/core/execution/executor';
import type { TVIREvent } from '../src/core/tvir/types';

function makeTrace(): TVIREvent[] {
  const result = planExecution(makeTinyMoEProfile(), GOLDEN_TASK);
  if (!result.ok) throw new Error('计划生成失败');
  return result.trace.events;
}

describe('Semantic Zoom 投影', () => {
  const events = makeTrace();

  it('五个级别全部可投影（L1-L5 均有非空焦点）', () => {
    // 取一个 KERNEL_LAUNCH 事件（携带完整 GEMM 元数据）
    const launch = events.find((e) => e.type === 'KERNEL_LAUNCH');
    expect(launch).toBeDefined();
    for (const level of ZOOM_LEVELS) {
      const focus = projectZoomFocus(launch ?? null, level);
      expect(focus, `级别 ${level} 应可投影`).not.toBeNull();
      expect(focus!.primary.length).toBeGreaterThan(0);
    }
  });

  it('同一事件在不同级别显示不同粒度，但锚点一致', () => {
    const launch = events.find((e) => e.type === 'KERNEL_LAUNCH')!;
    const focuses = ZOOM_LEVELS.map((lv) => projectZoomFocus(launch, lv)!);

    // 锚点（step / type / modelContext）在所有级别一致
    const firstAnchor = JSON.stringify(focuses[0]!.anchor);
    for (const f of focuses) {
      expect(JSON.stringify(f.anchor)).toBe(firstAnchor);
      expect(f.anchor.step).toBe(launch.step);
      expect(f.anchor.type).toBe(launch.type);
    }

    // 不同级别的主显示内容应有差异（粒度不同）
    const primaries = new Set(focuses.map((f) => f.primary));
    expect(primaries.size).toBeGreaterThan(1);
  });

  it('Model 级别聚焦 Embedding / Layers / LM Head', () => {
    const embedding = events.find((e) => e.metadata?.model && (e.metadata.model as Record<string, unknown>).layerType === 'embedding');
    const lmHead = events.find((e) => e.metadata?.model && (e.metadata.model as Record<string, unknown>).layerType === 'lm_head');
    if (embedding) expect(projectZoomFocus(embedding, 'model')!.primary).toBe('Embedding');
    if (lmHead) expect(projectZoomFocus(lmHead, 'model')!.primary).toBe('LM Head');
  });

  it('Kernel 级别显示 Grid / Block 信息', () => {
    const launch = events.find((e) => e.type === 'KERNEL_LAUNCH')!;
    const focus = projectZoomFocus(launch, 'kernel')!;
    expect(focus.details.join(' ')).toMatch(/Grid|Block/i);
  });

  it('GPU 级别在 MEMORY_MOVE 事件聚焦 Memory', () => {
    const move = events.find((e) => e.type === 'MEMORY_MOVE');
    if (move) {
      const focus = projectZoomFocus(move, 'gpu')!;
      expect(focus.primary).toBe('Memory');
    }
  });

  it('缩放级别步进切换（向内加深 / 向外变粗，边界返回 null）', () => {
    expect(nextZoomLevel('model', 'in')).toBe('layer');
    expect(nextZoomLevel('layer', 'in')).toBe('operator');
    expect(nextZoomLevel('operator', 'in')).toBe('kernel');
    expect(nextZoomLevel('kernel', 'in')).toBe('gpu');
    expect(nextZoomLevel('gpu', 'in')).toBeNull();
    expect(nextZoomLevel('model', 'out')).toBeNull();
    expect(nextZoomLevel('gpu', 'out')).toBe('kernel');
  });

  it('旧事件（无 metadata.model）在各级别仍可投影（向后兼容）', () => {
    const legacy: TVIREvent = {
      step: 1,
      type: 'GEMM_START',
      title: 'legacy',
      what: 'w',
      why: 'y',
      operator: 'Attention',
    };
    for (const level of ZOOM_LEVELS) {
      const focus = projectZoomFocus(legacy, level);
      expect(focus, `旧事件在 ${level} 级别应可投影`).not.toBeNull();
      // 锚点的 modelContext 为 null（旧事件无模型上下文）
      expect(focus!.anchor.modelContext).toBeNull();
    }
  });
});
