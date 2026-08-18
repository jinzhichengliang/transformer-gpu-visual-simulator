/**
 * Playback Engine 单元测试（实施手册 §12）。
 */

import { describe, expect, it, vi } from 'vitest';
import { PlaybackEngine } from '../src/core/playback/playbackEngine';
import { EXAMPLE_TVIR_TRACE } from '../src/core/tvir/exampleTrace';

describe('PlaybackEngine', () => {
  it('初始状态：索引 0、未播放、当前事件为第一个', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    expect(engine.getState().currentIndex).toBe(0);
    expect(engine.getState().playing).toBe(false);
    expect(engine.currentEvent?.id).toBe('event-0');
    engine.dispose();
  });

  it('next/previous 在边界内移动', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    engine.next();
    expect(engine.getState().currentIndex).toBe(1);
    engine.previous();
    expect(engine.getState().currentIndex).toBe(0);
    engine.previous();
    expect(engine.getState().currentIndex).toBe(0); // 不会越界
    engine.dispose();
  });

  it('seek 跳转到指定索引并钳制范围', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    engine.seek(5);
    expect(engine.getState().currentIndex).toBe(5);
    engine.seek(9999);
    expect(engine.getState().currentIndex).toBe(engine.totalEvents - 1);
    engine.seek(-3);
    expect(engine.getState().currentIndex).toBe(0);
    engine.dispose();
  });

  it('reset 回到起点并停止播放', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    engine.seek(3);
    engine.reset();
    expect(engine.getState().currentIndex).toBe(0);
    expect(engine.getState().playing).toBe(false);
    engine.dispose();
  });

  it('状态变化会通知订阅者', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    const listener = vi.fn();
    engine.subscribe(listener);
    engine.next();
    expect(listener).toHaveBeenCalledTimes(1);
    engine.dispose();
  });

  it('loadTrace 切换数据源并重置（架构验收：UI/Playback 与数据源解耦）', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    engine.seek(4);
    const shorterTrace = {
      description: 'mini',
      events: EXAMPLE_TVIR_TRACE.events.slice(0, 3),
    };
    engine.loadTrace(shorterTrace);
    expect(engine.totalEvents).toBe(3);
    expect(engine.getState().currentIndex).toBe(0);
    expect(engine.getState().playing).toBe(false);
    engine.dispose();
  });

  it('setSpeed 钳制在 [0.25, 8]', () => {
    const engine = new PlaybackEngine(EXAMPLE_TVIR_TRACE);
    engine.setSpeed(100);
    expect(engine.getState().speed).toBe(8);
    engine.setSpeed(0.01);
    expect(engine.getState().speed).toBe(0.25);
    engine.dispose();
  });
});
