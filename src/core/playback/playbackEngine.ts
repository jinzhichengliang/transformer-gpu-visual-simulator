/**
 * TVIR Playback Engine（实施手册 §12）。
 *
 * 职责只有三件事：
 *   1. 选择当前 TVIR Event
 *   2. 控制时间
 *   3. 向订阅者发布当前事件
 *
 * 铁律：Playback Engine 不理解任何 Operator。
 * 这里禁止出现任何 event.type 的业务分支（如 if type === "MMA"）。
 * 它也不 import React、不操作 DOM。
 */

import type { TVIREvent, TVIRTrace } from '../tvir/types';

export interface PlaybackState {
  /** 当前事件在 trace 中的索引（0-based） */
  currentIndex: number;
  /** 是否正在播放 */
  playing: boolean;
  /** 播放速度倍率（1 = 默认步进间隔） */
  speed: number;
}

export type PlaybackListener = (state: PlaybackState, event: TVIREvent | null) => void;

/** 默认每步间隔（毫秒，speed=1 时） */
export const DEFAULT_STEP_INTERVAL_MS = 1200;

export class PlaybackEngine {
  private trace: TVIRTrace;
  private state: PlaybackState;
  private timer: ReturnType<typeof setInterval> | null = null;
  private listeners = new Set<PlaybackListener>();

  constructor(trace: TVIRTrace, initialSpeed = 1) {
    this.trace = trace;
    this.state = { currentIndex: 0, playing: false, speed: initialSpeed };
  }

  // ---------- 查询 ----------

  getState(): PlaybackState {
    return { ...this.state };
  }

  getTrace(): TVIRTrace {
    return this.trace;
  }

  get currentEvent(): TVIREvent | null {
    return this.trace.events[this.state.currentIndex] ?? null;
  }

  get totalEvents(): number {
    return this.trace.events.length;
  }

  get isLast(): boolean {
    return this.state.currentIndex >= this.totalEvents - 1;
  }

  get isFirst(): boolean {
    return this.state.currentIndex <= 0;
  }

  // ---------- 订阅 ----------

  subscribe(listener: PlaybackListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(): void {
    const snapshot = this.getState();
    const event = this.currentEvent;
    for (const listener of this.listeners) {
      listener(snapshot, event);
    }
  }

  // ---------- 控制 ----------

  previous(): void {
    if (this.isFirst) return;
    this.state.currentIndex -= 1;
    this.emit();
  }

  next(): void {
    if (this.isLast) {
      this.stopTimer();
      return;
    }
    this.state.currentIndex += 1;
    this.emit();
  }

  seek(index: number): void {
    const clamped = Math.max(0, Math.min(index, this.totalEvents - 1));
    if (clamped === this.state.currentIndex) return;
    this.state.currentIndex = clamped;
    this.emit();
  }

  play(): void {
    if (this.state.playing) return;
    // 已在末尾时，从头播放
    if (this.isLast) {
      this.state.currentIndex = 0;
      this.emit();
    }
    this.state.playing = true;
    this.startTimer();
    this.emit();
  }

  pause(): void {
    if (!this.state.playing) return;
    this.state.playing = false;
    this.stopTimer();
    this.emit();
  }

  reset(): void {
    this.stopTimer();
    this.state.currentIndex = 0;
    this.state.playing = false;
    this.emit();
  }

  setSpeed(speed: number): void {
    const clamped = Math.max(0.25, Math.min(speed, 8));
    if (clamped === this.state.speed) return;
    this.state.speed = clamped;
    if (this.state.playing) {
      this.startTimer();
    }
    this.emit();
  }

  /** 更换数据源（架构关键：同一 Playback 可消费任意来源的 TVIR trace） */
  loadTrace(trace: TVIRTrace): void {
    this.stopTimer();
    this.trace = trace;
    this.state.currentIndex = 0;
    this.state.playing = false;
    this.emit();
  }

  dispose(): void {
    this.stopTimer();
    this.listeners.clear();
  }

  // ---------- 内部 ----------

  private startTimer(): void {
    this.stopTimer();
    const interval = Math.round(DEFAULT_STEP_INTERVAL_MS / this.state.speed);
    this.timer = setInterval(() => {
      if (this.isLast) {
        this.state.playing = false;
        this.stopTimer();
        this.emit();
        return;
      }
      this.state.currentIndex += 1;
      this.emit();
    }, interval);
  }

  private stopTimer(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}
