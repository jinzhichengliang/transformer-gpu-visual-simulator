/**
 * usePlayback — React 侧对 Playback Engine 的绑定。
 * 这是 UI 与核心逻辑之间唯一的桥梁：组件只消费本 hook 暴露的状态，
 * 不直接 import Simulation Engine。
 */

import { useEffect, useMemo, useState } from 'react';
import type { TVIREvent, TVIRTrace } from '../core/tvir/types';
import type { PlaybackState } from '../core/playback/playbackEngine';
import { PlaybackEngine } from '../core/playback/playbackEngine';

export interface PlaybackBinding {
  engine: PlaybackEngine;
  state: PlaybackState;
  event: TVIREvent | null;
  totalEvents: number;
  previous: () => void;
  next: () => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
  seek: (index: number) => void;
  setSpeed: (speed: number) => void;
  loadTrace: (trace: TVIRTrace) => void;
}

export function usePlayback(initialTrace: TVIRTrace): PlaybackBinding {
  const engine = useMemo(() => new PlaybackEngine(initialTrace), []);

  const [state, setState] = useState<PlaybackState>(() => engine.getState());
  const [event, setEvent] = useState<TVIREvent | null>(() => engine.currentEvent);

  useEffect(() => {
    const unsubscribe = engine.subscribe((nextState, nextEvent) => {
      setState(nextState);
      setEvent(nextEvent);
    });
    return () => {
      unsubscribe();
    };
  }, [engine]);

  useEffect(() => () => engine.dispose(), [engine]);

  return {
    engine,
    state,
    event,
    totalEvents: engine.totalEvents,
    previous: () => engine.previous(),
    next: () => engine.next(),
    play: () => engine.play(),
    pause: () => engine.pause(),
    reset: () => engine.reset(),
    seek: (index) => engine.seek(index),
    setSpeed: (speed) => engine.setSpeed(speed),
    loadTrace: (trace) => engine.loadTrace(trace),
  };
}
