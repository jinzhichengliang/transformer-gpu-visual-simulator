/**
 * ControlBar — 播放控制条（实施手册 §6 区域 1）。
 * 纯受控组件：只接收状态与回调，不含任何业务逻辑。
 */

import './ControlBar.css';

export interface ControlBarProps {
  playing: boolean;
  speed: number;
  currentIndex: number;
  totalEvents: number;
  onPrevious: () => void;
  onNext: () => void;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSeek: (index: number) => void;
  onSpeedChange: (speed: number) => void;
}

const SPEED_OPTIONS = [0.5, 1, 2, 4];

export function ControlBar(props: ControlBarProps) {
  const {
    playing,
    speed,
    currentIndex,
    totalEvents,
    onPrevious,
    onNext,
    onPlay,
    onPause,
    onReset,
    onSeek,
    onSpeedChange,
  } = props;

  return (
    <div className="control-bar">
      <button type="button" onClick={onPrevious} disabled={currentIndex <= 0} title="上一步">
        ⏮ Previous
      </button>
      <button type="button" onClick={onNext} disabled={currentIndex >= totalEvents - 1} title="下一步">
        Step ⏭
      </button>
      {playing ? (
        <button type="button" className="primary" onClick={onPause} title="暂停">
          ⏸ Pause
        </button>
      ) : (
        <button type="button" className="primary" onClick={onPlay} title="播放">
          ▶ Play
        </button>
      )}
      <button type="button" onClick={onReset} title="重置">
        ↺ Reset
      </button>

      <div className="control-progress">
        <input
          type="range"
          min={0}
          max={Math.max(totalEvents - 1, 0)}
          value={currentIndex}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="进度"
        />
        <span className="control-step-label">
          Step {currentIndex + 1} / {totalEvents}
        </span>
      </div>

      <div className="control-speed">
        <span>Speed</span>
        <div className="speed-buttons">
          {SPEED_OPTIONS.map((option) => (
            <button
              key={option}
              type="button"
              className={option === speed ? 'active' : ''}
              onClick={() => onSpeedChange(option)}
            >
              {option}×
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
