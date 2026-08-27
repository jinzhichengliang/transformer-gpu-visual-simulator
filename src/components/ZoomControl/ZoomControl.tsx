/**
 * ZoomControl — Semantic Zoom 级别控制条（Sprint 8, Task G1-G5）。
 *
 * 任务书要求：用户可在 Model → Layer → Operator → Kernel → GPU
 * 之间步进缩放；同一事件在不同级别显示不同粒度，锚点保持一致。
 */

import { ZOOM_LEVELS, ZOOM_LEVEL_LABELS, type ZoomLevel } from '../../core/zoom';
import './ZoomControl.css';

interface ZoomControlProps {
  level: ZoomLevel;
  onChange: (level: ZoomLevel) => void;
}

export function ZoomControl({ level, onChange }: ZoomControlProps) {
  return (
    <div className="zoom-control" role="tablist" aria-label="Semantic Zoom 级别">
      <span className="zoom-label">Semantic Zoom</span>
      <div className="zoom-levels">
        {ZOOM_LEVELS.map((lv) => (
          <button
            key={lv}
            role="tab"
            aria-selected={lv === level}
            className={`zoom-btn${lv === level ? ' zoom-btn-active' : ''}`}
            onClick={() => onChange(lv)}
          >
            {ZOOM_LEVEL_LABELS[lv]}
          </button>
        ))}
      </div>
      <span className="zoom-hint">粒度：{ZOOM_LEVEL_LABELS[level]}</span>
    </div>
  );
}
