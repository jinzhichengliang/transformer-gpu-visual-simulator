/**
 * ZoomFocusCard — 当前 Semantic Zoom 级别的焦点卡片（Sprint 8）。
 *
 * 任务书验证要求的可视化呈现：同一事件在不同缩放级别
 * 显示不同粒度的主内容与细节，但锚点（event id / model context /
 * execution position）保持一致。
 */

import { projectZoomFocus, ZOOM_LEVEL_LABELS, type ZoomLevel } from '../../core/zoom';
import type { TVIREvent } from '../../core/tvir/types';
import './ZoomFocusCard.css';

interface ZoomFocusCardProps {
  event: TVIREvent | null;
  level: ZoomLevel;
}

export function ZoomFocusCard({ event, level }: ZoomFocusCardProps) {
  const focus = projectZoomFocus(event, level);
  if (!focus) {
    return (
      <div className="zoom-focus-card zoom-focus-empty">
        <span className="zoom-focus-level">{ZOOM_LEVEL_LABELS[level]}</span>
        <span className="zoom-focus-empty-text">开始播放后显示当前焦点</span>
      </div>
    );
  }

  const anchor = focus.anchor;
  return (
    <div className="zoom-focus-card">
      <div className="zoom-focus-header">
        <span className="zoom-focus-level">{ZOOM_LEVEL_LABELS[level]} 级视图</span>
        <span className="zoom-focus-anchor">
          锚点：step {anchor.step} · {anchor.type}
          {anchor.modelContext ? ` · Layer ${anchor.modelContext.layerIndex}` : ''}
        </span>
      </div>
      <div className="zoom-focus-primary">{focus.primary}</div>
      {focus.details.length > 0 ? (
        <ul className="zoom-focus-details">
          {focus.details.map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
