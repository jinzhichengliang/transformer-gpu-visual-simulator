/**
 * EventExplanation — 事件解释区（实施手册 §6 区域 5）。
 * 展示当前事件的 What / Why 教学解释，以及事件上下文标签。
 * 只读当前 TVIR 事件。
 */

import type { TVIREvent } from '../../core/tvir/types';
import './EventExplanation.css';

export interface EventExplanationProps {
  event: TVIREvent | null;
  stepIndex: number;
  totalEvents: number;
}

export function EventExplanation(props: EventExplanationProps) {
  const { event, stepIndex, totalEvents } = props;

  if (!event) {
    return (
      <div className="event-explanation">
        <p className="ee-empty">暂无事件</p>
      </div>
    );
  }

  const contextTags: string[] = [];
  if (event.operator) contextTags.push(`Operator: ${event.operator}`);
  if (event.kernel) contextTags.push(`kernel: ${event.kernel}`);
  if (event.block !== undefined) contextTags.push(`Block ${event.block}`);
  if (event.sm !== undefined) contextTags.push(`SM ${event.sm}`);
  if (event.warp !== undefined) contextTags.push(`Warp ${event.warp}`);
  if (event.tile) contextTags.push(event.tile.label);
  if (event.source && event.destination) {
    contextTags.push(`${event.source} → ${event.destination}`);
  }

  return (
    <div className="event-explanation">
      <div className="ee-header">
        <span className="ee-step">
          Step {stepIndex + 1} / {totalEvents}
        </span>
        <span className={`ee-type ee-type-${event.type}`}>{event.type}</span>
      </div>

      <h3 className="ee-title">{event.title}</h3>

      <div className="ee-section">
        <span className="ee-label">What · 正在发生什么</span>
        <p>{event.what}</p>
      </div>

      <div className="ee-section">
        <span className="ee-label">Why · 为什么这么做</span>
        <p>{event.why}</p>
      </div>

      {contextTags.length > 0 ? (
        <div className="ee-tags">
          {contextTags.map((tag) => (
            <span key={tag} className="ee-tag">
              {tag}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
