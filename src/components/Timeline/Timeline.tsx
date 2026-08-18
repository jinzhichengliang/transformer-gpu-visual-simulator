/**
 * Timeline — 执行时间线（实施手册 §6 区域 6）。
 * 把整条 TVIR trace 按阶段分组展示为时间轴，点击任意事件可跳转（seek）。
 * V0.2：trace 含多个 Operator 时，顶部增加算子分段条，可整段跳转。
 * 只消费 TVIR 事件的 title/type/operator，不做业务理解。
 */

import type { TVIREvent } from '../../core/tvir/types';
import type { OperatorSegment } from '../../core/tvir/projection';
import './Timeline.css';

export interface TimelineProps {
  events: TVIREvent[];
  currentIndex: number;
  onSeek: (index: number) => void;
  /** 可选：按 operator 切分的连续段（V0.2 起由 App 传入） */
  operatorSegments?: OperatorSegment[] | undefined;
}

/** 按事件类型给时间轴分段着色 */
function typeClass(type: TVIREvent['type']): string {
  switch (type) {
    case 'GEMM_START':
    case 'GEMM_END':
    case 'TILE_CREATE':
    case 'KERNEL_LAUNCH':
      return 'tl-setup';
    case 'BLOCK_SCHEDULE':
    case 'WARP_SCHEDULE':
      return 'tl-schedule';
    case 'MEMORY_LOAD':
    case 'MEMORY_MOVE':
    case 'MEMORY_STORE':
      return 'tl-memory';
    case 'SYNC':
      return 'tl-sync';
    case 'MMA':
    case 'ACCUMULATE':
      return 'tl-compute';
  }
}

export function Timeline(props: TimelineProps) {
  const { events, currentIndex, onSeek } = props;
  const operatorSegments = props.operatorSegments ?? [];
  const totalEvents = events.length;
  const hasMultipleOperators = operatorSegments.filter((s) => s.operator !== '').length > 1;

  return (
    <div className="timeline">
      <div className="timeline-header">
        <h3>Timeline</h3>
        <div className="timeline-legend">
          <span className="tl-legend-item">
            <i className="tl-dot tl-setup" /> 准备
          </span>
          <span className="tl-legend-item">
            <i className="tl-dot tl-schedule" /> 调度
          </span>
          <span className="tl-legend-item">
            <i className="tl-dot tl-memory" /> 数据搬运
          </span>
          <span className="tl-legend-item">
            <i className="tl-dot tl-sync" /> 同步
          </span>
          <span className="tl-legend-item">
            <i className="tl-dot tl-compute" /> 计算
          </span>
        </div>
      </div>

      {hasMultipleOperators ? (
        <div className="timeline-operators">
          {operatorSegments.map((segment) => {
            const active = currentIndex >= segment.startIndex && currentIndex <= segment.endIndex;
            const widthPct = (segment.count / totalEvents) * 100;
            return (
              <button
                key={`${segment.operator}-${segment.startIndex}`}
                type="button"
                className={`tl-operator-seg ${active ? 'tl-operator-seg-active' : ''}`}
                style={{ width: `${widthPct}%` }}
                title={`${segment.operator}（${segment.count} 个事件，点击跳到该算子开头）`}
                onClick={() => onSeek(segment.startIndex)}
              >
                {segment.operator}
              </button>
            );
          })}
        </div>
      ) : null}

      <div className="timeline-track" role="listbox" aria-label="事件时间线">
        {events.map((event, index) => (
          <button
            key={event.id}
            type="button"
            role="option"
            aria-selected={index === currentIndex}
            className={`tl-cell ${typeClass(event.type)} ${index === currentIndex ? 'tl-current' : ''} ${index < currentIndex ? 'tl-past' : ''}`}
            title={`#${index + 1} ${event.title}`}
            onClick={() => onSeek(index)}
          />
        ))}
      </div>
    </div>
  );
}
