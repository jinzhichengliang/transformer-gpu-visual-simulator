/**
 * KernelTimeline — 真实 GPU kernel 时间轴视图（V0.5）。
 *
 * 按实测时间轴（横轴 = 时间 µs）展示每个 kernel 的执行区间，
 * 点击任意 kernel 条可 seek 到对应的 KERNEL_LAUNCH 事件。
 *
 * 数据可信度（CONCEPTS.md 规则 8）：
 * - 真实 trace：时间轴为 Measured（实测）数据
 * - 示例 trace：明确标注为示例数据（教学示意值），不标 Measured
 *
 * 本组件只消费 projectKernelTimeline() 的投影结果。
 */

import type { KernelTimelineSegment } from '../../core/tvir';
import './KernelTimeline.css';

export interface KernelTimelineProps {
  segments: KernelTimelineSegment[];
  currentIndex: number;
  onSeek: (index: number) => void;
  /** 是否为示例数据（true 时标注"示例数据"而非 Measured） */
  isSample: boolean;
}

export function KernelTimeline(props: KernelTimelineProps) {
  const { segments, currentIndex, onSeek, isSample } = props;

  if (segments.length === 0) return null;

  const endUs = Math.max(...segments.map((s) => s.startUs + s.durationUs));
  const totalUs = Math.max(endUs, 1);

  // SVG 布局
  const width = 900;
  const leftPad = 150;
  const rightPad = 16;
  const rowHeight = 26;
  const rowGap = 4;
  const topPad = 26;
  const height = topPad + segments.length * (rowHeight + rowGap) + 22;
  const trackWidth = width - leftPad - rightPad;

  const xOf = (us: number) => leftPad + (us / totalUs) * trackWidth;

  // 时间刻度（5 等分）
  const ticks = [];
  for (let i = 0; i <= 5; i++) {
    const us = (totalUs * i) / 5;
    ticks.push(
      <g key={`tick-${i}`}>
        <line
          x1={xOf(us)}
          y1={topPad - 4}
          x2={xOf(us)}
          y2={height - 18}
          className="kt-gridline"
        />
        <text x={xOf(us)} y={topPad - 10} className="kt-tick-label" textAnchor="middle">
          {us.toFixed(0)} µs
        </text>
      </g>,
    );
  }

  const bars = segments.map((segment) => {
    const y = topPad + segments.indexOf(segment) * (rowHeight + rowGap);
    const x = xOf(segment.startUs);
    const barWidth = Math.max((segment.durationUs / totalUs) * trackWidth, 3);
    const active = currentIndex === segment.eventIndex;
    const label = segment.operator ?? segment.kernel;

    return (
      <g key={`${segment.eventIndex}-${segment.kernel}`}>
        <text
          x={leftPad - 8}
          y={y + rowHeight / 2 + 3.5}
          className="kt-kernel-label"
          textAnchor="end"
        >
          {label.length > 22 ? `${label.slice(0, 21)}…` : label}
        </text>
        <rect
          x={x}
          y={y}
          width={barWidth}
          height={rowHeight}
          rx={3}
          className={`kt-bar ${active ? 'kt-bar-active' : ''}`}
          onClick={() => onSeek(segment.eventIndex)}
        >
          <title>
            {segment.kernel}
            {'\n'}start: {segment.startUs.toFixed(1)} µs · duration: {segment.durationUs.toFixed(1)} µs
            {'\n'}（点击跳转到对应事件）
          </title>
        </rect>
        {barWidth > 46 ? (
          <text
            x={x + barWidth / 2}
            y={y + rowHeight / 2 + 3.5}
            className="kt-bar-duration"
            textAnchor="middle"
          >
            {segment.durationUs.toFixed(0)}µs
          </text>
        ) : null}
      </g>
    );
  });

  return (
    <div className="kernel-timeline">
      <div className="kernel-timeline-header">
        <h3>Kernel Timeline</h3>
        <span className="kt-provenance">
          {isSample ? '示例数据（教学示意值，非实测）' : 'Measured · 来自真实 GPU profiler'}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="kt-svg">
        {ticks}
        {bars}
        <text x={width / 2} y={height - 4} className="kt-axis-label" textAnchor="middle">
          时间轴（µs，相对 trace 起点）
        </text>
      </svg>
    </div>
  );
}
