/**
 * MemoryView — 内存层级视图（实施手册 §15，区域 4）。
 *
 * 展示 HBM → L2 → L1 → Shared Memory → Register → Tensor Core 的教学层级，
 * 当前事件为内存类事件时，在 source 与 destination 之间播放搬运动画。
 *
 * 注意（CONCEPTS.md）：这是教学简化视图，真实 GPU 的数据通路远比
 * 单一路径复杂，组件内已显式声明。
 *
 * 本组件只消费 TVIR 事件，不维护执行状态。
 */

import type { MemoryLevel, TVIREvent } from '../../core/tvir/types';
import './MemoryView.css';

export interface MemoryViewProps {
  event: TVIREvent | null;
}

interface LevelDef {
  key: MemoryLevel | 'TENSOR_CORE';
  label: string;
  detail: string;
}

const LEVELS: LevelDef[] = [
  { key: 'HBM', label: 'HBM', detail: '显存 · 容量最大 · 带宽瓶颈' },
  { key: 'L2', label: 'L2 Cache', detail: '全 GPU 共享缓存' },
  { key: 'L1', label: 'L1 Cache', detail: 'SM 内缓存（与 SMEM 共享 SRAM）' },
  { key: 'SHARED_MEMORY', label: 'Shared Memory', detail: 'SM 片上 · Block 内共享' },
  { key: 'REGISTER', label: 'Register', detail: '线程私有 · 最快' },
  { key: 'TENSOR_CORE', label: 'Tensor Core', detail: 'MMA 计算单元' },
];

function isMemoryEvent(event: TVIREvent): boolean {
  return (
    event.type === 'MEMORY_LOAD' ||
    event.type === 'MEMORY_MOVE' ||
    event.type === 'MEMORY_STORE'
  );
}

export function MemoryView(props: MemoryViewProps) {
  const { event } = props;

  const active = event !== null && isMemoryEvent(event);
  const sourceKey = active ? event.source : undefined;
  const destKey = active ? event.destination : undefined;

  const sourceIndex = sourceKey ? LEVELS.findIndex((l) => l.key === sourceKey) : -1;
  const destIndex = destKey ? LEVELS.findIndex((l) => l.key === destKey) : -1;

  // SVG 布局
  const boxWidth = 200;
  const boxHeight = 44;
  const gap = 26;
  const svgWidth = 320;
  const topPad = 10;
  const svgHeight = topPad + LEVELS.length * boxHeight + (LEVELS.length - 1) * gap + 20;
  const centerX = svgWidth / 2;

  const yOf = (index: number) => topPad + index * (boxHeight + gap);

  // 搬运动画路径：非相邻层级走右侧弧线，相邻层级走中间直线
  const movePath =
    active && sourceIndex >= 0 && destIndex >= 0 && sourceIndex !== destIndex
      ? (() => {
          const y1 = yOf(sourceIndex) + boxHeight / 2;
          const y2 = yOf(destIndex) + boxHeight / 2;
          if (Math.abs(destIndex - sourceIndex) === 1) {
            return `M ${centerX} ${y1} L ${centerX} ${y2}`;
          }
          const bulge = centerX + boxWidth / 2 + 34;
          return `M ${centerX + boxWidth / 2} ${y1} Q ${bulge} ${(y1 + y2) / 2} ${centerX + boxWidth / 2} ${y2}`;
        })()
      : null;

  const movingLabel = event?.tile?.label ?? event?.tensor ?? 'data';

  return (
    <div className="memory-view">
      <div className="memory-view-title">
        <h3>Memory View</h3>
        <span className="memory-note">教学层级视图 · 真实数据通路更复杂</span>
      </div>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="memory-svg">
        {/* 层级间连接箭头 */}
        {LEVELS.map((level, i) =>
          i < LEVELS.length - 1 ? (
            <g key={`arrow-${level.key}`}>
              <line
                x1={centerX}
                y1={yOf(i) + boxHeight}
                x2={centerX}
                y2={yOf(i + 1)}
                className="memory-link"
              />
              <polygon
                points={`${centerX - 4},${yOf(i + 1) - 6} ${centerX + 4},${yOf(i + 1) - 6} ${centerX},${yOf(i + 1) - 1}`}
                className="memory-link-arrow"
              />
            </g>
          ) : null,
        )}

        {/* 层级盒子 */}
        {LEVELS.map((level, i) => {
          const isSource = level.key === sourceKey;
          const isDest = level.key === destKey;
          const cls = `memory-level ${isSource ? 'level-source' : ''} ${isDest ? 'level-dest' : ''}`;
          return (
            <g key={level.key}>
              <rect
                x={centerX - boxWidth / 2}
                y={yOf(i)}
                width={boxWidth}
                height={boxHeight}
                rx={6}
                className={cls}
              />
              <text x={centerX} y={yOf(i) + 18} className="memory-level-label">
                {level.label}
              </text>
              <text x={centerX} y={yOf(i) + 33} className="memory-level-detail">
                {level.detail}
              </text>
            </g>
          );
        })}

        {/* 搬运动画 */}
        {movePath !== null ? (
          <g>
            <path d={movePath} className="memory-move-path" />
            <circle r={5} className="memory-move-packet">
              <animateMotion dur="1.1s" repeatCount="indefinite" path={movePath} />
            </circle>
            <text className="memory-move-label">
              {movingLabel}
              <animateMotion dur="1.1s" repeatCount="indefinite" path={movePath} />
            </text>
          </g>
        ) : null}
      </svg>

      {active ? (
        <div className="memory-current">
          <span className="memory-current-route">
            {sourceKey} → {destKey}
          </span>
          <span className="memory-current-what">{event?.what}</span>
        </div>
      ) : (
        <div className="memory-current memory-current-idle">
          当前步骤没有数据搬运（或为非内存事件）
        </div>
      )}
    </div>
  );
}
