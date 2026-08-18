/**
 * MultiGpuView — 多 GPU 并行视图（实施手册 §27，V0.9）。
 *
 * 以环形拓扑展示 GPU0..N-1：
 *   - 计算事件（Compute）：高亮正在计算的 GPU；
 *   - 通信事件（Communication）：按 metadata.comm.transfers 绘制 GPU 间传输箭头，
 *     展示 AllReduce（ReduceScatter + AllGather）/ AllGather / ReduceScatter / P2P。
 *
 * 借鉴 Chakra + ASTRA-sim 的解析式建模：呈现通信模式与教学耗时估算，
 * 不模拟网络细节（见 CONCEPTS.md V0.9）。
 *
 * 本组件只消费当前 TVIR 事件（经 usePlayback），不模拟 GPU，不做计算。
 */

import type { TVIREvent } from '../../core/tvir/types';
import type { CommMeta } from '../../core/multigpu/commPrimitives';
import './MultiGpuView.css';

export interface MultiGpuViewProps {
  event: TVIREvent | null;
  numGpus: number;
}

/** 从事件中提取"正在计算"的 GPU 集合（计算事件用） */
function extractComputeGpus(event: TVIREvent | null, numGpus: number): number[] {
  if (!event) return [];
  // 优先：metadata.multigpu.gpu（DP 本地计算）
  const multigpu = event.metadata as { multigpu?: { gpu?: number } } | undefined;
  if (multigpu?.multigpu?.gpu !== undefined) {
    return [multigpu.multigpu.gpu];
  }
  // 其次：operator/kernel 名中显式带 GPU{n}（TP 分片 / PP 阶段）
  const text = `${event.operator ?? ''} ${event.kernel ?? ''}`;
  const match = text.match(/GPU\s*(\d+)/i);
  if (match) {
    const gpu = Number(match[1]);
    if (Number.isFinite(gpu) && gpu >= 0 && gpu < numGpus) return [gpu];
  }
  return [];
}

/** 通信事件元信息读取 */
function readCommMeta(event: TVIREvent | null): CommMeta | null {
  const comm = (event?.metadata as { comm?: CommMeta } | undefined)?.comm;
  return comm ?? null;
}

const COLLECTIVE_LABELS: Record<string, string> = {
  allreduce: 'AllReduce',
  allgather: 'AllGather',
  reduce_scatter: 'ReduceScatter',
  p2p: 'P2P',
  broadcast: 'Broadcast',
};

export function MultiGpuView(props: MultiGpuViewProps) {
  const { event, numGpus } = props;

  const comm = readCommMeta(event);
  const isCommEvent = comm !== null && (comm.transfers.length > 0 || comm.collective === 'allreduce');
  const computeGpus = isCommEvent ? [] : extractComputeGpus(event, numGpus);

  // 环形布局
  const svgSize = 420;
  const cx = svgSize / 2;
  const cy = svgSize / 2;
  const radius = 130;
  const nodeR = 42;

  const gpuPos = Array.from({ length: numGpus }, (_, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / numGpus;
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  // 发送/接收状态
  const sending = new Set<number>();
  const receiving = new Set<number>();
  if (comm) {
    for (const t of comm.transfers) {
      sending.add(t.from);
      receiving.add(t.to);
    }
  }

  // 通信箭头（沿环的弧线，端点裁剪到节点边缘）
  const arrows = comm
    ? comm.transfers.map((t, idx) => {
        const from = gpuPos[t.from];
        const to = gpuPos[t.to];
        if (!from || !to) return null;
        // 端点向节点边缘收缩，避免箭头被节点遮挡
        const edge = nodeR + 4;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const len = Math.max(Math.hypot(dx, dy), 1);
        const startX = from.x + (dx / len) * edge;
        const startY = from.y + (dy / len) * edge;
        const endX = to.x - (dx / len) * edge;
        const endY = to.y - (dy / len) * edge;
        // 控制点：向环外凸出
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        const dirX = midX - cx;
        const dirY = midY - cy;
        const dirLen = Math.max(Math.hypot(dirX, dirY), 1);
        const bulge = 30;
        const ctrlX = midX + (dirX / dirLen) * bulge;
        const ctrlY = midY + (dirY / dirLen) * bulge;
        return (
          <path
            key={`arrow-${idx}`}
            d={`M ${startX} ${startY} Q ${ctrlX} ${ctrlY} ${endX} ${endY}`}
            className="mg-arrow"
            fill="none"
          />
        );
      })
    : [];

  const commTitle = comm
    ? `${COLLECTIVE_LABELS[comm.collective] ?? comm.collective}${
        comm.phase === 'reduce_scatter'
          ? ' · ReduceScatter'
          : comm.phase === 'allgather'
            ? ' · AllGather'
            : ''
      }${comm.totalRingSteps > 0 ? `（第 ${comm.ringStep + 1}/${comm.totalRingSteps} 步）` : ''}`
    : null;

  return (
    <div className="multigpu-view">
      <div className="multigpu-view-title">
        <h3>Multi-GPU View</h3>
        <span className="multigpu-view-note">Educational simulation · 解析式通信建模（Simulated）</span>
      </div>

      <div className="multigpu-status-line">
        {commTitle ? (
          <span className="multigpu-comm-badge">{commTitle}</span>
        ) : computeGpus.length > 0 ? (
          <span className="multigpu-compute-badge">
            Compute · GPU {computeGpus.join(', ')}
          </span>
        ) : (
          <span className="multigpu-idle-badge">
            {event ? event.title : '等待事件'}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${svgSize} ${svgSize}`} className="multigpu-svg">
        <defs>
          <marker
            id="mg-arrowhead"
            markerWidth="8"
            markerHeight="8"
            refX="6"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M 0 0 L 6 3 L 0 6 Z" fill="var(--accent)" />
          </marker>
        </defs>
        {/* 环参考线 */}
        <circle cx={cx} cy={cy} r={radius} className="mg-ring" fill="none" />
        {arrows}
        {gpuPos.map((pos, i) => {
          const isSending = sending.has(i);
          const isReceiving = receiving.has(i);
          const isComputing = computeGpus.includes(i);
          const stateClass = isComputing
            ? 'mg-node-computing'
            : isSending
              ? 'mg-node-sending'
              : isReceiving
                ? 'mg-node-receiving'
                : '';
          const stateLabel = isComputing
            ? 'Compute'
            : isSending
              ? 'Send'
              : isReceiving
                ? 'Recv'
                : 'idle';
          return (
            <g key={`gpu-${i}`}>
              <circle cx={pos.x} cy={pos.y} r={nodeR} className={`mg-node ${stateClass}`} />
              <text x={pos.x} y={pos.y - 4} className="mg-node-label" textAnchor="middle">
                GPU {i}
              </text>
              <text x={pos.x} y={pos.y + 14} className="mg-node-state" textAnchor="middle">
                {stateLabel}
              </text>
            </g>
          );
        })}
        <text x={cx} y={cy} className="mg-center-label" textAnchor="middle">
          {numGpus} GPUs
        </text>
      </svg>

      {comm ? (
        <div className="multigpu-comm-detail">
          <span className="multigpu-detail-item">
            每 GPU 传输：{formatBytesShort(comm.bytesPerTransfer)}
          </span>
          <span className="multigpu-detail-item">本步耗时 ≈ {comm.durationUs.toFixed(1)} µs（估算）</span>
          <span className="multigpu-detail-item">{comm.label}</span>
        </div>
      ) : null}
    </div>
  );
}

function formatBytesShort(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes.toFixed(0)} B`;
}
