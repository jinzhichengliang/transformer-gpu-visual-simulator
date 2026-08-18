/**
 * GPUView — GPU / SM / Warp / Tensor Core / Memory 视图（实施手册 §14，区域 3）。
 * 使用 SVG 渲染。只消费 TVIR 事件（经 usePlayback），不模拟 GPU。
 *
 * 高亮规则（全部来自当前事件字段）：
 *  - event.sm        → 高亮对应 SM
 *  - event.block     → 在对应 SM 上标注当前 Block
 *  - event.warp      → 高亮对应 Warp
 *  - MMA 事件        → 高亮 Tensor Core
 *  - 内存类事件      → 高亮对应内存单元（Shared Memory / Register）
 */

import type { TVIREvent } from '../../core/tvir/types';
import type { SmDisplayState } from '../../core/tvir/projection';
import './GpuView.css';

export interface GpuViewProps {
  event: TVIREvent | null;
  numSM: number;
  warpsPerBlock: number;
  smStates: SmDisplayState[];
}

interface SmSvgProps {
  smIndex: number;
  x: number;
  y: number;
  width: number;
  height: number;
  isActive: boolean;
  currentBlock: number | null;
  activeWarp: number | null;
  tensorCoreActive: boolean;
  sharedMemoryActive: boolean;
  registerActive: boolean;
  warpsPerBlock: number;
  scheduledBlocks: number[];
}

function SmSvg(props: SmSvgProps) {
  const {
    smIndex,
    x,
    y,
    width,
    height,
    isActive,
    currentBlock,
    activeWarp,
    tensorCoreActive,
    sharedMemoryActive,
    registerActive,
    warpsPerBlock,
    scheduledBlocks,
  } = props;

  const innerPad = 10;
  const warpHeight = 16;
  const warpGap = 4;
  const warpsHeight = warpsPerBlock * (warpHeight + warpGap);
  const unitHeight = 24;
  const unitGap = 6;

  let cursorY = y + 28;

  // Warps
  const warpRects = [];
  for (let w = 0; w < warpsPerBlock; w++) {
    const wy = cursorY;
    const warpActive = isActive && activeWarp === w;
    warpRects.push(
      <g key={w}>
        <rect
          x={x + innerPad}
          y={wy}
          width={width - innerPad * 2}
          height={warpHeight}
          rx={3}
          className={`gpu-unit warp ${warpActive ? 'unit-active' : ''}`}
        />
        <text
          x={x + width / 2}
          y={wy + warpHeight / 2 + 3.5}
          className="gpu-unit-text"
        >
          Warp {w}
        </text>
      </g>,
    );
  }
  cursorY += warpsHeight + unitGap;

  // Tensor Core
  const tensorY = cursorY;
  cursorY += unitHeight + unitGap;

  // Shared Memory
  const sharedY = cursorY;
  cursorY += unitHeight + unitGap;

  // Register
  const registerY = cursorY;

  const blockLabel =
    currentBlock !== null ? `Block ${currentBlock}` : scheduledBlocks.length > 0 ? `Blocks: ${scheduledBlocks.join(', ')}` : 'idle';

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={8}
        className={`sm-box ${isActive ? 'sm-active' : ''}`}
      />
      <text x={x + 10} y={y + 17} className="sm-title">
        SM {smIndex}
      </text>
      <text x={x + width - 10} y={y + 17} className="sm-block-label" textAnchor="end">
        {blockLabel}
      </text>
      {warpRects}
      <rect
        x={x + innerPad}
        y={tensorY}
        width={width - innerPad * 2}
        height={unitHeight}
        rx={3}
        className={`gpu-unit tensor-core ${tensorCoreActive ? 'unit-active' : ''}`}
      />
      <text x={x + width / 2} y={tensorY + unitHeight / 2 + 3.5} className="gpu-unit-text">
        Tensor Core
      </text>
      <rect
        x={x + innerPad}
        y={sharedY}
        width={width - innerPad * 2}
        height={unitHeight}
        rx={3}
        className={`gpu-unit shared-memory ${sharedMemoryActive ? 'unit-active' : ''}`}
      />
      <text x={x + width / 2} y={sharedY + unitHeight / 2 + 3.5} className="gpu-unit-text">
        Shared Memory
      </text>
      <rect
        x={x + innerPad}
        y={registerY}
        width={width - innerPad * 2}
        height={unitHeight}
        rx={3}
        className={`gpu-unit register ${registerActive ? 'unit-active' : ''}`}
      />
      <text x={x + width / 2} y={registerY + unitHeight / 2 + 3.5} className="gpu-unit-text">
        Register
      </text>
    </g>
  );
}

export function GpuView(props: GpuViewProps) {
  const { event, numSM, warpsPerBlock, smStates } = props;

  const svgWidth = 640;
  const smWidth = 300;
  const smGap = 20;
  const smHeight = 28 + warpsPerBlock * 20 + 3 * 30 + 18;
  const cols = 2;
  const rows = Math.ceil(numSM / cols);
  const svgHeight = rows * (smHeight + smGap) + 30;

  const activeSm = event?.sm ?? null;
  const isMemoryEvent =
    event !== null &&
    (event.type === 'MEMORY_LOAD' || event.type === 'MEMORY_MOVE' || event.type === 'MEMORY_STORE');
  const sharedMemoryActive =
    isMemoryEvent &&
    (event?.source === 'SHARED_MEMORY' || event?.destination === 'SHARED_MEMORY');
  const registerActive =
    isMemoryEvent && (event?.source === 'REGISTER' || event?.destination === 'REGISTER');
  const tensorCoreActive = event?.type === 'MMA';

  const smSvgs = [];
  for (let sm = 0; sm < numSM; sm++) {
    const col = sm % cols;
    const row = Math.floor(sm / cols);
    const smState = smStates[sm];
    smSvgs.push(
      <SmSvg
        key={sm}
        smIndex={sm}
        x={10 + col * (smWidth + smGap)}
        y={10 + row * (smHeight + smGap)}
        width={smWidth}
        height={smHeight}
        isActive={activeSm === sm}
        currentBlock={activeSm === sm ? (event?.block ?? null) : (smState?.lastBlock ?? null)}
        activeWarp={activeSm === sm ? (event?.warp ?? null) : null}
        tensorCoreActive={tensorCoreActive && activeSm === sm}
        sharedMemoryActive={sharedMemoryActive && activeSm === sm}
        registerActive={registerActive && activeSm === sm}
        warpsPerBlock={warpsPerBlock}
        scheduledBlocks={smState?.blocks ?? []}
      />,
    );
  }

  return (
    <div className="gpu-view">
      <div className="gpu-view-title">
        <h3>GPU View</h3>
        <span className="gpu-view-note">Educational simulation · not cycle-accurate</span>
      </div>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="gpu-svg">
        <rect x={2} y={2} width={svgWidth - 4} height={svgHeight - 4} rx={10} className="gpu-box" />
        <text x={svgWidth / 2} y={svgHeight - 8} className="gpu-caption" textAnchor="middle">
          GPU（{numSM} SM，每 SM 展示 {warpsPerBlock} 个 Warp）
        </text>
        {smSvgs}
      </svg>
    </div>
  );
}
