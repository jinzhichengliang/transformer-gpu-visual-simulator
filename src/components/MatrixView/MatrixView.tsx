/**
 * MatrixView — A × B = C 矩阵视图（实施手册 §13，区域 2）。
 * 用 SVG 渲染矩阵网格与 Tile 划分；当前 TVIR 事件包含 tile 时高亮对应 Tile。
 *
 * 本组件只读取 TVIR 状态，禁止：自己计算 simulation、决定下一步、维护执行状态。
 */

import './MatrixView.css';

export interface MatrixViewProps {
  M: number;
  N: number;
  K: number;
  tileM: number;
  tileN: number;
  tileK: number;
  /** 当前事件中涉及的 tile 标签集合，如 ["A[0,1]", "B[1,2]"] */
  activeTiles: string[];
  /** 当前事件涉及的张量名（用于强调） */
  activeTensor?: string | undefined;
  /** 左矩阵标签（默认 "A"，Attention 场景下为 "Q"/"X" 等） */
  leftLabel?: string | undefined;
  /** 右矩阵标签（默认 "B"） */
  rightLabel?: string | undefined;
  /** 输出矩阵标签（默认 "C"） */
  outLabel?: string | undefined;
}

interface SingleMatrixProps {
  label: string;
  rows: number;
  cols: number;
  tileRows: number;
  tileCols: number;
  tileSize: number;
  activeTiles: string[];
  emphasized: boolean;
}

function SingleMatrix(props: SingleMatrixProps) {
  const { label, rows, cols, tileRows, tileCols, tileSize, activeTiles, emphasized } = props;

  // 渲染尺寸上限（px）：大模型维度（如 4096/8192）下按此上限等比缩小，
  // 保证矩阵整体完整显示在 Matrix View 区域内，不溢出、不变形。
  const MAX_RENDER_SIZE = 260;
  const size = Math.min(tileSize, MAX_RENDER_SIZE / Math.max(rows, cols, 1));
  // 单元格过小时网格线密度过高（大维度下会渲染上万条线），不画线
  const showGrid = size >= 0.75;
  const width = cols * size;
  const height = rows * size;
  // 标签移到 HTML 渲染（不随 SVG 缩放），SVG 仅画矩阵本体
  const pad = 6;
  const svgWidth = width + pad * 2;
  const svgHeight = height + pad * 2;

  const tilesRow = Math.ceil(rows / tileRows);
  const tilesCol = Math.ceil(cols / tileCols);

  // Tile 数量过多（大维度下可达上万）时，只渲染当前激活的 Tile，
  // 避免一次性渲染海量矩形导致卡顿；高亮定位功能不受影响。
  const totalTiles = tilesRow * tilesCol;
  const renderAllTiles = totalTiles <= 512;

  const tileRects = [];
  for (let tr = 0; tr < tilesRow; tr++) {
    for (let tc = 0; tc < tilesCol; tc++) {
      const tileLabel = `${label}[${tr},${tc}]`;
      const isActive = activeTiles.includes(tileLabel);
      if (!renderAllTiles && !isActive) continue;
      tileRects.push(
        <rect
          key={tileLabel}
          x={pad + tc * tileCols * size}
          y={pad + tr * tileRows * size}
          width={Math.min(tileCols, cols - tc * tileCols) * size}
          height={Math.min(tileRows, rows - tr * tileRows) * size}
          className={`matrix-tile ${isActive ? 'matrix-tile-active' : ''}`}
        >
          <title>{tileLabel}</title>
        </rect>,
      );
    }
  }

  const gridLines = [];
  if (showGrid) {
    for (let c = 1; c < cols; c++) {
      gridLines.push(
        <line
          key={`v${c}`}
          x1={pad + c * size}
          y1={pad}
          x2={pad + c * size}
          y2={pad + height}
          className="matrix-grid-line"
        />,
      );
    }
    for (let r = 1; r < rows; r++) {
      gridLines.push(
        <line
          key={`h${r}`}
          x1={pad}
          y1={pad + r * size}
          x2={pad + width}
          y2={pad + r * size}
          className="matrix-grid-line"
        />,
      );
    }
  }

  return (
    <div className={`single-matrix ${emphasized ? 'single-matrix-emphasized' : ''}`}>
      <div className="matrix-label">{label} [{rows}×{cols}]</div>
      <svg
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        preserveAspectRatio="xMidYMid meet"
        className="matrix-svg"
      >
        <rect x={pad} y={pad} width={width} height={height} className="matrix-body" />
        {tileRects}
        {gridLines}
      </svg>
      <div className="matrix-tile-info">tile {tileRows}×{tileCols} → {tilesRow}×{tilesCol} 块</div>
    </div>
  );
}

export function MatrixView(props: MatrixViewProps) {
  const { M, N, K, tileM, tileN, tileK, activeTiles, activeTensor } = props;
  const leftLabel = props.leftLabel ?? 'A';
  const rightLabel = props.rightLabel ?? 'B';
  const outLabel = props.outLabel ?? 'C';

  // 128×128 下每格约 2px，视觉清晰
  const scale = Math.max(1.2, Math.min(2.4, 260 / Math.max(M, N, K)));

  return (
    <div className="matrix-view">
      <div className="matrix-view-title">
        <h3>Matrix View</h3>
        <span className="matrix-formula">
          {outLabel}[{M}×{N}] = {leftLabel}[{M}×{K}] × {rightLabel}[{K}×{N}]
        </span>
      </div>
      <div className="matrix-row">
        <SingleMatrix
          label={leftLabel}
          rows={M}
          cols={K}
          tileRows={tileM}
          tileCols={tileK}
          tileSize={scale}
          activeTiles={activeTiles}
          emphasized={activeTensor === leftLabel}
        />
        <span className="matrix-op">×</span>
        <SingleMatrix
          label={rightLabel}
          rows={K}
          cols={N}
          tileRows={tileK}
          tileCols={tileN}
          tileSize={scale}
          activeTiles={activeTiles}
          emphasized={activeTensor === rightLabel}
        />
        <span className="matrix-op">=</span>
        <SingleMatrix
          label={outLabel}
          rows={M}
          cols={N}
          tileRows={tileM}
          tileCols={tileN}
          tileSize={scale}
          activeTiles={activeTiles}
          emphasized={activeTensor === outLabel}
        />
      </div>
    </div>
  );
}
