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

  const size = tileSize;
  const width = cols * size;
  const height = rows * size;
  const pad = 26;
  const svgWidth = width + pad * 2;
  const svgHeight = height + pad * 2 + 18;

  const tilesRow = Math.ceil(rows / tileRows);
  const tilesCol = Math.ceil(cols / tileCols);

  const tileRects = [];
  for (let tr = 0; tr < tilesRow; tr++) {
    for (let tc = 0; tc < tilesCol; tc++) {
      const tileLabel = `${label}[${tr},${tc}]`;
      const isActive = activeTiles.includes(tileLabel);
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

  return (
    <div className={`single-matrix ${emphasized ? 'single-matrix-emphasized' : ''}`}>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="matrix-svg">
        <rect x={pad} y={pad} width={width} height={height} className="matrix-body" />
        {tileRects}
        {gridLines}
        <text x={pad + width / 2} y={pad - 10} className="matrix-label">
          {label} [{rows}×{cols}]
        </text>
        <text x={pad + width / 2} y={pad + height + 14} className="matrix-tile-info">
          tile {tileRows}×{tileCols} → {tilesRow}×{tilesCol} 块
        </text>
      </svg>
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
