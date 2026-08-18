/**
 * InstructionView — SASS 指令级视图（V0.8，实施手册 §26）。
 *
 * 展示 NVBit 采集的 SASS 指令流（按 Warp 分组），高亮当前播放到的指令，
 * 点击任意指令可 seek 到对应事件。
 *
 * 定位声明（手册 §26 硬性要求）：
 *   Educational Simulation ≠ Architecture Simulation。
 *   本视图呈现指令顺序与数据通路（教学），不是 Accel-Sim 的
 *   cycle-accurate 微架构仿真——面板顶部必须常驻该声明。
 *
 * 本组件只消费 projectSassInstructions() 的投影结果，不理解指令语义。
 */

import type { SassInstructionRow } from '../../core/tvir';
import './InstructionView.css';

export interface InstructionViewProps {
  rows: SassInstructionRow[];
  currentIndex: number;
  onSeek: (index: number) => void;
  /** 是否为示例数据（true 时标注"示例数据"） */
  isSample: boolean;
}

/** 指令类别的显示标签（与 sassParser.SASS_CATEGORY_LABELS 保持一致） */
const CATEGORY_LABELS: Record<string, string> = {
  'global-memory': '全局内存',
  'shared-memory': '共享内存',
  'async-copy': '异步拷贝',
  'tensor-core': 'Tensor Core',
  'cuda-core': 'CUDA Core',
  sync: '同步',
  'address-calc': '地址计算',
  control: '控制流',
};

/** 图例中展示的类别（按教学重要性排序） */
const LEGEND_ORDER = [
  'global-memory',
  'shared-memory',
  'async-copy',
  'tensor-core',
  'cuda-core',
  'sync',
  'address-calc',
  'control',
] as const;

export function InstructionView(props: InstructionViewProps) {
  const { rows, currentIndex, onSeek, isSample } = props;

  if (rows.length === 0) return null;

  // 按 Warp 分组（保持 trace 中的出现顺序）
  const warpGroups: Array<{ warp: number; rows: SassInstructionRow[] }> = [];
  for (const row of rows) {
    const last = warpGroups[warpGroups.length - 1];
    if (last && last.warp === row.warp) {
      last.rows.push(row);
    } else {
      warpGroups.push({ warp: row.warp, rows: [row] });
    }
  }

  return (
    <div className="instruction-view">
      <div className="iv-header">
        <h3>SASS 指令级视图</h3>
        <span className="iv-provenance">
          {isSample ? '示例指令流（教学示意编排）' : 'NVBit 采集的指令流'}
        </span>
      </div>

      <div className="iv-disclaimer">
        Educational Simulation ≠ Architecture Simulation：本视图呈现指令顺序与数据通路
        （教学用途），不是 Accel-Sim 的 cycle-accurate 微架构仿真，指令间没有真实周期/流水线信息。
      </div>

      <div className="iv-legend">
        {LEGEND_ORDER.map((category) => (
          <span key={category} className={`iv-legend-item cat-${category}`}>
            {CATEGORY_LABELS[category] ?? category}
          </span>
        ))}
      </div>

      <div className="iv-body">
        {warpGroups.map((group) => (
          <div key={group.warp} className="iv-warp">
            <div className="iv-warp-title">Warp {group.warp}</div>
            <table className="iv-table">
              <thead>
                <tr>
                  <th className="iv-col-pc">PC</th>
                  <th className="iv-col-opcode">指令</th>
                  <th className="iv-col-operands">操作数</th>
                  <th className="iv-col-category">类别</th>
                </tr>
              </thead>
              <tbody>
                {group.rows.map((row) => {
                  const active = currentIndex === row.eventIndex;
                  return (
                    <tr
                      key={row.eventIndex}
                      className={`iv-row cat-${row.category} ${active ? 'iv-row-active' : ''}`}
                      onClick={() => onSeek(row.eventIndex)}
                    >
                      <td className="iv-col-pc">{row.pc}</td>
                      <td className="iv-col-opcode">{row.opcode}</td>
                      <td className="iv-col-operands">{row.operands ?? '—'}</td>
                      <td className="iv-col-category">
                        <span className={`iv-badge cat-${row.category}`}>
                          {CATEGORY_LABELS[row.category] ?? row.category}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
