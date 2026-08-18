/**
 * TensorCoreView — Tensor Core MMA 视图（实施手册 §16，区域 5 的扩展）。
 *
 * 展示 D = A × B + C 的 fragment 数据流：
 *   Input: A fragment / B fragment（来自寄存器）
 *   Accumulator: C fragment
 *   Output: D fragment
 *
 * MMA 事件时高亮。本组件只消费 TVIR，不模拟电路。
 */

import type { TVIREvent } from '../../core/tvir/types';
import './TensorCoreView.css';

export interface TensorCoreViewProps {
  event: TVIREvent | null;
}

export function TensorCoreView(props: TensorCoreViewProps) {
  const { event } = props;
  const mmaActive = event?.type === 'MMA';
  const accActive = event?.type === 'ACCUMULATE';
  const active = mmaActive || accActive;

  return (
    <div className={`tensor-core-view ${active ? 'tc-active' : ''}`}>
      <div className="tc-title">
        <h3>Tensor Core MMA</h3>
        <span className="tc-formula">D = A × B + C</span>
      </div>

      <div className="tc-diagram">
        <div className="tc-inputs">
          <div className={`tc-fragment tc-frag-a ${mmaActive ? 'tc-frag-active' : ''}`}>
            A fragment
            <span>Input（寄存器）</span>
          </div>
          <div className={`tc-fragment tc-frag-b ${mmaActive ? 'tc-frag-active' : ''}`}>
            B fragment
            <span>Input（寄存器）</span>
          </div>
        </div>

        <div className="tc-core">
          <div className={`tc-unit ${mmaActive ? 'tc-unit-firing' : ''}`}>
            Tensor Core
            <span>MMA</span>
          </div>
          <div className="tc-arrow">→</div>
        </div>

        <div className="tc-acc-output">
          <div className={`tc-fragment tc-frag-c ${accActive ? 'tc-frag-active' : ''}`}>
            C fragment
            <span>Accumulator（部分和）</span>
          </div>
          <div className="tc-arrow-down">＋</div>
          <div className={`tc-fragment tc-frag-d ${accActive ? 'tc-frag-active' : ''}`}>
            D fragment
            <span>Output</span>
          </div>
        </div>
      </div>

      <p className="tc-note">
        操作数必须来自寄存器 fragment，Tensor Core 不能直接从 HBM/Shared Memory 取数
        （CONCEPTS.md 规则 5）。
      </p>
    </div>
  );
}
