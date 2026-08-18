/**
 * CompilerView — 编译层级视图（V0.4，借鉴 tinygrad）。
 *
 * 展示当前算子的五层下钻链：
 *   Math（数学公式） → Operator（算子类型） → IR（中间表示指令）
 *   → Kernel（GPU kernel） → GPU（执行硬件单元）
 *
 * 本组件只消费 projectCompileChain() 的投影结果，
 * 不自己计算、不理解具体算子实现（见 ARCHITECTURE.md）。
 */

import type { CompileChain } from '../../core/compiler';
import './CompilerView.css';

export interface CompilerViewProps {
  chain: CompileChain | null;
}

const LAYER_LABELS = ['Math', 'Operator', 'IR', 'Kernel', 'GPU'] as const;

export function CompilerView(props: CompilerViewProps) {
  const { chain } = props;

  return (
    <div className="compiler-view">
      <div className="compiler-view-title">
        <h3>Compiler View</h3>
        <span className="compiler-note">Operator → IR → Kernel（借鉴 tinygrad）</span>
      </div>

      {chain === null ? (
        <div className="compiler-empty">
          <p>当前事件没有关联算子（operator）。</p>
          <p className="compiler-empty-hint">
            播放 GEMM / Attention / Transformer Block 仿真，即可看到每个算子从数学公式到 GPU
            硬件的完整编译下钻链。
          </p>
        </div>
      ) : (
        <div className="compiler-chain">
          {/* 当前算子名（来自事件的 operator 字段） */}
          <div className="compiler-operator-name">{chain.operator}</div>

          <div className="compiler-layers">
            {/* Layer 1: Math */}
            <div className="compiler-layer">
              <span className="compiler-layer-tag">{LAYER_LABELS[0]}</span>
              <div className="compiler-layer-body">
                <code className="compiler-math">{chain.math}</code>
              </div>
            </div>

            <div className="compiler-arrow">↓</div>

            {/* Layer 2: Operator */}
            <div className="compiler-layer">
              <span className="compiler-layer-tag">{LAYER_LABELS[1]}</span>
              <div className="compiler-layer-body">
                <span className="compiler-op-type">{chain.operatorType}</span>
              </div>
            </div>

            <div className="compiler-arrow">↓</div>

            {/* Layer 3: IR */}
            <div className="compiler-layer">
              <span className="compiler-layer-tag">{LAYER_LABELS[2]}</span>
              <div className="compiler-layer-body compiler-ir">
                {chain.ir.map((inst, index) => (
                  <div key={`${inst.op}-${index}`} className="compiler-ir-row">
                    <code className="compiler-ir-op">{inst.op}</code>
                    <span className="compiler-ir-meaning">{inst.meaning}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="compiler-arrow">↓</div>

            {/* Layer 4: Kernel */}
            <div className="compiler-layer">
              <span className="compiler-layer-tag">{LAYER_LABELS[3]}</span>
              <div className="compiler-layer-body">
                <span className="compiler-kernel-type">{chain.kernelType}</span>
                {chain.kernel ? <code className="compiler-kernel-name">{chain.kernel}</code> : null}
              </div>
            </div>

            <div className="compiler-arrow">↓</div>

            {/* Layer 5: GPU */}
            <div className="compiler-layer">
              <span className="compiler-layer-tag">{LAYER_LABELS[4]}</span>
              <div className="compiler-layer-body">
                <span className="compiler-gpu-unit">{chain.gpuUnit}</span>
              </div>
            </div>
          </div>

          {/* 瓶颈特征 */}
          <div className="compiler-bottleneck">
            <span className="compiler-bottleneck-label">瓶颈特征</span>
            <span>{chain.bottleneck}</span>
          </div>
        </div>
      )}
    </div>
  );
}
