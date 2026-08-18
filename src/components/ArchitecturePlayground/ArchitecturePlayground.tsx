/**
 * ArchitecturePlayground — GPU 架构实验场（V0.7，实施手册 §25）。
 *
 * 让用户修改五个硬件参数（SM count / Tensor Core throughput / Shared Memory
 * size / L2 size / HBM bandwidth），实时对比"基线 vs 修改后"的算子耗时与
 * 瓶颈变化，回答：HBM 带宽翻倍 Transformer 快多少？Tensor Core 翻倍呢？
 * 为什么有些 Operator 几乎没变？
 *
 * 本组件只消费 analyzePlayground() 的投影结果，不做任何计算。
 * 所有数值为 Simulated 教学估算（面板已标注）。
 */

import type { HardwareSpec } from '../../core/perf';
import type { PlaygroundAnalysis } from '../../core/perf';
import { PLAYGROUND_SLIDER_RANGES } from '../../core/perf';
import './ArchitecturePlayground.css';

export interface ArchitecturePlaygroundProps {
  baseline: HardwareSpec;
  modified: HardwareSpec;
  onChange: (patch: Partial<HardwareSpec>) => void;
  onReset: () => void;
  analysis: PlaygroundAnalysis;
  isDirty: boolean;
}

type SliderKey = keyof typeof PLAYGROUND_SLIDER_RANGES;

const SLIDER_KEYS: SliderKey[] = ['smCount', 'tensorCoreTflops', 'smemPerSMKB', 'l2SizeMB', 'hbmBandwidthGBps'];

function formatUs(us: number): string {
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(1)} µs`;
}

export function ArchitecturePlayground(props: ArchitecturePlaygroundProps) {
  const { baseline, modified, onChange, onReset, analysis, isDirty } = props;

  return (
    <div className="playground">
      <div className="playground-header">
        <h3>Architecture Playground</h3>
        <span className="playground-note">Simulated · 教学屋顶线模型（非真实硬件）</span>
        {isDirty ? (
          <button type="button" className="playground-reset" onClick={onReset}>
            恢复默认硬件
          </button>
        ) : null}
      </div>

      <div className="playground-body">
        {/* 左侧：硬件参数滑块 */}
        <div className="playground-sliders">
          {SLIDER_KEYS.map((key) => {
            const range = PLAYGROUND_SLIDER_RANGES[key];
            const value = modified[key];
            const baseValue = baseline[key];
            const changed = value !== baseValue;
            return (
              <label key={key} className={`playground-slider ${changed ? 'playground-slider-changed' : ''}`}>
                <div className="playground-slider-head">
                  <span className="playground-slider-label">{range.label}</span>
                  <span className="playground-slider-value">
                    {value} {range.unit}
                    {changed ? <span className="playground-slider-delta">（基线 {baseValue}）</span> : null}
                  </span>
                </div>
                <input
                  type="range"
                  min={range.min}
                  max={range.max}
                  step={range.step}
                  value={value}
                  onChange={(e) => onChange({ [key]: Number(e.target.value) } as Partial<HardwareSpec>)}
                />
              </label>
            );
          })}
        </div>

        {/* 右侧：对比结果 */}
        <div className="playground-results">
          <div className="playground-total">
            <span className="playground-total-label">整条工作负载</span>
            <span className="playground-total-value">
              {formatUs(analysis.totals.baselineUs)} → {formatUs(analysis.totals.modifiedUs)}
            </span>
            <span
              className={`playground-total-speedup ${
                analysis.totals.speedup > 1.01
                  ? 'speedup-good'
                  : analysis.totals.speedup < 0.99
                    ? 'speedup-bad'
                    : ''
              }`}
            >
              {analysis.totals.speedup > 1
                ? `快 ${analysis.totals.speedup.toFixed(2)}×`
                : analysis.totals.speedup < 1
                  ? `慢 ${(1 / analysis.totals.speedup).toFixed(2)}×`
                  : '无变化'}
            </span>
          </div>

          <div className="playground-impacts">
            <div className="playground-impacts-header">
              <span>算子</span>
              <span>基线</span>
              <span>修改后</span>
              <span>加速比</span>
              <span>瓶颈</span>
            </div>
            {analysis.impacts.map((impact) => (
              <div key={`${impact.operator}-${impact.count}`} className="playground-impact-row">
                <span className="playground-impact-name" title={impact.operator}>
                  {impact.operator}
                  {impact.count > 1 ? ` ×${impact.count}` : ''}
                </span>
                <span className="playground-impact-us">{formatUs(impact.baselineUs)}</span>
                <span className="playground-impact-us">{formatUs(impact.modifiedUs)}</span>
                <span
                  className={`playground-impact-speedup ${
                    impact.speedup > 1.01 ? 'speedup-good' : impact.speedup < 0.99 ? 'speedup-bad' : ''
                  }`}
                >
                  {impact.speedup > 1
                    ? `${impact.speedup.toFixed(2)}×`
                    : impact.speedup < 1
                      ? `0.${Math.round(impact.speedup * 100).toString().padStart(2, '0')}×`
                      : '1.00×'}
                </span>
                <span className={`playground-impact-bound ${impact.boundFlipped ? 'bound-flipped' : ''}`}>
                  {impact.modifiedBound === 'compute' ? '计算密集' : '访存密集'}
                  {impact.boundFlipped
                    ? `（原${impact.baselineBound === 'compute' ? '计算' : '访存'}密集）`
                    : ''}
                </span>
              </div>
            ))}
          </div>

          <ul className="playground-takeaways">
            {analysis.takeaways.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
