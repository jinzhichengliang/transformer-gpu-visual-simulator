/**
 * PerfPanel — 性能分析视图（V0.6，实施手册 §24）。
 *
 * 展示六项指标：Kernel duration / Tensor Core utilization / Memory bandwidth /
 * L2 hit rate / Occupancy / Arithmetic intensity。
 *
 * 数据可信度铁律（CONCEPTS.md 规则 8/26/30）：
 * 每项指标的 source 徽标（Measured/Estimated/Simulated/N-A）原样渲染，
 * 面板顶部另按 trace 数据类别给出总标注；示例数据绝不标 Measured。
 *
 * 本组件只消费 computePerfReport() 的投影结果，不做任何计算。
 */

import type { PerfMetric, PerfReport, MetricSource } from '../../core/perf';
import './PerfPanel.css';

export interface PerfPanelProps {
  report: PerfReport;
}

const SOURCE_LABELS: Record<MetricSource, string> = {
  measured: 'Measured',
  estimated: 'Estimated',
  simulated: 'Simulated',
  unavailable: 'N/A',
};

function SourceBadge({ source }: { source: MetricSource }) {
  return <span className={`perf-badge perf-badge-${source}`}>{SOURCE_LABELS[source]}</span>;
}

function formatValue(metric: PerfMetric): string {
  if (metric.value === null) return '—';
  const abs = Math.abs(metric.value);
  if (abs >= 1000) return metric.value.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (abs >= 100) return metric.value.toFixed(1);
  if (abs >= 1) return metric.value.toFixed(2);
  return metric.value.toFixed(3);
}

export function PerfPanel(props: PerfPanelProps) {
  const { report } = props;
  const { scopeLabel, dataClass, metrics, breakdown, totals, contextNote } = report;

  const dataClassLabel =
    dataClass === 'measured'
      ? 'Measured · 真实 GPU 实测'
      : dataClass === 'sample'
        ? '示例数据 · 教学示意值（非实测）'
        : 'Simulated · 教学仿真（屋顶线模型 + 假设参数）';

  const maxShareUs = breakdown.length > 0 ? (breakdown[0]?.totalUs ?? 0) : 0;

  return (
    <div className="perf-panel">
      <div className="perf-panel-header">
        <h3>Performance Analysis</h3>
        <span className={`perf-dataclass perf-dataclass-${dataClass}`}>{dataClassLabel}</span>
      </div>

      <div className="perf-scope">作用域：{scopeLabel}</div>

      <div className="perf-metrics-grid">
        {metrics.map((metric) => (
          <div key={metric.key} className={`perf-card ${metric.value === null ? 'perf-card-na' : ''}`}>
            <div className="perf-card-head">
              <span className="perf-card-label">{metric.label}</span>
              <SourceBadge source={metric.source} />
            </div>
            <div className="perf-card-value">
              {formatValue(metric)}
              <span className="perf-card-unit">{metric.value !== null ? ` ${metric.unit}` : ''}</span>
            </div>
            <p className="perf-card-note">{metric.note}</p>
          </div>
        ))}
      </div>

      {breakdown.length > 0 && totals ? (
        <div className="perf-breakdown">
          <div className="perf-breakdown-title">
            按算子的耗时分布（{totals.kernelCount} 个 kernel，合计 {totals.totalUs.toFixed(1)} µs）
          </div>
          {breakdown.map((share) => {
            const pct = totals.totalUs > 0 ? (share.totalUs / totals.totalUs) * 100 : 0;
            const barWidth = maxShareUs > 0 ? (share.totalUs / maxShareUs) * 100 : 0;
            return (
              <div key={share.operator} className="perf-breakdown-row">
                <span className="perf-breakdown-name" title={share.operator}>
                  {share.operator}
                </span>
                <div className="perf-breakdown-bar-wrap">
                  <div className="perf-breakdown-bar" style={{ width: `${barWidth}%` }} />
                </div>
                <span className="perf-breakdown-stats">
                  {share.totalUs.toFixed(1)} µs · {pct.toFixed(1)}% · ×{share.kernelCount}
                </span>
              </div>
            );
          })}
        </div>
      ) : null}

      <p className="perf-context-note">{contextNote}</p>
    </div>
  );
}
