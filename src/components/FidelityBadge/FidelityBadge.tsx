/**
 * FidelityBadge UI — Visual Fidelity Badge 卡片（Sprint 17, Task J1/J2）。
 *
 * 以分类行展示模型数据的可信来源，并显式声明"教学仿真，非实测"，
 * 杜绝测量级精度造成的虚假精确错觉。
 */

import { buildFidelityBadge } from '../../core/fidelity';
import type { ModelProfile } from '../../core/modelprofile';
import './FidelityBadge.css';

interface FidelityBadgeProps {
  profile: ModelProfile | null;
}

export function FidelityBadge({ profile }: FidelityBadgeProps) {
  if (!profile) {
    return (
      <div className="fidelity-badge fidelity-badge-empty">
        <span className="fidelity-empty-text">选择模型后显示数据可信度</span>
      </div>
    );
  }

  const summary = buildFidelityBadge(profile);
  return (
    <div className="fidelity-badge" role="group" aria-label="数据可信度标识">
      <div className="fidelity-header">
        <span className="fidelity-model">{summary.modelName}</span>
        <span className="fidelity-level">Fidelity {summary.fidelity}</span>
      </div>
      <div className="fidelity-rows">
        {summary.rows.map((row) => (
          <div key={row.category} className="fidelity-row">
            <span className="fidelity-category">{row.category}:</span>
            <span className={`fidelity-label${row.weak ? ' fidelity-label-weak' : ''}`}>
              {row.label}
              {row.weak ? ' ⚠' : ''}
            </span>
          </div>
        ))}
      </div>
      <p className="fidelity-disclaimer">{summary.disclaimer}</p>
    </div>
  );
}
