/**
 * ModelOverview — 模型执行位置总览（Sprint 7, Task F3）。
 *
 * 任务书要求：让用户先理解"现在运行的是哪个模型的哪一部分"：
 *   Model → Layers → Current Layer → Current Operator
 * 当前步骤变化时自动联动（只消费当前 TVIR 事件，不维护执行状态）。
 *
 * 附带 Fidelity/Source 标识（Sprint 17 Task J1）：
 *   显示 Execution Fidelity 与来源状态，Simulated 数据绝不标成 Measured。
 */

import type { TVIREvent } from '../../core/tvir/types';
import type { ModelProfile } from '../../core/modelprofile';
import {
  projectModelContext,
  projectModelBreadcrumb,
} from '../../core/model';
import './ModelOverview.css';

interface ModelOverviewProps {
  event: TVIREvent | null;
  /** 当前选中的模型 Profile（用于显示架构概要与保真度） */
  profile: ModelProfile | null;
}

/** Fidelity 层级的可读说明 */
function fidelityDescription(level: string): string {
  const map: Record<string, string> = {
    L1: 'L1 · 基于公开架构信息',
    L2: 'L2 · 基于开源实现',
    L3: 'L3 · 基于运行时框架',
    L4: 'L4 · 基于实测 trace',
    L5: 'L5 · 架构仿真（含估算）',
  };
  return map[level] ?? level;
}

export function ModelOverview({ event, profile }: ModelOverviewProps) {
  const ctx = projectModelContext(event);
  const crumb = projectModelBreadcrumb(event);

  return (
    <div className="model-overview">
      <div className="model-overview-header">模型执行位置（WHERE）</div>

      {crumb ? (
        <div className="model-breadcrumb">
          {crumb.parts.map((part, i) => (
            <span key={`${part}-${i}`} className={i === crumb.parts.length - 1 ? 'crumb-current' : 'crumb'}>
              {i > 0 ? <span className="crumb-sep">›</span> : null}
              {part}
            </span>
          ))}
        </div>
      ) : (
        <div className="model-overview-empty">尚未生成执行计划</div>
      )}

      {ctx && ctx.decodeStep !== undefined ? (
        <div className="model-decode-step">
          自回归生成：第 {ctx.decodeStep} / {ctx.decodeTotal} 个新 token
        </div>
      ) : null}

      {profile ? (
        <div className="model-profile-card">
          <div className="profile-row">
            <span className="profile-label">模型</span>
            <span className="profile-value">{profile.displayName}（{profile.family} {profile.version}）</span>
          </div>
          <div className="profile-row">
            <span className="profile-label">架构</span>
            <span className="profile-value">
              {profile.architecture.type === 'moe' ? 'MoE（混合专家）' : profile.architecture.type === 'hybrid' ? 'Hybrid（混合层）' : 'Dense'}
              {profile.architecture.hiddenSize ? ` · hidden ${profile.architecture.hiddenSize.value}` : ''}
            </span>
          </div>
          {profile.parameterInfo?.total ? (
            <div className="profile-row">
              <span className="profile-label">参数</span>
              <span className="profile-value">
                {formatParams(profile.parameterInfo.total.value)}
                {profile.parameterInfo.activated ? `（激活 ${formatParams(profile.parameterInfo.activated.value)}）` : ''}
              </span>
            </div>
          ) : null}
          {profile.contextLength ? (
            <div className="profile-row">
              <span className="profile-label">上下文</span>
              <span className="profile-value">{profile.contextLength.value.toLocaleString()} tokens</span>
            </div>
          ) : null}
          <div className="profile-row">
            <span className="profile-label">保真度</span>
            <span className={`profile-value fidelity-${profile.fidelity}`}>
              {fidelityDescription(profile.fidelity)}
            </span>
          </div>
          <div className="profile-row">
            <span className="profile-label">来源</span>
            <span className="profile-value">
              {profile.source.map((s) => s.sourceType).join(' / ')} · Simulated（教育仿真）
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/** 参数量的可读格式（如 2.8T / 104B） */
function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1).replace(/\.0$/, '')}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(0)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return String(n);
}
