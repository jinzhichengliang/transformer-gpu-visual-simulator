/**
 * ModelSelector — 模型选择与推理任务配置（Sprint 7, Task F1/F2）。
 *
 * Task F1：Model Selector 下拉（DeepSeek V4 Flash / V4 Pro / Kimi K3 / GLM-5.3 / Generic）。
 * Task F2：Task Selector（Phase / Prompt Tokens / Generation Tokens / Batch Size）
 *          + Generate Execution 按钮。
 *
 * 架构约束：
 *   - 组件只负责收集"模型 id + InferenceTask"，通过回调交给 App；
 *   - 不做仿真、不生成 TVIR（那是 Executor 的职责）；
 *   - 切换模型不得残留前一个模型的状态（App 侧负责重置）。
 */

import type { InferenceTask, InferencePhase } from '../../core/execution/task';
import type { ModelProfile } from '../../core/modelprofile';
import './ModelSelector.css';

export interface ModelSelectorProps {
  /** 可选模型列表（id + 显示名） */
  models: Array<{ id: string; displayName: string }>;
  /** 当前选中模型 id */
  selectedModelId: string;
  /** 当前推理任务配置 */
  task: InferenceTask;
  /** 当前模型 Profile（用于展示架构概要，可为空） */
  profile: ModelProfile | null;
  onModelChange: (id: string) => void;
  onTaskChange: (patch: Partial<InferenceTask>) => void;
  /** 点击 Generate Execution */
  onGenerate: () => void;
  /** 生成错误（任务非法时展示） */
  error?: string | null;
}

const PHASE_OPTIONS: Array<{ value: InferencePhase; label: string }> = [
  { value: 'prefill', label: 'Prefill（预填充）' },
  { value: 'decode', label: 'Decode（逐词生成）' },
  { value: 'prefill_decode', label: 'Prefill + Decode（完整推理）' },
];

const PROMPT_OPTIONS = [16, 32, 64, 128, 256, 512];
const OUTPUT_OPTIONS = [1, 2, 4, 8, 16];
const BATCH_OPTIONS = [1, 2, 4];

export function ModelSelector(props: ModelSelectorProps) {
  const { models, selectedModelId, task, profile, onModelChange, onTaskChange, onGenerate, error } = props;

  return (
    <div className="model-selector">
      <div className="model-selector-group">
        <label className="model-selector-label">Model（选择大模型）</label>
        <select
          className="model-selector-input"
          value={selectedModelId}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.displayName}</option>
          ))}
        </select>
      </div>

      <div className="model-selector-group">
        <label className="model-selector-label">推理阶段（Phase）</label>
        <select
          className="model-selector-input"
          value={task.phase}
          onChange={(e) => onTaskChange({ phase: e.target.value as InferencePhase })}
        >
          {PHASE_OPTIONS.map((p) => (
            <option key={p.value} value={p.value}>{p.label}</option>
          ))}
        </select>
      </div>

      <div className="model-selector-group">
        <label className="model-selector-label">Prompt Tokens</label>
        <select
          className="model-selector-input"
          value={task.promptTokens}
          onChange={(e) => onTaskChange({ promptTokens: Number(e.target.value) })}
        >
          {PROMPT_OPTIONS.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      <div className="model-selector-group">
        <label className="model-selector-label">Generation Tokens</label>
        <select
          className="model-selector-input"
          value={task.outputTokens}
          onChange={(e) => onTaskChange({ outputTokens: Number(e.target.value) })}
        >
          {OUTPUT_OPTIONS.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      <div className="model-selector-group">
        <label className="model-selector-label">Batch Size</label>
        <select
          className="model-selector-input"
          value={task.batchSize}
          onChange={(e) => onTaskChange({ batchSize: Number(e.target.value) })}
        >
          {BATCH_OPTIONS.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      <button type="button" className="model-generate-btn" onClick={onGenerate}>
        Generate Execution ▶
      </button>

      {error ? <div className="model-selector-error">{error}</div> : null}

      {profile ? (
        <div className="model-selector-hint">
          {profile.architecture.type === 'moe' ? 'MoE' : profile.architecture.type === 'hybrid' ? 'Hybrid' : 'Dense'}
          {' · '}
          {profile.layers.length} 层定义
          {profile.parameterInfo?.activated
            ? ` · 激活 ${formatParams(profile.parameterInfo.activated.value)}`
            : ''}
        </div>
      ) : null}
    </div>
  );
}

function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1).replace(/\.0$/, '')}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(0)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return String(n);
}
