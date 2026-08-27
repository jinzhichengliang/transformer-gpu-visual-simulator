/**
 * Fidelity Badge 投影层（Sprint 17, Task J1/J2）。
 *
 * 任务书要求：
 *   Task J1 — Visual Fidelity Badge：
 *     以分类行展示模型数据的可信来源，例如：
 *       Architecture: Official
 *       Runtime: vLLM based
 *       GPU timing: Simulated
 *   Task J2 — 禁止虚假精度：
 *     模拟数据禁止显示 "Latency = 3.427861 μs" 这类测量级精度；
 *     只能显示 "Illustrative execution" 或 "Estimated" 等限定表述。
 *
 * 架构约束：纯投影，不 import React，不新增事件类型。
 */

import type { ModelProfile, SourceMetadata, SourceType } from '../modelprofile/types';

/** Badge 的一行：分类 + 来源标签 */
export interface FidelityBadgeRow {
  category: string;
  label: string;
  /** 该标签是否为不可信来源（需要视觉弱化） */
  weak: boolean;
}

/** Badge 汇总结果 */
export interface FidelityBadgeSummary {
  modelName: string;
  rows: FidelityBadgeRow[];
  /** Profile 整体保真度（L1-L5） */
  fidelity: string;
  /** 全局警示语（J2：模拟数据必须显式声明） */
  disclaimer: string;
}

/** 来源类型的可读标签 */
const SOURCE_LABELS: Record<SourceType, string> = {
  official: 'Official',
  official_repo: 'Official Repo',
  technical_report: 'Technical Report',
  runtime_repo: 'Runtime Repo',
  inferred: 'Inferred',
  estimated: 'Estimated',
};

/** 可信来源（有公开依据） */
const STRONG_SOURCE_TYPES: SourceType[] = [
  'official',
  'official_repo',
  'technical_report',
  'runtime_repo',
];

/** 判断一组来源的最弱环节（只要有一个弱来源就标弱） */
function weakestSourceLabel(sources: SourceMetadata[]): { label: string; weak: boolean } {
  if (sources.length === 0) return { label: 'Unknown', weak: true };
  const hasWeak = sources.some((s) => !STRONG_SOURCE_TYPES.includes(s.sourceType));
  const allEstimated = sources.every((s) => s.sourceType === 'estimated');
  const allInferredOrEstimated = sources.every(
    (s) => s.sourceType === 'inferred' || s.sourceType === 'estimated',
  );
  if (allEstimated) return { label: 'Estimated', weak: true };
  if (allInferredOrEstimated) return { label: 'Inferred', weak: true };
  if (hasWeak) return { label: 'Mixed (incl. Inferred/Estimated)', weak: true };
  // 全部为可信来源：取第一个的标签
  const first = sources[0];
  return { label: first ? SOURCE_LABELS[first.sourceType] : 'Official', weak: false };
}

/**
 * 从 ModelProfile 计算 Fidelity Badge（Task J1）。
 *
 * 本项目为架构仿真器（L5 Architecture-Simulation-Based）：
 *   - 架构信息来源 = profile 自身的 source 元数据（诚实呈现）；
 *   - 执行模型与 GPU 时序恒为 Simulated（本项目不做实测，禁止标注为测量值）。
 */
export function buildFidelityBadge(profile: ModelProfile): FidelityBadgeSummary {
  const arch = weakestSourceLabel(profile.source);
  const rows: FidelityBadgeRow[] = [
    { category: 'Architecture', label: arch.label, weak: arch.weak },
    {
      category: 'Execution Model',
      label: 'Architecture Simulation',
      weak: false,
    },
    {
      category: 'GPU Timing',
      label: 'Simulated',
      weak: false,
    },
  ];
  return {
    modelName: profile.displayName,
    rows,
    fidelity: profile.fidelity,
    disclaimer:
      'Illustrative execution — 所有时序与性能数据为教学仿真估算（Simulated/Estimated），非实测。',
  };
}

/**
 * 禁止虚假精度（Task J2）：
 * 把模拟数值格式化为最多 2 位有效数字 + 限定标签。
 * 绝不输出 "3.427861 μs" 这类测量错觉格式。
 */
export function formatSimulatedValue(value: number, unit: string): string {
  if (!Number.isFinite(value)) return `~ ${unit}`;
  // 最多 2 位有效数字
  const rounded = Number(value.toPrecision(2));
  return `~ ${rounded} ${unit} (Simulated)`;
}

/**
 * 来源标签（供 UI 在参数旁显示），未知来源标记为 Estimated 而非留空。
 */
export function sourceLabelFor(sources: SourceMetadata[] | undefined): string {
  if (!sources || sources.length === 0) return 'Estimated';
  const { label } = weakestSourceLabel(sources);
  return label;
}
