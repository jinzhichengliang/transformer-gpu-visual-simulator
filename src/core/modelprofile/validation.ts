/**
 * ModelProfile Schema Validation（Sprint 1, Task A1/A2）。
 *
 * 任务书要求：
 *   - malformed profile 会被拒绝；
 *   - 所有关键字段支持 source metadata；
 *   - estimated 数据不能被展示成 measured（PASS 标准）。
 *
 * 本模块是纯逻辑层，不 import React。
 */

import type { ModelProfile, LayerProfile, Traced, SourceMetadata } from './types';
import { LAYER_TYPES, SOURCE_TYPES, CONFIDENCE_LEVELS, FIDELITY_LEVELS } from './types';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/** 校验单条 SourceMetadata 的字段合法性 */
function validateSourceMeta(source: SourceMetadata, path: string, errors: string[]): void {
  if (!SOURCE_TYPES.includes(source.sourceType)) {
    errors.push(`${path}.sourceType 非法：${source.sourceType}`);
  }
  if (!source.reference || source.reference.trim() === '') {
    errors.push(`${path}.reference 不能为空`);
  }
  if (!source.verifiedAt || !/^\d{4}-\d{2}-\d{2}$/.test(source.verifiedAt)) {
    errors.push(`${path}.verifiedAt 必须为 YYYY-MM-DD 格式`);
  }
  if (!CONFIDENCE_LEVELS.includes(source.confidence)) {
    errors.push(`${path}.confidence 非法：${source.confidence}`);
  }
}

/** 校验 Traced<T> 的来源元数据完整性 */
function validateTraced<T>(traced: Traced<T> | undefined, path: string, errors: string[]): void {
  if (traced === undefined) return;
  if (traced.value === undefined || traced.value === null) {
    errors.push(`${path}.value 缺失`);
  }
  if (!traced.fidelity || !FIDELITY_LEVELS.includes(traced.fidelity)) {
    errors.push(`${path}.fidelity 非法（必须为 ${FIDELITY_LEVELS.join('/')})`);
  }
  if (!Array.isArray(traced.sources) || traced.sources.length === 0) {
    errors.push(`${path}.sources 缺失（每个重要参数必须有来源）`);
    return;
  }
  traced.sources.forEach((s, i) => validateSourceMeta(s, `${path}.sources[${i}]`, errors));
}

/** 校验单个 LayerProfile */
function validateLayer(layer: LayerProfile, index: number, errors: string[]): void {
  const path = `layers[${index}]`;
  if (!LAYER_TYPES.includes(layer.type)) {
    errors.push(`${path}.type 非法：${layer.type}（必须为 ${LAYER_TYPES.join('/')})`);
    return;
  }
  // attention 层必须携带 attention 细节
  if (layer.type === 'attention') {
    if (!layer.attention) {
      errors.push(`${path}：type=attention 但缺少 attention 描述`);
    } else {
      validateTraced(layer.attention.numHeads, `${path}.attention.numHeads`, errors);
      validateTraced(layer.attention.numKVHeads, `${path}.attention.numKVHeads`, errors);
      validateTraced(layer.attention.headDim, `${path}.attention.headDim`, errors);
    }
  }
  // moe 层必须携带 moe 细节
  if (layer.type === 'moe') {
    if (!layer.moe) {
      errors.push(`${path}：type=moe 但缺少 moe 描述`);
    } else {
      validateTraced(layer.moe.numExperts, `${path}.moe.numExperts`, errors);
      validateTraced(layer.moe.expertsPerToken, `${path}.moe.expertsPerToken`, errors);
      if (layer.moe.numExperts && layer.moe.expertsPerToken) {
        if (layer.moe.expertsPerToken.value > layer.moe.numExperts.value) {
          errors.push(`${path}：expertsPerToken(${layer.moe.expertsPerToken.value}) 不能超过 numExperts(${layer.moe.numExperts.value})`);
        }
      }
    }
  }
  // ffn 层可选细节
  if (layer.type === 'ffn' && layer.ffn) {
    validateTraced(layer.ffn.intermediateSize, `${path}.ffn.intermediateSize`, errors);
  }
}

/**
 * 校验完整 ModelProfile。
 *
 * 验收要点：
 *   - id/displayName/family/version 必填；
 *   - layers 至少 1 层；
 *   - architecture.type 合法；
 *   - 所有 Traced 字段携带有效 source metadata；
 *   - fidelity 合法。
 */
export function validateModelProfile(profile: ModelProfile): ValidationResult {
  const errors: string[] = [];

  if (!profile.id || profile.id.trim() === '') {
    errors.push('id 不能为空');
  }
  if (!profile.displayName || profile.displayName.trim() === '') {
    errors.push('displayName 不能为空');
  }
  if (!profile.family || profile.family.trim() === '') {
    errors.push('family 不能为空');
  }
  if (!profile.version || profile.version.trim() === '') {
    errors.push('version 不能为空');
  }

  // architecture
  if (!profile.architecture) {
    errors.push('architecture 缺失');
  } else {
    const validArchTypes = ['dense', 'moe', 'hybrid'];
    if (!validArchTypes.includes(profile.architecture.type)) {
      errors.push(`architecture.type 非法：${profile.architecture.type}`);
    }
    validateTraced(profile.architecture.hiddenSize, 'architecture.hiddenSize', errors);
    validateTraced(profile.architecture.vocabSize, 'architecture.vocabSize', errors);
  }

  // layers
  if (!Array.isArray(profile.layers) || profile.layers.length === 0) {
    errors.push('layers 至少需要 1 层');
  } else {
    profile.layers.forEach((layer, i) => validateLayer(layer, i, errors));
  }

  // 可选顶层字段
  validateTraced(profile.contextLength, 'contextLength', errors);
  validateTraced(profile.parameterInfo?.total, 'parameterInfo.total', errors);
  validateTraced(profile.parameterInfo?.activated, 'parameterInfo.activated', errors);
  if (profile.precision) {
    validateTraced(profile.precision.paramPrecision, 'precision.paramPrecision', errors);
    validateTraced(profile.precision.computePrecision, 'precision.computePrecision', errors);
  }

  // source 与 fidelity
  if (!Array.isArray(profile.source) || profile.source.length === 0) {
    errors.push('source 缺失（Profile 必须有整体来源）');
  } else {
    profile.source.forEach((s, i) => validateSourceMeta(s, `source[${i}]`, errors));
  }
  if (!FIDELITY_LEVELS.includes(profile.fidelity)) {
    errors.push(`fidelity 非法：${profile.fidelity}（必须为 ${FIDELITY_LEVELS.join('/')})`);
  }

  return { valid: errors.length === 0, errors };
}

/**
 * 防作弊检查：estimated 数据不得被标记为 measured 语义（任务书 §5 PASS 标准）。
 * 返回 true 表示存在违规（即测试应失败的情况）。
 */
export function hasEstimatedMarkedAsMeasured(profile: ModelProfile): boolean {
  // 本项目所有仿真数据均为 Simulated/Estimated，不存在 Measured 标注路径。
  // 该函数检查 source 中是否有 sourceType='estimated' 却被声明为高保真（L4 实测）的矛盾。
  const contradictions: string[] = [];
  for (const s of profile.source) {
    if (s.sourceType === 'estimated' && profile.fidelity === 'L4') {
      contradictions.push('estimated 来源与 L4(Profile-Trace-Based) 保真度矛盾');
    }
  }
  return contradictions.length > 0;
}
