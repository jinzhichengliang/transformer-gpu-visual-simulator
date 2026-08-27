/**
 * Model Adapter Registry — 预置模型注册表（Sprint 11-14）。
 *
 * 任务书要求：不为每个模型分别开发一套动画，
 * 所有模型通过统一的 Adapter → ModelProfile → Planner → TVIR 管线。
 * 注册表只是"模型列表"，不包含任何渲染/仿真逻辑。
 */

import type { ModelProfile } from '../types';
import { makeGenericDenseProfile, makeGenericMoEProfile } from './generic';
import { makeDeepSeekV4FlashProfile, makeDeepSeekV4ProProfile } from './deepseekV4';
import { makeKimiK3Profile } from './kimiK3';
import { makeGLM53Profile } from './glm53';

export interface ModelRegistryEntry {
  /** Profile 工厂（惰性构造，避免启动时全部实例化） */
  factory: () => ModelProfile;
}

/** 预置模型注册表（顺序即 UI 展示顺序） */
const registry = new Map<string, ModelRegistryEntry>();

registry.set('generic-dense', { factory: makeGenericDenseProfile });
registry.set('generic-moe', { factory: () => makeGenericMoEProfile({ numLayers: 2 }) });
registry.set('deepseek-v4-flash', { factory: makeDeepSeekV4FlashProfile });
registry.set('deepseek-v4-pro', { factory: makeDeepSeekV4ProProfile });
registry.set('kimi-k3', { factory: makeKimiK3Profile });
registry.set('glm-5.3', { factory: makeGLM53Profile });

/** 列出全部已注册模型（id + 显示名） */
export function listRegisteredModels(): Array<{ id: string; displayName: string }> {
  const result: Array<{ id: string; displayName: string }> = [];
  for (const [id, entry] of registry) {
    const profile = entry.factory();
    result.push({ id, displayName: profile.displayName });
  }
  return result;
}

/** 按 id 获取 Profile（不存在时返回 undefined） */
export function getModelProfile(id: string): ModelProfile | undefined {
  const entry = registry.get(id);
  return entry?.factory();
}

/** 注册新模型（低成本扩展点，任务书 §0） */
export function registerModel(id: string, factory: () => ModelProfile): void {
  registry.set(id, { factory });
}
