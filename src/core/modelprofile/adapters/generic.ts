/**
 * Generic Transformer Profile 工厂（Sprint 2 Task B1 / Sprint 3 Task B2）。
 *
 * 目的（任务书 §6）：
 *   - 在做 DeepSeek/Kimi 之前，先用"通用"Profile 验证 ModelProfile schema
 *     不是为四个模型硬编码的；
 *   - Generic Dense：Embedding → Layer×N（Attention+Norm+FFN+Residual）→ LM Head；
 *   - Generic MoE：在 FFN 位置替换为 Router→Top-K→Dispatch→Expert GEMM→Combine；
 *   - 修改 numExperts / expertsPerToken 无需修改 Planner 或 UI（schema 可扩展性）。
 *
 * 本模块只产出 ModelProfile（纯数据），不生成事件、不 import React。
 */

import type { ModelProfile, LayerProfile } from '../types';
import { traced, officialSource } from '../helpers';

/** Generic Dense 配置 */
export interface GenericDenseConfig {
  id?: string;
  displayName?: string;
  numLayers?: number;
  hiddenSize?: number;
  vocabSize?: number;
  numHeads?: number;
  headDim?: number;
  ffnIntermediateSize?: number;
  contextLength?: number;
}

/** Generic MoE 配置 */
export interface GenericMoEConfig extends GenericDenseConfig {
  numExperts?: number;
  expertsPerToken?: number;
}

const GEN_SOURCE = () => officialSource('Generic Transformer (synthetic teaching profile)', '2026-08-26', 'high');

/**
 * 构造一个 Generic Dense Transformer 的 ModelProfile。
 * 验证点：能进入 Execution Planner 并最终生成 TVIR（Sprint 2 PASS）。
 */
export function makeGenericDenseProfile(config: GenericDenseConfig = {}): ModelProfile {
  const {
    id = 'generic-dense',
    displayName = 'Generic Dense Transformer',
    numLayers = 2,
    hiddenSize = 128,
    vocabSize = 1024,
    numHeads = 4,
    headDim = 32,
    ffnIntermediateSize = 512,
    contextLength = 2048,
  } = config;

  const src = GEN_SOURCE();
  const layers: LayerProfile[] = [{ type: 'embedding' }];

  for (let i = 0; i < numLayers; i++) {
    layers.push(
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(numHeads, [src]),
          numKVHeads: traced(numHeads, [src]),
          headDim: traced(headDim, [src]),
        },
      },
      { type: 'norm' },
      { type: 'ffn', ffn: { intermediateSize: traced(ffnIntermediateSize, [src]), activation: 'silu' } },
      { type: 'residual' },
    );
  }
  layers.push({ type: 'lm_head' });

  return {
    id,
    displayName,
    family: 'Generic',
    version: '1.0',
    architecture: {
      type: 'dense',
      hiddenSize: traced(hiddenSize, [src]),
      vocabSize: traced(vocabSize, [src]),
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers,
    contextLength: traced(contextLength, [src]),
    source: [src],
    fidelity: 'L1',
  };
}

/**
 * 构造一个 Generic MoE Transformer 的 ModelProfile。
 * 验证点：修改 numExperts / expertsPerToken 无需修改 Planner 或 UI（Sprint 3 PASS）。
 */
export function makeGenericMoEProfile(config: GenericMoEConfig = {}): ModelProfile {
  const {
    id = 'generic-moe',
    displayName = 'Generic MoE Transformer',
    numLayers = 2,
    hiddenSize = 128,
    vocabSize = 1024,
    numHeads = 4,
    headDim = 32,
    numExperts = 4,
    expertsPerToken = 2,
    contextLength = 2048,
  } = config;

  const src = GEN_SOURCE();
  const layers: LayerProfile[] = [{ type: 'embedding' }];

  for (let i = 0; i < numLayers; i++) {
    layers.push(
      {
        type: 'attention',
        attention: {
          attentionType: 'mha',
          numHeads: traced(numHeads, [src]),
          numKVHeads: traced(numHeads, [src]),
          headDim: traced(headDim, [src]),
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: {
          numExperts: traced(numExperts, [src]),
          expertsPerToken: traced(expertsPerToken, [src]),
          hasSharedExperts: traced(false, [src]),
        },
      },
      { type: 'residual' },
    );
  }
  layers.push({ type: 'lm_head' });

  return {
    id,
    displayName,
    family: 'Generic',
    version: '1.0',
    architecture: {
      type: 'moe',
      hiddenSize: traced(hiddenSize, [src]),
      vocabSize: traced(vocabSize, [src]),
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers,
    contextLength: traced(contextLength, [src]),
    source: [src],
    fidelity: 'L1',
  };
}
