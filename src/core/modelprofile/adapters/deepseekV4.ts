/**
 * DeepSeek V4 Flash / V4 Pro Model Adapter（Sprint 11 Task B3 / Sprint 12 Task B4）。
 *
 * Adapter 的唯一职责：公开模型信息 → ModelProfile。
 * 禁止：Adapter → GPUView（不得接触任何渲染/仿真逻辑）。
 *
 * 来源与可信度（Task A2 诚实标注）：
 *   - 总参数 / 激活参数 / 上下文：官方技术报告与官方仓库（official, high）
 *   - 层数 / 隐藏维度 / 专家数：来自第三方架构解析文章，非官方一手文档
 *     → 标注为 inferred（medium），不作为 verified 事实。
 *
 * Task B4 验收：V4 Flash 与 V4 Pro 必须产生实质不同的 Profile
 * （层数 43 vs 61、hidden 4096 vs 7168、专家 256 vs 384、参数 284B vs 1.6T），
 * 不得只有 displayName 不同。
 */

import type { ModelProfile, LayerProfile } from '../types';
import { traced } from '../helpers';
import type { SourceMetadata } from '../types';

const VERIFY_DATE = '2026-08-26';

function official(reference: string): SourceMetadata {
  return { sourceType: 'official', reference, verifiedAt: VERIFY_DATE, confidence: 'high' };
}
function inferred(reference: string): SourceMetadata {
  return { sourceType: 'inferred', reference, verifiedAt: VERIFY_DATE, confidence: 'medium' };
}

interface DeepSeekV4Spec {
  id: string;
  displayName: string;
  version: string;
  totalParams: number;
  activatedParams: number;
  numLayers: number;
  hiddenSize: number;
  numExperts: number;
  expertsPerToken: number;
  contextLength: number;
}

/** 依据统一规格构建 DeepSeek V4 系列 Profile（Dense/MoE 由规格决定） */
function buildDeepSeekV4Profile(spec: DeepSeekV4Spec): ModelProfile {
  const officialParamSrc = official('DeepSeek-V4 技术报告 / ModelScope deepseek-ai 官方仓库');
  const archSrc = inferred('DeepSeek V4 架构解析（第三方技术文章，层数/专家数等细节）');

  const layers: LayerProfile[] = [{ type: 'embedding' }];
  for (let i = 0; i < spec.numLayers; i++) {
    layers.push(
      {
        type: 'attention',
        attention: {
          // DeepSeek V4 采用混合注意力（压缩稀疏注意力 + 混合压缩注意力），
          // 架构解析资料描述为 GQA（num_kv_heads=1, head_dim=512）；
          // 教学模型以 GQA 表达，此处细节为推断（非官方一手文档）。
          attentionType: 'gqa',
          numHeads: traced(Math.max(1, Math.round(spec.hiddenSize / 512)), [archSrc]),
          numKVHeads: traced(1, [archSrc]),
          headDim: traced(512, [archSrc]),
          sources: [archSrc],
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: {
          numExperts: traced(spec.numExperts, [archSrc]),
          expertsPerToken: traced(spec.expertsPerToken, [archSrc]),
          hasSharedExperts: traced(true, [archSrc]),
          sources: [archSrc],
        },
      },
      { type: 'residual' },
    );
  }
  layers.push({ type: 'lm_head' });

  return {
    id: spec.id,
    displayName: spec.displayName,
    family: 'DeepSeek-V4',
    version: spec.version,
    architecture: {
      type: 'moe',
      hiddenSize: traced(spec.hiddenSize, [archSrc]),
      vocabSize: traced(128000, [archSrc]),
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers,
    contextLength: traced(spec.contextLength, [officialParamSrc]),
    parameterInfo: {
      total: traced(spec.totalParams, [officialParamSrc]),
      activated: traced(spec.activatedParams, [officialParamSrc]),
    },
    source: [officialParamSrc, archSrc],
    fidelity: 'L1',
  };
}

/** DeepSeek V4 Flash（284B 总参 / 13B 激活 / 43 层 / 256 专家） */
export function makeDeepSeekV4FlashProfile(): ModelProfile {
  return buildDeepSeekV4Profile({
    id: 'deepseek-v4-flash',
    displayName: 'DeepSeek V4 Flash',
    version: 'preview',
    totalParams: 284_000_000_000,
    activatedParams: 13_000_000_000,
    numLayers: 43,
    hiddenSize: 4096,
    numExperts: 256,
    expertsPerToken: 8,
    contextLength: 1_000_000,
  });
}

/** DeepSeek V4 Pro（1.6T 总参 / 49B 激活 / 61 层 / 384 专家） */
export function makeDeepSeekV4ProProfile(): ModelProfile {
  return buildDeepSeekV4Profile({
    id: 'deepseek-v4-pro',
    displayName: 'DeepSeek V4 Pro',
    version: 'preview',
    totalParams: 1_600_000_000_000,
    activatedParams: 49_000_000_000,
    numLayers: 61,
    hiddenSize: 7168,
    numExperts: 384,
    expertsPerToken: 8,
    contextLength: 1_000_000,
  });
}
