/**
 * Kimi K3 Model Adapter（Sprint 13, Task B5）。
 *
 * 重点验证（任务书）：多种 Layer 类型（Dense 层 + MoE 层）共存，
 * Execution Planner 能按 Layer 0, Layer 1... 正确读取不同类型 Layer。
 *
 * 来源与可信度：
 *   - 总参数 2.8T / 激活 104B / 93 层 / 1 个 Dense 层：官方仓库（github.com/MoonshotAI/Kimi-K3）
 *   - 896 专家 / top-16：官方技术报告与博客（official, high）
 *   - 注意力类型：Kimi Delta Attention（KDA）混合线性注意力，教学模型以最接近的
 *     GQA 表达 → 标注为 inferred（medium），不作为 verified 事实。
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

/** Kimi K3（2.8T 总参 / 104B 激活 / 93 层 / 896 专家 top-16） */
export function makeKimiK3Profile(): ModelProfile {
  const officialRepoSrc = official('github.com/MoonshotAI/Kimi-K3 官方仓库（Model Summary）');
  const officialReportSrc = official('Kimi K3 技术报告（arXiv:2607.24653）');
  const archSrc = inferred('Kimi K3 架构解析：KDA 混合线性注意力（第三方技术文章）');

  const totalLayers = 93;
  const denseLayers = 1; // 官方：Number of Dense Layers = 1

  const layers: LayerProfile[] = [{ type: 'embedding' }];
  for (let i = 0; i < totalLayers; i++) {
    const isDenseLayer = i < denseLayers; // 前 1 层为 Dense FFN

    layers.push(
      {
        type: 'attention',
        attention: {
          // KDA（Kimi Delta Attention）为混合线性注意力；
          // 教学模型以最接近的 GQA 表达，此处为推断。
          attentionType: 'gqa',
          numHeads: traced(64, [archSrc]),
          numKVHeads: traced(8, [archSrc]),
          headDim: traced(128, [archSrc]),
          sources: [archSrc],
        },
      },
      { type: 'norm' },
    );

    if (isDenseLayer) {
      // Dense 层：标准 FFN
      layers.push(
        {
          type: 'ffn',
          ffn: {
            intermediateSize: traced(16384, [archSrc]),
            activation: 'silu',
            sources: [archSrc],
          },
        },
        { type: 'residual' },
      );
    } else {
      // MoE 层：896 专家，每 token 激活 16 个
      layers.push(
        {
          type: 'moe',
          moe: {
            numExperts: traced(896, [officialReportSrc]),
            expertsPerToken: traced(16, [officialReportSrc]),
            hasSharedExperts: traced(false, [archSrc]),
            sources: [officialReportSrc],
          },
        },
        { type: 'residual' },
      );
    }
  }
  layers.push({ type: 'lm_head' });

  return {
    id: 'kimi-k3',
    displayName: 'Kimi K3',
    family: 'Kimi',
    version: 'release',
    architecture: {
      type: 'hybrid', // Dense 层 + MoE 层混合
      hiddenSize: traced(8192, [archSrc]),
      vocabSize: traced(163840, [archSrc]),
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers,
    contextLength: traced(1_000_000, [officialRepoSrc]),
    parameterInfo: {
      total: traced(2_800_000_000_000, [officialRepoSrc]),
      activated: traced(104_000_000_000, [officialRepoSrc]),
    },
    source: [officialRepoSrc, officialReportSrc, archSrc],
    fidelity: 'L1',
  };
}
