/**
 * GLM-5.3 Model Adapter（Sprint 14, Task B6）。
 *
 * 任务书核心要求：对于未完全公开或无法验证的细节，禁止假造，
 * 必须表示成 unknown / inferred / estimated；
 * 没有公开依据的字段不能以 verified = true 进入 Profile。
 *
 * 来源与可信度：
 *   - 总参数 ~743B、上下文 1M、激活 ~40B：官方发布与多家媒体一致（official）
 *   - GLM-5.3 与 GLM-5.2 共享同一基座（官方明确说明），基座为 256 专家、80 层
 *     的 MoE 结构 → 继承自 GLM-5 技术报告（official，但属于"基座规格"而非
 *     5.3 单独公布，故层数/专家数标注为 official_repo 来源 + medium 置信）
 *   - 隐藏维度、注意力头数、词表大小：无公开一手数据 → 标注为 estimated（low），
 *     绝不标成 verified / official。
 */

import type { ModelProfile, LayerProfile } from '../types';
import { traced } from '../helpers';
import type { SourceMetadata } from '../types';

const VERIFY_DATE = '2026-08-26';

function official(reference: string): SourceMetadata {
  return { sourceType: 'official', reference, verifiedAt: VERIFY_DATE, confidence: 'high' };
}
function estimated(reference: string): SourceMetadata {
  return { sourceType: 'estimated', reference, verifiedAt: VERIFY_DATE, confidence: 'low' };
}

/** GLM-5.3（~743B 总参 / ~40B 激活 / 80 层 / 256 专家，与 GLM-5.2 共享基座） */
export function makeGLM53Profile(): ModelProfile {
  const officialSrc = official('智谱 GLM-5.3 官方发布（2026-08-14，z.ai/blog/glm-5.3）');
  const baseSrc = {
    sourceType: 'official_repo' as const,
    reference: 'GLM-5 技术报告（arXiv:2602.15763）：256 专家、80 层基座规格',
    verifiedAt: VERIFY_DATE,
    confidence: 'medium' as const,
  };
  const estSrc = estimated('无公开数据，按同规模 MoE 模型常见配置估算（教学用途）');

  const numLayers = 80;
  const layers: LayerProfile[] = [{ type: 'embedding' }];
  for (let i = 0; i < numLayers; i++) {
    layers.push(
      {
        type: 'attention',
        attention: {
          // GLM 系列采用 MLA/GQA 变体；具体类型无 5.3 公开数据 → 以 GQA 教学表达
          attentionType: 'gqa',
          numHeads: traced(64, [estSrc]),
          numKVHeads: traced(4, [estSrc]),
          headDim: traced(128, [estSrc]),
          sources: [estSrc],
        },
      },
      { type: 'norm' },
      {
        type: 'moe',
        moe: {
          numExperts: traced(256, [baseSrc]),
          expertsPerToken: traced(8, [baseSrc]),
          hasSharedExperts: traced(true, [baseSrc]),
          sources: [baseSrc],
        },
      },
      { type: 'residual' },
    );
  }
  layers.push({ type: 'lm_head' });

  return {
    id: 'glm-5.3',
    displayName: 'GLM-5.3',
    family: 'GLM',
    version: '5.3',
    architecture: {
      type: 'moe',
      hiddenSize: traced(6144, [estSrc]), // 无公开数据，估算
      vocabSize: traced(151552, [estSrc]), // 无公开数据，估算
      normType: 'rmsnorm',
      positionalEncoding: 'rope',
    },
    layers,
    contextLength: traced(1_000_000, [officialSrc]),
    parameterInfo: {
      total: traced(743_000_000_000, [officialSrc]),
      activated: traced(40_000_000_000, [officialSrc]),
    },
    source: [officialSrc, baseSrc, estSrc],
    // 含估算字段，整体保真度降为 L5（架构仿真级），诚实反映数据混合来源
    fidelity: 'L5',
  };
}
