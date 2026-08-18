/**
 * Model 层结构模块（V1.0 六层联动 · Model 层）。
 *
 * 定义 Transformer Block 的模型结构树，并提供"算子 → 模型节点"的查找。
 * 这是纯静态领域知识 + 查找逻辑（与 compiler/operatorKnowledge 同一模式），
 * 不含仿真逻辑、不 import React。
 *
 * Model 层的职责（实施手册 §28 六层联动）：
 *   展示 Transformer 的层级结构（Layer → RMSNorm/Attention/FFN/Residual），
 *   并根据当前 TVIR 事件的 operator 字段高亮"激活路径"（从 Block 根到激活子层）。
 */

import type { TVIREvent } from '../tvir/types';

/** 模型节点分类（用于着色与图标） */
export type ModelNodeCategory =
  | 'block'
  | 'norm'
  | 'attention'
  | 'attention-sub'
  | 'residual'
  | 'ffn'
  | 'ffn-sub';

/** 模型结构树节点 */
export interface ModelNode {
  /** 唯一 id */
  id: string;
  /** 显示名 */
  label: string;
  /** 当前事件的 operator 命中其一则此节点激活 */
  operators: string[];
  /** 子节点 */
  children: ModelNode[];
  /** 分类 */
  category: ModelNodeCategory;
  /** 一句话说明（悬停提示） */
  hint?: string;
}

/**
 * Transformer Block 的模型结构树（Pre-Norm 结构）。
 *
 * operator 字符串与 transformerBlockEngine / attentionEngine 生成的事件一一对应
 * （见 CONCEPTS.md V0.3/V0.9 规则）。Attention 子算子以子节点展开，
 * 使 Model 层能下钻到 QKᵀ / Softmax / AV 粒度。
 */
export const TRANSFORMER_BLOCK_MODEL: ModelNode = {
  id: 'block',
  label: 'Transformer Block · Layer 0',
  operators: ['Transformer Block'],
  category: 'block',
  hint: '一个 Pre-Norm Transformer Block：RMSNorm → Attention → Residual → RMSNorm → FFN → Residual',
  children: [
    {
      id: 'norm1',
      label: 'RMSNorm',
      operators: ['RMSNorm (pre-Attention)'],
      category: 'norm',
      hint: '对输入做 RMS 归一化，稳定数值尺度（Pre-Norm，位于 Attention 前）',
      children: [],
    },
    {
      id: 'attention',
      label: 'Attention',
      operators: ['Attention'],
      category: 'attention',
      hint: '自注意力：Softmax(QKᵀ/√d) × V，让每个 token 关注序列中的其他 token',
      children: [
        { id: 'q-proj', label: 'Q Projection', operators: ['Q Projection'], category: 'attention-sub', hint: 'Q = X × Wq', children: [] },
        { id: 'k-proj', label: 'K Projection', operators: ['K Projection'], category: 'attention-sub', hint: 'K = X × Wk', children: [] },
        { id: 'v-proj', label: 'V Projection', operators: ['V Projection'], category: 'attention-sub', hint: 'V = X × Wv', children: [] },
        { id: 'qk', label: 'QKᵀ', operators: ['QK MatMul'], category: 'attention-sub', hint: 'S = Q × Kᵀ，计算注意力分数', children: [] },
        { id: 'scale', label: 'Scale', operators: ['Scale'], category: 'attention-sub', hint: 'S / √d，控制点积方差', children: [] },
        { id: 'mask', label: 'Mask', operators: ['Mask'], category: 'attention-sub', hint: 'causal mask，屏蔽未来位置', children: [] },
        { id: 'softmax', label: 'Softmax', operators: ['Softmax'], category: 'attention-sub', hint: '按行归一化为概率分布', children: [] },
        { id: 'av', label: 'AV', operators: ['AV MatMul'], category: 'attention-sub', hint: 'O = S × V，加权求和', children: [] },
        { id: 'out-proj', label: 'Output Projection', operators: ['Output Projection'], category: 'attention-sub', hint: '投影回 d_model', children: [] },
      ],
    },
    {
      id: 'residual1',
      label: 'Residual',
      operators: ['Residual 1 (+ Attention)'],
      category: 'residual',
      hint: 'X = X + AttnOut，残差连接',
      children: [],
    },
    {
      id: 'norm2',
      label: 'RMSNorm',
      operators: ['RMSNorm (pre-FFN)'],
      category: 'norm',
      hint: '对输入做 RMS 归一化（Pre-Norm，位于 FFN 前）',
      children: [],
    },
    {
      id: 'ffn',
      label: 'FFN',
      operators: ['FFN'],
      category: 'ffn',
      hint: '前馈网络：Up Projection → SiLU → Down Projection',
      children: [
        { id: 'ffn-up', label: 'Up Projection', operators: ['FFN Up Projection'], category: 'ffn-sub', hint: '升维 d → ffn_dim', children: [] },
        { id: 'ffn-silu', label: 'SiLU', operators: ['FFN SiLU'], category: 'ffn-sub', hint: '非线性激活', children: [] },
        { id: 'ffn-down', label: 'Down Projection', operators: ['FFN Down Projection'], category: 'ffn-sub', hint: '降维 ffn_dim → d', children: [] },
      ],
    },
    {
      id: 'residual2',
      label: 'Residual',
      operators: ['Residual 2 (+ FFN)'],
      category: 'residual',
      hint: 'X = X + F，残差连接',
      children: [],
    },
  ],
};

/**
 * 在模型树中查找命中给定 operator 的节点，返回从根到该节点的 id 路径。
 * 未命中返回空数组。
 *
 * 纯查找：深度优先，匹配节点的 operators 列表。命中子节点时，路径包含其所有祖先，
 * 使 Model 层能高亮"激活路径"（根 → … → 激活子层）。
 */
export function findActivePath(root: ModelNode, operator: string | undefined): string[] {
  if (!operator) return [];
  return findPathRecursive(root, operator, []);
}

function findPathRecursive(node: ModelNode, operator: string, acc: string[]): string[] {
  const current = [...acc, node.id];
  if (node.operators.includes(operator)) {
    return current;
  }
  for (const child of node.children) {
    const found = findPathRecursive(child, operator, current);
    if (found.length > 0) {
      return found;
    }
  }
  return [];
}

/**
 * 从当前 TVIR 事件提取激活的模型节点 id 路径。
 * 纯投影：只读 event.operator（见 ARCHITECTURE.md，Model 层不理解仿真细节）。
 */
export function projectActiveModelPath(event: TVIREvent | null): string[] {
  if (!event) return [];
  return findActivePath(TRANSFORMER_BLOCK_MODEL, event.operator);
}

/**
 * 判断当前事件是否属于 Transformer Block 仿真（决定 Model 层是否显示结构树）。
 * 纯投影：只读事件的 operator 前缀约定。
 */
export function isBlockModelEvent(event: TVIREvent | null): boolean {
  if (!event) return false;
  const op = event.operator ?? '';
  // Block 引擎与 Attention 子图（嵌入 Block 时 operator 无前缀）的算子均属于 Block 模型
  return findActivePath(TRANSFORMER_BLOCK_MODEL, op).length > 0;
}
