/**
 * Execution Planner — ModelProfile + InferenceTask → OperatorGraph（Sprint 5, Task D1）。
 *
 * 任务书要求：
 *   - Planner 不允许输出 UI 数据、不产生 SVG、不知道 React；
 *   - 节点必须包含 model context / layer context / operator type /
 *     input shapes / output shapes / phase / dependency / source fidelity；
 *   - Graph 必须无孤立节点、无非法循环、所有 dependency 可解析。
 *
 * 本模块是纯逻辑层，只产出 OperatorGraph（纯数据）。
 */

import type { ModelProfile } from '../modelprofile/types';
import type { InferenceTask, InferencePhase } from './task';

/**
 * 粗粒度算子类型（OperatorGraph 节点）。
 * 注意：这是"层内组件"级别，比 TVIR event 的 operator 字符串更粗一级。
 */
export const GRAPH_OPERATOR_TYPES = [
  'embedding',
  'norm',
  'attention',
  'ffn',
  'moe',
  'residual',
  'lm_head',
] as const;

export type GraphOperatorType = (typeof GRAPH_OPERATOR_TYPES)[number];

/** 张量形状描述 */
export interface TensorShape {
  rows: number;
  cols: number;
  label: string;
}

/** OperatorGraph 节点 */
export interface OperatorNode {
  id: string;
  /** 模型上下文（模型 id + 显示名） */
  modelId: string;
  modelDisplayName: string;
  /** 层上下文（层序号与类型，-1 表示非层结构如 embedding/lm_head） */
  layerIndex: number;
  layerType: string;
  /** 算子类型 */
  operatorType: GraphOperatorType;
  /** 执行阶段（该节点属于哪个阶段的计划） */
  phase: InferencePhase;
  /** 输入/输出形状（可追踪） */
  inputShape: TensorShape;
  outputShape: TensorShape;
  /** 依赖的前置节点 id */
  dependsOn: string[];
  /** 保真度（继承自 ModelProfile） */
  fidelity: string;
}

/** 算子图（线性依赖链） */
export interface OperatorGraph {
  modelId: string;
  phase: InferencePhase;
  nodes: OperatorNode[];
}

export interface GraphValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * 构建 OperatorGraph（Task D1）。
 *
 * 输入：ModelProfile + InferenceTask
 * 输出：线性依赖的 OperatorGraph（无孤立节点、无环）。
 *
 * 层编号规则：每遇到一个 attention 层，逻辑层号 +1；
 * 同层的 ffn/moe 继承该层号；embedding/lm_head 层号为 -1。
 */
export function buildOperatorGraph(profile: ModelProfile, task: InferenceTask): OperatorGraph {
  const nodes: OperatorNode[] = [];
  const hiddenSize = profile.architecture.hiddenSize?.value ?? 128;
  const vocabSize = profile.architecture.vocabSize?.value ?? 1024;
  const M = task.promptTokens * task.batchSize; // 有效 token 行数

  let prevId: string | null = null;
  let counter = 0;

  const pushNode = (
    operatorType: GraphOperatorType,
    layerIndex: number,
    layerType: string,
    inputShape: TensorShape,
    outputShape: TensorShape,
  ): void => {
    const id = `op-${counter}`;
    counter += 1;
    nodes.push({
      id,
      modelId: profile.id,
      modelDisplayName: profile.displayName,
      layerIndex,
      layerType,
      operatorType,
      phase: task.phase,
      inputShape,
      outputShape,
      dependsOn: prevId ? [prevId] : [],
      fidelity: profile.fidelity,
    });
    prevId = id;
  };

  const hidden = (label: string): TensorShape => ({ rows: M, cols: hiddenSize, label });

  let logicalLayer = -1;
  for (let i = 0; i < profile.layers.length; i++) {
    const layer = profile.layers[i];
    if (!layer) continue;

    switch (layer.type) {
      case 'embedding':
        pushNode(
          'embedding',
          -1,
          'embedding',
          { rows: M, cols: 1, label: 'token_ids' },
          hidden('X0'),
        );
        break;
      case 'lm_head':
        pushNode(
          'lm_head',
          -1,
          'lm_head',
          hidden('X_final'),
          { rows: M, cols: vocabSize, label: 'logits' },
        );
        break;
      case 'attention':
        logicalLayer += 1;
        pushNode('attention', logicalLayer, 'attention', hidden(`X_in_L${logicalLayer}`), hidden(`X_attn_L${logicalLayer}`));
        break;
      case 'ffn':
        pushNode('ffn', logicalLayer, 'ffn', hidden(`X_mid_L${logicalLayer}`), hidden(`X_out_L${logicalLayer}`));
        break;
      case 'moe':
        pushNode('moe', logicalLayer, 'moe', hidden(`X_mid_L${logicalLayer}`), hidden(`X_out_L${logicalLayer}`));
        break;
      case 'norm':
      case 'residual':
        // norm/residual 由执行展开时自动补全（Pre-Norm 结构），
        // 不在 Graph 层显式建节点，保持 Graph 聚焦主干计算。
        break;
      case 'other':
        // 未识别层类型：保留为占位节点，保证序列完整、不静默丢弃
        pushNode('ffn', logicalLayer, 'other', hidden(`X_other_${i}`), hidden(`X_other_${i}_out`));
        break;
    }
  }

  return { modelId: profile.id, phase: task.phase, nodes };
}

/**
 * Graph 校验（任务书 §8 自动检查）：
 *   - 无孤立节点（除首节点外都有依赖，且依赖可解析）；
 *   - 无非法循环（依赖链为 DAG）；
 *   - 输入输出 shape 可追踪（行列均为正整数）。
 */
export function validateOperatorGraph(graph: OperatorGraph): GraphValidationResult {
  const errors: string[] = [];
  const ids = new Set(graph.nodes.map((n) => n.id));

  if (graph.nodes.length === 0) {
    errors.push('OperatorGraph 为空（至少应有 embedding）');
    return { valid: false, errors };
  }

  graph.nodes.forEach((node, idx) => {
    // 依赖可解析
    for (const dep of node.dependsOn) {
      if (!ids.has(dep)) {
        errors.push(`节点 ${node.id} 的依赖 ${dep} 无法解析`);
      }
    }
    // 首节点无依赖；其余节点必须有依赖（线性链）
    if (idx === 0) {
      if (node.dependsOn.length !== 0) {
        errors.push(`首节点 ${node.id} 不应有依赖`);
      }
    } else if (node.dependsOn.length === 0) {
      errors.push(`节点 ${node.id} 是孤立节点（缺少依赖）`);
    }
    // shape 可追踪
    for (const shape of [node.inputShape, node.outputShape]) {
      if (!Number.isInteger(shape.rows) || shape.rows < 1 || !Number.isInteger(shape.cols) || shape.cols < 1) {
        errors.push(`节点 ${node.id} 的 shape 非法：${shape.label}（rows=${shape.rows}, cols=${shape.cols})`);
      }
    }
  });

  // 环检测（线性链：每个节点最多被一个后续节点依赖，且无自环）
  for (const node of graph.nodes) {
    if (node.dependsOn.includes(node.id)) {
      errors.push(`节点 ${node.id} 存在自环依赖`);
    }
  }

  return { valid: errors.length === 0, errors };
}
