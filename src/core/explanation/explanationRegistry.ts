/**
 * ExplanationRegistry — What/Why 教学解释注册表（Sprint 16, Task I1）。
 *
 * 任务书要求：
 *   - 不要把解释写死在 React component；
 *   - 建立 ExplanationRegistry（如 MOE_ROUTE / GEMM / MEMORY_LOAD / MMA /
 *     ATTENTION / CACHE_READ）；
 *   - 同一 Operator 不同模型可以重用 Explanation。
 *
 * 与 compiler/operatorKnowledge 同一模式：静态领域知识 + 纯查找，
 * 不 import React、不生成事件。模板展开时可调用 getExplanation() 获取
 * 统一文案；UI 侧可对未携带 what/why 的事件做投影回退。
 */

export interface Explanation {
  what: string;
  why: string;
}

/** 解释类别键（算子语义类别，非模型特定） */
export const EXPLANATION_KEYS = [
  'EMBEDDING',
  'GEMM',
  'ATTENTION',
  'MOE_ROUTE',
  'MOE_TOPK',
  'MOE_DISPATCH',
  'MOE_EXPERT',
  'MOE_COMBINE',
  'NORM',
  'RESIDUAL',
  'LM_HEAD',
  'MEMORY_LOAD',
  'MEMORY_STORE',
  'MMA',
  'CACHE_READ',
  'CACHE_WRITE',
  'PREFILL',
  'DECODE',
] as const;

export type ExplanationKey = (typeof EXPLANATION_KEYS)[number];

const registry = new Map<ExplanationKey, Explanation>();

function register(key: ExplanationKey, what: string, why: string): void {
  registry.set(key, { what, why });
}

register(
  'EMBEDDING',
  '把离散的 token id 查表映射为稠密向量。',
  '模型输入是离散符号，必须先变成连续向量才能参与矩阵运算。这是纯访存操作（gather），没有乘加计算。',
);
register(
  'GEMM',
  '执行一次通用矩阵乘法（GEMM）。',
  'GEMM 是 Transformer 中 Attention 与 FFN 的核心计算，由 Tensor Core 加速，是计算密集型操作。',
);
register(
  'ATTENTION',
  '计算注意力：Softmax(QKᵀ/√d) × V。',
  'Attention 让每个 token 关注序列中的其他 token，是 Transformer 交换信息的核心机制。',
);
register(
  'MOE_ROUTE',
  '执行 MoE Router：为每个 token 对全部专家计算匹配分数。',
  '并不是所有专家都会处理当前 Token。Router 根据当前 Token 表征计算专家选择分数，选择部分专家参与计算，从而降低激活计算量——这是 MoE 的核心思想。',
);
register(
  'MOE_TOPK',
  '执行 Top-K 选择：选出每个 token 分数最高的 K 个专家。',
  '只激活少数专家，让模型拥有巨大总参数量的同时，每个 token 的激活计算量只相当于一小部分专家——"大参数、低激活成本"。',
);
register(
  'MOE_DISPATCH',
  'Token Dispatch：把 token 按归属专家分组搬运。',
  '专家之间相互独立，必须先把 token 重排分组，每个专家才能连续处理属于它的 token 批。这是纯数据搬运，不做乘加。',
);
register(
  'MOE_EXPERT',
  'Expert Compute：被选中的专家对属于它的 token 做 FFN 计算。',
  '每个专家本质上是一个小型 FFN。被选中的专家并行计算各自的 token 子集——专家间无依赖，天然适合 GPU 并行。',
);
register(
  'MOE_COMBINE',
  'Combine：把多个专家的输出按路由权重加权求和。',
  '多专家的结果必须合并回每个 token 一条的隐藏向量，才能进入残差相加与下一层。',
);
register(
  'NORM',
  '对输入做归一化（如 RMSNorm），稳定数值尺度。',
  '归一化让每一层子层的输入保持稳定的数值尺度，显著提升训练与推理的稳定性。',
);
register(
  'RESIDUAL',
  '残差相加：X = X + 子层输出。',
  '残差连接提供梯度直达路径，让深层网络可训练，也让信息可以绕过子层直接传递。',
);
register(
  'LM_HEAD',
  '把最终隐藏向量投影到词表空间，得到下一个 token 的候选分数。',
  '语言模型的输出是"下一个词的概率分布"，LM Head 与词表嵌入相乘得到 logits，之后采样出下一个 token。',
);
register(
  'MEMORY_LOAD',
  '从低层级存储读取数据到更高层级缓存。',
  '计算前必须先把数据搬到更快的存储层级，这是 GPU 内存层级运作的基本规律。',
);
register(
  'MEMORY_STORE',
  '把计算结果写回存储。',
  '结果写回显存供下一个算子读取；写回时机与带宽是性能优化的关键考量。',
);
register(
  'MMA',
  'Tensor Core 执行矩阵乘累加（MMA）。',
  'Tensor Core 是专为矩阵乘累加设计的硬件单元，一条 MMA 指令完成一小块矩阵乘，远快于逐元素计算。',
);
register(
  'CACHE_READ',
  '读取 KV Cache 中历史 token 的 Key/Value。',
  'KV Cache 存放之前所有 token 的 K/V——Decode 不重算历史，但每步都要把它们读进来与新 Q 做注意力。序列越长带宽压力越大（Decode 是 memory-bound 的根源）。',
);
register(
  'CACHE_WRITE',
  '把新 token 的 K/V 追加写入 KV Cache。',
  '新 token 的 K/V 必须入缓存，后续步骤才能关注到它。KV Cache 因此每步增长。',
);
register(
  'PREFILL',
  'Prefill：并行处理整个输入序列。',
  '所有输入 token 同时参与计算，是计算密集型（compute-bound）阶段——与逐 token 生成的 Decode 形成鲜明对比。',
);
register(
  'DECODE',
  'Decode：逐 token 自回归生成。',
  '每次只前进一步，新 token 通过 KV Cache 看到所有历史。每步计算量小但要反复执行，是访存密集型（memory-bound）阶段。',
);

/** 按类别键获取解释（未登记返回 null，不臆测） */
export function getExplanation(key: ExplanationKey): Explanation | null {
  return registry.get(key) ?? null;
}

/**
 * operator 名 → 解释类别键的映射（统一复用，跨模型一致）。
 * 未命中返回 null（不硬归类，延续项目"不臆测"原则）。
 */
export function explanationKeyForOperator(operator: string | undefined): ExplanationKey | null {
  if (!operator) return null;
  const op = operator.toLowerCase();
  if (op.includes('router')) return 'MOE_ROUTE';
  if (op.includes('top-k') || op.includes('topk')) return 'MOE_TOPK';
  if (op.includes('dispatch')) return 'MOE_DISPATCH';
  if (op.includes('expert')) return 'MOE_EXPERT';
  if (op.includes('combine')) return 'MOE_COMBINE';
  if (op.includes('kv cache read')) return 'CACHE_READ';
  if (op.includes('kv cache write')) return 'CACHE_WRITE';
  if (op.includes('embedding')) return 'EMBEDDING';
  if (op.includes('lm head')) return 'LM_HEAD';
  if (op.includes('attention')) return 'ATTENTION';
  if (op.includes('projection') || op.includes('matmul') || op.includes('gemm')) return 'GEMM';
  if (op.includes('norm')) return 'NORM';
  if (op.includes('residual')) return 'RESIDUAL';
  if (op === 'prefill') return 'PREFILL';
  if (op === 'decode') return 'DECODE';
  return null;
}
