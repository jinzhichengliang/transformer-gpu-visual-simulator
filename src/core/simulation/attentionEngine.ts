/**
 * 教学型 Attention Simulation Engine（实施手册 §19/§20，V0.2）。
 * V0.3 重构：抽取 emitAttentionEvents 作为可嵌入子图原语，
 * Transformer Block 直接复用它，不复制任何 Attention 逻辑。
 *
 * 计算图（单头注意力，借鉴 llm.c 的透明实现思想）：
 *   Q = X × Wq      (Q Projection)   — 复用 GEMM 原语
 *   K = X × Wk      (K Projection)   — 复用 GEMM 原语
 *   V = X × Wv      (V Projection)   — 复用 GEMM 原语
 *   S = Q × Kᵀ      (QK MatMul)      — 复用 GEMM 原语
 *   S = S / √d      (Scale)          — 逐元素 kernel
 *   S = Mask(S)     (Mask)           — 逐元素 kernel（causal）
 *   S = Softmax(S)  (Softmax)        — 行归约 kernel
 *   O = S × V       (AV MatMul)      — 复用 GEMM 原语
 *   Y = O × Wo      (Output Projection) — 复用 GEMM 原语
 *
 * 架构要求（实施手册 §20）：
 *   - 不修改 TVIR 基础架构，仅使用 operator / metadata 可选字段；
 *   - GEMM 部分全部复用 emitGemmEvents，禁止复制 GEMM 可视化逻辑；
 *   - 每个 Operator 生成带 what/why 的 TVIR events。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder, type EventBuilder } from './eventBuilder';
import { emitGemmEvents } from './gemmPrimitives';
import { emitElementwiseEvents } from './elementwisePrimitives';

export interface AttentionConfig {
  /** 序列长度（token 数） */
  seqLen: number;
  /** 模型隐藏维度 */
  dModel: number;
  /** 注意力头维度（单头） */
  headDim: number;
  tileM: number;
  tileN: number;
  tileK: number;
  numSM: number;
  warpsPerBlock: number;
}

export const DEFAULT_ATTENTION_CONFIG: AttentionConfig = {
  seqLen: 64,
  dModel: 64,
  headDim: 64,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  numSM: 4,
  warpsPerBlock: 4,
};

/**
 * 生成教学型 Attention trace。
 */
export function simulateAttention(config: AttentionConfig): TVIRTrace {
  const { seqLen, dModel, headDim, tileM, tileN, tileK, numSM, warpsPerBlock } = config;

  const builder = createEventBuilder();
  emitAttentionEvents(builder, config);

  return {
    description: `Attention: Softmax(QKᵀ/√d)×V, seqLen=${seqLen}, d_model=${dModel}, headDim=${headDim}, tile=${tileM}×${tileN}×${tileK}, ${numSM} SM, ${warpsPerBlock} warps/block（Educational simulation, not cycle-accurate）`,
    events: builder.events,
  };
}

/**
 * Attention 可嵌入子图原语（V0.3 架构核心）。
 *
 * 向 builder 追加完整的 Attention 事件流（含 Attention_START/END 包裹事件）。
 * Transformer Block 直接调用本函数嵌入 Attention，不复制任何逻辑。
 *
 * @param builder 共享事件构建器（事件编号自动连续）
 * @param config Attention 配置
 * @param prefix 可选前缀：嵌入 Transformer Block 时为算子名加前缀
 *               （如 "Block."），保证分段与跳转不混淆
 * @param inputLabel 输入张量标签（默认 "X"；Block 场景传 "Xn" 表示归一化后的输入）
 */
export function emitAttentionEvents(
  builder: EventBuilder,
  config: AttentionConfig,
  prefix = '',
  inputLabel = 'X',
): void {
  const { seqLen, dModel, headDim, tileM, tileN, tileK, numSM, warpsPerBlock } = config;

  const gemmBase = { tileM, tileN, tileK, numSM, warpsPerBlock };
  const tag = (name: string) => `${prefix}${name}`;

  builder.push({
    type: 'GEMM_START',
    title: 'Attention 开始：Softmax(QKᵀ/√d) × V',
    what: `开始模拟单头注意力：输入 ${inputLabel}[${seqLen}×${dModel}]，头维度 d=${headDim}。依次执行 Q/K/V Projection → QKᵀ → Scale → Mask → Softmax → ×V → Output Projection。`,
    why: 'Attention 让每个 token 直接关注序列中所有其他 token：Q 是"我想查什么"，K 是"我能被查到什么"，V 是"查到后我给什么"。它是 Transformer 的核心算子，而其中每一步在 GPU 上最终都落到 GEMM 与逐元素 kernel 上。',
    operator: tag('Attention'),
    metadata: { seqLen, dModel, headDim },
  });

  // ---------- 1-3. Q / K / V Projection（三次 GEMM：X[seq×d] × W[ d×head]） ----------
  const projections: Array<{ name: 'Q' | 'K' | 'V'; why: string }> = [
    { name: 'Q', why: 'Q（Query）表示"每个 token 想检索什么"。Q = X × Wq 是一次标准 GEMM：每个 token 的隐藏向量与权重矩阵相乘，投影到注意力子空间。' },
    { name: 'K', why: 'K（Key）表示"每个 token 能被怎样检索到"。K = X × Wk 同样是 GEMM。Q 与 K 的点积大小决定了两个 token 之间的注意力强度。' },
    { name: 'V', why: 'V（Value）表示"检索命中后实际传递的内容"。V = X × Wv 也是 GEMM。最终的注意力输出是 V 的加权和。' },
  ];

  for (const { name, why } of projections) {
    emitGemmEvents(builder, {
      ...gemmBase,
      M: seqLen,
      N: headDim,
      K: dModel,
      left: inputLabel,
      right: `W${name.toLowerCase()}`,
      out: name,
      operator: tag(`${name} Projection`),
      kernel: `proj_${name.toLowerCase()}_kernel`,
      label: `${name} Projection`,
      startWhat: `计算 ${name}[${seqLen}×${headDim}] = ${inputLabel}[${seqLen}×${dModel}] × W${name.toLowerCase()}[${dModel}×${headDim}]。`,
      startWhy: why,
    });
  }

  // ---------- 4. QK MatMul：S = Q × Kᵀ ----------
  emitGemmEvents(builder, {
    ...gemmBase,
    M: seqLen,
    N: seqLen,
    K: headDim,
    left: 'Q',
    right: 'Kᵀ',
    out: 'S',
    operator: tag('QK MatMul'),
    kernel: 'qk_matmul_kernel',
    label: 'QK MatMul',
    startWhat: `计算注意力分数 S[${seqLen}×${seqLen}] = Q[${seqLen}×${headDim}] × Kᵀ[${headDim}×${seqLen}]。`,
    startWhy: 'S[i][j] 是 token i 的 Query 与 token j 的 Key 的点积，衡量"token i 应该关注 token j 多少"。Kᵀ 表示 K 转置后参与 GEMM。这一步是 Attention 中开销最大的 GEMM 之一。',
  });

  // ---------- 5. Scale：S = S / √d ----------
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: seqLen,
    numSM,
    warpsPerBlock,
    operator: tag('Scale'),
    kernel: 'scale_kernel',
    label: 'Scale',
    tensor: 'S',
    launchWhat: `启动逐元素 kernel：把 S 的每个元素除以 √${headDim} ≈ ${Math.sqrt(headDim).toFixed(2)}。`,
    launchWhy: '点积的方差随维度 d 增大而增大，除以 √d 把分数拉回合理范围，防止 Softmax 进入饱和区导致梯度消失。这是 Attention 论文（"Attention Is All You Need"）中的 Scaled Dot-Product 设计。',
    computeWhat: '每个线程把自己负责的分数元素乘以 1/√d（常数预计算好，避免逐元素除法）。',
    computeWhy: '逐元素操作是访存密集型（memory-bound）：计算量很小，瓶颈在 HBM 读写带宽。',
  });

  // ---------- 6. Mask：causal mask ----------
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: seqLen,
    numSM,
    warpsPerBlock,
    operator: tag('Mask'),
    kernel: 'causal_mask_kernel',
    label: 'Mask (causal)',
    tensor: 'S',
    launchWhat: `启动 mask kernel：对 j > i 的位置（未来 token）把 S[i][j] 置为 −∞。`,
    launchWhy: '自回归语言模型在预测第 i 个 token 时不能"偷看"后面的内容。causal mask 把未来位置的分数置为 −∞，Softmax 后这些位置的注意力权重变为 0。',
    computeWhat: `每个 Block 处理一行：把该行中列号大于行号的位置写入 −∞（实现上常用一个极大负数，如 -1e9）。`,
    computeWhy: 'mask 只在 j > i 的上三角生效，是逐元素操作，同样访存密集。',
  });

  // ---------- 7. Softmax（行归约） ----------
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: seqLen,
    numSM,
    warpsPerBlock,
    operator: tag('Softmax'),
    kernel: 'softmax_row_kernel',
    label: 'Softmax (row-wise)',
    tensor: 'S',
    launchWhat: `启动 Softmax kernel：对 S 的每一行独立做 Softmax，使每行的注意力权重非负且和为 1。`,
    launchWhy: 'Softmax 把原始分数转换为概率分布：exp(x)/Σexp。行内归一化意味着"token i 对所有 token 的注意力加起来等于 1"。这是 Attention 权重具有概率含义的关键。',
    computeWhat: '行内分三步（安全 Softmax）：① 求行内最大值 m；② 每个元素算 exp(x−m) 并求和；③ 每个元素除以总和。求最大值与求和是行内归约，需要线程间协作（warp shuffle / shared memory reduction）。',
    computeWhy: '先减最大值是为了数值稳定：防止 exp 溢出。归约操作让 Softmax 比纯逐元素 kernel 多一次行内同步。',
  });

  // ---------- 8. AV MatMul：O = S × V ----------
  emitGemmEvents(builder, {
    ...gemmBase,
    M: seqLen,
    N: headDim,
    K: seqLen,
    left: 'S',
    right: 'V',
    out: 'O',
    operator: tag('AV MatMul'),
    kernel: 'av_matmul_kernel',
    label: 'AV MatMul',
    startWhat: `计算注意力输出 O[${seqLen}×${headDim}] = S[${seqLen}×${seqLen}] × V[${seqLen}×${headDim}]。`,
    startWhy: 'O 的每一行是 V 所有行的加权和，权重就是 Softmax 后的注意力分数：每个 token 的输出 = 它"关注到的内容"的混合。这一步把"关注谁"转换为"拿到什么"。',
  });

  // ---------- 9. Output Projection：Y = O × Wo ----------
  emitGemmEvents(builder, {
    ...gemmBase,
    M: seqLen,
    N: dModel,
    K: headDim,
    left: 'O',
    right: 'Wo',
    out: 'Y',
    operator: tag('Output Projection'),
    kernel: 'proj_o_kernel',
    label: 'Output Projection',
    startWhat: `计算 Y[${seqLen}×${dModel}] = O[${seqLen}×${headDim}] × Wo[${headDim}×${dModel}]，把注意力输出投影回模型隐藏维度。`,
    startWhy: '多头 Attention 中，这一步还负责把各头的结果拼接并线性变换（单头下即一次普通投影）。它是 Attention 模块与残差连接之间的最后一道线性变换。',
  });

  builder.push({
    type: 'GEMM_END',
    title: 'Attention 完成',
    what: `9 个 Operator 全部执行完毕：输出 Y[${seqLen}×${dModel}] 已写入 HBM。`,
    why: '回顾整条链路：Q/K/V Projection、QK、AV、Output Projection 都是 GEMM（Tensor Core 密集），Scale/Mask/Softmax 是逐元素/归约 kernel（CUDA Core、访存密集）。真实实现中 FlashAttention 会把 QK→Scale→Mask→Softmax→AV 融合进一个 kernel，避免 S 反复读写 HBM——那是 V0.2 之后的优化话题。',
    operator: tag('Attention'),
    metadata: { seqLen, dModel, headDim },
  });
}
