/**
 * 教学型 Transformer Block Simulation Engine（实施手册 §21，V0.3）。
 *
 * 计算图（Pre-Norm 结构，借鉴 llm.c 的"透明实现"思想）：
 *   Input X[seq×d]
 *   ↓
 *   RMSNorm        → Xn
 *   ↓
 *   Attention      → A   （直接嵌入 V0.2 的 Attention 子图原语，零复制）
 *   ↓
 *   Residual       → X + A
 *   ↓
 *   RMSNorm        → Xn2
 *   ↓
 *   FFN
 *   ├─ Up Projection   H = Xn2 × Wup   (d → 4d, GEMM)
 *   ├─ SiLU            H = SiLU(H)     (逐元素激活)
 *   └─ Down Projection F = H × Wdown   (4d → d, GEMM)
 *   ↓
 *   Residual       → X + F
 *
 * 架构要求（实施手册 §22 前置 + §31）：
 *   - Attention 必须复用 emitAttentionEvents 子图原语，禁止复制逻辑；
 *   - GEMM 复用 emitGemmEvents，逐元素类复用 emitElementwiseEvents；
 *   - 不修改 TVIR 基础架构，仅使用 operator / metadata 可选字段。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder, type EventBuilder } from './eventBuilder';
import { emitGemmEvents } from './gemmPrimitives';
import { emitElementwiseEvents } from './elementwisePrimitives';
import { emitAttentionEvents, type AttentionConfig } from './attentionEngine';

export interface TransformerBlockConfig {
  /** 序列长度（token 数） */
  seqLen: number;
  /** 模型隐藏维度 */
  dModel: number;
  /** 注意力头维度（单头） */
  headDim: number;
  /** FFN 中间层维度（通常为 4×dModel） */
  ffnDim: number;
  tileM: number;
  tileN: number;
  tileK: number;
  numSM: number;
  warpsPerBlock: number;
}

export const DEFAULT_TRANSFORMER_BLOCK_CONFIG: TransformerBlockConfig = {
  seqLen: 64,
  dModel: 64,
  headDim: 64,
  ffnDim: 256,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  numSM: 4,
  warpsPerBlock: 4,
};

/**
 * 生成教学型 Transformer Block trace。
 */
export function simulateTransformerBlock(config: TransformerBlockConfig): TVIRTrace {
  const { seqLen, dModel, headDim, ffnDim, tileM, tileN, tileK, numSM, warpsPerBlock } = config;

  const builder = createEventBuilder();
  const gemmBase = { tileM, tileN, tileK, numSM, warpsPerBlock };

  builder.push({
    type: 'GEMM_START',
    title: 'Transformer Block 开始',
    what: `开始模拟一个 Pre-Norm Transformer Block：输入 X[${seqLen}×${dModel}]。数据流：RMSNorm → Attention → Residual → RMSNorm → FFN(Up→SiLU→Down) → Residual。`,
    why: 'Transformer 的每一层都是这样一个 Block 的堆叠。Pre-Norm（先归一化再进子层）训练更稳定，是 LLaMA/Qwen 等主流模型的结构。理解一个 Block，就理解了整个模型在 GPU 上的执行骨架。',
    operator: 'Transformer Block',
    metadata: { seqLen, dModel, headDim, ffnDim },
  });

  // ---------- 1. RMSNorm（第一次，Attention 前） ----------
  emitRmsNormEvents(builder, { ...config, input: 'X', output: 'Xn', tag: 'RMSNorm (pre-Attention)' });

  // ---------- 2. Attention（直接嵌入 V0.2 子图原语，架构验收点） ----------
  // 注意：Attention 的输入是 RMSNorm 之后的 Xn（Pre-Norm 结构），
  // 通过 inputLabel 参数传入，矩阵视图与公式随之正确显示。
  const attentionConfig: AttentionConfig = {
    seqLen,
    dModel,
    headDim,
    tileM,
    tileN,
    tileK,
    numSM,
    warpsPerBlock,
  };
  emitAttentionEvents(builder, attentionConfig, '', 'Xn');

  // ---------- 3. Residual 1：X = X + AttnOut ----------
  emitResidualEvents(builder, {
    ...config,
    branch: 'AttnOut',
    label: 'Residual 1 (+ Attention)',
    why: '残差连接把 Attention 的输出 A 加回原始输入：X ← X + A。它提供梯度直达路径，让深层网络可以训练；也让信息可以"绕过"Attention 直接传递。',
  });

  // ---------- 4. RMSNorm（第二次，FFN 前） ----------
  emitRmsNormEvents(builder, { ...config, input: 'X', output: 'Xn2', tag: 'RMSNorm (pre-FFN)' });

  // ---------- 5. FFN：Up Projection → SiLU → Down Projection ----------
  builder.push({
    type: 'KERNEL_LAUNCH',
    title: 'FFN 开始：Up Projection → SiLU → Down Projection',
    what: `FFN（前馈网络）对每个 token 独立做两次 GEMM 夹一次非线性激活：先把维度从 ${dModel} 升到 ${ffnDim}，激活后再降回 ${dModel}。`,
    why: 'Attention 负责"token 之间交换信息"，FFN 负责"对每个 token 独立加工信息"。研究表明模型的大部分知识存储在 FFN 的权重里。FFN 也是 Transformer 中参数量最大的部分。',
    operator: 'FFN',
    metadata: { seqLen, dModel, ffnDim },
  });

  // 5.1 Up Projection：H = Xn2 × Wup （d → 4d）
  emitGemmEvents(builder, {
    ...gemmBase,
    M: seqLen,
    N: ffnDim,
    K: dModel,
    left: 'Xn2',
    right: 'Wup',
    out: 'H',
    operator: 'FFN Up Projection',
    kernel: 'ffn_up_kernel',
    label: 'FFN Up Projection',
    startWhat: `计算 H[${seqLen}×${ffnDim}] = Xn2[${seqLen}×${dModel}] × Wup[${dModel}×${ffnDim}]，把每个 token 升维到 ${ffnDim}（通常 4 倍 d_model）。`,
    startWhy: '升维给模型更大的表达空间：在高维空间里做非线性变换，更容易分离复杂的特征模式。这一步是标准 GEMM，由 Tensor Core 加速。',
  });

  // 5.2 SiLU 激活（逐元素）
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: ffnDim,
    numSM,
    warpsPerBlock,
    operator: 'FFN SiLU',
    kernel: 'silu_kernel',
    label: 'SiLU Activation',
    tensor: 'H',
    launchWhat: `启动 SiLU 激活 kernel：对 H 的每个元素计算 SiLU(x) = x · σ(x) = x / (1 + e⁻ˣ)。`,
    launchWhy: '没有非线性激活，两次 GEMM 叠加仍只是线性变换，模型无法表达复杂函数。SiLU（即 Swish）是 LLaMA/Qwen 等现代模型在 FFN 中的选择，比 ReLU 更平滑。（注：部分实现用 SwiGLU，需要额外的 gate 分支，本仿真以 SiLU 示意。）',
    computeWhat: '每个线程对寄存器中的元素计算 x·σ(x)：一次 exp、一次除法、一次乘法，全部在 CUDA Core 上完成。',
    computeWhy: '激活函数是逐元素操作，计算强度低、访存密集——与 Scale/Mask 同类，瓶颈在 HBM 带宽。',
  });

  // 5.3 Down Projection：F = H × Wdown （4d → d）
  emitGemmEvents(builder, {
    ...gemmBase,
    M: seqLen,
    N: dModel,
    K: ffnDim,
    left: 'H',
    right: 'Wdown',
    out: 'F',
    operator: 'FFN Down Projection',
    kernel: 'ffn_down_kernel',
    label: 'FFN Down Projection',
    startWhat: `计算 F[${seqLen}×${dModel}] = H[${seqLen}×${ffnDim}] × Wdown[${ffnDim}×${dModel}]，把维度降回 ${dModel}。`,
    startWhy: '降维把高维空间中的加工结果压缩回模型主干维度，以便参与残差相加。这一步同样是标准 GEMM。',
  });

  // ---------- 6. Residual 2：X = X + F ----------
  emitResidualEvents(builder, {
    ...config,
    branch: 'F',
    label: 'Residual 2 (+ FFN)',
    why: '第二个残差连接把 FFN 的输出 F 加回主干：X ← X + F。至此一个 Transformer Block 完成，输出 X 的形状与输入完全相同，可以直接送入下一个 Block。',
  });

  builder.push({
    type: 'GEMM_END',
    title: 'Transformer Block 完成',
    what: `Block 执行完毕：输出 X[${seqLen}×${dModel}] 形状不变，可作为下一个 Block 的输入。`,
    why: '回顾整层结构：两次"RMSNorm → 子层 → 残差"的循环。Attention 部分复用 V0.2 的完整子图（这正是本工具架构的验收点：子模块可嵌入、不重写）；FFN 部分是两次大 GEMM 夹一次激活。真实模型把几十个这样的 Block 串起来执行。',
    operator: 'Transformer Block',
    metadata: { seqLen, dModel, headDim, ffnDim },
  });

  return {
    description: `Transformer Block: RMSNorm→Attention→Residual→RMSNorm→FFN→Residual, seqLen=${seqLen}, d_model=${dModel}, ffn=${ffnDim}, tile=${tileM}×${tileN}×${tileK}, ${numSM} SM, ${warpsPerBlock} warps/block（Educational simulation, not cycle-accurate）`,
    events: builder.events,
  };
}

// ---------------------------------------------------------------------------
// Block 内部辅助算子
// ---------------------------------------------------------------------------

interface RmsNormEmitOptions extends TransformerBlockConfig {
  /** 输入张量名 */
  input: string;
  /** 输出张量名 */
  output: string;
  /** 阶段标签（用于标题） */
  tag: string;
}

/** RMSNorm：行归约 kernel（每行求 RMS 后归一化，再乘缩放参数 γ） */
function emitRmsNormEvents(builder: EventBuilder, options: RmsNormEmitOptions): void {
  const { seqLen, dModel, numSM, warpsPerBlock, input, output, tag } = options;
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: dModel,
    numSM,
    warpsPerBlock,
    operator: tag,
    kernel: 'rmsnorm_kernel',
    label: tag,
    tensor: input,
    launchWhat: `启动 RMSNorm kernel：对 ${input} 的每一行计算 RMS（均方根），把该行除以 RMS，再逐元素乘以可学习参数 γ，结果写入 ${output}。`,
    launchWhy: '归一化让每一层子层的输入保持稳定的数值尺度，显著提升训练稳定性与速度。RMSNorm 比 LayerNorm 少算一个均值，更省算力，是 LLaMA/Qwen 等现代模型的选择。',
    computeWhat: '行内分两步归约：① 求该行平方和（需要线程协作，类似 Softmax 的行归约）；② 每个元素计算 x / √(mean(x²)+ε) · γ。ε 是防除零的小常数。',
    computeWhy: 'RMSNorm 是访存密集 kernel：计算量很小，瓶颈在把整行数据从 HBM 读进来、再写回去。',
  });
}

interface ResidualEmitOptions extends TransformerBlockConfig {
  /** 子层输出张量名（AttnOut / F） */
  branch: string;
  /** 阶段标签 */
  label: string;
  /** 教学解释：为什么需要这个残差连接 */
  why: string;
}

/** Residual：X = X + Branch，逐元素加法 */
function emitResidualEvents(builder: EventBuilder, options: ResidualEmitOptions): void {
  const { seqLen, dModel, numSM, warpsPerBlock, branch, label, why } = options;
  emitElementwiseEvents(builder, {
    rows: seqLen,
    cols: dModel,
    numSM,
    warpsPerBlock,
    operator: label,
    kernel: 'residual_add_kernel',
    label,
    tensor: 'X',
    launchWhat: `启动残差相加 kernel：逐元素计算 X = X + ${branch}，把子层输出加回主干。`,
    launchWhy: why,
    computeWhat: `每个线程读取 X 与 ${branch} 中对应位置的元素，相加后写回 X。两个输入各读一次、结果写一次，是典型的访存密集操作。`,
    computeWhy: '残差相加本身几乎没有计算量，开销全在 HBM 读写。真实实现常把它与相邻 kernel 融合（如 fused add + RMSNorm），减少一轮 HBM 往返。',
  });
}
