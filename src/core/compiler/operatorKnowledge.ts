/**
 * Compiler View 知识层（V0.4，借鉴 tinygrad 的 Operator → IR → Kernel 思想）。
 *
 * 职责：把 TVIR 事件中的 operator/kernel 字段投影为"编译下钻链"：
 *   Math（数学公式） → Operator（算子类型） → IR（中间表示指令序列）
 *   → Kernel（GPU kernel 类型） → GPU（执行硬件单元）
 *
 * 架构边界（见 ARCHITECTURE.md）：
 *   - 本模块是静态编译知识 + 对 TVIR 事件的纯投影，不生成执行事件；
 *   - 不 import React、不依赖 Simulation Engine 内部实现；
 *   - 只读取事件的 operator / kernel / metadata 公开字段。
 */

import type { TVIREvent } from '../tvir/types';

/** IR 指令（教学型中间表示，风格借鉴 tinygrad 的 LOAD/MUL/ACC） */
export interface IRInstruction {
  /** 指令助记符，如 LOAD / MUL / ACC / STORE */
  op: string;
  /** 该指令的教学解释 */
  meaning: string;
}

/** 一条完整的编译下钻链 */
export interface CompileChain {
  /** 算子名（来自事件的 operator 字段） */
  operator: string;
  /** 数学层：该算子表达的数学公式 */
  math: string;
  /** 算子层：算子的计算类型（MatMul / Elementwise / RowReduce） */
  operatorType: string;
  /** IR 层：中间表示指令序列 */
  ir: IRInstruction[];
  /** Kernel 层：映射到的 GPU kernel 类型 */
  kernelType: string;
  /** Kernel 名（来自事件的 kernel 字段，可能为空） */
  kernel?: string | undefined;
  /** GPU 层：执行该 kernel 的硬件单元 */
  gpuUnit: string;
  /** GPU 层补充：该 kernel 的瓶颈特征 */
  bottleneck: string;
}

/** 算子类型分类 */
export type OperatorType = 'MatMul' | 'Elementwise' | 'RowReduce';

interface OperatorRecipe {
  operatorType: OperatorType;
  /** 数学公式模板（GEMM 类由 metadata.gemm 动态生成，此处为静态回退） */
  math: string;
  ir: IRInstruction[];
  kernelType: string;
  gpuUnit: string;
  bottleneck: string;
}

const GEMM_IR: IRInstruction[] = [
  { op: 'LOAD', meaning: '从 Shared Memory 加载 A fragment 到寄存器' },
  { op: 'LOAD', meaning: '从 Shared Memory 加载 B fragment 到寄存器' },
  { op: 'MUL', meaning: 'Tensor Core 执行 MMA 矩阵乘（D = A × B + C）' },
  { op: 'ACC', meaning: '累加到 accumulator（沿 K 维度的部分和）' },
  { op: 'STORE', meaning: '累加完成后把结果 Tile 写回 HBM' },
];

const ELEMENTWISE_IR: IRInstruction[] = [
  { op: 'LOAD', meaning: '从内存加载输入元素到寄存器' },
  { op: 'COMPUTE', meaning: 'CUDA Core 逐元素计算（每个线程独立处理自己的元素）' },
  { op: 'STORE', meaning: '把结果写回内存' },
];

/**
 * 算子名 → 编译配方 知识表。
 *
 * 注意：算子名与 Simulation 层写入事件 operator 字段的值一一对应。
 * 新增算子时，Simulation 层与本表需同步登记（这是 V0.4 约定的唯一耦合点）。
 */
const OPERATOR_RECIPES: Record<string, OperatorRecipe> = {
  GEMM: {
    operatorType: 'MatMul',
    math: 'C = A × B',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）：Tile 化后每个元素被复用，瓶颈转移到 Tensor Core 吞吐。',
  },
  'Q Projection': {
    operatorType: 'MatMul',
    math: 'Q = X × Wq',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  'K Projection': {
    operatorType: 'MatMul',
    math: 'K = X × Wk',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  'V Projection': {
    operatorType: 'MatMul',
    math: 'V = X × Wv',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  'QK MatMul': {
    operatorType: 'MatMul',
    math: 'S = Q × Kᵀ',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）；S 的尺寸为 seq×seq，是 Attention 中最大的中间张量。',
  },
  'AV MatMul': {
    operatorType: 'MatMul',
    math: 'O = S × V',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  'Output Projection': {
    operatorType: 'MatMul',
    math: 'Y = O × Wo',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  'FFN Up Projection': {
    operatorType: 'MatMul',
    math: 'H = Xn2 × Wup',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）；升维 4× 使这次 GEMM 成为 Block 中 FLOPs 最大的算子之一。',
  },
  'FFN Down Projection': {
    operatorType: 'MatMul',
    math: 'F = H × Wdown',
    ir: GEMM_IR,
    kernelType: 'Tiled GEMM Kernel',
    gpuUnit: 'Tensor Core',
    bottleneck: '计算密集（compute-bound）。',
  },
  Scale: {
    operatorType: 'Elementwise',
    math: 'S = S / √d',
    ir: [
      { op: 'LOAD', meaning: '加载分数元素到寄存器' },
      { op: 'MUL', meaning: '乘以预计算常数 1/√d（避免逐元素除法）' },
      { op: 'STORE', meaning: '写回缩放后的分数' },
    ],
    kernelType: 'Elementwise Kernel',
    gpuUnit: 'CUDA Core',
    bottleneck: '访存密集（memory-bound）：每个元素只做一次乘法，瓶颈在 HBM 读写带宽。',
  },
  Mask: {
    operatorType: 'Elementwise',
    math: 'S[i][j] = −∞ (当 j > i)',
    ir: [
      { op: 'LOAD', meaning: '加载分数元素与行列坐标' },
      { op: 'CMP', meaning: '比较列号 j 与行号 i（判断是否在未来位置）' },
      { op: 'SELECT', meaning: 'j > i 则写入 −∞（极大负数），否则保留原值' },
      { op: 'STORE', meaning: '写回 mask 后的分数' },
    ],
    kernelType: 'Elementwise Kernel',
    gpuUnit: 'CUDA Core',
    bottleneck: '访存密集（memory-bound）。',
  },
  Softmax: {
    operatorType: 'RowReduce',
    math: 'S[i] = exp(S[i] − max) / Σ exp(S[i] − max)',
    ir: [
      { op: 'LOAD', meaning: '加载整行分数' },
      { op: 'REDUCE', meaning: '行内归约①：求最大值 max（数值稳定）' },
      { op: 'EXP', meaning: '每个元素计算 exp(x − max)' },
      { op: 'REDUCE', meaning: '行内归约②：求 exp 之和' },
      { op: 'DIV', meaning: '每个元素除以总和，得到概率' },
      { op: 'STORE', meaning: '写回注意力权重' },
    ],
    kernelType: 'Row-Reduction Kernel',
    gpuUnit: 'CUDA Core + Warp Shuffle',
    bottleneck: '访存密集 + 行内归约开销：两次归约需要线程间协作（warp shuffle / SMEM reduction）。',
  },
  'RMSNorm (pre-Attention)': {
    operatorType: 'RowReduce',
    math: 'Xn = X / √(mean(X²) + ε) · γ',
    ir: [
      { op: 'LOAD', meaning: '加载整行隐藏向量' },
      { op: 'REDUCE', meaning: '行内归约：求平方和 mean(X²)' },
      { op: 'RSQRT', meaning: '计算 1/√(mean + ε)' },
      { op: 'MUL', meaning: '逐元素乘归一化系数与可学习参数 γ' },
      { op: 'STORE', meaning: '写回归一化结果' },
    ],
    kernelType: 'Row-Reduction Kernel',
    gpuUnit: 'CUDA Core + Warp Shuffle',
    bottleneck: '访存密集（memory-bound）：计算量很小，瓶颈在整行数据的读写。',
  },
  'RMSNorm (pre-FFN)': {
    operatorType: 'RowReduce',
    math: 'Xn2 = X / √(mean(X²) + ε) · γ',
    ir: [
      { op: 'LOAD', meaning: '加载整行隐藏向量' },
      { op: 'REDUCE', meaning: '行内归约：求平方和 mean(X²)' },
      { op: 'RSQRT', meaning: '计算 1/√(mean + ε)' },
      { op: 'MUL', meaning: '逐元素乘归一化系数与可学习参数 γ' },
      { op: 'STORE', meaning: '写回归一化结果' },
    ],
    kernelType: 'Row-Reduction Kernel',
    gpuUnit: 'CUDA Core + Warp Shuffle',
    bottleneck: '访存密集（memory-bound）。',
  },
  'Residual 1 (+ Attention)': {
    operatorType: 'Elementwise',
    math: 'X = X + AttnOut',
    ir: [
      { op: 'LOAD', meaning: '加载主干 X 与 Attention 输出' },
      { op: 'ADD', meaning: '逐元素相加' },
      { op: 'STORE', meaning: '写回主干' },
    ],
    kernelType: 'Elementwise Kernel',
    gpuUnit: 'CUDA Core',
    bottleneck: '访存密集（memory-bound）：两个输入各读一次、结果写一次，常与相邻 kernel 融合。',
  },
  'Residual 2 (+ FFN)': {
    operatorType: 'Elementwise',
    math: 'X = X + F',
    ir: [
      { op: 'LOAD', meaning: '加载主干 X 与 FFN 输出' },
      { op: 'ADD', meaning: '逐元素相加' },
      { op: 'STORE', meaning: '写回主干' },
    ],
    kernelType: 'Elementwise Kernel',
    gpuUnit: 'CUDA Core',
    bottleneck: '访存密集（memory-bound），常与相邻 kernel 融合。',
  },
  'FFN SiLU': {
    operatorType: 'Elementwise',
    math: 'H = H · σ(H) = H / (1 + e⁻ᴴ)',
    ir: [
      { op: 'LOAD', meaning: '加载元素到寄存器' },
      { op: 'EXP', meaning: '计算 e⁻ˣ' },
      { op: 'ADD', meaning: '计算 1 + e⁻ˣ' },
      { op: 'DIV', meaning: '计算 x / (1 + e⁻ˣ)' },
      { op: 'STORE', meaning: '写回激活结果' },
    ],
    kernelType: 'Elementwise Kernel',
    gpuUnit: 'CUDA Core',
    bottleneck: '访存密集（memory-bound）：每元素几次浮点运算，瓶颈在带宽。',
  },
};

/** Attention / Transformer Block 等包裹型算子没有独立 kernel，给出概览配方 */
const WRAPPER_RECIPES: Record<string, OperatorRecipe> = {
  Attention: {
    operatorType: 'MatMul',
    math: 'Attn(X) = Softmax(QKᵀ/√d) × V',
    ir: [
      { op: 'CALL', meaning: 'Q/K/V Projection（3 次 GEMM）' },
      { op: 'CALL', meaning: 'QK MatMul（GEMM）' },
      { op: 'CALL', meaning: 'Scale / Mask / Softmax（逐元素与行归约）' },
      { op: 'CALL', meaning: 'AV MatMul（GEMM）+ Output Projection（GEMM）' },
    ],
    kernelType: '多个 kernel 的组合（FlashAttention 可融合为单 kernel）',
    gpuUnit: 'Tensor Core + CUDA Core',
    bottleneck: 'GEMM 部分计算密集；Scale/Mask/Softmax 访存密集。真实实现用 FlashAttention 融合以消除 S 的 HBM 往返。',
  },
  FFN: {
    operatorType: 'MatMul',
    math: 'FFN(x) = Down(SiLU(Up(x)))',
    ir: [
      { op: 'CALL', meaning: 'Up Projection（GEMM，d → 4d）' },
      { op: 'CALL', meaning: 'SiLU 激活（逐元素）' },
      { op: 'CALL', meaning: 'Down Projection（GEMM，4d → d）' },
    ],
    kernelType: 'GEMM + Elementwise 组合',
    gpuUnit: 'Tensor Core + CUDA Core',
    bottleneck: '两次大 GEMM 计算密集；激活与读写访存密集。',
  },
  'Transformer Block': {
    operatorType: 'MatMul',
    math: 'Block(X) = X + FFN(Norm(X + Attn(Norm(X))))',
    ir: [
      { op: 'CALL', meaning: 'RMSNorm → Attention（子图）' },
      { op: 'CALL', meaning: 'Residual → RMSNorm → FFN（子图）' },
      { op: 'CALL', meaning: 'Residual' },
    ],
    kernelType: '完整算子图的 kernel 序列',
    gpuUnit: 'Tensor Core + CUDA Core',
    bottleneck: '整体由 GEMM 主导；真实推理中逐元素算子常用 kernel 融合减少 HBM 往返。',
  },
};

/** 未知算子的回退配方（保证任意 trace 都可展示） */
const FALLBACK_RECIPE: OperatorRecipe = {
  operatorType: 'Elementwise',
  math: '（该算子暂无编译知识登记）',
  ir: ELEMENTWISE_IR,
  kernelType: 'Kernel',
  gpuUnit: 'GPU',
  bottleneck: '—',
};

/** GEMM 类元数据（与 metadata.gemm 约定一致） */
interface GemmMeta {
  left: string;
  right: string;
  out: string;
}

/**
 * 把当前 TVIR 事件投影为编译下钻链。
 *
 * 纯消费 TVIR：只读取 event.operator / event.kernel / event.metadata.gemm。
 * 无 operator 字段的事件（如手写示例 trace 的 GEMM_START）返回 null。
 */
export function projectCompileChain(event: TVIREvent | null): CompileChain | null {
  if (!event || !event.operator) return null;

  const recipe = OPERATOR_RECIPES[event.operator] ?? WRAPPER_RECIPES[event.operator] ?? FALLBACK_RECIPE;

  // GEMM 类算子：优先用事件的 metadata.gemm 动态生成精确数学公式
  let math = recipe.math;
  if (recipe.operatorType === 'MatMul' && !WRAPPER_RECIPES[event.operator]) {
    const gemm = (event.metadata as { gemm?: GemmMeta } | undefined)?.gemm;
    if (gemm) {
      math = `${gemm.out} = ${gemm.left} × ${gemm.right}`;
    }
  }

  return {
    operator: event.operator,
    math,
    operatorType: recipe.operatorType,
    ir: recipe.ir,
    kernelType: recipe.kernelType,
    ...(event.kernel !== undefined ? { kernel: event.kernel } : {}),
    gpuUnit: recipe.gpuUnit,
    bottleneck: recipe.bottleneck,
  };
}

/** 判断某算子名是否已登记编译知识（供测试与 UI 提示） */
export function hasCompileRecipe(operator: string): boolean {
  return operator in OPERATOR_RECIPES || operator in WRAPPER_RECIPES;
}
