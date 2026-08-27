/**
 * ModelProfile Framework — 统一模型描述 Schema（Sprint 1, Task A1/A2）。
 *
 * 架构要求（任务书 §4）：
 *   - 与 UI、具体模型代码完全解耦；
 *   - 不得出现 DeepSeek/Kimi 等特定模型逻辑；
 *   - 必须能表示 Dense Transformer、MoE Transformer、不同 Attention 类型、不同 Layer 类型；
 *   - 所有关键字段支持 source metadata（来源、可信度）；
 *   - 支持 fidelity level。
 *
 * 本模块不 import React，不 import 仿真引擎。
 */

// ─────────────────────────────────────────────
// Source Metadata 与 Fidelity（Task A2）
// ─────────────────────────────────────────────

/**
 * 参数来源类型（任务书 §5）。
 * - official：官方公开文档（技术报告/博客）
 * - official_repo：官方代码仓库（config.json 等）
 * - technical_report：技术报告（论文等）
 * - runtime_repo：运行时/推理框架仓库（vLLM 等）
 * - inferred：从公开信息推断
 * - estimated：估算（无直接依据）
 */
export const SOURCE_TYPES = [
  'official',
  'official_repo',
  'technical_report',
  'runtime_repo',
  'inferred',
  'estimated',
] as const;

export type SourceType = (typeof SOURCE_TYPES)[number];

/** 来源可信度（对 inferred/estimated 尤为重要） */
export const CONFIDENCE_LEVELS = ['high', 'medium', 'low'] as const;

export type ConfidenceLevel = (typeof CONFIDENCE_LEVELS)[number];

/**
 * 来源元数据。每个参数不能只有值，还必须有：
 * value + source + sourceType + verifiedAt + confidence。
 */
export interface SourceMetadata {
  sourceType: SourceType;
  /** 来源标识（文档标题、仓库地址等） */
  reference: string;
  /** 核验日期（ISO 8601，如 "2026-08-26"） */
  verifiedAt: string;
  confidence: ConfidenceLevel;
}

/**
 * 保真度层级（任务书 §5）。
 * - L1 Architecture-Based：基于公开架构信息
 * - L2 Open-Implementation-Based：基于开源实现
 * - L3 Runtime-Based：基于运行时框架行为
 * - L4 Profile-Trace-Based：基于实测 trace
 * - L5 Architecture-Simulation-Based：基于架构仿真（本项目所处层级）
 */
export const FIDELITY_LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5'] as const;

export type FidelityLevel = (typeof FIDELITY_LEVELS)[number];

/**
 * 带溯源的值包装（Task A2 核心）。
 * 任何重要参数都必须通过 Traced<T> 表达，携带来源与保真度。
 */
export interface Traced<T> {
  value: T;
  sources: SourceMetadata[];
  fidelity: FidelityLevel;
}

// ─────────────────────────────────────────────
// Layer 类型（Task A1）
// ─────────────────────────────────────────────

export const LAYER_TYPES = [
  'attention',
  'ffn',
  'moe',
  'norm',
  'residual',
  'embedding',
  'lm_head',
  'other',
] as const;

export type LayerType = (typeof LAYER_TYPES)[number];

/**
 * Attention 类型。
 * - mha：Multi-Head Attention（Q/K/V 头数相同）
 * - mqa：Multi-Query Attention（共享 1 组 K/V）
 * - gqa：Grouped-Query Attention（K/V 头数是 Q 的分组子集）
 * - mla：Multi-Latent Attention（低秩压缩 KV）
 */
export const ATTENTION_TYPES = ['mha', 'mqa', 'gqa', 'mla'] as const;

export type AttentionType = (typeof ATTENTION_TYPES)[number];

/** Attention 层结构参数 */
export interface AttentionProfile {
  attentionType: AttentionType;
  /** Query 头数 */
  numHeads?: Traced<number>;
  /** Key/Value 头数（MHA 时等于 numHeads，MQA=1，GQA 为分组数） */
  numKVHeads?: Traced<number>;
  /** 每头维度 */
  headDim?: Traced<number>;
  sources?: SourceMetadata[];
}

/** MoE 层结构参数 */
export interface MoEProfile {
  /** 专家总数 */
  numExperts: Traced<number>;
  /** 每 token 激活的专家数（Top-K） */
  expertsPerToken: Traced<number>;
  /** 是否有共享专家（Shared Expert） */
  hasSharedExperts?: Traced<boolean>;
  sources?: SourceMetadata[];
}

/** FFN 层结构参数 */
export interface FFNProfile {
  /** 中间层维度（intermediate size） */
  intermediateSize?: Traced<number>;
  /** 激活类型（如 silu、gelu） */
  activation?: string;
  sources?: SourceMetadata[];
}

/**
 * 层描述。一个 ModelProfile 由多个 LayerProfile 按序组成，
 * 支持 Dense（attention+ffn）与 MoE（attention+moe）以及混合结构。
 */
export interface LayerProfile {
  type: LayerType;
  /** Attention 层细节（type=attention 时） */
  attention?: AttentionProfile;
  /** MoE 层细节（type=moe 时） */
  moe?: MoEProfile;
  /** FFN 层细节（type=ffn 时） */
  ffn?: FFNProfile;
  sources?: SourceMetadata[];
}

// ─────────────────────────────────────────────
// 顶层 ModelProfile
// ─────────────────────────────────────────────

/** 精度描述 */
export interface PrecisionProfile {
  /** 参数精度（如 "fp16"、"bf16"、"fp8"） */
  paramPrecision: Traced<string>;
  /** 计算精度（推理时实际使用的精度） */
  computePrecision?: Traced<string>;
}

/**
 * ModelProfile — 与具体模型完全解耦的统一描述。
 *
 * 验收标准（任务书 §4 PASS）：
 *   - schema 能表达 Dense 和 MoE；
 *   - schema 不含特定模型名称判断；
 *   - 所有关键字段支持 source metadata。
 */
export interface ModelProfile {
  /** 唯一标识（如 "deepseek-v4-flash"） */
  id: string;
  /** 展示名称 */
  displayName: string;
  /** 模型家族（如 "DeepSeek-V4"、"Kimi"、"GLM"） */
  family: string;
  /** 版本 */
  version: string;

  /** 架构总览 */
  architecture: ArchitectureProfile;

  /** 按执行顺序排列的层序列（Embedding → Layers → LM Head） */
  layers: LayerProfile[];

  /** 上下文长度 */
  contextLength?: Traced<number>;

  /** 参数规模信息 */
  parameterInfo?: {
    /** 总参数量 */
    total?: Traced<number>;
    /** 激活参数量（MoE 模型每 token 实际参与计算的参数） */
    activated?: Traced<number>;
  };

  precision?: PrecisionProfile;

  /** Profile 整体来源 */
  source: SourceMetadata[];

  /** Profile 整体保真度 */
  fidelity: FidelityLevel;
}

/** 架构总览（Dense / MoE / Hybrid） */
export interface ArchitectureProfile {
  /** dense=全层 FFN；moe=含 MoE 层；hybrid=多种层类型混合 */
  type: 'dense' | 'moe' | 'hybrid';
  hiddenSize?: Traced<number>;
  vocabSize?: Traced<number>;
  /** Norm 类型（如 "rmsnorm"、"layernorm"） */
  normType?: string;
  /** 位置编码类型（如 "rope"、"alibi"） */
  positionalEncoding?: string;
}
