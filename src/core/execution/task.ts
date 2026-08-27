/**
 * InferenceTask — 推理任务定义（Sprint 4, Task C1）。
 *
 * 用户不只选模型，还必须选推理任务：
 *   phase（prefill / decode / prefill_decode）
 *   batchSize / promptTokens / outputTokens
 *   hardwareProfile（GPU 配置）
 *
 * 验收（任务书 §7）：
 *   - 合法输入成功；非法输入（负数、0、NaN）给出明确错误；
 *   - 不得产生 NaN / undefined execution plan。
 *
 * 本模块是纯逻辑层，不 import React。
 */

export const INFERENCE_PHASES = ['prefill', 'decode', 'prefill_decode'] as const;

export type InferencePhase = (typeof INFERENCE_PHASES)[number];

/** GPU 硬件配置（教学级抽象） */
export interface HardwareProfile {
  /** SM 数量 */
  numSM: number;
  /** 每 Block 的 Warp 数 */
  warpsPerBlock: number;
  /** GEMM tiling 尺寸（M/N/K 共用） */
  tileSize: number;
}

export const DEFAULT_HARDWARE_PROFILE: HardwareProfile = {
  numSM: 4,
  warpsPerBlock: 4,
  tileSize: 32,
};

/** 推理任务 */
export interface InferenceTask {
  phase: InferencePhase;
  /** 批大小（并发请求数） */
  batchSize: number;
  /** 输入提示词长度（token 数） */
  promptTokens: number;
  /** 生成长度（token 数） */
  outputTokens: number;
  hardwareProfile: HardwareProfile;
}

export const DEFAULT_INFERENCE_TASK: InferenceTask = {
  phase: 'prefill',
  batchSize: 1,
  promptTokens: 128,
  outputTokens: 4,
  hardwareProfile: DEFAULT_HARDWARE_PROFILE,
};

export interface TaskValidationResult {
  valid: boolean;
  errors: string[];
}

function isPositiveInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && Number.isInteger(v) && v >= 1;
}

/**
 * 校验推理任务。
 * 任务书 §7：promptTokens=-10、batch=0、outputTokens=-1 必须失败。
 */
export function validateInferenceTask(task: InferenceTask): TaskValidationResult {
  const errors: string[] = [];

  if (!INFERENCE_PHASES.includes(task.phase)) {
    errors.push(`phase 非法：${String(task.phase)}（必须为 ${INFERENCE_PHASES.join('/')})`);
  }

  if (!isPositiveInt(task.batchSize)) {
    errors.push(`batchSize 必须是 ≥1 的整数（收到 ${String(task.batchSize)}）`);
  }
  if (!isPositiveInt(task.promptTokens)) {
    errors.push(`promptTokens 必须是 ≥1 的整数（收到 ${String(task.promptTokens)}）`);
  }
  // outputTokens：prefill 阶段允许 0；包含 decode 的阶段必须 ≥1
  if (typeof task.outputTokens !== 'number' || !Number.isFinite(task.outputTokens) || !Number.isInteger(task.outputTokens) || task.outputTokens < 0) {
    errors.push(`outputTokens 必须是 ≥0 的整数（收到 ${String(task.outputTokens)}）`);
  } else if (task.outputTokens === 0 && task.phase !== 'prefill') {
    errors.push(`${task.phase} 阶段需要 outputTokens ≥1（decode 才有意义）`);
  }

  const hw = task.hardwareProfile;
  if (!hw) {
    errors.push('hardwareProfile 缺失');
  } else {
    if (!isPositiveInt(hw.numSM)) errors.push(`hardwareProfile.numSM 必须是 ≥1 的整数（收到 ${String(hw.numSM)}）`);
    if (!isPositiveInt(hw.warpsPerBlock)) errors.push(`hardwareProfile.warpsPerBlock 必须是 ≥1 的整数（收到 ${String(hw.warpsPerBlock)}）`);
    if (!isPositiveInt(hw.tileSize)) errors.push(`hardwareProfile.tileSize 必须是 ≥1 的整数（收到 ${String(hw.tileSize)}）`);
  }

  return { valid: errors.length === 0, errors };
}

/** 该任务是否包含 prefill 阶段 */
export function hasPrefill(task: InferenceTask): boolean {
  return task.phase === 'prefill' || task.phase === 'prefill_decode';
}

/** 该任务是否包含 decode 阶段 */
export function hasDecode(task: InferenceTask): boolean {
  return task.phase === 'decode' || task.phase === 'prefill_decode';
}
