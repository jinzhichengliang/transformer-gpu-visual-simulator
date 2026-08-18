/**
 * 教学型 GEMM Simulation Engine（实施手册 §10）。
 *
 * 输入：M N K / tileM tileN tileK / numSM / warpsPerBlock
 * 输出：TVIRTrace
 *
 * V0.2 起，事件生成委托给共享原语 emitGemmEvents（gemmPrimitives.ts），
 * 使 Attention 的 Q/K/V Projection、QK、AV 可以复用同一段逻辑
 * （实施手册 §20："已有 GEMM Engine 应该被重用，禁止复制 GEMM visualization code"）。
 *
 * 本引擎不模拟真实 CUDA cycle，只生成"教学上合理"的执行步骤。
 * 禁止：import React、操作 DOM、包含 CSS。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder } from './eventBuilder';
import { emitGemmEvents } from './gemmPrimitives';

export interface GemmConfig {
  M: number;
  N: number;
  K: number;
  tileM: number;
  tileN: number;
  tileK: number;
  numSM: number;
  warpsPerBlock: number;
}

export const DEFAULT_GEMM_CONFIG: GemmConfig = {
  M: 128,
  N: 128,
  K: 128,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  numSM: 4,
  warpsPerBlock: 4,
};

/**
 * 生成教学型 GEMM trace（C = A × B）。
 *
 * 事件结构（与实施手册 §17 的完整体验一致）：
 *  GEMM_START → TILE_CREATE → KERNEL_LAUNCH
 *  → 每个输出 Tile（= 一个 Block）：
 *      BLOCK_SCHEDULE → WARP_SCHEDULE
 *      → 每个 K Tile：
 *          MEMORY_LOAD(A HBM→L2) → MEMORY_MOVE(A L2→SMEM)
 *          MEMORY_LOAD(B HBM→L2) → MEMORY_MOVE(B L2→SMEM)
 *          SYNC → MEMORY_MOVE(SMEM→Register) → MMA → ACCUMULATE
 *      → MEMORY_STORE(C Register→HBM)
 *  → GEMM_END
 */
export function simulateGemm(config: GemmConfig): TVIRTrace {
  const { M, N, K, tileM, tileN, tileK, numSM, warpsPerBlock } = config;

  const builder = createEventBuilder();
  emitGemmEvents(builder, {
    ...config,
    left: 'A',
    right: 'B',
    out: 'C',
    operator: 'GEMM',
    kernel: 'gemm_tiled_kernel',
    label: 'GEMM',
  });

  return {
    description: `GEMM: C[${M}×${N}] = A[${M}×${K}] × B[${K}×${N}], tile=${tileM}×${tileN}×${tileK}, ${numSM} SM, ${warpsPerBlock} warps/block（Educational simulation, not cycle-accurate）`,
    events: builder.events,
  };
}
