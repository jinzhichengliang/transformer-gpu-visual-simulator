/**
 * 内置示例真实 trace（V0.5）。
 *
 * 数据形态与 Nsight Systems 导出的 JSON 一致，但数值是教学示意值
 * （sample: true），不代表任何真实硬件的实测结果。
 * 用途：让 Real Trace 模式开箱即用；用户可用自己的真实导出文件替换。
 *
 * 场景：一个 Transformer Block（seq=64, d_model=64, ffn=256）在示例 GPU 上
 * 执行一遍的 kernel 时间轴（与仿真模式的 Transformer Block 场景对应）。
 */

import type { NsightTraceFile } from './nsightParser';

export const SAMPLE_REAL_TRACE: NsightTraceFile = {
  sample: true,
  meta: {
    tool: 'Nsight Systems（示例导出格式）',
    gpu: 'Example GPU（示意，仅展示 4 个 SM）',
    capturedAt: '2026-08-17（示例）',
    command: 'python train.py --steps 1（示例命令）',
    smCount: 4,
  },
  kernels: [
    {
      name: 'rmsnorm_row_kernel',
      startNs: 0,
      durationNs: 3200,
      grid: [64, 1, 1],
      block: [64, 1, 1],
      operator: 'RMSNorm (pre-Attention)',
    },
    {
      name: 'cutlass_tensorop_gemm_q_proj',
      startNs: 3900,
      durationNs: 8500,
      grid: [4, 4, 1],
      block: [128, 1, 1],
      operator: 'Q Projection',
      metrics: { tensorCoreUtilization: 78, l2HitRate: 62, memoryBandwidthGBps: 890, occupancy: 55, arithmeticIntensity: 48 },
    },
    {
      name: 'cutlass_tensorop_gemm_k_proj',
      startNs: 12900,
      durationNs: 8400,
      grid: [4, 4, 1],
      block: [128, 1, 1],
      operator: 'K Projection',
    },
    {
      name: 'cutlass_tensorop_gemm_v_proj',
      startNs: 21800,
      durationNs: 8600,
      grid: [4, 4, 1],
      block: [128, 1, 1],
      operator: 'V Projection',
    },
    {
      name: 'cutlass_tensorop_gemm_qk',
      startNs: 31000,
      durationNs: 24000,
      grid: [8, 8, 1],
      block: [256, 1, 1],
      operator: 'QK MatMul',
      metrics: { tensorCoreUtilization: 86, l2HitRate: 41, memoryBandwidthGBps: 1320, occupancy: 60, arithmeticIntensity: 32 },
    },
    {
      name: 'fused_scale_mask_softmax_kernel',
      startNs: 55600,
      durationNs: 6800,
      grid: [64, 1, 1],
      block: [128, 1, 1],
      operator: 'Softmax',
    },
    {
      name: 'cutlass_tensorop_gemm_av',
      startNs: 63000,
      durationNs: 22600,
      grid: [8, 8, 1],
      block: [256, 1, 1],
      operator: 'AV MatMul',
    },
    {
      name: 'cutlass_tensorop_gemm_o_proj',
      startNs: 86200,
      durationNs: 8700,
      grid: [4, 4, 1],
      block: [128, 1, 1],
      operator: 'Output Projection',
    },
    {
      name: 'residual_add_kernel',
      startNs: 95400,
      durationNs: 2100,
      grid: [16, 1, 1],
      block: [256, 1, 1],
      operator: 'Residual 1 (+ Attention)',
    },
    {
      name: 'rmsnorm_row_kernel',
      startNs: 98000,
      durationNs: 3300,
      grid: [64, 1, 1],
      block: [64, 1, 1],
      operator: 'RMSNorm (pre-FFN)',
    },
    {
      name: 'cutlass_tensorop_gemm_ffn_up',
      startNs: 101900,
      durationNs: 31500,
      grid: [8, 8, 1],
      block: [256, 1, 1],
      operator: 'FFN Up Projection',
      metrics: { tensorCoreUtilization: 91, l2HitRate: 38, memoryBandwidthGBps: 1410, occupancy: 58, arithmeticIntensity: 52 },
    },
    {
      name: 'silu_elementwise_kernel',
      startNs: 134000,
      durationNs: 3900,
      grid: [32, 1, 1],
      block: [256, 1, 1],
      operator: 'FFN SiLU',
    },
    {
      name: 'cutlass_tensorop_gemm_ffn_down',
      startNs: 138500,
      durationNs: 30800,
      grid: [8, 8, 1],
      block: [256, 1, 1],
      operator: 'FFN Down Projection',
    },
    {
      name: 'residual_add_kernel',
      startNs: 169800,
      durationNs: 2100,
      grid: [16, 1, 1],
      block: [256, 1, 1],
      operator: 'Residual 2 (+ FFN)',
    },
  ],
};
