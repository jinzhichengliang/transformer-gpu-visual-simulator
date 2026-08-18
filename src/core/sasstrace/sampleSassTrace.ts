/**
 * 内置示例 SASS trace（V0.8）。
 *
 * 数据形态与 NVBit 采集的指令流一致，但指令序列是为教学编排的示意
 * （sample: true）：展示一个 tiled GEMM kernel 在单个 Warp 上的典型
 * 指令骨架——地址计算 → 异步搬运 tile → 同步 → LDS 喂 HMMA 循环 → STG
 * 写回。时长与顺序不代表任何真实采集结果。
 */

import type { SassTraceFile } from './sassParser';

export const SAMPLE_SASS_TRACE: SassTraceFile = {
  sample: true,
  meta: {
    tool: 'NVBit（示例）',
    gpu: '示例 GPU（教学示意）',
    capturedAt: '教学示例数据，非真实采集',
  },
  kernel: {
    name: 'gemm_tiled_kernel',
    operator: 'GEMM',
    grid: [16, 16, 1],
    block: [128, 1, 1],
    sm: 0,
  },
  warps: [
    {
      warp: 0,
      instructions: [
        { pc: '0x0040', opcode: 'IMAD.MOV.U32', operands: 'R1, RZ, RZ, c[0x0][0x28]' },
        { pc: '0x0050', opcode: 'IMAD', operands: 'R4, R2, 0x80, R6' },
        { pc: '0x0060', opcode: 'SHF.L.U32', operands: 'R8, R4, 0x5, R8' },
        { pc: '0x0070', opcode: 'ISETP.NE.AND', operands: 'P0, PT, R2, RZ, PT' },

        { pc: '0x0080', opcode: 'LDGSTS.E.BYPASS.128', operands: '[R20], [R10.64], 0x0' },
        { pc: '0x0090', opcode: 'LDGSTS.E.BYPASS.128', operands: '[R20+0x400], [R12.64], 0x1' },
        { pc: '0x00a0', opcode: 'BAR.SYNC', operands: '0x0' },

        { pc: '0x00b0', opcode: 'LDS.128', operands: 'R40, [R20]' },
        { pc: '0x00c0', opcode: 'LDS.128', operands: 'R44, [R20+0x100]' },
        { pc: '0x00d0', opcode: 'HMMA.16816.F32', operands: 'R60, R40, R44, R60' },
        { pc: '0x00e0', opcode: 'LDS.128', operands: 'R48, [R20+0x200]' },
        { pc: '0x00f0', opcode: 'HMMA.16816.F32', operands: 'R60, R48, R44, R60' },
        { pc: '0x0100', opcode: 'BRA', operands: '0x00b0' },

        { pc: '0x0110', opcode: 'STG.E.128', operands: '[R30.64], R60' },
        { pc: '0x0120', opcode: 'EXIT', operands: '' },
      ],
    },
    {
      warp: 1,
      instructions: [
        { pc: '0x0040', opcode: 'IMAD.MOV.U32', operands: 'R1, RZ, RZ, c[0x0][0x28]' },
        { pc: '0x0050', opcode: 'IMAD', operands: 'R4, R2, 0x80, R6' },
        { pc: '0x0070', opcode: 'ISETP.NE.AND', operands: 'P0, PT, R2, RZ, PT' },

        { pc: '0x0080', opcode: 'LDGSTS.E.BYPASS.128', operands: '[R22], [R14.64], 0x2' },
        { pc: '0x00a0', opcode: 'BAR.SYNC', operands: '0x0' },

        { pc: '0x00b0', opcode: 'LDS.128', operands: 'R50, [R22]' },
        { pc: '0x00d0', opcode: 'HMMA.16816.F32', operands: 'R70, R50, R54, R70' },
        { pc: '0x00f0', opcode: 'FFMA', operands: 'R70, R70, R2, R70' },
        { pc: '0x0100', opcode: 'BRA', operands: '0x00b0' },

        { pc: '0x0110', opcode: 'STG.E.128', operands: '[R32.64], R70' },
        { pc: '0x0120', opcode: 'EXIT', operands: '' },
      ],
    },
  ],
};
