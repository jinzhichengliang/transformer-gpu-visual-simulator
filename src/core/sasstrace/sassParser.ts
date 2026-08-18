/**
 * SASS Trace Adapter（V0.8，实施手册 §26）。
 *
 * 架构路径：CUDA → NVBit → SASS Trace → Accel-Sim → TVIR Adapter（本模块）→ Visualizer
 *
 * 定位声明（手册 §26 的硬性要求）：
 *   Educational Simulation ≠ Architecture Simulation。
 *   本 Adapter 只做"指令流顺序与数据通路"的教学呈现（Educational Simulation），
 *   不重新实现 Accel-Sim 的 cycle-accurate 微架构仿真（Architecture Simulation）。
 *   UI 中所有来自本模块的视图必须携带该声明（见 InstructionView）。
 *
 * 架构约束：
 *   - TVIR 12 种事件类型保持唯一词汇表（V0.1-V0.7 的零改动纪律延续到 V0.8）：
 *     SASS 指令通过显式映射表投影到既有事件类型（映射表见 CONCEPTS.md V0.8 章节）。
 *   - 本模块不 import React、不 import Simulation 业务逻辑（仅复用事件编号基础设施）。
 */

import type { TVIRTrace } from '../tvir/types';
import { createEventBuilder } from '../simulation/eventBuilder';

/** SASS 指令类别（Adapter 的教学分类） */
export type SassCategory =
  | 'global-memory' // LDG/STG：全局显存 ↔ 寄存器
  | 'shared-memory' // LDS/LDSM/STS：共享内存 ↔ 寄存器
  | 'async-copy' // LDGSTS（cp.async）：全局 → 共享（绕过寄存器，Ampere+）
  | 'tensor-core' // HMMA/QMMA/...：Tensor Core 矩阵乘加
  | 'cuda-core' // FFMA/FADD/...：CUDA Core 标量/向量浮点运算
  | 'sync' // BAR：Block 内 Warp 同步
  | 'address-calc' // MOV/IMAD/SHF/ISETP/...：地址计算与 Warp 内务
  | 'control'; // BRA/EXIT/RET：控制流

/** 单条 SASS 指令记录（NVBit/cuobjdump 导出格式的教学子集） */
export interface SassInstructionRecord {
  /** 指令地址（如 "0x0040"） */
  pc: string;
  /** 操作码（如 "LDG.E.128"、"HMMA.16816.F32"） */
  opcode: string;
  /** 操作数文本（可选） */
  operands?: string | undefined;
}

/** 一个 Warp 的指令流 */
export interface SassWarpRecord {
  /** Warp 编号（Block 内） */
  warp: number;
  instructions: SassInstructionRecord[];
}

/** SASS trace 的 kernel 元信息 */
export interface SassKernelMeta {
  /** kernel 名（如 "gemm_tiled_kernel"） */
  name: string;
  /** 可选：对应算子名 */
  operator?: string | undefined;
  /** Grid 维度 [x, y, z]（可选） */
  grid?: [number, number, number] | undefined;
  /** Block 维度 [x, y, z]（可选） */
  block?: [number, number, number] | undefined;
  /** 被采样的 Warp 所在 SM 编号（可选） */
  sm?: number | undefined;
}

/** SASS trace 文件顶层结构 */
export interface SassTraceFile {
  meta?:
    | {
        tool?: string | undefined;
        gpu?: string | undefined;
        capturedAt?: string | undefined;
      }
    | undefined;
  /** 是否为示例数据（指令流为教学示意编排，非真实采集） */
  sample?: boolean | undefined;
  kernel: SassKernelMeta;
  warps: SassWarpRecord[];
}

export type SassParseResult =
  | { ok: true; trace: TVIRTrace }
  | { ok: false; error: string };

/**
 * 指令类别元信息（标题、教学说明用的中文名）。
 * 供 Adapter 与 InstructionView 共用。
 */
export const SASS_CATEGORY_LABELS: Record<SassCategory, string> = {
  'global-memory': '全局内存',
  'shared-memory': '共享内存',
  'async-copy': '异步拷贝',
  'tensor-core': 'Tensor Core',
  'cuda-core': 'CUDA Core',
  sync: '同步',
  'address-calc': '地址计算',
  control: '控制流',
};

/** 操作码 → 类别规则表（按基础操作码精确匹配，基础码 = 第一个 '.' 之前的部分） */
const OPCODE_RULES: ReadonlyArray<{ base: string; category: SassCategory }> = [
  { base: 'LDGSTS', category: 'async-copy' },
  { base: 'LDG', category: 'global-memory' },
  { base: 'STG', category: 'global-memory' },
  { base: 'LDSM', category: 'shared-memory' },
  { base: 'LDS', category: 'shared-memory' },
  { base: 'STS', category: 'shared-memory' },
  { base: 'HMMA', category: 'tensor-core' },
  { base: 'QMMA', category: 'tensor-core' },
  { base: 'IMMA', category: 'tensor-core' },
  { base: 'BMMA', category: 'tensor-core' },
  { base: 'DMMA', category: 'tensor-core' },
  { base: 'FFMA', category: 'cuda-core' },
  { base: 'FADD', category: 'cuda-core' },
  { base: 'FMUL', category: 'cuda-core' },
  { base: 'FSUB', category: 'cuda-core' },
  { base: 'HFMA2', category: 'cuda-core' },
  { base: 'HFMA', category: 'cuda-core' },
  { base: 'DFMA', category: 'cuda-core' },
  { base: 'BAR', category: 'sync' },
  { base: 'BRA', category: 'control' },
  { base: 'JMP', category: 'control' },
  { base: 'CAL', category: 'control' },
  { base: 'RET', category: 'control' },
  { base: 'EXIT', category: 'control' },
];

/** 取基础操作码（第一个 '.' 之前的部分，大写；空串兜底） */
function baseOpcode(opcode: string): string {
  const first = opcode.split('.')[0];
  return (first ?? opcode).toUpperCase();
}

/** 按基础操作码对 SASS 指令分类；未登记的指令归入地址计算（教学映射，见 CONCEPTS.md） */
export function classifyOpcode(opcode: string): SassCategory {
  const base = baseOpcode(opcode);
  for (const rule of OPCODE_RULES) {
    if (base === rule.base) return rule.category;
  }
  return 'address-calc';
}

/** 判断基础操作码是否为 load 方向（LDG/LDS/LDSM/LDGSTS） */
function isLoadOpcode(opcode: string): boolean {
  return baseOpcode(opcode).startsWith('LD');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** 校验并解析未知输入为 SassTraceFile；失败时返回中文错误说明 */
function validateSassFile(input: unknown): { file: SassTraceFile } | { error: string } {
  if (!isRecord(input)) {
    return { error: 'JSON 顶层必须是对象（参考格式见 samples/sass-gemm.sample.json）' };
  }

  const kernel = input.kernel;
  if (!isRecord(kernel)) {
    return { error: '缺少 kernel 对象：SASS trace 必须声明所属 kernel' };
  }
  if (typeof kernel.name !== 'string' || kernel.name.length === 0) {
    return { error: 'kernel.name 必须是非空字符串（kernel 名）' };
  }
  if (kernel.operator !== undefined && (typeof kernel.operator !== 'string' || kernel.operator.length === 0)) {
    return { error: 'kernel.operator 若提供必须是非空字符串' };
  }
  for (const dimName of ['grid', 'block'] as const) {
    const dim = kernel[dimName];
    if (dim !== undefined) {
      if (!Array.isArray(dim) || dim.length !== 3 || dim.some((d) => typeof d !== 'number' || d <= 0)) {
        return { error: `kernel.${dimName} 必须是三个正数的数组 [x, y, z]` };
      }
    }
  }
  if (kernel.sm !== undefined && (typeof kernel.sm !== 'number' || kernel.sm < 0)) {
    return { error: 'kernel.sm 若提供必须是非负整数（SM 编号）' };
  }

  const warps = input.warps;
  if (!Array.isArray(warps) || warps.length === 0) {
    return { error: '缺少 warps 数组：SASS trace 必须包含至少一个 Warp 的指令流' };
  }
  for (let w = 0; w < warps.length; w++) {
    const warp = warps[w];
    if (!isRecord(warp)) {
      return { error: `warps[${w}] 必须是对象` };
    }
    if (typeof warp.warp !== 'number' || !Number.isFinite(warp.warp) || warp.warp < 0) {
      return { error: `warps[${w}].warp 必须是非负数字（Warp 编号）` };
    }
    const instructions = warp.instructions;
    if (!Array.isArray(instructions) || instructions.length === 0) {
      return { error: `warps[${w}].instructions 必须是非空数组` };
    }
    for (let i = 0; i < instructions.length; i++) {
      const inst = instructions[i];
      if (!isRecord(inst)) {
        return { error: `warps[${w}].instructions[${i}] 必须是对象` };
      }
      if (typeof inst.pc !== 'string' || inst.pc.length === 0) {
        return { error: `warps[${w}].instructions[${i}].pc 必须是非空字符串（指令地址）` };
      }
      if (typeof inst.opcode !== 'string' || inst.opcode.length === 0) {
        return { error: `warps[${w}].instructions[${i}].opcode 必须是非空字符串（操作码）` };
      }
      if (inst.operands !== undefined && typeof inst.operands !== 'string') {
        return { error: `warps[${w}].instructions[${i}].operands 若提供必须是字符串` };
      }
    }
  }

  if (input.meta !== undefined && !isRecord(input.meta)) {
    return { error: 'meta 若提供必须是对象' };
  }

  return {
    file: {
      ...(isRecord(input.meta) ? { meta: input.meta as SassTraceFile['meta'] } : {}),
      ...(typeof input.sample === 'boolean' ? { sample: input.sample } : {}),
      kernel: kernel as unknown as SassKernelMeta,
      warps: warps as SassWarpRecord[],
    },
  };
}

/** 每条 SASS 指令的 what/why 教学文案（按类别） */
function sassTeaching(
  category: SassCategory,
  opcode: string,
  operands: string | undefined,
  warp: number,
): { title: string; what: string; why: string } {
  const operandText = operands && operands.length > 0 ? `（${operands}）` : '';

  switch (category) {
    case 'global-memory':
      return isLoadOpcode(opcode)
        ? {
            title: `SASS ${opcode}：全局内存加载`,
            what: `Warp ${warp} 执行 ${opcode}${operandText}：从全局显存读取数据，沿 HBM → L2 → L1 → Register 通路进入寄存器。`,
            why: `LDG 是 SASS 层面的全局内存加载指令，是 V0.1 中"HBM → Register"数据通路在真实指令级的具体形态；访问会经过 L2/L1 缓存层级，未命中时延迟显著更高。`,
          }
        : {
            title: `SASS ${opcode}：全局内存写回`,
            what: `Warp ${warp} 执行 ${opcode}${operandText}：把寄存器中的结果沿 Register → L1/L2 → HBM 写回全局显存。`,
            why: `STG 是计算完成后的结果落地指令。写回同样经过缓存层级——这也是"为什么不是所有数据都直连 HBM"在指令级的体现。`,
          };
    case 'shared-memory':
      return isLoadOpcode(opcode)
        ? {
            title: `SASS ${opcode}：共享内存读取`,
            what: `Warp ${warp} 执行 ${opcode}${operandText}：从 Shared Memory 读取 fragment 到寄存器。`,
            why: `LDS 的延迟远低于 LDG：Shared Memory 是 SM 片上低延迟存储，这正是 V0.1 "为什么要 tiling + Shared Memory" 的指令级证据——tile 预加载后被反复 LDS 复用。`,
          }
        : {
            title: `SASS ${opcode}：写入共享内存`,
            what: `Warp ${warp} 执行 ${opcode}${operandText}：把寄存器数据写入 Shared Memory。`,
            why: `STS 把从全局内存搬来的 tile 放进片上 Shared Memory，供 Block 内所有 Warp 复用——这是 tiling 算法在指令级的落点。`,
          };
    case 'async-copy':
      return {
        title: `SASS ${opcode}：异步拷贝（cp.async）`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：数据从全局显存直接拷贝到 Shared Memory，绕过寄存器。`,
        why: `LDGSTS（对应 CUDA 的 cp.async，Ampere 起支持）让数据搬运不占用寄存器、并与计算重叠，是现代 GEMM kernel 隐藏访存延迟的关键手段。`,
      };
    case 'tensor-core':
      return {
        title: `SASS ${opcode}：Tensor Core 矩阵乘加`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：Tensor Core 执行 D = A × B + C，A/B fragment 必须已在寄存器中。`,
        why: `HMMA 是 Tensor Core 的 SASS 指令（如 16816 表示 m16n8k16 形状）。操作数只能来自寄存器（CONCEPTS.md 规则 10）——这就是数据必须先 HBM→Shared→Register 的原因。`,
      };
    case 'cuda-core':
      return {
        title: `SASS ${opcode}：CUDA Core 浮点运算`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：CUDA Core 执行标量/向量浮点乘加。`,
        why: `FFMA/FADD 等由 CUDA Core（而非 Tensor Core）执行。Softmax、SiLU 这类逐元素算子没有矩阵结构可喂给 Tensor Core，只能走 CUDA Core——这是 V0.4 "算子决定执行单元"的指令级体现。`,
      };
    case 'sync':
      return {
        title: `SASS ${opcode}：Block 内同步`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：等待 Block 内所有 Warp 到达该同步点。`,
        why: `BAR.SYNC 保证"Shared Memory 写完"对"后续读取"可见——没有它，Warp 可能读到别的 Warp 还没写好的 tile。对应 TVIR 的 SYNC 事件。`,
      };
    case 'control':
      return {
        title: `SASS ${opcode}：控制流`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：改变指令执行流（跳转/返回/退出）。`,
        why: `BRA 实现 K 维循环的回边，EXIT 让线程结束。控制流指令不触碰数据通路，但决定了上面那些访存/计算指令被执行的顺序与次数。`,
      };
    case 'address-calc':
    default:
      return {
        title: `SASS ${opcode}：地址计算/指令编排`,
        what: `Warp ${warp} 执行 ${opcode}${operandText}：计算访存地址、比较或搬运标量（CUDA Core 执行）。`,
        why: `IMAD/MOV/SHF/ISETP 等负责算出"从哪个地址搬数据"。真实 kernel 中这类指令占比很高——这正是 GPU 用大量 CUDA Core 做地址计算、把数据通路留给 LDG/HMMA 的原因。`,
      };
  }
}

/**
 * 把 SASS trace 解析为 TVIRTrace（TVIR Adapter 的主体）。
 *
 * 映射表（TVIR 12 类型保持唯一词汇表，见 CONCEPTS.md V0.8）：
 *   LDG            → MEMORY_LOAD（HBM → REGISTER）
 *   STG            → MEMORY_STORE（REGISTER → HBM）
 *   LDS/LDSM       → MEMORY_LOAD（SHARED_MEMORY → REGISTER）
 *   STS            → MEMORY_STORE（REGISTER → SHARED_MEMORY）
 *   LDGSTS         → MEMORY_MOVE（HBM → SHARED_MEMORY）
 *   HMMA/...       → MMA
 *   FFMA/FADD/...  → ACCUMULATE
 *   BAR            → SYNC
 *   其余（地址计算/控制流）→ WARP_SCHEDULE（文档化的教学映射）
 */
export function parseSassTrace(input: unknown): SassParseResult {
  const validated = validateSassFile(input);
  if ('error' in validated) {
    return { ok: false, error: validated.error };
  }
  const { file } = validated;

  const isSample = file.sample === true;
  const builder = createEventBuilder();
  const kernel = file.kernel;
  const grid = kernel.grid ?? [1, 1, 1];
  const block = kernel.block ?? [128, 1, 1];
  const threadsPerBlock = block[0] * block[1] * block[2];
  const totalBlocks = grid[0] * grid[1] * grid[2];
  const sm = kernel.sm ?? 0;

  const totalInstructions = file.warps.reduce((sum, warp) => sum + warp.instructions.length, 0);

  const dataSource = isSample
    ? '内置示例 SASS trace（指令流为教学示意编排，非真实采集）'
    : 'NVBit 采集的 SASS 指令流';

  builder.push({
    type: 'KERNEL_LAUNCH',
    title: `Kernel: ${kernel.name}（SASS 指令级视图）`,
    what: `kernel「${kernel.name}」在 SM ${sm} 上开始执行：grid ${grid[0]}×${grid[1]}×${grid[2]}（共 ${totalBlocks} 个 Block），block ${block[0]}×${block[1]}×${block[2]}（${threadsPerBlock} 线程/Block）。本 trace 展示其中 ${file.warps.length} 个 Warp 的 ${totalInstructions} 条 SASS 指令。`,
    why: `本事件由 ${dataSource} 经 TVIR Adapter 转换而来（CUDA → NVBit → SASS → Adapter → TVIR）。注意定位：这是 Educational Simulation——呈现指令顺序与数据通路，不是 Accel-Sim 的 cycle-accurate 架构仿真。`,
    ...(kernel.operator !== undefined ? { operator: kernel.operator } : {}),
    kernel: kernel.name,
    sm,
    metadata: {
      provenance: 'sass-trace',
      sassKernel: {
        name: kernel.name,
        gridX: grid[0],
        gridY: grid[1],
        gridZ: grid[2],
        blockX: block[0],
        blockY: block[1],
        blockZ: block[2],
        totalBlocks,
        threadsPerBlock,
        sm,
        totalInstructions,
      },
    },
  });

  let instructionIndex = 0;
  for (const warpRecord of file.warps) {
    for (const inst of warpRecord.instructions) {
      const category = classifyOpcode(inst.opcode);
      const teaching = sassTeaching(category, inst.opcode, inst.operands, warpRecord.warp);
      const base = baseOpcode(inst.opcode);

      let type:
        | 'MEMORY_LOAD'
        | 'MEMORY_STORE'
        | 'MEMORY_MOVE'
        | 'MMA'
        | 'ACCUMULATE'
        | 'SYNC'
        | 'WARP_SCHEDULE';
      let source: 'HBM' | 'SHARED_MEMORY' | 'REGISTER' | undefined;
      let destination: 'HBM' | 'SHARED_MEMORY' | 'REGISTER' | undefined;

      switch (category) {
        case 'global-memory':
          if (isLoadOpcode(inst.opcode)) {
            type = 'MEMORY_LOAD';
            source = 'HBM';
            destination = 'REGISTER';
          } else {
            type = 'MEMORY_STORE';
            source = 'REGISTER';
            destination = 'HBM';
          }
          break;
        case 'shared-memory':
          if (isLoadOpcode(inst.opcode)) {
            type = 'MEMORY_LOAD';
            source = 'SHARED_MEMORY';
            destination = 'REGISTER';
          } else {
            type = 'MEMORY_STORE';
            source = 'REGISTER';
            destination = 'SHARED_MEMORY';
          }
          break;
        case 'async-copy':
          type = 'MEMORY_MOVE';
          source = 'HBM';
          destination = 'SHARED_MEMORY';
          break;
        case 'tensor-core':
          type = 'MMA';
          break;
        case 'cuda-core':
          type = 'ACCUMULATE';
          break;
        case 'sync':
          type = 'SYNC';
          break;
        case 'address-calc':
        case 'control':
        default:
          type = 'WARP_SCHEDULE';
          break;
      }

      builder.push({
        type,
        title: teaching.title,
        what: teaching.what,
        why: teaching.why,
        ...(kernel.operator !== undefined ? { operator: kernel.operator } : {}),
        kernel: kernel.name,
        warp: warpRecord.warp,
        sm,
        ...(source !== undefined ? { source } : {}),
        ...(destination !== undefined ? { destination } : {}),
        metadata: {
          provenance: 'sass-trace',
          sass: {
            pc: inst.pc,
            opcode: inst.opcode,
            ...(inst.operands !== undefined ? { operands: inst.operands } : {}),
            category,
            baseOpcode: base,
            instructionIndex,
            totalInstructions,
            warp: warpRecord.warp,
          },
        },
      });
      instructionIndex++;
    }
  }

  const gpuLabel = file.meta?.gpu ?? '未知 GPU';
  const toolLabel = file.meta?.tool ?? 'NVBit';

  return {
    ok: true,
    trace: {
      description: `SASS Trace: ${kernel.name} · ${totalInstructions} 条指令（来源：${toolLabel} · ${gpuLabel}）${isSample ? ' · 示例数据（教学示意编排）' : ''} · Educational Simulation（非 cycle-accurate）`,
      events: builder.events,
      provenance: 'real-trace',
      ...(isSample ? { isSample: true } : {}),
    },
  };
}

/**
 * 从 SASS trace 推导 GPU View 需要的硬件参数。
 * - numSM：采样 SM 编号 + 1（教学视图聚焦单 SM 内部）
 * - warpsPerBlock：Block 线程数 / 32（回退 4）
 */
export function inferSassTraceHardware(trace: TVIRTrace): {
  numSM: number;
  warpsPerBlock: number;
} {
  for (const event of trace.events) {
    if (event.type !== 'KERNEL_LAUNCH') continue;
    const info = (event.metadata as { sassKernel?: { sm?: number; threadsPerBlock?: number } } | undefined)
      ?.sassKernel;
    if (info) {
      const numSM = Math.max(1, (info.sm ?? 0) + 1);
      const warpsPerBlock =
        info.threadsPerBlock !== undefined && info.threadsPerBlock > 0
          ? Math.max(1, Math.round(info.threadsPerBlock / 32))
          : 4;
      return { numSM, warpsPerBlock };
    }
  }
  return { numSM: 1, warpsPerBlock: 4 };
}
