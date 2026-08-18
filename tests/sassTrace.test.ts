/**
 * SASS Trace Adapter 单元测试（实施手册 §26，V0.8）。
 *
 * V0.8 架构验收点（手册 §9 的延续）：指令级数据源同样只替换 Source，
 * Playback 与 UI 零改动——Adapter 的输出是合法 TVIRTrace。
 * TVIR 12 种事件类型保持唯一词汇表（手册 §31）：SASS 指令通过显式
 * 映射表投影到既有事件类型，不新增事件类型。
 * 定位声明（手册 §26 硬性要求）：Educational Simulation ≠ Architecture Simulation。
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  parseSassTrace,
  classifyOpcode,
  inferSassTraceHardware,
  SAMPLE_SASS_TRACE,
  SASS_CATEGORY_LABELS,
} from '../src/core/sasstrace';
import { validateTVIRTrace } from '../src/core/tvir/validation';
import { projectSassInstructions, isSassInstructionEvent } from '../src/core/tvir/projection';
import { PlaybackEngine } from '../src/core/playback';
import { TVIR_EVENT_TYPES } from '../src/core/tvir/types';

describe('classifyOpcode（SASS 指令教学分类）', () => {
  it('全局内存指令：LDG/STG → global-memory', () => {
    expect(classifyOpcode('LDG.E.128')).toBe('global-memory');
    expect(classifyOpcode('STG.E.128')).toBe('global-memory');
  });

  it('共享内存指令：LDS/STS/LDSM → shared-memory', () => {
    expect(classifyOpcode('LDS.128')).toBe('shared-memory');
    expect(classifyOpcode('STS.128')).toBe('shared-memory');
    expect(classifyOpcode('LDSM.16.M88.4')).toBe('shared-memory');
  });

  it('LDGSTS（cp.async）优先匹配 async-copy 而非 global-memory', () => {
    expect(classifyOpcode('LDGSTS.E.BYPASS.128')).toBe('async-copy');
  });

  it('Tensor Core 指令：HMMA/QMMA 等 → tensor-core', () => {
    expect(classifyOpcode('HMMA.16816.F32')).toBe('tensor-core');
    expect(classifyOpcode('QMMA.16864.F32')).toBe('tensor-core');
    expect(classifyOpcode('DMMA.16816.F64')).toBe('tensor-core');
  });

  it('CUDA Core 浮点指令：FFMA/FADD → cuda-core', () => {
    expect(classifyOpcode('FFMA')).toBe('cuda-core');
    expect(classifyOpcode('FADD')).toBe('cuda-core');
    expect(classifyOpcode('HFMA2')).toBe('cuda-core');
  });

  it('BAR → sync，BRA/EXIT/RET → control', () => {
    expect(classifyOpcode('BAR.SYNC')).toBe('sync');
    expect(classifyOpcode('BRA')).toBe('control');
    expect(classifyOpcode('EXIT')).toBe('control');
    expect(classifyOpcode('RET')).toBe('control');
  });

  it('未登记的指令归入 address-calc（教学映射，不臆测）', () => {
    expect(classifyOpcode('IMAD')).toBe('address-calc');
    expect(classifyOpcode('MOV')).toBe('address-calc');
    expect(classifyOpcode('ISETP.NE.AND')).toBe('address-calc');
    expect(classifyOpcode('SHF.L.U32')).toBe('address-calc');
    expect(classifyOpcode('UNKNOWN.FOO')).toBe('address-calc');
  });

  it('每个类别都有中文标签', () => {
    for (const label of Object.values(SASS_CATEGORY_LABELS)) {
      expect(label.length).toBeGreaterThan(0);
    }
  });
});

describe('parseSassTrace（TVIR Adapter）', () => {
  it('内置示例 trace 解析成功，输出通过 TVIR schema 校验', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const validation = validateTVIRTrace(result.trace);
    expect(validation.valid, validation.errors.join('; ')).toBe(true);
  });

  it('输出只使用 TVIR 12 种既有事件类型（不新增词汇表）', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    for (const event of result.trace.events) {
      expect(TVIR_EVENT_TYPES).toContain(event.type);
    }
  });

  it('指令到事件类型的映射正确：LDG/STG/LDS/STS/LDGSTS/HMMA/FFMA/BAR', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const events = result.trace.events;

    // LDGSTS → MEMORY_MOVE（HBM → SHARED_MEMORY，绕过寄存器）
    const ldgsts = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'LDGSTS.E.BYPASS.128');
    expect(ldgsts?.type).toBe('MEMORY_MOVE');
    expect(ldgsts?.source).toBe('HBM');
    expect(ldgsts?.destination).toBe('SHARED_MEMORY');

    // LDS → MEMORY_LOAD（SHARED_MEMORY → REGISTER）
    const lds = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'LDS.128');
    expect(lds?.type).toBe('MEMORY_LOAD');
    expect(lds?.source).toBe('SHARED_MEMORY');
    expect(lds?.destination).toBe('REGISTER');

    // HMMA → MMA
    const hmma = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'HMMA.16816.F32');
    expect(hmma?.type).toBe('MMA');

    // FFMA → ACCUMULATE
    const ffma = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'FFMA');
    expect(ffma?.type).toBe('ACCUMULATE');

    // BAR.SYNC → SYNC
    const bar = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'BAR.SYNC');
    expect(bar?.type).toBe('SYNC');

    // STG → MEMORY_STORE（REGISTER → HBM）
    const stg = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'STG.E.128');
    expect(stg?.type).toBe('MEMORY_STORE');
    expect(stg?.source).toBe('REGISTER');
    expect(stg?.destination).toBe('HBM');

    // IMAD（地址计算）/ BRA / EXIT → WARP_SCHEDULE（文档化的教学映射）
    const imad = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'IMAD');
    expect(imad?.type).toBe('WARP_SCHEDULE');
    const bra = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'BRA');
    expect(bra?.type).toBe('WARP_SCHEDULE');
    const exit = events.find((e) => (e.metadata as { sass?: { opcode: string } }).sass?.opcode === 'EXIT');
    expect(exit?.type).toBe('WARP_SCHEDULE');
  });

  it('每条指令事件都携带教学文案与 sass 元数据', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const instructionEvents = result.trace.events.filter(
      (e) => e.type !== 'KERNEL_LAUNCH',
    );
    const totalInstructions = SAMPLE_SASS_TRACE.warps.reduce(
      (sum, warp) => sum + warp.instructions.length,
      0,
    );
    expect(instructionEvents.length).toBe(totalInstructions);
    for (const event of instructionEvents) {
      expect(event.what.length).toBeGreaterThan(0);
      expect(event.why.length).toBeGreaterThan(0);
      const meta = event.metadata as { sass: { pc: string; opcode: string; category: string } };
      expect(meta.sass.pc).toBeTruthy();
      expect(meta.sass.opcode).toBeTruthy();
      expect(meta.sass.category).toBeTruthy();
    }
  });

  it('定位声明：描述必须标明 Educational Simulation，且不得声称 cycle-accurate 或 Measured', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    expect(result.trace.description).toContain('Educational Simulation');
    expect(result.trace.description).not.toContain('cycle-accurate 仿真');
    // 示例数据标注
    expect(result.trace.description).toContain('示例数据');
    expect(result.trace.isSample).toBe(true);
    // KERNEL_LAUNCH 的 why 也要声明定位
    const launch = result.trace.events[0];
    expect(launch?.why).toContain('Educational Simulation');
  });

  it('非示例 trace：不带 isSample，描述不含示例字样', () => {
    const realFile = { ...SAMPLE_SASS_TRACE, sample: false };
    const result = parseSassTrace(realFile);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.trace.isSample).toBeUndefined();
    expect(result.trace.description).not.toContain('示例数据');
    expect(result.trace.description).toContain('Educational Simulation');
  });

  it('事件 step 全局连续递增，事件总数 = 1 + 指令数', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const totalInstructions = SAMPLE_SASS_TRACE.warps.reduce(
      (sum, warp) => sum + warp.instructions.length,
      0,
    );
    expect(result.trace.events.length).toBe(totalInstructions + 1);
    result.trace.events.forEach((event, index) => {
      expect(event.step).toBe(index);
    });
    expect(result.trace.events[0]?.type).toBe('KERNEL_LAUNCH');
  });

  it('拒绝非法输入并给出中文错误说明', () => {
    expect(parseSassTrace(null).ok).toBe(false);
    expect(parseSassTrace({}).ok).toBe(false);
    expect(parseSassTrace({ kernel: { name: 'k' } }).ok).toBe(false);

    const badKernelName = { kernel: { name: '' }, warps: [{ warp: 0, instructions: [{ pc: '0x0', opcode: 'MOV' }] }] };
    const r1 = parseSassTrace(badKernelName);
    expect(r1.ok).toBe(false);
    if (!r1.ok) expect(r1.error).toContain('kernel.name');

    const emptyWarps = { kernel: { name: 'k' }, warps: [] };
    const r2 = parseSassTrace(emptyWarps);
    expect(r2.ok).toBe(false);
    if (!r2.ok) expect(r2.error).toContain('warps');

    const badGrid = { kernel: { name: 'k', grid: [1, 2] }, warps: [{ warp: 0, instructions: [{ pc: '0x0', opcode: 'MOV' }] }] };
    const r3 = parseSassTrace(badGrid);
    expect(r3.ok).toBe(false);
    if (!r3.ok) expect(r3.error).toContain('grid');

    const badInstruction = { kernel: { name: 'k' }, warps: [{ warp: 0, instructions: [{ pc: '', opcode: 'MOV' }] }] };
    const r4 = parseSassTrace(badInstruction);
    expect(r4.ok).toBe(false);
    if (!r4.ok) expect(r4.error).toContain('pc');

    const emptyInstructions = { kernel: { name: 'k' }, warps: [{ warp: 0, instructions: [] }] };
    const r5 = parseSassTrace(emptyInstructions);
    expect(r5.ok).toBe(false);
    if (!r5.ok) expect(r5.error).toContain('instructions');
  });
});

describe('架构验收：Adapter 输出可直接驱动 Playback（UI 零改动）', () => {
  it('PlaybackEngine 无需任何修改即可消费 sass-trace', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const engine = new PlaybackEngine(result.trace);
    expect(engine.totalEvents).toBe(result.trace.events.length);
    expect(engine.currentEvent?.type).toBe('KERNEL_LAUNCH');
    engine.next();
    expect(engine.getState().currentIndex).toBe(1);
    engine.seek(engine.totalEvents - 1);
    expect(engine.isLast).toBe(true);
    engine.dispose();
  });

  it('isSassInstructionEvent 正确识别指令事件', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    expect(isSassInstructionEvent(result.trace.events[0] ?? null)).toBe(false); // KERNEL_LAUNCH
    expect(isSassInstructionEvent(result.trace.events[1] ?? null)).toBe(true);
    expect(isSassInstructionEvent(null)).toBe(false);
  });
});

describe('projectSassInstructions（指令视图投影）', () => {
  it('从 sass-trace 提取指令行，数量与源数据一致', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const rows = projectSassInstructions(result.trace.events);
    const totalInstructions = SAMPLE_SASS_TRACE.warps.reduce(
      (sum, warp) => sum + warp.instructions.length,
      0,
    );
    expect(rows.length).toBe(totalInstructions);
    // 每行携带 pc/opcode/category/warp 与事件索引
    for (const row of rows) {
      expect(row.pc).toBeTruthy();
      expect(row.opcode).toBeTruthy();
      expect(row.category).toBeTruthy();
      expect(row.eventIndex).toBeGreaterThan(0); // KERNEL_LAUNCH 不是指令行
    }
    // warp 分组保持出现顺序
    expect(rows[0]?.warp).toBe(0);
    const lastWarp = SAMPLE_SASS_TRACE.warps[SAMPLE_SASS_TRACE.warps.length - 1]?.warp ?? 0;
    expect(rows[rows.length - 1]?.warp).toBe(lastWarp);
  });

  it('仿真 trace（无 sass 元数据）返回空列表', () => {
    expect(
      projectSassInstructions([
        { id: 'a', step: 0, type: 'KERNEL_LAUNCH', title: 't', what: 'w', why: 'y' },
        { id: 'b', step: 1, type: 'MMA', title: 't', what: 'w', why: 'y' },
      ]),
    ).toEqual([]);
  });
});

describe('samples/sass-gemm.sample.json（格式参考示例）', () => {
  it('示例 JSON 文件可被 Adapter 解析并通过校验', () => {
    const raw = readFileSync(resolve(__dirname, '../samples/sass-gemm.sample.json'), 'utf-8');
    const parsed = JSON.parse(raw);
    const result = parseSassTrace(parsed);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(validateTVIRTrace(result.trace).valid).toBe(true);
    expect(result.trace.isSample).toBe(true);
  });
});

describe('inferSassTraceHardware', () => {
  it('从事件推导 SM 数量与 warps/Block', () => {
    const result = parseSassTrace(SAMPLE_SASS_TRACE);
    if (!result.ok) throw new Error(result.error);
    const hardware = inferSassTraceHardware(result.trace);
    expect(hardware.numSM).toBe(1); // sm=0，教学视图聚焦单 SM
    expect(hardware.warpsPerBlock).toBe(4); // 128 线程/Block ÷ 32
  });

  it('无 KERNEL_LAUNCH 事件时回退默认值', () => {
    const hardware = inferSassTraceHardware({
      description: 'empty',
      events: [],
    });
    expect(hardware.numSM).toBe(1);
    expect(hardware.warpsPerBlock).toBe(4);
  });
});
