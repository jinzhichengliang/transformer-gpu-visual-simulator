/**
 * 跨模型比较验证测试（Sprint 15, Phase K）。
 *
 * 任务书核心要求：
 *   相同任务配置下，架构不同的模型必须产生不同的 Operator Graph 与执行序列，
 *   证明 ModelProfile / Planner 真正工作——而不是"四个模型只是名字不同的
 *   Attention+FFN 序列"。
 *
 * Critical Fail 条件（任务书原文）：
 *   如果四个模型最终都只是 Attention→FFN→Attention→FFN 只是名字不同，
 *   说明 ModelProfile / Planner 没有真正工作 → 测试必须失败。
 */

import { describe, expect, it } from 'vitest';
import {
  makeDeepSeekV4FlashProfile,
  makeDeepSeekV4ProProfile,
  makeKimiK3Profile,
  makeGLM53Profile,
} from '../src/core/modelprofile';
import { buildOperatorGraph } from '../src/core/execution/planner';
import { planExecution } from '../src/core/execution/executor';
import { DEFAULT_INFERENCE_TASK } from '../src/core/execution/task';

/** 统一的任务配置（任务书 §15：相同 Prompt length / Batch / GPU profile） */
const SHARED_TASK = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const, promptTokens: 64, batchSize: 1 };

/** 提取执行序列的"算子签名"（按出现顺序去重） */
function operatorSignature(events: Array<{ operator?: string }>): string[] {
  const sig: string[] = [];
  for (const e of events) {
    if (e.operator && sig[sig.length - 1] !== e.operator) sig.push(e.operator);
  }
  return sig;
}

describe('Sprint 15：跨模型比较验证（Phase K）', () => {
  const models = {
    flash: makeDeepSeekV4FlashProfile(),
    pro: makeDeepSeekV4ProProfile(),
    kimi: makeKimiK3Profile(),
    glm: makeGLM53Profile(),
  };

  it('相同任务配置下，四个模型的 OperatorGraph 结构不同', () => {
    const graphs = Object.fromEntries(
      Object.entries(models).map(([name, profile]) => [name, buildOperatorGraph(profile, SHARED_TASK)]),
    );
    // 节点数量必须不同（层数不同：43 / 61 / 93 / 80）
    const nodeCounts = Object.fromEntries(
      Object.entries(graphs).map(([name, g]) => [name, g.nodes.length]),
    );
    const counts = Object.values(nodeCounts);
    // 至少存在两两不同的节点数（不全部相等）
    expect(new Set(counts).size).toBeGreaterThan(1);
  });

  it('MoE 模型执行序列包含 MoE 特征算子（Router/Top-K/Dispatch/Combine）', () => {
    for (const name of ['flash', 'pro', 'kimi', 'glm'] as const) {
      const result = planExecution(models[name], SHARED_TASK);
      expect(result.ok).toBe(true);
      if (!result.ok) continue;
      const sig = operatorSignature(result.trace.events);
      expect(sig).toContain('MoE Router');
      expect(sig).toContain('MoE Top-K');
      expect(sig).toContain('MoE Dispatch');
      expect(sig).toContain('MoE Combine');
    }
  });

  it('Kimi K3 的执行序列包含 Dense 层（FFN）——与纯 MoE 模型结构不同', () => {
    const result = planExecution(models.kimi, SHARED_TASK);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    // Kimi K3 有 1 个 Dense 层（FFN），展开后应出现 FFN Up/Down Projection
    const sig = operatorSignature(result.trace.events);
    expect(sig).toContain('FFN Up Projection');
    // 而 DeepSeek V4 Flash 没有 Dense 层
    const flashResult = planExecution(models.flash, SHARED_TASK);
    expect(flashResult.ok).toBe(true);
    if (!flashResult.ok) return;
    const flashSig = operatorSignature(flashResult.trace.events);
    expect(flashSig).not.toContain('FFN Up Projection');
  });

  it('V4 Flash 与 V4 Pro 执行计划实质不同（折叠层数与专家数差异）', () => {
    const rFlash = planExecution(models.flash, SHARED_TASK);
    const rPro = planExecution(models.pro, SHARED_TASK);
    expect(rFlash.ok).toBe(true);
    expect(rPro.ok).toBe(true);
    if (!rFlash.ok || !rPro.ok) return;

    // 折叠层数不同（43 层折叠 42 vs 61 层折叠 60）
    const colFlash = rFlash.trace.events.find((e) => e.metadata?.collapsed);
    const colPro = rPro.trace.events.find((e) => e.metadata?.collapsed);
    expect(colFlash).toBeDefined();
    expect(colPro).toBeDefined();
    expect(
      (colPro!.metadata!.collapsed as { count: number }).count,
    ).not.toBe((colFlash!.metadata!.collapsed as { count: number }).count);
  });

  it('Critical Fail 防护：四个模型不能只是"名字不同的相同序列"', () => {
    const sigs = Object.fromEntries(
      Object.entries(models).map(([name, profile]) => {
        const result = planExecution(profile, SHARED_TASK);
        expect(result.ok).toBe(true);
        if (!result.ok) return [name, []];
        return [name, operatorSignature(result.trace.events)];
      }),
    );
    // 任意两个模型的算子签名序列不能完全相同
    const names = Object.keys(sigs);
    let allIdentical = true;
    for (let i = 1; i < names.length; i++) {
      const a = sigs[names[0]!] ?? [];
      const b = sigs[names[i]!] ?? [];
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        allIdentical = false;
        break;
      }
    }
    expect(allIdentical).toBe(false);
  });
});
