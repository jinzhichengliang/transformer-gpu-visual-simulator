/**
 * Golden Trace 回归测试（Sprint 18, Task 18）。
 *
 * 任务书要求：
 *   - 固定 TinyMoETransformer（2 Layers / 2 Heads / 4 Experts / Top-2）；
 *   - 固定 Prefill 4 tokens + Decode 2 tokens；
 *   - 任何架构修改后重新生成 TVIR，自动比较 expected vs actual；
 *   - 非预期结构变化 → 测试失败，防止悄悄修改 execution semantics。
 *
 * 重新生成快照（仅当有意修改执行语义时）：
 *   npx tsx tests/golden/generateGolden.ts
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeTinyMoEProfile, GOLDEN_TASK, extractStructuralFingerprint } from './tinyMoe';
import { planExecution } from '../../src/core/execution/executor';
import { validateTVIRTrace } from '../../src/core/tvir';

const here = dirname(fileURLToPath(import.meta.url));

interface GoldenPayload {
  meta: Record<string, unknown>;
  totalEvents: number;
  fingerprint: Array<Record<string, unknown>>;
}

function loadGolden(): GoldenPayload {
  const raw = readFileSync(join(here, 'golden_trace.json'), 'utf-8');
  return JSON.parse(raw) as GoldenPayload;
}

describe('Golden Trace 回归（TinyMoETransformer）', () => {
  const golden = loadGolden();

  it('TinyMoE 执行计划生成成功且通过 TVIR 校验', () => {
    const result = planExecution(makeTinyMoEProfile(), GOLDEN_TASK);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(validateTVIRTrace(result.trace).valid).toBe(true);
  });

  it('执行语义与 golden 快照一致（非预期结构变化 → 失败）', () => {
    const result = planExecution(makeTinyMoEProfile(), GOLDEN_TASK);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const actual = extractStructuralFingerprint(result.trace.events);

    // 事件总数必须一致
    expect(actual.length, '事件总数与 golden 快照不一致').toBe(golden.totalEvents);

    // 逐事件结构指纹必须一致（首个差异处给出定位信息）
    for (let i = 0; i < actual.length; i++) {
      const exp = golden.fingerprint[i];
      const act = actual[i];
      expect(
        JSON.stringify(act),
        `事件 ${i} 结构指纹与 golden 不一致：\nexpected: ${JSON.stringify(exp)}\nactual:   ${JSON.stringify(act)}`,
      ).toBe(JSON.stringify(exp));
    }
  });

  it('golden 快照包含 Prefill 与 Decode 两阶段（语义完整性）', () => {
    const phases = new Set(golden.fingerprint.map((e) => e.phase).filter(Boolean));
    expect(phases.has('prefill')).toBe(true);
    expect(phases.has('decode')).toBe(true);
    // Decode 有 2 个 step
    const decodeSteps = new Set(
      golden.fingerprint.filter((e) => e.phase === 'decode').map((e) => e.decodeStep),
    );
    expect(decodeSteps.size).toBeGreaterThanOrEqual(2);
  });
});
