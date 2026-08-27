/**
 * Golden Trace 生成器（一次性运行）：
 *   生成 TinyMoETransformer 的执行计划并保存结构指纹到 golden_trace.json。
 * 运行：npx tsx tests/golden/generateGolden.ts
 */
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeTinyMoEProfile, GOLDEN_TASK, extractStructuralFingerprint } from './tinyMoe';
import { planExecution } from '../../src/core/execution/executor';

const here = dirname(fileURLToPath(import.meta.url));

const profile = makeTinyMoEProfile();
const result = planExecution(profile, GOLDEN_TASK);
if (!result.ok) {
  console.error('生成失败：', result.error);
  process.exit(1);
}

const fingerprint = extractStructuralFingerprint(result.trace.events);
const payload = {
  meta: {
    model: 'TinyMoE Transformer',
    task: GOLDEN_TASK,
    generatedAt: new Date().toISOString(),
    note: '结构指纹快照：任何架构修改导致执行语义变化时，本文件比对会失败。',
  },
  totalEvents: result.trace.events.length,
  fingerprint,
};

const outPath = join(here, 'golden_trace.json');
writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf-8');
console.log(`Golden trace 已写入 ${outPath}（${result.trace.events.length} 个事件）`);
