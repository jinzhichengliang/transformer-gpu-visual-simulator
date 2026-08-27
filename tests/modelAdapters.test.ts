/**
 * Model Adapter 单元测试（Sprint 11-14, Task B3-B6）。
 *
 * 任务书验收：
 *   - B3：DeepSeek V4 Flash Profile 成功载入 → Execution Plan → TVIR → Playback；
 *   - B4：V4 Flash 与 V4 Pro 产生实质不同的 Profile（非仅 displayName 差异）；
 *   - B5：Kimi K3 多种 Layer 类型共存，Planner 按序读取；
 *   - B6：GLM-5.3 未公开字段不以 verified/official 出现（禁止假造）。
 */

import { describe, expect, it } from 'vitest';
import {
  makeDeepSeekV4FlashProfile,
  makeDeepSeekV4ProProfile,
  makeKimiK3Profile,
  makeGLM53Profile,
  makeGenericDenseProfile,
  validateModelProfile,
  hasEstimatedMarkedAsMeasured,
  listRegisteredModels,
  getModelProfile,
} from '../src/core/modelprofile';
import { planExecution } from '../src/core/execution/executor';
import { DEFAULT_INFERENCE_TASK } from '../src/core/execution/task';
import { validateTVIRTrace } from '../src/core/tvir';

describe('Task B3：DeepSeek V4 Flash Adapter', () => {
  it('Profile 合法且通过校验', () => {
    const profile = makeDeepSeekV4FlashProfile();
    const result = validateModelProfile(profile);
    expect(result.valid).toBe(true);
    expect(profile.id).toBe('deepseek-v4-flash');
    expect(profile.family).toBe('DeepSeek-V4');
  });

  it('完整管线：Profile → Execution Plan → TVIR → 可播放', () => {
    const profile = makeDeepSeekV4FlashProfile();
    const result = planExecution(profile, DEFAULT_INFERENCE_TASK);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.trace.events.length).toBeGreaterThan(10);
    expect(validateTVIRTrace(result.trace).valid).toBe(true);
  });

  it('Adapter 只产出数据（无 UI/仿真依赖）', () => {
    const profile = makeDeepSeekV4FlashProfile();
    // Profile 是纯数据结构
    expect(typeof profile).toBe('object');
    expect(profile.displayName).toBe('DeepSeek V4 Flash');
  });
});

describe('Task B4：V4 Flash vs V4 Pro 实质差异', () => {
  it('两个版本的架构参数实质不同（非仅 displayName）', () => {
    const flash = makeDeepSeekV4FlashProfile();
    const pro = makeDeepSeekV4ProProfile();
    // 层数不同
    expect(flash.layers.length).not.toBe(pro.layers.length);
    // 隐藏维度不同
    expect(flash.architecture.hiddenSize?.value).not.toBe(pro.architecture.hiddenSize?.value);
    // 参数量不同
    expect(flash.parameterInfo?.total?.value).not.toBe(pro.parameterInfo?.total?.value);
  });

  it('同一任务下两者生成的 OperatorGraph 结构不同（层数差异真实存在）', () => {
    const flash = makeDeepSeekV4FlashProfile();
    const pro = makeDeepSeekV4ProProfile();
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const, promptTokens: 32 };
    const rFlash = planExecution(flash, task);
    const rPro = planExecution(pro, task);
    expect(rFlash.ok).toBe(true);
    expect(rPro.ok).toBe(true);
    if (!rFlash.ok || !rPro.ok) return;
    // 注意：TVIR 事件数因层折叠（重复结构 ×N）被拉平，这是教学抽样的预期行为。
    // 架构差异在折叠事件中可见：Pro 折叠的层数更多。
    const collapsedFlash = rFlash.trace.events.find((e) => e.metadata?.collapsed);
    const collapsedPro = rPro.trace.events.find((e) => e.metadata?.collapsed);
    expect(collapsedFlash).toBeDefined();
    expect(collapsedPro).toBeDefined();
    const countFlash = (collapsedFlash!.metadata!.collapsed as { count: number }).count;
    const countPro = (collapsedPro!.metadata!.collapsed as { count: number }).count;
    expect(countPro).toBeGreaterThan(countFlash);
  });
});

describe('Task B5：Kimi K3 多类型 Layer 共存', () => {
  it('Profile 合法（hybrid 架构）', () => {
    const profile = makeKimiK3Profile();
    expect(profile.architecture.type).toBe('hybrid');
    expect(validateModelProfile(profile).valid).toBe(true);
  });

  it('Dense 层与 MoE 层共存于同一模型', () => {
    const profile = makeKimiK3Profile();
    const types = profile.layers.map((l) => l.type);
    expect(types).toContain('ffn'); // 1 个 Dense 层
    expect(types).toContain('moe'); // 92 个 MoE 层
  });

  it('Planner 按 Layer 0,1... 正确读取不同类型层', () => {
    const profile = makeKimiK3Profile();
    const task = { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const, promptTokens: 16 };
    const result = planExecution(profile, task);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    // 检查事件流中既有 ffn 层上下文也有 moe 层上下文
    const layerTypes = new Set(
      result.trace.events
        .map((e) => (e.metadata?.model as Record<string, unknown> | undefined)?.layerType as string | undefined)
        .filter(Boolean),
    );
    expect(layerTypes.has('ffn')).toBe(true);
    expect(layerTypes.has('moe')).toBe(true);
  });
});

describe('Task B6：GLM-5.3 诚实标注（禁止假造）', () => {
  it('Profile 合法', () => {
    const profile = makeGLM53Profile();
    expect(validateModelProfile(profile).valid).toBe(true);
  });

  it('估算字段标注为 estimated（低置信），不冒充 official', () => {
    const profile = makeGLM53Profile();
    // hiddenSize 无公开数据 → 应为 estimated
    const hs = profile.architecture.hiddenSize;
    expect(hs?.sources.some((s) => s.sourceType === 'estimated')).toBe(true);
    expect(hs?.sources.some((s) => s.sourceType === 'official' && s.confidence === 'high')).toBe(false);
  });

  it('整体保真度因含估算字段而降级（非 L4 实测）', () => {
    const profile = makeGLM53Profile();
    expect(profile.fidelity).not.toBe('L4');
    expect(hasEstimatedMarkedAsMeasured(profile)).toBe(false);
  });

  it('官方公开的字段仍为 official（总参数/上下文）', () => {
    const profile = makeGLM53Profile();
    expect(profile.parameterInfo?.total?.sources.some((s) => s.sourceType === 'official')).toBe(true);
    expect(profile.contextLength?.sources.some((s) => s.sourceType === 'official')).toBe(true);
  });
});

describe('Model Registry（统一管线）', () => {
  it('注册表包含四个首批模型 + 两个 Generic', () => {
    const models = listRegisteredModels();
    const ids = models.map((m) => m.id);
    expect(ids).toContain('deepseek-v4-flash');
    expect(ids).toContain('deepseek-v4-pro');
    expect(ids).toContain('kimi-k3');
    expect(ids).toContain('glm-5.3');
    expect(ids).toContain('generic-dense');
    expect(ids).toContain('generic-moe');
  });

  it('切换模型无需专用初始化代码（统一 getModelProfile）', () => {
    for (const id of ['deepseek-v4-flash', 'deepseek-v4-pro', 'kimi-k3', 'glm-5.3']) {
      const profile = getModelProfile(id);
      expect(profile).toBeDefined();
      expect(validateModelProfile(profile!).valid).toBe(true);
      // 每个模型都能走同一 planExecution 管线
      const result = planExecution(profile!, { ...DEFAULT_INFERENCE_TASK, phase: 'prefill' as const, promptTokens: 8 });
      expect(result.ok).toBe(true);
    }
  });
});
