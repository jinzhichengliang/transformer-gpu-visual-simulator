import { describe, expect, it } from 'vitest';
import { simulateMultiGpu, DEFAULT_MULTI_GPU_CONFIG, isCollectiveDemo } from '../src/core/multigpu';
import { validateTVIRTrace } from '../src/core/tvir/validation';

describe('simulateMultiGpu', () => {
  for (const strategy of ['dp', 'tp', 'pp'] as const) {
    it(`${strategy} trace 通过 TVIR schema 校验`, () => {
      const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy });
      const validation = validateTVIRTrace(trace);
      expect(validation.valid, validation.errors.join('; ')).toBe(true);
    });

    it(`${strategy} trace 首尾为 GEMM_START/GEMM_END，step 连续`, () => {
      const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy });
      expect(trace.events[0]?.type).toBe('GEMM_START');
      expect(trace.events[trace.events.length - 1]?.type).toBe('GEMM_END');
      trace.events.forEach((e, i) => expect(e.step).toBe(i));
    });

    it(`${strategy} trace 每事件都有 what/why`, () => {
      const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy });
      for (const e of trace.events) {
        expect(e.what.length).toBeGreaterThan(0);
        expect(e.why.length).toBeGreaterThan(0);
      }
    });
  }

  it('dp 包含梯度 AllReduce（ReduceScatter + AllGather）', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'dp' });
    const commEvents = trace.events.filter((e) => (e.metadata as { comm?: object } | undefined)?.comm);
    expect(commEvents.length).toBeGreaterThan(0);
    const collectives = commEvents.map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('reduce_scatter');
    expect(collectives).toContain('allgather');
  });

  it('tp 包含部分和 AllReduce', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'tp' });
    const collectives = trace.events
      .filter((e) => (e.metadata as { comm?: object } | undefined)?.comm)
      .map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('reduce_scatter');
    expect(collectives).toContain('allgather');
  });

  it('pp 包含 P2P 通信', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'pp' });
    const collectives = trace.events
      .filter((e) => (e.metadata as { comm?: object } | undefined)?.comm)
      .map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('p2p');
  });

  it('AllReduce 环算法步数 = 2(N-1)，每步 N 次传输', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'dp', numGpus: 4 });
    const rsSteps = trace.events.filter((e) => {
      const comm = (e.metadata as { comm?: { collective: string; ringStep: number } } | undefined)?.comm;
      return comm?.collective === 'reduce_scatter';
    });
    const agSteps = trace.events.filter((e) => {
      const comm = (e.metadata as { comm?: { collective: string } } | undefined)?.comm;
      return comm?.collective === 'allgather';
    });
    // N=4: ReduceScatter 3 步 + AllGather 3 步
    expect(rsSteps.length).toBe(3);
    expect(agSteps.length).toBe(3);
    // 每步 4 次传输（环上每 GPU 发一次）
    for (const e of rsSteps) {
      const transfers = (e.metadata as { comm: { transfers: unknown[] } }).comm.transfers;
      expect(transfers.length).toBe(4);
    }
  });
});

describe('集合通信原语独立演示（手册 §27）', () => {
  for (const strategy of ['comm-allreduce', 'comm-allgather', 'comm-reducescatter'] as const) {
    it(`${strategy} trace 通过 TVIR schema 校验`, () => {
      const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy });
      const validation = validateTVIRTrace(trace);
      expect(validation.valid, validation.errors.join('; ')).toBe(true);
    });

    it(`${strategy} 首尾为 GEMM_START/GEMM_END，step 连续`, () => {
      const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy });
      expect(trace.events[0]?.type).toBe('GEMM_START');
      expect(trace.events[trace.events.length - 1]?.type).toBe('GEMM_END');
      trace.events.forEach((e, i) => expect(e.step).toBe(i));
    });
  }

  it('comm-allreduce 包含 ReduceScatter + AllGather 两段', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'comm-allreduce' });
    const collectives = trace.events
      .filter((e) => (e.metadata as { comm?: object } | undefined)?.comm)
      .map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('reduce_scatter');
    expect(collectives).toContain('allgather');
  });

  it('comm-allgather 只有 AllGather（无 ReduceScatter）', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'comm-allgather' });
    const collectives = trace.events
      .filter((e) => (e.metadata as { comm?: object } | undefined)?.comm)
      .map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('allgather');
    expect(collectives).not.toContain('reduce_scatter');
  });

  it('comm-reducescatter 只有 ReduceScatter（无 AllGather）', () => {
    const trace = simulateMultiGpu({ ...DEFAULT_MULTI_GPU_CONFIG, strategy: 'comm-reducescatter' });
    const collectives = trace.events
      .filter((e) => (e.metadata as { comm?: object } | undefined)?.comm)
      .map((e) => (e.metadata as { comm: { collective: string } }).comm.collective);
    expect(collectives).toContain('reduce_scatter');
    expect(collectives).not.toContain('allgather');
  });

  it('isCollectiveDemo 正确识别集合通信演示策略', () => {
    expect(isCollectiveDemo('comm-allreduce')).toBe(true);
    expect(isCollectiveDemo('comm-allgather')).toBe(true);
    expect(isCollectiveDemo('comm-reducescatter')).toBe(true);
    expect(isCollectiveDemo('dp')).toBe(false);
    expect(isCollectiveDemo('tp')).toBe(false);
    expect(isCollectiveDemo('pp')).toBe(false);
  });
});
