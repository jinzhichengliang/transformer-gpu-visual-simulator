/**
 * TVIR 事件序列构建器（Simulation 层共享基础设施）。
 *
 * 负责为事件分配全局唯一的 id 与连续的 step。
 * 所有 Simulation Engine（GEMM、Attention、未来的 Operator）共用，
 * 保证一条 trace 内事件编号连续、跨 Operator 不冲突。
 */

import type { TVIREvent } from '../tvir/types';

export interface EventBuilder {
  events: TVIREvent[];
  counter: number;
  push(partial: Omit<TVIREvent, 'id' | 'step'>): void;
}

export function createEventBuilder(): EventBuilder {
  const builder: EventBuilder = {
    events: [],
    counter: 0,
    push(partial) {
      builder.events.push({
        id: `event-${builder.counter}`,
        step: builder.counter,
        ...partial,
      });
      builder.counter += 1;
    },
  };
  return builder;
}
