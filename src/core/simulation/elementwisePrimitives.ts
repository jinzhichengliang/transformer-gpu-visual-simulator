/**
 * 逐元素 / 行归约 Kernel 事件生成原语（Scale、Mask、Softmax 共用）。
 *
 * Attention 中 Scale、Mask、Softmax 不涉及 Tensor Core 的 MMA，
 * 而是对中间张量（注意力分数 S）做逐元素或按行归约的计算，
 * 通常由普通 CUDA Core 执行。该原语生成这类 kernel 的"教学型"事件流：
 *
 * KERNEL_LAUNCH → 逐 Block：
 *   BLOCK_SCHEDULE → WARP_SCHEDULE
 *   → MEMORY_LOAD(HBM→L2) → MEMORY_MOVE(L2→SMEM) → SYNC
 *   → MEMORY_MOVE(SMEM→Register) → SYNC（跨线程协作）
 *   → MEMORY_STORE(Register→HBM)
 * → 结束
 */

import type { EventBuilder } from './eventBuilder';

export interface ElementwiseConfig {
  rows: number;
  cols: number;
  numSM: number;
  warpsPerBlock: number;
  /** 所属算子名（写入 operator 字段） */
  operator: string;
  kernel: string;
  /** 阶段标签，用于标题 */
  label: string;
  /** 操作的主张量名 */
  tensor: string;
  /** KERNEL_LAUNCH 的教学文案 */
  launchWhat: string;
  launchWhy: string;
  /** 每个 Block 内"计算"步骤的说明（写入 SYNC 前的等待解释） */
  computeWhat: string;
  computeWhy: string;
  /**
   * 教学抽样：只详细展示前 detailBlocks 个 Block 的完整流程，
   * 其余 Block 用一条汇总事件概括（它们的行为完全相同）。
   * 目的：控制 trace 长度，避免播放时间过长。默认 2。
   */
  detailBlocks?: number;
}

/**
 * 向 builder 追加一段逐元素 kernel 的事件流。
 */
export function emitElementwiseEvents(builder: EventBuilder, config: ElementwiseConfig): void {
  const { rows, cols, numSM, warpsPerBlock } = config;
  const { operator, kernel, label, tensor } = config;
  const detailBlocks = Math.min(config.detailBlocks ?? 2, rows);

  // 教学简化：每个 Block 负责一行，Grid 行数 = rows（见 CONCEPTS.md 简化声明）
  const numBlocks = rows;

  builder.push({
    type: 'KERNEL_LAUNCH',
    title: `${label}：启动 ${kernel}`,
    what: config.launchWhat,
    why: config.launchWhy,
    operator,
    kernel,
    metadata: { rows, cols, numBlocks, warpsPerBlock },
  });

  for (let blockId = 0; blockId < numBlocks; blockId++) {
    // 教学抽样：只详细展示前 detailBlocks 个 Block，其余行为完全相同，
    // 用一条汇总事件带过，避免 trace 过长。
    if (blockId >= detailBlocks) {
      if (blockId === detailBlocks) {
        builder.push({
          type: 'SYNC',
          title: `${label}：其余 ${numBlocks - detailBlocks} 个 Block 并行执行`,
          what: `Block ${detailBlocks}～${numBlocks - 1} 以与前面完全相同的方式并行处理各自的行（此处为教学抽样，不再逐一展示）。`,
          why: 'GPU 的并行性体现在大量 Block 同时做同一件事。为控制演示长度，本仿真只详细展示前几个 Block，其余 Block 的执行过程与之完全一致。',
          operator,
          metadata: { rows, cols, sampledBlocks: detailBlocks, totalBlocks: numBlocks },
        });
      }
      continue;
    }

    const sm = blockId % numSM;

    builder.push({
      type: 'BLOCK_SCHEDULE',
      title: `Block ${blockId} → SM ${sm}`,
      what: `调度器把 Block ${blockId}（负责 ${tensor} 的第 ${blockId} 行）分配到 SM ${sm}。`,
      why: '逐元素 kernel 也遵循同样的 Grid → Block → SM 调度规则：一个 Block 只会被调度到一个 SM 上执行。',
      operator,
      block: blockId,
      sm,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'WARP_SCHEDULE',
      title: `Block ${blockId}：Warp 0 就绪`,
      what: `Block ${blockId} 内的 Warp 0（32 个线程）进入执行队列。`,
      why: 'Warp 是 SM 内真正的调度与执行单位——硬件以 32 线程为一组发射指令。',
      operator,
      block: blockId,
      warp: 0,
      sm,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'MEMORY_LOAD',
      title: `取 ${tensor} 第 ${blockId} 行：HBM → L2`,
      what: `将 ${tensor} 的第 ${blockId} 行（${cols} 个元素）从 HBM 读入 L2 Cache。`,
      why: '与 GEMM 同理：计算前必须先把数据从显存（HBM）搬到更快的缓存层级。',
      operator,
      block: blockId,
      sm,
      source: 'HBM',
      destination: 'L2',
      tensor,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'MEMORY_MOVE',
      title: `Load ${tensor} 第 ${blockId} 行 → Shared Memory`,
      what: `将该行数据从 L2 搬运到 SM ${sm} 的 Shared Memory。`,
      why: '行内多个元素需要被 Block 内线程共享读取（例如 Softmax 需要整行参与归约），先搬入 Shared Memory 提高访问效率。',
      operator,
      block: blockId,
      sm,
      source: 'L2',
      destination: 'SHARED_MEMORY',
      tensor,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'SYNC',
      title: '线程同步 __syncthreads()',
      what: `Block ${blockId} 内所有线程等待该行数据全部写入 Shared Memory。`,
      why: '必须保证 Shared Memory 中的数据对所有线程可见后，才能开始计算。',
      operator,
      block: blockId,
      sm,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'MEMORY_MOVE',
      title: 'Shared Memory → Register',
      what: `Warp 0 把 ${tensor} 第 ${blockId} 行的元素从 Shared Memory 装入各线程的寄存器。`,
      why: '逐元素计算在 CUDA Core 上完成，操作数同样要先进入线程私有的寄存器。',
      operator,
      block: blockId,
      warp: 0,
      sm,
      source: 'SHARED_MEMORY',
      destination: 'REGISTER',
      metadata: { rows, cols },
    });

    builder.push({
      type: 'SYNC',
      title: `${label}：行内协作计算`,
      what: config.computeWhat,
      why: config.computeWhy,
      operator,
      block: blockId,
      sm,
      metadata: { rows, cols },
    });

    builder.push({
      type: 'MEMORY_STORE',
      title: `写回 ${tensor} 第 ${blockId} 行：Register → HBM`,
      what: `Block ${blockId} 把计算完成的第 ${blockId} 行从寄存器写回 HBM。`,
      why: '结果写回显存，供下一个 Operator 读取。',
      operator,
      block: blockId,
      sm,
      source: 'REGISTER',
      destination: 'HBM',
      tensor,
      metadata: { rows, cols },
    });
  }

  builder.push({
    type: 'GEMM_END',
    title: `${label} 完成`,
    what: `全部 ${numBlocks} 个 Block 执行完毕，${tensor}[${rows}×${cols}] 已就地更新并写入 HBM。`,
    why: '注意：本工具是教学仿真，步骤只表示逻辑先后顺序，不代表真实 cycle 数或时序。',
    operator,
    metadata: { rows, cols },
  });
}
