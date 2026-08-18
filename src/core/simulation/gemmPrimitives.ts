/**
 * GEMM 事件生成原语（V0.2 架构验收的关键复用点）。
 *
 * 实施手册 §20 要求：
 *   - 已有 GEMM Engine 必须被 Q/K/V Projection、QK、AV 重用；
 *   - 禁止复制 GEMM visualization code。
 *
 * 因此把"生成一段 GEMM 事件流"抽成可复用原语：
 *   - 张量名（left/right/out）、算子名、kernel 名、教学文案均可参数化；
 *   - 每个事件写入 operator 字段与 metadata.gemm，供 UI 泛化消费；
 *   - GPU 层面的教学解释（tiling / shared memory / warp / register→MMA）
 *     对所有 GEMM 通用，保持与 V0.1 完全一致。
 */

import type { EventBuilder } from './eventBuilder';
import type { TileRef } from '../tvir/types';

export interface GemmPrimitiveConfig {
  M: number;
  N: number;
  K: number;
  tileM: number;
  tileN: number;
  tileK: number;
  numSM: number;
  warpsPerBlock: number;
  /** 左矩阵名，如 "Q" */
  left: string;
  /** 右矩阵名，如 "Kᵀ" */
  right: string;
  /** 输出矩阵名，如 "S" */
  out: string;
  /** 所属算子名（写入事件的 operator 字段），如 "QK MatMul" */
  operator: string;
  /** kernel 名 */
  kernel: string;
  /** 阶段标签，用于标题，如 "QK MatMul" */
  label: string;
  /** 可选：覆盖 GEMM_START 的教学文案 */
  startWhat?: string;
  startWhy?: string;
}

function tileRef(tensor: string, tileRow: number, tileCol: number): TileRef {
  return { tensor, tileRow, tileCol, label: `${tensor}[${tileRow},${tileCol}]` };
}

/**
 * 向 builder 追加一段完整的 GEMM 事件流：
 * GEMM_START → TILE_CREATE → KERNEL_LAUNCH → 逐 Block（含 K 维循环）→ GEMM_END
 */
export function emitGemmEvents(builder: EventBuilder, config: GemmPrimitiveConfig): void {
  const { M, N, K, tileM, tileN, tileK, numSM, warpsPerBlock } = config;
  const { left, right, out, operator, kernel, label } = config;

  const tilesM = Math.ceil(M / tileM);
  const tilesN = Math.ceil(N / tileN);
  const tilesK = Math.ceil(K / tileK);
  const numBlocks = tilesM * tilesN;

  /** 每个事件统一携带的算子上下文（放在 metadata.gemm 下，供 UI 泛化展示） */
  const gemmMeta = { M, N, K, tileM, tileN, tileK, left, right, out };
  const meta = { gemm: gemmMeta };

  builder.push({
    type: 'GEMM_START',
    title: `${label}：${out} = ${left} × ${right}`,
    what:
      config.startWhat ??
      `准备计算 ${out}[${M}×${N}] = ${left}[${M}×${K}] × ${right}[${K}×${N}]。`,
    why:
      config.startWhy ??
      'GEMM（通用矩阵乘法）是 Transformer 中 Attention 与 FFN 的核心计算。理解它在 GPU 上的执行方式，是理解整个模型执行的起点。',
    operator,
    metadata: meta,
  });

  builder.push({
    type: 'TILE_CREATE',
    title: '矩阵切分为 Tiles',
    what: `${left} 切成 ${tilesM}×${tilesK} 个 Tile，${right} 切成 ${tilesK}×${tilesN} 个，${out} 切成 ${tilesM}×${tilesN} 个（每个输出 Tile ${tileM}×${tileN}）。`,
    why: 'GPU 一次无法把整个矩阵放进片上高速存储。Tiling 把大矩阵切成小块，让每个 Block 只处理一小块，并大幅提高数据复用率、减少对慢速 HBM 的重复访问。',
    operator,
    metadata: { gemm: { ...gemmMeta, tilesM, tilesN, tilesK } },
  });

  builder.push({
    type: 'KERNEL_LAUNCH',
    title: `启动 Kernel：${kernel}`,
    what: `启动 kernel，Grid 中共 ${numBlocks} 个 Thread Block（每个 Block 负责 ${out} 的一个 ${tileM}×${tileN} Tile），每个 Block 含 ${warpsPerBlock} 个 Warp。`,
    why: 'GPU 通过 Grid → Block → Warp 的层级把计算拆分给成千上万个线程并行执行。',
    operator,
    kernel,
    metadata: { gemm: { ...gemmMeta, numBlocks, warpsPerBlock } },
  });

  // ---------- 逐 Block 执行 ----------
  for (let blockRow = 0; blockRow < tilesM; blockRow++) {
    for (let blockCol = 0; blockCol < tilesN; blockCol++) {
      const blockId = blockRow * tilesN + blockCol;
      const sm = blockId % numSM;
      const outTile = tileRef(out, blockRow, blockCol);

      builder.push({
        type: 'BLOCK_SCHEDULE',
        title: `Block ${blockId} → SM ${sm}`,
        what: `调度器把 Block ${blockId}（负责 ${outTile.label}）分配到 SM ${sm}。`,
        why: 'Thread Block 是 GPU 调度的基本单位：一个 Block 只会被调度到一个 SM 上执行，不会跨 SM 拆分。',
        operator,
        block: blockId,
        sm,
        tile: outTile,
        metadata: meta,
      });

      for (let warp = 0; warp < warpsPerBlock; warp++) {
        builder.push({
          type: 'WARP_SCHEDULE',
          title: `Block ${blockId}：Warp ${warp} 就绪`,
          what: `Block ${blockId} 内的 Warp ${warp}（32 个线程）进入执行队列。`,
          why: 'Warp 是 SM 内真正的调度与执行单位——硬件以 32 线程为一组发射指令。',
          operator,
          block: blockId,
          warp,
          sm,
          metadata: meta,
        });
      }

      // ---------- K 维度循环（Tiling 的核心） ----------
      for (let kIter = 0; kIter < tilesK; kIter++) {
        const leftTile = tileRef(left, blockRow, kIter);
        const rightTile = tileRef(right, kIter, blockCol);
        const warp = kIter % warpsPerBlock;

        builder.push({
          type: 'MEMORY_LOAD',
          title: `取 ${leftTile.label}：HBM → L2`,
          what: `将 ${left} 的 Tile ${leftTile.label}（第 ${kIter + 1}/${tilesK} 段 K）从 HBM 读入 L2 Cache。`,
          why: '计算前必须先把数据从显存（HBM）搬到更快的缓存层级。L2 是全 GPU 共享的缓存，可能已命中其他 Block 取过的数据。',
          operator,
          block: blockId,
          sm,
          source: 'HBM',
          destination: 'L2',
          tensor: left,
          tile: leftTile,
          metadata: meta,
        });

        builder.push({
          type: 'MEMORY_MOVE',
          title: `Load ${leftTile.label} → Shared Memory`,
          what: `将 ${leftTile.label} 从 L2 搬运到 SM ${sm} 的 Shared Memory。`,
          why: '同一 Tile 会被 Block 内多个线程重复使用。先搬入 Shared Memory（片上、比 HBM 快约两个数量级），可大幅减少对慢速全局内存的重复访问——这正是 Tiling 的意义。',
          operator,
          block: blockId,
          sm,
          source: 'L2',
          destination: 'SHARED_MEMORY',
          tensor: left,
          tile: leftTile,
          metadata: meta,
        });

        builder.push({
          type: 'MEMORY_LOAD',
          title: `取 ${rightTile.label}：HBM → L2`,
          what: `将 ${right} 的 Tile ${rightTile.label} 从 HBM 读入 L2 Cache。`,
          why: `与 ${left} Tile 同理：${right} 的数据也要先搬到更快的缓存层级。`,
          operator,
          block: blockId,
          sm,
          source: 'HBM',
          destination: 'L2',
          tensor: right,
          tile: rightTile,
          metadata: meta,
        });

        builder.push({
          type: 'MEMORY_MOVE',
          title: `Load ${rightTile.label} → Shared Memory`,
          what: `将 ${rightTile.label} 从 L2 搬运到 Shared Memory。`,
          why: `${right} Tile 同样会被 Block 内线程重复使用，搬入 Shared Memory 提高复用。`,
          operator,
          block: blockId,
          sm,
          source: 'L2',
          destination: 'SHARED_MEMORY',
          tensor: right,
          tile: rightTile,
          metadata: meta,
        });

        builder.push({
          type: 'SYNC',
          title: '线程同步 __syncthreads()',
          what: `Block ${blockId} 内所有线程在 barrier 处等待，直到 ${left}/${right} Tile 全部写入 Shared Memory。`,
          why: '必须保证 Shared Memory 中的数据对所有线程可见后，才能开始计算，否则会读到未就绪的数据。',
          operator,
          block: blockId,
          sm,
          metadata: meta,
        });

        builder.push({
          type: 'MEMORY_MOVE',
          title: 'Shared Memory → Register',
          what: `Warp ${warp} 把 ${leftTile.label}、${rightTile.label} 的 fragment 从 Shared Memory 装入寄存器。`,
          why: 'Tensor Core 的操作数必须来自寄存器（fragment），不能直接从显存或 Shared Memory 取数。寄存器是线程私有的最快存储。',
          operator,
          block: blockId,
          warp,
          sm,
          source: 'SHARED_MEMORY',
          destination: 'REGISTER',
          metadata: meta,
        });

        builder.push({
          type: 'MMA',
          title: `Tensor Core 执行 MMA（K 段 ${kIter + 1}/${tilesK}）`,
          what: `Tensor Core 对寄存器中的 ${left}/${right} fragment 执行矩阵乘累加：D = ${left} × ${right} + C。`,
          why: 'Tensor Core 是专为矩阵乘累加设计的硬件单元，一条 MMA 指令完成一小块矩阵乘，远快于普通 CUDA Core 逐元素计算。',
          operator,
          block: blockId,
          warp,
          sm,
          metadata: { gemm: { ...gemmMeta, kIteration: kIter, outTile: outTile.label } },
        });

        builder.push({
          type: 'ACCUMULATE',
          title: `累加到 ${outTile.label} 的部分和`,
          what: `本段 K 的乘积累加进寄存器中 ${outTile.label} 的 accumulator（部分和）。`,
          why: `完整的 ${out} Tile 需要沿 K 维度累加 ${tilesK} 段的部分和。累加在寄存器中完成，避免中间结果反复搬运。`,
          operator,
          block: blockId,
          warp,
          sm,
          tile: outTile,
          metadata: { gemm: { ...gemmMeta, kIteration: kIter } },
        });
      }

      builder.push({
        type: 'MEMORY_STORE',
        title: `写回 ${outTile.label}：Register → HBM`,
        what: `Block ${blockId} 把累加完成的 ${outTile.label} 从寄存器写回 HBM。`,
        why: '所有 K 段累加完成后才一次性写回结果，避免中间结果反复占用宝贵的 HBM 带宽。',
        operator,
        block: blockId,
        sm,
        source: 'REGISTER',
        destination: 'HBM',
        tensor: out,
        tile: outTile,
        metadata: meta,
      });
    }
  }

  builder.push({
    type: 'GEMM_END',
    title: `${label} 完成`,
    what: `全部 ${numBlocks} 个 Block 执行完毕，${out}[${M}×${N}] 已写入 HBM。`,
    why: '注意：本工具是教学仿真，步骤只表示逻辑先后顺序，不代表真实 cycle 数或时序。',
    operator,
    metadata: meta,
  });
}
