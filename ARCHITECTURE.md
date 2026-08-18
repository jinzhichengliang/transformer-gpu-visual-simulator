# ARCHITECTURE.md — 架构边界（绝不能违反）

> 来源：实施手册 §3、§31、§41。TVIR 优先于 UI 是整个项目最重要的架构资产。

## 唯一合法数据流

```
Simulation Engine ──► TVIR events ──► Playback Engine ──► Visualization (UI)
```

- 前端永远不知道事件来自哪里（今天是 GEMM simulator，明天可以是 Nsight trace parser）。
- TVIR 是 Simulation 与 Visualization 之间**唯一**的数据接口。

## 模块职责与禁令

| 模块 | 职责 | 禁止 |
| --- | --- | --- |
| `core/tvir` | 定义事件类型与 schema 校验 | 不得包含 UI 样式、不得 import React |
| `core/simulation` | 生成 TVIREvent[]（教学合理即可） | 不得 import React、不得操作 DOM、不得包含 CSS |
| `core/playback` | 选择当前事件、控制时间、发布事件 | 不得理解任何 Operator：禁止 `if (event.type === "MMA")` 这类按类型做业务分支 |
| `components/*`（所有 View） | 只消费当前 TVIR event 渲染 | 不得自己计算 simulation、不得决定下一步、不得维护执行状态 |

### 明确禁止的调用关系

- ❌ `GPUView → calculate GEMM`
- ❌ `MatrixView → simulate warp`
- ❌ `PlaybackEngine → understand Attention/GEMM/Softmax`
- ❌ UI 组件绕过 Playback 直接 import Simulation Engine

## 关键验收问题（每个版本都要问）

> 如果未来把 GEMM simulator 删除，改成 Nsight trace parser，前端是否需要修改？
> **正确答案：基本不需要。** 若答案是需要大改 GPUView，说明架构已跑偏。

## 目录结构约定（V0.3 起，V1.0 更新）

```
src/
├── core/                    # 纯逻辑层，与 React 完全无关
│   ├── tvir/                # TVIR 类型定义、schema 校验、示例 trace、投影工具
│   ├── simulation/          # 教学仿真引擎（输出 TVIREvent[]）
│   │   ├── eventBuilder.ts        # 事件序列构建器（id/step 分配）
│   │   ├── gemmPrimitives.ts      # GEMM 事件生成原语（所有 GEMM 类算子复用）
│   │   ├── elementwisePrimitives.ts # 逐元素/行归约 kernel 事件原语
│   │   ├── gemmEngine.ts          # 独立 GEMM 场景（V0.1）
│   │   ├── attentionEngine.ts     # Attention 场景（V0.2）+ 可嵌入子图原语 emitAttentionEvents
│   │   └── transformerBlockEngine.ts # Transformer Block 场景（V0.3）
│   ├── compiler/            # 编译知识层（V0.4，静态知识 + 事件投影，非执行引擎）
│   │   └── operatorKnowledge.ts   # Math→Operator→IR→Kernel→GPU 五层映射
│   ├── realtrace/           # 真实 trace 解析层（V0.5）
│   │   ├── nsightParser.ts        # Nsight 风格 JSON → TVIRTrace（架构验收点）
│   │   └── sampleTrace.ts         # 内置示例 trace（sample: true，非实测）
│   ├── sasstrace/           # SASS trace 适配层（V0.8，TVIR Adapter）
│   │   ├── sassParser.ts          # NVBit 风格 SASS JSON → TVIRTrace（指令级映射）
│   │   └── sampleSassTrace.ts     # 内置示例指令流（sample: true，教学示意编排）
│   ├── perf/                # 性能分析层（V0.6，纯投影 + 屋顶线教学模型）
│   │   ├── metrics.ts             # computePerfReport：六项指标 + 数据来源标注；
│   │   │                          # V0.7 起接收 HardwareSpec，导出 modelGemm/modelElementwise
│   │   └── playground.ts          # Architecture Playground 对比分析（V0.7）
│   ├── multigpu/            # Multi-GPU 层（V0.9，解析式通信建模）
│   │   ├── commPrimitives.ts      # Ring 集合通信原语（ReduceScatter/AllGather/AllReduce/P2P）
│   │   └── multiGpuEngine.ts      # DP/TP/PP 策略编排 + 通信原语独立演示
│   ├── model/               # Model 层（V1.0，模型结构树 + 激活路径投影）
│   │   └── modelStructure.ts      # Transformer Block 结构树；operator → 模型节点查找
│   └── playback/            # Playback Engine（只认事件序号与时间）
├── components/              # React 视图层，只读 TVIR 状态
│   ├── ControlBar/
│   ├── MatrixView/
│   ├── GpuView/
│   ├── MemoryView/
│   ├── TensorCoreView/
│   ├── EventExplanation/
│   ├── Timeline/
│   ├── CompilerView/        # 编译层级视图（V0.4）
│   ├── KernelTimeline/      # kernel 实测时间轴视图（V0.5）
│   ├── PerfPanel/           # 性能分析面板（V0.6，六项指标 + 来源徽标）
│   ├── ArchitecturePlayground/ # 架构实验场（V0.7，硬件滑块 + 基线对比）
│   ├── InstructionView/     # SASS 指令级视图（V0.8，按 Warp 分组 + 类别着色）
│   ├── MultiGpuView/        # Multi-GPU 环形拓扑视图（V0.9）
│   └── ModelView/           # 模型结构树视图（V1.0，激活路径高亮）
└── state/                   # React 侧的播放状态绑定（usePlayback hook）
```

## V0.2 架构约定（Attention）

1. **TVIR 基础架构零改动**：V0.2 只使用 V0.1 预留的可选字段 `operator` 与 `metadata`，
   未新增事件类型、未修改 schema。这验证了 TVIR 作为稳定接口的价值。
2. **GEMM 复用铁律**（实施手册 §20）：Q/K/V Projection、QK MatMul、AV MatMul、
   Output Projection 全部通过 `emitGemmEvents()` 生成事件；
   **禁止复制 GEMM 事件生成/可视化逻辑**。新增算子时先检查能否复用现有原语。
3. **metadata.gemm 约定**：GEMM 类事件在 `metadata.gemm` 下携带
   `{ M, N, K, tileM, tileN, tileK, left, right, out }`；
   UI 通过 `projectMatrixScene()`（纯投影函数）消费它驱动矩阵视图，
   不得在 UI 里硬编码任何算子的矩阵形状。
4. **Operator 导航**：Timeline 通过 `projectOperatorSegments()`（按 `operator` 字段分段，
   不理解算子语义）渲染算子分段条；EventExplanation 展示 `operator` 标签。
5. **教学抽样**：逐元素 kernel（Scale/Mask/Softmax）只详细展示前 2 个 Block，
   其余 Block 用一条汇总事件概括（见 CONCEPTS.md 声明）。

## V0.3 架构约定（Transformer Block）

1. **TVIR 基础架构仍零改动**：V0.3 只使用既有的 `operator` 与 `metadata` 可选字段，
   未新增事件类型、未修改 schema。TVIR 作为稳定接口的价值再次得到验证。
2. **Attention 是可嵌入子图**（V0.3 架构验收点）：`emitAttentionEvents(builder, config, prefix?, inputLabel?)`
   是 Attention 的唯一实现。Transformer Block 直接调用它嵌入完整 Attention，
   **禁止复制任何 Attention 事件生成逻辑**。未来多层堆叠也通过该原语复用。
3. **`inputLabel` 参数**：Attention 的输入张量标签可配置（独立场景为 "X"，
   Block 场景为归一化后的 "Xn"），矩阵视图与公式随之正确显示 Pre-Norm 数据流。
4. **Block 内新算子复用现有原语**：RMSNorm、Residual、SiLU 复用 `emitElementwiseEvents()`
   （行归约/逐元素 kernel）；FFN Up/Down Projection 复用 `emitGemmEvents()`。
   V0.3 未引入任何新的底层原语——这正是手册 §40 预期的"GEMM 成为基础 primitive 后开发加速"。
5. **共享 builder 保证编号连续**：Block 内所有算子共用一个 `createEventBuilder()`，
   跨算子事件 id/step 全局连续，Playback 无需感知算子边界。

## V0.4 架构约定（Compiler View）

1. **TVIR 基础架构仍零改动**：V0.4 没有新增事件类型，编译视图完全由事件既有的
   `operator` / `kernel` / `metadata.gemm` 公开字段驱动。
2. **编译知识层是静态知识 + 纯投影，不是执行引擎**：`core/compiler/operatorKnowledge.ts`
   维护"算子名 → 编译配方"知识表，`projectCompileChain()` 把当前事件投影为
   Math → Operator → IR → Kernel → GPU 五层下钻链。它不生成执行事件、不进入播放序列，
   也不 import Simulation Engine 内部实现（只读事件公开字段）。
3. **唯一的登记耦合点**：新增算子时需同步在 `OPERATOR_RECIPES` 登记编译知识
   （算子名与事件 operator 字段一一对应）。`hasCompileRecipe()` + 单元测试
   （compilerView.test.ts 的覆盖度用例）保证任何新算子漏登记都会在 CI 暴露。
4. **交互方式**：不新增独立"选中"交互。用户在 Timeline 算子分段条点击某算子 →
   seek 到该算子事件 → `projectCompileChain` 自动投影出该算子的下钻链。
   这保持了"UI 只消费当前事件"的架构铁律。
5. **未登记算子的回退**：`projectCompileChain` 对未知算子返回回退配方（不报错），
   对未来 Nsight 真实 trace（V0.5）中的未知 kernel 同样适用。

## V0.5 架构约定（Real Trace）

1. **架构验收点兑现**（实施手册 §9/§23）：把数据源从仿真引擎换成 trace parser，
   Playback 与 UI **零改动**。`parseNsightTrace()` 的输出与 `simulateGemm()` 一样
   都是合法 `TVIRTrace`，经同一 `PlaybackEngine` 与同一组 View 消费。
2. **TVIR 仅做向后兼容扩展**：`TVIRTrace` 新增两个**可选**字段——
   `provenance?: 'simulation' | 'real-trace'` 与 `isSample?: boolean`。
   未提供 provenance 的既有 trace 行为不变；schema 校验对 real-trace
   豁免"GEMM_START/GEMM_END 首尾"教学完整性检查（真实 trace 不遵循该结构）。
3. **解析层独立于仿真层**：`core/realtrace/` 不 import 任何 Simulation 引擎
   （仅复用 `eventBuilder` 编号基础设施），不 import React。
   Simulation Mode 与 Real Trace Mode 走同一条 TVIR → Playback → UI 管线，
   符合手册"两个模式使用同一个 UI"的要求。
4. **诚实投影，不臆测**：profiler 不采集 Block 内部细节，parser 只生成
   KERNEL_LAUNCH 与 BLOCK_SCHEDULE 事件，**不生成** MMA/MEMORY 类事件。
   kernel 时间轴数据写入 `metadata.kernelInfo`（startNs/durationNs/grid/block），
   由 `projectKernelTimeline()`（纯投影）提取给 KernelTimeline 视图。
5. **数据可信度强制标注**（手册 §24）：trace 级 provenance + isSample 驱动
   页脚与 KernelTimeline 的标注——仿真=Simulated、真实=Measured、
   内置示例=示例数据（教学示意值，绝不标 Measured）。
   `realTrace.test.ts` 含防回归用例：示例数据描述不得出现 "Measured 数据"。
6. **硬件参数推导**：Real Trace 模式的 SM 数与 warps/Block 由
   `inferRealTraceHardware()` 从 trace 推导（smCount、Block 线程数÷32），
   不依赖仿真配置面板。
7. **导入格式边界**：当前接受的是 Nsight 风格的 JSON 教学子集
   （kernels[].name/startNs/durationNs/grid/block/operator + meta），
   不是 .nsys-rep/.sqlite 二进制格式；格式说明见 `samples/attention-block.sample.json`。
   V0.6 起 kernel 可携带可选 `metrics`（Nsight Compute 风格指标，缺失时 UI 显示 N/A）。

## V0.6 架构约定（Performance Analysis）

1. **TVIR 基础架构仍零改动**：V0.6 没有新增事件类型。性能指标来自对既有字段的
   纯投影——仿真模式读 `metadata.gemm`/`rows/cols` 走屋顶线教学模型，
   真实 trace 读 `metadata.kernelInfo`（含可选 `metrics`）。
2. **perf 层是纯投影，不是执行引擎**：`core/perf/metrics.ts` 的
   `computePerfReport(trace, event)` 返回六项指标（Kernel duration / Tensor Core
   utilization / Memory bandwidth / L2 hit rate / Occupancy / Arithmetic intensity），
   每项强制携带 `source: measured | estimated | simulated | unavailable`。
   该层不 import React，不生成事件，不进入播放序列。
3. **数据可信度逐项落地**（手册 §24）：
   - 仿真模式：duration/tcUtil/bandwidth/l2Hit/occupancy = **Simulated**，
     算术强度 = **Estimated**（由形状推导）；面板 dataClass = simulated。
   - 真实 trace（非示例）：kernel 时长与其携带的 metrics = **Measured**；
     trace 未提供的指标 = **N/A**，绝不用估算填充。
   - 内置示例 trace：dataClass = **sample**，所有数值按"示例数据（教学示意值）"
     标注，source 用 simulated，**绝不出现 Measured**。
   - PerfPanel 原样渲染 source 徽标；`perfMetrics.test.ts` 含防回归用例
     （示例 trace 的 duration 不得为 measured）。
4. **假设硬件参数集中管理**：屋顶线模型使用 `ASSUMED_HARDWARE` 教学常量
   （峰值算力、带宽、SMEM 容量、每 SM warp 上限、fp16 字节数），
   不代表真实 GPU；V0.7 Architecture Playground 将把这些参数开放给用户修改。
5. **PerfPanel 与数据源共用**：性能面板在 Simulation / Real Trace 两种模式下
   渲染同一组件，仅标注不同——再次验证"数据源可替换、UI 不改动"。

## V0.7 架构约定（Architecture Playground）

1. **TVIR 基础架构仍零改动**：V0.7 没有新增事件类型。Playground 直接复用 V0.6 的
   屋顶线建模函数（`modelGemm` / `modelElementwise`），只是把它们暴露为可接收
   任意 `HardwareSpec` 的纯函数——同一模型，参数不同而已。
2. **硬件参数化向后兼容**：`computePerfReport(trace, event, hardware?)` 的第三个
   参数可选，省略时用 `DEFAULT_HARDWARE_SPEC`，V0.6 的所有调用与测试无需改动。
   `ASSUMED_HARDWARE` 保留为兼容别名。
3. **硬件参数只影响仿真模式**：`hardware` 参数对真实 trace（Measured 数据）
   完全无效——修改硬件滑块不会"改变"实测时长。Playground 面板只在仿真
   数据源（GEMM / Attention / Transformer Block）下渲染。
4. **Playground 是纯投影层**：`core/perf/playground.ts` 的 `analyzePlayground()`
   从 trace 提取工作负载清单（`extractWorkloads`，读 `metadata.gemm`/`rows/cols`），
   用基线/修改后的 HardwareSpec 分别重算屋顶线，输出加速比、瓶颈判定与教学解读。
   不 import React，不生成事件，不修改 trace。
5. **SM 语义的物理正确性**（CONCEPTS.md 规则 35）：每 SM 峰值算力固定
   （= 滑块算力 ÷ 基线 SM 数），实际算力 = 忙碌 SM 数 × 每 SM 峰值，
   忙碌 SM 数 = min(SM 数, Block 数)。测试用例保证：加 SM 不会让任何负载变慢
   （并行扩展单调）。该语义在 V0.7 开发中修正过一次（原"并行效率"公式有物理缺陷）。
6. **UI 状态提升**：硬件规格状态在 App 中管理，同时驱动 PerfPanel（当前算子指标
   随硬件实时重算）与 Playground 面板（基线 vs 修改后对比）——两个面板共享
   同一份 HardwareSpec，用户拖滑块时两处同步更新。

## V0.8 架构约定（SASS Trace / Accel-Sim）

1. **TVIR 12 种事件类型保持唯一词汇表**：V0.8 没有新增事件类型。SASS 指令通过
   显式映射表投影到既有类型（LDG→MEMORY_LOAD、STG→MEMORY_STORE、LDS→MEMORY_LOAD、
   STS→MEMORY_STORE、LDGSTS→MEMORY_MOVE、HMMA→MMA、FFMA→ACCUMULATE、BAR→SYNC、
   其余→WARP_SCHEDULE）。映射表由 CONCEPTS.md 规则 41 锁定，
   `sassTrace.test.ts` 的映射用例防回归。
2. **架构验收点再次兑现**（手册 §9/§26）：指令级数据源同样只替换 Source。
   `parseSassTrace()` 输出合法 `TVIRTrace`，Playback 与 UI 零改动；
   InstructionView 与 KernelTimeline 一样只消费纯投影
   （`projectSassInstructions()`），不理解指令语义。
3. **适配层独立**：`core/sasstrace/` 不 import Simulation 业务逻辑与 React
   （仅复用 `eventBuilder` 编号基础设施）。指令的教学文案（what/why）在 Adapter
   内按类别生成，与 V0.5 realtrace 的文案生成方式一致。
4. **定位声明是硬性要求**（手册 §26）：Educational Simulation ≠ Architecture
   Simulation。Adapter 不做 cycle-accurate 仿真；InstructionView 顶部与 trace
   描述常驻该声明。sass-trace 的 provenance 复用 V0.5 的 `'real-trace'` 取值
   （页脚标注按 `source === 'sass-trace'` 单独分支，标"NVBit SASS 指令流
   （Educational Simulation，非 cycle-accurate）"，不标 Measured）。
5. **硬件推导聚焦单 SM**：`inferSassTraceHardware()` 从 KERNEL_LAUNCH 的
   `metadata.sassKernel` 推导 numSM（采样 SM+1）与 warps/Block（线程数÷32），
   教学视图聚焦 SM 内部，不代表真实占用。
6. **导入格式边界**：接受 NVBit/cuobjdump 风格 JSON（kernel + warps[].instructions[]，
   每条指令 pc/opcode/operands），格式参考 `samples/sass-gemm.sample.json`；
   不是 NVBit 原始二进制导出。
7. **未登记指令不臆测**：操作码分类基于基础码精确匹配（LDGSTS 优先于 LDG），
   未登记指令归入 address-calc，延续规则 29"不硬归类"原则。

## V0.9 架构约定（Multi-GPU / Chakra+ASTRA-sim 思想）

1. **TVIR 12 种事件类型保持唯一词汇表**：V0.9 没有新增事件类型。集合通信通过
   既有 `MEMORY_MOVE` 事件 + `metadata.comm` 表达（含 collective / phase /
   ringStep / transfers / bytesPerTransfer / durationUs 字段）；计算阶段复用
   `emitGemmEvents`；AllReduce 起始用 `SYNC` 事件标记。该约定由
   CONCEPTS.md 规则 46-47 与 `multiGpu.test.ts` 共同锁定。
2. **架构验收点继续兑现**（手册 §9）：Multi-GPU 是新增的一个仿真数据源，
   `simulateMultiGpu()` 输出合法 `TVIRTrace`，Playback 与既有 UI 零改动；
   MultiGpuView 与 GpuView 一样只消费当前 TVIR 事件（读 `metadata.comm` /
   `metadata.multigpu`），不理解策略语义。
3. **multigpu 模块独立**：`core/multigpu/` 不 import React，不 import
   Attention/Block 等业务引擎；`commPrimitives.ts`（集合通信原语）与
   `multiGpuEngine.ts`（DP/TP/PP 策略编排）分层，原语可被未来策略复用。
4. **解析式通信建模，不模拟网络**（借鉴 Chakra + ASTRA-sim，手册 §27）：
   通信耗时用 alpha-beta 模型估算（`ringStepDurationUs`），呈现通信量与耗时；
   不实现路由、拥塞、逐跳传输。所有通信耗时标注 Simulated，页脚按
   `provenance === 'simulation'` 标注。
5. **六种策略一个引擎入口**：`simulateMultiGpu(config)` 按 `config.strategy` 分派——
   三种计算编排（`emitDataParallel` / `emitTensorParallel` / `emitPipelineParallel`）
   与三种集合通信原语独立演示（`emitAllReduceDemo` / `emitAllGatherDemo` /
   `emitReduceScatterDemo`，对应 `comm-allreduce` / `comm-allgather` /
   `comm-reducescatter`）。通信演示复用 `commPrimitives` 原语，不编排模型计算，
   聚焦通信基元本身（手册 §27 要求展示 AllReduce/AllGather/ReduceScatter）。
   新增策略时只加分派分支与对应 emit 函数，复用既有原语；
   `isCollectiveDemo()` 辅助识别通信演示策略。
6. **UI 按策略切换视图**：`source === 'multigpu'` 时主视图区渲染
   `MultiGpuView`（环形拓扑 + 传输箭头），其余模式仍渲染 `GpuView`；
   配置面板提供策略选择与 GPU 数/SeqLen/d_model 调节。

## V1.0 架构约定（六层联动 · Model 层，手册 §28）

1. **TVIR 12 种事件类型保持唯一词汇表**：V1.0 没有新增事件类型。Model 层
   是既有 TVIR 事件的**纯投影消费者**——与 compiler/operatorKnowledge 同一
   模式：静态领域知识（模型结构树）+ 查找（operator → 模型节点），不产生、
   不改写任何 TVIR 事件。该约定由 CONCEPTS.md 规则 51-54 与
   `modelStructure.test.ts` 共同锁定。
2. **core/model 模块独立**：`core/model/`（`modelStructure.ts`）不 import
   React、不 import 任何仿真引擎；只依赖 `tvir/types` 的 `TVIREvent` 类型。
   对外导出 `TRANSFORMER_BLOCK_MODEL`（结构树）、`findActivePath()`、
   `projectActiveModelPath()`、`isBlockModelEvent()` 四个纯函数/常量。
3. **operator 精确匹配契约**：结构树节点的 `operators` 列表与
   transformerBlockEngine / attentionEngine 生成的 operator 字符串逐字符一致，
   命中即激活、未命中即不显示。修改引擎的 operator 命名时必须同步更新
   `TRANSFORMER_BLOCK_MODEL`，否则 Model 层将静默失配（由测试守护）。
4. **ModelView 是当前事件的纯函数**：组件只接收 `event: TVIREvent | null`，
   激活路径、节点展开/折叠全部由 `projectActiveModelPath()` 投影决定，
   组件无内部状态。仅 `source === 'block'` 时在左栏渲染；其余数据源下
   Model 层不显示（事件不属于 Block 模型）。
5. **六层联动的构成**（手册 §28）：Model（ModelView）↔ Operator/Math
   （CompilerView）↔ GPU（GpuView/TensorCoreView）↔ Kernel（编译链 Kernel 层）
   ↔ Memory（MemoryView）+ Timeline + What/Why（EventExplanation）。六个层面
   由同一 Playback 游标驱动，共享当前 TVIR 事件，层间无直接依赖。

## 数据可信度规则（来自 §24）

UI 中必须区分并标注三类数据来源，绝不混用：
- **Measured** — 真实测量（V0.5 起：用户上传的非 sample 真实 trace）
- **Estimated** — 估算
- **Simulated** — 教学仿真引擎产生的全部数据（含内置示例 trace 的数值）
