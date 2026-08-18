# CONCEPTS.md — GPU Concepts That MUST Remain Correct

> 来源：实施手册 §29-30。每次修改核心模块前必须检查本文件。
> 本文件列出所有不允许在可视化中被歪曲的 GPU 概念。
> 教学简化是允许的，但必须在文档中显式声明为"教学简化"，
> 且不得让初学者形成错误的心智模型。

## 硬性正确性规则（MUST）

1. **Block is scheduled to an SM.**
   Thread Block 由 GPU 调度器分配到某一个 SM 上执行；一个 Block 只在一个 SM 上运行，不会跨 SM 拆分。

2. **Warp is the scheduling/execution grouping within a block.**
   Warp（32 线程）是 SM 内实际的调度与执行单位；Block 内的线程以 Warp 为单位被发射执行。

3. **Shared Memory belongs to an SM and is shared within a block's threads.**
   Shared Memory 是 SM 上的片上资源，同一个 Block 内的线程共享；不同 Block 之间不可见。

4. **Registers are thread-local architectural state.**
   寄存器是每个线程私有的，是线程级别的架构状态，不在线程间共享。

5. **Tensor Core executes matrix multiply-accumulate operations.**
   Tensor Core 执行矩阵乘累加（MMA，D = A × B + C），操作数来自寄存器（fragment），而不是直接从显存取数。

6. **Tiling improves data reuse.**
   Tiling 的意义是提高数据复用率、降低对慢速全局内存（HBM）的重复访问，是 GEMM 优化的核心手段。

7. **Educational visualization must not imply cycle accuracy.**
   本工具是教学仿真，任何动画与步骤都不得暗示 cycle 级精确性。UI 中必须明确标注 "Educational simulation, not cycle-accurate"。

8. **Simulated metrics must never be labeled measured.**
   所有仿真产生的指标只能标注为 Simulated/Estimated，绝不能标注为 Measured。UI 中必须区分：Measured（真实测量）/ Estimated（估算）/ Simulated（仿真）。

## Attention 概念规则（V0.2，MUST）

9. **Attention 计算顺序固定**：Q/K/V Projection → QKᵀ → Scale → Mask → Softmax → ×V → Output Projection。
   不得颠倒或跳过步骤（除非显式说明是 FlashAttention 式融合优化）。

10. **Scaled Dot-Product**：分数除以 √d 是为了控制点积方差、防止 Softmax 饱和，
    这是 "Scaled" Dot-Product Attention 的定义性步骤，不能省略其解释。

11. **Causal mask 的方向**：causal（自回归）mask 屏蔽的是 j > i 的未来位置（上三角），
    置 −∞ 后 Softmax 权重为 0；不得画反方向。

12. **Softmax 按行归一化**：注意力 Softmax 对每一行（每个 query token）独立归一化，
    每行权重和为 1；不是对整个矩阵、也不是按列。

13. **Softmax 的数值稳定实现**：教学上应体现"先减行内最大值再 exp"的安全 Softmax，
    并说明这是防溢出，而非可选优化。

14. **Scale/Mask/Softmax 不是 Tensor Core 工作负载**：它们是逐元素/行归约 kernel，
    由 CUDA Core 完成，访存密集；可视化中不得出现 MMA 事件。

15. **GEMM 才是 Attention 的算力主体**：Q/K/V Projection、QK、AV、Output Projection
    都是 GEMM（Tensor Core 密集），这个"两类 kernel 分工"是 V0.2 的核心教学点。

## Transformer Block 概念规则（V0.3，MUST）

16. **Pre-Norm 结构**：本工具模拟的是 Pre-Norm（先归一化再进子层）：
    RMSNorm → Attention → Residual → RMSNorm → FFN → Residual。
    Attention 与 FFN 的输入都是归一化后的张量（Xn），不是原始主干 X；
    可视化与文案中不得画成 Post-Norm（先子层后归一化）。

17. **残差连接是逐元素相加**：Residual 是 X = X + Branch 的逐元素加法，
    由 CUDA Core 完成，不是 GEMM、不产生 MMA 事件。两个残差连接分别位于
    Attention 之后与 FFN 之后，顺序不得颠倒。

18. **RMSNorm 按行（每个 token）归一化**：对每个 token 的隐藏向量求均方根并归一化，
    再乘可学习缩放参数 γ；与 LayerNorm 的区别是不减均值、不做平移（无 β）。
    它是行归约 kernel（平方和归约 + 逐元素缩放），不使用 Tensor Core。

19. **FFN 是逐 token 独立的两层 GEMM 夹非线性激活**：
    Up Projection（d→ffnDim）→ 激活 → Down Projection（ffnDim→d）。
    FFN 不在 token 之间交换信息（那是 Attention 的职责）；
    升维倍数（通常 4×）与"FFN 是参数量最大的部分"这两个事实应体现于教学文案。

20. **Block 输入输出形状不变**：一个 Block 的输出形状与输入完全相同（seq×d_model），
    这是 Block 可以任意堆叠的前提，教学总结中必须点明。

21. **激活函数是非线性来源**：SiLU/GELU 等激活是逐元素操作；
    不得让学习者误以为两次 GEMM 之间没有非线性（否则整体仍是线性变换）。

## Compiler View 概念规则（V0.4，MUST）

22. **IR 是教学示意，不是真实编译器输出**：展示的 LOAD/MUL/ACC 等指令序列是
    为教学设计的"伪 IR"，不得声称是 PTX/SASS 或任何真实编译器的输出。
    V0.8 接入真实 SASS trace 时必须明确区分。

23. **算子类型与硬件单元的对应必须正确**：MatMul 类算子映射 Tensor Core，
    逐元素/行归约类算子映射 CUDA Core；不得把 Softmax/RMSNorm/Residual/SiLU
    标注为 Tensor Core 执行（与规则 14/18/17 一致）。

24. **瓶颈特征必须区分计算密集与访存密集**：GEMM 类为 compute-bound，
    逐元素/归约类为 memory-bound；这是 roofline 思维（V0.7）的前置概念，
    不得在 V0.4 中给出错误归类。

25. **kernel 融合是优化而非默认**：Compiler View 中提及 FlashAttention、
    fused add+RMSNorm 等融合技术时，必须说明它们是"真实实现的优化手段"，
    本仿真仍按独立 kernel 逐个展示，避免学习者以为融合是唯一正确形态。

## Real Trace 概念规则（V0.5，MUST）

26. **数据来源三分法必须显式标注**：每条 trace 的 provenance 决定标注——
    simulation=Simulated（教学仿真）、real-trace=Measured（真实 GPU 实测）、
    内置示例 trace=示例数据（教学示意值）。三者绝不混标；
    示例数据在任何 UI 位置都不得标成 Measured（手册 §24 数据可信度规则）。

27. **真实 trace 不臆测 Block 内部细节**：profiler 只记录 kernel 级时间轴与
    grid/block 配置，因此 Real Trace 模式不生成 MMA/MEMORY 类事件。
    不得为了"看起来完整"而虚构真实 trace 中不存在的访存或 Tensor Core 事件。

28. **真实 kernel 时间轴是 Measured 数据**：KernelTimeline 的横轴（µs）与
    kernel 条长度来自 profiler 实测，可如实比较 kernel 间相对耗时；
    但不得外推出本工具未采集的指标（如利用率、带宽），那是 V0.6 的话题。

29. **kernel 名 ≠ 算子语义**：真实 kernel 名（如 cutlass_tensorop_gemm_*）
    是编译器/库的命名，operator 字段是可选的人工标注；未标注时
    Compiler View 走回退配方，不得把未知 kernel 硬归类为某个算子。

## Performance 概念规则（V0.6，MUST）

30. **六项指标的数据来源必须逐项标注**：Kernel duration / Tensor Core
    utilization / Memory bandwidth / L2 hit rate / Occupancy / Arithmetic
    intensity 每项都必须携带 Measured / Estimated / Simulated / N-A 标注；
    UI 不得把不同来源的指标混在同一口径下比较（手册 §24）。

31. **仿真模式的性能数字是屋顶线估算**：仿真没有真实时序，所有性能数值由
    "教学屋顶线模型 + 假设硬件参数"推导（max(FLOPs/峰值算力, 访存量/带宽)），
    只能标注 Simulated；其中算术强度由形状推导，标注 Estimated。
    这些数字用于理解"计算密集 vs 访存密集"，不代表真实性能。

32. **真实 trace 的指标以采集范围为准**：Nsight Systems 类时间轴 trace 只提供
    kernel 时长（Measured）；Tensor Core 利用率、L2 命中率、带宽、occupancy、
    算术强度通常需要 Nsight Compute 采集。trace 未提供的指标必须如实显示 N/A，
    绝不用估算值填充成 Measured。

33. **Occupancy 的教学模型必须声明局限**：仿真模式 occupancy 只计 warp 槽位与
    共享内存两个约束，未计寄存器限制；文案中必须说明，避免给出"这就是真实
    occupancy"的错觉。

34. **假设硬件参数是教学常量**：`DEFAULT_HARDWARE_SPEC`（SM 数、峰值算力、带宽、
    L2/SMEM 容量、每 SM warp 上限、fp16 字节数）不代表任何真实 GPU；
    V0.7 Architecture Playground 允许用户修改这些参数并观察影响。

## Architecture Playground 概念规则（V0.7，MUST）

35. **SM 数量的并行语义必须物理合理**：每 SM 峰值算力固定（= 滑块算力 ÷ 基线
    SM 数），实际算力 = 忙碌 SM 数 × 每 SM 峰值，忙碌 SM 数 = min(SM 数, Block 数)。
    加 SM 只能让 Block 充足的大负载更快（或不变），**绝不能让任何负载变慢**；
    Block 数不足时收益饱和——这是"并行度收益递减"的正确表达。

36. **HBM 带宽是机器级资源**：带宽滑块改变的是整台机器的访存上限，对所有算子
    共用；带宽变化只显著影响访存密集算子，计算密集算子几乎不变——这正是
    Roofline 思维的核心，也是手册 §25"为什么有些 Operator 几乎没变"的答案。

37. **Tensor Core 算力只影响计算密集算子**：算力滑块（每 SM 峰值）加速 GEMM 类
    算子，但逐元素/归约算子走 CUDA Core 路径，不受 Tensor Core 算力影响；
    不得让访存密集算子"沾光"加速。

38. **L2/SMEM 的影响路径必须区分**：L2 容量影响"Tile 复用折扣"（有效访存量与
    命中率）；SMEM 容量影响"每 SM 可驻留 Block 数"（occupancy 与并发度）。
    两者作用机制不同，不得混为一谈；对已复用充分的 GEMM，L2 变化的影响应很小。

39. **Playground 的全部数值是 Simulated**：对比视图、加速比、瓶颈判定均为教学
    屋顶线估算，标注 Simulated；绝不得暗示这是对某款真实 GPU 的性能预测。
    教学解读（takeaways）必须回答手册 §25 的三个问题：带宽翻倍快多少、
    算力翻倍快多少、为什么有些算子几乎不变。

## SASS Trace 概念规则（V0.8，MUST）

40. **Educational Simulation ≠ Architecture Simulation**：V0.8 接入指令级数据，
    但 Adapter 只做"指令顺序与数据通路"的教学呈现，不重新实现 Accel-Sim 的
    cycle-accurate 微架构仿真。InstructionView 与 trace 描述必须常驻该声明，
    不得暗示本工具能给出指令级周期、流水线或 occupancy 的真实数值。

41. **TVIR 12 种事件类型保持唯一词汇表**：SASS 指令通过显式映射表投影到既有
    事件类型，不新增类型。映射表（Adapter 实现与测试共同锁定）：
    LDG→MEMORY_LOAD(HBM→REGISTER)、STG→MEMORY_STORE(REGISTER→HBM)、
    LDS/LDSM→MEMORY_LOAD(SHARED_MEMORY→REGISTER)、
    STS→MEMORY_STORE(REGISTER→SHARED_MEMORY)、
    LDGSTS→MEMORY_MOVE(HBM→SHARED_MEMORY)、HMMA/QMMA/IMMA/BMMA/DMMA→MMA、
    FFMA/FADD/FMUL/HFMA2 等→ACCUMULATE、BAR→SYNC、
    其余（地址计算/控制流）→WARP_SCHEDULE（文档化的教学映射）。

42. **指令分类基于基础操作码精确匹配**：分类取操作码第一个 '.' 之前的基础码
    大写匹配；LDGSTS 必须先于 LDG 规则匹配（否则 cp.async 会被误归入全局内存
    加载）。未登记的指令归入 address-calc，不得臆测其语义——这是规则 29
    "不硬归类"在指令级的延续。

43. **LDGSTS（cp.async）绕过寄存器**：LDGSTS 的数据通路是 HBM→SHARED_MEMORY，
    不经过寄存器，不得画成 MEMORY_LOAD（那是 LDG 的语义）；文案须点明
    它是 Ampere 起支持、用于隐藏访存延迟的异步拷贝。

44. **示例指令流是教学示意编排**：`sampleSassTrace.ts` 与
    `samples/sass-gemm.sample.json` 的指令序列是为教学编排的骨架，不对应任何
    真实采集；标注"示例指令流（教学示意编排）"，绝不声称是 NVBit 实测。

## Multi-GPU 概念规则（V0.9，MUST）

45. **三种并行策略的语义不得混淆**：
    - DP（数据并行）：每 GPU 持有完整模型副本，处理不同 micro-batch，
      反向后对梯度做 AllReduce——通信集中在反向阶段；
    - TP（张量并行）：单个 GEMM 沿张量维度切分，列并行输出分片独立、
      行并行输出部分和需 AllReduce 汇总——通信穿插在每层计算中；
    - PP（流水线并行）：不同层放在不同 GPU，micro-batch 沿阶段流动，
      阶段间用 P2P 传激活/梯度——通信是相邻阶段的点对点。
    视图与文案不得把三者画成同一种通信模式。

46. **AllReduce = ReduceScatter + AllGather（Ring 算法）**：本工具按 NCCL 的
    Ring 实现呈现 AllReduce——前半段 ReduceScatter（每 GPU 归约出 1/N 完整
    chunk）、后半段 AllGather（把完整 chunk 广播给所有人）。总通信量
    2(N-1)/N × S，与 GPU 数几乎无关；不得把 AllReduce 画成"所有 GPU 直接把
    全量数据发给一个中心节点"的星型模式。

47. **环算法每步的传输模式**：Ring 的第 k 步，每个 GPU 同时向下一个邻居发送
    一个 chunk（N 次并行传输）；ReduceScatter 与 AllGather 各 N-1 步。
    传输方向必须沿同一环向（不得出现双向对冲），chunk 编号随步数轮转。

48. **TP 的列/行并行配对**：列并行（如 QKV 投影）各 GPU 输出列独立、无需通信；
    行并行（如 Output Projection）各 GPU 输出部分和、需 AllReduce 汇总。
    两者配对使用把通信压缩到每层一次——不得只画其中一半。

49. **PP 的 bubble 与 micro-batch**：流水线的 GPU 空闲（bubble）比例约为
    (阶段数-1)/micro-batch 数；micro-batch 越多填得越满。文案须点明这是
    GPipe 式调度的核心权衡，不得呈现"流水线完全没有空闲"的错误印象。

50. **解析式通信建模，不模拟网络**（借鉴 Chakra + ASTRA-sim，手册 §27）：
    通信耗时用 alpha-beta 模型估算（latency + bytes/bandwidth），呈现通信量
    与耗时估算；**不模拟**路由、拥塞、逐跳传输等网络细节。所有通信耗时标注
    Simulated，不得声称是实测网络性能。

## Model 层概念规则（V1.0 六层联动，MUST）

51. **Model 层展示 Pre-Norm 结构，顺序不得更改**：结构树必须呈现
    RMSNorm → Attention → Residual → RMSNorm → FFN → Residual 的数据流顺序
    （Pre-Norm，归一化在子层之前）。Attention 下钻为 Q/K/V Projection →
    QKᵀ → Scale → Mask → Softmax → AV → Output Projection，FFN 下钻为
    Up Projection → SiLU → Down Projection。子算子的排列顺序必须与 V0.3
    Block 引擎的实际计算顺序一致，不得重排。

52. **激活路径是纯投影，只读 operator**：Model 层通过当前 TVIR 事件的
    `operator` 字段确定激活节点，高亮"Block 根 → 激活子层"的路径。
    Model 层不理解仿真细节、不读 metadata、不判断事件类型——这与
    compiler/operatorKnowledge 是同一模式（静态领域知识 + 查找）。

53. **operator 字符串精确匹配，禁止模糊归类**：模型树节点的 `operators`
    列表必须与 transformerBlockEngine / attentionEngine 生成的 operator
    字符串**逐字符一致**。命中即激活、未命中即不显示——不得做大小写折叠、
    子串包含或语义猜测匹配。

54. **展开/折叠是投影结果，不是交互状态**：Attention / FFN 父节点仅当
    激活路径命中自身时展开其子算子；组件不得持有内部展开状态或点击切换。
    这保证 Model 层是"当前事件的纯函数"，与其余视图一致。

## 允许的教学简化（可接受，但需声明）

- 内存层级以 HBM → L2 → L1 → Shared Memory → Register → Tensor Core 的层级视图呈现。
  **声明**：这是教学简化视图。真实 GPU 中 cache、shared memory 与显式数据搬运的关系远比单一路径复杂
  （例如 L1 与 Shared Memory 共享同一块片上 SRAM、存在 bypass/coalescing 等机制）。
  MemoryView 中不得暗示"所有访问必然严格沿这条路径逐级进行"。
- 为控制复杂度，V0.1 默认 4 个 SM、固定 warp 数，不代表真实 GPU 的配置。
- 不模拟真实 CUDA cycle 数与时序，步骤编号只表示教学上的先后顺序。
- **（V0.2）单头注意力**：当前只模拟单头 Attention，未展示多头拼接与并行；
  Output Projection 的 why 中已说明多头场景。
- **（V0.2）逐元素 kernel 教学抽样**：Scale/Mask/Softmax 每个 Block 负责一行，
  但只详细展示前 2 个 Block 的完整流程，其余 Block 用一条汇总事件概括。
  **声明**：这是为控制演示长度的教学简化，未展示的 Block 执行过程与展示的完全相同，
  不暗示它们以不同方式执行或不执行。
- **（V0.2）−∞ 的实现**：causal mask 中 −∞ 以"极大负数（如 -1e9）"实现，
  文案中已说明，不得让学习者误以为存在真正的负无穷浮点数。
- **（V0.3）FFN 激活用 SiLU 示意**：真实模型（如 LLaMA/Qwen）多用 SwiGLU（带额外 gate 分支），
  本仿真以 SiLU 示意激活环节，文案中已说明，不误导为所有模型都用 SiLU。
- **（V0.3）单层 Block**：当前只模拟一个 Block，不展示层间堆叠与参数变化；
  多层堆叠（及 KV Cache 等推理特性）是后续版本的话题。
- **（V0.3）算子融合未体现**：真实实现常把 Residual+RMSNorm 等融合为一个 kernel，
  本仿真按逻辑算子逐个展示以便教学，文案中已提示融合优化。
- **（V0.4）伪 IR 教学示意**：Compiler View 展示的 LOAD/MUL/ACC 等指令序列是为教学
  设计的中间表示示意，不是真实编译器的 PTX/SASS 输出。
  **声明**：这用于传达"数学 → 指令序列 → kernel"的 lowering 思想；
  真实指令级执行见 V0.8（SASS trace）。
- **（V0.4）编译知识为静态查表**：算子到 IR/Kernel 的映射是手工登记的知识表，
  不做真实的图优化、tiling 决策或 autotuning；这是教学工具而非编译器。
- **（V0.5）内置示例 trace 的数值是教学示意值**：`sampleTrace.ts` 与
  `samples/attention-block.sample.json` 中的 kernel 时长、grid 配置均为示意，
  不对应任何真实 GPU；UI 已标注"示例数据（教学示意值，非实测）"。
- **（V0.5）导入格式为 JSON 教学子集**：接受的是 Nsight 风格 JSON
  （kernels 数组 + meta），不是 .nsys-rep/.sqlite 二进制导出；
  真实场景需先用 Nsight Systems 导出 JSON（如 nsys export --type json 后提取）。
- **（V0.5）Block 分发为均匀示意**：parser 把 grid 中的 Block 按 SM 均分展示
  （每 SM 一条 BLOCK_SCHEDULE 汇总事件），真实硬件的 Block 调度顺序与分布
  更复杂；文案中已说明这是调度规则的示意，不代表真实调度序列。
- **（V0.6）屋顶线教学模型**：仿真模式的性能数字由理想化屋顶线模型推导
  （最少访存量 = A、B 各读一次 + C 写一次；L2 命中按理想 Tile 复用；
  occupancy 只计 warp 槽位与 SMEM 两个约束），不模拟流水、冲突、调度开销。
  **声明**：这些是理解"计算密集 vs 访存密集"的教学估算，非真实性能。
- **（V0.6）假设硬件参数**：峰值算力、带宽等取 `ASSUMED_HARDWARE` 教学常量，
  不对应任何真实 GPU；V0.7 将开放用户修改。
- **（V0.6）示例 trace 的指标为示意值**：`sampleTrace.ts` 中部分 kernel 携带的
  metrics（利用率、命中率等）是教学示意值，UI 按"示例数据"标注，绝不标 Measured。
- **（V0.7）SM 并行扩展为忙碌 SM 模型**：每 SM 峰值算力固定（滑块算力 ÷ 基线 SM 数），
  实际算力 = 忙碌 SM 数 × 每 SM 峰值，忙碌 SM 数 = min(SM 数, Block 数)。
  **声明**：这是教学简化——真实 GPU 还有 warp 级调度、尾部效应、L2 争用等，
  此处只表达"Block 不足则加 SM 收益饱和"这一核心规律。
- **（V0.7）L2 复用折扣为理想模型**：L2 命中率按 Tile 复用 × L2 容量覆盖率折扣，
  不模拟真实 L2 的替换策略、bank 冲突与跨 SM 争用。
- **（V0.7）Playground 仅作用于仿真模式**：硬件参数只影响仿真模式的屋顶线估算，
  对真实 trace 的 Measured 数据无任何影响——修改硬件不会"改变"实测时长。
- **（V0.8）指令流不含时序**：SASS trace 只呈现指令顺序与数据通路，不含
  周期、延迟或流水线信息；InstructionView 已声明非 cycle-accurate。
  **声明**：教学重点是"指令在做什么、数据从哪来到哪去"，不是执行快慢。
- **（V0.8）单 kernel、单 SM 聚焦**：示例 SASS trace 聚焦一个 kernel 的
  1-2 个 Warp，不展示 grid 全部 Block 与跨 SM 分发；硬件推导 numSM=采样 SM+1，
  用于教学视图聚焦 SM 内部，不代表真实占用。
- **（V0.8）操作码覆盖为教学子集**：分类规则表只登记常见指令（LDG/STG/LDS/STS/
  LDGSTS/HMMA/QMMA/FFMA/BAR/BRA 等），未登记指令统一归入地址计算；
  真实 SASS 指令集远大于此，Adapter 不追求全覆盖。
- **（V0.8）导入格式为 JSON 教学子集**：接受 NVBit/cuobjdump 风格 JSON
  （kernel + warps 数组），不是 NVBit 原始二进制导出；真实场景需先转换格式。
- **（V0.9）通信耗时为解析式估算**：多 GPU 通信耗时用 alpha-beta 模型
  （latency + bytes/bandwidth）估算，呈现通信量与耗时；不模拟路由、拥塞、
  逐跳传输等网络细节。所有通信耗时标注 Simulated。
- **（V0.9）通信链路为教学常量**：默认链路带宽 300 GB/s、延迟 5 µs 为
  NVLink 量级的教学示意值，不代表任何真实互连。
- **（V0.9）每策略只展示一个代表性计算**：DP 每 GPU 只展示一个代表性 GEMM
  （实际是完整前向+反向）；TP 只展示一层的列/行并行对；PP 只展示前 2 个
  micro-batch 的完整流动，其余 micro-batch 用一条 SYNC 事件概括。
  **声明**：这些是为控制演示长度的教学抽样，未展开部分的执行模式与展示的相同。
- **（V0.9）拓扑为环形简化**：MultiGpuView 以环形拓扑呈现 GPU，表达 Ring
  AllReduce 的邻居关系；真实互连可能是 NVLink mesh / NVSwitch / PCIe 树，
  本工具不区分具体互连拓扑。
- **（V1.0）单 Block 结构树**：Model 层只展示一个 Transformer Block（Layer 0），
  未呈现多 Block 堆叠与 embedding/LM head。**声明**：这是教学简化——多层
  Transformer 是同一 Block 结构的重复堆叠，每层内部数据流与展示的完全相同。
- **（V1.0）Model 层不感知 KV Cache / 批处理**：结构树展示的是逻辑数据流，
  不包含 KV Cache、批维度、位置编码等推理/训练细节。

## 会造成错误理解的简化（禁止）

- ❌ 让 Tensor Core 直接从 HBM/L2 读取操作数（操作数必须经 Register fragment）。
- ❌ 暗示 Block 可以跨多个 SM 执行。
- ❌ 暗示不同 Block 共享同一块 Shared Memory。
- ❌ 把 Warp 画成与 Block 无关的独立实体（Warp 必须归属于某个 Block）。
- ❌ 将仿真耗时/带宽数字标注为实测值。
- ❌（V0.2）把 Softmax 画成对整个矩阵归一化。
- ❌（V0.2）把 causal mask 的方向画反（屏蔽过去而非未来）。
- ❌（V0.2）让 Scale/Mask/Softmax 出现 Tensor Core MMA。
- ❌（V0.3）把 Pre-Norm 画成 Post-Norm（子层之后才归一化）。
- ❌（V0.3）让 RMSNorm/Residual/SiLU 出现 Tensor Core MMA。
- ❌（V0.3）暗示 FFN 会在 token 之间交换信息。
- ❌（V0.5）把示例 trace 数据标注为 Measured。
- ❌（V0.5）为真实 trace 虚构 profiler 未采集的事件（MMA/访存细节）。
- ❌（V0.5）把 kernel 名直接等同于算子语义（未标注 operator 时走回退，不硬归类）。
- ❌（V0.6）把屋顶线估算值标注为 Measured，或用估算值填充 trace 缺失的指标。
- ❌（V0.6）把不同数据来源（Measured/Estimated/Simulated）的指标混在同一口径比较。
- ❌（V0.6）暗示假设硬件参数代表某款真实 GPU。
- ❌（V0.7）让加 SM 导致任何负载变慢（并行扩展必须单调不减收益）。
- ❌（V0.7）让 Tensor Core 算力滑块加速访存密集/逐元素算子。
- ❌（V0.7）把 Playground 的估算加速比表述为对真实 GPU 的性能预测。
- ❌（V0.7）让硬件参数修改影响真实 trace 的 Measured 数据。
- ❌（V0.8）把 SASS trace 视图暗示为 cycle-accurate 架构仿真（它是 Educational Simulation）。
- ❌（V0.8）为 SASS 指令虚构周期/延迟/流水线数值。
- ❌（V0.8）把 LDGSTS（cp.async）画成经寄存器的 MEMORY_LOAD（它绕过寄存器）。
- ❌（V0.8）把示例指令流标注为 NVBit 实测数据。
- ❌（V0.8）新增 TVIR 事件类型来表达 SASS 指令（必须用既有 12 种类型映射）。
- ❌（V0.9）把 AllReduce 画成星型模式（所有 GPU 把全量数据发给一个中心节点）；
  必须呈现 Ring 算法的 ReduceScatter + AllGather 两段。
- ❌（V0.9）混淆 DP/TP/PP 三种策略的通信模式（DP 梯度 AllReduce、
  TP 每层部分和 AllReduce、PP 阶段间 P2P）。
- ❌（V0.9）呈现"流水线完全没有 bubble"（bubble 比例约为 (阶段数-1)/micro-batch 数）。
- ❌（V0.9）把通信耗时估算标注为实测网络性能，或模拟路由/拥塞等网络细节
  （本工具是解析式建模，借鉴 ASTRA-sim 思想而非重新实现）。
- ❌（V0.9）新增 TVIR 事件类型来表达集合通信（必须用既有 MEMORY_MOVE/SYNC 类型 +
  metadata.comm）。
- ❌（V1.0）把 Pre-Norm 结构树画成 Post-Norm（归一化节点出现在子层之后）。
- ❌（V1.0）用模糊匹配（子串/大小写折叠/语义猜测）把算子归入模型节点；
  operator 必须精确匹配，未命中即不高亮。
- ❌（V1.0）让 Model 层读取 operator 以外的仿真细节（metadata、耗时、tile 等）
  来决定高亮——激活路径只能由 event.operator 投影。
- ❌（V1.0）在 ModelView 中引入点击展开/折叠等交互状态（展开是激活路径的
  纯投影结果，Model 层必须是当前事件的纯函数）。
