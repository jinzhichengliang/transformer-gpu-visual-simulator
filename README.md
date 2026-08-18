# Transformer GPU Visual Simulator

> 🖥️ **在线体验（无需安装，浏览器直接打开）**：<https://jinzhichengliang.github.io/transformer-gpu-visual-simulator/>

**把 Transformer 的每一步计算，放到 GPU 的视角下看清。**

一个面向教学的交互式可视化工具：从模型算子、数学运算，一路追踪到 GPU 的 Kernel、内存与 Tensor Core 行为，帮助初学者建立"模型计算 → GPU 执行"的完整心智模型。

> **Educational simulation, not cycle-accurate.**
> 本项目是教育仿真工具，所有数值均为 **Simulated**（仿真推导），不声称任何性能测量或架构级精度。

## 当前版本

**V1.0 · 六层联动执行显微镜（Six-Layer Coordination Microscope）**

播放任意数据源时，六个层面随同一时间轴联动点亮：

```
Model（模型结构树）  ↔  GPU（SM / Warp / Tensor Core）
Operator / Math（算子与数学） ↔  Kernel（编译链与指令）
              + Timeline（时间轴） + What/Why（每步解释）
```

## 功能特性

### 六种数据源
| 数据源 | 内容 |
| --- | --- |
| **GEMM** | 矩阵乘法从数学到 Tiled Kernel、Shared Memory、Tensor Core MMA 的完整链路 |
| **Attention** | Q/K/V 投影、QKᵀ、Causal Mask、Softmax、AV、Output Projection 逐算子展开 |
| **Transformer Block** | Pre-Norm 完整 Block：RMSNorm → Attention → Residual → RMSNorm → FFN → Residual |
| **Multi-GPU** | TP / PP / DP 三种并行策略 + AllReduce / AllGather / ReduceScatter 集合通信（Ring 算法） |
| **Real Trace** | 导入 Nsight 风格的真实 trace（JSON），同一套可视化直接消费 |
| **SASS Trace** | 指令级 trace（SASS 风格）适配为统一事件流，展示指令调度视角 |

### 核心视图
- **ModelView** — Transformer Block 结构树，随执行逐步点亮激活路径（V1.0）
- **MatrixView** — 矩阵运算的数学层可视化
- **GpuView / TensorCoreView** — SM、Warp、Shared Memory、寄存器与 MMA 单元
- **CompilerView** — 算子 → Kernel 的编译链投影
- **MemoryView** — HBM / Shared Memory 数据搬运
- **Timeline / KernelTimeline** — 事件时间轴
- **EventExplanation** — 每一步的 What / Why 解释
- **PerfPanel** — 基于可配置硬件参数的性能指标推导（明确标注 Simulated）
- **ArchitecturePlayground** — 调整硬件架构参数，对比对 GEMM 执行的影响

### 架构原则（TVIR）
仿真与可视化之间只有唯一数据接口 —— **TVIR（Transformer Visual Intermediate Representation）**：

```
Simulation Engine ──► TVIR events ──► Playback Engine ──► Visualization (UI)
```

- TVIR 只含 12 种事件类型，是稳定契约；新增视图只是新的投影，不改动接口
- 前端永远不知道事件来自哪里（今天可以是教学仿真器，明天可以是真实 trace 解析器）
- 核心模块不依赖 React；组件只消费当前事件，不做仿真、不决定下一步

详见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) 与 [`CONCEPTS.md`](./CONCEPTS.md)（GPU 概念正确性守护清单）。

## 快速开始

```bash
npm install        # 安装依赖
npm run dev        # 启动开发服务器（默认 http://localhost:5173/）
npm test           # 运行全部测试（Vitest，132 tests）
npm run build      # 生产构建
npm run lint       # Oxlint 检查
```

技术栈：React 19 · TypeScript · Vite · Vitest。无后端，纯前端运行。

## 项目定位与边界

- ✅ 教学用途：解释 Transformer 算子如何映射到 GPU 执行机制（GEMM、Tiling、Warp、Tensor Core、集合通信）
- ✅ 所有教学简化均在 `CONCEPTS.md` 中显式声明，不误导初学者心智模型
- ❌ 不是性能模拟器：不承诺 cycle 精度，不替代 profiler 实测数据
- ❌ 不模拟真实硬件时序细节（占用率、bank conflict 等仅做概念性呈现）

## License

MIT
