import { useEffect, useMemo, useRef, useState } from 'react';
import { usePlayback } from './state/usePlayback';
import {
  simulateGemm,
  DEFAULT_GEMM_CONFIG,
  type GemmConfig,
  simulateAttention,
  DEFAULT_ATTENTION_CONFIG,
  type AttentionConfig,
  simulateTransformerBlock,
  DEFAULT_TRANSFORMER_BLOCK_CONFIG,
  type TransformerBlockConfig,
} from './core/simulation';
import {
  EXAMPLE_TVIR_TRACE,
  validateTVIRTrace,
  projectSmStates,
  projectMatrixScene,
  projectOperatorSegments,
  projectKernelTimeline,
  projectSassInstructions,
} from './core/tvir';
import { projectCompileChain } from './core/compiler';
import {
  computePerfReport,
  analyzePlayground,
  specsDiffer,
  DEFAULT_HARDWARE_SPEC,
  type HardwareSpec,
} from './core/perf';
import {
  parseNsightTrace,
  inferRealTraceHardware,
  SAMPLE_REAL_TRACE,
  type NsightTraceFile,
} from './core/realtrace';
import {
  parseSassTrace,
  inferSassTraceHardware,
  SAMPLE_SASS_TRACE,
  type SassTraceFile,
} from './core/sasstrace';
import {
  simulateMultiGpu,
  DEFAULT_MULTI_GPU_CONFIG,
  type MultiGpuConfig,
  type ParallelStrategy,
} from './core/multigpu';
import { ControlBar } from './components/ControlBar/ControlBar';
import { MatrixView } from './components/MatrixView/MatrixView';
import { GpuView } from './components/GpuView/GpuView';
import { MemoryView } from './components/MemoryView/MemoryView';
import { TensorCoreView } from './components/TensorCoreView/TensorCoreView';
import { EventExplanation } from './components/EventExplanation/EventExplanation';
import { Timeline } from './components/Timeline/Timeline';
import { CompilerView } from './components/CompilerView/CompilerView';
import { KernelTimeline } from './components/KernelTimeline/KernelTimeline';
import { InstructionView } from './components/InstructionView/InstructionView';
import { PerfPanel } from './components/PerfPanel/PerfPanel';
import { ArchitecturePlayground } from './components/ArchitecturePlayground/ArchitecturePlayground';
import { MultiGpuView } from './components/MultiGpuView/MultiGpuView';
import { ModelView } from './components/ModelView/ModelView';
import './App.css';

type SourceMode =
  | 'gemm'
  | 'attention'
  | 'block'
  | 'real-trace'
  | 'sass-trace'
  | 'multigpu'
  | 'example';

const SIZE_OPTIONS = [64, 128, 256];
const TILE_OPTIONS = [16, 32, 64];
const SM_OPTIONS = [2, 4, 8];
const WARP_OPTIONS = [2, 4, 8];
const SEQ_OPTIONS = [32, 64, 128];
const DIM_OPTIONS = [32, 64, 128];
const FFN_OPTIONS = [128, 256, 512];

/** 把上传文件的原始文本安全解析为 JSON（失败返回中文错误） */
function parseTraceFileText(text: string): { file: NsightTraceFile } | { error: string } {
  try {
    return { file: JSON.parse(text) as NsightTraceFile };
  } catch {
    return { error: '文件不是合法的 JSON（请检查是否为 Nsight 风格 JSON 导出）' };
  }
}

/** 把上传文件的原始文本安全解析为 SASS trace JSON（失败返回中文错误） */
function parseSassFileText(text: string): { file: SassTraceFile } | { error: string } {
  try {
    return { file: JSON.parse(text) as SassTraceFile };
  } catch {
    return { error: '文件不是合法的 JSON（请检查是否为 NVBit 风格 SASS trace 导出）' };
  }
}

interface NumberFieldProps {
  label: string;
  value: number;
  options: number[];
  onChange: (value: number) => void;
}

function NumberField(props: NumberFieldProps) {
  const { label, value, options, onChange } = props;
  return (
    <label className="config-field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function App() {
  const [config, setConfig] = useState<GemmConfig>(DEFAULT_GEMM_CONFIG);
  const [attentionConfig, setAttentionConfig] = useState<AttentionConfig>(DEFAULT_ATTENTION_CONFIG);
  const [blockConfig, setBlockConfig] = useState<TransformerBlockConfig>(DEFAULT_TRANSFORMER_BLOCK_CONFIG);
  const [source, setSource] = useState<SourceMode>('gemm');

  // V0.5：真实 trace 模式。默认使用内置示例 trace（sample: true），
  // 用户可上传自己的 Nsight 风格 JSON 文件替换。
  const [realTraceFile, setRealTraceFile] = useState<NsightTraceFile>(SAMPLE_REAL_TRACE);
  const [realTraceFileName, setRealTraceFileName] = useState<string>('内置示例 trace');
  const [realTraceError, setRealTraceError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // V0.8：SASS trace 模式。默认使用内置示例 trace（sample: true），
  // 用户可上传自己的 NVBit 风格 SASS trace JSON 文件替换。
  const [sassTraceFile, setSassTraceFile] = useState<SassTraceFile>(SAMPLE_SASS_TRACE);
  const [sassTraceFileName, setSassTraceFileName] = useState<string>('内置示例 SASS trace');
  const [sassTraceError, setSassTraceError] = useState<string | null>(null);
  const sassFileInputRef = useRef<HTMLInputElement>(null);

  // V0.9：Multi-GPU 模式。默认数据并行，可切换 TP/PP/DP。
  const [multiGpuConfig, setMultiGpuConfig] = useState<MultiGpuConfig>(DEFAULT_MULTI_GPU_CONFIG);

  // 上传文件处理（仅在用户主动选择文件时执行）
  const handleTraceFileSelect = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      const parsed = parseTraceFileText(text);
      if ('error' in parsed) {
        setRealTraceError(parsed.error);
        return;
      }
      const result = parseNsightTrace(parsed.file);
      if (!result.ok) {
        setRealTraceError(result.error);
        return;
      }
      setRealTraceFile(parsed.file);
      setRealTraceFileName(file.name);
      setRealTraceError(null);
      setSource('real-trace');
    };
    reader.onerror = () => {
      setRealTraceError('文件读取失败，请重试');
    };
    reader.readAsText(file);
  };

  // V0.8：SASS trace 文件上传处理（仅在用户主动选择文件时执行）
  const handleSassFileSelect = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : '';
      const parsed = parseSassFileText(text);
      if ('error' in parsed) {
        setSassTraceError(parsed.error);
        return;
      }
      const result = parseSassTrace(parsed.file);
      if (!result.ok) {
        setSassTraceError(result.error);
        return;
      }
      setSassTraceFile(parsed.file);
      setSassTraceFileName(file.name);
      setSassTraceError(null);
      setSource('sass-trace');
    };
    reader.onerror = () => {
      setSassTraceError('文件读取失败，请重试');
    };
    reader.readAsText(file);
  };

  // Simulation 只在参数变化时运行，产出 TVIR trace（UI 不直接理解仿真细节）
  const gemmTrace = useMemo(() => {
    const trace = simulateGemm(config);
    const validation = validateTVIRTrace(trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return trace;
  }, [config]);

  const attentionTrace = useMemo(() => {
    const trace = simulateAttention(attentionConfig);
    const validation = validateTVIRTrace(trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return trace;
  }, [attentionConfig]);

  const blockTrace = useMemo(() => {
    const trace = simulateTransformerBlock(blockConfig);
    const validation = validateTVIRTrace(trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return trace;
  }, [blockConfig]);

  // V0.5：真实 trace 解析为 TVIR（架构验收点：parser 与仿真引擎输出同为 TVIRTrace，
  // Playback 与 UI 无需任何改动）
  const realTrace = useMemo(() => {
    const result = parseNsightTrace(realTraceFile);
    if (!result.ok) {
      console.error('Real trace parse failed:', result.error);
      return null;
    }
    const validation = validateTVIRTrace(result.trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return result.trace;
  }, [realTraceFile]);

  // V0.8：SASS trace 经 TVIR Adapter 转换为 TVIR（同一架构验收点：
  // 指令级数据源同样只替换 Source，Playback 与 UI 零改动）
  const sassTrace = useMemo(() => {
    const result = parseSassTrace(sassTraceFile);
    if (!result.ok) {
      console.error('SASS trace parse failed:', result.error);
      return null;
    }
    const validation = validateTVIRTrace(result.trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return result.trace;
  }, [sassTraceFile]);

  // V0.9：Multi-GPU 仿真（TP/PP/DP），产出 TVIR trace
  const multiGpuTrace = useMemo(() => {
    const trace = simulateMultiGpu(multiGpuConfig);
    const validation = validateTVIRTrace(trace);
    if (!validation.valid) {
      console.error('TVIR validation failed:', validation.errors);
    }
    return trace;
  }, [multiGpuConfig]);

  const trace =
    source === 'gemm'
      ? gemmTrace
      : source === 'attention'
        ? attentionTrace
        : source === 'block'
          ? blockTrace
          : source === 'real-trace'
            ? (realTrace ?? EXAMPLE_TVIR_TRACE)
            : source === 'sass-trace'
              ? (sassTrace ?? EXAMPLE_TVIR_TRACE)
              : source === 'multigpu'
                ? multiGpuTrace
                : EXAMPLE_TVIR_TRACE;
  const playback = usePlayback(trace);

  // 数据源/参数变化时，把新 trace 交给 Playback Engine。
  // 这正是架构验收点的演示：同一 Playback/UI 可消费任意来源的 TVIR。
  useEffect(() => {
    playback.loadTrace(trace);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace]);

  // 从当前事件提取 MatrixView 需要的高亮信息（纯消费 TVIR）
  const activeTiles: string[] = [];
  if (playback.event?.tile) activeTiles.push(playback.event.tile.label);

  // 矩阵场景：由当前事件的 metadata.gemm 驱动（无则保持上一次场景）
  const [lastScene, setLastScene] = useState<{
    left: string;
    right: string;
    out: string;
    M: number;
    N: number;
    K: number;
    tileM: number;
    tileN: number;
    tileK: number;
  }>({ left: 'A', right: 'B', out: 'C', M: 128, N: 128, K: 128, tileM: 32, tileN: 32, tileK: 32 });

  const scene = useMemo(() => projectMatrixScene(playback.event), [playback.event]);

  useEffect(() => {
    if (scene) setLastScene(scene);
  }, [scene]);

  const effectiveScene = scene ?? lastScene;

  // 数据源切换时重置矩阵场景
  useEffect(() => {
    if (source === 'gemm') {
      setLastScene({ left: 'A', right: 'B', out: 'C', M: config.M, N: config.N, K: config.K, tileM: config.tileM, tileN: config.tileN, tileK: config.tileK });
    } else if (source === 'attention') {
      setLastScene({ left: 'X', right: 'Wq', out: 'Q', M: attentionConfig.seqLen, N: attentionConfig.headDim, K: attentionConfig.dModel, tileM: attentionConfig.tileM, tileN: attentionConfig.tileN, tileK: attentionConfig.tileK });
    } else if (source === 'block') {
      setLastScene({ left: 'Xn', right: 'Wq', out: 'Q', M: blockConfig.seqLen, N: blockConfig.headDim, K: blockConfig.dModel, tileM: blockConfig.tileM, tileN: blockConfig.tileN, tileK: blockConfig.tileK });
    } else if (source === 'real-trace') {
      // 真实 trace 不含矩阵 tiling 信息，矩阵视图显示中性占位场景
      setLastScene({ left: 'Input', right: 'Weight', out: 'Output', M: 64, N: 64, K: 64, tileM: 32, tileN: 32, tileK: 32 });
    } else if (source === 'sass-trace') {
      // SASS trace 聚焦单 kernel 的指令流，矩阵视图显示该 kernel 的 tiling 场景
      setLastScene({ left: 'A', right: 'B', out: 'C', M: 128, N: 128, K: 128, tileM: 32, tileN: 32, tileK: 32 });
    } else if (source === 'multigpu') {
      // Multi-GPU 模式：矩阵视图显示一个代表性 GEMM（单卡视角的分片前形状）
      setLastScene({ left: 'X', right: 'W', out: 'Y', M: multiGpuConfig.seqLen, N: multiGpuConfig.dModel, K: multiGpuConfig.dModel, tileM: multiGpuConfig.tileM, tileN: multiGpuConfig.tileN, tileK: multiGpuConfig.tileK });
    } else {
      setLastScene({ left: 'A', right: 'B', out: 'C', M: 4, N: 4, K: 4, tileM: 4, tileN: 4, tileK: 4 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // GPU View 硬件参数：真实 trace / SASS trace 模式从 trace 推导，其余用各场景配置
  const realTraceHardware = useMemo(
    () => (realTrace ? inferRealTraceHardware(realTrace) : { numSM: 4, warpsPerBlock: 4 }),
    [realTrace],
  );
  const sassTraceHardware = useMemo(
    () => (sassTrace ? inferSassTraceHardware(sassTrace) : { numSM: 1, warpsPerBlock: 4 }),
    [sassTrace],
  );
  const numSM =
    source === 'real-trace'
      ? realTraceHardware.numSM
      : source === 'sass-trace'
        ? sassTraceHardware.numSM
        : source === 'attention'
          ? attentionConfig.numSM
          : source === 'block'
            ? blockConfig.numSM
            : config.numSM;
  const warpsPerBlock =
    source === 'real-trace'
      ? realTraceHardware.warpsPerBlock
      : source === 'sass-trace'
        ? sassTraceHardware.warpsPerBlock
        : source === 'attention'
          ? attentionConfig.warpsPerBlock
          : source === 'block'
          ? blockConfig.warpsPerBlock
          : config.warpsPerBlock;

  const smStates = useMemo(
    () => projectSmStates(trace.events, playback.state.currentIndex, numSM),
    [trace, playback.state.currentIndex, numSM],
  );

  const operatorSegments = useMemo(() => projectOperatorSegments(trace.events), [trace]);

  // V0.5：kernel 时间轴投影（仅真实 trace 模式有意义）
  const kernelTimelineSegments = useMemo(
    () => projectKernelTimeline(trace.events),
    [trace],
  );

  // V0.8：SASS 指令投影（仅 sass-trace 模式有内容）
  const sassInstructions = useMemo(
    () => projectSassInstructions(trace.events),
    [trace],
  );

  // V0.7：Architecture Playground 硬件规格状态（仅影响仿真模式的屋顶线估算）
  const [hardwareSpec, setHardwareSpec] = useState<HardwareSpec>(DEFAULT_HARDWARE_SPEC);
  const hardwareDirty = useMemo(() => specsDiffer(DEFAULT_HARDWARE_SPEC, hardwareSpec), [hardwareSpec]);

  // V0.6：性能分析报告（仿真=Simulated，真实=Measured，示例数据绝不标 Measured）
  const perfReport = useMemo(
    () => computePerfReport(trace, playback.event, hardwareSpec),
    [trace, playback.event, hardwareSpec],
  );

  // V0.7：Playground 对比分析（基线 vs 修改后的硬件）
  const playgroundAnalysis = useMemo(
    () => analyzePlayground(trace, DEFAULT_HARDWARE_SPEC, hardwareSpec),
    [trace, hardwareSpec],
  );

  // 编译下钻链：由当前事件的 operator/kernel/metadata 投影（V0.4）
  const compileChain = useMemo(() => projectCompileChain(playback.event), [playback.event]);

  const updateConfig = (patch: Partial<GemmConfig>) => {
    setConfig((prev) => ({ ...prev, ...patch }));
    setSource('gemm');
  };

  const updateAttentionConfig = (patch: Partial<AttentionConfig>) => {
    setAttentionConfig((prev) => ({ ...prev, ...patch }));
    setSource('attention');
  };

  const updateBlockConfig = (patch: Partial<TransformerBlockConfig>) => {
    setBlockConfig((prev) => ({ ...prev, ...patch }));
    setSource('block');
  };

  const updateMultiGpuConfig = (patch: Partial<MultiGpuConfig>) => {
    setMultiGpuConfig((prev) => ({ ...prev, ...patch }));
    setSource('multigpu');
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <h1>Transformer GPU Visual Simulator</h1>
          <span className="app-version">
            V1.0 · 六层联动 · Model ↔ GPU ↔ Kernel ↔ Memory + Timeline + What/Why
          </span>
        </div>

        <div className="app-config">
          <div className="config-group">
            <span className="config-group-label">数据源</span>
            <div className="config-buttons">
              <button
                type="button"
                className={source === 'gemm' ? 'active' : ''}
                onClick={() => setSource('gemm')}
              >
                GEMM 仿真
              </button>
              <button
                type="button"
                className={source === 'attention' ? 'active' : ''}
                onClick={() => setSource('attention')}
              >
                Attention 仿真
              </button>
              <button
                type="button"
                className={source === 'block' ? 'active' : ''}
                onClick={() => setSource('block')}
              >
                Transformer Block
              </button>
              <button
                type="button"
                className={source === 'real-trace' ? 'active' : ''}
                onClick={() => setSource('real-trace')}
              >
                Real Trace
              </button>
              <button
                type="button"
                className={source === 'sass-trace' ? 'active' : ''}
                onClick={() => setSource('sass-trace')}
              >
                SASS Trace
              </button>
              <button
                type="button"
                className={source === 'multigpu' ? 'active' : ''}
                onClick={() => setSource('multigpu')}
              >
                Multi-GPU
              </button>
              <button
                type="button"
                className={source === 'example' ? 'active' : ''}
                onClick={() => setSource('example')}
              >
                示例 Trace
              </button>
            </div>
          </div>

          {source === 'gemm' ? (
            <>
              <NumberField label="M" value={config.M} options={SIZE_OPTIONS} onChange={(v) => updateConfig({ M: v })} />
              <NumberField label="N" value={config.N} options={SIZE_OPTIONS} onChange={(v) => updateConfig({ N: v })} />
              <NumberField label="K" value={config.K} options={SIZE_OPTIONS} onChange={(v) => updateConfig({ K: v })} />
              <NumberField label="Tile" value={config.tileM} options={TILE_OPTIONS} onChange={(v) => updateConfig({ tileM: v, tileN: v, tileK: v })} />
              <NumberField label="SM" value={config.numSM} options={SM_OPTIONS} onChange={(v) => updateConfig({ numSM: v })} />
              <NumberField label="Warps/Block" value={config.warpsPerBlock} options={WARP_OPTIONS} onChange={(v) => updateConfig({ warpsPerBlock: v })} />
            </>
          ) : null}

          {source === 'attention' ? (
            <>
              <NumberField label="SeqLen" value={attentionConfig.seqLen} options={SEQ_OPTIONS} onChange={(v) => updateAttentionConfig({ seqLen: v })} />
              <NumberField label="d_model" value={attentionConfig.dModel} options={DIM_OPTIONS} onChange={(v) => updateAttentionConfig({ dModel: v })} />
              <NumberField label="headDim" value={attentionConfig.headDim} options={DIM_OPTIONS} onChange={(v) => updateAttentionConfig({ headDim: v })} />
              <NumberField label="Tile" value={attentionConfig.tileM} options={TILE_OPTIONS} onChange={(v) => updateAttentionConfig({ tileM: v, tileN: v, tileK: v })} />
              <NumberField label="SM" value={attentionConfig.numSM} options={SM_OPTIONS} onChange={(v) => updateAttentionConfig({ numSM: v })} />
              <NumberField label="Warps/Block" value={attentionConfig.warpsPerBlock} options={WARP_OPTIONS} onChange={(v) => updateAttentionConfig({ warpsPerBlock: v })} />
            </>
          ) : null}

          {source === 'block' ? (
            <>
              <NumberField label="SeqLen" value={blockConfig.seqLen} options={SEQ_OPTIONS} onChange={(v) => updateBlockConfig({ seqLen: v })} />
              <NumberField label="d_model" value={blockConfig.dModel} options={DIM_OPTIONS} onChange={(v) => updateBlockConfig({ dModel: v })} />
              <NumberField label="headDim" value={blockConfig.headDim} options={DIM_OPTIONS} onChange={(v) => updateBlockConfig({ headDim: v })} />
              <NumberField label="FFN dim" value={blockConfig.ffnDim} options={FFN_OPTIONS} onChange={(v) => updateBlockConfig({ ffnDim: v })} />
              <NumberField label="Tile" value={blockConfig.tileM} options={TILE_OPTIONS} onChange={(v) => updateBlockConfig({ tileM: v, tileN: v, tileK: v })} />
              <NumberField label="SM" value={blockConfig.numSM} options={SM_OPTIONS} onChange={(v) => updateBlockConfig({ numSM: v })} />
              <NumberField label="Warps/Block" value={blockConfig.warpsPerBlock} options={WARP_OPTIONS} onChange={(v) => updateBlockConfig({ warpsPerBlock: v })} />
            </>
          ) : null}

          {source === 'multigpu' ? (
            <>
              <label className="config-field">
                <span>并行策略</span>
                <select
                  value={multiGpuConfig.strategy}
                  onChange={(e) =>
                    updateMultiGpuConfig({ strategy: e.target.value as ParallelStrategy })
                  }
                >
                  <option value="dp">数据并行（DP）</option>
                  <option value="tp">张量并行（TP）</option>
                  <option value="pp">流水线并行（PP）</option>
                  <option value="comm-allreduce">AllReduce 原语</option>
                  <option value="comm-allgather">AllGather 原语</option>
                  <option value="comm-reducescatter">ReduceScatter 原语</option>
                </select>
              </label>
              <NumberField
                label="GPU 数"
                value={multiGpuConfig.numGpus}
                options={[2, 4, 8]}
                onChange={(v) => updateMultiGpuConfig({ numGpus: v })}
              />
              <NumberField
                label="SeqLen"
                value={multiGpuConfig.seqLen}
                options={SEQ_OPTIONS}
                onChange={(v) => updateMultiGpuConfig({ seqLen: v })}
              />
              <NumberField
                label="d_model"
                value={multiGpuConfig.dModel}
                options={DIM_OPTIONS}
                onChange={(v) => updateMultiGpuConfig({ dModel: v })}
              />
            </>
          ) : null}

          {source === 'real-trace' ? (
            <div className="config-group realtrace-config">
              <span className="config-group-label">
                当前：{realTraceFileName}
                {realTraceFile.sample ? '（示例数据 · 教学示意值，非实测）' : '（Measured 数据）'}
              </span>
              <div className="config-buttons">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  上传 Nsight JSON…
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setRealTraceFile(SAMPLE_REAL_TRACE);
                    setRealTraceFileName('内置示例 trace');
                    setRealTraceError(null);
                  }}
                >
                  恢复示例
                </button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,application/json"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleTraceFileSelect(file);
                  // 允许重复选择同一文件
                  e.target.value = '';
                }}
              />
              {realTraceError ? (
                <span className="realtrace-error">{realTraceError}</span>
              ) : null}
            </div>
          ) : null}

          {source === 'sass-trace' ? (
            <div className="config-group realtrace-config">
              <span className="config-group-label">
                当前：{sassTraceFileName}
                {sassTraceFile.sample ? '（示例指令流 · 教学示意编排）' : '（NVBit 采集的指令流）'}
              </span>
              <div className="config-buttons">
                <button
                  type="button"
                  onClick={() => sassFileInputRef.current?.click()}
                >
                  上传 SASS trace JSON…
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSassTraceFile(SAMPLE_SASS_TRACE);
                    setSassTraceFileName('内置示例 SASS trace');
                    setSassTraceError(null);
                  }}
                >
                  恢复示例
                </button>
              </div>
              <input
                ref={sassFileInputRef}
                type="file"
                accept=".json,application/json"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleSassFileSelect(file);
                  // 允许重复选择同一文件
                  e.target.value = '';
                }}
              />
              {sassTraceError ? (
                <span className="realtrace-error">{sassTraceError}</span>
              ) : null}
            </div>
          ) : null}
        </div>
      </header>

      <ControlBar
        playing={playback.state.playing}
        speed={playback.state.speed}
        currentIndex={playback.state.currentIndex}
        totalEvents={playback.totalEvents}
        onPrevious={playback.previous}
        onNext={playback.next}
        onPlay={playback.play}
        onPause={playback.pause}
        onReset={playback.reset}
        onSeek={playback.seek}
        onSpeedChange={playback.setSpeed}
      />

      <main className="app-main">
        <div className="panel-left">
          {source === 'block' ? <ModelView event={playback.event} /> : null}
          <MatrixView
            M={effectiveScene.M}
            N={effectiveScene.N}
            K={effectiveScene.K}
            tileM={effectiveScene.tileM}
            tileN={effectiveScene.tileN}
            tileK={effectiveScene.tileK}
            activeTiles={activeTiles}
            activeTensor={playback.event?.tensor}
            leftLabel={effectiveScene.left}
            rightLabel={effectiveScene.right}
            outLabel={effectiveScene.out}
          />
          <TensorCoreView event={playback.event} />
        </div>

        <div className="panel-center">
          {source === 'multigpu' ? (
            <MultiGpuView event={playback.event} numGpus={multiGpuConfig.numGpus} />
          ) : (
            <GpuView
              event={playback.event}
              numSM={numSM}
              warpsPerBlock={warpsPerBlock}
              smStates={smStates}
            />
          )}
        </div>

        <div className="panel-right">
          <MemoryView event={playback.event} />
        </div>
      </main>

      <section className="app-compiler">
        <CompilerView chain={compileChain} />
      </section>

      {source === 'real-trace' && kernelTimelineSegments.length > 0 ? (
        <section className="app-kernel-timeline">
          <KernelTimeline
            segments={kernelTimelineSegments}
            currentIndex={playback.state.currentIndex}
            onSeek={playback.seek}
            isSample={realTraceFile.sample === true}
          />
        </section>
      ) : null}

      {source === 'sass-trace' && sassInstructions.length > 0 ? (
        <section className="app-kernel-timeline">
          <InstructionView
            rows={sassInstructions}
            currentIndex={playback.state.currentIndex}
            onSeek={playback.seek}
            isSample={sassTraceFile.sample === true}
          />
        </section>
      ) : null}

      <section className="app-perf">
        <PerfPanel report={perfReport} />
      </section>

      {source !== 'real-trace' && source !== 'sass-trace' && source !== 'example' ? (
        <section className="app-playground">
          <ArchitecturePlayground
            baseline={DEFAULT_HARDWARE_SPEC}
            modified={hardwareSpec}
            onChange={(patch) => setHardwareSpec((prev) => ({ ...prev, ...patch }))}
            onReset={() => setHardwareSpec(DEFAULT_HARDWARE_SPEC)}
            analysis={playgroundAnalysis}
            isDirty={hardwareDirty}
          />
        </section>
      ) : null}

      <section className="app-bottom">
        <EventExplanation
          event={playback.event}
          stepIndex={playback.state.currentIndex}
          totalEvents={playback.totalEvents}
        />
        <Timeline
          events={trace.events}
          currentIndex={playback.state.currentIndex}
          onSeek={playback.seek}
          operatorSegments={operatorSegments}
        />
      </section>

      <footer className="app-footer">
        <span>{trace.description}</span>
        <span>
          数据来源：
          {source === 'sass-trace'
            ? sassTraceFile.sample
              ? '示例指令流（教学示意编排）'
              : 'NVBit SASS 指令流（Educational Simulation，非 cycle-accurate）'
            : trace.provenance === 'real-trace'
              ? trace.isSample
                ? '示例数据（教学示意值，非实测）'
                : 'Measured（真实 GPU 实测）'
              : 'Simulated（教学仿真）'}
          {' · '}
          架构：Source → TVIR → Playback → UI（见 ARCHITECTURE.md）
        </span>
      </footer>
    </div>
  );
}
