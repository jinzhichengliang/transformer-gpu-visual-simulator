export {
  simulateMultiGpu,
  DEFAULT_MULTI_GPU_CONFIG,
  isCollectiveDemo,
  type MultiGpuConfig,
  type ParallelStrategy,
  type CollectiveDemoStrategy,
} from './multiGpuEngine';

export {
  emitAllReduce,
  emitReduceScatter,
  emitAllGather,
  emitP2P,
  reduceScatterTransfers,
  allGatherTransfers,
  ringStepDurationUs,
  formatBytes,
  DEFAULT_COMM_LINK,
  type CollectiveType,
  type CommTransfer,
  type CommMeta,
  type CommLinkSpec,
  type EmitCollectiveConfig,
} from './commPrimitives';
