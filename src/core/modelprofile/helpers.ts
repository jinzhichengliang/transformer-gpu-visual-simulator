/**
 * ModelProfile 构造辅助函数（Sprint 1）。
 * 简化 Traced<T> 与 SourceMetadata 的书写，避免样板代码。
 */

import type { SourceMetadata, Traced, ConfidenceLevel, FidelityLevel } from './types';

/** 构造 Traced<T>：值 + 来源 + 保真度 */
export function traced<T>(
  value: T,
  sources: SourceMetadata[],
  fidelity: FidelityLevel = 'L1',
): Traced<T> {
  return { value, sources, fidelity };
}

/** 官方公开文档来源 */
export function officialSource(reference: string, verifiedAt = '2026-08-26', confidence: ConfidenceLevel = 'high'): SourceMetadata {
  return { sourceType: 'official', reference, verifiedAt, confidence };
}

/** 推断来源 */
export function inferredSource(reference: string, verifiedAt = '2026-08-26', confidence: ConfidenceLevel = 'medium'): SourceMetadata {
  return { sourceType: 'inferred', reference, verifiedAt, confidence };
}

/** 估算来源 */
export function estimatedSource(reference: string, verifiedAt = '2026-08-26', confidence: ConfidenceLevel = 'low'): SourceMetadata {
  return { sourceType: 'estimated', reference, verifiedAt, confidence };
}
