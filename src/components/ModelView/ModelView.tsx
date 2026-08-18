/**
 * ModelView — 模型结构视图（V1.0 六层联动 · Model 层）。
 *
 * 展示 Transformer Block 的层级结构（Pre-Norm 数据流树），
 * 并根据当前 TVIR 事件高亮"激活路径"（Block 根 → 当前激活子层）。
 *
 * 本组件只消费 core/model 的纯投影结果（projectActiveModelPath），
 * 不自己计算、不理解仿真细节（见 ARCHITECTURE.md）：
 *   - 激活路径完全由 event.operator 决定；
 *   - Attention / FFN 的展开/折叠也是纯投影：激活路径命中父节点才展开其子算子，
 *     无需组件内部交互状态。
 */

import {
  TRANSFORMER_BLOCK_MODEL,
  projectActiveModelPath,
  isBlockModelEvent,
  type ModelNode,
} from '../../core/model';
import type { TVIREvent } from '../../core/tvir/types';
import './ModelView.css';

export interface ModelViewProps {
  event: TVIREvent | null;
}

/** 节点分类 → 着色标记（用于左侧色条） */
const CATEGORY_DOT: Record<string, string> = {
  block: 'model-dot-block',
  norm: 'model-dot-norm',
  attention: 'model-dot-attention',
  'attention-sub': 'model-dot-attention-sub',
  residual: 'model-dot-residual',
  ffn: 'model-dot-ffn',
  'ffn-sub': 'model-dot-ffn-sub',
};

export function ModelView(props: ModelViewProps) {
  const { event } = props;

  const isBlockEvent = isBlockModelEvent(event);
  const activePath = isBlockEvent ? projectActiveModelPath(event) : [];
  const activeIds = new Set(activePath);
  // 激活路径的末端节点 = 当前真正执行的模型子层
  const currentId = activePath.length > 0 ? activePath[activePath.length - 1] ?? null : null;
  const currentNode = currentId ? findNode(TRANSFORMER_BLOCK_MODEL, currentId) : null;

  return (
    <div className="model-view">
      <div className="model-view-title">
        <h3>Model View</h3>
        <span className="model-note">Transformer Block 结构 · 激活路径高亮</span>
      </div>

      {!isBlockEvent ? (
        <div className="model-empty">
          <p>当前事件不属于 Transformer Block 仿真。</p>
          <p className="model-empty-hint">
            播放「Transformer Block」数据源，即可看到模型结构树随执行逐步点亮：
            RMSNorm → Attention → Residual → RMSNorm → FFN → Residual。
          </p>
        </div>
      ) : (
        <div className="model-tree">
          {renderNode(TRANSFORMER_BLOCK_MODEL, activeIds, currentId, 0)}
        </div>
      )}

      {/* 当前激活子层的一句话说明（What/Why 联动的模型侧注脚） */}
      {isBlockEvent && currentNode ? (
        <div className="model-current-hint">
          <span className="model-current-hint-label">当前子层</span>
          <span className="model-current-hint-name">{currentNode.label}</span>
          {currentNode.hint ? <span className="model-current-hint-text">{currentNode.hint}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

/** 递归渲染模型树。激活路径上的节点高亮；父节点命中激活路径时展开其子算子 */
function renderNode(
  node: ModelNode,
  activeIds: Set<string>,
  currentId: string | null,
  depth: number,
) {
  const onPath = activeIds.has(node.id);
  const isCurrent = node.id === currentId;
  // 纯投影的展开规则：只有激活路径命中的父节点才展开子算子（下钻效果）
  const expanded = onPath && node.children.length > 0;

  const nodeClasses = [
    'model-node',
    `model-node-${node.category}`,
    onPath ? 'model-node-on-path' : '',
    isCurrent ? 'model-node-current' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div key={node.id} className="model-subtree" style={{ marginLeft: depth === 0 ? 0 : 14 }}>
      <div className={nodeClasses} title={node.hint}>
        <span className={`model-dot ${CATEGORY_DOT[node.category] ?? ''}`} />
        <span className="model-node-label">{node.label}</span>
        {node.children.length > 0 && !expanded ? (
          <span className="model-node-count">{node.children.length} 子算子</span>
        ) : null}
        {isCurrent ? <span className="model-node-active-tag">执行中</span> : null}
      </div>

      {expanded ? (
        <div className="model-children">
          {node.children.map((child) => renderNode(child, activeIds, currentId, depth + 1))}
        </div>
      ) : null}
    </div>
  );
}

/** 在模型树中按 id 查找节点（深度优先） */
function findNode(root: ModelNode, id: string): ModelNode | null {
  if (root.id === id) return root;
  for (const child of root.children) {
    const found = findNode(child, id);
    if (found) return found;
  }
  return null;
}
