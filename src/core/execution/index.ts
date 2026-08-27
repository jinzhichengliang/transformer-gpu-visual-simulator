/** Execution 模块导出（Sprint 4/5）：任务定义 + Planner + 模板注册表 + 执行器 */

export * from './task';
export * from './planner';
export { registerOperatorTemplate, getOperatorTemplate } from './templates';
export type { EmitContext, OperatorEmitter } from './templates';
export { planExecution } from './executor';
export type { ExecutionPlanResult } from './executor';
