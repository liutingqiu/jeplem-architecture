# MemoryS7 历史结论重验证报告（swap counterfactual 探针）

> 日期：2026-08-10
> 性质：标准审计（STANDARDS_AUDIT_20260809.md A2 项）的落地闭环——用 swap counterfactual 探针与 action-count 特征重验证全部 MemoryS7 历史结论
> 关联：JEPLM_ROUND1_REPORT.md、MASTER_PLAN.md（hybrid curriculum 段）、DIRECTION_PLAN.md、JEPA_PLWM_REDESIGN_REPORT.md

---

## 一、背景

审计发现 MiniGrid-MemoryS7 held-out 评估被**平凡策略**攻破：`action_counts=[0,64,221,...]`（每 episode 先 right 转向一次、再一路 forward），稳定命中 27/64——这些 worlds 的 `success_pos` 恰在直走路径上，评估测不出"记忆能力"。历史"架构优势"（hybrid 37/64 vs transformer 18/64）极可能是平凡策略的种子波动。

## 二、历史结论重验证清单

| 历史结论 | 来源 artifact | 重验证判定 | 证据 |
|---|---|---|---|
| hybrid PLWM 37/64（架构优势）| `minigrid-hybrid-curriculum-report.json` | **证伪** | action_counts=[64,0,292]（**left→forward 平凡策略**），与 27/64 的 [0,64,221]（right→forward）同为平凡策略；37 vs 27 只是转向方向不同 → **命中率差异 = seed 运气** |
| replication 36/64（全同分）| `minigrid-hybrid-curriculum-replication-report.json` | **证伪** | hybrid / PLWM-side-disabled / Transformer 全部 36/64 同分，无架构区分度（MASTER_PLAN 已标 rejected-current-controller）|
| JEPLM v1（30 updates）| `jeplem-hybrid-curriculum-report.json` | **证伪（无 probe，特征一致）** | 全部 27/64 + action_counts=[0,64,221] 平凡策略特征；无 memory_probe 字段但 eval 特征与 60-updates 版完全一致 |
| JEPLM v1（60 updates）swap probe | `jeplem-hybrid-curriculum-m60-probe-report.json` | **证伪** | hybrid_jeplem / hybrid_baseline / transformer 全部 `memory_probe={original:27, swapped:0, both:0}`——both=0 = 非真记忆 |
| JEPLM v2 + no_side 消融 probe | `jeplem-v2-hybrid-curriculum-probe-report.json` | **证伪** | v2 / baseline / no_side 全部 27/64 + both=0 |
| intrinsic 学习规则 probe | `hybrid-intrinsic-memory-probe-report.json` | **证伪** | 27/64 + both=0，accepted=False |
| transformer_capacity_control 18/64 | `minigrid-hybrid-curriculum-report.json` | **证伪（非记忆策略）** | counts=[0,35,163,0,1784]（大量转向动作，非记忆策略）|

**全部历史 MemoryS7 架构结论 = 证伪/不可诊断，无一幸免。**

## 三、方法论要点

- **平凡策略判定**：action_counts 显示"先转向一次再直走"结构即平凡策略；37/64 与 27/64 的差别仅转向方向（left vs right），说明 37/64 的"优势"是转向方向选择的 seed 运气。
- **swap counterfactual**：`memory_probe={original, swapped, both}`——`both`（原版+交换版都成功）才是真记忆；全部 0 = 无真记忆。
- **替代评估**：DecideAfterDelay 任务族的 swap 探针方法论（`DECIDE_AFTER_DELAY_REPORT.md` §六）已正式替代失效的 MemoryS7 评估——"决策必须依赖跨步身份记忆 + swap counterfactual"能干净分离真记忆/位置捷径/无记忆三态。

## 四、文档影响处置

| 文档 | 现状 | 处置 |
|---|---|---|
| MASTER_PLAN.md（hybrid 段）| 已标"deliberately not accepted"+"rejected-current-controller" | **补最终标注**：平凡策略攻破（27/64 恒定）坐实，37/64 亦为平凡策略 |
| DIRECTION_PLAN.md | 将 37/64 列为"最有力证据/成功先例"（4 处） | **修正**：标注证伪，最有力证据改为 DecideAfterDelay 家族 |
| JEPA_PLWM_REDESIGN_REPORT.md | 将 37/64 列为"目前最强证据"、环境选择理由 | **修正**：标注证伪，环境选择理由更新 |
| COGNITIVE_SYSTEM_WHITEPAPER.md | 12 区块不依赖 MemoryS7（区块 3 用 consolidation 实验）| **无影响** ✓ |
| DECIDE_AFTER_DELAY_REPORT.md | 已正确引用（MemoryS7 已替代）| 无影响 ✓ |

## 五、结论

MemoryS7 路线的全部架构主张（hybrid 优势、JEPLM v1/v2 增量、intrinsic 变体）在 swap 探针下**全部证伪**——这些结论从未测出记忆能力。项目的可信证据基础已完全转移到 DecideAfterDelay/DecideStream 家族（受控"记忆→决策"任务 + swap 探针，见 JEPLM_AUX_REPORT.md 四象限与 RMT_STREAMING_COLLAPSE_REPORT.md）。
