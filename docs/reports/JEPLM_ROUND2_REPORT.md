# JEPLM 第二轮实验：JEPLM v2 目标改进 + 完整负结果链归档

> 日期：2026-08-08
> 状态：MemoryS7 reward-only 路线停止（Completion Rule 适用）
> 关联：JEPLM_ROUND1_REPORT.md（第一轮）

---

## 一、本轮实验

**JEPLM v2**：辅助目标从"自举 belief 预测"改为 **SPR 原版"预测未来观测的编码"**（真实观测流，detach，无 EMA）。这是文献中 SPR（Atari 100k +55%）的机制，本应最有效。

| 控制器 | heldout | probe orig | probe swap | both（真记忆）| mean_steps |
|---|---|---|---|---|---|
| hybrid_jeplem_v2 | 27/64 | 27 | 0 | **0** | 4.5 |
| hybrid_baseline | 27/64 | 27 | 0 | **0** | 4.5 |
| hybrid_no_side | 27/64 | 27 | 0 | **0** | 4.5 |

新增 **hybrid_no_side 对照**：去掉 PLWM 侧状态后结果完全一致 → **在当前训练下 PLWM 侧对策略输出零贡献，Transformer 战术层完全主导**。

## 二、完整负结果链（5 种配置 × 多预算）

| # | 干预 | 结果 |
|---|---|---|
| 1 | 平凡基线（reward-only PPO）| 27/64，both=0 |
| 2 | 预算翻倍（30→60 updates）| 仍 27/64 |
| 3 | JEPLM v1（自举 belief 多步预测）| 仍 27/64，both=0 |
| 4 | JEPLM v2（SPR 式观测编码预测）| 仍 27/64，both=0 |
| 5 | intrinsic count 探索奖励 | 探索大增（790 toggle）仍 27/64，both=0 |

**任何干预都无法让任何控制器学会"记忆身份 → 选择走廊侧"**。

## 三、归因（总控判断）

1. **评估层**：MemoryS7 held-out 被平凡"右转直走"策略攻破（27/64 恒定）——已确认，swap probe 交付。
2. **训练层**：MemoryEnv 的 failure_pos 无惩罚（走错 = 0 分而非负分），平凡策略期望奖励（0.42）已是最优可行解 → **PPO 收敛到平凡解是理性行为**，不是探索不足（intrinsic 已证）。
3. **表征层**：delayed-object-identity v2 已证"记住身份"对 GRU 是 1.0 的易事。**"记住"不难，"用记忆做决策"在稀疏到达奖励下学不会**。
4. **架构层**：PLWM 侧状态在 PPO 梯度下未被策略利用（no_side 相同）——持久状态没有被"需要它才能拿奖励"的训练信号驱动。

**结论：MemoryS7 reward-only curriculum 是不可诊断的组合，停止此路线。** 依据 MASTER_PLAN Completion Rule（5 次受控评估无优势，成本超过信息收益）。

## 四、对主旨的含义（重要）

PLWM 主旨 = "持久认知系统能用经验获取并保留目标导向行为"。本轮证明的是：

- **用 PPO + 稀疏到达奖励"撞"出记忆行为不可行**（至少此预算内）；
- **记忆任务需要"结构性要求使用记忆"的任务设计**——决策点必须明确依赖跨步信息，且走错有代价；
- 下一轮任务必须满足：① 决策点明确（不是到达奖励）；② 平凡策略零分或负分；③ "记住"与"使用"被显式分离测量。

## 五、下一轮方向（已决策）

构造 **"记忆→决策"受控任务**（基于 delayed-object-identity 管线，它已证可生成身份帧+延迟+查询）：
- 结构：step 0 显示身份（Key/Ball）→ k 步延迟（distractor）→ step k+1 双选决策（左/右），选择与身份匹配者得 +1，错选 0/-1；
- 对比：no_memory / GRU / Transformer / hybrid(PLWM 侧) / hybrid_no_side；
- 指标：held-out 决策准确率 + swap counterfactual（身份匹配互换）；
- 验收：**有持久状态者准确率 > 无状态者**，且 swap 后仍正确（真用记忆，非位置捷径）。

此任务直接回答主旨问题："持久状态是否让 agent 能用跨步信息做决策"。
