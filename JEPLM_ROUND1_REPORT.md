# JEPLM 首轮实验报告：负结果链 + 评估诊断突破

> 日期：2026-08-08
> 实验执行人：总控（自主迭代）
> 状态：负结果归档（含重大诊断发现）

---

## 摘要

JEPLM（JEPA-Enhanced Persistent Latent Model）首轮实验在 MiniGrid-MemoryS7 延迟记忆任务上运行，**结论为负**：SPR 式多步 latent 预测辅助损失相对 hybrid baseline 无任何增量。但更重要的是，实验揭示了一个**影响所有历史结论的评估缺陷**——MemoryS7 held-out 评估被"先右转再直走"的平凡策略攻破，任何控制器都能稳定命中 27/64，**该评估从未测量过记忆能力**。

作为诊断工具交付：**paired counterfactual swap probe**（交换走廊两端 Key/Ball 物体 + 互换 success/failure 判定），只有原始与 swapped 双版本都成功才计为真记忆。所有已测控制器（含 JEPLM、hybrid baseline、Transformer、intrinsic 探索版）在该探针下 **both=0**——无一学会真记忆。

---

## 一、实验设置

| 项 | 值 |
|---|---|
| 基座架构 | HybridPLWMTransformerActorCritic（tactical Transformer + persistent PLWM side state）|
| JEPLM 增量 | action-conditioned 多步 latent 预测（jeplem_loss，从 belief_t + 动作序列预测 belief_{t+k}，detach 目标，无 EMA）+ SIGReg-lite 方差反坍缩 |
| 任务 | 两阶段 curriculum：Empty-5x5 导航 bootstrap（30 updates）→ MemoryS7 延迟记忆转移（30/60 updates）|
| 预算 | 24 episodes/update × (30+30) 与 × (30+60)；PPO external reward |
| 对照 | hybrid_jeplem vs hybrid_baseline vs transformer_reference（同 seed、同预算）|
| 评估 | held-out 64 worlds（seed 1_000_000 起）+ counterfactual swap probe |
| 环境 | RTX 4060 Laptop，CUDA，torch 2.13 |

## 二、关键发现 1：MemoryS7 评估被平凡策略攻破

所有控制器（无论模型、无论 30/60 updates、无论是否加 JEPLM/intrinsic）的 held-out 动作分布完全一致：

```text
action_counts = [0, 64, 221, 0, 0, 0, 0]
```

即每个 episode 恰好 right 转向 1 次 + forward 3.45 次——**"先右转再直走"固定路线**。该策略在 64 个 held-out worlds 上稳定命中 27/64（成功率 42%），因为其中有 27 个世界的 success_pos 恰好在直走路径上。

**铁证**：训练预算 30→60 updates 翻倍后，held-out 纹丝不动（27/64）；所有模型同分（27/27/27）。这不是模型能力差异，是评估的固定属性。

**结论：MemoryS7 held-out 评估从未测量过记忆能力。历史报告（hybrid 37/64 vs transformer 18/64）的"架构优势"极可能是平凡策略在不同随机种子下的命中率波动，不是真实记忆证据，需用 counterfactual 评估重新验证。**

## 三、关键发现 2：JEPLM 无增量

| 控制器 | heldout | probe orig | probe swap | **both（真记忆）** |
|---|---|---|---|---|
| hybrid_jeplem (α=0.1, h=2) | 27/64 | 27 | 0 | **0** |
| hybrid_baseline | 27/64 | 27 | 0 | **0** |
| transformer_reference | 27/64 | 27 | 0 | **0** |

SPR 式多步 latent 预测（在 Empty-5x5 单步自预测失败后的升级版）**依然无效**。与文献的对照：SPR 在 Atari 100k 上有效，其目标是从 belief 预测**未来观测的编码**（非自举），而我实现的 jeplem_loss 预测的是**模型自己 rollout 的未来 belief**（自举目标，虽然 detach）。自举目标可能鼓励惰性解（预测"belief 变化不大"）而非学习任务相关动态——这是下一轮改进的首选方向。

## 四、关键发现 3：intrinsic 探索也无法解锁记忆

用项目已有的 `ppo_intrinsic_count_v1`（内在计数奖励 bonus=0.05）重训，agent 探索显著增加（action_counts 出现 790 次 toggle、mean_steps 16.8），但 **held-out 仍 27/64、swap both=0**。

**结论：根因不是探索不足。** 当前"架构 + 任务 + reward-only PPO 范式"组合在预算内无法学会"记忆物体身份 → 选择走廊侧"的能力。

## 五、交付的新评估基础设施：paired counterfactual swap probe

`evaluate_memory_probe`（experiments/train_jeplem_hybrid_curriculum.py）：

- 对每个 held-out world 评估两次：原始布局 + **swap 布局**（交换走廊两端 Key/Ball 物体位置，并互换 success/failure 判定）
- 语义：真记忆策略跟随 matching 物体 → 双版本都成功（both=1）；固定路线策略 → 至多命中一个版本
- 输出：`orig / swap / both` 三元组
- 这是 MASTER_PLAN "counterfactual intervention on held-out worlds" 原则的落地，**应成为所有记忆类任务的默认评估**

## 六、决策（总控）

1. **归档本轮为负结果**（符合 Completion Rule：重复受控评估显示无优势）——MemoryS7 两阶段 curriculum 在当前范式下不可学，停止在此任务上继续堆辅助损失。
2. **保留 swap probe 作为评估基础设施**，后续所有"记忆能力"声明必须以 `both > 0` 为证据。
3. **重新审视历史证据**：37/64 等结论在 swap probe 下需重新验证；未验证前不得作为"PLWM 架构优势"引用。
4. **下一轮方向（待定）**：
   - A. 改进 JEPLM 目标：从"自举 belief 预测"改为"预测未来观测编码"（SPR 原版）或 TD-MPC 式任务导向目标；
   - B. 换任务：构造"决策点明确依赖记忆"的任务（如身份→显式选择），而非 MemoryS7 的稀疏到达奖励；
   - C. 承认该方向需要更强的任务信号，转向其他可诊断能力（如已有成功的 MiniGrid hybrid curriculum 导航类能力 + 用 swap probe 复核）。

## 七、文件清单

- `experiments/train_jeplem_hybrid_curriculum.py`（JEPLM + swap probe，v2）
- `experiments/train_hybrid_intrinsic_memory_probe.py`（intrinsic 诊断）
- `artifacts/jeplem-hybrid-curriculum-report.json`（30 updates）
- `artifacts/jeplem-hybrid-curriculum-m60-report.json`（60 updates）
- `artifacts/jeplem-hybrid-curriculum-m60-probe-report.json`（60 updates + probe，决定版）
- `artifacts/hybrid-intrinsic-memory-probe-report.json`（intrinsic + probe）
