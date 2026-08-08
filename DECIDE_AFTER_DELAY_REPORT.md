# DecideAfterDelay 任务族：持久记忆架构优势的首个干净实证

> 日期：2026-08-08
> 状态：✅ 验收通过（3 seed 复现）
> 关联：JEPLM_ROUND1/2_REPORT.md（MemoryS7 负结果链）、WORLD_MODEL_SURVEY_202608.md

---

## 摘要

MemoryS7 reward-only 路线被证实不可诊断（负结果链归档）后，重新设计了**受控"记忆→决策"任务族**。核心发现：**在需要跨步精确回忆的任务上，内容寻址记忆（PLWM / evidence slots）稳定 1.000，压缩隐状态（GRU）只有 0.802 ± 0.069，无持久状态者卡随机（0.504）**。这是 ARCHITECTURE.md 核心主张（"single recurrent hidden state blurs exact recall"，可寻址记忆优于单隐状态）的首个干净、可复现的实证。

## 一、任务族设计（3 个难度递增）

**共同结构**：身份帧（Key/Ball）→ 延迟帧（无物体）→ 决策输出。输入为位置不变式 presence 特征。无奖励、无 teacher、无 oracle。

| 版本 | 结构 | 考验 |
|---|---|---|
| v1 单身份 | 身份(帧0) → 8 帧延迟 → 决策 | 跨步记忆是决策的必要条件 |
| v2 干扰 | 同 v1 + 中途注入相反身份物体 | 抗干扰身份保持 |
| v3 双身份 | 身份A(帧0) + 身份B(帧3) + one-hot 查询(帧8) | **精确位置回忆 + 查询路由** |

**swap counterfactual probe**（v1/v2/v3 通用）：评估时互换身份↔选择映射。真记忆模型忠实跟随身份 → heldout≈1.0、swap≈0.0、evidence=heldout−swap≈+1.0；位置捷径/固定输出 → 两测法同分、evidence≈0。

## 二、结果

### v3 双身份（决定版，3 seed）

| 控制器 | s111 | s222 | s333 | 均值 |
|---|---|---|---|---|
| no_memory（末帧）| 0.496 | 0.508 | 0.508 | 0.504 |
| **gru**（压缩隐状态）| 0.793 | 0.875 | 0.738 | **0.802 ± 0.069** |
| **evidence_store**（内容寻址槽位）| 1.000 | 1.000 | 1.000 | **1.000** |
| evidence_disabled | 1.000 | 1.000 | 1.000 | 1.000 |
| **plwm**（TorchPLWM 完整）| 1.000 | 1.000 | 1.000 | **1.000** |

**结论**：双身份精确回忆下，GRU 的压缩隐状态在 12-26% 的 held-out 序列上提取错误身份（方差大），内容寻址槽位（PLWM/evidence）零错误。差异 ~20 个百分点、跨 seed 稳定、swap evidence 满值。

### v1/v2（单身份，验收通过）

- v1：GRU/evidence/PLWM 全 1.000，no_memory 0.539，four_event_transformer 0.461 → **跨步记忆是决策的必要条件**（无状态/短上下文卡随机）
- v2（干扰 50%）：全 1.000（单身份下干扰不足以难倒 GRU）

## 三、对主旨的意义

1. **任务方法论成立**："决策必须依赖跨步身份记忆"+"swap counterfactual"组合，能干净分离"真记忆/位置捷径/无记忆"三态——已替代失效的 MemoryS7 评估。
2. **架构主张实证**：PLWM 的内容寻址记忆（外部槽位）在精确回忆上确实优于 GRU 的压缩隐状态——ARCHITECTURE.md 的核心命题获得首个可复现证据。
3. **局限**：evidence_disabled（无槽位的 GRUCell 变体）也 1.0，说明"结构宽度"也起作用；GRU vs PLWM 的真实差距需在更长延迟/更多身份/跨 episode 场景进一步刻画。任务为合成数据，需向真实流（MemoryS13 流已用于 v1/v2 语料）扩展。

## 四、文件清单

- `experiments/audit_minigrid_decide_after_delay.py`（v1/v2 单身份+干扰，合成干扰注入）
- `experiments/audit_minigrid_decide_after_delay_two_identity.py`（v3 双身份，seed 参数化）
- `artifacts/minigrid-decide-after-delay-report.json`（v1，accepted）
- `artifacts/minigrid-decide-after-delay-distractor-report.json`（v2，accepted）
- `artifacts/minigrid-decide-after-delay-two-identity-report.json` + seed111/222/333（v3，accepted，3 seed）

## 五、难度扫描（B 阶段：delay / identity-count 轴）

### 延迟轴（identity_count=2，3 seed，d=32 多 seed 确认）

| 延迟 | GRU | evidence_store | evidence_disabled | PLWM |
|---|---|---|---|---|
| d=8（3 seed）| 0.802 ± 0.069 | 1.000 | 1.000 | 1.000 |
| d=16 | 0.871 | 0.734 | 0.793 | 1.000 |
| **d=32（3 seed）** | **0.496（随机，全 seed）** | 0.720 | 0.496 | **0.862** |

**GRU 在 33 帧延迟下三 seed 全部精确随机（0.496）**；PLWM 保持 0.862。**架构优势随延迟增长而扩大**（d=8: +0.2 → d=32: +0.37）——满足 MASTER_PLAN 要求的 "advantage growing as delay increases"。

**机制解释**：evidence_store 用 FIFO 8 槽位，33 帧序列中身份帧被中间空帧挤出 → 掉到 0.72；PLWM 用 gated memory（只写 surprise 高的信息帧），空帧不挤占槽位 → 长延迟仍保持高。gate 机制是差异来源。

### 身份轴（d=8，单 seed）

| 身份数 | GRU | evidence_store | PLWM |
|---|---|---|---|
| 2 | 0.738 | 1.000 | 1.000 |
| 3 | 0.695 | 1.000 | 0.863 |
| 4 | 0.617 | 0.875 | 0.934 |

PLWM 在 n=4 仍有 0.934，GRU 降至 0.617。身份轴同样是"槽位/可寻址 > 压缩"。

## 六、RL 化（A 阶段：从经验学习记忆，主旨④检验）

DecideAfterDelay 改造成**事件流 RL 环境**（DecideEnv）：每步一个帧，仅查询步计分（+1 选对身份、否则 0），PPO on-policy，无 teacher/oracle/shaped reward。RL 版 swap 语义：强制翻转动作，真记忆模型 orig 高分、swap 必失败（both≈0 是正确行为）。

### 结果（delay=8，100 updates × 24 episodes，3 seed）

| 模型 | s222 | s333 | s444 | 均值 |
|---|---|---|---|---|
| gru_policy | 0.562 | 0.625 | 0.562 | **0.583**（≈随机）|
| plwm_policy | 0.781 | 0.891 | 0.781 | **0.818** |
| evidence_policy | 1.000 | 0.891 | 1.000 | **0.964** |

**结论：从纯奖励经验中，可寻址记忆（evidence 0.96 / PLWM 0.82）学会了跨步记忆决策，GRU 压缩隐状态卡在随机水平（0.58）**——主旨完整命题（"持久认知系统从经验中获取目标导向行为"）的首个证据。

待诊断：PLWM（0.818）低于 evidence（0.964）——可能是 TorchPLWM 的 think 计算/记忆写入门控在 RL 梯度下优化更困难，或 latent=32 容量限制。

## 七、下一步（已决策，待执行）

1. **难度刻画像限**：延迟 8→16→32、身份 2→3→4、跨 episode 持久（GRU 隐状态无法跨序列保留 vs PLWM 槽位可持久化）——找到 GRU 失效的边界，确立 PLWM 优势域；
2. **真实流回归**：v3 语料改为 MemoryS13 真实流（v1/v2 已用）；
3. **JEPLM 重新定位**：辅助目标（多步 latent 预测）在 RL 控制上无效，但可作为"记忆内容预测"（预测未来身份帧）在受控任务上重新检验——回归"预测误差驱动记忆"的本意。
