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
| d=16（1 seed）| 0.871 | 0.734 | 0.793 | 1.000 |
| **d=32（3 seed）** | **0.496（随机，全 seed）** | 0.720 | 0.496 | **0.862** |

**GRU 在 33 帧延迟下三 seed 全部精确随机（0.496）**；PLWM 保持 0.862。**架构优势随延迟增长而扩大**（d=8: +0.2 → d=32: +0.37）——满足 MASTER_PLAN 要求的 "advantage growing as delay increases"。

> **⚠️ 补测修正（2026-08-09）**：d=32 的 PLWM 数值在三个独立批次（scan / rmt / rmt-full）中共测得 **9 个 seed**：`0.793, 0.793, 1.0, 1.0, 0.875, 0.793, 1.0, 0.734, 0.609`（中位 **0.793**、均值 0.844±0.136、范围 0.61–1.0；各批均值 0.862/0.889/0.781）。**方向性结论稳（9/9 seed PLWM 均胜 GRU 的 0.496），但 PLWM d=32 的单批精确值方差大、不应作为"0.86"级数字引用**；本表 0.862 为 scan 批 3 seed 均值。GRU d=32 为 9/9 seed 精确随机（0.496，0 方差）。汇总见 `artifacts/d32-variance-aggregate.json`、`experiments/audit_d32_variance_aggregate.py`。

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

### 差距诊断（3 seed，100 updates）

| 模型 | s222 | s333 | s444 | 均值 |
|---|---|---|---|---|
| evidence_policy（无预测头）| 1.000 | 0.891 | 1.000 | **0.964** |
| **evidence_pred_policy（+predictor/surprise）** | 1.000 | 0.797 | 0.797 | **0.865（↓0.10）**|
| plwm_policy（latent=32, 全写）| 0.781 | 0.891 | 0.797 | 0.823 |
| plwm_latent64_policy | 0.781 | 0.891 | 0.891 | 0.854 |
| plwm_gated_policy | 0.797 | 1.000 | 0.891 | 0.896 |

**结论（架构修正证据）**：
1. **主因是 surprise 重建目标**：给 evidence 加 predictor 头后掉 0.10（0.964→0.865）——预测误差目标在纯 RL 策略学习中是噪声/负担；PLWM 自带 predictor 所以系统性低于 evidence；
2. 容量有影响（latent=32→64: 0.823→0.854）；
3. 写入门控非主因（gated_memory 效果随 seed 波动，0.854-0.896）；
4. **surprise 的价值是任务依赖的**：长延迟监督任务（d=32 scan）中 PLWM 的 surprise 门控优于 FIFO（0.862 vs 0.720），但在 RL 策略学习中是负担。
5. **架构含义**：surprise 应作为门控/置信度/新奇检测的辅助信号（决策侧），而非作为通用训练目标强制混合——预测误差驱动写入保留，但不应进入策略梯度主损失。

## 七、跨 episode 持久（连续流，units=3 gap=4，6 seed 补测）

连续事件流：每 episode [身份帧, gap 空帧, 查询帧]，查询 one-hot 指定当前或上一 episode 的身份（cross=上一 episode）。流不 reset，GRU 隐状态被后续 episode 稀释。

### 7.1 原 3 seed（41123/222/333）

| 模型 | query 均值 | cross 均值 |
|---|---|---|
| gru_policy | 0.595 | **0.517（≈随机）** |
| evidence_policy | 0.889 | 0.796 |
| **plwm_policy** | **0.924** | **0.821** |

### 7.2 ⚠️ 补测修正（2026-08-09）：6 seed 同脚本同 seed 重出（41123/222/333 重跑为 6 模型版 + 444/555/666）

| 模型 | n | query 均值 | cross 均值 | 脚本门达标（cross≥0.6 且超 GRU+0.2）|
|---|---|---|---|---|
| gru_policy | 6 | 0.521 | 0.478（6/6 ≈随机）| — |
| evidence_policy | 6 | 0.932 | 0.806 | 4/6 |
| plwm_policy | 6 | 0.909 | 0.751 | **6/6** |
| plwm_slots16_policy | 6 | 0.892 | 0.779 | 6/6 |
| rmt_mem4 / rmt_mem8 | 6 | 0.823 / 0.823 | **0.492 / 0.492（≈随机）** | **0/6** |

**结论修正**：
1. **原"PLWM query 精度第一（0.924）"在 6 seed 下不成立**：PLWM query 0.909 ≈ evidence 0.932（打平），cross 0.751 低于 evidence 0.806——3 seed 的"第一"是抽样波动。**稳的结论是"可寻址记忆（PLWM/evidence/slots16）≫ GRU"，不是"PLWM 优于 evidence"**。
2. **RMT 在跨 episode 连续流上结构性崩盘（gate 0/6，cross≈随机）**——RMT 的 query（段内）0.82 正常但 cross（跨段）全丢：段边界 detach（truncated BPTT）使跨段记忆不可学。RMT 只在监督精确回忆/规则修订（有界序列）与 PLWM 打平；**流式无界场景是 PLWM 相对 attention 记忆家族的实证差异化**（详见 `RMT_STREAMING_COLLAPSE_REPORT.md`）。
3. GRU 6/6 seed 全部 ≈随机（cross 0.478），"压缩隐状态被后续 episode 稀释"的机制结论成立。
4. 汇总：`artifacts/stream-seed-aggregate.json`、`experiments/audit_stream_seed_aggregate.py`。

至此证据链完整：**监督（3 任务）→ 难度扫描（延迟/身份轴）→ RL 单 episode → RL 连续流**，四个维度一致——需要精确跨步回忆时，**可寻址记忆 > 压缩隐状态（GRU/RMT 流式版）**；PLWM 与 evidence 在连续流上等价，PLWM 的差异化价值是 gated memory 的认知语义与长间隔鲁棒性（§八），而非绝对精度。

## 八、连续流外推（更长流 / 更大间隔，3 seed）

> 注：下表中 u=3 g=4 基线行为 3 seed 口径（41123/222/333，见 §7.1）；u=5 与 u=3 g=8 为各自 3 seed。外推结论（间隔是 PLWM 正向轴、超长流是容量/干扰边界）未受 6 seed 补测影响。

| 配置 | GRU query/cross | evidence query/cross | **PLWM query/cross** |
|---|---|---|---|
| u=3 g=4（基线）| 0.595 / 0.517 | 0.889 / 0.796 | 0.924 / 0.821 |
| u=5 g=4（更长流）| 0.529 / 0.520 | 0.726 / 0.604 | 0.775 / 0.504 |
| **u=3 g=8（更大间隔）** | 0.507 / 0.493 | 0.590 / 0.517 | **0.896 / 0.716** |

**两个结论**：
1. **间隔（gap）是 PLWM 优势的正向轴**：g=4→8 时 evidence 的 cross 暴跌（0.796→0.517），PLWM 相对跌幅小得多（0.821→0.716），g=8 下 PLWM 优势扩大（cross +0.20 vs evidence）；u=3 g=8 三 seed 全部 0.90/0.72（query/cross），极其稳定。
2. **超长流（u=5，25 帧）是容量/干扰边界**：所有模型 cross 崩（PLWM 0.504、evidence 0.604）；**slots=16 不恢复（0.501）——非槽位数问题**，是更长流中干扰 episode 稀释 + cross 信号占比下降（u=5 时 cross 查询 40% vs u=3 时 33%）导致的学习/检索困难，非 PLWM 特有问题。

## 九、下一步（已决策，待执行）

1. **难度刻画像限**：延迟 8→16→32、身份 2→3→4、跨 episode 持久（GRU 隐状态无法跨序列保留 vs PLWM 槽位可持久化）——找到 GRU 失效的边界，确立 PLWM 优势域；
2. **真实流回归**：v3 语料改为 MemoryS13 真实流（v1/v2 已用）；
3. **JEPLM 重新定位**：辅助目标（多步 latent 预测）在 RL 控制上无效，但可作为"记忆内容预测"（预测未来身份帧）在受控任务上重新检验——回归"预测误差驱动记忆"的本意。
