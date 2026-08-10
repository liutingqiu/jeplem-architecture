# JEPLM 系统集成层设计文档（2026-08-11）

> 性质：集成层（图纸验证层之后）的决策与验证记录。原则：需求驱动、最小可行接线、每次只接一个模块、带开/关对照 + 3 seed + swap 探针。
> 关联：CURRENT_STATE.md（交接指针）、JEPLM_IMPLEMENTATION.md（蓝图/待集成项）、NIGHT_RUN_20260809.md（阶段证据）、COGNITIVE_SYSTEM_WHITEPAPER.md（12 区块）。

---

## 一、集成原则（从图纸层负结果提炼）

1. **需求驱动，不理论驱动**：接口只在任务需求出现时设计。前车之鉴——JEPA latent 预测（理论合理、5 轮中性）、surprise 清槽修订（理论合理、RL 有害）都是"无任务支撑的机制先行"的教训。
2. **最小可行接线（MVP）**：每次只接入一个模块到一个接缝点，用主战场任务验证，不做大爆炸式组装。
3. **修订动作 ≠ 触发信号**：阶段 35 证明"连续失败检测"（信号）是有效的，但"清槽"（动作）有害（after 0.292 vs 0.396，记忆存翻转不变的身份，清槽纯丢信息）。集成时信号与动作分开设计。

## 二、勘察结论（2026-08-11）

- **骨架已组装**：JEPLMAgent = 冻结 CNN 感知（可选）→ JEPLMMemory（内容寻址读 + 学习门控写 + 语义 aux）→ policy head。数据流：obs → 感知 → 记忆（读-更新-写）→ 策略头 → 动作。
- **待集成模块 11 个**（belief/commitments/fact_ledger/continual/adaptive/multiscale/regime/latent_planner/world_model/events/evidence/core），全部独立 toy 验证、未接入骨架；runtime/world_runtime 是另一条已自组装的子线（参考实现）。
- **关键缺口（勘察发现）**：JEPLMMemory 内容寻址读的 query **只来自观测 encoder**（`self.query(encoded)`），无独立 query 注入点——显式记忆系统缺少"目标驱动检索（主动回忆）"能力。

## 三、集成第①刀：目标驱动检索（goal → query 调制）

### 3.1 动机与证据

- **接缝**：内容寻址读的 query 来源（jeplem_memory.py:113-117）。加性调制不改变主干结构，可插拔。
- **任务零改动**：DecideStream 观测 = `[身份线索 2 维, selector 2 维]`；查询帧的 selector（0=问当前 unit 身份, 1=问上一 unit 身份）**天然是目标通道**。不需要构造新任务即可验证。
- **认知对应**：cue-based memory retrieval——查询目标（问谁）显式调制记忆寻址，区别于"整帧观测被动编码为 query"。
- **增益假设（可证伪）**：显式目标调制让 attention 聚焦"当前/上一"槽位 → recalled 更干净 → **cross-episode 精度提升**（骨架弱点：0.941/0.831，cross 0.831）。
- **接口基础**：本刀把"selector（任务给定的目标）→ query"接成可插拔接口，后续 gap_goals（区块 4，自主目标）直接复用同一注入点——目标来源从任务给定升级为自主生成，接口不变。

### 3.2 设计

```
JEPLMMemory(goal_dim=0)         # 可选：目标投影层 goal_proj: Linear(goal_dim, latent)
step(obs, valid, goal=None)     # goal [B, goal_dim]；query_vec = W_q(encoded) + W_g(goal)
```

- `goal_dim=0` 时行为与现状完全一致（默认关闭，向后兼容）。
- JEPLMAgent 增加 `use_goal_query`：从观测的 query 段拆出 selector 通道作为 goal。
- 实验入口：`--goal-query` 开关（jeplem_agent_policy）。

### 3.3 验证协议

| 项 | 设定 |
|---|---|
| 任务 | DecideStream u3g4（合成流主战场） |
| 对照 | goal-query 开 / 关（同 seed 同初始化结构） |
| seed | 444 / 555 / 666（项目纪律 3 seed） |
| 指标 | q 均值 / c 均值（cross-episode）/ swap 探针 |
| 判定 | 增益 ≥ +0.02 且 swap 探针不退化 → 保留；中性 → 记录但降级；有害 → 关闭 |

### 3.4 风险与诚实预期

- selector 信息本就在观测里（W_q 理论上可学），此刀是**归纳偏置 + 接口铺设**，增益可能是小或中性——中性也接受（接口为后续 gap_goals 铺路），但须诚实记录。
- 不改任务、不改评估纪律，无混淆变量。

### 3.5 第①刀结果（2026-08-11，中性）

| 模式 | s444 | s555 | s666 | q 均值 | c 均值 |
|---|---|---|---|---|---|
| off（基线） | 0.969/0.910 | 0.958/0.881 | 0.958/0.881 | **0.962** | **0.891** |
| on（goal-query） | 0.958/0.881 | 0.958/0.881 | 0.958/0.881 | **0.958** | **0.881** |

**结论（诚实中性）**：无增益（-0.004/-0.010，seed 波动范围内）、无损害。与预期一致——selector 信息本就在观测里，加性 query 调制不改变信息内容。**机制降级为可选组件**（`--goal-query` 保留供研究），**接口保留**：goal 注入点是后续 gap_goals（自主目标）的寻址基础设施，目标来源从"任务给定 selector"升级为"自主生成"时接口不变。
- 产物：`artifacts/int1-goal-{off,on}-s{444,555,666}.json`

**诊断启示**：query 来源不是记忆瓶颈 → 寻址瓶颈在写入侧（FIFO 无条件轮转 vs 学习门控写），引出第②刀。

## 六、第②刀：write_mode gate vs fifo（RL 验证，负结果）

**动机**：蓝图声称"学习门控写"是主干（GRU 融合 + FIFO 位置 + sigmoid gate），但实验默认是 fifo（无条件轮转），gate 从未在 RL 下对照验证。

| write_mode | s444 | s555 | s666 | q 均值 | c 均值 |
|---|---|---|---|---|---|
| **fifo**（默认） | 0.969/0.910 | 0.958/0.881 | 0.958/0.881 | **0.962** | **0.891** |
| gate（学习门控写） | 0.755/0.612 | 0.797/0.687 | 0.807/0.448 | **0.786** | **0.582** |

**结论（诚实负结果）**：gate 大幅有害（q -0.176 / c -0.309，3 seed 一致方向）。RL 稀疏梯度将学习式门控学坏（无法从稀疏奖励学出"该写谁"），无条件 FIFO 是 RL 下的稳健正确默认——与阶段 35（门控/修订类机制 RL 有害）一脉相承。**fifo 保留为唯一默认**，gate/delta_rule 标记为 RL 下已证伪（`--jeplem-write-mode` 保留供研究）。
- 产物：`artifacts/int2-gate-s{444,555,666}.json`（对照基线 int1-goal-off-s*）
- 蓝图修订：JEPLM_IMPLEMENTATION.md 的"学习门控写"表述更新为"无条件 FIFO 为 RL 实证正确（2026-08-11 gate 对照证伪）"。

**元结论（两刀后）**：合成流主战场已局部饱和——显式机制（goal query、gate 写）均无增益或有害。**骨架在简单合成流上接近最优，集成验证需要更难任务**（真实流 / 复合任务）——见下一步。

## 七、下一步（第③刀候选）

1. **真实流验证**（train_decide_stream_real_rl.py）：合成流饱和掩盖机制价值；真实流（0.774/0.577）改进空间大，goal-query/gate 可在真实流复测（低成本，复用现有开关）。
2. **复合任务**（CURRENT_STATE 推荐）：多规则/多目标任务，让 gap_goals/信念/世界模型同时被需要——任务构造投入大，需谨慎防退化捷径。
3. **与 evidence_aux 对齐**：本次 off 基线 0.962/0.891 已接近 evidence_aux（0.965/0.900），差距基本被 seed 效应覆盖，优先级降低。

## 八、复合任务设计：EvidenceSeek（主动证据收集，2026-08-11）

**任务**：K=2 身份（Key/Ball），每 unit = [线索帧, gap 空帧, 查询帧]。线索帧显示某个身份的带噪线索（p=0.65 真通道）；查询帧 agent 四类动作：答身份 0 / 答身份 1（+1 并结束）/ 求线索 0 / 求线索 1（下一 unit 该身份线索）。总 units=6 超时无奖励。

**需求驱动论证**（每个模块为何真实被需要）：
| 模块 | 任务需求 | 退化捷径（被机制惩罚） |
|---|---|---|
| 显式槽位记忆 | 跨帧累积带噪线索（无记忆正确率 0.65 封顶） | 读输入抄单帧 → 噪声不可靠 |
| 信念（CategoricalBelief） | 每身份后验置信度 → 缺口 = 1 - confidence | — |
| 自主目标（gap_goals） | 请求缺口最大身份的线索（max-gap，原样迁移 toy 逻辑） | 随机请求浪费在已确定身份 |
| 置信度阈值决策 | 何时停止收集、回答 | 永远请求 → 超时无奖励；只答一个 → 被问另一必错 |

**设计决策**：
- 信念用 `plwm/belief.py` 的 `CategoricalBelief`（连续贝叶斯、可微），gap 特征拼入策略 head 输入（决策侧接线，同 conf_feature 模式）；不用 BeliefCell（二值翻转与连续缺口不匹配）
- 请求动作的"目标选择" = gap_goals 的 choose_goal 逻辑，评估时用探针验证"请求目标 == max-gap 身份"的匹配率
- 对照：JEPLM+gap 请求 / JEPLM+随机请求 / JEPLM+固定顺序请求 / GRU 无记忆基线；3 seed + 探针
- 预期：gap 驱动请求 > 随机/固定（效率，同样单位数内正确率更高）；JEPLM > GRU（记忆价值）

**实现清单**：①环境类 EvidenceSeekEnv（参考 DecideStreamEnv）②训练脚本 train_evidence_seek_rl.py（PPO 循环参考 train_decide_stream_rl.py）③对照策略 + 评估探针。**先验证无捷径基线**（GRU 应 ≤0.65 正确率上限验证任务确实需要记忆），再跑模块对照。

## 九、复合任务 EvidenceSeek 诊断（v1-v4，学习失败 + 架构发现，2026-08-11）

**结果**：RL 100 updates 三模型均停 0.5（随机）；监督探针（含 25600 样本 batch 训练）JEPLMMemory aux 头 loss 恒 0.693（均匀分布）——**任务学习失败，诊断为架构级**。

**v1-v4 修正链**（每版修一个独立缺陷，全部排除后问题仍在）：
1. v1：线索帧每次 `_observation` 调用重新随机采样（观测不稳定）→ 预生成修复
2. v1：求动作的线索与随机线索可靠性相同（**边际价值 ≈ 0**，RL 学"直接答"合理）→ v4 改"线索只由请求产生"（不请求 = 无信息，答只能猜）
3. 记忆 mask bug：`model.step(obs, active)` 把"流存活"当"记忆写入 mask"，空帧/查询帧冲掉线索槽 → 改只在线索帧写；GRU head 缺当前观测（查询帧 X 通道丢失）→ 修复
4. **终极发现（架构级）**：v4 信息完整 + 归属通道 + goal 调制下，监督 25600 样本仍 loss 恒定——JEPLMMemory 的 **attention 内容寻址未学到**

**架构发现（本诊断最重要的产出）**：
- DecideStream 的 0.94 **从未真正考验内容寻址**——aux 预测"最近身份"可由 `state`（GRU 融合）兜底，recalled 混不混合不影响
- EvidenceSeek 首次逼出"**从多槽按 X 精确寻址**"的真需求（身份混叠必须靠 query 聚焦正确槽），而隐式 attention 对齐（encoder 学 query↔key 映射）监督下不收敛
- **"内容寻址"支柱的实证基础弱于预期**：显式槽位记忆的"寻址"能力需要独立验证/设计（候选：显式 key 解耦、槽 key 与内容分离、寻址预训练），列为集成层后续专题

**诚实记录**：任务学习失败不归因于任务构造（v4 信息完整、env 逐帧验证正确、oracle 上界 ~0.61），归因于 RL 延迟 credit（求→答链）+ 记忆寻址不收敛的双重困难。**EvidenceSeek 任务暂停**，`experiments/train_evidence_seek_rl.py` 保留为诊断资产（含完整探针），后续若做"显式寻址"研究可复用。

## 四、决策记录（已排除方向及理由）

| 方向 | 排除理由 | 证据 |
|---|---|---|
| 信念 → 清槽记忆修订 | 修订动作有害（记忆存翻转不变的身份） | NIGHT_RUN 阶段 35（after 0.292 vs 0.396） |
| 信念 → 规则跟踪切换 | DecideStream 的 switch_at 是 **episode 内分段**（每 episode 翻转一次、规则由 unit 位置确定性决定），无"世界变化不确定"可学——信念无观测空间 | 任务结构勘察（train_decide_stream_rl.py:90-95） |
| 信念软门控（surprise_feature 接入 gate） | surprise 信号探针已验证中性 | JEPLM_IMPLEMENTATION 第四节（2026-08-10 探针） |
| JEPA latent 预测接入 | 5 轮中性 + 与记忆保持冲突 | NIGHT_RUN 阶段 28-32 |

## 五、后续候选（P1，待任务需求出现再接）

1. **gap_goals（区块 4）→ 自主目标生成**：复用第①刀的 goal 注入点，目标来源从任务 selector 升级为置信度缺口驱动的自主目标。需要"多候选目标"任务场景。
2. **执行控制（区块 9）→ 预算注意力**：写入门控加"值得写才写"的预算约束。需要长流高噪声任务。
3. **世界模型（区块 7/12）→ 反事实规划**：world_runtime 管道已存在，接入骨架需多步规划任务。
4. **与 evidence_aux 的微小差距**（-0.03/-0.07，JEPLM_IMPLEMENTATION 待集成项 1）：初始化/gate 输入/read query 来源对齐——调优项，可随时做。
