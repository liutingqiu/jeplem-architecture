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
