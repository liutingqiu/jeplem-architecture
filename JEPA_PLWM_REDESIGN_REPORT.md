# PLWM × JEPA 架构重设计报告

> 版本：v2.0
> 日期：2026-08-08
> 性质：架构重设计 + 定制化实验规划（本报告交付"架构设计"与"实验内容"两项，不含代码实现）
> 状态：待评审

---

## 摘要

本报告基于 JEPA 家族最新进展（V-JEPA 2、LeJEPA、JEPA-SCORE、线性可识别性证明）对 PLWM 进行架构重设计。**核心结论：采纳 JEPA 的"学习目标层思想"（latent 空间预测、层级多尺度、显式反坍缩、mask 辅助目标、离线预训练补充），拒绝 JEPA 的"架构缺失"（无记忆、无显式世界模型、纯离线、EMA 训练脆）与"与任务无关的 mask"。**

新架构命名为 **JEPLM (JEPA-Enhanced Persistent Latent Model)**，是 PLWM 的增强而非重写：保留内容寻址记忆、显式世界模型、信念修订、在线 RL 四根支柱，把 JEPA 的预测思想注入训练目标与表征组织，直接瞄准当前 L1 阻塞（延迟关联任务所有模型 50%、short_delay 37.5%）。

配套设计 7 项定制化实验（E-J1~E-J7），全部遵循 MASTER_PLAN.md 证据标准（held-out seed 分离、匹配 baseline、多次运行、失败归档），每项有明确验收线，可在 MiniGrid（已验证的稳定环境）上于 CPU 运行。

---

## 一、现状诊断

### 1.1 PLWM 现有架构（主分支代码）

`plwm/torch_model.py` 的 `TorchPLWM` 是当前可训练参考实现，其单步转换：

```text
encoded = Encoder(observation)                    # latent 表征
prediction = Predictor(belief) -> input_dim       # ⚠️ 在输入空间重建
surprise  = MSE(prediction, observation)          # 输入空间预测误差
retrieved = MemoryRead(QueryHead(encoded), state) # 内容寻址记忆
belief    = GRU(encoded, retrieved, belief)       # 信念更新
workspace = GRU(belief, retrieved, workspace)     # latent-only 思考
```

四大状态域（latent / belief / commitment / control）分离是设计不变量；`plwm/world_model.py` 提供显式 action-conditioned 世界模型与 counterfactual planning；`plwm/multiscale.py` 已有 fast/slow 分离雏形但未与主模型集成；`plwm/regime.py` 提供可审计信念修订。

### 1.2 当前阻塞点

| 任务 | 结果 | 判定 |
|---|---|---|
| addressed_balanced_2id（6 步延迟，Luanti）| 所有模型 50% | 负结果，已归档（任务设计问题）|
| short_delay_2id（3 步延迟，Luanti）| PLWM 3/8 = 37.5% | 未通过（验收线 ≥ 0.75）|
| MiniGrid hybrid curriculum | PLWM 37/64 vs 无记忆 0/64 | ✅ 目前最强证据 |

**根因（已在 DIRECTION_PLAN.md 归档）**：延迟关联任务的终端奖励太稀疏，credit assignment 失效；身份(step 0)与查询(step k)永不同帧，中间无视觉线索。任何 controller 都学不会 → 说明**不是记忆容量问题，而是训练信号问题**。

### 1.3 为什么引入 JEPA

JEPA 的全部价值集中在**训练目标**：它证明"预测语义 latent（而非重建输入）+ 显式分布正则"能学到可识别的世界表征，且不需要稀疏奖励、不需要标签。这恰好是 PLWM 当前最缺的东西——一个能为延迟关联提供密集、自监督训练信号的辅助目标。

---

## 二、JEPA 批判性采纳分析

> 原则：**采纳学习目标层的思想，拒绝架构层的缺失与工程层的脆弱**。
> "不全部接收"体现在三处：① mask 目标从"无任务任意块"裁剪为"任务相关语义时刻"；② 反坍缩拒绝 EMA/stop-grad 启发式，改用显式正则；③ 离线学习从"唯一路径"降级为"在线训练的预训练补充"。

### 2.1 采纳清单（6 项）

| # | 采纳项 | JEPA 出处 | 对 PLWM 的改造 |
|---|---|---|---|
| A1 | **Latent 空间预测**：预测 latent 而非重建输入 | I-JEPA/V-JEPA 2：mask-denoising feature prediction；计算高效、忽略不可预测细节 | 现有 `predictor: latent→input` 改为 `latent→latent`；surprise 在 latent 空间计算，仍驱动记忆写入门控（保留 PLWM 的 surprise-gated 设计，见文献综述门控 2）|
| A2 | **层级多尺度 latent**：fast/slow 时间尺度分离 | V-JEPA 2 多尺度特征（空间 patch 多尺度 → 时序化）| 把 `multiscale.py` 的 fast/slow 提升为一等公民：fast 处理短期动态与即时决策，slow 经 surprise 门控更新，承载持久语义（身份/规则）|
| A3 | **显式反坍缩正则**：表征不能坍缩 | LeJEPA SIGReg（各向同性高斯）；JEPA-SCORE（反坍缩项=密度估计）；A-P1c 已有 key 正交 | 拒绝 EMA/stop-grad（训练脆，LeJEPA §1 明确批评）；扩展已有的 `enforce_key_orthogonality` 为双路正则：记忆键正交（已有）+ latent 方差/各向同性正则（新增）|
| A4 | **Masked 预测辅助目标**：强制学 context→target 映射 | I-JEPA/V-JEPA 2 的 mask 策略 | **裁剪**：不 mask 任意空间块，而是 mask 任务相关语义时刻（身份帧/查询帧），预测其 latent。直接弥补 DQN 稀疏奖励的 credit assignment 缺陷 |
| A5 | **离线预训练阶段**：从历史轨迹自监督学动态 | V-JEPA 2 纯离线自监督 | **降级为补充**：用 replay buffer 预训练 predictor，再在线微调。拒绝"纯离线"，保留在线 RL |
| A6 | **可识别性作为诊断指标**：表征不混淆世界自由度 | Klindt et al.: 预测目标+分布正则 ⇒ 线性可识别性 ⇒ 支持规划/组合泛化 | 把"latent 对世界自由度（身份）的线性探针可恢复性"纳入实验指标，作为架构是否学到干净表征的判据 |

### 2.2 拒绝清单（5 项）

| # | 拒绝项 | 理由 | 替代/保留 |
|---|---|---|---|
| R1 | **纯离线、无动作学习** | 预测目标与决策脱节；V-JEPA 2 需两阶段（预训练+AC 后训练）才能行动 | 保留在线 RL 交互；离线只做预训练补充（A5）|
| R2 | **无显式记忆** | JEPA 无记忆模块，长时预测性能随 horizon 陡降（V-JEPA 2 §13.2）；这正是 PLWM 记忆机制要解决的 | 保留内容寻址记忆（key/value/confidence），已通过 needle-in-slot 检索考试 |
| R3 | **EMA target / stop-gradient 启发式** | 训练脆，存在 complete/dimensional collapse 失败模式（LeJEPA §1）| 用显式正则（A3）：方差/正交约束，每步 QR 投影（已有 E-001 集成）|
| R4 | **无显式世界模型** | JEPA 世界模型是隐式的，无法做可靠多步 counterfactual planning | 保留 `ActionConditionedPLWM` 显式世界模型，升级为 latent 空间 rollout |
| R5 | **强高斯嵌入假设** | 最优性证明依赖高斯/平稳加性噪声；流形数据上非最优（SPHERE-JEPA）| 只保留轻量反坍缩正则，不强求嵌入严格各向同性高斯 |

### 2.3 采纳/拒绝依据的论文勘误

研究过程中勘误了用户笔记 `JEPA_RESEARCH_NOTE.md` 中的论文编号：
- V-JEPA 2 实为 **arXiv:2506.09985**（不是 2505.17594，那是数学优化论文；也非 Nature 发表，Nature 2025 是 V-JEPA 1.5 系另一篇）
- LeJEPA 实为 **arXiv:2511.08544**（不是 2511.00668，那是天体物理论文）

---

## 三、新架构：JEPLM

### 3.1 架构总览

```text
                    ┌──────────────────────────────────────────────┐
                    │              JEPLM Core (persistent)          │
                    │                                              │
 observation ──→ [Encoder] ──→ fast latent f_t                     │
                    │            │                                 │
                    │            ▼                                 │
                    │   [Latent Predictor] ◄─── action a_t         │
                    │     f̂_{t+1}, ŝ_{t+1} = P(f_t, s_t, a_t)      │
                    │            │            (JEPA: latent 空间)  │
                    │            ▼                                 │
                    │   surprise = d([f̂,ŝ], [f_{t+1},s_{t+1}])    │
                    │            │                                 │
                    │   ┌────────┴───────────┐                     │
                    │   ▼                    ▼                     │
                    │ [Slow Gate]     [Memory Write Gate]          │
                    │  s 按 surprise 更新   episodic slot 写入       │
                    │                                              │
                    │  episodic memory: keys / values / confidences│
                    │  retrieved = Attention(M, query(f,s))        │
                    │            │                                 │
                    │            ▼                                 │
                    │  [Masked Predictor]  (task-relevant 语义帧)    │
                    │  [World Model]  (latent 空间 rollout 规划)     │
                    │  [Policy Head]   (DQN / actor-critic)        │
                    └──────────────────────────────────────────────┘

训练目标（加权和）:
  L = L_policy + α·L_latent_pred + β·L_masked + γ·L_anti_collapse + δ·L_world
```

### 3.2 五个核心组件

**C1. 层级潜在状态（A2）**
- `fast latent f_t`：每步更新（现有 belief_cell 升级），编码短期动态、感知细节。
- `slow latent s_t`：surprise 门控更新，编码持久语义（身份、规则、目标）。
- 与现有 `MultiScalePLWM` 的差异：新增组件内部状态流转闭环——slow 更新既驱动记忆写入也参与预测，而非孤立实验。

**C2. Latent 空间预测器（A1）**
- `P: (f_t, s_t, a_t) → (f̂_{t+1}, ŝ_{t+1})`，action-conditioned。
- surprise 定义从 `MSE(predictor(belief), observation)`（输入空间）改为 `d(latent_pred, latent_next)`（latent 空间）。
- 预测误差三重用途保留并强化：① slow gate 触发；② 记忆写入门控（surprise-gated memory，保留文献门控 2 的设计立场）；③ 作为自监督损失 `L_latent_pred`。

**C3. 内容寻址记忆（保留 + R2 反驳）**
- 结构不变（keys/values/confidences + confidence-weighted softmax + 正交初始化）。
- 新增：检索结果**注入预测器**（`P` 的输入含 `retrieved`），使预测可利用精确回忆——这是"记忆 × 预测"协同，是 JEPA 完全没有的能力。
- 反坍缩扩展为双路：记忆键正交（已有 A-P1c）+ latent 方差正则（A3）。

**C4. 显式世界模型（R4 反驳）**
- `ActionConditionedPLWM` 升级：`imagine` 在 latent 空间 rollout（不再只在离散 state logits 上），支持多步 counterfactual planning。
- 规划成本用规范化目标 log-probability（现有 `plan_costed` 保留）。

**C5. 任务条件化 Masked 预测器（A4 裁剪）**
- 在 trajectory 上 mask 任务相关语义帧（身份帧、查询帧），用 context 帧的 latent 预测被 mask 帧的 latent。
- 与 JEPA 的区别：JEPA 自监督 mask 任意块（无任务目标），JEPLM 的 mask 由任务契约决定（task-relevant），直接对准延迟关联能力。
- 实现为辅助损失 `L_masked`，不改变主策略损失。

### 3.3 训练流程（在线为主、离线补充）

```text
阶段 0（可选）：离线自监督预训练
  replay buffer 轨迹 → 训练 C2 预测器 + C5 masked 预测器（纯自监督，无奖励）
阶段 1（在线）：RL 联合训练
  与环境交互 → 轨迹入库 → 优化 L（policy + 各辅助项）
  （α, β, γ, δ 从 0 逐步 warm-up，避免辅助目标干扰主任务初期学习）
阶段 2（评估）：held-out worlds + 多 seed + 消融
```

### 3.4 与现有代码的映射（迁移路径）

| 新组件 | 现有基础 | 改动 |
|---|---|---|
| C1 层级 latent | `plwm/multiscale.py` + `TorchPLWM.belief_cell` | 集成 fast/slow 到 `TorchPLWM` 主状态；slow 门控复用 surprise |
| C2 latent 预测器 | `TorchPLWM.predictor`（latent→input）| 重写为 latent→latent；surprise 定义改动 |
| C3 记忆 | `TorchPLWM` keys/values/confidences | 保留；检索注入预测器输入 |
| C4 世界模型 | `plwm/world_model.py` | `imagine` 升级 latent 空间 rollout |
| C5 masked 预测器 | 无 | 新增模块 + `masked_prediction_loss` |
| A3 反坍缩 | `enforce_key_orthogonality`（E-001 已集成）| 新增 latent 方差正则项 |
| A5 离线预训练 | 训练脚本的 replay buffer | 新增预训练入口脚本 |

新文件建议：`plwm/jeplem.py`（模型）+ `experiments/train_jeplem_*.py`（实验脚本，位于主分支 experiments/ 目录）。

---

## 四、定制化实验体系

> 环境选择：**MiniGrid 为主环境**（理由：hybrid curriculum 已有 37/64 成功先例；CPU 可运行；Luanti 服务器稳定性问题已消耗过多调试时间）。L2 阶段扩展到 Crafter（`plwm/crafter_env.py` 已存在）。所有实验遵循 Evidence Standards：契约先行、held-out seed 分离、匹配 baseline、多 seed 运行、失败报告归档。

### 4.1 任务族：延迟关联（Delayed Association）

`DelayedObjectIdentity`：身份对象在 step 0 可见 → k 步延迟（中间含 distractor）→ step k 查询 → step k+1 决策。
变体轴：延迟长度 k∈{3,5,7}、distractor 数量、双身份（2id）、观测噪声。

### 4.2 实验卡（E-J1 ~ E-J7）

**E-J1｜门控验证（Gating）**
- 目的：JEPLM 是否解决 short_delay 阻塞（3 步延迟）
- 对比：PLWM（现架构）/ JEPLM（无辅助损失）/ JEPLM-full / GRU / 无记忆
- 验收：JEPLM-full held-out ≥ 0.75 且 vs PLWM gap ≥ 0.10；vs 无记忆 gap ≥ 0.15

**E-J2｜延迟缩放（Delay Scaling）**
- 目的：架构优势是否随延迟增长（直接检验 Completion Rule 的 "advantage growing as delay increases"）
- 任务：k = 3 / 5 / 7
- 验收：JEPLM 在 k=7 held-out ≥ 0.60；优势随 k 单调扩大（gap(k=7) > gap(k=3)）

**E-J3｜辅助损失消融（Auxiliary Loss Ablation）**
- 目的：每个 JEPA 采纳项（α/β/γ）的独立贡献
- 配置：full / -α / -β / -γ / 纯 policy（=JEPLM 无辅助）
- 验收：每项非负贡献；full 最优；若某项为负 → 记录并调整权重而非掩盖

**E-J4｜预测空间对照（Prediction Space）**
- 目的：**直接检验 JEPA 核心主张**——latent 空间预测 vs 输入空间重建预测（同容量、同预算）
- 对比：现有 `predictor: latent→input` vs 新 `latent→latent`
- 验收：latent 预测在 k≥5 延迟任务上 held-out 显著更优（≥ 0.05）；若打平 → JEPA 主张在低维特征输入下不成立，记录为对该主张的边界刻画

**E-J5｜离线预训练价值（Offline Pretraining）**
- 目的：检验 A5（离线自监督预训练是否提升样本效率）
- 配置：纯在线 vs 离线预训练 C2+C5 + 在线微调（相同交互预算）
- 验收：预训练版本在相同 held-out 达标所需交互步数 ≤ 纯在线 × 0.7（样本效率提升 ≥ 30%）

**E-J6｜可识别性探针（Identifiability Probe）**
- 目的：检验 A6——训练后 latent 是否线性可识别地编码世界自由度（身份）
- 方法：冻结 latent，训练 linear probe 预测身份；对比 PLWM vs JEPLM
- 验收：JEPLM 探针准确率 > PLWM ≥ 0.10；并与下游 held-out 表现相关（探针是规划的代理指标）

**E-J7｜记忆 × 预测协同（Memory-Prediction Synergy）**
- 目的：检验 C3 的"检索注入预测器"是否优于"检索只喂策略"
- 配置：检索注入预测 vs 不注入（其余相同）
- 验收：注入版本在需要精确回忆的任务（带 distractor 的 k=5）上 held-out 更优 ≥ 0.05

### 4.3 证据与判定规则

- 每个实验：契约先行（写入 dispatch 卡）→ 训练与 held-out seed 分离 → ≥ 3 seed 运行记录方差 → 报告 JSON 到 `artifacts/` + replayable trajectory。
- 任一实验未达验收线 → 按 MASTER_PLAN Completion Rule 写失败报告，区分"机制失效"与"任务不可学"（如 addressed_balanced_2id 已证后者）。
- **E-J1 失败即止损**：若 JEPLM-full 在 3 步延迟仍 < 0.60，停止 JEPA 路线，回到架构级重新评估，不继续堆超参。

---

## 五、实施路线图

| 阶段 | 内容 | 工期 | 出口 |
|---|---|---|---|
| P0 | 代码勘误：`JEPA_RESEARCH_NOTE.md` 论文编号更新 | 0.5 天 | 文档修订 commit |
| P1 | 实现 `plwm/jeplem.py`（C1+C2+C3 集成，先不带辅助损失）| 2-3 天 | 单测通过（复用 tests/ 骨架）|
| P2 | C5 masked 预测器 + A3 latent 反坍缩 + 损失加权 | 2 天 | 训练脚本可跑通 smoke |
| P3 | 实验 E-J1~E-J4（MiniGrid）| 3-5 天 | 报告 JSON + 裁决 |
| P4 | 实验 E-J5~E-J7 + 消融 | 3-5 天 | 完整实验报告 |
| P5 | 视 P3/P4 结果决定：通过 → 扩展到 Crafter/Luanti；失败 → 归档 + 重新设计 | 1 天 | 决策记录 |

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| JEPLM 在 3 步延迟仍失败 | 中 | 高 | E-J1 止损线；失败报告区分机制/任务；MiniGrid 已有成功先例降低整体风险 |
| 辅助损失干扰主任务 | 中 | 中 | 权重 warm-up（α/β/γ 从 0 渐增）；E-J3 消融定位 |
| latent 空间 surprise 定义退化（预测误差失去信息量）| 低 | 中 | E-J4 对照输入空间版本；保留输入空间距离作为诊断辅助指标 |
| 离线预训练样本效率提升不达 30% | 中 | 低 | E-J5 是增益验证不是硬依赖；不达标则说明在线已足够，记录边界 |
| 训练不稳定 | 低 | 中 | 梯度裁剪、QR 投影（已有）、损失缩放 |

---

## 七、结论

1. **JEPA 的价值是训练目标，不是架构**——"预测语义 latent + 显式反坍缩 + mask 辅助"直接针对 PLWM 当前阻塞（稀疏奖励 credit assignment 失效）的根因。
2. **JEPLM 是 PLWM 的增强不是重写**——四根支柱（记忆/世界模型/信念/在线 RL）全部保留，JEPA 只改"怎么学"，不改"系统是什么"。
3. **裁剪而非照搬**——mask 任务条件化、反坍缩用显式正则、离线只做预训练补充、不强求高斯嵌入。这四处是"不全部接收"的落点。
4. **每项采纳都有可证伪实验**——E-J1~E-J7 逐项检验，验收线明确，失败即归档。符合项目 Completion Rule 的证据导向精神。

**下一步（待确认后执行）**：P0 代码勘误 + P1 实现 `plwm/jeplem.py`。

---

### 附录：关键参考文献（勘误后）

1. Assran et al. "Self-Supervised Learning of Visual Representations with I-JEPA." CVPR 2023 (arXiv:2301.08243).
2. Bardes et al. "V-JEPA 2: Self-Supervised Learning of Hierarchical World Models." arXiv:2506.09985, 2025.
3. Balestriero et al. "Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density." arXiv:2510.05949, 2025.
4. LeCun et al. "LeJEPA: Toward a Universal and Simple Joint-Embedding Predictive Architecture." arXiv:2511.08544, 2025.
5. Klindt et al. "When Does LeJEPA Learn a World Model?" arXiv:2605.26379, 2026.
6. V-JEPA 2-AC / action-conditioned latent world models（零样本 MPC 规划）, arXiv:2506.09985 §3-4.
7. Balestriero et al. "SPHERE-JEPA"（流形数据上高斯嵌入非最优）, arXiv:2605.26900.
