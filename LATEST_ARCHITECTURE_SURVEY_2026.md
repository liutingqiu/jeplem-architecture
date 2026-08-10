# 2025–2026 最新架构情报调研（2026-08-09）

> 4 路子代理并行调研（神经记忆 / 世界模型-JEPA / 持续学习-agent 记忆 / RL 记忆-稀疏奖励），
> 全部 arXiv 编号经官方页验证。目的：核查 PLWM 差异化定位在当下最新架构面前是否成立。

## 一、最重要的三个结论

### 1. surprise 门控已"主流化"——不是我们的独有卖点（威胁）
2025–2026 至少 4 个独立系统用"预测误差/梯度=惊奇"做记忆写入：
- **Titans 升级为统一框架 Miras**（2504.13173）：记忆=在线优化，Moneta/Yaad/Memora
- **SuRe**（surprise 优先回放）、**2603.04029**（世界模型残差触发微调）、**WMI**（预测差做域判定）
- 神经科学（PLOS Comp Biol 2025）：meta-RL 统一 surprise/value/control

→ **PLWM 若继续把 surprise 门控当唯一卖点，会被视为 Miras 的特例。** 必须把卖点收敛到
调研 12+ 工作中**无人做全**的组合：独立信念域 + 证据门隔离 + commitment 审计 + 参数冻结外部修订。

### 2. "latent 预测 + 显式可检索记忆"是多方独立收敛的共识（方向背书）
- **HERA**（2608.05523，2026-08-06）：给冻结 V-JEPA 2 加"历史证据路由适配器"
  （Structured Memory Bank + Memory/Workspace Registers）→ 物理预测显著提升——**与我们思路最近的同类工作**
- **Genie 3**：约 1 分钟环境记忆换一致性；**HWM**（2604.03208）：层级子目标=记忆锚点
- JEPA-for-control 谱系 2025–2026 爆发：V-JEPA 2.1（2603.14482）、TD-JEPA、Causal-JEPA、WCM

→ "JEPA 优点 + 记忆"正是主流方向，不是野路子。

### 3. "规则修订：参数冻结 + 外部规则记忆"是无人占据的结构空位（最大机会）
2026 年"规则变化后快速适应"成为显式研究线，但**全部在权重/价值域做手术**：
- **BAPR**：变点检测+信念切换（权重域）；**Value Decay**：显式遗忘；**CPR**（2607.24996）：防塑性崩溃
- **CPR 给了我们关键机制解释**：持续权重训练累积 dormant neurons → plasticity 丧失 → policy collapse——
  我们"纯 PPO 在线适应失败"的实验结论有机械论背书（不是信用分配，是权重续训本身的结构性后果）
- **AlphaEvolve**（2506.13131）：外部解表示迭代修订——"参数不动、外部修订"范式的工业级背书
- **LongMemEval-V2**：451 道"规则变化后是否被旧规则误导"评测题——我们规则切换实验的评测蓝本

→ 我们"知识在状态、修订在记忆"的路线在 2026 无直接先例、也无人竞争。

## 二、其他关键发现

| 领域 | 工作 | 要点 |
|---|---|---|
| 记忆训练效率 | **TNT**（2511.07343） | Titans/TTT 分块并行提速 17×；承认 Titans 仍不如 SOTA Transformer |
| 记忆信号替代 | **TTCD**（2608.01672, 2026-08） | 长窗-短窗预测差异比预测误差绝对值更稳——建议 JEPLM v2 写入门控对照 |
| 事件边界写入 | **HiMem-WAM / MemoryWAM / WorldScape 2.0**（2026-06/07） | 状态突变点写入已工业落地，但无信念域/无修订/无证据门 |
| 记忆可信度 | **TrustMem**（2606.25161） | 记忆更新可信度可用 RL 端到端优化；三类失败（省略/污染/幻觉） |
| 世界模型决策 | **Dreamer 4**（2509.24527） | 世界模型解耦 + imagination RL；离线数据 Minecraft 钻石 |
| JEPA 语言局限 | **The JEPA Paradox**（2607.23531） | 多义文本需保留分布而非单点 latent 预测 |
| 反坍缩深化 | **LeJEPA/SIGReg**（2511.08544） | 各向同性高斯最优先验；我们 JEPLM 已用 SIGReg-lite |
| 认知架构 | **KC-Agent**（2608.02351, 2026-08） | 双过程 + 漂移检测 + 知识更新——与我们同一设计骨架 |
| 记忆系统 | **mem0/Zep/Hindsight** | LLM 居中记忆服务；无 surprise 门控、无版本化验证门 |
| 持续 RL | **Safe Continual RL / AllDayNav / WorldCycle**（2026） | 非平稳 RL 活跃但都在权重域 |

## 三、对 PLWM 定位的最终裁定

1. **surprise 门控**：从"独有设计"降级为"主流共识中的一员"——必须叠加信念域/证据门/修订审计才构成差异化。
2. **latent 预测 + 记忆**：方向被 HERA/Genie 3/HWM 独立验证——但 HERA 是最接近的竞争者，需对照（我们：流式槽位+门控写入+信念重构+RL 规则修订；HERA：冻结 V-JEPA 2 + 记忆路由，纯感知评估）。
3. **参数冻结 + 外部规则记忆的规则修订**：2026 无竞争者（全部在权重域做手术）——这是 PLWM 当前最独特、最有论文价值的空位。
4. **方法论**：TTT 需行为证据校准（2607.00368）背书我们的 swap counterfactual 探针——PLWM 是少数"行为级验证记忆"的系统。

## 四、下一步建议（更新优先级）

1. **（最高）规则修订差异化实验**：把 surprise 门控升级为"证据门 + 信念域 + commitment 审计"组合
   ——与 BAPR/CPR 权重域路线直接对照（它们没有的我们做）
2. **对照 HERA**：明确我们 vs HERA 的差异（流式 RL 修订 vs 冻结感知记忆路由）
3. **JEPLM v2 写入门控对照**：预测误差 vs TTCD（长-短窗差异）vs 梯度（Titans）
4. 规则修订评测对齐 LongMemEval-V2 思路（旧规则保持率 + 新规则正确率双指标——我们已做）

## 五、来源
4 路子代理调研结果（2026-08-09），arXiv 编号均已验证；详见各子代理报告。

## 六、追加调研：Kimi K3 + CCH（2026-08-11）

### Kimi K3（arXiv 2607.24653, 月之暗面 2026-07-27, 权重开源）
- 2.8T 总参数 MoE，104B 激活/token，1M context，原生视觉；缩放效率较 K2 +2.5×
- **KDA（Kimi Delta Attention）**：delta-rule 递归 + 通道级 forget gate
  `S_t = (I − β_t k_t k_tᵀ)·Diag(α_t)·S_{t−1} + β_t k_t v_tᵀ`
  每步：通道衰减旧态（α）→ 在 key 方向定向擦除（delta-rule）→ 外积写入（β 写强度）→ 线性读取
  = 学习式"遗忘旋钮 + 定向擦写"；记忆载体 = 固定矩阵 S（**仍是压缩态**）
- 混合注意：每 block **3×KDA（压缩）+ 1×Gated MLA（无损检索）**，末尾强制全局 attention
  → **工业级独立验证"压缩+无损检索混合"**（与我们 RMT/GRU 对比结论同构）
- AttnRes（跨层 pseudo-query 检索，preprint 59）；Stable LatentMoE（896 专家激活 16）
- 后训练：多域 RL × 多推理强度，multi-teacher 蒸馏合并；性能领先其他开源，落后 Claude Fable 5/GPT-5.6 Sol
- 对项目：① delta-rule 擦写 = 未测的第三类写入门控 → 实验队列；② 压缩态不可审计/无体外修订 → 差异化不变

### CCH 能力收敛假说（arXiv 2607.14144）
- 主张：固定 per-token 推理预算下，能力收敛到 **access-complete 混合** = compressive O(1) 状态通道 + scalable verbatim-index 通道
- 三道资源墙：Shannon wall（o(N^b) 状态）、horizon wall（固定窗口）、circuit wall（定深 attention-only, 条件 TC0≠NC1）
- 预注册实验：64 标量状态 + 一层 global attention → 精确检索误差 0.994 → 0.000（剪刀差实测）
- 对项目：槽位（verbatim-index）+ 隐状态（compressive）= access-complete 方向的理论背书；难度边界实验可基于三道墙设计

### 行动项
1. JEPLMMemory 加 delta-rule 擦写门控变体对照（中优先级，1-2 天）
2. CCH 三道墙作为任务难度边界设计依据（写论文/设计实验时引用）
