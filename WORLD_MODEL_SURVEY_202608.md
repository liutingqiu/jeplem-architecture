# 世界模型与预测性学习全景调研（JEPA 及同族/互补架构）

> 日期：2026-08-08
> 目的：为 PLWM 架构重设计扩大"借鉴池"——不止 JEPA，调研其同族（预测性表征学习）与互补（记忆、世界模型、认知架构、预测性编码）的权威机构工作
> 方法：6 路并行子代理 web 调研，所有 arXiv 编号均经官方页面验证；多处纠正了常见编号误记（DreamerV2=2010.02193 非 2010.03593；SIMA=2404.10179 非 2402.12549；Memory3=2407.01178 非 2502.19453；LeJEPA=2511.08544 非 2511.00668）
> 状态：调研素材，供 JEPLM v2 设计参考

---

## 〇、一句话结论（给决策者）

JEPA 家族解决的是"**如何从观察中无监督地学到可规划的表征**"；但 PLWM 需要的"**什么值得记住、如何用记忆做预测、如何让预测误差驱动信念**"在 JEPA 家族之外有更成熟、更实证的答案（SPR/TD-MPC2/Titans/Memory3/FEP）。**JEPA 是借"目标"不借"骨架"，真正的架构补强来自 RL 表征学习与神经记忆两条线。**

---

## 一、JEPA 家族全景（Meta AI/FAIR + NYU LeCun 组主导）

### 主线

| 工作 | 机构 | 编号 | 核心 |
|---|---|---|---|
| I-JEPA | Meta FAIR | 2301.08243 | 大目标块 + 分散上下文块，EMA teacher 防崩坍；"预测粒度=语义粒度" |
| V-JEPA | Meta FAIR | 2404.08471 | 视频版；跨时间帧预测 latent，把时序一致性编进表示 |
| V-JEPA 2 / 2-AC | Meta FAIR | 2506.09985 | 1M 小时视频预训练；**2-AC 用 <62h 无标签机器人视频微调出 action-conditioned latent 世界模型，零样本规划 Franka 抓取** |
| LeJEPA | Brown + Meta | 2511.08544 | 各向同性高斯=下游预测风险最优嵌入分布；**SIGReg 显式分布正则替代 EMA/stop-grad 全部启发式**，单一超参 |
| JEPA-SCORE | Meta | 2510.05949 | 反崩坍项隐式估计数据密度；**Jacobian 闭式计算样本概率**→免费的新奇/异常信号 |

### 变体与延伸（解决 JEPA 已知局限）

| 工作 | 机构 | 编号 | 解决什么 | 对 PLWM 的启示 |
|---|---|---|---|---|
| SPHERE-JEPA | ENS Paris-Saclay | 2605.26900 | 高斯嵌入在检索场景有各向异性偏差；**超球面均匀才是检索最优** | **检索/记忆的嵌入几何用均匀化，预测/规划用高斯**——记忆槽 key 该用哪种有理论依据了 |
| VJ-VCR | NYU LeCun 组 | 2412.10925 | 未来不确定性 | 长时预测要**分布化 latent**（不确定性进 latent 而非点预测）|
| LeWorldModel (LeWM) | Meta/Brown | 2603.19312 | **首个端到端可训的 JEPA 世界模型**：仅两项损失（下一嵌入预测+高斯正则），~15M 参数单卡数小时，latent 可探测物理量、可检测不 plausible 事件（surprise）| 轻量常驻世界模型的直接模板 |
| JEDI | NUS 等 | 2605.13013 | JEPA 纯离线、不做在线 MBRL | **在线端到端 JEPA + diffusion 世界模型**，Atari100k 有竞争力 |
| PAM (Inward-JEPA) | — | 2602.11322 | 无记忆、检索靠相似度 | **"记忆检索=预测"而非"检索=余弦相似度"**——概念上最新颖（跨房间 Recall@20 0.421 vs 余弦 0）|
| Worth Remembering | MIT Carlone 组 | 2606.03787 | JEPA 无记忆 | **V-JEPA 语义 latent 算 Bayesian surprise 作为记忆门控** + 4D 场景图外部记忆；机器人问答超 SOTA ≥12% |
| HERA | — | 2608.05523 | 记忆不改变权重 | 历史证据路由适配器：**记忆以 register 方式选择性注入冻结预测器** |
| Branch-JEPA | — | 2607.05238 | 单点转移 | 上下文加权的有限分支预测，推理时全保留 |
| UWM-JEPA | — | 2605.25313 | 部分可观测长时想象 | **密度矩阵 latent + 酉预测器**保持不确定性不消散 |
| AV-JEPA | Aalto | 2607.15295 | 多模态 | modality dropout 当 masking，SIGReg 对齐 |
| VL-JEPA | HKUST | 2512.10942 | 语言 | **预测文本的连续嵌入**而非自回归 token |

**JEPA 家族 5 条最新鲜机制**：① SIGReg 显式最优分布正则（在线训练稳定首选）；② 几何按场景选（高斯预测/均匀检索）；③ 预测器即世界模型（V-JEPA 2-AC/LeWM 三段式：观测预训练→少量交互微调→latent 规划）；④ JEPA-SCORE 免费密度信号；⑤ 记忆即预测（PAM/HERA/Worth Remembering 三路同向）。

---

## 二、DeepMind 世界模型线

| 工作 | 编号 | 核心 | 与 JEPA 的分歧 |
|---|---|---|---|
| Dreamer V1 | 1912.01603 | latent 空间想象 rollout 学行为 | 生成式重建（离散 latent token）vs JEPA 联合嵌入 |
| Dreamer V2 | 2010.02193 | 离散 token latent，Atari 55 人类级 | |
| Dreamer V3 | 2301.04104 | 单配置 150+ 任务，**从零像素采集 Minecraft 钻石（稀疏奖励）** | **稀疏奖励 + latent 世界模型的实证范本** |
| Dreamer 4 | 2509.24527 | shortcut-forcing 目标，单 GPU 实时推理 | |
| Genie 1/2/3 | DeepMind 博客 | 可交互生成式世界模型（生成式，非 JEPA 路线）| 生成式仍是 DeepMind 主流 |
| SIMA | 2404.10179 | 语言指令 agent | |
| MuZero | 1911.08265 | 学习奖励/策略/价值隐式模型做树规划 | 隐式模型 + 规划 |

要点：DeepMind 的 latent 世界模型（Dreamer 系）**已经在做"latent 空间预测 + 稀疏奖励 RL"**，且实证最强（Minecraft 钻石）——这比 JEPA 更接近 PLWM 的目标。JEPLM 的 C2（latent 预测器）实为 Dreamer/SPR 路线的机制，JEPA 只是理论包装。

---

## 三、RL 预测性表征学习（稀疏奖励的辅助目标设计，与 JEPA 思想最接近的一条线）

| 工作 | 机构 | 编号 | 辅助目标 | 证据 |
|---|---|---|---|---|
| **SPR** | Mila | 2007.05929 | **多步 latent 预测**（EMA target 编码器 + learned transition，5 步 latent 自预测 + 随机裁剪一致性）| Atari 100k 人类归一化 0.415，比 SOTA 高 55%，7/26 超人类 |
| CURL | UC Berkeley | 2004.04136 | InfoNCE 对比 | DMControl 100k 提 1.9x |
| DrQ | NYU | 2004.13649 | 仅 random-shift 增强正则 Q | DMControl 100k 超 Dreamer |
| SGI | Mila | 2106.04799 | latent dynamics + 无监督 goal-conditioned 预训练 | Atari 100k 超此前预训练法 |
| TD-MPC2 | UCSD | 2310.16828 | 任务导向 latent world model + **TD target（r+γV）作 latent 预测目标** | 104 任务单超参，80 任务单 agent |
| RPC | CMU/Berkeley | 2109.03214 | 信息瓶颈使策略与 latent 模型**自洽**（策略避开不可预测状态）| 比标准 IB 高 5x 奖励 |

**5 条对 PLWM 阻塞（3-7 步延迟关联）的直接结论**：
1. **多步 latent 预测是稀疏终端奖励最直接解药**，必须做在 belief/hidden 空间而非单帧分类头——PLWM 的 α=0.1 单帧检索辅助头失败（3/8=37.5%）正是"做错层级"的证据；
2. 预测目标宜**任务导向**（TD target 或自洽约束），避免学到无关熵；
3. 增强一致性（random-shift/裁剪）成本最低且必要（SPR/DrQ 双证）；
4. 可先无监督预训练 latent dynamics 再微调，彻底解耦表征与奖励；
5. 6 步延迟全模型 50% 还提示需要**多时间尺度 belief（fast/slow）分层**。

**与 JEPA 最接近**：SPR（预测语义 latent 而非重建、EMA target、自预测一致性，作者 Bachman 属该思想谱系）；TD-MPC2 次之。

---

## 四、外置记忆与持久状态架构

| 工作 | 机构 | 编号 | 存储机制 | 与预测的结合 |
|---|---|---|---|---|
| **Titans** | Google Research | 2501.00663 | 神经长期记忆模块（注意力=短期、神经记忆=长期）+ persistent memory | **surprise 门控写入**（联想记忆损失的梯度=意外度；momentum 融合当下+过去；遗忘门=weight decay）|
| **Memory3** | 含鄂维南团队 | 2407.01178 | 显式外置稀疏化可索引 KV 记忆库（参数=隐式记忆、KV=工作记忆、显式库=第三记忆）| 2.4B 外置记忆胜过更大参数模型与 RAG——**"事实进显式槽、规律留参数"的第一原则实证** |
| Mamba 系 | CMU/Cartesia | 2312.00752 / 2405.21060 / 2603.15569 | 权重内压缩状态（全神经）| 内容门控，非预测驱动——**压缩型反例**（无外置槽也能工作，但事实与规律在状态内混合）|
| Gated DeltaNet | NVIDIA | 2412.06464 | 矩阵值状态=线性注意力快权重 | **delta rule 是 error-driven 在线学习**（等价于优化联想记忆损失）→ 记忆槽定点覆写算子 |
| HiPPO/LMU | Stanford/CMU | 2008.07669 / 2110.02402 | 历史最优投影到 Legendre 多项式基 | 结构化记忆的规范形式（系数=固定格式槽位）|
| MemGPT | UC Berkeley | 2310.08560 | OS 式虚拟上下文管理，显式外置、权重不变 | 写入由 LLM 决策，非预测信号——缺"什么该写"的准则 |

结论：**最接近 PLWM 设计的是 Memory3**（唯一把"事实外置"作为第一原则并实证的）。可移植机制：① surprise/momentum 门控（Titans）——回答"什么该写入记忆槽"，与 PLWM 的 surprise-gated write **互相印证**；② persistent memory 作规律参数载体；③ delta rule 作记忆槽读写算子；④ HiPPO 系数作槽位编码；⑤ MemGPT 分层上下文管理作检索组织层。Mamba/RWKV 的"权重内压缩"是对"事实必须外置"的挑战，PLWM 应通过 needle-in-slot 检索考试守住立场。

---

## 五、认知架构与持续学习

| 工作 | 机构 | 编号 | 核心 |
|---|---|---|---|
| CoALA | Princeton | 2309.02427 | **"LLM 无状态、外部状态是主体"**；工作/情景/语义/程序四层 + 决策循环 + 记忆更新是一等动作 |
| MemGPT | UC Berkeley | 2310.08560 | 记忆操作=可审计工具调用、上下文可重建、跨会话演化 |
| Generative Agents | Stanford | 2305.16291 | 反思/自验证是修订入口，但**无证据门**（会把推断写成事实，论文自陈）|
| Agent Smith | — | 2402.08567 | 无来源门控的记忆会被注入并指数传播——**记忆即攻击面** |
| Voyager | NVIDIA | 2305.16291 | 知识外置（代码库）避免遗忘的工业验证 |
| Bayes-Adaptive RL | UMass | Duff 2002 / 1609.04436 | **信念持续可修订且不遗忘**的数学基石（hyperstate 含环境参数）|

对比结论：**全部 12 个被调研工作里只有 PLWM 明确分离 belief（世界如何）与 commitment（我要做什么）并有证据门隔离**——这是 PLWM 的差异化优势，调研中无人做。Agent Smith 证明 PLWM 的证据门是结构性安全防御。BAMDP 证明 commitment 独立成域正是避免 hyperstate 爆炸的分层。

---

## 六、预测性编码与自由能原理（把"预测误差"升级为"置信度"）

| 工作 | 机构 | 编号 | 核心 |
|---|---|---|---|
| Rao & Ballard | Salk + Rochester | Nature Neurosci 1999 | 反馈传预测、前馈传残差；**同一误差双用（推断+学习）** |
| FEP | UCL Friston | 2005 / 2010 / 2017 | 变分自由能=surprise 上界；**precision（逆方差）加权误差**；active inference 的 **epistemic value=主动采样（选能最大降低不确定性的动作）** |
| PredNet | MIT | 1605.08104 | 每层只向上转发残差，视频预测自监督 |
| PC≈Backprop | Sussex/Oxford | 2006.04182 / 2202.09467 | PC 在任意计算图上渐近等价反传，局部 Hebbian |
| Neural Generative Coding | RIT/Penn State | Nat Commun 2022 | 神经元"预测邻居、按误差调参"，性能匹敌 VAE |

**对 PLWM 的直接启示（补上 JEPLM 缺的 belief 侧）**：
1. **置信度=precision（逆方差），不是启发式计数**——surprise 用误差方差标准化，噪声大时抑制写入；
2. **同一误差双用**：surprise 不只门控记忆写入，还应反传更新预测器（PC≈Backprop）；
3. **主动采样**：引入 epistemic value 作为探索项——"主动制造 surprise"，直接缓解稀疏奖励的 credit assignment；
4. **fast/slow 可正式化为层级 precision**：慢层低 precision 长时预测、快层高 precision 短时预测；
5. **风险警示**：precision 估计错误→过度加权→幻觉（REBUS 模型），**必须给 precision 加上限/饱和度**。

---

## 七、综合：对 JEPLM v2 设计的影响

结合上一版报告的四点自我批判，本次调研给出明确修正弹药：

| 上版问题 | 调研给出的修正 |
|---|---|
| ① JEPA 核心主张（latent 预测）在低维输入下可能不成立 | 不必赌 JEPA——**SPR（Mila）与 TD-MPC2（UCSD）已经在 RL 侧实证"多步 latent 预测 + 任务导向目标"是稀疏奖励的最直接解药**，这是比 JEPA 更可靠的借鉴源；JEPA 只贡献"显式分布正则"这一理论包装 |
| ② 偏科：belief/commitment 未接入 | FEP 给出答案：**预测误差→precision→置信度**，belief 更新与记忆写入共用同一个信号；active inference 的 epistemic value 给 exploration 一个理论锚 |
| ③ 离线预训练独立阶段违背 One Operating System | 改为 **SGI/TD-MPC 式的"预训练 latent dynamics 并入在线闭环"**：replay buffer 上的自监督目标成为总损失的一部分，而非单独阶段 |
| ④ MiniGrid 转向未标注 | 保持，标注为临时止损（调研确认 MiniGrid/BabyAI 正是 SPR/CURL 等工作的标准环境，证据链兼容）|
| 新增：记忆层升级 | **PAM 的"记忆检索=预测"**、Titans surprise 门控、Memory3 稀疏化外置库、SPHERE-JEPA 的"检索用均匀几何"——四个机制可直接进 C3 |

**JEPLM v2 的组件排序建议（按实证强度）**：
1. **多步 latent 预测辅助损失**（SPR 机制，实证最强）——替代/先于 masked prediction；
2. **TD target 任务导向预测**（TD-MPC2）；
3. **fast/slow 分层**（多时间尺度 belief，配合多步预测的尺度对齐）；
4. **记忆层**：surprise 门控（Titans/Worth Remembering）+ 检索=预测（PAM）+ 均匀几何（SPHERE-JEPA）；
5. **belief 侧**：precision 置信度 + epistemic value 探索（FEP）；
6. JEPA 专属贡献收敛为一项：**SIGReg 式显式分布正则**替代 EMA 启发式。

---

## 八、权威机构分布速查

- **Meta AI/FAIR（LeCun 系）**：JEPA 全部主线 + DINOv2 + LeWorldModel
- **NYU（LeCun/Klindt）**：VJ-VCR、可识别性证明
- **Google DeepMind**：Dreamer 系、Genie 系、SIMA、MuZero
- **Google Research**：Titans
- **UC Berkeley**：CURL、MemGPT、RPC（合著）
- **CMU/Cartesia（Albert Gu）**：Mamba 系、HiPPO（合著）
- **Stanford**：HiPPO（合著）、Generative Agents
- **MIT**：PredNet、Worth Remembering（Carlone 组）
- **UCL（Friston）**：FEP/Active Inference 全部
- **Mila**：SPR、SGI（Bachman）
- **UCSD**：TD-MPC2
- **NVIDIA**：Voyager、Gated DeltaNet
- **Princeton**：CoALA
- **Oxford/Sussex**：PC≈Backprop
- **Brown**：LeJEPA（Balestriero）、LeWorldModel（合著）

---

## 九、遗留事项

1. task-2 部分输出（Genie/SIMA 细节）因截断未完整收录，如需可单独补查；
2. PAM（2602.11322）、Branch-JEPA（2607.05238）、UWM-JEPA（2605.25313）、HERA（2608.05523）等较新工作仅有摘要级信息，实施前需读全文验证；
3. 建议下一步：基于本调研 + 上版报告的四点批判，修订 `JEPA_PLWM_REDESIGN_REPORT.md` 为 JEPLM v2（把 SPR/TD-MPC2/Titans/FEP 的机制并入，JEPA 收敛为 SIGReg 一项）。
