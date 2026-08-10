# JEPLM 记忆组件实现蓝图（2026-08-10）

> 性质：从"证据"到"系统"的过渡文档——把 JEPLM_MEMORY_DESIGN 的 6 条设计决策 + 2026-08-10 的新证据落成可实现的组件蓝图。
> 关联：JEPLM_MEMORY_DESIGN.md（决策）、JEPLM_AUX_REPORT.md（四象限）、RMT_STREAMING_COLLAPSE_REPORT.md（流式）、MEMORYS7_REVERIFICATION_REPORT.md（评估纪律）、DECIDE_AFTER_DELAY_REPORT.md（swap 方法论）

---

## 一、状态：组件已落地并端到端验证

`plwm/jeplem_memory.py` 已实现（2026-08-10）：
- **显式槽位主干**：内容寻址读（softmax attention over slot keys）+ 学习门控写（GRU 融合 + FIFO 位置 + sigmoid gate）
- **语义内容预测头**（可选多时标）：aux_head 从 state+recalled 特征预测最近 k 个身份的 one-hot
- **无重建式 surprise**：门控为学习式，surprise 仅作可选特征（默认关闭）
- API：`step(obs, valid, surprise_feature?) -> (features, aux_logits)`；`forward_sequence`；`export_state/load_state` 可插拔

自测（合成流监督）与端到端（连续流 RL）：
| 场景 | 结果 |
|---|---|
| 监督 d=32, slots=16 | 1.000 |
| 监督 d=64, slots=16, 无 aux | 0.422（容量边界）|
| **监督 d=64 + aux** | **1.000（D2 决定性复现）** |
| RL 连续流 u3g4, slots=16 | 0.832/0.706（槽多稀释）|
| **RL 连续流 u3g4, slots=8** | **0.941/0.831**（对照 evidence 0.958/0.881）|

## 二、集成指南（供后续完整 JEPLM 使用）

1. **槽数配置**（D1 容量规律）：槽数 ≥ 需记住的时间跨度；8 槽适合 u3g4（跨度 5 帧/身份），16 槽在短跨度任务会稀释（写入分散）。配置规则：`slots = max(8, span_in_frames)`。
2. **aux 时标数**（D4 精确化）：aux 头数 ≥ 任务查询时标数（2 时标需求 ≥2 头；3 时标需求 ≥3 头）。**aux 目标必须放在"空帧"**（记忆内容预测语义），放身份帧/查询帧是"读输入"伪目标（0.422 与 1.000 的差别）。
3. **aux 系数**：0.1-0.5 稳健（JEPLM_AUX_REPORT）；RL 下 aux 进总损失不进 PPO advantage。
4. **门控**：学习式 write_gate；TTCD/surprise 信号替代无增益（2026-08-10 探针），保持简单。
5. **评估纪律**：一切记忆声明带 swap 探针；达标率 + 口径披露；任务构造干预须带"打破假设的对照"。

## 三、感知层（2026-08-10 已实证，JEPA 编码器职责）

- **实现**：CNN 感知器（7×7 图像卷积 + 辅助通道 MLP）从原始 157 维观测学 Key/Ball 身份区分，heldout 0.961（key 召回 0.983 vs MLP 0.555）——`experiments/train_perception_layer.py`。
- **精度要求（传导量化）**：身份区分精度是记忆可行性的硬前置——<0.9 → 记忆不可学（0.43）；0.96 → 部分可学（0.74/0.54）；oracle 1.0 → 全可学（0.84/0.75）。cross 对感知噪声更敏感。
- **全链路验证**：原始观测 → CNN 感知器 → JEPLM 记忆 → RL 决策（0.743/0.537，swap evidence 0.49），"感知→语义→记忆"链路闭环。
- **设计要点**：空间归纳偏置必需（CNN）；感知瓶颈形态 = 身份区分（Key vs Ball）而非存在性检测。
- **分层训练决策（2026-08-10 联合训练证伪）**：感知器**冻结**是正确设计——RL 稀疏梯度会破坏预训练感知器（身份信号退化，记忆 0.743→0.427 不可学）；感知层自监督预训练 + 冻结、记忆层 RL 优化（与 JEPA 预训练编码器主张一致）。自监督感知（JEPA latent 预测）列后续。

## 四、待集成项（诚实清单）

1. **与 evidence 的微小差距**（-0.03/-0.07）：jeplem8 0.941/0.831 vs evidence_aux 0.965/0.900——结构等价但实现细节（初始化、gate 输入、read query 来源）未完全对齐，留待集成期微调。
2. **感知-记忆联合训练**：当前感知器冻结；联合微调（低学习率）或自监督感知（JEPA 式 latent 预测替代监督目标）是下一层实现。
3. **完整 JEPLM 体系**：记忆+感知之外的部分（JEPA 式预测、信念更新、多尺度）按 WORLD_MODEL_SURVEY 与白皮书 Block 1-12 分层接入。
4. **surprise 决策侧应用**：置信度/门控信号接出到执行控制（白皮书 Block 8/9），组件已留 `surprise_feature` 接口。

## 四、证据链完整性（为什么可以进入实现）

所有设计决策都有 3+ seed + swap 探针证据（JEPLM_MEMORY_DESIGN 证据矩阵）；历史 MemoryS7 结论已全部证伪并替换（MEMORYS7_REVERIFICATION_REPORT）；PLWM RL 差距诊断闭环（seed 效应）；3 时标上限与槽位免先验优势已量化。**项目证据基础无已知悬空项**（唯一未决 = 真实流重构，属任务设计非架构）。
