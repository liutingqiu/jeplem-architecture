# 标准审计与补测记录（2026-08-09）

> 性质：阶段反思的落地文档 —— 审计发现 + 补测结果 + 冻结协议
> 关联：DECIDE_AFTER_DELAY_REPORT.md、DECIDE_STREAM_RULE_ADAPTATION_REPORT.md、JEPLM_ROUND1_REPORT.md

---

## 一、审计发现摘要（三档）

### A 级（最接近"让通过"）

| # | 发现 | 证据 | 处置 |
|---|---|---|---|
| A1 | rule-adapt 报告以 3 seed（1/3 严格达标）宣布"架构优势实证"，用"爬升"代替"达标" | `train_decide_stream_rule_adaptation.py:121-124` 验收门 vs 报告结论；6 seed 中 3/6 达标 | **已补测**：6 seed 双口径重出，见本文件 §二.1 与报告 §六 |
| A2 | MemoryS7 held-out 被平凡策略攻破（27/64 恒定），`minigrid-hybrid-curriculum-report.json` 曾标 accepted=true（37/64）——假阳性通过 | JEPLM_ROUND1_REPORT.md:34-41 | 已由项目自行证伪 + swap probe 修复（此前完成，非本次） |

### B 级（验收标准弱于宣称结论）

| # | 发现 | 证据 |
|---|---|---|
| B1 | v1/v2/v3 验收只验 `max(gru,evidence,plwm)≥0.8 且超 shortcut 0.2`，不验 PLWM>GRU；v1 全模型 1.0（零区分度）仍 accepted | `audit_minigrid_decide_after_delay.py:251-256` |
| B2 | RL 验收 `max(plwm,evidence)≥gru+0.1`，evidence 单独达标即可通过 | `train_decide_after_delay_rl.py:425-429` |
| B3 | stream 验收只要求 PLWM≥0.6 且超 GRU+0.2，不要求胜 evidence，报告却写"query 第一"（0.924 vs 0.889 差距在噪声内） | `train_decide_stream_rl.py:497-500` |

### C 级（数字口径与透明性）

| # | 发现 | 处置 |
|---|---|---|
| C1 | PLWM d=32 三个报告引用不同批次（0.862/0.889/0.781），未披露跨批次方差（9 seed 范围 0.61–1.0） | **已补测**：9 seed 汇总，见 §二.2 |
| C2 | d=16 单 seed 未标注；身份轴单 seed 下结论 | 已加注（d=16） |

### 已排除的疑点

- **预算公平**：6 个对照脚本均同循环/同 updates/同 lr（subagent 核查），无 PLWM 独享预算。
- **阈值事后放宽（rule-adapt/stream）**：git blame 确认阈值随文件创建写入、从未修改。
- **stream seed 挑选**：seed 集 41123/222/333 是实验开始时固定的（会话记录 message 452），非失败后替换。
- **self_model 判据放宽（0.1→0.05）**：主会话确实放宽过判据，但**放宽后 3 seed 仍全部不达标**（连 ECE≤0.2 上限都未过）——未改变任何验收结果，区块 8 的 🟡 为诚实失败。此项从"改判据让过"降级为"判据调整未遂"。

---

## 二、补测结果

### 2.1 rule-adapt 6 seed 双口径重出

见 `DECIDE_STREAM_RULE_ADAPTATION_REPORT.md` §六 与 `artifacts/rule-adapt-recompute-6seed.json`。

要点：严格达标（终值≥0.7 且超 GRU+0.2）= **slots16 5/6、默认 PLWM 3/6、GRU 0/6**；爬升 = slots16 6/6、默认 6/6、GRU 3/6。"PLWM 优势"按严格口径需分级表述；GRU 结构性失败 0/6 成立。

### 2.2 d=32 九 seed 方差汇总

见 `artifacts/d32-variance-aggregate.json`。PLWM d=32 9 seed = 中位 0.793、均值 0.844±0.136、范围 0.609–1.000；GRU 9/9 = 0.496（0 方差）。**方向性结论（可寻址>压缩）稳，单批精确值不可作精确优势引用**。

### 2.3 stream 补测 seed 444/555/666（6 seed 重出）

见 `artifacts/stream-seed-aggregate.json`。要点：
- **PLWM 脚本门达标 5/6**（原 3 seed 为 2/3；唯一失败 seed222 差 0.006）；slots16 3/3。
- **原"PLWM query 第一（0.924）"被证伪**：6 seed 下 PLWM query 0.909 ≈ evidence 0.910（打平），cross 0.769 < evidence 0.799。3 seed 的"第一"是抽样波动。
- **新增对照发现：RMT 连续流 0/3 崩盘**（cross≈0.5 随机）——RMT 只在监督精确回忆/规则修订与 PLWM 打平，跨 episode 连续流场景不成立（此前 RMT 对照缺该场景，本次 6 模型脚本补齐）。
- GRU 6/6 ≈随机（cross 0.48–0.52），机制结论稳。

### 2.4 self_model 判据澄清

现有 3 seed（222/333/444）在宽松判据（softmax margin 0.05）下 accepted 全部为 False；严格判据（0.1）同样全部 False。无重跑必要，区块 8 维持 🟡。

---

## 三、冻结协议（自 2026-08-09 起生效）

1. **结论强度 ≤ 验收强度**：报告结论（如"PLWM > GRU"）必须逐条对应验收标准中的同项；标准未含的差异不得作为结论宣称。
2. **达标率披露**：所有"通过/优势/实证"类声明必须注明 `X/N seed 达标` + 口径定义（达标/爬升/终值>θ），并附失败 seed 数值。
3. **seed 集冻结**：实验 seed 集与验收逻辑在首个实验运行前 git 提交；运行后只允许新增实验，不允许改口径后重算通过。
4. **增强配置须披露默认**：以调优后配置（如 slots16、latent64、gated）作为主张依据时，必须同时披露默认配置同指标。
5. **补测必须沉淀**：补跑/巩固类实验（如 6 seed 巩固）的结果必须写入对应报告文件，不得只留在会话记录。
6. **评估防御**：记忆/决策类任务一律带 swap counterfactual 探针（`both` 或 `orig−swap` 语义），防止平凡策略攻破（A2 教训）。

---

## 四、产物清单

- `experiments/audit_rule_adapt_recompute.py` → `artifacts/rule-adapt-recompute-6seed.json`
- `experiments/audit_d32_variance_aggregate.py` → `artifacts/d32-variance-aggregate.json`
- `experiments/audit_stream_seed_aggregate.py` → `artifacts/stream-seed-aggregate.json`（stream 补测后）
- 报告修订：`DECIDE_AFTER_DELAY_REPORT.md`（d=32 区间表述）、`DECIDE_STREAM_RULE_ADAPTATION_REPORT.md`（§六 6 seed 重出）
