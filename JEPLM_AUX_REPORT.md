# JEPLM 辅助目标重定位：记忆内容预测在长延迟监督任务上的决定性帮助

> 日期：2026-08-10（夜间运行）
> 性质：JEPLM 辅助目标重定位的强正结果（与此前负结果链互补）
> 关联：JEPLM_ROUND1/2_REPORT.md（MemoryS7 负结果）、DECIDE_AFTER_DELAY_REPORT.md（§6 RL 诊断）、NIGHT_RUN_20260809.md

---

## 一、背景与动机

JEPLM 辅助目标此前全是负结果：
- MemoryS7 RL 上 SPR 式 latent 预测无增量（round 1/2）；
- DecideAfterDelay RL 上给 evidence 加 predictor 头掉 0.10（0.964→0.865）——aux 进 RL 策略梯度是**负担**；
- 真实 v2 语料（8 帧延迟）上 aux 可学但主任务饱和无增量。

**重定位假说**：aux 的价值是任务依赖的——在**监督 + 记忆保持压力**场景（长延迟回忆）下，"记忆内容预测"（aux 头持续预测被记住的身份）应帮助**压缩派**架构（GRU/evidence）克服隐式压缩瓶颈。

## 二、任务与设计

- 任务：单身份延迟回忆（合成）——[身份 one-hot(帧0), (delay-1) 空帧]，末帧决策输出身份。swap 探针（翻转身份→选择映射）。
- 对照：gru_base / gru_aux / evidence_base / evidence_aux（aux = 帧 1..T-1 每帧预测身份 one-hot；α=0.5 主，扫描 0.1-1.0；同预算同循环）。
- 语料：synthetic_single（`train_aux_memory_content.py`），heldout 256，300 updates。

## 三、结果（3 seed 全确认）

| 延迟 | gru_base | **gru_aux** | evidence_base | **evidence_aux** |
|---|---|---|---|---|
| d=8/16 | 1.000 | 1.000 | 1.000 | 1.000 |
| **d=32** | **0.508（随机）** | **1.000** | 1.000 | 1.000 |
| **d=64** | **0.508（随机）** | **1.000** | **0.508（随机）** | **1.000** |

- 3/3 seed（93_337/222/333）完全一致；
- α 扫描 0.1/0.2/1.0：全部 1.000（效果对 α 稳健，0.1 即够）；
- swap evidence：gru_aux d=32 = 1.0（真记忆，非捷径）；
- 无泄漏：aux 预测帧 1..T-1 全为空帧（身份仅帧 0 出现）。

**机制**：aux 在每帧施加"身份保持"监督 → 压缩派隐状态被训练成**持续携带记忆内容** → 决策帧可直接读出。相当于给"隐式压缩"加显式内容锚点——正是 JEPA"预测记忆内容驱动记忆"的本意。

## 四、与既有结论的整合（不矛盾）

| 场景 | aux 效果 | 结论 |
|---|---|---|
| MemoryS7 RL（sparse reward）| 无增量 | 任务不可诊断（评估攻破）|
| DecideAfterDelay RL | **负担（-0.10）** | aux 进策略梯度有害 |
| 真实 v2 监督（d=8）| 无增量 | 主任务饱和 |
| **单身份合成监督（d≥32）** | **决定性帮助（0.508→1.0）** | **记忆保持压力场景有效** |

**架构含义**：surprise/预测类辅助目标应**双轨**——作为**记忆写入门控/内容锚点**（监督侧，本报告正结果）而非**策略梯度目标**（RL 侧，此前诊断负结果）。JEPLM 的"记忆内容预测"定位成立。

## 五、遗留（诚实记录）

- **RMT + aux 未测**：流式 RMT 的跨段压缩"不可学"结论是否因 aux 而松动？——流式 RMT 是 RL 策略，aux 进 RL 落入"负担"区（已有诊断），需 RL+aux 混合设计，工程量大，列后续。
- 双身份/干扰场景下 aux 的帮助未测（单身份已验证核心机制）。
- d 边界：GRU 崩点（d≥32 单身份）处 aux 生效；d≤16 无增量空间。

## 六、产物

- `experiments/train_aux_memory_content.py`（corpus: real/synthetic_single；--delay/--alpha/--experiment-seed）
- `artifacts/aux-mem-content-synth-d{32,64}.json`、`aux-mem-synth-d{32,64}-seed{222,333}.json`、`aux-mem-d32-alpha{0.1,0.2,1.0}-s222.json`、`aux-mem-d16-s222.json`
