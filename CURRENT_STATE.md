# CURRENT_STATE — 项目交接文档（新会话从这里读起）

> 用途：让新会话用几分钟读完"项目现在在哪、下一步做什么"，避免重读历史大文件。
> 更新规则：每个阶段闭环后更新本节头部；历史细节写进 NIGHT_RUN/各 REPORT，不搬进本文件。
> 最后更新：2026-08-11（阶段 31 闭环）

---

## 一、项目北极星（一行）

验证"显式槽位记忆 + 语义感知层"的 JEPLM 认知架构，用白皮书（COGNITIVE_SYSTEM_WHITEPAPER.md）12 个认知区块的最小可证伪实验逐块攻克；当前在记忆/感知层（区块 3/12 相关）。

## 二、当前主线（2026-08 以来）

JEPLM 记忆架构的组件验证链（按依赖顺序已完成）：

1. **记忆层**：内容寻址槽位 > 压缩隐状态（DecideAfterDelay 家族，3-6 seed）
2. **感知层**：CNN 感知（0.961）→ 记忆传导量化；冻结是正确设计（联合训练证伪 0.427）
3. **辅助目标三态矩阵（阶段 28-30 闭环）**：
   | 辅助类型 | RL 效果 |
   |---|---|
   | 重建式（像素/输入空间）| 有害（evidence -0.10）|
   | 语义内容预测（记忆内容 aux）| 帮助（evidence 0→+0.56）|
   | latent next-state 预测（JEPA 式，作训练目标）| 中性（0.941/0.831 与无预测完全一致）|
   | JEPA 预测误差 → 决策侧置信度特征 | 中性（同上，分诊：信息与 one-hot 观测冗余）|
4. **混淆观测 POMDP 攻击（阶段 31 闭环）**：把身份帧改为 4 帧带噪线索（单帧 65% 可确定、贝叶斯上限 0.846）——JEPA predict/conf 仍全中性（3 seed 噪声内）；双层分诊：state/features 信号源下翻转线索 conf 均无区分（0.458≈0.455、0.497≈0.497），预测误差（帧间变化量）与任务所需信息（线索-身份静态一致性）类型错配；u3g4 回归确认阶段 28 结论不受实现修正影响
5. **架构组装**：JEPLMAgent v1（冻结 CNN + JEPLMMemory + 策略头）在合成/真实流可用

## 三、最新闭环结论（阶段 31，2026-08-11）

- **混淆观测 POMDP 上 JEPA 仍全中性**：身份帧改为 4 帧带噪线索（单帧不可确定、多帧可贝叶斯推断 0.846）后，predict（0.712/0.672）、conf_feature（0.727/0.706）与基线（0.736/0.687）全在 seed 噪声内；
- **分诊根因（双层）**：① state 表示平滑线索差异（翻转/干净线索 conf 0.458≈0.455）；② 修正信号源为 features（含 recalled）后仍无区分（理想 predictor 0.497≈0.497）——**预测误差（帧间变化量）与线索-身份静态一致性（任务所需信息）类型错配**；
- **定位结论更新**：JEPA 价值位不是"观测混淆"（记忆可绕过：存线索+融合读出），而是"状态随时间演化且演化规律隐藏"的真动力学外推（世界模型规划/反事实）。混淆观测攻击关闭了 JEPA 在静态记忆任务族上的最后一道门。

## 四、下一步候选（按推荐序）

1. **真动力学外推任务（区块 12 剩余位）**：混淆观测已证伪（阶段 31）——JEPA 剩余价值位只剩"状态随演化且规律隐藏"（隐蔽转移/世界模型规划）。**不要**做"记忆型 POMDP"（观测混淆但状态静态=记忆可绕过，已关闭）；要做"动力学型"：观测给部分状态，下一步状态由隐藏规律决定，预测误差携带纯动力学信息。相关资产：train_world_model_rollout.py（转移表学习，非 latent 预测）、train_causal_world.py/train_multistep_world.py（隐藏上下文=记忆保持，也非价值位）。
2. **RMT + aux 混合**（JEPLM_AUX_REPORT 遗留）：流式 RMT 跨段压缩在 RL 下不可学，语义 aux 能否松动——工程量大，需 RL+aux 混合设计。
3. **双身份/干扰场景 aux**（JEPLM_AUX_REPORT 遗留）：单身份已验证核心机制，干扰场景未测。

## 五、关键文件索引（新会话按需读，不读历史档案）

| 文件 | 角色 | 何时读 |
|---|---|---|
| **本文件** | 交接指针 | 必读 |
| COGNITIVE_SYSTEM_WHITEPAPER.md (15KB) | 北极星/12 区块 | 定方向时 |
| NIGHT_RUN_20260809.md (32KB) | 阶段 1-31 实验日志 | 查具体证据时（有目录可跳） |
| JEPLM_AUX_REPORT.md | 辅助目标三态完整矩阵 | 做 aux/JEPA 相关时 |
| JEPLM_MEMORY_DESIGN.md / JEPLM_IMPLEMENTATION.md | 记忆组件设计/集成 | 改记忆代码时 |
| JEPLM_WRITE_MODE_REPORT.md | write-mode 消融（FIFO>gate>delta） | 动写入门控时 |
| MEMORYS7_REVERIFICATION_REPORT.md | 历史结论证伪记录 | 引用 MemoryS7 结论前必读 |
| **不要读**：MASTER_PLAN.md (107KB，Luanti/MiniGrid 历史档案)、PLWM_BLUEPRINT_V1/PLAIN、RESEARCH_SURVEY_202608 | 已完结历史 | 除非查旧结论 |

## 六、实验入口（常用命令）

```bash
# 合成流连续 RL（主战场）：--jeplem-only 只看 JEPLM，--jeplem-predict/--jeplem-conf-feature 测 JEPA
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json

# 混淆观测 POMDP（阶段 31）：--confounded --clue-frames 4 --clue-reliability 0.65，--probe-conf 跑 conf 分诊
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --confounded --clue-frames 4 --clue-reliability 0.65 --jeplem-predict --jeplem-conf-feature --probe-conf --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json

# 感知层
python experiments/train_perception_layer.py

# 世界模型（下一步候选，注意：要动力学型不是记忆型）
python experiments/train_world_model_rollout.py

# 测试
python -m unittest discover -s tests -v
```

> 纪律：3 seed（444/555/666 或 222/333/41123）、swap 探针、诚实负结果归档。
