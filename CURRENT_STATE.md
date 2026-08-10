# CURRENT_STATE — 项目交接文档（新会话从这里读起）

> 用途：让新会话用几分钟读完"项目现在在哪、下一步做什么"，避免重读历史大文件。
> 更新规则：每个阶段闭环后更新本节头部；历史细节写进 NIGHT_RUN/各 REPORT，不搬进本文件。
> 最后更新：2026-08-11（阶段 35 闭环：承重件 4 修订 RL 集成不成立，架构承重件最终确定）

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
5. **查询提示混淆（阶段 32 闭环，JEPA 支柱审计终局）**：查询帧携带身份提示（30% 翻转），最优=信记忆抗提示——JEPA predict/conf 仍中性（3 seed 均值），且带 predictor 配置跨段记忆崩（s555/666 cross 0.493=随机）——预测监督与记忆保持冲突；探针：查询帧 features 被随机选择器噪声主导，conf 测不到 hint-记忆一致性
6. **架构组装**：JEPLMAgent v1（冻结 CNN + JEPLMMemory + 策略头）在合成/真实流可用
7. **surprise 修订 RL 集成（阶段 35 闭环，负结果）**：规则翻转（switch_at=2）下清槽修订有害（after 0.292 vs 基线 0.396）——记忆存身份（翻转不变信息）清掉纯损失；翻转适应靠权重学习；one_shot==always（触发一次后模型自适）；监督版 one-shot 1.0 不迁移到 RL（无 teacher）
8. **架构承重件最终确定**：JEPA 感知层（冻结 CNN 0.961）+ PLWM 显式槽位记忆 + 语义 aux（单身份/单时标）。JEPA latent 预测（阶段 28-32）与 surprise 清槽修订（阶段 35）均降级——RL 下无实证支持。

## 三、最新闭环结论（阶段 35，2026-08-11）

- **surprise 清槽修订 RL 集成不成立**：规则翻转任务（switch_at=2）上清槽修订 after_switch 0.292 < 基线 0.396——记忆存身份（翻转不变），清槽丢信息；适应靠权重学习；one_shot==always（一次触发后自适）；监督版结论（阶段 27）不迁移到 RL；
- **架构承重件最终确定**（贯穿 28-35 审计）：JEPA 感知层（冻结 CNN）+ PLWM 显式槽位记忆 + 语义 aux。**JEPA latent 预测（5 轮中性 + 记忆冲突）与 surprise 清槽修订（RL 有害）均非承重件**——RL 下无实证支持，默认关闭；
- **架构整合状态**：合成流最终配置（无 predictor）早已验证（jeplem8 0.941/0.831）；真实流端到端（感知+记忆+策略）阶段 21/25 已验证（0.840/0.748、0.774/0.577）——完整链路已闭环，无需重做。

## 四、下一步候选（按推荐序）

1. **真实流端到端补测（需 minigrid 环境）**：JEPLMAgent 最终配置（无预测无修订）在真实流的完整验证——阶段 21/25 已部分验证（0.840/0.748、0.774/0.577），如需补强在 Linux/服务器跑（Windows 无 minigrid，hlj-server 当前离线）。
2. **新方向候选（JEPA 审计 + 遗留清零后的项目走向）**：白皮书 12 区块中尚未攻克的（区块 5 因果、6 组合泛化），或架构整合后的应用级任务（多模块协同的复杂流）。建议开新会话时与主管定方向。
3. ~~感知层自监督探针~~ **搁置**（latent 预测审计全中性，无实证基础；环境不可行）。

## 五、关键文件索引（新会话按需读，不读历史档案）

| 文件 | 角色 | 何时读 |
|---|---|---|
| **本文件** | 交接指针 | 必读 |
| COGNITIVE_SYSTEM_WHITEPAPER.md (15KB) | 北极星/12 区块 | 定方向时 |
| NIGHT_RUN_20260809.md (32KB) | 阶段 1-32 实验日志 | 查具体证据时（有目录可跳） |
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

# 查询提示混淆（阶段 32，JEPA 审计终局）：--query-hint --query-hint-reliability 0.7
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --query-hint --query-hint-reliability 0.7 --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json

# 感知层
python experiments/train_perception_layer.py

# 测试
python -m unittest discover -s tests -v
```

> 纪律：3 seed（444/555/666 或 222/333/41123）、swap 探针、诚实负结果归档。
