# 规则修订实验：两阶段适应（信念修订 / 持续学习维度首个证据）

日期：2026-08-08 · 设备：CUDA (RTX 4060 Laptop) · 3 seeds (222/333/444)

## 一、动机与设计

DecideStream 规则切换实验（`stream-switch-*`）发现：**PPO 在线适应规则突变失败**
（切换后所有模型 ≤ 0.54，evidence 固化在旧规则 0.370）。但那混入了"切换检测"难度。

本实验分离出干净的**规则修订**问题（`train_decide_stream_rule_adaptation.py`）：

- **阶段 1**：4-episode 连续流（gap=4），规则 0（choice == identity），PPO 训练 100 updates 到收敛
- **阶段 2**：规则翻转为 1（choice == complement），继续 PPO 训练 100 updates（每 20 updates 评估）
- 指标：stage2 适应曲线——从经验（仅奖励反馈）修订已学规则的速度
- 无 teacher / oracle / 特权状态；replay 无（纯在线）

## 二、结果（3 seed 均值）

| 模型 | stage1 (rule0) | S2块1 | 块2 | 块3 | 块4 | 块5(终) | 3 seed 终值 |
|---|---|---|---|---|---|---|---|
| GRU | 0.788 | 0.514 | 0.540 | 0.520 | 0.578 | **0.538** | 0.512/0.559/0.543 |
| evidence | 0.788 | 0.457 | 0.457 | 0.457 | 0.457 | 0.586 | 0.457/0.457/0.844 |
| PLWM | 0.788 | 0.332 | 0.440 | 0.448 | 0.602 | **0.747** | 0.629/0.879/0.734 |
| PLWM slots16 | 0.788 | 0.382 | 0.467 | 0.664 | 0.798 | **0.836** | 0.832/0.684/0.992 |

## 三、解读

1. **GRU 结构性无法修订规则**：100 updates 持续卡在随机（0.54），压缩隐状态在规则翻转下既不能保留新映射也无法丢弃旧映射——3/3 seed 失败。
2. **evidence 记忆固化**：前 4 块（80 updates）完全僵化在旧规则（0.457），无 surprise 门控的 FIFO 槽位不因预测失败而重写；seed 444 偶然突破到 0.844，不可靠。
3. **PLWM 架构性适应**：3/3 seed 爬升（0.63-0.99），slots16 达到 0.836 均值。机制解释：规则翻转 → 旧规则下决策失败 → **surprise 高 → 新证据经 gated memory 写入 → belief 重构推翻旧规则**。slots16 更多槽位缓冲新旧证据 → 恢复最快。
4. **初期"过拟合"模式**：PLWM 切换第一块 0.332（低于随机）——高置信度旧信念难撼动，证据累积后快速重写。这是"先坚持后修订"的可解释行为，与 evidence 的"永远僵化"有本质区别。

## 四、结论

**信念修订 / 持续学习维度：PLWM 架构优势首个实证（继记忆优势后的第二个独立维度）**。

- 优势机制：gated memory（surprise 驱动写入）+ belief 重构 > 压缩隐状态（GRU）> 无门控 FIFO（evidence）
- 对架构设计的支持：规则的检测/修订需要"预测失败 → 证据重写"的机制，这与 surprise 重建目标在策略主损失中是负担（此前结论）不矛盾——**surprise 应作为记忆写入门控信号，而非策略梯度目标**
- 下一步：① 更多 seed 巩固（5-10）② 将 VersionedFactLedger/regime 符号组件与神经 belief 混合，测试显式规则修订能否进一步提升 ③ 中间块粒度分析（适应曲线形状）

## 五、产物

- 代码：`experiments/train_decide_stream_rule_adaptation.py`、`train_decide_stream_rl.py`（switch_at 支持）
- 数据：`artifacts/rule-adapt-u4g4-seed{222,333,444}.json`、`stream-switch-*`、`stream-noswitch-*`
- 关联：`DECIDE_AFTER_DELAY_REPORT.md`（记忆维度）、`WORLD_MODEL_SURVEY_202608.md`（JEPA/世界模型调研）
