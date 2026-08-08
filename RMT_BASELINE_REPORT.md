# 当代基线对照报告：RMT vs PLWM（2026-08-08）

## 问题
PLWM 的记忆/规则修订优势 vs GRU 是否成立？若成立，是否独有？
（此前评估中"当代基线"是第二大风险：只赢 GRU 说服力不足。）

## 基线
**RMT（Recurrent Memory Transformer）**：因果 transformer + 可学习前缀 memory token，
memory token 跨段持久（段边界 detach，truncated BPTT）。监督版用于 DecideAfterDelay，
流式版（PolicyRMTStream）用于连续流/规则修订。与 PLWM/GRU 同 corpus、同训练循环、同 swap 探针。

## 结果（3 seed 均值）

### 1. 监督精确回忆（DecideAfterDelay two-identity）
| 模型 | 参数 | d=8 heldout | d=32 heldout |
|---|---|---|---|
| GRU | 13.5K | 0.875 | **0.496（随机）** |
| RMT mem4/8/16 | 84K | 1.000 | **1.000** |
| PLWM | 25.4K | 1.000 | 0.889 |

### 2. RL 规则修订（两阶段适应，终值）
| 模型 | 终值 |
|---|---|
| GRU | 0.538 |
| evidence | 0.586 |
| PLWM / slots16 | 0.747 / **0.836** |
| RMT mem4 / mem8 | 0.790 / **0.815** |

### 3. u=5 长流（PLWM 已知崩点）
| 模型 | cross |
|---|---|
| PLWM / slots16 | 0.553 / 0.528 |
| RMT mem4 / mem8 | 0.417 / 0.507 |
| GRU | 0.520 |

## 结论

1. **"可寻址记忆 > 压缩隐状态"是机制级结论**：RMT 与 PLWM 共同在 GRU 崩盘的点
   （d=32 精确回忆、规则修订）上成功——GRU 是弱基线。
2. **PLWM 与当代 attention 记忆基线等价**（统计上无差异），记忆/规则修订优势**非独有**。
3. **u=5 长流全部模型崩**（RMT 不例外）——跨 episode 长流是 PPO 共同难点，非 PLWM 缺陷。
4. **PLWM 独有（相对 RMT）价值**：
   - 轻量：25K vs 84K 参数（1/3）
   - 流式 O(1)/帧，无 attention 的 O(L²) 段内开销
   - surprise 门控 + belief 重构的认知语义（可解释槽位写入、规则修订机制可解释）

## 定位调整
"独有架构优越" → **"轻量流式认知系统，记忆机制与当代 attention 记忆等价"**。
PLWM 的差异化卖点从"记忆能力"转移到"成本 + 流式在线 + 认知语义（surprise 门控）"。

## 待办
- 像素版 RMT 对照（生态 + 当代组合）
- surprise 门控差异化场景（RMT 无 surprise 概念——找它失效而 PLWM 生效的任务）
- 显式 fact_ledger/regime 符号组件与神经 belief 混合

## 产物
- `experiments/train_rmt_baseline.py`（监督版 RMT）
- `train_decide_stream_rl.py` / `train_decide_stream_rule_adaptation.py`（PolicyRMTStream）
- `artifacts/rmt-d{8,32}-seed*.json`、`rule-adapt-rmt-u4g4-seed*.json`、`stream-u5g4-rmt-seed*.json`
