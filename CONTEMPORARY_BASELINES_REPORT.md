# 当代架构对比报告：RMT / Titans / MoE-GRU vs PLWM（2026-08-08）

## 问题
PLWM 的记忆/规则修订优势是否成立？是否独有？（用户直接要求与 transformer、MoE 等当代架构对比）

## 基线（全部同 corpus、同训练循环、同 swap 探针）
| 基线 | 机制 | 参数 |
|---|---|---|
| GRU | 压缩隐状态（1994） | 13.5K |
| **MoE-GRU** | GRU + Top-2 稀疏专家（容量扩展） | 16.5K |
| **Titans** | surprise 门控神经记忆（Google 2024 简化） | 50K |
| **Titans-nosurprise** | 同上但去掉 surprise 门控（FIFO 写入） | 50K |
| **RMT** | 因果 transformer + 前缀 memory token（Bulatov 2022） | 84K |
| PLWM | surprise 门控内容寻址槽位 + belief 重构 | 25K |

## 结果（3 seed 均值，heldout / swap-evidence）

### 监督精确回忆 d=8（基准）
gru 0.875 / moe_gru 0.820 / titans 1.000 / titans_nosurprise 1.000 / rmt 1.000 / plwm 1.000

### 监督精确回忆 d=32（区分点，GRU 崩盘处）
| 模型 | heldout | 结论 |
|---|---|---|
| GRU | 0.496 | 随机 |
| **MoE-GRU** | **0.508** | **容量扩展救不了压缩隐状态** |
| Titans-nosurprise | 0.496 | 去掉 surprise 门控即崩 |
| **Titans** | **0.773** | surprise 门控是承重机制 |
| **PLWM** | 0.781 | 与 Titans 打平 |
| **RMT** | **1.000** | attention 无损检索最强 |

### 规则修订（此前数据）
GRU 0.538 / evidence 0.586 / PLWM slots16 0.836 / RMT mem8 0.815

### 像素生态效度（此前数据）
GRU 0.801 / evidence 1.000 / PLWM 1.000

## 结论（机制层级，d=32 精确回忆）

```
RMT (attention 无损记忆) 1.000
   > PLWM 0.781 ≈ Titans (surprise 门控) 0.773
   > GRU ≈ MoE-GRU ≈ Titans-nosurprise ≈ 0.50 (随机)
```

1. **"可寻址记忆 > 压缩隐状态"牢固成立**：GRU 系（含 MoE 容量扩展）在长延迟精确回忆上结构性失败，容量不是解药——优势来自"可寻址存储"而非参数。
2. **surprise 门控是可分离的真实组件**：Titans 0.773 vs Titans-nosurprise 0.496——同一架构仅去掉 surprise 门控就崩。PLWM 的 gated memory 设计（此前 d=32 扫描 0.862 vs evidence 无门控 0.734 的间接证据）被直接对照证实。
3. **PLWM ≈ Titans**：surprise 门控家族的等价实现。
4. **RMT 最强但最贵**：attention 无损检索在记忆任务上胜过 surprise 门控槽位（RMT 84K 参数 + O(L²)；PLWM 25K + 流式 O(1)）。
5. **PLWM 相对当代架构的定位**：不是"最强记忆"，而是"轻量流式 + surprise 门控认知语义"的记忆系统——比 RMT 便宜 3 倍参数且流式在线，比 Titans 多 belief 重构/规则修订语义。

## 对项目定位的最终调整
- "独有架构优越" ❌ → 已排除
- "机制级结论（可寻址 > 压缩）+ surprise 门控实证 + 轻量流式差异点" ✅

## 产物
- `experiments/train_rmt_baseline.py`（RMT + Titans + MoE-GRU + 对照）
- `artifacts/rmt-full-d{8,32}-seed*.json`、`rmt-d8-moe-titans-smoke.json`
- 关联：`RMT_BASELINE_REPORT.md`（首版）、`DECIDE_STREAM_RULE_ADAPTATION_REPORT.md`
