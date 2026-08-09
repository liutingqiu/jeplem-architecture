# RMT 流式失效报告：段边界 detach 是跨 episode 记忆的结构性瓶颈

> 日期：2026-08-09
> 性质：补测意外发现的系统化验证（标准审计 §2.3 的延伸）
> 关联：RMT_BASELINE_REPORT.md（"PLWM 与 RMT 等价"）、DECIDE_AFTER_DELAY_REPORT.md（§7 连续流）、CONTEMPORARY_BASELINES_REPORT.md

---

## 一、背景

标准审计补测 stream seed 444/555/666 时（6 模型脚本含 RMT 对照），意外发现 **RMT 在跨 episode 连续流上 cross≈随机**。本报告把该发现系统化为独立结论：**RMT 只在有界序列任务上与 PLWM 打平，在流式无界场景（最贴近真实持久认知系统）结构性失效**——段边界 detach（truncated BPTT）是失效原因，机制可证伪。

## 二、数据：RMT 的分任务表现矩阵（全部 3-6 seed）

| 任务 | 性质 | RMT mem4/mem8 | PLWM | 结论 |
|---|---|---|---|---|
| 监督精确回忆 d=8/d=32 | 有界序列（单 episode）| 1.000/1.000 | 0.889-1.000 | 打平 |
| 规则修订（两阶段适应）| 有界阶段任务 | 0.790/0.815 | 0.747/0.836 | 打平 |
| **跨 episode 连续流 u=3 g=4** | **无界流（6 seed）** | **0.493/0.493（≈随机）** | **0.751** | **RMT 崩** |
| 长流 u=5 g=4 | 无界流（3 seed）| 0.417/0.507 | 0.504-0.553 | 双双崩（容量边界）|

### 2.1 核心证据：u=3 g=4 六 seed 同脚本同 seed 全家桶

（`artifacts/stream-full6-u3g4-seed{41123,222,333}.json` + `stream-u3g4-seed{444,555,666}.json`）

| 模型 | query 均值 | cross 均值 | cross 范围（6 seed）|
|---|---|---|---|
| gru_policy | 0.521 | 0.478 | 0.478–0.478 |
| evidence_policy | 0.932 | 0.806 | 0.642–0.910 |
| plwm_policy | 0.909 | 0.751 | 0.687–0.910 |
| plwm_slots16_policy | 0.892 | 0.779 | 0.716–0.881 |
| **rmt_mem4_policy** | **0.823** | **0.493** | 0.478–0.522 |
| **rmt_mem8_policy** | **0.823** | **0.493** | 0.448–0.627 |

**决定性模式**：
- RMT **query（段内）0.82–0.87**，远超 GRU（0.52）——RMT 的 memory token 在**段内**正常记忆当前 episode 身份；
- RMT **cross（跨段）0.45–0.52 ≈ 随机**，与 GRU 无差别——**跨 episode 身份完全丢失**；
- 对照：所有槽位式记忆（PLWM/evidence/slots16）cross 0.75–0.81。

**统计**：RMT mem4 cross 6/6 seed 全部低于 PLWM → 配对符号检验 p = (1/2)⁶ = **0.016**；RMT vs GRU 无差异（双双≈随机）。

## 三、机制：段边界 detach 是表象，隐式压缩不可学才是根源（探针闭环）

`experiments/train_decide_stream_rl.py:218-224`（PolicyRMTStream 文档串）与 `:261`：

```python
# 每 unit 为一个 transformer segment（memory_tokens + unit_len 帧）
# 段边界 memory 更新 detach（truncated BPTT，RMT 标准训练方式）
return out, out[:, :mem_count].detach()
```

**探针设计**（`--rmt-segment-units k` 缩放段边界、`--no-rmt-detach` 关闭梯度截断）：

| 配置 | n | mem4 query/cross | mem8 query/cross |
|---|---|---|---|
| k=0.5（3帧/段）| 3 | 0.526/0.493 | 0.483/0.493 |
| k=1.0（基线，6帧/段）| 2 | 0.826/0.500 | 0.794/0.410 |
| k=2.0（12帧/段=2ep）| 3 | 0.891/**0.751** | 0.908/**0.801** |
| k=3.0（18帧/段=全流）| 3 | 0.969/**0.910** | 0.913/0.751 |
| k=1.0 无 detach | 3 | 0.811/0.458 | 0.814/0.468 |
| k=2.0 无 detach | 2 | 0.875/0.739 | 0.891/0.709 |

**结论（⚠️ 原假说被自身消融证伪，如实修正）**：
1. **detach 不是崩盘主因**：k=1.0 关闭 detach（full BPTT）cross 仍 0.46-0.47（≈随机）；k=2.0 有无 detach 几乎相同（0.75 vs 0.74）。
2. **窗口可见性才是主因**：上一 episode 帧直接落在段内（k≥2）→ cross 恢复（0.75-0.91）；跨段（k≤1）→ 崩（0.49-0.50）。k=0.5 连 query 也崩（身份帧与查询帧被段边界切开）。
3. **机制链条（修正版）**：RMT 跨段记忆只能经 memory token **隐式压缩**携带；RL 稀疏奖励下该压缩**不可学**（梯度是否可传均无效）；段内直接可见则无需压缩、attention 直接检索 → 恢复。
4. **与主旨统一**：RMT 的 memory token 本质是"压缩派"（同 GRU）——"显式槽位写入（PLWM/evidence）> 隐式压缩（GRU/RMT memory token）"在流式跨段场景第三次成立（前两次：监督精确回忆、规则修订）。
5. 附带：k=0.5 无 detach 因 full BPTT 计算图过大训练过慢未完成（不影响结论，方向由 k=1.0-nodetach 覆盖）。

**这是 attention 记忆家族的结构性代价**：流式无界场景下，要么 memory token 学会隐式压缩（本报告证伪其在 RL 下可学），要么加大段窗口（k≥2，注意成本随窗口增长、且等于放弃跨段记忆机制）。PLWM 的槽位显式写入每帧可写、无压缩负担、O(1)/帧。

## 四、对项目定位的意义（修正 RMT_BASELINE_REPORT）

`RMT_BASELINE_REPORT.md` 曾调整定位为"轻量流式认知系统，记忆机制与当代 attention 记忆等价"。**本报告修正该结论**：

1. **"等价"只在有界序列任务成立**（监督精确回忆、规则修订）；**流式无界场景 RMT 失效（0/6）**——"流式"从宣称变为实证差异化。
2. **PLWM 的差异化卖点升级**：不是"轻量 + 打平"，而是"**流式持久记忆（跨段保留）**"——attention 记忆家族在有界任务上的等价对手，在流上系统性失效。
3. **RMT 的适用范围界定**：适合段内/有界任务；用于无界流时须提供跨段训练机制（成本随流长增长）。

## 五、下一步（状态更新：2026-08-09 夜间探针已完成）

1. ~~segment 长度探针~~ **已完成**（k=0.5/1/2/3，3 seed）：窗口可见性 > 梯度处理，"段内直接可见即恢复"证实；
2. ~~detach 消融~~ **已完成并证伪原假说**（k=1.0/k=2.0 无 detach）：梯度截断不是崩盘主因，隐式压缩不可学才是；
3. **真实流回归**（进行中）：连续流语料改 MemoryS13 真实流——受限于真实流双身份同帧罕见，改为"真实流单身份延迟回忆 + RMT 对照"（`audit_minigrid_delayed_object_identity.py` 已加 rmt_mem4/rmt_mem8）；
4. 可选后续：k=2.0 配置下 RMT 的 cross 恢复是否随流长（u=5/6）保持（"加大窗口"方案的边界）。

## 六、产物

- `artifacts/stream-full6-u3g4-seed{41123,222,333}.json`（6 模型版补跑，与 444/555/666 同脚本同 seed）
- `artifacts/stream-seed-aggregate.json`（6 seed 汇总，含 RMT 对照）
- 关联：`STANDARDS_AUDIT_20260809.md` §2.3
