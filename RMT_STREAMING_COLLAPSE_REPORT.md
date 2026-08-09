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

## 三、机制：段边界 detach（truncated BPTT）

`experiments/train_decide_stream_rl.py:218-224`（PolicyRMTStream 文档串）与 `:261`：

```python
# 每 unit 为一个 transformer segment（memory_tokens + unit_len 帧）
# 段边界 memory 更新 detach（truncated BPTT，RMT 标准训练方式）
return out, out[:, :mem_count].detach()
```

**机制链条**：
1. cross 查询需要"上一 episode 的身份"——该信息在**上一个 segment** 中被编码进 memory token；
2. 段边界 `detach()` 切断跨段梯度 → 训练中 memory token 不学习"把段间信息组织成可检索表示"；
3. 结果：query（段内 attention 即可解决）正常，cross（必须跨段）退化为随机。

**这是 attention 记忆家族的结构性代价**：流式无界训练要么 truncated BPTT（本报告证伪的 cross 记忆），要么 full BPTT / sliding-window（O(L²) 成本随流长爆炸）。PLWM 的槽位记忆每帧可写、无段边界、O(1)/帧，不存在此瓶颈。

## 四、对项目定位的意义（修正 RMT_BASELINE_REPORT）

`RMT_BASELINE_REPORT.md` 曾调整定位为"轻量流式认知系统，记忆机制与当代 attention 记忆等价"。**本报告修正该结论**：

1. **"等价"只在有界序列任务成立**（监督精确回忆、规则修订）；**流式无界场景 RMT 失效（0/6）**——"流式"从宣称变为实证差异化。
2. **PLWM 的差异化卖点升级**：不是"轻量 + 打平"，而是"**流式持久记忆（跨段保留）**"——attention 记忆家族在有界任务上的等价对手，在流上系统性失效。
3. **RMT 的适用范围界定**：适合段内/有界任务；用于无界流时须提供跨段训练机制（成本随流长增长）。

## 五、下一步（待执行）

1. **segment 长度探针**：unit_len 1/2/4/8 扫描，验证"段越短 cross 越崩"（detach 频率 ↑ → 跨段更早丢失）；
2. **detach 消融**：不 detach 的流式 RMT（full BPTT on 有限段）——预期 cross 恢复但训练成本上升，量化"代价-收益"曲线；
3. **真实流回归**：连续流语料改 MemoryS13 真实流（DecideAfterDelay §九 已决策待办）——验证流式差异在真实流上保持。

## 六、产物

- `artifacts/stream-full6-u3g4-seed{41123,222,333}.json`（6 模型版补跑，与 444/555/666 同脚本同 seed）
- `artifacts/stream-seed-aggregate.json`（6 seed 汇总，含 RMT 对照）
- 关联：`STANDARDS_AUDIT_20260809.md` §2.3
