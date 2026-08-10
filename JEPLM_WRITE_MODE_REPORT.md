# JEPLM 写入门控对照报告：FIFO vs learned-gate vs delta-rule（2026-08-11）

## 背景（为什么做）
Kimi K3（2607.24653）的 KDA = delta-rule 递归 + 通道级 forget gate（内容定向擦写）。
调研结论："delta-rule 擦写 = 未测的第三类写入门控，列入实验队列"（LATEST_ARCHITECTURE_SURVEY §六）。
同时审计 `plwm/jeplem_memory.py` 发现 **write_gate 死代码**：gate 计算了但从未接入写路径，
实际行为是无条件 FIFO 写入（文档声称的"learned gate"未实现）——这解释了 jeplem8 ≈ evidence
（两者都 FIFO）。

## 本次改动
1. 修复实现缺口：`JEPLMMemory` 新增 `write_mode` 参数（`fifo`/`gate`/`delta_rule`）：
   - `fifo`：无条件 FIFO 轮换（2026-08-11 之前的实际行为，基线）
   - `gate`：文档声称的学习门控——门控调制"被 recall 槽"的替换强度（dead-code 补接线）
   - `delta_rule`：KDA 式内容寻址擦写——按 key 相似度 argmax 选槽，相似度调制 erase 强度 + 学习 add 强度
2. 透传至 PolicyJEPLM / JEPLMAgent；`--jeplem-write-mode` 参数。

## 结果（u3g4 连续流 RL，100 updates×24，3 seed，标准协议）
| 写模式 | q 均值 | c 均值 | 逐 seed (q/c) |
|---|---|---|---|
| **fifo（无条件）** | **0.941** | **0.831** | 444:0.958/0.881, 555:0.958/0.881, 666:0.906/0.731 |
| gate（学习门控） | 0.821 | 0.547 | 444:0.849/0.687, 555:0.807/0.448, 666:0.807/0.507 |
| delta_rule（KDA 擦写）| 0.793 | 0.657 | 444:0.750/0.836, 555:0.823/0.687, 666:0.807/0.448 |

对照（历史）：evidence 0.958/0.881。

## 结论（诚实负结果 + 边界条件）
1. **高信息密度事件流（u3g4：每帧身份/查询）上，无条件 FIFO > 学习门控 > delta-rule**。
   - FIFO 保证每个身份帧必然写入（时间结构完整）；选择性写入（gate/delta_rule）在 RL 稀疏奖励下
     学不好"什么值得记"，引入**漏写风险**（跨 episode 记忆被覆盖，c 掉 0.2-0.3）。
2. **位置寻址写 > 内容寻址写**（在此任务）：身份帧按时间到达，FIFO 保持时间顺序；内容寻址选槽+
   擦写会覆盖同内容槽，破坏跨 episode 时序。
3. **边界条件（不过度外推）**：
   - 低信息密度流（d=32 大量空帧）：PLWM 门控防空帧挤占仍有效（0.862 vs evidence 0.720）——
     门控的用武之地是"冗余帧"，不是"每帧都重要"的流。
   - KDA/delta-rule 在语言建模（大量冗余 token 需选择性压缩）场景合理，事件流任务不适用。
4. **D3 延伸证据**：学习式写入门控在 RL 下是负担而非帮助（surprise/TTCD/delta-rule 三连负结果），
   与"写什么值得记"难以从稀疏奖励学习一致。fifo 是稳健默认。

## 对架构的影响
- JEPLMMemory 默认写模式保持 `fifo`（当前最优）。
- `gate` 实现已真实接线（供未来"低信息密度流 + 门控"场景复用），但非默认。
- 记忆组件文档更新：显式标注"fifo 无条件写 = 事件流默认；门控适用于冗余帧流"。

## 产物
- `plwm/jeplem_memory.py`（write_mode 参数 + 3 模式）
- `plwm/jeplem_agent.py`、`experiments/train_decide_stream_rl.py`（透传）
- `artifacts/stream-wmode-{fifo,gate,delta_rule}-s{444,555,666}.json`（9 个）
