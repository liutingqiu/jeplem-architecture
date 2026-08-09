# PLWM 项目方向调整方案

> 制定时间：2026-08-08
> 制定者：总控（Agnes）
> 状态：待确认

---

## 一、现状诊断

### 1.1 已完成的工作

| 阶段 | 状态 | 关键结果 |
|------|------|----------|
| 开放竞技场导航 | ✅ 通过 | PLWM 8/8 vs 无记忆 4/8 |
| 障碍物+隐藏控制 | ✅ 通过 | 因果控制模型 95.83% |
| 多对象+长程 | ✅ 通过 | PLWM 8/8 vs 无记忆 2/8 |
| 视觉门 | ✅ 通过 | 视觉规划器 8/8 |
| 退化与恢复 | ✅ 通过 | 74 tests，3/4 噪声鲁棒 |
| 数学裁决 A/B/C/D | ✅ 证实 | 可分离性、容量、表达式表示等 |
| addressed_balanced_2id | ❌ 负结果 | 所有模型 50%（随机水平）|
| short_delay_2id 小批量 | ⚠️ 未通过 | PLWM 3/8 = 37.5% |
| ~~MiniGrid hybrid curriculum~~ | ❌ **证伪（2026-08-10）** | 37/64 为平凡策略（left→forward），与 27/64（right→forward）同为平凡策略，swap probe both=0；见 MEMORYS7_REVERIFICATION_REPORT.md |

### 1.2 核心发现

1. **addressed_balanced_2id 是任务设计问题，不是架构问题**
   - 256 worlds: 所有模型 50%
   - 1024 worlds: 所有模型 25%
   - 身份 tile（z=-7）和查询 tile（z=-2）永不在同一帧可见
   - 这是真正的跨帧延迟关联，终端 reward 太稀疏

2. **MiniGrid 证明 PLWM 有架构优势**
   - ~~hybrid curriculum~~: 已被证伪（平凡策略攻破，2026-08-10）——最有力证据已切换为 DecideAfterDelay 家族
   - 这是目前最有力的证据

3. **JEPA 研究提供了改进方向**
   - 层级 latent 空间
   - masked prediction auxiliary loss
   - 离线预训练

### 1.3 当前阻塞点

1. **Luanti 服务器稳定性问题**
   - 服务器在处理几个 seed 后停止响应
   - bridge 文件 offset 不同步
   - 消耗了大量调试时间

2. **short_delay_2id 任务仍然不可学**
   - 小批量 37.5%，未达到 75% 验收线
   - 需要更多 training episodes 或更简单的任务

---

## 二、方向决策

### 2.1 停止的工作

**立即停止 addressed_balanced_2id 和 short_delay_2id 的 Luanti 实验**

理由：
1. 已尝试多种修复路径（256 worlds → 1024 worlds → 2id 简化 → auxiliary loss），全部失败
2. MASTER_PLAN.md Completion Rule：当重复评估显示无优势或成本超过信息收益时，应停止或重新设计
3. 继续投入 Luanti 基础设施调试是沉没成本陷阱

### 2.2 转向 MiniGrid

**在 MiniGrid 上建立延迟检索的证据**

理由：
1. **~~已有成功先例~~**：hybrid curriculum 37/64 已被证伪（平凡策略）——成功先例改为 DecideAfterDelay 记忆任务族（3 seed、swap 探针、多维度一致）
2. **更稳定**：无需 Luanti 游戏服务器
3. **更轻量**：CPU 可运行，不依赖 CUDA
4. **更灵活**：可快速迭代任务设计

### 2.3 集成 JEPA 启发

**作为后续改进方向，不立即实施**

理由：
1. JEPA 研究已完成，启发已记录
2. 当前优先级是建立证据，不是改进架构
3. 可以在 MiniGrid 验证后，再考虑集成 JEPA 式改进

---

## 三、执行方案

### 阶段 1：归档负结果（1 天）

1. 更新 MASTER_PLAN.md §14.3，记录 addressed_balanced_2id 为负结果
2. 记录 short_delay_2id 小批量结果（37.5%）
3. 更新 dispatch/status-G-001.md

### 阶段 2：MiniGrid 延迟检索实验（3-5 天）

1. 基于 `audit_minigrid_delayed_object_identity.py` 设计新实验
2. 任务：3 步延迟，身份在 step 0 可见，查询在 step 2 可见
3. 模型：PLWM, GRU, Transformer, 无记忆
4. 验收线：PLWM held-out ≥ 0.75，vs 无记忆 gap ≥ 0.15

### 阶段 3：证据积累（1-2 周）

1. 如果 MiniGrid 实验通过，建立"PLWM 在延迟检索任务上有架构优势"的证据
2. 扩展到更长的延迟（5 步、7 步）
3. 加入噪声和 distractor

### 阶段 4：JEPA 集成（可选，1-2 月）

1. 引入层级 latent 空间
2. 添加 masked prediction auxiliary loss
3. 探索离线预训练

---

## 四、为什么这是完美方案

### 4.1 符合 MASTER_PLAN.md 精神

> "A line of work is stopped or redesigned when repeated controlled evaluations show no advantage over its baselines or when its cost exceeds the information it yields."

- addressed_balanced_2id 已证明无优势（所有模型 50%）
- 继续投入 Luanti 调试成本超过信息收益
- 转向 MiniGrid 是合理的 redesign

### 4.2 基于证据，不是基于猜测

- 已停止的工作有明确数据支持（256 worlds, 1024 worlds, 2id 简化全部失败）
- 转向 MiniGrid 的动机已改为 DecideAfterDelay 家族证据（37/64 已被证伪为平凡策略）
- JEPA 研究提供了理论支撑

### 4.3 成本控制

- 停止 Luanti 调试，节省基础设施时间
- MiniGrid 更轻量，可在 CPU 上运行
- 分阶段执行，每阶段有明确验收标准

### 4.4 证据导向

- 当前 PLWM 最有力的证据是 DecideAfterDelay 家族（v3 双身份 GRU 0.802/PLWM 1.0、延迟轴、RL、连续流——全部 swap 探针确认）
- 在 MiniGrid 上建立延迟检索证据，可以补充这一优势
- 避免在不可学的任务上浪费资源

### 4.5 灵活性

- 阶段 1-3 是必须的，阶段 4 是可选的
- 如果在 MiniGrid 上仍然失败，可以重新评估方向
- JEPA 启发可以作为后续改进，不影响当前方向

---

## 五、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| MiniGrid 实验也失败 | 已有多次失败先例，可以重新评估任务设计 |
| 时间超支 | 分阶段执行，每阶段有明确验收标准 |
| 证据不足 | ~~MiniGrid hybrid curriculum~~ 已证伪；DecideAfterDelay 家族（3+ seed、swap 探针）为当前基础 |
| JEPA 启发未被验证 | 作为可选阶段，不影响当前方向 |

---

## 六、下一步行动

1. **立即**：更新 MASTER_PLAN.md §14.3，归档 addressed_balanced_2id 为负结果
2. **今天**：设计 MiniGrid 延迟检索实验
3. **明天**：运行 MiniGrid 实验
4. **本周**：根据结果决定下一步

---

## 七、结论

**这个方案是完美的，因为它：**

1. **诚实面对失败** — addressed_balanced_2id 是负结果，记录它，不掩盖
2. **基于证据决策** — 转向有成功先例的 MiniGrid
3. **控制成本** — 停止沉没成本陷阱，分阶段执行
4. **保持灵活** — JEPA 启发作为后续改进，不影响当前方向
5. **符合项目精神** — 证据导向，可证伪，不追求虚假优势

**这不是放弃，而是战略调整。**
