# HJPLWM: Hierarchical JEPA-PLWM 架构设计报告

> 制定时间：2026-08-08
> 设计者：总控（Agnes）
> 状态：待实施

---

## 一、问题诊断

### 1.1 当前 PLWM 的局限

基于对 `torch_model.py` 的分析和实验结果，当前 PLWM 存在以下问题：

| 问题 | 表现 | 根因 |
|------|------|------|
| **单一 latent state** | 长延迟任务（6步）所有模型卡在 50% | belief/workspace 混合处理不同时间尺度 |
| **无辅助训练目标** | short_delay_2id 仅 37.5% | DQN terminal reward 太稀疏，credit assignment 失效 |
| **记忆扁平化** | 无法区分"近期观察"和"持久事实" | 单一 episodic memory 无层级 |
| **无 latent prediction** | 世界模型仅用于 policy | 缺少 JEPA 式的 latent prediction 辅助损失 |

### 1.2 JEPA 的启发

JEPA 的核心创新：
1. **层级 latent 预测** — 多层级表征处理不同时间尺度
2. **Masked prediction** — 强制模型学习上下文到目标的映射
3. **Latent space prediction** — 不重建像素，只预测 latent
4. **Anti-collapse** — 防止表征坍缩

### 1.3 设计原则

**采纳 JEPA 的优点：**
- ✅ 层级 latent 空间（fast/slow）
- ✅ Masked prediction auxiliary loss
- ✅ Latent space prediction（不重建像素）
- ✅ Anti-collapse 正则化

**保留 PLWM 的优点：**
- ✅ 显式世界模型（counterfactual planning）
- ✅ Episodic memory（跨 episode 知识保留）
- ✅ Evidence-based belief revision
- ✅ Online RL（与 JEPA 的 offline 互补）

**拒绝 JEPA 的缺点：**
- ❌ 无显式世界模型 → PLWM 保留
- ❌ 无持久记忆 → PLWM 保留
- ❌ 无信念更新 → PLWM 保留
- ❌ 纯离线学习 → PLWM 在线 RL 互补

---

## 二、HJPLWM 架构设计

### 2.1 核心架构

```
                    ┌─────────────────────────────────────┐
                    │         HJPLWM Architecture         │
                    ├─────────────────────────────────────┤
                    │                                     │
  Observation ──→ [Encoder] ──→ Fast Latent (f)          │
                    │               ↓                     │
                    │         [Fast Cell] ──→ Fast State  │
                    │               ↓                     │
                    │         [Slow Cell] ──→ Slow State  │
                    │               ↓                     │
                    │         [Hierarchical Memory]       │
                    │           ├─ Fast Memory (recent)   │
                    │           └─ Slow Memory (persistent)│
                    │               ↓                     │
                    │         [Prediction Heads]          │
                    │           ├─ Latent Prediction      │
                    │           ├─ Observation Prediction │
                    │           └─ Policy Head            │
                    │                                     │
                    └─────────────────────────────────────┘
```

### 2.2 层级 Latent 空间

**Fast State (f)**:
- 维度：`latent_dim`（当前 48）
- 更新频率：每步
- 作用：处理短期视觉动态、即时决策
- 类比：JEPA 的 low-level latent

**Slow State (s)**:
- 维度：`latent_dim`（当前 48）
- 更新频率：按需（高 surprise 时）
- 作用：处理长期身份记忆、规则学习
- 类比：JEPA 的 high-level latent

**切换机制**：
```python
# 基于 surprise 的动态切换
surprise = ||f_t - Predict(f_{t-1})||
if surprise > threshold:
    s = Update(s, f_t)  # 写入 slow state
else:
    s = s  # 保持 slow state
```

### 2.3 Hierarchical Memory

**Fast Memory**:
- 容量：`memory_slots`（当前 8）
- 作用：存储近期观察（最近 N 步）
- 写入：每步或高 surprise 时
- 读取：基于当前 fast state query

**Slow Memory**:
- 容量：`memory_slots`（当前 8）
- 作用：存储持久事实（身份、规则）
- 写入：仅当 slow state 更新时
- 读取：基于 slow state query

**记忆检索**：
```python
# Fast memory retrieval
fast_query = QueryHead(f)
fast_retrieval = Attention(fast_memory, fast_query)

# Slow memory retrieval
slow_query = QueryHead(s)
slow_retrieval = Attention(slow_memory, slow_query)

# 融合
retrieved = Concat(fast_retrieval, slow_retrieval)
```

### 2.4 训练目标

**主损失（DQN Policy Loss）**：
```python
L_policy = SmoothL1Loss(Q(s, f, retrieved), target_Q)
```

**辅助损失 1（Latent Prediction Loss）**：
```python
# JEPA-style: predict next latent from context
f_pred = Predictor(f_t, action_t)
L_latent = MSE(f_pred, f_{t+1})
```

**辅助损失 2（Masked Prediction Loss）**：
```python
# Mask random timesteps, predict masked latents
f_masked = Mask(f_t, mask_ratio=0.25)
f_recon = Reconstructor(f_context, f_masked)
L_masked = MSE(f_recon, f_t)
```

**辅助损失 3（Anti-collapse Loss）**：
```python
# JEPA-SCORE style: encourage diverse representations
L_anti_collapse = GradientPenalty(f, f_t)
```

**总损失**：
```python
L_total = L_policy + α * L_latent + β * L_masked + γ * L_anti_collapse
```

**推荐超参数**：
- α = 0.1（latent prediction weight）
- β = 0.05（masked prediction weight）
- γ = 0.01（anti-collapse weight）

---

## 三、实验设计

### 3.1 实验 1：短延迟验证（Baseline）

**目的**：验证 HJPLWM 在短延迟任务上优于当前 PLWM

**任务**：short_delay_2id（3 步延迟）
- 身份在 step 0 可见
- 查询在 step 2 可见
- 终端决策在 step 3

**模型对比**：
1. PLWM（当前架构）
2. HJPLWM（新架构，无辅助损失）
3. HJPLWM-full（新架构，含所有辅助损失）

**验收标准**：
- PLWM held-out ≥ 0.75
- HJPLWM vs PLWM gap ≥ 0.10

### 3.2 实验 2：中延迟扩展

**目的**：验证层级 latent 是否帮助处理更长延迟

**任务**：extended_delay_2id（5 步延迟）
- 身份在 step 0 可见
- 中间有视觉线索（step 2, 3）
- 查询在 step 4 可见
- 终端决策在 step 5

**验收标准**：
- HJPLWM held-out ≥ 0.60
- HJPLWM vs PLWM gap ≥ 0.15

### 3.3 实验 3：长延迟对比

**目的**：验证 HJPLWM 是否能解决 addressed_balanced_2id 的阻塞

**任务**：addressed_balanced_2id（6 步延迟，已失败）
- 身份在 step 0 可见（z=-7）
- 查询在 step 5 可见（z=-2）
- 终端决策在 step 6

**假设**：
- 如果 HJPLWM 的 slow state 能持久保留身份信息，可能突破 50% 瓶颈
- 如果仍然失败，证明是任务设计问题，不是架构问题

**验收标准**：
- HJPLWM held-out > 0.50（当前所有模型 50%）
- 如果通过，证明架构改进有效

### 3.4 实验 4：辅助损失 ablation

**目的**：验证每个辅助损失的贡献

**配置**：
1. 无辅助损失（baseline）
2. + Latent prediction loss
3. + Masked prediction loss
4. + Anti-collapse loss
5. 全部辅助损失

**验收标准**：
- 每个辅助损失应有正向贡献
- 全部辅助损失效果最佳

---

## 四、实现计划

### 4.1 阶段 1：架构实现（3-5 天）

**文件**：`plwm/hjplwm.py`

**实现内容**：
1. `HierarchicalLatentState` 数据类
   - fast_state: Tensor
   - slow_state: Tensor
   - fast_memory: Tensor (B, M, d)
   - slow_memory: Tensor (B, M, d)

2. `HJPLWM` 模型
   - Fast cell: GRUCell(latent_dim * 2, latent_dim)
   - Slow cell: GRUCell(latent_dim * 2, latent_dim)
   - Hierarchical memory with separate query heads
   - Prediction heads for latent, observation, policy

3. 辅助损失函数
   - latent_prediction_loss
   - masked_prediction_loss
   - anti_collapse_loss

### 4.2 阶段 2：实验脚本（2-3 天）

**文件**：
- `experiments/train_hjplwm_short_delay.py`
- `experiments/train_hjplwm_extended_delay.py`
- `experiments/train_hjplwm_addressed.py`

**实现内容**：
1. 基于 `train_luanti_short_delay_2id.py` 改编
2. 替换模型为 HJPLWM
3. 添加辅助损失训练逻辑
4. 添加 ablation 实验配置

### 4.3 阶段 3：实验运行（5-7 天）

**实验顺序**：
1. 实验 1：短延迟验证（快速，1-2 天）
2. 实验 2：中延迟扩展（2-3 天）
3. 实验 3：长延迟对比（3-5 天）
4. 实验 4：Ablation（2-3 天）

### 4.4 阶段 4：报告撰写（1-2 天）

**输出**：
- 实验报告（JSON + Markdown）
- 架构改进建议
- 下一步方向

---

## 五、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| HJPLWM 在短延迟失败 | 中 | 高 | 回退到 PLWM，分析失败原因 |
| 辅助损失干扰主任务 | 中 | 中 | 调整 α, β, γ 超参数 |
| 训练不稳定 | 低 | 中 | 添加梯度裁剪、学习率调度 |
| 计算资源不足 | 低 | 低 | MiniGrid 可 CPU 运行 |

---

## 六、预期收益

### 6.1 如果成功

1. **架构改进验证**：层级 latent + 辅助损失有效
2. **长延迟突破**：解决 addressed_balanced_2id 阻塞
3. **证据增强**：PLWM 在更多任务上有架构优势
4. **理论贡献**：JEPA + PLWM 的融合架构

### 6.2 如果失败

1. **负结果也有价值**：证明当前架构方向有误
2. **重新评估目标**：可能需要完全不同的架构
3. **积累知识**：对 JEPA 和 PLWM 的理解更深入

---

## 七、结论

HJPLWM 是 PLWM 的自然演进，它：

1. **采纳 JEPA 的优点**：层级 latent、masked prediction、latent space prediction
2. **保留 PLWM 的优点**：显式世界模型、持久记忆、信念更新、在线 RL
3. **解决当前阻塞**：通过层级 latent 处理不同时间尺度
4. **可证伪**：每个假设都有明确的验收标准

**这不是对 JEPA 的模仿，而是对 PLWM 的增强。**

---

## 八、下一步

1. **立即**：实现 `plwm/hjplwm.py`
2. **今天**：运行实验 1（短延迟验证）
3. **本周**：完成所有实验并撰写报告
