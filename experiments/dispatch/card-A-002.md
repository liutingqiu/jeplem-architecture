---
name: card-A-002
card-id: A-002
assignee: 卡A会话
created: 2026-08-07
priority: high
depends-on: A-001 (已完成)
status: pending
---

# 卡 A 任务卡 A-002：v1.1 更新 — QR 投影验证 + 最终裁决

## 背景

A-001 已完成（v1.0），但工程实施后有了新的验证结果需要纳入报告。

## 任务

### 任务 1：补充 QR 投影验证结果

在 `PLWM_A_RESEARCH_REPORT.md` 中新增 §二.1（P1c 补充）：

```
工程实施：A-P1c 要求键正交初始化，但我们发现 Adam 训练步会破坏正交性。
解决方案：每个 optimizer.step() 后调用 enforce_key_orthogonality()，
使用 QR 投影 W ← Q（保证行向量正交归一）。

验证结果：
- 初始正交性：max off-diag = 0.000000
- 训练 10 步后：max off-diag = 0.000000（全程保持）
- train_associative_memory.py：1.0 准确率（不变）
- train_continuity.py：1.0 准确率（不变）
- 梯度流正常：backward + enforce 闭环无问题
```

### 任务 2：更新判定表

在 §三 综合判定表中更新 P1c 行：

| 命题 | 判定 | 关键条件 | 可复算依据 |
|------|------|---------|-----------|
| **P1c** | **证实（QR 投影保正交）** | 键正交初始化 + 每步 QR 投影 | 训练 10 步后 orth=0.000000 |

### 任务 3：更新 §五 开放问题

删除"键正交初始化方式"问题（已解决）。新增：
1. β 自适应学习：能否让 β 成为可学习参数，使其自动适应键的分离度？
2. 值向量正交性：C-P3c 建议值也需正交初始化，这会影响 P1c 的结论。

### 任务 4：更新 §六 对工程的影响

补充：
- `enforce_key_orthogonality()` 方法已加入 `TorchPLWM`，所有训练脚本需调用
- QR 投影的额外计算成本：O(d³) 每步，d=16 时 ≈ 4096 次乘法，可忽略

## 交付物

更新 `PLWM_A_RESEARCH_REPORT.md` 版本为 v1.1。

## 回报格式

在 `experiments/dispatch/status-A.md` 覆盖写入：
```markdown
---
card-id: A-002
status: completed
timestamp: 2026-08-07
summary: v1.1 更新完成，QR 投影验证通过
---

关键结论：
- P1c 工程实现：QR 投影保正交，训练全程 orth=0.000000
- 训练脚本兼容性：所有现有训练通过
```
