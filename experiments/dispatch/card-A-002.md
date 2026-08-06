---
name: card-A-002
card-id: A-002
assignee: 卡A会话
created: 2026-08-07
priority: high
depends-on: A-001 (已完成)
status: pending
---

# 卡 A 任务卡 A-002：v1.1 收尾 — 工程验证结果纳入裁决

## 评估

卡 A 原始使命（P1a/P1b/P1c 数学裁决）已完成（v1.0）。
本任务仅补充工程验证结果，使报告从"纯数学"升级为"数学+工程验证"的完整裁决。

## 任务（轻量，~15分钟）

### 任务 1：在 PLWM_A_RESEARCH_REPORT.md 追加 §七

```markdown
## 七、工程验证（v1.1 补充）

A-P1c 预测：键正交初始化是训练收敛的必要条件。
工程验证结果（2026-08-07）：

1. QR 投影保正交：每次 optimizer.step() 后调用 enforce_key_orthogonality()
   训练 10 步后正交性误差仍为 0.000000（初始值即 0.000000）

2. 训练兼容性：
   - train_associative_memory.py: 1.0 准确率（不变）
   - train_continuity.py: 1.0 准确率（不变）
   - 梯度流正常，backward + QR 投影闭环无问题

3. 结论：A-P1c 的"需正交初始化"假设在工程上可通过 QR 投影实现，
   不破坏训练动力学，且与 A-P1a 的 β 缩放兼容。
```

### 任务 2：更新判定表

在 §三 综合判定表中 P1c 行的"可复算依据"列追加：
```
A-P1c 工程验证：QR 投影保正交，训练 10 步后 orth=0.000000
```

### 任务 3：更新 §六 对工程的影响

追加：
```
6. QR 投影的额外计算成本：O(d³) 每训练步，d=16 时 ≈ 4096 次乘法，可忽略。
```

### 任务 4：将报告版本更新为 v1.1

修改头部：
```markdown
> 版本：v1.1（数学裁决 + 工程验证）
> 状态：**已完成（v1.0 数学裁决 + v1.1 工程验证）**
```

## 回报

完成后在 `experiments/dispatch/status-A.md` 覆盖写入：
```markdown
---
card-id: A-002
status: completed
timestamp: 2026-08-07
summary: v1.1 收尾完成，QR 投影验证纳入裁决
---

关键结论：
- P1c 工程验证：QR 投影保正交，训练全程 orth=0.000000
- A 的使命：已完成（v1.0 数学裁决 + v1.1 工程验证）
```
