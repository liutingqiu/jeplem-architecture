---
name: card-B-002
card-id: B-002
assignee: 卡B会话
created: 2026-08-07
priority: high
depends-on: A-001 (P1c 正交初始化), B-001 (P2a/P2c)
status: pending
---

# 卡 B 任务卡 B-002：起草辅助检索监督的预声明合同

## 背景

`MASTER_PLAN.md` 末尾已明确指出下一个预声明合同：
> "The next predeclared contract must therefore grade that binding directly:
> an auxiliary retrieval objective that, at the query step, trains the
> write-then-read path"

这是 L1 验收的关键步骤——当前 2id balanced smoke 中所有模型塌陷到 50%（机会水平），
根因是身份↔query 永不同帧，绑定步骤无梯度信号。

## 数学前提（来自 A/B/C/D）

1. **A-P1a**：content-based read 检索误差 `err ≤ (M-1)·exp(-βδ/√d)`，正交键时 δ 足够大
2. **A-P1c**：键正交初始化 + QR 投影保正交 → 检索路径可微
3. **B-P2a**：Luanti 离散网格噪声 ε≈0.3~0.5 ≪ ε_max≈2.3，检索路径稳定
4. **B-P2c**：confidence-weighted softmax 已实现，同 key 冲突不退化
5. **D-P4a**：U₂ 贝叶斯合并保证修订收敛（用于辅助监督的信号传递）

## 任务

起草一份**预声明合同**（frozen contract），包含：

### 1. 合同基本信息
- 课程：`addressed_balanced_2id`（已修复 controls-inversion，seed 范围 18,000,000..19,000,000）
- 设备：CUDA（复用已有 smoke 设置）
- 冻结种子：训练 seed 范围 + held-out seed 范围

### 2. 模型定义
- 4 模型对比：no_memory / GRU / Transformer(2-layer) / PLWM(full)
- 参数量匹配（±5%）
- 所有模型共享同一 ObjectTokenEventEncoder（公平性）
- PLWM 新增：auxiliary retrieval loss（见下）

### 3. Auxiliary Retrieval Loss 设计
基于 A-P1a 和 B-P2a 的数学约束：
- **形式**：`L_aux = -log P(top-1 correct)` 或 MSE(position prediction)
- **监督信号**：在 query 步，给定 query token，预测被查询身份的第一帧位置
- **权重**：α ∈ [0.01, 0.1] 扫参（A-P1c 收敛结构指导）
- **公平性**：所有模型共享同一辅助头（GRU/Transformer 也有等价的 read head）

### 4. 验收线（predeclared，实验前冻结）
- PLWM held-out ≥ 0.75（当前 50%，机会水平）
- PLWM ≥ no_memory + 15pp（显著间隔）
- PLWM ≥ GRU + 10pp
- PLWM ≥ Transformer + 10pp
- 5 模型对比报告 + 独立审计

### 5. 冻结合同落盘
- 合同 JSON 写入 `artifacts/addressed-2id-aux-retrieval-contract.json`
- 包含：seed 范围、模型定义、loss 权重、验收线、冻结时间戳
- **禁止**：实验前不得修改任何参数

### 6. 预期输出
- `experiments/train_luanti_addressed_balanced_2id_aux_retrieval.py`（新训练脚本）
- `experiments/audit_luanti_addressed_balanced_2id_aux_retrieval.py`（审计脚本）
- 合同 JSON 文件
- 结果报告 JSON 文件

## 约束
- 只做合同设计，不写训练代码
- 所有参数必须在实验前冻结
- 公平性论证必须书面化（§8 蓝图框架）

## 交付物

创建以下文件：
1. `experiments/dispatch/contract-addressed-2id-aux-retrieval.json` — 预声明合同
2. `experiments/dispatch/summary-B-002.md` — 合同摘要（1-2 页）

## 回报格式

在 `experiments/dispatch/status-B.md` 覆盖写入：
```markdown
---
card-id: B-002
status: completed
timestamp: 2026-08-07
summary: 预声明合同起草完成
---

关键结论：
- 合同已冻结，验收线预声明
- 辅助检索 loss 设计基于 A-P1a/B-P2a 数学约束
- 公平性论证：所有模型共享辅助头
```
