---
name: card-F-001
card-id: F-001
assignee: 文档整合子agent
created: 2026-08-07
priority: medium
depends-on: none
status: pending
---

# 文档整合任务 F-001：MASTER_PLAN 更新 + 裁决综合

## 背景

A/B/C/D 四个数学卡已全部完成，报告已提交。但 `MASTER_PLAN.md` 尚未反映这些裁决结果，
导致总控无法基于数学结论做工程决策。

## 任务

### 任务 1：更新 MASTER_PLAN.md 末尾

在 `MASTER_PLAN.md` 末尾（`## Completion Rule` 之前）新增章节：

```markdown
## 14. 数学探索裁决综合（2026-08-07）

### A/B/C/D 四卡裁决摘要

| 卡 | 命题 | 判定 | 工程影响 |
|---|---|---|---|
| A-P1a | 可分离性 | 证实 | β 控制检索尖锐度 |
| A-P1b | 容量上限 | 证实 | d=16 理论容量充足 |
| A-P1c | 训练可得多性 | 需正交初始化 | key_head QR 投影 |
| B-P2a | 模糊查询 | 证实 | Luanti 噪声安全 |
| B-P2b | 谓词筛选 | 证实 | L1 单 score 足够 |
| B-P2c | 一致性 | 需元数据 | confidence-weighted softmax |
| C-P3a | 表达式表示 | 证实 | 矩阵半群 = 最小结构 |
| C-P3b | 时序必要性 | 证实 | 非交换性必须 |
| C-P3c | 值→命题统一 | 需类型系统 | value/rule 标签 |
| D-P4a | 修订收敛 | 证实 | U₂ 贝叶斯合并推荐 |
| D-P4b | 新旧并存 | 证实（条件） | c_e 需可靠 |
| D-P4c | 冲突检测 | P（简化） | 先做同 key 比对 |

### 工程修复状态

- ✅ β 缩放（torch_model.py:110）
- ✅ 键正交初始化（torch_model.py:99-108）
- ✅ QR 投影保正交（torch_model.py:137-143）
- ✅ confidence-weighted retrieval（torch_model.py:105-114）
- ✅ 向后兼容旧 checkpoint

### L1 下一步

预声明合同已起草（见 experiments/dispatch/contract-addressed-2id-aux-retrieval.json）：
- 课程：addressed_balanced_2id
- 辅助检索监督 loss
- 验收线：PLWM held-out ≥ 0.75
```

### 任务 2：创建裁决综合摘要文档

创建 `PLWM_MATH_SUMMARY.md`（1-2 页），面向决策者：
- 一句话总结：四个数学卡全部完成，核心结论是"检索可行但需正交初始化+高β"
- 三个关键数字：β≥12（检索）、ε_max≈2.3（噪声容忍）、U₂ 收敛（修订）
- 对蓝图的三个影响：L1 检索修复（已完成）、L2 信念修订（U₂ 推荐）、L3 推理（矩阵半群）
- 风险登记：β 自适应学习（开放）、值正交初始化（C 建议但未验证）

### 任务 3：更新 dispatch 协议

在 `experiments/dispatch/PROTOCOL.md` 中：
- 标记 A/B/C/D 全部完成
- 新增 E/F 任务状态
- 更新当前阶段说明

## 验收标准
- MASTER_PLAN.md 新增 §14
- PLWM_MATH_SUMMARY.md 存在且内容完整
- dispatch/PROTOCOL.md 已更新

## 回报

完成后在 `experiments/dispatch/status-F-001.md` 写入：
```markdown
---
card-id: F-001
status: completed
timestamp: 2026-08-07
summary: MASTER_PLAN 更新 + 裁决综合摘要
---

交付物：
- MASTER_PLAN.md §14 已添加
- PLWM_MATH_SUMMARY.md 已创建
- dispatch/PROTOCOL.md 已更新
```
