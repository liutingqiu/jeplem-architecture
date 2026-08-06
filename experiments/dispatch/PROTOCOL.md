# 会话协调协议

## 架构

```
总控（本会话）
  │
  ├── 派发 → experiments/dispatch/card-*.md
  │
  ├── 卡A会话 ← 读取 card-A-*.md → 执行 → 写 status-A.md
  │
  └── 卡B会话 ← 读取 card-B-*.md → 执行 → 写 status-B.md
```

## 文件约定

| 文件 | 用途 |
|---|---|
| `experiments/dispatch/card-{A,B}-NNN.md` | 总控派发的任务卡 |
| `experiments/dispatch/status-{A,B}.md` | 会话回报状态 |
| `PLWM_A_RESEARCH_REPORT.md` | 卡 A 正式裁决（随迭代更新） |
| `PLWM_MATH_B_REPORT.md` | 卡 B 正式裁决（随迭代更新） |

## 协议

1. 总控写任务卡 → 你（A/B 会话）读任务卡 → 执行 → 回报 status
2. 总控读取 status 后决定下一步任务卡或工程决策
3. 任务卡有 `depends-on` 字段，依赖未满足时不应开始执行
4. 所有报告遵循 `PLWM_MATH_EXPLORATION.md` §结论模板格式

## 当前任务卡

- [A-001](card-A-001.md) ✅ **已完成** — P1c 形式化 + 最小反例
- [B-001](card-B-001.md) ✅ **已完成** — 非高斯噪声 + 索引紧性 + A∩B 联合分析
- [C-001](card-C-001.md) ✅ **已完成** — 矩阵半群表示 + 非交换性 + 类型系统
- [D-001](card-D-001.md) ✅ **已完成** — U₂ 贝叶斯合并推荐 + 冲突检测复杂度
