---
card-id: G-001
status: active
timestamp: 2026-08-08
depends-on: []
priority: high
---

# 任务卡 G-001: short_delay_2id 诊断实验

## 背景

addressed_balanced_2id 六步延迟检索任务失败，根因已定位：
- 身份 tile（z=-7）和查询 tile（z=-2）**永不在同一帧可见**
- 这是一个真正的跨帧延迟关联，终端 reward 太稀疏，credit assignment 失效
- 所有控制器（PLWM/GRU/Transformer/无记忆）都卡在 50%

## 目标

运行 `experiments/train_luanti_short_delay_2id.py` 诊断任务：
- 3步 episode，身份在 step 0 可见，查询在 step 2 可见
- 加入辅助检索损失（α=0.1）监督
- 验证 PLWM 机制在**可学习任务**上是否有效

## 验收标准（contract-short-delay-2id.json）

```
plwm held-out >= 0.75
plwm vs no_memory gap >= 0.15
plwm vs gru gap >= 0.10
plwm vs transformer gap >= 0.10
no_memory held-out <= 0.60
```

## 执行方式

```bash
cd "D:\Documents\探索新架构"
python experiments/train_luanti_short_delay_2id.py --workers 4
```

## 输出

- 报告写入 `artifacts/luanti-short-delay-2id-report.json`
- 如果通过验收，更新 `MASTER_PLAN.md` 当前承诺部分
- 如果失败，分析原因并记录到 `experiments/dispatch/status-G-001.md`

## 约束

- 使用 CUDA（RTX 4060）
- 使用 4 workers 并行（加速收集）
- action_seed 固定为 9042337（与之前实验一致）
- 不要修改 contract 中的任何参数
