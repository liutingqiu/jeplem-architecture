# ARC 基准评估报告（ARC-1 + ARC-2，2026-08-12）

> 性质：身体 15/15 收官后，P6 档3 组合泛化终极判据的 ARC 摸底（ARC-1 全量 + ARC-2 公开分割）。
> 方法：数据 = ARC-AGI-1 官方（HF 缓存，训练 400 + 评估 400）+ ARC-AGI-2（HF 社区镜像 `arc-agi-community/arc-agi-2`，train 1000 + test 120），已提取 `data/arc/`。机制 = 显式符号原语库 + 暴力组合搜索（P6 混合组合器哲学：显式骨架 + 参数化接口）。
> 纪律：诚实覆盖报告（信息性，非 pass/fail）+ 受控协议（验收判据）。

---

## 一、覆盖扫描（原语库在真实 ARC 上的表达力）

**depth≤2 原语组合搜索**（recolor/swap/flip/rotate/transpose/scale/crop/keep + 两层组合）：

| 基准 | 分割 | 任务数 | train 拟合 | test 命中 | 命中率 | 记忆回放基线 |
|---|---|---|---|---|---|---|
| ARC-1 | training | 400 | 13 | 13 | **3.25%** | 0.0 |
| ARC-1 | evaluation | 400 | 1 | 1 | **0.25%** | 0.0 |
| ARC-2 | train | 1000 | 14 | 14 | **1.4%** | 0.0 |
| ARC-2 | test | 120 | 0 | 0 | **0.0%** | 0.0 |

- 全部命中都是简单几何/颜色变换（transpose/rotate/flip/recolor/scale/swap + 少数 depth-2 组合）。
- train 拟合 = test 命中（无"拟合但测试错"）——命中是真实规则泛化而非过拟合。
- **诚实结论**：depth≤2 原语库只能表达 ~1-3% 的 ARC 任务，ARC-2 比 ARC-1 更难（简单任务更少）。ARC 高度异构（物体识别/计数/条件逻辑/多步推理），远超两层原语组合。**瓶颈在原语库表达力，不在组合机制**。

## 二、P6 档3 受控协议（组合泛化验收判据）

**判据**：训练规则子集 + 测试**未见组合**，成功率 > 2x 记忆回放基线且 > 0.5。

**5 个组合族**（recolor→scale / scale→recolor / flip→recolor / rotate→recolor / crop→scale），参数组合按值 60/40 切分 seen/unseen。此协议不依赖具体基准数据（合成规则族），ARC-1/ARC-2 共用同一机制验证：

| 族 | seen 成功率 | unseen 成功率 | 回放 |
|---|---|---|---|
| recolor_scale | 1.0 | 1.0 | 0 |
| scale_recolor | 1.0 | 1.0 | 0 |
| flip_recolor | 1.0 | 1.0 | 0 |
| rotate_recolor | 1.0 | 1.0 | 0 |
| crop_scale | 0.667 | 1.0 | 0 |
| **合计** | | **1.0** | **0.0** |

**accepted: true** —— unseen 成功率 1.0 > 2×0.0 且 > 0.5 ✓。

**解读**：显式符号组合搜索在未见参数组合上 100% 泛化，回放基线 0（非记忆）——**组合泛化的完整性来自显式符号机制**（与第一阶段 P6 混合组合器、第二阶段 P6 档1/2 的结论同构）。搜索式归纳从 demo 直接推断规则、天然组合泛化（无记忆）。

## 三、诚实边界

1. **P6 档3 通过 ≠ 通过 ARC 基准**：受控协议在"原语库可表达的规则族"上验证组合机制；真实 ARC-1 覆盖 ~3%、ARC-2 覆盖 ~1%。ARC 作为**覆盖面**验收远未达成——需大幅扩展原语库（物体提取/连通域/计数/条件）才有意义。
2. **搜索式归纳的普适性**：本 harness 的归纳器是穷举搜索（显式符号路径）；若未来换成"学习型原子"（P6 混合组合器的可学习叶子），才涉及真正的可学习组合。当前验证的是**显式符号路径**。
3. **P6 档2 深度嵌套仍未达标**（0.583→0.8，PARTS_AUDIT）：与 ARC 独立，仍开放。
4. **ARC-2 公开分割 vs 官方验收**：本报告用社区镜像的 train/test 分割；官方 ARC-AGI-2 基准（arcprize 主持）的正式排行榜评估需按官方提交规则，不在此列。

## 四、产物

- `data/arc/`：`arc1_training.json`（400）、`arc1_evaluation.json`（400）、`arc2_train.json`（1000）、`arc2_test.json`（120）
- `experiments/arc/`：`grid.py`、`primitives.py`、`induction.py`、`extract_arc1.py`、`extract_arc2.py`、`run_coverage.py`、`run_protocol_p6d3.py`
- `artifacts/arc1-coverage.json`、`arc2-coverage.json`、`arc1-p6d3-controlled.json`

## 五、下一步

- **扩展原语库**：物体提取/连通域/计数/条件原语 → 重跑覆盖扫描（把 ARC-1 覆盖从 3% 抬到有意义水平）
- **可学习原子**：把搜索式归纳换/并成 P6 混合组合器的学习型原子映射，检验"学习型组合"路径
- **P6 档2 深度嵌套**（0.583→0.8）：与 ARC 并行的开放项
