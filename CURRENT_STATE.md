# CURRENT_STATE — 项目交接文档（新会话从这里读起）

> 用途：让新会话用几分钟读完"项目现在在哪、下一步做什么"，避免重读历史大文件。
> 更新规则：每个阶段闭环后更新本节头部；历史细节写进 NIGHT_RUN/各 REPORT，不搬进本文件。
> 最后更新：2026-08-11（第二阶段主体性判据全绿：P1/P2/P3/P4/P5 🟢（P3 任务级自知 v2）+ P6 🟢（混合组合器）；M1/M2/M5/M6 里程碑 + M6c 随机压力流全过；下一步：完整主体整合——感知+记忆+决策+运行时）

---

## 一、项目北极星（一行）

验证"显式槽位记忆 + 语义感知层"的 JEPLM 认知架构：白皮书 12 个认知区块的最小可证伪实验已逐块验证（**11/12 ✅**），下一阶段把已验证的机制 + JEPLM 三承重柱**组装为完整自主认知系统**（系统集成层）。

## 二、当前主线（图纸验证层已收官，进入系统集成层）

**A. 图纸验证层（2026-08 至 08-11，已完成）**：
1. **JEPLM 三承重柱验证完**：JEPA 感知层（冻结 CNN 0.961，联合训练证伪）| 显式槽位记忆（内容寻址 > 压缩）| 语义 aux（单身份/单时标决定性）
2. **两根候选柱淘汰（负结果驱动）**：JEPA latent 预测（阶段 28-32 五轮中性 + 记忆冲突）、surprise 清槽修订（阶段 35 RL 有害）——均降级，默认关闭
3. **白皮书 11/12 区块 ✅**：1-5、7-12 验证通过；区块 6（组合泛化）toy 不可判，可判条件已标注（需 ARC-AGI-2 类外部基准）
4. **遗留项全清零**：RMT 流式崩盘、真实流回归、aux 三态矩阵、混淆观测 POMDP、查询提示、RMT+aux、双身份干扰 aux

**B. 系统集成层（进行中，2026-08-11 探路完成）**：
1. **待集成模块勘察完成**：骨架已组装（JEPLMAgent = 冻结 CNN → JEPLMMemory → 策略头）；11 个区块机制模块独立 toy 验证未接入；runtime/world_runtime 是已自组装子线（参考实现）
2. **集成接口文档落盘**（JEPLM_INTEGRATION_DESIGN.md）：需求驱动/最小可行原则 + 第①刀设计 + 已排除方向及证据
3. **第①刀 goal→query 调制：中性**（q 0.962→0.958，接口保留为 gap_goals 基础设施）
4. **第②刀 gate 学习门控写：RL 下有害**（q 0.786 vs fifo 0.962）——fifo 确认为实证正确默认，蓝图已修订
5. **元结论**：合成流已饱和，显式机制无增益——集成验证需**复合任务**（下一步）
6. **复合任务 EvidenceSeek 完整闭环（v1-v5 + 核心 bug 修复）**：诊断翻转（监督标签）→ 评估 bug 修复 → 分层负结果（边界洞见）→ gap_goals 机制决策（显式接线）→ 显式接线实证（explicit_key 0.589）→ **9.5 核心任务 bug（v1 起线索与值无关）修复**：线索-值相关性 0.655 ✓、显式策略 0.49→0.646、修复后 RL 300updates ~0.60 上升——修复前 0.565 为伪学习上限。**任务构造纪律：观测须验证与目标变量相关性**。修复后全量重跑为下一步
7. **任务 1：零件健康审计完成**（PARTS_AUDIT_REPORT.md，4 项补测 3 seed）：①显式记忆寻址 K=4 监督 jeplem 0.747 > gru 0.663（**承重件寻址健康确认**）②区块 4 子目标分解 mono 1.0 vs 0.63（✓ 补全）③区块 3 高维 consolidation 1.0 但 neural 0.99 不忘（无冲突任务局限，高维冲突性遗忘仍空白）④区块 6 v2 组合泛化负结果（组合模型 heldout 0.31，非自回归解码实现待改进）
8. **任务 2：第二阶段白皮书**（JEPLM_PHASE2_WHITEPAPER.md，造人计划）：P1-P6 判据（含新增 P6 组合泛化）+ 里程碑 M1-M6 + 验收协议
9. **第二阶段执行结果**（2026-08-11，FUNCTIONAL_SUBJECT_REPORT.md 验收）：**主体性判据全绿**——P1 连续性/P2 自我驱动/P4 自我修订/P5 自我一致性（M1/M2/M6a/M6b/M6c 随机压力流，3 seed 全过）/ P3 自知（任务级 v2：注册表预测每任务强弱 max_err≤0.023）/ P6 组合泛化（混合组合器：符号骨架+学习参数 = 100%+学习性）；M5 meta-controller ✅（内部状态路由）；样本级 P3 与学习型 P6 的机制边界已记录；**下一步：完整主体整合**（感知+记忆+决策+运行时）

## 三、关键结论（供集成层设计引用）

- **架构承重件**（JEPLM 最终配置）：感知（冻结 CNN）→ 语义 → 显式槽位记忆 + 语义 aux → 策略。合成流 0.941/0.831、真实流 0.774/0.577 端到端已验证。
- **JEPA latent 预测 / surprise 清槽修订：RL 下无实证支持，默认关闭**（审计证据：NIGHT_RUN 阶段 28-35）。
- **分层训练是特征非缺陷**：感知自监督/监督预训练 + 冻结、记忆 RL 优化（联合训练证伪 0.427）；预测监督与记忆保持冲突（阶段 32）。
- **辅助目标三态**：重建式有害（-0.10）| 语义内容预测帮助（0→+0.56）| latent next-state 中性。
- **区块机制已验证清单**：1 信念（BeliefCell 分叉）/2 驱力（置信度缺口）/3 持续学习（consolidation 账本）/4 自主目标（gap_goals）/5 因果（CI 测试）/7 反事实（转移泛化）/8 自我模型（能力边界 ECE）/9 执行控制（预算注意力）/10 价值锚定/11 社会认知/12 世界模型（显式转移）。

## 四、下一步：系统集成层（推荐规划路径）

**目标**：把 11 个已验证区块机制 + JEPLM 三承重柱组装为完整自主认知系统，用 ARC-AGI-2 类外部基准作为最终验收（区块 6 可判条件已标注）。

**已有集成资产**（plwm/ 下待勘察）：
- **骨架已组装**：`plwm/jeplem_agent.py`（JEPLMAgent：感知+记忆+策略）、`jeplem_memory.py`、`jeplem_predictor.py`（降级保留）
- **区块机制模块**（toy 级独立验证，未与骨架集成）：`belief.py`（区块 1/2）、`commitments.py`/`fact_ledger.py`/`continual.py`（区块 3）、`adaptive.py`/`multiscale.py`/`regime.py`（多尺度/自适应）、`latent_planner.py`（规划）、`world_model.py`（区块 7/12）、`events.py`/`runtime.py`（运行时）
- **建议规划步骤**：①勘察上述模块的接口与 toy 验证状态 → ②设计模块间接口（信念如何驱动记忆写入/修订、目标如何选择查询、执行控制如何分配注意力、驱力如何触发探索）→ ③选一个"多模块协同"的复合任务验证集成（非单 toy）→ ④ARC 摸底/验收
- **大方向变更才汇报**：集成层是否改变架构承重件（如信念组件接入记忆门控）属架构决策，按授权自主拍板；ARC 外部基准采购需用户定夺

## 五、关键文件索引（新会话按需读，不读历史档案）

| 文件 | 角色 | 何时读 |
|---|---|---|
| **本文件** | 交接指针 | 必读 |
| **JEPLM_INTEGRATION_DESIGN.md** | 集成层决策与验证记录（两刀结果 + EvidenceSeek 诊断 + 下一步候选） | 集成工作时 |
| **PARTS_AUDIT_REPORT.md** | 零件健康审计（全零件数据 + 4 项补测结果） | 修零件/集成时 |
| **JEPLM_PHASE2_WHITEPAPER.md** | 第二阶段白皮书草稿（造人计划：P1-P5 主体性判据 + 路线） | 定第二阶段方向时 |
| COGNITIVE_SYSTEM_WHITEPAPER.md (15KB) | 北极星/12 区块（11 ✅ + 区块 6 可判条件） | 定方向时 |
| JEPLM_IMPLEMENTATION.md | 蓝图（含 JEPA 审计章节、待集成项） | 集成设计时 |
| JEPLM_MEMORY_DESIGN.md / JEPLM_AUX_REPORT.md | 记忆设计决策/aux 矩阵 | 改记忆代码时 |
| NIGHT_RUN_20260809.md (32KB) | 阶段 1-37 实验日志 | 查具体证据时（有目录可跳） |
| **不要读**：MASTER_PLAN.md (107KB，Luanti/MiniGrid 历史档案)、PLWM_BLUEPRINT_V1/PLAIN、RESEARCH_SURVEY_202608 | 已完结历史 | 除非查旧结论 |

## 六、实验入口（常用命令）

```bash
# 合成流连续 RL（主战场）：--jeplem-only 只看 JEPLM
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json

# 混淆观测 POMDP（阶段 31）/ 查询提示（阶段 32）模式
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --confounded --clue-frames 4 --clue-reliability 0.65 --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json
python experiments/train_decide_stream_rl.py --jeplem-only --aux-memory --query-hint --query-hint-reliability 0.7 --updates 100 --experiment-seed 444 --artifact artifacts/xxx.json

# 区块实验（图纸验证层资产，集成时复用）
python experiments/train_perception_layer.py   # 感知层
python experiments/train_causal_structure_learning.py --scenario collider  # 区块 5
python experiments/train_self_model_v2.py      # 区块 8
python experiments/train_belief_bifurcation.py # 区块 1

# 测试
python -m unittest discover -s tests -v
```

> 纪律：3 seed（444/555/666 或 222/333/41123）、swap 探针、诚实负结果归档；集成层改动仍须带可证伪对照。
