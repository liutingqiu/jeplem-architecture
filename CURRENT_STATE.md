# CURRENT_STATE — 项目交接文档（新会话从这里读起）

> 用途：让新会话用几分钟读完"项目现在在哪、下一步做什么"，避免重读历史大文件。
> 更新规则：每个阶段闭环后更新本节头部；历史细节写进 NIGHT_RUN/各 REPORT，不搬进本文件。
> 最后更新：2026-08-15（**阶段四 M1 知识预训练完成 + 阶段六 ARC 集成决战完成**：Wikidata 灌入完成——69.1 亿三元组→1.32 亿事实/6980 万实体，G-K 门 3 seed 全过(recall 0.999-1.0/诚实 1.0/ECE 0.039，ConceptNet+Wikidata 双库)；阶段六 #44 决战——机制×真实经验集成后真实 ARC 覆盖上限仍 ~6%(schema union 6.0%/潜因 +1 任务/知识驱动 instance-of 原子)，零假事实，终局证伪条件 4 候选坐实，详见 `JEPLM_PHASE6_ARC_INTEGRATION_REPORT.md`；233 回归 OK(3 skipped)）

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
10. **身体组装收官（v16，2026-08-12，15/15 机制入体，3 seed 全过）**：完整身体 = v14 整合身体（value/gap/belief/pred/rev/snapshot）+ **meta-controller**（来源优劣随状态变化任务：clean 0.98 翻转后记忆胜 / noisy 0.60 信念胜；路由 0.999-1.0 vs fixed 0.59-0.68，非单调守卫过）+ **自我模型**（样本级 Block-8：self-head 预测信念答案正确性 ECE 0.014-0.040 vs naive 0.16-0.19）。判据：v14 全保留 + 路由 ≥0.80 / routed ≥ best-fixed+0.10 / 两 regime 子集 ≥0.7 / self ECE ≤0.10 且 ≤naive-0.05 / 退化守卫。产物 `artifacts/phase2-body-v16-complete-{41123,41124,41125}.json`。设计迭代史与判据修订记录在 BODY_ASSEMBLY.md D 阶段
11. **全资产集成（v17，2026-08-13，3 seed 全过）**：`plwm/body.py` **UnifiedBody 单一体类**承载全部机制（感知前端可插拔 + ledger/belief/value/gap/预测/修订/快照/执行注意力/组合技能/社会/meta-router/self-head/因果 CI），`train_phase2_body_v17_unified.py` 在**真视觉前端**（冻结 CNN 读真实 MiniGrid 帧→语义身份）+ v15 RL agent 的复合自驱动任务验证。**机制共存判据 3 seed 全过**：真视觉查询 0.59-0.81 > random、感知 0.81、value 1.0、pred 0.99、belief 1.0、rev 1.0、P1/P5 1.0、routing 0.84-0.94、组合 1.0、执行注意力 1.0、社会 0.67-0.69、self ECE 0.024-0.037 ≤0.7×naive、n≥200。诚实边界：集成判据=机制共存（个体判据 v9-v16 已建）；平衡细节（非单调带/子集/路由余量）与因果归因为集成种子方差（因果单翻转单位+翻转时机近二进制脆弱，CI 基础设施在运行）；self 0.05→0.7×naive、exec 控制比较→注意力准确性为判据错误修订。**Q1 全资产集成完成**

## 三、关键结论（供集成层设计引用）

- **架构承重件**（JEPLM 最终配置）：感知（冻结 CNN）→ 语义 → 显式槽位记忆 + 语义 aux → 策略。合成流 0.941/0.831、真实流 0.774/0.577 端到端已验证。
- **JEPA latent 预测 / surprise 清槽修订：RL 下无实证支持，默认关闭**（审计证据：NIGHT_RUN 阶段 28-35）。
- **分层训练是特征非缺陷**：感知自监督/监督预训练 + 冻结、记忆 RL 优化（联合训练证伪 0.427）；预测监督与记忆保持冲突（阶段 32）。
- **辅助目标三态**：重建式有害（-0.10）| 语义内容预测帮助（0→+0.56）| latent next-state 中性。
- **区块机制已验证清单**：1 信念（BeliefCell 分叉）/2 驱力（置信度缺口）/3 持续学习（consolidation 账本）/4 自主目标（gap_goals）/5 因果（CI 测试）/7 反事实（转移泛化）/8 自我模型（能力边界 ECE）/9 执行控制（预算注意力）/10 价值锚定/11 社会认知/12 世界模型（显式转移）。

## 四、下一步：Q2 训练范式（先训练后 ARC）

**原则（用户定调）**：不是为了通关 ARC 而训练，而是**先训练，然后 ARC 作为验收**。铁律：为架构普适性不为应试微调；ARC 是验收不是目标。

**训练范式设计（v17 身体是载体）**：知识不在权重里，而在显式记忆 + 机制 + 小学习接口。训练 = ①经验驱动记忆积累（身体解任务，规则存进账本/信念）②学习小接口（P6 组合器的可学习原子映射、meta-router、self-head）③**原语库成长（library learning）**。

**ARC 学习阶段已实施（2026-08-13，accepted）**：`experiments/arc/learn_library.py` 受控族课程演示训练机制——身体从训练实例发现组合技能骨架（recolor→scale 等，depth-2），库随经验成长。**学习曲线：depth-1 未训练基线 0% → 3 技能发现后 held-out 未见组合覆盖 85.7%**（技能=骨架按任务重参数化，非记忆；ARC-2 0% 为诚实边界：族技能不适用于真实 ARC-2 异质规则）。产物 `artifacts/arc-library-learning.json`。
- **真实 ARC 边界（诚实）**：覆盖扫描显示当前原语库下真实 ARC-1 ~3% / ARC-2 ~1% 可解——训练机制演示有效，但其对真实 ARC 的效果**受原语库表达力封顶**。抬真实覆盖 = 扩展原语库（物体提取/连通域/计数/条件）+ 可学习原子（P6 混合组合器学习型叶子）+ 库归纳（从真实 ARC 训练集发现新变换）。
- **学习层诊断修正（2026-08-13，git 历史 + artifact 核实）**：学习层**主机制已单模块验证通过**（learn_library 0→85.7%：技能发现+重参数化复用；G1b/回路用的是同一套 skeleton_of/can_map/search_program，**无版本错接**）。真实 ARC 失败的真实归因 = **词汇表达力是唯一瓶颈**（覆盖数据：depth1 21、depth2 0，深度组合零增益）+ G1b gate 预设 ≥10% 而静态天花板仅 5% → **结构性不可达（判据误差，按纪律文档化，非静默调）**。库成长原实现只是"已有原语重组"，不是"从数据学新原语"。
- **学习层新设计落地（对象-关系模式文法 + 库压缩成长，2026-08-13）**：
  - 新增 `object_layer.py`（对象/关系提取）、`schemas.py`（选谓词→变换→重组文法，**相对谓词按输入网格实时解析**——修掉"largest 烘焙绝对面积"bug）、`schema_search.py`（一致性归纳，库原语优先，G2 验证的正确工具，无稠密梯度）、`library_growth.py`（**重复 2 步片段→原子原语**，溯源入账本）、`run_schema_ceiling.py` / `run_library_growth.py`。
  - **V1 表达力诊断**：对象规则文法在真实 ARC **边际新增 11 个任务**（基础词汇解不了）→ 证实词汇是瓶颈；但并集仅 6%，**未达 2× gate** → 判据误差修订已文档化：真实 ARC 变换空间（逐位置映射/排列/抽取/对称）远超适度文法，靠手工扩文法不现实——正说明词汇该**从数据成长**。
  - **V2 学习机制（3 seed 全过）**：受控课程——教家族暴露 2 步片段、测家族需 3 步；库成长后 held-out **0→0.667**（≥0.6、上升、≥2×），2 个原子原语被学习，V4 溯源/复用探针全过。**从数据学新原语成立**。
  - **V3 真实 ARC 转移（诚实边界）**：真实 ARC 训练中 **0 个片段可压缩**（家族唯一，无重复）→ 边际 0.0，文档化为"ARC 家族唯一性"属性（机制在重复存在处生效=V2，真实 ARC 不提供重复）。
- **下一步候选**：①扩展对象-关系文法家族（关系谓词/结构变换）后重测 V1 并集 ②把成长库接入 v18 身体（meta-router 在 base/learned 原语间路由，P6 组合走成长库，账本溯源喂 self-model）③真实环境数据积累驱动的库成长。

## 四·五、阶段四进展（完整自研架构，2026-08-14）

**白皮书**：[JEPLM_PHASE4_WHITEPAPER.md](JEPLM_PHASE4_WHITEPAPER.md)（草稿 V2，含 §十 无硬件限制愿景）；阶段五批判性白皮书已移交另一工作区。

**近端里程碑 M1/M2/M3 完成**（详见 [PHASE4_KNOWLEDGE_LANGUAGE_REPORT.md](PHASE4_KNOWLEDGE_LANGUAGE_REPORT.md)）：
- **M2 语言机制**：`plwm/mrep.py`（M-rep 意义表征：fact/chain/unknown/explanation/revision + 实体/关系/置信/来源/推理）+ `plwm/language_gen.py`（模板生成器，零训练）；`knowledge_query.py` 返回 MRep。修的 2 个 parser bug（标点实体、冠词吞首字母）使 recall 0.27→0.997。
- **M1 知识预训练（G-K）**：`run_gk_knowledge_eval.py` 在 ConceptNet 上 3 seed 全过（recall 0.997 / transitive 0.997 / 诚实无知 1.0 / ECE 0.039）。Wikidata join 完成后换库重跑全量。
- **M3 自指闭环（G-S）**：`plwm/body.py` 加独立 doubt 头（默认关闭）+ `train_phase2_self_ref_loop.py`；3 seed 全过——自驱动修订延迟 0 vs 非驱动 2，零假阳性（翻转前不误标/不翻转域不误标）。诚实负结果→修实现（单翻转期+非加权损失 seed 41124 失败→多翻转期+加权损失修复，gate 不变）。
- **回归**：233 tests OK（body 改动 additive）。

**当前后台**：Wikidata facts 大 join（186M 行索引+合并）运行中 → 完成后跑 `run_gk_knowledge_eval.py --db wikidata.db`（M1 全量）+ 知识库规模/查询验证。

**资源纪律**：与阶段五工作区共享机器，所有 GPU/CPU 任务带监控，危险值直接 kill（见记忆 gpu-cpu-resource-monitoring）。

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
