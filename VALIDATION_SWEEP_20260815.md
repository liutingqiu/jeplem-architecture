# 验证扫尾报告——待办全验证(VALIDATION SWEEP,2026-08-15)

> 用户指令:"等后台结束后把所有待办,没有验证的,全都验证一遍"(2026-08-14/15,自主执行)。
> 纪律:预写达标线、3 seed、诚实负结果、零假事实、判据修订文档化。

---

## 一、待办 → 验证结果一览

| # | 待办 | 验证 | 结果 |
|---|---|---|---|
| #26 | Wikidata facts 遍(98.8% 部分) | 69.1 亿三元组摄入 | ✅ **完成**:spool 1.87 亿 rawfacts → join **1.32 亿事实**(去重 740 万)→ 43GB 库 + 索引。**教训**:join 期间禁止对库跑长读(读锁阻塞写提交,2 次 stall 已记录)。 |
| #27 | 验证 Wikidata 知识库 + 查询演示 | `run_wikidata_validation.py` | ✅ **完成**:规模(1.32 亿事实/6980 万实体/7683 万标签)+ 真实查询演示(置信 0.98 + 溯源)。 |
| #43 | 验证 Wikidata 知识库(G-K + 规模 + 查询) | `run_gk_knowledge_eval.py --db wikidata.db --relmap wikidata` 3 seed | ✅ **3 seed 全过**:recall 0.9997-1.0 / transitive 0.9997-1.0 / honest 1.0 / ECE 0.0394(达标 0.95/0.85/0.98/0.10)。**查询层修复 4 处**见 PHASE4_KNOWLEDGE_LANGUAGE_REPORT。 |
| #44 | 决定性 ARC 集成测试(机制×真实经验) | `run_arc_integration.py` Part A/B + `run_knowledge_latent_cause.py` Part C | ✅ **完成(记录型)**:Part A 潜因机制边际 +0.3% train(1 任务)/+0 eval,零假事实 3 seed;Part B body 集成 core 全保真;Part C 知识驱动 instance-of 原子(0.03% 恢复)。**终局证伪条件 4 候选坐实**。详见 `JEPLM_PHASE6_ARC_INTEGRATION_REPORT.md`。 |
| #45 | 全量回归 + 验证报告 + 提交 | `python -m unittest discover -s tests` + M1/M2/M3 复认 | ✅ **本报告 + 提交**:233 tests OK(3 skipped)。M1 G-K 双库(CN+WD)3 seed 全过;M2 语言机制(经 G-K + 演示);M3 自指闭环 3 seed 全过(自驱动延迟 0 vs 非驱动 2,零假阳性)。 |

## 二、回归细项

- **测试套件**:233 tests OK,3 skipped(环境相关:luanti/minigrid 等模拟器),2.9s。
- **M1 复认**:ConceptNet G-K 3 seed gate True(recall 0.9993-1.0/trans 0.9993-0.9996/honest 1.0/ECE 0.039);Wikidata G-K 3 seed gate True(recall 0.9997-1.0)。
- **M2 复认**:M-rep + 模板渲染经 G-K 评估与查询演示全链路(未知→"I don't know"、fact→Yes+置信、explanation→枚举)。
- **M3 复认**:`train_phase2_self_ref_loop.py` 3 seed gate True,自驱动修订延迟 0(非驱动 2),零假阳性。

## 三、本日代码改动(已自洽,随提交入库)

1. `experiments/arc/ingest_wikidata.py`:journal_mode=DELETE(防 WAL 爆盘)+ batch-IN join(避嵌套循环病理)+ `_unescape` + qid 尖括号修复。
2. `experiments/arc/join_only.py`(新):独立 join 重跑脚本。
3. `experiments/arc/knowledge_query.py`:实体格式容错(GREEDY 主语/LAZY 末实体/IGNORECASE 保留大小写/对齐形式组合),修复 Wikidata recall 0.81→1.0 与 ConceptNet BFS 灾难(>2min→4s)。
4. `experiments/arc/run_arc_integration.py`(新):#44 Part A(覆盖扫描+潜因管线)/Part B(body 集成)。
5. `experiments/arc/run_knowledge_latent_cause.py`(新):#44 Part C(知识驱动潜因,真实 Wikidata)。
6. `experiments/arc/run_wikidata_validation.py`(新):#43 规模+演示,带 -journal 中间写守卫。
7. 机制模块拷入:grounded_manifold_o1 / new_latent_cause / latent_cause_world_model / resource_guard(phase-5 潜因机制,供 #44 集成)。

## 四、文档更新(随提交入库)

- `JEPLM_PHASE4_WHITEPAPER.md`:M1 完成 + 九节步骤 1 勾除。
- `PHASE4_KNOWLEDGE_LANGUAGE_REPORT.md`:M1 Wikidata 全量验证 + parser 修复记录 + 待办勾除。
- `JEPLM_PHASE6_WHITEPAPER.md`:M5(真实 ARC 记录)完成。
- `JEPLM_PHASE6_ARC_INTEGRATION_REPORT.md`(新):#44 总裁决。
- `CURRENT_STATE.md`:头部交接更新。

## 五、诚实边界

- **#44 是记录型裁决**(M5),不是通过门:真实 ARC 覆盖 ~6% 上限是阶段六 G5 与终局证伪条件 4 的诚实证据,不是失败——它精确化了墙("词汇广度与经验规模",非机制缺陷)。
- **知识轴刚起步**:Wikidata 知识引擎(G-K)全过但"知识 × 网格世界模型"集成未做——下一步杠杆在真实环境 + LLM 皮层生成经验 + 更大规模网格词汇学习。
- 判据未降:所有达标线原样保留;查询层修复是让引擎在真实标签上工作(非放宽判据)。
