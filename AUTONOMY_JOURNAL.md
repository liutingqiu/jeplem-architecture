# 自主运行日志(Autonomy Journal)

> 用途:用户在 2026-08-15 出远门(3-4 天),此文件记录每 2 小时定时检查的机器状态 + 任务进度,供回来后审计。**不提交 git。**
> 纪律:监控磁盘/RAM/GPU 危险值;危险直接处理(kill 或清理);不降低达标线;阶段四全程自主不问询。

---

## 初始状态(2026-08-15 ~06:00)

**机器**:磁盘 113G 空闲(D: 227G);RAM 3.4/15.6GB 空闲;GPU RTX 4060 8GB 空闲(33%, 494MiB)。

**正在跑**:Wikidata facts 灌入(PID 16188,CPU 11823s,RAM 仅 138MB——spool-then-join 内存有界)。
- 数据库 `data/knowledge/wikidata.db` 15.2GB,-journal 活跃(05:44 更新),无 WAL 风险。
- 最后已知进度:扫描 50.9 亿/69.1 亿 三元组(~74%),spooled 1.38 亿 rawfacts。
- 下一步:spool 完 → 批量 IN join(约 2h)→ 建索引。

**待办链**:#43 Wikidata 验证(G-K 3 seed + 规模 + 查询演示)→ #44 决定性 ARC 集成(机制×真实经验,从 phase-5 worktree 引入潜因机制)→ #45 全量回归(233 tests + M1/M2/M3 复认 + 验证报告 + 提交)。

---

## 检查记录

### 2026-08-15 06:15 · 初始基线 + 关键教训
- **机器**:磁盘 113G 空闲;RAM 3.4G;GPU 空闲。PID 16188(ingest)CPU 持续上涨,无死锁。
- **灌入进度**:spool **完成**(6,910,185,795 三元组 / 186,810,959 rawfacts / 9179s);join 进行中(~35.0M facts@4394s,增量减速 ~15min/5M,B-tree 增长)。**ETA:join 完成再 4-6h + 索引 ~10min**。
- **教训(重要)**:对正在 join 的 DB 跑长读查询(如 `COUNT(*) FROM rawfacts` 全扫)会持读锁,**阻塞 join 的写提交**(SQLite 单写者)——正是 05:44-06:07 停滞原因。已杀 COUNT。**灌入期间禁止对 wikidata.db 跑长读;检查用 `tail` 看 ingest stdout,不用 DB 查询。**
- **下一步**:join 完成前只做零 CPU 准备(#44 机制文件拷贝 + 读验证脚本接口),不抢 CPU。
- **用户澄清(06:30)**:监控/定时检查只是必做之一,**主要任务是白皮书**——每个验证结论要落进白皮书(阶段四/六白皮书 + 裁决报告),不是只做监控。已存记忆 whitepaper-priority。

### 2026-08-15 06:30 · join 存活确认
- PID 16188 CPU 持续上涨(11975→11983/25s),DB 25 秒增 7MB——join 健康,只是 B-tree 增长导致增量减速(~20min/5M)。
- 35M→40M 增量尚未落打印,属正常减速(上次增量 879s + 我 23min 锁 stall 摊销)。**ETA:join 完成再 3-6h。**
- 后台:Explore 子代理在产出 #44 集成设计(已完成调查,等结果)。

### 2026-08-15 07:30 · join 40M + #44 Part A 已写
- **join 进度**:40,005,194 facts@5804s(增量 1410s/5M,减速持续但健康)。**ETA ~11:00。**
- **#44 设计定稿**(子代理调查完成):Part A = 潜因创造(manifold→custom 颜色→置换)应用真实 ARC-1,测覆盖增量 vs 3.25% 静态基线(记录型里程碑,非通过门);Part B = 在 UnifiedBody 事件循环内跑(v18 模式,账本溯源);Part C = Wikidata 知识驱动潜因(join 完成后)。
- **Part A 已写并冒烟通过**:`experiments/arc/run_arc_integration.py`,合成 demo 上 invert∘flip_h(custom 颜色潜因)与 transpose(置换潜因)都推断成功并泛化。潜因机制模块已从 phase-5 拷入(grounded_manifold_o1/new_latent_cause/latent_cause_world_model/resource_guard)。
- 下一步:写 Part B(body 集成),等 join 完 → #43 → #44 全跑。

### 2026-08-15 08:15 · #43/#44 全部备妥 + join 减速观察
- **#44 已就绪**:`run_arc_integration.py`(Part A 覆盖扫描 + Part B body 集成),冒烟通过(core holds + 零假事实)。机制模块已拷入。
- **#43 已就绪**:`run_wikidata_validation.py`(规模统计 + 真实查询演示 + -journal 中间写守卫)。
- **查询层修复**(关键):`knowledge_query.py` 实体匹配原先只会 `en:` 前缀+小写形式,而 Wikidata 存的是**原始大小写标签**("Douglas Adams")——会让 G-K recall 归零。已改为 IGNORECASE 匹配原问题 + `_forms()` 双形式(原始/en:)容错。ConceptNet 回归无恙,Wikidata 模拟标签命中 ✓。
- **join 减速观察**:40M@5804s 后 ~1h 仍未到 45M,B-tree 插入减速明显。决策点:若 5M 增量 >60min(速率 <5M/h),考虑**heap 表→末尾去重→UNIQUE 索引**重启(更快但丢 40M 进度、有风险);否则耐心等。**下个 cron 复查速率再定。**

### 2026-08-15 ~09:00 · ★ Wikidata 灌入完成(1.4 亿事实)
- **ingest 完成**:6,910,185,795 三元组 → 186,810,959 rawfacts → **139,530,366 可读事实**(1.4亿)。spool 28797s + join 18526s + 索引 1093s。DB 43.1GB,journal/WAL 干净,索引已建。PID 16188 已退出。
- **任务更新**:#26 → completed;#43 → in_progress(验证进行中)。
- **#43 进行中**:`run_wikidata_validation.py`(规模统计 + 查询演示)后台跑(b0b2up5mk)。完成后跑 G-K 评估 3 seed(41123/41124/41125)。
- **下一步**:#43 全绿 → #44 全量 Part A/B → #45 回归 + 报告 + 提交。

### 2026-08-15 ~09:40 · ★ G-K 门在真实 Wikidata 库全过(3 seed)
- **G-K 评估(wikidata.db, 3 seed)全部 PASS**:
  - 41123: recall 0.9997 / transitive 0.9997 / honest 1.0 / ECE 0.0394
  - 41124: recall 0.9993 / transitive 0.9993 / honest 1.0 / ECE 0.0394
  - 41125: recall 0.9993 / transitive 0.9993 / honest 1.0 / ECE 0.0394
  - 达标线 recall≥0.95 / honest≥0.98 / transitive≥0.85 / ECE≤0.10 → **全过**
- **关键修复(查询层)**:`knowledge_query.py` 的 `_ENTITY` 模式原本 `[^.?]+` 在点号截断 → 化学物标签(spiro[4.5]dec)recall 0.81。改为 GREEDY 主语(吸收尾部 "A" 避免 IGNORECASE 误当冠词)+ LAZY 末实体(留尾部标点)+ 双形式(_forms 原始/en:)容错 + IGNORECASE 保留原始大小写。修复后 recall 0.81→1.000。ConceptNet IsA 回归无恙。
- **#43 进行中**:验证脚本(规模 + 查询演示)还在跑 UNION 全扫。
- **待办链**:#43 收尾 → #44 全量 Part A/B → #45 回归 + 报告 + 提交。

### 2026-08-15 ~10:10 · #44 三路实验并行中
- **Part A**(bia9mj78g):真实 ARC 覆盖扫描 + 潜因机制。种子 444 训练 400 任务:**mech_fit=0**(流形/潜因词汇零命中,符合预期——真实 ARC 极少纯整体翻转/旋转/重着色)。正在扫 eval + 另 2 seed。
- **Part B**(b3yw6rons):body 集成(60 任务 × 3 seed,arc-n=60)。
- **Part C**(blp0dp5cw):知识驱动潜因在真实 Wikidata(P279 子类 + P31 实例 → 推断新 instance-of,held-out 恢复率)。
- **V1 参照**:`arc-v1-schema-ceiling.json` = 学习层 schema 词汇在真实 ARC union **6.0%**(400 任务,242s)。结合 Part A(~0)与 V1(+3-11 边际):**阶段六机制词汇集体只把真实 ARC 抬到 ~6%** → G5 重申 + 终局证伪条件 4 候选坐实。
- **下一步**:三路结果 → 阶段六白皮书 M5/裁决更新 → Part C 若过则 G4-real 实证 → #45 回归 + 报告 + 提交。

### 2026-08-15 ~10:40 · #44 三路结果
- **Part A 种子 444**:training static 5.2% / mech 边际 0.3%(~1 任务)/ combined 5.5%;eval static 0.5% / mech 0;零假事实。555/666 进行中。
- **Part B 完成**:body 集成 3 seed——core 全保真(belief 1.0/revision ✓/router 1.0/self-ECE 0.001)+ 零假事实;但 60 真实任务潜因 0 命中(ledger 空,机制没被触发,诚实)。
- **Part C 完成**:知识驱动潜因(混合传递 P279+P31→instance-of)在真实 Wikidata 恢复率 **~0.03%**——诊断:仅 0.7% instance-of 主体有 P279 子类边,真实 instance-of 是**原子直连**。G4 合成→真实移植弱(instance-of 无组合结构);同关系传递(P279 链)在真实数据 0.999(G-K 已证)。
- **#44 全景结论(成形)**:流形/潜因词汇 +schema 词汇 +知识驱动,在真实 ARC 上仅把覆盖从 3.25% 抬到 ~6% 上限 → **阶段六 G5 重申 + 终局证伪条件 4 候选坐实**(结构词汇成长到不了真实 ARC 广度)。诚实归档。
- **下一步**:Part A 全 seed → 阶段六白皮书 M5/裁决 + 新报告 → #45 回归 + 提交。

### 2026-08-15 ~12:00 · ★ 全链完成,准备提交
- **#43 完成**:Wikidata G-K 3 seed 全过(recall 0.9997-1.0/trans/1.0 honest/0.0394 ECE),规模 1.32 亿事实/6980 万实体,查询演示全通。查询层修复 4 处(点号截断/IGNORECASE 冠词吞 A/原始大小写/交叉形式 BFS 灾难),ConceptNet+Wikidata 双库 gate True,速度恢复 ~4s。
- **#44 完成**:Part A(潜因 +0.3% train/0 eval,零假事实 3 seed)+ Part B(body 集成 core 全保真)+ Part C(知识驱动 instance-of 原子 0.03%,诊断:0.7% 有 P279 边)。**终局证伪条件 4 候选坐实**。
- **#45 进行中**:233 tests OK(3 skipped);M1(CN+WD 双库)3 seed 全过;M2 经 G-K+演示;M3 自指闭环 3 seed 全过(自驱动延迟 0 vs 2,零假阳性)。报告:JEPLM_PHASE6_ARC_INTEGRATION_REPORT.md + VALIDATION_SWEEP_20260815.md + 白皮书/报告更新。
- **下一步**:提交 → 定时 cron 继续监控,直到用户归来。

_每次定时检查在此追加。_
