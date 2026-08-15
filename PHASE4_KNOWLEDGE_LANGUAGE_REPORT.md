# PHASE4_KNOWLEDGE_LANGUAGE_REPORT — 阶段四近端里程碑 M1/M2/M3

> 2026-08-14 初稿;2026-08-15 **更新:Wikidata 灌入完成(1.32 亿事实),M1 全量验证通过**。阶段四(完整自研架构)近端三里程碑实现与验证记录。

## M2 语言机制(M-rep + 模板生成器)——完成

**新增资产**:
- `plwm/mrep.py`:M-rep 意义表征——回答类型 fact/chain/unknown/explanation/revision + 实体/关系/置信/来源/推理步骤,结构无损(`structure_lossless`)。
- `plwm/language_gen.py`:模板生成器 M-rep → 自然语言(零训练,保留置信+来源,可审计)。
- `experiments/arc/knowledge_query.py`:改为返回 MRep(问句→查询→结构化回答),模板渲染输出。

**修的 parser bug(2 个,影响 recall 从 0.27 → 0.997)**:
1. 实体字符类不含标点 → ConceptNet 含点/逗号的实体("0.22 inch calibre"、"1,4 dioxane")无法解析 → 放宽为 `[^.?]+` + norm 保留标点。
2. 可选冠词 `(?:an?|the)?\s*` 无空格约束 → 吞掉 a/an/the 开头实体的首字母("accepted"→"ccepted")→ 冠词必须带空格(`a `/`an `/`the `)。

## M1 知识预训练(G-K 判据)——ConceptNet 上 3 seed 全过

`experiments/arc/run_gk_knowledge_eval.py`:从知识库生成 QA 对(IsA 精确 + 2-3 跳传递 + 随机未知负样本),测召回/传递/诚实无知/置信 ECE(10 bins)。

| seed | recall | transitive | honest-unknown | ECE | gate |
|---|---|---|---|---|---|
| 41123 | 0.9970 | 0.9968 | 1.0 | 0.0388 | ✅ |
| 41124 | 0.9973 | 0.9979 | 1.0 | 0.0388 | ✅ |
| 41125 | 0.9963 | 0.9968 | 1.0 | 0.0388 | ✅ |

判据:recall≥0.95 / 传递≥0.85 / 诚实≥0.98 / ECE≤0.10。产物 `artifacts/phase4-gk-knowledge-eval-{seed}.json`。

### M1 Wikidata 全量验证(2026-08-15)——3 seed 全过 ✅

**知识库**:`wikidata.db` 43GB——69.1 亿三元组摄入 → **1.32 亿事实**(insert OR IGNORE 去重 740 万跨块重复)、**6980 万 distinct 实体**、7683 万标签。关系分布:P31 instance-of 7300 万 / P50 author 2300 万 / P106 occupation 960 万 / P17 country 960 万 / P131 located-in 750 万 / P279 subclass 380 万 / P361 part-of 300 万 / P527 has-part 170 万 / P123 publisher 52 万。

**G-K 门(wikidata.db,3 seed)**:

| seed | recall | transitive | honest-unknown | ECE | gate |
|---|---|---|---|---|---|
| 41123 | 1.0000 | 1.0000 | 1.0 | 0.0394 | ✅ |
| 41124 | 0.9997 | 0.9997 | 1.0 | 0.0394 | ✅ |
| 41125 | 0.9997 | 0.9997 | 1.0 | 0.0394 | ✅ |

**查询演示**:真实事实→自然问句→MRep 回答(置信 0.98 + 溯源),含点号化学标签正确解析(如 "is (+)-(7S,8S)-guaiacylglycerol a guaiacylglycerol → Yes")。产物 `artifacts/phase4-wikidata-validation.json`。

**本日 parser 修复(3 个,影响 Wikidata recall 从 0.81 → 1.0)**:
1. **实体点号截断**:`_ENTITY` 原为 `[^.?]+` → 含点号标签("spiro[4.5]dec")截断。改为 GREEDY 主语(`(?:[^?]+)`)+ LAZY 末实体(`(?:[^?]+?)`),靠结构锚定边界。
2. **IGNORECASE 冠词误吞**:主语尾部大写 "A" 被 `an?`(IGNORECASE)误当冠词 → GREEDY 主语取最后一个有效冠词位。
3. **原始大小写丢失**:整体 `.lower()` 使实体小写,匹配不到存库原大小写("Douglas Adams")→ 改 IGNORECASE 匹配原问题保留实体大小写。
4. **跨形式 BFS 灾难**:交叉形式组合触发无提前退出的全图 BFS(ConceptNet 超时 >2min)→ 改**对齐组合**((raw,raw)+(norm,norm)),恢复 ~4s,两库 gate 均过。

## M3 自指闭环(G-S)——3 seed 全过,零假阳性

**机制**:`plwm/body.py` 新增**独立 doubt 头**(5 维 state_features → 预测"当前信念答案可能是错的")+ `self_doubt()` 方法 + 修订触发接线(默认关闭,不破坏 v17)。doubt 头低置信 + 信念/账本分歧 + 信念高置信 → 立即 mark(不等 3 次连续矛盾)。

`experiments/train_phase2_self_ref_loop.py`:训练 doubt 头(多翻转期 300/450/600 + 加权损失,测试翻转 700 泛化),测翻转后修订延迟。

| seed | 非驱动延迟 | 自驱动延迟 | 假阳性(翻转前误标/不翻转域误标) | gate |
|---|---|---|---|---|
| 41123 | 2 | 0 | 无 | ✅ |
| 41124 | 2 | 0 | 无 | ✅ |
| 41125 | 2 | 0 | 无 | ✅ |

判据:自驱动 ≤ 非驱动 − 2 且无假阳性(3 seed)。产物 `artifacts/phase4-m3-self-ref-loop-{seed}.json`。

**诚实负结果 → 修实现(非降标)**:初版单翻转期 + 非加权损失,seed 41124 自驱动延迟 = 2(无提速)。诊断为训练样本少且类别不平衡(信念错误样本稀缺)→ doubt 头置信不足。修复:多翻转期训练 + 加权 BCE。gate 不变,三 seed 全过。

## 回归

`python -m unittest discover -s tests`:233 tests OK(3 skipped)——body 改动(additive,默认关闭)不破坏 v16/v17。

## 待办(不阻塞)

- ~~**Wikidata join 完成** → `run_gk_knowledge_eval.py --db wikidata.db` 跑全量 G-K + 知识库规模/查询验证~~ ✅ **2026-08-15 完成**(见上)。
- M2 输入侧(自然语言 → M-rep)v1 为正则;更鲁棒解析器属流畅核心(M4/M5)。
- Ollama 过渡皮层:等资源/阶段五空闲再挂(监控纪律)。
