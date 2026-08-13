# PHASE4_KNOWLEDGE_LANGUAGE_REPORT — 阶段四近端里程碑 M1/M2/M3

> 2026-08-14。阶段四(完整自研架构)近端三里程碑实现与验证记录。知识库(Wikidata)大 join 后台运行中,M1 全量验证待其完成后执行。

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

判据:recall≥0.95 / 传递≥0.85 / 诚实≥0.98 / ECE≤0.10。产物 `artifacts/phase4-gk-knowledge-eval-{seed}.json`(在 ConceptNet 上;Wikidata join 完成后换库重跑)。

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

- **Wikidata join 完成** → `run_gk_knowledge_eval.py --db wikidata.db` 跑全量 G-K + 知识库规模/查询验证。
- M2 输入侧(自然语言 → M-rep)v1 为正则;更鲁棒解析器属流畅核心(M4/M5)。
- Ollama 过渡皮层:等资源/阶段五空闲再挂(监控纪律)。
