# 阶段 R 报告——向外探索:知识引擎走出快照,向真实世界获取

> 2026-08-16 · 用户方向:"他不应该局限于现有的知识库，而是向外去探索。"系统不再止步于 1.32 亿快照——快照答不了的问题,它去**真实世界(维基百科)**获取,带来源、可验证、不瞎编。已接入部署服务。

---

## 一、机制(`experiments/arc/run_outward_exploration.py` + 服务 `/ask`)

```
快照(1.32 亿事实,wikidata)→ 精确 + 传递推理 → 答
  ↓ 答不了(unknown)
向外探索:fetch 实时维基 API(实时的世界,非快照)
  → 答案 = 真实摘录,标注 [向外探索] + 来源 URL(可验证)
  → 抓取失败 → 诚实"不知道"(绝不瞎编)
```

**关键**:系统明确区分"来自知识库(已验证)"与"来自网络(探索,带来源)"——探索**不冒充知识**。

## 二、结果

**实验(10 实体)**:快照缺口 2 个,向外探索全部填补,接地率 **1.0**(摘录含实体 + 真实维基来源)。

**部署服务实测(2026-08-16)**:

| 问题 | 快照 | 向外探索 |
|---|---|---|
| who is Carl Sagan | 不知道 | **"Carl Edward Sagan was an American astronomer, planetary scientist and science communicator..."**(来源:维基) |
| what is cryonics | 不知道 | **"Cryonics is the low-temperature freezing and storage of human remains..."** |
| who is Mary Shelley | 不知道 | **"Mary Wollstonecraft Shelley was an English novelist who wrote the Gothic novel Frankenstein..."** |
| what is astrobiology | 答(有类目) | 不需要(快照已答) |

## 三、裁决

**PASS:系统向外探索真实世界,不局限于知识库快照**——

1. **走出快照**:快照答不了 → fetch 实时维基(真实世界数据)。
2. **可验证**:答案带来源 URL(用户可核对),摘录是真实内容不是编造。
3. **诚实边界**:网络也答不了 → 诚实"不知道";探索结果标注 [向外探索],不冒充知识库的已验证知识。

**这是"向外探索"的落地**:从"被 1.32 亿快照锁死"到"需要时向真实世界伸手"。

## 四、下一步(用户:"你让他自己去做，自己完善")

1. **自训练 × 向外探索**:自训练时发现知识缺口 → 向外获取 → 纳入候选/知识。**系统用真实世界补自己的短板**。
2. **更广的向外源**:维基之外(实时数据、其他语料)。
3. **探索沉淀**:向外获取的知识经验证后沉淀进账本/库(向外 → 内化)。

**证据**:`artifacts/phase-r-outward-exploration.json` + `experiments/arc/run_outward_exploration.py` + 服务实测。
