# 知识预训练进展报告（2026-08-13）——真实知识图谱摄入身体

> 性质：知识预训练的第一批积木。G4 在合成数据验证"账本即知识库"；本批把整条管线搬到**真实知识图谱**，并做成**资源安全**（磁盘化、内存有界）。
> 位置：本地 D 盘。数据 `data/knowledge/`，脚本 `experiments/arc/` + `plwm/knowledge_store.py`。
> 数据：ConceptNet 5.7（常识 KG，3.4M 英文边，498MB 下载）；Wikidata truthy（71GB，**下载中**）。

## Step 1：摄入管线（真实数据验证）

- `run_knowledge_ingest.py`：ConceptNet → 解析 → 内存版账本。
- 结果：3.4M 英文事实，61s，1.5GB RAM；召回 1.0、不知检测 ~99.3%、修订（P4）正确。
- **诚实发现**：账本精确知道摄入的，但自然问答需"匹配/推理层"（"dog is an animal" 精确键不存在，但 `IsA(dog,mammal)` 在）。

## Step 2：磁盘化知识库（资源安全，为 Wikidata 打地基）

- `plwm/knowledge_store.py`：**SQLite 磁盘库**（索引精确查询 + BFS 传递闭包），摄入流式、内存有界。
- `run_knowledge_store.py`：全英文摄入 3.3M 事实 → **71.6s / 528MB RAM / 460MB 磁盘**（vs 内存版 1.5GB）。
- 验证：召回 1.0、不知检测 97.6%（0.7-2.4% 误报 = 随机三元组撞上真实知识，非缺陷）；传递推理：apple→fruit KNOW、**dog→animal 传递 KNOW**、**dog→planet 诚实 DON'T KNOW**。
- **资源安全意义**：内存从 1.5GB → 528MB 且持久化磁盘 → **Wikidata 全量（~10⁸ 事实，内存版必爆 15.6GB 机器）变得可能**。

## Step 2b：语言接口（第一迭代，模式匹配版）

- `knowledge_query.py`：自然问题 → 知识查询 + 推理。模式："is X a Y"（精确+传递 IsA）、"what is X"（IsA 集合）、"do you know X"（自知）、"X used for Y"。
- 迭代修正：`an?` 冠词匹配（否则 "is an apple" 变 "n apple"）、specific-first 处理顺序（否则 "what is X" 被吞）。
- 效果：body 从 3.3M 事实回答常识问题，且**精确 + 传递推理 + 诚实不知**。

## 下一步

- **Wikidata truthy（71GB）**：下载中。完成后**分片实测**（先样本测 DB 规模/性能再决定摄入全量），流式摄入保持内存有界。
- **深度 NLP**：语言接口从模式匹配 → 真正的语义解析。
- **推理接入身体**：把知识推理（传递闭包/信念）接进 UnifiedBody，让身体"基于记忆推理"。

## 产物

- `plwm/knowledge_store.py`、`experiments/arc/run_knowledge_ingest.py`、`run_knowledge_store.py`、`knowledge_query.py`
- 数据（gitignore）：`data/knowledge/conceptnet-assertions-5.7.0.csv.gz`（498MB）、`conceptnet.db`（460MB）、`latest-truthy.nt.gz`（71GB 下载中）
- 报告：`artifacts/knowledge-ingest-step1.json`、`knowledge-store-step2.json`
