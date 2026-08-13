# 知识预训练第一步报告（2026-08-13）——真实知识图谱摄入账本

> 性质：知识预训练的第一块积木。G4 在合成数据上验证了"账本即知识库"；本步把整条管线搬到**真实知识图谱**（ConceptNet 5.7）。
> 位置：本地 D 盘（数据 `data/knowledge/`，脚本 `experiments/arc/run_knowledge_ingest.py`）。
> 数据：ConceptNet 5.7（常识知识图谱，34M 边全部语言 / 3.4M 英文边），官方 S3 直链 498MB 下载。

## 一、结果（真实数据）

| 指标 | 数值 |
|---|---|
| 摄入英文事实 | **3,423,004 条**（唯一键 3.32M） |
| 摄入时间 | **61 秒**（本地 D 盘） |
| 内存峰值 | **~1.5GB**（本机装得下） |
| 精确召回（已摄入） | **1.0** |
| 不知检测（未见） | **~99.3%**（0.7% 误报 = 随机三元组撞上真实知识，非缺陷） |
| 修订（P4） | 正确（冲突事实修订、其余完好） |

**本机可行性**：全英文知识图谱 3.4M 事实摄入账本，61 秒 + 1.5GB RAM——完全本地可行，不需要云/硬盘。

## 二、关键诚实发现：精确记忆需要"匹配/推理层"

自然语言查询 → 精确键的匹配是脆弱的：

| 查询 | 精确键存在？ | 账本直接答 | 传递闭包推理答 |
|---|---|---|---|
| apple is a fruit | ✅ 存在 | **KNOW** | ✓ |
| **dog is an animal** | ❌ 不存在（ConceptNet 存的是 IsA(dog, mammal)、IsA(dog, domestic animal)） | DON'T KNOW | **YES**（dog→mammal→animal 链） |
| car is for transportation | ✅ | KNOW | ✓ |
| dog is a planet（假） | ❌ | DON'T KNOW | **NO**（正确自知） |

**含义**：账本**精确知道摄入的每条事实**，但不做推断。自然问答需要：
1. **匹配/归一化层**（同义词、词形、"animal"↔"mammal"）；
2. **显式推理层**（传递闭包、信念）——**这正是我们架构的机制该出场的地方**。

**"精确账本 + 显式推理"组合的独特价值被演示**：它既不会像 LLM 那样编造（dog→planet 正确说 NO），又能推出账本里没逐字存的事实（dog→animal 链式 YES）。

## 三、下一步（缩放路径）

1. **更大知识源**：Wikidata truthy（~30-40GB、~1 亿+ 事实）——本机内存需扩展（~10-20GB）或分片摄入，这是"全知"的主料。
2. **语言接口**：接 `plwm/language.py`——自然问题 → 归一化 → 账本查询 + 推理。
3. **推理接入身体**：把传递闭包/信念推理接进 UnifiedBody，让身体能"基于记忆推理"而非仅"检索记忆"。
4. **持久化**：账本导出为可复用知识库文件（当前摄入 61 秒可重建，暂不需持久化）。

## 四、产物

- `experiments/arc/run_knowledge_ingest.py`（摄入管线 + 验证）
- `data/knowledge/conceptnet-assertions-5.7.0.csv.gz`（498MB，**gitignore，可重下**）
- `artifacts/knowledge-ingest-step1.json`
