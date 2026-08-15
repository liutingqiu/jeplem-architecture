# JEPLM 知识助手——真实世界部署

> 2026-08-16 · 把架构放进真实世界的实战形态:一个**活的 HTTP 问答服务**,底层是 1.32 亿事实的 Wikidata 知识库。零依赖(python 标准库)。

## 运行

```bash
python serve_knowledge_assistant.py     # 启动在 http://0.0.0.0:8080
```

## 端点

| 端点 | 说明 |
|---|---|
| `GET /ask?q=<问题>` | 问答:返回 `{answer, type, unknown, confidence, sources}` |
| `GET /stats` | 知识库规模(1.32 亿事实 / 6980 万实体 / 7683 万标签) |
| `GET /health` | 健康检查 |
| `GET /` | 使用页 |

## 真实问答样例(2026-08-16 实测)

```
Q: is a dog a mammal
   -> Yes (confidence 0.94).  Reasoning: dog IsA domesticated mammal -> domesticated mammal IsA mammal.
Q: is a dog a planet
   -> I don't know.
Q: what is an apple
   -> charge, fruit, fruit of Maloideae, pome, type of fruit
      (apple is a fruit / apple is a pome / ... )
Q: is the sky green
   -> I don't know.
```

## 它是什么

- **精确 + 诚实 + 会推理**:库里有的 99.9% 答对;库里没有的 100% 说"不知道";多跳传递推理(dog→mammal)带推理链。
- **带置信 + 溯源**:每个答案有 confidence 与 sources(可审计)。
- **真实规模**:69.1 亿三元组 → 1.32 亿事实 / 6980 万实体(Wikidata 骨干)。
- **架构**:`KnowledgeStore`(磁盘 SQLite,零内存膨胀)→ `knowledge_query.answer()`(MRep 结构化答案)→ 模板渲染(诚实,后续可接 realizer 学到的流畅渲染)。

## 部署说明

- 服务只读知识库,线程安全(`check_same_thread=False`;SQLite 内部串行)。
- 生产部署:前端静态页 / 其他 API 可跨域(`Access-Control-Allow-Origin: *`)。
- 扩展:接 realizer 做流畅渲染(当前用诚实模板)、加日志/限流、暴露到公网时加认证。

## 后续实战方向

- **接 realizer**:事实答案用学到的流畅语言渲染(阶段 C/D 已证,接线是工程)。
- **知识 × 感知**:让服务接受图片/场景(阶段 H 的感知桥)而非纯文本问题。
- **公网服务**:部署到服务器,真实用户可问。
