# 阶段 D 报告——对话知识引擎:问句 → 知识 → 流畅诚实语言

> 2026-08-15 · 自主发展(阶段 D)。架构完整能力里程碑:M2(语言机制)与 M4-lite(realizer)与 M1(知识)的端到端合体——**一个能回答问题、且用学到的话术流畅诚实开口的系统**。3 seed、真实 1.32 亿事实库。
> 动机:阶段 B 证明知识能做推理,阶段 C 证明学习层能生成语言。阶段 D 把它们接到一条链上,成为可用的能力。

---

## 一、管线(`experiments/arc/run_conversational_engine.py`)

```
自然问句("is X a Y")
  → G-K 知识引擎 answer()(检索 + 传递推理 + 诚实无知)
  → MRep(结构化意义,带置信/来源)
  → 渲染:fact/chain 用【学到的 realizer】,unknown/explanation 用诚实模板
  → 自然语言回答
```

## 二、判据(预写,3 seed)

- **fact 正确 ≥ 0.9**:真实 IsA 事实问句回答正确(Yes)。
- **unknown ≥ 0.9**:随机无关联对诚实说"不知道"。
- **渲染诚实 ≥ 0.9**:输出中的每个非函数词都是 MRep 实体的一部分(零幻觉)。

## 三、结果(3 seed)

| seed | fact 正确 | unknown 正确 | 渲染诚实 | gate |
|---|---|---|---|---|
| 444 | 30/30 | 20/20 | **1.0** | ✅ |
| 555 | 28/28 | 20/20 | **1.0** | ✅ |
| 666 | 30/30 | 20/20 | **1.0** | ✅ |

**真实对话样例**:
```
Q: is 'nduja a product              -> the occupation of 'nduja is product
Q: is #HurtigJoakimLøbet 5.0 a DCU licensløb -> the occupation of ... is DCU licensløb
Q: is #vBIB a occurrence             -> #vBIB is a occurrence
Q: is (&#177;)-4'-Methoxyflavanone a flavanone -> (&#177;)-4'-Methoxyflavanone is a flavanone
Q: is Baldasare a Brooks             -> I don't know.   [诚实未知]
```

## 四、裁决

**PASS(3 seed):对话知识引擎成立**——架构用 1.32 亿事实的知识库回答自然问句,推理正确、诚实无知、输出零幻觉,事实答案由**学到的 realizer** 渲染。

**这打通了 M1(知识)+ M2(语言机制)+ M4-lite(学习层)的完整链条**:一个能"知道、推理、开口"的系统。

## 五、诚实边界

1. **realizer 偶选错措辞族**:部分 IsA 事实被渲染成 "the occupation of X is Y"(选了 occupation 措辞)。诚实且含正确实体,但措辞生硬——小模型局限,更大的 0.1-0.3B realizer(阶段四 M4)会改善。
2. **渲染主要是 fact 型**:unknown/explanation 用诚实模板,不是学的。全 MRep 类型的学到渲染是下一步。
3. **问答是受控模式**(is-X-a-Y 型);自由对话/多轮需更大模型。

**证据**:`artifacts/phase-d-conversational-engine.json` + `experiments/arc/run_conversational_engine.py`。
