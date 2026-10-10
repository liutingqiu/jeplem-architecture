# 阶段 BC 集成报告——知识通道 + 学习层 + v17 核心,同一身体

> 2026-08-15 · 自主发展收官一环。把阶段 B(知识条件化推理)与阶段 C(realizer 学习层)接进 UnifiedBody,演示架构完整形态。3 seed、真实 Wikidata。
> 背景:阶段 B 证明"知识轴启用纯感知做不到的推理";阶段 C 证明"可扩展学习层从真实经验学到语言能力"。本报告:两者与 v17 核心在**同一事件循环**共存,零干扰。

---

## 一、设计(`experiments/arc/run_body_integration.py`)

v18 集成模式:hosted 能力与身份观测交错,决策写**共享账本**,事后打分。

- **身份核(v17)**:信念/修订/路由/self-head(翻转场景)。
- **知识通道(阶段 B)**:知识条件化选择任务(writer 选择,经 KnowledgeStore `known_any(P106, entity, writer)`),决策写账本。
- **学习层(阶段 C)**:word-LSTM realizer(真实语料训练),把事实 M-rep 渲染为自然语言(评分式生成),测 round-trip。

## 二、结果(3 seed 一致)

| seed | belief | revision | router | self-ECE | 知识选择 | 渲染 roundtrip | 账本 | gate |
|---|---|---|---|---|---|---|---|---|
| 444 | 1.0 | ✓ | 1.0 | 0.001 | **1.0** | **1.0** | 40 | ✅ |
| 555 | 1.0 | ✓ | 1.0 | 0.001 | **1.0** | **1.0** | 40 | ✅ |
| 666 | 1.0 | ✓ | 1.0 | 0.001 | **1.0** | **1.0** | 40 | ✅ |

**渲染样例(真实 Wikidata 事实)**:`a Maki is a human` / `a Maki counts as a human`。

## 三、裁决

**PASS(3 seed):知识通道 + 学习层 + v17 核心在同一身体共存,零干扰。**

1. **v17 core 全保真**(belief 1.0/revision/router 1.0/self-ECE 0.001)——新增两个通道不破坏既有机制(G3 集成判据延续)。
2. **知识选择正确**(1.0):body 的语义推理通道用真实知识库选对对象。
3. **渲染 round-trip**(1.0):body 输出经学习层 realizer 渲染为自然语言,主语/宾语完整。
4. **账本溯源**(40 条):知识决策写入共享账本。

**架构完整形态草图成立**:感知/机制/知识/语言四件套在一具身体里协调工作。

## 四、诚实边界

1. **集成是"托管能力"而非深度耦合**:知识通道与 realizer 是 body 事件循环里被调用的能力(hosted),不是重写 body 内部动态。深度集成(知识进 belief/revision 的反馈环)是下一步。
2. **渲染是评分式生成**:自由自回归生成需更大模型(阶段四 M4 0.1-0.3B)。
3. **任务族有限**:知识条件化选择是受控任务;更复杂的两通道协作任务(视觉+知识同时需要)未测。

**证据**:`artifacts/phase-bc-body-integration.json` + `experiments/arc/run_body_integration.py`。
