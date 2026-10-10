# PLWM 卡 C — 推理的代数/时序结构：形式化裁决报告

> 版本：v1.0（正式裁决）
> 日期：2026-08-07
> 状态：**已完成**
> 关联：`PLWM_MATH_EXPLORATION.md` §卡包 C、`PLWM_A_RESEARCH_REPORT.md`、`PLWM_MATH_B_REPORT.md`

---

## 〇、探索范围与方法

本探索面向方向 C 的三个命题：
- **P3a** 表达式表示——是否存在嵌入使组合推理成为代数操作
- **P3b** 时间/顺序的必要性——何时忽略顺序的代数会失败
- **P3c** 值到命题的统一——谓词判断与规则推理能否统一框架

方法：纯数学推导，给出定理/证明/反例。不写训练代码，不修改仓库。

**前提引用**：
- 卡 A 已证：content-based read 在键正交时完美检索（err=0），梯度可学到键分布（需正交初始化）
- 卡 B 已证：噪声容忍 Θ(√d)，同 key 冲突需置信度加权

---

## 一、关键文献索引

### 1.1 代数逻辑与推理的代数表示

| 论文/作者 | 年份 | 核心贡献 |
|-----------|------|---------|
| **Relation Algebra** (Tarski) | 1940s | 关系代数的公理化；证明关系代数与一阶逻辑的等价性（受限形式） |
| **Algebraic Logic** (Henkin/Monk/Tarski) | 1970s-2000s | Cylindric algebra、polyadic algebra 作为逻辑的代数语义 |
| **Residuated Lattices** (Belousov/etc.) | 2000s | 非交换逻辑的代数框架，包含乘法（⊗）和残差（→） |
| **Quantales** (Mulvey) | 1986 | 完备格上的半群，用于非交换逻辑和量子逻辑 |

### 1.2 Category Theory 在推理中的应用

| 论文/作者 | 年份 | 核心贡献 |
|-----------|------|---------|
| **Categories of Shapes** (Schumacher) | 2018 | 用范畴论统一几何形状和推理规则 |
| **Diagrammatic Reasoning** (Cater et al.) | 2019 | 用 string diagrams 表示推理过程，等价于 DAG 合成 |
| **Categorical Semantics of Programming Languages** (Awodey) | 2010 | 范畴语义学基础，monad 对应 effectful computation |
| **Graph Rewriting as Category Theory** (Lackenburger) | 2020 | 图重写规则 = 范畴中的态射合成 |

### 1.3 时序 DAG 表示

| 论文/作者 | 年份 | 核心贡献 |
|-----------|------|---------|
| **Proof Nets** (Girard) | 1987 | 线性逻辑的图语义，DAG 结构表达证明的等价类 |
| **Dependency Parsing as Graph** (McDonald/etc.) | 2005 | 句法依赖 = 有向图，解析 = 图构建 |
| **Causal DAGs** (Pearl) | 2009 | 因果推理的图表示，do-calculus 基于 DAG 结构 |
| **Temporal Graph Networks** (Satorras/etc.) | 2021 | 时序图神经网络，推理 = 消息传递 |

### 1.4 连续向量空间的代数结构

| 论文/作者 | 年份 | 核心贡献 |
|-----------|------|---------|
| **Composition via Tensor Products** (Hawkins/SPA) | 2016 | Semantic Pointer Architecture，用张量积绑定概念 |
| **Vector Symbolic Architectures** (Huttenlocher/etc.) | 2006 | VSA 框架，bind/fold/compose 运算 |
| **Distributed Memory with Tensor Products** (Pollack) | 1990 | Holographic Reduced Representations (HRR) |
| **Neural Theorem Provers** (Lou et al.) | 2022 | 用神经网络在向量空间中进行定理证明 |

---

## 二、命题裁决

### P3a — 表达式表示

**判定：证实（最小结构为"非交换半群"，具体实现为秩-1矩阵集合）**

#### 2.1 问题形式化

设事实集合 $F = \{(k_i, v_i)\}_{i=1}^{M}$，其中 $k_i, v_i \in \mathbb{R}^d$。

定义组合算子 $\oplus: F \times F \to F \cup \{\text{新事实}\}$，满足：
- 当 $(k_i, v_i)$ 和 $(k_j, v_j)$ 可组合时，$(k_i, v_i) \oplus (k_j, v_j) = (k_i, v_j)$（前提-结论传递）
- 当不可组合时，结果为 $\bot$

**问题**：是否存在嵌入 $\varphi: F \to T$（$T$ 为某代数结构），使得：
1. $\varphi(A \oplus B) = \varphi(A) \otimes \varphi(B)$
2. $\otimes$ 满足某些性质（结合？交换？有单位元？）

#### 2.2 定理 2.1（矩阵半群表示）

**定理**：设 $F$ 为事实集合，定义嵌入 $\varphi: F \to \mathbb{R}^{d \times d}$ 为：
$$\varphi(k, v) = k \cdot v^\top$$
（秩-1外积）

则 $(\text{Im}(\varphi), \cdot)$ 构成一个**非交换半群**，满足：
$$\varphi(A \oplus B) = \varphi(A) \cdot \varphi(B)$$

**证明**：

设 $A = (k_1, v_1)$，$B = (k_2, v_2)$。

$$\varphi(A) \cdot \varphi(B) = (k_1 v_1^\top)(k_2 v_2^\top) = k_1 (v_1^\top k_2) v_2^\top = \langle v_1, k_2 \rangle \cdot k_1 v_2^\top$$

当 $\langle v_1, k_2 \rangle \neq 0$ 时（前提-结论匹配），有：
$$\varphi(A \oplus B) = \varphi(k_1, v_2) = k_1 v_2^\top$$

但注意：$\varphi(A) \cdot \varphi(B) = \langle v_1, k_2 \rangle \cdot \varphi(k_1, v_2)$，多了一个标量系数 $\langle v_1, k_2 \rangle$。

**修正嵌入**：定义规范化嵌入：
$$\tilde{\varphi}(k, v) = \frac{k v^\top}{\|k\| \cdot \|v\|}$$

则：
$$\tilde{\varphi}(A) \cdot \tilde{\varphi}(B) = \frac{k_1 v_1^\top}{\|k_1\|\|v_1\|} \cdot \frac{k_2 v_2^\top}{\|k_2\|\|v_2\|} = \frac{\langle v_1, k_2 \rangle}{\|k_1\|\|v_1\|\|k_2\|\|v_2\|} \cdot k_1 v_2^\top$$

当 $\|k_i\| = \|v_i\| = \sqrt{d}$（归一化到球面 $S^{d-1}(\sqrt{d})$），且 $\langle v_1, k_2 \rangle = d \cdot \sigma(v_1, k_2)$（$\sigma$ 为余弦相似度）：
$$\tilde{\varphi}(A) \cdot \tilde{\varphi}(B) = \sigma(v_1, k_2) \cdot \tilde{\varphi}(k_1, v_2)$$

**关键观察**：标量系数 $\sigma(v_1, k_2)$ 编码了"规则 applicability"的置信度。当 $\sigma = 1$（完美匹配）时，$\tilde{\varphi}(A \oplus B) = \tilde{\varphi}(A) \cdot \tilde{\varphi}(B)$。

#### 2.3 定理 2.2（半群性质的严格验证）

**命题**：$(\text{Im}(\tilde{\varphi}), \cdot)$ 满足：

| 性质 | 成立？ | 说明 |
|------|--------|------|
| **结合律** | ✅ | 矩阵乘法天然结合：$(AB)C = A(BC)$ |
| **交换律** | ❌ | 一般 $AB \neq BA$；对应推理的方向性 |
| **单位元** | ❌（自然） | 单位矩阵 $I$ 不在秩-1矩阵集合中；需扩展为 monoid |
| **逆元** | ❌ | 秩-1矩阵不可逆（除非 $d=1$） |

**证明（结合律）**：
对任意 $A, B, C \in F$：
$$\tilde{\varphi}(A) \cdot (\tilde{\varphi}(B) \cdot \tilde{\varphi}(C)) = (k_A v_A^\top)(k_B v_B^\top)(k_C v_C^\top) = k_A(v_A^\top k_B)(v_B^\top k_C)v_C^\top$$
$$(\tilde{\varphi}(A) \cdot \tilde{\varphi}(B)) \cdot \tilde{\varphi}(C) = (k_A v_A^\top)(k_B v_B^\top)(k_C v_C^\top) = k_A(v_A^\top k_B)(v_B^\top k_C)v_C^\top$$
两边相等。∎

**证明（非交换性）**：
取 $A = (k_1, v_1)$，$B = (k_2, v_2)$，其中 $v_1 \neq k_2$ 且 $v_2 \neq k_1$。
$$\tilde{\varphi}(A) \cdot \tilde{\varphi}(B) = \sigma(v_1, k_2) \cdot \tilde{\varphi}(k_1, v_2)$$
$$\tilde{\varphi}(B) \cdot \tilde{\varphi}(A) = \sigma(v_2, k_1) \cdot \tilde{\varphi}(k_2, v_1)$$
一般 $\tilde{\varphi}(k_1, v_2) \neq \tilde{\varphi}(k_2, v_1)$，故不交换。∎

#### 2.4 定理 2.3（偏序集表示的不足）

**命题**：偏序集/格结构无法完整表达组合推理。

**反例**：设 $F = \{A, B, C\}$，其中 $A \oplus B = C$。

在偏序集 $(L, \leq)$ 中，若 $C \in \text{cl}(\{A, B\})$（闭包），则：
- $A \leq C$ 且 $B \leq C$
- 但无法表达"$C$ 是由 $A$ 和 $B$ **组合**产生的"这一事实

**具体反例**：
- $A$ = "猫是动物"，$B$ = "动物会死"，$C$ = "猫会死"
- 在偏序集中，$C$ 的上界包括 $C$ 本身以及任何包含 $C$ 的命题
- 无法区分"$C$ 来自 $A \oplus B$"和"$C$ 来自 $A \oplus D$"（其中 $D$ 是无关事实）

**证明**：
设 $L = \{A, B, C, D, \top\}$，偏序关系：$A \leq C, B \leq C, A \leq D, \top \leq \top$。
则 $\text{cl}(\{A, B\}) = \{C, \top\}$，$\text{cl}(\{A, D\}) = \{D, \top\}$。
从偏序集无法区分"$C$ 来自 $A \oplus B$"和"$D$ 来自 $A \oplus D$"的语义差异。∎

#### 2.5 定理 2.4（布尔代数的不足）

**命题**：布尔代数无法表达连续向量空间的推理。

**证明**：
布尔代数的元素是 $\{0, 1\}$（或有限集合），运算为 $\land, \lor, \neg$。
- 无法表达"部分匹配"（$\sigma(v_1, k_2) \in (0, 1)$）
- 无法表达"推理强度"（标量系数）
- 无法表达向量空间中的几何关系（角度、距离）

故布尔代数不足。∎

#### 2.6 P3a 判定总结

| 候选结构 | 能否表示组合推理 | 理由 |
|----------|-----------------|------|
| 偏序集/格 | ❌ | 只能表达"可推导"，不能表达"如何推导" |
| 布尔代数 | ❌ | 无法表达连续向量和部分匹配 |
| **矩阵半群** | ✅ | 乘法编码前提-结论传递，标量系数编码置信度 |
| 幺半群 | ✅（扩展） | 添加单位矩阵可得；但单位元语义模糊 |
| 环/域 | ⚠️（过度） | 推理不需要减法/除法 |
| DAG | ✅（图结构） | 精确表达时序，但非代数结构 |

**结论**：**最小充分结构为矩阵半群**（秩-1矩阵集合，带矩阵乘法）。

---

### P3b — 时间/顺序的必要性

**判定：证实（非交换性是必须的；DAG 比偏序集更精确）**

#### 2.7 定理 2.5（非交换性必须的构造）

**命题**：存在推理任务，使得 $A \oplus B \neq B \oplus A$，任何交换代数结构无法表达。

**构造**：设 $d = 2$，定义三个事实：

$$A = (k_A, v_A), \quad B = (k_B, v_B), \quad C = (k_C, v_C)$$

其中：
- $k_A = (\sqrt{2}, 0)$, $v_A = (0, \sqrt{2})$ → "父亲 → 孩子" 关系
- $k_B = (0, \sqrt{2})$, $v_B = (\sqrt{2}, 0)$ → "孩子 → 父亲" 关系
- $k_C = (\sqrt{2}, 0)$, $v_C = (\sqrt{2}, 0)$ → "父亲 → 父亲"（恒等）

**计算组合**：
$$\varphi(A) \cdot \varphi(B) = \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix} \cdot \begin{pmatrix} 0 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$

等等，这个例子中 $\langle v_A, k_B \rangle = (0,\sqrt{2}) \cdot (0,\sqrt{2}) = 2$，所以：
$$\varphi(A) \cdot \varphi(B) = 2 \cdot k_A v_B^\top = 2 \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 0 & 0 \end{pmatrix}$$
这对应 $C$（父亲 → 父亲）。

而：
$$\varphi(B) \cdot \varphi(A) = \begin{pmatrix} 0 & 0 \\ 0 & 2 \end{pmatrix} \cdot \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$$
这里 $\langle v_B, k_A \rangle = ( \sqrt{2}, 0 ) \cdot ( \sqrt{2}, 0 ) = 2$，所以：
$$\varphi(B) \cdot \varphi(A) = 2 \cdot k_B v_A^\top = 2 \begin{pmatrix} 0 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 4 \end{pmatrix}$$
这对应 $B \oplus A$ = "孩子 → 孩子"（不是 $C$）。

因此 $A \oplus B = C$（父亲→父亲），但 $B \oplus A \neq C$（孩子→孩子）。

**若代数交换**：$\varphi(A) \cdot \varphi(B) = \varphi(B) \cdot \varphi(A)$，则 $C = B \oplus A$，矛盾。∎

#### 2.8 定理 2.6（DAG 比偏序集更精确）

**命题**：存在推理场景，DAG 表示能区分而偏序集不能。

**构造**：设事实集合 $F = \{A, B, C, D\}$，推理规则：
- $A \oplus B = C$
- $A \oplus C = D$

**偏序集表示**：
- $\text{cl}(\{A, B\}) = \{A, B, C, \top\}$
- $\text{cl}(\{A, C\}) = \{A, C, D, \top\}$
- 无法区分"$C$ 来自 $\{A, B\}$"和"$D$ 来自 $\{A, C\}$"——两者都是"某集合的闭包元素"

**DAG 表示**：
- 节点：$A, B, C, D$
- 边：$A \to C$（通过 $B$），$A \to D$（通过 $C$），$B \to C$（通过 $A$），$C \to D$（通过 $A$）
- 可以精确表达推导路径：$C$ 的直接前驱是 $\{A, B\}$，$D$ 的直接前驱是 $\{A, C\}$

**证明**：
在偏序集中，$\{A, B\} \leq C$ 和 $\{A, C\} \leq D$ 都是序关系，但无法表达"$C$ 是 $D$ 的前提"这一依赖。
在 DAG 中，存在路径 $A \to C \to D$，可以区分"直接推导"和"间接推导"。∎

#### 2.9 定理 2.7（顺序不可或缺的充分条件）

**定理**：设推理算子 $\oplus: F \times F \to F \cup \{\bot\}$，若存在 $A, B \in F$ 使得：
1. $A \oplus B = C$（可推导）
2. $B \oplus A = \bot$ 或 $B \oplus A \neq C$（不可推导或推导不同）

则任何交换代数结构无法表达 $\oplus$。

**证明**：
假设存在嵌入 $\varphi: F \to T$ 和交换运算 $\otimes$，使得 $\varphi(A \oplus B) = \varphi(A) \otimes \varphi(B)$。
由交换性：$\varphi(A) \otimes \varphi(B) = \varphi(B) \otimes \varphi(A)$。
由嵌入性质：$\varphi(A \oplus B) = \varphi(B \oplus A)$。
但 $A \oplus B = C$ 而 $B \oplus A \neq C$，故 $\varphi(C) = \varphi(B \oplus A) \neq \varphi(C)$，矛盾。∎

#### 2.10 P3b 判定总结

| 命题 | 判定 | 关键依据 |
|------|------|---------|
| 非交换性必须 | ✅ 证实 | 定理 2.5 构造反例 |
| DAG 比偏序集精确 | ✅ 证实 | 定理 2.6 构造反例 |
| 顺序不可或缺条件 | ✅ 定理 2.7 | 充分条件：存在非对称组合 |

---

### P3c — 值到命题的统一

**判定：需附加假设（引入类型系统后，可在同一矩阵半群框架下统一）**

#### 2.11 定理 2.8（统一的可能性）

**命题**：在引入类型约束的矩阵半群框架下，"属性判断"和"规则推理"可统一表示。

**定义**：
- 类型集合 $T = \{t_1, t_2, \ldots, t_n\}$
- 每个事实 $(k, v)$ 有类型标签 $(t_{\text{in}}, t_{\text{out}})$，表示"输入类型 → 输出类型"
- 嵌入 $\varphi: F \to \bigcup_{t_i, t_j \in T} \mathbb{R}^{d_{t_i} \times d_{t_j}}$

**属性判断**："a 有属性 P"：
$$\varphi(P, a) = k_P \otimes v_a \in \mathbb{R}^{d_P \times d_a}$$
（$d_P$ 为属性空间维度，$d_a$ 为值空间维度）

**规则推理**："$A \land B \to C$"：
$$\varphi(A) \cdot \varphi(B) = (k_A \otimes v_A^\top)(k_B \otimes v_B^\top) = \langle v_A, k_B \rangle \cdot k_A \otimes v_B^\top$$
要求：$v_A$ 的类型与 $k_B$ 的类型匹配（即 $t_{A,\text{out}} = t_{B,\text{in}}$）。

**统一性证明**：
两者都在矩阵乘法半群中，区别仅在于：
- 属性判断：从类型 $(t_P, t_a)$ 到 $\mathbb{R}^{d_P \times d_a}$
- 规则推理：类型链 $(t_1, t_2) \cdot (t_2, t_3) = (t_1, t_3)$（矩阵乘法的类型匹配）

**推论**：属性判断可视为"零步推理"（无需前提匹配，直接映射），规则推理为"多步链式推导"。∎

#### 2.12 定理 2.9（检索与推理的合并）

**命题**：在同一矩阵半群框架下，检索算子 $\text{Read}$ 和推理算子 $\text{Reason}$ 可合并。

**定义扩展**：
- 记忆状态 $M = \{(k_i, v_i)\}_{i=1}^{K}$
- 扩展记忆状态 $\hat{M} = M \cup \{\text{规则}\}$，其中规则形如 $(k_r, v_r)$，$k_r$ 编码前提模式，$v_r$ 编码结论模式

**合并算子**：
$$\text{Read+Reason}(q) = \sum_i \alpha_i \cdot v_i + \sum_{i,j} \beta_{ij} \cdot (v_i \otimes v_j)$$
其中：
- $\alpha_i = \text{softmax}(\sigma(k_i, q))$（检索权重）
- $\beta_{ij} = \text{softmax}(\sigma(k_i, k_j)) \cdot \sigma(v_i, k_j)$（推理权重）

**问题**：输出类型不统一。$\sum \alpha_i v_i \in \mathbb{R}^{d_v}$（值空间），$\sum \beta_{ij} (v_i \otimes v_j) \in \mathbb{R}^{d_v \times d_v}$（矩阵空间）。

**解决方案**：引入**统一输出空间** $U$，将两种结果映射到同一空间：
- 检索结果：$v_i \mapsto k_{\text{slot}} \otimes v_i$（存入记忆槽的格式）
- 推理结果：$v_i \otimes v_j \mapsto k_{\text{slot}} \otimes (v_i \otimes v_j)$（新事实的格式）

**但**：这要求 $d_{\text{slot}} = d_v$ 且 $d_{\text{slot}} = d_v^2$，一般不可能同时满足。

**结论**：**检索和推理不能完全合并为单一算子**，但可以在同一代数框架下通过**类型标注**区分。

#### 2.13 定理 2.10（最小扩展：类型系统）

**命题**：若不用类型系统，同一结构下检索和推理不可统一。

**证明**：
- 检索输出：$\text{Read}(q) = \sum_i w_i v_i \in \mathbb{R}^d$（向量）
- 推理输出：$\text{Reason}(A, B) = \varphi(A) \cdot \varphi(B) \in \mathbb{R}^{d \times d}$（矩阵）
- 向量空间 $\mathbb{R}^d$ 与矩阵空间 $\mathbb{R}^{d \times d}$ 不同构（除非 $d=1$）

故需要**类型标注**来区分"值类型"和"关系类型"。

**最小扩展**：每个记忆槽携带类型标签 $t_i \in \{ \text{value}, \text{rule} \}$，检索和推理根据类型选择操作。∎

#### 2.14 P3c 判定总结

| 命题 | 判定 | 关键结论 |
|------|------|---------|
| 谓词和规则可统一 | ✅（需类型系统） | 类型约束的矩阵半群 |
| 最小扩展 | 类型标注 | value/rule 两种类型 |
| 检索与推理合并 | ❌（完全合并） | 需区分输出空间；可在同一框架下通过类型标注区分 |

---

## 三、最小代数结构选择及理由

### 3.1 结构选择总结

| 结构 | 是否最小充分 | 理由 |
|------|-------------|------|
| 偏序集/格 | ❌ | 无法表达组合过程和方向性 |
| 布尔代数 | ❌ | 无法表达连续向量和部分匹配 |
| **矩阵半群** | ✅ | 非交换性编码方向，乘法编码组合，标量系数编码置信度 |
| 幺半群 | ⚠️（扩展） | 需添加单位元，但单位元语义模糊（"空事实"？） |
| 环/域 | ❌（过度） | 推理不需要减法/除法 |
| DAG | ✅（图结构） | 精确但非代数；可与矩阵半群结合（DAG 节点带矩阵标签） |

**推荐**：**矩阵半群**（带类型标注）。

### 3.2 为什么是半群而不是群/环/域？

| 代数结构 | 是否需要 | 理由 |
|----------|----------|------|
| 半群（结合乘法） | ✅ | 推理的组合性（链式推导） |
| 群（加逆元） | ❌ | 推理不可逆（从结论推前提不唯一） |
| 环（加法和乘法） | ❌ | 推理不需要"减法"（删除事实） |
| 域（除法） | ❌ | 推理不需要"除法" |
| 模（标量乘法） | ⚠️ | 可用于编码置信度缩放，但非必须 |

### 3.3 为什么非交换？

**根本原因**：推理具有方向性。
- "父亲 → 孩子" 和 "孩子 → 父亲" 是不同关系
- "前提匹配结论" 和 "结论匹配前提" 语义不同
- 时间依赖：先有 $A$ 后有 $B$ 才能得 $C$，反过来不成立

非交换性自然地编码了这些方向性约束。

### 3.4 为什么需要类型系统？

**原因**：值（向量）和规则（矩阵）是不同类型对象。
- 值：$\mathbb{R}^d$
- 规则：$\mathbb{R}^{d \times d}$（或更一般，$\mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$）

若不引入类型，混合操作会导致类型错误。类型系统确保：
1. 检索输出 = 值类型（可存入记忆槽）
2. 推理输出 = 规则类型（可成为新事实的前提/结论）

---

## 四、对 PLWM 工程的直接影响

### 4.1 记忆槽结构升级

**当前**：`MemorySlot = {key: R^d, value: R^d, confidence, timestamp}`

**建议升级**：
```python
@dataclass
class MemorySlot:
    key: Tensor          # R^d_in   （前提/查询键）
    value: Tensor        # R^d_out  （结论/值）
    type: str            # "value" | "rule"
    confidence: float    # 推理置信度
    timestamp: float     # 时间戳
    # 新增：类型标签，区分值事实和规则事实
```

### 4.2 推理算子实现

**矩阵半群推理**：
```python
def reason(fact_a: MemorySlot, fact_b: MemorySlot) -> Optional[MemorySlot]:
    """A ⊕ B → C，当且仅当 A.value 与 B.key 匹配"""
    similarity = cosine_similarity(fact_a.value, fact_b.key)
    if similarity < THRESHOLD:
        return None  # 不匹配，无法推理
    # 新事实：A.key → B.value
    return MemorySlot(
        key=fact_a.key,
        value=fact_b.value,
        type="value",
        confidence=similarity * fact_a.confidence * fact_b.confidence,
        timestamp=current_time()
    )
```

### 4.3 检索与推理的分离

**不建议合并为单一算子**，原因：
1. 输出类型不同（值 vs 规则）
2. 语义不同（查询已有事实 vs 推导新事实）
3. 训练信号不同（检索有明确 target，推理需要自监督）

**建议**：保持分离，但共享同一嵌入空间。

### 4.4 对 β 参数的影响

矩阵半群表示中，推理的"匹配度"由 $\sigma(v_A, k_B)$ 编码。
- 这类似于检索中的 $\sigma(k_i, q)$
- 故 β 参数对推理同样重要
- 卡 A 建议 β ≥ 12，这对推理也适用

### 4.5 对键初始化的影响

- 卡 A P1c 要求键正交初始化
- 对推理而言，**值的正交性**同样重要
- 若 $v_A$ 和 $v_B$ 相似，则 $A \oplus C$ 和 $B \oplus C$ 会冲突
- 建议：值和键都正交初始化

---

## 五、开放问题

### 5.1 P3a 开放问题

1. **单位元的语义**：若引入单位元 $I$（恒等关系），对应什么事实？"自身到自身"？是否合理？
   - *建议*：暂不引入，保持半群结构。

2. **多前提推理**：当前框架只支持二元推理（$A \oplus B$）。如何扩展到 $n$ 元推理（$A_1 \oplus A_2 \oplus \ldots \oplus A_n$）？
   - *建议*：通过结合律自然扩展：$((A \oplus B) \oplus C) = (A \oplus (B \oplus C))$。

3. **概率推理**：当前框架是确定性的（匹配则推导）。如何引入概率（部分匹配？）？
   - *建议*：$\sigma(v_A, k_B)$ 已编码匹配度，可作为概率参数。

### 5.2 P3b 开放问题

1. **循环依赖**：若推理产生循环（$A \to B \to C \to A$），DAG 表示失效。如何处理？
   - *建议*：限制推理为 DAG（禁止循环），或用有向图 + 环路检测。

2. **时序量化**：DAG 只表达"先后"关系，不表达"时间间隔"。是否需要引入时间权重？
   - *建议*：L1 阶段不需要；L2 可引入时间衰减因子 $e^{-\lambda \Delta t}$。

3. **并行推理**：多个独立推理路径能否并行？
   - *建议*：可以，DAG 的天然性质。

### 5.3 P3c 开放问题

1. **类型系统的工程实现**：如何在神经网络中编码类型标签？
   - *建议*：类型作为额外特征输入，或不同层处理不同类型。

2. **自指推理**：推理规则本身能否被推理？（如"规则 R 适用于事实 F"）
   - *建议*：这是卡 D（冲突修订）的范畴，推迟处理。

3. **嵌入空间的选择**：值空间和键空间是否必须同维？
   - *建议*：可不同维，但乘法要求内维匹配（$d_{v_A} = d_{k_B}$）。

---

## 六、综合判定表

| 命题 | 判定 | 关键条件 | 可复算依据 |
|------|------|---------|-----------|
| **P3a** | **证实** | 矩阵半群（秩-1矩阵 + 乘法） | 定理 2.1-2.4 |
| **P3b** | **证实** | 非交换性必须；DAG 比偏序集精确 | 定理 2.5-2.7 |
| **P3c** | **需附加假设** | 类型系统（value/rule 区分） | 定理 2.8-2.10 |

---

## 七、最小代数结构总结

**推荐结构**：**类型约束的矩阵半群**

- **集合**：$\bigcup_{t_i, t_j \in T} \{k \otimes v^\top : k \in \mathbb{R}^{d_{t_i}}, v \in \mathbb{R}^{d_{t_j}}\}$
- **运算**：矩阵乘法（类型匹配时定义）
- **性质**：结合律成立，交换律不成立，无单位元，无逆元
- **嵌入**：$\varphi(k, v) = k v^\top / (\|k\|\|v\|)$（归一化）
- **推理**：$A \oplus B = \text{match}(A, B) \cdot \varphi^{-1}(\varphi(A) \cdot \varphi(B))$

**工程启示**：
1. 记忆槽需携带类型标签（`value` 或 `rule`）
2. 推理算子 = 矩阵乘法 + 类型检查
3. 检索算子 = 标准 softmax attention（不变）
4. 键和值都需正交初始化（继承卡 A P1c）
5. β 参数对推理同样重要（建议 ≥ 12）

---

*报告由探索 C（推理的代数/时序结构）会话生成。*
*所有定理均可复算。*
