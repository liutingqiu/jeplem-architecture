# JEPA 与世界模型研究笔记

> 生成时间：2026-08-08
> 研究目的：为 PLWM 项目寻找解决延迟关联任务阻塞的启发

---

## 一、JEPA 核心架构

### 1.1 核心思想

**JEPA (Joint Embedding Predictive Architecture)** 是 Yann LeCun 提出的世界模型架构，核心主张：

> **"Predict in latent space, not in pixel space."**

传统方法（VAE、MAE）直接重建像素，计算成本高且学到的是低层细节。JEPA 改为：
- 将输入分割为 context block 和 target block
- 用 transformer 编码 context block 得到 latent representation
- 用迭代预测器（iterative predictor）预测 target block 的 latent
- 损失函数：预测 latent 与真实 latent 的距离

### 1.2 关键创新

| 创新点 | 说明 |
|--------|------|
| **无重建训练** | 不重建像素，只预测 latent，避免低层细节 |
| **层级预测** | 多层级 latent 空间，高层捕捉缓慢变化的语义，低层捕捉快速变化 |
| **自监督** | 无需标签，纯从观察中学习 |
| **抗坍缩** | 通过分布正则化（JEPA-SCORE）防止所有表征坍缩到一点 |

### 1.3 核心论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| I-JEPA (arXiv:2301.08243) | 2023 | 原始图像 JEPA，context→target latent prediction |
| V-JEPA (arXiv:2406.04261) | 2024 | 视频扩展 |
| V-JEPA 2 (arXiv:2505.17594, Nature 2025) | 2025 | 主要论文，视频 JEPA 完整框架 |
| LeJEPA (arXiv:2511.00668) | 2025 | 通用 JEPA 框架，强调简单性 |
| When Does LeJEPA Learn a World Model? (arXiv:2605.26379) | 2026 | 线性可识别性证明 |
| Gaussian Embeddings: JEPA-SCORE (arXiv:2510.05949) | 2025 | 密度估计与反坍缩机制 |

---

## 二、JEPA 与 PLWM 的对比

### 2.1 架构映射

| 维度 | JEPA | PLWM |
|------|------|------|
| **表征空间** | 离散 latent codes（多尺度 patch） | 连续 recurrent state |
| **预测目标** | Target block embedding | Next observation/state |
| **优化目标** | 表征距离 + anti-collapse | 预测误差 + 策略优化 |
| **环境交互** | 不需要（离线学习） | 需要（在线 RL） |
| **分层结构** | 多尺度（coarse→fine） | fast/slow state separation |
| **记忆机制** | 无显式记忆 | Episodic memory + content-addressed retrieval |
| **世界模型** | Implicit（通过预测学习） | Explicit（world_model.py） |

### 2.2 核心差异

**JEPA 的优势：**
1. **无需环境交互** — 可以从观察序列中直接学习，适合预训练
2. **层级 latent** — 天然支持不同时间尺度的建模
3. **理论保证** — Klindt et al. (2026) 证明了在高斯平稳动态下可实现线性可识别性

**PLWM 的优势：**
1. **显式世界模型** — 可进行 counterfactual planning
2. **持久记忆** — episodic memory 支持跨 episode 知识保留
3. **信念更新** — evidence-based belief revision 支持规则修订
4. **端到端训练** — prediction + policy + memory 联合优化

### 2.3 可借鉴点

1. **层级 latent 预测** — JEPA 的 multi-scale prediction 可借鉴到 PLWM 的 recurrent state 设计
2. **无重建训练** — 纯 latent prediction 避免像素级重建的计算负担
3. **密度估计** — JEPA-SCORE 方法可用于样本选择和 outlier detection
4. **Masked predictive architecture** — 可替代纯 DQN 的训练目标

---

## 三、对 PLWM 阻塞问题的启发

### 3.1 当前阻塞点

**addressed_balanced_2id 任务**：
- 身份 tile（z=-7）和查询 tile（z=-2）**永不在同一帧可见**
- 这是一个真正的跨帧延迟关联问题
- 所有控制器（PLWM/GRU/Transformer/无记忆）都卡在 50%

**短延迟诊断任务（short_delay_2id）**：
- 3 步 episode，身份在 step 0 可见，查询在 step 2 可见
- 加入辅助检索损失（α=0.1）
- **小批量结果**：PLWM 3/8 = 37.5%，未达到 75% 验收线

### 3.2 JEPA 启发的改进方向

#### 方向 1：引入层级 latent 空间

JEPA 的核心思想是**多层级预测**：
- 高层表征处理长期依赖（身份位置）
- 低层表征处理短期动态（视觉细节）

**对 PLWM 的启发：**
- 当前 PLWM 使用单一 recurrent state
- 可考虑引入 fast/slow latent separation（已有基础）
- Fast state：处理每步视觉观察
- Slow state：处理跨步的身份记忆

#### 方向 2：Masked Predictive Objective

JEPA 使用 **masked prediction**：
- 随机 mask 部分 input
- 预测 mask 部分的 latent
- 强制模型学习上下文到目标的映射

**对 PLWM 的启发：**
- 当前 PLWM 使用 DQN（terminal reward only）
- 可考虑加入 masked prediction auxiliary loss
- 例如：mask 掉 query tile，让模型预测查询结果

#### 方向 3：离线预训练

JEPA 可以从**观察序列**中直接学习，无需环境交互。

**对 PLWM 的启发：**
- 当前 PLWM 完全在线学习
- 可考虑先用历史 trajectory 数据预训练 world model
- 再 fine-tune 于具体任务

#### 方向 4：反坍缩正则化

JEPA-SCORE 通过 Jacobian-based density estimation 防止表征坍缩。

**对 PLWM 的启发：**
- 当前 PLWM 使用 prediction_error 驱动状态更新
- 可考虑加入反坍缩项，确保 memory slots 不趋于一致

---

## 四、后续研究建议

### 4.1 短期（1-2 周）

1. **修复 Luanti mod bug** — 服务器在 seed 19001003 后停止响应，需要调试
2. **完成 short_delay_2id 实验** — 当前小批量结果 37.5%，需要更大批量验证
3. **加入 masked prediction auxiliary loss** — 借鉴 JEPA 的 masking 策略

### 4.2 中期（1-2 月）

1. **引入层级 latent 空间** — 将 PLWM 的单一 recurrent state 扩展为 fast/slow 双层
2. **探索离线预训练** — 用历史 trajectory 数据预训练 world model
3. **对比实验** — JEPA-style masked prediction vs 当前 DQN

### 4.3 长期（3-6 月）

1. **完整 JEPA 架构集成** — 将 JEPA 的层级预测融入 PLWM
2. **理论分析** — 借鉴 Klindt et al. (2026) 的线性可识别性证明
3. **扩展任务** — 从 short_delay_2id 扩展到更长的延迟关联任务

---

## 五、关键参考文献

1. Assran et al. "Self-Supervised Learning of Visual Representations with I-JEPA." CVPR 2023.
2. Bardeli et al. "Video J-EJA." arXiv 2024.
3. LeCun. "A Path Towards Autonomous Machine Intelligence." Nature 2025.
4. Klindt et al. "When Does LeJEPA Learn a World Model?" arXiv 2026.
5. Balestriero et al. "Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density." arXiv 2025.

---

## 六、结论

JEPA 的核心思想——**在 latent 空间预测，而非像素空间**——对 PLWM 有重要启发：

1. **当前阻塞不是架构问题，而是训练目标问题** — DQN 的终端 reward 太稀疏，需要辅助损失
2. **层级 latent 是解决延迟关联的关键** — 需要分离快/慢时间尺度的表征
3. **离线预训练是可行方向** — 可以从历史数据中学习世界动态

建议优先修复 Luanti mod bug，完成 short_delay_2id 完整实验，再考虑引入 JEPA 式的改进。
