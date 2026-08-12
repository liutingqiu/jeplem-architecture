# 会话交接文档 — 2026-08-12(架构体检完成 + ARC 落地 + 能力路线定调)

> 用途：新会话从这里读起，用几分钟接续本会话全部工作。本会话完成了三件大事：①架构级判据体检与深修 ②ARC-AGI 落地(自制+真实+官方 ARC-2) ③主体能力路线定调(为架构微调,ARC-2 是最难验收)。
> 工作区：`D:\Documents\探索新架构\.claude\worktrees\zealous-lamport-398898`(git worktree,分支 `claude/zealous-lamport-398898`,工作树干净)

---

## 一、本会话三件大事(按时间顺序)

### 大事 1：架构级判据体检 + 全量深修(CRITERIA_AUDIT.md)

**发现**：项目约 2/3 的"全 1.0/全绿"验证数据是**测量设计幻影**,不是真实能力。三线交叉审计(artifact 完美数据 / 脚本判据实现 / 文档声称)确认 6 种判据缺陷,全部深修:
1. **自洽循环**——判据用与选择函数相同的表达式比较自己(P2/gap/audit)→ 独立熵验证
2. **脚本喂证据**——翻转被脚本连喂触发条件(P4)→ 单次世界变化 + window-majority 自主检测
3. **账本恒真**——append-only 无删除使 P1/P5 必真 → 外部 truth 对照
4. **空样本豁免**——`if samples else 1.0` → 无样本记 fail
5. **宣称判据≠实现判据** → accepted 门对齐 docstring
6. **必败/摆设对照** → 有意义基线

**深修后真实基线**：M6c 自主修订 0.80-0.86(原脚本喂 1.0)、full_subject 连续性 0.17→1.0(修请求帧接线)、v14 gap 恒真 1.0→0.39-0.66、v13 社会认知优势在公平对照下消失(诚实负结果)、v10 因果/v12 价值/v11 执行全部重写为真实机制(各建 v10b/v12b/v11b)。

**可信保留**：m2b 修订(检测 1.0/弱稳定 1.0/误报 0)、m6a/m6b 受控流、v8/v9 路由 0.70-0.71、P6 混合、m3b/m3a/m1/a3。

### 大事 2：ARC-AGI 落地(P6 档3)

**自制基准**：`experiments/arc_grid_generator.py`(7 规则族)+ `arc_perception.py`(物体提取)+ `arc_rule_induction.py`(模板枚举+参数搜索)+ `train_arc_benchmark.py`。自制基准 **heldout 0.976-1.0**(>2x 基线且 >0.5,达标)。

**真实 ARC-1**：`data/arc_training.json`(dataartist/arc-agi 400 任务,代理 7890 从 HuggingFace 拉取)。覆盖 **1.3%**(模板枚举极限),covered 任务 100% 精确。

**官方 ARC-2**：`data/arc2/`(arcprize/ARC-AGI-2 官方 1000 训练 + 120 评估,git 代理克隆)。覆盖 **0/120**——ARC-2 对抗性生成使模板枚举死路,连顶尖模型都 <5%。

**两个参考仓库(已调研)**：
- `arcprize/ARC-AGI-2`——官方数据,已接入
- `poetiq-ai/poetiq-arc-agi-solver`——破纪录方案=LLM 生成解题代码,依赖外部 API,与项目纯本地方向不符,仅作方法参考

### 大事 3：主体能力路线定调(SUBJECT_CAPABILITIES.md + SUBJECT_ACCEPTANCE.md)

**定调**：不是"为 ARC-2 应试微调",是"为架构微调"。ARC-2 是主体能力清单里**最难一档验收**,不是独立目标。用户明确:架构应天然包含度过 ARC-2 的能力。

**能力清单**(SUBJECT_CAPABILITIES.md)：12 项已验证(记忆/自驱/修订/一致性/自知/价值/因果/组合/社会/注意力/路由/感知)+ 5 项部分(视觉组合/示例学规则/抽象视觉/长时运行/真实探索)+ 5 项缺失(语言/规则归纳/长期规划/反事实/主动提问)。

**关键发现**：**C2 规则归纳是总开关**——同时解锁"视觉组合泛化""从示例学规则""抽象视觉推理"三项 + ARC 覆盖。做法=规则 DSL(基础操作×组合)+ 符号执行器,从"预装 7 条公式"升级为"当场拼装新规则"。这是 P6 混合组合器思想("显式骨架+可学内容")的自然延伸,符合项目哲学,纯本地零 API。

**验收关卡**(SUBJECT_ACCEPTANCE.md,全部免费零 API)：
1. ARC-1(现 1.3%)→ 2. ARC-2(现 0%,最难)→ 3. LongMemEval-V2(修订,架构强项)→ 4. GAIA(需工具层,远期)→ 5. BIG-bench/SUPERBENCH(广度)→ 6. 持续运行压力(部分完成)

---

## 二、下一步(新会话从这里开始)

### 核心攻坚：C2 规则归纳层(解锁 ARC-1/ARC-2 + 主体能力)

```
示例对(3-4个)
  → 感知:网格→物体结构(已有 arc_perception)
  → 规则DSL搜索:基础操作(翻转/旋转/换色/平移/取子图/填充)×组合 → 枚举程序
  → 符号执行器跑程序(已有 P6 混合思想)
  → 选择覆盖全部示例的最简程序(Occam)
  → 应用到测试输入
```

**第一步验证**：设计 3-4 个操作的规则 DSL + 搜索,在 ARC-1(400 任务)上测覆盖能否从 1.3% 显著提升且保持 100% 精确。若 DSL 有效,ARC-2 是下一步。

### 并行(不冲突)
- **LongMemEval-V2**(修订关卡):架构强项,性价比高,需拉数据
- **持续运行压力**:主体长时活着验证

---

## 三、关键文件索引

| 文件 | 角色 | 何时读 |
|---|---|---|
| **SUBJECT_CAPABILITIES.md** | 能力清单(12 验证/5 部分/5 缺失),C2 规则归纳是总开关 | 定方向时 |
| **SUBJECT_ACCEPTANCE.md** | 6 道免费验收关卡 + 攻关顺序 | 排任务时 |
| **CRITERIA_AUDIT.md** | 判据可信度矩阵 + 深修前后分数 | 改判据/看基线时 |
| **CURRENT_STATE.md** | 项目交接头部(已更新) | 必读 |
| experiments/arc_*.py | ARC 管线(生成器/感知/推断/基准) | 攻 ARC 时 |
| experiments/train_arc2_official.py | 官方 ARC-2 评估(0/120) | 攻 ARC-2 时 |
| JEPLM_PHASE2_WHITEPAPER.md | P6 档3 判据(已更新) | 定 ARC 判据时 |
| data/arc*.json, data/arc2/ | 缓存数据(ARC-1 400 + ARC-2 1000/120) | 评估时 |

## 四、纪律(保持)

- **判据预写、诚实负结果、3 seed + 对照**(判据审计的核心教训:全满分先怀疑判据)
- **为架构微调,不为应试微调**:ARC-2 是验收不是目标
- **纯本地零 API**:验证用免费基准;API 仅在"与大模型比高低"时可选
- **显式骨架 + 可学内容**:一切机制符合此哲学,不依赖端到端涌现(已证伪)

## 五、本会话提交摘要(关键)

```
1458533 docs: subject acceptance gates
667f8ca docs: functional-subject capability list
5785ab3 ARC-AGI-2 official: 0/120 coverage
1528d4e data: cache real ARC-AGI tasks
34b3a65 docs: P6 tier-3 ARC landing
5a46589 ARC template expansion (+recolor/+bbox)
610df42 ARC real-data eval: 1.3% coverage
267de66 ARC P6 tier-3 self-built benchmark 0.98-1.0
f498aa7 v11b executive control (real mechanism)
259e4e7 v12b value anchoring learned
227cfc3 v10b causality real CI test
... (前置:判据深修 B1-B6、M6c 自主修订、full_subject 修复 等)
```
