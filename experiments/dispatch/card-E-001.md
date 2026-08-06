---
name: card-E-001
card-id: E-001
assignee: 工程集成子agent
created: 2026-08-07
priority: critical
depends-on: none
status: pending
---

# 工程集成任务 E-001：正交性保真 + 置信度检索集成

## 背景

数学探索 A/B/C/D 已完成，三项工程修复已实现于 `plwm/torch_model.py`：
1. β 缩放（A-P1a）
2. 键正交初始化 + QR 投影（A-P1c）
3. confidence-weighted softmax（B-P2c）

但 **7 个训练脚本** 尚未集成 `enforce_key_orthogonality()` 调用，
需要逐一适配。

## 任务

### 任务 1：集成正交性保真到所有训练脚本

文件列表（使用 TorchPLWM 的）：
1. `experiments/train_associative_memory.py`
2. `experiments/train_continuity.py`
3. `experiments/train_luanti_gated_memory_dqn.py`
4. `experiments/train_luanti_reward_only.py`
5. `experiments/train_minigrid_memory_actor_critic.py`
6. `experiments/train_object_token_memory.py`
7. `experiments/train_public_vision_candidate_memory.py`

修改模式（统一）：
```python
# 在 optimizer.step() 之后添加：
if hasattr(model.core, 'enforce_key_orthogonality'):
    model.core.enforce_key_orthogonality()
```

注意：`StatelessBaseline` 和基线模型（no_memory/GRU/Transformer）不需要此调用。

### 任务 2：验证所有训练脚本
- 每个脚本跑一次 smoke（不跑完整训练，只验证 import + 单步 forward + backward 不报错）
- 输出：通过/失败 + 错误信息

### 任务 3：写兼容性测试
在 `tests/test_torch_plwm_fixes.py` 中新增测试：
- test_orthogonal_initialization：验证 key_head 初始化为正交矩阵
- test_confidence_weighted_read：验证高置信度 slot 检索更准确
- test_enforce_key_orthogonality：验证训练后正交性保持
- test_checkpoint_backward_compat：验证旧 checkpoint 可恢复
- test_beta_default：验证默认 beta=1.0（与原代码行为一致）

## 验收标准
- 所有 7 个训练脚本 smoke 通过（import + 单步）
- 所有 5 个测试通过
- `train_associative_memory.py` 和 `train_continuity.py` 准确率不变（1.0）

## 回报

完成后在 `experiments/dispatch/status-E-001.md` 写入：
```markdown
---
card-id: E-001
status: completed
timestamp: 2026-08-07
summary: 7 个训练脚本集成完成，5 个测试通过
---

集成结果：
- train_associative_memory.py: ✅
- train_continuity.py: ✅
- train_luanti_gated_memory_dqn.py: ✅
- train_luanti_reward_only.py: ✅
- train_minigrid_memory_actor_critic.py: ✅
- train_object_token_memory.py: ✅
- train_public_vision_candidate_memory.py: ✅
- tests: 5/5 通过
```
