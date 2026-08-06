---
card-id: E-001
status: completed
timestamp: 2026-08-07
summary: 7训练脚本集成完成，6测试全部通过
---

关键结论：
- train_associative_memory.py: ✅ hook present, accuracy=1.0
- train_continuity.py: ✅ hook present, accuracy=1.0
- train_luanti_gated_memory_dqn.py: ✅ hook present (in _learn function)
- train_luanti_reward_only.py: ✅ hook present
- train_minigrid_memory_actor_critic.py: ✅ hook present (2 update loops)
- train_object_token_memory.py: ✅ hook present
- train_public_vision_candidate_memory.py: ✅ hook present
- tests/test_torch_plwm_fixes.py: 6/6 passed
  - test_orthogonal_init, test_confidence_weighted
  - test_enforcement, test_training_with_enforcement
  - test_checkpoint_backward_compat, test_beta_default
