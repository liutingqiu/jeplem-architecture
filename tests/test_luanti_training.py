import unittest

import torch

from plwm.luanti_training import (
    CausalControlWorldModel,
    ControlMechanismMemory,
    DIRECTIONS,
    LuantiInteraction,
    LuantiLiveClient,
    GRUContextGoalPolicy,
    GoalDirectedPLWM,
    LuantiState,
    MaskedSensorGoalPolicy,
    NoMemoryGoalPolicy,
    NoMemoryObjectPolicy,
    ObjectTaskState,
    GRUObjectPolicy,
    TransformerObjectPolicy,
    padded_histories,
    encode_vision,
    augment_visual_observations,
    VISION_TILES,
    VisionGoalDirectedPLWM,
    VisionObjectGoalDirectedPLWM,
    NoMemoryVisionObjectPolicy,
    RewardMemoryVisionObjectPLWM,
    NoMemoryRewardVisionObjectPolicy,
    FactMemoryVisionObjectPLWM,
    FactMemoryCurrentVisionObjectPolicy,
    TwoFactMemoryCurrentVisionObjectPolicy,
    object_two_fact_memory_visual_frame_features,
    PolarityFactMemoryCurrentVisionObjectPolicy,
    object_polarity_fact_memory_visual_frame_features,
    MultiChoiceFactMemoryCurrentVisionObjectPolicy,
    object_multi_choice_fact_memory_visual_frame_features,
    object_multi_choice_public_visual_frame_features,
    VerifiedObjectFactMemory,
    VerifiedFactObjectPlanner,
    padded_object_visual_histories,
    reconcile_visual_sensors,
    temporal_consensus_vision,
)


class LuantiTrainingTests(unittest.TestCase):
    def test_goal_teacher_probes_then_plans_through_hidden_controls_and_barrier(self):
        # Seed zero has a gate at x=3 and inverted controls.  The first east
        # request is a safe probe; its observed westward movement identifies
        # the inversion before the teacher requests north (actual south).
        state = LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False)
        self.assertEqual(state.oracle_direction(), "east")
        after_probe = LuantiState(20, 20, -1, -5, 3, 5, 0, False, False, False, False, False, "east", False)
        self.assertEqual(after_probe.oracle_direction(), "north")
        reached = LuantiState(20, 20, 3, 5, 3, 5, 0, True, False, False, False, False, "south", False)
        self.assertIsNone(reached.oracle_direction())

    def test_padded_histories_preserve_fixed_shape_and_action_labels(self):
        states = [
            LuantiState(20, 20, 0, -5, -3, 5, 1, False, False, False, False, False, None, False),
            LuantiState(20, 19, 1, -5, -3, 5, 1, False, False, False, False, False, "east", False),
            LuantiState(20, 19, -3, 5, -3, 5, 1, True, False, False, False, False, "south", False),
        ]
        windows, labels = padded_histories(states, history=3)
        self.assertEqual(tuple(windows.shape), (2, 3, 16))
        self.assertEqual(labels.tolist(), [DIRECTIONS.index("east"), DIRECTIONS.index("south")])
        self.assertEqual(windows[0, 0].tolist(), windows[0, 1].tolist())

    def test_padded_histories_can_omit_an_environment_only_transition_label(self):
        states = [
            LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False),
            LuantiState(20, 20, -1, -5, 3, 5, 0, False, False, False, False, False, "east", False),
            LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, "east", False),
        ]
        _windows, labels = padded_histories(states, history=3, skip_label_indices=frozenset({1}))
        self.assertEqual(labels.tolist(), [DIRECTIONS.index("east"), DIRECTIONS.index("north")])

    def test_policy_uses_its_fixed_training_context_at_inference(self):
        class CapturingCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(()))
                self.last_shape = None

            def forward(self, observations, *, think_on_last=True):
                self.last_shape = tuple(observations.shape)
                return torch.zeros((1, len(DIRECTIONS))), torch.zeros((1, observations.shape[1])), None

        model = GoalDirectedPLWM(history_length=4)
        capturing_core = CapturingCore()
        model.core = capturing_core
        state = LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False)
        self.assertEqual(model.choose([state]), "north")
        self.assertEqual(capturing_core.last_shape, (1, 4, 16))

    def test_policy_session_retains_state_when_a_new_observation_arrives(self):
        model = GoalDirectedPLWM(history_length=2)
        initial = LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False)
        after_probe = LuantiState(20, 20, -1, -5, 3, 5, 0, False, False, False, False, False, "east", False)
        session = model.begin_episode(initial)
        before = session.cognitive_state.belief.clone()
        session.observe(after_probe)
        self.assertFalse(torch.equal(before, session.cognitive_state.belief))

    def test_causal_control_model_updates_from_a_probe_without_hidden_seed(self):
        initial = LuantiState(20, 20, 0, -5, 3, 5, 0, False, False, False, False, False, None, False)
        after_probe = LuantiState(20, 20, -1, -5, 3, 5, 0, False, False, False, False, False, "east", False)
        model = CausalControlWorldModel()
        self.assertEqual((model.predict(initial, "east").delta_x, model.predict(initial, "east").delta_z), (1, 0))
        model.observe(initial, "east", after_probe)
        predicted = model.predict(after_probe, "north")
        self.assertEqual((predicted.delta_x, predicted.delta_z, predicted.blocked), (0, 1, False))

    def test_causal_control_model_revises_a_control_rule_after_contradiction(self):
        initial = LuantiState(20, 20, 0, -5, 3, 5, 1, False, False, False, False, False, None, False)
        normal = LuantiState(20, 20, 1, -5, 3, 5, 1, False, False, False, False, False, "east", False)
        switched = LuantiState(20, 20, 0, -5, 3, 5, 1, False, False, False, False, False, "east", False)
        model = CausalControlWorldModel()
        model.observe(initial, "east", normal)
        self.assertEqual(model.predict(normal, "east").delta_x, 1)
        model.observe(normal, "east", switched)
        self.assertEqual(model.revision_count, 1)
        self.assertEqual(model.predict(switched, "east").delta_x, -1)

    def test_causal_control_snapshot_restores_verified_mapping_and_revision_count(self):
        initial = LuantiState(20, 20, 0, -5, 3, 5, 1, False, False, False, False, False, None, False)
        normal = LuantiState(20, 20, 1, -5, 3, 5, 1, False, False, False, False, False, "east", False)
        switched = LuantiState(20, 20, 0, -5, 3, 5, 1, False, False, False, False, False, "east", False)
        model = CausalControlWorldModel()
        model.observe(initial, "east", normal)
        model.observe(normal, "east", switched)
        restored = CausalControlWorldModel.restore(model.snapshot())
        self.assertEqual(restored.revision_count, 1)
        self.assertTrue(restored.inferred_inversion)

    def test_control_mechanism_memory_scopes_verified_rules_by_world(self):
        memory = ControlMechanismMemory()
        inverted = memory.begin("world-a")
        inverted.seed_mapping(True)
        memory.commit("world-a", inverted)
        normal = memory.begin("world-b")
        normal.seed_mapping(False)
        memory.commit("world-b", normal)
        self.assertTrue(memory.mapping_for("world-a"))
        self.assertFalse(memory.mapping_for("world-b"))
        self.assertTrue(memory.begin("world-a").inferred_inversion)

    def test_policy_baselines_produce_four_action_logits(self):
        observations = torch.zeros((2, 4, 16))
        for policy in (NoMemoryGoalPolicy(), MaskedSensorGoalPolicy(history_length=4), GRUContextGoalPolicy(history_length=4)):
            logits, surprise = policy(observations)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertEqual(tuple(surprise.shape), (2, 4))

    def test_object_task_state_preserves_long_horizon_collection_facts(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 2_000_000, "completed": False,
            "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
            "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": 3, "key_z": -3, "decoy_x": -3, "decoy_z": -2,
            "key_collected": False, "decoy_collected": False,
        }
        state = ObjectTaskState.from_payload(payload)
        self.assertEqual(len(state.features()), 22)
        self.assertEqual(len(state.memory_features()), 23)
        self.assertEqual(state.oracle_direction(), "east")
        self.assertEqual(state.oracle_actual_direction(), "east")

    def test_object_task_state_accepts_public_indexed_multi_object_positions(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 5_000_000, "completed": False, "failed": False,
            "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
            "last_direction": None, "last_blocked": False, "object_mode": True,
            "key_x": -5, "key_z": -5, "decoy_x": -5, "decoy_z": -3,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
            "candidate_positions": [
                {"index": 1, "x": -5, "z": -3}, {"index": 2, "x": -2, "z": -3},
                {"index": 3, "x": 2, "z": -3}, {"index": 4, "x": 5, "z": -3},
            ],
        }
        state = ObjectTaskState.from_payload(payload)
        self.assertEqual(state.candidate_positions, ((-5, -3), (-2, -3), (2, -3), (5, -3)))
        self.assertFalse(state.base.failed)
        with self.assertRaises(ValueError):
            ObjectTaskState.from_payload({**payload, "candidate_positions": [{"index": 2, "x": -5, "z": -5}]})

    def test_object_policy_baselines_produce_four_action_logits(self):
        observations = torch.zeros((2, 4, 22))
        for policy in (NoMemoryObjectPolicy(), GRUObjectPolicy(history_length=4), TransformerObjectPolicy(history_length=4)):
            logits, surprise = policy(observations)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertEqual(tuple(surprise.shape), (2, 4))

    def test_object_visual_history_uses_actual_direction_labels(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 2_500_000, "completed": False,
            "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
            "last_direction": None, "last_blocked": False, "object_mode": True,
            "key_x": 3, "key_z": -3, "decoy_x": -3, "decoy_z": -2,
            "key_collected": False, "decoy_collected": False,
        }
        state = ObjectTaskState.from_payload(payload)
        vision = {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": ["air"] * 60 + ["agent"] + ["air"] * 60}
        windows, labels = padded_object_visual_histories([state], [vision], history=4, actual_direction_labels=True)
        self.assertEqual(tuple(windows.shape), (1, 4, VisionObjectGoalDirectedPLWM.input_dim))
        self.assertEqual(labels.tolist(), [DIRECTIONS.index("east")])

    def test_no_memory_visual_object_baseline_produces_actions(self):
        observations = torch.zeros((2, 4, VisionObjectGoalDirectedPLWM.input_dim))
        logits, surprise = NoMemoryVisionObjectPolicy()(observations)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 4))

    def test_reward_memory_visual_object_policies_produce_actions(self):
        observations = torch.zeros((2, 16, RewardMemoryVisionObjectPLWM.input_dim))
        for policy in (RewardMemoryVisionObjectPLWM(), NoMemoryRewardVisionObjectPolicy()):
            logits, surprise = policy(observations)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertEqual(tuple(surprise.shape), (2, 16))

    def test_verified_object_fact_memory_latches_delayed_facts_and_restores(self):
        memory = VerifiedObjectFactMemory()
        memory.observe_reward({"value": -0.01})
        self.assertFalse(memory.key_verified)
        self.assertFalse(memory.charge_verified)
        memory.observe_reward({"value": 0.15, "_plwm_provenance": {
            "episode_id": "heldout-7", "tick": 42, "source": "luanti-reward",
            "request_id": "request-42", "fact_schema_version": 1,
        }})
        self.assertTrue(memory.charge_verified)
        self.assertEqual(memory.charge_mode, "key")
        self.assertEqual(memory.charge_choice, 1)
        memory.observe_reward({"value": 0.2})
        restored = VerifiedObjectFactMemory.restore(memory.snapshot())
        self.assertTrue(restored.key_verified)
        self.assertTrue(restored.charge_verified)
        self.assertEqual(restored.charge_mode, "key")
        self.assertEqual(len(restored.provenance), 2)
        self.assertEqual(restored.provenance[0].episode_id, "heldout-7")
        self.assertEqual(restored.provenance[0].tick, 42)
        self.assertEqual(restored.provenance[0].request_id, "request-42")
        old_restored = VerifiedObjectFactMemory.restore({"key_verified": True})
        self.assertTrue(old_restored.key_verified)
        self.assertFalse(old_restored.charge_verified)
        self.assertIsNone(old_restored.charge_mode)
        memory = VerifiedObjectFactMemory()
        memory.observe_reward({"value": 0.18})
        self.assertTrue(memory.charge_verified)
        self.assertIsNone(memory.charge_mode)
        self.assertEqual(memory.charge_choice, 4)
        self.assertEqual(VerifiedObjectFactMemory.restore(memory.snapshot()).charge_choice, 4)
        payload = {
            "object_mode": True, "key_x": 1, "key_z": 2, "decoy_x": -1, "decoy_z": -2,
            "key_collected": False, "decoy_collected": False, "candidate_positions": None,
            "x": 0, "z": 0, "goal_x": 3, "goal_z": 3, "task_seed": 3_000_000,
            "completed": False, "failed": False,
            "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
            "last_direction": None, "last_blocked": False,
        }
        self.assertEqual(ObjectTaskState.from_payload(payload).candidate_positions, ())
        observations = torch.zeros((2, 16, FactMemoryVisionObjectPLWM.input_dim))
        logits, surprise = FactMemoryVisionObjectPLWM()(observations)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 16))
        logits, surprise = FactMemoryCurrentVisionObjectPolicy()(observations)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 16))

    def test_two_fact_memory_policy_uses_two_verified_inputs(self):
        observations = torch.zeros((2, 1, TwoFactMemoryCurrentVisionObjectPolicy.input_dim))
        logits, surprise = TwoFactMemoryCurrentVisionObjectPolicy()(observations)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 1))
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 3_000_000, "completed": False, "blocked_north": False, "blocked_south": False,
            "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": -3, "key_z": -3, "decoy_x": -3, "decoy_z": 3,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
        }
        state = ObjectTaskState.from_payload(payload)
        features = object_two_fact_memory_visual_frame_features(
            state,
            {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": ["air"] * 60 + ["agent"] + ["air"] * 60},
            verified_key=True,
            verified_charge=True,
        )
        self.assertEqual(len(features), TwoFactMemoryCurrentVisionObjectPolicy.input_dim)

    def test_polarity_fact_memory_policy_uses_reward_verified_mode(self):
        observations = torch.zeros((2, 1, PolarityFactMemoryCurrentVisionObjectPolicy.input_dim))
        logits, surprise = PolarityFactMemoryCurrentVisionObjectPolicy()(observations)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 1))
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 4_000_000, "completed": False, "blocked_north": False, "blocked_south": False,
            "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": -3, "key_z": -3, "decoy_x": -3, "decoy_z": -5,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
        }
        state = ObjectTaskState.from_payload(payload)
        vision = {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": ["air"] * 60 + ["agent"] + ["air"] * 60}
        features = object_polarity_fact_memory_visual_frame_features(
            state, vision, verified_key=False, charge_mode="decoy"
        )
        self.assertEqual(len(features), PolarityFactMemoryCurrentVisionObjectPolicy.input_dim)
        with self.assertRaises(ValueError):
            object_polarity_fact_memory_visual_frame_features(state, vision, verified_key=False, charge_mode="invalid")

    def test_multi_choice_fact_memory_policy_requires_indexed_candidates(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 5_000_000, "completed": False, "failed": False,
            "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
            "last_direction": None, "last_blocked": False, "object_mode": True,
            "key_x": -5, "key_z": -5, "decoy_x": -5, "decoy_z": -3,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
            "candidate_positions": [
                {"index": 1, "x": -5, "z": -3}, {"index": 2, "x": -2, "z": -3},
                {"index": 3, "x": 2, "z": -3}, {"index": 4, "x": 5, "z": -3},
            ],
        }
        state = ObjectTaskState.from_payload(payload)
        vision = {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": ["air"] * 60 + ["agent"] + ["air"] * 60}
        self.assertEqual(
            len(object_multi_choice_public_visual_frame_features(state, vision)),
            23 + 8 + 11 * 11 * len(VISION_TILES),
        )
        features = object_multi_choice_fact_memory_visual_frame_features(state, vision, verified_key=False, charge_choice=4)
        self.assertEqual(len(features), MultiChoiceFactMemoryCurrentVisionObjectPolicy.input_dim)
        logits, surprise = MultiChoiceFactMemoryCurrentVisionObjectPolicy()(torch.zeros((2, 1, len(features))))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(surprise.shape), (2, 1))
        with self.assertRaises(ValueError):
            object_multi_choice_fact_memory_visual_frame_features(state, vision, verified_key=False, charge_choice=5)

    def test_verified_fact_planner_switches_target_without_hidden_seed(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 2_700_000, "completed": False, "blocked_north": False, "blocked_south": False,
            "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": 3, "key_z": -3, "decoy_x": -3, "decoy_z": -2,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
        }
        state = ObjectTaskState.from_payload(payload)
        planner = VerifiedFactObjectPlanner()
        self.assertEqual(planner.choose_actual(state, verified_key=False), "east")
        at_key = ObjectTaskState.from_payload({**payload, "x": 3, "z": -3})
        self.assertEqual(planner.choose_actual(at_key, verified_key=True), "south")

    def test_verified_fact_planner_requires_charge_then_key_then_goal(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 3_000_000, "completed": False, "blocked_north": False, "blocked_south": False,
            "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": -3, "key_z": -3, "decoy_x": -3, "decoy_z": 3,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
        }
        state = ObjectTaskState.from_payload(payload)
        planner = VerifiedFactObjectPlanner()
        self.assertEqual(planner.choose_actual(state, verified_key=False, verified_charge=False), "east")
        self.assertEqual(planner.choose_actual(state, verified_key=False, verified_charge=True), "west")
        self.assertEqual(planner.choose_actual(state, verified_key=True, verified_charge=True), "east")
        decoy_first = ObjectTaskState.from_payload({**payload, "decoy_x": 3, "decoy_z": -3})
        self.assertEqual(
            planner.choose_actual(decoy_first, verified_key=False, verified_charge=True, charge_mode="decoy"), "east"
        )
        multi = ObjectTaskState.from_payload({
            **payload, "task_seed": 5_000_000,
            "candidate_positions": [
                {"index": 1, "x": -5, "z": -3}, {"index": 2, "x": -2, "z": -3},
                {"index": 3, "x": 2, "z": -3}, {"index": 4, "x": 5, "z": -3},
            ],
        })
        self.assertEqual(planner.choose_actual(multi, verified_key=False, verified_charge=True, charge_choice=3), "east")
        with self.assertRaises(ValueError):
            planner.choose_actual(multi, verified_key=False, verified_charge=True)

    def test_live_interaction_contract_keeps_vision_separate_from_state(self):
        interaction = LuantiInteraction("r", {"status": "accepted"}, {"x": 0}, {"cells": ["agent"]}, None)
        self.assertEqual(interaction.observation["x"], 0)
        self.assertEqual(interaction.vision["cells"], ["agent"])

    def test_object_observe_wrapper_requests_a_fresh_observation_without_direction(self):
        payload = {
            "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
            "task_seed": 4_000_000, "completed": False, "blocked_north": False, "blocked_south": False,
            "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False,
            "object_mode": True, "key_x": -3, "key_z": -3, "decoy_x": -3, "decoy_z": -5,
            "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
        }
        vision = {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": ["air"] * 60 + ["agent"] + ["air"] * 60}
        client = object.__new__(LuantiLiveClient)
        captured = []

        def submit(actions):
            captured.extend(actions)
            return {"observe": LuantiInteraction("observe", {"status": "accepted"}, payload, vision, None)}, []

        client.submit_batch = submit  # type: ignore[method-assign]
        interaction, state, received_vision = client.observe_object_with_vision()
        self.assertEqual(interaction.outcome["status"], "accepted")
        self.assertEqual(captured, [{"name": "observe", "expect_observation": True}])
        self.assertEqual(state.base.x, 0)
        self.assertEqual(received_vision, vision)

    def test_vision_encoder_requires_a_valid_single_agent_grid(self):
        payload = {"encoding": "tile-grid-v1", "width": 3, "height": 3, "cells": ["air"] * 4 + ["agent"] + ["wall"] * 4}
        encoded = encode_vision(payload)
        self.assertEqual(len(encoded), 9 * len(VISION_TILES))
        self.assertEqual(sum(encoded), 9.0)
        with self.assertRaises(ValueError):
            encode_vision({**payload, "cells": ["air"] * 9})

    def test_visual_noise_augmentation_preserves_shape_and_agent_channel(self):
        observations = torch.zeros((2, 4, VisionGoalDirectedPLWM.input_dim))
        visual = observations[:, :, 16:].reshape(2, 4, 121, len(VISION_TILES))
        visual[..., 2] = 1.0  # one-hot agent tile at every location for this contract test
        augmented = augment_visual_observations(observations, flip_rate=1.0, generator=torch.Generator().manual_seed(7))
        self.assertEqual(tuple(augmented.shape), tuple(observations.shape))
        augmented_visual = augmented[:, :, 16:].reshape(2, 4, 121, len(VISION_TILES))
        self.assertTrue(torch.equal(augmented_visual[..., 2], visual[..., 2]))

    def test_visual_noise_augmentation_rejects_invalid_rate(self):
        observations = torch.zeros((1, 4, VisionGoalDirectedPLWM.input_dim))
        with self.assertRaises(ValueError):
            augment_visual_observations(observations, flip_rate=1.1)

    def test_visual_sensor_reconciliation_changes_only_immediate_conflict(self):
        state = LuantiState(20, 20, 0, -5, 3, 5, 0, False, True, False, False, False, None, False)
        cells = ["air"] * 121
        cells[60] = "agent"
        cells[49] = "air"  # north conflicts with the public blocked sensor
        cells[61] = "wall"  # east conflicts with the public unblocked sensor
        corrected, corrections = reconcile_visual_sensors(
            state, {"encoding": "tile-grid-v1", "width": 11, "height": 11, "cells": cells}
        )
        self.assertEqual(corrections, 2)
        self.assertEqual(corrected["cells"][49], "wall")
        self.assertEqual(corrected["cells"][61], "air")
        self.assertEqual(corrected["cells"][0], "air")

    def test_temporal_consensus_aligns_previous_view_by_observed_displacement(self):
        previous = LuantiState(20, 20, 0, 0, 3, 5, 0, False, False, False, False, False, "east", False)
        current = LuantiState(20, 20, 1, 0, 3, 5, 0, False, False, False, False, False, "east", False)
        previous_cells = ["air"] * 25
        previous_cells[12] = "agent"
        # A wall two cells east becomes one cell east after the body moves east.
        previous_cells[14] = "wall"
        current_cells = ["air"] * 25
        current_cells[12] = "agent"
        current_cells[13] = "air"  # noisy conflict with the aligned prior frame
        previous_vision = {"encoding": "tile-grid-v1", "width": 5, "height": 5, "cells": previous_cells}
        current_vision = {"encoding": "tile-grid-v1", "width": 5, "height": 5, "cells": current_cells}
        fused, conflicts = temporal_consensus_vision(previous, previous_vision, current, current_vision)
        self.assertGreaterEqual(conflicts, 1)
        self.assertEqual(fused["cells"][13], "wall")


if __name__ == "__main__":
    unittest.main()
