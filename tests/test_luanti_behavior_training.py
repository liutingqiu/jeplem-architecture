import unittest

import torch

from experiments.train_luanti_behavior import (
    ExecutedAction,
    Trajectory,
    evaluate_action_world_model,
    training_tensors,
    transition_tensors,
)


class LuantiBehaviorTrainingTests(unittest.TestCase):
    def test_training_tensors_use_real_state_schema_and_skip_terminal_labels(self):
        trajectory = Trajectory(
            episode_id="t1",
            task_seed=1,
            states=[
                {"health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5, "task_seed": 1, "completed": False, "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False},
                {"health": 20.0, "energy": 19.95, "x": -1, "z": -5, "goal_x": 3, "goal_z": 5, "task_seed": 1, "completed": True, "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False, "last_direction": "east", "last_blocked": False},
            ],
        )
        observations, labels = training_tensors([trajectory])
        self.assertEqual(tuple(observations.shape), (1, 4, 16))
        self.assertEqual(labels.tolist(), [2])

    def test_transition_tensors_keep_requested_action_and_its_observed_effect_together(self):
        trajectory = Trajectory(
            episode_id="t2",
            task_seed=0,
            states=[
                {"health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5, "task_seed": 0, "completed": False, "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False, "last_direction": None, "last_blocked": False},
                {"health": 20.0, "energy": 19.95, "x": -1, "z": -5, "goal_x": 3, "goal_z": 5, "task_seed": 0, "completed": False, "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False, "last_direction": "east", "last_blocked": False},
            ],
            actions=[ExecutedAction("east", "r1", "accepted", -0.01, False)],
        )
        histories, actions, outcomes = transition_tensors([trajectory])
        self.assertEqual(tuple(histories.shape), (1, 4, 16))
        self.assertEqual(actions.tolist(), [2])
        self.assertEqual(outcomes.tolist(), [[-1.0, 0.0, 0.0]])

    def test_world_model_evaluation_decodes_discrete_displacements(self):
        class FixedWorldModel(torch.nn.Module):
            def forward(self, histories, actions):
                batch = histories.shape[0]
                return (
                    torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch, 1),
                    torch.tensor([[0.0, 1.0, 0.0]]).repeat(batch, 1),
                    torch.tensor([-2.0]).repeat(batch),
                    torch.zeros((batch, histories.shape[1])),
                )

        evaluation = evaluate_action_world_model(
            FixedWorldModel(),
            torch.zeros((1, 4, 16)),
            torch.tensor([2]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            device=torch.device("cpu"),
        )
        self.assertEqual(evaluation.displacement_accuracy, 1.0)
        self.assertEqual(evaluation.blocked_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
