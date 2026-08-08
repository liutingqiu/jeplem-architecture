import unittest

from experiments.scale_luanti_polarity_delay import _is_polarity_only_world, _select_polarity_only_seeds
from plwm.luanti_training import ObjectTaskState


def _state(*, candidates: bool) -> ObjectTaskState:
    payload = {
        "health": 20.0, "energy": 20.0, "x": 0, "z": -5, "goal_x": 3, "goal_z": 5,
        "task_seed": 1, "completed": False, "failed": False,
        "blocked_north": False, "blocked_south": False, "blocked_east": False, "blocked_west": False,
        "last_direction": None, "last_blocked": False, "object_mode": True,
        "key_x": -3, "key_z": -3, "decoy_x": 3, "decoy_z": -3,
        "key_collected": False, "decoy_collected": False, "last_reward": 0.0,
    }
    if candidates:
        payload["candidate_positions"] = [
            {"index": 1, "x": -5, "z": -3}, {"index": 2, "x": -2, "z": -3},
            {"index": 3, "x": 2, "z": -3}, {"index": 4, "x": 5, "z": -3},
        ]
    return ObjectTaskState.from_payload(payload)


class _ScreeningClient:
    def __init__(self, multi_seeds: set[int]) -> None:
        self.multi_seeds = multi_seeds
        self.calls: list[tuple[str, int]] = []

    def begin_object_episode_with_vision(self, episode_id: str, seed: int):
        self.calls.append((episode_id, seed))
        return _state(candidates=seed in self.multi_seeds), {"cells": []}


class PolarityDelaySelectionTests(unittest.TestCase):
    def test_public_layout_screen_excludes_only_multi_choice_worlds(self):
        client = _ScreeningClient({101, 103})
        selected = _select_polarity_only_seeds(
            client, seed_start=100, count=3, stride=1, episode_prefix="test",
        )
        self.assertEqual(selected, [100, 102, 104])
        self.assertEqual([seed for _episode, seed in client.calls], [100, 101, 102, 103, 104])

    def test_layout_predicate_uses_candidate_positions(self):
        self.assertTrue(_is_polarity_only_world(_state(candidates=False)))
        self.assertFalse(_is_polarity_only_world(_state(candidates=True)))

    def test_invalid_selection_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            _select_polarity_only_seeds(_ScreeningClient(set()), seed_start=1, count=0, stride=1, episode_prefix="test")


if __name__ == "__main__":
    unittest.main()
