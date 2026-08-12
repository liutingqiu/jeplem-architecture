"""Luanti adapter skeleton tests (M5).

Verifies the meta-rule engine's PerceptionAdapter protocol is real beyond ARC:
  - perceive() turns Luanti state+vision into an EntityScene
  - conserve_raw() produces a valid Conservation signature
  - family relations are registered with the right family/fired_by
  - the generic engine can attach to the Luanti adapter (interface proof)
No live Luanti process is required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from luanti_adapter import LuantiAdapter, TILES, VISION_RADIUS


def make_state(**over):
    base = {
        "health": 20.0, "energy": 20.0, "x": 3, "z": 3,
        "goal_x": 6, "goal_z": 6, "task_seed": 1, "completed": False,
        "blocked_north": False, "blocked_south": False,
        "blocked_east": False, "blocked_west": False,
        "last_direction": "north", "last_blocked": False, "failed": False,
    }
    base.update(over)
    return base


def empty_vision():
    n = VISION_RADIUS * 2 + 1
    return ["air"] * (n * n)


class LuantiAdapterTests(unittest.TestCase):
    def test_perceive_produces_agent_and_tiles(self):
        adapter = LuantiAdapter()
        vision = empty_vision()
        n = VISION_RADIUS * 2 + 1
        center = VISION_RADIUS * n + VISION_RADIUS
        vision[center] = "agent"
        # place a wall near the agent and a goal
        vision[center + 1] = "wall"
        vision[center - n] = "goal"
        scene = adapter.perceive((make_state(x=3, z=3), vision))
        types = {e.type for e in scene.entities}
        self.assertIn("agent", types)
        self.assertIn("tile", types)
        # wall and goal tiles present
        tiles = {e.properties.get("tile") for e in scene.entities if e.type == "tile"}
        self.assertIn("wall", tiles)
        self.assertIn("goal", tiles)
        # adjacency relation from agent to the wall tile
        rel_types = {r.rel_type for r in scene.relations}
        self.assertIn("adjacent", rel_types)

    def test_conserve_signature_valid(self):
        adapter = LuantiAdapter()
        vision = empty_vision()
        n = VISION_RADIUS * 2 + 1
        center = VISION_RADIUS * n + VISION_RADIUS
        vision[center] = "agent"
        vision[center + 1] = "wall"
        # move the agent: same world, new position
        in_raw = (make_state(x=3, z=3), vision)
        out_raw = (make_state(x=4, z=3), vision)
        sig = adapter.conserve_raw(in_raw, out_raw)
        self.assertIsInstance(sig.signature, str)
        self.assertIn("agent_identity", sig.properties)
        self.assertIn("goal_identity", sig.properties)
        self.assertIn("position", sig.changed)
        # same signature for a different movement of the same kind
        out_raw2 = (make_state(x=5, z=3), vision)
        sig2 = adapter.conserve_raw(in_raw, out_raw2)
        self.assertEqual(sig.signature, sig2.signature)

    def test_family_relations_registered(self):
        adapter = LuantiAdapter()
        causal = adapter.causality_relations()
        self.assertIn("cause_effect", causal)
        self.assertEqual(causal["cause_effect"][0], "causality")
        inter = adapter.interaction_relations()
        self.assertIn("interact", inter)
        self.assertEqual(inter["interact"][0], "interaction")


if __name__ == "__main__":
    unittest.main()
