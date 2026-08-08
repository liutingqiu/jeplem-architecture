"""Audit the public-vision contract for Luanti addressed retrieval.

This inspection does not train or evaluate a controller. It reads only the
same 11x11 tile tensor exposed to a reward-only learner and verifies that the
course actually presents four initial identities, five cue-free forced moves,
and one later identity-only query.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from experiments.audit_luanti_delayed_goal_qlearning import public_valid_actions
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE
from plwm.luanti_training import VISION_TILES


ROOT = Path(__file__).resolve().parents[1]
COURSE_MINIMUM = 12_000_000
COURSE_MAXIMUM = 13_000_000
IDENTITIES = ("goal", "key", "decoy", "beacon")
SOUTH = 1
EAST_WEST = {2, 3}


@dataclass(frozen=True)
class VisualContractAudit:
    protocol: str
    seed: int
    initial_identity_counts: dict[str, int]
    intermediate_identity_counts: list[dict[str, int]]
    query_identity_counts: dict[str, int]
    forced_action_sets: list[list[int]]
    final_action_set: list[int]
    accepted: bool
    failures: list[str]
    artifact: str


def identity_counts(observation: np.ndarray) -> dict[str, int]:
    if not isinstance(observation, np.ndarray) or observation.shape != VISION_SHAPE or observation.dtype != np.uint8:
        raise ValueError("expected one public Luanti vision tensor")
    return {identity: int(observation[..., VISION_TILES.index(identity)].sum()) for identity in IDENTITIES}


def audit(*, seed: int, bridge_root: Path = DEFAULT_BRIDGE_ROOT, artifact: Path | None = None) -> VisualContractAudit:
    if not COURSE_MINIMUM <= seed < COURSE_MAXIMUM:
        raise ValueError("addressed visual audit seed must use the addressed course")
    environment = GymVisionOnlyLuanti(seed=seed, max_steps=6, bridge_root=bridge_root)
    failures: list[str] = []
    try:
        observation, info = environment.reset(seed=seed)
        if info != {}:
            failures.append("reset exposed non-public info")
        initial = identity_counts(observation)
        forced_action_sets: list[list[int]] = []
        intermediate: list[dict[str, int]] = []
        for move in range(5):
            allowed = sorted(public_valid_actions(observation))
            forced_action_sets.append(allowed)
            if allowed != [SOUTH]:
                failures.append(f"forced move {move + 1} did not expose only south")
            observation, _reward, terminated, truncated, info = environment.step(SOUTH)
            if info != {}:
                failures.append(f"forced move {move + 1} exposed non-public info")
            if terminated or truncated:
                failures.append(f"forced move {move + 1} ended the addressed course")
                break
            counts = identity_counts(observation)
            if move < 4:
                intermediate.append(counts)
        query = identity_counts(observation)
        final_actions = sorted(public_valid_actions(observation))
    finally:
        environment.close()

    if initial != {identity: 1 for identity in IDENTITIES}:
        failures.append("initial public frame did not contain each addressed identity exactly once")
    if any(sum(counts.values()) != 0 for counts in intermediate):
        failures.append("an intermediate forced-move frame exposed a cue or query")
    if sum(query.values()) != 1:
        failures.append("post-delay public frame did not contain exactly one query identity")
    if set(final_actions) != EAST_WEST:
        failures.append("query decision did not expose exactly east/west")
    target = artifact or ROOT / "artifacts" / "luanti-addressed-visual-contract-audit.json"
    outcome = VisualContractAudit(
        "luanti_addressed_visual_contract_v1: audits only public 11x11 tiles and public legal actions; no symbolic state, seed answer, query answer, teacher, planner, or reward enters the inspection.",
        seed, initial, intermediate, query, forced_action_sets, final_actions, not failures, failures, str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12_400_000)
    parser.add_argument("--artifact", type=Path, default=None)
    arguments = parser.parse_args()
    print(json.dumps(asdict(audit(seed=arguments.seed, artifact=arguments.artifact)), indent=2))
