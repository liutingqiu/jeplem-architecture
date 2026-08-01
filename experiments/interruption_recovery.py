"""Structural benchmark for PLWM's interruption-and-recovery contract.

This is intentionally a state-continuity benchmark, not a language benchmark:
the same model instance must preserve an unresolved commitment through a
checkpoint/restart without replaying prior observations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tempfile import TemporaryDirectory

from plwm import CognitiveAction, PLWM, PLWMConfig


def run() -> dict[str, object]:
    config = PLWMConfig(input_dim=3, latent_dim=12, thought_steps=5, seed=19)
    model = PLWM(config)
    model.set_goal([1.0, 0.0, 0.0], commitment="return to the original objective")

    # A salient cue and unrelated work occur before the process disappears.
    cue = model.step([6.0, -4.0, 2.0], action=CognitiveAction.THINK)
    distractor = model.step([-1.0, 5.0, -3.0], action=CognitiveAction.THINK)

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "persistent-state.json"
        model.save(str(checkpoint))
        resumed = PLWM(config)
        resumed.load(str(checkpoint))

        restored_goal = list(resumed.state.goal)
        resumed_step = resumed.step(action=CognitiveAction.THINK)

    return {
        "checkpoint_recovery": {
            "commitment_preserved": resumed.state.pending_commitment == "return to the original objective",
            "goal_preserved": restored_goal == model.state.goal,
            "memory_preserved": len(resumed.state.memory) == len(model.state.memory),
            "time_advanced_after_restore": resumed_step.time == model.state.time + 1,
        },
        "before_interruption": [asdict(cue), asdict(distractor)],
        "after_restore": asdict(resumed_step),
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, default=str))
    if not all(report["checkpoint_recovery"].values()):
        raise SystemExit("interruption recovery contract failed")
