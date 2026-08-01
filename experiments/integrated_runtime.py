"""Exercise one PLWM runtime tick path across all persistent state layers."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.commitments import Commitment
from plwm.runtime import PersistentRuntime


@dataclass
class Report:
    action_under_uncertainty: str
    selected_before_completion: str | None
    completed_after_evidence: int
    selected_after_completion: str | None
    snapshot_has_all_layers: bool
    action_without_observation: str


def run() -> Report:
    runtime = PersistentRuntime(input_dim=2, belief_states=2)
    runtime.add_commitment(Commitment("reach-state-one", target_state=1, priority=0.9))
    runtime.add_commitment(Commitment("fallback", target_state=0, priority=0.2))
    observe_tick = runtime.ingest(None)
    weak = math.log(0.55)
    uncertain = runtime.ingest([0.1, 0.0], [math.log(0.45), weak])
    for _ in range(5):
        certain = runtime.ingest([0.9, 0.1], [math.log(0.05), math.log(0.95)])
    snapshot = runtime.snapshot()
    return Report(
        action_under_uncertainty=uncertain.action.value,
        selected_before_completion=uncertain.selected_commitment,
        completed_after_evidence=certain.completed_commitments,
        selected_after_completion=certain.selected_commitment,
        snapshot_has_all_layers={"latent_state", "belief_log_probs", "ledger"}.issubset(snapshot),
        action_without_observation=observe_tick.action.value,
    )


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not (
        outcome.action_under_uncertainty == "think"
        and outcome.selected_before_completion == "reach-state-one"
        and outcome.completed_after_evidence == 1
        and outcome.selected_after_completion == "fallback"
        and outcome.snapshot_has_all_layers
        and outcome.action_without_observation == "observe"
    ):
        raise SystemExit("integrated runtime gate failed")
