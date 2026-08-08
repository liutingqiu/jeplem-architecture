"""Independently validate the public-feature micro-navigation Q-learning report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINING_EPISODES = 128
EVALUATION_EPISODES = 16
TRAINING_SEED_START = 6_300_000
HELDOUT_SEED_START = 6_400_000


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def audit(path: Path) -> Audit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Q-learning report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    if not isinstance(protocol, str) or "real Luanti visible-goal micro-navigation" not in protocol:
        failures.append("missing real-Luanti micro-navigation declaration")
    for boundary in ("no bridge state", "teacher trajectory", "external planner", "PLWM"):
        if boundary not in protocol:
            failures.append(f"missing declared boundary: {boundary}")
    if payload.get("executor") != "local-cpu":
        failures.append("baseline executor is not declared local-cpu")
    if payload.get("training_episodes") != TRAINING_EPISODES or payload.get("max_steps") != 10:
        failures.append("training budget differs from frozen baseline")
    expected_train = list(range(TRAINING_SEED_START, TRAINING_SEED_START + TRAINING_EPISODES))
    expected_heldout = list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + EVALUATION_EPISODES))
    if payload.get("training_seed_range") != expected_train:
        failures.append("training seed split differs from frozen baseline")
    if payload.get("heldout_seed_range") != expected_heldout:
        failures.append("held-out seed split differs from frozen baseline")
    if set(payload.get("training_seed_range", [])).intersection(payload.get("heldout_seed_range", [])):
        failures.append("training and held-out seeds overlap")

    evaluations: dict[str, dict[str, Any]] = {}
    for name in ("random_evaluation", "q_learning_evaluation"):
        evaluation = payload.get(name)
        if not isinstance(evaluation, dict):
            failures.append(f"missing {name}")
            continue
        evaluations[name] = evaluation
        successes = evaluation.get("successes")
        action_counts = evaluation.get("action_counts")
        if evaluation.get("episodes") != EVALUATION_EPISODES or not isinstance(successes, int) or not 0 <= successes <= EVALUATION_EPISODES:
            failures.append(f"invalid {name} episode accounting")
        if not isinstance(action_counts, list) or len(action_counts) != 4 or any(not isinstance(value, int) or value < 0 for value in action_counts):
            failures.append(f"invalid {name} action accounting")
    recomputed = False
    if not failures and len(evaluations) == 2:
        random_successes = evaluations["random_evaluation"]["successes"]
        q_successes = evaluations["q_learning_evaluation"]["successes"]
        recomputed = q_successes >= 12 and q_successes >= random_successes + 4
        if payload.get("accepted_public_feature_baseline") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed gate")
    elif payload.get("accepted_public_feature_baseline") is True:
        failures.append("incomplete report cannot be accepted")
    return Audit(
        "luanti_micro_navigation_qlearning_audit_v1: independently validates the fixed public-feature standard baseline split and its held-out gate.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-micro-navigation-qlearning-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-micro-navigation-qlearning-audit.json")
    arguments = parser.parse_args()
    print(json.dumps(asdict(main(arguments.source, arguments.artifact)), indent=2))
