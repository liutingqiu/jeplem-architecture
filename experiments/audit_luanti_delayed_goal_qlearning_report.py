"""Independently validate the delayed-goal Q-learning task diagnostic report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINING_EPISODES = 128
EVALUATION_EPISODES = 16
TRAINING_SEED_START = 7_100_000
HELDOUT_SEED_START = 7_200_000


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def audit(path: Path, *, course: str = "long_delay") -> Audit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("delayed-goal report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    contracts = {
        "long_delay": ("long_delay", 7_100_000, 7_200_000, 10, EVALUATION_EPISODES, None),
        "compact_binary": ("compact_binary", 8_100_000, 8_200_000, 4, EVALUATION_EPISODES, None),
        "gated_compact_binary": ("gated_compact_binary", 9_100_000, 9_200_000, 4, EVALUATION_EPISODES, None),
        "gated_compact_binary_replication": ("gated_compact_binary_replication", 9_300_000, 9_400_000, 4, 64, 1_942_337),
    }
    if course not in contracts:
        raise ValueError("unsupported delayed-goal audit course")
    protocol_name, training_seed_start, heldout_seed_start, max_steps, evaluation_episodes, action_seed_base = contracts[course]
    if not isinstance(protocol, str) or "real Luanti delayed-goal micro-navigation" not in protocol or protocol_name not in protocol:
        failures.append("missing real-Luanti delayed-goal declaration")
    for boundary in ("Reactive and retained-cue", "no bridge state", "teacher trajectory", "external planner", "PLWM"):
        if boundary not in protocol:
            failures.append(f"missing declared boundary: {boundary}")
    if payload.get("executor") != "local-cpu":
        failures.append("diagnostic executor is not declared local-cpu")
    if payload.get("training_episodes") != TRAINING_EPISODES or payload.get("max_steps") != max_steps:
        failures.append("training budget differs from frozen delayed-goal contract")
    if payload.get("training_seed_range") != list(range(training_seed_start, training_seed_start + TRAINING_EPISODES)):
        failures.append("training seeds differ from frozen delayed-goal split")
    if payload.get("heldout_seed_range") != list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)):
        failures.append("held-out seeds differ from frozen delayed-goal split")
    if action_seed_base is not None and payload.get("action_seed_base") != action_seed_base:
        failures.append("action random seed differs from frozen replication contract")
    if set(payload.get("training_seed_range", [])).intersection(payload.get("heldout_seed_range", [])):
        failures.append("training and held-out seeds overlap")
    evaluations: dict[str, dict[str, Any]] = {}
    for name in ("random_evaluation", "reactive_q_evaluation", "retained_cue_q_evaluation"):
        evaluation = payload.get(name)
        if not isinstance(evaluation, dict):
            failures.append(f"missing {name}")
            continue
        evaluations[name] = evaluation
        successes = evaluation.get("successes")
        counts = evaluation.get("action_counts")
        if evaluation.get("episodes") != evaluation_episodes or not isinstance(successes, int) or not 0 <= successes <= evaluation_episodes:
            failures.append(f"invalid {name} episode accounting")
        if not isinstance(counts, list) or len(counts) != 4 or any(not isinstance(value, int) or value < 0 for value in counts):
            failures.append(f"invalid {name} action accounting")
    recomputed = False
    if not failures and len(evaluations) == 3:
        reactive = evaluations["reactive_q_evaluation"]["successes"]
        retained = evaluations["retained_cue_q_evaluation"]["successes"]
        recomputed = retained >= 12 and retained >= reactive + 4
        if payload.get("accepted_memory_diagnostic") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed gate")
    elif payload.get("accepted_memory_diagnostic") is True:
        failures.append("incomplete report cannot be accepted")
    return Audit(
        "luanti_delayed_goal_qlearning_audit_v1: independently validates the fixed delayed-goal public-memory task split and its reactive-versus-retained gate.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path, *, course: str = "long_delay") -> Audit:
    outcome = audit(source, course=course)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-delayed-goal-qlearning-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-delayed-goal-qlearning-audit.json")
    parser.add_argument("--course", choices=("long_delay", "compact_binary", "gated_compact_binary", "gated_compact_binary_replication"), default="long_delay")
    arguments = parser.parse_args()
    print(json.dumps(asdict(main(arguments.source, arguments.artifact, course=arguments.course)), indent=2))
