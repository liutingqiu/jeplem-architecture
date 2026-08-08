"""Independently validate the gated Luanti reward-only DQN smoke/formal report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def audit(path: Path, *, formal: bool = False) -> Audit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gated DQN report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    if not isinstance(protocol, str) or "real Luanti gated delayed-goal course" not in protocol:
        failures.append("missing real-Luanti gated-course declaration")
    for boundary in ("reward-only", "11x11 public vision", "previous chosen action", "previous scalar reward", "legal-action masks", "no cue labels", "symbolic bridge state", "teacher trajectory", "external planner", "shaped reward"):
        if boundary not in protocol:
            failures.append(f"missing declared boundary: {boundary}")
    training = payload.get("training_seed_range")
    heldout = payload.get("heldout_seed_range")
    if not isinstance(training, list) or len(training) < 128 or any(not isinstance(seed, int) or not 9_000_000 <= seed < 10_000_000 for seed in training):
        failures.append("invalid gated DQN training seeds")
    if not isinstance(heldout, list) or len(heldout) < 16 or any(not isinstance(seed, int) or not 9_000_000 <= seed < 10_000_000 for seed in heldout):
        failures.append("invalid gated DQN held-out seeds")
    if isinstance(training, list) and isinstance(heldout, list) and set(training).intersection(heldout):
        failures.append("training and held-out seeds overlap")
    if formal:
        if training != list(range(9_500_000, 9_500_512)):
            failures.append("training seeds differ from frozen formal DQN block")
        if heldout != list(range(9_600_000, 9_600_064)):
            failures.append("held-out seeds differ from frozen formal DQN block")
        if payload.get("action_seed") != 2_042_337:
            failures.append("action seed differs from frozen formal DQN block")
    if payload.get("transformer_context") != 2:
        failures.append("Transformer context differs from declared gated-course bound")
    results = payload.get("results")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(results, list):
        failures.append("missing model results")
    else:
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED_MODELS:
            failures.append("model set differs from frozen comparison")
    heldout_episodes = len(heldout) if isinstance(heldout, list) else 0
    for name, result in by_name.items():
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("episodes") != heldout_episodes:
            failures.append(f"invalid {name} held-out accounting")
            continue
        successes = evaluation.get("successes")
        counts = evaluation.get("action_counts")
        if not isinstance(successes, int) or not 0 <= successes <= heldout_episodes:
            failures.append(f"invalid {name} success count")
        if not isinstance(counts, list) or len(counts) != 4 or any(not isinstance(value, int) or value < 0 for value in counts):
            failures.append(f"invalid {name} action accounting")
    random_evaluation = payload.get("random_evaluation")
    if not isinstance(random_evaluation, dict) or random_evaluation.get("episodes") != heldout_episodes:
        failures.append("invalid random held-out accounting")
    recomputed = False
    if not failures:
        recomputed = all(by_name[name]["evaluation"]["successes"] / heldout_episodes >= 0.75 for name in ("gru", "transformer_2", "plwm"))
        if payload.get("accepted_neural_task_diagnostic") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed gate")
    return Audit(
        "luanti_gated_memory_dqn_audit_v1: independently validates the fixed reward-only DQN task diagnostic; never an architecture acceptance audit.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path, *, formal: bool = False) -> Audit:
    outcome = audit(source, formal=formal)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-gated-memory-dqn-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-gated-memory-dqn-audit.json")
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(asdict(main(arguments.source, arguments.artifact, formal=arguments.formal)), indent=2))
