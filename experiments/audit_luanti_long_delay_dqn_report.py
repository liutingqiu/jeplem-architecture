"""Independently validate the frozen Luanti long-delay DQN task diagnostic."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
TRAINING_SEED_START = 10_500_000
TRAINING_EPISODES = 512
HELDOUT_SEED_START = 10_600_000
HELDOUT_EPISODES = 64
ACTION_SEED = 3_042_337


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
        raise ValueError("long-delay DQN report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    if not isinstance(protocol, str) or "real Luanti gated delayed-goal course" not in protocol:
        failures.append("missing real-Luanti gated-course declaration")
    for boundary in (
        "reward-only", "11x11 public vision", "previous chosen action",
        "previous scalar reward", "legal-action masks", "no cue labels",
        "symbolic bridge state", "teacher trajectory", "external planner", "shaped reward",
    ):
        if not isinstance(protocol, str) or boundary not in protocol:
            failures.append(f"missing declared boundary: {boundary}")
    if payload.get("course") != "long_gated":
        failures.append("report does not declare the frozen long-gated course")
    if payload.get("device") != "cuda":
        failures.append("formal result was not recorded on CUDA")
    if payload.get("training_seed_range") != list(range(TRAINING_SEED_START, TRAINING_SEED_START + TRAINING_EPISODES)):
        failures.append("training seeds differ from frozen long-delay block")
    if payload.get("heldout_seed_range") != list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + HELDOUT_EPISODES)):
        failures.append("held-out seeds differ from frozen long-delay block")
    if payload.get("action_seed") != ACTION_SEED:
        failures.append("action seed differs from frozen long-delay block")
    if payload.get("max_steps") != 8:
        failures.append("long-delay horizon differs from the frozen eight-step contract")
    if payload.get("transformer_context") != 2:
        failures.append("Transformer context differs from the declared two-event bound")

    by_name: dict[str, dict[str, Any]] = {}
    results = payload.get("results")
    if not isinstance(results, list):
        failures.append("missing model results")
    else:
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED_MODELS:
            failures.append("model set differs from frozen comparison")
    for name, result in by_name.items():
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("episodes") != HELDOUT_EPISODES:
            failures.append(f"invalid {name} held-out accounting")
            continue
        successes = evaluation.get("successes")
        counts = evaluation.get("action_counts")
        if not isinstance(successes, int) or not 0 <= successes <= HELDOUT_EPISODES:
            failures.append(f"invalid {name} success count")
        if (
            not isinstance(counts, list) or len(counts) != 4
            or any(not isinstance(value, int) or value < 0 for value in counts)
            or sum(counts) != HELDOUT_EPISODES * 6
        ):
            failures.append(f"invalid {name} six-event action accounting")
    random_evaluation = payload.get("random_evaluation")
    if not isinstance(random_evaluation, dict) or random_evaluation.get("episodes") != HELDOUT_EPISODES:
        failures.append("invalid random held-out accounting")

    recomputed = False
    if not failures:
        successes = {name: int(result["evaluation"]["successes"]) for name, result in by_name.items()}
        recomputed = (
            successes["gru"] / HELDOUT_EPISODES >= 0.75
            and successes["plwm"] / HELDOUT_EPISODES >= 0.75
            and successes["no_memory"] / HELDOUT_EPISODES <= 0.55
            and successes["plwm_state_off"] / HELDOUT_EPISODES <= 0.55
            and successes["transformer_2"] / HELDOUT_EPISODES <= 0.55
        )
        if payload.get("accepted_neural_task_diagnostic") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed long-delay gate")
    return Audit(
        "luanti_long_delay_dqn_audit_v1: independently validates the frozen real-Luanti long-delay reward-only DQN task diagnostic; never an architecture acceptance audit.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-long-delay-dqn-formal-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-long-delay-dqn-formal-audit.json")
    arguments = parser.parse_args()
    print(json.dumps(asdict(main(arguments.source, arguments.artifact)), indent=2))
