"""Independently audit the repaired Luanti addressed-memory calibration."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
TRAINING_SEEDS = list(range(12_900_000, 12_900_128))
HELDOUT_SEEDS = list(range(12_910_000, 12_910_016))
ACTION_SEED = 8_042_337


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
        raise ValueError("addressed calibration report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    required_boundary = (
        "real Luanti gated delayed-goal course", "reward-only", "11x11 public vision",
        "previous chosen action", "previous scalar reward", "legal-action masks",
        "no cue labels", "symbolic bridge state", "teacher trajectory", "external planner", "shaped reward",
        "parameter-capacity-matched",
    )
    if not isinstance(protocol, str) or any(item not in protocol for item in required_boundary):
        failures.append("missing repaired reward-only public-input declaration")
    if payload.get("course") != "addressed_gated":
        failures.append("wrong course")
    if payload.get("device") != "cuda":
        failures.append("calibration did not run on local CUDA")
    if payload.get("training_seed_range") != TRAINING_SEEDS or payload.get("heldout_seed_range") != HELDOUT_SEEDS:
        failures.append("frozen addressed calibration seeds differ")
    if payload.get("action_seed") != ACTION_SEED:
        failures.append("frozen addressed calibration action seed differs")
    if payload.get("max_steps") != 6 or payload.get("transformer_context") != 2:
        failures.append("wrong course horizon or Transformer context")
    if payload.get("acceptance_mode") != "addressed_memory_diagnostic":
        failures.append("wrong acceptance mode")
    tolerance = payload.get("parameter_match_tolerance")
    errors = payload.get("parameter_match_errors")
    if not isinstance(tolerance, (int, float)) or not 0.0 < tolerance <= 0.05 or not isinstance(errors, dict):
        failures.append("missing strict capacity-match declaration")
    elif any(not isinstance(errors.get(name), (int, float)) or errors[name] > tolerance for name in REQUIRED_MODELS):
        failures.append("one or more models exceed declared capacity tolerance")
    results = payload.get("results")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(results, list):
        failures.append("missing model results")
    else:
        for item in results:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                by_name[item["name"]] = item
        if set(by_name) != REQUIRED_MODELS:
            failures.append("model set differs from calibration contract")
    for name, item in by_name.items():
        evaluation = item.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("episodes") != len(HELDOUT_SEEDS):
            failures.append(f"invalid {name} held-out accounting")
    recomputed = False
    if not failures:
        rate = {name: by_name[name]["evaluation"]["successes"] / len(HELDOUT_SEEDS) for name in REQUIRED_MODELS}
        recomputed = (
            rate["plwm"] >= 0.75 and rate["gru"] >= 0.75
            and rate["no_memory"] <= 0.60 and rate["plwm_state_off"] <= 0.60 and rate["transformer_2"] <= 0.60
        )
        if payload.get("accepted_neural_task_diagnostic") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed calibration gate")
    return Audit(
        "luanti_addressed_memory_diagnostic_audit_v1: independent recomputation of the repaired seven-channel current-writer calibration; never an architecture-advantage audit.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-addressed-repaired-current-writer-dqn-smoke-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-addressed-repaired-current-writer-dqn-smoke-audit.json")
    args = parser.parse_args()
    print(json.dumps(asdict(main(args.source, args.artifact)), indent=2))
