"""Independently audit the frozen object-token Luanti addressed balanced smoke."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
TRAINING_SEEDS = list(range(16_000_000, 16_001_024))
HELDOUT_SEEDS = list(range(16_002_000, 16_002_032))
ACTION_SEED = 9_042_337


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def audit(path: Path) -> Audit:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("object-token balanced smoke report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    boundary = (
        "real repaired seven-channel luanti addressed balanced retrieval", "reward-only", "11x11 public vision",
        "previous chosen action", "previous scalar reward", "legal-action masks", "same object-token visual event encoder",
        "counterbalanced course", "50/50", "no cue/query labels", "coordinates", "symbolic bridge state",
        "seed", "teacher trajectory", "external planner", "shaped reward", "plwm-only input",
    )
    if not isinstance(protocol, str) or any(item not in protocol.casefold() for item in boundary):
        failures.append("missing object-token balanced reward-only public-input declaration")
    if payload.get("course") != "addressed_balanced" or payload.get("device") != "cuda":
        failures.append("wrong course or non-CUDA executor")
    if payload.get("training_seed_range") != TRAINING_SEEDS or payload.get("heldout_seed_range") != HELDOUT_SEEDS or payload.get("action_seed") != ACTION_SEED:
        failures.append("frozen object-token balanced smoke seeds differ")
    if payload.get("max_steps") != 6 or payload.get("transformer_context") != 2 or payload.get("acceptance_mode") != "object_token_writer_smoke":
        failures.append("wrong horizon, Transformer bound, or acceptance mode")
    tolerance, errors = payload.get("parameter_match_tolerance"), payload.get("parameter_match_errors")
    if not isinstance(tolerance, (int, float)) or not 0.0 < tolerance <= 0.05 or not isinstance(errors, dict) or any(not isinstance(errors.get(name), (int, float)) or errors[name] > tolerance for name in REQUIRED_MODELS):
        failures.append("capacity match is missing or exceeds tolerance")
    by_name = {item.get("name"): item for item in payload.get("results", []) if isinstance(item, dict) and isinstance(item.get("name"), str)}
    if set(by_name) != REQUIRED_MODELS:
        failures.append("model set differs from frozen balanced smoke contract")
    if any(not isinstance(item.get("evaluation"), dict) or item["evaluation"].get("episodes") != len(HELDOUT_SEEDS) for item in by_name.values()):
        failures.append("invalid held-out accounting")
    recomputed = False
    if not failures:
        rate = {name: by_name[name]["evaluation"].get("successes", -1) / len(HELDOUT_SEEDS) for name in REQUIRED_MODELS}
        recomputed = rate["plwm"] >= 0.75 and rate["gru"] >= 0.75 and rate["transformer_2"] >= 0.75 and rate["no_memory"] <= 0.50 and rate["plwm_state_off"] <= 0.50
        if payload.get("accepted_neural_task_diagnostic") is not recomputed:
            failures.append("reported acceptance disagrees with recomputed balanced smoke gate")
    return Audit("luanti_addressed_balanced_object_token_smoke_audit_v1: independent recomputation of one capacity-matched CUDA smoke on the side-bias-separated balanced course; never an architecture-advantage audit.", str(path), not failures and recomputed, recomputed, failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-addressed-balanced-object-token-dqn-smoke-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-addressed-balanced-object-token-dqn-smoke-audit.json")
    args = parser.parse_args()
    outcome = audit(args.source)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))
