"""Independently validate the frozen Luanti reward-only PPO report contract.

The trainer records an ``accepted`` field for convenience.  This audit repeats
the boundary and acceptance checks from the completed JSON artifact so a later
registry entry never relies on that self-reported boolean alone.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODELS = ("plwm", "no_memory", "gru", "transformer_16")
TRAIN_SEED_START = 4_300_000
HELDOUT_SEED_START = 4_600_000
UPDATES = 16
EPISODES_PER_UPDATE = 4
EVALUATION_EPISODES = 8
TRANSFORMER_CONTEXT = 16


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reward-only report must be a JSON object")
    return payload


def _result_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("report results must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("each result must have a name")
        if item["name"] in indexed:
            raise ValueError(f"duplicate controller result: {item['name']}")
        indexed[item["name"]] = item
    return indexed


def audit(path: Path) -> Audit:
    """Validate the complete, fixed first-block protocol and recompute its gate."""
    payload = _load(path)
    failures: list[str] = []
    protocol = payload.get("protocol")
    if not isinstance(protocol, str) or "real two-object Luanti worlds" not in protocol:
        failures.append("missing real-Luanti protocol declaration")
    for forbidden in ("no teacher trajectories", "external planning", "shaped reward", "enters training/evaluation"):
        if forbidden not in protocol:
            failures.append(f"protocol does not declare boundary: {forbidden}")

    if payload.get("device") != "cuda":
        failures.append("formal result was not recorded on cuda")
    if payload.get("updates") != UPDATES or payload.get("episodes_per_update") != EPISODES_PER_UPDATE:
        failures.append("training budget differs from frozen formal block")
    if payload.get("transformer_context") != TRANSFORMER_CONTEXT:
        failures.append("Transformer context differs from frozen 16-event baseline")

    expected_train = list(range(TRAIN_SEED_START, TRAIN_SEED_START + UPDATES * EPISODES_PER_UPDATE))
    expected_heldout = [HELDOUT_SEED_START + index * 31 for index in range(EVALUATION_EPISODES)]
    if payload.get("train_seed_range") != expected_train:
        failures.append("training seeds differ from frozen formal split")
    if payload.get("heldout_seed_range") != expected_heldout:
        failures.append("held-out seeds differ from frozen formal split")
    if set(payload.get("train_seed_range", [])).intersection(payload.get("heldout_seed_range", [])):
        failures.append("training and held-out seeds overlap")

    try:
        indexed = _result_by_name(payload)
    except ValueError as error:
        failures.append(str(error))
        indexed = {}
    if tuple(indexed) != REQUIRED_MODELS:
        failures.append("controller set or order differs from PLWM/no-memory/GRU/Transformer contract")

    for name in REQUIRED_MODELS:
        result = indexed.get(name)
        if result is None:
            continue
        if not isinstance(result.get("parameters"), int) or result["parameters"] < 1:
            failures.append(f"{name} lacks a positive parameter count")
        if result.get("training_episodes") != UPDATES * EPISODES_PER_UPDATE:
            failures.append(f"{name} has an incomplete training episode count")
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("episodes") != EVALUATION_EPISODES:
            failures.append(f"{name} has an incomplete held-out evaluation")
            continue
        successes = evaluation.get("successes")
        if not isinstance(successes, int) or not 0 <= successes <= EVALUATION_EPISODES:
            failures.append(f"{name} has invalid held-out successes")
        action_counts = evaluation.get("action_counts")
        if (
            not isinstance(action_counts, list)
            or len(action_counts) != 4
            or any(not isinstance(count, int) or count < 0 for count in action_counts)
        ):
            failures.append(f"{name} has invalid four-action accounting")

    recomputed = False
    if not failures and all(name in indexed for name in REQUIRED_MODELS):
        successes = {name: indexed[name]["evaluation"]["successes"] for name in REQUIRED_MODELS}
        recomputed = (
            successes["plwm"] / EVALUATION_EPISODES >= 0.55
            and successes["plwm"] >= successes["no_memory"] + 1
            and successes["plwm"] >= successes["gru"] + 1
            and successes["plwm"] >= successes["transformer_16"] + 1
        )
        if payload.get("accepted") is not recomputed:
            failures.append("reported accepted field disagrees with recomputed gate")
    elif payload.get("accepted") is True:
        failures.append("incomplete report cannot be accepted")
    return Audit(
        "luanti_reward_only_ppo_audit_v1: validates the frozen 16x4 real-Luanti vision/action/reward-only comparison and recomputes the declared held-out gate.",
        str(path),
        not failures and recomputed,
        recomputed,
        failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "luanti-reward-only-ppo-formal-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "luanti-reward-only-ppo-formal-audit.json")
    arguments = parser.parse_args()
    print(json.dumps(asdict(main(arguments.source, arguments.artifact)), indent=2))
