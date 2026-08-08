"""Independently audit an object-token episodic-memory diagnostic report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.train_object_token_memory import ACCEPTANCE_ACCURACY, ACCEPTANCE_MARGIN


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"object_token_multi_writer", "object_token_single_writer", "object_token_memory_disabled"}


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
        raise ValueError("object-token memory report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    required_boundary = (
        "bounded supervised storage/readout diagnostic", "11x11x7 public vision", "object masks derive solely",
        "grid-index parameters are internal model weights", "labels", "identity ids", "coordinates", "seeds",
        "teachers", "planners", "never transfer to luanti control",
    )
    if not isinstance(protocol, str) or any(term not in protocol.casefold() for term in required_boundary):
        failures.append("missing object-token memory visual-only and non-control boundary declaration")
    if not isinstance(payload.get("train_seed"), int) or not isinstance(payload.get("heldout_seed"), int) or payload["train_seed"] == payload["heldout_seed"]:
        failures.append("training and held-out seeds are not isolated")
    if payload.get("train_steps") != 240 or payload.get("batch_size") != 64 or payload.get("device") != "cpu" or payload.get("parameter_match_error") != 0.0:
        failures.append("frozen comparison configuration differs")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(payload.get("results"), list):
        failures.append("missing object-token memory results")
    else:
        for result in payload["results"]:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED:
            failures.append("wrong object-token memory comparison set")
        elif len({by_name[name].get("parameters") for name in REQUIRED}) != 1:
            failures.append("object-token memory parameter counts differ")
    recomputed = False
    if not failures:
        values = {name: by_name[name].get("heldout_accuracy") for name in REQUIRED}
        if any(not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in values.values()):
            failures.append("invalid held-out accuracy")
        else:
            multi = float(values["object_token_multi_writer"])
            single = float(values["object_token_single_writer"])
            disabled = float(values["object_token_memory_disabled"])
            recomputed = multi >= ACCEPTANCE_ACCURACY and multi - single >= ACCEPTANCE_MARGIN and multi - disabled >= ACCEPTANCE_MARGIN
            if payload.get("accepted_object_token_memory") is not recomputed:
                failures.append("reported acceptance disagrees with frozen gate")
    return Audit(
        "object_token_episodic_memory_audit_v1: independent recomputation of an object-token storage/readout comparison; never reward-only control or architecture evidence.",
        str(path), not failures and recomputed, recomputed, failures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "object-token-episodic-memory-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "object-token-episodic-memory-audit.json")
    args = parser.parse_args()
    outcome = audit(args.source)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))
