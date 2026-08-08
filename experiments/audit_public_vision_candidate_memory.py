"""Independently recompute a public-vision candidate-memory diagnostic result."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.train_public_vision_candidate_memory import ACCEPTANCE_ACCURACY, ACCEPTANCE_MARGIN


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"multi_candidate", "single_candidate"}


@dataclass(frozen=True)
class Audit:
    protocol: str
    source_artifact: str
    accepted: bool
    recomputed_accepted: bool
    failures: list[str]


def audit(path: Path) -> Audit:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if not isinstance(payload, dict):
        raise ValueError("candidate-memory report must be a JSON object")
    protocol = payload.get("protocol")
    required_boundary = (
        "bounded supervised storage/readout diagnostic only", "11x11x7 public vision",
        "previous chosen action", "previous scalar reward", "labels", "coordinates", "seeds", "teachers", "planners",
        "never transfer to Luanti control",
    )
    if not isinstance(protocol, str) or any(term not in protocol for term in required_boundary):
        failures.append("missing public-input and non-control boundary declaration")
    if not isinstance(payload.get("train_seed"), int) or not isinstance(payload.get("heldout_seed"), int) or payload["train_seed"] == payload["heldout_seed"]:
        failures.append("training and held-out seeds are not isolated")
    if payload.get("parameter_match_error") != 0.0:
        failures.append("candidate ablations are not exactly parameter matched")
    results = payload.get("results")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(results, list):
        failures.append("missing candidate results")
    else:
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED:
            failures.append("wrong candidate comparison set")
        elif by_name["multi_candidate"].get("parameters") != by_name["single_candidate"].get("parameters"):
            failures.append("reported parameter counts differ")
    recomputed = False
    if not failures:
        multi = by_name["multi_candidate"].get("heldout_accuracy")
        single = by_name["single_candidate"].get("heldout_accuracy")
        if not isinstance(multi, (int, float)) or not isinstance(single, (int, float)) or not 0.0 <= multi <= 1.0 or not 0.0 <= single <= 1.0:
            failures.append("invalid held-out accuracy")
        else:
            recomputed = multi >= ACCEPTANCE_ACCURACY and multi - single >= ACCEPTANCE_MARGIN
            if payload.get("accepted_mechanism_diagnostic") is not recomputed:
                failures.append("reported acceptance disagrees with the frozen gate")
    return Audit(
        "public_vision_sparse_candidate_memory_audit_v2: independently recomputes the bounded public-vision storage/readout gate; never architecture or Luanti-control evidence.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "public-vision-sparse-candidate-memory-smoke-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "public-vision-sparse-candidate-memory-smoke-audit.json")
    args = parser.parse_args()
    print(json.dumps(asdict(main(args.source, args.artifact)), indent=2))
