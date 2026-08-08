"""Independently audit a query-conditioned public-vision binding report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.train_query_conditioned_vision_binding import ACCEPTANCE_ACCURACY, ACCEPTANCE_MARGIN


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"query_conditioned", "late_fusion"}


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
        raise ValueError("query-conditioned report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    required_boundary = (
        "bounded supervised visual readout only", "11x11x7 public visual frames", "labels", "identity IDs",
        "coordinates", "seeds", "actions", "rewards", "teachers", "planners", "never transfer to Luanti control",
    )
    if not isinstance(protocol, str) or any(term not in protocol for term in required_boundary):
        failures.append("missing visual-only and non-control boundary declaration")
    if not isinstance(payload.get("train_seed"), int) or not isinstance(payload.get("heldout_seed"), int) or payload["train_seed"] == payload["heldout_seed"]:
        failures.append("training and held-out seeds are not isolated")
    if payload.get("train_steps") != 240 or payload.get("batch_size") != 64 or payload.get("device") != "cpu" or payload.get("parameter_match_error") != 0.0:
        failures.append("frozen comparison configuration differs")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(payload.get("results"), list):
        failures.append("missing visual-binding results")
    else:
        for result in payload["results"]:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED:
            failures.append("wrong visual-binding comparison set")
        elif by_name["query_conditioned"].get("parameters") != by_name["late_fusion"].get("parameters"):
            failures.append("visual-binding parameter counts differ")
    recomputed = False
    if not failures:
        query = by_name["query_conditioned"].get("heldout_accuracy")
        late = by_name["late_fusion"].get("heldout_accuracy")
        if not isinstance(query, (int, float)) or not isinstance(late, (int, float)) or not 0.0 <= query <= 1.0 or not 0.0 <= late <= 1.0:
            failures.append("invalid held-out accuracy")
        else:
            recomputed = query >= ACCEPTANCE_ACCURACY and query - late >= ACCEPTANCE_MARGIN
            if payload.get("accepted_query_conditioned_binding") is not recomputed:
                failures.append("reported acceptance disagrees with frozen gate")
    return Audit(
        "query_conditioned_public_vision_binding_audit_v1: independent recomputation of a visual readout comparison; never reward-only control or architecture evidence.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "query-conditioned-public-vision-binding-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "query-conditioned-public-vision-binding-audit.json")
    args = parser.parse_args()
    print(json.dumps(asdict(main(args.source, args.artifact)), indent=2))
