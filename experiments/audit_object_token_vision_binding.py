"""Independently audit an object-token public-vision binding report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.train_object_token_vision_binding import ACCEPTANCE_ACCURACY, ACCEPTANCE_MARGIN


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"object_token_query_matching", "query_independent_object_tokens"}


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
        raise ValueError("object-token visual-binding report must be a JSON object")
    failures: list[str] = []
    protocol = payload.get("protocol")
    required_boundary = (
        "bounded supervised visual readout only", "11x11x7 public visual frames", "object masks are derived only",
        "labels", "identity IDs", "coordinates", "seeds", "actions", "rewards", "teachers", "planners",
        "never transfer to Luanti control",
    )
    if not isinstance(protocol, str) or any(term.casefold() not in protocol.casefold() for term in required_boundary):
        failures.append("missing object-token visual-only and non-control boundary declaration")
    if not isinstance(payload.get("train_seed"), int) or not isinstance(payload.get("heldout_seed"), int) or payload["train_seed"] == payload["heldout_seed"]:
        failures.append("training and held-out seeds are not isolated")
    if payload.get("train_steps") != 240 or payload.get("batch_size") != 64 or payload.get("device") != "cpu" or payload.get("parameter_match_error") != 0.0:
        failures.append("frozen comparison configuration differs")
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(payload.get("results"), list):
        failures.append("missing object-token visual-binding results")
    else:
        for result in payload["results"]:
            if isinstance(result, dict) and isinstance(result.get("name"), str):
                by_name[result["name"]] = result
        if set(by_name) != REQUIRED:
            failures.append("wrong object-token visual-binding comparison set")
        elif by_name["object_token_query_matching"].get("parameters") != by_name["query_independent_object_tokens"].get("parameters"):
            failures.append("object-token visual-binding parameter counts differ")
    recomputed = False
    if not failures:
        matching = by_name["object_token_query_matching"].get("heldout_accuracy")
        ablation = by_name["query_independent_object_tokens"].get("heldout_accuracy")
        if not isinstance(matching, (int, float)) or not isinstance(ablation, (int, float)) or not 0.0 <= matching <= 1.0 or not 0.0 <= ablation <= 1.0:
            failures.append("invalid held-out accuracy")
        else:
            recomputed = matching >= ACCEPTANCE_ACCURACY and matching - ablation >= ACCEPTANCE_MARGIN
            if payload.get("accepted_object_token_binding") is not recomputed:
                failures.append("reported acceptance disagrees with frozen gate")
    return Audit(
        "object_token_public_vision_binding_audit_v1: independent recomputation of an object-token visual readout comparison; never reward-only control or architecture evidence.",
        str(path), not failures and recomputed, recomputed, failures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "object-token-public-vision-binding-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "object-token-public-vision-binding-audit.json")
    args = parser.parse_args()
    outcome = audit(args.source)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))
