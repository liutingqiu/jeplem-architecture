"""Independently audit the bounded public-vision binding calibration."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.train_public_vision_binding_calibration import ACCEPTANCE_ACCURACY


ROOT = Path(__file__).resolve().parents[1]


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
        raise ValueError("visual-binding report must be a JSON object")
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
    if payload.get("train_steps") != 240 or payload.get("batch_size") != 64 or payload.get("device") != "cpu":
        failures.append("frozen calibration budget or device differs")
    accuracy = payload.get("heldout_accuracy")
    recomputed = False
    if not isinstance(accuracy, (int, float)) or not 0.0 <= accuracy <= 1.0:
        failures.append("invalid held-out accuracy")
    else:
        recomputed = accuracy >= ACCEPTANCE_ACCURACY
        if payload.get("accepted_public_visual_binding") is not recomputed:
            failures.append("reported acceptance disagrees with frozen gate")
    return Audit(
        "public_vision_binding_calibration_audit_v1: independent recomputation of a supervised visual readout calibration; never reward-only control or architecture evidence.",
        str(path), not failures and recomputed, recomputed, failures,
    )


def main(source: Path, artifact: Path) -> Audit:
    outcome = audit(source)
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts" / "public-vision-binding-calibration-report.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "public-vision-binding-calibration-audit.json")
    args = parser.parse_args()
    print(json.dumps(asdict(main(args.source, args.artifact)), indent=2))
