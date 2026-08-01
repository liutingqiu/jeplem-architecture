"""Run the feasibility gate across seeds and interruption lengths."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_continuity import run


def main() -> None:
    rows = []
    for distractors in (1, 5, 12):
        for seed in (5, 17, 41):
            report = run(steps=220, distractors=distractors, seed=seed)
            rows.append({"distractors": distractors, "seed": seed, **asdict(report)})
    print(json.dumps(rows, indent=2))

    valid = all(
        row["plwm_accuracy"] >= 0.9
        and row["baseline_accuracy"] <= 0.65
        and row["checkpoint_exact"]
        for row in rows
    )
    if not valid:
        raise SystemExit("feasibility sweep failed")


if __name__ == "__main__":
    main()
