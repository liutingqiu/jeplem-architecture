"""Extract ARC-AGI-2 into data/arc/arc2_{train,test}.json in the same
standard ARC format as arc1: {id: {"train": [{input, output}..],
                                   "test":  [{input, output}..]}}.

The community mirror exposes columns ``fewshots`` (train demos) and
``question`` (test pairs); task ids are not present, so we synthesize
sequential ids per split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from datasets import load_dataset
    out_dir = ROOT / "data" / "arc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, fname in (("train", "arc2_train.json"), ("test", "arc2_test.json")):
        ds = load_dataset("arc-agi-community/arc-agi-2", split=split)
        tasks = {}
        for i, row in enumerate(ds):
            tasks[f"arc2_{split}_{i:05d}"] = {
                "train": [{"input": g["input"], "output": g["output"]} for g in row["fewshots"]],
                "test": [{"input": g["input"], "output": g["output"]} for g in row["question"]],
            }
        path = out_dir / fname
        path.write_text(json.dumps(tasks), encoding="utf-8")
        print(f"wrote {path} ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
