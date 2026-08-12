"""Extract ARC-AGI-1 structured data from the HuggingFace cache into
data/arc/arc1_{training,evaluation}.json (standard ARC format:
{task_id: {"train": [{"input": .., "output": ..}, ..],
           "test":  [{"input": .., "output": ..}, ..]}}).

The two parquet blobs in the cache are the training (400) and evaluation
(400) splits; they are identified here by checking task ids against a few
canonical ARC-AGI-1 TRAINING ids (any hit => training split).
"""
import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get(
    "HF_CACHE",
    r"C:/Users/Administrator/.cache/huggingface/hub/datasets--dataartist--arc-agi/blobs"))

# canonical ARC-AGI-1 training task ids (spot-check anchors)
KNOWN_TRAIN_IDS = {"007bbfb7", "00d62c1b", "025d127b", "0520fde7",
                   "08ed6ac7", "0a938d79", "1c0d4a94"}


def read_blobs() -> list[dict]:
    tables = []
    for p in sorted(glob.glob(str(CACHE / "*"))):
        if os.path.getsize(p) < 1000:
            continue
        with open(p, "rb") as f:
            if f.read(4) != b"PAR1":
                continue
        tables.append(pq.read_table(p))
    return tables


def rows_to_arc(t: "pyarrow.Table") -> dict:
    d = t.to_pydict()
    out = {}
    for tid, train, test in zip(d["id"], d["train"], d["test"]):
        out[tid] = {
            "train": [{"input": g["input"], "output": g["output"]} for g in train],
            "test": [{"input": g["input"], "output": g["output"]} for g in test],
        }
    return out


def main() -> None:
    tables = read_blobs()
    print(f"found {len(tables)} parquet tables")
    splits: dict[str, dict] = {}
    for t in tables:
        arc = rows_to_arc(t)
        ids = set(arc)
        is_train = bool(ids & KNOWN_TRAIN_IDS)
        name = "training" if is_train else "evaluation"
        splits[name] = arc
        print(f"  {name}: {len(arc)} tasks, sample ids {sorted(ids)[:3]}")
    if "training" not in splits or "evaluation" not in splits:
        raise SystemExit("could not identify both splits; check anchors")
    out_dir = ROOT / "data" / "arc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("training", "evaluation"):
        path = out_dir / f"arc1_{name}.json"
        path.write_text(json.dumps(splits[name]), encoding="utf-8")
        print(f"wrote {path} ({len(splits[name])} tasks)")


if __name__ == "__main__":
    main()
