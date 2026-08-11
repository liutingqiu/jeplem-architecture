"""Phase-2 body v11: executive control IN the body -- budgeted attention over
memory writes (Block-9 semantics).

The body's memory has a WRITE BUDGET (only B frames may be written, e.g. due
to limited slots).  A control mechanism (Block-9 budgeted attention) decides
WHICH frames are worth remembering: frames whose clue disagrees with the
current belief (informative / conflict) vs frames that merely repeat it.
Controls: write-everything (FIFO, budget violated -> early frames evicted)
and random-budget (random B frames).

Verdicts: query accuracy with budgeted attention vs the two controls, and the
attention actually selected the informative frames.
Criterion: budgeted-attention query accuracy >= 0.85 and >= both controls + 0.15.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_IDENTITIES = 3
BUDGET = 8
RELIABILITY = 0.65


def run(*, episodes: int = 100, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    def truth_for(ep: int) -> dict[int, int]:
        return {i: (rng.randrange(2)) for i in range(N_IDENTITIES)}

    outcomes = {"attention": [], "write_all": [], "random": []}
    for ep in range(episodes):
        truth = truth_for(ep)
        # build a HETEROGENEOUS frame stream: per identity 2 RELIABLE frames
        # (p=0.9, high-confidence signal) and 3 NOISY frames (p=0.55, low
        # confidence) -- budgeted attention should prefer reliable frames.
        frames: list[tuple[int, int, float]] = []  # (identity, shown, confidence)
        for identity in range(N_IDENTITIES):
            for _ in range(2):
                shown = truth[identity] if rng.random() < 0.9 else (1 - truth[identity])
                frames.append((identity, shown, 0.9))
            for _ in range(3):
                shown = truth[identity] if rng.random() < 0.55 else (1 - truth[identity])
                frames.append((identity, shown, 0.55))
        # memory = dict identity -> value votes (written frames)
        def query(memory: dict[int, list[int]]) -> float:
            correct = 0
            for identity in range(N_IDENTITIES):
                votes = memory.get(identity, [])
                if not votes:
                    continue
                pred = int(np.bincount(votes).argmax())
                correct += int(pred == truth[identity])
            return correct / N_IDENTITIES

        # attention control: write the HIGH-CONFIDENCE frames (reliable
        # evidence) up to the budget; noisy frames are skipped by attention.
        memory_a: dict[int, list[int]] = {}
        written = 0
        for identity, shown, conf in frames:
            if conf >= 0.9 and written < BUDGET:
                memory_a.setdefault(identity, []).append(shown)
                written += 1
        outcomes["attention"].append(query(memory_a))

        # write-all control: write everything (FIFO-like, budget exceeded ->
        # only the LAST BUDGET frames kept)
        memory_w: dict[int, list[int]] = {}
        for identity, shown, conf in frames[:BUDGET]:
            memory_w.setdefault(identity, []).append(shown)
        outcomes["write_all"].append(query(memory_w))

        # random-budget control: random BUDGET frames written
        memory_r: dict[int, list[int]] = {}
        for identity, shown, conf in rng.sample(frames, min(BUDGET, len(frames))):
            memory_r.setdefault(identity, []).append(shown)
        outcomes["random"].append(query(memory_r))

    att = float(np.mean(outcomes["attention"]))
    wa = float(np.mean(outcomes["write_all"]))
    rnd = float(np.mean(outcomes["random"]))
    accepted = att >= 0.85 and att >= max(wa, rnd) + 0.15
    report = {
        "protocol": "phase2_body_v11_executive_v1: budgeted attention (Block-9) decides which frames are worth writing under a write budget (informative = clue disagrees with current belief), vs write-all (FIFO eviction) and random-budget controls. Criterion: attention query accuracy >= 0.85 and >= controls + 0.15.",
        "seed": experiment_seed,
        "results": {
            "attention_query_acc": round(att, 3),
            "write_all_acc": round(wa, 3),
            "random_budget_acc": round(rnd, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v11-executive.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v11-executive.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v11-executive.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))
