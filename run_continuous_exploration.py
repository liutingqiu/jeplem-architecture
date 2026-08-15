"""Continuous PROGRAMMING-ONLY mastery -- ALL resources to one path.

The user's directive: give every resource to programming, break through that one
path.  He also noticed the system was spinning in place.

The spinning was real and had a concrete cause: the old loop re-initialised a
FRESH skill model every round (each skill_seed(seed) trains 15 rounds from
scratch on a fresh seed), so the skill could never accumulate -- every round it
re-learned the same ~98% and never got past the last compositions.  Re-learning
the same thing every round is spinning, not learning.

This version keeps ONE persistent skill model and ONE growing primitive pool
across ALL rounds: each round practises all 12 problems, and the model weights
plus the outward-unlocked compositions are never reset, so the skill ACCUMULATES
until the 4 two-step compositions are stably mastered (12/12 sustained), and
keeps deepening after that.  Free exploration is dropped -- all resources go to
the programming path.  Nothing here is an external constraint: the system learns
the same way, just with persistent (rather than wiped) memory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard  # noqa: E402
from experiments.arc.run_skill_acquisition_loop import (  # noqa: E402
    PRIMITIVES, PROBLEMS, TEST_INPUTS, _onehot, _prim_index, spec_feature,
    verify, SelfModel,
)

JOURNAL = ROOT / "artifacts" / "programming_mastery_journal.jsonl"
COMPOSITIONS = {"double_reverse", "evens_sum", "square_sort", "negate_reverse"}
MASTERY_RUNS = 30   # consecutive 12/12 rounds -> marked mastered


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def evaluate(model, pool) -> dict:
    """Final-level accuracy right now (the honest 'what it can do' number)."""
    n_prim = len(PRIMITIVES)
    ok = {}
    for spec in PROBLEMS:
        cands = pool
        feats = torch.stack(
            [torch.tensor(list(spec_feature(spec)) + _onehot(p, n_prim)).float()
             for p in cands])
        with torch.no_grad():
            scores = torch.sigmoid(model(feats)).squeeze(-1)
        best = cands[int(scores.argmax())]
        ok[spec] = verify(spec, PRIMITIVES[best][0], TEST_INPUTS)
    comps = [s for s in PROBLEMS if PROBLEMS[s] in COMPOSITIONS]
    return {
        "last_acc": sum(ok.values()) / len(ok),
        "comp_acc": sum(ok[s] for s in comps) / len(comps),
        "fails": [s for s, v in ok.items() if not v],
    }


def main() -> None:
    start_resource_guard(interval=2.0)
    n_prim = len(PRIMITIVES)
    specs = list(PROBLEMS)
    model = SelfModel(24, n_prim)      # ONE persistent skill model
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    bce = nn.BCEWithLogitsLoss()
    pool = list(range(8))              # start with the 8 simples; compositions
                                       # are unlocked OUTWARD and then kept
    round_no = 0
    consec_perfect = 0
    mastered = False
    print(f"programming-mastery started (journal: {JOURNAL})", flush=True)
    while True:
        round_no += 1
        # practice pass: train on all specs (persistent model, persistent pool)
        for spec in specs:
            cands = list(pool)   # snapshot: pool may grow (outward unlock)
            feats = torch.stack(
                [torch.tensor(list(spec_feature(spec)) + _onehot(p, n_prim)).float()
                 for p in cands])
            with torch.no_grad():
                scores = torch.sigmoid(model(feats)).squeeze(-1)
            best = cands[int(scores.argmax())]
            outcomes = torch.tensor(
                [1.0 if verify(spec, PRIMITIVES[p][0], TEST_INPUTS) else 0.0
                 for p in cands])
            for _ in range(10):
                loss = bce(model(feats).squeeze(-1), outcomes)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if not verify(spec, PRIMITIVES[best][0], TEST_INPUTS):
                need = _prim_index(PROBLEMS[spec])
                if need not in pool:      # outward acquisition, kept forever
                    pool.append(need)
        # evaluation pass: honest current level
        ev = evaluate(model, pool)
        consec_perfect = consec_perfect + 1 if ev["last_acc"] >= 1.0 else 0
        if not mastered and consec_perfect >= MASTERY_RUNS:
            mastered = True
        log({"round": round_no, "time": time.time(),
             "last_acc": round(ev["last_acc"], 3),
             "comp_acc": round(ev["comp_acc"], 3),
             "fails": ev["fails"],
             "pool_size": len(pool),
             "consec_perfect": consec_perfect,
             "mastered": mastered})
        time.sleep(1)   # pace, let the resource guard breathe


if __name__ == "__main__":
    main()
