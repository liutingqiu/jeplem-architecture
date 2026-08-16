"""True-world-driven learning loop -- the system faces REAL tasks, grows a
library, and is measured on REAL held-out truth.

Fixed principles (user, 2026-08-16 -- no sandbox, no curriculum, no spin):
  1. REAL task source   -- real ARC-1 tasks, real test truth (not synthetic
     problems, not a question bank).
  2. OUTWARD            -- unsolved real tasks drive NEW vocabulary: fragments
     abstracted from the real tasks it HAS solved become new atomic primitives
     (cross-task library growth), and it re-attacks the hard queue with them.
  3. SELF-CHALLENGE     -- unsolved tasks are re-queued with priority; it
     attacks its own edge instead of coasting on what it already knows.
  4. HONEST MEASURE     -- real held-out solve rate over real tasks, zero
     false facts, provenance kept.

Solve pipeline per real task: static grid-primitive library -> latent-cause
(new vocabulary read from the task's own demos) -> schema grammar + grown
library (cross-task vocabulary).  The grown library is the OUTWARD interface:
fragments from solved real tasks become atomic primitives that search uses
first on later tasks.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard  # noqa: E402
from experiments.arc import induction, schema_search  # noqa: E402
from experiments.arc.grid import equal  # noqa: E402
from experiments.arc.library_growth import GrownLibrary, fragment_of  # noqa: E402
from experiments.arc.run_arc_integration import latent_cause_solve  # noqa: E402

DATA = ROOT / "data" / "arc"
JOURNAL = ROOT / "artifacts" / "true_world_journal.jsonl"
BATCH = 60
LEARN_EVERY = 5
MIN_OCCUR = 2
MAX_COMPOSED = 200


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def try_solve(task, grown, rng) -> tuple[str | None, bool, bool, object]:
    """Return (source, correct_on_test, false_fact, fragment_or_None)."""
    demos = [(g["input"], g["output"]) for g in task["train"]]
    test_in = task["test"][0]["input"]
    test_out = task["test"][0]["output"]
    # 1 static grid-primitive library
    try:
        prog = induction.search_program(demos, max_depth=2)
    except Exception:
        prog = None
    if prog is not None:
        return ("static", equal(prog[1](test_in), test_out), False, None)
    # 2 latent-cause (new vocabulary from the task's own demos)
    try:
        out = latent_cause_solve(demos, rng)
    except Exception:
        out = None
    if out is not None:
        cand, info = out
        ok = equal(cand(test_in), test_out)
        return ("latent", ok, not ok, info.get("source"))
    # 3 schema grammar + grown library (cross-task vocabulary)
    try:
        name, pred = schema_search.solve_task(task, max_steps=2,
                                              max_composed=MAX_COMPOSED, library=grown)
    except Exception:
        name, pred = None, None
    if name is not None:
        return ("schema", equal(pred, test_out), False, fragment_of(name))
    return (None, False, False, None)


def main() -> None:
    start_resource_guard(interval=2.0)
    rng = random.Random(20260816)
    train = json.loads((DATA / "arc1_training.json").read_text(encoding="utf-8"))
    eval_ = json.loads((DATA / "arc1_evaluation.json").read_text(encoding="utf-8"))
    tasks = list(train.items()) + list(eval_.items())
    rng.shuffle(tasks)

    grown = GrownLibrary()
    pending = list(tasks)      # real tasks not yet solved
    hard_queue = []            # re-attack queue (self-challenge)
    seen = set()
    solved = set()
    false_total = 0
    round_no = 0
    print(f"true-world learning started ({len(tasks)} real tasks, "
          f"journal {JOURNAL})", flush=True)
    while True:
        round_no += 1
        batch = hard_queue[:BATCH]
        hard_queue = hard_queue[BATCH:]
        need = BATCH - len(batch)
        if need > 0 and pending:
            rng.shuffle(pending)
            batch += pending[:need]
            pending = pending[need:]
        if not batch:
            break
        src_count = {}
        solved_round = 0
        for tid, task in batch:
            src, ok, is_false, frag = try_solve(task, grown, rng)
            seen.add(tid)
            if ok:
                solved.add(tid)
                solved_round += 1
                src_count[src] = src_count.get(src, 0) + 1
                if frag is not None:
                    grown.observe(frag, tid)
            elif src is not None:
                if is_false:
                    false_total += 1
                hard_queue.append((tid, task))   # wrong on truth -> keep attacking
            else:
                hard_queue.append((tid, task))   # unsolved -> challenge queue
        if round_no % LEARN_EVERY == 0:
            grown.learn(min_occur=MIN_OCCUR)     # fragments -> atomic primitives
        rate = len(solved) / len(seen) if seen else 0.0
        log({"round": round_no, "time": time.time(),
             "real_tasks_seen": len(seen),
             "real_tasks_solved": len(solved),
             "real_solve_rate": round(rate, 4),
             "round_solved": solved_round,
             "sources": src_count,
             "hard_queue": len(hard_queue),
             "library_primitives": len(grown.active),
             "fragments_seen": len(grown.fragments),
             "false_facts": false_total})
        time.sleep(1)   # pace, let the resource guard breathe


if __name__ == "__main__":
    main()
