"""Programming FREE EXPLORATION -- the system sets its OWN problems.

The user's rule: never constrain the goal from outside -- no fixed question
bank, no curriculum.  Free exploration means the system decides WHAT to learn,
by generating its own tasks.  The programming path works exactly that way:

  * the TOOL SPACE (the primitives + their two-step compositions, from the
    body's primitive library) is the architecture's executable vocabulary --
    an asset, not a curriculum;
  * each round the system GENERATES a task: it picks a random target program G
    from the tool space, builds random input lists, and computes the outputs --
    the (input, output) pairs are the problem it sets for itself;
  * it SOLVES its own problem: it scores every program in the tool space
    against the examples (fit evidence) with a persistent self-model, proposes
    H, and VERIFIES by execution on held-out inputs;
  * the verification result trains the persistent self-model, so mastery
    ACCUMULATES across rounds (no wipe, no spin);
  * the journal records what it explored and how much it can now solve.

Nothing external fixes a target set of problems -- the question stream is
self-generated, the targets are self-chosen, the correctness is execution.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard  # noqa: E402
from experiments.arc.run_skill_acquisition_loop import PRIMITIVES  # noqa: E402

JOURNAL = ROOT / "artifacts" / "programming_mastery_journal.jsonl"

# held-out inputs used ONLY for verifying a proposed program generalises
# (a verification set, not a question bank)
TEST_INPUTS = [[3, 1, 4, 1, 5], [2, 7, 1, 8, 2], [0, -3, 5, -1], [1, 1, 2, 2], [5]]
EXAMPLES = 2          # examples the system gives itself per task
WINDOW = 20           # running solve-rate window


def run_code(code: str, L):
    """SAFE execution: primitive codes use only list/number ops on ints."""
    return eval(code, {"__builtins__": {}},
                {"L": L, "sum": sum, "sorted": sorted,
                 "max": max, "min": min, "len": len})


# ---- the tool space: primitives + all two-step compositions (executable) ----
def make_fn(code):
    return lambda L: run_code(code, L)


def eval_program(prog, L):
    """prog is a list of primitive names (depth 1 or 2), applied in order."""
    x = L
    for name in prog:
        f = make_fn(dict(PRIMITIVES)[name])
        try:
            x = f(x)
        except Exception:
            return None
    return x


PRIM_NAMES = [n for n, _ in PRIMITIVES]
PROGRAM_SPACE = PRIM_NAMES + [a + "_" + b for a in PRIM_NAMES for b in PRIM_NAMES]
PROG_LIST = [([a], n) for a, n in zip(PRIM_NAMES, PRIM_NAMES)] + \
            [([a, b], a + "_" + b) for a in PRIM_NAMES for b in PRIM_NAMES]


def gen_task(rng) -> tuple:
    """The system GENERATES its own task: a random target program (depth 1 or 2
    -- with depth-2 slightly preferred so it keeps climbing) + random inputs."""
    depth = 2 if rng.random() < 0.6 else 1
    while True:
        if depth == 1:
            g = [rng.choice(PRIM_NAMES)]
        else:
            g = [rng.choice(PRIM_NAMES), rng.choice(PRIM_NAMES)]
        ex = []
        ok = True
        for _ in range(EXAMPLES):
            x = [rng.randint(-5, 9) for _ in range(rng.randint(2, 6))]
            y = eval_program(g, x)
            if y is None:
                ok = False
                break
            ex.append((x, y))
        if ok:
            return g, ex


def fit_vec(progs, examples) -> np.ndarray:
    """The evidence the system 'sees': which programs fit all its examples."""
    v = np.zeros(len(progs))
    for i, (g, name) in enumerate(progs):
        if all(eval_program(g, x) == y for x, y in examples):
            v[i] = 1.0
    return v


def correctness(prog, target, inputs=TEST_INPUTS) -> bool:
    """VERIFY by execution on held-out inputs: does prog match the target G?"""
    for x in inputs:
        if eval_program(prog, x) != eval_program(target, x):
            return False
    return True


class SelfModel(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 48), nn.ReLU(), nn.Linear(48, n_out))

    def forward(self, x):
        return self.net(x)


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def main() -> None:
    start_resource_guard(interval=2.0)
    rng = random.Random(1000)
    n_prog = len(PROG_LIST)
    model = SelfModel(n_prog, n_prog)      # ONE persistent self-model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss()

    round_no = 0
    solved_hist = []                       # running window of solved tasks
    solved_targets = set()                 # programs it has verified it can solve
    print(f"programming free-exploration started (tool space: {n_prog} programs, "
          f"journal: {JOURNAL})", flush=True)
    while True:
        round_no += 1
        # --- the system sets its own problem ---
        target, examples = gen_task(rng)
        # --- it solves it: score every program on the example evidence ---
        fv = torch.tensor(fit_vec(PROG_LIST, examples)).float()
        with torch.no_grad():
            scores = model(fv)
        best = int(scores.argmax())
        proposed, pname = PROG_LIST[best]
        solved = correctness(proposed, target)
        # --- it learns from the verification (execution is the truth) ---
        labels = torch.tensor(
            [1.0 if correctness(p, target) else 0.0 for p, _ in PROG_LIST]).float()
        for _ in range(6):
            loss = bce(model(fv), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # --- accumulate mastery ---
        if solved:
            solved_targets.add(tuple(target))
        solved_hist.append(int(solved))
        if len(solved_hist) > WINDOW:
            solved_hist.pop(0)
        tname = "_".join(target)
        log({"round": round_no, "time": time.time(),
             "target": tname,
             "solved": bool(solved),
             "window_solve_rate": round(sum(solved_hist) / len(solved_hist), 3),
             "mastered_programs": len(solved_targets),
             "fit_count": int(fv.sum())})
        time.sleep(1)   # pace, let the resource guard breathe


if __name__ == "__main__":
    main()
