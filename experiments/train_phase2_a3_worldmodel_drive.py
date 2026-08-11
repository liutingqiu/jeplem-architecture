"""Phase-2 A3: world-model prediction head inside the body, prediction-error
drives exploration (Block-2 drive semantics + Block-7 world model).

The body lives on a state stream with a deterministic transition
(s' = (s + k) mod N).  A small prediction head learns the transition from the
body's own observations.  The prediction error on a state is the body's
'internal gap' (surprise): the body preferentially revisits high-error states
to learn them (curiosity-driven exploration, no external reward).

Verdicts:
  - transition prediction accuracy (the head learned the world's law);
  - exploration coverage: curiosity-driven visits reach all states faster
    than random (or at higher coverage in the same steps);
  - the prediction-error signal is well-defined (high-error states get
    revisited).
Criterion: prediction >= 0.9, curiosity coverage >= random coverage + 0.2.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PredictionHead(nn.Module):
    def __init__(self, n_states: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_states, 16), nn.ReLU(), nn.Linear(16, n_states))

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        return self.net(onehot)


def run(*, n_states: int = 6, transition_k: int = 2, steps: int = 240,
        experiment_seed: int = 41_123, artifact: Path | None = None) -> dict:
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    rng = random.Random(experiment_seed)

    def transition(s: int) -> int:
        return (s + transition_k) % n_states

    # --- curiosity-driven body: learns transition, explores high-error states
    head = PredictionHead(n_states)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    visited = set()
    current = rng.randrange(n_states)
    error_history: list[float] = []
    pred_correct = pred_total = 0
    # per-state error estimates (updated online)
    state_error: dict[int, float] = {s: 1.0 for s in range(n_states)}
    visits: dict[int, int] = {s: 0 for s in range(n_states)}
    steps_to_full_c: int | None = None
    for step in range(steps):
        # predict next state from current
        onehot = torch.zeros(1, n_states)
        onehot[0, current] = 1.0
        logits = head(onehot)
        pred = int(logits.argmax(-1).item())
        # observe the actual next state (transition happens when 'acting')
        # the body CHOOSES the next state to visit; the transition law gives
        # the observed outcome of acting on the current state
        nxt = transition(current)
        # update prediction head on (current -> nxt)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([nxt]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        pred_correct += int(pred == nxt)
        pred_total += 1
        # prediction error on current state (softmax entropy as surprise)
        probs = torch.softmax(logits, -1).squeeze(0)
        err = float((1.0 - probs[nxt]).item())
        state_error[current] = 0.7 * state_error.get(current, 1.0) + 0.3 * err
        error_history.append(state_error[current])
        visited.add(current)
        visits[current] += 1
        if steps_to_full_c is None and len(visited) == n_states:
            steps_to_full_c = step + 1
        # curiosity choice: next state = argmax state_error among all
        current = max(range(n_states), key=lambda s: state_error.get(s, 1.0))
    pred_acc = pred_correct / pred_total

    # --- random control (same transition, random next-state choice)
    head_r = PredictionHead(n_states)
    opt_r = torch.optim.AdamW(head_r.parameters(), lr=1e-2)
    visited_r = set()
    cur = rng.randrange(n_states)
    steps_to_full_r: int | None = None
    for step in range(steps):
        onehot = torch.zeros(1, n_states)
        onehot[0, cur] = 1.0
        logits = head_r(onehot)
        nxt = transition(cur)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([nxt]))
        opt_r.zero_grad(set_to_none=True)
        loss.backward()
        opt_r.step()
        visited_r.add(cur)
        if steps_to_full_r is None and len(visited_r) == n_states:
            steps_to_full_r = step + 1
        cur = rng.randrange(n_states)  # random exploration

    coverage_c = len(visited) / n_states
    coverage_r = len(visited_r) / n_states
    stc = steps_to_full_c if steps_to_full_c is not None else steps
    str_ = steps_to_full_r if steps_to_full_r is not None else steps
    # criterion aligned with the docstring claim (was only stc < str_; the
    # docstring promised coverage >= random + 0.2)
    accepted = pred_acc >= 0.9 and stc < str_ and coverage_c >= coverage_r + 0.2
    report = {
        "protocol": "phase2_a3_worldmodel_drive_v1: body learns a deterministic transition (s'=(s+k) mod N) with a prediction head; prediction error (surprise) drives exploration to high-error states (Block-2 drive + Block-7 world model in the body). Criterion: prediction >= 0.9, curiosity coverage >= random + 0.2.",
        "seed": experiment_seed,
        "results": {
            "prediction_accuracy": round(pred_acc, 3),
            "curiosity_coverage": round(coverage_c, 3),
            "random_coverage": round(coverage_r, 3),
            "curiosity_steps_to_full": stc,
            "random_steps_to_full": str_,
            "min_visits": min(visits.values()),
            "max_visits": max(visits.values()),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-a3-worldmodel-drive.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-a3-worldmodel-drive.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-a3-worldmodel-drive.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))
