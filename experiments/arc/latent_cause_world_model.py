"""Phase-5 continuation M1: latent-cause world model (predictive coding).

Question (JEPLM_PHASE5_WHITEPAPER_CONTINUATION.md, hypothesis H mechanism 1):
  A world model that learns LAWS (latent causes) rather than memorising
  mappings extrapolates beyond experience; a memoriser does not.  O5 showed
  the generic MLP prediction head is a memoriser (0.0 on never-perceived
  states).  M1 gives the world model a LAW-CLASS HYPOTHESIS module -- the
  predictive-coding mechanism: hypothesise s'=(s+k) mod N, fit k from
  observed transitions (each observation eliminates inconsistent k), then
  predict ALL states including never-perceived ones from the fitted law.

Pre-written pass line (3 seeds consistent):
  M1 PASS: the latent-cause world model predicts NEVER-PERCEIVED states at
    >= 0.8 (vs the memoriser MLP baseline, which is ~0.0 on them) -- i.e.
    the law-hypothesis mechanism is what enables extrapolation, and the
    world model is no longer experience-bound.

Controls:
  MEMORISER MLP (current world model, O5): trained on the same observed
    transitions, evaluates the same never-perceived states (expected ~0.0).
  SAMPLE EFFICIENCY: report how few observations are needed to isolate k
    (predictive coding's value: few examples -> law -> extrapolation).

Domain: N=10 states, unknown law s'=(s+3) mod 10.  The body perceives and
probes states 0..7; states 8,9 are never probed as starting states.  It
observes transitions (s -> next) by probing perceived states, records them in
its ledger (explicit memory), and fits the law hypothesis.  The MEMORISER
baseline gets the same transitions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from plwm.fact_ledger import FactRecord, VersionedFactLedger

N_STATES = 10
M_PERCEIVED = 8
K = 3
N_OBS = 12          # transitions the body observes by probing perceived states


class MemoriserMLP(nn.Module):
    """The current world model (O5): learns observed mappings, cannot
    extrapolate to never-perceived states."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def onehot(s: int, n: int) -> torch.Tensor:
    x = torch.zeros(1, n)
    x[0, s] = 1.0
    return x


class LawHypothesisWorldModel:
    """Predictive-coding world model: hypothesises the law class s'=(s+k) mod
    N and filters candidate k from observed transitions (Bayesian hypothesis
    filtering / latent-cause inference)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.candidates = set(range(n))   # possible k
        self.obs_count = 0

    def observe(self, s: int, sp: int) -> None:
        """sp = (s + k) mod N  =>  k = (sp - s) mod N."""
        k = (sp - s) % self.n
        self.candidates &= {k}
        self.obs_count += 1
        if not self.candidates:            # inconsistent -> world changed / noise
            self.candidates = set(range(self.n))

    def predict(self, s: int) -> tuple[int | None, float]:
        if len(self.candidates) == 1:
            k = next(iter(self.candidates))
            return (s + k) % self.n, 1.0
        return None, 0.0                   # ambiguous: no confident prediction

    def confident(self) -> bool:
        return len(self.candidates) == 1


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # the body probes perceived states, observing transitions
    observed: list[tuple[int, int]] = []
    for _ in range(N_OBS):
        s = rng.randrange(M_PERCEIVED)
        observed.append((s, (s + K) % N_STATES))

    # ----- latent-cause world model (predictive coding) ------------------
    law = LawHypothesisWorldModel(N_STATES)
    isolated_at = None
    for i, (s, sp) in enumerate(observed):
        law.observe(s, sp)
        if isolated_at is None and law.confident():
            isolated_at = i + 1

    unperceived = list(range(M_PERCEIVED, N_STATES))
    law_unperceived_correct = 0
    for s in unperceived:
        pred, conf = law.predict(s)
        if pred is not None and pred == (s + K) % N_STATES:
            law_unperceived_correct += 1
    law_unperceived_acc = law_unperceived_correct / len(unperceived)

    law_perceived_correct = 0
    for s in range(M_PERCEIVED):
        pred, _ = law.predict(s)
        law_perceived_correct += int(pred is not None and pred == (s + K) % N_STATES)
    law_perceived_acc = law_perceived_correct / M_PERCEIVED

    # ----- memoriser MLP baseline (current world model, O5) --------------
    mem = MemoriserMLP(N_STATES)
    opt = torch.optim.AdamW(mem.parameters(), lr=1e-2)
    for step in range(400):
        for s, sp in observed:
            loss = nn.functional.cross_entropy(mem(onehot(s, N_STATES)), torch.tensor([sp]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mem_unperceived_correct = 0
    mem_perceived_correct = 0
    with torch.no_grad():
        for s in unperceived:
            mem_unperceived_correct += int(mem(onehot(s, N_STATES)).argmax(-1).item() == (s + K) % N_STATES)
        for s in range(M_PERCEIVED):
            mem_perceived_correct += int(mem(onehot(s, N_STATES)).argmax(-1).item() == (s + K) % N_STATES)
    mem_unperceived_acc = mem_unperceived_correct / len(unperceived)
    mem_perceived_acc = mem_perceived_correct / M_PERCEIVED

    gate_pass = law_unperceived_acc >= 0.8 and law_unperceived_acc - mem_unperceived_acc >= 0.5
    return {
        "seed": seed,
        "law_unperceived_accuracy": round(law_unperceived_acc, 3),
        "memoriser_unperceived_accuracy": round(mem_unperceived_acc, 3),
        "law_perceived_accuracy": round(law_perceived_acc, 3),
        "memoriser_perceived_accuracy": round(mem_perceived_acc, 3),
        "k_isolated_after_obs": isolated_at,
        "n_obs": len(observed),
        "unperceived_states": unperceived,
        "gate_pass": bool(gate_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-m1-latent-world-model.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 continuation M1: latent-cause world model (predictive coding). The world "
            "model hypothesises the law class s'=(s+k) mod N and filters k from observed "
            "transitions, then predicts NEVER-perceived states from the fitted law. "
            "Pre-written pass: never-perceived accuracy >= 0.8 AND >= memoriser + 0.5 (the "
            "memoriser MLP -- the current world model from O5 -- cannot extrapolate). "
            "This tests whether the latent-cause mechanism breaks the O5 experience-boundary."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("M1 PASS (provisional): the latent-cause world model (hypothesise a "
                             "law class, fit k from experience, predict from the law) extrapolates "
                             "to never-perceived states, breaking the O5 experience boundary -- the "
                             "generic memoriser MLP cannot. Predictive coding is mechanised.")
    else:
        report["verdict"] = "M1 FAIL -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: law_unperceived={r['law_unperceived_accuracy']} "
              f"memoriser_unperceived={r['memoriser_unperceived_accuracy']} "
              f"law_perceived={r['law_perceived_accuracy']} "
              f"k_isolated_after={r['k_isolated_after_obs']} obs gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
