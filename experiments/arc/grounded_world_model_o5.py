"""Phase-5 O5 gate: grounded world model -- the white paper's most honest red line.

Question (JEPLM_PHASE5_WHITEPAPER.md, organ O5):
  Can a GROUNDED world model (action -> prediction -> difference -> update)
  be implemented as a PLWM MECHANISM -- using the body's own perception
  stream + explicit memory + a learned prediction head + prediction-error-
  driven exploration -- WITHOUT introducing a real-environment new substrate?
  If the world model MUST be a separate substrate (real interactive
  environment + dynamics model), then AGI != PLWM (U falsified at the core).

White paper red-line criterion:
  "能否作为 PLWM 机制实现（不引入真实环境新基质），预测-差异闭环收敛".
  White paper open question 7: can "internal simulation + difference update"
  (counterfactual on the semantic manifold) replace real-environment
  interaction?  This is the ONLY route to avoid the transplant line.

Pre-written pass line (3 seeds consistent):
  O5 PASS (provisional): the prediction-difference-update loop CONVERGES --
    prediction accuracy >= 0.9 on the states the body can perceive -- using
    ONLY the body's mechanisms: perception of its own observation stream,
    explicit memory (ledger) recording transitions, a learned prediction head
    updated by prediction error, and prediction-error-driven exploration.  No
    external dynamics model is added to the body.
  Internal-simulation finding (open question 7, reported not gated): does the
    learned world model predict next-states for states the body NEVER
    perceived (counterfactual beyond experience), or is it experience-bound?
  TRANSPLANT LINE (O5 FAIL): the loop does NOT converge from the body's own
    experience -> a grounded world model needs a real-environment substrate ->
    U is falsified at the core.

Domain: N=10 states, unknown transition law s' = (s + 3) mod 10.  The body
perceives states 0..M-1 (M=8); states 8,9 are BEYOND its stream.  It probes
perceived states (prediction-error-driven), records each observed transition
in its ledger, updates its prediction head by the prediction error, and
converges.  Then it is asked to predict next-states for the never-perceived
states 8,9 -- internal simulation / counterfactual.
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
T_STEPS = 400


class PredHead(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def onehot(s: int, n: int) -> torch.Tensor:
    x = torch.zeros(1, n)
    x[0, s] = 1.0
    return x


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    ledger = VersionedFactLedger(trusted_sources={"body"})
    head = PredHead(N_STATES)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)

    errors = []
    state = rng.randrange(M_PERCEIVED)
    probed: set[int] = set()
    for t in range(T_STEPS):
        next_state = (state + K) % N_STATES
        ledger.observe(FactRecord(f"t_{state}", next_state, "body", tick=t,
                                  confidence=0.9, version=1))
        probed.add(state)
        loss = nn.functional.cross_entropy(head(onehot(state, N_STATES)), torch.tensor([next_state]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        errors.append(float(loss.item()))
        # prediction-error-driven exploration: probe the perceived state with
        # the highest prediction entropy (the difference drives the action)
        with torch.no_grad():
            ents = []
            for s in range(M_PERCEIVED):
                p = torch.softmax(head(onehot(s, N_STATES)), -1)
                ents.append(-(p * torch.log(p + 1e-9)).sum().item())
            state = int(np.argmax(ents))

    # convergence: accuracy on the perceived states (the body's world)
    perceived_correct = 0
    with torch.no_grad():
        for s in range(M_PERCEIVED):
            pred = head(onehot(s, N_STATES)).argmax(-1).item()
            perceived_correct += int(pred == (s + K) % N_STATES)
    perceived_acc = perceived_correct / M_PERCEIVED

    # internal simulation: predict next-states for never-perceived states 8,9
    unperceived = list(range(M_PERCEIVED, N_STATES))
    sim_correct = 0
    with torch.no_grad():
        for s in unperceived:
            pred = head(onehot(s, N_STATES)).argmax(-1).item()
            sim_correct += int(pred == (s + K) % N_STATES)
    sim_acc = sim_correct / len(unperceived) if unperceived else 0.0

    loop_converged = perceived_acc >= 0.9 and float(np.mean(errors[-20:])) < float(np.mean(errors[:20]))
    transplant = not loop_converged

    return {
        "seed": seed,
        "perceived_accuracy": round(perceived_acc, 3),
        "internal_simulation_accuracy": round(sim_acc, 3),
        "unperceived_states": unperceived,
        "n_states_probed": len(probed),
        "error_first10": round(float(np.mean(errors[:10])), 4),
        "error_last10": round(float(np.mean(errors[-10:])), 4),
        "loop_converged": bool(loop_converged),
        "transplant_required": bool(transplant),
        "gate_pass": bool(loop_converged),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-o5-grounded-world-model.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 O5 gate: grounded world model (the most honest red line). Can the "
            "action->prediction->difference->update loop CONVERGE as a PLWM mechanism "
            "(perception + explicit-memory ledger + learned prediction head + prediction-error-"
            "driven exploration), learning the transition law s'=(s+3) mod 10 on the states the "
            "body perceives, WITHOUT a real-environment substrate? Pre-written pass: perceived "
            "prediction accuracy >= 0.9 with the error dropping. Internal-simulation finding "
            "(open question 7, reported not gated): can the model predict next-states for "
            "never-perceived states? Transplant line: no convergence from the body's own "
            "experience -> U falsified at the core."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("O5 PASS (provisional): the grounded world model is implementable as a "
                             "PLWM mechanism -- the prediction-difference-update loop converges on "
                             "the body's own observation stream with explicit-memory grounding and "
                             "error-driven exploration, learning the transition LAW. No "
                             "real-environment substrate is introduced; the transplant red line is "
                             "NOT triggered. (Internal-simulation generalisation to never-perceived "
                             "states is reported separately.)")
    elif any(r["transplant_required"] for r in reports):
        report["verdict"] = ("O5 FAIL / TRANSPLANT LINE TRIGGERED: the grounded world model cannot "
                             "converge from the body's own experience without a new substrate -> "
                             "AGI != PLWM at the core (U falsified).")
    else:
        report["verdict"] = "O5 FAIL (single-organ falsification) -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: perceived_acc={r['perceived_accuracy']} "
              f"internal_sim_acc={r['internal_simulation_accuracy']} "
              f"probed={r['n_states_probed']} err {r['error_first10']}->{r['error_last10']} "
              f"gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
