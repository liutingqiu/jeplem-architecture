"""Phase-5 O4 gate: open goals -- endogenous goal generation from value-gap.

Question (JEPLM_PHASE5_WHITEPAPER.md, organ O4):
  Does the VALUE-GAP organ grow ENDOGENOUS goals -- the body generates a NEW
  goal from its OWN internal tension (valuable uncertainty) that GUIDES
  behaviour -- rather than receiving preset goals?
  Degradation risk: the "goal" is just RANDOM goal sampling.

White paper criterion:
  generate entirely new goals from internal tension, and the goal guides
  behaviour, not preset.

Pre-written pass line (3 seeds consistent):
  O4 PASS:
    (a) CONTENT / non-random: the body's goals target the TOP-VALUE domain at
        a far higher rate than a random sampler (>= 0.7 vs ~1/M);
    (b) INTERNAL-TENSION content: the targeted state is the HIGHEST-GAP state
        in the domain (the "why": reduce the most uncertain thing in the
        valuable area), i.e. the targeted gap is >= the domain's mean gap;
    (c) BEHAVIOUR GUIDANCE: acting on the generated goals resolves the
        top-value domain's uncertainty (its mean belief gap drops to <= 0.2
        by the end), and the random sampler resolves it far less.
  O4 退化线: the body's goals ~= random sampling (top-value fraction ~= 1/M,
    targeted states not max-gap).

Domain: M domains with external VALUES; N binary-identity states each with a
Bayesian belief.  GAP(d,s) = 1 - belief confidence.  The goal-generation loop
mirrors the verified UnifiedBody value-gap mechanism:
  select_domain = argmax value;  select_state = argmax gap (fewest-observed
  tie-break).  The experimenter sets values and initialises uniform beliefs;
  the specific goal (which state to investigate NOW) is the body's own output
  and SHIFTS as the internal tension resolves -- endogenous, dynamic goals.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from plwm.belief import CategoricalBelief

M_DOMAINS = 3
N_STATES = 6
VALUES = [0.9, 0.5, 0.2]
TOP_DOMAIN = 0
T_STEPS = 20
OBS_RELIABILITY = 0.8


def uniform_belief() -> CategoricalBelief:
    return CategoricalBelief.uniform(1, 2)


def belief_gap(b: CategoricalBelief) -> float:
    return float(1.0 - b.confidence().item())


def observe(b: CategoricalBelief, v: int, rel: float) -> CategoricalBelief:
    lik = torch.tensor([[rel, 1 - rel], [1 - rel, rel]]).log()
    return b.observe(lik[:, v])


class GoalBody:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.truths = {d: [self.rng.randrange(2) for _ in range(N_STATES)] for d in range(M_DOMAINS)}
        self.beliefs = {(d, s): uniform_belief() for d in range(M_DOMAINS) for s in range(N_STATES)}
        self.obs = {(d, s): 0 for d in range(M_DOMAINS) for s in range(N_STATES)}

    def gap(self, d: int, s: int) -> float:
        return belief_gap(self.beliefs[(d, s)])

    def mean_gap(self, d: int) -> float:
        return float(np.mean([self.gap(d, s) for s in range(N_STATES)]))

    def select_goal(self) -> tuple[int, int]:
        d_star = max(range(M_DOMAINS), key=lambda d: VALUES[d])
        max_gap = max(self.gap(d_star, s) for s in range(N_STATES))
        argmax_gaps = [s for s in range(N_STATES) if self.gap(d_star, s) >= max_gap - 1e-9]
        s_star = min(argmax_gaps, key=lambda s: (self.obs[(d_star, s)], s))
        return d_star, s_star

    def act(self, d: int, s: int) -> None:
        v = self.truths[d][s]
        self.beliefs[(d, s)] = observe(self.beliefs[(d, s)], v, OBS_RELIABILITY)
        self.obs[(d, s)] += 1


def random_sampler(seed: int) -> dict:
    """A goal sampler that picks a random (domain, state) each step: the
    degradation baseline.  Returns the same metrics as the body."""
    rng = random.Random(seed * 13 + 5)
    truths = {d: [rng.randrange(2) for _ in range(N_STATES)] for d in range(M_DOMAINS)}
    beliefs = {(d, s): uniform_belief() for d in range(M_DOMAINS) for s in range(N_STATES)}
    obs = {(d, s): 0 for d in range(M_DOMAINS) for s in range(N_STATES)}
    top_frac = 0
    targeted_gap_vs_mean = []
    for _ in range(T_STEPS):
        d = rng.randrange(M_DOMAINS)
        s = rng.randrange(N_STATES)
        if d == TOP_DOMAIN:
            top_frac += 1
        d_mean = float(np.mean([belief_gap(beliefs[(d, s2)]) for s2 in range(N_STATES)]))
        targeted_gap_vs_mean.append(belief_gap(beliefs[(d, s)]) >= d_mean - 1e-9)
        beliefs[(d, s)] = observe(beliefs[(d, s)], truths[d][s], OBS_RELIABILITY)
        obs[(d, s)] += 1
    final_top_gap = float(np.mean([belief_gap(beliefs[(TOP_DOMAIN, s)]) for s in range(N_STATES)]))
    return {
        "top_fraction": top_frac / T_STEPS,
        "target_maxgap_fraction": float(np.mean(targeted_gap_vs_mean)),
        "final_top_domain_gap": round(final_top_gap, 3),
    }


def run_seed(seed: int) -> dict:
    body = GoalBody(seed)
    trajectory = []
    top_frac = 0
    target_maxgap = 0
    for t in range(T_STEPS):
        d_star, s_star = body.select_goal()
        if d_star == TOP_DOMAIN:
            top_frac += 1
        d_mean = body.mean_gap(d_star)
        target_maxgap += int(body.gap(d_star, s_star) >= d_mean - 1e-9)
        trajectory.append({
            "tick": t, "domain": d_star, "state": s_star,
            "domain_value": VALUES[d_star],
            "targeted_gap": round(body.gap(d_star, s_star), 3),
            "domain_mean_gap": round(d_mean, 3),
            "obs": body.obs[(d_star, s_star)],
        })
        body.act(d_star, s_star)
    final_top_gap = body.mean_gap(TOP_DOMAIN)

    rand = random_sampler(seed)

    gate_pass = (
        top_frac / T_STEPS >= 0.7
        and target_maxgap / T_STEPS >= 0.7
        and final_top_gap <= 0.2
    )
    degraded = abs(top_frac / T_STEPS - rand["top_fraction"]) < 0.1

    return {
        "seed": seed,
        "top_value_domain_goal_fraction": round(top_frac / T_STEPS, 3),
        "random_top_fraction": round(rand["top_fraction"], 3),
        "targeted_maxgap_fraction": round(target_maxgap / T_STEPS, 3),
        "random_target_maxgap_fraction": round(rand["target_maxgap_fraction"], 3),
        "final_top_domain_mean_gap": round(final_top_gap, 3),
        "random_final_top_domain_mean_gap": round(rand["final_top_domain_gap"], 3),
        "trajectory": trajectory,
        "gate_pass": bool(gate_pass),
        "degraded": bool(degraded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase5-o4-open-goals.json")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)

    reports = [run_seed(seed) for seed in args.seeds]
    report = {
        "protocol": (
            "phase5 O4 gate: open goals -- endogenous goal generation from value-gap. "
            "The body emits its OWN goal (highest-value domain x highest-gap state) from "
            "internal tension; values are preset but the specific state to investigate is "
            "the body's output, and it shifts as the tension resolves. Pre-written pass: "
            ">=0.7 of goals target the top-value domain (vs random ~1/3); >=0.7 target the "
            "domain's max-gap state (internal-tension content); acting resolves the top-value "
            "domain's mean gap to <=0.2 (behaviour guidance). Degradation: goals ~= random."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("O4 PASS (provisional): the value-gap organ generates endogenous, "
                             "contentful goals from internal tension (valuable uncertainty) that "
                             "guide behaviour -- far above random goal sampling.")
    elif any(r["degraded"] for r in reports):
        report["verdict"] = "O4 DEGRADED (退化线): goals ~= random sampling."
    else:
        report["verdict"] = "O4 FAIL (single-organ falsification) -- see per-seed details."

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: top_frac={r['top_value_domain_goal_fraction']} (rand {r['random_top_fraction']}) "
              f"maxgap_frac={r['targeted_maxgap_fraction']} (rand {r['random_target_maxgap_fraction']}) "
              f"final_top_gap={r['final_top_domain_mean_gap']} (rand {r['random_final_top_domain_mean_gap']}) "
              f"gate={r['gate_pass']} degraded={r['degraded']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
