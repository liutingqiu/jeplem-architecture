"""Phase-2 body v9 (2026-08-12): meta-controller with BALANCED source conflicts
-- window memory vs Bayesian belief.

v8 showed routing can EMERGE from reward (REINFORCE beats both fixed sources
on 2/3 seeds) but gains were only 1-2%: the ledger's unconditional latest-write
kept memory accurate in nearly every state bucket, so 'memory wrong, belief
right' conflicts were rare.  The probe showed a WINDOW memory (majority of the
last K observations) creates the missing scenes: after a single noisy frame the
window briefly holds the wrong value (belief, accumulating evidence, is right);
right after a flip the window flips fast (memory right, belief lags).  Both
directions of conflict are present, so the router has real work to do.

v9 replaces the ledger-backed memory with a window-majority memory and keeps
the Bayesian belief.  Everything else (REINFORCE, no labels, held-out eval) is
unchanged from v8.  Hypothesis: with balanced conflicts, RL routing beats both
fixed sources by a large margin (approaching the oracle, which is the fraction
of queries where at least one source is right).
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

from plwm.belief import CategoricalBelief

N_IDENTITIES = 2
RELIABILITY = 0.65
WINDOW = 4  # window memory holds the majority of the last WINDOW observations


class BodyRuntime:
    """Window memory (majority of last K obs) + Bayesian belief per identity."""

    def __init__(self, seed: int, window: int = WINDOW) -> None:
        self.seed = seed
        self.window = window
        self.beliefs: dict[int, CategoricalBelief] = {}
        self.recent: dict[int, list[int]] = {}  # identity -> recent values
        self.mem_since: dict[int, int] = {}  # steps since memory value last changed
        self.bel_since: dict[int, int] = {}  # steps since belief argmax last flipped
        self._last_mem: dict[int, int] = {}
        self._last_bel: dict[int, int] = {}
        self.lik = torch.tensor([[RELIABILITY, 1.0 - RELIABILITY],
                                 [1.0 - RELIABILITY, RELIABILITY]]).log()

    def _belief(self, identity: int) -> CategoricalBelief:
        if identity not in self.beliefs:
            self.beliefs[identity] = CategoricalBelief.uniform(1, 2)
        return self.beliefs[identity]

    def observe(self, identity: int, value: int) -> None:
        self.beliefs[identity] = self._belief(identity).observe(self.lik[:, value])
        self.recent.setdefault(identity, []).append(value)
        if len(self.recent[identity]) > self.window:
            self.recent[identity].pop(0)
        # recency of memory value change (observable: the body knows its own
        # window history).  Increment every identity's counter, reset the one
        # whose memory value just flipped.
        for i in self.recent:
            self.mem_since[i] = self.mem_since.get(i, 0) + 1
            self.bel_since[i] = self.bel_since.get(i, 0) + 1
        mv = self.memory_value(identity)
        bv = self.belief_value(identity)
        if mv != self._last_mem.get(identity):
            self.mem_since[identity] = 0
            self._last_mem[identity] = mv
        if bv != self._last_bel.get(identity):
            self.bel_since[identity] = 0
            self._last_bel[identity] = bv

    def memory_value(self, identity: int) -> int | None:
        r = self.recent.get(identity)
        if not r:
            return None
        return 1 if sum(r) >= len(r) / 2 else 0

    def belief_value(self, identity: int) -> int:
        return int(self._belief(identity).probs.argmax(-1).item())

    def belief_conf(self, identity: int) -> float:
        return float(self._belief(identity).confidence())

    def state_features(self, identity: int, obs_count: dict[int, int]) -> torch.Tensor:
        mv = self.memory_value(identity)
        bv = self.belief_value(identity)
        return torch.tensor([
            float(mv if mv is not None else 0.0),
            float(bv),
            self.belief_conf(identity),
            float(obs_count.get(identity, 0)) / 8.0,   # evidence count, normalized
            1.0 if mv is not None and mv == bv else 0.0,  # agreement
            0.0 if mv is None else 1.0,                  # memory present
            float(self.mem_since.get(identity, 0)) / 8.0,  # steps since memory flipped
            float(self.bel_since.get(identity, 0)) / 8.0,  # steps since belief flipped
        ], dtype=torch.float32)


class Router(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 2))
        self.value = nn.Linear(feat_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def collect_stream(seed: int, n_steps: int, flip_gap: int, window: int = WINDOW) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
    rng = random.Random(seed)
    rt = BodyRuntime(seed, window=window)
    obs_count: dict[int, int] = {}
    truth: dict[int, int] = {i: rng.randrange(2) for i in range(N_IDENTITIES)}
    feats: list[torch.Tensor] = []
    src_correct: list[torch.Tensor] = []
    truths: list[int] = []
    queried: list[int] = []
    next_flip = flip_gap
    for step in range(n_steps):
        for i in range(N_IDENTITIES):
            v = truth[i]
            shown = v if rng.random() < RELIABILITY else (1 - v)
            rt.observe(i, shown)
            obs_count[i] = obs_count.get(i, 0) + 1
        if step + 1 >= next_flip:
            fid = rng.randrange(N_IDENTITIES)
            truth[fid] = 1 - truth[fid]
            next_flip += flip_gap
        q = rng.randrange(N_IDENTITIES)
        mv = rt.memory_value(q)
        bv = rt.belief_value(q)
        mem_c = 1.0 if mv is not None and mv == truth[q] else 0.0
        bel_c = 1.0 if bv == truth[q] else 0.0
        feats.append(rt.state_features(q, obs_count))
        src_correct.append(torch.tensor([mem_c, bel_c], dtype=torch.float32))
        truths.append(truth[q])
        queried.append(q)
    return feats, src_correct, truths, queried


def train_rl(feats: list[torch.Tensor], src_correct: list[torch.Tensor],
             steps: int, seed: int, device: torch.device) -> Router:
    torch.manual_seed(seed)
    X = torch.stack(feats).to(device)
    # precompute correctness of each source for every query: [n, 2]
    correct = torch.stack(src_correct).to(device)
    router = Router(X.shape[1]).to(device)
    opt = torch.optim.AdamW(list(router.parameters()), lr=5e-3)
    n = len(X)
    batch = 512
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        xb = X[idx]
        logits = router.net(xb)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # [B]
        # vectorized reward: correct[action_indices, chosen_action]
        rewards = correct[idx, action]
        baseline = router.value(xb).squeeze(-1)
        adv = rewards - baseline.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -(dist.log_prob(action) * adv).mean()
        ent = -probs * torch.log(probs + 1e-8)
        ent = ent.sum(-1).mean()
        v = (rewards - baseline).square().mean()
        loss = pg + 0.5 * v - 0.05 * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return router


@torch.no_grad()
def evaluate(router: Router, feats: list[torch.Tensor], src_correct: list[torch.Tensor],
             device: torch.device) -> dict:
    X = torch.stack(feats).to(device)
    logits = router.net(X)
    route = logits.argmax(-1)
    routed_c = sum(1 for t, r in enumerate(route) if src_correct[t][r] == 1.0)
    n = len(feats)
    mem_acc = float(sum(src_correct[t][0] for t in range(n)) / n)
    bel_acc = float(sum(src_correct[t][1] for t in range(n)) / n)
    oracle = float(sum(1 for t in range(n) if src_correct[t][0] == 1.0 or src_correct[t][1] == 1.0) / n)
    return {"routed_acc": float(routed_c / n), "fixed_memory_acc": mem_acc,
            "fixed_belief_acc": bel_acc, "oracle_acc": oracle, "n_queries": n}


def run(*, train_steps: int = 20_000, n_steps: int = 90, flip_gap: int = 14,
        episodes: int = 64, window: int = WINDOW, experiment_seed: int = 41_123,
        device: torch.device | None = None, artifact: Path | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_feats, all_correct = [], []
    for ep in range(episodes):
        feats, correct, _, _ = collect_stream(experiment_seed * 31 + ep, n_steps, flip_gap, window)
        all_feats.extend(feats)
        all_correct.extend(correct)
    # split by EPISODE, not by index into the flattened feats list (each
    # episode contributes n_steps queries -- cutting the flat list at
    # 0.75*episodes sliced into the middle of an episode and gave the
    # trainer only ~48 queries).
    per_ep = n_steps
    cut = int(0.75 * episodes) * per_ep
    train_f = all_feats[:cut]
    train_c = all_correct[:cut]
    eval_f = all_feats[cut:]
    eval_c = all_correct[cut:]
    router = train_rl(train_f, train_c, train_steps, experiment_seed, device)
    metrics = evaluate(router, eval_f, eval_c, device)
    routed_wins = metrics["routed_acc"] > max(metrics["fixed_memory_acc"], metrics["fixed_belief_acc"])
    report = {
        "protocol": "phase2_body_v9: meta-controller with BALANCED conflicts -- window-majority memory vs Bayesian belief. Window memory holds a stale wrong value after a single noisy frame (belief right), flips fast after a world change (memory right, belief lags) -- both directions present. REINFORCE no-label routing, held-out eval. Hypothesis: RL routing >> fixed sources, approaching oracle.",
        "seed": experiment_seed,
        "results": {
            "routed_acc": round(metrics["routed_acc"], 3),
            "fixed_memory_acc": round(metrics["fixed_memory_acc"], 3),
            "fixed_belief_acc": round(metrics["fixed_belief_acc"], 3),
            "oracle_acc": round(metrics["oracle_acc"], 3),
            "routed_beats_best_fixed": bool(routed_wins),
            "routed_vs_oracle_gap": round(metrics["oracle_acc"] - metrics["routed_acc"], 3),
            "n_train_queries": len(train_f),
            "n_eval_queries": len(eval_f),
        },
        "accepted": bool(routed_wins and metrics["routed_acc"] >= max(metrics["fixed_memory_acc"], metrics["fixed_belief_acc"]) + 0.02),
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v9-router-window.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v9-router-window.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v9-router-window.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(json.dumps(run(train_steps=args.train_steps, window=args.window,
                         experiment_seed=args.experiment_seed, device=device,
                         artifact=args.artifact), indent=2))
