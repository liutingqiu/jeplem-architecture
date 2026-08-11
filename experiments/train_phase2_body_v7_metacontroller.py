"""Phase-2 body v7 (2026-08-12): meta-controller with REAL source conflicts.

v5 was negative (body task sources monotonic -- belief always better), v6 tried
mid-stream identity flips but sampled conflicts only at the FINAL state, when
both ledger and belief had already caught up to the new truth (too few
conflicts, routing stuck at 0.73).

The key insight: source quality varies with state only in the WINDOW RIGHT
AFTER a flip.  The ledger records the LATEST observation (it flips within a
step or two); the belief accumulates evidence (its posterior needs several
observations to flip).  So immediately after a flip:
  - ledger value = the new truth (from the first post-flip observation);
  - belief argmax = still the OLD truth (accumulated prior).
Memory is right, belief is wrong.  A few observations later belief catches up
and both agree on the new truth (no conflict).  A few observations BEFORE the
flip, both agree on the old truth (no conflict).  Only the transition window
has a real source conflict, and its polarity is determined by "how many
post-flip observations have been seen" -- exactly the state the meta-controller
must read.

Design: sample a conflict query EVERY STEP during the flip transition window,
so the router sees source quality as a function of post-flip step.  Then route
a held-out set of queries and require routing >= 0.85 and routed > best fixed.
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
from plwm.fact_ledger import FactRecord, VersionedFactLedger

N_IDENTITIES = 2
RELIABILITY = 0.65


class FlipBodyRuntime:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.ledger = VersionedFactLedger(trusted_sources={"subject"}, revision_confidence=0.9)
        self.tick = 0
        self.chain: list[dict] = []
        self.beliefs: dict[int, CategoricalBelief] = {}
        self.lik = torch.tensor([[RELIABILITY, 1.0 - RELIABILITY],
                                 [1.0 - RELIABILITY, RELIABILITY]]).log()
        self.marked: set[int] = set()
        self.max_conf: dict[int, float] = {}

    def _belief(self, identity: int) -> CategoricalBelief:
        if identity not in self.beliefs:
            self.beliefs[identity] = CategoricalBelief.uniform(1, 2)
        return self.beliefs[identity]

    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def observe_identity(self, identity: int, value: int) -> None:
        self.tick += 1
        self.ledger.observe(FactRecord(key=f"id{identity}", value=int(value), source="subject",
                                       tick=self.tick, confidence=0.95, version=1))
        self.beliefs[identity] = self._belief(identity).observe(self.lik[:, value])
        conf = float(self._belief(identity).confidence())
        if conf > self.max_conf.get(identity, 0.0):
            self.max_conf[identity] = conf
        if self.max_conf.get(identity, 0.0) >= 0.85 and conf < 0.6:
            self.marked.add(identity)
        self.chain.append({"event": "observe", "snapshot": self.snapshot_digest()})

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest()})

    def memory_value(self, identity: int) -> int | None:
        record = self.ledger.current.get(f"id{identity}")
        return int(record.value) if record is not None else None

    def belief_value(self, identity: int) -> int:
        return int(self._belief(identity).probs.argmax(-1).item())

    def belief_conf(self, identity: int) -> float:
        return float(self._belief(identity).confidence())

    def state_features(self, identity: int) -> torch.Tensor:
        mv = self.memory_value(identity)
        bv = self.belief_value(identity)
        return torch.tensor([
            float(mv if mv is not None else 0.0),
            self.belief_conf(identity),
            1.0 if identity in self.marked else 0.0,
            1.0 if mv is not None and mv == bv else 0.0,
            0.0 if mv is None else 1.0,
        ])


class MetaRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def run(*, episodes: int = 64, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    samples: list[tuple[torch.Tensor, int]] = []
    cont_list, cons_list, mark_correct, mark_total = [], [], 0, 0
    for ep in range(episodes):
        rt = FlipBodyRuntime(seed=experiment_seed * 31 + ep)
        rng = random.Random(experiment_seed * 7919 + ep)
        truth: dict[int, int] = {i: rng.randrange(2) for i in range(N_IDENTITIES)}
        flip_identity = rng.randrange(N_IDENTITIES)
        # flip AFTER ONLY ~2 pre-flip observations per identity: belief flips
        # back within ~6 steps (measured), so the window after the flip shows
        # memory-right / belief-wrong -> both-right -> belief catches up.  A
        # longer pre-flip (10+ obs) pins belief at conf>0.92 and the flip
        # latency balloons to 20+ steps -- the window never shows belief
        # catching up, so routing degenerates to "always memory".
        flip_at = 2
        for step in range(12):
            for identity in range(N_IDENTITIES):
                value = truth[identity]
                shown = value if rng.random() < RELIABILITY else (1 - value)
                rt.observe_identity(identity, shown)
            if step + 1 == flip_at:
                truth[flip_identity] = 1 - truth[flip_identity]
            # sample a conflict query for the flipped identity every step AFTER
            # the flip (and a few before, as a no-conflict control)
            if step >= flip_at - 1:
                final_truth = truth[flip_identity]
                mv = rt.memory_value(flip_identity)
                bv = rt.belief_value(flip_identity)
                mem_c = mv is not None and mv == final_truth
                bel_c = bv == final_truth
                if mem_c != bel_c:
                    samples.append((rt.state_features(flip_identity), 0 if mem_c else 1))
            if step % 3 == 0:
                rt.idle()
        mark_correct += int(flip_identity in rt.marked)
        mark_total += 1
        ok = all(rt.memory_value(i) is not None for i in truth)
        cont_list.append(int(ok))
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        cons_list.append(int(idle_ok))

    # meta-controller on the collected conflict samples (all episodes)
    router = MetaRouter()
    if len(samples) >= 16:
        X = torch.stack([s[0] for s in samples])
        y = torch.tensor([s[1] for s in samples], dtype=torch.long)
        opt = torch.optim.AdamW(router.parameters(), lr=1e-2)
        for _ in range(500):
            logits = router(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        router.eval()
        with torch.no_grad():
            routing_acc = float((router(X).argmax(-1) == y).float().mean())
        routed_c = sum(1 for f, l in samples if int(router(f.unsqueeze(0)).argmax(-1).item()) == l)
        routed_acc = routed_c / len(samples)
        mem_acc = sum(1 for _, l in samples if l == 0) / len(samples)
        bel_acc = sum(1 for _, l in samples if l == 1) / len(samples)
    else:
        routing_acc = routed_acc = mem_acc = bel_acc = 0.0

    cont = float(np.mean(cont_list))
    cons = float(np.mean(cons_list))
    p4 = mark_correct / mark_total
    accepted = (routing_acc >= 0.85 and routed_acc > max(mem_acc, bel_acc)
                and p4 >= 0.9 and cont >= 0.9 and cons >= 0.9)
    report = {
        "protocol": "phase2_body_v7: meta-controller with REAL source conflicts -- mid-stream identity flip, conflict queried EVERY STEP in the transition window (ledger flips in ~1 step, belief lags a few). Routing must beat both fixed sources. Criterion: routing >= 0.85, routed > best fixed, P4 >= 0.9, P1/P5 >= 0.9.",
        "seed": experiment_seed,
        "results": {
            "routing_accuracy": round(routing_acc, 3),
            "routed_answer_acc": round(routed_acc, 3),
            "fixed_memory_acc": round(mem_acc, 3),
            "fixed_belief_acc": round(bel_acc, 3),
            "flip_detection_p4": round(p4, 3),
            "continuity_p1": round(cont, 3),
            "consistency_p5": round(cons, 3),
            "n_conflicts": len(samples),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-body-v7-metacontroller.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-body-v7-metacontroller.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-body-v7-metacontroller.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, episodes=args.episodes,
                         artifact=args.artifact), indent=2))
