"""Phase-2 full-subject v6: the body on a composite task -- identity flips
mid-stream create a REAL memory-vs-belief source conflict.

The v5 negative showed the body task was too simple: sources monotonic
(belief always better).  v6 adds MID-STREAM IDENTITY FLIPS: right after a
flip the MEMORY ledger (last observation) is correct while the BELIEF
(accumulated statistics) lags on the old value; after several new
observations the belief catches up.  Source quality now VARIES with state ->
the meta-controller (M5 mechanism) has real routing room.

Body: 2 identities, observation streams with a random-time flip on one
identity; memory (ledger) + Bayesian belief both updated; flip detection
(P4: consecutive observation/belief mismatches -> stale mark); query routing
(memory vs belief) by the meta-controller on body-internal state features.

Verdicts: routing accuracy, routed query accuracy vs fixed sources, P4 flip
detection, P1 ledger retrievability, P5 idle stability.
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
FLIP_THRESHOLD = 3


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
        self.max_conf: dict[int, float] = {}  # highest belief confidence reached

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
        # flip detection (P4): CONFIDENCE DROP -- the belief reached high
        # confidence then collapsed (Block-1 bifurcation semantics: strong
        # evidence flips a formerly-confident belief; a single noisy
        # observation cannot collapse an accumulated posterior)
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
            self.belief_conf(identity),
            1.0 if identity in self.marked else 0.0,       # flip marked?
            1.0 if mv is not None and mv == bv else 0.0,   # sources agree?
            0.0 if mv is None else 1.0,
        ])


class MetaRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def run(*, episodes: int = 48, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> dict:
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    samples: list[tuple[torch.Tensor, int]] = []
    cont_list, cons_list, mark_correct, mark_total = [], [], 0, 0
    for ep in range(episodes):
        rt = FlipBodyRuntime(seed=experiment_seed * 31 + ep)
        rng = random.Random(experiment_seed * 7919 + ep)
        observed: dict[int, int] = {}
        truth: dict[int, int] = {i: rng.randrange(2) for i in range(N_IDENTITIES)}
        flip_identity = rng.randrange(N_IDENTITIES)
        flip_at = rng.randrange(4, 10)  # flip mid-stream
        for step in range(12):
            for identity in range(N_IDENTITIES):
                value = truth[identity]
                shown = value if rng.random() < RELIABILITY else (1 - value)
                rt.observe_identity(identity, shown)
                observed[identity] = shown
            if step + 1 == flip_at:
                truth[flip_identity] = 1 - truth[flip_identity]  # world flips
            if step % 3 == 0:
                rt.idle()
        # P4: the flipped identity should be marked
        mark_correct += int(flip_identity in rt.marked)
        mark_total += 1
        # query samples: for each identity, which source is correct w.r.t.
        # the FINAL truth (post-flip, latest observations)
        for identity in range(N_IDENTITIES):
            final_truth = truth[identity]
            mv = rt.memory_value(identity)
            bv = rt.belief_value(identity)
            mem_c = mv == final_truth
            bel_c = bv == final_truth
            if mem_c != bel_c:
                samples.append((rt.state_features(identity), 0 if mem_c else 1))
        ok = all(rt.memory_value(i) is not None for i in observed)
        cont_list.append(int(ok))
        idle_ok = all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(rt.chain[1:], rt.chain)
        )
        cons_list.append(int(idle_ok))

    # meta-controller on the body's conflict samples
    router = MetaRouter()
    if len(samples) >= 8:
        X = torch.stack([s[0] for s in samples])
        y = torch.tensor([s[1] for s in samples], dtype=torch.long)
        opt = torch.optim.AdamW(router.parameters(), lr=1e-2)
        for _ in range(400):
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
        "protocol": "phase2_full_subject_v6: composite body task -- mid-stream identity flips create memory-vs-belief conflicts; meta-controller routes on body state; P4 flip detection, P1 retrievability, P5 idle stability. Criterion: routing >= 0.85, routed > best fixed, P4 >= 0.9, P1/P5 >= 0.9.",
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
        "artifact": str(artifact or ROOT / "artifacts" / "phase2-full-subject-v6-composite.json"),
    }
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v6-composite.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "phase2-full-subject-v6-composite.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(experiment_seed=args.experiment_seed, artifact=args.artifact), indent=2))
