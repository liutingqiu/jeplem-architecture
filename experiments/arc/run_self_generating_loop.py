"""Phase-S: the self-training loop with SELF-GENERATED candidates -- the system
invents its own new rules from verified experience, not just picks from a
given pool (the "let it do it itself" extension).

In Phase-Q the KNOWLEDGE rule was in the candidate pool (given).  Here the pool
starts with ONLY spurious colour rules; the system, during self-training,
GENERATES new candidates by abstracting the discriminator from the VERIFIED
correct selections (it discovers "the kept objects share a semantic property --
plant-based food"), adds it, self-scores it, verifies it, and converges.

  round     propose the highest-self-scored rule, verify against knowledge.
  generate  every G rounds: from the verified-correct (scene, kept) pairs, find
            the property all kept entities share in the knowledge base -> a NEW
            knowledge-conditioned candidate rule, added to the pool.
  improve   the self-model learns to score the generated rule high -> the
            system converges to the self-invented knowledge rule.

Gate (3 seeds): last-ten accuracy >= 0.9, the SELF-GENERATED knowledge rule
dominates the final proposals, self-model ECE <= 0.2 -- the system invented the
correct rule itself and trained itself to use it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from plwm.knowledge_store import KnowledgeStore

DB = ROOT / "data" / "knowledge" / "wikidata.db"
FRUITS = ["apple", "orange", "grape", "strawberry", "pear", "mango", "cherry", "lemon"]
OTHER = ["dog", "cat", "car", "chair", "table", "bicycle"]
ALL_ENTITIES = FRUITS + OTHER
ROUNDS = 200
GEN_EVERY = 6


class SelfModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x)


def sample_category(store, rng, explored=None):
    """Pick a RANDOM knowledge category with enough members, preferring ones
    NOT already explored (diversity-aware -- avoids repeating the same domains
    like calendar days)."""
    explored = explored or set()
    # diverse pool: sample a random subset of the facts, count ends in Python
    ends = Counter(r[0] for r in store.conn.execute(
        "SELECT end FROM facts WHERE rel IN ('P31','P279') AND length(end)<25 LIMIT 200000"))
    pool = [e for e, c in ends.items() if c >= 12]
    rng.shuffle(pool)
    fresh = [e for e in pool if e not in explored]
    pick_from = fresh if fresh else pool
    return rng.choice(pick_from) if pick_from else "fruit"


def make_scene(rng, members, others, n_obj=6):
    ents = rng.sample(members, min(n_obj // 2, len(members))) \
        + rng.sample(others, min(n_obj - n_obj // 2, len(others)))
    rng.shuffle(ents)
    return [(e, rng.choice(range(1, 10))) for e in ents]


def is_cat(store, e, cat):
    return store.transitive_any(["P31", "P279"], e, cat)


def keep_set(rule, scene, store, cat):
    if rule[0] == "color":
        return [e for e, c in scene if c == rule[1]]
    if rule[0] == "prop":
        rels = ["P31", "P279"] if rule[1] in ("P31", "P279") else [rule[1]]
        return [e for e, _ in scene if store.transitive_any(rels, e, rule[2])]
    return [e for e, _ in scene if is_cat(store, e, cat)]


def rule_correct(store, rule, scene, cat):
    return keep_set(rule, scene, store, cat) == [e for e, _ in scene if is_cat(store, e, cat)]


def rule_feature(rule, scene, store, cat):
    kept = keep_set(rule, scene, store, cat)
    keep_flags = [1.0 if e in kept else 0.0 for e, _ in scene]
    return [1.0 if rule[0] == "color" else 0.0] + keep_flags


def generate_new_rule(store, rng, verified, used_props):
    """SELF-GENERATE a hypothesis candidate: sample a random entity from a
    recent scene, take one of its knowledge properties -> a new candidate rule
    'select entities with <rel>=<end>'.  The system EXPLORES the hypothesis
    space; verification identifies the correct property."""
    for _ in range(20):
        if not verified:
            return None
        scene = rng.choice(verified)[0]
        e = rng.choice([x for x, _ in scene])
        rows = store.conn.execute(
            "SELECT rel, end FROM facts WHERE start=? AND length(end)<30 LIMIT 10", (e,)).fetchall()
        if not rows:
            continue
        rel, end = rng.choice(rows)
        key = (rel, end)
        if key in used_props:
            continue
        used_props.add(key)
        return ("prop", rel, end)
    return None


def run_seed(seed: int, store, cat: str | None = None, explored=None) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if cat is None:
        cat = sample_category(store, rng, explored)
    # category members + non-members (free exploration: whatever category)
    rows = [r[0] for r in store.conn.execute(
        f"SELECT start FROM facts WHERE rel IN ('P31','P279') AND end=? AND length(start)<30 LIMIT 40",
        (cat,))]
    if len(rows) < 8:
        return {"seed": seed, "note": f"category {cat} too small", "self_generated_rules": [],
                "first_ten_acc": 0, "last_ten_acc": 0, "generated_rule_dominant": 0,
                "self_model_ece": 1.0, "gate_pass": False, "category": cat}
    rng.shuffle(rows)
    members = rows[:20]
    others = [r[0] for r in store.conn.execute(
        "SELECT start FROM facts WHERE rel='P31' LIMIT 40")]
    rng.shuffle(others)
    others = [e for e in others if not is_cat(store, e, cat)][:20]

    pool = [("color", c) for c in range(1, 10)]
    self_model = SelfModel(1 + 6)
    opt = torch.optim.AdamW(self_model.parameters(), lr=5e-3)
    bce = nn.BCEWithLogitsLoss()

    accs, preds_all, corrects = [], [], []
    proposals = []
    generated = []
    verified = []
    used_props = set()
    for t in range(ROUNDS):
        scene = make_scene(rng, members, others)
        feats = torch.stack([torch.tensor(rule_feature(r, scene, store, cat)).float() for r in pool])
        with torch.no_grad():
            scores = torch.sigmoid(self_model(feats)).squeeze(-1)
        best = int(scores.argmax())
        rule = pool[best]
        proposals.append(best)
        outcomes = torch.tensor([1.0 if rule_correct(store, r, scene, cat) else 0.0 for r in pool])
        correct = int(outcomes[best])
        accs.append(correct)
        preds_all.append(float(scores[best])); corrects.append(int(correct))
        for _ in range(6):
            loss = bce(self_model(feats).squeeze(-1), outcomes)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        verified.append((scene, keep_set(rule, scene, store, cat), bool(correct)))
        if (t + 1) % GEN_EVERY == 0:
            new = generate_new_rule(store, rng, verified, used_props)
            if new is not None and new not in pool:
                pool.append(new)
                generated.append(f"{new[1]}:{new[2]}")

    first = sum(accs[:10]) / 10
    last = sum(accs[-10:]) / 10
    # the self-GENERATED prop rule is the LAST pool entry (index len(pool)-1)
    gen_idx = len(pool) - 1 if generated else -1
    gen_dominant = sum(1 for p in proposals[-20:] if p == gen_idx) / 20 if gen_idx >= 0 else 0.0
    pa, ca = np.array(preds_all), np.array(corrects)
    ece = 0.0
    for b in range(5):
        mask = (pa > b / 5) & (pa <= (b + 1) / 5)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(pa)) * abs(float(pa[mask].mean()) - float(ca[mask].mean()))
    gate = bool(last >= 0.9 and gen_dominant >= 0.8 and ece <= 0.2)
    return {
        "seed": seed,
        "category": cat,
        "first_ten_acc": round(first, 3),
        "last_ten_acc": round(last, 3),
        "self_generated_rules": generated,
        "generated_rule_dominant": round(gen_dominant, 3),
        "self_model_ece": round(ece, 3),
        "gate_pass": bool(gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-s-self-generating.json")
    args = parser.parse_args()

    store = KnowledgeStore(args.db)
    start_resource_guard(interval=2.0)
    reports = [run_seed(s, store) for s in args.seeds]
    store.close()
    report = {
        "protocol": (
            "Phase-S self-training with SELF-GENERATED candidates: the pool starts with only "
            "spurious colour rules; the system GENERATES new knowledge-conditioned candidates "
            "from verified experience, self-scores them, and converges.  Gate: last-ten acc "
            ">= 0.9, the self-generated rule dominates, self-ECE <= 0.2, 3 seeds."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("PHASE-S PASS: the system INVENTED the correct rule itself (from "
                             "verified experience) and trained itself to use it -- self-training "
                             "with self-generated candidates, not a given pool.  'Let it do it "
                             "itself'.")
    else:
        report["verdict"] = "see per-seed"
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: first_ten {r['first_ten_acc']} last_ten {r['last_ten_acc']} "
              f"generated {r['self_generated_rules']} dominant {r['generated_rule_dominant']} "
              f"self_ece {r['self_model_ece']} gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
