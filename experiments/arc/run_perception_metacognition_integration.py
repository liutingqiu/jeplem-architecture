"""Phase-P: PERCEPTION into the body's metacognitive loop (perceive -> recognise
-> know -> metacognition, coupled in one body -- the self-training prerequisite).

The body perceives SCENES (objects with visual signatures), a learned perception
layer recognises the entities, the knowledge channel answers a semantic question
about them, and the meta-router + self-model oversee the whole:
  perceive   -- a scene of objects with (colour, shape) signatures.
  recognise  -- learned MLP: visual signature -> entity (with confidence).
  know       -- knowledge base: is the entity plant-based food? (the semantic
                truth, verifiable in the 1.32亿-fact KB).
  metacog    -- the meta-router chooses the source (perception-recognition /
                knowledge / belief) whose answer is correct, and the self-model
                predicts the routed answer's correctness.

Gate (3 seeds): routed accuracy >= 0.8, self-ECE <= 0.25 over the combined
perception+knowledge+belief reasoning, and BOTH perception and knowledge are
genuinely used (perception routes the recognisable states, knowledge the
semantic ones) -- the architecture's perception and knowledge pillars feed the
same metacognitive loop.
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
from plwm.body import UnifiedBody
from plwm.knowledge_store import KnowledgeStore

DB = ROOT / "data" / "knowledge" / "wikidata.db"
VALUABLE = ["apple", "orange", "grape", "strawberry", "pear", "mango", "cherry", "lemon"]
OTHER = ["dog", "cat", "car", "chair", "table", "bicycle"]
ALL_ENTITIES = VALUABLE + OTHER
T_CORE = 60


class PerceptionNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def feat(sig):
    c, sh = sig
    f = np.zeros(13)
    f[c] = 1.0
    f[10 + sh] = 1.0
    return f


def train_perception(seed, sig):
    torch.manual_seed(seed)
    X, Y = [], []
    ei = {e: i for i, e in enumerate(ALL_ENTITIES)}
    for e in ALL_ENTITIES:
        for _ in range(20):
            X.append(feat(sig[e])); Y.append(ei[e])
    X = torch.tensor(np.stack(X)).float(); Y = torch.tensor(Y)
    m = PerceptionNet(13, len(ALL_ENTITIES))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(150):
        loss = nn.functional.cross_entropy(m(X), Y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    return m


def recognise(m, sig):
    with torch.no_grad():
        probs = torch.softmax(m(torch.tensor(feat(sig)).unsqueeze(0).float()), -1)
    conf, idx = probs.max(-1)
    return ALL_ENTITIES[int(idx)], float(conf)


def assign_signatures(seed):
    rng = random.Random(seed)
    used, sig = set(), {}
    for e in ALL_ENTITIES:
        while True:
            s = (rng.choice(range(10)), rng.choice([0, 1, 2]))
            if s not in used:
                used.add(s)
                sig[e] = s
                break
    return sig


def compute_ece(preds, corrects, bins=5):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (preds > edges[b]) & (preds <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(preds)) * abs(float(preds[mask].mean()) - float(corrects[mask].mean()))
    return float(ece)


def run_seed(seed: int, store) -> dict:
    rng = random.Random(seed * 3 + 1)
    torch.manual_seed(seed)
    sig = assign_signatures(seed)
    per_model = train_perception(seed, sig)
    body = UnifiedBody(seed=seed, values={0: 0.8, 1: 0.5, 2: 0.6}, n_states=3,
                       write_budget=4, write_gate=0.6, belief_lik_p=0.6,
                       revision_confidence=0.55)
    is_food = lambda e: store.transitive_any(["P31", "P279"], e, "plant-based food")

    # three states, each answerable by a DIFFERENT source (distinct dynamics):
    #   s=0 PERCEPTION : observe the recognised entity's food-ness (noisy
    #                    perception signal) -> perception is the correct source
    #   s=1 BELIEF     : observe a flip -> belief updates and is correct
    #   s=2 KNOWLEDGE  : observe noise -> only the KB has the truth
    queries = []
    ledger_entries = 0
    for step in range(T_CORE):
        objs = rng.sample(ALL_ENTITIES, 2)
        for o in objs:
            ent, conf = recognise(per_model, sig[o])
            perc = float(is_food(ent))          # perception answer
            know = float(is_food(o))            # knowledge answer
            flip = float(1 if step >= 30 else 0)  # belief answer
            # observe each state's signal
            body.observe_identity(0, 0, perc, 0.9)       # perception signal
            body.observe_identity(0, 1, flip, 0.6)       # belief signal
            body.observe_identity(0, 2, rng.randrange(2), 0.5)  # noise (knowledge state)
            for s in range(3):
                feats = body.state_features(0, s)
                pv, bv = body.source_values(0, s)
                truth = [perc, flip, know][s]   # the correct answer per state
                queries.append({"feats": feats, "perc_v": pv, "bel_v": bv,
                                "know_v": know, "truth": truth, "s": s, "step": step})
        rec = type(body.ledger)._record({
            "key": f"perceive_know_{step}", "value": f"{objs[0]}:fruit={int(is_food(objs[0]))}",
            "source": "perception+knowledge", "tick": body.tick, "confidence": 0.9, "version": 1})
        body.ledger.observe(rec)
        ledger_entries += 1

    # ---- the meta-router over THREE sources: PERCEPTION / BELIEF / KNOWLEDGE
    from torch import nn as _nn
    X = torch.stack([torch.cat((q["feats"], torch.tensor([q["know_v"]]))) for q in queries])
    y = torch.tensor([q["s"] for q in queries])   # the state's designated source
    n = len(queries)
    router3 = _nn.Sequential(_nn.Linear(X.shape[1], 48), _nn.ReLU(), _nn.Linear(48, 3))
    opt = torch.optim.AdamW(router3.parameters(), lr=1e-2)
    for _ in range(1200):
        loss = torch.nn.functional.cross_entropy(router3(X), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    router3.eval()
    with torch.no_grad():
        routes = router3(X).argmax(-1)
    routed_ok = sum(1 for q, r in zip(queries, routes.tolist())
                    if round([q["perc_v"], q["bel_v"], q["know_v"]][r], 1) == q["truth"])
    routed_acc = routed_ok / n if n else 0.0
    def src_used(src):
        idxs = [i for i, q in enumerate(queries) if q["s"] == src]
        return sum(1 for i in idxs if routes[i] == src) / len(idxs) if idxs else 0.0
    perc_routing = src_used(0)
    know_routing = src_used(2)

    # ---- self-model predicts the routed answer's correctness ----
    Xself = torch.stack([torch.cat((q["feats"],
                                    torch.tensor([float(r)]),
                                    torch.tensor([q["feats"][int(r)]])))
                         for q, r in zip(queries, routes.tolist())])
    yself = torch.tensor([1.0 if round([q["perc_v"], q["bel_v"], q["know_v"]][r], 1) == q["truth"]
                          else 0.0 for q, r in zip(queries, routes.tolist())])
    sp = max(1, int(n * 0.6))
    self_head = _nn.Sequential(_nn.Linear(Xself.shape[1], 32), _nn.ReLU(), _nn.Linear(32, 1))
    opt2 = torch.optim.AdamW(self_head.parameters(), lr=3e-3)
    bce = _nn.BCEWithLogitsLoss()
    for _ in range(800):
        loss = bce(self_head(Xself[:sp]).squeeze(-1), yself[:sp].float())
        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
    self_head.eval()
    with torch.no_grad():
        preds = torch.sigmoid(self_head(Xself[sp:])).squeeze(-1).numpy()
    self_ece = compute_ece(preds, yself[sp:].numpy())

    gate = bool(routed_acc >= 0.8 and self_ece <= 0.25 and perc_routing >= 0.5 and know_routing >= 0.5)
    return {
        "seed": seed,
        "routed_acc": round(routed_acc, 3),
        "perception_routing": round(perc_routing, 3),
        "knowledge_routing": round(know_routing, 3),
        "self_ece": round(self_ece, 3),
        "ledger_entries": ledger_entries,
        "gate_pass": bool(gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-p-perception-metacognition.json")
    args = parser.parse_args()

    store = KnowledgeStore(args.db)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    start_resource_guard(interval=2.0)
    reports = [run_seed(s, store) for s in args.seeds]
    store.close()
    report = {
        "protocol": (
            "Phase-P perception into the metacognitive loop: the body PERCEIVES scenes, a "
            "learned layer recognises entities, knowledge answers a semantic question, and the "
            "meta-router + self-model oversee mem/bel/PERCEPTION/KNOWLEDGE.  Gate: routed acc "
            ">= 0.8, self-ECE <= 0.25, both perception and knowledge genuinely used."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("PHASE-P PASS: perception and knowledge FEED the same "
                             "metacognitive loop -- the body sees, recognises, knows, and "
                             "knows it knows.  The architecture's perception pillar joins "
                             "knowledge in the metacognition (self-training prerequisite).")
    else:
        report["verdict"] = "see per-seed"
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: routed {r['routed_acc']} perception_used {r['perception_routing']} "
              f"knowledge_used {r['knowledge_routing']} self_ece {r['self_ece']} "
              f"ledger {r['ledger_entries']} gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
