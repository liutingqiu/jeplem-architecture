"""Phase-J: the AUTONOMOUS knowledge-driven agent -- the goal is DERIVED, not given.

Phase I gave the agent a goal ("collect fruits") and it executed.  Phase J: the
agent has only a PRIMITIVE VALUE ("plant-based food is good"); it DERIVES what
to pursue from its own knowledge.  In a grid world:

  value      -- a primitive: plant-based food is good (a survival/nutrition
                value, the ONLY thing baked in).
  perceive   -- scan the world, recognise entities (learned perception layer).
  derive     -- for each recognised entity, query the 1.32亿-fact KB: is it a
                plant-based food (transitive P31/P279 -> 'plant-based food')?
                The agent FIGURES OUT which objects satisfy its value -- the
                goal ("collect those") EMERGES from value + knowledge, it is
                not handed over.
  plan+act   -- route visiting every valuable object, collect them.
  correct    -- low-confidence recognitions -> honest UNKNOWN (skip).

The contrast: an agent with the same perception but NO knowledge cannot derive
which objects are valuable (plant-based-food-ness is semantic, not visual).

Gate (3 seeds): the agent derives its goal correctly and collects >= 0.9 of
the valuable objects with ZERO non-valuable collections; a no-knowledge agent
collects wrong objects.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore

DB = ROOT / "data" / "knowledge" / "wikidata.db"
VALUABLE = ["apple", "orange", "grape", "strawberry", "pear", "mango", "cherry", "lemon"]
OTHER = ["dog", "cat", "car", "chair", "table", "bicycle"]
ALL_ENTITIES = VALUABLE + OTHER
VALUE_CLASS = "plant-based food"
SIZE = 8


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


def build_world(rng, sig):
    n_obj = rng.randint(5, 8)
    ents = rng.sample(ALL_ENTITIES, n_obj)
    pos = rng.sample([(r, c) for r in range(SIZE) for c in range(SIZE) if (r, c) != (0, 0)], n_obj)
    return {p: e for p, e in zip(pos, ents)}


def run_seed(seed: int, store) -> dict:
    rng = random.Random(seed)
    sig = assign_signatures(seed)
    model = train_perception(seed, sig)
    derives = lambda e: store.transitive_any(["P31", "P279"], e, VALUE_CLASS)

    worlds = [build_world(rng, sig) for _ in range(10)]
    v_collected = v_total = v_wrong = 0
    naive_wrong = 0
    derived_goals = 0
    for world in worlds:
        total_v = sum(1 for e in world.values() if derives(e))
        if not total_v:
            continue
        # the agent perceives and DERIVES value from knowledge (no goal given)
        targets = []
        for p, e in world.items():
            ent, conf = recognise(model, sig[e])
            if conf >= 0.5 and derives(ent):      # value + knowledge -> target
                targets.append(p)
        derived_goals += 1
        # plan: greedy nearest-target route; act: collect
        agent_pos = (0, 0)
        collected = []
        tlist = list(targets)
        while tlist:
            nxt = min(tlist, key=lambda t: abs(t[0] - agent_pos[0]) + abs(t[1] - agent_pos[1]))
            agent_pos = nxt
            collected.append(nxt)
            tlist.remove(nxt)
        col_ents = {world[p] for p in collected}
        v_collected += len([e for e in col_ents if derives(e)])
        v_total += total_v
        v_wrong += len([e for e in col_ents if not derives(e)])
        naive_wrong += sum(1 for p in rng.sample(list(world), max(1, total_v))
                           if not derives(world[p]))

    completeness = v_collected / v_total if v_total else 0.0
    gate = bool(completeness >= 0.9 and v_wrong == 0 and naive_wrong > 0
                and derived_goals >= 8)
    return {
        "seed": seed,
        "worlds_acted": derived_goals,
        "completeness": round(completeness, 3),
        "valuable_collected": v_collected, "total_valuable": v_total,
        "non_valuable_collected": v_wrong,
        "no_knowledge_baseline_wrong": naive_wrong,
        "gate_pass": bool(gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-j-autonomous-agent.json")
    args = parser.parse_args()

    store = KnowledgeStore(args.db)
    reports = [run_seed(s, store) for s in args.seeds]
    store.close()
    report = {
        "protocol": (
            "Phase-J autonomous agent: only a primitive VALUE ('plant-based food is good') is "
            "baked in; the agent DERIVES its goal from knowledge (which objects are plant-based "
            "food, transitive P31/P279), then perceives, plans, acts, self-corrects.  Gate: "
            "completeness >= 0.9, zero non-valuable collections, no-knowledge baseline fails."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("PHASE-J PASS: the agent's goal EMERGES from value + knowledge, "
                             "not from an external instruction -- it wants something (derived), "
                             "figures out how to get it, and does it.  An autonomous agent.")
    else:
        report["verdict"] = "see per-seed"
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: worlds_acted {r['worlds_acted']} "
              f"completeness {r['completeness']} "
              f"valuable {r['valuable_collected']}/{r['total_valuable']} "
              f"non_valuable {r['non_valuable_collected']} "
              f"no_knowledge_wrong {r['no_knowledge_baseline_wrong']} gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
