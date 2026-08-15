"""#44 decisive ARC integration: mechanisms x real experience.

Phase-6 决战 (JEPLM_PHASE6_WHITEPAPER M5 + verdict): the G1-G4 mechanisms
(latent-cause creation M2, grounded manifold O1) applied to REAL ARC-1 tasks.
The static primitive library ceilings at ~3-5% (phase-3 G1: depth-2 skill
REUSE cannot exceed the vocabulary ceiling).  The question is whether NEW-
VOCABULARY creation -- a colour / permutation latent cause READ FROM THE
TASK'S OWN DEMOS when existing structure fails -- lifts real coverage.

RECORDING milestone (phase-6 discipline: real-ARC coverage is a record, not a
pass gate): before/after coverage, mechanism-only solutions, false facts
(must be 0), 3 seeds (exploration-order robustness).

Part A: coverage scan with the latent-cause pipeline vs the static library.
Part B: the same discovery inside UnifiedBody (v18 event-loop pattern),
        ledger traceability + v17 criteria regression.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from plwm.body import UnifiedBody
from experiments.arc.grid import dims, equal
from experiments.arc import induction
from experiments.arc.grounded_manifold_o1 import (
    manifold_generate, adjudicate, color_map, read_cmap, POS, SPATIAL_FN,
    spatial_candidates)
from experiments.arc.new_latent_cause import (
    classify_color_op as nc_classify_color_op,
    apply_color_op, infer_permutation, apply_perm,
    KNOWN_SPATIAL, KNOWN_COLOR)


# --------------------------------------------------------------------- #
# latent-cause pipeline (mechanism): NEW vocabulary from a task's demos   #
# --------------------------------------------------------------------- #
def latent_cause_solve(demos, rng):
    """Try the mechanism pipeline on ONE task's demos.  Returns
    (candidate_fn, info) confirmed by the explicit core, or None (hold).
    Order: grounded manifold -> custom colour latent cause -> permutation
    latent cause.  Ambiguity (>=2 consistent spatial ops) HOLDS (no candidate
    -> no false fact can be confirmed, phase-5 O1 ambiguity lesson)."""
    # 1. grounded manifold: known spatial+colour classes with analogy
    #    extrapolation (flip_h->flip_v, rotate90->rotate180; recolor->keep)
    gen = manifold_generate(demos, KNOWN_SPATIAL, KNOWN_COLOR)
    if gen is not None:
        cand, info = gen
        if adjudicate(cand, demos) == "confirm":
            return cand, {**info, "source": "manifold"}

    # 2. custom colour latent cause: a per-pixel colour function NOT in the
    #    known classes (M2a).  Only on UNambiguous tasks (exactly one
    #    consistent spatial op) -- else the demos underdetermine the colour op
    #    and the adjudicator cannot prevent a wrong-on-test candidate.
    cand_ops = spatial_candidates(KNOWN_SPATIAL)
    consistent = [n for n in cand_ops if color_map(demos, POS[n])[0]]
    if len(consistent) == 1:
        sname = consistent[0]
        cmap = read_cmap(demos, POS[sname])
        cls = nc_classify_color_op(cmap, KNOWN_COLOR, allow_custom=True)
        if cls is not None:
            cname, params, novel = cls
            if novel:
                def cand_custom(g, s=sname, cn=cname, p=params):
                    return apply_color_op(cn, p, SPATIAL_FN[s](g))
                if adjudicate(cand_custom, demos) == "confirm":
                    return cand_custom, {"spatial": sname, "color": cname,
                                         "source": "custom_color", "novel": novel}

    # 3. permutation latent cause: a NEW position permutation inferred from
    #    cell correspondence when NO known spatial op explains the demos (M2b).
    if len(consistent) == 0:
        perm = infer_permutation(demos)
        if perm is not None:
            def cand_perm(g, p=perm):
                return apply_perm(g, p)
            if adjudicate(cand_perm, demos) == "confirm":
                return cand_perm, {"source": "permutation", "perm_size": len(perm)}

    return None


# --------------------------------------------------------------------- #
# Part A: coverage scan                                                  #
# --------------------------------------------------------------------- #
def scan_with_mechanism(tasks, rng, use_mechanism: bool = True) -> dict:
    n = len(tasks)
    static_fit = mech_fit = 0
    static_test = mech_test = 0
    mech_false = 0
    sources: dict[str, int] = {}
    tasks_with_mech = []
    for i, (tid, task) in enumerate(tasks.items()):
        if i % 100 == 0:
            print(f"  ...{i}/{n} (static_fit={static_fit} mech_fit={mech_fit})", flush=True)
        demos = [(g["input"], g["output"]) for g in task["train"]]
        test_in = task["test"][0]["input"]
        test_out = task["test"][0]["output"]

        # static primitive library first (the phase-3 ceiling)
        prog = None
        try:
            prog = induction.search_program(demos, max_depth=2)
        except Exception:
            prog = None
        if prog is not None:
            static_fit += 1
            fn = prog[1]
            if equal(fn(test_in), test_out):
                static_test += 1
            continue  # mechanism only runs where the static library fails

        # mechanism: NEW vocabulary for tasks the static library cannot fit
        if use_mechanism:
            try:
                out = latent_cause_solve(demos, rng)
            except Exception:
                out = None
            if out is not None:
                cand, info = out
                mech_fit += 1
                src = info.get("source")
                sources[src] = sources.get(src, 0) + 1
                tasks_with_mech.append(tid)
                if equal(cand(test_in), test_out):
                    mech_test += 1
                else:
                    mech_false += 1  # confirmed on demos, wrong on test = false fact

    return {
        "n_tasks": n,
        "static_train_fit": static_fit,
        "static_test_correct": static_test,
        "mech_train_fit": mech_fit,
        "mech_test_correct": mech_test,
        "combined_test_correct": static_test + mech_test,
        "mech_false_facts": mech_false,
        "mech_sources": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "static_rate": round(static_test / n, 3) if n else 0.0,
        "mech_only_rate": round(mech_test / n, 3) if n else 0.0,
        "combined_rate": round((static_test + mech_test) / n, 3) if n else 0.0,
    }


# --------------------------------------------------------------------- #
# Part B: the discovery pipeline inside UnifiedBody (v18 event loop)     #
# --------------------------------------------------------------------- #
N_STATES = 2
VALUES = {0: 0.8, 1: 0.5}
T_CORE = 40
FLIP_STEP = 20
RELIABILITY = {0: 0.9, 1: 0.6}


def compute_ece(preds, corrects, bins=5):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (preds > edges[b]) & (preds <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(preds)) * abs(float(preds[mask].mean()) - float(corrects[mask].mean()))
    return float(ece)


def run_body_seed(seed: int, arc_tasks: dict, arc_n: int = 60) -> dict:
    rng = random.Random(seed * 3 + 1)
    torch.manual_seed(seed)
    body = UnifiedBody(seed=seed, values=VALUES, n_states=N_STATES,
                       write_budget=4, write_gate=0.6, belief_lik_p=0.6,
                       revision_confidence=0.55)

    truths = {d: [rng.randrange(2) for _ in range(N_STATES)] for d in VALUES}

    def truth_at(d, s, step):
        if d == 0 and s == 0 and step >= FLIP_STEP:
            return 1 - truths[d][s]
        return truths[d][s]

    task_ids = list(arc_tasks)[:arc_n]
    queries = []
    arc_test = arc_false = arc_tot = ledger_written = 0
    provenance: dict = {}
    task_idx = 0

    def arc_discovery_task():
        nonlocal arc_test, arc_false, arc_tot, ledger_written, task_idx
        if task_idx >= len(task_ids):
            return
        tid = task_ids[task_idx]
        task = arc_tasks[tid]
        demos = [(g["input"], g["output"]) for g in task["train"]]
        test_in = task["test"][0]["input"]
        test_out = task["test"][0]["output"]
        task_idx += 1
        try:
            out = latent_cause_solve(demos, rng)
        except Exception:
            out = None
        if out is None:
            return
        cand, info = out
        arc_tot += 1
        if equal(cand(test_in), test_out):
            arc_test += 1
        else:
            arc_false += 1
        # the confirmed transform fact is written to the SHARED ledger
        src = info.get("source", "unknown")
        key = f"arc_lc_{tid}"
        rec = type(body.ledger)._record({
            "key": key, "value": src, "source": "body",
            "tick": body.tick, "confidence": 0.9, "version": 1})
        body.ledger.observe(rec)
        ledger_written += 1
        provenance[key] = {"task": tid, "source": src}

    for step in range(T_CORE):
        for d in VALUES:
            for s in range(N_STATES):
                body.observe_identity(d, s, truth_at(d, s, step), RELIABILITY[d])
        arc_discovery_task()
        for d in VALUES:
            for s in range(N_STATES):
                feats = body.state_features(d, s)
                mem_v, bel_v = body.source_values(d, s)
                queries.append({"feats": feats, "mem_v": mem_v, "bel_v": bel_v,
                                "truth": truth_at(d, s, step)})

    # ---- score the v17 core (mirror of the v18 G3 protocol) ----
    disagree = [q for q in queries if q["mem_v"] != q["bel_v"]]
    rng.shuffle(disagree)
    router_acc = routed_acc = 0.0
    self_ece = naive_ece = 0.0
    if disagree:
        split = max(1, int(len(disagree) * 0.6))
        train_d, eval_d = disagree[:split], disagree[split:]
        if eval_d:
            Xtr = torch.stack([q["feats"] for q in train_d])
            ytr = torch.tensor([0 if q["mem_v"] == q["truth"] else 1 for q in train_d],
                               dtype=torch.long)
            opt = torch.optim.AdamW(body.router.parameters(), lr=1e-2)
            for _ in range(500):
                loss = torch.nn.functional.cross_entropy(body.router(Xtr), ytr)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            body.router.eval()
            Xev = torch.stack([q["feats"] for q in eval_d])
            with torch.no_grad():
                routes = body.router(Xev).argmax(-1)
            labels = torch.tensor([0 if q["mem_v"] == q["truth"] else 1 for q in eval_d])
            router_acc = float((routes == labels).float().mean())
            routed_acc = sum(1 for q, r in zip(eval_d, routes.tolist())
                             if (q["bel_v"] if r else q["mem_v"]) == q["truth"]) / len(eval_d)
        qs = list(queries); rng.shuffle(qs)
        sp = max(1, int(len(qs) * 0.6))
        tr, ev = qs[:sp], qs[sp:]
        if tr and ev:
            with torch.no_grad():
                tr_routes = body.router(torch.stack([q["feats"] for q in tr])).argmax(-1)
                ev_routes = body.router(torch.stack([q["feats"] for q in ev])).argmax(-1)
            Xself = torch.stack([torch.cat((q["feats"],
                                            torch.tensor([float(r)]),
                                            torch.tensor([q["feats"][int(r)]])))
                                 for q, r in zip(tr, tr_routes.tolist())])
            yself = torch.tensor([1.0 if (q["bel_v"] if r else q["mem_v"]) == q["truth"]
                                  else 0.0 for q, r in zip(tr, tr_routes.tolist())])
            opt2 = torch.optim.AdamW(body.self_head.parameters(), lr=3e-3)
            for _ in range(600):
                loss = torch.nn.functional.binary_cross_entropy(body.self_head(Xself).squeeze(-1), yself)
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
            body.self_head.eval()
            Xev2 = torch.stack([torch.cat((q["feats"],
                                           torch.tensor([float(r)]),
                                           torch.tensor([q["feats"][int(r)]])))
                                for q, r in zip(ev, ev_routes.tolist())])
            with torch.no_grad():
                preds = body.self_head(Xev2).squeeze(-1).numpy()
            corrects = np.array([1.0 if (q["bel_v"] if r else q["mem_v"]) == q["truth"]
                                 else 0.0 for q, r in zip(ev, ev_routes.tolist())])
            self_ece = compute_ece(preds, corrects)
            naive_ece = compute_ece(np.array([q["feats"][int(r)].item() for q, r in zip(ev, ev_routes.tolist())]), corrects)

    revision_detected = 0 in body.marked
    with torch.no_grad():
        all_routes = body.router(torch.stack([q["feats"] for q in queries])).argmax(-1).tolist()
    routed_all = sum(1 for q, r in zip(queries, all_routes)
                     if (q["bel_v"] if r else q["mem_v"]) == q["truth"]) / len(queries)

    arc_rate = arc_test / arc_tot if arc_tot else 0.0
    return {
        "seed": seed,
        "belief_query_acc": round(routed_all, 3),
        "revision_detected": bool(revision_detected),
        "router_acc": round(router_acc, 3),
        "self_ece": round(self_ece, 3),
        "arc_tasks_confirmed": arc_tot,
        "arc_tasks_test_correct": arc_test,
        "arc_test_rate": round(arc_rate, 3),
        "arc_false_facts": arc_false,
        "ledger_written": ledger_written,
        "provenance_traceable": bool(ledger_written) and len(provenance) == ledger_written,
        "ledger_sample": [{"key": k, **v} for k, v in list(provenance.items())[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B"], default="A",
                        help="A: coverage scan; B: UnifiedBody integration")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "arc")
    parser.add_argument("--files", nargs="*", default=["arc1_training.json", "arc1_evaluation.json"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--arc-n", type=int, default=60, help="Part B: tasks the body processes")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase6-arc-integration.json")
    parser.add_argument("--mechanism", action="store_true", default=True,
                        help="apply the latent-cause pipeline (default on)")
    parser.add_argument("--no-mechanism", dest="mechanism", action="store_false")
    args = parser.parse_args()

    start_resource_guard(interval=2.0)
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 4))

    if args.part == "A":
        data = {name: json.loads((args.data_dir / name).read_text(encoding="utf-8"))
                for name in args.files}
        per_seed = []
        for seed in args.seeds:
            rng = random.Random(seed)
            res = {"seed": seed}
            for name in args.files:
                res[name] = scan_with_mechanism(data[name], rng, use_mechanism=args.mechanism)
            per_seed.append(res)
            print(f"seed {seed}: " + " | ".join(
                f"{n}: static {res[n]['static_rate']} mech {res[n]['mech_only_rate']} "
                f"combined {res[n]['combined_rate']} false {res[n]['mech_false_facts']}"
                for n in args.files))

        all_false_zero = all(r[n]["mech_false_facts"] == 0 for r in per_seed for n in args.files)
        report = {
            "part": "A",
            "protocol": (
                "phase6 decisive ARC integration (#44): apply latent-cause creation (M2) + "
                "grounded manifold (O1) to REAL ARC-1 tasks.  Static primitive library ceilings "
                "at ~3-5%; the mechanism reads a NEW colour/permutation latent cause from a "
                "task's own demos when existing structure fails.  RECORDING milestone: "
                "before/after coverage + mechanism-only solutions + zero false facts, 3 seeds."
            ),
            "baseline_note": (
                "arc1-coverage.json static baseline: training 0.0325 test_correct, "
                "evaluation 0.0025.  mech_only_rate = tasks solved ONLY by the mechanism."
            ),
            "false_facts_zero_all_seeds": bool(all_false_zero),
            "per_seed": per_seed,
        }
        print("VERDICT: ", "zero false facts all seeds" if all_false_zero else "FALSE FACTS PRESENT")
    else:
        train_tasks = json.loads((args.data_dir / "arc1_training.json").read_text(encoding="utf-8"))
        per_seed = [run_body_seed(seed, train_tasks, arc_n=args.arc_n) for seed in args.seeds]
        core_ok = all(r["belief_query_acc"] >= 0.8 and r["revision_detected"]
                      and r["router_acc"] >= 0.7 and r["self_ece"] <= 0.15 for r in per_seed)
        false_zero = all(r["arc_false_facts"] == 0 for r in per_seed)
        report = {
            "part": "B",
            "protocol": (
                "phase6 decisive ARC integration (#44) Part B: the latent-cause discovery "
                "pipeline runs INSIDE UnifiedBody (v18 event-loop pattern) on real ARC-1 "
                "training tasks, confirmed transform facts written to the SHARED ledger.  "
                "v17-core criteria (belief/revision/router/self-ECE) must hold alongside the "
                "ARC discovery -- no interference, zero false facts, provenance traceable."
            ),
            "false_facts_zero_all_seeds": bool(false_zero),
            "core_holds_all_seeds": bool(core_ok),
            "per_seed": per_seed,
        }
        print("VERDICT:", "core holds + zero false facts" if (core_ok and false_zero)
              else ("core OK, false facts present" if core_ok else "core degraded"))

    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.artifact}")


if __name__ == "__main__":
    main()
