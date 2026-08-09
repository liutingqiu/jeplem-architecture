"""Aggregate DecideStream (u=3 g=4) results over 6 seeds (original 41123/222/333
+ the 444/555/666 retest) under the script's acceptance criterion and report
pass rates per model, replacing the 3-seed mean narrative.

Acceptance criterion (train_decide_stream_rl.py:497-500):
    plwm cross >= 0.6  and  plwm cross >= gru cross + 0.2

Note: the original 3 seeds ran the 3-model script (gru/evidence/plwm); the
retest 3 seeds ran the 6-model script (adds plwm_slots16/rmt_mem4/rmt_mem8).
per-model n reflects which seeds contain that model.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [41123, 222, 333, 444, 555, 666]
FILE = {41123: "stream-seed41123.json", 222: "stream-seed222.json", 333: "stream-seed333.json",
        444: "stream-u3g4-seed444.json", 555: "stream-u3g4-seed555.json", 666: "stream-u3g4-seed666.json"}


def main() -> None:
    query: dict[str, dict[int, float]] = {}
    cross: dict[str, dict[int, float]] = {}
    per_seed: dict[int, bool] = {}
    for seed in SEEDS:
        data = json.loads((ROOT / "artifacts" / FILE[seed]).read_text(encoding="utf-8"))
        per_seed[seed] = data.get("accepted", False)
        for r in data["results"]:
            query.setdefault(r["name"], {})[seed] = r["query_accuracy"]
            cross.setdefault(r["name"], {})[seed] = r["cross_episode_accuracy"]

    per_model: dict[str, dict] = {}
    for model in query:
        q, c = query[model], cross[model]
        seeds_have = sorted(q)
        vals_q = [q[s] for s in seeds_have]
        vals_c = [c[s] for s in seeds_have]
        if model == "gru_policy":
            pass_strict = [False] * len(seeds_have)
        else:
            pass_strict = [c[s] >= 0.6 and c[s] >= cross["gru_policy"][s] + 0.2
                           for s in seeds_have]
        per_model[model] = {
            "n": len(seeds_have),
            "seeds": seeds_have,
            "query": [round(v, 4) for v in vals_q],
            "query_mean": round(sum(vals_q) / len(vals_q), 4),
            "cross": [round(v, 4) for v in vals_c],
            "cross_mean": round(sum(vals_c) / len(vals_c), 4),
            "plwm_gate_pass": [bool(v) for v in pass_strict],
            "plwm_gate_pass_rate": f"{sum(pass_strict):.0f}/{len(seeds_have)}",
        }
    summary = {"protocol": "decide_stream_u3g4_6seed_aggregate_v1",
               "seeds": SEEDS,
               "per_seed_script_accepted": per_seed,
               "criterion": "cross>=0.6 and cross>=gru_cross+0.2 (script gate)",
               "note": "old 3 seeds: 3-model script; new 3 seeds (444/555/666): 6-model script (adds slots16/RMT)",
               "per_model": per_model}
    out = ROOT / "artifacts" / "stream-seed-aggregate.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
