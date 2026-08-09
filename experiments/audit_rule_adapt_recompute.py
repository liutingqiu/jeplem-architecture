"""Recompute rule-adaptation results under two explicit criteria (6 seeds).

Criterion A (strict gate, matches the script's accepted logic):
    final accuracy >= 0.7 and >= gru_final + 0.2
Criterion B (improvement / "adapted" as used in the session summary):
    final block accuracy > first block accuracy (curve rose)

Outputs a single summary JSON with per-seed finals and per-criterion pass
rates for both the default PLWM and the slots16 configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [222, 333, 444, 555, 666, 777]
MODELS = ["gru_policy", "evidence_policy", "plwm_policy", "plwm_slots16_policy"]


def strict_pass(final: float, gru_final: float) -> bool:
    return final >= 0.7 and final >= gru_final + 0.2


def main() -> None:
    rows: dict[str, list[float]] = {m: [] for m in MODELS}
    curves: dict[str, list[list[float]]] = {m: [] for m in MODELS}
    script_gates: list[bool] = []
    for seed in SEEDS:
        path = ROOT / "artifacts" / f"rule-adapt-u4g4-seed{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        script_gates.append(data.get("accepted", False))
        for model in MODELS:
            curve = data["adaptation_curve"][model]
            curves[model].append(curve)
            rows[model].append(curve[-1])

    summary: dict = {"protocol": "rule_adapt_recompute_v1", "seeds": SEEDS,
                     "criteria": {"A_strict": "final>=0.7 and final>=gru+0.2",
                                  "B_improve": "final block > first block"}}
    per_model: dict[str, dict] = {}
    for model in MODELS:
        finals = rows[model]
        rises = [1.0 if c[-1] > c[0] else 0.0 for c in curves[model]]
        if model == "gru_policy":
            pass_strict = [1.0 if strict_pass(f, f) else 0.0 for f in finals]
        else:
            pass_strict = [1.0 if strict_pass(finals[i], rows["gru_policy"][i]) else 0.0
                           for i in range(len(SEEDS))]
        per_model[model] = {
            "finals": [round(f, 4) for f in finals],
            "mean_final": round(sum(finals) / len(finals), 4),
            "strict_pass": [bool(v) for v in pass_strict],
            "strict_pass_rate": f"{sum(pass_strict):.0f}/{len(SEEDS)}",
            "improve_pass": [bool(v) for v in rises],
            "improve_pass_rate": f"{sum(rises):.0f}/{len(SEEDS)}",
        }
    summary["per_model"] = per_model
    summary["script_accepted"] = script_gates
    summary["script_accepted_rate"] = f"{sum(script_gates):.0f}/{len(SEEDS)}"

    out = ROOT / "artifacts" / "rule-adapt-recompute-6seed.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
