"""C2: tool_prior control-group experiment (2026-08-12).

Question: does the tool prior (learn 'which tool is effective under which
condition') ADD value to exploration, or does blind try-all suffice?

Control design:
  - WITH tool_prior: the agent maintains per-condition per-tool success
    statistics (ToolPrior) and picks the tool most likely to succeed given
    the current condition.
  - WITHOUT: try all tools every time (no learned effectiveness).

Environment (controlled): a condition space (e.g. content-type: text/numeric/
image).  Each condition has a 'correct' tool that succeeds (value+1); wrong
tools fail (value-0.5).  The tool_prior must learn the condition->tool map.

Criterion: WITH > WITHOUT by a meaningful margin, else the prior adds no
value over blind search.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.tool_prior import Tool, ToolPrior


CONDITIONS = ["text", "numeric", "image"]
TOOLS = ["parse", "sum", "render"]


def make_environment() -> dict:
    """condition -> correct tool (the ground truth the prior must learn)."""
    return {
        "text": "parse",
        "numeric": "sum",
        "image": "render",
    }


def make_tools_with_truth(truth: dict) -> dict:
    """Tools that succeed only on their condition (1.0) else fail (-0.5)."""
    tools = {}
    for tool_name in TOOLS:
        cond = next(c for c, t in truth.items() if t == tool_name)
        tools[tool_name] = Tool(
            tool_name,
            call=lambda args, _cond=cond: 1.0 if args.get("kind") == _cond else -0.5,
        )
    return tools


def run_episode(*, use_prior: bool, steps: int = 60, seed: int = 1) -> float:
    rng = np.random.RandomState(seed)
    truth = make_environment()
    tp = ToolPrior(make_tools_with_truth(truth)) if use_prior else None
    total = 0.0
    for t in range(steps):
        cond = CONDITIONS[t % 3]
        args = {"kind": cond}
        if use_prior:
            # epsilon-greedy: mostly exploit learned effectiveness, occasionally
            # explore a random tool (so every tool gets tried and can be learned)
            if rng.rand() < 0.2:
                choice = TOOLS[rng.randint(0, len(TOOLS))]
            else:
                rates = {tn: tp.success_rate(cond, tn) for tn in TOOLS}
                choice = max(TOOLS, key=lambda tn: rates[tn])
            outcome = tp.call(choice, cond, args)
        else:
            # TRUE blind: random tool (no alignment, no learned routing)
            choice = TOOLS[rng.randint(0, len(TOOLS))]
            outcome = 1.0 if choice == truth[cond] else -0.5
        total += outcome
    return total


def run(*, seeds=(1, 2, 3), steps: int = 60, artifact: Path | None = None) -> dict:
    results = {"with_prior": [], "without_prior": []}
    for seed in seeds:
        w = run_episode(use_prior=True, steps=steps, seed=seed)
        wo = run_episode(use_prior=False, steps=steps, seed=seed)
        results["with_prior"].append(round(w, 2))
        results["without_prior"].append(round(wo, 2))
    mean_w = float(np.mean(results["with_prior"]))
    mean_wo = float(np.mean(results["without_prior"]))
    report = {
        "protocol": "C2_tool_prior_control: learned tool effectiveness (with) vs blind try-all (without) over 3 conditions x 3 tools. If the prior cannot beat blind search HERE, it adds no value to the web pipeline.",
        "steps": steps,
        "results": results,
        "mean_with_prior": round(mean_w, 2),
        "mean_without_prior": round(mean_wo, 2),
        "gain": round(mean_w - mean_wo, 2),
        "conclusion": ("tool_prior ADDS value" if mean_w > mean_wo * 1.1
                       else "tool_prior does NOT add value over blind search"),
        "honest_boundary": "Blind try-all cycles tools uniformly, so it spends 1/3 of time on the wrong tool -> -0.5. The prior learns the map and prefers the right tool. Real-web analogue: fetch_text vs fetch_json vs fetch_image routing by content-type. This controls the toy map-learning.",
        "artifact": str(artifact or ROOT / "artifacts" / "c2-tool-prior.json"),
    }
    (artifact or ROOT / "artifacts" / "c2-tool-prior.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "c2-tool-prior.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
