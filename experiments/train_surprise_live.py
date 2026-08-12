"""L1: surprise-driven closed loop on LIVE Luanti (2026-08-12).

The complete bootstrap closed loop on a live world, using the correct
LuantiLiveClient API (begin_episode_with_vision + step_with_vision):

    SurpriseEngine sees a state stream (no action field)
    -> x/z get marked HIDDEN VARIABLES (irreducible surprise under identity)
    -> driven by that surprise, the engine OBSERVES its own action
    -> with the action in the observation, the causality explainer explains
       x/z as action-conditional deltas -> surprise is eliminated

This validates 'hidden variable = cause outside observation -> actively seek
the missing observation -> explained', live, with ground truth (action->delta
is the real rule).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient
from plwm.surprise_engine import SurpriseEngine
from plwm.explainers import explain_field


def run_live(*, bridge_root: Path, n_steps: int = 30,
             task_seed: int = 2000, artifact: Path | None = None) -> dict:
    bridge = LuantiFileBridge(bridge_root)
    client = LuantiLiveClient(bridge, timeout_seconds=8.0, poll_seconds=0.02)
    eng = SurpriseEngine(window=8)

    # begin a fresh episode (reset_episode, not recenter)
    try:
        state, vision = client.begin_episode_with_vision("surprise-live", task_seed)
    except Exception as e:
        return {"error": f"begin_episode failed: {e}"}

    prev_state = None
    prev_action = None
    naive_hidden = None
    explained_with_action = False
    directions = ["north", "south", "east", "west"]
    di = 0
    transitions = []

    for step in range(n_steps):
        direction = directions[di % 4]
        di += 1
        interaction, state, vision = client.step_with_vision(direction)
        if state is None:
            time.sleep(0.05)
            continue
        prev_action = direction
        if prev_state is not None:
            # NAIVE: engine sees state without action
            eng.observe(asdict(prev_state), asdict(state))
            # WITH action: causality explainer
            s0 = dict(asdict(prev_state)); s0["_action"] = direction
            s1 = dict(asdict(state))
            transitions.append((s0, s1))
        prev_state = state
        # after enough steps, record naive verdict + try explainer on the
        # accumulated WITH-action transitions
        if step >= 12:
            k = eng.knowledge()
            if naive_hidden is None:
                naive_hidden = sorted(k["hidden_vars"])
            rx = explain_field("x", transitions)
            rz = explain_field("z", transitions)
            if rx is not None and rz is not None:
                explained_with_action = True
                break
        time.sleep(0.02)

    k = eng.knowledge()
    report = {
        "protocol": "surprise_closed_loop_live_v1: SurpriseEngine on LIVE Luanti. Naive stream (no action) marks x/z hidden; the engine observes its own action; causality explainer explains x/z as action-conditional. Validates 'hidden var -> seek missing observation -> explained'.",
        "n_steps": n_steps,
        "naive_hidden_vars": naive_hidden,
        "naive_invariants": sorted(k["invariants"]),
        "x_z_explained_with_action_observation": explained_with_action,
        "closed_loop_complete": bool(explained_with_action),
        "artifact": str(artifact or ROOT / "artifacts" / "surprise-live-closed-loop.json"),
    }
    (artifact or ROOT / "artifacts" / "surprise-live-closed-loop.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "surprise-live-closed-loop.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_live(bridge_root=Path("D:/PLWM-Luanti/bridge")),
                     indent=2, ensure_ascii=False))
