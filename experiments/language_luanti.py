"""Verify a strict language goal crossing the real Luanti object task boundary."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.device import resolve_device
from plwm.language import LuantiLanguageGoal, parse_luanti_language_goal
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, ObjectGoalDirectedPLWM


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


@dataclass
class LanguageLuantiReport:
    parsed_goal: LuantiLanguageGoal
    actions: int
    completed: bool
    key_collected: bool
    decoy_collected: bool
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> LanguageLuantiReport:
    resolved_device = resolve_device(device)
    goal = parse_luanti_language_goal(
        '{"task":"object_delivery","target":"goal","require_key":true,"avoid_decoy":true}'
    )
    checkpoint = torch.load(ROOT / "artifacts" / "luanti-object-policy.pt", map_location=resolved_device, weights_only=True)
    model = ObjectGoalDirectedPLWM(history_length=4).to(resolved_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    state = client.begin_object_episode("language-object-goal", 2_300_000)
    history = [state]
    actions = 0
    while not state.completed and actions < 28:
        direction = model.choose(history[-4:])
        interaction, next_state = client.step_object(direction)
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            break
        state = next_state
        history.append(state)
        actions += 1
        if actions == 1:
            recenter, recentered_state = client.recenter_object()
            if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                break
            state = recentered_state
            history.append(state)
    accepted = (
        goal.require_key
        and goal.avoid_decoy
        and state.completed
        and state.key_collected
        and not state.decoy_collected
    )
    artifact = ROOT / "artifacts" / "luanti-language-goal-report.json"
    report = LanguageLuantiReport(goal, actions, state.completed, state.key_collected, state.decoy_collected, accepted, str(artifact))
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(asdict(report), indent=2))
    if not report.accepted:
        raise SystemExit("language Luanti goal did not meet its evidence gate")
