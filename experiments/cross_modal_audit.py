"""Checkpoint, cross-modal, and rule-revision audit for the current PLWM loop."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.language_luanti import run as run_language_goal
from experiments.train_luanti_behavior import evaluate_visual_causal_policy, run_rule_switch
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, VisionGoalDirectedPLWM


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


@dataclass
class CrossModalAudit:
    visual_restart_successes: int
    visual_restart_episodes: int
    rule_switch_accepted: bool
    language_goal_accepted: bool
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> CrossModalAudit:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(ROOT / "artifacts" / "luanti-vision-policy.pt", map_location=resolved_device, weights_only=True)
    model = VisionGoalDirectedPLWM(history_length=int(checkpoint["history_length"])).to(resolved_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_450_000 + index * 31 for index in range(4)]
    visual = evaluate_visual_causal_policy(
        client,
        seeds=seeds,
        chooser=lambda history: model.choose(history[-4:]),
        prefix="restart-visual",
        max_steps=28,
    )
    rule_switch = run_rule_switch(episode_count=4, steps=12, bridge_root=bridge_root)
    language = run_language_goal(device=resolved_device, bridge_root=bridge_root)
    accepted = (
        visual.successes == visual.episodes
        and rule_switch.accepted
        and language.accepted
    )
    artifact = ROOT / "artifacts" / "cross-modal-audit.json"
    report = CrossModalAudit(visual.successes, visual.episodes, rule_switch.accepted, language.accepted, accepted, str(artifact))
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(asdict(report), indent=2))
    if not report.accepted:
        raise SystemExit("cross-modal audit did not meet its evidence gate")
