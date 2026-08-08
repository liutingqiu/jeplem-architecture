"""Held-out robustness comparison for clean and noise-augmented policies."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.luanti_degradation_audit import BRIDGE_ROOT, DegradedVisualClient, NoiseSchedule, _run_visual_recovery
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, VisionGoalDirectedPLWM


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HeldoutCondition:
    noise_rate: float
    stale_replay_steps: int
    clean_successes: int
    augmented_successes: int
    episodes: int


@dataclass
class HeldoutRobustnessReport:
    device: str
    heldout_seeds: list[int]
    conditions: list[HeldoutCondition]
    accepted: bool
    artifact: str


def _load_policy(path: Path, device: torch.device) -> VisionGoalDirectedPLWM:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = VisionGoalDirectedPLWM(history_length=int(payload["history_length"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> HeldoutRobustnessReport:
    resolved = resolve_device(device)
    clean = _load_policy(ROOT / "artifacts" / "luanti-vision-policy.pt", resolved)
    augmented = _load_policy(ROOT / "artifacts" / "luanti-vision-noise-mixture-policy.pt", resolved)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    # These seeds are outside both the original visual training range and the
    # small comparison set used to select the augmentation configuration.
    seeds = [1_480_000 + index * 37 for index in range(8)]
    conditions: list[HeldoutCondition] = []
    for condition_index, (rate, stale) in enumerate(((0.08, 1), (0.16, 2))):
        clean_successes = 0
        augmented_successes = 0
        for index, task_seed in enumerate(seeds):
            schedule = NoiseSchedule(cell_flip_rate=rate, noisy_steps=6, stale_replay_steps=stale)
            clean_client = DegradedVisualClient(client, schedule, seed=11_000 + condition_index * 100 + index)
            clean_success, _ = _run_visual_recovery(
                clean, clean_client, task_seed=task_seed, episode_id=f"robustness-clean-{condition_index}-{index}"
            )
            clean_successes += int(clean_success)
            augmented_client = DegradedVisualClient(client, schedule, seed=11_000 + condition_index * 100 + index)
            augmented_success, _ = _run_visual_recovery(
                augmented, augmented_client, task_seed=task_seed, episode_id=f"robustness-augmented-{condition_index}-{index}"
            )
            augmented_successes += int(augmented_success)
        conditions.append(HeldoutCondition(rate, stale, clean_successes, augmented_successes, len(seeds)))
    artifact = ROOT / "artifacts" / "luanti-noise-mixture-generalization-report.json"
    report = HeldoutRobustnessReport(
        device=str(resolved),
        heldout_seeds=seeds,
        conditions=conditions,
        accepted=(
            all(condition.augmented_successes / condition.episodes >= 0.75 for condition in conditions)
            and all(condition.augmented_successes >= condition.clean_successes + 2 for condition in conditions)
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("held-out noise generalization did not meet its evidence gate")
