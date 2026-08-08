"""Behavioral held-out evaluation of the learned temporal visual denoiser."""

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
from plwm.vision_denoiser import TemporalVisionDenoiser


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DenoiserCondition:
    policy: str
    noise_rate: float
    stale_replay_steps: int
    raw_successes: int
    denoised_successes: int
    episodes: int


@dataclass
class DenoiserEvaluationReport:
    device: str
    heldout_seeds: list[int]
    conditions: list[DenoiserCondition]
    accepted: bool
    artifact: str


def _load_policy(path: Path, device: torch.device) -> VisionGoalDirectedPLWM:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = VisionGoalDirectedPLWM(history_length=int(payload["history_length"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _load_denoiser(device: torch.device) -> TemporalVisionDenoiser:
    payload = torch.load(ROOT / "artifacts" / "luanti-temporal-vision-denoiser.pt", map_location=device, weights_only=True)
    model = TemporalVisionDenoiser().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run(
    *,
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
    seed_start: int = 1_520_000,
    seed_count: int = 4,
    artifact_name: str = "luanti-temporal-denoiser-evaluation.json",
) -> DenoiserEvaluationReport:
    resolved = resolve_device(device)
    policies = {
        "clean": _load_policy(ROOT / "artifacts" / "luanti-vision-policy.pt", resolved),
        "noise-mixture": _load_policy(ROOT / "artifacts" / "luanti-vision-noise-mixture-policy.pt", resolved),
    }
    denoiser = _load_denoiser(resolved)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [seed_start + index * 47 for index in range(seed_count)]
    conditions: list[DenoiserCondition] = []
    for policy_index, (name, policy) in enumerate(policies.items()):
        for condition_index, (rate, stale) in enumerate(((0.08, 1), (0.16, 2))):
            raw_successes = 0
            denoised_successes = 0
            for index, task_seed in enumerate(seeds):
                schedule = NoiseSchedule(cell_flip_rate=rate, noisy_steps=6, stale_replay_steps=stale)
                raw_client = DegradedVisualClient(client, schedule, seed=17_000 + policy_index * 1_000 + condition_index * 100 + index)
                raw, _ = _run_visual_recovery(
                    policy, raw_client, task_seed=task_seed, episode_id=f"denoiser-raw-{policy_index}-{condition_index}-{index}"
                )
                raw_successes += int(raw)
                filtered_client = DegradedVisualClient(client, schedule, seed=17_000 + policy_index * 1_000 + condition_index * 100 + index)
                denoised, _ = _run_visual_recovery(
                    policy,
                    filtered_client,
                    task_seed=task_seed,
                    episode_id=f"denoiser-filtered-{policy_index}-{condition_index}-{index}",
                    denoiser=denoiser,
                )
                denoised_successes += int(denoised)
            conditions.append(DenoiserCondition(name, rate, stale, raw_successes, denoised_successes, len(seeds)))
    artifact = ROOT / "artifacts" / artifact_name
    report = DenoiserEvaluationReport(
        device=str(resolved),
        heldout_seeds=seeds,
        conditions=conditions,
        accepted=(
            all(item.denoised_successes >= item.raw_successes for item in conditions)
            and all(item.denoised_successes / item.episodes >= 0.75 for item in conditions if item.noise_rate == 0.16)
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("temporal denoiser did not meet its evidence gate")
