"""Held-out test of action-result-aligned temporal visual confidence."""

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
class TemporalCondition:
    model: str
    noise_rate: float
    stale_replay_steps: int
    raw_successes: int
    temporal_successes: int
    episodes: int


@dataclass
class TemporalConfidenceReport:
    device: str
    heldout_seeds: list[int]
    conditions: list[TemporalCondition]
    accepted: bool
    artifact: str


def _load(path: Path, device: torch.device) -> VisionGoalDirectedPLWM:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = VisionGoalDirectedPLWM(history_length=int(payload["history_length"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> TemporalConfidenceReport:
    resolved = resolve_device(device)
    models = {
        "clean": _load(ROOT / "artifacts" / "luanti-vision-policy.pt", resolved),
        "noise-mixture": _load(ROOT / "artifacts" / "luanti-vision-noise-mixture-policy.pt", resolved),
    }
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_510_000 + index * 43 for index in range(4)]
    results: list[TemporalCondition] = []
    for model_index, (name, model) in enumerate(models.items()):
        for condition_index, (rate, stale) in enumerate(((0.08, 1), (0.16, 2))):
            raw_successes = 0
            temporal_successes = 0
            for index, task_seed in enumerate(seeds):
                schedule = NoiseSchedule(cell_flip_rate=rate, noisy_steps=6, stale_replay_steps=stale)
                raw_client = DegradedVisualClient(client, schedule, seed=15_000 + model_index * 1_000 + condition_index * 100 + index)
                raw, _ = _run_visual_recovery(
                    model, raw_client, task_seed=task_seed, episode_id=f"temporal-raw-{model_index}-{condition_index}-{index}"
                )
                raw_successes += int(raw)
                temporal_client = DegradedVisualClient(client, schedule, seed=15_000 + model_index * 1_000 + condition_index * 100 + index)
                temporal, _ = _run_visual_recovery(
                    model,
                    temporal_client,
                    task_seed=task_seed,
                    episode_id=f"temporal-filtered-{model_index}-{condition_index}-{index}",
                    temporal_consensus=True,
                )
                temporal_successes += int(temporal)
            results.append(TemporalCondition(name, rate, stale, raw_successes, temporal_successes, len(seeds)))
    artifact = ROOT / "artifacts" / "luanti-temporal-confidence-report.json"
    report = TemporalConfidenceReport(
        device=str(resolved),
        heldout_seeds=seeds,
        conditions=results,
        accepted=(
            all(item.temporal_successes >= item.raw_successes for item in results)
            and all(item.temporal_successes / item.episodes >= 0.75 for item in results if item.noise_rate == 0.16)
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("temporal confidence did not meet its evidence gate")

