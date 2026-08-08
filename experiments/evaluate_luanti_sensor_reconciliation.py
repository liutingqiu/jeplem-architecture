"""Evaluate fresh local sensor/vision reconciliation on held-out noise worlds."""

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
class ReconciliationCondition:
    model: str
    noise_rate: float
    stale_replay_steps: int
    raw_successes: int
    reconciled_successes: int
    episodes: int


@dataclass
class ReconciliationReport:
    device: str
    heldout_seeds: list[int]
    conditions: list[ReconciliationCondition]
    accepted: bool
    artifact: str


def _load(path: Path, device: torch.device) -> VisionGoalDirectedPLWM:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = VisionGoalDirectedPLWM(history_length=int(payload["history_length"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> ReconciliationReport:
    resolved = resolve_device(device)
    models = {
        "clean": _load(ROOT / "artifacts" / "luanti-vision-policy.pt", resolved),
        "noise-mixture": _load(ROOT / "artifacts" / "luanti-vision-noise-mixture-policy.pt", resolved),
    }
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_490_000 + index * 41 for index in range(4)]
    results: list[ReconciliationCondition] = []
    for model_index, (name, model) in enumerate(models.items()):
        for condition_index, (rate, stale) in enumerate(((0.08, 1), (0.16, 2))):
            raw_successes = 0
            reconciled_successes = 0
            for index, task_seed in enumerate(seeds):
                schedule = NoiseSchedule(cell_flip_rate=rate, noisy_steps=6, stale_replay_steps=stale)
                raw_client = DegradedVisualClient(client, schedule, seed=13_000 + model_index * 1_000 + condition_index * 100 + index)
                raw, _ = _run_visual_recovery(
                    model, raw_client, task_seed=task_seed, episode_id=f"sensor-raw-{model_index}-{condition_index}-{index}"
                )
                raw_successes += int(raw)
                filtered_client = DegradedVisualClient(client, schedule, seed=13_000 + model_index * 1_000 + condition_index * 100 + index)
                reconciled, _ = _run_visual_recovery(
                    model,
                    filtered_client,
                    task_seed=task_seed,
                    episode_id=f"sensor-filtered-{model_index}-{condition_index}-{index}",
                    reconcile_sensors=True,
                )
                reconciled_successes += int(reconciled)
            results.append(ReconciliationCondition(name, rate, stale, raw_successes, reconciled_successes, len(seeds)))
    artifact = ROOT / "artifacts" / "luanti-sensor-reconciliation-report.json"
    report = ReconciliationReport(
        device=str(resolved),
        heldout_seeds=seeds,
        conditions=results,
        accepted=(
            all(item.reconciled_successes >= item.raw_successes for item in results)
            and all(item.reconciled_successes / item.episodes >= 0.75 for item in results if item.noise_rate == 0.16)
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("sensor reconciliation did not meet its evidence gate")

