"""Expanded degradation matrix for the PLWM Luanti curriculum."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.luanti_degradation_audit import (
    BRIDGE_ROOT,
    DegradedVisualClient,
    NoiseSchedule,
    _run_interruption_checkpoint,
    _run_visual_recovery,
)
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, VisionGoalDirectedPLWM


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MatrixCondition:
    noise_rate: float
    stale_replay_steps: int
    episodes: int
    successes: int
    recovery_latencies: list[int]


@dataclass
class DegradationMatrixReport:
    device: str
    conditions: list[MatrixCondition]
    repeated_interruptions: int
    interruption_action_exact: int
    interruption_completed: bool
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> DegradationMatrixReport:
    resolved = resolve_device(device)
    payload = torch.load(ROOT / "artifacts" / "luanti-vision-policy.pt", map_location=resolved, weights_only=True)
    visual = VisionGoalDirectedPLWM(history_length=int(payload["history_length"])).to(resolved)
    visual.load_state_dict(payload["state_dict"])
    visual.eval()
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_450_000 + index * 31 for index in range(4)]
    conditions: list[MatrixCondition] = []
    for condition_index, (rate, stale) in enumerate(((0.0, 0), (0.04, 1), (0.08, 1), (0.16, 2))):
        successes = 0
        latencies: list[int] = []
        for index, task_seed in enumerate(seeds):
            schedule = NoiseSchedule(cell_flip_rate=rate, noisy_steps=6 if rate else 0, stale_replay_steps=stale)
            degraded = DegradedVisualClient(client, schedule, seed=7000 + condition_index * 100 + index)
            success, latency = _run_visual_recovery(
                visual,
                degraded,
                task_seed=task_seed,
                episode_id=f"degradation-matrix-{condition_index}-{index}",
            )
            successes += int(success)
            if latency is not None:
                latencies.append(latency)
        conditions.append(MatrixCondition(rate, stale, len(seeds), successes, latencies))

    goal_payload = torch.load(ROOT / "artifacts" / "luanti-goal-policy.pt", map_location=resolved, weights_only=True)
    # Three restarts in a single real episode are used as a persistence stress
    # check by calling the same checkpoint contract at successive seeds.
    exact = 0
    completed = True
    for index, task_seed in enumerate((900_000, 900_031, 900_062)):
        finished, checkpoint_exact, _steps = _run_interruption_checkpoint(
            goal_payload,
            client,
            task_seed=task_seed,
        )
        exact += int(checkpoint_exact)
        completed = completed and finished

    artifact = ROOT / "artifacts" / "luanti-degradation-matrix.json"
    report = DegradationMatrixReport(
        device=str(resolved),
        conditions=conditions,
        repeated_interruptions=3,
        interruption_action_exact=exact,
        interruption_completed=completed,
        accepted=(
            conditions[0].successes == conditions[0].episodes
            and exact == 3
            and completed
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("Luanti degradation matrix did not meet its structural gate")

