"""Train and evaluate a visual policy with sensor-noise augmentation."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.luanti_degradation_audit import BRIDGE_ROOT, DegradedVisualClient, NoiseSchedule, _run_visual_recovery
from experiments.train_luanti_behavior import ExecutedAction, Trajectory, collect_teacher_trajectory, visual_training_tensors
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, VisionGoalDirectedPLWM, augment_visual_observations


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Comparison:
    noise_rate: float
    stale_replay_steps: int
    episodes: int
    clean_successes: int
    augmented_successes: int


@dataclass
class NoiseAugmentedReport:
    device: str
    training_samples: int
    augmentation_rate: float
    training_steps: int
    comparisons: list[Comparison]
    clean_checkpoint: str
    augmented_checkpoint: str
    accepted: bool
    artifact: str


def _load_teacher_trajectories() -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    with (ROOT / "artifacts" / "luanti-rollouts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("collection") != "teacher":
                continue
            actions = [ExecutedAction(**action) for action in payload["actions"]]
            trajectories.append(
                Trajectory(
                    episode_id=payload["episode_id"],
                    task_seed=int(payload["task_seed"]),
                    collection=payload.get("collection", "teacher"),
                    states=payload["states"],
                    actions=actions,
                    visions=payload.get("visions", []),
                )
            )
    if not trajectories:
        raise RuntimeError("no teacher visual trajectories found")
    return trajectories


def _train(model: VisionGoalDirectedPLWM, observations: torch.Tensor, labels: torch.Tensor, *, device: torch.device, steps: int, seed: int, augmentation_rates: tuple[float, ...]) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = min(128, len(labels))
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(labels), (batch_size,), generator=generator)
        batch = observations[indices]
        if augmentation_rates:
            rate_index = int(torch.randint(len(augmentation_rates), (1,), generator=generator).item())
            batch = augment_visual_observations(batch, flip_rate=augmentation_rates[rate_index], generator=generator)
        logits, surprises = model(batch.to(device))
        loss = criterion(logits, labels[indices].to(device)) + 0.05 * surprises.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT, training_steps: int = 700) -> NoiseAugmentedReport:
    resolved = resolve_device(device)
    trajectories = _load_teacher_trajectories()
    if not any(trajectory.visions for trajectory in trajectories):
        collector = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
        trajectories = [
            collect_teacher_trajectory(
                collector,
                task_seed=1_400_000 + index * 17,
                episode_id=f"noise-augmented-teacher-{index}",
                max_steps=28,
            )
            for index in range(24)
        ]
    observations, labels = visual_training_tensors(trajectories)
    clean_payload = torch.load(ROOT / "artifacts" / "luanti-vision-policy.pt", map_location=resolved, weights_only=True)
    clean_model = VisionGoalDirectedPLWM(history_length=int(clean_payload["history_length"])).to(resolved)
    clean_model.load_state_dict(clean_payload["state_dict"])
    clean_model.eval()
    augmented_model = VisionGoalDirectedPLWM(history_length=int(clean_payload["history_length"])).to(resolved)
    augmentation_rates = (0.04, 0.08, 0.16)
    _train(augmented_model, observations, labels, device=resolved, steps=training_steps, seed=442, augmentation_rates=augmentation_rates)

    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [1_450_000 + index * 31 for index in range(4)]
    comparisons: list[Comparison] = []
    for condition_index, (rate, stale) in enumerate(((0.0, 0), (0.04, 1), (0.08, 1), (0.16, 2))):
        clean_successes = 0
        augmented_successes = 0
        for index, task_seed in enumerate(seeds):
            clean_client = DegradedVisualClient(
                client,
                NoiseSchedule(cell_flip_rate=rate, noisy_steps=6 if rate else 0, stale_replay_steps=stale),
                seed=8100 + condition_index * 100 + index,
            )
            success, _latency = _run_visual_recovery(
                clean_model,
                clean_client,
                task_seed=task_seed,
                episode_id=f"noise-compare-clean-{condition_index}-{index}",
            )
            clean_successes += int(success)
            augmented_client = DegradedVisualClient(
                client,
                NoiseSchedule(cell_flip_rate=rate, noisy_steps=6 if rate else 0, stale_replay_steps=stale),
                seed=9100 + condition_index * 100 + index,
            )
            success, _latency = _run_visual_recovery(
                augmented_model,
                augmented_client,
                task_seed=task_seed,
                episode_id=f"noise-compare-augmented-{condition_index}-{index}",
            )
            augmented_successes += int(success)
        comparisons.append(Comparison(rate, stale, len(seeds), clean_successes, augmented_successes))

    augmented_checkpoint = ROOT / "artifacts" / "luanti-vision-noise-mixture-policy.pt"
    torch.save(
        {
            "state_dict": augmented_model.state_dict(),
            "input_dim": VisionGoalDirectedPLWM.input_dim,
            "history_length": int(clean_payload["history_length"]),
            "augmentation_rates": augmentation_rates,
            "training_samples": len(labels),
        },
        augmented_checkpoint,
    )
    artifact = ROOT / "artifacts" / "luanti-noise-augmentation-report.json"
    report = NoiseAugmentedReport(
        device=str(resolved),
        training_samples=len(labels),
        augmentation_rate=max(augmentation_rates),
        training_steps=training_steps,
        comparisons=comparisons,
        clean_checkpoint=str(ROOT / "artifacts" / "luanti-vision-policy.pt"),
        augmented_checkpoint=str(augmented_checkpoint),
        accepted=(
            comparisons[0].clean_successes == comparisons[0].episodes
            and comparisons[0].augmented_successes == comparisons[0].episodes
            and comparisons[2].augmented_successes >= comparisons[2].clean_successes
            and comparisons[3].augmented_successes >= comparisons[3].clean_successes
        ),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(asdict(result), indent=2))
    if not result.accepted:
        raise SystemExit("noise augmentation did not meet its comparison gate")
