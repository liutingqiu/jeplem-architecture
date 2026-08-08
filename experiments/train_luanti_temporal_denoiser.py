"""Train a supervised temporal visual denoiser from real Luanti frames."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import collect_teacher_trajectory
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import LuantiLiveClient, LuantiState, VISION_TILES
from plwm.vision_denoiser import TemporalVisionDenoiser, align_previous_vision, corrupt_vision, vision_tensor


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


def _dataset(client: LuantiLiveClient) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_frames: list[torch.Tensor] = []
    aligned_frames: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in range(24):
        trajectory = collect_teacher_trajectory(
            client,
            task_seed=1_400_000 + index * 17,
            episode_id=f"temporal-denoiser-teacher-{index}",
            max_steps=28,
        )
        states = [LuantiState(**payload) for payload in trajectory.states]
        for prior_index in range(len(states) - 1):
            previous = vision_tensor(trajectory.visions[prior_index])
            current = vision_tensor(trajectory.visions[prior_index + 1])
            aligned, mask = align_previous_vision(states[prior_index], previous, states[prior_index + 1])
            current_frames.append(current)
            aligned_frames.append(aligned)
            masks.append(mask)
            targets.append(current.argmax(dim=0))
    if not targets:
        raise RuntimeError("temporal denoiser collected no frame pairs")
    return torch.stack(current_frames), torch.stack(aligned_frames), torch.stack(masks), torch.stack(targets)


def _corrupt_batch(batch: torch.Tensor, *, rate: float, generator: torch.Generator) -> torch.Tensor:
    return torch.stack([corrupt_vision(frame, flip_rate=rate, generator=generator) for frame in batch])


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT, steps: int = 1200) -> dict:
    resolved = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    current, aligned, masks, targets = _dataset(client)
    model = TemporalVisionDenoiser().to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    weights = torch.tensor([1.0, 2.0, 2.0, 4.0, 3.0, 3.0], device=resolved)
    criterion = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator(device="cpu").manual_seed(683)
    model.train()
    batch_size = min(64, len(targets))
    final_loss = 0.0
    for _ in range(steps):
        indices = torch.randint(len(targets), (batch_size,), generator=generator)
        rate = (0.04, 0.08, 0.16)[int(torch.randint(3, (1,), generator=generator).item())]
        noisy_current = _corrupt_batch(current[indices], rate=rate, generator=generator).to(resolved)
        noisy_aligned = _corrupt_batch(aligned[indices], rate=rate, generator=generator).to(resolved)
        logits = model(noisy_current, noisy_aligned, masks[indices].to(resolved))
        loss = criterion(logits, targets[indices].to(resolved))
        final_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    checkpoint = ROOT / "artifacts" / "luanti-temporal-vision-denoiser.pt"
    torch.save({"state_dict": model.state_dict(), "training_pairs": len(targets), "noise_rates": (0.04, 0.08, 0.16)}, checkpoint)
    artifact = ROOT / "artifacts" / "luanti-temporal-denoiser-training-report.json"
    report = {
        "device": str(resolved),
        "training_pairs": len(targets),
        "training_steps": steps,
        "final_loss": final_loss,
        "checkpoint": str(checkpoint),
        "artifact": str(artifact),
    }
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
