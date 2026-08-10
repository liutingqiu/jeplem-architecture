"""JEPLM perception layer: learn Key/Ball presence from RAW MemoryS13
observations (157-dim encode_observation), replacing the hand-crafted
visible_object_presence feature.  This is the 'perception -> semantics'
layer of the JEPLM architecture (JEPA-encoder role): the agent must extract
'what object is visible' from its sensor stream on its own.

The target (visible-object class) is extracted from the same observation
image -- observable, not privileged.  After pretraining, the encoder output
can replace the hand-crafted presence feature in the real-stream RL loop
(train_decide_stream_real_rl.py --use-perception).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.minigrid_features import visible_object_presence
from experiments.train_minigrid_memory_actor_critic import encode_observation

ENVIRONMENT = "MiniGrid-MemoryS13-v0"
SENSORY_DIM = 159  # encode_observation(observation_only)


def _collect_frames(seed: int, count: int) -> tuple[Tensor, Tensor]:
    """Collect raw observations and their visible-object class labels
    (0=none, 1=key, 2=ball) from a MemoryS13 random walk."""
    rng = random.Random(77_331 + seed)
    environment = gym.make(ENVIRONMENT, max_steps=64)
    frames: list[Tensor] = []
    labels: list[int] = []
    try:
        observation, _info = environment.reset(seed=seed)
        while len(frames) < count:
            presence = visible_object_presence(
                encode_observation(observation).unsqueeze(0)).squeeze(0)
            if presence.sum() == 1:
                label = 1 + int(presence.argmax().item())  # 1=key, 2=ball
            else:
                label = 0  # none visible (empty or both - both is rare)
            frames.append(encode_observation(observation))
            labels.append(label)
            observation, _reward, terminated, truncated, _info = environment.step(rng.randrange(3))
            if terminated or truncated:
                observation, _info = environment.reset(seed=seed + len(frames))
    finally:
        environment.close()
    return torch.stack(frames), torch.tensor(labels, dtype=torch.long)


class PerceptionNet(nn.Module):
    """CNN over the 7x7x3 image prefix + MLP over the auxiliary channels.
    Image part: conv layers exploit the spatial color structure (Key vs Ball);
    the rest (direction/action/reward) goes through an MLP branch."""

    def __init__(self, input_dim: int = 159, hidden: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, 3, padding=1), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden), nn.Tanh(),
        )
        self.aux = nn.Linear(input_dim - 147, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        image = x[..., :147].reshape(*x.shape[:-1], 7, 7, 3).permute(0, 3, 1, 2)
        rest = x[..., 147:]
        feat = torch.cat((self.cnn(image), self.aux(rest)), dim=-1)
        return self.head(feat)


@dataclass
class Report:
    protocol: str
    device: str
    training_examples: int
    updates: int
    heldout_accuracy: float
    accepted_perception_contract: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, training_examples: int = 2048,
    updates: int = 800, batch_size: int = 128, experiment_seed: int = 93_337,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    train_frames, train_labels = _collect_frames(50_000, training_examples)
    heldout_frames, heldout_labels = _collect_frames(6_000_000, 512)
    counts = train_labels.bincount(minlength=3).float()
    weights = (train_labels.shape[0] / (3 * counts.clamp_min(1.0))).to(resolved)
    model = PerceptionNet(SENSORY_DIM).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(92_337)
    model.train()
    for _update in range(updates):
        indices = torch.randint(train_frames.shape[0], (batch_size,), generator=generator)
        logits = model(train_frames[indices].to(resolved))
        loss = nn.functional.cross_entropy(logits, train_labels[indices].to(resolved), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        heldout = float((model(heldout_frames.to(resolved)).argmax(-1).cpu() == heldout_labels).float().mean())
    accepted = heldout >= 0.95
    weight_path = ROOT / "artifacts" / "perception-layer.pt"
    torch.save(model.state_dict(), weight_path)
    target = artifact or ROOT / "artifacts" / "perception-layer-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "jeplem_perception_layer_v1: learn Key/Ball presence class from RAW MemoryS13 observation encoding (159-dim); target class extracted from the same image (observable); this replaces the hand-crafted presence feature with a learned perception layer (JEPA-encoder role) for the real-stream RL loop.",
        str(resolved), training_examples, updates, heldout, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--training-examples", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--experiment-seed", type=int, default=93_337)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, training_examples=args.training_examples,
                  updates=args.updates, batch_size=args.batch_size,
                  experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
