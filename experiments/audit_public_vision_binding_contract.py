"""Validate that delayed visual-binding labels are determined by public pixels."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from experiments.train_public_vision_candidate_memory import GRID_SIZE, IDENTITIES, VISION_CHANNELS, make_batch


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SEED = 13_800_000
EPISODES = 2_048


@dataclass(frozen=True)
class Audit:
    protocol: str
    generator_seed: int
    episodes: int
    public_pixel_oracle_accuracy: float
    accepted: bool


def public_pixel_labels(vision: Tensor) -> Tensor:
    """Derive each answer from rendered public tiles for audit only."""
    if vision.ndim != 5 or vision.shape[2:] != (GRID_SIZE, GRID_SIZE, VISION_CHANNELS):
        raise ValueError("expected [batch, time, 11, 11, 7] visual sequences")
    initial, query = vision[:, 0, :, :, 3:], vision[:, -1, :, :, 3:]
    initial_counts = initial.sum(dim=(1, 2))
    query_counts = query.sum(dim=(1, 2))
    if not bool((initial_counts == 1).all()) or not bool((query_counts.sum(dim=1) == 1).all()):
        raise ValueError("visual contract does not expose four initial identities and one query identity")
    query_identity = query_counts.argmax(dim=1)
    selected = initial.gather(3, query_identity[:, None, None, None].expand(-1, GRID_SIZE, GRID_SIZE, 1)).squeeze(3)
    left = selected[:, :, : GRID_SIZE // 2].sum(dim=(1, 2))
    right = selected[:, :, GRID_SIZE // 2 + 1 :].sum(dim=(1, 2))
    if not bool((left + right == 1).all()):
        raise ValueError("queried public identity is not rendered exactly once off centre")
    return right.to(dtype=torch.float32).unsqueeze(1)


def run(*, episodes: int = EPISODES, seed: int = AUDIT_SEED) -> Audit:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    batch = make_batch(episodes, generator=torch.Generator().manual_seed(seed), device=torch.device("cpu"))
    oracle = public_pixel_labels(batch.vision)
    accuracy = float((oracle == batch.labels).float().mean())
    return Audit(
        "public_vision_binding_contract_audit_v1: read-only public-pixel label reconstruction for the synthetic delayed-query calibration; never a model input, control policy, or architecture result.",
        seed, episodes, accuracy, accuracy == 1.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "public-vision-binding-contract-audit.json")
    args = parser.parse_args()
    outcome = run(episodes=args.episodes, seed=args.seed)
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(asdict(outcome), indent=2), encoding="utf-8")
    print(json.dumps(asdict(outcome), indent=2))
