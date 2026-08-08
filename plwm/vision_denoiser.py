"""Action-result-aligned learned visual reconstruction for Luanti frames."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .luanti_training import LuantiState, VISION_TILES, VISION_WIDTH, encode_vision


def vision_tensor(vision: Mapping[str, Any]) -> Tensor:
    """Return a validated ``[tile, y, x]`` one-hot CPU tensor."""
    encoded = encode_vision(vision)
    return torch.tensor(encoded, dtype=torch.float32).reshape(VISION_WIDTH, VISION_WIDTH, len(VISION_TILES)).permute(2, 0, 1)


def align_previous_vision(
    previous_state: LuantiState,
    previous: Tensor,
    current_state: LuantiState,
) -> tuple[Tensor, Tensor]:
    """Project previous egocentric cells into the current frame by observed motion."""
    if tuple(previous.shape) != (len(VISION_TILES), VISION_WIDTH, VISION_WIDTH):
        raise ValueError("previous vision tensor has unexpected shape")
    aligned = torch.zeros_like(previous)
    mask = torch.zeros((1, VISION_WIDTH, VISION_WIDTH), dtype=previous.dtype, device=previous.device)
    shift_x = current_state.x - previous_state.x
    shift_z = current_state.z - previous_state.z
    agent_index = VISION_TILES.index("agent")
    for z in range(VISION_WIDTH):
        for x in range(VISION_WIDTH):
            prior_x, prior_z = x + shift_x, z + shift_z
            if not (0 <= prior_x < VISION_WIDTH and 0 <= prior_z < VISION_WIDTH):
                continue
            if previous[agent_index, prior_z, prior_x] == 1:
                continue
            aligned[:, z, x] = previous[:, prior_z, prior_x]
            mask[:, z, x] = 1.0
    return aligned, mask


def corrupt_vision(tensor: Tensor, *, flip_rate: float, generator: torch.Generator | None = None) -> Tensor:
    """Inject the same independent tile noise used in live Luanti audits."""
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError("flip_rate must be between zero and one")
    if tensor.ndim != 3 or tensor.shape[0] != len(VISION_TILES):
        raise ValueError("vision tensor must have tile channels")
    result = tensor.clone()
    indices = result.argmax(dim=0)
    selected = torch.rand(indices.shape, generator=generator, device=tensor.device) < flip_rate
    selected &= indices != VISION_TILES.index("agent")
    ys, xs = selected.nonzero(as_tuple=True)
    original = indices[ys, xs]
    replacement = torch.where(
        original == VISION_TILES.index("air"),
        torch.full_like(original, VISION_TILES.index("wall")),
        torch.full_like(original, VISION_TILES.index("air")),
    )
    result[:, ys, xs] = 0.0
    result[replacement, ys, xs] = 1.0
    return result


class TemporalVisionDenoiser(nn.Module):
    """Predict the fresh clean tile class from current and aligned prior input."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        channels = len(VISION_TILES) * 2 + 1
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, len(VISION_TILES), kernel_size=1),
        )

    def forward(self, current: Tensor, aligned_previous: Tensor, overlap: Tensor) -> Tensor:
        return self.network(torch.cat((current, aligned_previous, overlap), dim=1))


@torch.no_grad()
def denoise_frame(
    model: TemporalVisionDenoiser,
    previous_state: LuantiState,
    previous_vision: Mapping[str, Any],
    current_state: LuantiState,
    current_vision: Mapping[str, Any],
) -> dict[str, Any]:
    """Denoise one frame without changing any nonvisual observation fields."""
    device = next(model.parameters()).device
    current = vision_tensor(current_vision).unsqueeze(0).to(device)
    previous = vision_tensor(previous_vision).to(device)
    aligned, mask = align_previous_vision(previous_state, previous, current_state)
    logits = model(current, aligned.unsqueeze(0), mask.unsqueeze(0))
    tiles = logits.argmax(dim=1).squeeze(0).cpu()
    center = VISION_WIDTH // 2
    agent = VISION_TILES.index("agent")
    tiles[tiles == agent] = VISION_TILES.index("air")
    tiles[center, center] = agent
    payload = dict(current_vision)
    payload["cells"] = [VISION_TILES[int(tile)] for tile in tiles.reshape(-1)]
    encode_vision(payload)
    return payload

