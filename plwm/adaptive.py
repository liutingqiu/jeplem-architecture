"""Differentiable adaptive latent-compute controller."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdaptivePonderer(nn.Module):
    """Selects a distribution over recurrent thought depths before decoding.

    The transition core and readout do not receive the difficulty hint. Only
    the controller sees it, so the task can only be solved cheaply when the
    controller routes an example to an adequate latent depth.
    """

    def __init__(self, state_count: int = 8, latent_dim: int = 32, max_steps: int = 4):
        super().__init__()
        self.state_count = state_count
        self.max_steps = max_steps
        self.state_encoder = nn.Sequential(nn.Linear(state_count, latent_dim), nn.Tanh())
        self.transition = nn.GRUCell(latent_dim, latent_dim)
        self.readout = nn.Linear(latent_dim, state_count)
        self.halt_controller = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, max_steps + 1),
        )

    def forward(self, start_state: Tensor, difficulty: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        latent = self.state_encoder(torch.nn.functional.one_hot(start_state.long(), self.state_count).float())
        depth_logits = []
        for _ in range(self.max_steps + 1):
            depth_logits.append(self.readout(latent))
            latent = self.transition(latent, latent)
        depth_logits = torch.stack(depth_logits, dim=1)
        halt_weights = torch.softmax(self.halt_controller(difficulty.unsqueeze(1)), dim=-1)
        mixed_logits = torch.einsum("bk,bks->bs", halt_weights, depth_logits)
        steps = torch.arange(self.max_steps + 1, device=start_state.device, dtype=halt_weights.dtype)
        expected_steps = (halt_weights * steps).sum(dim=-1)
        return mixed_logits, expected_steps, halt_weights, depth_logits
