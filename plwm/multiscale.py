"""Trainable fast/slow latent state separation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MultiScaleState:
    fast: Tensor
    slow: Tensor


class MultiScalePLWM(nn.Module):
    """Fast state reacts to events; slow state updates through a learned gate."""

    def __init__(self, input_dim: int = 2, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.Tanh())
        self.fast_cell = nn.GRUCell(latent_dim + latent_dim, latent_dim)
        self.slow_candidate = nn.GRUCell(latent_dim, latent_dim)
        self.slow_gate = nn.Sequential(nn.Linear(latent_dim * 3, latent_dim), nn.Tanh(), nn.Linear(latent_dim, 1))
        self.fast_head = nn.Linear(latent_dim, 1)
        self.slow_head = nn.Linear(latent_dim, 1)

    def initial_state(self, batch: int, device: torch.device | str) -> MultiScaleState:
        zeros = torch.zeros(batch, self.latent_dim, device=device)
        return MultiScaleState(zeros, zeros.clone())

    def step(self, observation: Tensor, state: MultiScaleState) -> tuple[MultiScaleState, Tensor]:
        encoded = self.encoder(observation)
        fast = self.fast_cell(torch.cat((encoded, state.slow), dim=-1), state.fast)
        gate = torch.sigmoid(self.slow_gate(torch.cat((encoded, fast, state.slow), dim=-1)))
        candidate = self.slow_candidate(fast, state.slow)
        slow = (1.0 - gate) * state.slow + gate * candidate
        return MultiScaleState(fast, slow), gate

    def decode(self, state: MultiScaleState) -> tuple[Tensor, Tensor]:
        return self.fast_head(state.fast), self.slow_head(state.slow)
