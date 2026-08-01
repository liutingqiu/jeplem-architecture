"""Evidence-traceable categorical belief state for PLWM modules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CategoricalBelief:
    log_probs: Tensor

    @classmethod
    def uniform(cls, batch_size: int, states: int, device: torch.device | str = "cpu") -> "CategoricalBelief":
        return cls(torch.full((batch_size, states), -math.log(states), device=device))

    @property
    def probs(self) -> Tensor:
        return self.log_probs.exp()

    def observe(self, log_likelihood: Tensor) -> "CategoricalBelief":
        return CategoricalBelief(torch.log_softmax(self.log_probs + log_likelihood, dim=-1))

    def transition(self, log_matrix: Tensor) -> "CategoricalBelief":
        # log_matrix[from_state, to_state]
        next_probs = torch.logsumexp(self.log_probs.unsqueeze(-1) + log_matrix, dim=-2)
        return CategoricalBelief(torch.log_softmax(next_probs, dim=-1))

    def entropy(self) -> Tensor:
        return -(self.probs * self.log_probs).sum(dim=-1)

    def confidence(self) -> Tensor:
        return self.probs.max(dim=-1).values
