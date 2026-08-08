"""Action-conditioned latent dynamics for bounded counterfactual planning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionConditionedLatentPlanner(nn.Module):
    """Alternates sensory and action events in one persistent latent state.

    The model predicts an observed transition during training. At inference it
    rolls its own action-conditioned latent states forward; it never receives a
    future observation or an action label from a planner.
    """

    def __init__(self, input_dim: int, *, action_count: int, latent_dim: int = 96) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_count = action_count
        self.latent_dim = latent_dim
        self.observation_encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.Tanh())
        self.observation_cell = nn.GRUCell(latent_dim, latent_dim)
        self.action_encoder = nn.Embedding(action_count, latent_dim)
        self.action_cell = nn.GRUCell(latent_dim, latent_dim)
        self.observation_decoder = nn.Linear(latent_dim, input_dim)
        self.reward_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, 1))

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def observe(self, observation: Tensor, state: Tensor) -> Tensor:
        return self.observation_cell(self.observation_encoder(observation), state)

    def transition(self, state: Tensor, action: Tensor) -> Tensor:
        if action.dtype != torch.long:
            action = action.long()
        if action.ndim != 1 or action.shape[0] != state.shape[0]:
            raise ValueError("action batch must match latent state")
        if (action < 0).any() or (action >= self.action_count).any():
            raise ValueError("action is outside the planner action space")
        return self.action_cell(self.action_encoder(action), state)

    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        next_state = self.transition(state, action)
        return next_state, self.observation_decoder(next_state), self.reward_head(next_state).squeeze(-1)

    def rollout(self, state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.ndim != 2 or actions.shape[0] != state.shape[0]:
            raise ValueError("actions must have shape [batch, horizon]")
        rewards = []
        for index in range(actions.shape[1]):
            state, _observation, reward = self.predict(state, actions[:, index])
            rewards.append(reward)
        return state, torch.stack(rewards, dim=1)

    @torch.no_grad()
    def plan(self, state: Tensor, *, horizon: int, discount: float = 0.99) -> tuple[Tensor, Tensor]:
        if horizon < 1 or horizon > 8:
            raise ValueError("planning horizon must be in [1, 8]")
        action_axes = [torch.arange(self.action_count, device=state.device) for _ in range(horizon)]
        sequences = torch.cartesian_prod(*action_axes)
        if horizon == 1:
            sequences = sequences.unsqueeze(1)
        choices = sequences.shape[0]
        repeated_state = state.repeat_interleave(choices, dim=0)
        expanded_actions = sequences.repeat(state.shape[0], 1)
        _final, reward_logits = self.rollout(repeated_state, expanded_actions)
        discounts = torch.tensor([discount**index for index in range(horizon)], device=state.device)
        scores = (torch.sigmoid(reward_logits) * discounts).sum(dim=1).view(state.shape[0], choices)
        return sequences[scores.argmax(dim=1)], scores
