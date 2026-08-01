"""Action-conditioned latent dynamics and one-step counterfactual planning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class WorldState:
    belief: Tensor

    def detached_clone(self) -> "WorldState":
        return WorldState(self.belief.detach().clone())


class ActionConditionedPLWM(nn.Module):
    """Persistent context encoder plus an action-conditioned world model.

    `advance` integrates observations into a durable belief. `imagine` predicts
    the next discrete world state for a proposed action without changing that
    belief. `plan` compares all possible actions against a goal state.
    """

    def __init__(self, input_dim: int, *, latent_dim: int = 32, state_count: int = 3, action_count: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_count = state_count
        self.action_count = action_count
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.Tanh())
        self.belief_cell = nn.GRUCell(latent_dim, latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_count, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_count),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> WorldState:
        return WorldState(torch.zeros(batch_size, self.latent_dim, device=device))

    def advance(self, observation: Tensor, state: WorldState) -> WorldState:
        return WorldState(self.belief_cell(self.encoder(observation), state.belief))

    def imagine(self, state: WorldState, action: Tensor) -> Tensor:
        one_hot = torch.nn.functional.one_hot(action.long(), self.action_count).to(state.belief.dtype)
        return self.transition(torch.cat((state.belief, one_hot), dim=-1))

    def plan(self, state: WorldState, goal_state: Tensor) -> tuple[Tensor, Tensor]:
        actions = torch.arange(self.action_count, device=state.belief.device)
        candidate_logits = []
        for action in actions:
            expanded = torch.full((state.belief.shape[0],), action, device=state.belief.device)
            candidate_logits.append(self.imagine(state, expanded))
        imagined = torch.stack(candidate_logits, dim=1)
        goal_index = goal_state.long().view(-1, 1, 1).expand(-1, self.action_count, 1)
        goal_scores = torch.log_softmax(imagined, dim=-1).gather(2, goal_index).squeeze(-1)
        return goal_scores.argmax(dim=1), imagined

    def plan_costed(self, state: WorldState, goal_state: Tensor, action_costs: Tensor) -> tuple[Tensor, Tensor]:
        """Counterfactual action selection with an explicit action-cost budget."""
        if tuple(action_costs.shape) != (self.action_count,):
            raise ValueError("action_costs must contain one cost per action")
        _actions, imagined = self.plan(state, goal_state)
        goal_index = goal_state.long().view(-1, 1, 1).expand(-1, self.action_count, 1)
        goal_scores = torch.log_softmax(imagined, dim=-1).gather(2, goal_index).squeeze(-1)
        utility = goal_scores - action_costs.to(goal_scores.device).view(1, -1)
        return utility.argmax(dim=1), utility


class StatelessWorldModel(nn.Module):
    """Ablation that receives no earlier environmental context."""

    def __init__(self, observation_dim: int, *, latent_dim: int = 32, state_count: int = 3, action_count: int = 2):
        super().__init__()
        self.action_count = action_count
        self.state_count = state_count
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_count, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_count),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        one_hot = torch.nn.functional.one_hot(action.long(), self.action_count).to(observation.dtype)
        return self.network(torch.cat((observation, one_hot), dim=-1))

    def plan(self, observation: Tensor, goal_state: Tensor) -> Tensor:
        candidates = []
        for action in range(self.action_count):
            action_tensor = torch.full((observation.shape[0],), action, device=observation.device)
            candidates.append(self.forward(observation, action_tensor))
        imagined = torch.stack(candidates, dim=1)
        goal_index = goal_state.long().view(-1, 1, 1).expand(-1, self.action_count, 1)
        return imagined.gather(2, goal_index).squeeze(-1).argmax(dim=1)


class RolloutWorldModel(nn.Module):
    """Context-persistent discrete dynamics with differentiable multi-step rollout."""

    def __init__(self, *, state_count: int = 5, latent_dim: int = 32, action_count: int = 2, use_context: bool = True):
        super().__init__()
        self.state_count = state_count
        self.action_count = action_count
        self.use_context = use_context
        self.context_encoder = nn.Sequential(nn.Linear(1, latent_dim), nn.Tanh())
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + state_count + action_count, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, state_count),
        )

    def encode_context(self, context: Tensor) -> Tensor:
        if not self.use_context:
            return torch.zeros(context.shape[0], self.context_encoder[0].out_features, device=context.device)
        return self.context_encoder(context)

    def predict(self, belief: Tensor, state_distribution: Tensor, action: Tensor) -> Tensor:
        one_hot = torch.nn.functional.one_hot(action.long(), self.action_count).to(belief.dtype)
        return self.transition(torch.cat((belief, state_distribution, one_hot), dim=-1))

    def rollout(self, belief: Tensor, start_state: Tensor, actions: Tensor) -> Tensor:
        """Return state probabilities for every action in a batch sequence."""
        state = torch.nn.functional.one_hot(start_state.long(), self.state_count).to(belief.dtype)
        trajectories = []
        for index in range(actions.shape[1]):
            state = torch.softmax(self.predict(belief, state, actions[:, index]), dim=-1)
            trajectories.append(state)
        return torch.stack(trajectories, dim=1)

    def plan(self, belief: Tensor, start_state: Tensor, goal_state: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
        sequences = torch.cartesian_prod(*[torch.arange(self.action_count, device=belief.device) for _ in range(horizon)])
        if horizon == 1:
            sequences = sequences.unsqueeze(1)
        choices = sequences.shape[0]
        expanded_belief = belief.repeat_interleave(choices, dim=0)
        expanded_start = start_state.repeat_interleave(choices, dim=0)
        expanded_goal = goal_state.repeat_interleave(choices, dim=0)
        expanded_actions = sequences.repeat(belief.shape[0], 1)
        terminal = self.rollout(expanded_belief, expanded_start, expanded_actions)[:, -1, :]
        scores = terminal.gather(1, expanded_goal.long().unsqueeze(1)).squeeze(1).view(belief.shape[0], choices)
        best = scores.argmax(dim=1)
        return sequences[best], scores
