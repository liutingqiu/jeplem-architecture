"""Trainable PLWM reference implementation built on PyTorch.

The model is intentionally small. It preserves the architectural contract of
the dependency-free prototype while making the state transition trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass
class TorchCognitiveState:
    belief: Tensor
    workspace: Tensor
    keys: Tensor
    values: Tensor

    def detached_clone(self) -> "TorchCognitiveState":
        return TorchCognitiveState(
            belief=self.belief.detach().clone(),
            workspace=self.workspace.detach().clone(),
            keys=self.keys.detach().clone(),
            values=self.values.detach().clone(),
        )

    def checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "belief": self.belief.detach().cpu(),
                "workspace": self.workspace.detach().cpu(),
                "keys": self.keys.detach().cpu(),
                "values": self.values.detach().cpu(),
            },
            path,
        )

    @classmethod
    def restore(cls, path: str | Path, device: torch.device | str) -> "TorchCognitiveState":
        payload = torch.load(path, map_location=device, weights_only=True)
        return cls(
            belief=payload["belief"],
            workspace=payload["workspace"],
            keys=payload["keys"],
            values=payload["values"],
        )


class TorchPLWM(nn.Module):
    """A recurrent predictive model with persistent state and episodic recall."""

    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 48,
        memory_slots: int = 8,
        thought_steps: int = 3,
        output_dim: int = 1,
        use_memory: bool = True,
        gated_memory: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_slots = memory_slots
        self.thought_steps = thought_steps
        self.use_memory = use_memory
        self.gated_memory = gated_memory

        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.Tanh())
        self.predictor = nn.Linear(latent_dim, input_dim)
        self.belief_cell = nn.GRUCell(latent_dim * 2, latent_dim)
        self.key_head = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value_head = nn.Linear(latent_dim, latent_dim)
        self.memory_write_gate = nn.Linear(latent_dim + 1, 1)
        if gated_memory:
            nn.init.constant_(self.memory_write_gate.bias, 2.0)
        self.query_head = nn.Linear(latent_dim, latent_dim, bias=False)
        self.workspace_cell = nn.GRUCell(latent_dim * 2, latent_dim)
        self.readout = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, output_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> TorchCognitiveState:
        zeros = torch.zeros(batch_size, self.latent_dim, device=device)
        memory = torch.zeros(batch_size, self.memory_slots, self.latent_dim, device=device)
        return TorchCognitiveState(zeros, zeros.clone(), memory, memory.clone())

    def _read_memory(self, query: Tensor, state: TorchCognitiveState) -> Tensor:
        if not self.use_memory:
            return torch.zeros_like(state.belief)
        scores = torch.einsum("bd,bmd->bm", query, state.keys) / (self.latent_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bm,bmd->bd", weights, state.values)

    def step(
        self,
        observation: Tensor,
        state: TorchCognitiveState,
        *,
        think: bool = False,
        write_candidates: Tensor | None = None,
        write_mask: Tensor | None = None,
    ) -> tuple[TorchCognitiveState, Tensor]:
        """Advance one event, optionally writing several latent candidates.

        The default path writes the original single event summary exactly as
        before. Multi-candidate writers supply latent vectors derived only
        from their current public event. Each ``write_mask`` entry controls
        allocation of its corresponding candidate, so inactive candidates do
        not consume or evict an episodic slot.
        """
        if observation.ndim != 2 or observation.shape != (state.belief.shape[0], self.encoder[0].in_features):
            raise ValueError("observation must match the state batch and input dimension")
        encoded = self.encoder(observation)
        prediction = self.predictor(state.belief)
        surprise = (prediction - observation).square().mean(dim=-1, keepdim=True)

        query = self.query_head(encoded)
        retrieved = self._read_memory(query, state)
        correction = torch.cat((encoded, retrieved), dim=-1)
        belief = self.belief_cell(correction, state.belief)

        if write_candidates is None:
            write_source = encoded.unsqueeze(1)
            write_values = self.value_head(belief).unsqueeze(1)
        else:
            if (
                write_candidates.ndim != 3
                or write_candidates.shape[0] != observation.shape[0]
                or write_candidates.shape[2] != self.latent_dim
                or not 1 <= write_candidates.shape[1] <= self.memory_slots
            ):
                raise ValueError("write candidates must have [batch, 1..memory_slots, latent_dim] shape")
            if write_candidates.device != observation.device or write_candidates.dtype != encoded.dtype:
                raise ValueError("write candidates must share the observation device and floating-point dtype")
            write_source = write_candidates
            write_values = self.value_head(write_source)
        writes = write_source.shape[1]
        if write_mask is None:
            active_writes = torch.ones(observation.shape[0], writes, device=observation.device, dtype=observation.dtype)
        else:
            if write_mask.shape != (observation.shape[0], writes):
                raise ValueError("write mask must match the batch and candidate count")
            active_writes = write_mask.to(device=observation.device, dtype=observation.dtype)
            if not torch.isfinite(active_writes).all() or bool((active_writes < 0.0).any()) or bool((active_writes > 1.0).any()):
                raise ValueError("write mask values must be finite probabilities")
        keys, values = state.keys, state.values
        salience = observation.abs().sum(dim=-1, keepdim=True).clamp(max=1.0)
        for index in range(writes):
            candidate_keys = torch.cat((keys[:, 1:, :], self.key_head(write_source[:, index]).unsqueeze(1)), dim=1)
            candidate_values = torch.cat((values[:, 1:, :], write_values[:, index].unsqueeze(1)), dim=1)
            if self.gated_memory:
                # A differentiable per-candidate gate preserves existing facts
                # when a public candidate is absent or not salient.
                learned_gate = torch.sigmoid(
                    self.memory_write_gate(torch.cat((write_source[:, index], surprise), dim=-1))
                ).unsqueeze(1)
                write_gate = learned_gate * salience.unsqueeze(1) * active_writes[:, index : index + 1].unsqueeze(-1)
            else:
                write_gate = active_writes[:, index : index + 1].unsqueeze(-1)
            keys = keys * (1.0 - write_gate) + candidate_keys * write_gate
            values = values * (1.0 - write_gate) + candidate_values * write_gate
        workspace = state.workspace
        if think:
            for _ in range(self.thought_steps):
                workspace = self.workspace_cell(torch.cat((belief, retrieved), dim=-1), workspace)

        next_state = TorchCognitiveState(belief, workspace, keys, values)
        return next_state, surprise

    def classify(self, state: TorchCognitiveState) -> Tensor:
        query = self.query_head(state.belief)
        retrieved = self._read_memory(query, state)
        return self.readout(torch.cat((state.belief, state.workspace, retrieved), dim=-1))

    def think_only(self, state: TorchCognitiveState, *, steps: int | None = None) -> TorchCognitiveState:
        """Advance workspace without an observation, decoder call, or memory write."""
        workspace = state.workspace
        query = self.query_head(state.belief)
        retrieved = self._read_memory(query, state)
        for _ in range(self.thought_steps if steps is None else steps):
            workspace = self.workspace_cell(torch.cat((state.belief, retrieved), dim=-1), workspace)
        return TorchCognitiveState(state.belief, workspace, state.keys, state.values)

    def forward(self, observations: Tensor, *, think_on_last: bool = True) -> tuple[Tensor, Tensor, TorchCognitiveState]:
        """Run an event sequence shaped ``[batch, time, input_dim]``."""
        state = self.initial_state(observations.shape[0], observations.device)
        surprises = []
        for index in range(observations.shape[1]):
            state, surprise = self.step(
                observations[:, index, :],
                state,
                think=think_on_last and index == observations.shape[1] - 1,
            )
            surprises.append(surprise)
        return self.classify(state), torch.cat(surprises, dim=1), state


class StatelessBaseline(nn.Module):
    """Deliberately lacks an across-event state; it sees only the final event."""

    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations[:, -1, :])
