"""DecideStream RL: cross-episode persistent memory in a continuous stream.

A continuous event stream is composed of ``units`` episodes, each
[identity frame, gap empty frames, query frame].  Each query's one-hot channel
selects either the CURRENT episode's identity (recency) or the PREVIOUS
episode's identity (cross-episode persistence).  Reward is +1 only when the
chosen identity matches the queried target.

The stream never resets between episodes, so a controller must retain earlier
identities while newer episodes keep arriving.  This is the regime where a
compressed recurrent hidden state (GRU) is diluted by intervening episodes,
while content-addressable slots (evidence / PLWM) can still retrieve the
earlier identity.  The counterfactual swap flips the identity->choice mapping.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.torch_model import TorchPLWM
from plwm.jeplem_memory import JEPLMMemory
from plwm.jeplem_agent import JEPLMAgent
from plwm.jeplem_predictor import JEPLMPredictor


class DecideStreamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, units: int = 3, gap: int = 4, identity_count: int = 2, seed: int = 0, switch_at: int | None = None, query_horizon: int = 2, confounded: bool = False, clue_frames: int = 1, clue_reliability: float = 0.65) -> None:
        if units < 2:
            raise ValueError("stream needs at least two episodes for cross-episode queries")
        self.units, self.gap, self.identity_count = units, gap, identity_count
        self.switch_at = switch_at if switch_at is not None else units + 1
        # confounded mode: the identity is revealed through ``clue_frames``
        # noisy clues instead of a single clean one-hot frame.  Each clue is
        # the identity one-hot with probability ``clue_reliability`` and the
        # OTHER identity channel with probability (1 - clue_reliability), so a
        # single clue cannot identify the identity (P(o|s) overlap) but the
        # clue history can (Bayesian accumulation).  This turns the task from
        # "observation == state" into a genuine POMDP: prediction error now
        # carries information (identity belief) not directly readable from a
        # single observation -- the JEPA value niche identified in phase 30.
        self.confounded = confounded
        self.clue_frames = max(1, clue_frames if confounded else 1)
        if confounded and self.clue_frames > gap:
            raise ValueError(f"clue_frames ({self.clue_frames}) must be <= gap ({gap}) so clue block does not collide with the query frame")
        self.clue_reliability = clue_reliability
        self.unit_len = gap + 1  # query frame sits at position == gap
        self.total_steps = units * self.unit_len
        self.query_horizon = query_horizon
        self._clue_flips: list[list[bool]] | None = None
        # Note: in confounded mode with clue_frames == gap there are NO empty
        # frames, so the memory-content aux head silently sees no aux frames
        # (frame_mask is empty).  Phase-31 used this configuration; the three
        # compared policies all had aux equally disabled, so the comparison
        # stays controlled, but the phase-31 numbers are aux-free baselines.
        self.input_dim = identity_count + max(2, query_horizon)  # presence + query channels
        self.action_space = gym.spaces.Discrete(identity_count)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.input_dim,), dtype=np.float32)
        self._seed = seed

    def target_action(self, unit: int, target_unit: int) -> int:
        """Identity->choice mapping.  Rule 0 (before switch_at): identity and
        choice agree.  Rule 1 (from switch_at): choice is the complement, so
        the controller must detect the rule change from reward feedback."""
        identity = self._identities[target_unit]
        rule = 0 if unit < self.switch_at else 1
        return identity if rule == 0 else (1 - identity)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = random.Random(seed if seed is not None else self._seed)
        self._identities = [rng.randrange(self.identity_count) for _ in range(self.units)]
        self._query_types = [
            rng.randrange(1) if unit == 0 else rng.randrange(self.query_horizon)
            for unit in range(self.units)
        ]  # 0 = current, 1 = previous, 2 = two-back (query_horizon=3)
        # Confounded mode: per-episode clue flip masks.  Seeded by a string
        # hash of the episode seed so re-evaluation of the same episode
        # regenerates identical clues, while adjacent integer seeds get fully
        # decorrelated patterns (plain integer seeds share MT initial-state
        # correlation, which made the flip-pattern distribution non-uniform).
        if self.confounded:
            clue_rng = random.Random(f"clue-{seed if seed is not None else self._seed}")
            self._clue_flips = [
                [clue_rng.random() >= self.clue_reliability for _ in range(self.clue_frames)]
                for _ in range(self.units)
            ]
        self._frame = 0
        return self._observation(0), {}

    def _observation(self, frame: int) -> np.ndarray:
        unit, position = divmod(frame, self.unit_len)
        vector = np.zeros(self.input_dim, dtype=np.float32)
        if position < self.clue_frames:
            # identity clue frame.  Confounded mode: each clue independently
            # shows the true identity channel with probability p and the
            # opposite channel with probability 1-p (flip mask decided at
            # reset, per episode).  Single-frame identity is ambiguous; the
            # multi-frame clue history is not.
            identity = self._identities[unit]
            if not self.confounded:
                vector[identity] = 1.0
            else:
                shown = identity if not self._clue_flips[unit][position] else (1 - identity)
                vector[shown] = 1.0
        elif position == self.gap:
            vector[self.identity_count + self._query_types[unit]] = 1.0
        return vector
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        unit, position = divmod(frame, self.unit_len)
        reward = 0.0
        if position == self.gap:  # query frame
            target_unit = max(0, unit - self._query_types[unit])
            if int(action) == self.target_action(unit, target_unit):
                reward = 1.0
        self._frame += 1
        terminated = self._frame >= self.total_steps
        observation = self._observation(min(self._frame, self.total_steps - 1))
        return observation, reward, terminated, False, {}


class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, width)
        self.head = nn.Linear(width, num_actions + 1)
        self._hidden: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one frame for a batch of episodes; returns (logits, value)."""
        candidate = self.cell(observation_step, self._hidden)
        active = valid_step.unsqueeze(-1)
        self._hidden = torch.where(active, candidate, self._hidden)
        output = self.head(self._hidden)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PolicyGRUAux(nn.Module):
    """Streaming GRU policy with a memory-content prediction head: at every
    empty frame (between identity and query) the aux head predicts the current
    episode's identity one-hot.  Tests whether explicit memory-content
    prediction rescues GRU's cross-episode memory under RL (as it does in the
    single-episode setting, JEPLM_AUX_REPORT.md)."""

    def __init__(self, input_dim: int, num_actions: int, width: int = 64, aux_coefficient: float = 0.5, aux_timescales: int = 1) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, width)
        self.head = nn.Linear(width, num_actions + 1)
        self.aux_head = nn.Linear(width, 2 * aux_timescales)
        self.aux_coefficient = aux_coefficient
        self.aux_timescales = aux_timescales
        self._hidden: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        candidate = self.cell(observation_step, self._hidden)
        active = valid_step.unsqueeze(-1)
        self._hidden = torch.where(active, candidate, self._hidden)
        output = self.head(self._hidden)
        return output[..., :-1], output[..., -1], self.aux_head(self._hidden)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
            aux_list.append(aux)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1), torch.stack(aux_list, dim=1)


class PolicyGRUAux2(nn.Module):
    """Dual-timescale memory-content prediction: the aux head outputs both the
    current and the previous episode's identity (4 logits); loss supervises
    both.  Tests whether GRU's implicit state can hold two timescales when
    explicitly trained to (single-head aux kept only one: query xor cross)."""

    def __init__(self, input_dim: int, num_actions: int, width: int = 64, aux_coefficient: float = 0.5) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, width)
        self.head = nn.Linear(width, num_actions + 1)
        self.aux_head = nn.Linear(width, 4)  # [cur(2), prev(2)]
        self.aux_coefficient = aux_coefficient
        self._hidden: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        candidate = self.cell(observation_step, self._hidden)
        active = valid_step.unsqueeze(-1)
        self._hidden = torch.where(active, candidate, self._hidden)
        output = self.head(self._hidden)
        return output[..., :-1], output[..., -1], self.aux_head(self._hidden)

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
            aux_list.append(aux)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1), torch.stack(aux_list, dim=1)


class PolicyEvidence(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, width: int = 64, slots: int = 8) -> None:
        super().__init__()
        self.slots = slots
        self.num_actions = num_actions
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width * 2, num_actions + 1)
        self._state: Tensor | None = None
        self._keys: Tensor | None = None
        self._values: Tensor | None = None
        self._final_state: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        width = self.encoder[0].out_features
        self._state = torch.zeros(batch, width, device=device)
        self._keys = torch.zeros(batch, self.slots, width, device=device)
        self._values = torch.zeros_like(self._keys)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(encoded), self._keys) / self.encoder[0].out_features**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", weights, self._values)
        candidate_state = self.state(torch.cat((encoded, recalled), dim=-1), self._state)
        gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
        candidate_keys = torch.cat((self._keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
        candidate_values = torch.cat((self._values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
        active = valid_step.unsqueeze(-1)
        active_slots = active.unsqueeze(-1)
        self._state = torch.where(active, candidate_state, self._state)
        self._keys = torch.where(active_slots, candidate_keys, self._keys)
        self._values = torch.where(active_slots, candidate_values, self._values)
        self._final_state = self._state
        output = self.head(torch.cat((self._state, recalled), dim=-1))
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PolicyEvidenceAux(PolicyEvidence):
    """Evidence slot memory + dual-timescale memory-content prediction head
    (current and previous identity).  JEPLM D1+D2 combination test: does the
    semantic aux strengthen or disturb the slots' native multi-timescale?"""

    def __init__(self, input_dim: int, num_actions: int, *, width: int = 64, slots: int = 8, aux_coefficient: float = 0.5) -> None:
        super().__init__(input_dim, num_actions, width=width, slots=slots)
        self.aux_head = nn.Linear(width * 2, 4)  # [cur(2), prev(2)]
        self.aux_coefficient = aux_coefficient

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(encoded), self._keys) / self.encoder[0].out_features**0.5, dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", weights, self._values)
        candidate_state = self.state(torch.cat((encoded, recalled), dim=-1), self._state)
        gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
        candidate_keys = torch.cat((self._keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
        candidate_values = torch.cat((self._values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
        active = valid_step.unsqueeze(-1)
        active_slots = active.unsqueeze(-1)
        self._state = torch.where(active, candidate_state, self._state)
        self._keys = torch.where(active_slots, candidate_keys, self._keys)
        self._values = torch.where(active_slots, candidate_values, self._values)
        self._final_state = self._state
        output = self.head(torch.cat((self._state, recalled), dim=-1))
        return output[..., :-1], output[..., -1], self.aux_head(torch.cat((self._state, recalled), dim=-1))

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
            aux_list.append(aux)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1), torch.stack(aux_list, dim=1)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)



class PolicyJEPLM(nn.Module):
    """JEPLM memory component (plwm.jeplem_memory) wrapped as a stream policy:
    content-addressed slots + learned-gate write + dual-timescale semantic aux.
    First end-to-end assembly of the architecture on the continuous stream."""

    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64,
                 slots: int = 16, aux_timescales: int = 2, aux_coefficient: float = 0.5,
                 write_mode: str = "fifo", predict: bool = False,
                 pred_coefficient: float = 0.1, conf_feature: bool = False) -> None:
        super().__init__()
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=num_actions, aux_timescales=aux_timescales,
                                  write_mode=write_mode)
        # conf_feature: JEPA decision-side wiring -- the per-frame prediction
        # error (predictor vs current latent, stop-grad) mapped to a [0,1]
        # confidence scalar is concatenated to the head input.  This tests
        # phase-28's positioning claim that JEPA prediction belongs on the
        # decision side (as a belief signal), NOT as an RL training target.
        self.conf_feature = conf_feature
        head_in = self.memory.head_in + (1 if conf_feature else 0)
        self.head = nn.Linear(head_in, num_actions + 1)
        self.aux_coefficient = aux_coefficient
        self.predict = predict
        self.pred_coefficient = pred_coefficient
        if predict or conf_feature:
            # predictor learns the dynamics of the FULL features (state +
            # recalled), not just the state: recalled carries the slot
            # contents (e.g. noisy clues), so features dynamics are what
            # change when a clue flips.  Phase-31 probe showed state-only
            # conf cannot separate flipped/clean clues (state smooths them).
            self.predictor = JEPLMPredictor(self.memory.head_in)
        else:
            self.predictor = None
        self._pred_loss: Tensor | None = None
        self._prev_features: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)
        self._pred_loss = None
        self._prev_features = None

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features, aux = self.memory.step(observation_step, valid_step)
        self._raw_features = features.detach()
        if self.conf_feature and self.predictor is not None:
            if self._prev_features is not None:
                pred = self.predictor(self._prev_features).detach()
                err = JEPLMPredictor.prediction_error(pred, features)
                conf = JEPLMPredictor.confidence(err).unsqueeze(-1)  # [B, 1]
            else:
                conf = torch.zeros(features.shape[0], 1, device=features.device)
            features = torch.cat((features, conf), dim=-1)
            self._last_conf = conf
            self._prev_features = self._raw_features
        output = self.head(features)
        return output[..., :-1], output[..., -1], aux

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        feature_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
            if aux is not None:
                aux_list.append(aux)
            feature_list.append(self._raw_features)
        features = torch.stack(feature_list, dim=1)  # [B, T, head_in]
        if self.predictor is not None and length > 1:
            pred = self.predictor(features[:, :-1])
            target = features[:, 1:]
            valid_pair = valid[:, :-1] & valid[:, 1:]
            err = JEPLMPredictor.prediction_error(pred, target)
            self._pred_loss = err[valid_pair].mean()
        out = torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)
        return out + (torch.stack(aux_list, dim=1) if aux_list else None,)


class PolicyPLWM(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64, memory_slots: int = 8, thought_steps: int = 1, gated_memory: bool = True) -> None:
        super().__init__()
        self.core = TorchPLWM(
            input_dim=input_dim, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_actions + 1,
            gated_memory=gated_memory,
        )

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = self.core.initial_state(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        active_mask = valid_step.unsqueeze(-1)
        candidate, _surprise = self.core.step(observation_step, self._state, think=True)
        self._state = TorchCognitiveState(
            torch.where(active_mask, candidate.belief, self._state.belief),
            torch.where(active_mask, candidate.workspace, self._state.workspace),
            torch.where(active_mask.unsqueeze(-1), candidate.keys, self._state.keys),
            torch.where(active_mask.unsqueeze(-1), candidate.values, self._state.values),
            torch.where(active_mask.unsqueeze(-1), candidate.confidences, self._state.confidences),
        )
        output = self.core.classify(self._state)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


from plwm.torch_model import TorchCognitiveState


class PolicyRMTStream(nn.Module):
    """Recurrent Memory Transformer as a streaming policy.

    Learnable memory tokens persist across episodes (units); each unit is one
    transformer segment of (memory_tokens + unit_len) frames.  Memory updates
    are detached at segment boundaries (truncated BPTT), mirroring the standard
    RMT training regime.  This is the contemporary attention-with-memory
    baseline for the stream / rule-revision tasks.
    """

    def __init__(self, input_dim: int, num_actions: int, *, unit_len: int,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 memory_tokens: int = 4, dropout: float = 0.0,
                 segment_units: float = 1.0, detach_mem: bool = True) -> None:
        super().__init__()
        self.unit_len, self.memory_tokens = unit_len, memory_tokens
        # probe knobs: segment_units scales the segment boundary (1.0 = one
        # episode per segment); detach_mem=False keeps the full gradient across
        # segment boundaries (full-BPTT regime) instead of truncated BPTT.
        self.segment_units = segment_units
        self.detach_mem = detach_mem
        self._seg_target = max(1, int(round(segment_units * unit_len)))
        self.proj = nn.Linear(input_dim, d_model)
        self.memory = nn.Parameter(torch.randn(memory_tokens, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, num_actions + 1)
        self._mem: Tensor | None = None
        self._seg_len = 0
        self._seg_frames: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._mem = self.memory.unsqueeze(0).expand(batch, -1, -1).to(device)
        self._seg_len = 0
        self._seg_frames = None

    def _run(self, seg_frames: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        batch = seg_frames.shape[0]
        mem_count = mem.shape[1]
        embedded = torch.cat((mem, self.proj(seg_frames)), dim=1)  # [B, M+L, d]
        total = mem_count + seg_frames.shape[1]
        embedded = embedded + self.pos[:, :total]
        causal = torch.triu(
            torch.full((total, total), float("-inf"), device=embedded.device), diagonal=1,
        )
        out = self.encoder(embedded, mask=causal)
        new_mem = out[:, :mem_count]
        if self.detach_mem:
            new_mem = new_mem.detach()
        return out, new_mem

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        if self._seg_frames is None:
            self._seg_frames = observation_step.unsqueeze(1)
        else:
            self._seg_frames = torch.cat((self._seg_frames, observation_step.unsqueeze(1)), dim=1)
        self._seg_len += 1
        out, new_mem = self._run(self._seg_frames, self._mem)
        if self._seg_len >= self._seg_target:
            self._mem = torch.where(valid_step.unsqueeze(-1).unsqueeze(-1), new_mem, self._mem)
            self._seg_len = 0
            self._seg_frames = None
        output = self.head(out[:, -1])
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PolicyRMTStreamAux(PolicyRMTStream):
    """RMT streaming policy with a memory-content prediction head on the
    current frame representation (out[:, -1]).  Aux targets the PREVIOUS
    episode's identity (cross-episode content) -- the decisive test whether
    explicit content supervision can make the RMT memory-token compression
    learnable (vs the measured 'implicit compression unlearnable under RL')."""

    def __init__(self, input_dim: int, num_actions: int, *, unit_len: int,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                 memory_tokens: int = 4, dropout: float = 0.0,
                 segment_units: float = 1.0, detach_mem: bool = True,
                 aux_coefficient: float = 0.5) -> None:
        super().__init__(input_dim, num_actions, unit_len=unit_len, d_model=d_model,
                         n_layers=n_layers, n_heads=n_heads, memory_tokens=memory_tokens,
                         dropout=dropout, segment_units=segment_units, detach_mem=detach_mem)
        self.aux_head = nn.Linear(d_model, 2)
        self.aux_coefficient = aux_coefficient

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._seg_frames is None:
            self._seg_frames = observation_step.unsqueeze(1)
        else:
            self._seg_frames = torch.cat((self._seg_frames, observation_step.unsqueeze(1)), dim=1)
        self._seg_len += 1
        out, new_mem = self._run(self._seg_frames, self._mem)
        if self._seg_len >= self._seg_target:
            self._mem = torch.where(valid_step.unsqueeze(-1).unsqueeze(-1), new_mem, self._mem)
            self._seg_len = 0
            self._seg_frames = None
        output = self.head(out[:, -1])
        return output[..., :-1], output[..., -1], self.aux_head(out[:, -1])

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
            aux_list.append(aux)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1), torch.stack(aux_list, dim=1)


def _pad(episodes: list[list[Tensor]], device: torch.device, input_dim: int) -> tuple[Tensor, Tensor]:
    length = max(len(episode) for episode in episodes)
    values = torch.zeros((len(episodes), length, input_dim), device=device)
    valid = torch.zeros((len(episodes), length), dtype=torch.bool, device=device)
    for row, episode in enumerate(episodes):
        values[row, : len(episode)] = torch.stack(episode).to(device)
        valid[row, : len(episode)] = True
    return values, valid


@torch.no_grad()
def _collect(
    model: nn.Module, env: DecideStreamEnv, *, count: int, seed_start: int,
    device: torch.device, stochastic: bool = True,
) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]]]:
    """Batch-parallel streaming collection (O(T), full batch per frame)."""
    was_training = model.training
    model.eval()
    import copy
    envs = [copy.deepcopy(env) for _ in range(count)]
    observations = [envs[i].reset(seed=seed_start + i)[0] for i in range(count)]
    model.reset(count, device)
    histories: list[list[Tensor]] = [[] for _ in range(count)]
    action_lists: list[list[int]] = [[] for _ in range(count)]
    reward_lists: list[list[float]] = [[] for _ in range(count)]
    labels = [envs[i]._identities[:] for i in range(count)]
    done = [False] * count
    for _step in range(env.total_steps):
        active = torch.tensor([not d for d in done], dtype=torch.bool, device=device)
        if not bool(active.any()):
            break
        observation_step = torch.stack([torch.tensor(obs) for obs in observations]).to(device)
        outputs = model.step(observation_step, active)
        logits = outputs[0]
        for index in range(count):
            if done[index]:
                continue
            distribution = Categorical(logits=logits[index])
            action = int(distribution.sample().item()) if stochastic else int(logits[index].argmax().item())
            observation, reward, terminated, truncated, _info = envs[index].step(action)
            histories[index].append(observation_step[index])
            action_lists[index].append(action)
            reward_lists[index].append(reward)
            observations[index] = observation
            if terminated or truncated:
                done[index] = True
    if was_training:
        model.train()
    return histories, action_lists, reward_lists, labels


def _gae(rewards: list[list[float]], values: Tensor, valid: Tensor, *, gamma: float = 0.99, lambda_: float = 0.95) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    for row, episode in enumerate(rewards):
        estimate = torch.zeros((), device=values.device)
        for index in reversed(range(len(episode))):
            next_value = values[row, index + 1] if index + 1 < len(episode) else torch.zeros((), device=values.device)
            delta = episode[index] + gamma * next_value - values[row, index]
            estimate = delta + gamma * lambda_ * estimate
            advantages[row, index] = estimate
            returns[row, index] = estimate + values[row, index]
    return advantages, returns


def _train_ppo(
    model: nn.Module, env: DecideStreamEnv, *, updates: int, episodes_per_update: int,
    seed_start: int, device: torch.device, ppo_epochs: int = 4, entropy_coefficient: float = 0.05,
    weight_decay: float = 1e-5,
) -> tuple[int, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    total_episodes = total_successes = 0
    for update in range(updates):
        model.train()
        histories, action_lists, reward_lists, labels = _collect(
            model, env, count=episodes_per_update, seed_start=seed_start + update * episodes_per_update, device=device,
        )
        observations, valid = _pad(histories, device, env.input_dim)
        action_tensor = torch.full(valid.shape, -1, dtype=torch.long, device=device)
        for row, actions in enumerate(action_lists):
            action_tensor[row, : len(actions)] = torch.tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            old_outputs = model(observations, valid)
            old_logits, old_values = old_outputs[0], old_outputs[1]
            old_distribution = Categorical(logits=old_logits)
            old_selected = old_distribution.log_prob(action_tensor)
            advantages, returns = _gae(reward_lists, old_values, valid)
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
        for _ in range(ppo_epochs):
            outputs = model(observations, valid)
            logits, values = outputs[0], outputs[1]
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - entropy_coefficient * entropy
            if hasattr(model, "aux_head") and hasattr(model, "aux_coefficient") and len(outputs) > 2:
                # memory-content prediction on empty frames (position 1..gap-1
                # within each unit).  Single-head (2-dim) predicts the previous
                # episode's identity; dual-head (4-dim) supervises BOTH the
                # current and the previous identity.
                length = outputs[2].shape[1]
                frame_units = torch.arange(length, device=observations.device) // env.unit_len
                pos_in_unit = torch.arange(length, device=observations.device) % env.unit_len
                labels_t = torch.tensor(labels, device=observations.device)  # [B, units]
                cur_targets = labels_t[:, frame_units.clamp(max=labels_t.shape[1] - 1)]
                prev_units = (frame_units - 1).clamp(min=0)
                prev_targets = labels_t[:, prev_units]
                # empty frames = after the clue block, before the query frame
                # (position >= clue_frames keeps the aux off the noisy clues)
                frame_mask = (pos_in_unit >= env.clue_frames) & (pos_in_unit < env.gap) & (frame_units > 0)
                aux_mask = valid & frame_mask.unsqueeze(0)
                if bool(aux_mask.any()):
                    aux_logits = outputs[2]
                    if aux_logits.shape[-1] >= 6:
                        cur_loss = nn.functional.cross_entropy(aux_logits[..., :2][aux_mask], cur_targets[aux_mask])
                        prev_loss = nn.functional.cross_entropy(aux_logits[..., 2:4][aux_mask], prev_targets[aux_mask])
                        prev2_units = (frame_units - 2).clamp(min=0)
                        prev2_targets = labels_t[:, prev2_units]
                        prev2_loss = nn.functional.cross_entropy(aux_logits[..., 4:6][aux_mask], prev2_targets[aux_mask])
                        aux_loss = cur_loss + prev_loss + prev2_loss
                    elif aux_logits.shape[-1] == 4:
                        cur_loss = nn.functional.cross_entropy(aux_logits[..., :2][aux_mask], cur_targets[aux_mask])
                        prev_loss = nn.functional.cross_entropy(aux_logits[..., 2:][aux_mask], prev_targets[aux_mask])
                        aux_loss = cur_loss + prev_loss
                    else:
                        prev_onehot = torch.eye(2, device=observations.device)[prev_targets]
                        aux_loss = nn.functional.cross_entropy(aux_logits[aux_mask], prev_onehot[aux_mask].argmax(-1))
                    loss = loss + model.aux_coefficient * aux_loss
            if getattr(model, "_pred_loss", None) is not None:
                loss = loss + model.pred_coefficient * model._pred_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        successes = sum(1 for rewards in reward_lists if sum(rewards) >= env.units * 0.5)
        total_episodes += episodes_per_update
        total_successes += successes
    return total_episodes, total_successes


@torch.no_grad()
def _probe_conf_evidence(model: nn.Module, env: DecideStreamEnv, *, episodes: int, seed_start: int, device: torch.device) -> dict:
    """Phase-31 diagnosis: does the JEPA confidence feature carry information
    beyond a single observation?  In confounded mode a clue frame's conf is
    the prediction error of the previous frame's latent vs the current clue's
    latent -- if the predictor learned identity persistence, flipped (noisy)
    clues should show LOWER conf than clean ones.  If conf separates
    flipped/clean clues, the mechanism is alive and the decision-side wiring
    (current-frame conf only) is the limiting factor; if it does not, the
    predictor never learned the persistence dynamics.

    Also records conf on gap/query frames as a frame-type baseline.
    """
    conf_flipped: list[float] = []
    conf_clean: list[float] = []
    conf_query: list[float] = []
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            model.step(observation_step, valid_step)
            conf = float(model._last_conf[0, 0].item())
            if position < env.clue_frames:
                if env._clue_flips[unit][position]:
                    conf_flipped.append(conf)
                else:
                    conf_clean.append(conf)
            elif position == env.gap:
                conf_query.append(conf)
            observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")
    return {
        "conf_flipped_mean": mean(conf_flipped),
        "conf_clean_mean": mean(conf_clean),
        "conf_flipped_n": len(conf_flipped),
        "conf_clean_n": len(conf_clean),
        "conf_query_mean": mean(conf_query),
        "conf_query_n": len(conf_query),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, env: DecideStreamEnv, *, episodes: int, seed_start: int, device: torch.device) -> tuple[float, float, float, float]:
    """Return (query_accuracy, cross_episode_accuracy, before_switch_acc, after_switch_acc)."""
    model.eval()
    correct = correct_prev = total = total_prev = 0
    before_correct = before_total = after_correct = after_total = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            if position == env.gap:  # query frame
                action = int(outputs[0][0].argmax().item())
                target_unit = unit - env._query_types[unit]
                target = env.target_action(unit, target_unit)
                hit = int(action == target)
                correct += hit
                total += 1
                if env._query_types[unit] > 0:  # cross-episode (previous / two-back)
                    correct_prev += hit
                    total_prev += 1
                if unit < env.switch_at:
                    before_correct += hit
                    before_total += 1
                else:
                    after_correct += hit
                    after_total += 1
            observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
    return (
        correct / max(total, 1), correct_prev / max(total_prev, 1),
        before_correct / max(before_total, 1), after_correct / max(after_total, 1),
    )


@dataclass
class Result:
    name: str
    parameters: int
    query_accuracy: float
    cross_episode_accuracy: float
    before_switch_accuracy: float
    after_switch_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    identity_count: int
    switch_at: int | None
    updates: int
    episodes_per_update: int
    results: list[Result]
    accepted: bool
    artifact: str
    conf_probe: dict | None = None


def run(
    *, device: str | torch.device | None = None, units: int = 3, gap: int = 4,
    identity_count: int = 2, updates: int = 100, episodes_per_update: int = 24,
    evaluation_episodes: int = 64, experiment_seed: int = 41_123,
    train_seed_start: int = 10_000, heldout_seed_start: int = 1_000_000,
    switch_at: int | None = None, artifact: Path | None = None,
    rmt_segment_units: float = 1.0, rmt_detach_mem: bool = True,
    rmt_probe: bool = False, aux_memory: bool = False,
    aux_coefficient: float = 0.5, gru_focus: bool = False, rmt_aux_only: bool = False,
    evidence_aux_only: bool = False, query_horizon: int = 2, aux_timescales: int = 1,
    evidence_slots: int = 8, jeplem_only: bool = False, jeplem_slots: int = 8,
    jeplem_aux_timescales: int = 2, jeplem_write_mode: str = "fifo", jeplem_predict: bool = False,
    jeplem_conf_feature: bool = False, confounded: bool = False,
    clue_frames: int = 1, clue_reliability: float = 0.65, probe_conf: bool = False,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env = DecideStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=switch_at, query_horizon=query_horizon, confounded=confounded, clue_frames=clue_frames, clue_reliability=clue_reliability)
    input_dim = env.input_dim
    if rmt_probe:
        # probe mode: only RMT (+GRU as the gate baseline) -- cheap and
        # focused on the segment-boundary mechanism.
        factories = (
            ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
            ("rmt_mem4_policy", lambda: PolicyRMTStream(
                input_dim, identity_count, unit_len=env.unit_len, memory_tokens=4,
                segment_units=rmt_segment_units, detach_mem=rmt_detach_mem)),
            ("rmt_mem8_policy", lambda: PolicyRMTStream(
                input_dim, identity_count, unit_len=env.unit_len, memory_tokens=8,
                segment_units=rmt_segment_units, detach_mem=rmt_detach_mem)),
        )
    else:
        factories = (
            ("gru_policy", lambda: PolicyGRU(input_dim, identity_count)),
            ("evidence_policy", lambda: PolicyEvidence(input_dim, identity_count, slots=evidence_slots)),
            ("plwm_policy", lambda: PolicyPLWM(input_dim, identity_count)),
            ("plwm_slots16_policy", lambda: PolicyPLWM(input_dim, identity_count, memory_slots=16)),
            ("rmt_mem4_policy", lambda: PolicyRMTStream(
                input_dim, identity_count, unit_len=env.unit_len, memory_tokens=4,
                segment_units=rmt_segment_units, detach_mem=rmt_detach_mem)),
            ("rmt_mem8_policy", lambda: PolicyRMTStream(
                input_dim, identity_count, unit_len=env.unit_len, memory_tokens=8,
                segment_units=rmt_segment_units, detach_mem=rmt_detach_mem)),
        )
    if aux_memory:
        factories = factories + (
            ("gru_aux_policy", lambda: PolicyGRUAux(input_dim, identity_count, aux_coefficient=aux_coefficient, aux_timescales=aux_timescales)),
            ("gru_aux3_policy", lambda: PolicyGRUAux(input_dim, identity_count, aux_coefficient=aux_coefficient, aux_timescales=3)),
            ("gru_aux2_policy", lambda: PolicyGRUAux2(input_dim, identity_count, aux_coefficient=aux_coefficient)),
            ("rmt_aux_mem4_policy", lambda: PolicyRMTStreamAux(
                input_dim, identity_count, unit_len=env.unit_len, memory_tokens=4,
                aux_coefficient=aux_coefficient)),
            ("evidence_aux_policy", lambda: PolicyEvidenceAux(
                input_dim, identity_count, aux_coefficient=aux_coefficient)),
            ("jeplem_policy", lambda: PolicyJEPLM(
                input_dim, identity_count, slots=jeplem_slots, aux_coefficient=aux_coefficient,
                write_mode=jeplem_write_mode, predict=jeplem_predict, pred_coefficient=0.1,
                conf_feature=jeplem_conf_feature)),
            ("jeplem_agent_policy", lambda: JEPLMAgent(
                input_dim, identity_count, slots=jeplem_slots, aux_coefficient=aux_coefficient,
                aux_timescales=jeplem_aux_timescales, write_mode=jeplem_write_mode)),
        )
    if gru_focus:
        factories = [f for f in factories if f[0] in ("gru_policy", "gru_aux_policy", "gru_aux2_policy")]
    if rmt_aux_only:
        factories = [f for f in factories if f[0] in ("rmt_mem4_policy", "rmt_aux_mem4_policy")]
    if evidence_aux_only:
        factories = [f for f in factories if f[0] in ("evidence_policy", "evidence_aux_policy")]
    if jeplem_only:
        factories = [f for f in factories if f[0] in ("jeplem_policy", "jeplem_agent_policy")]
    results = []
    probe: dict | None = None
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(
            model, env, updates=updates, episodes_per_update=episodes_per_update,
            seed_start=train_seed_start, device=resolved,
        )
        query_acc, cross_acc, before_acc, after_acc = _evaluate(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        results.append(Result(
            name, sum(parameter.numel() for parameter in model.parameters()),
            query_acc, cross_acc, before_acc, after_acc,
        ))
        # phase-31 conf diagnosis: only meaningful for the conf_feature model
        # (jeplem_policy runs before jeplem_agent_policy, so leaving probe
        # unchanged on non-conf models keeps the diagnosis attached)
        if probe_conf and name == "jeplem_policy" and hasattr(model, "_last_conf"):
            probe = _probe_conf_evidence(
                model, env, episodes=evaluation_episodes * 4, seed_start=heldout_seed_start + 10_000, device=resolved,
            )
    by_name = {result.name: result for result in results}
    if rmt_probe or gru_focus or rmt_aux_only or evidence_aux_only or jeplem_only:
        # probe/focus modes lack the full controller set; the gate is
        # meaningless there.
        accepted = False
    else:
        accepted = (
            by_name["plwm_policy"].cross_episode_accuracy >= 0.6
            and by_name["plwm_policy"].cross_episode_accuracy >= by_name["gru_policy"].cross_episode_accuracy + 0.2
        )
    target = artifact or ROOT / "artifacts" / "decide-stream-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        f"minigrid_decide_stream_rl_v1: continuous stream of episodes, each [identity frame, gap empty frames, query frame]; query one-hot selects current or previous episode's identity; reward only on query steps; PPO on-policy; no teacher, oracle, shaped reward, or privileged state; reports overall query accuracy, cross-episode accuracy, and before/after rule-switch accuracy (switch_at); rule 0: choice == identity, rule 1: choice == complement. rmt_segment_units={rmt_segment_units}, rmt_detach_mem={rmt_detach_mem}, rmt_probe={rmt_probe}",
        str(resolved), units, gap, identity_count, switch_at, updates, episodes_per_update,
        results, accepted, str(target), conf_probe=probe,
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--switch-at", type=int, default=None, help="unit index where the identity->choice rule flips (None = never)")
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--query-horizon", type=int, default=2)
    parser.add_argument("--aux-timescales", type=int, default=1)
    parser.add_argument("--evidence-slots", type=int, default=8)
    parser.add_argument("--jeplem-only", action="store_true")
    parser.add_argument("--jeplem-slots", type=int, default=8)
    parser.add_argument("--jeplem-aux-timescales", type=int, default=2)
    parser.add_argument("--jeplem-predict", action="store_true")
    parser.add_argument("--jeplem-conf-feature", action="store_true")
    parser.add_argument("--jeplem-write-mode", type=str, default="fifo", choices=["fifo", "gate", "delta_rule"])
    parser.add_argument("--confounded", action="store_true", help="POMDP mode: identity revealed through noisy clue frames instead of one clean one-hot frame")
    parser.add_argument("--clue-frames", type=int, default=4, help="clue frames per unit in confounded mode")
    parser.add_argument("--clue-reliability", type=float, default=0.65, help="per-clue probability of showing the true identity channel")
    parser.add_argument("--probe-conf", action="store_true", help="phase-31: record JEPA conf by clue-flip status (diagnosis probe)")
    parser.add_argument("--rmt-segment-units", type=float, default=1.0)
    parser.add_argument("--no-rmt-detach", action="store_true")
    parser.add_argument("--rmt-probe", action="store_true")
    parser.add_argument("--aux-memory", action="store_true")
    parser.add_argument("--aux-coefficient", type=float, default=0.5)
    parser.add_argument("--gru-focus", action="store_true")
    parser.add_argument("--rmt-aux-only", action="store_true")
    parser.add_argument("--evidence-aux-only", action="store_true")
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, switch_at=args.switch_at, artifact=args.artifact,
        rmt_segment_units=args.rmt_segment_units,
        rmt_detach_mem=not args.no_rmt_detach, rmt_probe=args.rmt_probe,
        aux_memory=args.aux_memory, aux_coefficient=args.aux_coefficient,
        gru_focus=args.gru_focus, rmt_aux_only=args.rmt_aux_only,
        evidence_aux_only=args.evidence_aux_only, query_horizon=args.query_horizon,
        aux_timescales=args.aux_timescales, evidence_slots=args.evidence_slots,
        jeplem_only=args.jeplem_only, jeplem_slots=args.jeplem_slots,
        jeplem_aux_timescales=args.jeplem_aux_timescales,
        jeplem_write_mode=args.jeplem_write_mode,
        jeplem_predict=args.jeplem_predict,
        jeplem_conf_feature=args.jeplem_conf_feature,
        confounded=args.confounded, clue_frames=args.clue_frames,
        clue_reliability=args.clue_reliability, probe_conf=args.probe_conf,
    )
    print(json.dumps(asdict(outcome), indent=2))
