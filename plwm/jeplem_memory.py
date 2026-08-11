"""JEPLM 记忆组件：显式槽位记忆主干（可插拔，独立于策略）。

设计依据：JEPLM_MEMORY_DESIGN.md D1/D2/D3（2026-08 实验证据）
- D1 显式槽位主干：内容寻址读 + 学习门控写；槽数须覆盖任务时间跨度
  （d=64 需 16 槽、d=128 需 32 槽；跨段多时标唯一免 aux 达标的架构）
- D2 语义内容预测 aux：可选多时标头（cur/prev/two-back...），监督/RL 通用，
  目标必须可达成（真实流目标不可学时 aux 是噪声）
- D3 无重建式 surprise：写入门控为学习式（surprise 仅作可选特征），
  重建目标永不进入策略梯度

API：JEPLMMemory 是纯记忆模块——step 接收一帧观测，返回
(features, aux_logits)。策略头/aux 头由宿主模型外挂，本模块不含任务
特定头，可插拔到任何模型（RL 策略、监督分类、世界模型预测头）。
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor


class JEPLMMemory(nn.Module):
    """Explicit slot memory: content-addressed read + learned-gate write.

    Slots hold (key, value) pairs written from encoded observations; read is
    softmax attention over slot keys against a query derived from the current
    observation.  The write path fuses the encoded observation with the
    recalled content through a GRU, then writes the fused representation into
    a FIFO position under a learned gate (per-frame salience), so an
    unimportant frame does not evict a remembered one.

    Optional aux head: predicts the identity content of the last ``k``
    written identities (one one-hot block per timescale).  Loss is
    cross-entropy; the caller decides how to weight it.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        latent: int = 64,
        slots: int = 8,
        identity_count: int = 2,
        aux_timescales: int = 0,
        use_surprise_feature: bool = False,
        write_mode: str = "fifo",
        goal_dim: int = 0,
        shared_address: bool = False,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.latent = latent
        self.identity_count = identity_count
        self.aux_timescales = aux_timescales
        self.write_mode = write_mode  # 'fifo' | 'gate' | 'delta_rule'
        self.shared_address = shared_address
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent), nn.Tanh())
        # goal-driven retrieval (integration layer, 2026-08-11): an optional
        # goal vector is projected and added to the read query, so addressing
        # is modulated by WHAT is being asked (e.g. the stream selector:
        # current vs previous episode identity), not only by the raw frame.
        # goal_dim=0 keeps the exact pre-integration behaviour (back-compat).
        self.goal_proj = nn.Linear(goal_dim, latent, bias=False) if goal_dim > 0 else None
        # shared_address: the read query uses the SAME projection as the slot
        # key, so semantic closeness in the encoded space transfers to
        # content-addressed recall. Without it, two independent random linear
        # layers (key vs query) can rotate a multi-modal semantic apart and
        # break cross-modal retrieval (Blueprint §3.3 "no bridging head").
        self.key = nn.Linear(latent, latent, bias=False)
        self.query = self.key if shared_address else nn.Linear(latent, latent, bias=False)
        self.value = nn.Linear(latent, latent)
        gate_in = latent + (1 if use_surprise_feature else 0)
        self.write_gate = nn.Linear(gate_in, 1)
        self.state_cell = nn.GRUCell(latent * 2, latent)
        self.head_in = latent * 2  # state + recalled
        if write_mode == "delta_rule":
            # erase strength (modulated by content similarity) + add strength
            self.erase_gate = nn.Linear(latent, 1)
            self.add_gate = nn.Linear(latent, 1)
        if aux_timescales > 0:
            self.aux_head = nn.Linear(self.head_in, identity_count * aux_timescales)
        else:
            self.aux_head = None
        self.use_surprise_feature = use_surprise_feature
        self._state: Tensor | None = None
        self._keys: Tensor | None = None
        self._values: Tensor | None = None

    # -- lifecycle ----------------------------------------------------------

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = torch.zeros(batch, self.latent, device=device)
        self._keys = torch.zeros(batch, self.slots, self.latent, device=device)
        self._values = torch.zeros_like(self._keys)

    def initial_state(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.zeros(batch, self.latent, device=device),
            torch.zeros(batch, self.slots, self.latent, device=device),
            torch.zeros(batch, self.slots, self.latent, device=device),
        )

    def load_state(self, state: tuple[Tensor, Tensor, Tensor]) -> None:
        self._state, self._keys, self._values = state

    def export_state(self) -> tuple[Tensor, Tensor, Tensor]:
        assert self._state is not None
        return self._state, self._keys, self._values

    # -- core ---------------------------------------------------------------

    def step(
        self,
        observation_step: Tensor,
        valid_step: Tensor,
        surprise_feature: Tensor | None = None,
        goal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """One frame: read, update state, gated write.

        ``goal`` (optional): [B, goal_dim] vector that modulates the content
        addressing (added to the projected query).  Ignored when the memory
        was built without ``goal_dim``.

        Returns (features, aux_logits): features = cat(state, recalled) for
        the host's head; aux_logits = None unless aux_timescales > 0.
        """
        assert self._state is not None and self._keys is not None and self._values is not None
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        query_vec = self.query(encoded)
        if goal is not None and self.goal_proj is not None:
            query_vec = query_vec + self.goal_proj(goal)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", query_vec, self._keys) / self.latent**0.5,
            dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", weights, self._values)
        candidate_state = self.state_cell(torch.cat((encoded, recalled), dim=-1), self._state)
        if self.use_surprise_feature:
            gate_input = torch.cat(
                (encoded, surprise_feature if surprise_feature is not None else torch.zeros(batch, 1, device=encoded.device)),
                dim=-1,
            )
        else:
            gate_input = encoded
        gate = torch.sigmoid(self.write_gate(gate_input)).view(batch, 1, 1)
        candidate_keys = torch.cat((self._keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
        candidate_values = torch.cat((self._values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
        active = valid_step.unsqueeze(-1)
        active_slots = active.unsqueeze(-1)
        new_keys, new_values = self._keys, self._values
        if self.write_mode == "fifo":
            # unconditional FIFO rotation (the de-facto behavior before 2026-08-11)
            new_keys, new_values = candidate_keys, candidate_values
        elif self.write_mode == "gate":
            # learned per-frame gate controls how much of the new content
            # replaces the recalled slot (the documented-but-unwired behavior)
            recalled_slot = recalled.unsqueeze(1)  # [B,1,D]
            new_values = (1 - gate) * recalled_slot + gate * self.value(encoded).unsqueeze(1)
            new_keys = (1 - gate) * recalled_slot + gate * self.key(encoded).unsqueeze(1)
        elif self.write_mode == "delta_rule":
            # content-addressed erase + add (KDA-style): pick the most similar
            # slot, erase it proportionally to similarity, then add new content
            # with learned strength.  Keeps the explicit slot structure.
            sim = torch.sigmoid(torch.einsum("bd,bmd->bm", self.key(encoded), self._keys))
            erase = torch.sigmoid(self.erase_gate(encoded))  # [B,1]
            add = torch.sigmoid(self.add_gate(encoded))  # [B,1]
            selected = sim.argmax(dim=-1)  # [B]
            idx = torch.arange(self.slots, device=selected.device)
            mask = (idx.unsqueeze(0) == selected.unsqueeze(-1)).float().unsqueeze(-1)  # [B,S,1]
            new_values = (1 - erase.unsqueeze(-1) * mask) * self._values + mask * (add * self.value(encoded)).unsqueeze(1)
            new_keys = (1 - erase.unsqueeze(-1) * mask) * self._keys + mask * (add * self.key(encoded)).unsqueeze(1)
        self._state = torch.where(active, candidate_state, self._state)
        self._keys = torch.where(active_slots, new_keys, self._keys)
        self._values = torch.where(active_slots, new_values, self._values)
        features = torch.cat((self._state, recalled), dim=-1)
        aux_logits = self.aux_head(features) if self.aux_head is not None else None
        return features, aux_logits

    def forward_sequence(
        self,
        observations: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Whole-sequence pass.  observations: [B, T, input_dim].

        Returns (features, aux_logits) with features [B, T, latent*2].
        """
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        feature_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            features, aux_logits = self.step(observations[:, index], valid[:, index])
            feature_list.append(features)
            if aux_logits is not None:
                aux_list.append(aux_logits)
        features = torch.stack(feature_list, dim=1)
        aux = torch.stack(aux_list, dim=1) if aux_list else None
        return features, aux
