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
    ) -> None:
        super().__init__()
        self.slots = slots
        self.latent = latent
        self.identity_count = identity_count
        self.aux_timescales = aux_timescales
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent), nn.Tanh())
        self.query = nn.Linear(latent, latent, bias=False)
        self.key = nn.Linear(latent, latent, bias=False)
        self.value = nn.Linear(latent, latent)
        gate_in = latent + (1 if use_surprise_feature else 0)
        self.write_gate = nn.Linear(gate_in, 1)
        self.state_cell = nn.GRUCell(latent * 2, latent)
        self.head_in = latent * 2  # state + recalled
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
    ) -> tuple[Tensor, Tensor | None]:
        """One frame: read, update state, gated write.

        Returns (features, aux_logits): features = cat(state, recalled) for
        the host's head; aux_logits = None unless aux_timescales > 0.
        """
        assert self._state is not None and self._keys is not None and self._values is not None
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(encoded), self._keys) / self.latent**0.5,
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
        self._state = torch.where(active, candidate_state, self._state)
        self._keys = torch.where(active_slots, candidate_keys, self._keys)
        self._values = torch.where(active_slots, candidate_values, self._values)
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
