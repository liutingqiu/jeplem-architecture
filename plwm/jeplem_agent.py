"""JEPLM Agent：架构第一版完整实现（统一入口）。

分层结构（JEPLM_MEMORY_DESIGN D1-D3 + 2026-08-10 感知层实证）：
  perception : CNN 感知器（原始观测 → 语义身份特征），预训练 + 冻结
               （联合训练被证伪：RL 梯度破坏感知器，0.743→0.427）
  memory     : JEPLMMemory（内容寻址槽位 + 学习门控写 + 语义内容预测 aux）
  policy     : 决策/值头（features → actions + value）

用途：可插拔到任何任务（RL 策略 / 监督分类 / 世界模型前段）。冻结的感知器
可在构造时切换（use_perception=False 时直接接受语义特征输入，如 presence）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from plwm.jeplem_memory import JEPLMMemory

ROOT = Path(__file__).resolve().parents[2]


class JEPLMAgent(nn.Module):
    """Perception + memory + policy head, one entry point.

    If ``use_perception`` the input is the raw observation (sensory dims +
    query dims); the frozen CNN perception module extracts identity
    semantics.  Otherwise the input is already semantic features + query.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        latent: int = 64,
        slots: int = 8,
        aux_timescales: int = 2,
        aux_coefficient: float = 0.5,
        use_perception: bool = False,
        perception_state_dict: str | None = None,
        sensory_dim: int = 159,
        query_dim: int = 2,
        write_mode: str = "fifo",
        use_goal_query: bool = False,
        goal_dim: int = 0,
        semantic_dim: int | None = None,
        semantic_space: "SemanticSpace | None" = None,
    ) -> None:
        super().__init__()
        self.use_perception = use_perception
        self.sensory_dim = sensory_dim
        self.use_goal_query = use_goal_query
        self.goal_dim = goal_dim
        if use_perception:
            from experiments.train_perception_layer import PerceptionNet

            self.perception = PerceptionNet(sensory_dim)
            if perception_state_dict:
                self.perception.load_state_dict(torch.load(perception_state_dict, map_location="cpu"))
            if semantic_dim is not None:
                # unified multimodal body: the frozen CNN's semantic (2-dim)
                # is projected into a shared semantic space before memory.
                from plwm.semantic_space import ModalityEncoder, SemanticSpace

                if semantic_space is None:
                    semantic_space = SemanticSpace(dim=semantic_dim)
                    semantic_space.register(
                        "vision", ModalityEncoder(2, dim=semantic_dim, state_dim=2)
                    )
                self.semantic_space = semantic_space
                memory_input_dim = semantic_dim + query_dim
            else:
                self.semantic_space = None
                memory_input_dim = 2 + query_dim
        else:
            self.perception = None
            self.semantic_space = None
            memory_input_dim = input_dim
        self.memory = JEPLMMemory(
            memory_input_dim, latent=latent, slots=slots,
            identity_count=num_actions, aux_timescales=aux_timescales,
            write_mode=write_mode, goal_dim=goal_dim if use_goal_query else 0,
        )
        self.head = nn.Linear(self.memory.head_in, num_actions + 1)
        self.aux_coefficient = aux_coefficient

    def reset(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor,
             modulation: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        """One frame -> (policy_logits, value, aux_logits|None).

        ``modulation`` (optional): [B, K] internal-feeling vector forwarded to
        the memory's write gate / salience buffer.  None keeps current
        behaviour.
        """
        if self.use_perception:
            raw = observation_step[..., : self.sensory_dim]
            query = observation_step[..., self.sensory_dim:]
            with torch.no_grad():  # frozen perception
                semantic = torch.softmax(self.perception(raw), dim=-1)[..., 1:3]
            if self.semantic_space is not None:
                semantic = self.semantic_space("vision", semantic)
            memory_input = torch.cat((semantic, query), dim=-1)
        else:
            memory_input = observation_step
        goal = None
        if self.use_goal_query and self.goal_dim > 0:
            # goal-driven retrieval: the trailing goal channels (e.g. the
            # stream selector: current vs previous identity) address memory.
            goal = observation_step[..., -self.goal_dim :]
        features, aux = self.memory.step(memory_input, valid_step, goal=goal, modulation=modulation)
        output = self.head(features)
        return output[..., :-1], output[..., -1], aux

    def forward(self, observations: Tensor, valid: Tensor,
                modulation: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        aux_list: list[Tensor] = []
        for index in range(length):
            logits, value, aux = self.step(observations[:, index], valid[:, index], modulation=modulation)
            logits_list.append(logits)
            values_list.append(value)
            if aux is not None:
                aux_list.append(aux)
        out = torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)
        return out + (torch.stack(aux_list, dim=1) if aux_list else None,)
