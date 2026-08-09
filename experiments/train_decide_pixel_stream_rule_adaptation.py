"""Pixel-stream rule revision: two-stage adaptation under raw RGB input.

Combines the ecological-validity port (real 7x7x3 RGB identity frames and border
query signals, from DecidePixel) with the rule-revision protocol (stage 1 learns
rule 0, stage 2 flips to rule 1 and measures the adaptation curve on reward
feedback only).  Answers: does PLWM's memory-layer rule-revision advantage hold
when perception is real pixels in front of the memory, or is it an artifact of
the presence-feature toy stream?
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.torch_model import TorchPLWM, TorchCognitiveState
from train_decide_pixel import _frame, KEY_COLOR, BALL_COLOR, QUERY_FIRST_BORDER, QUERY_SECOND_BORDER, GRID

import gymnasium as gym

INPUT_DIM = GRID * GRID * 3  # 147
EMBED = 32


class PixelStreamEnv(gym.Env):
    """Continuous stream of pixel episodes with a rule flip at switch_at.

    Each unit is [identity RGB frame, gap empty frames, query border frame].
    Query border (red/blue) selects current or previous episode's identity.
    Rule 0 (before switch_at): choice == identity.  Rule 1: choice == complement.
    """

    def __init__(self, *, units: int = 4, gap: int = 4, identity_count: int = 2,
                 seed: int = 0, switch_at: int | None = None) -> None:
        if units < 2:
            raise ValueError("stream needs at least two episodes")
        self.units, self.gap, self.identity_count = units, gap, identity_count
        self.switch_at = switch_at if switch_at is not None else units + 1
        self.unit_len = gap + 1
        self.total_steps = units * self.unit_len
        self.input_dim = INPUT_DIM
        self.action_space = gym.spaces.Discrete(identity_count)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (INPUT_DIM,), dtype=np.float32)
        self._seed = seed

    def target_action(self, unit: int, target_unit: int) -> int:
        identity = self._identities[target_unit]
        rule = 0 if unit < self.switch_at else 1
        return identity if rule == 0 else (1 - identity)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = random.Random(seed if seed is not None else self._seed)
        self._identities = [rng.randrange(self.identity_count) for _ in range(self.units)]
        self._query_types = [
            rng.randrange(1) if unit == 0 else rng.randrange(2)
            for unit in range(self.units)
        ]
        self._frame = 0
        return self._observation(0), {}

    def _observation(self, frame: int) -> np.ndarray:
        unit, position = divmod(frame, self.unit_len)
        if position == 0:  # identity frame: colored center cell
            identity = self._identities[unit]
            return _frame(center=KEY_COLOR if identity == 0 else BALL_COLOR).numpy()
        if position == self.gap:  # query frame: colored border
            query = self._query_types[unit]
            return _frame(border=QUERY_FIRST_BORDER if query == 0 else QUERY_SECOND_BORDER).numpy()
        return _frame().numpy()  # empty frame

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        unit, position = divmod(frame, self.unit_len)
        reward = 0.0
        if position == self.gap:
            target_unit = unit - self._query_types[unit]
            if int(action) == self.target_action(unit, target_unit):
                reward = 1.0
        self._frame += 1
        terminated = self._frame >= self.total_steps
        observation = self._observation(min(self._frame, self.total_steps - 1))
        return observation, reward, terminated, False, {}


class _PixelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(), nn.Linear(64, EMBED), nn.Tanh(),
        )

    def forward(self, events: Tensor) -> Tensor:
        return self.net(events)


class PixelPolicyGRU(nn.Module):
    def __init__(self, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.encoder = _PixelEncoder()
        self.cell = nn.GRUCell(EMBED, width)
        self.head = nn.Linear(width, num_actions + 1)
        self._hidden: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._hidden = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(observation_step)
        candidate = self.cell(encoded, self._hidden)
        active = valid_step.unsqueeze(-1)
        self._hidden = torch.where(active, candidate, self._hidden)
        output = self.head(self._hidden)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list, values_list = [], []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits); values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PixelPolicyEvidence(nn.Module):
    def __init__(self, num_actions: int, *, width: int = 64, slots: int = 8) -> None:
        super().__init__()
        self.encoder = _PixelEncoder()
        self.width = width
        self.query = nn.Linear(EMBED, width, bias=False)
        self.query_final = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(EMBED, width, bias=False)
        self.value = nn.Linear(EMBED, width)
        self.write_gate = nn.Linear(EMBED, 1)
        self.state = nn.GRUCell(EMBED + width, width)
        self.head = nn.Linear(width * 2, num_actions + 1)
        self._state: Tensor | None = None
        self._keys: Tensor | None = None
        self._values: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = torch.zeros(batch, self.width, device=device)
        self._keys = torch.zeros(batch, 8, self.width, device=device)
        self._values = torch.zeros_like(self._keys)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        batch = observation_step.shape[0]
        encoded = self.encoder(observation_step)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query(encoded), self._keys) / self.width**0.5, dim=-1,
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
        final_weights = torch.softmax(
            torch.einsum("bd,bmd->bm", self.query_final(self._state), self._keys) / self.width**0.5, dim=-1,
        )
        recalled_final = torch.einsum("bm,bmd->bd", final_weights, self._values)
        output = self.head(torch.cat((self._state, recalled_final), dim=-1))
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        self.reset(batch, observations.device)
        logits_list, values_list = [], []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits); values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class PixelPolicyPLWM(nn.Module):
    def __init__(self, num_actions: int, *, latent: int = 32, memory_slots: int = 8, thought_steps: int = 1, gated_memory: bool = True) -> None:
        super().__init__()
        self.encoder = _PixelEncoder()
        self.core = TorchPLWM(
            input_dim=EMBED, latent_dim=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, output_dim=num_actions + 1, gated_memory=gated_memory,
        )

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = self.core.initial_state(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(observation_step)
        active_mask = valid_step.unsqueeze(-1)
        candidate, _surprise = self.core.step(encoded, self._state, think=True)
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
        self.core._reset_ttcd(batch, observations.device)
        logits_list, values_list = [], []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits); values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


# ---- reuse streaming collect / PPO from train_decide_stream_rl ----
from train_decide_stream_rl import _collect, _pad, _train_ppo  # noqa: E402


def _evaluate(model: nn.Module, env: PixelStreamEnv, *, episodes: int, seed_start: int, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            if position == env.gap:
                action = int(outputs[0][0].argmax().item())
                target_unit = unit - env._query_types[unit]
                correct += int(action == env.target_action(unit, target_unit))
                total += 1
            observation, _r, terminated, truncated, _i = env.step(0)
            if terminated or truncated:
                break
    return correct / max(total, 1)


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    revision: str
    stage1_updates: int
    stage2_updates: int
    adaptation_curve: dict[str, list[float]]
    accepted: bool
    artifact: str


def run(
    *, device: str | torch.device | None = None, units: int = 4, gap: int = 4,
    identity_count: int = 2, stage1_updates: int = 100, stage2_updates: int = 100,
    eval_every: int = 20, episodes_per_update: int = 24, evaluation_episodes: int = 64,
    experiment_seed: int = 41_123, train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000, artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env_rule0 = PixelStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=None)
    env_rule1 = PixelStreamEnv(units=units, gap=gap, identity_count=identity_count, switch_at=0)
    factories = (
        ("gru_policy", lambda: PixelPolicyGRU(identity_count)),
        ("evidence_policy", lambda: PixelPolicyEvidence(identity_count)),
        ("plwm_policy", lambda: PixelPolicyPLWM(identity_count)),
        ("plwm_slots16_policy", lambda: PixelPolicyPLWM(identity_count, memory_slots=16)),
    )
    curves: dict[str, list[float]] = {}
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(model, env_rule0, updates=stage1_updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start, device=resolved)
        curve: list[float] = []
        for _block in range(stage2_updates // eval_every):
            _train_ppo(model, env_rule1, updates=eval_every, episodes_per_update=episodes_per_update,
                       seed_start=train_seed_start + stage1_updates + _block * eval_every, device=resolved)
            curve.append(_evaluate(model, env_rule1, episodes=evaluation_episodes,
                                   seed_start=heldout_seed_start, device=resolved))
        curves[name] = curve
    finals = {name: curve[-1] if curve else 0.0 for name, curve in curves.items()}
    accepted = (
        finals.get("plwm_policy", 0.0) >= 0.7
        and finals["plwm_policy"] >= finals["gru_policy"] + 0.2
    )
    target = artifact or ROOT / "artifacts" / "pixel-stream-rule-adaptation-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "pixel_stream_rule_adaptation_v1: continuous stream of raw 7x7x3 RGB episodes (identity center cell, border query) with rule flip; stage 1 learns rule 0, stage 2 flips to rule 1 and measures adaptation on reward feedback only; PPO on-policy; no privileged state; ecological-validity port of the rule-revision protocol.",
        str(resolved), units, gap, "continue", stage1_updates, stage2_updates,
        curves, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--stage1-updates", type=int, default=100)
    parser.add_argument("--stage2-updates", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        stage1_updates=args.stage1_updates, stage2_updates=args.stage2_updates,
        eval_every=args.eval_every, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, artifact=args.artifact,
    )
    print(f"accepted: {outcome.accepted}")
    for name, curve in outcome.adaptation_curve.items():
        print(f"  {name}: {[round(c, 3) for c in curve]}")
