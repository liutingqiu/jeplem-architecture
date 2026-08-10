"""Real-stream cross-episode continuous RL: decide with real MemoryS13
observations, injected query signal, and persistent state across episodes.

Each episode is [real identity frame (Key or Ball visible) -> gap real empty
frames -> query frame (real empty observation + injected one-hot: current or
previous episode)].  The stream does not reset between episodes, so GRU hidden
state / slot memory / RMT segments persist.  Reward only on query steps (choose
the remembered identity).  PPO on-policy, no teacher / oracle / shaped reward.

This is the real-observation analogue of DecideStreamEnv
(train_decide_stream_rl.py): it answers whether the cross-episode memory
difference (addressable slots > implicit compression, RMT streaming collapse)
holds on genuine MemoryS13 sensor streams instead of synthetic presence frames.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device
from plwm.minigrid_features import visible_object_presence
from plwm.jeplem_memory import JEPLMMemory
from experiments.train_perception_layer import PerceptionNet, SENSORY_DIM
from experiments.train_minigrid_memory_actor_critic import encode_observation
from experiments.train_decide_stream_rl import (
    PolicyGRU, PolicyEvidence, PolicyPLWM, PolicyRMTStream, PolicyGRUAux,
    _collect, _train_ppo, _evaluate,
)

ENVIRONMENT = "MiniGrid-MemoryS13-v0"
SENSORY_DIM = 159  # encode_observation(observation_only): 147 image + 4 direction + 7 action + 1 reward
QUERY_DIM = 2
INPUT_DIM = SENSORY_DIM + QUERY_DIM

_FRAME_CACHE: dict[int, list[Tensor]] = {}


def _identity_label(frames: list[Tensor], frame_index: int) -> int | None:
    """Key/Ball presence from the real encoded frame; None if neither/both."""
    present = visible_object_presence(frames[frame_index].unsqueeze(0)).squeeze(0)
    if present.sum() != 1:
        return None
    return int(present.argmax().item())


def _collect_real_frames(seed: int, max_frames: int = 512) -> list[Tensor]:
    """Run a MemoryS13 random walk and return encoded observations (cached)."""
    if seed in _FRAME_CACHE:
        return _FRAME_CACHE[seed]
    rng = random.Random(77_331 + seed)
    environment = gym.make(ENVIRONMENT, max_steps=64)
    frames: list[Tensor] = []
    try:
        observation, _info = environment.reset(seed=seed)
        while len(frames) < max_frames:
            frames.append(encode_observation(observation))
            observation, _reward, terminated, truncated, _info = environment.step(rng.randrange(3))
            if terminated or truncated:
                observation, _info = environment.reset(seed=seed + len(frames))
    finally:
        environment.close()
    _FRAME_CACHE[seed] = frames
    return frames


def _extract_episode(frames: list[Tensor], start: int, gap: int) -> list[int] | None:
    """Try to build one episode [identity, (gap-1) empties, query] starting at
    an identity frame; returns frame indices (length gap+1), or None if the
    surrounding frames don't satisfy the clean structure.  Query sits at
    position ``gap`` (the last frame), matching DecideStreamEnv semantics."""
    identity = _identity_label(frames, start)
    if identity is None:
        return None
    for offset in range(1, gap + 1):
        if start + offset >= len(frames):
            return None
        if _identity_label(frames, start + offset) is not None:
            return None  # an object appears inside the gap or at the query slot
    return list(range(start, start + gap + 1))


class DecideStreamRealEnv(gym.Env):
    """Continuous stream of real-observation episodes with injected queries."""

    metadata = {"render_modes": []}

    def __init__(self, *, units: int = 3, gap: int = 4, seed: int = 0,
                 use_presence: bool = False, use_perception: bool = False,
                 perception: PerceptionNet | None = None, hard_perception: bool = False) -> None:
        self.units, self.gap, self.seed = units, gap, seed
        self.use_presence = use_presence
        self.use_perception = use_perception
        self.perception = perception
        self.hard_perception = hard_perception
        self.switch_at = units + 1
        self.unit_len = gap + 1
        self.total_steps = units * self.unit_len
        self.input_dim = (2 if (use_presence or use_perception) else SENSORY_DIM) + QUERY_DIM
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (INPUT_DIM,), dtype=np.float32)
        self._sequence: list[Tensor] | None = None
        self._labels: list[int] | None = None
        self._query_types: list[int] | None = None

    def target_action(self, unit: int, target_unit: int) -> int:
        return self._labels[target_unit]

    @property
    def _identities(self) -> list[int]:
        """Alias used by the shared streaming _collect/_train_ppo (which read
        env._identities for the aux target)."""
        return self._labels

    def _build_stream(self, seed: int) -> tuple[list[Tensor], list[int], list[int]]:
        # Collect frames from several walk seeds until we have enough clean
        # episodes of BOTH identities (interleaving needs label coverage).
        frames_all: list[Tensor] = []
        candidates: list[tuple[int, int]] = []  # (frame_start, label)
        for attempt in range(4):
            frames = _collect_real_frames(seed + attempt * 997, max_frames=512)
            base = len(frames_all)
            frames_all.extend(frames)
            index = 0
            while index < len(frames) - self.gap - 1:
                episode = _extract_episode(frames, index, self.gap)
                if episode is None:
                    index += 1
                    continue
                label = _identity_label(frames, episode[0])
                assert label is not None
                candidates.append((base + episode[0], label))
                index = episode[-1] + 1
            by_label = {0: sum(1 for _, l in candidates if l == 0),
                        1: sum(1 for _, l in candidates if l == 1)}
            if len(candidates) >= self.units and by_label[0] >= (self.units + 1) // 2 and by_label[1] >= self.units // 2:
                break
        frames = frames_all
        if len(candidates) < self.units:
            raise RuntimeError(f"insufficient clean real-stream episodes (got {len(candidates)}/{self.units})")
        by_label_starts: dict[int, list[int]] = {0: [], 1: []}
        for frame_start, label in candidates:
            by_label_starts[label].append(frame_start)
        if len(by_label_starts[0]) == 0 or len(by_label_starts[1]) == 0:
            raise RuntimeError("degenerate stream: only one identity present in this seed's frames")
        # Natural order: slide a window over the candidates (stream order,
        # no alternation forcing) and take the first window that contains BOTH
        # identities.  A single-identity stream would be degenerate (solvable
        # by a constant output); a deterministic alternation would be a
        # learnable pattern shortcut.  Window keeps relative stream order.
        episodes: list[list[int]] | None = None
        for start in range(len(candidates) - self.units + 1):
            window = candidates[start:start + self.units]
            labels_in_window = {label for _frame_start, label in window}
            if len(labels_in_window) == 2:
                episodes = [list(range(fs, fs + self.gap + 1)) for fs, _label in window]
                break
        if episodes is None:
            raise RuntimeError("degenerate stream: no window contains both identities")
        sequence: list[Tensor] = []
        labels: list[int] = []
        query_types: list[int] = []
        for unit, episode in enumerate(episodes):
            identity = _identity_label(frames, episode[0])
            assert identity is not None
            labels.append(identity)
            query_type = 1 if (unit > 0 and random.Random(seed + unit).randrange(2) == 1) else 0
            query_types.append(query_type)
            for position, frame_index in enumerate(episode):
                if self.use_presence:
                    obs = visible_object_presence(frames[frame_index].unsqueeze(0)).squeeze(0).clone()
                elif self.use_perception:
                    assert self.perception is not None
                    with torch.no_grad():
                        device = next(self.perception.parameters()).device
                        frame_t = frames[frame_index].unsqueeze(0).to(device)
                        logits = self.perception(frame_t).squeeze(0).cpu()
                    if self.hard_perception:
                        argmax = int(logits.argmax().item())
                        obs = torch.zeros(2)
                        if argmax in (1, 2):
                            obs[argmax - 1] = 1.0
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        obs = probs[1:3].clone()  # [p_key, p_ball]
                else:
                    obs = frames[frame_index].clone()
                query_onehot = torch.zeros(QUERY_DIM)
                if position == self.gap:  # query frame (last position)
                    query_onehot[query_type] = 1.0
                sequence.append(torch.cat((obs, query_onehot)))
        return sequence, labels, query_types

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        seed_value = seed if seed is not None else self.seed
        self._sequence, self._labels, self._query_types = self._build_stream(seed_value)
        self._frame = 0
        return self._sequence[0].numpy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        self._frame += 1
        unit, position = divmod(frame, self.unit_len)
        reward = 0.0
        if position == self.gap:  # query frame
            target_unit = unit - self._query_types[unit]
            if int(action) == self._labels[target_unit]:
                reward = 1.0
        terminated = self._frame >= len(self._sequence)
        observation = self._sequence[min(self._frame, len(self._sequence) - 1)].numpy()
        return observation, reward, terminated, False, {}


@dataclass
class Result:
    name: str
    parameters: int
    query_accuracy: float
    cross_episode_accuracy: float
    swap_query_accuracy: float
    swap_cross_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    updates: int
    episodes_per_update: int
    results: list[Result]
    artifact: str


def _evaluate_swap(
    model: nn.Module, env: DecideStreamRealEnv, *, episodes: int, seed_start: int, device: torch.device,
) -> tuple[float, float]:
    """Swap-counterfactual evaluation (RL semantics): force the flipped
    action.  A genuine memory policy scores high on orig and collapses on swap
    (evidence = orig - swap); a positional/visual shortcut scores the same on
    both.  Required by the frozen protocol for all memory claims."""
    model.eval()
    correct = correct_prev = total = total_prev = 0
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        for frame in range(env.total_steps):
            unit, position = divmod(frame, env.unit_len)
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            if position == env.gap:  # query frame
                action = 1 - int(outputs[0][0].argmax().item())  # flipped
                target_unit = unit - env._query_types[unit]
                target = env.target_action(unit, target_unit)
                hit = int(action == target)
                correct += hit
                total += 1
                if env._query_types[unit] == 1:
                    correct_prev += hit
                    total_prev += 1
            observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
    return (correct / max(total, 1), correct_prev / max(total_prev, 1))



class PolicyJEPLM(nn.Module):
    """JEPLM memory component as a stream policy (same wrapper as the
    synthetic-stream script): content-addressed slots + learned-gate write +
    dual-timescale semantic aux.  Eco-validity check on real timing + presence."""

    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64,
                 slots: int = 8, aux_timescales: int = 2, aux_coefficient: float = 0.5) -> None:
        super().__init__()
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=num_actions, aux_timescales=aux_timescales)
        self.head = nn.Linear(self.memory.head_in, num_actions + 1)
        self.aux_coefficient = aux_coefficient

    def reset(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features, aux = self.memory.step(observation_step, valid_step)
        output = self.head(features)
        return output[..., :-1], output[..., -1], aux

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
            if aux is not None:
                aux_list.append(aux)
        out = torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)
        return out + (torch.stack(aux_list, dim=1) if aux_list else None,)


def run(
    *, device: str | torch.device | None = None, units: int = 3, gap: int = 4,
    updates: int = 100, episodes_per_update: int = 24, evaluation_episodes: int = 32,
    experiment_seed: int = 41_123, train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000, use_presence: bool = False,
    rmt_probe: bool = False, aux_memory: bool = False, aux_coefficient: float = 0.5,
    jeplem_only: bool = False, use_perception: bool = False, hard_perception: bool = False,
    artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    perception = None
    if use_perception:
        perception = PerceptionNet(SENSORY_DIM).to(resolved)
        perception.load_state_dict(torch.load(ROOT / "artifacts" / "perception-layer.pt", map_location=resolved))
        perception.eval()
    env = DecideStreamRealEnv(units=units, gap=gap, seed=experiment_seed,
                              use_presence=use_presence, use_perception=use_perception,
                              perception=perception, hard_perception=hard_perception)
    input_dim = env.input_dim
    if rmt_probe:
        factories = (
            ("gru_policy", lambda: PolicyGRU(input_dim, 2)),
            ("rmt_mem4_policy", lambda: PolicyRMTStream(input_dim, 2, unit_len=env.unit_len, memory_tokens=4)),
        )
    else:
        factories = (
            ("gru_policy", lambda: PolicyGRU(input_dim, 2)),
            ("evidence_policy", lambda: PolicyEvidence(input_dim, 2)),
            ("plwm_policy", lambda: PolicyPLWM(input_dim, 2)),
            ("rmt_mem4_policy", lambda: PolicyRMTStream(input_dim, 2, unit_len=env.unit_len, memory_tokens=4)),
        )
    if aux_memory:
        factories = factories + (
            ("gru_aux_policy", lambda: PolicyGRUAux(input_dim, 2, aux_coefficient=aux_coefficient)),
            ("jeplem_policy", lambda: PolicyJEPLM(input_dim, 2, aux_coefficient=aux_coefficient)),
        )
    if jeplem_only:
        factories = [f for f in factories if f[0] == "jeplem_policy"]
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(
            model, env, updates=updates, episodes_per_update=episodes_per_update,
            seed_start=train_seed_start, device=resolved,
        )
        query_acc, cross_acc, _before, _after = _evaluate(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        swap_query, swap_cross = _evaluate_swap(
            model, env, episodes=evaluation_episodes, seed_start=heldout_seed_start, device=resolved,
        )
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()),
                              query_acc, cross_acc, swap_query, swap_cross))
    target = artifact or ROOT / "artifacts" / "decide-stream-real-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "decide_stream_real_rl_v1: continuous stream of REAL MemoryS13 observation episodes [identity frame -> gap empty frames -> query frame (real obs + injected one-hot current/previous)]; reward only on query steps; PPO on-policy; no teacher, oracle, shaped reward, or privileged state; real-observation analogue of DecideStreamEnv.",
        str(resolved), units, gap, updates, episodes_per_update, results, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--use-presence", action="store_true")
    parser.add_argument("--use-perception", action="store_true")
    parser.add_argument("--hard-perception", action="store_true")
    parser.add_argument("--jeplem-only", action="store_true")
    parser.add_argument("--rmt-probe", action="store_true")
    parser.add_argument("--aux-memory", action="store_true")
    parser.add_argument("--aux-coefficient", type=float, default=0.5)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap, updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed,
        use_presence=args.use_presence, rmt_probe=args.rmt_probe,
        use_perception=args.use_perception, hard_perception=args.hard_perception,
        jeplem_only=args.jeplem_only, aux_memory=args.aux_memory, aux_coefficient=args.aux_coefficient, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
