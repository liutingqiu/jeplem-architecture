"""EvidenceSeek: composite-task RL for the integration layer (2026-08-11).

Task (JEPLM_INTEGRATION_DESIGN.md section 8): K identities (Key/Ball), each
unit = [clue frame, gap empty frames, query frame].  A clue frame shows a
noisy one-hot of SOME identity's value (p=reliability true channel).  The
query frame asks "what is identity X" (X drawn uniformly, given in the
observation selector channels); the agent either ANSWERS (submit X's value,
+1 if correct, episode ends) or REQUESTS a clue for a SELF-CHOSEN identity i
(small penalty, next unit's clue frame shows identity i).

Why this composite task needs multiple verified modules:
  - explicit slot memory : noisy clues must be accumulated across frames (a
    memoryless agent is capped at reliability ~0.65);
  - belief (gap)        : each identity's confidence evolves independently;
    the "confidence gap" (1 - max posterior) is the natural request signal;
  - autonomous goal     : which identity to request = max-gap choice
    (gap_goals toy logic, transferred unchanged);
  - threshold decision  : when to stop collecting and answer.

Controls: mlp_no_memory (state-less, no memory), gru_policy (compressed
hidden state), jeplem_policy (explicit slot memory).  Probes (evaluation
only, no oracle in training): answer accuracy, answer rate, mean episode
reward, and for jeplem the request-target vs oracle-max-gap match rate.

PPO on-policy; no teacher, no oracle, no shaped reward beyond the request
penalty (which is intrinsic to the task design).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributions import Categorical

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.jeplem_memory import JEPLMMemory

ACTION_ANSWER_0 = 0
ACTION_ANSWER_1 = 1
ACTION_REQUEST_0 = 2
ACTION_REQUEST_1 = 3
N_ACTIONS = 4


class EvidenceSeekEnv(gym.Env):
    """Continuous unit stream: [clue(if requested), query] per unit.

    Clues exist ONLY when requested: an un-requested clue frame is all zeros,
    so answering without requesting is pure guessing (0.5).  This makes the
    REQUEST the only information source -- the request target choice (which
    identity) is the autonomous-goal (gap) decision.

    Observation channels (input_dim = K + K + 1 + 4 + K):
      [0:K]         clue VALUE (noisy one-hot; all-zero when un-requested)
      [K:2K]        clue ATTRIBUTION (which identity's value; zero when none)
      [2K]          query-frame marker (1 only on query frames)
      [2K+1:2K+5]   previous QUERY action one-hot (zero if none yet)
      [2K+5:2K+5+K] asked identity X one-hot (only on query frames)
    """

    def __init__(self, *, units: int = 4, identity_count: int = 2,
                 reliability: float = 0.65, seed: int = 0) -> None:
        super().__init__()
        self.units = units
        self.reliability = reliability  # p(true channel)
        self.identity_count = identity_count  # K
        self.gap = 0
        self.unit_len = 2  # [clue (if requested), query]
        self.total_steps = units * self.unit_len
        self.input_dim = identity_count + identity_count + 1 + 4 + identity_count
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._seed = seed
        self._rng: random.Random | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = random.Random(seed if seed is not None else self._seed)
        self._rng = rng
        # EvidenceSeek: K identities are fixed for the whole episode.
        self._identities = [rng.randrange(self.identity_count) for _ in range(self.identity_count)]
        self._frame = 0
        self._last_action: int | None = None
        self._asked: list[int] = [self._rng.randrange(self.identity_count) for _ in range(self.units)]
        self._clue_history: list[tuple[int, int]] = []  # (identity, noisy content), recorded when shown
        # Clues exist only after a request; unit 0 always has an empty clue frame.
        self._clues: list[tuple[int, int] | None] = [None] * self.units
        return self._observation(0), {}

    def _noisy_clue(self, identity: int) -> tuple[int, int]:
        content = identity if self._rng.random() < self.reliability else (1 - identity)
        return identity, content

    def _observation(self, frame: int) -> np.ndarray:
        unit, position = divmod(frame, self.unit_len)
        vector = np.zeros(self.input_dim, dtype=np.float32)
        k = self.identity_count
        if position == 0:
            # clue frame: value (noisy) + attribution, ONLY if a clue exists
            clue = self._clues[unit]
            if clue is not None:
                identity, content = clue
                vector[content] = 1.0
                vector[k + identity] = 1.0
                self._clue_history.append((identity, content))
        elif position == self.unit_len - 1:
            # query frame: asked identity selector
            vector[2 * k] = 1.0  # query marker
            vector[2 * k + 1 + 4 + self._asked[unit]] = 1.0
        if self._last_action is not None:
            vector[2 * k + 1 + self._last_action] = 1.0
        return vector

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        frame = self._frame
        unit, position = divmod(frame, self.unit_len)
        reward = 0.0
        terminated = False
        if position == self.unit_len - 1:  # query frame: action is meaningful
            if action in (ACTION_ANSWER_0, ACTION_ANSWER_1):
                asked = self._asked[unit]
                reward = 1.0 if action == self._identities[asked] else -1.0
                terminated = True
            else:  # request a clue for self-chosen identity
                requested = action - ACTION_REQUEST_0
                reward = -0.01
                if unit + 1 < self.units:
                    # a request CREATES the next unit's clue (the only source)
                    self._clues[unit + 1] = self._noisy_clue(requested)
                if unit == self.units - 1:
                    terminated = True  # timeout on the last unit
            self._last_action = action
        self._frame += 1
        if self._frame >= self.total_steps:
            terminated = True
        observation = self._observation(min(self._frame, self.total_steps - 1))
        return observation, reward, terminated, False, {}


class MLPNoMemory(nn.Module):
    """State-less baseline: only the current frame (memoryless, capped at reliability)."""

    def __init__(self, input_dim: int, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.head = nn.Linear(width, num_actions + 1)

    def reset(self, batch: int, device: torch.device) -> None:
        pass

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        features = self.net(observation_step)
        output = self.head(features)
        return output[..., :-1], output[..., -1]

    def forward(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = observations.shape
        logits_list: list[Tensor] = []
        values_list: list[Tensor] = []
        for index in range(length):
            logits, value = self.step(observations[:, index], valid[:, index])
            logits_list.append(logits)
            values_list.append(value)
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


class GRUPolicy(nn.Module):
    """Compressed-hidden-state baseline (GRU)."""

    def __init__(self, input_dim: int, num_actions: int, width: int = 64) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, width)
        self.head = nn.Linear(width + input_dim, num_actions + 1)
        self._state: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._state = torch.zeros(batch, self.cell.hidden_size, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        new_state = self.cell(observation_step, self._state)
        self._state = torch.where(valid_step.unsqueeze(-1), new_state, self._state)
        # head sees the current observation too: the asked-identity channels
        # live on the query frame, which never updates the hidden state
        features = torch.cat((self._state, observation_step), dim=-1)
        output = self.head(features)
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


class ExplicitKeyMemory(nn.Module):
    """Explicit-slot memory with STRUCTURALLY-ALIGNED addressing (diagnosis
    control for the content-addressing finding, 2026-08-11).

    JEPLMMemory's implicit attention alignment (one shared encoder learns
    both the query mapping from the asked-identity channels and the key
    mapping from the clue-attribution channels) did not converge under
    supervision in EvidenceSeek (loss constant 0.693).  This variant removes
    the alignment burden: slot i belongs to identity i, key = projection of
    the attribution one-hot, query = projection of the asked X one-hot,
    value = projection of the clue VALUE one-hot.  If this learns, the
    bottleneck is the implicit alignment, not the task or the attention.
    """

    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64,
                 identity_count: int = 2, value_classes: int = 2) -> None:
        super().__init__()
        self.identity_count = identity_count
        self.value_classes = value_classes
        self.key_proj = nn.Linear(identity_count, latent)
        self.value_proj = nn.Linear(value_classes, latent)
        self.query_proj = nn.Linear(identity_count, latent)
        self.head = nn.Linear(latent, num_actions + 1)
        self._keys: Tensor | None = None
        self._values: Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._keys = self.key_proj(
            torch.eye(self.identity_count, device=device).unsqueeze(0).expand(batch, -1, -1)
        )  # [B, K, latent]
        self._values = torch.zeros(batch, self.identity_count, self.key_proj.out_features, device=device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        batch = observation_step.shape[0]
        k = self.identity_count
        value = observation_step[:, : self.value_classes]
        attribution = observation_step[:, k : 2 * k]
        asked = observation_step[:, 2 * k + 1 + 4 : 2 * k + 1 + 4 + k]
        # write: on clue frames, accumulate the projected value into slot i
        written_value = self.value_proj(value)  # [B, latent]
        write_mask = valid_step.unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        self._values = self._values + write_mask * written_value.unsqueeze(1) * attribution.unsqueeze(-1)
        # read: query from asked X, attention over identity slots
        query_vec = self.query_proj(asked)
        weights = torch.softmax(
            torch.einsum("bd,bmd->bm", query_vec, self._keys) / self.key_proj.out_features**0.5,
            dim=-1,
        )
        recalled = torch.einsum("bm,bmd->bd", weights, self._values)
        output = self.head(recalled)
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


class JEPLMEvidenceSeek(nn.Module):
    """Explicit slot memory: JEPLMMemory stores clue frames (valid only on
    clue frames); policy head reads features and answers/requests.

    Goal-driven retrieval (integration cut 1, wired here for real): the asked
    identity X (trailing K channels of the observation) modulates the content
    addressing so the query frame recalls X's clue, not a mix of both
    identities' clues -- this is the case that made cut 1's interface
    necessary (DecideStream was too easy to show it)."""

    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64,
                 slots: int = 8, write_mode: str = "fifo", goal_dim: int = 2) -> None:
        super().__init__()
        self.goal_dim = goal_dim
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=num_actions, aux_timescales=0,
                                  write_mode=write_mode, goal_dim=goal_dim)
        self.head = nn.Linear(self.memory.head_in, num_actions + 1)

    def reset(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)

    def step(self, observation_step: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor]:
        # asked-identity channels = trailing goal_dim channels (zero on
        # non-query frames, where it must not perturb the read either)
        goal = observation_step[..., -self.goal_dim:]
        features, _aux = self.memory.step(observation_step, valid_step, goal=goal)
        output = self.head(features)
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


# -- collection / training --------------------------------------------------

def _clue_valid(env: EvidenceSeekEnv, frame: int) -> bool:
    return divmod(frame, env.unit_len)[1] == 0


def _collect(model: nn.Module, env: EvidenceSeekEnv, *, count: int, seed_start: int,
             device: torch.device, exploring: bool = False, explore_ratio: float = 0.5) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]], list[list[int]]]:
    """Batch-parallel episode collection.  Each stream resets per episode
    (episodes are independent tasks; no cross-episode memory requirement).

    ``exploring``: early-training forced exploration -- with probability
    ``explore_ratio`` the query-frame action is replaced by a random REQUEST.
    Without this, the high-variance +-1 answer rewards collapse the policy
    onto "always answer at the first query" and the request action is never
    sampled again (exploration collapse, observed 2026-08-11)."""
    import copy
    was_training = model.training
    model.eval()
    envs = [copy.deepcopy(env) for _ in range(count)]
    observations = [envs[i].reset(seed=seed_start + i)[0] for i in range(count)]
    model.reset(count, device)
    histories: list[list[Tensor]] = [[] for _ in range(count)]
    action_lists: list[list[int]] = [[] for _ in range(count)]
    reward_lists: list[list[float]] = [[] for _ in range(count)]
    identities = [envs[i]._identities[:] for i in range(count)]
    done = [False] * count
    for _step in range(env.total_steps):
        active = torch.tensor([not d for d in done], dtype=torch.bool, device=device)
        if not bool(active.any()):
            break
        observation_step = torch.stack([torch.tensor(obs) for obs in observations]).to(device)
        # only clue frames enter memory / update state; the same mask gates
        # per-stream liveness (done streams hold valid=False).
        valid_step = torch.tensor([(not d) and _clue_valid(envs[i], _step) for i, d in enumerate(done)],
                                  dtype=torch.bool, device=device)
        outputs = model.step(observation_step, valid_step)
        logits = outputs[0]
        for index in range(count):
            if done[index]:
                continue
            distribution = Categorical(logits=logits[index])
            action = int(distribution.sample().item())
            if exploring and random.random() < explore_ratio:
                # forced exploration of the request actions (anti collapse)
                action = random.choice((ACTION_REQUEST_0, ACTION_REQUEST_1))
            observation, reward, terminated, truncated, _info = envs[index].step(action)
            histories[index].append(observation_step[index])
            action_lists[index].append(action)
            reward_lists[index].append(reward)
            observations[index] = observation
            if terminated or truncated:
                done[index] = True
    if was_training:
        model.train()
    return histories, action_lists, reward_lists, identities


def _pad(histories: list[list[Tensor]], device: torch.device, input_dim: int) -> tuple[Tensor, Tensor]:
    length = max(len(h) for h in histories)
    batch = len(histories)
    observations = torch.zeros(batch, length, input_dim, device=device)
    valid = torch.zeros(batch, length, dtype=torch.bool, device=device)
    for row, history in enumerate(histories):
        if history:
            observations[row, : len(history)] = torch.stack(history)
            valid[row, : len(history)] = True
    return observations, valid


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


def _train_ppo(model: nn.Module, env: EvidenceSeekEnv, *, updates: int, episodes_per_update: int,
               seed_start: int, device: torch.device, ppo_epochs: int = 4,
               entropy_coefficient: float = 0.1, weight_decay: float = 1e-5,
               explore_updates: int = 30, explore_ratio: float = 0.5) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    for update in range(updates):
        model.train()
        exploring = update < explore_updates
        histories, action_lists, reward_lists, _ = _collect(
            model, env, count=episodes_per_update, seed_start=seed_start + update * episodes_per_update, device=device,
            exploring=exploring, explore_ratio=explore_ratio,
        )
        observations, valid = _pad(histories, device, env.input_dim)
        action_tensor = torch.full(valid.shape, -1, dtype=torch.long, device=device)
        for row, actions in enumerate(action_lists):
            action_tensor[row, : len(actions)] = torch.tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            old_outputs = model(observations, valid)
            old_logits, old_values = old_outputs[0], old_outputs[1]
            old_distribution = Categorical(logits=old_logits)
            # padding frames hold -1 (excluded later via `valid`); clamp keeps
            # log_prob inside the support for the batch call.
            old_selected = old_distribution.log_prob(action_tensor.clamp(min=0))
            advantages, returns = _gae(reward_lists, old_values, valid)
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
        for _ in range(ppo_epochs):
            outputs = model(observations, valid)
            logits, values = outputs[0], outputs[1]
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(action_tensor.clamp(min=0)) - old_selected).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# -- evaluation --------------------------------------------------------------

def _oracle_gap(env: EvidenceSeekEnv) -> list[float]:
    """Oracle posterior per identity from the clue history (evaluation probe
    only; never used in training).  gap = 1 - max posterior."""
    p = env.reliability
    gaps = []
    for identity in range(env.identity_count):
        log_pos = 0.0  # P(identity value = 0) in log space, uniform prior
        log_neg = 0.0
        for shown_identity, content in env._clue_history:
            if shown_identity != identity:
                continue
            # P(content | value): content==0 shown -> consistent with value 0
            l_consistent = math.log(p)
            l_inconsistent = math.log(1.0 - p)
            if content == 0:
                log_pos += l_consistent
                log_neg += l_inconsistent
            else:
                log_pos += l_inconsistent
                log_neg += l_consistent
        # identity value 0 vs 1 (K=2); log odds
        prob = 1.0 / (1.0 + math.exp(log_neg - log_pos))
        confidence = max(prob, 1.0 - prob)
        gaps.append(1.0 - confidence)
    return gaps


@torch.no_grad()
def _evaluate(model: nn.Module, env: EvidenceSeekEnv, *, episodes: int, seed_start: int,
              device: torch.device) -> dict:
    """Returns answer accuracy, answer rate, mean episode reward, mean steps,
    and (jeplem only) request-target vs oracle-max-gap match rate."""
    answered = 0
    correct = 0
    total_reward = 0.0
    total_steps = 0
    gap_matches = 0
    gap_requests = 0
    request_gaps: list[float] = []  # gap of the REQUESTED identity when requested
    max_gap_requests: list[float] = []  # oracle max gap at that moment
    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed_start + episode_index)
        model.reset(1, device)
        episode_reward = 0.0
        for frame in range(env.total_steps):
            observation_step = torch.tensor(observation).unsqueeze(0).to(device)
            valid_step = torch.ones(1, dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            distribution = Categorical(logits=outputs[0][0])
            action = int(distribution.sample().item())
            observation, reward, terminated, truncated, _info = env.step(action)
            episode_reward += reward
            unit, position = divmod(frame, env.unit_len)
            if position == env.unit_len - 1:
                if action in (ACTION_ANSWER_0, ACTION_ANSWER_1):
                    answered += 1
                    correct += int(reward > 0)
                else:
                    gaps = _oracle_gap(env)
                    requested = action - ACTION_REQUEST_0
                    gap_matches += int(requested == max(range(env.identity_count), key=gaps.__getitem__))
                    gap_requests += 1
                    request_gaps.append(gaps[requested])
                    max_gap_requests.append(max(gaps))
            if terminated or truncated:
                break
        total_reward += episode_reward
        total_steps += frame + 1
    report = {
        "answer_accuracy": correct / answered if answered else 0.0,
        "answer_rate": answered / episodes,
        "episode_reward": total_reward / episodes,
        "mean_steps": total_steps / episodes,
        "gap_match_rate": gap_matches / gap_requests if gap_requests else 0.0,
        "request_gap_mean": float(np.mean(request_gaps)) if request_gaps else 0.0,
        "max_gap_mean": float(np.mean(max_gap_requests)) if max_gap_requests else 0.0,
    }
    return report


# -- run / main --------------------------------------------------------------

@dataclass
class Report:
    protocol: str
    device: str
    units: int
    gap: int
    identity_count: int
    reliability: float
    updates: int
    episodes_per_update: int
    results: list[dict]
    accepted: bool
    artifact: str


def resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run(*, device: str | torch.device | None = None, units: int = 6, gap: int = 2,
        identity_count: int = 2, reliability: float = 0.65, updates: int = 100,
        episodes_per_update: int = 24, evaluation_episodes: int = 64,
        experiment_seed: int = 41_123, train_seed_start: int = 10_000,
        heldout_seed_start: int = 1_000_000, slots: int = 8,
        jeplem_only: bool = False, artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    env = EvidenceSeekEnv(units=units, identity_count=identity_count,
                          reliability=reliability)
    input_dim = env.input_dim
    factories = (
        ("mlp_no_memory", lambda: MLPNoMemory(input_dim, N_ACTIONS)),
        ("gru_policy", lambda: GRUPolicy(input_dim, N_ACTIONS)),
        ("jeplem_policy", lambda: JEPLMEvidenceSeek(input_dim, N_ACTIONS, slots=slots, goal_dim=identity_count)),
        ("explicit_key_policy", lambda: ExplicitKeyMemory(input_dim, N_ACTIONS, identity_count=identity_count)),
    )
    if jeplem_only:
        factories = [f for f in factories if f[0] == "jeplem_policy"]
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train_ppo(model, env, updates=updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start, device=resolved)
        metrics = _evaluate(model, env, episodes=evaluation_episodes,
                            seed_start=heldout_seed_start, device=resolved)
        results.append({"name": name, "parameters": sum(p.numel() for p in model.parameters()), **metrics})
    by_name = {r["name"]: r for r in results}
    accepted = (
        "jeplem_policy" in by_name
        and by_name["jeplem_policy"]["answer_accuracy"] >= 0.8
        and (by_name["jeplem_policy"]["answer_accuracy"]
             >= by_name["mlp_no_memory"]["answer_accuracy"] + 0.15)
    )
    target = artifact or ROOT / "artifacts" / "evidence-seek-rl-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "evidence_seek_v1: composite task for the integration layer -- K identities, noisy clue frames (p=reliability), query frames ask 'what is identity X' (X uniform in observation); agent answers X's value (+1 correct, ends) or requests a clue for a self-chosen identity (-0.01, next unit shows it); timeout at units units. PPO on-policy, no teacher/oracle. Controls: mlp_no_memory (memoryless, capped at reliability), gru_policy (compressed state), jeplem_policy (explicit slots). Probes: answer accuracy/rate, episode reward, request-vs-oracle-max-gap match.",
        str(resolved), units, gap, identity_count, reliability, updates, episodes_per_update,
        results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--units", type=int, default=6)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--identity-count", type=int, default=2)
    parser.add_argument("--reliability", type=float, default=0.65)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--jeplem-slots", type=int, default=8)
    parser.add_argument("--jeplem-only", action="store_true")
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, reliability=args.reliability,
        updates=args.updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, slots=args.jeplem_slots,
        jeplem_only=args.jeplem_only, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
