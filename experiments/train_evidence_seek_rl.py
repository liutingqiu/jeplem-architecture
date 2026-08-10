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
import torch.nn.functional as F
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
                 reliability: float = 0.65, seed: int = 0, max_requests: int = 3) -> None:
        super().__init__()
        self.units = units
        self.reliability = reliability  # p(true channel)
        self.identity_count = identity_count  # K
        self.gap = 0
        self.max_requests = max_requests  # budget: opportunity cost, not penalty
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
        self._requests_left = self.max_requests
        self._asked: list[int] = [self._rng.randrange(self.identity_count) for _ in range(self.units)]
        self._clue_history: list[tuple[int, int]] = []  # (identity, noisy content), recorded when shown
        # Clues exist only after a request; unit 0 always has an empty clue frame.
        self._clues: list[tuple[int, int] | None] = [None] * self.units
        return self._observation(0), {}

    def _noisy_clue(self, identity: int) -> tuple[int, int]:
        # BUG FIX (2026-08-11, found during explicit-policy debugging): the
        # clue VALUE must be the identity's VALUE (identities[identity]) with
        # noise, NOT the identity number itself -- with identity_count=2 the
        # number and the value are the same domain but independent, so the old
        # code made clue content uncorrelated with the asked value (all
        # downstream learning was capped at 0.5 for the value prediction).
        value = self._identities[identity]
        content = value if self._rng.random() < self.reliability else (1 - value)
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
                if self._requests_left <= 0:
                    terminated = True  # budget exhausted: must concede (0)
                else:
                    self._requests_left -= 1
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
    necessary (DecideStream was too easy to show it).

    ``actions=3`` (3-class policy): actions are {answer 0, answer 1, request};
    the request TARGET is fully outside the policy gradient (explicit wiring
    or random) -- clean test of mechanism wiring vs end-to-end (9.4)."""

    def __init__(self, input_dim: int, num_actions: int, *, latent: int = 64,
                 slots: int = 8, write_mode: str = "fifo", goal_dim: int = 2,
                 actions: int = 4) -> None:
        super().__init__()
        self.goal_dim = goal_dim
        self.num_policy_actions = actions  # 4 = {ans0,ans1,req0,req1}, 3 = {ans0,ans1,req}
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=num_actions, aux_timescales=0,
                                  write_mode=write_mode, goal_dim=goal_dim)
        self.head = nn.Linear(self.memory.head_in, actions + 1)

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


def _gap_argmax_request(env: EvidenceSeekEnv) -> int:
    """Explicit gap-driven request target (integration-layer mechanism wiring,
    2026-08-11): Bayesian posterior per identity from the agent's OWN observed
    clue history (no oracle -- attribution+value come from observations),
    request the identity with the largest confidence gap.  Replaces the
    end-to-end-learned request target; the policy only learns WHEN to answer."""
    p = env.reliability
    best_i, best_gap = 0, -1.0
    for i in range(env.identity_count):
        vals = [c for (ident, c) in env._clue_history if ident == i]
        if not vals:
            gap = 1.0
        else:
            n = len(vals)
            c0 = vals.count(0)
            log_odd = (2 * c0 - n) * (math.log(p) - math.log(1.0 - p))
            prob0 = 1.0 / (1.0 + math.exp(-log_odd))
            gap = 1.0 - max(prob0, 1.0 - prob0)
        if gap > best_gap:
            best_gap, best_i = gap, i
    return ACTION_REQUEST_0 + best_i


def _map_policy_action(model: nn.Module, action: int, env: EvidenceSeekEnv,
                       explicit_gap: bool) -> int:
    """Map a policy action (4-class {ans0,ans1,req0,req1} or 3-class
    {ans0,ans1,req}) to an env action.  3-class policies leave the request
    TARGET outside the policy gradient: explicit max-gap wiring or random."""
    if getattr(model, "num_policy_actions", 4) == 4:
        if explicit_gap and action in (ACTION_REQUEST_0, ACTION_REQUEST_1):
            return _gap_argmax_request(env)
        return action
    # 3-class policy
    if action == 2:  # request
        if explicit_gap:
            return _gap_argmax_request(env)
        return random.choice((ACTION_REQUEST_0, ACTION_REQUEST_1))
    return action  # answer 0 / answer 1 map directly


def _bayes_target(clue_values: list[int], reliability: float) -> int | None:
    """Bayesian posterior argmax of an identity's value given its noisy clue
    values (p=true channel).  None when there is no clue (un-answerable)."""
    if not clue_values:
        return None
    n = len(clue_values)
    c0 = clue_values.count(0)
    log_odd = (2 * c0 - n) * (math.log(reliability) - math.log(1.0 - reliability))
    prob0 = 1.0 / (1.0 + math.exp(-log_odd))
    return 0 if prob0 >= 0.5 else 1


@torch.no_grad()
def _bayes_batch(env: EvidenceSeekEnv, n: int, seed0: int) -> tuple[Tensor, Tensor, list[list[tuple[int, int, int]]]]:
    """Collect n episodes under a fixed 'request both then answer' policy and
    return (observations, valid, labels) where labels[b] = [(X, frame_idx,
    bayes_target), ...] -- the SUPERVISABLE targets (never ground truth)."""
    obs_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    lab_list: list[list[tuple[int, int, int]]] = []
    tmax = 0
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        frames: list[np.ndarray] = []
        valid: list[bool] = []
        labels: list[tuple[int, int, int]] = []
        clues: dict[int, list[int]] = {}
        acts = [ACTION_REQUEST_0, ACTION_REQUEST_1, ACTION_REQUEST_0, ACTION_ANSWER_0]
        for f in range(env.total_steps):
            unit, pos = divmod(f, env.unit_len)
            frames.append(obs.copy())
            valid.append(_clue_valid(env, f))
            if pos == 0 and env._clues[unit] is not None:
                ident, content = env._clues[unit]
                clues.setdefault(ident, []).append(content)
            a = acts.pop(0) if pos == env.unit_len - 1 else 0
            if pos == env.unit_len - 1:
                x = env._asked[unit]
                tgt = _bayes_target(clues.get(x, []), env.reliability)
                if tgt is not None:
                    labels.append((x, len(frames) - 1, tgt))
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        tmax = max(tmax, len(frames))
        obs_list.append(np.stack(frames))
        val_list.append(np.array(valid, dtype=bool))
        lab_list.append(labels)
    obs_t = torch.zeros(n, tmax, env.input_dim)
    valid_t = torch.zeros(n, tmax, dtype=torch.bool)
    for b, o in enumerate(obs_list):
        obs_t[b, : len(o)] = torch.tensor(o)
        valid_t[b, : len(o)] = torch.tensor(val_list[b])
    return obs_t, valid_t, lab_list


def pretrain_memory_stage1(model: JEPLMEvidenceSeek, env: EvidenceSeekEnv, *, steps: int = 600,
                    batch: int = 32, seed0: int = 5000, lr: float = 1e-3,
                    device: torch.device) -> None:
    """Stage 1 of hierarchical training: supervise the ANSWER path (memory +
    head's answer logits) with BAYESIAN-POSTERIOR targets (never ground truth
    -- noisy clues make ground truth un-supervisable, integration design 9.1).
    Freezes the memory afterwards; the head starts stage 2 from this answer
    path, so RL only has to learn the request/answer gating."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        obs_t, valid_t, lab_list = _bayes_batch(env, batch, seed0 + step * batch)
        obs_t, valid_t = obs_t.to(device), valid_t.to(device)
        model.reset(batch, device)
        loss = torch.tensor(0.0, device=device)
        cnt = 0
        for f in range(valid_t.shape[1]):
            logits, _ = model.step(obs_t[:, f], valid_t[:, f])
            for b in range(batch):
                for x, fidx, tgt in lab_list[b]:
                    if fidx == f:
                        loss = loss + F.cross_entropy(logits[b, :2].unsqueeze(0), torch.tensor([tgt], device=device))
                        cnt += 1
        if cnt == 0:
            continue
        opt.zero_grad(set_to_none=True)
        (loss / cnt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    for param in model.memory.parameters():
        param.requires_grad_(False)
    model.memory.eval()


def _collect(model: nn.Module, env: EvidenceSeekEnv, *, count: int, seed_start: int,
             device: torch.device, exploring: bool = False, explore_ratio: float = 0.5,
             explicit_gap: bool = False) -> tuple[list[list[Tensor]], list[list[int]], list[list[float]], list[list[int]]]:
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
            policy_action = int(distribution.sample().item())
            if exploring and random.random() < explore_ratio:
                # forced exploration of the request actions (anti collapse);
                # the recorded target-policy action is 'request' for 3-class
                # policies (request target is outside the gradient anyway)
                action = random.choice((ACTION_REQUEST_0, ACTION_REQUEST_1))
                if getattr(model, "num_policy_actions", 4) == 3:
                    policy_action = 2
            else:
                action = _map_policy_action(model, policy_action, envs[index], explicit_gap)
            observation, reward, terminated, truncated, _info = envs[index].step(action)
            histories[index].append(observation_step[index])
            action_lists[index].append(policy_action)
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
               explore_updates: int = 30, explore_ratio: float = 0.5,
               explicit_gap: bool = False) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=weight_decay)
    for update in range(updates):
        model.train()
        exploring = update < explore_updates
        histories, action_lists, reward_lists, _ = _collect(
            model, env, count=episodes_per_update, seed_start=seed_start + update * episodes_per_update, device=device,
            exploring=exploring, explore_ratio=explore_ratio, explicit_gap=explicit_gap,
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
              device: torch.device, explicit_gap: bool = False) -> dict:
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
            # only clue frames may write memory (same mask as training)
            valid_step = torch.tensor([_clue_valid(env, frame)], dtype=torch.bool, device=device)
            outputs = model.step(observation_step, valid_step)
            distribution = Categorical(logits=outputs[0][0])
            action = int(distribution.sample().item())
            observation, reward, terminated, truncated, _info = env.step(action)
            episode_reward += reward
            unit, position = divmod(frame, env.unit_len)
            if position == env.unit_len - 1:
                action = _map_policy_action(model, action, env, explicit_gap)
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
        jeplem_only: bool = False, pretrain_memory: bool = False,
        pretrain_steps: int = 600, explicit_gap: bool = False,
        policy_actions: int = 4, artifact: Path | None = None) -> Report:
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
        ("jeplem_policy", lambda: JEPLMEvidenceSeek(input_dim, N_ACTIONS, slots=slots, goal_dim=identity_count, actions=policy_actions)),
        ("explicit_key_policy", lambda: ExplicitKeyMemory(input_dim, N_ACTIONS, identity_count=identity_count)),
    )
    if jeplem_only:
        factories = [f for f in factories if f[0] == "jeplem_policy"]
    # hierarchical training (integration follow-up): stage 1 = Bayesian-
    # supervised memory pretraining (aux head), frozen; stage 2 = RL on the
    # policy head only.
    pretrained = False
    _mem_state: dict | None = None
    _head_state: dict | None = None
    if pretrain_memory:
        probe = JEPLMEvidenceSeek(input_dim, N_ACTIONS, slots=slots, goal_dim=identity_count).to(resolved)
        pretrain_memory_stage1(probe, env, steps=pretrain_steps, device=resolved)
        _mem_state = {k: v.detach().clone() for k, v in probe.memory.state_dict().items()}
        _head_state = {k: v.detach().clone() for k, v in probe.head.state_dict().items()}
        pretrained = True
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        if pretrained and name == "jeplem_policy":
            model.memory.load_state_dict(_mem_state, strict=False)
            model.head.load_state_dict(_head_state)
            for param in model.memory.parameters():
                param.requires_grad_(False)
            model.memory.eval()
        _train_ppo(model, env, updates=updates, episodes_per_update=episodes_per_update,
                   seed_start=train_seed_start, device=resolved, explicit_gap=explicit_gap)
        metrics = _evaluate(model, env, episodes=evaluation_episodes,
                            seed_start=heldout_seed_start, device=resolved, explicit_gap=explicit_gap)
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
    parser.add_argument("--pretrain-memory", action="store_true", help="hierarchical: Bayesian-supervised memory pretrain (aux, frozen) then RL policy head")
    parser.add_argument("--pretrain-steps", type=int, default=600)
    parser.add_argument("--explicit-gap", action="store_true", help="integration wiring: request target = explicit max-gap identity (Bayesian from observations); policy learns only when to answer")
    parser.add_argument("--policy-actions", type=int, default=4, choices=[3, 4], help="3 = policy outputs {answer 0, answer 1, request} (request target outside gradient)")
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, units=args.units, gap=args.gap,
        identity_count=args.identity_count, reliability=args.reliability,
        updates=args.updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes,
        experiment_seed=args.experiment_seed, slots=args.jeplem_slots,
        jeplem_only=args.jeplem_only, pretrain_memory=args.pretrain_memory,
        pretrain_steps=args.pretrain_steps, explicit_gap=args.explicit_gap,
        policy_actions=args.policy_actions, artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
