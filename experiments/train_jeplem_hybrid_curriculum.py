"""JEPLM hybrid curriculum: SPR-style multi-step latent prediction auxiliary.

JEPLM keeps the validated hybrid controller (tactical causal Transformer plus
persistent PLWM side state) and adds two JEPA-inspired objectives on the
persistent latent:

1. ``jeplem_loss``: action-conditioned multi-step latent prediction.  From the
   persistent belief at time t and the intervening action one-hot sequence,
   predict the belief the real rollout reaches at t+k.  The target is the real
   future belief, detached.  No EMA teacher, no target network (JEPA's fragile
   EMA heuristic is deliberately rejected); supervision comes from the actual
   episode, so the objective stays causal at action time and online.

2. ``anti_collapse_loss``: a variance hinge on the persistent latent across the
   batch (SIGReg-lite), so latent states cannot collapse to a single point.

Both objectives are online auxiliary losses inside the same reward-only PPO
loop (One Operating System); there is no separate offline pretraining phase.

Reward-only curriculum, sampled actions, external environment reward, fixed
separated train/held-out seeds, no teacher action, path oracle, future
observation at action time, seed answer, reward shaping, or replayed teacher
trajectory.
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
import torch.nn.functional as F
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.device import resolve_device

try:
    from .train_minigrid_memory_actor_critic import (
        ACTION_COUNT, MAX_STEPS, HybridPLWMTransformerActorCritic, NoMemoryActorCritic,
        TransformerActorCritic, _collect, _gae, _mask_actions, _pad,
        encode_observation, evaluate, evaluate_random, make_environment,
    )
except ImportError:
    from train_minigrid_memory_actor_critic import (
        ACTION_COUNT, MAX_STEPS, HybridPLWMTransformerActorCritic, NoMemoryActorCritic,
        TransformerActorCritic, _collect, _gae, _mask_actions, _pad,
        encode_observation, evaluate, evaluate_random, make_environment,
    )


BOOTSTRAP_ENVIRONMENT = "MiniGrid-Empty-5x5-v0"
MEMORY_ENVIRONMENT = "MiniGrid-MemoryS7-v0"


class HybridJEPLMActorCritic(HybridPLWMTransformerActorCritic):
    """Hybrid controller with SPR-style multi-step latent prediction on the
    persistent PLWM side."""

    def __init__(
        self,
        width: int = 64,
        *,
        latent: int = 64,
        memory_slots: int = 8,
        thought_steps: int = 2,
        use_plwm_side: bool = True,
        max_steps: int = 64,
        predict_horizon: int = 2,
        latent_pred_width: int = 64,
        predict_observation: bool = True,
    ) -> None:
        super().__init__(
            width, latent=latent, memory_slots=memory_slots,
            thought_steps=thought_steps, use_plwm_side=use_plwm_side,
            max_steps=max_steps,
        )
        if predict_horizon < 1:
            raise ValueError("predict_horizon must be at least one")
        self.predict_horizon = predict_horizon
        self.predict_observation = predict_observation
        self.latent_predictor = nn.Sequential(
            nn.Linear(latent + predict_horizon * ACTION_COUNT, latent_pred_width),
            nn.Tanh(),
            nn.Linear(latent_pred_width, latent),
        )

    def _rollout_states(self, observations: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Persistent belief sequence plus the per-event sensory encodings."""
        state = self.core.initial_state(observations.shape[0], observations.device)
        beliefs: list[Tensor] = []
        encodings: list[Tensor] = []
        for index in range(observations.shape[1]):
            encodings.append(self.core.encoder(observations[:, index]))
            state = self._advance(observations[:, index], state, valid[:, index])
            beliefs.append(state.belief)
        return torch.stack(beliefs, dim=1), torch.stack(encodings, dim=1)

    def _rollout_beliefs(self, observations: Tensor, valid: Tensor) -> Tensor:
        """Return the persistent belief sequence of the real episode rollout."""
        return self._rollout_states(observations, valid)[0]

    def jeplem_loss(self, observations: Tensor, actions: Tensor, valid: Tensor) -> Tensor:
        """Action-conditioned multi-step prediction in persistent latent space.

        target_mode "observation_encoding" (SPR-style): the predictor learns to
        forecast the sensory encoding of a future observation from the current
        persistent belief and the intervening actions.  The target comes from
        the real observation stream, never from the model's own state, so the
        objective cannot settle into a self-bootstrapping shortcut.  target_mode
        "belief" keeps the original self-prediction variant for ablation.
        """
        beliefs, encodings = self._rollout_states(observations, valid)  # [B, T, d]
        if self.predict_observation:
            targets = encodings  # sensory encoding of future observations
        else:
            targets = beliefs  # self-bootstrapped persistent state
        horizon_total = beliefs.shape[1]
        onehot = F.one_hot(actions.clamp(min=0).long(), ACTION_COUNT).float()  # [B, T, 7]
        losses = []
        for k in range(1, self.predict_horizon + 1):
            horizon = horizon_total - k
            if horizon <= 0:
                continue
            src = beliefs[:, :horizon]  # [B, H, d]
            # actions[t:t+k] per target t: offset j reads onehot[:, j:j+H];
            # pad the block to predict_horizon*7 so the predictor input width
            # is fixed for every k.
            act_parts = []
            for j in range(self.predict_horizon):
                if j < k:
                    act_parts.append(onehot[:, j : j + horizon])
                else:
                    act_parts.append(torch.zeros_like(onehot[:, :horizon]))
            action_block = torch.cat(act_parts, dim=-1)  # [B, H, horizon*7]
            prediction = self.latent_predictor(torch.cat((src, action_block), dim=-1))
            target = targets[:, k:].detach()  # [B, H, d]
            active = valid[:, k:].unsqueeze(-1).float()
            masked = ((prediction - target) ** 2).mean(dim=-1, keepdim=True) * active
            losses.append(masked.sum() / active.sum().clamp_min(1.0))
        if not losses:
            return torch.zeros((), device=beliefs.device)
        return sum(losses) / len(losses)

    def anti_collapse_loss(self, beliefs: Tensor, valid: Tensor) -> Tensor:
        """Variance hinge on the persistent latent across the batch."""
        active = valid.unsqueeze(-1).float()
        count = active.sum(dim=0).clamp_min(1.0)
        mean = (beliefs * active).sum(dim=0) / count  # [T, d]
        variance = ((beliefs - mean.unsqueeze(0)) ** 2 * active).sum(dim=0) / count
        # sqrt at zero has an infinite gradient; epsilon keeps padding steps
        # (zero variance) from poisoning the backward pass.
        return torch.relu(1.0 - (variance + 1e-8).sqrt().mean())


@dataclass
class Stage:
    name: str
    environment: str
    action_mode: str
    updates: int
    episodes_per_update: int
    seed_start: int
    train_episodes: int
    train_successes: int


@dataclass
class ControllerResult:
    name: str
    parameters: int
    active_parameters: int
    jeplem_coefficient: float
    anti_collapse_coefficient: float
    predict_horizon: int
    stages: list[Stage]
    evaluation: object
    stochastic_evaluation: object
    memory_probe: object  # {original, swapped, both}


@dataclass
class MemoryProbe:
    original: int
    swapped: int
    both: int


@dataclass
class Report:
    protocol: str
    device: str
    experiment_seed: int
    bootstrap_updates: int
    memory_updates: int
    episodes_per_update: int
    heldout_seed_range: list[int]
    random_evaluation: object
    results: list[ControllerResult]
    jeplem_increment_successes: int
    jeplem_memory_increment: int
    accepted: bool
    artifact: str


def _jeplem_train(
    model: HybridJEPLMActorCritic,
    *,
    updates: int,
    episodes_per_update: int,
    start_seed: int,
    device: torch.device,
    ppo_epochs: int = 4,
    entropy_coefficient: float = 0.02,
    action_mode: str = "all",
    environment: str = MEMORY_ENVIRONMENT,
    jeplem_coefficient: float = 0.1,
    anti_collapse_coefficient: float = 0.01,
) -> tuple[int, int]:
    """Reward-only PPO with online JEPLM auxiliary objectives."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    total_episodes = total_successes = 0
    for update in range(updates):
        histories, actions, _external_rewards, learning_rewards, successes = _collect(
            model, count=episodes_per_update, start_seed=start_seed + update * episodes_per_update,
            device=device, intrinsic_bonus=0.0, action_mode=action_mode,
            environment=environment, observation_mode="observation_only",
        )
        observations, valid = _pad(histories, device)
        action_tensor = torch.zeros(valid.shape, dtype=torch.long, device=device)
        for index, episode in enumerate(actions):
            action_tensor[index, : len(episode)] = torch.tensor(episode, dtype=torch.long, device=device)
        with torch.no_grad():
            old_logits, old_values = model(observations, valid)
            old_distribution = torch.distributions.Categorical(logits=_mask_actions(old_logits, action_mode))
            old_selected = old_distribution.log_prob(action_tensor)
            advantages, returns = _gae(learning_rewards, old_values, valid)
            normalized = advantages[valid]
            advantages = (advantages - normalized.mean()) / normalized.std().clamp_min(1e-5)
        for _ in range(ppo_epochs):
            logits, values = model(observations, valid)
            distribution = torch.distributions.Categorical(logits=_mask_actions(logits, action_mode))
            ratio = (distribution.log_prob(action_tensor) - old_selected).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(0.8, 1.2) * advantages
            actor = -torch.minimum(unclipped[valid], clipped[valid]).mean()
            critic = (returns[valid] - values[valid]).square().mean()
            entropy = distribution.entropy()[valid].mean()
            loss = actor + 0.5 * critic - entropy_coefficient * entropy
            if jeplem_coefficient > 0.0:
                loss = loss + jeplem_coefficient * model.jeplem_loss(observations, action_tensor, valid)
            if anti_collapse_coefficient > 0.0:
                beliefs = model._rollout_beliefs(observations, valid)
                loss = loss + anti_collapse_coefficient * model.anti_collapse_loss(beliefs, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if hasattr(model.core, "enforce_key_orthogonality"):
                model.core.enforce_key_orthogonality()
        total_episodes += episodes_per_update
        total_successes += successes
    return total_episodes, total_successes


def _swap_matching_objects(env: gym.Env) -> None:
    """Counterfactual intervention: exchange the two hallway-end objects
    (Key/Ball) and swap the success/failure determination.

    A policy that genuinely uses the remembered object identity follows the
    matching object and still succeeds after the swap.  A fixed-route policy
    (e.g. turn-right-then-forward, which the current held-out evaluation lets
    score ~27/64) hits at most one of the two versions, so ``both`` separates
    memory-driven success from route-hitting success.
    """
    u = env.unwrapped
    key_pos = ball_pos = None
    for y in range(u.height):
        for x in range(u.width):
            obj = u.grid.get(x, y)
            if obj is None or x < 2:  # x=1 holds the start-room object
                continue
            if obj.type == "key" and key_pos is None:
                key_pos = (x, y)
            elif obj.type == "ball" and ball_pos is None:
                ball_pos = (x, y)
    if key_pos is None or ball_pos is None:
        raise RuntimeError("failed to locate the two hallway-end objects")
    key_obj, ball_obj = u.grid.get(*key_pos), u.grid.get(*ball_pos)
    u.grid.set(*key_pos, ball_obj)
    u.grid.set(*ball_pos, key_obj)
    u.success_pos, u.failure_pos = list(u.failure_pos), list(u.success_pos)


def _episode_success(
    model: Any,
    instance: gym.Env,
    observation: dict[str, Any],
    *,
    device: torch.device,
    action_mode: str = "all",
    environment: str = MEMORY_ENVIRONMENT,
) -> bool:
    history = [encode_observation(observation)]
    total = 0.0
    for step in range(MAX_STEPS):
        observations, valid = _pad([history], device)
        logits, _value = model(observations, valid)
        current_logits = _mask_actions(logits[0, len(history) - 1], action_mode)
        action = int(current_logits.argmax().item())
        observation, reward, terminated, truncated, _info = instance.step(action)
        total += float(reward)
        if terminated or truncated:
            break
        history.append(encode_observation(observation))
    instance.close()
    return total > 0.0


def evaluate_memory_probe(
    model: Any,
    *,
    episodes: int,
    start_seed: int,
    device: torch.device,
    action_mode: str = "all",
    environment: str = MEMORY_ENVIRONMENT,
) -> MemoryProbe:
    """Paired counterfactual memory probe on held-out worlds.

    Each world is evaluated twice: once as generated, once after swapping the
    hallway-end objects and the success/failure determination.  ``both`` counts
    worlds where the policy succeeds in both versions, which requires tracking
    the object identity rather than following a fixed route.
    """
    model.eval()
    original = swapped = both = 0
    for episode_index in range(episodes):
        instance = make_environment(environment)
        observation, _info = instance.reset(seed=start_seed + episode_index)
        succeeded = _episode_success(
            model, instance, observation, device=device,
            action_mode=action_mode, environment=environment,
        )
        original += int(succeeded)
        # Rebuild the same world and apply the counterfactual intervention.
        swapped_instance = make_environment(environment)
        observation, _info = swapped_instance.reset(seed=start_seed + episode_index)
        _swap_matching_objects(swapped_instance)
        succeeded_swapped = _episode_success(
            model, swapped_instance, observation, device=device,
            action_mode=action_mode, environment=environment,
        )
        swapped += int(succeeded_swapped)
        both += int(succeeded and succeeded_swapped)
    return MemoryProbe(original, swapped, both)


def run(
    *,
    device: str | torch.device | None = None,
    bootstrap_updates: int = 30,
    memory_updates: int = 30,
    episodes_per_update: int = 24,
    evaluation_episodes: int = 64,
    experiment_seed: int = 41_123,
    train_seed_start: int = 10_000,
    heldout_seed_start: int = 1_000_000,
    jeplem_coefficient: float = 0.1,
    anti_collapse_coefficient: float = 0.01,
    predict_horizon: int = 2,
    predict_observation: bool = True,
    artifact: Path | None = None,
) -> Report:
    if (bootstrap_updates < 1 or memory_updates < 1 or episodes_per_update < 2
            or evaluation_episodes < 1 or jeplem_coefficient < 0.0
            or anti_collapse_coefficient < 0.0):
        raise ValueError("invalid run configuration")
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    def train_one(model: HybridJEPLMActorCritic | TransformerActorCritic | NoMemoryActorCritic,
                  *, jeplem: float, anticollapse: float) -> list[Stage]:
        seed_start = train_seed_start
        stages: list[Stage] = []
        for stage_name, environment, action_mode, updates in (
            ("reward_only_navigation_bootstrap", BOOTSTRAP_ENVIRONMENT, "movement_only", bootstrap_updates),
            ("delayed_memory_transfer", MEMORY_ENVIRONMENT, "all", memory_updates),
        ):
            if isinstance(model, HybridJEPLMActorCritic) and jeplem > 0.0:
                train_episodes, train_successes = _jeplem_train(
                    model, updates=updates, episodes_per_update=episodes_per_update,
                    start_seed=seed_start, device=resolved, ppo_epochs=4,
                    entropy_coefficient=0.02, action_mode=action_mode,
                    environment=environment, jeplem_coefficient=jeplem,
                    anti_collapse_coefficient=anticollapse,
                )
            else:
                from train_minigrid_memory_actor_critic import _train
                train_episodes, train_successes = _train(
                    model, updates=updates, episodes_per_update=episodes_per_update,
                    start_seed=seed_start, device=resolved,
                    learning_rule="ppo_external_reward_v1", ppo_epochs=4,
                    entropy_coefficient=0.02, action_mode=action_mode,
                    environment=environment, observation_mode="observation_only",
                )
            stages.append(Stage(
                stage_name, environment, action_mode, updates, episodes_per_update,
                seed_start, train_episodes, train_successes,
            ))
            seed_start += updates * episodes_per_update
        return stages

    results: list[ControllerResult] = []
    factories = (
        ("hybrid_jeplem_v2", HybridJEPLMActorCritic(predict_horizon=predict_horizon, predict_observation=predict_observation), jeplem_coefficient, anti_collapse_coefficient),
        ("hybrid_baseline", HybridPLWMTransformerActorCritic(), 0.0, 0.0),
        ("hybrid_no_side", HybridPLWMTransformerActorCritic(use_plwm_side=False), 0.0, 0.0),
    )
    for name, model, jeplem, anticollapse in factories:
        model = model.to(resolved)
        stages = train_one(model, jeplem=jeplem, anticollapse=anticollapse)
        evaluation = evaluate(
            model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
            device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        stochastic = evaluate(
            model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
            device=resolved, stochastic=True, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        memory_probe = evaluate_memory_probe(
            model, episodes=evaluation_episodes, start_seed=heldout_seed_start,
            device=resolved, action_mode="all", environment=MEMORY_ENVIRONMENT,
        )
        results.append(ControllerResult(
            name,
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            jeplem, anticollapse, predict_horizon, stages, evaluation, stochastic,
            memory_probe,
        ))
    by_name = {result.name: result for result in results}
    jeplem = by_name["hybrid_jeplem_v2"].evaluation
    baseline = by_name["hybrid_baseline"].evaluation
    increment = jeplem.successes - baseline.successes
    # The architecture-relevant gate is the counterfactual memory probe: how
    # many worlds the controller solves under both the original and the
    # swapped layout (i.e. genuine object-identity memory, not route-hitting).
    jeplem_memory = by_name["hybrid_jeplem_v2"].memory_probe
    baseline_memory = by_name["hybrid_baseline"].memory_probe
    memory_increment = jeplem_memory.both - baseline_memory.both
    accepted = (
        jeplem_memory.both >= int(evaluation_episodes * 0.40)
        and jeplem_memory.both >= baseline_memory.both + int(evaluation_episodes * 0.05)
        and increment >= 0
    )
    target = artifact or ROOT / "artifacts" / "jeplem-hybrid-curriculum-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "jeplem_hybrid_curriculum_v3: two-stage reward-only curriculum; tactical causal Transformer plus persistent PLWM side state; JEPLM v2 switches the auxiliary target from self-bootstrapped belief prediction to SPR-style prediction of the sensory encoding of future observations (real observation stream, detached, no EMA teacher); SIGReg-lite variance hinge; evaluation includes a paired counterfactual memory probe that swaps the hallway-end objects and the success/failure determination on every held-out world, so route-hitting policies (turn-right-then-forward scores ~27/64 on the plain held-out) are separated from genuine object-identity memory (both versions solved); no separate offline pretraining phase, no teacher action, path oracle, future observation at action time, seed answer, reward shaping, or replayed trajectory; fixed separated train/held-out seeds.",
        str(resolved), experiment_seed, bootstrap_updates, memory_updates, episodes_per_update,
        list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)),
        evaluate_random(episodes=evaluation_episodes, start_seed=heldout_seed_start, action_mode="all", environment=MEMORY_ENVIRONMENT),
        results, increment, memory_increment, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-updates", type=int, default=30)
    parser.add_argument("--memory-updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--train-seed-start", type=int, default=10_000)
    parser.add_argument("--heldout-seed-start", type=int, default=1_000_000)
    parser.add_argument("--jeplem-coefficient", type=float, default=0.1)
    parser.add_argument("--anti-collapse-coefficient", type=float, default=0.01)
    parser.add_argument("--predict-horizon", type=int, default=2)
    parser.add_argument("--predict-observation", type=lambda v: v.lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(
        device=args.device, bootstrap_updates=args.bootstrap_updates,
        memory_updates=args.memory_updates, episodes_per_update=args.episodes_per_update,
        evaluation_episodes=args.evaluation_episodes, experiment_seed=args.experiment_seed,
        train_seed_start=args.train_seed_start, heldout_seed_start=args.heldout_seed_start,
        jeplem_coefficient=args.jeplem_coefficient,
        anti_collapse_coefficient=args.anti_collapse_coefficient,
        predict_horizon=args.predict_horizon, predict_observation=args.predict_observation,
        artifact=args.artifact,
    )
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("jeplem hybrid curriculum did not meet the held-out increment gate")
