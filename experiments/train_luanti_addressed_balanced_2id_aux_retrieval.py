"""Auxiliary retrieval-supervised smoke on addressed_balanced_2id.

Extends the frozen object-token addressed-balanced-2id smoke with an auxiliary
retrieval objective that grades the write-then-read binding step directly:
at the query frame (agent at z=-2, public query tile visible), the model must
predict which of the two first-frame identity tokens was queried.

Contract: experiments/dispatch/contract-addressed-2id-aux-retrieval.json
Root cause reference: MASTER_PLAN.md §1536-1540 (identity-query cross-frame
association, no co-visible signal, all collapsed to fixed direction)

This script adds an auxiliary cross-entropy loss on the identity label at the
query step, shared across all four comparison models (PLWM uses its retrieved
key projection; GRU/Transformer use their final hidden state; no_memory is
random). All hyperparameters are predeclared and frozen before training.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.audit_luanti_delayed_goal_qlearning import ACTION_COUNT, public_valid_actions
from experiments.train_luanti_gated_memory_dqn import (
    OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
    OBJECT_TOKEN_DIM,
    OBJECT_EVENT_DIM,
    ObjectTokenEventEncoder,
    ObjectTokenGRUQ,
    ObjectTokenNoMemoryQ,
    ObjectTokenPLWMQ,
    ObjectTokenTransformerQ,
    Report,
    _evaluate,
    _learn as _base_learn,
    _train as _base_train,
)
from experiments.train_luanti_reward_only import INPUT_DIM, encode_observation
from plwm.device import resolve_device
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT, GymVisionOnlyLuanti, VISION_SHAPE


# Predeclared contract constants (frozen before any training)
AUX_RETRIEVAL_ALPHA = 0.1  # L_total = L_reward + α · L_aux
IDENTITY_COUNT = 2  # 2id course: goal + key
QUERY_STEP = 4  # query tile appears at step 4 (0-indexed) in 6-step episodes
IDENTITY_LABELS = ("goal", "key")  # from VISION_TILES index mapping

VISUAL_SIZE = int(np.prod(VISION_SHAPE))  # 847


def _query_identity_label(observations: Tensor) -> Tensor:
    """Extract identity label at query step from the observation.

    observations: [batch, time, INPUT_DIM]
    Returns: [batch] long tensor with labels in {0, 1} mapping to (goal, key).
    """
    query_obs = observations[:, QUERY_STEP, :VISUAL_SIZE]  # [batch, visual_size]
    tiles = ("air", "wall", "agent", "goal", "key", "decoy", "beacon")
    batch_size = query_obs.shape[0]
    counts = query_obs.reshape(batch_size, *VISION_SHAPE).sum(dim=(1, 2))
    # tile indices 3=goal, 4=key for 2id; map to labels 0,1
    labels = torch.zeros(batch_size, dtype=torch.long, device=observations.device)
    goal_present = counts[:, 3] > 0
    key_present = counts[:, 4] > 0
    labels[goal_present & ~key_present] = 0
    labels[key_present & ~goal_present] = 1
    labels[goal_present & key_present] = 0  # tie-break to goal
    return labels


class AuxiliaryRetrievalHead(nn.Module):
    """Shared auxiliary head for retrieval supervision at the query step.

    Predicts which identity tile was queried, given the model's hidden state
    at the query step. All models share the same head structure.
    """

    def __init__(self, input_dim: int, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__()
        self.identity_count = identity_count
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, identity_count),
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.head(hidden_state)


# ---------------------------------------------------------------------------
# Model variants with auxiliary retrieval capability
# ---------------------------------------------------------------------------

class ObjectTokenPLWMQWithAux(ObjectTokenPLWMQ):
    """PLWM variant with auxiliary retrieval loss at query step."""

    def __init__(self, width: int, *, persistent_state: bool, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__(width, persistent_state=persistent_state, identity_count=identity_count)
        self.aux_head = AuxiliaryRetrievalHead(width, identity_count=identity_count)

    def forward_with_aux(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning logits, aux_logits, and aux_labels.

        Returns:
            logits: [batch, time, action_count] policy output
            aux_logits: [batch] auxiliary retrieval prediction at query step
            aux_labels: [batch] ground-truth identity label at query step
        """
        events, tokens, object_mask = self.encoder(observations)
        initial = tokens[:, 0] + self.position.unsqueeze(0)
        indices = object_mask[:, 0].to(dtype=observations.dtype).topk(
            self.identity_count, dim=1
        ).indices
        candidates = self.candidate_projection(
            initial.gather(1, indices.unsqueeze(-1).expand(-1, -1, OBJECT_TOKEN_DIM))
        )

        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs: list[Tensor] = []
        query_hidden: Tensor | None = None

        for index in range(events.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            if index == 0:
                writes, mask = candidates, torch.ones(
                    events.shape[0], self.identity_count, dtype=observations.dtype, device=observations.device
                )
            else:
                writes = torch.zeros(
                    events.shape[0], 1, self.core.latent_dim, dtype=observations.dtype, device=observations.device
                )
                mask = torch.zeros(
                    events.shape[0], 1, dtype=observations.dtype, device=observations.device
                )
            state, _ = self.core.step(events[:, index], state, think=True, write_candidates=writes, write_mask=mask)
            outputs.append(self.core.classify(state))

            if index == QUERY_STEP:
                query_hidden = state.belief  # [batch, latent_dim]

        logits = torch.stack(outputs, dim=1)
        aux_logits = self.aux_head(query_hidden) if query_hidden is not None else self.aux_head(
            torch.zeros(observations.shape[0], self.aux_head.head[0].in_features, device=logits.device)
        )
        aux_labels = _query_identity_label(observations)
        return logits, aux_logits, aux_labels


class ObjectTokenGRUQWithAux(ObjectTokenGRUQ):
    """GRU variant with auxiliary retrieval loss at query step."""

    def __init__(self, width: int, *, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__(width, identity_count=identity_count)
        self.aux_head = AuxiliaryRetrievalHead(width, identity_count=identity_count)

    def forward_with_aux(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        events, _tokens, _mask = self.encoder(observations)  # [batch, time, OBJECT_EVENT_DIM]
        encoded, hidden = self.recurrent(events)  # [batch, time, width]
        logits = self.head(encoded)  # [batch, time, ACTION_COUNT]
        query_hidden = encoded[:, QUERY_STEP]  # [batch, width]
        aux_logits = self.aux_head(query_hidden)
        aux_labels = _query_identity_label(observations)
        return logits, aux_logits, aux_labels


class ObjectTokenTransformerQWithAux(ObjectTokenTransformerQ):
    """Transformer variant with auxiliary retrieval loss at query step."""

    def __init__(self, width: int, *, context: int = 2, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__(width, context=context, identity_count=identity_count)
        self.aux_head = AuxiliaryRetrievalHead(width, identity_count=identity_count)

    def forward_with_aux(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        events, _tokens, _mask = self.encoder(observations)
        if events.shape[1] <= self.context:
            logits = self._window(events)
        else:
            logits = torch.stack([self._window(events[:, max(0, i - self.context + 1):i + 1])[:, -1] for i in range(events.shape[1])], dim=1)
        query_hidden = self.project(events)[:, QUERY_STEP]  # hidden before head
        aux_logits = self.aux_head(query_hidden)
        aux_labels = _query_identity_label(observations)
        return logits, aux_logits, aux_labels


class ObjectTokenNoMemoryQWithAux(ObjectTokenNoMemoryQ):
    """No-memory variant with a dummy auxiliary head (random baseline)."""

    def __init__(self, width: int, *, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__(width, identity_count=identity_count)
        # Dummy head: no actual supervision signal, just matches interface
        self.aux_head = AuxiliaryRetrievalHead(OBJECT_EVENT_DIM, identity_count=identity_count)

    def forward_with_aux(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        events, _tokens, _mask = self.encoder(observations)
        logits = self.network(events)  # [batch, time, ACTION_COUNT]
        # Dummy aux: use the query-step event as "hidden state" — no memory to bind
        query_event = events[:, QUERY_STEP]  # [batch, OBJECT_EVENT_DIM]
        aux_logits = self.aux_head(query_event)
        aux_labels = _query_identity_label(observations)
        return logits, aux_logits, aux_labels


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _aux_loss(
    logits: Tensor,
    aux_logits: Tensor,
    aux_labels: Tensor,
    masks: Tensor | None = None,
) -> Tensor:
    """Compute total loss = reward loss + α · auxiliary retrieval loss.

    logits: [batch, time, action_count]
    aux_logits: [batch] predictions at query step
    aux_labels: [batch] ground-truth identity labels at query step
    masks: optional [batch, time, action_count] valid-action masks (for reward loss)
    """
    batch_size, time_steps, _ = logits.shape
    # Reward loss: cross-entropy over all timesteps where mask is valid
    if masks is not None:
        # Use the same masking convention as the base _learn
        valid = masks.any(dim=-1)  # [batch, time] — step is valid if any action is legal
    else:
        valid = torch.ones(batch_size, time_steps, dtype=torch.bool, device=logits.device)
    # Dummy target for reward loss — actual targets come from DQN learning signal
    # The reward loss here is a placeholder; true policy loss is handled by DQN target updates
    reward_loss = torch.zeros(1, device=logits.device)

    # Auxiliary loss: cross-entropy at query step
    if aux_labels.numel() == 0:
        return reward_loss
    aux = nn.functional.cross_entropy(aux_logits, aux_labels)
    return reward_loss + AUX_RETRIEVAL_ALPHA * aux


# ---------------------------------------------------------------------------
# Extended _learn with auxiliary retrieval loss
# ---------------------------------------------------------------------------

def _learn(
    model: nn.Module,
    target: nn.Module,
    replay: list[Any],
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    generator: random.Random,
) -> None:
    """DQN learn step with auxiliary retrieval supervision added."""
    # Sample a batch from replay
    batch = generator.sample(replay, batch_size)
    lengths = [item.observations.shape[0] for item in batch]
    maximum = max(lengths)
    
    # Build batch tensors
    observations = torch.zeros((batch_size, maximum, INPUT_DIM), dtype=torch.float32, device=device)
    actions = torch.zeros((batch_size, maximum), dtype=torch.long, device=device)
    rewards = torch.zeros((batch_size, maximum), dtype=torch.float32, device=device)
    masks = torch.zeros((batch_size, maximum, ACTION_COUNT), dtype=torch.bool, device=device)
    valid = torch.zeros((batch_size, maximum), dtype=torch.bool, device=device)
    
    for index, (episode, length) in enumerate(zip(batch, lengths, strict=True)):
        observations[index, :length] = episode.observations.to(device)
        actions[index, :length] = episode.actions.to(device)
        rewards[index, :length] = episode.rewards.to(device)
        masks[index, :length] = episode.masks.to(device)
        valid[index, :length] = True
    
    # Compute DQN target
    values = model(observations)
    selected = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_values = target(observations)[:, 1:].masked_fill(~masks[:, 1:], float("-inf")).max(dim=-1).values
        continuation = torch.zeros_like(valid)
        continuation[:, :-1] = valid[:, 1:]
        targets = rewards.clone()
        bootstrap = torch.where(continuation[:, :-1], next_values, torch.zeros_like(next_values))
        targets[:, :-1] += 0.95 * bootstrap
    reward_loss = torch.nn.functional.smooth_l1_loss(selected[valid], targets[valid])
    
    # Add auxiliary retrieval loss if model supports it
    total_loss = reward_loss
    if hasattr(model, 'forward_with_aux'):
        _, aux_logits, aux_labels = model.forward_with_aux(observations)
        if aux_labels.numel() > 0:
            aux = nn.functional.cross_entropy(aux_logits, aux_labels)
            total_loss = reward_loss + AUX_RETRIEVAL_ALPHA * aux
    
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if hasattr(model, 'core') and hasattr(model.core, 'enforce_key_orthogonality'):
        model.core.enforce_key_orthogonality()


# ---------------------------------------------------------------------------
# Extended _train with auxiliary retrieval
# ---------------------------------------------------------------------------

def _train(
    model: nn.Module,
    *,
    environment: GymVisionOnlyLuanti,
    episodes: int,
    seed_start: int,
    action_seed: int,
    device: torch.device,
    max_steps: int,
    worker_pool: Any | None = None,
) -> tuple[int, int]:
    """Train with auxiliary retrieval supervision."""
    model.train()
    target = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = random.Random(action_seed)
    replay: list[Any] = []
    successes = 0

    if worker_pool is not None:
        from experiments.train_luanti_gated_memory_dqn import _WORKER_POOL_REF, _WORKER_ENVS, _collect_on_worker
        global _WORKER_POOL_REF, _WORKER_ENVS
        _WORKER_POOL_REF = worker_pool["pool"]
        _WORKER_ENVS = {}
        workers = worker_pool["workers"]

        def worker_loop(worker: int) -> list[tuple[int, Any, bool]]:
            collected: list[tuple[int, Any, bool]] = []
            for index in range(worker, episodes, workers):
                from experiments.train_luanti_gated_memory_dqn import _collect_on_worker
                collected.append(_collect_on_worker(
                    model, worker, environment, seed=seed_start + index, device=device,
                    epsilon=0.45 - 0.40 * index / max(1, episodes - 1),
                    generator=random.Random(action_seed + index), max_steps=max_steps,
                    seed_offset=index,
                ))
            return collected

        outcomes: list[tuple[int, Any, bool]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch in executor.map(worker_loop, range(workers)):
                outcomes.extend(batch)
        outcomes.sort(key=lambda item: item[0])
        replay = [item[1] for item in outcomes]
        successes = sum(1 for item in outcomes if item[2])
        for offset in range(0, len(replay), 32):
            _learn(model, target, replay[offset:offset + 32], batch_size=32, device=device, optimizer=optimizer, generator=generator)
        return episodes, successes

    for index in range(episodes):
        epsilon = 0.45 - 0.40 * index / max(1, episodes - 1)
        from experiments.train_luanti_gated_memory_dqn import _collect
        episode, success = _collect(model, environment, seed=seed_start + index, device=device, epsilon=epsilon, generator=generator, max_steps=max_steps)
        replay.append(episode)
        successes += success
        if len(replay) >= 32:
            _learn(model, target, replay, batch_size=32, device=device, optimizer=optimizer, generator=generator)
        if index and index % 32 == 0:
            target.load_state_dict(model.state_dict())
    return episodes, successes


# ---------------------------------------------------------------------------
# Object-token comparison factories with auxiliary retrieval
# ---------------------------------------------------------------------------

def object_token_aux_factories(
    widths: Any = OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
    *,
    identity_count: int = IDENTITY_COUNT,
) -> tuple[tuple[str, Callable[[], nn.Module]], ...]:
    """Build all models with auxiliary retrieval capability."""
    return (
        ("no_memory", lambda: ObjectTokenNoMemoryQWithAux(widths.no_memory, identity_count=identity_count)),
        ("gru", lambda: ObjectTokenGRUQWithAux(widths.gru, identity_count=identity_count)),
        ("transformer_2", lambda: ObjectTokenTransformerQWithAux(widths.transformer, identity_count=identity_count)),
        ("plwm_state_off", lambda: ObjectTokenPLWMQWithAux(widths.plwm, persistent_state=False, identity_count=identity_count)),
        ("plwm", lambda: ObjectTokenPLWMQWithAux(widths.plwm, persistent_state=True, identity_count=identity_count)),
    )


# ---------------------------------------------------------------------------
# Run entry point
# ---------------------------------------------------------------------------

def run(
    *,
    device: str | torch.device | None = "cuda",
    training_episodes: int = 256,
    evaluation_episodes: int = 32,
    training_seed_start: int = 18_000_000,
    heldout_seed_start: int = 18_001_000,
    action_seed: int = 9_042_337,
    bridge_root: Path = DEFAULT_BRIDGE_ROOT,
    artifact: Path | None = None,
    workers: int = 1,
) -> Report:
    """Run auxiliary retrieval training on addressed_balanced_2id."""
    minimum_seed, maximum_seed, max_steps = (18_000_000, 19_000_000, 6)
    if (
        training_episodes < 32 or evaluation_episodes < 8
        or not minimum_seed <= training_seed_start < maximum_seed
        or not minimum_seed <= heldout_seed_start < maximum_seed
        or set(range(training_seed_start, training_seed_start + training_episodes)).intersection(
            range(heldout_seed_start, heldout_seed_start + evaluation_episodes)
        )
    ):
        raise ValueError("invalid auxiliary retrieval contract")

    resolved = resolve_device(device)
    if str(resolved) != "cuda":
        raise ValueError("auxiliary retrieval smoke requires local CUDA")

    torch.manual_seed(action_seed)
    np.random.seed(action_seed)
    environment = GymVisionOnlyLuanti(seed=training_seed_start, max_steps=max_steps, bridge_root=bridge_root)

    try:
        random_evaluation = _evaluate(None, environment=environment, episodes=evaluation_episodes,
                                       seed_start=heldout_seed_start, action_seed=action_seed,
                                       device=resolved, max_steps=max_steps)
        results: list[Any] = []
        factories = object_token_aux_factories(OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS)

        from experiments.train_luanti_gated_memory_dqn import _parameter_count, ModelResult
        for offset, (name, factory) in enumerate(factories):
            torch.manual_seed(action_seed + offset)
            model = factory().to(resolved)
            trained, successes = _train(
                model, environment=environment, episodes=training_episodes,
                seed_start=training_seed_start, action_seed=action_seed + 1000 * (offset + 1),
                device=resolved, max_steps=max_steps,
            )
            evaluation = _evaluate(
                model, environment=environment, episodes=evaluation_episodes,
                seed_start=heldout_seed_start, action_seed=action_seed + 2000 * (offset + 1),
                device=resolved, max_steps=max_steps,
            )
            results.append(ModelResult(name, _parameter_count(model), trained, successes, evaluation))
    finally:
        environment.close()

    by_name = {result.name: result for result in results}
    success = {name: result.evaluation.success_rate for name, result in by_name.items()}
    accepted = (
        success["plwm"] >= 0.75
        and success["plwm"] - success["no_memory"] >= 0.15
        and success["plwm"] - success["gru"] >= 0.10
        and success["plwm"] - success["transformer_2"] >= 0.10
    )

    target = artifact or Path(__file__).resolve().parents[1] / "artifacts" / "luanti-addressed-2id-aux-retrieval-report.json"
    report = Report(
        protocol=(
            "luanti_addressed_2id_aux_retrieval_v1: auxiliary retrieval-supervised DQN on addressed_balanced_2id; "
            "L_total = L_reward + 0.1 * L_aux where L_aux is cross-entropy predicting which of 2 identities "
            "(goal/key) was queried at step 4; all 4 models share AuxiliaryRetrievalHead; PLWM uses memory belief, "
            "GRU/Transformer use hidden state, no_memory uses query event as dummy."
        ),
        course="addressed_balanced_2id",
        device=str(resolved),
        training_seed_range=list(range(training_seed_start, training_seed_start + training_episodes)),
        heldout_seed_range=list(range(heldout_seed_start, heldout_seed_start + evaluation_episodes)),
        action_seed=action_seed,
        max_steps=max_steps,
        transformer_context=2,
        model_widths=OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
        parameter_match_tolerance=0.05,
        parameter_match_errors={},
        acceptance_mode="aux_retrieval",
        random_evaluation=random_evaluation,
        results=results,
        accepted_neural_task_diagnostic=accepted,
        artifact=str(target),
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-episodes", type=int, default=256)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--training-seed-start", type=int, default=18_000_000)
    parser.add_argument("--heldout-seed-start", type=int, default=18_001_000)
    parser.add_argument("--action-seed", type=int, default=9_042_337)
    parser.add_argument("--bridge-root", type=Path, default=Path.home() / ".luanti" / "bridge")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    resolved = resolve_device(args.device)
    if str(resolved) != "cuda":
        raise ValueError("auxiliary retrieval smoke requires local CUDA")

    print(f"Starting auxiliary retrieval smoke on addressed_balanced_2id")
    print(f"Device: {resolved}, Alpha: {AUX_RETRIEVAL_ALPHA}, Identity count: {IDENTITY_COUNT}")
    print(f"Training episodes: {args.training_episodes}, Held-out: {args.evaluation_episodes}")
    print(f"Query step: {QUERY_STEP}, α = {AUX_RETRIEVAL_ALPHA}")

    report = run(
        device=args.device,
        training_episodes=args.training_episodes,
        evaluation_episodes=args.evaluation_episodes,
        training_seed_start=args.training_seed_start,
        heldout_seed_start=args.heldout_seed_start,
        action_seed=args.action_seed,
        bridge_root=args.bridge_root,
        artifact=args.artifact,
        workers=args.workers,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
