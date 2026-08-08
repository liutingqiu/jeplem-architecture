"""Train and verify goal-directed behavior in real partially hidden Luanti worlds.

Each seeded arena contains an obstacle barrier and a hidden control inversion.
The model observes only local sensors and the consequences of its own actions;
the shortest-path teacher remains an evaluation-only source of labels.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import (
    DIRECTION_TO_INDEX,
    DIRECTIONS,
    ActionConditionedWorldModel,
    CausalControlWorldModel,
    ControlMechanismMemory,
    GRUContextGoalPolicy,
    GoalDirectedPLWM,
    LuantiInteraction,
    LuantiLiveClient,
    LuantiState,
    MaskedSensorGoalPolicy,
    NoMemoryGoalPolicy,
    NoMemoryObjectPolicy,
    ObjectGoalDirectedPLWM,
    ObjectTaskState,
    GRUObjectPolicy,
    PredictedTransition,
    StatelessControlBaseline,
    padded_histories,
    padded_object_histories,
    VisionGoalDirectedPLWM,
    padded_visual_histories,
)


BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"
HISTORY = 4
MAX_STEPS = 18
VISUAL_MAX_STEPS = 28


@dataclass
class ExecutedAction:
    direction: str
    request_id: str
    status: str
    reward: float | None
    terminal: bool | None


@dataclass
class Trajectory:
    episode_id: str
    task_seed: int
    collection: str = "exploration"
    states: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ExecutedAction] = field(default_factory=list)
    visions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ObjectTrajectory:
    episode_id: str
    task_seed: int
    states: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ExecutedAction] = field(default_factory=list)


@dataclass
class Evaluation:
    successes: int
    episodes: int
    mean_steps: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


@dataclass
class TransitionEvaluation:
    transitions: int
    displacement_accuracy: float
    blocked_accuracy: float
    revisions: int = 0


@dataclass
class RuleSwitchReport:
    episodes: int
    causal: TransitionEvaluation
    stateless: TransitionEvaluation
    accepted: bool
    artifact: str


@dataclass
class RetentionReport:
    initial_probe_correct: bool
    retained_probe_correct: bool
    fresh_probe_correct: bool
    isolated_world_mapping: bool
    accepted: bool
    artifact: str


@dataclass
class ObjectPolicyReport:
    device: str
    training_seed_count: int
    heldout_seed_count: int
    training_samples: int
    learned: Evaluation
    no_memory: Evaluation
    gru_context: Evaluation
    random: Evaluation
    reactive: Evaluation
    accepted: bool
    checkpoint: str
    artifact: str


@dataclass
class VisualPolicyReport:
    device: str
    training_seed_count: int
    heldout_seed_count: int
    training_samples: int
    visual: Evaluation
    symbolic: Evaluation
    masked_symbolic: Evaluation
    random: Evaluation
    reactive: Evaluation
    accepted: bool
    checkpoint: str
    artifact: str


@dataclass
class Report:
    device: str
    training_seed_count: int
    heldout_seed_count: int
    trajectories: int
    training_samples: int
    final_policy_loss: float
    final_prediction_loss: float
    action_world_model: TransitionEvaluation
    stateless_action_world_model: TransitionEvaluation
    learned: Evaluation
    no_memory: Evaluation
    gru_context: Evaluation
    random: Evaluation
    reactive: Evaluation
    accepted: bool
    checkpoint: str
    rollout_artifact: str


def collect_random_trajectory(
    client: LuantiLiveClient,
    *,
    task_seed: int,
    episode_id: str,
    rng: random.Random,
    max_steps: int,
) -> Trajectory:
    state, vision = client.begin_episode_with_vision(episode_id, task_seed)
    trajectory = Trajectory(episode_id=episode_id, task_seed=task_seed, collection="exploration", states=[asdict(state)], visions=[dict(vision)])
    for step in range(max_steps):
        if state.completed:
            break
        # Establish a causally informative observation before exploration.
        # The policy must later infer the control mapping from this outcome.
        direction = "east" if step == 0 else rng.choice(DIRECTIONS)
        interaction, next_state, next_vision = client.step_with_vision(direction)
        trajectory.actions.append(_executed_action(direction, interaction))
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            break
        trajectory.states.append(asdict(next_state))
        trajectory.visions.append(dict(next_vision or {}))
        state = next_state
        if step == 0:
            recenter, recentered_state, recentered_vision = client.recenter_with_vision()
            trajectory.actions.append(_executed_action("recenter", recenter))
            if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                break
            trajectory.states.append(asdict(recentered_state))
            trajectory.visions.append(dict(recentered_vision or {}))
            state = recentered_state
    return trajectory


def collect_teacher_trajectory(
    client: LuantiLiveClient,
    *,
    task_seed: int,
    episode_id: str,
    max_steps: int,
) -> Trajectory:
    """Execute the hidden-rule teacher in the real game to cover viable paths."""
    state, vision = client.begin_episode_with_vision(episode_id, task_seed)
    trajectory = Trajectory(episode_id=episode_id, task_seed=task_seed, collection="teacher", states=[asdict(state)], visions=[dict(vision)])
    for step in range(max_steps):
        direction = state.oracle_direction()
        if direction is None:
            break
        interaction, next_state, next_vision = client.step_with_vision(direction)
        trajectory.actions.append(_executed_action(direction, interaction))
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            break
        trajectory.states.append(asdict(next_state))
        trajectory.visions.append(dict(next_vision or {}))
        state = next_state
        if step == 0:
            recenter, recentered_state, recentered_vision = client.recenter_with_vision()
            trajectory.actions.append(_executed_action("recenter", recenter))
            if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                break
            trajectory.states.append(asdict(recentered_state))
            trajectory.visions.append(dict(recentered_vision or {}))
            state = recentered_state
    return trajectory


def collect_switch_trajectory(
    client: LuantiLiveClient,
    *,
    task_seed: int,
    episode_id: str,
    max_steps: int,
) -> Trajectory:
    """Probe then repeatedly request east to expose the hidden regime switch."""
    state = client.begin_episode(episode_id, task_seed)
    trajectory = Trajectory(episode_id=episode_id, task_seed=task_seed, collection="rule-switch", states=[asdict(state)])
    for step in range(max_steps):
        interaction, next_state = client.step("east")
        trajectory.actions.append(_executed_action("east", interaction))
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            break
        trajectory.states.append(asdict(next_state))
        state = next_state
        if step == 0:
            recenter, recentered_state = client.recenter()
            trajectory.actions.append(_executed_action("recenter", recenter))
            if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                break
            trajectory.states.append(asdict(recentered_state))
            state = recentered_state
    return trajectory


def collect_object_teacher_trajectory(
    client: LuantiLiveClient,
    *,
    task_seed: int,
    episode_id: str,
    max_steps: int,
) -> ObjectTrajectory:
    state = client.begin_object_episode(episode_id, task_seed)
    trajectory = ObjectTrajectory(episode_id=episode_id, task_seed=task_seed, states=[_object_state_dict(state)])
    for step in range(max_steps):
        direction = state.oracle_direction()
        if direction is None:
            break
        interaction, next_state = client.step_object(direction)
        trajectory.actions.append(_executed_action(direction, interaction))
        if interaction.outcome.get("status") != "accepted" or next_state is None:
            break
        trajectory.states.append(_object_state_dict(next_state))
        state = next_state
        if step == 0:
            recenter, recentered_state = client.recenter_object()
            trajectory.actions.append(_executed_action("recenter", recenter))
            if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                break
            trajectory.states.append(_object_state_dict(recentered_state))
            state = recentered_state
    return trajectory


def _object_state_dict(state: ObjectTaskState) -> dict[str, Any]:
    payload = asdict(state.base)
    payload.update(
        {
            "object_mode": True,
            "key_x": state.key_x,
            "key_z": state.key_z,
            "decoy_x": state.decoy_x,
            "decoy_z": state.decoy_z,
            "key_collected": state.key_collected,
            "decoy_collected": state.decoy_collected,
            "last_reward": state.last_reward,
        }
    )
    return payload


def _executed_action(direction: str, interaction: LuantiInteraction) -> ExecutedAction:
    reward_payload = interaction.reward or {}
    reward = reward_payload.get("value")
    terminal = interaction.outcome.get("terminal")
    return ExecutedAction(
        direction=direction,
        request_id=interaction.request_id,
        status=str(interaction.outcome.get("status")),
        reward=float(reward) if isinstance(reward, (int, float)) else None,
        terminal=bool(terminal) if isinstance(terminal, bool) else None,
    )


def training_tensors(trajectories: Sequence[Trajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    windows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for trajectory in trajectories:
        states = [LuantiState(**state) for state in trajectory.states]
        skip_labels = frozenset(index for index, action in enumerate(trajectory.actions) if action.direction == "recenter")
        episode_windows, episode_labels = padded_histories(states, history=HISTORY, skip_label_indices=skip_labels)
        if len(episode_labels):
            windows.append(episode_windows)
            labels.append(episode_labels)
    if not windows:
        raise ValueError("no trainable Luanti states were collected")
    return torch.cat(windows), torch.cat(labels)


def object_training_tensors(trajectories: Sequence[ObjectTrajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    windows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(state) for state in trajectory.states]
        skip_labels = frozenset(index for index, action in enumerate(trajectory.actions) if action.direction == "recenter")
        episode_windows, episode_labels = padded_object_histories(
            states, history=HISTORY, skip_label_indices=skip_labels
        )
        if len(episode_labels):
            windows.append(episode_windows)
            labels.append(episode_labels)
    if not windows:
        raise ValueError("no trainable Luanti object states were collected")
    return torch.cat(windows), torch.cat(labels)


def visual_training_tensors(trajectories: Sequence[Trajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    windows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for trajectory in trajectories:
        states = [LuantiState(**state) for state in trajectory.states]
        skip_labels = frozenset(index for index, action in enumerate(trajectory.actions) if action.direction == "recenter")
        episode_windows, episode_labels = padded_visual_histories(
            states,
            trajectory.visions,
            history=HISTORY,
            skip_label_indices=skip_labels,
            actual_direction_labels=True,
        )
        if len(episode_labels):
            windows.append(episode_windows)
            labels.append(episode_labels)
    if not windows:
        raise ValueError("no trainable Luanti visual states were collected")
    return torch.cat(windows), torch.cat(labels)


def transition_tensors(
    trajectories: Sequence[Trajectory], *, history: int = HISTORY
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build causal `(history, requested action, next outcome)` training rows."""
    windows: list[list[tuple[float, ...]]] = []
    actions: list[int] = []
    outcomes: list[tuple[float, float, float]] = []
    for trajectory in trajectories:
        states = [LuantiState(**state) for state in trajectory.states]
        for index, action in enumerate(trajectory.actions):
            if index + 1 >= len(states) or action.status != "accepted" or action.direction not in DIRECTION_TO_INDEX:
                continue
            current, following = states[index], states[index + 1]
            prefix = list(states[max(0, index - history + 1) : index + 1])
            prefix = [prefix[0]] * (history - len(prefix)) + prefix
            windows.append([item.features() for item in prefix])
            actions.append(DIRECTION_TO_INDEX[action.direction])
            outcomes.append((float(following.x - current.x), float(following.z - current.z), float(following.last_blocked)))
    if not windows:
        raise ValueError("no causal Luanti transitions were collected")
    return (
        torch.tensor(windows, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(outcomes, dtype=torch.float32),
    )


def train_policy(
    model: torch.nn.Module,
    observations: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    model.train()
    policy_loss_value = 0.0
    prediction_loss_value = 0.0
    batch_size = min(128, len(labels))
    for _ in range(steps):
        indices = torch.randint(len(labels), (batch_size,), generator=generator)
        batch_observations = observations[indices].to(device)
        batch_labels = labels[indices].to(device)
        logits, surprises = model(batch_observations)
        policy_loss = cross_entropy(logits, batch_labels)
        prediction_loss = surprises.mean()
        loss = policy_loss + 0.05 * prediction_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_loss_value = float(policy_loss.detach().cpu())
        prediction_loss_value = float(prediction_loss.detach().cpu())
    return policy_loss_value, prediction_loss_value


def train_action_world_model(
    model: ActionConditionedWorldModel,
    histories: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    binary_cross_entropy = nn.BCEWithLogitsLoss()
    model.train()
    loss_value = 0.0
    batch_size = min(128, len(actions))
    for _ in range(steps):
        indices = torch.randint(len(actions), (batch_size,), generator=generator)
        predicted_x, predicted_z, predicted_blocked, surprises = model(histories[indices].to(device), actions[indices].to(device))
        target = outcomes[indices].to(device)
        loss = (
            cross_entropy(predicted_x, (target[:, 0] + 1).long())
            + cross_entropy(predicted_z, (target[:, 1] + 1).long())
            + binary_cross_entropy(predicted_blocked, target[:, 2])
            + 0.05 * surprises.mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    return loss_value


@torch.no_grad()
def evaluate_action_world_model(
    model: ActionConditionedWorldModel,
    histories: torch.Tensor,
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    device: torch.device,
) -> TransitionEvaluation:
    model.eval()
    predicted_x, predicted_z, predicted_blocked, _surprises = model(histories.to(device), actions.to(device))
    target = outcomes.to(device)
    predicted_delta = torch.stack((predicted_x.argmax(dim=-1) - 1, predicted_z.argmax(dim=-1) - 1), dim=-1)
    displacement_accuracy = float((predicted_delta == target[:, :2]).all(dim=1).float().mean().cpu())
    blocked_accuracy = float(((predicted_blocked.sigmoid() >= 0.5) == target[:, 2].bool()).float().mean().cpu())
    return TransitionEvaluation(len(actions), displacement_accuracy, blocked_accuracy)


def evaluate_causal_control_model(
    trajectories: Sequence[Trajectory],
    model_factory: Callable[[], CausalControlWorldModel | StatelessControlBaseline],
) -> TransitionEvaluation:
    """Score online predictions before each real action outcome is revealed."""
    predicted: list[PredictedTransition] = []
    targets: list[PredictedTransition] = []
    revisions = 0
    for trajectory in trajectories:
        model = model_factory()
        model.reset()
        states = [LuantiState(**state) for state in trajectory.states]
        for index, action in enumerate(trajectory.actions):
            if index + 1 >= len(states) or action.status != "accepted" or action.direction not in DIRECTIONS:
                continue
            before, after = states[index], states[index + 1]
            predicted.append(model.predict(before, action.direction))
            targets.append(PredictedTransition(after.x - before.x, after.z - before.z, after.last_blocked))
            model.observe(before, action.direction, after)
        revisions += getattr(model, "revision_count", 0)
    if not targets:
        raise ValueError("no causal Luanti transitions were collected")
    displacement_accuracy = sum(
        prediction.delta_x == target.delta_x and prediction.delta_z == target.delta_z
        for prediction, target in zip(predicted, targets)
    ) / len(targets)
    blocked_accuracy = sum(prediction.blocked == target.blocked for prediction, target in zip(predicted, targets)) / len(targets)
    return TransitionEvaluation(len(targets), displacement_accuracy, blocked_accuracy, revisions)


def run_rule_switch(
    *,
    episode_count: int = 8,
    steps: int = 12,
    bridge_root: Path = BRIDGE_ROOT,
) -> RuleSwitchReport:
    """Evaluate online rule revision in real high-seed Luanti switch worlds."""
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    trajectories = [
        collect_switch_trajectory(
            client,
            task_seed=1_500_000 + index * 31,
            episode_id=f"rule-switch-{index}",
            max_steps=steps,
        )
        for index in range(episode_count)
    ]
    causal = evaluate_causal_control_model(trajectories, CausalControlWorldModel)
    stateless = evaluate_causal_control_model(trajectories, StatelessControlBaseline)
    accepted = (
        causal.revisions == episode_count
        and causal.displacement_accuracy >= 0.8
        and causal.blocked_accuracy >= 0.9
        and causal.displacement_accuracy >= stateless.displacement_accuracy + 0.3
    )
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    artifact = ARTIFACT_ROOT / "luanti-rule-switch-report.json"
    report = RuleSwitchReport(episode_count, causal, stateless, accepted, str(artifact))
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def _probe_control_mapping(
    client: LuantiLiveClient,
    *,
    episode_id: str,
    task_seed: int,
    model: CausalControlWorldModel,
) -> bool:
    state = client.begin_episode(episode_id, task_seed)
    prediction = model.predict(state, "east")
    interaction, after = client.step("east")
    if interaction.outcome.get("status") != "accepted" or after is None:
        raise RuntimeError(f"Luanti probe was rejected: {interaction.outcome}")
    actual = PredictedTransition(after.x - state.x, after.z - state.z, after.last_blocked)
    model.observe(state, "east", after)
    return prediction == actual


def run_rule_retention(*, bridge_root: Path = BRIDGE_ROOT) -> RetentionReport:
    """Verify a confirmed rule survives an unrelated world's episode."""
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    memory = ControlMechanismMemory()
    inverse_seed, normal_seed = 0, 1
    inverse_world, normal_world = "retention-inverse", "retention-normal"

    initial = memory.begin(inverse_world)
    initial_probe_correct = _probe_control_mapping(
        client, episode_id="retention-initial", task_seed=inverse_seed, model=initial
    )
    memory.commit(inverse_world, initial)

    unrelated = memory.begin(normal_world)
    _probe_control_mapping(client, episode_id="retention-unrelated", task_seed=normal_seed, model=unrelated)
    memory.commit(normal_world, unrelated)

    retained = memory.begin(inverse_world)
    retained_probe_correct = _probe_control_mapping(
        client, episode_id="retention-return", task_seed=inverse_seed, model=retained)
    fresh_probe_correct = _probe_control_mapping(
        client, episode_id="retention-fresh", task_seed=inverse_seed, model=CausalControlWorldModel())
    isolated_world_mapping = memory.mapping_for(inverse_world) is True and memory.mapping_for(normal_world) is False
    accepted = not initial_probe_correct and retained_probe_correct and not fresh_probe_correct and isolated_world_mapping

    ARTIFACT_ROOT.mkdir(exist_ok=True)
    artifact = ARTIFACT_ROOT / "luanti-rule-retention-report.json"
    report = RetentionReport(
        initial_probe_correct,
        retained_probe_correct,
        fresh_probe_correct,
        isolated_world_mapping,
        accepted,
        str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def evaluate(
    client: LuantiLiveClient,
    *,
    seeds: Sequence[int],
    chooser: Callable[[Sequence[LuantiState]], str],
    prefix: str,
    max_steps: int,
    persistent_model: GoalDirectedPLWM | None = None,
) -> Evaluation:
    successes = 0
    steps_taken: list[int] = []
    for index, task_seed in enumerate(seeds):
        state = client.begin_episode(f"{prefix}-{task_seed}-{index}", task_seed)
        history = [state]
        session = persistent_model.begin_episode(state) if persistent_model is not None else None
        for step in range(max_steps):
            if state.completed:
                break
            direction = session.choose() if session is not None else chooser(history)
            interaction, next_state = client.step(direction)
            if interaction.outcome.get("status") != "accepted" or next_state is None:
                break
            state = next_state
            history.append(state)
            if session is not None:
                session.observe(state)
            if step == 0:
                recenter, recentered_state = client.recenter()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                    break
                state = recentered_state
                history.append(state)
                if session is not None:
                    session.observe(state)
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(max_steps)
    return Evaluation(successes=successes, episodes=len(seeds), mean_steps=sum(steps_taken) / len(seeds))


def evaluate_object_policy(
    client: LuantiLiveClient,
    *,
    seeds: Sequence[int],
    chooser: Callable[[Sequence[ObjectTaskState]], str],
    prefix: str,
    max_steps: int,
) -> Evaluation:
    successes = 0
    steps_taken: list[int] = []
    for index, task_seed in enumerate(seeds):
        state = client.begin_object_episode(f"{prefix}-{task_seed}-{index}", task_seed)
        history = [state]
        for step in range(max_steps):
            if state.completed:
                break
            direction = chooser(history)
            interaction, next_state = client.step_object(direction)
            if interaction.outcome.get("status") != "accepted" or next_state is None:
                break
            state = next_state
            history.append(state)
            if step == 0:
                recenter, recentered_state = client.recenter_object()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                    break
                state = recentered_state
                history.append(state)
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(max_steps)
    return Evaluation(successes=successes, episodes=len(seeds), mean_steps=sum(steps_taken) / len(seeds))


def run_object_behavior(
    *,
    training_seed_count: int = 16,
    heldout_seed_count: int = 8,
    training_steps: int = 600,
    seed: int = 223,
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
) -> ObjectPolicyReport:
    """Train and evaluate long-horizon object consequences in real Luanti."""
    torch.manual_seed(seed)
    resolved_device = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    training_seeds = [2_000_000 + index * 17 for index in range(training_seed_count)]
    heldout_seeds = [2_100_000 + index * 31 for index in range(heldout_seed_count)]
    trajectories = [
        collect_object_teacher_trajectory(
            client,
            task_seed=task_seed,
            episode_id=f"object-teacher-{task_seed}",
            max_steps=MAX_STEPS,
        )
        for task_seed in training_seeds
    ]
    observations, labels = object_training_tensors(trajectories)
    model = ObjectGoalDirectedPLWM(history_length=HISTORY).to(resolved_device)
    train_policy(model, observations, labels, device=resolved_device, steps=training_steps, seed=seed)
    no_memory_model = NoMemoryObjectPolicy().to(resolved_device)
    train_policy(no_memory_model, observations, labels, device=resolved_device, steps=training_steps, seed=seed + 101)
    gru_context_model = GRUObjectPolicy(history_length=HISTORY).to(resolved_device)
    train_policy(gru_context_model, observations, labels, device=resolved_device, steps=training_steps, seed=seed + 102)
    model.eval()
    learned = evaluate_object_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: model.choose(history[-HISTORY:]),
        prefix="object-learned",
        max_steps=MAX_STEPS,
    )
    no_memory_model.eval()
    no_memory = evaluate_object_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: no_memory_model.choose(history),
        prefix="object-no-memory",
        max_steps=MAX_STEPS,
    )
    gru_context_model.eval()
    gru_context = evaluate_object_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: gru_context_model.choose(history[-HISTORY:]),
        prefix="object-gru-context",
        max_steps=MAX_STEPS,
    )
    random_rng = random.Random(seed + 1)
    random_baseline = evaluate_object_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda _history: random_rng.choice(DIRECTIONS),
        prefix="object-random",
        max_steps=MAX_STEPS,
    )
    reactive = evaluate_object_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: history[-1].oracle_direction() or "north",
        prefix="object-reactive",
        max_steps=MAX_STEPS,
    )
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    checkpoint = ARTIFACT_ROOT / "luanti-object-policy.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": ObjectGoalDirectedPLWM.input_dim, "history_length": HISTORY}, checkpoint)
    artifact = ARTIFACT_ROOT / "luanti-object-training-report.json"
    report = ObjectPolicyReport(
        device=str(resolved_device),
        training_seed_count=training_seed_count,
        heldout_seed_count=heldout_seed_count,
        training_samples=len(labels),
        learned=learned,
        random=random_baseline,
        reactive=reactive,
        no_memory=no_memory,
        gru_context=gru_context,
        accepted=(learned.success_rate >= 0.9 and learned.success_rate >= no_memory.success_rate + 0.3 and reactive.success_rate == 1.0),
        checkpoint=str(checkpoint),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def evaluate_visual_policy(
    client: LuantiLiveClient,
    *,
    seeds: Sequence[int],
    chooser: Callable[[Sequence[tuple[LuantiState, dict[str, Any]]]], str],
    prefix: str,
    max_steps: int,
) -> Evaluation:
    successes = 0
    steps_taken: list[int] = []
    for index, task_seed in enumerate(seeds):
        state, vision = client.begin_episode_with_vision(f"{prefix}-{task_seed}-{index}", task_seed)
        history: list[tuple[LuantiState, dict[str, Any]]] = [(state, dict(vision))]
        for step in range(max_steps):
            if state.completed:
                break
            direction = chooser(history)
            interaction, next_state, next_vision = client.step_with_vision(direction)
            if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
                break
            state, vision = next_state, dict(next_vision)
            history.append((state, vision))
            if step == 0:
                recenter, recentered_state, recentered_vision = client.recenter_with_vision()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                    break
                state, vision = recentered_state, dict(recentered_vision)
                history.append((state, vision))
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(max_steps)
    return Evaluation(successes=successes, episodes=len(seeds), mean_steps=sum(steps_taken) / len(seeds))


def evaluate_visual_causal_policy(
    client: LuantiLiveClient,
    *,
    seeds: Sequence[int],
    chooser: Callable[[Sequence[tuple[LuantiState, dict[str, Any]]]], str],
    prefix: str,
    max_steps: int,
) -> Evaluation:
    """Use visual planning for actual directions and causal memory for requests."""
    successes = 0
    steps_taken: list[int] = []
    for index, task_seed in enumerate(seeds):
        state, vision = client.begin_episode_with_vision(f"{prefix}-{task_seed}-{index}", task_seed)
        history: list[tuple[LuantiState, dict[str, Any]]] = [(state, dict(vision))]
        mechanism = CausalControlWorldModel()
        for step in range(max_steps):
            if state.completed:
                break
            actual_direction = chooser(history)
            requested_direction = mechanism.request_for_actual(actual_direction)
            before = state
            interaction, next_state, next_vision = client.step_with_vision(requested_direction)
            if interaction.outcome.get("status") != "accepted" or next_state is None or next_vision is None:
                break
            mechanism.observe(before, requested_direction, next_state)
            state, vision = next_state, dict(next_vision)
            history.append((state, vision))
            if step == 0:
                recenter, recentered_state, recentered_vision = client.recenter_with_vision()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None or recentered_vision is None:
                    break
                state, vision = recentered_state, dict(recentered_vision)
                history.append((state, vision))
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(max_steps)
    return Evaluation(successes=successes, episodes=len(seeds), mean_steps=sum(steps_taken) / len(seeds))


def run_visual_behavior(
    *,
    training_seed_count: int = 16,
    heldout_seed_count: int = 8,
    training_steps: int = 600,
    seed: int = 331,
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
) -> VisualPolicyReport:
    """Train a vision-augmented policy and compare a symbolic state policy."""
    torch.manual_seed(seed)
    resolved_device = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    training_seeds = [1_400_000 + index * 17 for index in range(training_seed_count)]
    heldout_seeds = [1_450_000 + index * 31 for index in range(heldout_seed_count)]
    trajectories = [
        collect_teacher_trajectory(
            client,
            task_seed=task_seed,
            episode_id=f"visual-teacher-{task_seed}",
            max_steps=VISUAL_MAX_STEPS,
        )
        for task_seed in training_seeds
    ]
    visual_observations, labels = visual_training_tensors(trajectories)
    visual_model = VisionGoalDirectedPLWM(history_length=HISTORY).to(resolved_device)
    train_policy(visual_model, visual_observations, labels, device=resolved_device, steps=training_steps, seed=seed)
    symbolic_observations, symbolic_labels = training_tensors(trajectories)
    symbolic_model = GoalDirectedPLWM(history_length=HISTORY).to(resolved_device)
    train_policy(symbolic_model, symbolic_observations, symbolic_labels, device=resolved_device, steps=training_steps, seed=seed + 1)
    masked_symbolic_model = MaskedSensorGoalPolicy(history_length=HISTORY).to(resolved_device)
    train_policy(masked_symbolic_model, symbolic_observations, symbolic_labels, device=resolved_device, steps=training_steps, seed=seed + 2)
    visual_model.eval()
    symbolic_model.eval()
    masked_symbolic_model.eval()
    visual = evaluate_visual_causal_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: visual_model.choose(history[-HISTORY:]),
        prefix="visual-learned",
        max_steps=VISUAL_MAX_STEPS,
    )
    symbolic = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: symbolic_model.choose(history[-HISTORY:]),
        prefix="visual-symbolic-control",
        max_steps=VISUAL_MAX_STEPS,
    )
    masked_symbolic = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: masked_symbolic_model.choose(history[-HISTORY:]),
        prefix="visual-masked-symbolic-control",
        max_steps=VISUAL_MAX_STEPS,
    )
    random_rng = random.Random(seed + 2)
    random_baseline = evaluate_visual_policy(
        client,
        seeds=heldout_seeds,
        chooser=lambda _history: random_rng.choice(DIRECTIONS),
        prefix="visual-random",
        max_steps=VISUAL_MAX_STEPS,
    )
    reactive = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: history[-1].oracle_direction() or "north",
        prefix="visual-reactive",
        max_steps=VISUAL_MAX_STEPS,
    )
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    checkpoint = ARTIFACT_ROOT / "luanti-vision-policy.pt"
    torch.save({"state_dict": visual_model.state_dict(), "input_dim": VisionGoalDirectedPLWM.input_dim, "history_length": HISTORY}, checkpoint)
    artifact = ARTIFACT_ROOT / "luanti-vision-training-report.json"
    report = VisualPolicyReport(
        device=str(resolved_device),
        training_seed_count=training_seed_count,
        heldout_seed_count=heldout_seed_count,
        training_samples=len(labels),
        visual=visual,
        symbolic=symbolic,
        masked_symbolic=masked_symbolic,
        random=random_baseline,
        reactive=reactive,
        accepted=(
            visual.success_rate >= 0.9
            and visual.success_rate >= masked_symbolic.success_rate + 0.3
            and random_baseline.success_rate <= 0.2
            and reactive.success_rate == 1.0
        ),
        checkpoint=str(checkpoint),
        artifact=str(artifact),
    )
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def run(
    *,
    training_seed_count: int = 72,
    heldout_seed_count: int = 12,
    collection_steps: int = 12,
    training_steps: int = 700,
    seed: int = 113,
    device: str | torch.device | None = None,
    bridge_root: Path = BRIDGE_ROOT,
) -> Report:
    """Collect, train, and evaluate entirely through a running Luanti server."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    resolved_device = resolve_device(device)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    training_seeds = [1000 + index * 17 for index in range(training_seed_count)]
    heldout_seeds = [900_000 + index * 31 for index in range(heldout_seed_count)]
    exploration_trajectories = [
        collect_random_trajectory(
            client,
            task_seed=task_seed,
            episode_id=f"collect-{task_seed}",
            rng=rng,
            max_steps=collection_steps,
        )
        for task_seed in training_seeds
    ]
    teacher_trajectories = [
        collect_teacher_trajectory(
            client,
            task_seed=task_seed,
            episode_id=f"teacher-{task_seed}",
            max_steps=MAX_STEPS,
        )
        for task_seed in training_seeds
    ]
    observations, labels = training_tensors(teacher_trajectories)
    model = GoalDirectedPLWM(history_length=HISTORY).to(resolved_device)
    policy_loss, prediction_loss = train_policy(
        model,
        observations,
        labels,
        device=resolved_device,
        steps=training_steps,
        seed=seed,
    )
    no_memory_model = NoMemoryGoalPolicy().to(resolved_device)
    train_policy(
        no_memory_model,
        observations,
        labels,
        device=resolved_device,
        steps=training_steps,
        seed=seed + 101,
    )
    gru_context_model = GRUContextGoalPolicy(history_length=HISTORY).to(resolved_device)
    train_policy(
        gru_context_model,
        observations,
        labels,
        device=resolved_device,
        steps=training_steps,
        seed=seed + 102,
    )
    heldout_transitions = [
        collect_random_trajectory(
            client,
            task_seed=task_seed,
            episode_id=f"world-model-{task_seed}",
            rng=random.Random(seed + task_seed),
            max_steps=collection_steps,
        )
        for task_seed in heldout_seeds
    ]
    action_world_evaluation = evaluate_causal_control_model(heldout_transitions, CausalControlWorldModel)
    stateless_world_evaluation = evaluate_causal_control_model(heldout_transitions, StatelessControlBaseline)
    model.eval()
    learned = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: model.choose(history[-HISTORY:]),
        prefix="learned",
        max_steps=MAX_STEPS,
    )
    no_memory_model.eval()
    no_memory = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: no_memory_model.choose(history),
        prefix="no-memory",
        max_steps=MAX_STEPS,
    )
    gru_context_model.eval()
    gru_context = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: gru_context_model.choose(history[-HISTORY:]),
        prefix="gru-context",
        max_steps=MAX_STEPS,
    )
    random_baseline_rng = random.Random(seed + 1)
    random_baseline = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda _history: random_baseline_rng.choice(DIRECTIONS),
        prefix="random",
        max_steps=MAX_STEPS,
    )
    reactive = evaluate(
        client,
        seeds=heldout_seeds,
        chooser=lambda history: history[-1].oracle_direction() or "north",
        prefix="reactive",
        max_steps=MAX_STEPS,
    )

    ARTIFACT_ROOT.mkdir(exist_ok=True)
    rollout_path = ARTIFACT_ROOT / "luanti-rollouts.jsonl"
    all_trajectories = exploration_trajectories + teacher_trajectories
    rollout_path.write_text("".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in all_trajectories), encoding="utf-8")
    checkpoint_path = ARTIFACT_ROOT / "luanti-goal-policy.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "no_memory_state_dict": no_memory_model.state_dict(),
            "gru_context_state_dict": gru_context_model.state_dict(),
            "action_world_model_type": "causal-control-hypotheses-v1",
            "input_dim": GoalDirectedPLWM.input_dim,
            "history_length": HISTORY,
            "directions": DIRECTIONS,
            "training_seed_count": training_seed_count,
            "training_seed_first": training_seeds[0],
            "training_seed_last": training_seeds[-1],
        },
        checkpoint_path,
    )
    accepted = (
        learned.success_rate >= 0.9
        and learned.success_rate >= no_memory.success_rate + 0.3
        and learned.success_rate >= random_baseline.success_rate + 0.5
        and reactive.success_rate == 1.0
        and action_world_evaluation.displacement_accuracy >= 0.8
        and action_world_evaluation.blocked_accuracy >= 0.9
        and action_world_evaluation.displacement_accuracy >= stateless_world_evaluation.displacement_accuracy + 0.3
    )
    report = Report(
        device=str(resolved_device),
        training_seed_count=training_seed_count,
        heldout_seed_count=heldout_seed_count,
        trajectories=len(all_trajectories),
        training_samples=len(labels),
        final_policy_loss=policy_loss,
        final_prediction_loss=prediction_loss,
        action_world_model=action_world_evaluation,
        stateless_action_world_model=stateless_world_evaluation,
        learned=learned,
        no_memory=no_memory,
        gru_context=gru_context,
        random=random_baseline,
        reactive=reactive,
        accepted=accepted,
        checkpoint=str(checkpoint_path),
        rollout_artifact=str(rollout_path),
    )
    (ARTIFACT_ROOT / "luanti-training-report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("Luanti behavior training did not meet its evidence gate")
