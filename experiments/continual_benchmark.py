"""Five-stage continual-cognition benchmark for the PLWM prototype.

The benchmark intentionally reports comparable controllers rather than treating
the hand-designed persistent controller as a universal baseline.  All models
receive the same route cue and delayed distractor stream; only their allowed
state mechanisms differ.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plwm.continual import (
    BayesianPersistentController,
    EndToEndPersistentPLWM,
    EvidenceTask,
    FixedLanguagePort,
    LearnedPersistentRuntime,
    NoMemoryController,
    RecurrentPolicy,
    TaskAction,
    TransformerContextPolicy,
    WindowedContextController,
    batches,
)
from plwm.device import resolve_device


@dataclass
class ContinualBenchmarkResult:
    plwm_success: float
    plwm_mean_cost: float
    plwm_observation_rate: float
    no_memory_success: float
    windowed_context_success: float
    recurrent_success: float
    transformer_context_success: float
    end_to_end_plwm_success: float
    end_to_end_commitment_recovery: float
    stale_memory_correction: float
    long_delay_recovery: float
    language_port_is_stateless: bool
    commitment_survives_delay: bool


def _tasks(count: int, *, distractors: int, seed: int) -> list[EvidenceTask]:
    generator = torch.Generator().manual_seed(seed)
    return [EvidenceTask.sample(generator, distractor_count=distractors) for _ in range(count)]


def _control_targets(tasks: list[EvidenceTask]) -> torch.Tensor:
    # 0 means purchase a verification; 1/2 are direct left/right proposals.
    return torch.tensor([0 if item.cheap_reliability < 0.82 else item.route + 1 for item in tasks])


def _train(
    model: nn.Module,
    pre_observation: torch.Tensor,
    post_observation: torch.Tensor,
    route_targets: torch.Tensor,
    control_targets: torch.Tensor,
    *,
    end_to_end: bool,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    for _ in range(180):
        optimizer.zero_grad()
        if end_to_end:
            action_logits, _, _ = model(pre_observation)
            _, route_logits, _ = model(post_observation)
        else:
            action_logits, _ = model(pre_observation)
            _, route_logits = model(post_observation)
        loss = nn.functional.cross_entropy(action_logits, control_targets) + nn.functional.cross_entropy(route_logits, route_targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _accuracy(model: nn.Module, pre_observation: torch.Tensor, post_observation: torch.Tensor, y: torch.Tensor, *, end_to_end: bool) -> float:
    if end_to_end:
        actions, _, _ = model(pre_observation)
    else:
        actions, _ = model(pre_observation)
    proposals = actions.argmax(dim=-1)
    prediction = proposals - 1
    observe_mask = proposals == 0
    if observe_mask.any():
        if end_to_end:
            _, routes, _ = model(post_observation[observe_mask])
        else:
            _, routes = model(post_observation[observe_mask])
        prediction[observe_mask] = routes.argmax(dim=-1)
    return float((prediction == y).float().mean().item())


def run(*, seed: int = 73, device: str | torch.device | None = None) -> ContinualBenchmarkResult:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    device = resolve_device(device)
    train = _tasks(1024, distractors=2, seed=seed)
    evaluation = _tasks(512, distractors=10, seed=seed + 1)
    train_x, train_y = batches(train)
    train_verified_x, _ = batches(train, include_verification=True)
    eval_x, eval_y = batches(evaluation)
    eval_verified_x, _ = batches(evaluation, include_verification=True)
    train_control = _control_targets(train)
    train_x, train_y, train_verified_x, train_control = (
        train_x.to(device),
        train_y.to(device),
        train_verified_x.to(device),
        train_control.to(device),
    )
    eval_x, eval_y, eval_verified_x = eval_x.to(device), eval_y.to(device), eval_verified_x.to(device)

    recurrent = RecurrentPolicy().to(device)
    transformer = TransformerContextPolicy().to(device)
    learned_plwm = EndToEndPersistentPLWM().to(device)
    _train(recurrent, train_x, train_verified_x, train_y, train_control, end_to_end=False)
    _train(transformer, train_x, train_verified_x, train_y, train_control, end_to_end=False)
    _train(learned_plwm, train_x, train_verified_x, train_y, train_control, end_to_end=True)

    persistent = BayesianPersistentController()
    no_memory = NoMemoryController()
    windowed = WindowedContextController(window=3)
    persistent_results = [persistent.run(task) for task in evaluation]
    stale_results = [persistent.run(task, inject_stale_memory=True) for task in evaluation]
    long_delay_results = [
        persistent.run(
            EvidenceTask(
                route=item.route,
                cheap_signal=item.cheap_signal,
                verify_signal=item.verify_signal,
                distractors=(0,) * 100,
                cheap_reliability=item.cheap_reliability,
                verify_reliability=item.verify_reliability,
            )
        )
        for item in evaluation[:128]
    ]
    learned_runtime = LearnedPersistentRuntime(learned_plwm)
    learned_results = [learned_runtime.run(task) for task in evaluation]
    port = FixedLanguagePort()
    encoded, _ = port.encode_observation("route right")
    emitted_before = list(port.emitted)
    port.decode_action(TaskAction.RIGHT)

    return ContinualBenchmarkResult(
        plwm_success=sum(item.success for item in persistent_results) / len(persistent_results),
        plwm_mean_cost=sum(item.total_cost for item in persistent_results) / len(persistent_results),
        plwm_observation_rate=sum(item.observed for item in persistent_results) / len(persistent_results),
        no_memory_success=sum(no_memory.run(item).success for item in evaluation) / len(evaluation),
        windowed_context_success=sum(windowed.run(item).success for item in evaluation) / len(evaluation),
        recurrent_success=_accuracy(recurrent, eval_x, eval_verified_x, eval_y, end_to_end=False),
        transformer_context_success=_accuracy(transformer, eval_x, eval_verified_x, eval_y, end_to_end=False),
        end_to_end_plwm_success=_accuracy(learned_plwm, eval_x, eval_verified_x, eval_y, end_to_end=True),
        end_to_end_commitment_recovery=sum(item.commitment_completed and item.selected_before_delay == item.selected_after_delay for item in learned_results) / len(learned_results),
        stale_memory_correction=sum(item.corrected_stale_memory for item in stale_results) / len(stale_results),
        long_delay_recovery=sum(item.success and item.commitment_completed for item in long_delay_results) / len(long_delay_results),
        language_port_is_stateless=bool(encoded.any()) and emitted_before == [] and port.emitted == ["take right route"],
        commitment_survives_delay=all(
            item.selected_before_delay == "reach-correct-route" and item.selected_after_delay == "reach-correct-route"
            for item in persistent_results
        ),
    )


if __name__ == "__main__":
    print(asdict(run()))
