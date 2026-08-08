"""Isolate reward-backed fact memory in the hidden-fact Luanti object world."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_behavior import Evaluation
from experiments.train_luanti_hidden_reward_memory import MemoryTrajectory, _read_trajectories, _reward_value, _teacher_direction
from plwm.device import resolve_device
from plwm.luanti import LuantiFileBridge
from plwm.luanti_training import CausalControlWorldModel, DIRECTIONS, LuantiLiveClient, ObjectTaskState, VerifiedObjectFactMemory


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = Path(r"D:\PLWM-Luanti\bridge")


def _features(state: ObjectTaskState, *, verified_key: bool) -> tuple[float, ...]:
    return state.memory_features() + (float(verified_key),)


class _CurrentPolicy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 48), nn.Tanh(), nn.Linear(48, len(DIRECTIONS)))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.network(observations[:, -1, :])
        surprise = torch.zeros((observations.shape[0], observations.shape[1]), dtype=observations.dtype, device=observations.device)
        return logits, surprise

    @torch.no_grad()
    def choose(self, features: Sequence[float]) -> str:
        device = next(self.parameters()).device
        logits, _ = self(torch.tensor(features, dtype=torch.float32, device=device).view(1, 1, -1))
        return DIRECTIONS[int(logits.argmax(dim=-1).item())]


def _tensors(trajectories: Sequence[MemoryTrajectory]) -> tuple[Tensor, Tensor, Tensor]:
    fact_rows: list[tuple[float, ...]] = []
    current_rows: list[tuple[float, ...]] = []
    labels: list[int] = []
    for trajectory in trajectories:
        states = [ObjectTaskState.from_payload(payload) for payload in trajectory.states]
        for index, direction in enumerate(trajectory.actual_labels):
            if direction is None:
                continue
            fact_rows.append(_features(states[index], verified_key=trajectory.verified_keys[index]))
            current_rows.append(states[index].memory_features())
            labels.append(DIRECTIONS.index(direction))
    if not labels:
        raise RuntimeError("no fact-memory labels")
    return torch.tensor(fact_rows).unsqueeze(1), torch.tensor(current_rows).unsqueeze(1), torch.tensor(labels, dtype=torch.long)


def _train(model: _CurrentPolicy, observations: Tensor, labels: Tensor, *, device: torch.device, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    batch = min(128, len(labels))
    for _ in range(1_000):
        indices = torch.randint(len(labels), (batch,), generator=generator)
        logits, surprise = model(observations[indices].to(device))
        loss = criterion(logits, labels[indices].to(device)) + 0.05 * surprise.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


def _evaluate(client: LuantiLiveClient, *, fact_policy: _CurrentPolicy, no_memory: _CurrentPolicy, seeds: Sequence[int], use_fact: bool) -> tuple[Evaluation, int]:
    successes, revisions = 0, 0
    steps_taken: list[int] = []
    for index, seed in enumerate(seeds):
        state = client.begin_object_episode(f"symbolic-fact-{'fact' if use_fact else 'current'}-{index}", seed)
        causal = CausalControlWorldModel()
        facts = VerifiedObjectFactMemory()
        for step in range(28):
            actual = (
                fact_policy.choose(_features(state, verified_key=facts.key_verified))
                if use_fact
                else no_memory.choose(state.memory_features())
            )
            requested = causal.request_for_actual(actual)
            before = state
            interaction, next_state = client.step_object(requested)
            if interaction.outcome.get("status") != "accepted" or next_state is None:
                break
            causal.observe(before.base, requested, next_state.base)
            if use_fact:
                facts.observe_reward(interaction.reward)
            state = next_state
            # Both policies encounter the same physical recenter; only the
            # fact policy carries its causally verified reward across it.
            if step == 0 or ((interaction.reward or {}).get("value") == 0.2):
                recenter, recentered_state = client.recenter_object()
                if recenter.outcome.get("status") != "accepted" or recentered_state is None:
                    break
                state = recentered_state
            if state.completed:
                successes += 1
                steps_taken.append(step + 1)
                break
        else:
            steps_taken.append(28)
        revisions += causal.revision_count
    return Evaluation(successes, len(seeds), sum(steps_taken) / len(seeds)), revisions


def run(*, device: str | torch.device | None = None, bridge_root: Path = BRIDGE_ROOT) -> dict[str, Any]:
    resolved = resolve_device(device)
    trajectories = _read_trajectories(ROOT / "artifacts" / "luanti-hidden-reward-memory-rollouts.jsonl")
    fact_rows, current_rows, labels = _tensors(trajectories)
    fact_policy = _CurrentPolicy(fact_rows.shape[-1]).to(resolved)
    current_policy = _CurrentPolicy(current_rows.shape[-1]).to(resolved)
    _train(fact_policy, fact_rows, labels, device=resolved, seed=1001)
    _train(current_policy, current_rows, labels, device=resolved, seed=1002)
    client = LuantiLiveClient(LuantiFileBridge(bridge_root), timeout_seconds=8.0)
    seeds = [2_800_000 + index * 31 for index in range(8)]
    fact, revisions = _evaluate(client, fact_policy=fact_policy, no_memory=current_policy, seeds=seeds, use_fact=True)
    current, _ = _evaluate(client, fact_policy=fact_policy, no_memory=current_policy, seeds=seeds, use_fact=False)
    checkpoint = ROOT / "artifacts" / "luanti-symbolic-fact-memory-policy.pt"
    torch.save({"fact_state_dict": fact_policy.state_dict(), "current_state_dict": current_policy.state_dict()}, checkpoint)
    artifact = ROOT / "artifacts" / "luanti-symbolic-fact-memory-report.json"
    report = {
        "device": str(resolved), "training_samples": len(labels), "fact_memory": asdict(fact),
        "current_frame": asdict(current), "control_revisions": revisions,
        "accepted": fact.success_rate >= 0.75 and fact.success_rate >= current.success_rate + 0.25 and revisions == len(seeds),
        "checkpoint": str(checkpoint), "artifact": str(artifact),
    }
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit("symbolic fact memory did not meet its evidence gate")

