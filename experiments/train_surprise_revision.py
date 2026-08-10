"""JEPA 支柱最小闭环：surprise 驱动的记忆修订（预测误差 → 清槽重写）。

假设：记忆系统遇到"世界规则变了"时，应检测到变化（身份预测误差飙升）并
清空/重写记忆，让新映射快速写入；不清的话旧映射残留阻碍适应。

设计（监督版，机制验证）：
- 语料：DecideAfterDelay 合成流（身份→延迟→查询），label = 选择
- 阶段 1：映射 A（选择 = 身份）训练到收敛
- 阶段 2：映射 B（选择 = 补集），逐 batch 算身份预测误差（surprise）；
  - no_reset：什么都不做（旧映射残留）
  - surprise_reset：surprise 连续超过阈值 → 清空槽位（JEPA 检测驱动）
  - always_reset：每 batch 清空（无差别重写，对照）
- 指标：阶段 2 适应速度（query 精度恢复曲线）

预期：surprise_reset 快于 no_reset；若与 always_reset 相当，说明"选择性检测"
无增益（诚实记录）。
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
from plwm.jeplem_memory import JEPLMMemory
from experiments.audit_minigrid_decide_after_delay_scan import make_corpus


class SurpriseRevisionModel(nn.Module):
    """JEPLMMemory + identity predictor (surprise source) + choice head."""

    def __init__(self, input_dim: int, *, slots: int = 8, latent: int = 64) -> None:
        super().__init__()
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=2, aux_timescales=2)
        self.identity_head = nn.Linear(self.memory.head_in, 2)  # surprise via identity pred
        self.choice_head = nn.Linear(self.memory.head_in, 2)

    def forward(self, events: Tensor) -> tuple[Tensor, Tensor]:
        feats, _aux = self.memory.forward_sequence(events, torch.ones(events.shape[:2], dtype=torch.bool, device=events.device))
        final = feats[:, -1]
        return self.choice_head(final), self.identity_head(final)

    def reset_memory(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)


def make_rule_batch(seed: int, delay: int, mapping: int, n: int = 64, gap: int = 3) -> tuple[Tensor, Tensor, Tensor]:
    """DecideAfterDelay sequence [identity, empties, query]; label = choice under mapping."""
    seqs, idents, labels = [], [], []
    for _ in range(n):
        ident = random.Random(seed * 131 + _).randrange(2)
        events = torch.zeros(delay + 1, 2)
        events[0, ident] = 1.0
        events[delay, 0] = 1.0  # query
        seqs.append(events)
        idents.append(ident)
        labels.append(ident if mapping == 0 else 1 - ident)
    return torch.stack(seqs), torch.tensor(labels), torch.tensor(idents)


@torch.no_grad()
def identity_surprise(model: SurpriseRevisionModel, events: Tensor, idents: Tensor, device: torch.device) -> float:
    """Mean identity-prediction error over the batch (surprise signal)."""
    model.eval()
    _, ident_logits = model(events.to(device))
    err = nn.functional.cross_entropy(ident_logits, idents.to(device))
    model.train()
    return float(err)


def run(
    *, device: str | torch.device | None = None, delay: int = 8, gap: int = 3,
    stage1_updates: int = 300, stage2_updates: int = 100, batch_size: int = 64,
    surprise_threshold: float = 0.5, eval_every: int = 20,
    experiment_seed: int = 93_337, artifact: Path | None = None,
) -> dict:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    curves: dict[str, list[float]] = {}
    for mode in ("no_reset", "surprise_reset", "surprise_one_shot", "always_reset"):
        model = SurpriseRevisionModel(2).to(resolved)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        # stage 1: mapping A
        for u in range(stage1_updates):
            X, y, ids = make_rule_batch(u, delay, mapping=0, n=batch_size)
            X, y, ids = X.to(resolved), y.to(resolved), ids.to(resolved)
            choice, ident = model(X)
            loss = nn.functional.cross_entropy(choice, y) + 0.5 * nn.functional.cross_entropy(ident, ids)
            opt.zero_grad(); loss.backward(); opt.step()
        # stage 2: mapping B
        surprise_streak = 0
        one_shot_done = False
        for u in range(stage2_updates):
            X, y, ids = make_rule_batch(10_000 + u, delay, mapping=1, n=batch_size)
            X, y, ids = X.to(resolved), y.to(resolved), ids.to(resolved)
            if (mode in ("surprise_reset", "surprise_one_shot")) and u % eval_every == 0:
                if mode == "surprise_one_shot" and one_shot_done:
                    pass
                else:
                    s = identity_surprise(model, X, ids, resolved)
                    if s > surprise_threshold:
                        surprise_streak += 1
                    else:
                        surprise_streak = 0
                    if surprise_streak >= 2:
                        model.reset_memory(X.shape[0], resolved)
                        surprise_streak = 0
                        if mode == "surprise_one_shot":
                            one_shot_done = True
            if mode == "always_reset":
                model.reset_memory(X.shape[0], resolved)
            choice, ident = model(X)
            loss = nn.functional.cross_entropy(choice, y) + 0.5 * nn.functional.cross_entropy(ident, ids)
            opt.zero_grad(); loss.backward(); opt.step()
            if (u + 1) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    acc = float((model(X)[0].argmax(-1) == y).float().mean())
                model.train()
                curves.setdefault(mode, []).append(acc)
    result = {"protocol": "jeplem_surprise_revision_v1", "device": str(resolved),
              "delay": delay, "surprise_threshold": surprise_threshold,
              "stage1_updates": stage1_updates, "stage2_updates": stage2_updates,
              "adaptation_curves": curves}
    target = artifact or ROOT / "artifacts" / "surprise-revision-report.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--stage1-updates", type=int, default=300)
    parser.add_argument("--stage2-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--surprise-threshold", type=float, default=0.5)
    parser.add_argument("--experiment-seed", type=int, default=93_337)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    out = run(device=args.device, delay=args.delay, stage1_updates=args.stage1_updates,
              stage2_updates=args.stage2_updates, batch_size=args.batch_size,
              surprise_threshold=args.surprise_threshold,
              experiment_seed=args.experiment_seed, artifact=args.artifact)
    print(json.dumps(out, indent=2))
