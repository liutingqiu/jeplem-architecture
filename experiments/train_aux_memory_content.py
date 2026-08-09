"""JEPLM auxiliary-target repositioning: does explicit "memory-content
prediction" (predicting the remembered identity at every frame after the
identity frame) help a controller USE memory for the decision?

Thesis context: the SPR-style multi-step latent auxiliary was a burden under
RL (DecideAfterDelay §6) and useless on MemoryS7 (JEPLM round 1/2).  The
repositioning is to test the auxiliary as *memory-content prediction* on a
controlled supervised task where the main loss is a clean memory->decision
demand: real-stream single-identity delayed recall (v2 corpus).

Design:
  base      : GRU / evidence, decision head only (control)
  +aux      : same + a head that predicts the identity one-hot at every frame
              after the identity frame (aux = memory content carried forward)
  metric    : heldout decision accuracy + swap-counterfactual evidence
  hypothesis: if explicit memory-content prediction helps, +aux > base under
              equal budget; if it is a distraction (RL diagnosis), +aux == base.
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
from experiments.audit_minigrid_delayed_object_identity import _corpus as _real_corpus
from plwm.minigrid_features import visible_object_presence


class AuxGRUDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, width: int = 64, alpha: float = 0.5) -> None:
        super().__init__()
        self.core = nn.GRU(input_dim, width, batch_first=True)
        self.head = nn.Linear(width, num_classes)
        self.aux = nn.Linear(width, 2)  # identity one-hot predictor
        self.alpha = alpha

    def forward(self, events: Tensor) -> tuple[Tensor, Tensor]:
        outputs, _state = self.core(events)
        return self.head(outputs[:, -1]), self.aux(outputs[:, :-1])


class AuxEvidenceDecision(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, width: int = 64, slots: int = 8, alpha: float = 0.5) -> None:
        super().__init__()
        self.width = width
        self.slots = slots
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)
        self.write_gate = nn.Linear(width, 1)
        self.state = nn.GRUCell(width * 2, width)
        self.head = nn.Linear(width, num_classes)
        self.aux = nn.Linear(width, 2)
        self.alpha = alpha

    def forward(self, events: Tensor) -> tuple[Tensor, Tensor]:
        batch = events.shape[0]
        state = torch.zeros(batch, self.width, device=events.device)
        keys = torch.zeros(batch, self.slots, self.width, device=events.device)
        values = torch.zeros_like(keys)
        states: list[Tensor] = []
        for index in range(events.shape[1]):
            encoded = self.encoder(events[:, index])
            read_weights = torch.softmax(torch.einsum("bd,bmd->bm", self.query(encoded), keys) / self.width**0.5, dim=-1)
            recalled = torch.einsum("bm,bmd->bd", read_weights, values)
            state = self.state(torch.cat((encoded, recalled), dim=-1), state)
            gate = torch.sigmoid(self.write_gate(encoded)).view(batch, 1, 1)
            candidate_keys = torch.cat((keys[:, 1:], self.key(encoded).unsqueeze(1)), dim=1)
            candidate_values = torch.cat((values[:, 1:], self.value(encoded).unsqueeze(1)), dim=1)
            keys = keys * (1.0 - gate) + candidate_keys * gate
            values = values * (1.0 - gate) + candidate_values * gate
            states.append(state)
        stacked = torch.stack(states, dim=1)
        return self.head(stacked[:, -1]), self.aux(stacked[:, :-1])


@dataclass
class Result:
    name: str
    parameters: int
    heldout_accuracy: float
    swapped_accuracy: float
    memory_evidence: float
    aux_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    corpus: str
    training_examples: int
    updates: int
    alpha: float
    results: list[Result]
    accepted: bool
    artifact: str


def _identity_onehot(labels: Tensor) -> Tensor:
    return torch.eye(2)[labels]


def _train(model: nn.Module, events: Tensor, labels: Tensor, *, updates: int, batch_size: int, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(92_337)
    model.train()
    identity = _identity_onehot(labels)
    for _update in range(updates):
        indices = torch.randint(events.shape[0], (batch_size,), generator=generator)
        batch_events = events[indices].to(device)
        decision_logits, aux_logits = model(batch_events)
        main_loss = nn.functional.cross_entropy(decision_logits, labels[indices].to(device))
        aux_target = identity[indices].unsqueeze(1).expand(-1, aux_logits.shape[1], -1).to(device)
        aux_loss = nn.functional.cross_entropy(
            aux_logits.reshape(-1, 2), aux_target.reshape(-1, 2).argmax(-1))
        loss = main_loss + model.alpha * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _evaluate(model: nn.Module, events: Tensor, labels: Tensor, swapped_labels: Tensor, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    decision_logits, aux_logits = model(events.to(device))
    decision = decision_logits.argmax(-1).cpu()
    heldout = float((decision == labels).float().mean())
    swapped = float((decision == swapped_labels).float().mean())
    aux_target = _identity_onehot(labels).argmax(-1)  # [B]
    aux_pred = aux_logits.argmax(-1).cpu()  # [B, T-1]
    aux_acc = float((aux_pred == aux_target.unsqueeze(1)).all(dim=-1).float().mean())
    return heldout, swapped, aux_acc


def run(
    *, device: str | torch.device | None = None, corpus: str = "real",
    training_examples: int = 512, updates: int = 300, batch_size: int = 32,
    alpha: float = 0.5, delay: int = 8, artifact: Path | None = None,
) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(93_337)
    np.random.seed(93_337)
    random.seed(93_337)
    if corpus == "real":
        train_events, train_labels = _real_corpus(examples=training_examples, seed_start=50_000, sequence_length=delay + 1)
        heldout_events, heldout_labels = _real_corpus(examples=256, seed_start=6_000_000, sequence_length=delay + 1)
        train_events = visible_object_presence(train_events)
        heldout_events = visible_object_presence(heldout_events)
    else:
        raise ValueError("only the real-stream corpus is supported in this experiment")
    input_dim = train_events.shape[-1]
    factories = (
        ("gru_base", lambda: AuxGRUDecision(input_dim, 2, alpha=0.0)),
        ("gru_aux", lambda: AuxGRUDecision(input_dim, 2, alpha=alpha)),
        ("evidence_base", lambda: AuxEvidenceDecision(input_dim, 2, alpha=0.0)),
        ("evidence_aux", lambda: AuxEvidenceDecision(input_dim, 2, alpha=alpha)),
    )
    swapped_labels = 1 - heldout_labels
    results = []
    for name, factory in factories:
        model = factory().to(resolved)
        _train(model, train_events, train_labels, updates=updates, batch_size=batch_size, device=resolved)
        heldout, swapped, aux_acc = _evaluate(model, heldout_events, heldout_labels, swapped_labels, resolved)
        results.append(Result(name, sum(parameter.numel() for parameter in model.parameters()),
                              heldout, swapped, heldout - swapped, aux_acc))
    by_name = {r.name: r for r in results}
    # gate: the repositioning hypothesis is that memory-content prediction
    # HELPS; require both controllers to improve by >= 0.02 under equal budget.
    accepted = (by_name["gru_aux"].heldout_accuracy >= by_name["gru_base"].heldout_accuracy + 0.02
                and by_name["evidence_aux"].heldout_accuracy >= by_name["evidence_base"].heldout_accuracy + 0.02)
    target = artifact or ROOT / "artifacts" / "aux-memory-content-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "jeplem_aux_memory_content_v1: real/synthetic v2 delayed-identity corpus; base controllers (GRU/evidence) vs +aux head predicting the remembered identity at every post-identity frame (memory-content prediction); alpha=0 means aux loss disabled; heldout decision accuracy + swap-counterfactual evidence; tests whether explicit memory-content prediction helps memory-use under equal budget.",
        str(resolved), corpus, training_examples, updates, alpha, results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus", default="real", choices=("real", "synthetic"))
    parser.add_argument("--training-examples", type=int, default=512)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, corpus=args.corpus, training_examples=args.training_examples,
                  updates=args.updates, batch_size=args.batch_size, alpha=args.alpha,
                  delay=args.delay, artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
