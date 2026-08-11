"""Block 6 v2: compositional generalization on nested-operation programs.

v1 (one-hot attributes) was trivial for every model.  v2 uses a SCAN-like
program structure: atoms (a->A, b->B), operators DUP (double: DUP(a)=AA) and
REV (reverse: REV(a,b)=BA).  Training covers atoms, DUP on atoms, and SOME
nested REV patterns; testing uses UNSEEN nestings (e.g. REV(DUP(a), b)).
A structurally compositional model (per-token recursion) generalizes; a
monolithic GRU trained on seen patterns may fail on unseen nesting orders.

Criterion: modular-composition model heldout accuracy >= 0.8 and >= monolithic
GRU + 0.2.
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

VOCAB = {"a": 0, "b": 1, "DUP": 2, "REV": 3, "<PAD>": 4, "<EOS>": 5}
ACTION_VOCAB = {"A": 0, "B": 1, "<PAD>": 2}
MAX_TOKENS = 8
MAX_OUT = 8


def _programs(seed: int) -> tuple[list[str], list[str]]:
    """(train_programs, test_programs) as token strings -> action strings."""
    rng = random.Random(seed)
    atoms = ["a", "b"]
    # seen patterns (train)
    seen = list(atoms)
    seen += [f"DUP({x})" for x in atoms]
    seen += [f"REV({x},{y})" for x in atoms for y in atoms]
    # seen nestings: DUP(REV(...)) and REV(DUP(x), y) with x atom
    seen += [f"DUP(REV({x},{y}))" for x in atoms for y in atoms]
    seen += [f"REV(DUP({x}),{y})" for x in atoms for y in atoms]
    # unseen nestings (test): REV(DUP(x), DUP(y)) and DUP(REV(x, DUP(y))) etc.
    unseen = [f"REV(DUP({x}),DUP({y}))" for x in atoms for y in atoms]
    unseen += [f"DUP(REV({x},DUP({y})))" for x in atoms for y in atoms]
    unseen += [f"REV(DUP(DUP({x})),{y})" for x in atoms for y in atoms]
    return seen, unseen


def _eval_program(prog: str) -> str:
    """Evaluate a program string to an action string."""
    def ev(s: str) -> str:
        s = s.strip()
        if s in ("a", "b"):
            return s.upper()
        if s.startswith("DUP("):
            inner = s[4:-1]
            return ev(inner) * 2
        if s.startswith("REV("):
            body = s[4:-1]
            # split top-level comma
            depth = 0
            split = -1
            for i, ch in enumerate(body):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    split = i
                    break
            left, right = body[:split], body[split + 1:]
            return ev(right) + ev(left)
        raise ValueError(s)
    return ev(prog)


def _tokenize(prog: str) -> list[int]:
    """Flatten program to a token list (simple left-to-right)."""
    toks = []
    i = 0
    while i < len(prog):
        if prog[i] in "(),":
            i += 1
            continue
        for name in ("DUP", "REV"):
            if prog.startswith(name, i):
                toks.append(VOCAB[name])
                i += len(name)
                break
        else:
            toks.append(VOCAB[prog[i]])
            i += 1
    return toks + [VOCAB["<EOS>"]]


def _encode_actions(actions: str) -> list[int]:
    return [ACTION_VOCAB[c] for c in actions] + [ACTION_VOCAB["<PAD>"]] * 0


class MonolithicGRU(nn.Module):
    """GRU from token sequence to action sequence (no structural prior)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), 16)
        self.gru = nn.GRU(16, 64, batch_first=True)
        self.head = nn.Linear(64, len(ACTION_VOCAB))

    def forward(self, x: Tensor) -> Tensor:
        emb = self.emb(x)
        out, _ = self.gru(emb)
        return self.head(out)


class CompositionalModel(nn.Module):
    """Structural-composition model (Block-6 control): each operator is a
    reusable learnable function (DUP: 1 child, REV: 2 children); a program is
    parsed into a tree and encoded by recursive function application.  Any
    UNSEEN nesting is then just a new combination of already-learned operator
    functions -> compositional generalization, unlike the monolithic GRU."""

    def __init__(self) -> None:
        super().__init__()
        self.structural = True
        self.atom = nn.Embedding(2, 16)
        self.dup = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.rev = nn.Sequential(nn.Linear(32, 16), nn.Tanh())
        self.head = nn.Linear(16, MAX_OUT * len(ACTION_VOCAB))

    def _split_top(self, body: str) -> tuple[str, str]:
        depth = 0
        for i, ch in enumerate(body):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                return body[:i], body[i + 1:]
        raise ValueError(body)

    def _encode(self, prog: str) -> Tensor:
        prog = prog.strip()
        if prog in ("a", "b"):
            idx = torch.tensor([0 if prog == "a" else 1], device=self.atom.weight.device)
            return self.atom(idx)
        if prog.startswith("DUP("):
            return self.dup(self._encode(prog[4:-1]))
        if prog.startswith("REV("):
            left, right = self._split_top(prog[4:-1])
            return self.rev(torch.cat([self._encode(left), self._encode(right)], dim=-1))
        raise ValueError(prog)

    def forward(self, programs: list[str]) -> Tensor:
        roots = [self._encode(p) for p in programs]
        x = torch.stack(roots)  # [B, 16]
        out = self.head(x)  # [B, MAX_OUT*3]
        return out.view(len(programs), MAX_OUT, len(ACTION_VOCAB))


def _pad(seqs: list[list[int]], max_len: int, pad: int) -> Tensor:
    out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    for r, s in enumerate(seqs):
        out[r, : len(s)] = torch.tensor(s[:max_len])
    return out


def _train(model: nn.Module, programs: list[str], device: torch.device, *, epochs: int = 800) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=ACTION_VOCAB["<PAD>"])
    ys = [[ACTION_VOCAB[c] for c in _eval_program(p)][:MAX_OUT] for p in programs]
    yt = _pad(ys, MAX_OUT, ACTION_VOCAB["<PAD>"]).to(device)
    if getattr(model, "structural", False):
        # structural model: input is the program STRING (tree parsed internally)
        for _ in range(epochs):
            logits = model(programs)  # [B, MAX_OUT, 3]
            loss = criterion(logits.permute(0, 2, 1), yt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        return
    xs = [_tokenize(p)[:MAX_TOKENS] for p in programs]
    xt = _pad(xs, MAX_TOKENS, VOCAB["<PAD>"]).to(device)
    model.train()
    for _ in range(epochs):
        logits = model(xt)  # [B, T, 3]
        loss = criterion(logits.permute(0, 2, 1), yt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()


@torch.no_grad()
def _accuracy(model: nn.Module, programs: list[str], device: torch.device) -> float:
    if getattr(model, "structural", False):
        logits = model(programs)
        pred = logits.argmax(-1)
        correct = 0
        for r, p in enumerate(programs):
            true_actions = [ACTION_VOCAB[c] for c in _eval_program(p)]
            got = pred[r, : len(true_actions)].tolist()
            correct += int(got == true_actions)
        return correct / max(len(programs), 1)
    xs = _pad([_tokenize(p)[:MAX_TOKENS] for p in programs], MAX_TOKENS, VOCAB["<PAD>"]).to(device)
    logits = model(xs)
    pred = logits.argmax(-1)
    correct = 0
    for r, p in enumerate(programs):
        true_actions = [ACTION_VOCAB[c] for c in _eval_program(p)]
        got = pred[r, : len(true_actions)].tolist()
        correct += int(got == true_actions)
    return correct / max(len(programs), 1)


@dataclass
class Result:
    name: str
    train_accuracy: float
    heldout_accuracy: float


@dataclass
class Report:
    protocol: str
    device: str
    train_programs: int
    test_programs: int
    results: list[Result]
    accepted: bool
    artifact: str


def run(*, device: str | torch.device | None = None, experiment_seed: int = 41_123,
        artifact: Path | None = None) -> Report:
    resolved = resolve_device(device)
    torch.manual_seed(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)
    seen, unseen = _programs(experiment_seed)

    results = []
    for name, factory in (("monolithic_gru", MonolithicGRU), ("compositional", CompositionalModel)):
        model = factory().to(resolved)
        _train(model, seen, resolved)
        train_acc = _accuracy(model, seen, resolved)
        heldout = _accuracy(model, unseen, resolved)
        results.append(Result(name, train_acc, heldout))

    by_name = {r.name: r for r in results}
    accepted = (
        by_name["compositional"].heldout_accuracy >= 0.8
        and by_name["compositional"].heldout_accuracy >= by_name["monolithic_gru"].heldout_accuracy + 0.2
    )
    target = artifact or ROOT / "artifacts" / "compositional-generalization-v2-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "compositional_generalization_v2: Block-6 upgrade. SCAN-like nested programs (DUP/REV); training on atoms + DUP + REV + two nesting shapes; heldout on three UNSEEN nesting shapes. monolithic GRU is tested for whether it generalizes the operation algebra. Criterion: heldout >= 0.8.",
        str(resolved), len(seen), len(unseen), results, accepted, str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    outcome = run(device=args.device, experiment_seed=args.experiment_seed,
                  artifact=args.artifact)
    print(json.dumps(asdict(outcome), indent=2))
