"""Phase-2 addressing probe (2026-08-12): does JEPLMMemory read do precise
multi-slot content addressing under supervised learning?

Background: DecideStream only ever asked about the most recent identity, so the
0.94 end-to-end number never tested 'from N slots, recall slot X by query'.
The 9.1 diagnosis proved the aux head learns under Bayesian supervision (0.96),
but that supervised the ANSWER path on K=2.  This probe tests the READ path
directly at larger K, under SUPERVISED training (the regime where 9.1 showed
the memory CAN learn precise addressing -- RL end-to-end was shown to fail on
the request chain, integration design 9.3).

Setup (per episode):
  - K identities, each with a hidden binary value v[i].
  - Each identity is written into a slot via a clue frame: obs = onehot(value
    v[i]) ++ onehot(identity i) ++ noise.  Clue content is the value corrupted
    with noise at p=reliability (the real EvidenceSeek noise model).
  - Query frame: asked identity X is one-hot in the goal channel.  The memory
    reads and the head must output the BAYESIAN posterior argmax of v[X] given
    the clues seen so far (NOT the ground truth -- noisy clues make ground
    truth un-supervisable, integration 9.1).
  - Loss: cross-entropy on the query frame's answer logits (the head's first 2
    logits) against the Bayes target.

Control: explicit_key_memory (structurally-aligned slots: slot i <-> identity i)
vs JEPLMMemory (content-addressed).  If content addressing is precise, JEPLM
matches explicit_key; if it degrades with K, the addressing pillar has a real
gap that DecideStream never exposed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.jeplem_memory import JEPLMMemory


def bayes_target(clue_values: list[int], reliability: float) -> int | None:
    """Bayesian posterior argmax of an identity's value given its noisy clues
    (uniform prior; clues are P(value) with reliability).  Returns None if no
    clues (un-supervisable)."""
    if not clue_values:
        return None
    p = reliability
    log_pos = 0.0  # P(value=0)
    log_neg = 0.0  # P(value=1)
    for c in clue_values:
        # P(clue c | value): c==0 consistent with value 0
        if c == 0:
            log_pos += np.log(p)
            log_neg += np.log(1.0 - p)
        else:
            log_pos += np.log(1.0 - p)
            log_neg += np.log(p)
    prob0 = 1.0 / (1.0 + np.exp(log_neg - log_pos))
    return 0 if prob0 >= 0.5 else 1


class ExplicitKeyMemory(nn.Module):
    """Structurally-aligned control: slot i <-> identity i, no content
    addressing.  The upper bound for addressing precision."""

    def __init__(self, k: int, latent: int = 32) -> None:
        super().__init__()
        self.k = k
        self.latent = latent
        self.value = nn.Linear(latent, latent)
        self.head = nn.Linear(latent, 2)
        self._acc: torch.Tensor | None = None

    def reset(self, batch: int, device: torch.device) -> None:
        self._acc = torch.zeros(batch, self.k, device=device)

    def step(self, obs: torch.Tensor, valid: torch.Tensor, clue_identity: torch.Tensor | None,
             clue_value: torch.Tensor | None, goal: torch.Tensor | None) -> torch.Tensor:
        """obs: [B, 2k+2]; clue_identity one-hot [B,k]; goal one-hot [B,k]."""
        if clue_identity is not None and clue_value is not None:
            # accumulate the clue value into slot i with ±latent energy
            contrib = torch.where(clue_value.unsqueeze(-1) == 0, -1.0, 1.0)
            self._acc = self._acc + clue_identity * contrib.squeeze(-1)
        # read: on query frames, the asked identity's accumulator
        if goal is not None:
            asked_acc = (self._acc * goal).sum(dim=-1, keepdim=True)  # [B,1]
        else:
            asked_acc = torch.zeros(obs.shape[0], 1, device=obs.device)
        return self.head(asked_acc)


def make_env_batch(k: int, reliability: float, n: int, seed0: int) -> list[dict]:
    """Generate n episodes: each writes every identity once (with noisy clue),
    then queries each identity once.  Returns list of {obs, valid, clue_i,
    clue_v, goal, target} tensors."""
    episodes = []
    for ep in range(n):
        rng = np.random.default_rng(seed0 + ep)
        values = rng.integers(0, 2, size=k)
        obs_all: list[np.ndarray] = []
        valid_all: list[bool] = []
        clue_i_all: list[np.ndarray] = []
        clue_v_all: list[np.ndarray] = []
        goal_all: list[np.ndarray] = []
        target_all: list[int | None] = []
        clues_seen: dict[int, list[int]] = {}
        # write each identity once (clue frame), then query each identity
        order = np.arange(k)
        rng.shuffle(order)
        for i in order:
            obs = np.zeros(2 * k + 2, dtype=np.float32)
            obs[values[i]] = 1.0  # value one-hot
            obs[2 + i] = 1.0  # identity index one-hot
            obs_all.append(obs)
            valid_all.append(True)
            clue_i = np.zeros(k, dtype=np.float32); clue_i[i] = 1.0
            content = values[i] if rng.random() < reliability else (1 - values[i])
            clue_v = np.array([content], dtype=np.float32)
            clue_i_all.append(clue_i); clue_v_all.append(clue_v)
            goal = np.zeros(k, dtype=np.float32)
            goal_all.append(goal)
            target_all.append(None)
            clues_seen.setdefault(i, []).append(content)
        for i in order:
            # query frame: empty obs, goal = asked identity
            obs = np.zeros(2 * k + 2, dtype=np.float32)
            obs_all.append(obs); valid_all.append(False)
            clue_i = np.zeros(k, dtype=np.float32); clue_v = np.array([0.0], dtype=np.float32)
            clue_i_all.append(clue_i); clue_v_all.append(clue_v)
            goal = np.zeros(k, dtype=np.float32); goal[i] = 1.0
            goal_all.append(goal)
            tgt = bayes_target(clues_seen[i], reliability)
            target_all.append(tgt)
        episodes.append({
            "obs": torch.tensor(np.stack(obs_all)),
            "valid": torch.tensor(np.array(valid_all, dtype=bool)),
            "clue_i": torch.tensor(np.stack(clue_i_all)),
            "clue_v": torch.tensor(np.stack(clue_v_all)),
            "goal": torch.tensor(np.stack(goal_all)),
            "target": target_all,
        })
    return episodes


class JEPLMProbe(nn.Module):
    def __init__(self, input_dim: int, k: int, latent: int = 32, slots: int = 16,
                 goal_dim: int = 0) -> None:
        super().__init__()
        self.memory = JEPLMMemory(input_dim, latent=latent, slots=slots,
                                  identity_count=2, aux_timescales=0,
                                  write_mode="fifo", goal_dim=goal_dim)
        self.head = nn.Linear(self.memory.head_in, 2)

    def reset(self, batch: int, device: torch.device) -> None:
        self.memory.reset(batch, device)

    def step(self, obs: torch.Tensor, valid: torch.Tensor,
             goal: torch.Tensor | None) -> torch.Tensor:
        feats, _aux = self.memory.step(obs, valid, goal=goal)
        return self.head(feats)


def run(*, k: int, reliability: float, latent: int, slots: int, goal: bool,
        train_steps: int, batch: int, seed: int, device: torch.device,
        artifact: Path | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_dim = 2 * k + 2
    goal_dim = k if goal else 0
    model = JEPLMProbe(input_dim, k, latent=latent, slots=slots, goal_dim=goal_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(train_steps):
        episodes = make_env_batch(k, reliability, batch, seed * 1000 + step * batch)
        opt.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)
        total_cnt = 0
        for ep in episodes:
            obs = ep["obs"].unsqueeze(0).to(device)
            valid = ep["valid"].unsqueeze(0).to(device)
            goal_t = ep["goal"].unsqueeze(0).to(device) if goal else None
            model.reset(1, device)
            for f in range(obs.shape[1]):
                goal_slice = goal_t[:, f] if goal else None
                logits = model.step(obs[:, f], valid[:, f], goal_slice)
                tgt = ep["target"][f]
                if tgt is not None:
                    total_loss = total_loss + F.cross_entropy(
                        logits[:, :2], torch.tensor([tgt], device=device))
                    total_cnt += 1
        if total_cnt:
            (total_loss / total_cnt).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    # evaluate on held-out episodes
    model.eval()
    ev = make_env_batch(k, reliability, 64, seed * 999 + 777_777)
    correct = total = 0
    with torch.no_grad():
        for ep in ev:
            obs = ep["obs"].unsqueeze(0).to(device)
            valid = ep["valid"].unsqueeze(0).to(device)
            goal_t = ep["goal"].unsqueeze(0).to(device) if goal else None
            model.reset(1, device)
            for f in range(obs.shape[1]):
                goal_slice = goal_t[:, f] if goal else None
                logits = model.step(obs[:, f], valid[:, f], goal_slice)
                tgt = ep["target"][f]
                if tgt is not None:
                    correct += int(logits[0, :2].argmax(-1).item() == tgt)
                    total += 1
    acc = correct / total if total else 0.0
    report = {
        "protocol": f"phase2_addressing_probe: JEPLMMemory supervised read-path probe at K={k} (reliability {reliability}, slots {slots}, goal={'on' if goal else 'off'}). Bayes-posterior-supervised answer logits on query frames. Control: random-init + supervised; explicit_key is the structural upper bound.",
        "seed": seed,
        "results": {"K": k, "goal_modulated": goal, "supervised_acc": round(acc, 3),
                    "correct": correct, "total": total},
    }
    (artifact or ROOT / "artifacts" / "addressing-probe.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "addressing-probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--reliability", type=float, default=0.9)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--goal", action="store_true")
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--experiment-seed", type=int, default=41_123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(json.dumps(run(k=args.k, reliability=args.reliability, latent=args.latent,
                         slots=args.slots, goal=args.goal, train_steps=args.train_steps,
                         batch=args.batch, seed=args.experiment_seed, device=device,
                         artifact=args.artifact), indent=2))
