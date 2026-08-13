"""The unified complete body: all verified mechanisms coexist in one event loop.

Mechanisms hosted (each verified separately in the phase-2 body series v9-v16):
  - knowledge ledger (VersionedFactLedger) with executive-controlled writes
  - per-state Bayesian belief
  - value anchoring (Block-10) -> value-ordered goal selection
  - gap goals (P2) -> belief-uncertainty state selection
  - world-model prediction head (Block-7) + prediction-error exploration signal
  - event-level revision (Block-1, sustained contradiction detector)
  - snapshot chain (P5)
  - executive control (Block-9): write budget + confidence gate
  - compositional skills (P6 hybrid): symbolic order skeleton + learned atoms
  - social cognition (Block-11): model of another agent's behavior
  - meta-controller (A2): route queries between memory and belief on internal state
  - self-model (Block-8): predict the body's own belief-answer correctness

The body never sees ground truth: it only receives noisy ``shown`` values and
a per-observation reliability.  It is perception-agnostic: identity events can
come from a frozen CNN on real frames (v15 front) or a symbolic stream.
"""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .belief import CategoricalBelief
from .fact_ledger import FactRecord, VersionedFactLedger


class PredictionHead(nn.Module):
    """Learns a domain's transition law s' = (s + k) mod N."""

    def __init__(self, n_states: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_states, 16), nn.ReLU(), nn.Linear(16, n_states))

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        return self.net(onehot)


class SkillNet(nn.Module):
    """One atomic skill: input features -> {none, skill}.  Learned from single
    examples; composed by the symbolic order skeleton (P6 hybrid)."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaRouter(nn.Module):
    """Routes a query answer to memory-ledger (0) or belief (1) from
    body-internal state features."""

    def __init__(self, n_feat: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SelfHead(nn.Module):
    """Predicts whether the body's ROUTED answer is correct from body-internal
    features + the routing decision + the routed source's confidence; beats the
    source's own confidence (naive), which cannot capture routing errors."""

    def __init__(self, n_feat: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def compute_ece(preds: np.ndarray, corrects: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        mask = (preds > edges[b]) & (preds <= edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(preds)) * abs(float(preds[mask].mean()) - float(corrects[mask].mean()))
    return float(ece)


class UnifiedBody:
    """All mechanisms in one body.  Identity observations arrive via
    ``observe_identity``; composite skill inputs via ``observe_composite``;
    the other agent's actions via ``observe_other_action``; queries are
    answered from the ledger (``query``) and routed post-hoc by the
    meta-controller."""

    def __init__(
        self,
        *,
        seed: int,
        values: dict[int, int],
        n_states: int,
        write_budget: int,
        write_gate: float,
        belief_lik_p: float = 0.52,
        belief_lik_noisy: float = 0.58,
        mark_belief_conf: float = 0.7,
        revision_confidence: float = 0.55,
        skill_n_feat: int = 8,
        laws: dict[int, int] | None = None,
        noisy_states: set[int] | frozenset = frozenset(),
    ) -> None:
        self.seed = seed
        self.values = values
        self.n_states = n_states
        self.write_budget = write_budget
        self.write_gate = write_gate
        self.belief_lik_p = belief_lik_p
        self.mark_belief_conf = mark_belief_conf
        self.laws = laws or {}   # world transition s' = (s + k) mod n per domain
        self.noisy_states = frozenset(noisy_states)
        # clean states use a weak calibration (slow re-convergence -> long
        # flip-lag memory-wins); noisy states use a matched calibration so the
        # belief converges and the gap-driven selection is not monopolised.
        self.lik_noisy = torch.tensor([[belief_lik_noisy, 1.0 - belief_lik_noisy],
                                       [1.0 - belief_lik_noisy, belief_lik_noisy]]).log()

        self.ledger = VersionedFactLedger(trusted_sources={"body"}, revision_confidence=revision_confidence)
        self.tick = 0
        self.chain: list[dict] = []
        self.beliefs: dict[tuple[int, int], CategoricalBelief] = {}
        self.marked: set[int] = set()
        self.conflict: dict[tuple[int, int], int] = {}
        self.obs_count: dict[tuple[int, int], int] = {}
        self.budget_spent = 0
        self.lik = torch.tensor([[belief_lik_p, 1.0 - belief_lik_p],
                                 [1.0 - belief_lik_p, belief_lik_p]]).log()
        # executive-controlled WORKING memory (Block-9): a bounded, gated
        # vote memory distinct from the durable fact ledger -- the ledger is
        # the P1/P5/routing substrate (records all observations), attention
        # is the budgeted store used for fast queries (executive verdict).
        self.attention: dict[int, list[int]] = {}
        self.attention_budget = write_budget
        self.attention_gate = write_gate
        self.attention_spent = 0
        self._attention_ts: dict[int, int] = {}

        # world-model prediction heads per domain
        self.pred_heads: dict[int, PredictionHead] = {d: PredictionHead(n_states) for d in values}
        self.pred_opts: dict[int, torch.optim.Optimizer] = {
            d: torch.optim.AdamW(self.pred_heads[d].parameters(), lr=1e-2) for d in values
        }
        self.prev_state: dict[int, int] = {}
        self.pred_correct = 0
        self.pred_total = 0

        # compositional skills (learned by the caller from single-pattern
        # examples; the order skeleton is symbolic)
        self.skills: dict[str, SkillNet] = {"a": SkillNet(skill_n_feat), "b": SkillNet(skill_n_feat)}
        self.skill_correct = 0
        self.skill_total = 0

        # social: belief over the OTHER agent's actions
        self.n_actions = 0
        self.other_counts: np.ndarray | None = None
        self.other_pred_correct = 0
        self.other_pred_total = 0
        self.other_marg_correct = 0

        # meta-controller + self-model (trained by the caller on collected data)
        self.router = MetaRouter()
        self.self_head = SelfHead()
        # Phase-4 M3 self-reference: an independent self-DOUBT head predicts
        # whether the current BELIEF answer is wrong (from body-internal state
        # features).  When it fires during belief/ledger disagreement it marks
        # the domain at once -- faster self-driven revision.  Disabled by
        # default so the v17 behaviour is unchanged.
        self.doubt_head = SelfHead(n_feat=5)
        self.self_head_enabled = False
        self.self_doubt_threshold = 0.5

    # ------------------------------------------------------------------ #
    # identity observation (the core epistemic step)
    # ------------------------------------------------------------------ #
    def _belief(self, domain: int, state: int) -> CategoricalBelief:
        return self.beliefs.get((domain, state), CategoricalBelief.uniform(1, 2))

    def _unit_id(self, domain: int, state: int) -> int:
        return domain * self.n_states + state

    def attention_query(self, domain: int, state: int) -> int | None:
        """Executive working-memory answer (majority vote of gated writes)."""
        votes = self.attention.get(self._unit_id(domain, state), [])
        if not votes:
            return None
        return int(np.bincount(votes).argmax())

    def observe_identity(self, domain: int, state: int, shown: int, reliability: float) -> None:
        self.tick += 1
        key = (domain, state)
        # durable fact ledger: records every observation (P1 retrievability,
        # the routing substrate); confidence = the observation's reliability
        self.ledger.observe(FactRecord(key=f"d{domain}_s{state}", value=int(shown), source="body",
                                       tick=self.tick, confidence=reliability, version=1))
        # executive working memory (Block-9): budgeted, gated vote store --
        # only reliable observations are committed; PER-UNIT FIFO (each fact
        # channel keeps its own recent evidence, so an active channel cannot
        # starve the others).
        unit = self._unit_id(domain, state)
        if reliability >= self.attention_gate:
            votes = self.attention.setdefault(unit, [])
            votes.append(int(shown))
            if len(votes) > self.attention_budget:
                votes.pop(0)
            self.attention_spent = sum(len(v) for v in self.attention.values())
        # belief always accumulates (evidence regardless of the write gate);
        # the calibration is matched to the state's sensing regime
        lik = self.lik_noisy if state in self.noisy_states else self.lik
        self.beliefs[key] = self._belief(*key).observe(lik[:, int(shown)])
        self.obs_count[key] = self.obs_count.get(key, 0) + 1

        # world-model prediction head: from the previous state of the domain,
        # predict the WORLD's next state per the transition law (Block-7).
        # Without a law the target falls back to the observed state.
        if domain in self.prev_state:
            onehot = torch.zeros(1, self.n_states)
            onehot[0, self.prev_state[domain]] = 1.0
            logits = self.pred_heads[domain](onehot)
            if domain in self.laws:
                nxt = (self.prev_state[domain] + self.laws[domain]) % self.n_states
            else:
                nxt = state
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([nxt]))
            self.pred_opts[domain].zero_grad(set_to_none=True)
            loss.backward()
            self.pred_opts[domain].step()
            self.pred_correct += int(logits.argmax(-1).item() == nxt)
            self.pred_total += 1
        self.prev_state[domain] = state

        # revision detector: a high-confidence belief disagreeing with the
        # just-written ledger value for 3 consecutive observations means the
        # world flipped (Block-1).  Noisy-state disagreement is ignored.
        if reliability >= 0.9:
            bel_v = int(self._belief(*key).probs.argmax(-1).item())
            mem_v = self.ledger.current.get(f"d{domain}_s{state}")
            mem_val = int(mem_v.value) if mem_v is not None else -1
            conf = float(self._belief(*key).confidence()[0])
            if bel_v != mem_val and conf >= self.mark_belief_conf:
                self.conflict[key] = self.conflict.get(key, 0) + 1
            else:
                self.conflict[key] = 0
            if self.conflict.get(key, 0) >= 3 and domain not in self.marked:
                self.marked.add(domain)
            # self-reference (M3): the doubt head says the belief is likely
            # wrong while it disagrees with memory -> mark at once (faster than
            # waiting for 3 consecutive contradictions).  Only when enabled,
            # and only once the belief is confident (avoids firing during the
            # initial low-confidence convergence).
            if (self.self_head_enabled and domain not in self.marked
                    and bel_v != mem_val and conf >= self.mark_belief_conf):
                doubt = self.self_doubt(domain, state)
                if doubt is not None and doubt < self.self_doubt_threshold:
                    self.marked.add(domain)
        self.chain.append({"event": "observe", "snapshot": self.snapshot_digest()})

    # ------------------------------------------------------------------ #
    # compositional skill input (P6 hybrid)
    # ------------------------------------------------------------------ #
    def compose_skills(self, x: torch.Tensor) -> tuple[int, int]:
        """Run the symbolic order skeleton (A then B) with learned atoms."""
        with torch.no_grad():
            a_out = int(self.skills["a"](x.unsqueeze(0)).argmax(-1).item())
            b_out = int(self.skills["b"](x.unsqueeze(0)).argmax(-1).item())
        return a_out, b_out

    def record_skill_verdict(self, a_out: int, b_out: int, label: tuple[int, int]) -> None:
        self.skill_correct += int(a_out == label[0] and b_out == label[1])
        self.skill_total += 1

    # ------------------------------------------------------------------ #
    # social cognition (Block-11): model of ANOTHER agent
    # ------------------------------------------------------------------ #
    def set_other_actions(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.other_counts = np.zeros(n_actions)

    def observe_other_action(self, action: int, next_known: int | None = None) -> None:
        """Observe the other's action; optionally score the body's prediction
        of it (call BEFORE observing when next_known is the action)."""
        if self.other_counts is None:
            self.set_other_actions(action + 1)
        if next_known is not None and self.other_counts.sum() > 0:
            pred = int(self.other_counts.argmax())
            self.other_pred_correct += int(pred == next_known)
            self.other_pred_total += 1
            self.other_marg_correct += int(random.randrange(self.n_actions) == next_known)
        self.other_counts[action] += 1

    # ------------------------------------------------------------------ #
    # goals / queries
    # ------------------------------------------------------------------ #
    def belief_gap(self, domain: int, state: int) -> float:
        return float(1.0 - self._belief(domain, state).confidence()[0])

    def available_domains(self) -> list[int]:
        available = [d for d in self.values if d not in self.marked]
        return available or list(self.values)

    def select_domain(self, available: list[int]) -> int:
        # value anchoring (Block-10): highest-value available domain
        return min(available, key=lambda d: -self.values[d])

    def select_state(self, domain: int, tie_break: Iterable[float] | None = None) -> int:
        """gap selection (P2) with fewest-observed tie-break; optional
        prediction-error signal breaks near-equal gaps (Block-7 drive)."""
        gaps = {s: self.belief_gap(domain, s) for s in range(self.n_states)}
        max_gap = max(gaps.values())
        argmax_gaps = [s for s, g in gaps.items() if g >= max_gap - 1e-9]
        obs = {s: self.obs_count.get((domain, s), 0) for s in argmax_gaps}
        return min(argmax_gaps, key=lambda s: (obs[s], s))

    def query(self, domain: int, state: int) -> int | None:
        record = self.ledger.current.get(f"d{domain}_s{state}")
        return int(record.value) if record is not None else None

    def state_features(self, domain: int, state: int) -> torch.Tensor:
        record = self.ledger.current.get(f"d{domain}_s{state}")
        mem_conf = float(record.confidence) if record is not None else 0.0
        mem_v = int(record.value) if record is not None else -1
        bel = self._belief(domain, state)
        bel_conf = float(bel.confidence()[0])
        bel_v = int(bel.probs.argmax(-1).item())
        agreement = 1.0 if (record is not None and mem_v == bel_v) else 0.0
        flipped = 1.0 if domain in self.marked else 0.0
        obs = self.obs_count.get((domain, state), 0) / 10.0
        return torch.tensor([mem_conf, bel_conf, agreement, obs, flipped], dtype=torch.float)

    def source_values(self, domain: int, state: int) -> tuple[int, int]:
        record = self.ledger.current.get(f"d{domain}_s{state}")
        mem_v = int(record.value) if record is not None else -1
        bel_v = int(self._belief(domain, state).probs.argmax(-1).item())
        return mem_v, bel_v

    def self_doubt(self, domain: int, state: int) -> float | None:
        """Phase-4 M3: does the body think its current BELIEF answer might be
        wrong?  The doubt head predicts belief-correctness from body-internal
        state features; low prediction = self-doubt.  None when disabled."""
        if not self.self_head_enabled:
            return None
        with torch.no_grad():
            x = self.state_features(domain, state).unsqueeze(0)
            return float(self.doubt_head(x).squeeze(-1).item())

    # ------------------------------------------------------------------ #
    # snapshots (P5)
    # ------------------------------------------------------------------ #
    def snapshot_digest(self) -> str:
        current = sorted((k, str(v["value"])) for k, v in self.ledger.snapshot()["current"].items())
        return repr(current)

    def idle(self) -> None:
        self.chain.append({"event": "idle", "snapshot": self.snapshot_digest()})

    def p5_consistent(self) -> bool:
        return all(
            ch["event"] != "idle" or ch["snapshot"] == prev["snapshot"]
            for ch, prev in zip(self.chain[1:], self.chain)
        )
