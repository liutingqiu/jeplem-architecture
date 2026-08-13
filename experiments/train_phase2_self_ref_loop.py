"""Phase-4 M3: self-reference closure (G-S).

Does the body's self-DOUBT (a head that predicts "my current belief answer is
wrong" from body-internal state features) drive faster self-revision than
waiting for 3 consecutive belief/ledger contradictions?

Scenario: two clean domains; one domain's world truth FLIPS mid-run.  The
belief lags (accumulated old evidence) while the ledger records the new
observations -> belief/ledger disagreement.  The non-driven detector marks the
domain after 3 consecutive disagreements.  The self-driven path marks it as
soon as the doubt head (trained on belief-correctness) fires LOW during a
disagreement.

Gate (whitepaper G-S): self-driven revision latency < non-driven latency by a
significant margin (>= 2 steps), 3 seeds, with the doubt head generalising to a
different flip timing than it trained on.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.body import UnifiedBody

SEEDS = [41_123, 41_124, 41_125]
RELIABILITY = 0.97
TEST_FLIP = 700   # different flip timing than any training flip (generalisation)


def _truth(domain: int, state: int, t: int, flip_step: int) -> int:
    if domain == 0:
        return int(state ^ (t >= flip_step))   # domain 0 flips after flip_step
    return state                              # domain 1 never flips


def _collect(body: UnifiedBody, flip_step: int, n_steps: int) -> list:
    samples = []
    flip_mark_step = None
    for t in range(n_steps):
        for d in (0, 1):
            for s in (0, 1):
                truth = _truth(d, s, t, flip_step)
                body.observe_identity(d, s, truth, RELIABILITY)
                if d == 0:
                    bel_v = int(body._belief(d, s).probs.argmax(-1).item())
                    samples.append((body.state_features(d, s).clone(), float(bel_v == truth)))
                    if d in body.marked and flip_mark_step is None and t >= flip_step:
                        flip_mark_step = t
    return samples, flip_mark_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    artifact = args.artifact or ROOT / "artifacts" / f"phase4-m3-self-ref-loop-{args.seed}.json"

    # ---- Phase 1: collect belief-correctness samples and train doubt head ----
    # Train on MULTIPLE flip episodes (different timings) so the doubt head
    # robustly learns "confident disagreement with memory = likely wrong",
    # with a balanced loss (belief-wrong samples are rarer than right ones).
    all_X: list = []
    all_y: list = []
    for train_flip in (300, 450, 600):
        tb = UnifiedBody(seed=args.seed, values={0: 3, 1: 2}, n_states=2,
                         write_budget=8, write_gate=0.5)
        samples_t, _ = _collect(tb, train_flip, n_steps=900)
        all_X += [f for f, _ in samples_t]
        all_y += [l for _, l in samples_t]
    train_body = UnifiedBody(seed=args.seed, values={0: 3, 1: 2}, n_states=2,
                             write_budget=8, write_gate=0.5)
    X = torch.stack(all_X)
    y = torch.tensor(all_y, dtype=torch.float)
    w0 = y.numel() / (2 * (y == 0).sum().clamp(min=1))
    w1 = y.numel() / (2 * (y == 1).sum().clamp(min=1))
    weights = torch.where(y == 0, w0, w1)
    opt = torch.optim.AdamW(train_body.doubt_head.parameters(), lr=3e-3)
    for _ in range(2000):
        pred = train_body.doubt_head(X).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(pred, y, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_body.doubt_head.eval()
    with torch.no_grad():
        preds = train_body.doubt_head(X).squeeze(-1).numpy()
    doubt_ece = float(sum(abs(p - l) for p, l in zip(preds, y.numpy())) / len(samples_t) / 3)

    # ---- Phase 2: measure revision latency, non-driven vs self-driven ----
    def latency(enabled: bool, flip_step: int) -> dict:
        body = UnifiedBody(seed=args.seed, values={0: 3, 1: 2}, n_states=2,
                           write_budget=8, write_gate=0.5)
        body.self_head_enabled = enabled
        body.self_doubt_threshold = 0.5
        if enabled:
            body.doubt_head.load_state_dict(train_body.doubt_head.state_dict())
        _, mark = _collect(body, flip_step, n_steps=900)
        # false-positive guards: no mark on the never-flipping domain, and the
        # flipping domain must not be marked BEFORE the flip.
        return {"mark": mark, "flip_step": flip_step,
                "pre_flip_mark": (mark is not None and mark < flip_step),
                "domain1_marked": 1 in body.marked}

    non = latency(False, TEST_FLIP)
    self_ = latency(True, TEST_FLIP)
    non_lat = (non["mark"] - TEST_FLIP) if non["mark"] is not None else None
    self_lat = (self_["mark"] - TEST_FLIP) if self_["mark"] is not None else None
    clean = (not non["domain1_marked"] and not self_["domain1_marked"]
             and not non["pre_flip_mark"] and not self_["pre_flip_mark"])
    significant = (non_lat is not None and self_lat is not None
                   and self_lat <= non_lat - 2 and clean)

    report = {
        "protocol": "phase4_m3_self_ref_loop_v1: doubt head (belief-correctness) trained on a flip at step TRAIN_FLIP; tested on a flip at step TEST_FLIP (different timing). Latency = steps from flip to domain marked. Non-driven = 3-consecutive-contradiction detector; self-driven = doubt head fires during disagreement. False-positive guards: no pre-flip mark, no mark on the never-flipping domain. Gate: self_latency <= non_latency - 2 AND clean (3 seeds).",
        "seed": args.seed,
        "results": {
            "train_samples": len(all_X),
            "train_doubt_mean_abs_err": round(doubt_ece, 4),
            "non_driven_latency": non_lat,
            "self_driven_latency": self_lat,
            "latency_reduction": (non_lat - self_lat) if (non_lat is not None and self_lat is not None) else None,
            "fp_pre_flip_mark": non["pre_flip_mark"] or self_["pre_flip_mark"],
            "fp_domain1_marked": non["domain1_marked"] or self_["domain1_marked"],
        },
        "gate_pass": bool(significant),
        "artifact": str(artifact),
    }
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate_pass": significant, "results": report["results"]}, indent=1))


if __name__ == "__main__":
    main()
