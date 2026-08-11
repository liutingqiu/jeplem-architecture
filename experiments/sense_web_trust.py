"""Phase-2 body internet senses: TRUST (TrustRegistry evolution) -> body.

Two sources compete: one always reports correctly (rises in trust), the other
always errs (falls).  TrustRegistry score modulates FactRecord.confidence so
the ledger's weak/strong isolation becomes continuous trust modulation --
high-trust sources' evidence is accepted, low-trust sources' is isolated.

Verdicts:
  C1 divergence: correct source trust -> >0.8, wrong source trust -> <0.3.
  C2 ledger: high-trust evidence lands in current, low-trust is isolated.
  C3 confidence tracks trust over time.
Protocol: 3 seeds (444/555/666).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import FactRecord, VersionedFactLedger
from plwm.trust import TrustRegistry


def run(*, seeds: tuple[int, ...] = (444, 555, 666), rounds: int = 40,
        artifact: Path | None = None) -> dict:
    results = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        trust = TrustRegistry(prior=0.5, alpha=0.2, beta=0.2)
        ledger = VersionedFactLedger(trusted_sources={"good", "bad"}, revision_confidence=0.6)
        # C3 trace: confidence of a fact from each source over rounds
        conf_good: list[float] = []
        conf_bad: list[float] = []
        for r in range(rounds):
            truth = int(rng.rand() < 0.5)
            # good source always correct, bad source always wrong
            good_correct = True
            bad_correct = False
            trust.report("good", good_correct)
            trust.report("bad", bad_correct)
            # write a fact from each source with trust-modulated confidence
            conf_g = trust.score("good")
            conf_b = trust.score("bad")
            conf_good.append(conf_g)
            conf_bad.append(conf_b)
            ledger.observe(FactRecord(key=f"k{r}", value=truth, source="good",
                                      tick=r, confidence=conf_g, version=1))
            ledger.observe(FactRecord(key=f"k{r}", value=1 - truth, source="bad",
                                      tick=r, confidence=conf_b, version=1))
        final_good = trust.score("good")
        final_bad = trust.score("bad")
        # C2: good-source facts in current, bad-source facts isolated
        good_in_current = sum(1 for k, rec in ledger.current.items()
                              if rec.source == "good")
        bad_in_current = sum(1 for k, rec in ledger.current.items()
                             if rec.source == "bad")
        bad_isolated = sum(1 for rec in ledger.isolated if rec.source == "bad")
        # C3: confidence trend: good rises, bad falls
        good_rise = conf_good[-1] > conf_good[0]
        bad_fall = conf_bad[-1] < conf_bad[0]
        results.append({
            "seed": seed, "final_good": final_good, "final_bad": final_bad,
            "good_in_current": good_in_current, "bad_in_current": bad_in_current,
            "bad_isolated": bad_isolated, "good_rise": good_rise, "bad_fall": bad_fall,
        })

    final_good = float(np.mean([r["final_good"] for r in results]))
    final_bad = float(np.mean([r["final_bad"] for r in results]))
    c1 = final_good >= 0.8 and final_bad <= 0.3
    good_in_current = float(np.mean([r["good_in_current"] for r in results]))
    bad_in_current = float(np.mean([r["bad_in_current"] for r in results]))
    bad_isolated = float(np.mean([r["bad_isolated"] for r in results]))
    c2 = good_in_current >= rounds * 0.9 and bad_in_current <= rounds * 0.1 and bad_isolated >= rounds * 0.5
    good_rise = float(np.mean([r["good_rise"] for r in results]))
    bad_fall = float(np.mean([r["bad_fall"] for r in results]))
    c3 = good_rise >= 1.0 and bad_fall >= 1.0
    accepted_verdict = c1 and c2 and c3
    report = {
        "protocol": "sense_web_trust_v1: TrustRegistry with a correct source (rises) vs an always-wrong source (falls); score modulates FactRecord.confidence -> ledger weak/strong isolation becomes continuous trust. C1 divergence (good>0.8, bad<0.3); C2 good facts in current, bad isolated; C3 confidence trend. Criterion: C1-C3 per seed.",
        "seeds": list(seeds),
        "results": {
            "good_trust": round(final_good, 3),
            "bad_trust": round(final_bad, 3),
            "good_in_current": round(good_in_current / rounds, 3),
            "bad_isolated": round(bad_isolated / rounds, 3),
            "confidence_trend": round((good_rise + bad_fall) / 2, 3),
        },
        "accepted": accepted_verdict,
        "artifact": str(artifact or ROOT / "artifacts" / "sense-web-trust.json"),
    }
    (artifact or ROOT / "artifacts" / "sense-web-trust.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "sense-web-trust.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
