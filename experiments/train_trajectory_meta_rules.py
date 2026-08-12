"""Unified trajectory meta-rule driver (2026-08-12).

Runs BOTH implemented non-pair families -- causality (2) and interaction (4) --
through ONE shared RuleLedger over real Luanti episodes.  This is the
integration step: the two families are one pipeline, not two scripts.

Pipeline per episode:
    causality  : learn action->(dx,dz) causal map; compile/reuse per signature
    interaction: learn agent-goal distance -> outcome rule; compile/reuse
Both write to the same ledger, so cross-family reuse and world-change
detection are unified.

Criterion: each episode either REUSES a known rule (hit) or COMPILES a new one
(induction); no family should spuriously re-compile the same signature.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.rule_ledger import RuleLedger
from plwm.trajectory_rules import learn_trajectory_rules


def load_episodes(rollout_path: Path) -> list[dict]:
    return [json.loads(line) for line in rollout_path.open(encoding="utf-8")]


def run(*, rollout_path: Path, ledger_path: Path | None = None,
        artifact: Path | None = None) -> dict:
    episodes = load_episodes(rollout_path)
    ledger = RuleLedger()

    total = {"compiled": 0, "reused": 0, "gaps": 0, "world_changes": 0}
    for ep in episodes:
        r = learn_trajectory_rules(ep, ledger)
        for k in total:
            total[k] += r[k]

    if ledger_path is not None:
        ledger.save(ledger_path)

    report = {
        "protocol": "trajectory_meta_rules_v1: causality (action->effect causal maps) + interaction (agent-goal -> outcome) learned from REAL Luanti episodes through ONE shared RuleLedger. Compile-once-reuse-many across episodes; world changes (new causal signature) compiled as new rules.",
        "n_episodes": len(episodes),
        "results": total,
        "ledger": ledger.stats(),
        "distinct_signatures": len(ledger.current),
        "rules": [
            {"family": r.relation_key, "signature": sig, "version": r.version,
             "confirms": r.n_confirm}
            for sig, bucket in sorted(ledger.current.items())
            for r in bucket
        ],
        "artifact": str(artifact or ROOT / "artifacts" / "trajectory-meta-rules.json"),
    }
    (artifact or ROOT / "artifacts" / "trajectory-meta-rules.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "trajectory-meta-rules.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(
        rollout_path=ROOT / "artifacts" / "luanti-rollouts.jsonl",
        ledger_path=ROOT / "artifacts" / "luanti-trajectory-ledger.json",
    ), indent=2, ensure_ascii=False))
