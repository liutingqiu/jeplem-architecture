"""Continuous self-exploration: the system explores its OWN skills and rules
autonomously, in a loop, logging progress.  The user's directive: "let it
explore itself, you just monitor the machine."

Runs, round after round with fresh seeds:
  * the skill-acquisition loop (Phase T: programming skills via absorb +
    practice + explore),
  * the self-generating loop (Phase S: invents new candidate rules from
    verified experience).
Logs each round's discovery to the exploration journal.  A long-running
background process -- the system trains itself while we monitor.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from experiments.resource_guard import start_resource_guard
from experiments.arc.run_skill_acquisition_loop import run_seed as skill_seed
from experiments.arc.run_self_generating_loop import run_seed as gen_seed
from plwm.knowledge_store import KnowledgeStore

DB = ROOT / "data" / "knowledge" / "wikidata.db"
JOURNAL = ROOT / "artifacts" / "exploration_journal.jsonl"


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False))


def main() -> None:
    start_resource_guard(interval=2.0)
    store = KnowledgeStore(DB)
    seed = 1000
    skills_solved_total = 0
    rules_generated_total = 0
    rounds = 0
    print(f"continuous self-exploration started (journal: {JOURNAL})", flush=True)
    while True:
        seed += 1
        try:
            # PROGRAMMING priority: 2 skill rounds per exploration round
            s1 = skill_seed(seed)
            s2 = skill_seed(seed + 500)
            g = gen_seed(seed + 1000, store)          # random category (free)
            skills_solved_total += int(s1["total_acc"] >= 0.9)
            rules_generated_total += len(g["self_generated_rules"])
            rounds += 1
            log({"round": rounds, "seed": seed, "time": time.time(),
                 "programming_1_acc": s1["total_acc"], "programming_2_acc": s2["total_acc"],
                 "programming_gate": s1["gate_pass"] and s2["gate_pass"],
                 "programming_outward": s1["outward_acquired"],
                 "free_explore_category": g["category"],
                 "free_generated_rules": g["self_generated_rules"][:4],
                 "free_gate": g["gate_pass"],
                 "cumulative_skills_solved": skills_solved_total,
                 "cumulative_rules_generated": rules_generated_total})
        except Exception as e:
            log({"round": rounds + 1, "seed": seed, "time": time.time(), "error": str(e)})
        time.sleep(2)   # pace, and let the resource guard breathe


if __name__ == "__main__":
    main()
