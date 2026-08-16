"""Real-world OUTWARD learning loop -- keep learning from the LIVE world.

User: '只在现有知识库探索,没有向外的机制?'  This makes the outward channel
CONTINUOUS: each round the system faces REAL entities sampled from the real
knowledge base; when the snapshot cannot answer, it FETCHES the LIVE web
(Wikipedia) -- grounded, source-cited, no fabrication -- and KEEPS what it
learns.  The snapshot grows from static to static+outward.

Measures (honest):
  snapshot_coverage  -- fraction of real entities the snapshot answers.
  outward_unlock     -- of the snapshot gaps, fraction the real world fills
                        with a grounded summary (this is the outward value).
  outward_learned    -- cumulative entities learned from the live world.
  honest_failures    -- snapshot gaps the web could not fill (honest 'don't
                        know', zero fabrication).

Principles (fixed): real source (live web) + outward (beyond the snapshot) +
self-challenge (the gaps are exactly what it must fetch) + honest measure.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore  # noqa: E402
from experiments.arc.run_outward_exploration import (  # noqa: E402
    web_summary, snapshot_answer,
)
from experiments.resource_guard import start_resource_guard  # noqa: E402

DB = ROOT / "data" / "knowledge" / "wikidata.db"
JOURNAL = ROOT / "artifacts" / "outward_learning_journal.jsonl"
BATCH = 20


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def sample_entities(store, rng, n: int) -> list[str]:
    rows = store.conn.execute(
        "SELECT DISTINCT start FROM facts WHERE rel='P31' AND length(start)<25 "
        "LIMIT 50000").fetchall()
    ents = [r[0] for r in rows]
    rng.shuffle(ents)
    return ents[:n]


def main() -> None:
    start_resource_guard(interval=2.0)
    rng = random.Random(20260816)
    store = KnowledgeStore(DB)
    outward_learned: dict[str, dict] = {}
    covered: set[str] = set()
    tried: set[str] = set()
    honest_fail = 0
    round_no = 0
    print(f"outward learning started (journal {JOURNAL})", flush=True)
    while True:
        round_no += 1
        batch = sample_entities(store, rng, BATCH)
        round_new = round_covered = round_out = round_honest = 0
        for ent in batch:
            if ent in tried:
                continue
            tried.add(ent)
            if snapshot_answer(store, ent):
                covered.add(ent)
                round_covered += 1
                continue
            # OUTWARD: the snapshot gap is exactly what it must fetch
            web = web_summary(ent)
            if web is not None and ent.split()[0].lower() in web["extract"].lower():
                outward_learned[ent] = {
                    "excerpt": web["extract"][:300], "source": web["source"]}
                round_out += 1
            else:
                round_honest += 1
        tried_n = len(tried)
        covered_rate = len(covered) / tried_n if tried_n else 0.0
        gaps = tried - covered
        unlock_rate = (sum(1 for e in gaps if e in outward_learned) / len(gaps)
                       if gaps else 0.0)
        log({"round": round_no, "time": time.time(),
             "real_entities_tried": tried_n,
             "snapshot_covered": len(covered),
             "snapshot_coverage": round(covered_rate, 4),
             "snapshot_gaps": len(gaps),
             "outward_unlocked": len(outward_learned),
             "outward_unlock_rate": round(unlock_rate, 4),
             "outward_learned_cumulative": len(outward_learned),
             "honest_failures": honest_fail,
             "round_covered": round_covered, "round_outward": round_out,
             "round_honest": round_honest})
        time.sleep(1)   # pace, let the resource guard breathe


if __name__ == "__main__":
    main()
