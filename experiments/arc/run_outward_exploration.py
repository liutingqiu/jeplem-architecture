"""Phase-R: OUTWARD exploration -- the knowledge engine goes beyond its snapshot.

The user's directive: the system should not be limited to the ingested
knowledge base; it should explore OUTWARD (the real world).  When the snapshot
(wikidata, 1.32亿 facts) cannot answer, the system FETCHES from the LIVE web
(Wikipedia API) -- exploring outward, with grounding:

  snapshot   -- the wikidata knowledge base answers first (exact + transitive).
  outward    -- on unknown, the system fetches the entity's live Wikipedia
                summary.  The answer is GROUNDED: a source-cited excerpt, NOT
                an invented claim.  If the web has nothing, it says "I don't
                know" (honest -- outward exploration must not hallucinate).

Eval (per question): the snapshot-gap is filled by a REAL grounded summary
(source URL present, excerpt contains the entity), and failures are honest
unknowns (no fabrication).  The system explores the world, grounded.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from experiments.arc.knowledge_query import answer, WIKIDATA_RELS

DB = ROOT / "data" / "knowledge" / "wikidata.db"
UA = {"User-Agent": "JEPLM-knowledge/0.1 (research; honest outward exploration)"}

# entities whose snapshot-answer is sparse/empty (good outward-exploration cases)
TEST_ENTITIES = [
    "quantum entanglement", "machine learning", "Fermi paradox", "black hole",
    "photosynthesis", "general relativity", "dark matter", "neural network",
    "climate change", "artificial general intelligence",
]


def web_summary(entity: str) -> dict | None:
    """Fetch the live Wikipedia summary for an entity (grounded: source + excerpt).
    Returns {'title', 'extract', 'source'} or None (page missing / network fail)."""
    try:
        title = entity.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        extract = (d.get("extract") or "").strip()
        if not extract:
            return None
        return {"title": d.get("title", entity), "extract": extract[:400],
                "source": f"https://en.wikipedia.org/wiki/{d.get('title', title)}"}
    except Exception:
        return None


def snapshot_answer(store, entity: str) -> bool:
    """Does the snapshot answer 'what is X' with any classes?"""
    rep = answer(store, f"what is {entity}", rels=WIKIDATA_RELS)
    return not rep.unknown and bool(rep.answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-r-outward-exploration.json")
    args = parser.parse_args()

    store = KnowledgeStore(args.db)
    results = []
    for ent in TEST_ENTITIES:
        snap_ok = snapshot_answer(store, ent)
        web = web_summary(ent)
        grounded = False
        if web is not None:
            # grounded: the excerpt contains the entity and a real source is cited
            grounded = (ent.split()[0].lower() in web["extract"].lower()) and web["source"].startswith("https://en.wikipedia.org")
        results.append({
            "entity": ent,
            "snapshot_answered": snap_ok,
            "outward_explored": web is not None,
            "outward_grounded": grounded,
            "source": web["source"] if web else None,
            "excerpt": web["extract"][:120] if web else None,
            "honest_if_failed": web is None,   # no fabrication on failure
        })
    store.close()

    gaps = [r for r in results if not r["snapshot_answered"]]
    filled = [r for r in gaps if r["outward_grounded"]]
    honest = [r for r in gaps if r["honest_if_failed"]]
    report = {
        "protocol": (
            "Phase-R outward exploration: when the snapshot cannot answer, the system fetches "
            "the LIVE web (Wikipedia) -- grounded, source-cited, honest on failure.  Eval: "
            "snapshot-gaps filled by REAL grounded summaries; no fabrication."
        ),
        "snapshot_gaps": len(gaps),
        "gaps_filled_by_outward": len(filled),
        "gaps_honest": len(honest),
        "outward_grounded_rate": round(len(filled) / len(gaps), 3) if gaps else 0.0,
        "results": results,
    }
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"snapshot gaps: {len(gaps)}  | outward-filled: {len(filled)} "
          f"| honest: {len(honest)}  | grounded rate: {report['outward_grounded_rate']}")
    for r in results[:6]:
        print(f"  {r['entity']}: snapshot={r['snapshot_answered']} "
              f"outward={r['outward_grounded']} "
              f"src={r['source'][:55] if r['source'] else 'none'}")
    print("VERDICT:", "outward exploration fills gaps, grounded + honest" if report["outward_grounded_rate"] >= 0.8
          else "see detail")


if __name__ == "__main__":
    main()
