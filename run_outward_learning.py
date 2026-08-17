"""Real-world OUTWARD learning v2 -- PRECISE extraction from LIVE Wikidata.

v1 used Wikipedia TEXT summaries, which rarely state a structured relation --
outward unlock plateaued at ~3.6% (the honest ceiling of free-text luck).  v2
fetches LIVE Wikidata STRUCTURE instead:

  entity -> QID -> claims (P106 occupation / P27 citizenship / P17 country /
  P36 capital / P30 continent / P31 instance) -> value labels.

This gives PRECISE answers beyond the snapshot, source-cited by QID.  Honest:
no value found -> "don't know" (no fabrication).  Snapshot first (the KB
answers what it knows), outward fills the real gaps exactly.

Real-world semantics matter here: a person's citizenship is P27, not P17
(P17 is country for places/organisations), occupation is P106 -- the mapping
below is the real Wikidata relation grammar.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore  # noqa: E402
from experiments.arc.knowledge_query import answer, WIKIDATA_RELS  # noqa: E402
from experiments.resource_guard import start_resource_guard  # noqa: E402

DB = ROOT / "data" / "knowledge" / "wikidata.db"
JOURNAL = ROOT / "artifacts" / "outward_learning_journal.jsonl"
BATCH = 8
UA = {"User-Agent": "JEPLM-knowledge/0.1 (research; outward precise extraction)"}

REL_QS = [
    ("instance", "what is {}"),
    ("occupation", "what is the occupation of {}"),
    ("country", "what country is {} in"),
    ("capital", "what is the capital of {}"),
    ("continent", "what continent is {} in"),
]
# real Wikidata property grammar for each relation type (order = preference)
REL_P = {
    "instance": ["P31"],
    "occupation": ["P106"],
    "country": ["P27", "P17"],
    "capital": ["P36"],
    "continent": ["P30"],
}


def log(entry: dict) -> None:
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def _get(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def sample_entities(store, rng, n: int) -> list[str]:
    rows = store.conn.execute(
        "SELECT DISTINCT start FROM facts WHERE rel='P31' AND length(start)<25 "
        "AND start NOT GLOB '*[0-9]*' AND start NOT LIKE '(%)' "
        "AND start NOT LIKE '#%' AND start NOT LIKE '@%' LIMIT 200000").fetchall()
    ents = [r[0] for r in rows]
    rng.shuffle(ents)
    return ents[:n]


def outward_exact(ent: str, rel: str) -> dict | None:
    """PRECISE outward: entity -> QID -> claims -> value labels.  Returns
    {'values', 'qid', 'source'} or None (nothing found / rate-limited)."""
    try:
        d = _get("https://www.wikidata.org/w/api.php?action=wbsearchentities"
                 "&search=%s&language=en&limit=1&format=json" % urllib.parse.quote(ent))
    except Exception:
        return None
    if not d.get("search"):
        return None
    qid = d["search"][0]["id"]
    time.sleep(1)
    try:
        doc = _get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json?flavor=dump")
    except Exception:
        return None
    e = doc.get("entities", {}).get(qid, {})
    value_qids = []
    for prop in REL_P[rel]:
        for claim in e.get("claims", {}).get(prop, []):
            try:
                v = claim["mainsnak"]["datavalue"]["value"]
                if isinstance(v, dict) and "id" in v and v["id"].startswith("Q"):
                    value_qids.append(v["id"])
            except Exception:
                continue
    value_qids = list(dict.fromkeys(value_qids))
    if not value_qids:
        return None
    # batch labels for the value QIDs
    labels = {}
    for i in range(0, len(value_qids), 10):
        time.sleep(1)
        chunk = value_qids[i:i + 10]
        try:
            dl = _get("https://www.wikidata.org/w/api.php?action=wbgetentities"
                      "&ids=%s&props=labels&languages=en&format=json"
                      % "|".join(chunk))
            for q in chunk:
                labels[q] = (dl.get("entities", {}).get(q, {})
                             .get("labels", {}).get("en", {}).get("value", q))
        except Exception:
            continue
    values = [labels.get(q, q) for q in value_qids]
    return {"values": values[:5], "qid": qid,
            "source": f"https://www.wikidata.org/wiki/{qid}"}


def main() -> None:
    start_resource_guard(interval=2.0)
    rng = random.Random(20260817)
    store = KnowledgeStore(DB)
    outward_learned: dict = {}
    covered = set()
    tried = set()
    honest_fail = rate_limited = 0
    round_no = 0
    print(f"outward v2 (precise Wikidata) started (journal {JOURNAL})", flush=True)
    while True:
        round_no += 1
        batch = sample_entities(store, rng, BATCH)
        round_covered = round_out = round_honest = 0
        for ent in batch:
            rel, qfmt = rng.choice(REL_QS)
            key = (ent, rel)
            if key in tried:
                continue
            tried.add(key)
            rep = answer(store, qfmt.format(ent), rels=WIKIDATA_RELS)
            if not rep.unknown and rep.answer:
                covered.add(key)
                round_covered += 1
                continue
            out = outward_exact(ent, rel)
            if out is not None:
                outward_learned[key] = out
                round_out += 1
            else:
                round_honest += 1
                honest_fail += 1
        tried_n = len(tried)
        covered_rate = len(covered) / tried_n if tried_n else 0.0
        gaps = tried - covered
        unlock_rate = (sum(1 for k in gaps if k in outward_learned) / len(gaps)
                       if gaps else 0.0)
        log({"round": round_no, "time": time.time(),
             "real_questions_tried": tried_n,
             "snapshot_answered": len(covered),
             "snapshot_coverage": round(covered_rate, 4),
             "snapshot_gaps": len(gaps),
             "outward_unlocked": len(outward_learned),
             "outward_unlock_rate": round(unlock_rate, 4),
             "outward_learned_cumulative": len(outward_learned),
             "honest_failures": honest_fail,
             "round_covered": round_covered, "round_outward": round_out,
             "round_honest": round_honest})
        time.sleep(2)


if __name__ == "__main__":
    main()
