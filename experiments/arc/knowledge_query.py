"""Language interface (Phase-4 M2): natural-ish questions -> MRep answers.

The body answers from its disk-backed KnowledgeStore with exact-match retrieval
AND transitive reasoning, and it is SELF-AWARE: it says "I don't know" when the
fact (or a reasoned derivation) is not in its knowledge.  Every answer is an
MRep (plwm/mrep.py) -- structured, confidence-bearing, source-traceable -- and
rendered to natural language by the template mechanism (plwm/language_gen.py).

Question patterns handled (regex):
  "is <subj> a <obj>" / "is a <subj> a <obj>"   -> IsA(subj,obj) exact + transitive
  "<subj> is used for <obj>" / "is <subj> used for <obj>" -> UsedFor(subj,obj)
  "is <subj> <prop>" / "<subj> is <prop>"       -> HasProperty(subj,prop)
  "what is <subj>" / "what is a <subj>"          -> IsA(subj) reachable set
  "do you know <subj>"                            -> self-aware existence check
  anything else                                   -> "I don't know" (honest)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plwm.knowledge_store import KnowledgeStore
from plwm.mrep import MRep, FACT, CHAIN, UNKNOWN, EXPLANATION, Source
from plwm.language_gen import render

DEFAULT_DB = ROOT / "data" / "knowledge" / "conceptnet.db"

# canonical relation -> KB relations.  ConceptNet uses IsA/UsedFor/HasProperty;
# Wikidata stores subclass/instance as P279/P31 (and others for located/country).
CONCEPTNET_RELS: dict[str, list[str]] = {
    "IsA": ["IsA"], "UsedFor": ["UsedFor"], "HasProperty": ["HasProperty"],
}
WIKIDATA_RELS: dict[str, list[str]] = {
    "IsA": ["P279", "P31"], "LocatedIn": ["P131"], "Country": ["P17"],
    "Occupation": ["P106"], "PartOf": ["P361", "P527"], "Author": ["P50"],
}

# patterns: (regex, handler) -- ordered specific-first.  Entity capture is
# permissive (anything but a question mark; NOT greedy so a trailing "."/"?"
# stays out of the entity) because Wikidata labels carry dots, brackets and
# punctuation ("(1,4,6-Trioxaspiro[4.6]undecan-2-yl)methanol",
# "N-(...spiro[4.5]dec-...)formamide") and ConceptNet entities carry commas.
# Entity boundaries are the structural anchors (`\s+a\s+` separators, " used
# for ", `[.?]?$`).  Leading articles are only consumed WITH their space
# (`a `/`an `/`the `) so an entity that starts with "a"/"the" is not
# truncated.  IGNORECASE: questions are matched against the ORIGINAL-case
# question so the captured entity keeps its stored spelling ("Douglas Adams").
# non-capturing wrappers: `({_EG}?)` expands to `((?:[^?]+)?)` -- the trailing
# `?` is a GROUP quantifier (optional), not adjacent to `+`, so no "multiple
# repeat".  GREEDY for entities followed by a structural separator (the
# separator anchors the boundary, and greedy makes the subject absorb a
# trailing "A" that IGNORECASE would otherwise mis-read as an article, e.g.
# "(-)-Ampelopsin A a ampelopsin A"); LAZY for the LAST entity so a trailing
# "." / "?" stays out of the capture.
_EG = r"(?:[^?]+)"
_EL = r"(?:[^?]+?)"
_ART = r"(?:a |an |the )?"
_WHAT_IS = re.compile(rf"^what\s+is\s+{_ART}\s*({_EL}?)[.?]?$", re.IGNORECASE)
_DO_YOU_KNOW = re.compile(rf"^do\s+you\s+know\s+(?:about\s+)?{_ART}\s*({_EL}?)[.?]?$", re.IGNORECASE)
_USED_FOR = re.compile(rf"^(?:is\s+)?({_EG}?)\s+(?:is\s+)?used\s+for\s+{_ART}\s*({_EL}?)[.?]?$", re.IGNORECASE)
_IS_A = re.compile(rf"^is\s+{_ART}\s*({_EG}?)\s+an?\s+({_EL}?)[.?]?$", re.IGNORECASE)
_IS_A2 = re.compile(rf"^(.+?)\s+is\s+an?\s+({_EL}?)[.?]?$", re.IGNORECASE)
_HAS_PROP = re.compile(rf"^(?:is\s+)?{_ART}\s*({_EG}?)\s+is\s+({_EL}?)[.?]?$", re.IGNORECASE)
# relation-specific questions (use the full 9-relation KB, not just IsA)
_WHO_WROTE = re.compile(rf"^who\s+wrote\s+({_EL}?)[.?]?$", re.IGNORECASE)
_WHERE_IS = re.compile(rf"^where\s+is\s+({_EL}?)(?:\s+located)?[.?]?$", re.IGNORECASE)
_WHAT_OCC = re.compile(rf"^(?:what\s+is\s+the\s+occupation\s+of\s+|what\s+does\s+)({_EL}?)(?:\s+do)?[.?]?$", re.IGNORECASE)
_WHAT_COUNTRY = re.compile(rf"^what\s+country\s+is\s+({_EL}?)\s+in[.?]?$", re.IGNORECASE)
_WHAT_PARTOF = re.compile(rf"^what\s+is\s+{_ART}\s*({_EL}?)\s+part\s+of[.?]?$", re.IGNORECASE)
_WHAT_HAS_PART = re.compile(rf"^what\s+are\s+the\s+parts?\s+of\s+({_EL}?)[.?]?$", re.IGNORECASE)

# confidence model: exact fact near-certain, transitive decays with chain length
EXACT_CONF = 0.98


def norm(word: str) -> str:
    # preserve punctuation to match the stored entity format (en:... carries
    # dots/apostrophes); only lowercase + collapse whitespace.
    return "en:" + " ".join(word.lower().split())


def _forms(word: str) -> list[str]:
    """Candidate entity forms to match against the KB.  ConceptNet stores
    'en:...' canonical entities; the Wikidata ingest stores RAW English labels
    ('Douglas Adams').  A query can originate from either format, so lookups
    try both the raw form and the en:-normalized form."""
    raw = " ".join(word.split())
    n = norm(word)
    return [n] if n == raw else [raw, n]


def _aligned_pairs(s_raw: str, o_raw: str):
    """(raw, raw) and (norm, norm) entity pairs.  NOT the cross product: a KB
    stores entities in ONE format, so a mixed (raw start, en: target) pair can
    never match -- and trying it triggers a full-graph BFS with no early exit
    (the target never matches), which is catastrophic on ConceptNet."""
    yield s_raw, o_raw
    sn, on = norm(s_raw), norm(o_raw)
    if (sn, on) != (s_raw, o_raw):
        yield sn, on


def _known_pair(store, rels, s_raw: str, o_raw: str):
    """known_any across aligned entity-form pairs; returns the matched (s, o)
    form (for MRep entities) or None."""
    for s, o in _aligned_pairs(s_raw, o_raw):
        if store.known_any(rels, s, o):
            return s, o
    return None


def _transitive_any(store, rels, s_raw: str, o_raw: str) -> bool:
    for s, o in _aligned_pairs(s_raw, o_raw):
        if store.transitive_any(rels, s, o):
            return True
    return False


def _reachable(store, rels, s_raw: str) -> set[str]:
    out: set[str] = set()
    for s in _forms(s_raw):
        out |= store.reachable_any(rels, s, max_hops=1)
    return out


def _chain_confidence(k: int) -> float:
    return round(EXACT_CONF ** (k + 1), 4)


def answer(store: KnowledgeStore, question: str,
           rels: dict[str, list[str]] | None = None) -> MRep:
    rels = rels or CONCEPTNET_RELS
    is_a = rels["IsA"]
    q = question.strip()

    # ---- relation-specific questions FIRST (more specific than _WHAT_IS /
    # _HAS_PROP, which would swallow them: "what is the occupation of X") ----
    def _explain(subj, relname, rows, verb):
        labels = sorted(rows)[:8]
        text = ", ".join(k[3:] if k.startswith("en:") else k for k in labels)
        reps = [Source(relname, subj, k) for k in labels]
        return MRep(answer_type=EXPLANATION, question=question, answer=text,
                    confidence=EXACT_CONF, entities=[subj] + labels,
                    relations=[(relname, subj, k) for k in labels],
                    reasoning=[f"{subj} {verb} {k[3:] if k.startswith('en:') else k}"
                               for k in labels], sources=reps)

    m = _WHO_WROTE.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(rels.get("Author", ["P50"]), subj)
        if rows:
            return _explain(subj, "Author", rows, "was written by")
        return MRep.honest_unknown(question)
    m = _WHAT_OCC.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(rels.get("Occupation", ["P106"]), subj)
        if rows:
            return _explain(subj, "Occupation", rows, "works as a")
        return MRep.honest_unknown(question)
    m = _WHERE_IS.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(rels.get("LocatedIn", ["P131"]), subj)
        rows += store.successors_any(rels.get("Country", ["P17"]), subj)
        if rows:
            return _explain(subj, "LocatedIn", rows, "is located in")
        return MRep.honest_unknown(question)
    m = _WHAT_COUNTRY.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(rels.get("Country", ["P17"]), subj)
        if rows:
            return _explain(subj, "Country", rows, "is in the country")
        return MRep.honest_unknown(question)
    m = _WHAT_PARTOF.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(rels.get("PartOf", ["P361"]), subj)
        if rows:
            return _explain(subj, "PartOf", rows, "is part of")
        return MRep.honest_unknown(question)
    m = _WHAT_HAS_PART.match(q)
    if m:
        subj = m.group(1)
        rows = store.successors_any(["P527"], subj)
        if rows:
            return _explain(subj, "HasPart", rows, "has part")
        return MRep.honest_unknown(question)

    m = _WHAT_IS.match(q)
    if m:
        subj = m.group(1)
        kinds = _reachable(store, is_a, subj)
        if kinds:
            labels = sorted(kinds)
            text = ", ".join(k[3:] if k.startswith("en:") else k for k in labels[:8])
            reps = [Source("IsA", subj, k) for k in sorted(kinds)[:8]]
            return MRep(answer_type=EXPLANATION, question=question, answer=text,
                        confidence=EXACT_CONF, entities=[subj] + list(kinds)[:8],
                        relations=[("IsA", subj, k) for k in sorted(kinds)[:8]],
                        reasoning=[f"{subj} is a {k[3:] if k.startswith('en:') else k}"
                                   for k in sorted(kinds)[:8]],
                        sources=reps)
        return MRep.honest_unknown(question)
    m = _DO_YOU_KNOW.match(q)
    if m:
        subj = norm(m.group(1))
        known = (store.reachable_any(is_a, subj, max_hops=1)
                 or store.successors_any(rels.get("UsedFor", ["UsedFor"]), subj)
                 or store.successors_any(rels.get("HasProperty", ["HasProperty"]), subj))
        label = subj[3:] if subj.startswith("en:") else subj
        if known:
            return MRep.fact(question, f"Yes, I know {label}", "exists", subj, label)
        return MRep.honest_unknown(question)
    m = _USED_FOR.match(q)
    if m:
        subj, obj = norm(m.group(1)), norm(m.group(2))
        used = rels.get("UsedFor", ["UsedFor"])
        if store.known_any(used, subj, obj):
            return MRep.fact(question, "Yes", "UsedFor", subj, obj)
        return MRep.honest_unknown(question)
    m = _IS_A.match(q) or _IS_A2.match(q)
    if m:
        subj, obj = m.group(1), m.group(2)
        pair = _known_pair(store, is_a, subj, obj)
        if pair is not None:
            s, o = pair
            return MRep.fact(question, "Yes", "IsA", s, o)
        if _transitive_any(store, is_a, subj, obj):
            path = _transitive_path(store, subj, obj, is_a)
            return MRep.chain(question, "Yes", subj, "IsA", obj,
                              [(s, "IsA", o) for s, o in path],
                              _chain_confidence(len(path)))
        return MRep.honest_unknown(question)
    m = _HAS_PROP.match(q)
    if m:
        subj, prop = norm(m.group(1)), norm(m.group(2))
        hp = rels.get("HasProperty", ["HasProperty"])
        if store.known_any(hp, subj, prop):
            return MRep.fact(question, "Yes", "HasProperty", subj, prop)
        return MRep.honest_unknown(question)

    return MRep.honest_unknown(question)


def _transitive_path(store: KnowledgeStore, start: str, target: str,
                     is_a: list[str], max_hops: int = 4) -> list[tuple[str, str]]:
    """BFS path start -> ... -> target via is_a rels (for auditable reasoning),
    across entity-form candidates (raw + en:-normalized)."""
    from collections import deque

    def bfs(st, tg):
        frontier = deque([st])
        parent: dict[str, str] = {}
        seen = {st}
        while frontier:
            s = frontier.popleft()
            for e in store.successors_any(is_a, s):
                if e == tg:
                    parent[e] = s
                    path = [e]
                    while path[-1] != st:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                if e not in seen:
                    seen.add(e)
                    parent[e] = s
                    frontier.append(e)
        return []

    for st, tg in _aligned_pairs(start, target):
        path = bfs(st, tg)
        if path:
            return path
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--relmap", choices=["conceptnet", "wikidata"], default="conceptnet")
    args = parser.parse_args()
    rels = CONCEPTNET_RELS if args.relmap == "conceptnet" else WIKIDATA_RELS
    store = KnowledgeStore(args.db)
    print(f"knowledge base: {store.count():,} facts loaded from {args.db.name}")
    if args.question:
        rep = answer(store, args.question, rels=rels)
        print(json.dumps({"mrep": {
            "type": rep.answer_type, "answer": rep.answer,
            "confidence": rep.confidence, "unknown": rep.unknown,
            "sources": [(s.rel, s.subject, s.object) for s in rep.sources],
        }, "rendered": render(rep)}, ensure_ascii=False, indent=1))
    else:
        demo = [
            "is an apple a fruit",
            "is a dog an animal",
            "is a dog a planet",
            "what is an apple",
            "is a car used for transportation",
            "do you know a giraffe",
            "is the sky green",
        ]
        for q in demo:
            rep = answer(store, q, rels=rels)
            print(f"Q: {q}\n  -> {render(rep)}\n     (type={rep.answer_type}, "
                  f"conf={rep.confidence}, unknown={rep.unknown})")
    store.close()


if __name__ == "__main__":
    main()
