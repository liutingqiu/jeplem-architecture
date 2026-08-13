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
# permissive (`[^.?]+`: anything but sentence enders) because ConceptNet
# entities carry arbitrary punctuation ("1,4 dioxane").  Leading articles are
# only consumed WITH their space (`a `/`an `/`the `) so an entity that starts
# with "a"/"the" is not truncated ("accepted" must not lose its leading "a").
_ENTITY = r"[^.?]+"
_ART = r"(?:a |an |the )?"
_WHAT_IS = re.compile(rf"^what\s+is\s+{_ART}\s*({_ENTITY}?)[.?]?$")
_DO_YOU_KNOW = re.compile(rf"^do\s+you\s+know\s+(?:about\s+)?{_ART}\s*({_ENTITY}?)[.?]?$")
_USED_FOR = re.compile(rf"^(?:is\s+)?({_ENTITY}?)\s+(?:is\s+)?used\s+for\s+{_ART}\s*({_ENTITY}?)[.?]?$")
_IS_A = re.compile(rf"^is\s+{_ART}\s*({_ENTITY}?)\s+an?\s+({_ENTITY}?)[.?]?$")
_IS_A2 = re.compile(rf"^([a-z][^.?]+?)\s+is\s+an?\s+({_ENTITY}?)[.?]?$")
_HAS_PROP = re.compile(rf"^(?:is\s+)?{_ART}\s*({_ENTITY}?)\s+is\s+({_ENTITY}?)[.?]?$")

# confidence model: exact fact near-certain, transitive decays with chain length
EXACT_CONF = 0.98


def norm(word: str) -> str:
    # preserve punctuation to match the stored entity format (en:... carries
    # dots/apostrophes); only lowercase + collapse whitespace.
    return "en:" + " ".join(word.lower().split())


def _chain_confidence(k: int) -> float:
    return round(EXACT_CONF ** (k + 1), 4)


def answer(store: KnowledgeStore, question: str,
           rels: dict[str, list[str]] | None = None) -> MRep:
    rels = rels or CONCEPTNET_RELS
    is_a = rels["IsA"]
    q = question.lower().strip()
    m = _WHAT_IS.match(q)
    if m:
        subj = norm(m.group(1))
        kinds = store.reachable_any(is_a, subj, max_hops=1)
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
        subj, obj = norm(m.group(1)), norm(m.group(2))
        if store.known_any(is_a, subj, obj):
            return MRep.fact(question, "Yes", "IsA", subj, obj)
        if store.transitive_any(is_a, subj, obj):
            return MRep.chain(question, "Yes", subj, "IsA", obj,
                              [(s, "IsA", o) for s, o in _transitive_path(store, subj, obj, is_a)],
                              _chain_confidence(_transitive_path_len(store, subj, obj, is_a)))
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
    """BFS path start -> ... -> target via is_a rels (for auditable reasoning)."""
    from collections import deque
    frontier = deque([start])
    parent: dict[str, str] = {}
    seen = {start}
    found = False
    while frontier and not found:
        s = frontier.popleft()
        for e in store.successors_any(is_a, s):
            if e == target:
                parent[e] = s
                found = True
                break
            if e not in seen:
                seen.add(e)
                parent[e] = s
                frontier.append(e)
    if not found:
        return []
    path = [target]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _transitive_path_len(store: KnowledgeStore, start: str, target: str,
                         is_a: list[str]) -> int:
    return len(_transitive_path(store, start, target, is_a))


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
