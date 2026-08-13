"""Language interface (first iteration): natural-ish questions -> the body's
knowledge base + explicit reasoning.

The body answers from its disk-backed KnowledgeStore with exact-match retrieval
AND transitive reasoning, and it is SELF-AWARE: it says "I don't know" when the
fact (or a reasoned derivation) is not in its knowledge.  This is the first
language adapter (pattern-based); deeper NLP is a later iteration.

Question patterns handled (regex):
  "is <subj> a <obj>" / "is a <subj> a <obj>"   -> IsA(subj,obj) exact + transitive
  "<subj> is used for <obj>" / "is <subj> used for <obj>" -> UsedFor(subj,obj)
  "is <subj> <prop>" / "<subj> is <prop>"       -> HasProperty(subj,prop)
  "what is <subj>" / "what is a <subj>"          -> return IsA(subj) reachable set
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

DEFAULT_DB = ROOT / "data" / "knowledge" / "conceptnet.db"

# patterns: (regex, handler) -- ordered specific-first; `an?` matches "an"
# greedily so "is an apple" does not become "n apple"
_WHAT_IS = re.compile(r"^what\s+is\s+(?:an?|the)?\s*([a-z0-9 _-]+?)[.?]?$")
_DO_YOU_KNOW = re.compile(r"^do\s+you\s+know\s+(?:about\s+)?(?:an?|the)?\s*([a-z0-9 _-]+?)[.?]?$")
_USED_FOR = re.compile(r"^(?:is\s+)?([a-z0-9 _-]+?)\s+(?:is\s+)?used\s+for\s+(?:an?|the)?\s*([a-z0-9 _-]+?)[.?]?$")
_IS_A = re.compile(r"^is\s+(?:an?|the)?\s*([a-z0-9 _-]+?)\s+an?\s+([a-z0-9 _-]+?)[.?]?$")
_IS_A2 = re.compile(r"^([a-z][a-z0-9 _-]+?)\s+is\s+an?\s+([a-z0-9 _-]+?)[.?]?$")
_HAS_PROP = re.compile(r"^(?:is\s+)?(?:the\s+)?([a-z0-9 _-]+?)\s+is\s+([a-z0-9 _-]+?)[.?]?$")

_WORD = re.compile(r"[a-z0-9]+")


def norm(word: str) -> str:
    return "en:" + " ".join(_WORD.findall(word.lower()))


def answer(store: KnowledgeStore, question: str) -> dict:
    q = question.lower().strip()
    m = _WHAT_IS.match(q)
    if m:
        subj = norm(m.group(1))
        kinds = store.reachable_sets("IsA", subj, max_hops=1)
        if kinds:
            labels = [k[3:] if k.startswith("en:") else k for k in sorted(kinds)[:8]]
            return {"question": question, "answer": ", ".join(labels),
                    "reason": "IsA reachable set", "confidence": "high"}
        return {"question": question, "answer": "I don't know", "reason": "no knowledge of it",
                "confidence": "honest-unknown"}
    m = _DO_YOU_KNOW.match(q)
    if m:
        subj = norm(m.group(1))
        known = (store.reachable_sets("IsA", subj, max_hops=1) or store.successors("UsedFor", subj)
                 or store.successors("HasProperty", subj))
        label = subj[3:] if subj.startswith("en:") else subj
        return {"question": question, "answer": "Yes, I know " + label if known else "No, I don't know " + label,
                "reason": "self-aware", "confidence": "high"}
    m = _USED_FOR.match(q)
    if m:
        subj, obj = norm(m.group(1)), norm(m.group(2))
        if store.known("UsedFor", subj, obj):
            return {"question": question, "answer": "Yes", "reason": "exact fact", "confidence": "high"}
        return {"question": question, "answer": "I don't know", "reason": "no such fact", "confidence": "honest-unknown"}
    m = _IS_A.match(q) or _IS_A2.match(q)
    if m:
        subj, obj = norm(m.group(1)), norm(m.group(2))
        exact = store.known("IsA", subj, obj)
        trans = store.transitive("IsA", subj, obj)
        if exact or trans:
            return {"question": question, "answer": "Yes",
                    "reason": "exact fact" if exact else "transitive (chained)",
                    "confidence": "high"}
        return {"question": question, "answer": "I don't know", "reason": "no such fact",
                "confidence": "honest-unknown"}
    m = _HAS_PROP.match(q)
    if m:
        subj, prop = norm(m.group(1)), norm(m.group(2))
        if store.known("HasProperty", subj, prop):
            return {"question": question, "answer": "Yes", "reason": "exact fact", "confidence": "high"}
        return {"question": question, "answer": "I don't know", "reason": "no such fact", "confidence": "honest-unknown"}
    return {"question": question, "answer": "I don't know", "reason": "question not understood",
            "confidence": "honest-unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()
    store = KnowledgeStore(args.db)
    print(f"knowledge base: {store.count():,} facts loaded from {args.db.name}")
    if args.question:
        print(json.dumps(answer(store, args.question), ensure_ascii=False, indent=1))
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
            print(json.dumps(answer(store, q), ensure_ascii=False))
    store.close()


if __name__ == "__main__":
    main()
