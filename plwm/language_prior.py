"""Language prior (B3, 2026-08-12): innate bias to extract text structure.

Infant-analogy: a newborn hears speech and pulls out word boundaries,
frequencies, and co-occurrences long before it knows what words mean.  Our
architecture needs the same innate bias -- from raw text it EXTRACTS basic
linguistic structure (token boundaries, frequencies, co-occurrence), so
surprise can bootstrap on the LANGUAGE dimension.

This is a learning BIAS (what textual structure to notice), not knowledge:
it counts words and co-occurrences but does not know what they MEAN.  The
meaning (e.g. 'this word signals a question') is grown by the surprise engine.

Implementation: LanguagePrior tokenizes text, computes per-token and per-doc
features (word count, unique ratio, token frequencies, bigram co-occurrence,
symbol density), producing a feature dict the surprise engine can run on.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:()\"]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def language_features(text: str) -> dict:
    """Per-text linguistic features for the surprise engine."""
    toks = tokenize(text)
    if not toks:
        return {"n_words": 0, "unique_ratio": 0.0, "word_freq": 0.0,
                "symbol_density": 0.0, "n_bigrams": 0}
    counts = Counter(toks)
    n = len(toks)
    words = [t for t in toks if t.isalnum()]
    n_words = len(words)
    n_symbols = n - n_words
    feats = {
        "n_words": n_words,
        "unique_ratio": round(len(counts) / n, 3),
        "top_freq": round(counts.most_common(1)[0][1] / n, 3),
        "symbol_density": round(n_symbols / n, 3),
        "n_bigrams": max(0, n - 1),
    }
    # per-token frequency of the most common word (structure signal)
    return feats


def bigram_features(text: str) -> dict:
    """Co-occurrence: count of distinct bigrams (structure/diversity)."""
    toks = tokenize(text)
    if len(toks) < 2:
        return {"n_distinct_bigrams": 0, "bigram_ratio": 0.0}
    bigrams = Counter(zip(toks, toks[1:]))
    n = len(toks) - 1
    return {
        "n_distinct_bigrams": len(bigrams),
        "bigram_ratio": round(len(bigrams) / n, 3),
    }


def text_stream(docs: list[str]) -> list[tuple[dict, dict]]:
    """Build a stream of language-feature pairs from consecutive docs."""
    pairs = []
    prev = None
    for doc in docs:
        cur = language_features(doc)
        cur.update(bigram_features(doc))
        if prev is not None:
            pairs.append((prev, cur))
        prev = cur
    return pairs


def make_sample_docs() -> list[str]:
    """Controlled docs: several 'question-like' and 'statement-like' texts."""
    return [
        "what is the capital of france ?",
        "where does the sun rise ?",
        "why do we sleep at night ?",
        "the cat sat on the mat .",
        "the dog ran in the park .",
        "the sun rose in the east .",
        "who invented the telephone ?",
        "how does gravity work ?",
    ]


if __name__ == "__main__":
    docs = make_sample_docs()
    for d in docs:
        print(f"{language_features(d)} | {bigram_features(d)}")
