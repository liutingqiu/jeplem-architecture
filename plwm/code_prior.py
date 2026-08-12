"""Code prior (B7, 2026-08-12): innate bias to extract symbolic structure.

The architecture needs an innate bias for code-like content -- from raw
source text it EXTRACTS structural features (indentation depth, identifier
density, keyword presence, syntactic element counts) without knowing what
the code MEANS.  This lets the surprise engine bootstrap on the CODE
dimension (e.g. 'functions are nested under def', 'control flow uses
keywords').

This is a learning BIAS (what syntactic structure to notice), not knowledge:
it counts indentation and keywords but does not know the program's behavior.

Implementation: CodePrior tokenizes source into structural features -- line
count, indent depth (max/mean), keyword hits, identifier density, bracket
balance.  A minimal keyword set covers common languages.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while", "import",
    "from", "function", "var", "let", "const", "public", "private", "static",
    "int", "void", "struct", "fn", "let", "use", "package", "print", "echo",
}
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def code_features(source: str) -> dict:
    lines = source.splitlines()
    if not lines or all(not l.strip() for l in lines):
        return {"n_lines": 0, "max_indent": 0, "keyword_density": 0.0,
                "ident_density": 0.0, "bracket_balance": 0, "n_blocks": 0}
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    max_indent = max(indents) if indents else 0
    toks = IDENT_RE.findall(source)
    n_ident = len(toks)
    kw = [t for t in toks if t in KEYWORDS]
    n_kw = len(kw)
    # bracket balance
    balance = source.count("{") - source.count("}") + source.count("(") - source.count(")")
    # block starts: lines whose stripped form ends with ':' or '{'
    n_blocks = sum(1 for l in lines
                   if l.strip() and (l.strip().endswith(":") or l.strip().endswith("{")))
    return {
        "n_lines": len(lines),
        "max_indent": max_indent,
        "keyword_density": round(n_kw / max(n_ident, 1), 3),
        "ident_density": round(n_ident / max(len(toks) + n_ident, 1), 3),
        "bracket_balance": balance,
        "n_blocks": n_blocks,
    }


def make_sample_code() -> list[str]:
    return [
        # a function-heavy snippet
        "def add(a, b):\n    return a + b\ndef main():\n    x = add(1, 2)\n    print(x)",
        # a class-heavy snippet
        "class Dog:\n    def bark(self):\n        return 'woof'\n    def run(self):\n        pass",
        # a loop-heavy snippet
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)\n    else:\n        continue",
    ]


if __name__ == "__main__":
    for s in make_sample_code():
        print(code_features(s))
