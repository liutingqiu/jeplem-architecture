"""Train + save the realizer used by the deployed knowledge assistant.

The service loads the cached model at startup (instant) and renders fact
answers with the LEARNED fluent language layer instead of templates.
Run once; the service picks up artifacts/realizer_cache.pt.
"""
from __future__ import annotations

import sqlite3
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from experiments.arc.run_realizer import sample_facts, train_realizer

DB = ROOT / "data" / "knowledge" / "wikidata.db"
OUT = ROOT / "artifacts" / "realizer_cache.pt"


def main() -> None:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    facts = sample_facts(conn, "P31", None, 2000 // 3, 1)
    facts += sample_facts(conn, "P106", None, 2000 // 3, 2)
    facts += sample_facts(conn, "P17", None, 2000 // 3, 3)
    conn.close()
    print(f"training realizer on {len(facts)} facts...")
    model, w2i, i2w, pad_id = train_realizer(facts, seed=444, hidden=128, epochs=4)
    torch.save({"model": model.state_dict(), "w2i": w2i, "i2w": i2w},
               OUT, _use_new_zipfile_serialization=False)
    print(f"saved {OUT}  (vocab {len(w2i)})")


if __name__ == "__main__":
    main()
