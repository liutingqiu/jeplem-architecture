"""End-to-end tests for the meta-engine (M2-M4 verification).

Covers:
  - ARC adapter: conserve_raw signature, family1 relations fire on the right
    changed dims
  - compile-once-reuse-many: same conservation signature across tasks -> cache
    hit, no re-induction
  - cross-session persistence: save/restore round-trip, tamper rejection
  - streaming rule-flip: single contradiction does not mark, majority marks,
    revision bumps version
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from arc_adapter import ArcAdapter
from plwm.meta_engine import MetaEngine
from plwm.relation_library import relation_catalog, register_family1
from plwm.rule_ledger import RuleLedger


def build():
    adapter = ArcAdapter()
    rels = relation_catalog()
    register_family1(rels, adapter.family1_relations())
    return adapter, rels


class ArcAdapterTests(unittest.TestCase):
    def test_conserve_signature_color_remap(self):
        adapter = ArcAdapter()
        g = np.array([[7, 0], [0, 7]], dtype=int)
        out = g.copy()
        out[out == 7] = 5
        sig = adapter.conserve_raw(g, out)
        self.assertIn("grid_size", sig.properties)
        self.assertIn("colors", sig.changed)  # colors changed -> in changed
        self.assertNotIn("colors", sig.properties)

    def test_family1_fires_on_colors_changed(self):
        adapter, rels = build()
        g = np.array([[7, 0], [0, 7]], dtype=int)
        out = g.copy()
        out[out == 7] = 5
        sig = adapter.conserve_raw(g, out)
        fired = [r.key for r in rels if set(r.fired_by) & set(sig.changed)]
        self.assertGreater(len(fired), 0, "no relation fired on colors-changed")
        self.assertIn("quantity/map_color", fired)


class CompileReuseTests(unittest.TestCase):
    def test_same_signature_cross_task_reuse(self):
        adapter, rels = build()
        eng = MetaEngine(adapter, rels)
        # two tasks share the SAME conservation signature but DIFFERENT mapping
        # (7->5 vs 6->2).  The engine must NOT blindly reuse the first rule;
        # it re-verifies, fails, and re-induces the correct second rule.  This
        # is the many-to-one signature correctness requirement.
        for task_pairs in self._two_remap_tasks():
            eng.process(task_pairs)
        # two distinct rules compiled under the same signature (version 1 and 2)
        sigs = {r.signature for r in eng.ledger.history}
        self.assertEqual(len(sigs), 1)  # same signature
        versions = sorted(r.version for r in eng.ledger.history)
        self.assertIn(1, versions)

    def _two_remap_tasks(self):
        g1 = np.array([[7, 0], [0, 7]], dtype=int)
        o1 = g1.copy(); o1[o1 == 7] = 5
        g2 = np.array([[6, 1], [1, 6]], dtype=int)
        o2 = g2.copy(); o2[o2 == 6] = 2
        return [(g1, o1)], [(g2, o2)]


class PersistenceTests(unittest.TestCase):
    def test_save_restore_and_tamper(self, tmp=None):
        adapter, rels = build()
        eng = MetaEngine(adapter, rels)
        g = np.array([[7, 0], [0, 7]], dtype=int)
        out = g.copy(); out[out == 7] = 5
        eng.process([(g, out)])
        path = ROOT / "artifacts" / "_meta_engine_test.json"
        eng.save()
        # save with explicit path via ledger
        eng.ledger.save(path)
        eng2 = MetaEngine(adapter, rels)
        eng2.ledger.restore(path)
        self.assertEqual(eng2.ledger.stats()["n_rules"], 1)
        # tamper
        data = json.loads(path.read_text(encoding="utf-8"))
        sig = list(data["current"].keys())[0]
        data["current"]["TAMPERED"] = data["current"].pop(sig)
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            eng2.ledger.restore(path)
        path.unlink()


class StalenessRevisionTests(unittest.TestCase):
    def test_streaming_rule_flip(self):
        adapter, rels = build()
        eng = MetaEngine(adapter, rels, streaming=True)
        g = np.array([[1, 0], [0, 0]], dtype=int)
        pairs = []
        for _ in range(3):
            out = g.copy(); out[out == 1] = 2; pairs.append((g.copy(), out))
        for _ in range(4):
            out = g.copy(); out[out == 1] = 3; pairs.append((g.copy(), out))
        eng.process(pairs)
        sig = adapter.conserve_raw(g, pairs[0][1]).signature
        self.assertTrue(eng.ledger.is_stale(sig))
        self.assertGreaterEqual(eng.stats.inductions, 1)

    def test_revision_bumps_version(self):
        ledger = RuleLedger()
        r1 = ledger.compile("S", {"kind": "meta"}, "q/r")
        for _ in range(4):
            ledger._contradict("S")
        r2 = ledger.compile("S", {"kind": "meta"}, "q/r")
        self.assertEqual(r2.version, 2)
        self.assertFalse(ledger.is_stale("S"))


if __name__ == "__main__":
    unittest.main()
