"""Contract tests for the relation library + rule ledger + inducer.

The generic core (plwm/relation_library.py, plwm/rule_ledger.py, plwm/inducer.py,
plwm/meta_engine.py) must have ZERO dependency on any arc_* module.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.relation_library import RelationFunc, relation_catalog, register_family1
from plwm.rule_ledger import CompiledRule, RuleLedger


class RelationCatalogTests(unittest.TestCase):
    def test_catalog_has_six_families(self):
        cats = relation_catalog()
        families = {c.family for c in cats}
        self.assertEqual(families, {
            "causality", "geometry", "interaction", "social", "logic",
        })
        for c in cats:
            self.assertIn("/", c.key)

    def test_register_family1_injects(self):
        catalog = relation_catalog()
        def _fit(a, b):
            return {"m": 1}
        def _apply(p, a):
            return a
        register_family1(catalog, {"map_color": ("quantity", ("colors",), _fit, _apply)})
        keys = {c.key for c in catalog}
        self.assertIn("quantity/map_color", keys)
        # no arc_* import in the generic core modules
        for modname in ("plwm.relation_library", "plwm.rule_ledger",
                        "plwm.inducer", "plwm.meta_engine"):
            mod = __import__(modname, fromlist=["*"])
            self.assertNotIn("arc_", getattr(mod, "__file__", "") or "")
            import re
            src = Path(mod.__file__).read_text(encoding="utf-8")
            imports = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
            self.assertFalse(any("arc_" in l for l in imports),
                             f"{modname} imports arc_*: {imports}")


class RuleLedgerTests(unittest.TestCase):
    def _mk(self, sig="P:colors|X:pos"):
        return CompiledRule(signature=sig, rule={"kind": "meta", "relation": "map_color", "params": {1: 2}}, relation_key="quantity/map_color")

    def test_compile_then_hit(self):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta"}, "quantity/map_color")
        rule = ledger.lookup("SIG", lambda r: True)
        self.assertIsNotNone(rule)
        self.assertEqual(rule.n_confirm, 1)

    def test_miss_on_verify_fail(self):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta"}, "quantity/map_color")
        self.assertIsNone(ledger.lookup("SIG", lambda r: False))

    def test_single_contradiction_does_not_mark(self):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta"}, "q/r")
        ledger._contradict("SIG")
        self.assertFalse(ledger.is_stale("SIG"))

    def test_majority_marks_stale(self):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta"}, "q/r")
        for _ in range(4):
            ledger._contradict("SIG")
        self.assertTrue(ledger.is_stale("SIG"))
        # stale rules are NOT deleted, just marked
        self.assertIn("SIG", ledger.current)

    def test_revision_bumps_version(self):
        ledger = RuleLedger()
        r1 = ledger.compile("SIG", {"kind": "meta", "v": 1}, "q/r")
        # mark stale, then re-compile = revision
        for _ in range(4):
            ledger._contradict("SIG")
        self.assertTrue(ledger.is_stale("SIG"))
        r2 = ledger.compile("SIG", {"kind": "meta", "v": 2}, "q/r")
        self.assertEqual(r2.version, 2)
        self.assertGreater(r2.version, r1.version)
        self.assertFalse(ledger.is_stale("SIG"))  # stale cleared on revision

    def test_snapshot_restore_roundtrip(self, tmp_path=None):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta", "params": {"a": 1}}, "q/r")
        path = ROOT / "artifacts" / "_test_rules.json"
        ledger.save(path)
        ledger2 = RuleLedger()
        ledger2.restore(path)
        self.assertEqual(ledger2.stats()["n_rules"], 1)
        rule = ledger2.lookup("SIG", lambda r: True)
        self.assertEqual(rule.rule["params"]["a"], 1)
        path.unlink()

    def test_restore_rejects_tampered_signature(self):
        ledger = RuleLedger()
        ledger.compile("SIG", {"kind": "meta"}, "q/r")
        path = ROOT / "artifacts" / "_test_tamper.json"
        ledger.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        # tamper: change the stored signature key
        data["current"]["OTHER"] = data["current"].pop("SIG")
        path.write_text(json.dumps(data), encoding="utf-8")
        ledger2 = RuleLedger()
        with self.assertRaises(ValueError):
            ledger2.restore(path)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
