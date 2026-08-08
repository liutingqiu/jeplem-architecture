import unittest
from copy import deepcopy

from plwm.fact_ledger import FactRecord, LuantiRewardFactLedger, VersionedFactLedger


class FactLedgerTests(unittest.TestCase):
    def test_weak_conflict_isolated_and_strong_newer_evidence_revises(self):
        ledger = VersionedFactLedger(trusted_sources={"trusted"})
        self.assertEqual(ledger.observe(FactRecord("rule", "a", "trusted", 1, 1.0, 1)), "accepted")
        self.assertEqual(ledger.observe(FactRecord("rule", "b", "weak", 2, 0.2, 1)), "isolated-conflict")
        self.assertEqual(ledger.current["rule"].value, "a")
        self.assertEqual(ledger.observe(FactRecord("rule", "b", "trusted", 3, 1.0, 1)), "revised")
        self.assertEqual(ledger.current["rule"].version, 2)

    def test_snapshot_restores_isolated_and_current_records(self):
        ledger = VersionedFactLedger(trusted_sources={"trusted"})
        ledger.observe(FactRecord("rule", True, "trusted", 1, 1.0, 1))
        ledger.observe(FactRecord("rule", False, "weak", 2, 0.2, 1))
        self.assertEqual(VersionedFactLedger.restore(ledger.snapshot()).snapshot(), ledger.snapshot())

    def test_luanti_reward_adapter_requires_provenance_and_maps_only_prerequisites(self):
        adapter = LuantiRewardFactLedger()
        charge = {"value": 0.15, "_plwm_provenance": {"source": "luanti-reward", "tick": 5, "request_id": "charge"}}
        key = {"value": 0.2, "_plwm_provenance": {"source": "luanti-reward", "tick": 8, "request_id": "key"}}
        self.assertEqual(adapter.observe_reward(charge), "accepted")
        self.assertEqual(adapter.observe_reward(key), "accepted")
        self.assertTrue(adapter.ledger.current["charge_verified"].value)
        self.assertTrue(adapter.ledger.current["key_verified"].value)
        with self.assertRaises(ValueError):
            adapter.observe_reward({"value": 0.15})

    def test_restore_rejects_tampered_current_key_and_history(self):
        ledger = VersionedFactLedger(trusted_sources={"trusted"})
        ledger.observe(FactRecord("rule", "a", "trusted", 1, 1.0, 1))
        snapshot = ledger.snapshot()
        wrong_key = deepcopy(snapshot)
        wrong_key["current"] = {"other": wrong_key["current"]["rule"]}
        with self.assertRaises(ValueError):
            VersionedFactLedger.restore(wrong_key)
        forged = deepcopy(snapshot)
        forged["current"]["rule"]["value"] = "forged"
        with self.assertRaises(ValueError):
            VersionedFactLedger.restore(forged)
        untrusted = deepcopy(snapshot)
        untrusted["history"][0]["source"] = "untrusted"
        with self.assertRaises(ValueError):
            VersionedFactLedger.restore(untrusted)


if __name__ == "__main__":
    unittest.main()
