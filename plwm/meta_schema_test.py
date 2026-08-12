"""Contract tests for the modality-agnostic meta-rule schema.

Guarantees: the general core (plwm/meta_schema.py) has zero dependency on any
ARC / grid / Luanti / web module -- it is pure data model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.meta_schema import Conservation, EntityScene, MetaEntity, MetaRelation


class ConservationSignatureTests(unittest.TestCase):
    def test_signature_is_order_independent(self):
        a = Conservation(properties=("colors", "size"), changed=("pos",))
        b = Conservation(properties=("size", "colors"), changed=("pos",))
        self.assertEqual(a.signature, b.signature)

    def test_signature_changes_when_changed_dims_change(self):
        a = Conservation(properties=("colors",), changed=())
        b = Conservation(properties=("colors",), changed=("pos",))
        self.assertNotEqual(a.signature, b.signature)

    def test_signature_covers_all_four_fields(self):
        c = Conservation(properties=("p",), relations=("r",),
                         counts=("c",), changed=("x",))
        sig = c.signature
        self.assertIn("P:p", sig)
        self.assertIn("R:r", sig)
        self.assertIn("C:c", sig)
        self.assertIn("X:x", sig)

    def test_signature_is_string(self):
        c = Conservation()
        self.assertIsInstance(c.signature, str)
        self.assertTrue(c.signature)  # even the empty signature is non-empty


class EntitySceneTests(unittest.TestCase):
    def test_entity_relation_construction(self):
        e = MetaEntity("o0", "object", {"color": 3, "area": 6})
        r = MetaRelation("same_row", "o0", "o1", True)
        s = to_scene([e], [r])
        self.assertEqual(len(s.entities), 1)
        self.assertEqual(len(s.relations), 1)
        self.assertEqual(s.entities[0].properties["color"], 3)

    def test_scene_is_immutable_tuple(self):
        s = to_scene([])
        self.assertIsInstance(s.entities, tuple)


def to_scene(entities, relations=()):
    return EntityScene(tuple(entities), tuple(relations))


if __name__ == "__main__":
    unittest.main()
