import unittest

import torch

from experiments.audit_learned_fact_revision import (
    EVENT_DIM,
    FourEventRevisionTransformer,
    GRURevision,
    LearnedFactStore,
    _corpus,
)


class LearnedFactRevisionTests(unittest.TestCase):
    def test_corpus_contains_only_visible_event_fields_and_query(self):
        events, labels = _corpus(examples=12, seed_start=3, distractors=8)
        self.assertEqual(events.shape, (12, 17, EVENT_DIM))
        self.assertEqual(labels.shape, (12,))
        self.assertTrue(torch.all(events[:, -1, -1] == 1.0))
        self.assertTrue(torch.all(events[:, :-1, -1] == 0.0))
        self.assertTrue(torch.all((0 <= labels) & (labels < 4)))

    def test_models_return_value_logits(self):
        events, _labels = _corpus(examples=3, seed_start=10, distractors=8)
        for model in (LearnedFactStore(), LearnedFactStore(use_fact_addressing=False), GRURevision(), FourEventRevisionTransformer()):
            self.assertEqual(tuple(model(events).shape), (3, 4))

    def test_inconsistent_distractor_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            _corpus(examples=2, seed_start=1, distractors=3)

    def test_invalid_conflict_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _corpus(examples=2, seed_start=1, distractors=8, conflict_mode="unknown")


if __name__ == "__main__":
    unittest.main()
