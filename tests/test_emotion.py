"""Emotion / Effort / TimeSense 单测。"""
import math
import unittest

from plwm.emotion import Effort, Emotion, TimeSense


class EmotionTests(unittest.TestCase):
    def test_positive_outcome_drives_positive_valence(self):
        emotion = Emotion()
        for _ in range(20):
            emotion.observe(reward=1.0)
        self.assertGreater(emotion.state.valence, 0.5)

    def test_negative_outcome_drives_negative_valence(self):
        emotion = Emotion()
        for _ in range(20):
            emotion.observe(reward=-1.0)
        self.assertLess(emotion.state.valence, -0.5)

    def test_surprise_drives_arousal(self):
        emotion = Emotion()
        for _ in range(20):
            emotion.observe(surprise=1.0)
        self.assertGreater(emotion.state.arousal, 0.5)

    def test_no_signal_keeps_state(self):
        emotion = Emotion()
        emotion.observe(reward=1.0)
        before = emotion.state
        emotion.observe()  # no signal
        self.assertEqual(emotion.state, before)

    def test_none_safe(self):
        emotion = Emotion()
        # all None must not crash and keeps initial state
        state = emotion.observe()
        self.assertEqual(state.valence, 0.0)
        self.assertEqual(state.arousal, 0.0)

    def test_state_matches_current(self):
        emotion = Emotion()
        emotion.observe(reward=1.0)
        state = emotion.observe(reward=1.0)
        self.assertEqual(state, emotion.state)


class EffortTests(unittest.TestCase):
    def test_spend_increases_and_decays(self):
        effort = Effort(decay=0.5)
        effort.spend(1.0)
        self.assertGreater(effort.level, 0.5)
        level = effort.level
        effort.tick()
        self.assertLess(effort.level, level)
        self.assertGreater(effort.level, 0.0)

    def test_level_bounded(self):
        effort = Effort()
        for _ in range(100):
            effort.spend(10.0)
        self.assertLessEqual(effort.level, 1.0)


class TimeSenseTests(unittest.TestCase):
    def test_elapsed_accumulates(self):
        ts = TimeSense()
        ts.tick(1.0)
        ts.tick(1.5)
        self.assertAlmostEqual(ts.elapsed, 2.5)

    def test_stamp_and_since(self):
        ts = TimeSense()
        ts.tick(1.0)
        t0 = ts.stamp()
        ts.tick(3.0)
        self.assertAlmostEqual(ts.since(t0), 3.0)

    def test_recency_decays_with_age(self):
        ts = TimeSense(tau=10.0)
        ts.tick(1.0)
        old = ts.stamp()
        ts.tick(20.0)
        recent = ts.stamp()
        self.assertLess(ts.recency(old), ts.recency(recent))
        self.assertAlmostEqual(ts.recency(recent), 1.0)
        # old event has strongly decayed
        self.assertLess(ts.recency(old), math.exp(-1.0))


class BodyEmotionIntegration(unittest.TestCase):
    def test_multimodal_body_feel_wires_emotion(self):
        from plwm.emotion import Effort, Emotion, TimeSense
        from plwm.fact_ledger import VersionedFactLedger
        from plwm.multimodal_body import MultimodalBody
        from plwm.runtime import PersistentRuntime
        from plwm.semantic_space import ModalityEncoder, SemanticSpace

        space = SemanticSpace(dim=16)
        space.register("vision", ModalityEncoder(2, dim=16))
        runtime = PersistentRuntime(16, belief_states=2, latent_dim=16)
        ledger = VersionedFactLedger(trusted_sources={"vision"}, revision_confidence=0.9)
        emotion, effort, ts = Emotion(), Effort(), TimeSense()
        body = MultimodalBody("w", space, runtime, ledger=ledger,
                              emotion=emotion, effort=effort, time_sense=ts)
        # a THINK (effort) + good outcome -> arousal up + valence positive
        body.feel(reward=1.0, surprise=0.8, think=True)
        self.assertGreater(emotion.state.valence, 0.0)
        self.assertGreater(emotion.state.arousal, 0.0)
        self.assertGreater(effort.level, 0.0)
        self.assertGreater(ts.elapsed, 0.0)

    def test_runtime_emotion_from_gap(self):
        from plwm.emotion import Emotion, TimeSense
        from plwm.runtime import PersistentRuntime

        emotion, ts = Emotion(), TimeSense()
        runtime = PersistentRuntime(4, belief_states=2, latent_dim=4,
                                    emotion=emotion, time_sense=ts)
        runtime.ingest([0.1, 0.2, 0.3, 0.4])
        # gap = 1 - confidence drives arousal; time ticks on ingest
        self.assertGreater(ts.elapsed, 0.0)
        # with low confidence, arousal should rise
        self.assertGreater(emotion.state.arousal, 0.0)


if __name__ == "__main__":
    unittest.main()
