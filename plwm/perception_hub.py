"""PerceptionHub (A3, 2026-08-12): the assembled innate-prior perception layer.

Combines ALL innate priors into one perception front-end, so the surprise
engine sees a rich, unified feature stream from real multimodal input:

    raw input
      -> [NumberPrior]     quantity features (tile/object counts)
      -> [SpatialPrior]    spatial relations (dist/direction/adjacency)
      -> [ImagePrior]      visual features (regions/edges/color counts)
      -> [AgentPrior]      autonomous-agent detection
      -> [SafetyPrior]     danger flags/aversion
      -> [CuriosityPrior]  priority over which surprise to chase
      -> merged feature dict -> SurpriseEngine (bootstrap)

The hub is the 'learning readiness' the architecture brings to a real world:
it counts, locates, sees, detects agents, avoids danger, and knows what to be
curious about -- all without any knowledge of WHAT things mean.  Meaning is
grown by the surprise engine on top of these features.

This module is the assembly point: each prior stays independent and swappable,
but together they give the engine a complete set of lenses.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.number_prior import vision_tile_counts, object_counts
from plwm.spatial_prior import spatial_features
from plwm.image_prior import visual_features
from plwm.agent_prior import AgentPrior
from plwm.safety_prior import SafetyPrior
from plwm.curiosity_prior import CuriosityPrior
from plwm.language_prior import language_features, bigram_features
from plwm.surprise_engine import SurpriseEngine


@dataclass
class PerceptionHub:
    """Assembled innate-prior perception front-end."""

    agents: AgentPrior = field(default_factory=AgentPrior)
    safety: SafetyPrior = field(default_factory=SafetyPrior)
    curiosity: CuriosityPrior = field(default_factory=CuriosityPrior)

    def perceive(self, raw: dict) -> dict:
        """raw: a dict describing one observation frame.  Returns the merged
        feature dict.

        Supported raw shapes (dispatch on content):
          - {'vision': vision}      : Luanti 11x11 vision -> spatial + number
          - {'image': np.ndarray}   : pixel matrix -> image features
          - {'objects': [...], 'positions': {...}} : object list + positions
          - {'signals': {...}}      : scalar signal channels -> safety
        """
        feats = {}
        if "vision" in raw:
            feats.update(spatial_features(raw["vision"]))
            cells = raw["vision"].get("cells") if isinstance(raw["vision"], dict) else raw["vision"]
            feats.update(vision_tile_counts(cells))
        if "image" in raw:
            feats.update(visual_features(np.asarray(raw["image"])))
        if "objects" in raw:
            feats.update(object_counts(raw["objects"]))
        if "positions" in raw and raw["positions"]:
            agent_scores = self.agents.observe_frame(raw["positions"])
            for oid, score in agent_scores.items():
                feats[f"agent_{oid}"] = round(score, 3)
        if "signals" in raw:
            danger = self.safety.observe(raw["signals"])
            for ch, score in danger.items():
                feats[f"danger_{ch}"] = round(score, 3)
        if "text" in raw:
            feats.update(language_features(raw["text"]))
            feats.update(bigram_features(raw["text"]))
        return feats

    def note_explained(self, field: str) -> None:
        """Tell the curiosity prior a field got a rule (its surprise is being
        resolved)."""
        self.curiosity.observe_field(field, True, True)


def run_on_stream(hub: PerceptionHub, engine: SurpriseEngine,
                  stream: list[dict]) -> dict:
    """Run the hub + surprise engine over a stream of raw frames."""
    prev_feats = None
    for raw in stream:
        feats = hub.perceive(raw)
        if prev_feats is not None:
            rep = engine.observe(prev_feats, feats)
            # feed surprise into curiosity
            for f, s in rep.items():
                hub.curiosity.observe_field(f, s > 0.5, False)
        prev_feats = feats
    return {
        "engine_knowledge": engine.knowledge(),
        "curiosity_top": hub.curiosity.top(5),
        "safety_flags": {k: round(v, 2) for k, v in
                         hub.safety._worst.items()} if hasattr(hub.safety, "_worst") else {},
    }


def make_luanti_stream(episodes: list[dict]) -> list[dict]:
    """Build raw frames from composite Luanti episodes (visions + states)."""
    frames = []
    for ep in episodes:
        visions = ep.get("visions") or []
        for v in visions:
            frames.append({"vision": v})
    return frames


if __name__ == "__main__":
    import json
    episodes = [json.loads(l) for l in
                Path(ROOT / "artifacts" / "luanti-composite-rollouts.jsonl").open(encoding="utf-8")]
    hub = PerceptionHub()
    engine = SurpriseEngine(window=6)
    stream = make_luanti_stream(episodes)
    result = run_on_stream(hub, engine, stream)
    print("engine knowledge:", result["engine_knowledge"])
    print("curiosity top:", result["curiosity_top"])
