"""Modality-identify prior (B8, 2026-08-12): innate bias to route input to the
right lens.

The real world presents content of many modalities (text, image, audio,
video, code, structured data).  For the architecture to explore on its own --
with no guidance per content type -- it needs an innate MODALITY RECOGNIZER:
given a raw input, decide WHICH lens (prior) applies, then route to it.

This is a learning BIAS (how to recognize input type), not knowledge: it
identifies 'this is an image' but does not know what the image means.  The
meaning is grown by the surprise engine through the routed lens.

Recognition signals (deliberately simple and modular):
  - bytes with known magic numbers -> image (PNG/JPEG/GIF), audio (WAV/MP3),
    video (MP4/WebM), PDF
  - text: high printable-char ratio, whitespace, no binary
  - code: text with programming keywords + indentation structure
  - JSON/structured: parseable JSON
  - else: unknown

Returns the modality tag + the routed feature extractor, so the rest of the
pipeline (PerceptionHub) can dispatch without knowing content types a priori.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.language_prior import language_features, bigram_features
from plwm.image_prior import visual_features
from plwm.audio_prior import audio_features
from plwm.code_prior import code_features

MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image",
    b"\xff\xd8\xff": "image",           # JPEG
    b"GIF8": "image",
    b"RIFF": "audio",                    # WAV
    b"ID3": "audio",                     # MP3
    b"\x1aE\xdf\xa3": "video",           # WebM/Matroska
    b"\x00\x00\x00\x18ftyp": "video",    # MP4
    b"%PDF": "document",
}

CODE_KEYWORD_HINT = re.compile(
    r"\b(def|class|function|import|return|for|while|if|else|const|let|var|fn|use)\b",
    re.MULTILINE,
)


def identify(raw) -> tuple[str, str]:
    """Return (modality, subtype) for a raw input.
    raw: bytes, str, numpy array, or dict."""
    # dicts / structured
    if isinstance(raw, dict):
        if "cells" in raw:
            return "spatial", "luanti_vision"
        if "text" in raw:
            return "text", "dict_text"
        return "structured", "dict"
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], str):
            return "spatial", "tile_list"
        return "structured", "list"
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            return "audio", "waveform"
        return "image", "ndarray"
    # bytes: magic-number sniff
    if isinstance(raw, (bytes, bytearray)):
        head = bytes(raw[:16])
        for magic, mod in MAGIC.items():
            if head.startswith(magic):
                return mod, "bytes"
        # try JSON
        try:
            json.loads(raw.decode("utf-8"))
            return "structured", "json_bytes"
        except Exception:
            pass
        # else: decode as text and fall through
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return "unknown", "bytes"
    # str: text or code
    if isinstance(raw, str):
        if CODE_KEYWORD_HINT.search(raw):
            return "code", "text"
        # JSON string?
        try:
            json.loads(raw)
            return "structured", "json_str"
        except Exception:
            pass
        return "text", "plain"
    return "unknown", str(type(raw))


def route(raw) -> tuple[str, dict | None]:
    """Identify modality and return (modality, feature_dict)."""
    mod, subtype = identify(raw)
    if mod == "text":
        feats = language_features(raw)
        feats.update(bigram_features(raw))
        return mod, feats
    if mod == "code":
        return mod, code_features(raw)
    if mod == "image" and isinstance(raw, np.ndarray):
        return mod, visual_features(raw)
    if mod == "audio" and isinstance(raw, np.ndarray):
        return mod, audio_features(raw)
    if mod == "spatial" and isinstance(raw, dict) and "cells" in raw:
        from plwm.spatial_prior import spatial_features, parse_vision
        from plwm.number_prior import vision_tile_counts
        feats = spatial_features(raw)
        feats.update(vision_tile_counts(raw.get("cells", [])))
        return mod, feats
    return mod, None


def identify_batch(inputs: list) -> dict:
    """Identify a batch of mixed inputs; count modalities (routing test)."""
    counts = {}
    for raw in inputs:
        mod, _ = identify(raw)
        counts[mod] = counts.get(mod, 0) + 1
    return counts


if __name__ == "__main__":
    # mixed input test
    from plwm.image_prior import make_sample_images
    from plwm.audio_prior import make_tones
    from plwm.code_prior import make_sample_code
    mixed = [
        "hello world this is text",
        np.zeros((8, 8), dtype=int),
        make_tones()[0],
        make_sample_code()[0],
        b"\x89PNG\r\n\x1a\nfake-image-bytes",
        '{"a": 1, "b": 2}',
    ]
    for raw in mixed:
        mod, sub = identify(raw)
        print(f"  -> {mod} ({sub})")
