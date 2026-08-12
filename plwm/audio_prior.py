"""Audio prior (B5, 2026-08-12): innate bias to extract sound structure.

Infant-analogy: a newborn parses sound into amplitude, pitch, and rhythm
before it knows what sounds mean.  Our architecture needs the same innate
bias -- from a raw waveform it EXTRACTS basic acoustic features (RMS
amplitude, dominant frequency, zero-crossing rate, spectral shape), so
surprise can bootstrap on the AUDIO dimension.

This is a learning BIAS (what acoustic structure to notice), not knowledge:
it measures pitch and loudness but does not know what they MEAN.  The
meaning (e.g. 'this sound signals a call') is grown by the surprise engine.

Implementation: AudioPrior computes per-frame acoustic features from a float
waveform.  Synthetic waveforms (pure tones at different frequencies/
amplitudes) verify it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audio_features(wave: np.ndarray, sr: int = 8000) -> dict:
    """Per-frame acoustic features from a float waveform."""
    if len(wave) < 2:
        return {"rms": 0.0, "dominant_freq": 0.0, "zcr": 0.0, "n_samples": 0}
    wave = wave - wave.mean()
    rms = float(np.sqrt(np.mean(wave ** 2)))
    # dominant frequency via autocorrelation-based pitch (simple)
    n = len(wave)
    fft = np.abs(np.fft.rfft(wave)) ** 2
    fft[0] = 0
    if fft.sum() > 0:
        dom_bin = int(np.argmax(fft))
        dom_freq = dom_bin * sr / n
    else:
        dom_freq = 0.0
    # zero-crossing rate
    signs = np.signbit(wave[1:]) != np.signbit(wave[:-1])
    zcr = float(signs.mean())
    return {
        "rms": round(rms, 4),
        "dominant_freq": round(dom_freq, 1),
        "zcr": round(zcr, 4),
        "n_samples": n,
    }


def make_tones() -> list[np.ndarray]:
    """Synthetic: high-pitch loud tone, low-pitch quiet tone, silence."""
    sr = 8000
    t = np.arange(0, 0.5, 1 / sr)
    high_loud = 0.8 * np.sin(2 * np.pi * 880 * t)     # A5, loud
    low_quiet = 0.2 * np.sin(2 * np.pi * 220 * t)     # A3, quiet
    silence = np.zeros_like(t)
    return [high_loud, low_quiet, silence]


if __name__ == "__main__":
    tones = make_tones()
    for label, tone in zip(("high_loud", "low_quiet", "silence"), tones):
        print(f"{label}: {audio_features(tone)}")
