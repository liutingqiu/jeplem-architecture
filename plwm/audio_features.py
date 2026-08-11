"""纯标准库的 WAV 解码 + 波形特征提取。

decode_wav:io.BytesIO + wave 模块读 16-bit PCM,1/2 声道(立体声取均值)。
压缩/异常 WAV → WebFetchError(严格门)。

WaveFeatureExtractor:确定性 [0,1] 特征,正弦/方波/噪声三者特征 L1 距离
显著(身体能区分不同的声音)。
"""
from __future__ import annotations

import io
import wave

import torch
from torch import Tensor

from .web import WebFetchError


def decode_wav(data: bytes) -> tuple[list[int], int]:
    """解码 WAV → (samples 列表, sample_rate)。仅 16-bit PCM 1/2 声道。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            rate = handle.getframerate()
            if channels not in (1, 2):
                raise WebFetchError(f"unsupported WAV channels {channels}")
            if sampwidth != 2:
                raise WebFetchError(f"unsupported WAV sample width {sampwidth}")
            frames = handle.readframes(handle.getnframes())
    except wave.Error as error:
        raise WebFetchError(f"invalid WAV: {error}") from error
    if channels == 1:
        samples: list[int] = []
        for i in range(0, len(frames) - 1, 2):
            samples.append(int.from_bytes(frames[i:i + 2], "little", signed=True))
        return samples, rate
    # stereo: parse interleaved L/R, average per frame
    samples = []
    for i in range(0, len(frames) - 3, 4):
        left = int.from_bytes(frames[i:i + 2], "little", signed=True)
        right = int.from_bytes(frames[i + 2:i + 4], "little", signed=True)
        samples.append((left + right) // 2)
    return samples, rate


class WaveFeatureExtractor:
    """确定性波形特征:[1,16]。能区分正弦/方波/噪声。"""

    def extract(self, samples: list[int], sample_rate: int) -> Tensor:
        n = max(1, len(samples))
        duration = n / max(1, sample_rate)
        rms = (sum(s * s for s in samples) / n) ** 0.5
        peak = max(abs(s) for s in samples) if samples else 0
        # zero-crossing rate
        zcr = sum(1 for i in range(1, len(samples)) if samples[i - 1] < 0 <= samples[i]) / n
        # amplitude histogram (8 buckets of |x|/32768)
        buckets = [0.0] * 8
        for s in samples:
            buckets[min(7, int(abs(s) / 32768.0 * 8.0))] += 1.0
        hist = [b / n for b in buckets]
        # peak ratio: fraction of samples at high amplitude (square wave high)
        peak_ratio = sum(1 for s in samples if abs(s) > 0.8 * 32768) / n
        # smoothness: mean |delta| (sine low, noise high)
        smooth = sum(abs(samples[i] - samples[i - 1]) for i in range(1, len(samples))) / (n * 32768.0)
        features = [
            min(1.0, duration / 30.0),
            sample_rate / 48000.0,
            rms / 32768.0,
            peak / 32768.0,
            zcr,
            *hist,
            peak_ratio,
            smooth,
            1.0,
        ]
        return torch.tensor([features], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return 16
