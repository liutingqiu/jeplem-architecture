"""视频帧提取(ffmpeg) + 运动特征。

extract_frames:把视频 bytes 经 stdin 喂给 imageio_ffmpeg 的 ffmpeg,
用 -frames:v 限制帧数、scale 限分辨率,输出 PPM 管道,增量解析出 PIL
图像帧。帧数/时长不足 → WebFetchError(严格门)。

VideoFeatureExtractor:每帧用 PixelFeatureExtractor 算 16 维特征,聚合为
18 维 [帧均16, motion(相邻帧L1均值=运动量), head_tail(首尾帧L1)]。
motion 单独暴露,供"身体感知动态"判据。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg
import torch

from .image_features import PixelFeatureExtractor
from .web import WebFetchError


@dataclass
class VideoFrames:
    frames: list  # list[PIL.Image]
    duration: float


def extract_frames(data: bytes, *, n_frames: int = 8, max_dim: int = 160) -> VideoFrames:
    """从视频 bytes 提取 ≤n_frames 帧(时间均匀)。返回 (frames, duration)。"""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        process = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-i", "-",
             "-vf", f"scale={max_dim}:{max_dim}",
             "-frames:v", str(n_frames * 4),
             "-vcodec", "ppm", "-f", "image2pipe", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as error:
        raise WebFetchError(f"failed to start ffmpeg: {error}") from error
    stdout, stderr = process.communicate(data, timeout=60)
    if process.returncode != 0:
        raise WebFetchError(f"ffmpeg frame extraction failed: {stderr[:200]!r}")
    ppm_frames = _parse_ppm_stream(stdout)
    if len(ppm_frames) < 2:
        raise WebFetchError(f"too few frames decoded ({len(ppm_frames)})")
    duration = _parse_duration(stderr)
    # time-uniform subsample to n_frames
    sampled = _subsample(ppm_frames, n_frames)
    return VideoFrames(frames=sampled, duration=duration)


def _parse_ppm_stream(raw: bytes) -> list:
    """解析连续的 P6 PPM 流为 PIL 图像列表。

    头格式:P6\\n<w> <h>\\n<maxval>\\n。逐帧:magic、dims、maxval、像素。
    """
    from PIL import Image
    images: list = []
    pos = 0
    length = len(raw)
    while pos + 2 <= length and raw[pos:pos + 2] == b"P6":
        pos += 2
        # skip the newline right after the magic (P6\n)
        if raw[pos:pos + 1] == b"\n":
            pos += 1
        # dims line (may be preceded by comment lines)
        dims: list[str] = []
        for _ in range(4):
            nl = raw.index(b"\n", pos)
            line = raw[pos:nl].decode("ascii", errors="replace").strip()
            pos = nl + 1
            if line.startswith("#"):
                continue
            dims = line.split()
            break
        if len(dims) < 2:
            break
        width, height = int(dims[0]), int(dims[1])
        # maxval line
        nl = raw.index(b"\n", pos)
        pos = nl + 1
        pixel_bytes = width * height * 3
        if pos + pixel_bytes > length:
            break
        frame = raw[pos:pos + pixel_bytes]
        images.append(Image.frombytes("RGB", (width, height), frame))
        pos += pixel_bytes
    return images


def _subsample(frames: list, n: int) -> list:
    if len(frames) <= n:
        return frames
    step = (len(frames) - 1) / (n - 1) if n > 1 else 0
    return [frames[int(round(i * step))] for i in range(n)]


def _parse_duration(stderr: bytes) -> float:
    match = re.search(rb"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
    if not match:
        return 0.0
    h, m, s, frac = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s + frac / 100.0


class VideoFeatureExtractor:
    """视频级特征:[1,18]。帧均16 + 运动量 + 首尾差异。"""

    def __init__(self) -> None:
        self._frame_extractor = PixelFeatureExtractor()

    def extract(self, frames: list) -> list:
        """frames: list[PIL.Image]。返回 18 维视频特征列表。"""
        frame_features = [self._frame_extractor.extract(
            frame.width, frame.height,
            list(frame.getdata()),
        ) for frame in frames]
        if frame_features:
            mean = torch.stack(frame_features).mean(dim=0)  # [1,16]
        else:
            mean = torch.zeros(1, 16)
        motion = 0.0
        for i in range(1, len(frame_features)):
            motion += float((frame_features[i] - frame_features[i - 1]).abs().sum().item())
        motion = motion / max(1, len(frame_features) - 1)
        head_tail = 0.0
        if len(frame_features) > 1:
            head_tail = float((frame_features[-1] - frame_features[0]).abs().sum().item())
        return mean.flatten().tolist() + [motion, head_tail]

    @property
    def dim(self) -> int:
        return 18
