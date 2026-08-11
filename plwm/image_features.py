"""纯标准库的真实图片解码 + 像素特征提取。

decode_png:zlib 解 IDAT,逐扫描线解 filter(0/1/2/3/4),支持 color_type
0(灰)/2(RGB)/4(灰+alpha)/6(RGBA)、8bit、非隔行。type3(调色板)/隔行
→ WebFetchError(严格门:不支持就当拉取失败,绝不让原始二进制进认知)。

PixelFeatureExtractor:确定性 [0,1] 特征,不同图片必有不同特征(颜色
直方图 + 尺寸 + 亮度)。
"""
from __future__ import annotations

import struct
import zlib

import torch
from torch import Tensor

from .web import WebFetchError


def decode_png(data: bytes) -> tuple[int, int, list[tuple[int, int, int]]]:
    """解码 PNG → (width, height, list[(r,g,b)])。仅非隔行 8bit。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise WebFetchError("not a PNG")
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IHDR":
            (width, height, bit_depth, color_type, compression,
             filter_method, interlace) = struct.unpack(">IIBBBBB", chunk_data)
            if bit_depth != 8:
                raise WebFetchError("only 8-bit PNG supported")
            if interlace != 0:
                raise WebFetchError("interlaced PNG not supported")
            if color_type == 3:
                raise WebFetchError("palette PNG not supported")
        elif chunk_type == b"IDAT":
            idat += chunk_data
        elif chunk_type == b"IEND":
            break
        pos += 12 + length
    if width is None or height is None:
        raise WebFetchError("PNG missing IHDR")
    if color_type not in (0, 2, 4, 6):
        raise WebFetchError(f"unsupported PNG color type {color_type}")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    pixels: list[tuple[int, int, int]] = []
    previous = bytearray(stride)
    offset = 0
    for _row in range(height):
        if offset >= len(raw):
            raise WebFetchError("PNG data truncated")
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset:offset + stride])
        offset += stride
        _unfilter(line, previous, filter_type, channels, width)
        for x in range(width):
            base = x * channels
            if color_type == 0:
                gray = line[base]
                pixels.append((gray, gray, gray))
            elif color_type == 2:
                pixels.append((line[base], line[base + 1], line[base + 2]))
            elif color_type == 4:
                gray = line[base]
                pixels.append((gray, gray, gray))
            else:  # 6 RGBA
                pixels.append((line[base], line[base + 1], line[base + 2]))
        previous = line
    return width, height, pixels


def _unfilter(line: bytearray, previous: bytearray, filter_type: int,
              channels: int, width: int) -> None:
    """按 PNG filter 类型解码一行。filter 0-4,通道数 per-pixel。"""
    bpp = channels
    for i in range(width * channels):
        left = line[i - bpp] if i >= bpp else 0
        up = previous[i]
        up_left = previous[i - bpp] if i >= bpp else 0
        if filter_type == 1:  # Sub
            line[i] = (line[i] + left) & 0xFF
        elif filter_type == 2:  # Up
            line[i] = (line[i] + up) & 0xFF
        elif filter_type == 3:  # Average
            line[i] = (line[i] + ((left + up) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            predictor = _paeth(left, up, up_left)
            line[i] = (line[i] + predictor) & 0xFF
        elif filter_type != 0:  # None
            raise WebFetchError(f"unknown PNG filter type {filter_type}")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


class PixelFeatureExtractor:
    """确定性像素特征:[1,16]。RGB 直方图(各 4 桶)+ 尺寸 + 亮度。"""

    def extract(self, width: int, height: int,
                pixels: list[tuple[int, int, int]]) -> Tensor:
        buckets = [0.0] * 12  # R0-3, G0-3, B0-3
        total = max(1, len(pixels))
        brightness = 0.0
        for r, g, b in pixels:
            buckets[r // 64] += 1.0
            buckets[4 + g // 64] += 1.0
            buckets[8 + b // 64] += 1.0
            brightness += (r + g + b) / 3.0
        hist = [value / total for value in buckets]
        area = width * height
        features = hist + [
            width / max(1, width + height),
            height / max(1, width + height),
            brightness / (255.0 * total),
            (area ** 0.5) / 256.0,
        ]
        return torch.tensor([features], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return 16
