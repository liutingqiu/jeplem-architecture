"""互联网感官:文本(网页)+ 结构化数据(JSON)经统一身体入口进入认知。

两个传感器都复用 SemanticSpace(register 新编码器)+ multimodal_body
(observe),零改动既有组件。WebTextExtractor / WebStructEncoder 把真实
数据降为 N 维声明式语义向量,经统一语义空间 D 维投影后进 body.observe
→ gate(审计含 URL+sha256)→ runtime.ingest → 账本跨模态汇聚。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .image_features import decode_png
from .semantic_space import ModalityEncoder
from .web import JsonSchemaValidator, WebFetchBackend, WebFetchError, WebResource

DEFAULT_DIM = 16


def _number_features(values: Mapping[str, Any], keys: list[str]) -> list[float]:
    """把 schema 校验过的数值字段规范化到 [0,1] 特征(分桶直方图)。"""
    features: list[float] = []
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            features.append(0.0)
            continue
        features.append(max(0.0, min(1.0, float(value) / 100.0)))
    return features


class WebTextExtractor:
    """把网页文本降为 N 维事实特征向量。

    这是一个极简、声明式、可审计的提取器——不是"理解"网页,而是提取
    可验证的事实信号。rich=True 时在旧前缀([has_number]+[keyword*K])后
    追加丰富度特征(词/字符/句/数字饱和度、标点多样性、字符熵、URL 长
    度),让"读内容"不是退化信号;rich=False 完全复刻旧向量(控制组)。
    """

    def __init__(self, keywords: list[str] | None = None, *, rich: bool = True) -> None:
        self.keywords = keywords or ["temperature", "price", "score", "level"]
        self.rich = rich

    def _rich_features(self, text: str, url: str | None = None) -> list[float]:
        words = [w for w in text.split() if w.strip()]
        chars = len(text)
        sentences = max(1, len(re.findall(r"[.!?。！？]", text)) + (1 if text.strip() else 0))
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        word_sat = min(1.0, len(words) / 200.0)
        char_sat = min(1.0, chars / 2000.0)
        sent_sat = min(1.0, sentences / 20.0)
        num_sat = min(1.0, len(numbers) / 20.0)
        punct = len(re.findall(r"[.,;:!?。，；：！？]", text))
        punct_div = min(1.0, punct / 40.0)
        # normalized character entropy (diversity)
        from collections import Counter
        import math
        counts = Counter(text.lower())
        total = max(1, len(text))
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        entropy_norm = min(1.0, entropy / 4.0)
        url_len = min(1.0, len(url or "") / 100.0)
        return [word_sat, char_sat, sent_sat, num_sat, punct_div, entropy_norm, url_len]

    def extract(self, text: str, *, url: str | None = None) -> Tensor:
        """文本 → 特征向量。rich 时前缀保序 + 追加丰富度特征。"""
        lowered = text.lower()
        numbers = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]
        number_feature = 1.0 if numbers else 0.0
        keyword_feature = [1.0 if word in lowered else 0.0 for word in self.keywords]
        prefix = [number_feature] + keyword_feature
        if self.rich:
            return torch.tensor([prefix + self._rich_features(text, url)], dtype=torch.float32)
        return torch.tensor([prefix], dtype=torch.float32)

    def richness(self, text: str) -> float:
        """文本丰富度 [0,1]:词/句/字符/数字饱和度的均值。"""
        words = [w for w in text.split() if w.strip()]
        sentences = min(1.0, len(re.findall(r"[.!?。！？]", text)) / 10.0)
        return min(1.0, (min(1.0, len(words) / 100.0) + sentences + min(1.0, len(text) / 1000.0)) / 3.0)

    def confidence(self, text: str) -> float:
        """按内容丰富度给感知置信度。

        首次感知总能记入账本(0.95 起),丰富度提高置信(重读升高置信 →
        gap 下降 → 驱动深化理解)。richness 低只影响"多深",不影响"记不记"。
        """
        return min(1.0, 0.95 + 0.05 * self.richness(text))

    @property
    def dim(self) -> int:
        return 1 + len(self.keywords) + (7 if self.rich else 0)

    def extract_value(self, text: str) -> str | None:
        """提取网页中的数字作为事实值(如 'temperature is 23' -> '23')。"""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[0] if numbers else None


class WebStructEncoder:
    """把 schema 校验过的 JSON 规范化到特征向量。"""

    def __init__(self, numeric_keys: list[str] | None = None) -> None:
        self.numeric_keys = numeric_keys or []

    def encode(self, data: Mapping[str, Any]) -> Tensor:
        return torch.tensor([_number_features(data, self.numeric_keys)], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return max(1, len(self.numeric_keys))


class WebTextSensor:
    """文本感官:真实网页 → 事实特征 → 统一语义空间 → 身体。"""

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, dim: int = DEFAULT_DIM, extractor: WebTextExtractor | None = None,
                 confidence: bool = True, statement_extractor=None,
                 extract_statements: bool = True) -> None:
        self.backend = backend
        self.body = body
        self.space = space
        self.use_confidence = confidence
        self.extract_statements = extract_statements
        if statement_extractor is None and extract_statements:
            from .statement_extractor import StatementExtractor
            statement_extractor = StatementExtractor()
        self.statement_extractor = statement_extractor
        extractor = extractor or WebTextExtractor()
        self.extractor = extractor
        encoder = ModalityEncoder(extractor.dim, dim=dim, state_dim=extractor.dim)
        if "web-text" not in self.space.encoders:
            self.space.register("web-text", encoder)

    def _observe_statements(self, resource, semantic, url: str) -> int:
        """提取网页知识陈述并写入账本 fact: 键。返回写入条数。"""
        if (not self.extract_statements or self.statement_extractor is None
                or resource.html_raw is None):
            return 0
        statements = self.statement_extractor.extract(resource.html_raw, url=url)
        n = 0
        for st in statements:
            norm_key = re.sub(r"\s+", " ", st.statement).strip()[:120]
            fact_key = f"fact:{norm_key}"
            # statements are deterministic rule extractions (auditable), so
            # they enter the ledger as strong facts; rule strength is kept in
            # the payload for audit, not used to gate ledger acceptance.
            self.body.observe(
                "web-text",
                semantic.squeeze(0),
                payload={"url": resource.url, "hash": resource.content_hash,
                         "rule": st.rule},
                fact_value=st.statement,
                fact_key_override=fact_key,
                confidence=0.95,
            )
            n += 1
        return n

    def sense(self, url: str, *, fact_value_override: str | None = None) -> bool:
        """fetch 一页 → 提取 → 进身体。返回是否成功进身体。

        fact_value_override:跨模态汇聚时强制用同一事实键(如 URL),默认
        None 沿用网页数字提取值(旧行为)。
        confidence=True 时:按内容丰富度给置信度(重读升高置信 → gap 下降
        → 驱动深化理解)。
        extract_statements 时:从 HTML 提取可读知识陈述进账本 fact: 键。
        """
        try:
            resource = self.backend.fetch_text(url)
        except WebFetchError:
            return False
        semantic = self.extractor.extract(resource.content, url=url)
        fact_value = fact_value_override or self.extractor.extract_value(resource.content)
        confidence = None
        if self.use_confidence:
            # max with any previous confidence so re-reading deepens (rises
            # confidence, lowers gap) instead of being treated as fresh
            prev_conf = 0.0
            if self.body.ledger is not None:
                record = self.body.ledger.current.get(f"identity:{fact_value}")
                if record is not None:
                    prev_conf = float(record.confidence)
            confidence = max(prev_conf, self.extractor.confidence(resource.content))
        self._observe_statements(resource, semantic, url)
        self.body.observe(
            "web-text",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash},
            fact_value=fact_value,
            confidence=confidence,
        )
        return True


class WebStructSensor:
    """结构化数据感官:真实 JSON API → schema 校验 → 进身体。"""

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, schema_name: str, schema_fields: dict[str, type],
                 dim: int = DEFAULT_DIM, encoder: WebStructEncoder | None = None,
                 subset: bool = False) -> None:
        self.backend = backend
        self.body = body
        self.space = space
        self.schema_name = schema_name
        self.subset = subset
        self.validator = JsonSchemaValidator({schema_name: schema_fields})
        encoder = encoder or WebStructEncoder(numeric_keys=list(schema_fields.keys()))
        self.encoder = encoder
        mod = ModalityEncoder(encoder.dim, dim=dim, state_dim=encoder.dim)
        if "web-json" not in self.space.encoders:
            self.space.register("web-json", mod)

    def sense(self, url: str) -> bool:
        """fetch 一个 JSON API → schema 校验 → 进身体。未过校验则拒绝。"""
        try:
            resource = self.backend.fetch_json(url)
        except WebFetchError:
            return False
        try:
            validated = self.validator.validate(self.schema_name, resource.content, subset=self.subset)
        except ValueError:
            return False
        semantic = self.encoder.encode(validated)
        fact_value = str(validated.get(list(self.encoder.numeric_keys)[0])) if self.encoder.numeric_keys else None
        self.body.observe(
            "web-json",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash, "schema": self.schema_name},
            fact_value=fact_value,
        )
        return True


class WebImageSensor:
    """图像感官:真实图片 → 像素特征 → 统一语义空间 → 身体。

    用纯标准库解码 PNG 并提取确定性像素特征(零新增依赖)。
    fact_value = URL,账本键 identity:{url}(URL 作身份便于跨模态)。
    """

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, dim: int = DEFAULT_DIM, extractor=None) -> None:
        from .image_features import PixelFeatureExtractor

        self.backend = backend
        self.body = body
        self.space = space
        self.extractor = extractor or PixelFeatureExtractor()
        encoder = ModalityEncoder(self.extractor.dim, dim=dim, state_dim=self.extractor.dim)
        if "web-image" not in self.space.encoders:
            self.space.register("web-image", encoder)

    def sense(self, url: str) -> bool:
        try:
            resource = self.backend.fetch_image(url)
        except WebFetchError:
            return False
        try:
            width, height, pixels = decode_png(resource.content)
        except WebFetchError:
            return False
        semantic = self.extractor.extract(width, height, pixels)
        self.body.observe(
            "web-image",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash, "w": width, "h": height},
            fact_value=url,
        )
        return True


class WebLinkSensor:
    """超链接感官:真实 HTML → (source->target) 边入 LinkGraph + 进身体。

    fact_value = f"{source}->{target}"。write_ledger=False(默认,内容优先)
    时边只进 LinkGraph、账本零链接键(skip_ledger);write_ledger=True 保留
    旧行为(链接也写账本)。
    """

    def __init__(self, backend: WebFetchBackend, body, space, graph,
                 *, dim: int = DEFAULT_DIM, write_ledger: bool = False) -> None:
        from .link_graph import LinkGraph

        self.backend = backend
        self.body = body
        self.space = space
        self.graph = graph
        self.write_ledger = write_ledger
        # 1 (has-link) + 2 buckets for url length / anchor length
        encoder = ModalityEncoder(3, dim=dim, state_dim=3)
        if "web-link" not in self.space.encoders:
            self.space.register("web-link", encoder)

    @staticmethod
    def _link_features(source: str, target: str, anchor: str) -> Tensor:
        return torch.tensor([[1.0, len(source) / 100.0, len(anchor) / 100.0]], dtype=torch.float32)

    def sense(self, url: str) -> bool:
        try:
            links = self.backend.fetch_links(url)
        except WebFetchError:
            return False
        for target, anchor in links:
            self.graph.add_edge(url, target, anchor)
            semantic = self._link_features(url, target, anchor)
            self.body.observe(
                "web-link",
                semantic.squeeze(0),
                payload={"source": url, "target": target, "anchor": anchor},
                fact_value=f"{url}->{target}",
                skip_ledger=not self.write_ledger,
            )
        return bool(links)


class WebAudioSensor:
    """音频感官:真实 WAV → 波形特征 → 统一语义空间 → 身体。

    fact_value = url,账本键 identity:{url}(与文本/图像同 URL 汇聚)。
    """

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, dim: int = DEFAULT_DIM, extractor=None) -> None:
        from .audio_features import WaveFeatureExtractor, decode_wav

        self.backend = backend
        self.body = body
        self.space = space
        self.decode_wav = decode_wav
        self.extractor = extractor or WaveFeatureExtractor()
        encoder = ModalityEncoder(self.extractor.dim, dim=dim, state_dim=self.extractor.dim)
        if "web-audio" not in self.space.encoders:
            self.space.register("web-audio", encoder)

    def sense(self, url: str) -> bool:
        try:
            resource = self.backend.fetch_audio(url)
        except WebFetchError:
            return False
        try:
            samples, rate = self.decode_wav(resource.content)
        except WebFetchError:
            return False
        semantic = self.extractor.extract(samples, rate)
        self.body.observe(
            "web-audio",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash, "sr": rate, "n": len(samples)},
            fact_value=url,
        )
        return True


class SocialEngagementEncoder:
    """社交 engagement 特征:[like, comment, view] 归一化 + 互动率。"""

    def encode(self, data: Mapping[str, Any]) -> Tensor:
        like = float(data.get("like_count", 0))
        comment = float(data.get("comment_count", 0))
        view = float(data.get("view_count", 0))
        interaction = (like + comment) / max(1.0, view)
        return torch.tensor([[
            like / 10000.0,
            comment / 1000.0,
            view / 1000000.0,
            interaction,
            1.0,
        ]], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return 5


class WebSocialSensor:
    """社交信号感官:真实 JSON API → schema 校验 → engagement 特征 → 身体。

    复用 JsonSchemaValidator 精确键集,多余键/凭据拒之门外。
    fact_value = content_url,账本键 identity:{content_url}。
    """

    SOCIAL_SCHEMA = {"content_url": str, "like_count": int, "comment_count": int, "view_count": int}

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, dim: int = DEFAULT_DIM, encoder: SocialEngagementEncoder | None = None,
                 subset: bool = False) -> None:
        self.backend = backend
        self.body = body
        self.space = space
        self.subset = subset
        self.validator = JsonSchemaValidator({"engagement": self.SOCIAL_SCHEMA})
        self.encoder = encoder or SocialEngagementEncoder()
        mod = ModalityEncoder(self.encoder.dim, dim=dim, state_dim=self.encoder.dim)
        if "web-social" not in self.space.encoders:
            self.space.register("web-social", mod)

    def sense(self, url: str) -> bool:
        try:
            resource = self.backend.fetch_json(url)
        except WebFetchError:
            return False
        try:
            validated = self.validator.validate("engagement", resource.content, subset=self.subset)
        except ValueError:
            return False
        semantic = self.encoder.encode(validated)
        self.body.observe(
            "web-social",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash,
                     "content_url": validated["content_url"]},
            fact_value=validated["content_url"],
        )
        return True


class WebVideoSensor:
    """视频感官:真实视频 → ffmpeg 帧提取 → 运动特征 → 统一语义空间 → 身体。

    fact_value = url,账本键 identity:{url}。帧数不足 2 视为拉取失败。
    """

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, dim: int = DEFAULT_DIM, extractor=None, n_frames: int = 8) -> None:
        from .video_features import VideoFeatureExtractor, extract_frames

        self.backend = backend
        self.body = body
        self.space = space
        self.extract_frames = extract_frames
        self.n_frames = n_frames
        self.extractor = extractor or VideoFeatureExtractor()
        encoder = ModalityEncoder(self.extractor.dim, dim=dim, state_dim=self.extractor.dim)
        if "web-video" not in self.space.encoders:
            self.space.register("web-video", encoder)

    def sense(self, url: str) -> bool:
        try:
            resource = self.backend.fetch_video(url)
        except WebFetchError:
            return False
        try:
            video = self.extract_frames(resource.content, n_frames=self.n_frames)
        except WebFetchError:
            return False
        semantic = self.extractor.extract(video.frames)
        self.body.observe(
            "web-video",
            torch.tensor(semantic, dtype=torch.float32),
            payload={"url": resource.url, "hash": resource.content_hash,
                     "n_frames": len(video.frames), "duration": video.duration},
            fact_value=url,
        )
        return True
