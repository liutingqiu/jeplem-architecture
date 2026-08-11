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
    """把网页文本降为 N 维事实特征向量(标题/数字/关键词 one-hot)。

    这是一个极简、声明式、可审计的提取器——不是"理解"网页,而是提取
    可验证的事实信号(数字、已知关键词),供身体经统一语义空间使用。
    """

    def __init__(self, keywords: list[str] | None = None) -> None:
        self.keywords = keywords or ["temperature", "price", "score", "level"]
        self._word_index: dict[str, int] = {w: i for i, w in enumerate(self.keywords)}

    def extract(self, text: str) -> Tensor:
        """文本 → [1 + 数字桶 + len(keywords)] 维 one-hot 特征。"""
        lowered = text.lower()
        numbers = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]
        number_feature = 1.0 if numbers else 0.0
        keyword_feature = [1.0 if word in lowered else 0.0 for word in self.keywords]
        return torch.tensor([[number_feature] + keyword_feature], dtype=torch.float32)

    @property
    def dim(self) -> int:
        return 1 + len(self.keywords)

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
                 *, dim: int = DEFAULT_DIM, extractor: WebTextExtractor | None = None) -> None:
        self.backend = backend
        self.body = body
        self.space = space
        extractor = extractor or WebTextExtractor()
        self.extractor = extractor
        encoder = ModalityEncoder(extractor.dim, dim=dim, state_dim=extractor.dim)
        if "web-text" not in self.space.encoders:
            self.space.register("web-text", encoder)

    def sense(self, url: str) -> bool:
        """fetch 一页 → 提取 → 进身体。返回是否成功进身体。"""
        try:
            resource = self.backend.fetch_text(url)
        except WebFetchError:
            return False
        semantic = self.extractor.extract(resource.content)
        fact_value = self.extractor.extract_value(resource.content)
        self.body.observe(
            "web-text",
            semantic.squeeze(0),
            payload={"url": resource.url, "hash": resource.content_hash},
            fact_value=fact_value,
        )
        return True


class WebStructSensor:
    """结构化数据感官:真实 JSON API → schema 校验 → 进身体。"""

    def __init__(self, backend: WebFetchBackend, body, space,
                 *, schema_name: str, schema_fields: dict[str, type],
                 dim: int = DEFAULT_DIM, encoder: WebStructEncoder | None = None) -> None:
        self.backend = backend
        self.body = body
        self.space = space
        self.schema_name = schema_name
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
            validated = self.validator.validate(self.schema_name, resource.content)
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

    fact_value = f"{source}->{target}",账本键 identity:{source}->{target}。
    """

    def __init__(self, backend: WebFetchBackend, body, space, graph,
                 *, dim: int = DEFAULT_DIM) -> None:
        from .link_graph import LinkGraph

        self.backend = backend
        self.body = body
        self.space = space
        self.graph = graph
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
            )
        return bool(links)
