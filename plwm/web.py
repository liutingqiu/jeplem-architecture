"""真实互联网数据接入前端:WebFetchBackend + JsonSchemaValidator。

设计依据:项目纪律"schema 严格验证 + 凭据不进门"。从真实 URL 拉取文本
/JSON,只把通过声明式校验的、规范化后的语义喂给身体。原始网页/任意 JSON
绝不进入认知状态;错误统一抛 WebFetchError,绝不崩溃身体。

仅用标准库 urllib,不新增重型依赖。
"""
from __future__ import annotations

import hashlib
import html
import json
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError


class WebFetchError(Exception):
    """web 拉取/解析/校验失败。身体捕获后安全降级,绝不崩溃。"""


@dataclass(frozen=True)
class WebResource:
    """一次经审计的 web 资源:URL + 模态 + 内容 + 内容哈希 + 元数据。"""

    url: str
    modality: str
    content: str | dict[str, Any]
    content_hash: str
    etag: str | None = None
    fetched_at: float = 0.0


class WebFetchBackend:
    """urllib GET 前端。fetch_text 去 HTML 标签, fetch_json 解析 dict。

    未过 schema 的 JSON 不进身体(由调用方的 JsonSchemaValidator 把关)。
    """

    def __init__(self, *, timeout: float = 10.0, max_bytes: int = 262144,
                 user_agent: str = "plwm-research/0.1") -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent

    def _open(self, url: str, accept: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": accept},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    raise WebFetchError(f"response exceeds max_bytes={self.max_bytes}")
                return data
        except (URLError, OSError, ValueError) as error:
            raise WebFetchError(f"failed to fetch {url}: {error}") from error

    def fetch_text(self, url: str) -> WebResource:
        raw = self._open(url, "text/html,text/plain")
        text = _strip_html(raw.decode("utf-8", errors="replace"))
        return WebResource(url, "web-text", text, _sha256(text))

    def fetch_json(self, url: str) -> WebResource:
        raw = self._open(url, "application/json")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise WebFetchError(f"invalid JSON from {url}: {error}") from error
        if not isinstance(payload, dict):
            raise WebFetchError(f"JSON from {url} is not a mapping")
        return WebResource(url, "web-json", payload, _sha256(json.dumps(payload, sort_keys=True)))


class JsonSchemaValidator:
    """声明式 JSON schema 校验:精确键集匹配。

    仿 parse_luanti_language_goal:每个 schema 是 {field: type} 精确键集。
    多余键 / 缺失键 / 类型不符 → ValueError(凭证或无关数据永不进门)。
    """

    def __init__(self, schemas: dict[str, dict[str, type]]) -> None:
        if not schemas:
            raise ValueError("at least one schema is required")
        self.schemas = schemas

    def validate(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        if name not in self.schemas:
            raise ValueError(f"unknown schema: {name!r}")
        if not isinstance(data, dict):
            raise ValueError(f"data for {name!r} must be a mapping")
        fields = self.schemas[name]
        if set(data) != set(fields):
            raise ValueError(f"schema {name!r} requires exactly keys {sorted(fields)}, got {sorted(data)}")
        cleaned: dict[str, Any] = {}
        for key, field_type in fields.items():
            value = data[key]
            if isinstance(value, bool):
                if field_type is not bool:
                    raise ValueError(f"{name}.{key} must be {field_type.__name__}, not bool")
            elif not isinstance(value, field_type):
                raise ValueError(f"{name}.{key} must be {field_type.__name__}, got {type(value).__name__}")
            cleaned[key] = value
        return cleaned


def _strip_html(raw: str) -> str:
    """极简 HTML→文本:去标签 + 实体解码 + 压缩空白。"""
    import re
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
