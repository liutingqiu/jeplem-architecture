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
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
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
    html_raw: str | None = None


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

    def _open(self, url: str, accept: str, *, max_bytes: int | None = None) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": accept},
        )
        limit = max_bytes if max_bytes is not None else self.max_bytes
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise WebFetchError(f"response exceeds max_bytes={limit}")
                return data
        except (URLError, OSError, ValueError) as error:
            raise WebFetchError(f"failed to fetch {url}: {error}") from error

    def probe(self, url: str) -> tuple[str, bytes]:
        """探测一个 URL 的内容类型与原始字节(全模态路由用)。

        返回 (content_type, raw)。Content-Type 缺失时用魔数尽力推断;
        拉取失败抛 WebFetchError。
        """
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_bytes + 1)
                if len(raw) > self.max_bytes:
                    raise WebFetchError(f"response exceeds max_bytes={self.max_bytes}")
                content_type = response.headers.get("Content-Type", "") or ""
        except (URLError, OSError, ValueError) as error:
            raise WebFetchError(f"failed to probe {url}: {error}") from error
        if content_type:
            return content_type.lower(), raw
        # no Content-Type: infer from magic bytes
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png", raw
        if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
            return "audio/wav", raw
        if _is_video(raw):
            return "video/mp4", raw
        if raw[:1] in (b"{", b"["):
            return "application/json", raw
        return "text/html", raw

    def fetch_text(self, url: str) -> WebResource:
        raw = self._open(url, "text/html,text/plain")
        decoded = raw.decode("utf-8", errors="replace")
        text = _strip_html(decoded)
        return WebResource(url, "web-text", text, _sha256(text), html_raw=decoded)

    def fetch_image(self, url: str) -> WebResource:
        """拉取图片字节,校验 PNG 魔数。返回 WebResource(content=bytes)。"""
        raw = self._open(url, "image/png")
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise WebFetchError(f"{url} is not a PNG image")
        return WebResource(url, "web-image", raw, _sha256(raw.hex()))

    def fetch_audio(self, url: str) -> WebResource:
        """拉取音频字节,校验 RIFF/WAVE 魔数。返回 WebResource(content=bytes)。"""
        raw = self._open(url, "audio/wav")
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            raise WebFetchError(f"{url} is not a WAV audio file")
        return WebResource(url, "web-audio", raw, _sha256(raw.hex()))

    def fetch_video(self, url: str) -> WebResource:
        """拉取视频字节,校验常见容器魔数(视频允许更大上限)。"""
        raw = self._open(url, "video/mp4", max_bytes=8 * 1024 * 1024)
        if not _is_video(raw):
            raise WebFetchError(f"{url} is not a supported video container")
        return WebResource(url, "web-video", raw, _sha256(raw.hex()))

    def fetch_links(self, url: str) -> list[tuple[str, str]]:
        """从 HTML 提取 (target_url, anchor_text) 对。用 html.parser 而非正则。

        返回绝对 URL,去重,过滤 javascript:/空链接。
        """
        raw = self._open(url, "text/html,text/plain")
        html_text = raw.decode("utf-8", errors="replace")
        parser = _LinkParser()
        parser.feed(html_text)
        base = url
        seen: set[str] = set()
        links: list[tuple[str, str]] = []
        for href, anchor in parser.links:
            resolved = urllib.parse.urljoin(base, href)
            if resolved.startswith(("javascript:", "mailto:", "#")) or not resolved:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            links.append((resolved, anchor.strip()))
        return links

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

    def validate(self, name: str, data: dict[str, Any], *, subset: bool = False) -> dict[str, Any]:
        """校验 data 是否符合 schema。

        subset=False(默认):精确键集匹配,多余键/缺失键都拒绝(凭证或无关
        数据永不进门)。
        subset=True:只要求 schema 的已知键都在 data 中(缺已知键拒,多余键
        忽略),用于真实互联网数据(网页/API 常带多余字段)。
        """
        if name not in self.schemas:
            raise ValueError(f"unknown schema: {name!r}")
        if not isinstance(data, dict):
            raise ValueError(f"data for {name!r} must be a mapping")
        fields = self.schemas[name]
        if subset:
            if not all(key in data for key in fields):
                raise ValueError(f"schema {name!r} requires keys {sorted(fields)}, missing from data")
        elif set(data) != set(fields):
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


def _is_video(raw: bytes) -> bool:
    """常见视频容器魔数:mp4/mov(ftyp), avi(RIFF), webm(WEBM), mkv/mka。"""
    if len(raw) >= 8 and raw[4:8] == b"ftyp":
        return True
    return raw[:4] in (b"RIFF", b"WEBM") or raw[:4] == b"\x1a\x45\xdf\xa3"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _LinkParser(HTMLParser):
    """提取 <a href> 与锚文本。"""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._current_anchor: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._current_href = None
            self._current_anchor = []
            for key, value in attrs:
                if key == "href" and value:
                    self._current_href = value

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_anchor.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, "".join(self._current_anchor)))
            self._current_href = None
