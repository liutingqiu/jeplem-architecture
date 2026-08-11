"""语义理解层:从网页 HTML 提取可读的知识陈述(规则版,零依赖)。

StatementExtractor 用确定性规则从网页提取"关于世界的陈述"(如
"Python is a programming language"),并给每个陈述按规则强度打分。
这让身体积累的不再只是 URL 和不可解释的向量,而是可读的知识。

规则(按强度):copula 句式(最强) > <title> > <h1> > <h2> > 锚文本。
垃圾过滤:词数 <3 或 >32、纯数字、导航词丢弃;去重取高置信。
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

# nav / junk anchor words that are not knowledge
_NAV_WORDS = {"home", "about", "contact", "click", "here", "more", "read",
              "login", "register", "sign", "up", "menu", "next", "previous",
              "back", "top", "bottom", "search", "submit", "ok", "close"}
_COPULA_VERBS = re.compile(r"\b(is|are|was|were|means|refers to|describes|defines)\b", re.IGNORECASE)
_NAVIGATION = re.compile(r"^(home|about|contact|click here|read more|more|menu|login|register|sign in|sign up)$", re.IGNORECASE)
_PURE_DIGITS = re.compile(r"^[\d.,%\s-]+$")


@dataclass(frozen=True)
class Statement:
    """一条可读的知识陈述 + 置信度 + 来源规则。"""

    statement: str
    confidence: float
    rule: str  # copula | title | h1 | h2 | anchor


class StatementExtractor:
    """从网页 HTML 提取知识陈述(确定性规则)。"""

    def __init__(self, *, min_words: int = 3, max_words: int = 32,
                 max_statements: int = 8) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.max_statements = max_statements

    def _strip_tags(self, html_text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html_text)
        return re.sub(r"\s+", " ", html.unescape(text)).strip()

    def _word_count(self, text: str) -> int:
        return len([w for w in text.split() if w.strip()])

    def _clean(self, text: str) -> str | None:
        """清洗并过滤。返回规范化的陈述,或 None(垃圾)。"""
        text = self._strip_tags(text).strip()
        if not text:
            return None
        words = self._word_count(text)
        if words < self.min_words or words > self.max_words:
            return None
        if _PURE_DIGITS.match(text):
            return None
        if _NAVIGATION.match(text.lower()):
            return None
        return text

    def _copula(self, raw: str) -> list[tuple[str, str]]:
        """用 HTMLParser 提取正文句子中含 copula 动词的陈述。

        只取 <p>/<div>/正文文本块(不含锚文本、标题),按句拆分。
        """
        from html.parser import HTMLParser

        texts: list[str] = []
        in_script_style = False

        class _TextCollector(HTMLParser):
            def handle_starttag(self, tag, attrs):
                nonlocal in_script_style
                if tag in ("script", "style"):
                    in_script_style = True

            def handle_endtag(self, tag):
                nonlocal in_script_style
                if tag in ("script", "style"):
                    in_script_style = False

            def handle_data(self, data):
                if not in_script_style and data.strip():
                    texts.append(data.strip())

        parser = _TextCollector()
        parser.feed(raw)
        found: list[tuple[str, str]] = []
        for block in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", block):
                sentence = sentence.strip()
                if not sentence or self._word_count(sentence) < self.min_words:
                    continue
                if _COPULA_VERBS.search(sentence):
                    found.append((sentence, "copula"))
        return found

    def _title(self, raw: str) -> list[tuple[str, str]]:
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
        return [(m.group(1), "title")] if m else []

    def _heading(self, raw: str, tag: str) -> list[tuple[str, str]]:
        return [(m, tag) for m in re.findall(
            rf"<{tag}[^>]*>(.*?)</{tag}>", raw, re.IGNORECASE | re.DOTALL)]

    def _anchors(self, raw: str) -> list[tuple[str, str]]:
        return [(m, "anchor") for m in re.findall(
            r"<a[^>]*>(.*?)</a>", raw, re.IGNORECASE | re.DOTALL)]

    def _confidence(self, rule: str, words: int) -> float:
        base = {"copula": 0.85, "title": 0.80, "h1": 0.70, "h2": 0.65, "anchor": 0.55}[rule]
        return round(min(1.0, base * min(1.0, words / 8.0)), 3)

    def extract(self, raw: str, url: str | None = None) -> list[Statement]:
        """从 HTML 提取 ≤max_statements 条知识陈述。"""
        candidates: list[tuple[str, str]] = []
        candidates += self._title(raw)
        candidates += self._heading(raw, "h1")
        candidates += self._heading(raw, "h2")
        candidates += self._copula(raw)
        candidates += self._anchors(raw)
        seen: dict[str, Statement] = {}
        for text, rule in candidates:
            cleaned = self._clean(text)
            if cleaned is None:
                continue
            # skip pure nav anchors even if they passed word count
            lower = cleaned.lower()
            if all(w in _NAV_WORDS for w in re.findall(r"[a-z]+", lower)):
                continue
            key = lower
            conf = self._confidence(rule, self._word_count(cleaned))
            if key not in seen or conf > seen[key].confidence:
                seen[key] = Statement(cleaned, conf, rule)
        statements = sorted(seen.values(), key=lambda s: -s.confidence)
        return statements[: self.max_statements]
