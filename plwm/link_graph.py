"""超链接关系图:零依赖有向图,(source -> target) 带锚文本。

让身体"感知网页间关系":抓取一个页面,把它的每条出链存入图,之后能
查询某节点的邻居、判断两条 URL 是否有边。账本只存 key/value,不承载
图查询,故单独一个极简结构。
"""
from __future__ import annotations


class LinkGraph:
    """有向图:node -> set[target]。anchor 存映射 (source,target)->anchor。"""

    def __init__(self) -> None:
        self._adj: dict[str, set[str]] = {}
        self._anchors: dict[tuple[str, str], str] = {}

    def add_edge(self, source: str, target: str, anchor: str | None = None) -> None:
        if not source or not target:
            raise ValueError("edge requires non-empty source and target")
        self._adj.setdefault(source, set()).add(target)
        if anchor is not None:
            self._anchors[(source, target)] = anchor

    def query_neighbors(self, node: str) -> list[str]:
        return sorted(self._adj.get(node, set()))

    def has_edge(self, source: str, target: str) -> bool:
        return target in self._adj.get(source, set())

    def anchor_of(self, source: str, target: str) -> str | None:
        return self._anchors.get((source, target))

    def edges(self) -> list[tuple[str, str]]:
        return [(s, t) for s, targets in self._adj.items() for t in sorted(targets)]

    def node_count(self) -> int:
        return len(self._adj)

    def edge_count(self) -> int:
        return len(self.edges())
