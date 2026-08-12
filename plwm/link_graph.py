"""超链接关系图:零依赖有向图,(source -> target) 带锚文本。

让身体"感知网页间关系":抓取一个页面,把它的每条出链存入图,之后能
查询某节点的邻居、判断两条 URL 是否有边。账本只存 key/value,不承载
图查询,故单独一个极简结构。
"""
from __future__ import annotations


class LinkGraph:
    """有向图:node -> set[target]。anchor 存映射 (source,target)->anchor。

    LRU-bounded: when over max_nodes, evict the LEAST-RECENTLY-USED source
    (not the oldest-added).  Frequently re-visited hubs (important sites) are
    kept; one-off pages are evicted.  This preserves the important
    exploration topology while bounding memory.
    """

    def __init__(self, max_nodes: int = 200_000) -> None:
        self._adj: dict[str, set[str]] = {}
        self._anchors: dict[tuple[str, str], str] = {}
        self._order: list[str] = []       # LRU order: front = least recently used
        self.max_nodes = max_nodes

    def add_edge(self, source: str, target: str, anchor: str | None = None) -> None:
        if not source or not target:
            raise ValueError("edge requires non-empty source and target")
        # LRU touch: (re)inserting source marks it most-recently-used
        if source in self._adj:
            try:
                self._order.remove(source)
            except ValueError:
                pass
        else:
            self._adj.setdefault(source, set())
        self._order.append(source)  # back = most recently used
        self._adj[source].add(target)
        if anchor is not None:
            self._anchors[(source, target)] = anchor
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Bound the graph: evict LEAST-recently-used sources when over cap.
        Keeps frequently-explored hubs, drops one-off pages."""
        while len(self._order) > self.max_nodes:
            lru = self._order.pop(0)  # front = least recently used
            targets = self._adj.pop(lru, set())
            for t in targets:
                self._anchors.pop((lru, t), None)

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
