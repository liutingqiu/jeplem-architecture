"""BodyRunner:把全部零件组装成一个完整身体,支持长时运行 + 完整 checkpoint。

这是"造人"路线中第一次把 10 种互联网感官 + 统一语义空间 + 记忆 + 情绪 +
内感受 + 时间感 + 可信度 + 关系图组装进一个身体,并让它在时间里连续运行。

run_round:对一组 URL 目标用对应传感器感知 → trust 演化 → 情绪/时间更新。
snapshot/restore:完整保存/重建身体全部状态(runtime/body/ledger/gate/
emotion/effort/time_sense/trust/link_graph),供长时运行跨 checkpoint 存活。
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .emotion import Effort, Emotion, TimeSense
from .link_graph import LinkGraph
from .multimodal_body import MultimodalBody
from .semantic_space import SemanticSpace
from .trust import TrustRegistry
from .web import WebFetchBackend
from .web_sensors import (
    WebAudioSensor,
    WebImageSensor,
    WebLinkSensor,
    WebSocialSensor,
    WebStructSensor,
    WebTextSensor,
    WebVideoSensor,
)


class BodyRunner:
    """完整身体的运行时:组装全部感官 + 内部感受层,支持长跑 + checkpoint。"""

    def __init__(
        self,
        body: MultimodalBody,
        *,
        backend: WebFetchBackend,
        trust: TrustRegistry | None = None,
        link_graph: LinkGraph | None = None,
        struct_schema: Mapping[str, type] | None = None,
        social_subset: bool = False,
        struct_subset: bool = False,
    ) -> None:
        self.body = body
        self.backend = backend
        self.trust = trust if trust is not None else TrustRegistry()
        self.link_graph = link_graph if link_graph is not None else LinkGraph()
        self.struct_schema = struct_schema or {"temperature": int, "unit": str, "city": str}
        self.social_subset = social_subset
        self.struct_subset = struct_subset
        self.sensors = self._build_sensors()
        self.round_stats: list[dict] = []

    def _build_sensors(self) -> dict[str, object]:
        """组装全部 10 种感官(复用各 sensor 的 register 模式,幂等)。"""
        body, backend, space = self.body, self.backend, self.body.space
        return {
            "web-text": WebTextSensor(backend, body, space),
            "web-json": WebStructSensor(backend, body, space,
                                        schema_name="weather",
                                        schema_fields=self.struct_schema,
                                        subset=self.struct_subset),
            "web-image": WebImageSensor(backend, body, space),
            "web-link": WebLinkSensor(backend, body, space, self.link_graph),
            "web-audio": WebAudioSensor(backend, body, space),
            "web-social": WebSocialSensor(backend, body, space, subset=self.social_subset),
            "web-video": WebVideoSensor(backend, body, space),
        }

    def sense_one(self, modality: str, url: str, *, override: Mapping[str, Any] | None = None) -> bool:
        """用对应传感器感知一个 URL。失败返回 False,绝不崩溃身体。"""
        sensor = self.sensors.get(modality)
        if sensor is None:
            return False
        try:
            if modality == "web-text" and override and "fact_value" in override:
                return bool(sensor.sense(url, fact_value_override=override["fact_value"]))
            return bool(sensor.sense(url))
        except Exception:  # noqa: BLE001 -- any web failure must not kill the body
            return False

    def _track_source(self, modality: str, url: str, ok: bool) -> None:
        """来源可信度演化:成功升分,失败降分。"""
        source = f"{modality}:{url}"
        self.trust.report(source, correct=ok)

    def run_round(self, targets: Sequence[Mapping[str, Any]]) -> dict:
        """一轮感知:对每个 target(含 modality/url/可选 override)感知 + 演化。"""
        stats = {"round": len(self.round_stats), "ok": 0, "fail": 0}
        for target in targets:
            modality = target["modality"]
            url = target["url"]
            ok = self.sense_one(modality, url, override=target.get("override"))
            self._track_source(modality, url, ok)
            if ok:
                stats["ok"] += 1
            else:
                stats["fail"] += 1
            # internal feeling: outcome valence + effort from the round
            self.body.feel(outcome=1.0 if ok else -1.0)
        self.round_stats.append(stats)
        return stats

    # -- checkpoint ----------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "version": 1,
            "body": self.body.snapshot(),
            "trust": self.trust.snapshot(),
            "link_graph": {"edges": self.link_graph.edges()},
            "round_stats": self.round_stats,
        }

    def restore(self, snap: dict) -> None:
        self.body.restore(snap["body"])
        trust_snap = snap["trust"]
        self.trust._scores = dict(trust_snap["scores"])
        self.trust._events = list(trust_snap["events"])
        self.link_graph = LinkGraph()
        for source, target in snap["link_graph"]["edges"]:
            self.link_graph.add_edge(source, target)
        self.round_stats = list(snap["round_stats"])

    def snapshot_digest(self) -> str:
        """稳定字段的确定性摘要(idle 不漂移度量)。"""
        body = self.body.snapshot()
        stable = {
            "tick": body["tick"],
            "runtime": body["runtime"].get("integrity", ""),
            "ledger_current": sorted(
                (k, str(v["value"]), v["confidence"]) for k, v in body["ledger"]["current"].items()
            ) if body["ledger"] else [],
            "trust": sorted(self.trust.snapshot()["scores"].items()),
        }
        return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def build_default_body(*, dim: int = 16, belief_states: int = 4,
                       world_id: str = "body",
                       emotion: Emotion | None = None,
                       effort: Effort | None = None,
                       time_sense: TimeSense | None = None,
                       ledger=None) -> MultimodalBody:
    """构造一个默认身体(MultimodalBody + 统一语义空间 + 运行时 + 情绪)。"""
    from .fact_ledger import VersionedFactLedger
    from .runtime import PersistentRuntime

    space = SemanticSpace(dim=dim)
    runtime = PersistentRuntime(dim, belief_states=belief_states, latent_dim=dim)
    ledger = ledger or VersionedFactLedger(
        trusted_sources={"web-text", "web-json", "web-image", "web-link",
                         "web-audio", "web-social", "web-video"},
        revision_confidence=0.9,
    )
    return MultimodalBody(world_id, space, runtime, ledger=ledger,
                          emotion=emotion, effort=effort, time_sense=time_sense)
