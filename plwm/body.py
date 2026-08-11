"""BodyRunner:把全部零件组装成一个完整身体,支持长时运行 + 完整 checkpoint。

这是"造人"路线中第一次把 10 种互联网感官 + 统一语义空间 + 记忆 + 情绪 +
内感受 + 时间感 + 可信度 + 关系图组装进一个身体,并让它在时间里连续运行。

run_round:对一组 URL 目标用对应传感器感知 → trust 演化 → 情绪/时间更新。
snapshot/restore:完整保存/重建身体全部状态(runtime/body/ledger/gate/
emotion/effort/time_sense/trust/link_graph),供长时运行跨 checkpoint 存活。
"""
from __future__ import annotations

import json
import math
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
        whitelist: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.body = body
        self.backend = backend
        self.trust = trust if trust is not None else TrustRegistry()
        self.link_graph = link_graph if link_graph is not None else LinkGraph()
        self.struct_schema = struct_schema or {"temperature": int, "unit": str, "city": str}
        self.social_subset = social_subset
        self.struct_subset = struct_subset
        self.whitelist = list(whitelist) if whitelist else []
        self.sensors = self._build_sensors()
        self.round_stats: list[dict] = []
        self.choice_log: list[dict] = []

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

    def _candidate_signals(self, modality: str, url: str, value: float) -> dict:
        """单个候选的驱力信号:gap / recency / value / trust。"""
        key = f"identity:{url}"
        record = self.body.ledger.current.get(key) if self.body.ledger is not None else None
        if record is None:
            gap, tick = 1.0, 0
        else:
            gap = 1.0 - float(record.confidence)
            tick = int(record.tick)
        if self.body.time_sense is not None and tick > 0:
            recency = self.body.time_sense.recency(float(tick))
        else:
            elapsed = self.body.tick - tick
            recency = math.exp(-elapsed / 20.0)
        trust = self.trust.score(f"{modality}:{url}")
        return {"modality": modality, "url": url, "value": float(value),
                "gap": float(gap), "recency": float(recency), "trust": float(trust)}

    def choose_targets(self, *, whitelist: Sequence[Mapping[str, Any]] | None = None,
                       k: int | None = None) -> list[dict]:
        """自主选择下一轮感知目标。

        候选 = 白名单 ∪ ledger 已知事实。决策(v14 式分层):
        score = (-value, gap, -recency, trust) 字典序 argmax——
        价值选"哪个 URL",gap+recency 选"感知什么",trust 破平。
        每候选记入 choice_log 审计。
        """
        wl = list(whitelist) if whitelist is not None else self.whitelist
        candidates: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for entry in wl:
            modality = entry["modality"]
            url = entry["url"]
            seen.add((modality, url))
            candidates.append(self._candidate_signals(modality, url, entry.get("value", 1.0)))
        if self.body.ledger is not None:
            for key in self.body.ledger.current:
                if not key.startswith("identity:"):
                    continue
                url = key[len("identity:"):]
                # find the modality that observed it (from observations)
                modality = "web-text"
                for obs in reversed(self.body.observations):
                    if obs.fact_key == key:
                        modality = obs.modality
                        break
                if (modality, url) in seen:
                    continue
                seen.add((modality, url))
                candidates.append(self._candidate_signals(modality, url, 1.0))
        # layered ranking: unexplored (high gap) first -- the body is curious
        # about the unknown; within same gap, higher value; then older
        # (low recency); trust breaks ties.  This keeps value as a preference
        # while allowing exploration to move past already-perceived facts.
        ranked = sorted(candidates, key=lambda c: (-c["gap"], -c["value"], c["recency"], -c["trust"]))
        chosen = ranked if k is None else ranked[:k]
        override_by: dict[tuple[str, str], dict] = {
            (e["modality"], e["url"]): e.get("override") or {}
            for e in wl if e.get("override")
        }
        for c in chosen:
            self.choice_log.append({**c, "round": len(self.round_stats)})
        return [{"modality": c["modality"], "url": c["url"],
                 "value": c["value"],
                 "override": override_by.get((c["modality"], c["url"]))}
                for c in chosen]

    def run_round(self, targets: Sequence[Mapping[str, Any]] | None = None) -> dict:
        """一轮感知:对每个 target(含 modality/url/可选 override)感知 + 演化。

        targets=None 时自主选择(self.choose_targets)。显式传 targets 保持
        旧行为(向后兼容)。感知结果同时驱动情绪:outcome(效价)+ surprise
        (失败=意外/高唤起),使 arousal 有内部信号。
        """
        if targets is None:
            targets = self.choose_targets()
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
            # internal feeling: success -> positive valence + low surprise;
            # failure -> negative valence + surprise (arousal rises)
            surprise = 0.0 if ok else 1.0
            gap = float(1.0 - self.body.runtime.belief.confidence().item()) \
                if hasattr(self.body.runtime, "belief") else None
            self.body.feel(outcome=1.0 if ok else -1.0, surprise=surprise, gap=gap)
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
