"""模态无关的统一身体入口:所有感官经同一管道进入认知核心。

MultimodalBody 是"健全身体"的门面——它不重造任何机制,只做接线:
  1. EvidenceFreshnessGate 接受 VersionedEvidence 信封(模态无关,按模态去重)
  2. SemanticSpace 把该模态语义投影到统一 D 维空间
  3. runtime.ingest 是唯一入口,belief 累积证据、ledger 修订
  4. VersionedFactLedger 记录"语义标签"事实(同一事实跨模态命中同一键)

设计依据:PLWM_BLUEPRINT_V1 §3.3 统一意义空间——身体"一出生就是人模态",
所有感官进入记忆前已投影到共享意义空间,无模态级分批次训练。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .evidence import EvidenceFreshnessGate, VersionedEvidence
from .emotion import Effort, Emotion, TimeSense
from .fact_ledger import FactRecord, VersionedFactLedger
from .runtime import PersistentRuntime, RuntimeTick, TrainablePersistentRuntime
from .semantic_space import SemanticSpace

Runtime = PersistentRuntime | TrainablePersistentRuntime


@dataclass
class BodyObservation:
    """一次经统一入口进入身体的观测记录(可审计)。"""

    tick: int
    modality: str
    semantic_dim: int
    projected_norm: float
    accepted: bool
    fact_key: str | None = None


class MultimodalBody:
    """统一身体入口:gate → space → runtime → ledger。"""

    def __init__(
        self,
        world_id: str,
        space: SemanticSpace,
        runtime: Runtime,
        *,
        ledger: VersionedFactLedger | None = None,
        gate: EvidenceFreshnessGate | None = None,
        tick: int = 0,
        emotion: Emotion | None = None,
        effort: Effort | None = None,
        time_sense: TimeSense | None = None,
    ) -> None:
        self.world_id = world_id
        self.space = space
        self.runtime = runtime
        self.ledger = ledger
        self.gate = gate if gate is not None else EvidenceFreshnessGate(world_id)
        self.tick = tick
        self.observations: list[BodyObservation] = []
        # internal-feeling layer (optional, back-compat): emotion / effort /
        # time-sense give the body interoception + time awareness.
        self.emotion = emotion
        self.effort = effort
        self.time_sense = time_sense

    def feel(
        self,
        *,
        reward: float | None = None,
        outcome: float | None = None,
        surprise: float | None = None,
        gap: float | None = None,
        think: bool = False,
    ) -> "EmotionalState | None":
        """Update the internal-feeling layer from an event's outcome/surprise.

        THINK actions spend effort (honest fatigue).  A timestamp is stamped
        so time-sense tracks when this feeling happened.
        """
        if self.effort is not None and think:
            self.effort.spend()
        if self.time_sense is not None:
            self.time_sense.tick()
            self.time_sense.stamp()
        if self.emotion is None:
            return None
        effort_level = self.effort.level if self.effort is not None else None
        return self.emotion.observe(
            surprise=surprise, gap=gap, reward=reward, outcome=outcome, effort=effort_level,
        )

    def observe(
        self,
        modality: str,
        semantic: Tensor,
        *,
        log_likelihood: Sequence[float] | None = None,
        payload: Mapping[str, Any] | None = None,
        fact_value: str | int | bool | None = None,
        confidence: float | None = None,
        skip_ledger: bool = False,
    ) -> RuntimeTick:
        """一次多模态观测。semantic 是该模态的预训练语义向量(非原始感官)。

        fact_value 非 None 时,以 ``{modality}:{value}`` 为键写入账本(强证据
        修订),使同一事实的不同模态呈现汇聚到同一事实键。
        confidence 可选:由来源可信度(TrustRegistry)调制 FactRecord
        confidence;None 时沿用 log_likelihood 推导(向后兼容)。
        skip_ledger:图发现/去重场景不写账本(可选)。
        """
        self.tick += 1
        evidence = VersionedEvidence(
            world_id=self.world_id,
            mechanism_version=self.gate.mechanism_version,
            tick=self.tick,
            modality=modality,
            payload=payload if payload is not None else {},
        )
        decision = self.gate.accept(evidence)
        if not decision.accepted:
            self.observations.append(BodyObservation(self.tick, modality, semantic.shape[-1], 0.0, False))
            return self._noop_tick()
        projected = self.space(modality, semantic)  # 统一 D 维语义向量
        tick_result = self.runtime.ingest(projected.tolist(), log_likelihood)
        fact_key: str | None = None
        if self.ledger is not None and fact_value is not None and not skip_ledger:
            # shared semantic identity key (NOT modality-prefixed): the same
            # fact presented in vision or text converges to the same ledger
            # key (Blueprint §3.3 "no bridging head"). Source stays the
            # modality, so ownership/audit is preserved.
            fact_key = f"identity:{fact_value}"
            if confidence is None:
                if log_likelihood is None:
                    conf = 1.0
                else:
                    conf = min(1.0, max(0.0, float(sum(log_likelihood)) * 2 + 0.5))
            else:
                conf = max(0.0, min(1.0, confidence))
            self.ledger.observe(
                FactRecord(
                    key=fact_key,
                    value=fact_value,
                    source=modality,
                    tick=self.tick,
                    confidence=conf,
                    version=1,
                )
            )
        self.observations.append(
            BodyObservation(
                self.tick, modality, semantic.shape[-1],
                float(projected.norm(dim=-1).mean().item()), True, fact_key,
            )
        )
        return tick_result

    def _noop_tick(self) -> RuntimeTick:
        return self.runtime.ingest(None, None)

    def snapshot(self) -> dict:
        """完整保存 body 状态(runtime + body + ledger + gate + 情绪/时间)。

        供 BodyRunner 长时运行 checkpoint。restore() 反向重建。
        """
        snapshot: dict[str, Any] = {
            "world_id": self.world_id,
            "tick": self.tick,
            "observations": [asdict(obs) for obs in self.observations],
            "runtime": self.runtime.snapshot(),
            "ledger": self.ledger.snapshot() if self.ledger is not None else None,
            "gate": {
                "mechanism_version": self.gate.mechanism_version,
                "last_tick": dict(self.gate._last_tick),
                "accepted_count": self.gate.accepted_count,
                "rejected_stale": self.gate.rejected_stale,
            },
        }
        if self.emotion is not None:
            snapshot["emotion"] = {"_valence": self.emotion._valence, "_arousal": self.emotion._arousal}
        if self.effort is not None:
            snapshot["effort"] = {"_level": self.effort._level}
        if self.time_sense is not None:
            snapshot["time_sense"] = {"elapsed": self.time_sense.elapsed}
        return snapshot

    def restore(self, snapshot: dict) -> None:
        """从 snapshot 重建 body 状态。"""
        self.world_id = snapshot["world_id"]
        self.tick = snapshot["tick"]
        self.observations = [BodyObservation(**obs) for obs in snapshot["observations"]]
        # runtime + ledger restored via their own restore
        restored_runtime = type(self.runtime).restore(
            snapshot["runtime"], self.runtime.core.config.input_dim
            if hasattr(self.runtime, "core") else self.runtime.model.core.config.input_dim,
            self.runtime.belief.log_probs.shape[-1],
            latent_dim=self.runtime.core.config.latent_dim
            if hasattr(self.runtime, "core") else self.runtime.model.core.config.latent_dim,
        )
        # restore runtime state in place (keep the same runtime object)
        self.runtime.core.state = restored_runtime.core.state
        self.runtime.belief = restored_runtime.belief
        self.runtime.ledger = restored_runtime.ledger
        if self.ledger is not None and snapshot.get("ledger") is not None:
            self.ledger.current = {k: FactRecord(**r) for k, r in snapshot["ledger"]["current"].items()}
            self.ledger.history = [FactRecord(**r) for r in snapshot["ledger"]["history"]]
            self.ledger.isolated = [FactRecord(**r) for r in snapshot["ledger"]["isolated"]]
        gate = snapshot["gate"]
        self.gate.mechanism_version = gate["mechanism_version"]
        self.gate._last_tick = dict(gate["last_tick"])
        self.gate.accepted_count = gate["accepted_count"]
        self.gate.rejected_stale = gate["rejected_stale"]
        if self.emotion is not None and "emotion" in snapshot:
            self.emotion._valence = snapshot["emotion"]["_valence"]
            self.emotion._arousal = snapshot["emotion"]["_arousal"]
        if self.effort is not None and "effort" in snapshot:
            self.effort._level = snapshot["effort"]["_level"]
        if self.time_sense is not None and "time_sense" in snapshot:
            self.time_sense.elapsed = snapshot["time_sense"]["elapsed"]
