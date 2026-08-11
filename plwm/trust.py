"""来源可信度:连续信任分的演化 registry。

账本的 trusted_sources 是硬白名单(是否可信的布尔),这里升级为连续信任
分数:每个来源有 [0,1] 的 trust,被证实(confirm)升分、被证伪(refute)
降分。传感器读 score() 作 FactRecord.confidence,使账本的弱/强证据隔离
变为连续信任调制——高信任来源的证据更易被接受,低信任来源被隔离。
"""
from __future__ import annotations


class TrustRegistry:
    """source -> 连续信任分 [0,1],随 confirm/refute 演化。"""

    def __init__(self, *, prior: float = 0.5, alpha: float = 0.2, beta: float = 0.2) -> None:
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0,1]")
        if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("alpha/beta must be in (0,1)")
        self.prior = prior
        self.alpha = alpha
        self.beta = beta
        self._scores: dict[str, float] = {}
        self._events: list[dict] = []

    def score(self, source: str) -> float:
        return self._scores.get(source, self.prior)

    def report(self, source: str, correct: bool) -> float:
        """记录一次来源的表现。correct=True 升分,False 降分。"""
        current = self.score(source)
        if correct:
            updated = current + self.alpha * (1.0 - current)
        else:
            updated = current - self.beta * current
        self._scores[source] = max(0.0, min(1.0, updated))
        self._events.append({"source": source, "correct": correct,
                             "score": self._scores[source]})
        return self._scores[source]

    def is_trusted(self, source: str, threshold: float = 0.8) -> bool:
        return self.score(source) >= threshold

    def snapshot(self) -> dict:
        return {
            "prior": self.prior,
            "alpha": self.alpha,
            "beta": self.beta,
            "scores": dict(self._scores),
            "events": list(self._events),
        }
