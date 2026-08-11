"""身体的内在感受层:情绪(valence+arousal)、内感受(Effort)、时间感(TimeSense)。

设计依据:
- 情绪 = 内在标注机制(非主观感受,项目 Non-Claims)。valence(效价:结果
  好坏)由外部 outcome/reward 驱动;arousal(唤醒:内部强度)由内部信号
  (预测误差/belief gap/Effort)驱动——两者分离,符合情绪本质。
- 内感受不造假:Effort 是 THINK 动作的诚实疲劳累加器(不是"饥饿"类假模拟)。
- 时间感 = elapsed(真实持续时长)+ recency(记忆随时间衰减),感知"活过多久"。

确定性状态更新(非可学习),指数滑动平均使情绪有时间动力学(衰减),
与 belief.py / fact_ledger.py 的纪律一致。全部参数可空(None 安全),
Persistent 和 Trainable runtime 都可用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalState:
    """情绪的当前内部状态。valence 在 [-1,1](负=坏,正=好),
    arousal 在 [0,1](低=平静,高=唤起)。"""

    valence: float
    arousal: float


def _lowpass(previous: float, incoming: float, tau: float) -> float:
    """指数滑动平均(低通):alpha = 1 - exp(-1/tau),使状态有衰减动力学。"""
    if tau <= 0:
        return incoming
    alpha = 1.0 - math.exp(-1.0 / tau)
    return (1.0 - alpha) * previous + alpha * incoming


class Emotion:
    """内在标注机制:把外部结果(valence)与内部强度(arousal)汇聚成状态。

    用法:emotion.observe(reward=..., surprise=..., gap=..., effort=...)
    返回 EmotionalState。任何参数为 None 时该通道不变。
    """

    def __init__(self, v_tau: float = 4.0, a_tau: float = 3.0) -> None:
        self.v_tau = v_tau
        self.a_tau = a_tau
        self._valence = 0.0
        self._arousal = 0.0

    def observe(
        self,
        *,
        surprise: float | None = None,
        gap: float | None = None,
        reward: float | None = None,
        outcome: float | None = None,
        effort: float | None = None,
    ) -> EmotionalState:
        """更新并返回情绪状态。

        valence: 低通(reward 或 outcome 的符号值)。好结果→正效价,坏→负。
        arousal: 低通(max(surprise, gap, |reward|, effort))。任一强→高唤醒。
        """
        if reward is not None:
            self._valence = _lowpass(self._valence, max(-1.0, min(1.0, reward)), self.v_tau)
        elif outcome is not None:
            self._valence = _lowpass(self._valence, max(-1.0, min(1.0, outcome)), self.v_tau)
        intensity: float | None = None
        for value in (surprise, gap, None if reward is None else abs(reward), effort):
            if value is None:
                continue
            intensity = value if intensity is None else max(intensity, value)
        if intensity is not None:
            self._arousal = _lowpass(self._arousal, max(0.0, min(1.0, intensity)), self.a_tau)
        return EmotionalState(self._valence, self._arousal)

    @property
    def state(self) -> EmotionalState:
        return EmotionalState(self._valence, self._arousal)


class Effort:
    """内感受:THINK 动作的泄漏疲劳累加器——"我连续工作多久"。

    这是诚实的内部负荷信号(不是虚构的饥饿/口渴)。THINK 时 spend() 增加,
    每 tick 按 decay 衰减,模拟疲劳的积累与恢复。
    """

    def __init__(self, decay: float = 0.9) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self._level = 0.0

    def spend(self, amount: float = 1.0) -> None:
        self._level = min(1.0, self._level + amount)

    def tick(self) -> None:
        self._level *= self.decay

    @property
    def level(self) -> float:
        return self._level


class TimeSense:
    """时间感:elapsed(真实持续时长)+ recency(记忆随时间衰减)。

    tick(dt) 累计 elapsed。THINK 可比观察消耗更大 dt(思考更"耗时")。
    stamp() 给事件盖时间戳;since() 回答"距现在多久";recency() 让旧记忆
    渐淡(exp 衰减),从而"感觉到时间流逝"。
    """

    def __init__(self, tau: float = 20.0) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.elapsed = 0.0

    def tick(self, dt: float = 1.0) -> None:
        self.elapsed += dt

    def stamp(self) -> float:
        return self.elapsed

    def since(self, t: float) -> float:
        return max(0.0, self.elapsed - t)

    def recency(self, t: float) -> float:
        return math.exp(-self.since(t) / self.tau)
