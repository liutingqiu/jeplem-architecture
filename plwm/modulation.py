"""情绪 + 可信度 → 记忆调制向量。

modulation_vector:把 Emotion(valence/arousal)与 TrustRegistry 连续信任分
拼成 [B,K] 调制张量,喂给 JEPLMMemory.step 的 modulation 参数(进写门)。
任一源缺失(None)对应通道置 0,通道可选。

salience_from_modulation:确定性映射 arousal/trust → [0,1] salience,用于
每槽显著性缓冲 + 检索重加权。与学习式 write_gate 解耦,保证实验可判据。
valence 只进调制向量供学习(符号编码效价),不进 salience(避免负效价歧义)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .emotion import EmotionalState


def modulation_vector(
    emotion: "EmotionalState | None" = None,
    trust: float | None = None,
    *,
    valence: bool = True,
    device: torch.device | str = "cpu",
) -> Tensor:
    """拼 [1,K] 调制向量 [valence, arousal, trust](K=3)。

    emotion 缺省 → valence/arousal 置 0;trust 缺省 → 置 0。
    valence=False 时跳过 valence 通道,返回 [1,2](K=2)。
    """
    if emotion is None:
        v, a = 0.0, 0.0
    else:
        v, a = emotion.valence, emotion.arousal
    t = trust if trust is not None else 0.0
    if valence:
        return torch.tensor([[v, a, t]], dtype=torch.float32, device=device)
    return torch.tensor([[a, t]], dtype=torch.float32, device=device)


def salience_from_modulation(m: Tensor) -> float:
    """确定性 salience:clamp(arousal + trust, 0, 2),中性 1.0。

    高唤起(0.9)+ 高信任(0.9) → 1.8(强记忆);低唤起/低信任 → <1.0(弱)。
    无调制(None)时 salience=1.0(中性)。m 为 modulation_vector 输出。
    """
    if m.numel() == 0:
        return 1.0
    arousal = float(m[0, 1]) if m.shape[-1] >= 3 else 0.0
    trust_val = float(m[0, -1]) if m.shape[-1] >= 2 else 0.0
    return max(0.0, min(2.0, arousal + trust_val))
