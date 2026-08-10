"""JEPLM 预测/信念层：JEPA 式 latent 预测头（架构最后一根支柱的组件）。

设计依据（JEPLM_MEMORY_DESIGN D3 + 2026-08 证据）：
- JEPA 核心：在 latent 空间预测未来（不预测像素），预测误差驱动学习
- 我们的实证：重建式 surprise 预测在 RL 是负担（噪声）；预测应做**决策侧**
  （置信度/门控信号），不混入策略梯度主损失
- 本模块：从当前 features（state + recalled）预测下一帧的 latent（target
  encoder 的 stop-grad 输出），prediction error 作为置信度/信念更新信号

API：
- JEPLMPredictor(projector) 封装 predict_next(features_t) -> pred_latent
- prediction_error(pred, target_latent) -> scalar per-sample
- 置信度特征：conf = 1 - sigmoid(err)（或直接用 err 特征），供宿主决策侧使用

与记忆协同：预测误差大 = 世界状态出乎模型意料 = 信念置信度应下调 /
该帧内容值得注意（决策侧关注度信号），但**不直接进策略梯度主损失**。
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor


class JEPLMPredictor(nn.Module):
    """Latent next-frame predictor with a stop-grad target (JEPA style).

    Trained on a latent-space regression loss against the *next* frame's
    representation produced by the same encoder (stop-gradient), so the
    predictor learns the dynamics of the semantic space without collapsing
    the encoder (no backprop through the target).
    """

    def __init__(self, latent: int, hidden: int = 128) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent),
        )

    def forward(self, features_t: Tensor) -> Tensor:
        """Predict the next-frame latent from current features."""
        return self.predictor(features_t)

    @staticmethod
    def prediction_error(predicted: Tensor, target_latent: Tensor) -> Tensor:
        """Per-sample L2 error between predicted and (stop-grad) target."""
        return (predicted - target_latent.detach()).pow(2).mean(dim=-1)

    @staticmethod
    def confidence(prediction_error: Tensor) -> Tensor:
        """Map prediction error to a [0,1] confidence signal (belief side)."""
        return 1.0 - torch.sigmoid(prediction_error)
