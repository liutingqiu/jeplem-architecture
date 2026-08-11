"""统一语义空间:多模态 → 同一 D 维语义空间,汇入同一认知管道。

设计依据:PLWM_BLUEPRINT_V1 §3.3"统一意义空间/人类原生模态"——跨模态
一致性 = 同一事实的文本/视觉等价命中同一条记忆,无桥接头。每个模态是
一个可插拔编码器,把预训练语义(视觉 2 维、文本 5 维…)投影到共享的
D 维语义向量,然后进 JEPLMMemory / runtime.ingest。

未来模态(Sound、Touch…)只需 SemanticSpace.register(name, encoder),
不需要改任何认知管线。
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class ModalityEncoder(nn.Module):
    """一个模态的可插拔编码器:预训练语义 → D 维语义向量。

    ``projection`` 把上游语义(如视觉 softmax 2 维 / 文本事件 5 维)线性
    投影到统一语义空间。``state_dim`` 可选,供宿主构造 RuntimeTick 用。
    """

    def __init__(
        self,
        semantic_dim: int,
        *,
        dim: int,
        projection: nn.Module | None = None,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.dim = dim
        self.state_dim = state_dim
        if projection is None:
            projection = nn.Sequential(nn.Linear(semantic_dim, dim), nn.Tanh())
        self.projection = projection

    def encode(self, semantic: Tensor) -> Tensor:
        return self.projection(semantic)


class SemanticSpace(nn.Module):
    """注册表:modality 名字 → ModalityEncoder,统一投影到同一 D 维空间。

    ``__call__(modality, semantic)`` 返回 D 维语义向量;未知模态抛
    ValueError(防拼写错误静默通过)。

    跨模态一致性不是"独立随机投影"能保证的——两个独立 Linear 层不会把
    同一事实自动映射到同一区域。``fit_anchors`` 用一组共享的"身份锚"
    作为监督目标,把每个模态的编码器对齐到同一语义区域(Blueprint §3.3
    无桥接头:锚在语义空间,不在感知前端)。
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("semantic space dim must be positive")
        self.dim = dim
        self.encoders: nn.ModuleDict = nn.ModuleDict()
        self._anchors_fixed: Tensor | None = None  # [N_identities, D] aligned anchors

    def register(self, name: str, encoder: ModalityEncoder) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("modality name must be a non-empty string")
        if encoder.dim != self.dim:
            raise ValueError(f"encoder dim {encoder.dim} != semantic space dim {self.dim}")
        self.encoders[name] = encoder

    def fit_anchors(
        self,
        examples: dict[str, list[tuple[Tensor, int]]],
        *,
        updates: int = 200,
        lr: float = 1e-2,
        anchor_distance: float = 2.0,
    ) -> dict[str, float]:
        """Align every registered encoder to a shared identity-anchor set.

        ``examples[modality]`` is a list of (semantic, identity) pairs.  Uses
        a CLIP-style contrastive alignment: same-identity cross-modal pairs
        are pulled together, different-identity pairs are pushed apart.  No
        bridging head -- the encoders learn to land the same fact (presented
        in any modality) in the same semantic region directly.
        Returns per-modality final cosine-to-own-anchor after alignment.
        """
        if not self.encoders:
            raise ValueError("no encoders registered before fit_anchors")
        if not examples or not all(m in self.encoders for m in examples):
            raise ValueError("examples must cover exactly the registered modalities")
        modalities = list(examples.keys())
        identities = sorted({ident for pairs in examples.values() for _, ident in pairs})
        if not identities:
            raise ValueError("no identities in examples")
        n_identities = max(identities) + 1
        optimizer = torch.optim.Adam(self.encoders.parameters(), lr=lr)
        # CLIP-style InfoNCE alignment: for each pair of modalities, build the
        # cross-modal similarity matrix of all examples.  The loss is the mean
        # cross-entropy of the diagonal (same-identity pairs) -- each vector
        # must retrieve its own-identity partner among every identity.  This
        # provably does not collapse (unlike pulling to anchors), because
        # collapsing makes every row uniform and cross-entropy stays high.
        for _ in range(updates):
            optimizer.zero_grad()
            total = torch.tensor(0.0)
            for m_a in range(len(modalities)):
                for m_b in range(m_a + 1, len(modalities)):
                    ma, mb = modalities[m_a], modalities[m_b]
                    # align every example of ma against every example of mb;
                    # labels = the identity of each example (one-hot matching)
                    a_sem = [sem for sem, _ in examples[ma]]
                    b_sem = [sem for sem, _ in examples[mb]]
                    a_ids = [ident for _, ident in examples[ma]]
                    b_ids = [ident for _, ident in examples[mb]]
                    a_vecs = torch.stack([self.encoders[ma].encode(s.unsqueeze(0)).squeeze(0) for s in a_sem])
                    b_vecs = torch.stack([self.encoders[mb].encode(s.unsqueeze(0)).squeeze(0) for s in b_sem])
                    # normalized cosine matrix [A, B]
                    na = a_vecs / a_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    nb = b_vecs / b_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    logits = na @ nb.t() / 0.1
                    a_targets = torch.tensor([b_ids.index(ident) for ident in a_ids], dtype=torch.long)
                    b_targets = torch.tensor([a_ids.index(ident) for ident in b_ids], dtype=torch.long)
                    total = total + torch.nn.functional.cross_entropy(logits, a_targets)
                    total = total + torch.nn.functional.cross_entropy(logits.t(), b_targets)
                    # direct same-fact pull WITH cross-fact repulsion: each a_vec
                    # toward its own identity's mean b_vec (detached), away from
                    # every other identity's mean b_vec.  The repulsion prevents
                    # the pull from dragging different facts together.
                    b_means = {}
                    for ident in identities:
                        b_sel = torch.stack([v for v, idt in zip(b_vecs, b_ids) if idt == ident])
                        if b_sel.numel() > 0:
                            b_means[ident] = b_sel.mean(0).detach()
                    for v, idt in zip(a_vecs, a_ids):
                        if idt not in b_means:
                            continue
                        total = total + 1.5 * (1.0 - cosine(v, b_means[idt]))
                        for other, center in b_means.items():
                            if other == idt:
                                continue
                            total = total + 0.5 * cosine(v, center).clamp_min(0.0)
            total.backward()
            optimizer.step()
        # final report: same-fact cross-modal cosine per modality
        report: dict[str, float] = {}
        for ma in modalities:
            cosines = []
            for identity in identities:
                a_sem = [sem for sem, ident in examples[ma] if ident == identity][0]
                other_mod = next(m for m in modalities if m != ma)
                b_sem = [sem for sem, ident in examples[other_mod] if ident == identity][0]
                va = self.encoders[ma].encode(a_sem.unsqueeze(0)).squeeze(0)
                vb = self.encoders[other_mod].encode(b_sem.unsqueeze(0)).squeeze(0)
                cosines.append(float(cosine(va, vb).item()))
            report[ma] = float(sum(cosines) / len(cosines))
        self._anchors_fixed = None
        return report

    def __call__(self, modality: str, semantic: Tensor) -> Tensor:
        if modality not in self.encoders:
            raise ValueError(f"unknown modality: {modality!r} (registered: {sorted(self.encoders)})")
        encoder = self.encoders[modality]
        if semantic.shape[-1] != encoder.semantic_dim:
            raise ValueError(
                f"semantic dim {semantic.shape[-1]} != encoder {encoder.semantic_dim} for {modality!r}"
            )
        return encoder.encode(semantic)


def cosine(a: Tensor, b: Tensor) -> Tensor:
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (a * b).sum(dim=-1)


def _orthonormal(rows: Tensor) -> Tensor:
    """Gram-Schmidt orthonormalize a [N, D] matrix (rows pairwise orthogonal)."""
    rows = rows.clone()
    for i in range(rows.shape[0]):
        for j in range(i):
            rows[i] = rows[i] - (rows[i] @ rows[j]) * rows[j]
        rows[i] = rows[i] / rows[i].norm().clamp_min(1e-8)
    return rows
