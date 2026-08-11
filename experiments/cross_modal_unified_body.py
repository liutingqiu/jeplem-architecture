"""跨模态一致性实验:统一语义空间让"同一事实"跨模态命中同一记忆。

Blueprint §3.3 I 族课程:同一事实(如"当前可见身份是 key")分别用
modality="vision"(冻结 CNN 语义)与 modality="text"(语言端口事件)呈现,
经 SemanticSpace 投影到统一 D 维空间。判据:
  C1 模态距离:同一事实的 vision/text 投影余弦 > 0.9;异事实对 < 0.6。
  C2 记忆命中:视觉事实写入 JEPLMMemory 槽位后,文本呈现经同一语义向量
     在内容寻址下召回同一事实的召回强度最高。
  C3 无桥接头:只用 SemanticSpace 投影,无跨模态直接拼接(代码审查)。
Criterion: C1 同 > 0.9 且异 < 0.6; C2 同事实召回为 max; C3 恒真(架构保证)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.semantic_space import ModalityEncoder, SemanticSpace, cosine
from plwm.jeplem_memory import JEPLMMemory
from plwm.multimodal_body import MultimodalBody
from plwm.runtime import PersistentRuntime
from plwm.fact_ledger import VersionedFactLedger
from plwm.evidence import EvidenceFreshnessGate

DIM = 16
N_FACTS = 2  # key(0) / ball(1)


def vision_semantic(identity: int, rng: random.Random) -> Tensor:
    """Frozen-CNN-like semantic: 2-dim key/ball one-hot with small noise."""
    sem = torch.zeros(2)
    sem[identity] = 1.0
    return sem + rng.uniform(-0.02, 0.02)


def text_semantic(identity: int, rng: random.Random) -> Tensor:
    """Language-port-like event: 5-dim (route one-hot + confidence).

    Matches JsonLanguagePort.decode_evidence: 2-class route one-hot at
    [0]/[1] + confidence at [4].  Indices 2-3 are unused by the real port.
    """
    event = torch.zeros(5)
    event[identity] = 1.0
    event[4] = rng.uniform(0.8, 1.0)
    return event


def build_space(seed: int) -> SemanticSpace:
    """Vision(2→D) + Text(5→D) encoders, aligned via shared identity anchors."""
    torch.manual_seed(seed)
    space = SemanticSpace(dim=DIM)
    space.register("vision", ModalityEncoder(2, dim=DIM, state_dim=2))
    space.register("text", ModalityEncoder(5, dim=DIM, state_dim=5))
    rng = random.Random(seed * 7 + 1)
    examples: dict[str, list[tuple[Tensor, int]]] = {
        "vision": [(vision_semantic(f, rng), f) for f in range(N_FACTS) for _ in range(8)],
        "text": [(text_semantic(f, rng), f) for f in range(N_FACTS) for _ in range(8)],
    }
    space.fit_anchors(examples, updates=200)
    return space


def run(*, seeds: tuple[int, ...] = (444, 555, 666),
        episodes: int = 24, artifact: Path | None = None) -> dict:
    all_c1_same: list[float] = []
    all_c1_diff: list[float] = []
    all_c2_hits: list[bool] = []
    ledger = VersionedFactLedger(trusted_sources={"vision", "text"}, revision_confidence=0.9)
    for seed in seeds:
        rng = random.Random(seed)
        space = build_space(seed)
        # C1: pairwise cosine within a fact (vision vs text), and across facts
        for fact in range(N_FACTS):
            v = space("vision", vision_semantic(fact, rng))
            t = space("text", text_semantic(fact, rng))
            all_c1_same.append(float(cosine(v, t).item()))
        for fa in range(N_FACTS):
            for fb in range(N_FACTS):
                if fa == fb:
                    continue
                v = space("vision", vision_semantic(fa, rng))
                t = space("text", text_semantic(fb, rng))
                all_c1_diff.append(float(cosine(v, t).item()))
        # C2: memory hit — vision writes slots, text recall retrieves the same fact
        # shared_address=True makes the read query share the slot-key projection,
        # so multi-modal semantic closeness transfers to content-addressed recall.
        memory = JEPLMMemory(DIM, latent=32, slots=4, identity_count=N_FACTS,
                             shared_address=True)
        memory.reset(1, torch.device("cpu"))
        valid = torch.ones(1, dtype=torch.bool)
        slot_of_fact: dict[int, int] = {}
        for fact in range(N_FACTS):
            vis_sem = space("vision", vision_semantic(fact, rng))  # D-dim
            memory.step(vis_sem.unsqueeze(0), valid)
            # FIFO: the just-written key lands in the last written slot
            slot_of_fact[fact] = memory.slots - N_FACTS + fact
        # text probe for the same fact must recall that fact's memory strongest
        for fact in range(N_FACTS):
            txt_sem = space("text", text_semantic(fact, rng))  # D-dim
            encoded = memory.encoder(txt_sem.unsqueeze(0))
            qv = memory.query(encoded)
            keys = memory.export_state()[1]
            sims = torch.einsum("bd,bmd->bm", qv, keys) / (keys.shape[-1] ** 0.5)
            hit = bool(sims[0].argmax(-1).item() == slot_of_fact[fact])
            all_c2_hits.append(hit)

    c1_same = float(np.mean(all_c1_same))
    c1_diff = float(np.mean(all_c1_diff))
    c2_hit = float(np.mean(all_c2_hits))
    accepted = c1_same > 0.9 and c1_diff < 0.6 and c2_hit >= 0.9
    report = {
        "protocol": "cross_modal_unified_body_v1: same fact presented as vision (frozen-CNN 2-dim semantic) and text (language-port 5-dim event) both project into one D=16 SemanticSpace and hit the same memory slot. C1 modal cosine (same > 0.9, diff < 0.6); C2 cross-modal memory recall hit; C3 no bridging head (architecture).",
        "seeds": list(seeds),
        "results": {
            "c1_same_cosine": round(c1_same, 3),
            "c1_diff_cosine": round(c1_diff, 3),
            "c2_memory_hit": round(c2_hit, 3),
        },
        "accepted": accepted,
        "artifact": str(artifact or ROOT / "artifacts" / "cross-modal-unified-body.json"),
    }
    (artifact or ROOT / "artifacts" / "cross-modal-unified-body.json").parent.mkdir(exist_ok=True)
    (artifact or ROOT / "artifacts" / "cross-modal-unified-body.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact=args.artifact), indent=2))
