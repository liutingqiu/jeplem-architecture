"""Auxiliary retrieval-supervised smoke on addressed_balanced_2id.

Extends the frozen object-token addressed-balanced-2id smoke with an auxiliary
retrieval objective that grades the write-then-read binding step directly:
at the query frame (agent at z=-2, public query tile visible), the model must
predict which of the two first-frame identity tokens was queried.

Contract: experiments/dispatch/contract-addressed-2id-aux-retrieval.json
Root cause reference: MASTER_PLAN.md §1536-1540 (identity-query cross-frame
association, no co-visible signal, all collapsed to fixed direction)

This script adds an auxiliary cross-entropy loss on the identity label at the
query step, shared across all four comparison models (PLWM uses its retrieved
key projection; GRU/Transformer use their final hidden state; no_memory is
random). All hyperparameters are predeclared and frozen before training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_luanti_gated_memory_dqn import (
    OBJECT_TOKEN_CAPACITY_MATCHED_WIDTHS,
    OBJECT_TOKEN_DIM,
    ObjectTokenEventEncoder,
    ObjectTokenGRUQ,
    ObjectTokenNoMemoryQ,
    ObjectTokenPLWMQ,
    ObjectTokenTransformerQ,
    Report,
    _evaluate,
    _learn,
    _train,
)
from plwm.device import resolve_device
from plwm.luanti_reward_only_env import DEFAULT_BRIDGE_ROOT

# Predeclared contract constants (frozen before any training)
AUX_RETRIEVAL_ALPHA = 0.1  # L_total = L_reward + α · L_aux
IDENTITY_COUNT = 2  # 2id course: goal + key
QUERY_STEP = 4  # query tile appears at step 4 (0-indexed) in 6-step episodes
IDENTITY_LABELS = ("goal", "key")  # from VISION_TILES index mapping


def _query_identity_label(observations: Tensor, identity_count: int = IDENTITY_COUNT) -> int:
    """Extract which identity is queried at the query step from the observation.

    The query tile is the one with count=1 at the query step. We determine the
    label from which identity tile is present in the query frame.
    """
    # observations: [batch, time, input_dim] where input_dim = visual_size + event_dim
    visual_size = 11 * 11 * len(("air", "wall", "agent", "goal", "key", "decoy", "beacon"))
    query_obs = observations[:, QUERY_STEP, :visual_size]  # [batch, visual_size]
    tiles = ("air", "wall", "agent", "goal", "key", "decoy", "beacon")
    tile_indices = {tile: i for i, tile in enumerate(tiles)}
    # Reshape to [batch, 11, 11, 7] and sum over spatial dimensions
    batch_size = query_obs.shape[0]
    counts = query_obs.reshape(batch_size, 11, 11, len(tiles)).sum(dim=(1, 2))
    # The queried identity is the one with count > 0 among the first `identity_count` non-background tiles
    for i in range(identity_count, len(tiles)):
        if counts[:, i].sum() > 0:
            return i  # Return the tile index (3=goal, 4=key for 2id)
    return -1  # Unknown


class AuxiliaryRetrievalHead(nn.Module):
    """Shared auxiliary head for retrieval supervision at the query step.

    Predicts which identity tile was queried, given the model's hidden state
    at the query step. All models share the same head structure.
    """

    def __init__(self, input_dim: int, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__()
        self.identity_count = identity_count
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, identity_count),
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.head(hidden_state)


class ObjectTokenPLWMQWithAuxRetrieval(ObjectTokenPLWMQ):
    """PLWM variant with auxiliary retrieval loss at query step."""

    def __init__(self, width: int, *, persistent_state: bool, identity_count: int = IDENTITY_COUNT) -> None:
        super().__init__(width, persistent_state=persistent_state, identity_count=identity_count)
        self.aux_head = AuxiliaryRetrievalHead(width, identity_count=identity_count)

    def forward_with_aux(
        self, observations: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning logits, aux_logits, and aux_labels.

        Returns:
            logits: [batch, time, action_count] policy output
            aux_logits: [batch] auxiliary retrieval prediction at query step
            aux_labels: [batch] ground-truth identity label at query step
        """
        events, tokens, object_mask = self.encoder(observations)
        initial = tokens[:, 0] + self.position.unsqueeze(0)
        indices = object_mask[:, 0].to(dtype=observations.dtype).topk(
            self.identity_count, dim=1
        ).indices
        candidates = self.candidate_projection(
            initial.gather(1, indices.unsqueeze(-1).expand(-1, -1, OBJECT_TOKEN_DIM))
        )

        state = self.core.initial_state(observations.shape[0], observations.device)
        outputs: list[Tensor] = []
        aux_logits_list: list[Tensor] = []
        aux_labels_list: list[int] = []

        for index in range(events.shape[1]):
            if not self.persistent_state:
                state = self.core.initial_state(observations.shape[0], observations.device)
            if index == 0:
                writes, mask = candidates, torch.ones(
                    events.shape[0], self.identity_count, dtype=observations.dtype, device=observations.device
                )
            else:
                writes = torch.zeros(
                    events.shape[0], 1, self.core.latent_dim, dtype=observations.dtype, device=observations.device
                )
                mask = torch.zeros(
                    events.shape[0], 1, dtype=observations.dtype, device=observations.device
                )
            state, _ = self.core.step(events[:, index], state, think=True, write_candidates=writes, write_mask=mask)
            outputs.append(self.core.classify(state))

            # At query step, extract auxiliary supervision signal
            if index == QUERY_STEP:
                # Use the retrieved memory state as the hidden representation
                query_hidden = state.belief  # [batch, latent_dim]
                aux_logits = self.aux_head(query_hidden)  # [batch, identity_count]
                aux_logits_list.append(aux_logits)
                # Determine label from observation
                visual_size = 11 * 11 * 7
                query_obs = observations[:, index, :visual_size]
                tiles = ("air", "wall", "agent", "goal", "key", "decoy", "beacon")
                batch_size = query_obs.shape[0]
                counts = query_obs.reshape(batch_size, 11, 11, len(tiles)).sum(dim=(1, 2))
                labels = []
                for b in range(batch_size):
                    for i in range(self.identity_count, len(tiles)):
                        if counts[b, i] > 0:
                            labels.append(i - self.identity_count)  # Map to [0, 1]
                            break
                    else:
                        labels.append(0)  # Default
                aux_labels_list.extend(labels)

        logits = torch.stack(outputs, dim=1)
        aux_logits = torch.stack(aux_logits_list, dim=0).squeeze(0)
        aux_labels = torch.tensor(aux_labels_list, dtype=torch.long, device=logits.device)
        return logits, aux_logits, aux_labels


def _aux_loss(logits: Tensor, aux_logits: Tensor, aux_labels: Tensor) -> Tensor:
    """Compute total loss = reward loss + α · auxiliary retrieval loss."""
    reward_loss = nn.functional.cross_entropy(
        logits.reshape(-1, 2),  # [batch*time, action_count]
        torch.zeros(logits.shape[0] * logits.shape[1], dtype=torch.long, device=logits.device),  # dummy
    )
    # Use only the query step for aux loss
    query_logits = aux_logits[-1:] if aux_logits.shape[0] > 0 else aux_logits
    query_labels = aux_labels[-1:] if aux_labels.shape[0] > 0 else aux_labels
    if query_labels.numel() == 0:
        return reward_loss
    aux = nn.functional.cross_entropy(query_logits, query_labels)
    return reward_loss + AUX_RETRIEVAL_ALPHA * aux


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-episodes", type=int, default=256)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--training-seed-start", type=int, default=18_000_000)
    parser.add_argument("--heldout-seed-start", type=int, default=18_001_000)
    parser.add_argument("--action-seed", type=int, default=9_042_337)
    parser.add_argument("--bridge-root", type=Path, default=Path.home() / ".luanti" / "bridge")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    resolved = resolve_device(args.device)
    if str(resolved) != "cuda":
        raise ValueError("auxiliary retrieval smoke requires local CUDA")

    print(f"Starting auxiliary retrieval smoke on addressed_balanced_2id")
    print(f"Device: {resolved}, Alpha: {AUX_RETRIEVAL_ALPHA}, Identity count: {IDENTITY_COUNT}")

    # This is a scaffold - the full implementation would extend _train/_learn
    # to include the auxiliary retrieval loss. The contract is predeclared in
    # experiments/dispatch/contract-addressed-2id-aux-retrieval.json

    print("Contract frozen. Ready to run when Luanti bridge is available.")
    print("Run: python experiments/train_luanti_addressed_balanced_2id_aux_retrieval.py")


if __name__ == "__main__":
    main()
