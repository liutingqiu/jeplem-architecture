"""Persistent Latent World Model research prototype."""

from .core import (
    CognitiveAction,
    MemorySlot,
    PLWM,
    PLWMConfig,
    PersistentState,
    StepResult,
)

try:
    from .adaptive import AdaptivePonderer
    from .belief import CategoricalBelief
    from .commitments import Commitment, CommitmentLedger, CommitmentStatus
    from .runtime import PersistentRuntime, RuntimeTick, TrainablePersistentRuntime, TrainableRuntimeTick
    from .multiscale import MultiScalePLWM, MultiScaleState
    from .torch_model import StatelessBaseline, TorchCognitiveState, TorchPLWM
    from .world_model import ActionConditionedPLWM, StatelessWorldModel, WorldState
    from .world_runtime import WorldModelPersistentRuntime, WorldRuntimeTick
    from .language import JsonLanguagePort, LanguageEvidence, OllamaBackend
except ModuleNotFoundError:  # The dependency-free core remains importable.
    pass

__all__ = [
    "CognitiveAction",
    "MemorySlot",
    "PLWM",
    "PLWMConfig",
    "PersistentState",
    "StepResult",
    "StatelessBaseline",
    "TorchCognitiveState",
    "TorchPLWM",
    "ActionConditionedPLWM",
    "StatelessWorldModel",
    "WorldState",
    "AdaptivePonderer",
    "MultiScalePLWM",
    "MultiScaleState",
    "CategoricalBelief",
    "Commitment",
    "CommitmentLedger",
    "CommitmentStatus",
    "PersistentRuntime",
    "RuntimeTick",
    "TrainablePersistentRuntime",
    "TrainableRuntimeTick",
    "WorldModelPersistentRuntime",
    "WorldRuntimeTick",
    "JsonLanguagePort",
    "LanguageEvidence",
    "OllamaBackend",
]
