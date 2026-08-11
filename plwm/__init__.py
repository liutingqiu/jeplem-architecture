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
    from .events import EnvironmentEvent, EpisodeTrace, observation_payload
    from .minecraft import MinecraftFileBridge, MinecraftRuntimeSession, MinecraftStateEncoder
    from .emotion import Effort, Emotion, EmotionalState, TimeSense
    from .semantic_space import ModalityEncoder, SemanticSpace, cosine
    from .multimodal_body import BodyObservation, MultimodalBody
    from .web import JsonSchemaValidator, WebFetchBackend, WebFetchError, WebResource
    from .web_sensors import WebStructSensor, WebTextSensor
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
    "EnvironmentEvent",
    "EpisodeTrace",
    "observation_payload",
    "MinecraftFileBridge",
    "MinecraftRuntimeSession",
    "MinecraftStateEncoder",
    "Effort",
    "Emotion",
    "EmotionalState",
    "TimeSense",
    "ModalityEncoder",
    "SemanticSpace",
    "cosine",
    "BodyObservation",
    "MultimodalBody",
    "JsonSchemaValidator",
    "WebFetchBackend",
    "WebFetchError",
    "WebResource",
    "WebStructSensor",
    "WebTextSensor",
]
