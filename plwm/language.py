"""Language as a bounded I/O surface for a persistent cognitive runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Protocol
from urllib.request import Request, urlopen


class TextBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class LanguageEvidence:
    event: tuple[float, float, float, float, float]
    log_likelihood: tuple[float, float]
    raw: str


class OllamaBackend:
    """Minimal Ollama HTTP backend; credentials and state never live here."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: float = 30.0,
        json_mode: bool = True,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.json_mode = json_mode

    def complete(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False}
        if self.json_mode:
            body["format"] = "json"
        payload = json.dumps(body).encode("utf-8")
        request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: local user-selected endpoint
            result = json.loads(response.read().decode("utf-8"))
        text = result.get("response")
        if not isinstance(text, str):
            raise ValueError("Ollama response did not contain text")
        return text


@dataclass
class JsonLanguagePort:
    """Converts text into declared evidence; it owns neither beliefs nor goals."""

    backend: TextBackend | None = None
    emitted: list[str] = field(default_factory=list)

    EXTRACTION_PROMPT = (
        "Extract route evidence. Reply with one JSON object only, with exactly "
        'the keys "route" ("left" or "right") and "confidence" (a number from 0.5 to 1.0). Text: '
    )

    def decode_evidence(self, raw: str) -> LanguageEvidence:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("language evidence must be strict JSON") from error
        route = payload.get("route")
        confidence = payload.get("confidence")
        if route not in {"left", "right"} or not isinstance(confidence, (int, float)) or not 0.5 <= confidence <= 1.0:
            raise ValueError("language evidence has invalid route or confidence")
        value = 1 if route == "right" else 0
        event = [0.0] * 5
        event[value] = 1.0
        event[4] = float(confidence)
        likelihood = (math.log(1.0 - confidence + 1e-8), math.log(confidence))
        if value == 0:
            likelihood = tuple(reversed(likelihood))
        return LanguageEvidence(tuple(event), likelihood, raw)

    def encode_observation(self, text: str) -> LanguageEvidence:
        if self.backend is None:
            raise RuntimeError("no language backend configured")
        return self.decode_evidence(self.backend.complete(self.EXTRACTION_PROMPT + text))

    def emit_action(self, action: str) -> str:
        if action not in {"observe", "left", "right", "sleep", "think"}:
            raise ValueError("unsupported cognitive action")
        text = f"action:{action}"
        self.emitted.append(text)
        return text

    def apply_to_runtime(self, runtime, evidence: LanguageEvidence):
        """Forward declared evidence; runtime authority decides all state changes."""
        return runtime.ingest(evidence.event, evidence.log_likelihood)
