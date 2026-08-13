"""M-rep: the kernel's structured meaning representation (Phase-4 M2).

Every answer the knowledge engine produces is an MRep object, not raw text.
The language mechanism renders MRep -> natural language (plwm/language_gen.py);
a later fluency core (M4/M5) learns that mapping.  M-rep is the interface
between the explicit kernel and the language layer, so its shape is decided
now: answer type, entities, relations, confidence, sources, reasoning -- all
preserved losslessly so any rendering is auditable back to the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FACT = "fact"              # exact stored fact
CHAIN = "chain"            # derived by transitive reasoning
UNKNOWN = "unknown"        # honest unknown (no fact, no derivation)
EXPLANATION = "explanation"  # structured multi-part answer
REVISION = "revision"      # a belief revision with trace


@dataclass(frozen=True)
class Source:
    """One exact fact the answer rests on."""
    rel: str
    subject: str
    object: str


@dataclass
class MRep:
    answer_type: str = UNKNOWN
    question: str = ""
    answer: str | None = None
    confidence: float | None = None        # calibrated correctness probability
    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # (rel, start, end)
    reasoning: list[str] = field(default_factory=list)   # human-readable steps
    sources: list[Source] = field(default_factory=list)
    unknown: bool = False                   # honest-unknown flag
    revision: dict | None = None            # for REVISION: {'from','to','why'}

    # --- constructors ---------------------------------------------------- #
    @classmethod
    def fact(cls, question: str, answer: str, rel: str, subj: str, obj: str,
             confidence: float = 0.98) -> "MRep":
        return cls(answer_type=FACT, question=question, answer=answer,
                   confidence=confidence, entities=[subj, obj],
                   relations=[(rel, subj, obj)], sources=[Source(rel, subj, obj)])

    @classmethod
    def chain(cls, question: str, answer: str, start: str, rel: str, target: str,
              steps: list[tuple[str, str, str]], confidence: float) -> "MRep":
        return cls(answer_type=CHAIN, question=question, answer=answer,
                   confidence=confidence, entities=[start, target],
                   relations=list(steps),
                   reasoning=[f"{s} {r} {o}" for s, r, o in steps],
                   sources=[Source(r, s, o) for s, r, o in steps])

    @classmethod
    def honest_unknown(cls, question: str) -> "MRep":
        return cls(answer_type=UNKNOWN, question=question, answer=None,
                   confidence=None, unknown=True)

    # --- structure guarantee -------------------------------------------- #
    def structure_lossless(self) -> bool:
        """Every rendered answer must trace back to exact sources/confidence."""
        if self.answer_type in (FACT, CHAIN):
            return (self.answer is not None and self.confidence is not None
                    and bool(self.sources) and not self.unknown)
        if self.answer_type == UNKNOWN:
            return self.unknown and self.answer is None
        return True
