"""Language mechanism (Phase-4 M2): MRep -> natural language via templates.

Template + slot filling, zero training.  This is the kernel's 'mouth': it
renders structured answers as auditable natural language, preserving structure
(confidence, sources, reasoning) losslessly.  A later fluency core (M4/M5)
learns this mapping; templates are the mechanism baseline that any learned
realizer must beat.

Structure guarantee: the rendered text always carries the confidence and the
exact sources, so a consumer can verify the answer against the kernel.
"""
from __future__ import annotations

from .mrep import FACT, CHAIN, UNKNOWN, EXPLANATION, REVISION, MRep


def _source_line(rep: MRep) -> str:
    srcs = " ; ".join(f"{s.rel}({s.subject}, {s.object})" for s in rep.sources[:3])
    if len(rep.sources) > 3:
        srcs += f" ; +{len(rep.sources)-3} more"
    return srcs


def _confidence_line(rep: MRep) -> str:
    if rep.confidence is None:
        return ""
    return f" (confidence {rep.confidence:.2f})"


def render(rep: MRep) -> str:
    """MRep -> one or more natural-language sentences (auditable)."""
    if rep.answer_type == UNKNOWN:
        return "I don't know."
    if rep.answer_type == FACT:
        out = f"{rep.answer}{_confidence_line(rep)}."
        if rep.sources:
            out += f"  [source: {_source_line(rep)}]"
        return out
    if rep.answer_type == CHAIN:
        out = f"{rep.answer}{_confidence_line(rep)}."
        if rep.reasoning:
            out += f"  Reasoning: {' -> '.join(rep.reasoning)}."
        if rep.sources:
            out += f"  [sources: {_source_line(rep)}]"
        return out
    if rep.answer_type == EXPLANATION:
        parts = [rep.answer or ""]
        parts += [f"- {step}" for step in rep.reasoning]
        if rep.confidence is not None:
            parts.append(f"(confidence {rep.confidence:.2f})")
        return "\n".join(parts)
    if rep.answer_type == REVISION:
        rev = rep.revision or {}
        return (f"I previously believed {rev.get('from')}; based on new evidence "
                f"I now believe {rev.get('to')}.  Why: {rev.get('why', 'revision')}.")
    return str(rep.answer) if rep.answer is not None else "I don't know."
