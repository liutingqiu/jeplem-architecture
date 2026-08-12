"""Tool prior (B4, 2026-08-12): innate bias to use and learn tools.

Infant-analogy: a child picks up a stick to reach, a cup to drink -- it learns
"what tool works in what situation" by trying and observing outcomes, before
it has full knowledge.  Our architecture needs the same innate bias: a TOOL
ABstraction where an action invokes an external tool/function and the result
comes back as an observation, so the surprise engine can learn 'which tool is
effective under which condition'.

This is a learning BIAS (how to interact with external functions), not
knowledge: it does not know which tool is good for what; it only defines the
action -> tool -> observation loop and lets surprise learn effectiveness.

Implementation: ToolPrior exposes a register of tools (name + callable that
returns an outcome signal).  A tool call is (condition_features, tool_name,
outcome).  The prior tracks per-condition per-tool outcome statistics, giving
the surprise engine fields like 'tool_X_success_rate' to bootstrap on.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Tool:
    name: str
    call: Callable[[dict], float]     # (args) -> outcome signal (e.g. -1..1)


@dataclass
class ToolPrior:
    """Tool-call abstraction + per-condition effectiveness tracking."""

    tools: Mapping[str, Tool] = field(default_factory=dict)
    _outcomes: dict = field(default_factory=lambda: defaultdict(list))  # (cond,tool)->[outcome]
    _calls: dict = field(default_factory=lambda: defaultdict(int))

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def call(self, tool_name: str, condition: str, args: dict) -> float:
        """Invoke a tool; record the outcome under (condition, tool)."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return 0.0
        outcome = tool.call(args)
        self._outcomes[(condition, tool_name)].append(outcome)
        self._calls[(condition, tool_name)] += 1
        return outcome

    def success_rate(self, condition: str, tool_name: str) -> float:
        """Fraction of outcomes above 0 (success) for this condition+tool."""
        outcomes = self._outcomes.get((condition, tool_name), [])
        if not outcomes:
            return 0.0
        return sum(1 for o in outcomes if o > 0) / len(outcomes)

    def tool_features(self, condition: str, tool_names: list[str]) -> dict:
        """Per-tool success rates under a condition, as surprise-engine fields."""
        feats = {}
        for tn in tool_names:
            feats[f"tool_{tn}_rate"] = round(self.success_rate(condition, tn), 3)
            feats[f"tool_{tn}_calls"] = self._calls.get((condition, tn), 0)
        return feats


def make_tools() -> dict:
    """Synthetic tools: 'search' works on text conditions, 'calc' on numeric."""
    tools = {}
    tools["search"] = Tool("search", lambda args: 1.0 if args.get("kind") == "text" else -1.0)
    tools["calc"] = Tool("calc", lambda args: 1.0 if args.get("kind") == "numeric" else -1.0)
    return tools


if __name__ == "__main__":
    tp = ToolPrior(make_tools())
    # text condition: search should work, calc fail
    print("text condition:", tp.tool_features("text", ["search", "calc"]))
    print("numeric condition:", tp.tool_features("numeric", ["search", "calc"]))
