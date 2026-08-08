"""Read-only replay audit of real Luanti reward records into the fact ledger."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plwm.fact_ledger import LuantiRewardFactLedger, VersionedFactLedger


BRIDGE_LOG = Path(r"D:\PLWM-Luanti\bridge\observations.jsonl")


@dataclass
class EpisodeResult:
    episode_id: str
    prerequisite_values: list[float]
    facts_complete: bool
    restored_complete: bool


@dataclass
class Report:
    protocol: str
    source_log: str
    selected_episodes: int
    episodes: list[EpisodeResult]
    accepted: bool
    artifact: str


def _records(path: Path) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("kind") != "reward" or payload.get("modality") != "luanti-reward":
                continue
            episode_id = payload.get("episode_id")
            reward = payload.get("payload")
            tick = payload.get("tick")
            if not isinstance(episode_id, str) or not isinstance(reward, dict) or not isinstance(tick, int):
                continue
            value = reward.get("value")
            if value not in (0.15, 0.2, 0.20000000000000001):
                continue
            selected.setdefault(episode_id, []).append({
                "value": float(value),
                "_plwm_provenance": {
                    "source": "luanti-reward",
                    "tick": tick,
                    "request_id": reward.get("caused_by"),
                },
            })
    return selected


def run(*, source_log: Path = BRIDGE_LOG, episodes: int = 8, artifact: Path | None = None) -> Report:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not source_log.exists():
        raise FileNotFoundError(source_log)
    candidates = _records(source_log)
    qualified = [(episode_id, rewards) for episode_id, rewards in candidates.items() if [item["value"] for item in rewards] == [0.15, 0.2]]
    if len(qualified) < episodes:
        raise RuntimeError("insufficient real two-fact reward episodes in bridge replay")
    rows = []
    for episode_id, rewards in qualified[-episodes:]:
        ledger = LuantiRewardFactLedger()
        for reward in rewards:
            ledger.observe_reward(reward)
        restored = VersionedFactLedger.restore(ledger.ledger.snapshot())
        facts_complete = (
            ledger.ledger.current.get("charge_verified") is not None
            and ledger.ledger.current.get("key_verified") is not None
            and all(record.request_id and record.source == "luanti-reward" for record in ledger.ledger.current.values())
        )
        rows.append(EpisodeResult(
            episode_id, [item["value"] for item in rewards], facts_complete,
            restored.snapshot() == ledger.ledger.snapshot(),
        ))
    target = artifact or ROOT / "artifacts" / "luanti-reward-ledger-replay-report.json"
    target.parent.mkdir(exist_ok=True)
    report = Report(
        "luanti_reward_ledger_replay_v1: read-only replay of existing local Luanti bridge reward records; existing reward envelope source/tick/caused_by fields become bounded ledger provenance, with no game action or hidden-state access.",
        str(source_log), episodes, rows, all(item.facts_complete and item.restored_complete for item in rows), str(target),
    )
    target.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(asdict(outcome), indent=2))
    if not outcome.accepted:
        raise SystemExit("Luanti reward ledger replay did not meet its audit contract")
