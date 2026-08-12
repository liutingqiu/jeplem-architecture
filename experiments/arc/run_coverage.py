"""ARC-1 coverage scan: how much of the real ARC-AGI-1 benchmark the
primitive-library inducer can solve exactly (train-fit + test-match), with a
memory-replay baseline for the P6 档3 comparison.

Honest report: coverage is the library's expressive limit on real ARC tasks,
NOT a score-chasing number.  Low coverage on heterogeneous ARC-1 is expected;
the controlled family protocol (run_protocol_p6d3.py) is where the P6 档3
acceptance criterion is tested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.arc import induction
from experiments.arc.grid import equal


def memory_replay_baseline(task: dict) -> bool:
    """Memorize train input->output pairs; succeed only if the test input is
    an exact copy of a train input (a trivial non-generalizing baseline)."""
    test_in = task["test"][0]["input"]
    test_out = task["test"][0]["output"]
    for demo in task["train"]:
        if demo["input"] == test_in and demo["output"] == test_out:
            return True
    return False


def scan(path: Path, max_depth: int = 2) -> dict:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    n = len(tasks)
    train_fit = 0
    test_correct = 0
    replay = 0
    depth1_ok = 0
    depth2_ok = 0
    by_name: dict[str, int] = {}
    unsolved = []
    for i, (tid, task) in enumerate(tasks.items()):
        if i % 100 == 0:
            print(f"  ...{i}/{n} (train_fit={train_fit})", flush=True)
        demos = [(g["input"], g["output"]) for g in task["train"]]
        prog = induction.search_program(demos, max_depth=max_depth)
        if prog is None:
            unsolved.append(tid)
            continue
        name, fn = prog
        train_fit += 1
        by_name[name] = by_name.get(name, 0) + 1
        if name.startswith("("):
            depth2_ok += 1
        else:
            depth1_ok += 1
        test_in = task["test"][0]["input"]
        test_out = task["test"][0]["output"]
        if equal(fn(test_in), test_out):
            test_correct += 1
        if memory_replay_baseline(task):
            replay += 1
    return {
        "split": path.stem,
        "n_tasks": n,
        "train_fit": train_fit,
        "train_fit_rate": round(train_fit / n, 3) if n else 0.0,
        "test_correct": test_correct,
        "test_correct_rate": round(test_correct / n, 3) if n else 0.0,
        "memory_replay_baseline": round(replay / n, 3) if n else 0.0,
        "depth1": depth1_ok,
        "depth2": depth2_ok,
        "top_programs": dict(sorted(by_name.items(), key=lambda kv: -kv[1])[:15]),
        "n_unsolved": len(unsolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "arc")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "arc1-coverage.json")
    parser.add_argument("--files", nargs="*", default=["arc1_training.json", "arc1_evaluation.json"])
    args = parser.parse_args()
    results = {}
    for name in args.files:
        path = args.data_dir / name
        print(f"scanning {path}")
        results[name] = scan(path, max_depth=args.max_depth)
        print(f"  -> {results[name]['split']}: train_fit "
              f"{results[name]['train_fit']}/{results[name]['n_tasks']} "
              f"test_correct {results[name]['test_correct']}/{results[name]['n_tasks']} "
              f"replay {results[name]['memory_replay_baseline']}")
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.artifact}")


if __name__ == "__main__":
    main()
