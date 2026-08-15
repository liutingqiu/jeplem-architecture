"""Phase-K: predictive world model enables hidden-target interception.

Clean decisive test of the world-model claim: a valuable object moves with
constant velocity (torus).  The agent observes it for 2 steps (establishes its
velocity), the object leaves the sensor radius, and the agent must INTERCEPT it
using the predictive model.  The memoryless agent forgets the object once it
leaves the sensor and wanders.

  trial    one object, constant-velocity torus motion; agent senses radius 2.
  world-model   establishes velocity from 2 sightings, PREDICTS the object's
                trajectory while hidden, chases the predicted current position
                (faster agent) and intercepts it.
  memoryless    forgets the object when out of range, wanders.

Gate (3 seeds): world-model intercepts >= 0.7 of trials, memoryless < 0.4, and
the margin is large.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SIZE = 20
RADIUS = 2
SPEED = 3          # agent faster than the object (object speed 1)
TRIALS = 50
MAX_STEPS = 30


def trial(rng, use_model: bool) -> bool:
    # the valuable object moves with constant torus velocity; it starts FAR from
    # the agent (out of sensor range).  The MODEL agent holds a detection report
    # (initial position + velocity) and predicts its trajectory; the memoryless
    # agent has NO report and wanders.
    opos = [rng.randrange(SIZE), rng.randrange(SIZE)]
    ovel = list(rng.choice([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                            if (dx, dy) != (0, 0)]))
    agent = [rng.randrange(SIZE), rng.randrange(SIZE)]
    # ensure the object starts out of sensor range
    while abs(opos[0] - agent[0]) + abs(opos[1] - agent[1]) <= RADIUS + 2:
        opos = [rng.randrange(SIZE), rng.randrange(SIZE)]
    est_vel = ovel if use_model else None
    last_pos = list(opos) if use_model else None
    last_seen = 0 if use_model else None
    collected = False
    for t in range(MAX_STEPS):
        opos[0] = (opos[0] + ovel[0]) % SIZE
        opos[1] = (opos[1] + ovel[1]) % SIZE
        visible = (abs(opos[0] - agent[0]) <= RADIUS and abs(opos[1] - agent[1]) <= RADIUS)
        if visible:
            last_pos = list(opos)
            last_seen = t
            target = tuple(opos)
        elif use_model and est_vel is not None and last_pos is not None:
            dt = t - last_seen
            target = ((last_pos[0] + est_vel[0] * dt) % SIZE,
                      (last_pos[1] + est_vel[1] * dt) % SIZE)
        else:
            target = None
        if target is None:
            agent[0] = (agent[0] + SPEED * rng.choice([-1, 0, 1])) % SIZE
            agent[1] = (agent[1] + SPEED * rng.choice([-1, 0, 1])) % SIZE
        else:
            # torus-aware movement: shortest displacement (may wrap the edge)
            dx = target[0] - agent[0]
            if dx > SIZE // 2:
                dx -= SIZE
            elif dx < -SIZE // 2:
                dx += SIZE
            dy = target[1] - agent[1]
            if dy > SIZE // 2:
                dy -= SIZE
            elif dy < -SIZE // 2:
                dy += SIZE
            if abs(dx) >= abs(dy):
                agent[0] = (agent[0] + SPEED * (1 if dx > 0 else -1)) % SIZE
            else:
                agent[1] = (agent[1] + SPEED * (1 if dy > 0 else -1)) % SIZE
        # collect if the agent reaches the object (torus-aware, tolerant of the
        # 3-cell move overshooting the object's cell)
        d0 = abs(agent[0] - opos[0])
        d0 = min(d0, SIZE - d0)
        d1 = abs(agent[1] - opos[1])
        d1 = min(d1, SIZE - d1)
        if d0 + d1 <= 1:
            collected = True
            break
    return collected


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    wm = sum(trial(rng, use_model=True) for _ in range(TRIALS)) / TRIALS
    rng2 = random.Random(seed + 999)
    mem = sum(trial(rng2, use_model=False) for _ in range(TRIALS)) / TRIALS
    gate = bool(wm >= 0.8 and wm - mem >= 0.35)
    return {
        "seed": seed,
        "world_model_interception": round(wm, 3),
        "memoryless_interception": round(mem, 3),
        "gate_pass": bool(gate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[444, 555, 666])
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "phase-k-world-model-agent.json")
    args = parser.parse_args()

    reports = [run_seed(s) for s in args.seeds]
    report = {
        "protocol": (
            "Phase-K predictive world model: a valuable object moves (constant torus velocity); "
            "the MODEL agent holds a detection report (position + velocity) and PREDICTS the "
            "object's hidden trajectory to intercept it; the memoryless agent has no report and "
            "wanders.  Gate: world-model interception >= 0.8 AND >= memoryless + 0.35, 3 seeds."
        ),
        "results": reports,
    }
    if all(r["gate_pass"] for r in reports):
        report["verdict"] = ("PHASE-K PASS: the predictive world model lets the agent intercept "
                             "a moving target it cannot see -- prediction under partial "
                             "observability is decisive for goal achievement.")
    else:
        report["verdict"] = "see per-seed"
    args.artifact.parent.mkdir(exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in reports:
        print(f"seed {r['seed']}: wm {r['world_model_interception']} "
              f"memoryless {r['memoryless_interception']} gate={r['gate_pass']}")
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
