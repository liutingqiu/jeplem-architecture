"""Phase-31 second-order diagnosis: does the JEPLM state representation itself
distinguish flipped (noisy) clues from clean ones?

Ideal-predictor probe: with the identity predictor (predict(x)=x), conf =
1 - sigmoid(||state_t - state_{t+1}||^2).  If the state representation reacts
differently to flipped vs clean clues, an ideal predictor separates them and
the phase-31 neutral result is a training issue (predictor under-trained at
pred_coef=0.1).  If the state does NOT separate them, the representation
smooths away clue reliability and the mechanism is dead for this task.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from train_decide_stream_rl import DecideStreamEnv, PolicyJEPLM, _train_ppo
from plwm.jeplem_predictor import JEPLMPredictor
from plwm.device import resolve_device


def run(seed: int = 444, updates: int = 100) -> dict:
    device = resolve_device(None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = DecideStreamEnv(units=3, gap=4, identity_count=2, confounded=True,
                          clue_frames=4, clue_reliability=0.65, seed=0)
    model = PolicyJEPLM(env.input_dim, env.identity_count, slots=8,
                        aux_coefficient=0.5, predict=True, conf_feature=True).to(device)
    _train_ppo(model, env, updates=updates, episodes_per_update=24,
               seed_start=10_000, device=device)
    model.eval()

    # ideal-predictor conf on the TRAINED features representation (state +
    # recalled: recalled carries slot contents, which is what changes when a
    # clue flips -- phase-31 probe showed state-only conf cannot separate)
    ideal_flip: list[float] = []
    ideal_clean: list[float] = []
    # trained-predictor conf, same frames (reference)
    real_flip: list[float] = []
    real_clean: list[float] = []
    with torch.no_grad():
        for ep in range(256):
            obs, _ = env.reset(seed=1_000_000 + ep)
            model.reset(1, device)
            prev_features: torch.Tensor | None = None
            for frame in range(env.total_steps):
                unit, position = divmod(frame, env.unit_len)
                o = torch.tensor(obs).unsqueeze(0).to(device)
                v = torch.ones(1, dtype=torch.bool, device=device)
                model.step(o, v)
                features = model._raw_features
                if prev_features is not None:
                    err_ideal = (features - prev_features).pow(2).mean(-1)
                    conf_ideal = JEPLMPredictor.confidence(err_ideal)[0].item()
                    if position < env.clue_frames:
                        flipped = env._clue_flips[unit][position]
                        (ideal_flip if flipped else ideal_clean).append(conf_ideal)
                        real_conf = float(model._last_conf[0, 0].item())
                        (real_flip if flipped else real_clean).append(real_conf)
                prev_features = features
                obs, *_ = env.step(0)
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else float("nan")
    return {
        "ideal_flipped_conf": mean(ideal_flip),
        "ideal_clean_conf": mean(ideal_clean),
        "ideal_flip_n": len(ideal_flip),
        "ideal_clean_n": len(ideal_clean),
        "real_flipped_conf": mean(real_flip),
        "real_clean_conf": mean(real_clean),
        "real_flip_n": len(real_flip),
        "real_clean_n": len(real_clean),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
