# Persistent Latent World Model (PLWM / JEPLM Architecture)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![Tests: 240 Passed](https://img.shields.io/badge/Tests-240%20Passed-success.svg)]()

Research prototype exploring **Persistent Latent World Models (PLWM)** — cognitive world-model architectures that maintain recurrent latent states across time, update beliefs via prediction error, and perform internal reasoning without relying on continuous text token emission.

---

## 1. Project Objectives

Modern LLM-based autonomous agents typically operate as stateless completion functions: past context must be re-ingested as a linear token history, thoughts must be externalized into text, and state persistence relies on external databases or prompt re-stuffing.

The **PLWM / JEPLM Architecture** project explores a fundamental alternative:
1. **Persistent Latent State**: Maintaining an explicit, structured cognitive state (belief, goal, workspace, and episodic memory slots) that survives session boundaries and checkpoints without replaying full dialogue or trajectory histories.
2. **Prediction Error as a Driver**: Driving state belief updates and episodic memory writes through predictive discrepancy rather than passive context appending.
3. **Latent Cognitive Operations**: Enabling recurrent, multi-step thought transitions entirely in latent space without text decoding.
4. **Explicit Cognitive Action Space**: Supporting structured internal actions (`observe`, `think`, `retrieve`, `commit`, `speak`, `sleep`) with formal state transition boundaries.

---

## 2. Currently Implemented Capabilities

The repository contains both a dependency-free pure-Python reference specification and an end-to-end trainable PyTorch runtime (`plwm/`):

- **Structured Cognitive State (`plwm/state.py`, `plwm/torch_model.py`)**:
  - `latent_state`: Continuous vector representation across persistent memory slots.
  - `belief_state`: Categorical probability distributions and entropy measures over environmental states.
  - `commitment_state`: Ledger-managed, verified obligations with audit trails.
  - `control_state`: Learnable adaptive compute halting distributions.
- **Differentiable Transition Contract (`plwm/torch_model.py`)**:
  - Observation prediction: `predicted_obs = P(belief)`
  - Error-driven update: `prediction_error = distance(predicted_obs, actual_obs)`
  - Belief blending: `belief_next = blend(belief, Core(belief, goal, encoded_obs))`
  - Cognitive action policy: `action = Policy(belief, goal, workspace, prediction_error)`
- **Memory & Evidence Systems (`plwm/lifelong.py`, `plwm/fact_ledger.py`)**:
  - Content-addressed episodic recall with decay and reinforcement.
  - Versioned fact memory preventing unverified low-confidence inputs from overwriting verified commitments.
  - Latent planning (`plwm/latent_planner.py`) supporting action-conditioned counterfactual rollouts.
- **Benchmark & Environment Bridges (`experiments/`, `minecraft-mod/`, `luanti-game/`)**:
  - Minigrid partially observable delayed cue tasks.
  - Crafter vision-only pixel environment bindings.
  - Luanti / Minetest local bridge and Minecraft Fabric mod (`minecraft-mod/`) for real-time state extraction.

---

## 3. Research Prototype vs. Production System

> **Important Positioning Notice**:
> This repository is an **experimental research prototype and architectural testbed**, NOT a commercial or production-ready agent product.

| Dimension | PLWM Research Prototype | Production Agent Platform |
| :--- | :--- | :--- |
| **Primary Goal** | Falsifying architectural hypotheses on latent world models | Scalable user-facing workflow execution |
| **Model Scale** | Lightweight exploratory dimensions (d=16 to d=128) | Multi-billion parameter frontier LLMs |
| **State Storage** | Local JSON / PyTorch tensor checkpoints | Distributed databases, Vector DBs, KV caches |
| **Maturity** | Validated contracts across 240+ deterministic unit tests | Beta/GA production availability |

---

## 4. Installation & Requirements

### Prerequisites
- Python 3.10+ (tested through Python 3.13)
- PyTorch 2.0+ (CPU is sufficient for test suite; CUDA auto-detected for training sweeps)

### Setup
```bash
git clone https://github.com/liutingqiu/jeplem-architecture.git
cd jeplem-architecture

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 5. Running Demos & Tests

### Running the Test Suite
The full suite covers 240+ unit and integration tests across state transitions, backward-compatible checkpoints, counterfactual rollouts, and sensory bindings:

```bash
# Set PYTHONPATH to project root
export PYTHONPATH=.       # Linux / macOS
$env:PYTHONPATH = "."     # Windows PowerShell

pytest tests/ -v
```

### Running the Interruption & Recovery Demo
Demonstrates checkpointing persistent cognitive state to disk, interrupting the process, and recovering exact state without prior history replay:

```bash
python demo.py
```

### Running Verification Script (Windows)
```powershell
.\verify.ps1
```

---

## 6. Known Limitations & Roadmap

1. **High-Dimensional Visual End-to-End Bottleneck**: While pixel-level Crafter and Luanti bindings function correctly, end-to-end joint training of high-resolution visual encoders with deep latent planning remains computationally expensive.
2. **Open-World Representation Ceiling**: On real-world structured knowledge extraction (Wikidata/ConceptNet), outward expansion is bound by external relation density; the system cannot hallucinate facts beyond available observations.
3. **Language Interface Boundary**: The language module (`plwm/language.py`) acts strictly as a perceptual and action-emission port. It deliberately does not allow untrusted natural language text to directly mutate internal state tensors without passing through the belief ledger.

---

## 7. License & Citation

This project is licensed under the Apache License 2.0. Detailed phase logs, mathematical proofs, and historical experimental surveys are archived under `docs/reports/`.
