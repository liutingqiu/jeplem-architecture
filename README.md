# PLWM research prototype

This repository is a dependency-free executable sketch of the Persistent
Latent World Model discussed in the research conversation.

It tests the architectural contract before choosing a large neural backend:

- a persistent belief, goal, workspace, and episodic memory state;
- prediction error as a driver of state updates and memory writes;
- recurrent latent thought steps that do not emit text;
- explicit cognitive actions: observe, think, retrieve, commit, speak, sleep;
- JSON checkpointing and recovery without replaying the prior conversation.

The matrices are deterministic seeded weights. This is intentionally not a
claim of a trained language model. The next research step is to replace the
small pure-Python operators with trainable PyTorch modules while preserving the
state/action/checkpoint contract and adding a continual partially observable
task benchmark.

That trainable reference is now in `plwm/torch_model.py`. It has a persistent
belief state, FIFO episodic memory with content-addressed reads, latent-only
recurrent thought on query events, and a disk-checkpointable runtime state.

## Model contract

The transition is deliberately split into prediction, correction, and optional
cognitive work:

```text
predicted_observation = P(belief)
prediction_error      = distance(predicted_observation, observation)
belief_next           = blend(belief, Core(belief, goal, encoded_observation))
action                = Policy(belief, goal, workspace, prediction_error)
```

`think` iterates the workspace without decoding text. `retrieve` can reactivate
episodic slots. `sleep` preserves state while avoiding an unnecessary thought
loop. The action interface is the stable boundary for a future learned policy.

Run the tests from this directory:

```text
python -m unittest discover -s tests -v
```

Run the interruption-and-recovery demo:

```text
python demo.py
```

Run the structural interruption-and-recovery benchmark:

```text
python experiments/interruption_recovery.py
```

Train and evaluate the first feasibility experiment (PyTorch CPU required):

```text
python experiments/train_continuity.py
```

The task presents an early binary cue, several distractors, and a final query
that contains no cue. The acceptance gate is persistent-model accuracy of at
least 90%, stateless-baseline accuracy no greater than 65%, and exactly equal
predictions before versus after a cognitive-state checkpoint/restart.

The suite also covers action-conditioned causal prediction and a three-step
counterfactual rollout task. Both use a hidden environmental rule revealed
before the decision event; the planner learns transition prediction only, then
selects actions by evaluating imagined outcomes.

It also validates a differentiable adaptive-compute controller: a recurrent
latent core is supervised at each depth while a learned halt distribution trades
task loss against expected compute.

Run the stronger feasibility sweep across three seeds and three distractor
lengths:

```text
python experiments/sweep_continuity.py
```

Run the complete current evidence suite; it writes
`artifacts/feasibility-report.json` on success:

```text
python experiments/run_all.py
```

On Windows, run the same verification flow with:

```text
.\verify.ps1
```

## Current Architecture

The prototype keeps four state domains intentionally separate:

```text
latent state       PLWM / recurrent workspace / episodic memory
belief state       normalized categorical probabilities and entropy
commitment state   active, completed, or cancelled auditable obligations
control state      learned halt distribution and fast/slow update gates
```

`PersistentRuntime` composes these domains for one event loop. A latent vector
does not directly complete a commitment: only the explicit belief layer can do
that, and completion leaves evidence in the commitment ledger. This separation
is a design invariant, not a UI convention.

The evidence suite currently checks continuity after checkpointing, associative
recall, adaptive compute, multi-timescale retention, belief revision, causal
planning, counterfactual rollout, commitment recovery, and unified-runtime
composition.

## Five-Stage Expansion

The current expansion route is fixed as five falsifiable gates:

1. A partially observable task with noisy evidence, active observation cost,
   delayed commitments, and distractors.
2. Same-budget comparisons against finite-context, recurrent, and no-memory
   controllers.
3. End-to-end trainable persistent state and control connected to the PLWM
   world-model path.
4. Stale-memory correction, interruption recovery, and long-run drift tests.
5. A language adapter limited to perception and action emission; it cannot
   complete commitments or mutate cognitive state by text alone.

Run the expanded benchmark directly with:

```text
python experiments/continual_benchmark.py
```

The benchmark's recurrent baseline is intentionally retained even when it
matches or exceeds PLWM on a task. That result is evidence about the task, not
silently discarded.

The lifelong memory benchmark adds versioned provenance and sleep-time replay:

```text
python experiments/lifelong_benchmark.py
```

It measures whether a verified change to one fact survives a stale low-confidence
burst while unrelated facts remain available. Unverified evidence is not allowed
to overwrite a verified commitment merely because it arrived later.

`plwm/language.py` supplies an `OllamaBackend` and strict `JsonLanguagePort`.
The backend is optional: no model weights, endpoint credentials, belief, or
commitment state are stored in the adapter. Its recovery contract is tested by:

```text
python experiments/language_runtime.py
```

When a local Ollama server and model are available, run a real backend call
through the same checkpoint/recovery contract (it is intentionally not part of
the offline suite):

```text
python experiments/ollama_integration.py --required
```

The trainable runtime also exposes `think_only()`: a latent-only workspace
transition that has no observation input and cannot write evidence, memory, or
commitment state. `WorldModelPersistentRuntime` combines action-conditioned
counterfactual planning, explicit action costs, checkpoint recovery, and
ledger-selected multi-goal sequencing. Its integrated experiment is:

```text
python experiments/world_runtime.py
```
