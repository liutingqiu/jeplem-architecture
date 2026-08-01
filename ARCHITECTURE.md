# PLWM Research Specification

## Thesis

PLWM treats language as one optional action of a persistent cognitive system.
The primary computational object is a durable state transition, not a chat
turn. The implementation is a research prototype, not a claim of sentience or
a replacement for a pretrained language model.

## State Domains

| Domain | Representation | Authority |
| --- | --- | --- |
| Latent | recurrent belief, workspace, episodic slots | neural prediction and local deliberation |
| Belief | normalized categorical log probabilities | evidence, uncertainty, and revision |
| Commitment | auditable active/completed/cancelled records | goals and completion conditions |
| Control | halt distribution and fast/slow gates | allocation of internal computation |

No domain can silently impersonate another: latent similarity cannot complete a
commitment, and a commitment cannot override evidence without a state update.

## Verified Claims

- Persistent state survives in-memory and disk checkpoint/restart.
- Latent-only recursive thought can be separated from visible emission.
- Content-addressed episodic memory improves associative recall over a
  no-retrieval ablation.
- Action-conditioned dynamics support one-step and three-step
  counterfactual planning under a hidden environmental rule.
- Adaptive computation allocates more latent depth to deeper transformations.
- Fast/slow state separation resists short contradictory bursts.
- Explicit probabilistic beliefs accumulate and reverse under contrary
  evidence.

## Non-Claims

The experiments use small synthetic environments. They do not establish
open-world generalization, subjective experience, reliable autonomous goal
formation, or long-horizon safety. Each such claim requires a separate
environment, metric, baseline, and failure analysis.

## Research Roadmap

The next stages extend the prototype through separately falsifiable gates:

1. Evaluate under partial observability, noisy evidence, delayed commitments,
   action costs, and active observation.
2. Compare against long-context Transformer, recurrent, and no-memory
   baselines under the same task distribution and compute budget.
3. Connect the trainable world model to the persistent runtime so state and
   control are learned end-to-end rather than only structurally composed.
4. Test stale-memory correction, interruption recovery, and long-run state
   drift without losing unrelated information.
5. Introduce a language model only as a perception/action interface, keeping
   the persistent cognitive runtime as the primary computational object.

The current benchmark also includes a 100-event interruption probe, explicit
stale-memory correction, and a language-port boundary test. These are
intermediate gates, not evidence of open-world competence.

The lifelong memory gate additionally requires versioned provenance, verified
non-stationary updates, stale-burst resistance, and sleep-time consolidation of
repeated consistent evidence.

Language adapters must produce declared, schema-validated evidence. A backend
may generate or parse text, but it cannot mutate runtime state directly; the
runtime checkpoint remains the only source of continuity across backend calls.

`TrainablePersistentRuntime` now makes the trainable latent state a runtime
layer instead of a one-shot experiment artifact. Its `think_only` transition
updates workspace without any observation, text emission, belief update, memory
write, or commitment completion. `WorldModelPersistentRuntime` separately joins
action-conditioned counterfactual dynamics with the same evidence-authoritative
commitment ledger; costed planning uses normalized goal log-probabilities, not
raw neural logits.

Completion of these stages would establish a stronger empirical candidate for
continuous cognition. It would not by itself establish human equivalence or
subjective experience.
