"""Execute the current PLWM feasibility suite and persist its evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.train_associative_memory import run as run_associative_memory
from experiments.train_adaptive_compute import run as run_adaptive_compute
from experiments.train_multiscale_state import run as run_multiscale_state
from experiments.belief_revision import run as run_belief_revision
from experiments.commitment_recovery import run as run_commitment_recovery
from experiments.integrated_runtime import run as run_integrated_runtime
from experiments.train_causal_world import run as run_causal_world
from experiments.train_continuity import run as run_continuity
from experiments.train_multistep_world import run as run_multistep_world
from experiments.continual_benchmark import run as run_continual_benchmark
from experiments.lifelong_benchmark import run as run_lifelong_benchmark
from experiments.language_runtime import run as run_language_runtime
from experiments.world_runtime import run as run_world_runtime


def main() -> None:
    continuity = []
    for distractors in (1, 5, 12):
        for seed in (5, 17, 41):
            continuity.append(
                {
                    "distractors": distractors,
                    "seed": seed,
                    **asdict(run_continuity(steps=220, distractors=distractors, seed=seed)),
                }
            )
    associative = asdict(run_associative_memory())
    adaptive = asdict(run_adaptive_compute())
    multiscale = asdict(run_multiscale_state())
    belief = asdict(run_belief_revision())
    commitments = asdict(run_commitment_recovery())
    runtime = asdict(run_integrated_runtime())
    causal = asdict(run_causal_world())
    multistep = asdict(run_multistep_world())
    continual = asdict(run_continual_benchmark())
    lifelong = asdict(run_lifelong_benchmark())
    language_runtime = asdict(run_language_runtime())
    world_runtime = asdict(run_world_runtime())
    accepted = {
        "continuity": all(
            row["plwm_accuracy"] >= 0.9
            and row["baseline_accuracy"] <= 0.65
            and row["checkpoint_exact"]
            for row in continuity
        ),
        "episodic_memory": (
            associative["memory_accuracy"] >= 0.85
            and associative["memory_accuracy"] >= associative["no_memory_accuracy"] + 0.15
            and associative["gated_memory_accuracy"] >= 0.85
            and associative["gated_memory_accuracy"] >= associative["no_memory_accuracy"] + 0.15
        ),
        "causal_planning": (
            causal["persistent_transition_accuracy"] >= 0.95
            and causal["persistent_planning_accuracy"] >= 0.95
            and causal["stateless_planning_accuracy"] <= 0.65
        ),
        "multistep_rollout": (
            multistep["persistent_transition_accuracy"] >= 0.98
            and multistep["persistent_rollout_success"] >= 0.98
            and multistep["stateless_rollout_success"] <= 0.65
        ),
        "adaptive_compute": (
            adaptive["accuracy"] >= 0.98
            and adaptive["learned_mean_compute"] < adaptive["fixed_max_compute"] - 0.5
            and all(
                right > left
                for left, right in zip(
                    adaptive["expected_steps_by_difficulty"],
                    adaptive["expected_steps_by_difficulty"][1:],
                )
            )
        ),
        "multiscale_state": min(
            multiscale["fast_accuracy"],
            multiscale["slow_accuracy"],
            multiscale["slow_retention_after_contradiction"],
        ) >= 0.9,
        "belief_revision": (
            belief["weak_evidence_confidence"] < 0.6
            and belief["strong_evidence_confidence"] > 0.98
            and belief["final_revised_state"] == 0
            and belief["entropy_declines_with_confirming_evidence"]
        ),
        "commitment_recovery": (
            commitments["selected_initially"] == "ship"
            and commitments["selected_after_interruption"] == "ship"
            and commitments["completed_status"] == "completed"
            and commitments["cancelled_status"] == "cancelled"
        ),
        "integrated_runtime": (
            runtime["action_under_uncertainty"] == "think"
            and runtime["selected_before_completion"] == "reach-state-one"
            and runtime["completed_after_evidence"] == 1
            and runtime["snapshot_has_all_layers"]
        ),
        "continual_partial_observability": (
            continual["plwm_success"] >= 0.9
            and 0.2 <= continual["plwm_observation_rate"] <= 0.8
            and continual["no_memory_success"] <= 0.65
            and continual["windowed_context_success"] <= 0.65
            and continual["commitment_survives_delay"]
        ),
        "end_to_end_persistent_control": (
            continual["end_to_end_plwm_success"] >= 0.9
            and continual["end_to_end_commitment_recovery"] >= 0.9
        ),
        "stale_memory_and_language_boundary": (
            continual["stale_memory_correction"] >= 0.85
            and continual["long_delay_recovery"] >= 0.9
            and continual["language_port_is_stateless"]
        ),
        "lifelong_memory_consolidation": (
            lifelong["stable_retention"] >= 0.99
            and lifelong["changed_fact_revision"]
            and lifelong["stale_burst_resistance"]
            and lifelong["provenance_complete"]
            and lifelong["consolidation_count"] >= 1
        ),
        "fair_transformer_baseline": continual["transformer_context_success"] >= 0.9,
        "language_runtime_boundary": (
            language_runtime["checkpoint_continuity"]
            and language_runtime["runtime_ticks"] == 2
            and language_runtime["strict_parser_rejects_invalid"]
            and language_runtime["emits_without_mutating_runtime"]
        ),
        "world_model_persistent_runtime": (
            world_runtime["runtime_planning_accuracy"] >= 0.95
            and world_runtime["checkpoint_action_exact"]
            and world_runtime["completion_requires_evidence"]
            and world_runtime["commitment_switches_after_evidence"]
        ),
    }
    report = {
        "continuity": continuity,
        "associative_memory": associative,
        "adaptive_compute": adaptive,
        "multiscale_state": multiscale,
        "belief_revision": belief,
        "commitment_recovery": commitments,
        "integrated_runtime": runtime,
        "causal_planning": causal,
        "multistep_rollout": multistep,
        "continual_partial_observability": continual,
        "lifelong_memory": lifelong,
        "language_runtime": language_runtime,
        "world_model_runtime": world_runtime,
        "accepted": accepted,
    }
    output = Path(__file__).resolve().parents[1] / "artifacts" / "feasibility-report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(accepted.values()):
        raise SystemExit("PLWM feasibility suite failed")


if __name__ == "__main__":
    main()
