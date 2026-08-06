"""Build a machine-readable boundary between supported and rejected PLWM claims.

This registry consumes completed artifacts only.  It deliberately records
counterexamples alongside bounded successes so downstream reports cannot turn a
teacher-scoped memory interface result into an end-to-end architecture claim.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Evidence:
    identifier: str
    status: str
    scope: str
    artifact: str
    summary: str


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _result(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for item in payload.get("results", []):
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise ValueError(f"missing result {name!r}")


def build_registry(artifact_dir: Path) -> list[Evidence]:
    navigation_path = artifact_dir / "minigrid-empty-navigation-30x24-report.json"
    navigation = _load(navigation_path)
    plwm = _result(navigation, "plwm")
    transformer = _result(navigation, "transformer")
    plwm_successes = int(plwm["evaluation"]["successes"])
    transformer_successes = int(transformer["evaluation"]["successes"])
    evidence = [Evidence(
        "reward_only_public_navigation",
        "rejected-current-controller" if plwm_successes == 0 and transformer_successes == 64 else "requires-review",
        "same-budget MiniGrid Empty-5x5 reward-only control",
        str(navigation_path),
        f"PLWM greedy {plwm_successes}/64; Transformer greedy {transformer_successes}/64.",
    )]

    for identifier, filename, plwm_name in (
        ("current_observation_skip", "minigrid-plwm-observation-skip-report.json", "plwm_observation_skip"),
        ("causal_action_event", "minigrid-plwm-causal-event-sequence-report.json", "plwm_separate_action_event"),
        ("concurrent_predictive_auxiliary", "minigrid-plwm-predictive-auxiliary-report.json", "plwm_predictive_auxiliary"),
        ("predictive_pretraining", "minigrid-plwm-predictive-pretraining-report.json", "plwm_predictive_pretrained"),
    ):
        path = artifact_dir / filename
        payload = _load(path)
        result = _result(payload, plwm_name)
        successes = int(result["evaluation"]["successes"])
        evidence.append(Evidence(
            identifier,
            "rejected-current-controller" if successes == 0 else "requires-review",
            "switchable MiniGrid control mechanism audit",
            str(path),
            f"Held-out greedy success {successes}/{result['evaluation']['episodes']}.",
        ))

    planner_path = artifact_dir / "minigrid-model-based-planning-report.json"
    planner = _load(planner_path)
    planner_successes = int(planner["planner"]["successes"])
    evidence.append(Evidence(
        "action_conditioned_model_based_planning",
        "rejected-current-controller" if planner_successes == 0 else "requires-review",
        "random-transition-trained latent world model with self-rolled counterfactual planning",
        str(planner_path),
        f"Held-out planning success {planner_successes}/{planner['planner']['episodes']}; one-step loss {planner['final_loss']:.4f}; six-step reward Brier {planner['heldout_six_step_reward_brier']:.4f}.",
    ))

    multistep_path = artifact_dir / "minigrid-model-based-multistep-report.json"
    multistep = _load(multistep_path)
    multistep_successes = int(multistep["planner"]["successes"])
    evidence.append(Evidence(
        "multistep_action_conditioned_model_based_planning",
        "rejected-current-controller" if multistep_successes == 0 else "requires-review",
        "six-step self-rolled world-model training and six-step counterfactual planning",
        str(multistep_path),
        "Held-out planning success "
        f"{multistep_successes}/{multistep['planner']['episodes']}; "
        f"six-step reward Brier {multistep['heldout_six_step_reward_brier']:.4f}; "
        f"training loss {multistep['initial_loss']:.4f}->{multistep['final_loss']:.4f}.",
    ))

    hybrid_path = artifact_dir / "minigrid-hybrid-curriculum-report.json"
    hybrid = _load(hybrid_path)
    hybrid_result = next(item for item in hybrid["results"] if item["name"] == "hybrid_plwm_transformer")
    hybrid_successes = int(hybrid_result["evaluation"]["successes"])
    replication_path = artifact_dir / "minigrid-hybrid-curriculum-replication-report.json"
    if replication_path.exists():
        replication = _load(replication_path)
        replication_result = next(item for item in replication["results"] if item["name"] == "hybrid_plwm_transformer")
        replication_successes = int(replication_result["evaluation"]["successes"])
        status = (
            "requires-complexity-scaling"
            if hybrid.get("accepted_hybrid_controller") and replication.get("accepted_hybrid_controller")
            else "rejected-current-controller"
        )
        if status == "requires-complexity-scaling":
            summary = (
                f"Independent held-out blocks: {hybrid_successes}/{hybrid_result['evaluation']['episodes']} and "
                f"{replication_successes}/{replication_result['evaluation']['episodes']}; "
                "replication clears only the prerequisite for delay/rule/object/noise scaling."
            )
        else:
            summary = (
                f"Independent held-out blocks: {hybrid_successes}/{hybrid_result['evaluation']['episodes']} and "
                f"{replication_successes}/{replication_result['evaluation']['episodes']}; "
                "the replicated block did not meet the PLWM-side contribution gate."
            )
        artifact = f"{hybrid_path}; {replication_path}"
    else:
        status = "requires-independent-replication" if hybrid.get("accepted_hybrid_controller") else "rejected-current-controller"
        summary = (
            f"First held-out block hybrid success {hybrid_successes}/{hybrid_result['evaluation']['episodes']}; "
            "acceptance is not architecture evidence until independently seeded replication succeeds."
        )
        artifact = str(hybrid_path)
    evidence.append(Evidence(
        "plwm_transformer_hybrid_curriculum",
        status,
        "reward-only two-stage navigation-to-delayed-memory curriculum with capacity-matched Transformer and switchable PLWM side state",
        artifact,
        summary,
    ))

    fact_path = artifact_dir / "luanti-two-fact-composition-report.json"
    fact = _load(fact_path)
    learned = int(fact["fact_memory_successes"])
    no_memory = int(fact["no_fact_successes"])
    evidence.append(Evidence(
        "verified_delayed_fact_interface",
        "supported-bounded-interface" if fact.get("accepted") and fact.get("all_fact_provenance_complete") and learned > no_memory else "requires-review",
        "source-bound, checkpoint-restored delayed facts in Luanti; not end-to-end control",
        str(fact_path),
        f"Fact memory {learned}/8; no-fact {no_memory}/8; provenance={bool(fact.get('all_fact_provenance_complete'))}.",
    ))

    polarity_path = artifact_dir / "luanti-polarity-delay-32-report.json"
    polarity_replication_path = artifact_dir / "luanti-polarity-delay-32-replication-report.json"
    if polarity_path.exists() and polarity_replication_path.exists():
        polarity = _load(polarity_path)
        polarity_replication = _load(polarity_replication_path)
        first_point = polarity["scale_points"][0]
        replication_point = polarity_replication["scale_points"][0]
        first_seeds = set(polarity["train_seeds"] + polarity["heldout_seeds"])
        replication_seeds = set(polarity_replication["train_seeds"] + polarity_replication["heldout_seeds"])
        independent = not first_seeds.intersection(replication_seeds)
        status = (
            "supported-bounded-interface"
            if polarity.get("accepted") and polarity_replication.get("accepted") and independent
            else "requires-review"
        )
        evidence.append(Evidence(
            "replicated_luanti_polarity_delay_fact_interface",
            status,
            "two independent real-Luanti two-object delayed-reward fact-memory blocks; teacher-labelled policy, not end-to-end control",
            f"{polarity_path}; {polarity_replication_path}",
            "32-hidden-observation fact-memory / no-memory / public-event GRU / 16-event Transformer "
            f"successes: {first_point['fact_memory']['successes']}/{first_point['fact_memory']['episodes']}, "
            f"{first_point['no_memory']['successes']}/{first_point['no_memory']['episodes']}, "
            f"{first_point['recurrent']['successes']}/{first_point['recurrent']['episodes']}, "
            f"{first_point['transformer']['successes']}/{first_point['transformer']['episodes']}; replication "
            f"{replication_point['fact_memory']['successes']}/{replication_point['fact_memory']['episodes']}, "
            f"{replication_point['no_memory']['successes']}/{replication_point['no_memory']['episodes']}, "
            f"{replication_point['recurrent']['successes']}/{replication_point['recurrent']['episodes']}, "
            f"{replication_point['transformer']['successes']}/{replication_point['transformer']['episodes']}; "
            f"disjoint-seeds={independent}.",
        ))

    ledger_path = artifact_dir / "versioned-fact-ledger-report.json"
    ledger = _load(ledger_path)
    evidence.append(Evidence(
        "versioned_source_bound_fact_ledger",
        "supported-bounded-interface" if ledger.get("accepted") else "requires-review",
        "source/time/confidence/version fact revision with weak-conflict isolation; not end-to-end control",
        str(ledger_path),
        "Strong revision, weak/stale conflict isolation, unrelated fact retention, and snapshot restoration="
        f"{bool(ledger.get('accepted'))}.",
    ))

    ledger_matrix_path = artifact_dir / "fact-ledger-distractor-matrix-report.json"
    ledger_matrix = _load(ledger_matrix_path)
    evidence.append(Evidence(
        "versioned_fact_ledger_long_distractor_matrix",
        "supported-bounded-interface" if ledger_matrix.get("accepted") else "requires-review",
        "source-bound fact revision under weak/stale conflict distractors; not end-to-end control",
        str(ledger_matrix_path),
        f"Ledger {ledger_matrix['ledger_successes']}/{ledger_matrix['episodes']}; "
        f"naive last-write {ledger_matrix['naive_last_write_successes']}/{ledger_matrix['episodes']}; "
        f"restored={bool(ledger_matrix.get('all_restored'))}.",
    ))

    ledger_replay_path = artifact_dir / "luanti-reward-ledger-replay-report.json"
    ledger_replay = _load(ledger_replay_path)
    evidence.append(Evidence(
        "luanti_reward_ledger_replay",
        "supported-bounded-interface" if ledger_replay.get("accepted") else "requires-review",
        "read-only replay of local Luanti reward provenance into versioned facts; not end-to-end control",
        str(ledger_replay_path),
        f"Real reward episodes replayed={ledger_replay['selected_episodes']}; "
        f"source-bound restore complete={bool(ledger_replay.get('accepted'))}.",
    ))

    visual_path = artifact_dir / "luanti-multi-object-visual-evidence-report.json"
    visual = _load(visual_path)
    moderate = next(
        (item for item in visual.get("conditions", []) if item.get("condition", {}).get("name") == "moderate_noise_and_replay"),
        None,
    )
    if not isinstance(moderate, dict):
        raise ValueError("missing moderate visual evidence condition")
    evidence.append(Evidence(
        "visual_evidence_replacement",
        "rejected-default-robustness-path",
        "Luanti stale/replayed visual evidence replacement",
        str(visual_path),
        f"Conservative filter success {moderate['evidence_filtered']['successes']}/8 versus raw {moderate['raw']['successes']}/8.",
    ))

    crafter_path = artifact_dir / "crafter-reward-only-sentinel-report.json"
    crafter_retry_path = artifact_dir / "crafter-reward-only-sentinel-v2-smoke-report.json"
    if crafter_path.exists() and crafter_retry_path.exists():
        crafter = _load(crafter_path)
        retry = _load(crafter_retry_path)
        first_random = float(crafter["random_evaluation"]["mean_return"])
        retry_random = float(retry["random_evaluation"]["mean_return"])
        first_best = max(float(result["evaluation"]["mean_return"]) for result in crafter["results"])
        retry_best = max(float(result["evaluation"]["mean_return"]) for result in retry["results"])
        evidence.append(Evidence(
            "crafter_short_budget_ppo_baseline",
            "rejected-baseline-protocol" if not crafter.get("accepted_learnability") and not retry.get("accepted_learnability") else "requires-review",
            "pixels-only 17-action Crafter short-budget PPO sentinel; not a PLWM architecture test",
            f"{crafter_path}; {crafter_retry_path}",
            "Two independently written artifacts failed the basic greedy-return gate: "
            f"v1 best={first_best:.4f} versus random={first_random:.4f}; "
            f"streaming-inference v2 best={retry_best:.4f} versus random={retry_random:.4f}. "
            "The protocol is rejected, not the environment or PLWM architecture.",
        ))

    learned_revision_path = artifact_dir / "learned-fact-revision-report.json"
    learned_revision_replication_path = artifact_dir / "learned-fact-revision-replication-report.json"
    if learned_revision_path.exists() and learned_revision_replication_path.exists():
        first = _load(learned_revision_path)
        replication = _load(learned_revision_replication_path)
        first_by_name = {item["name"]: item for item in first["results"]}
        replication_by_name = {item["name"]: item for item in replication["results"]}
        first_margin = float(first_by_name["learned_fact_store"]["long_distractor_accuracy"]) - max(
            float(first_by_name["gru"]["long_distractor_accuracy"]),
            float(first_by_name["four_event_transformer"]["long_distractor_accuracy"]),
        )
        replication_margin = float(replication_by_name["learned_fact_store"]["long_distractor_accuracy"]) - max(
            float(replication_by_name["gru"]["long_distractor_accuracy"]),
            float(replication_by_name["four_event_transformer"]["long_distractor_accuracy"]),
        )
        status = (
            "replicated-diagnostic-signal-not-control-ready"
            if first_margin >= 0.1 and replication_margin >= 0.1
            else "requires-review"
        )
        evidence.append(Evidence(
            "learned_source_bound_fact_revision",
            status,
            "supervised long-conflict fact-memory diagnostic; not reward-only control",
            f"{learned_revision_path}; {learned_revision_replication_path}",
            "64-distractor learned-store margin over the strongest standard baseline: "
            f"first={first_margin:.4f}, replication={replication_margin:.4f}; "
            "both reports failed their absolute reliability gate and remain excluded from control.",
        ))

    sb3_paths = (
        artifact_dir / "crafter-sb3-pixels-baseline-report.json",
        artifact_dir / "crafter-sb3-pixels-baseline-replication-report.json",
        artifact_dir / "crafter-sb3-pixels-baseline-third-block-report.json",
    )
    if all(path.exists() for path in sb3_paths):
        sb3_blocks = [_load(path) for path in sb3_paths]
        deltas = [
            float(block["ppo_evaluation"]["mean_return"]) - float(block["random_evaluation"]["mean_return"])
            for block in sb3_blocks
        ]
        passed = sum(bool(block.get("accepted_standard_baseline")) for block in sb3_blocks)
        evidence.append(Evidence(
            "crafter_sb3_short_budget_stability",
            "rejected-baseline-protocol" if passed < 2 else "requires-review",
            "three independent Stable-Baselines3 PPO blocks on RGB-only Crafter; not a PLWM test",
            "; ".join(str(path) for path in sb3_paths),
            f"Fixed 16,384-step PPO deltas over random={deltas}; mean={sum(deltas) / len(deltas):.4f}; "
            f"gate passes={passed}/3. The fixed-budget baseline is not stable enough for PLWM comparison.",
        ))

    reward_only_path = artifact_dir / "luanti-reward-only-ppo-formal-report.json"
    reward_only_audit_path = artifact_dir / "luanti-reward-only-ppo-formal-audit.json"
    if reward_only_path.exists() and reward_only_audit_path.exists():
        reward_only = _load(reward_only_path)
        reward_only_audit = _load(reward_only_audit_path)
        reward_only_results = {item.get("name"): item for item in reward_only.get("results", []) if isinstance(item, dict)}
        required = ("plwm", "no_memory", "gru", "transformer_16")
        if all(name in reward_only_results for name in required):
            successes = {name: int(reward_only_results[name]["evaluation"]["successes"]) for name in required}
            audit_clean = not reward_only_audit.get("failures")
            if audit_clean and not reward_only_audit.get("accepted") and not reward_only.get("accepted"):
                status = "rejected-current-controller"
            elif audit_clean and reward_only_audit.get("accepted") and reward_only.get("accepted"):
                status = "requires-independent-replication"
            else:
                status = "requires-review"
            evidence.append(Evidence(
                "luanti_reward_only_end_to_end_ppo_first_block",
                status,
                "real-Luanti local-vision/action/reward-only PPO controller comparison with no teacher, symbolic state, or external planner",
                f"{reward_only_path}; {reward_only_audit_path}",
                "Held-out success PLWM/no-memory/GRU/16-event Transformer="
                f"{successes['plwm']}/8, {successes['no_memory']}/8, {successes['gru']}/8, {successes['transformer_16']}/8; "
                f"independent audit accepted={bool(reward_only_audit.get('accepted'))}."
            ))

    micro_q_path = artifact_dir / "luanti-micro-navigation-qlearning-report.json"
    micro_q_audit_path = artifact_dir / "luanti-micro-navigation-qlearning-audit.json"
    if micro_q_path.exists() and micro_q_audit_path.exists():
        micro_q = _load(micro_q_path)
        micro_q_audit = _load(micro_q_audit_path)
        random_successes = int(micro_q["random_evaluation"]["successes"])
        q_successes = int(micro_q["q_learning_evaluation"]["successes"])
        audit_clean = not micro_q_audit.get("failures")
        status = (
            "supported-standard-baseline-only"
            if audit_clean and micro_q_audit.get("accepted") and micro_q.get("accepted_public_feature_baseline")
            else "requires-review"
        )
        evidence.append(Evidence(
            "luanti_visible_goal_public_feature_qlearning_baseline",
            status,
            "real-Luanti visible-goal public-feature standard baseline; not raw visual representation or PLWM",
            f"{micro_q_path}; {micro_q_audit_path}",
            f"Held-out Q-learning/random success={q_successes}/16/{random_successes}/16; "
            f"independent audit accepted={bool(micro_q_audit.get('accepted'))}."
        ))

    gated_path = artifact_dir / "luanti-delayed-goal-gated-compact-binary-qlearning-report.json"
    gated_audit_path = artifact_dir / "luanti-delayed-goal-gated-compact-binary-qlearning-audit.json"
    if gated_path.exists() and gated_audit_path.exists():
        gated = _load(gated_path)
        gated_audit = _load(gated_audit_path)
        random_successes = int(gated["random_evaluation"]["successes"])
        reactive_successes = int(gated["reactive_q_evaluation"]["successes"])
        retained_successes = int(gated["retained_cue_q_evaluation"]["successes"])
        audit_clean = not gated_audit.get("failures")
        status = (
            "requires-independent-replication"
            if audit_clean and gated_audit.get("accepted") and gated.get("accepted_memory_diagnostic")
            else "requires-review"
        )
        evidence.append(Evidence(
            "luanti_gated_delayed_goal_public_cue_diagnostic",
            status,
            "real-Luanti public-vision retained-cue versus reactive tabular Q diagnostic; not a PLWM or neural-controller result",
            f"{gated_path}; {gated_audit_path}",
            "Held-out retained-cue/reactive/random success="
            f"{retained_successes}/16/{reactive_successes}/16/{random_successes}/16. "
            "The random reference is unusually high in this 16-world block, so a disjoint larger held-out replication is mandatory."
        ))
        gated_replication_path = artifact_dir / "luanti-delayed-goal-gated-compact-binary-replication-qlearning-report.json"
        gated_replication_audit_path = artifact_dir / "luanti-delayed-goal-gated-compact-binary-replication-qlearning-audit.json"
        if gated_replication_path.exists() and gated_replication_audit_path.exists():
            replication = _load(gated_replication_path)
            replication_audit = _load(gated_replication_audit_path)
            first_seeds = set(gated["training_seed_range"] + gated["heldout_seed_range"])
            replication_seeds = set(replication["training_seed_range"] + replication["heldout_seed_range"])
            replication_audit_clean = not replication_audit.get("failures")
            independent = not first_seeds.intersection(replication_seeds)
            status = (
                "supported-task-diagnostic-only"
                if audit_clean and gated_audit.get("accepted") and gated.get("accepted_memory_diagnostic")
                and replication_audit_clean and replication_audit.get("accepted")
                and replication.get("accepted_memory_diagnostic") and independent
                else "requires-review"
            )
            evidence.append(Evidence(
                "replicated_luanti_gated_delayed_goal_public_cue_diagnostic",
                status,
                "two real-Luanti public-vision retained-cue versus reactive tabular Q blocks; validated task diagnostic, not PLWM or neural-controller evidence",
                f"{gated_path}; {gated_audit_path}; {gated_replication_path}; {gated_replication_audit_path}",
                "Held-out retained-cue/reactive/random success: first="
                f"{retained_successes}/16/{reactive_successes}/16/{random_successes}/16; replication="
                f"{int(replication['retained_cue_q_evaluation']['successes'])}/64/"
                f"{int(replication['reactive_q_evaluation']['successes'])}/64/"
                f"{int(replication['random_evaluation']['successes'])}/64; "
                f"disjoint-seeds={independent}; replication action-seed-base={replication.get('action_seed_base')}."
            ))
    dqn_path = artifact_dir / "luanti-gated-memory-dqn-formal-report.json"
    dqn_audit_path = artifact_dir / "luanti-gated-memory-dqn-formal-audit.json"
    if dqn_path.exists() and dqn_audit_path.exists():
        dqn = _load(dqn_path)
        dqn_audit = _load(dqn_audit_path)
        results = {item.get("name"): item for item in dqn.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if required.issubset(results):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            status = (
                "supported-neural-task-diagnostic-only"
                if dqn_audit.get("accepted") and dqn.get("accepted_neural_task_diagnostic")
                else "requires-review"
            )
            evidence.append(Evidence(
                "luanti_gated_reward_only_dqn_neural_diagnostic",
                status,
                "real-Luanti two-step public-cue reward-only DQN neural task diagnostic; not an architecture comparison result",
                f"{dqn_path}; {dqn_audit_path}",
                "Held-out no-memory/GRU/Transformer/PLWM-state-off/PLWM success="
                f"{successes['no_memory']}/64/{successes['gru']}/64/{successes['transformer_2']}/64/"
                f"{successes['plwm_state_off']}/64/{successes['plwm']}/64. "
                "GRU and Transformer match full PLWM, so this result cannot support a PLWM advantage claim."
            ))

    long_delay_path = artifact_dir / "luanti-long-delay-dqn-formal-report.json"
    long_delay_audit_path = artifact_dir / "luanti-long-delay-dqn-formal-audit.json"
    if long_delay_path.exists() and long_delay_audit_path.exists():
        long_delay = _load(long_delay_path)
        long_delay_audit = _load(long_delay_audit_path)
        results = {item.get("name"): item for item in long_delay.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if (
            required.issubset(results)
            and long_delay.get("accepted_neural_task_diagnostic")
            and long_delay_audit.get("accepted")
            and not long_delay_audit.get("failures")
        ):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            evidence.append(Evidence(
                "luanti_long_delay_reward_only_dqn_memory_span_diagnostic",
                "supported-neural-task-diagnostic-only",
                "real-Luanti six-event public-cue reward-only DQN memory-span diagnostic; not a PLWM architecture comparison result",
                f"{long_delay_path}; {long_delay_audit_path}",
                "Held-out no-memory/GRU/2-event Transformer/PLWM-state-off/PLWM success="
                f"{successes['no_memory']}/64/{successes['gru']}/64/{successes['transformer_2']}/64/"
                f"{successes['plwm_state_off']}/64/{successes['plwm']}/64. "
                "The result validates a delay beyond the Transformer context, not superiority over GRU."
            ))

    paired_delay_path = artifact_dir / "luanti-paired-delay-dqn-formal-report.json"
    paired_delay_audit_path = artifact_dir / "luanti-paired-delay-dqn-formal-audit.json"
    if paired_delay_path.exists() and paired_delay_audit_path.exists():
        paired_delay = _load(paired_delay_path)
        paired_delay_audit = _load(paired_delay_audit_path)
        results = {item.get("name"): item for item in paired_delay.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if (
            required.issubset(results)
            and paired_delay.get("accepted_neural_task_diagnostic")
            and paired_delay_audit.get("accepted")
            and not paired_delay_audit.get("failures")
        ):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            evidence.append(Evidence(
                "luanti_paired_delay_reward_only_dqn_memory_span_diagnostic",
                "supported-neural-task-diagnostic-only",
                "real-Luanti paired-cue eight-event reward-only DQN memory-span diagnostic; not a PLWM architecture comparison result",
                f"{paired_delay_path}; {paired_delay_audit_path}",
                "Held-out no-memory/GRU/2-event Transformer/PLWM-state-off/PLWM success="
                f"{successes['no_memory']}/128/{successes['gru']}/128/{successes['transformer_2']}/128/"
                f"{successes['plwm_state_off']}/128/{successes['plwm']}/128. "
                "The paired-cue task is a complexity point, not a claim of superiority over GRU."
            ))

    addressed_smoke_path = artifact_dir / "luanti-addressed-gated-dqn-smoke-report.json"
    predictive_smoke_path = artifact_dir / "luanti-addressed-predictive-dqn-smoke-report.json"
    addressed_visual_contract_path = artifact_dir / "luanti-addressed-visual-contract-audit.json"
    visual_contract = _load(addressed_visual_contract_path) if addressed_visual_contract_path.exists() else None
    legacy_addressed_invalid = isinstance(visual_contract, dict) and visual_contract.get("accepted") is True
    if legacy_addressed_invalid and (addressed_smoke_path.exists() or predictive_smoke_path.exists()):
        evidence.append(Evidence(
            "luanti_addressed_legacy_smokes_invalidated_by_public_visual_contract",
            "invalidated-course-visual-contract",
            "legacy six-channel addressed retrieval smokes; invalidated before architecture interpretation",
            f"{addressed_smoke_path}; {predictive_smoke_path}; {addressed_visual_contract_path}",
            "The later real-Luanti public-vision audit proved that the repaired course requires a seventh beacon channel, "
            "four initial rendered identities, five forced south moves, and an identity-only query. The legacy smoke inputs "
            "did not meet that contract, so their scores are retained only as traceability artifacts."
        ))
    elif addressed_smoke_path.exists():
        addressed_smoke = _load(addressed_smoke_path)
        results = {item.get("name"): item for item in addressed_smoke.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if addressed_smoke.get("course") == "addressed_gated" and required.issubset(results):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            random_successes = int(addressed_smoke["random_evaluation"]["successes"])
            evidence.append(Evidence(
                "luanti_addressed_gated_retrieval_smoke_precheck",
                "rejected-smoke-precheck",
                "real-Luanti four-cue addressed retrieval reward-only DQN precheck; no formal block launched",
                str(addressed_smoke_path),
                "Held-out random/no-memory/GRU/2-event Transformer/PLWM-state-off/PLWM success="
                f"{random_successes}/16/{successes['no_memory']}/16/{successes['gru']}/16/"
                f"{successes['transformer_2']}/16/{successes['plwm_state_off']}/16/{successes['plwm']}/16. "
                "Full PLWM showed no learning or relative-advantage signal, so the frozen formal blocks were not launched."
            ))

    if not legacy_addressed_invalid and predictive_smoke_path.exists():
        predictive_smoke = _load(predictive_smoke_path)
        results = {item.get("name"): item for item in predictive_smoke.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if predictive_smoke.get("course") == "addressed_gated" and required.issubset(results):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            evidence.append(Evidence(
                "luanti_addressed_predictive_auxiliary_smoke_precheck",
                "rejected-smoke-precheck",
                "real-Luanti addressed retrieval with fixed 0.1 causal predictive auxiliary; no formal block launched",
                str(predictive_smoke_path),
                "Held-out no-memory/GRU/2-event Transformer/predictive-state-off/full-predictive-PLWM success="
                f"{successes['no_memory']}/32/{successes['gru']}/32/{successes['transformer_2']}/32/"
                f"{successes['plwm_state_off']}/32/{successes['plwm']}/32. "
                "Direct predictive supervision did not create a full-PLWM learning or relative-advantage signal."
            ))

    repaired_addressed_path = artifact_dir / "luanti-addressed-repaired-current-writer-dqn-smoke-report.json"
    repaired_addressed_audit_path = artifact_dir / "luanti-addressed-repaired-current-writer-dqn-smoke-audit.json"
    if repaired_addressed_path.exists() and repaired_addressed_audit_path.exists():
        repaired = _load(repaired_addressed_path)
        repaired_audit = _load(repaired_addressed_audit_path)
        results = {item.get("name"): item for item in repaired.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if (
            repaired.get("course") == "addressed_gated"
            and repaired.get("acceptance_mode") == "addressed_memory_diagnostic"
            and required.issubset(results)
            and not repaired_audit.get("failures")
        ):
            successes = {name: int(results[name]["evaluation"]["successes"]) for name in required}
            evidence.append(Evidence(
                "luanti_addressed_repaired_current_writer_smoke_precheck",
                "rejected-smoke-precheck",
                "real-Luanti repaired seven-channel addressed retrieval current-writer calibration; no architecture claim",
                f"{repaired_addressed_path}; {repaired_addressed_audit_path}",
                "Held-out no-memory/GRU/2-event Transformer/PLWM-state-off/full current PLWM success="
                f"{successes['no_memory']}/16/{successes['gru']}/16/{successes['transformer_2']}/16/"
                f"{successes['plwm_state_off']}/16/{successes['plwm']}/16. "
                "The repaired visual contract passed, but the predeclared current-writer memory-diagnostic gate did not."
            ))

    candidate_path = artifact_dir / "public-vision-multi-candidate-memory-smoke-report.json"
    candidate_audit_path = artifact_dir / "public-vision-multi-candidate-memory-smoke-audit.json"
    if candidate_path.exists() and candidate_audit_path.exists():
        candidate = _load(candidate_path)
        candidate_audit = _load(candidate_audit_path)
        results = {item.get("name"): item for item in candidate.get("results", []) if isinstance(item, dict)}
        required = {"multi_candidate", "single_candidate"}
        if required.issubset(results) and not candidate_audit.get("failures"):
            multi_accuracy = float(results["multi_candidate"]["heldout_accuracy"])
            single_accuracy = float(results["single_candidate"]["heldout_accuracy"])
            status = (
                "requires-independent-replication"
                if candidate_audit.get("accepted") and candidate.get("accepted_mechanism_diagnostic")
                else "rejected-smoke-precheck"
            )
            evidence.append(Evidence(
                "public_vision_multi_candidate_writer_smoke_precheck",
                status,
                "bounded supervised public-vision storage/readout diagnostic; not a Luanti-control or architecture result",
                f"{candidate_path}; {candidate_audit_path}",
                "Held-out multi-candidate/single-candidate accuracy="
                f"{multi_accuracy:.4f}/{single_accuracy:.4f}; exact parameter match="
                f"{results['multi_candidate']['parameters'] == results['single_candidate']['parameters']}. "
                "The predeclared multi-candidate advantage gate did not pass, so no Luanti CUDA smoke was launched."
            ))

    sparse_candidate_path = artifact_dir / "public-vision-sparse-candidate-memory-smoke-report.json"
    sparse_candidate_audit_path = artifact_dir / "public-vision-sparse-candidate-memory-smoke-audit.json"
    if sparse_candidate_path.exists() and sparse_candidate_audit_path.exists():
        sparse = _load(sparse_candidate_path)
        sparse_audit = _load(sparse_candidate_audit_path)
        results = {item.get("name"): item for item in sparse.get("results", []) if isinstance(item, dict)}
        required = {"multi_candidate", "single_candidate"}
        if required.issubset(results) and not sparse_audit.get("failures"):
            multi_accuracy = float(results["multi_candidate"]["heldout_accuracy"])
            single_accuracy = float(results["single_candidate"]["heldout_accuracy"])
            status = (
                "requires-independent-replication"
                if sparse_audit.get("accepted") and sparse.get("accepted_mechanism_diagnostic")
                else "rejected-smoke-precheck"
            )
            evidence.append(Evidence(
                "public_vision_sparse_candidate_writer_smoke_precheck",
                status,
                "bounded supervised public-vision sparse-allocation storage/readout diagnostic; not a Luanti-control or architecture result",
                f"{sparse_candidate_path}; {sparse_candidate_audit_path}",
                "Held-out sparse-multi/single-candidate accuracy="
                f"{multi_accuracy:.4f}/{single_accuracy:.4f}; exact parameter match="
                f"{results['multi_candidate']['parameters'] == results['single_candidate']['parameters']}. "
                "The predeclared sparse-allocation advantage gate did not pass, so no Luanti CUDA smoke was launched."
            ))

    binding_path = artifact_dir / "public-vision-binding-calibration-report.json"
    binding_audit_path = artifact_dir / "public-vision-binding-calibration-audit.json"
    if binding_path.exists() and binding_audit_path.exists():
        binding = _load(binding_path)
        binding_audit = _load(binding_audit_path)
        accuracy = binding.get("heldout_accuracy")
        if isinstance(accuracy, (int, float)) and not binding_audit.get("failures"):
            status = (
                "supported-visual-readout-calibration-only"
                if binding_audit.get("accepted") and binding.get("accepted_public_visual_binding")
                else "rejected-visual-readout-calibration"
            )
            evidence.append(Evidence(
                "public_vision_query_binding_short_budget_calibration",
                status,
                "supervised public-vision initial-cue/delayed-query binding calibration; not reward-only control or PLWM evidence",
                f"{binding_path}; {binding_audit_path}",
                f"Held-out visual binding accuracy={float(accuracy):.4f}; "
                "the fixed short-budget visual readout gate did not pass, so candidate-writer failures do not isolate memory writing."
            ))

    query_binding_path = artifact_dir / "query-conditioned-public-vision-binding-report.json"
    query_binding_audit_path = artifact_dir / "query-conditioned-public-vision-binding-audit.json"
    if query_binding_path.exists() and query_binding_audit_path.exists():
        query_binding = _load(query_binding_path)
        query_binding_audit = _load(query_binding_audit_path)
        results = {item.get("name"): item for item in query_binding.get("results", []) if isinstance(item, dict)}
        required = {"query_conditioned", "late_fusion"}
        if required.issubset(results) and not query_binding_audit.get("failures"):
            query_accuracy = float(results["query_conditioned"]["heldout_accuracy"])
            late_accuracy = float(results["late_fusion"]["heldout_accuracy"])
            status = (
                "supported-visual-readout-calibration-only"
                if query_binding_audit.get("accepted") and query_binding.get("accepted_query_conditioned_binding")
                else "rejected-visual-readout-calibration"
            )
            evidence.append(Evidence(
                "query_conditioned_public_vision_binding_short_budget_calibration",
                status,
                "same-parameter query-conditioned versus late-fusion supervised public-vision binding; not reward-only control or PLWM evidence",
                f"{query_binding_path}; {query_binding_audit_path}",
                "Held-out query-conditioned/late-fusion visual binding accuracy="
                f"{query_accuracy:.4f}/{late_accuracy:.4f}; exact parameter match="
                f"{results['query_conditioned']['parameters'] == results['late_fusion']['parameters']}. "
                "The fixed query-conditioned visual readout gate did not pass."
            ))

    binding_contract_path = artifact_dir / "public-vision-binding-contract-audit.json"
    if binding_contract_path.exists():
        binding_contract = _load(binding_contract_path)
        if binding_contract.get("accepted") and binding_contract.get("public_pixel_oracle_accuracy") == 1.0:
            evidence.append(Evidence(
                "public_vision_delayed_query_contract",
                "supported-course-contract-only",
                "read-only public-pixel reconstruction of the synthetic delayed-query label; not a model input, control result, or architecture claim",
                str(binding_contract_path),
                f"Public-pixel oracle reconstructed {binding_contract.get('episodes')} generated labels exactly; "
                "this validates task determinacy, not learned visual binding."
            ))

    fixed_identity_path = artifact_dir / "public-vision-fixed-identity-calibration-report.json"
    fixed_identity_audit_path = artifact_dir / "public-vision-fixed-identity-calibration-audit.json"
    if fixed_identity_path.exists() and fixed_identity_audit_path.exists():
        fixed_identity = _load(fixed_identity_path)
        fixed_identity_audit = _load(fixed_identity_audit_path)
        accuracy = fixed_identity.get("heldout_accuracy")
        if isinstance(accuracy, (int, float)) and fixed_identity_audit.get("accepted") and fixed_identity.get("accepted_fixed_identity_readout"):
            evidence.append(Evidence(
                "public_vision_fixed_identity_side_short_budget_calibration",
                "supported-visual-readout-calibration-only",
                "supervised one-frame fixed-identity public-vision side readout; not variable query binding, reward-only control, or PLWM evidence",
                f"{fixed_identity_path}; {fixed_identity_audit_path}",
                f"Held-out fixed-identity side accuracy={float(accuracy):.4f}; "
                "this localizes the rejected short-budget visual task to variable query-to-object binding."
            ))

    object_token_path = artifact_dir / "object-token-public-vision-binding-report.json"
    object_token_audit_path = artifact_dir / "object-token-public-vision-binding-audit.json"
    if object_token_path.exists() and object_token_audit_path.exists():
        object_token = _load(object_token_path)
        object_token_audit = _load(object_token_audit_path)
        results = {item.get("name"): item for item in object_token.get("results", []) if isinstance(item, dict)}
        required = {"object_token_query_matching", "query_independent_object_tokens"}
        if required.issubset(results) and not object_token_audit.get("failures"):
            matching_accuracy = float(results["object_token_query_matching"]["heldout_accuracy"])
            ablation_accuracy = float(results["query_independent_object_tokens"]["heldout_accuracy"])
            status = (
                "supported-visual-readout-calibration-only"
                if object_token_audit.get("accepted") and object_token.get("accepted_object_token_binding")
                else "rejected-visual-readout-calibration"
            )
            evidence.append(Evidence(
                "object_token_public_vision_binding_short_budget_calibration",
                status,
                "same-parameter object-token query matching versus query-independent object-token readout; not reward-only control or PLWM evidence",
                f"{object_token_path}; {object_token_audit_path}",
                "Held-out object-token matching/query-independent object-token accuracy="
                f"{matching_accuracy:.4f}/{ablation_accuracy:.4f}; exact parameter match="
                f"{results['object_token_query_matching']['parameters'] == results['query_independent_object_tokens']['parameters']}. "
                "This establishes an object-token visual-binding signal only."
            ))

    object_memory_path = artifact_dir / "object-token-episodic-memory-report.json"
    object_memory_audit_path = artifact_dir / "object-token-episodic-memory-audit.json"
    if object_memory_path.exists() and object_memory_audit_path.exists():
        object_memory = _load(object_memory_path)
        object_memory_audit = _load(object_memory_audit_path)
        results = {item.get("name"): item for item in object_memory.get("results", []) if isinstance(item, dict)}
        required = {"object_token_multi_writer", "object_token_single_writer", "object_token_memory_disabled"}
        if required.issubset(results) and not object_memory_audit.get("failures"):
            multi_accuracy = float(results["object_token_multi_writer"]["heldout_accuracy"])
            single_accuracy = float(results["object_token_single_writer"]["heldout_accuracy"])
            disabled_accuracy = float(results["object_token_memory_disabled"]["heldout_accuracy"])
            status = (
                "supported-visual-memory-diagnostic-only"
                if object_memory_audit.get("accepted") and object_memory.get("accepted_object_token_memory")
                else "rejected-smoke-precheck"
            )
            evidence.append(Evidence(
                "object_token_episodic_memory_short_budget_diagnostic",
                status,
                "same-parameter object-token multi-writer versus aggregated-writer and memory-disabled storage/readout diagnostic; not reward-only control or PLWM architecture evidence",
                f"{object_memory_path}; {object_memory_audit_path}",
                "Held-out multi/single/memory-disabled accuracy="
                f"{multi_accuracy:.4f}/{single_accuracy:.4f}/{disabled_accuracy:.4f}; exact parameter match="
                f"{len({results[name]['parameters'] for name in required}) == 1}. "
                "This establishes an object-token writer/readout signal only."
            ))
    object_token_smoke_path = artifact_dir / "luanti-addressed-object-token-dqn-smoke-report.json"
    object_token_smoke_audit_path = artifact_dir / "luanti-addressed-object-token-dqn-smoke-audit.json"
    if object_token_smoke_path.exists() and object_token_smoke_audit_path.exists():
        smoke = _load(object_token_smoke_path)
        smoke_audit = _load(object_token_smoke_audit_path)
        results = {item.get("name"): item for item in smoke.get("results", []) if isinstance(item, dict)}
        required = {"no_memory", "gru", "transformer_2", "plwm_state_off", "plwm"}
        if required.issubset(results) and not smoke_audit.get("failures"):
            rates = {name: int(results[name]["evaluation"]["successes"]) / int(results[name]["evaluation"]["episodes"]) for name in required}
            status = "requires-independent-replication" if smoke_audit.get("accepted") and smoke.get("accepted_neural_task_diagnostic") else "rejected-smoke-precheck"
            evidence.append(Evidence(
                "luanti_addressed_object_token_writer_cuda_smoke",
                status,
                "single capacity-matched CUDA object-token PLWM writer smoke on repaired addressed retrieval; never an architecture conclusion",
                f"{object_token_smoke_path}; {object_token_smoke_audit_path}",
                "Held-out no-memory/GRU/2-event Transformer/state-off/full object-token PLWM success rates="
                f"{rates['no_memory']:.4f}/{rates['gru']:.4f}/{rates['transformer_2']:.4f}/{rates['plwm_state_off']:.4f}/{rates['plwm']:.4f}. "
                "One smoke can at most require an independent replication."
            ))
    return evidence


def run(*, artifact_dir: Path | None = None, output: Path | None = None) -> dict[str, Any]:
    directory = artifact_dir or ROOT / "artifacts"
    entries = build_registry(directory)
    payload = {
        "protocol": "plwm_evidence_registry_v1: completed artifact classification; bounded interfaces never imply end-to-end controller superiority",
        "entries": [asdict(entry) for entry in entries],
        "counts": {status: sum(entry.status == status for entry in entries) for status in sorted({entry.status for entry in entries})},
    }
    target = output or directory / "plwm-evidence-registry.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(artifact_dir=args.artifact_dir, output=args.output), indent=2))
