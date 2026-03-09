#!/usr/bin/env python3
# ruff: noqa: E402
"""Minimal cue discrimination benchmark for oNeuro.

This benchmark sits between the mechanistic corticostriatal pair protocol and
the action-bias task. It asks a narrower question:

Can a tiny benchmark-safe microcircuit learn to map two cues onto two competing
readout populations, and do clean controls abolish that effect?

Compared with the action-bias benchmark, this surface is intentionally simpler:
- two cues and two class labels only
- symmetric cue -> output wiring
- continuous classification readout instead of action semantics
- teacher-assisted training with reward-modulated plasticity
- matched-seed controls for no learning, shuffled labels, and NMDA block
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from experiments.benchmark_shared import (
    DEFAULT_PHASE_SEED_OFFSETS,
    restore_nmda_conductance_scale,
    set_nmda_conductance_scale,
)
from experiments.benchmark_shared import (
    add_projection as _add_projection,
)
from experiments.benchmark_shared import (
    apply_task_reward_modulated_plasticity as _apply_task_reward_modulated_plasticity,
)
from experiments.benchmark_shared import (
    brain_seed as _brain_seed,
)
from experiments.benchmark_shared import (
    collect_pathways as _collect_pathways_for_groups,
)
from experiments.benchmark_shared import (
    cue_schedule as _cue_schedule,
)
from experiments.benchmark_shared import (
    delta_summary as _delta_summary,
)
from experiments.benchmark_shared import (
    frozen_plasticity as _frozen_plasticity,
)
from experiments.benchmark_shared import (
    git_commit as _shared_git_commit,
)
from experiments.benchmark_shared import (
    mean as _mean,
)
from experiments.benchmark_shared import (
    mechanistic_totals as _mechanistic_totals_for_groups,
)
from experiments.benchmark_shared import (
    phase_seed as _phase_seed,
)
from experiments.benchmark_shared import (
    population_activity as _population_activity,
)
from experiments.benchmark_shared import (
    sample_std as _sample_std,
)
from experiments.benchmark_shared import (
    stimulate_population as _stimulate_population,
)
from experiments.benchmark_shared import (
    warmup_network as _warmup_network,
)
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import NeuronArchetype

ConditionName = str
CueName = str

DEFAULT_CONDITIONS: tuple[ConditionName, ...] = (
    "full_learning",
    "no_learning",
    "label_shuffle",
    "nmda_block",
)
PHASE_SEED_OFFSETS = {
    **DEFAULT_PHASE_SEED_OFFSETS,
}


@dataclass(slots=True, frozen=True)
class BenchmarkConfig:
    """Simulation settings for one paired-condition discrimination run."""

    dt: float = 0.1
    warmup_steps: int = 100
    baseline_trials: int = 20
    training_trials: int = 100
    evaluation_trials: int = 20
    cue_steps: int = 30
    response_steps: int = 20
    feedback_steps: int = 10
    inter_trial_steps: int = 10
    stimulation_period: int = 2
    cue_intensity: float = 55.0
    teaching_current: float = 16.0
    teaching_steps: int = 8
    connection_probability: float = 0.55
    recurrent_inhibition_probability: float = 0.35
    task_stdp_factor: float = 0.0
    cue_population_size: int = 6
    output_population_size: int = 6
    reward_amount: float = 2.0
    learning_rate: float = 0.2
    shuffle_within_blocks: bool = True
    trial_balance_block: int = 4

    def normalized(self) -> "BenchmarkConfig":
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.cue_steps <= 0 or self.response_steps <= 0:
            raise ValueError("cue_steps and response_steps must be positive")
        if self.feedback_steps < 0 or self.inter_trial_steps < 0:
            raise ValueError("feedback_steps and inter_trial_steps must be non-negative")
        if self.stimulation_period <= 0:
            raise ValueError("stimulation_period must be positive")
        if self.connection_probability <= 0.0 or self.connection_probability > 1.0:
            raise ValueError("connection_probability must be in (0, 1]")
        if (
            self.recurrent_inhibition_probability < 0.0
            or self.recurrent_inhibition_probability > 1.0
        ):
            raise ValueError("recurrent_inhibition_probability must be in [0, 1]")
        if self.task_stdp_factor < 0.0 or self.task_stdp_factor > 1.0:
            raise ValueError("task_stdp_factor must be in [0, 1]")
        if self.teaching_current < 0.0 or self.teaching_steps < 0:
            raise ValueError("teaching_current and teaching_steps must be non-negative")
        if self.cue_population_size <= 1:
            raise ValueError("cue_population_size must exceed 1")
        if self.output_population_size <= 1:
            raise ValueError("output_population_size must exceed 1")
        if self.reward_amount <= 0.0:
            raise ValueError("reward_amount must be positive")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        return self


@dataclass(slots=True, frozen=True)
class TaskTopology:
    """Neuron groups that define the benchmark cue and readout pathways."""

    cue_a_ids: tuple[int, ...]
    cue_b_ids: tuple[int, ...]
    output_a_ids: tuple[int, ...]
    output_b_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class TrialOutcome:
    """Serializable output for one cue presentation."""

    phase: str
    trial_index: int
    cue: CueName
    expected_label: str
    decision: str
    correct: bool
    output_a_activity: float
    output_b_activity: float
    output_a_spikes: int
    output_b_spikes: int
    decision_margin: float
    task_consistent_margin: float
    applied_reward: float


@dataclass(slots=True)
class BenchmarkRecord:
    """Serializable output for one seed-condition pair."""

    pair_index: int
    condition: ConditionName
    seed: int
    config: dict[str, Any]
    topology_summary: dict[str, Any]
    pre_training_summary: dict[str, float]
    training_summary: dict[str, float]
    post_training_summary: dict[str, float]
    discrimination_margin_improvement: float
    accuracy_improvement: float
    mechanistic_summary: dict[str, Any]
    trial_records: list[dict[str, Any]]
    wall_time_s: float


def _build_microcircuit(
    config: BenchmarkConfig,
    seed: int,
) -> tuple[MolecularNeuralNetwork, TaskTopology, dict[str, int]]:
    rng = np.random.default_rng(seed)
    net = MolecularNeuralNetwork(
        initial_neurons=0,
        size=(8.0, 8.0, 4.0),
        enable_advanced_neurons=True,
        benchmark_safe_mode=True,
        psc_scale=45.0,
    )
    net.learning_rate = config.learning_rate

    cue_a_ids: list[int] = []
    cue_b_ids: list[int] = []
    output_a_ids: list[int] = []
    output_b_ids: list[int] = []

    for _ in range(config.cue_population_size):
        cue_a_ids.append(
            net.create_neuron_at(
                1.5 + rng.uniform(-0.6, 0.6),
                2.0 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )
        cue_b_ids.append(
            net.create_neuron_at(
                1.5 + rng.uniform(-0.6, 0.6),
                5.5 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )

    for _ in range(config.output_population_size):
        output_a_ids.append(
            net.create_neuron_at(
                5.6 + rng.uniform(-0.5, 0.5),
                2.2 + rng.uniform(-0.7, 0.7),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.D1_MSN,
            )
        )
        output_b_ids.append(
            net.create_neuron_at(
                5.6 + rng.uniform(-0.5, 0.5),
                5.0 + rng.uniform(-0.7, 0.7),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.D2_MSN,
            )
        )

    topology = TaskTopology(
        cue_a_ids=tuple(cue_a_ids),
        cue_b_ids=tuple(cue_b_ids),
        output_a_ids=tuple(output_a_ids),
        output_b_ids=tuple(output_b_ids),
    )

    created = {
        "cue_a_to_output_a": _add_projection(
            net,
            topology.cue_a_ids,
            topology.output_a_ids,
            config.connection_probability,
            rng,
        ),
        "cue_a_to_output_b": _add_projection(
            net,
            topology.cue_a_ids,
            topology.output_b_ids,
            config.connection_probability,
            rng,
        ),
        "cue_b_to_output_a": _add_projection(
            net,
            topology.cue_b_ids,
            topology.output_a_ids,
            config.connection_probability,
            rng,
        ),
        "cue_b_to_output_b": _add_projection(
            net,
            topology.cue_b_ids,
            topology.output_b_ids,
            config.connection_probability,
            rng,
        ),
        "output_a_to_output_b": _add_projection(
            net,
            topology.output_a_ids,
            topology.output_b_ids,
            config.recurrent_inhibition_probability,
            rng,
            "gaba",
        ),
        "output_b_to_output_a": _add_projection(
            net,
            topology.output_b_ids,
            topology.output_a_ids,
            config.recurrent_inhibition_probability,
            rng,
            "gaba",
        ),
        "output_a_to_output_a": _add_projection(
            net,
            topology.output_a_ids,
            topology.output_a_ids,
            0.15,
            rng,
            "gaba",
        ),
        "output_b_to_output_b": _add_projection(
            net,
            topology.output_b_ids,
            topology.output_b_ids,
            0.15,
            rng,
            "gaba",
        ),
    }
    return net, topology, created


def _warmup(net: MolecularNeuralNetwork, config: BenchmarkConfig) -> None:
    _warmup_network(net, warmup_steps=config.warmup_steps, dt=config.dt)


def _task_synapse_keys(topology: TaskTopology) -> set[tuple[int, int]]:
    return {
        (pre, post)
        for pre_ids, post_ids in (
            (topology.cue_a_ids, topology.output_a_ids),
            (topology.cue_a_ids, topology.output_b_ids),
            (topology.cue_b_ids, topology.output_a_ids),
            (topology.cue_b_ids, topology.output_b_ids),
        )
        for pre in pre_ids
        for post in post_ids
    }


def _freeze_non_task_plasticity(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
    task_stdp_factor: float,
) -> None:
    _task_keys = _task_synapse_keys(topology)
    for key, syn in net._molecular_synapses.items():
        syn._plasticity_factor = task_stdp_factor if key in _task_keys else 0.0


def _expected_label(cue: CueName) -> str:
    return "A" if cue == "cue_a" else "B"


def _task_consistent_margin(cue: CueName, margin: float) -> float:
    return margin if cue == "cue_a" else -margin


def _assigned_label(cue: CueName, condition: ConditionName) -> str:
    label = _expected_label(cue)
    if condition != "label_shuffle":
        return label
    return "B" if label == "A" else "A"


def _teaching_population(topology: TaskTopology, assigned_label: str) -> tuple[int, ...]:
    return topology.output_a_ids if assigned_label == "A" else topology.output_b_ids


def _apply_reward_modulated_plasticity(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
) -> None:
    _apply_task_reward_modulated_plasticity(net, _task_synapse_keys(topology))


def _training_reward(
    condition: ConditionName,
    assigned_label: str,
    config: BenchmarkConfig,
) -> float:
    if condition == "no_learning":
        return 0.0
    return config.reward_amount if assigned_label == "A" else -config.reward_amount


def _run_trials(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
    config: BenchmarkConfig,
    n_trials: int,
    phase: str,
    schedule_seed: int,
    condition: ConditionName,
    enable_learning: bool,
) -> list[TrialOutcome]:
    schedule = _cue_schedule(
        n_trials,
        schedule_seed,
        ("cue_a", "cue_b"),
        shuffle_within_blocks=config.shuffle_within_blocks,
        block_size=config.trial_balance_block,
    )
    outcomes: list[TrialOutcome] = []

    for trial_index, cue in enumerate(schedule):
        cue_population = topology.cue_a_ids if cue == "cue_a" else topology.cue_b_ids
        assigned_label = _assigned_label(cue, condition)
        teacher_population = _teaching_population(topology, assigned_label)
        output_a_trace: list[float] = []
        output_b_trace: list[float] = []
        output_a_spikes_total = 0
        output_b_spikes_total = 0

        total_steps = config.cue_steps + config.response_steps
        readout_start = max(0, config.cue_steps // 3)
        for step_i in range(total_steps):
            if step_i < config.cue_steps and step_i % config.stimulation_period == 0:
                _stimulate_population(net, cue_population, config.cue_intensity)
            if (
                enable_learning
                and config.teaching_steps > 0
                and config.teaching_current > 0.0
                and config.cue_steps <= step_i < (config.cue_steps + config.teaching_steps)
                and step_i % config.stimulation_period == 0
            ):
                _stimulate_population(net, teacher_population, config.teaching_current)
            net.step(config.dt)
            if enable_learning:
                net.update_eligibility_traces(dt=config.dt)
            if step_i >= readout_start:
                output_a_activity, output_a_spikes = _population_activity(
                    net, topology.output_a_ids
                )
                output_b_activity, output_b_spikes = _population_activity(
                    net, topology.output_b_ids
                )
                output_a_trace.append(output_a_activity)
                output_b_trace.append(output_b_activity)
                output_a_spikes_total += output_a_spikes
                output_b_spikes_total += output_b_spikes

        output_a_activity = _mean(output_a_trace)
        output_b_activity = _mean(output_b_trace)
        decision_margin = output_a_activity - output_b_activity
        decision = "A" if decision_margin > 0.0 else "B"
        expected_label = _expected_label(cue)
        correct = decision == expected_label
        applied_reward = 0.0

        if enable_learning:
            applied_reward = _training_reward(condition, assigned_label, config)
            if applied_reward != 0.0:
                net.release_dopamine(applied_reward)
                _apply_reward_modulated_plasticity(net, topology)
                for _ in range(config.feedback_steps):
                    net.step(config.dt)

        for _ in range(config.inter_trial_steps):
            net.step(config.dt)

        outcomes.append(
            TrialOutcome(
                phase=phase,
                trial_index=trial_index,
                cue=cue,
                expected_label=expected_label,
                decision=decision,
                correct=correct,
                output_a_activity=output_a_activity,
                output_b_activity=output_b_activity,
                output_a_spikes=output_a_spikes_total,
                output_b_spikes=output_b_spikes_total,
                decision_margin=decision_margin,
                task_consistent_margin=_task_consistent_margin(cue, decision_margin),
                applied_reward=applied_reward,
            )
        )

    return outcomes


def _phase_summary(outcomes: list[TrialOutcome]) -> dict[str, float]:
    if not outcomes:
        return {
            "n_trials": 0.0,
            "accuracy": 0.0,
            "a_accuracy": 0.0,
            "b_accuracy": 0.0,
            "a_margin": 0.0,
            "b_margin": 0.0,
            "task_consistent_margin": 0.0,
            "mean_output_a_activity": 0.0,
            "mean_output_b_activity": 0.0,
            "mean_output_a_spikes": 0.0,
            "mean_output_b_spikes": 0.0,
        }

    cue_a = [trial for trial in outcomes if trial.cue == "cue_a"]
    cue_b = [trial for trial in outcomes if trial.cue == "cue_b"]
    return {
        "n_trials": float(len(outcomes)),
        "accuracy": _mean(float(trial.correct) for trial in outcomes),
        "a_accuracy": _mean(float(trial.correct) for trial in cue_a),
        "b_accuracy": _mean(float(trial.correct) for trial in cue_b),
        "a_margin": _mean(float(trial.decision_margin) for trial in cue_a),
        "b_margin": _mean(float(trial.decision_margin) for trial in cue_b),
        "task_consistent_margin": _mean(
            float(trial.task_consistent_margin) for trial in outcomes
        ),
        "mean_output_a_activity": _mean(
            float(trial.output_a_activity) for trial in outcomes
        ),
        "mean_output_b_activity": _mean(
            float(trial.output_b_activity) for trial in outcomes
        ),
        "mean_output_a_spikes": _mean(float(trial.output_a_spikes) for trial in outcomes),
        "mean_output_b_spikes": _mean(float(trial.output_b_spikes) for trial in outcomes),
    }


def _training_summary(outcomes: list[TrialOutcome]) -> dict[str, float]:
    summary = _phase_summary(outcomes)
    if not outcomes:
        summary["first_block_accuracy"] = 0.0
        summary["last_block_accuracy"] = 0.0
        summary["first_block_margin"] = 0.0
        summary["last_block_margin"] = 0.0
        summary["mean_applied_reward"] = 0.0
        return summary

    block_size = min(10, len(outcomes))
    summary["first_block_accuracy"] = _mean(
        float(trial.correct) for trial in outcomes[:block_size]
    )
    summary["last_block_accuracy"] = _mean(
        float(trial.correct) for trial in outcomes[-block_size:]
    )
    summary["first_block_margin"] = _mean(
        float(trial.task_consistent_margin) for trial in outcomes[:block_size]
    )
    summary["last_block_margin"] = _mean(
        float(trial.task_consistent_margin) for trial in outcomes[-block_size:]
    )
    summary["mean_applied_reward"] = _mean(
        float(trial.applied_reward) for trial in outcomes
    )
    return summary


def _pathway_groups(
    topology: TaskTopology,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    return {
        "cue_a_to_output_a": (topology.cue_a_ids, topology.output_a_ids),
        "cue_a_to_output_b": (topology.cue_a_ids, topology.output_b_ids),
        "cue_b_to_output_a": (topology.cue_b_ids, topology.output_a_ids),
        "cue_b_to_output_b": (topology.cue_b_ids, topology.output_b_ids),
    }


def _collect_pathways(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
) -> dict[str, dict[str, float]]:
    return _collect_pathways_for_groups(net, _pathway_groups(topology))


def _mechanistic_totals(delta: dict[str, dict[str, float]]) -> dict[str, float]:
    return _mechanistic_totals_for_groups(
        delta,
        aligned_groups=("cue_a_to_output_a", "cue_b_to_output_b"),
        cross_groups=("cue_a_to_output_b", "cue_b_to_output_a"),
    )


# Public task helpers used by the consolidated capability suite.
brain_seed = _brain_seed
phase_seed = _phase_seed
build_microcircuit = _build_microcircuit
freeze_non_task_plasticity = _freeze_non_task_plasticity
population_activity = _population_activity
run_trials = _run_trials
warmup_microcircuit = _warmup


def run_condition(
    pair_index: int,
    condition: ConditionName,
    config: BenchmarkConfig,
    master_seed: int = 1234,
) -> BenchmarkRecord:
    """Run one seed-condition benchmark instance."""
    start = time.perf_counter()
    config = config.normalized()
    seed = _brain_seed(master_seed, pair_index)
    net, topology, created = _build_microcircuit(
        config,
        seed=_phase_seed(seed, "topology"),
    )

    task_stdp_factor = 0.0 if condition == "no_learning" else config.task_stdp_factor
    _freeze_non_task_plasticity(net, topology, task_stdp_factor)
    if condition == "no_learning":
        net.learning_rate = 0.0

    _warmup(net, config)
    initial_pathways = _collect_pathways(net, topology)

    pre_probe_net = deepcopy(net)
    with _frozen_plasticity(pre_probe_net):
        pre_training_trials = _run_trials(
            pre_probe_net,
            topology,
            config,
            n_trials=config.baseline_trials,
            phase="pre_training",
            schedule_seed=_phase_seed(seed, "pre_training"),
            condition=condition,
            enable_learning=False,
        )

    nmda_saved: dict[int, float] | None = None
    if condition == "nmda_block":
        nmda_saved = set_nmda_conductance_scale(net, 0.0)

    try:
        training_trials = _run_trials(
            net,
            topology,
            config,
            n_trials=config.training_trials,
            phase="training",
            schedule_seed=_phase_seed(seed, "training"),
            condition=condition,
            enable_learning=True,
        )
    finally:
        if nmda_saved is not None:
            restore_nmda_conductance_scale(net, nmda_saved)

    final_pathways = _collect_pathways(net, topology)

    post_probe_net = deepcopy(net)
    with _frozen_plasticity(post_probe_net):
        post_training_trials = _run_trials(
            post_probe_net,
            topology,
            config,
            n_trials=config.evaluation_trials,
            phase="post_training",
            schedule_seed=_phase_seed(seed, "post_training"),
            condition=condition,
            enable_learning=False,
        )

    pre_summary = _phase_summary(pre_training_trials)
    training_summary = _training_summary(training_trials)
    post_summary = _phase_summary(post_training_trials)
    delta = _delta_summary(initial_pathways, final_pathways)

    topology_summary = {
        "cue_a_count": len(topology.cue_a_ids),
        "cue_b_count": len(topology.cue_b_ids),
        "output_a_count": len(topology.output_a_ids),
        "output_b_count": len(topology.output_b_ids),
        "created_cue_a_to_output_a": created["cue_a_to_output_a"],
        "created_cue_a_to_output_b": created["cue_a_to_output_b"],
        "created_cue_b_to_output_a": created["cue_b_to_output_a"],
        "created_cue_b_to_output_b": created["cue_b_to_output_b"],
        "n_neurons": len(net._molecular_neurons),
        "n_synapses": len(net._molecular_synapses),
    }

    trial_records = [
        asdict(trial)
        for trial in (pre_training_trials + training_trials + post_training_trials)
    ]

    return BenchmarkRecord(
        pair_index=pair_index,
        condition=condition,
        seed=seed,
        config=asdict(config),
        topology_summary=topology_summary,
        pre_training_summary=pre_summary,
        training_summary=training_summary,
        post_training_summary=post_summary,
        discrimination_margin_improvement=float(
            post_summary["task_consistent_margin"]
            - pre_summary["task_consistent_margin"]
        ),
        accuracy_improvement=float(post_summary["accuracy"] - pre_summary["accuracy"]),
        mechanistic_summary={
            "initial_pathways": initial_pathways,
            "final_pathways": final_pathways,
            "pathway_deltas": delta,
            "delta_summary": _mechanistic_totals(delta),
        },
        trial_records=trial_records,
        wall_time_s=float(time.perf_counter() - start),
    )


def _condition_summary(records: list[BenchmarkRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(record.condition, []).append(record)

    summary: dict[str, dict[str, float]] = {}
    for condition, group in grouped.items():
        margin_improvements = [
            record.discrimination_margin_improvement for record in group
        ]
        acc_improvements = [record.accuracy_improvement for record in group]
        pre_margin = [
            record.pre_training_summary["task_consistent_margin"] for record in group
        ]
        post_margin = [
            record.post_training_summary["task_consistent_margin"] for record in group
        ]
        pre_acc = [record.pre_training_summary["accuracy"] for record in group]
        post_acc = [record.post_training_summary["accuracy"] for record in group]
        summary[condition] = {
            "n_runs": float(len(group)),
            "mean_pre_training_margin": _mean(pre_margin),
            "mean_post_training_margin": _mean(post_margin),
            "mean_margin_improvement": _mean(margin_improvements),
            "std_margin_improvement": _sample_std(margin_improvements),
            "mean_pre_training_accuracy": _mean(pre_acc),
            "mean_post_training_accuracy": _mean(post_acc),
            "mean_accuracy_improvement": _mean(acc_improvements),
            "std_accuracy_improvement": _sample_std(acc_improvements),
            "mean_training_last_block_accuracy": _mean(
                record.training_summary["last_block_accuracy"] for record in group
            ),
            "mean_training_last_block_margin": _mean(
                record.training_summary["last_block_margin"] for record in group
            ),
            "mean_aligned_weight_delta": _mean(
                record.mechanistic_summary["delta_summary"]["aligned_weight_delta"]
                for record in group
            ),
            "mean_cross_weight_delta": _mean(
                record.mechanistic_summary["delta_summary"]["cross_weight_delta"]
                for record in group
            ),
        }
    return summary


def _paired_differences(
    records: list[BenchmarkRecord],
    anchor_condition: str = "full_learning",
) -> dict[str, dict[str, Any]]:
    by_pair: dict[int, dict[str, BenchmarkRecord]] = {}
    for record in records:
        by_pair.setdefault(record.pair_index, {})[record.condition] = record

    contrasts: dict[str, dict[str, Any]] = {}
    comparison_conditions = sorted(
        {record.condition for record in records if record.condition != anchor_condition}
    )
    for condition in comparison_conditions:
        margin_diffs: list[float] = []
        accuracy_diffs: list[float] = []
        aligned_weight_diffs: list[float] = []
        for pair_records in by_pair.values():
            anchor = pair_records.get(anchor_condition)
            other = pair_records.get(condition)
            if anchor is None or other is None:
                continue
            margin_diffs.append(
                anchor.discrimination_margin_improvement
                - other.discrimination_margin_improvement
            )
            accuracy_diffs.append(anchor.accuracy_improvement - other.accuracy_improvement)
            aligned_weight_diffs.append(
                anchor.mechanistic_summary["delta_summary"]["aligned_weight_delta"]
                - other.mechanistic_summary["delta_summary"]["aligned_weight_delta"]
            )
        contrasts[condition] = {
            "n_pairs": len(margin_diffs),
            "mean_margin_improvement_difference": _mean(margin_diffs),
            "raw_margin_improvement_differences": margin_diffs,
            "mean_accuracy_improvement_difference": _mean(accuracy_diffs),
            "mean_aligned_weight_delta_difference": _mean(aligned_weight_diffs),
        }
    return contrasts


def run_experiment(
    conditions: Iterable[ConditionName] = DEFAULT_CONDITIONS,
    n_seeds: int = 10,
    config: BenchmarkConfig = BenchmarkConfig(),
    output_dir: str | Path = REPO_ROOT / "experiments" / "results",
    workers: int = 1,
    master_seed: int = 1234,
) -> dict[str, Any]:
    """Run the benchmark across conditions and seeds, saving JSON output."""
    config = config.normalized()
    conditions = tuple(conditions)
    tasks = [
        (pair_index, condition)
        for pair_index in range(n_seeds)
        for condition in conditions
    ]
    records: list[BenchmarkRecord] = []

    if workers <= 1:
        for pair_index, condition in tasks:
            records.append(
                run_condition(
                    pair_index=pair_index,
                    condition=condition,
                    config=config,
                    master_seed=master_seed,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_condition,
                    pair_index,
                    condition,
                    config,
                    master_seed,
                ): (pair_index, condition)
                for pair_index, condition in tasks
            }
            for future in as_completed(futures):
                records.append(future.result())

    records.sort(key=lambda record: (record.pair_index, record.condition))
    summary = _condition_summary(records)
    paired = _paired_differences(records)

    timestamp = int(time.time())
    output_path = Path(output_dir) / f"stimulus_discrimination_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "stimulus_discrimination_benchmark",
        "config": asdict(config),
        "conditions": list(conditions),
        "n_seeds": n_seeds,
        "master_seed": master_seed,
        "summary": summary,
        "paired_differences": paired,
        "results": [asdict(record) for record in records],
        "metadata": {
            "git_commit": _shared_git_commit(REPO_ROOT),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timestamp": timestamp,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    payload["result_path"] = str(output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=1234)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--baseline-trials", type=int, default=20)
    parser.add_argument("--training-trials", type=int, default=100)
    parser.add_argument("--evaluation-trials", type=int, default=20)
    parser.add_argument("--cue-steps", type=int, default=30)
    parser.add_argument("--response-steps", type=int, default=20)
    parser.add_argument("--feedback-steps", type=int, default=10)
    parser.add_argument("--inter-trial-steps", type=int, default=10)
    parser.add_argument("--stimulation-period", type=int, default=2)
    parser.add_argument("--cue-intensity", type=float, default=55.0)
    parser.add_argument("--connection-probability", type=float, default=0.55)
    parser.add_argument(
        "--recurrent-inhibition-probability",
        type=float,
        default=0.35,
    )
    parser.add_argument("--teaching-current", type=float, default=16.0)
    parser.add_argument("--teaching-steps", type=int, default=8)
    parser.add_argument("--reward-amount", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--task-stdp-factor", type=float, default=0.0)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(DEFAULT_CONDITIONS),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "results"),
    )
    args = parser.parse_args()

    payload = run_experiment(
        conditions=args.conditions,
        n_seeds=args.n_seeds,
        config=BenchmarkConfig(
            warmup_steps=args.warmup_steps,
            baseline_trials=args.baseline_trials,
            training_trials=args.training_trials,
            evaluation_trials=args.evaluation_trials,
            cue_steps=args.cue_steps,
            response_steps=args.response_steps,
            feedback_steps=args.feedback_steps,
            inter_trial_steps=args.inter_trial_steps,
            stimulation_period=args.stimulation_period,
            cue_intensity=args.cue_intensity,
            connection_probability=args.connection_probability,
            recurrent_inhibition_probability=args.recurrent_inhibition_probability,
            teaching_current=args.teaching_current,
            teaching_steps=args.teaching_steps,
            reward_amount=args.reward_amount,
            learning_rate=args.learning_rate,
            task_stdp_factor=args.task_stdp_factor,
        ),
        output_dir=args.output_dir,
        workers=args.workers,
        master_seed=args.master_seed,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved results to {payload['result_path']}")


if __name__ == "__main__":
    main()
