#!/usr/bin/env python3
# ruff: noqa: E402
"""Initial cue-conditioned corticostriatal action-bias benchmark for oNeuro.

This benchmark intentionally starts smaller than the regional-brain Go/No-Go demo.
It asks a narrower question:

Can a tiny corticostriatal microcircuit learn cue-specific D1-versus-D2 action
biases, and does that effect depend on dopamine and NMDA-dependent plasticity?

Compared with the earlier benchmark, this version:
- uses symmetric cue -> D1 and cue -> D2 projections
- uses continuous population activity instead of sparse spike-count voting
- separates probe, training, feedback, and evaluation phases
- runs in benchmark-safe mode to avoid structural drift during short assays
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from oneuro.molecular.ion_channels import IonChannelType
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import NeuronArchetype
from oneuro.molecular.receptors import ReceptorType

ConditionName = str
CueName = str

DEFAULT_CONDITIONS: tuple[ConditionName, ...] = (
    "full_learning",
    "no_dopamine",
    "nmda_block",
    "reward_shuffle",
)
PHASE_SEED_OFFSETS = {
    "pre_training": 1_001,
    "training": 2_003,
    "post_training": 3_007,
    "topology": 4_009,
}


@dataclass(slots=True, frozen=True)
class BenchmarkConfig:
    """Simulation settings for one paired-condition action-bias run."""

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
    task_stdp_factor: float = 0.0
    cue_population_size: int = 6
    d1_population_size: int = 6
    d2_population_size: int = 6
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
        if self.task_stdp_factor < 0.0 or self.task_stdp_factor > 1.0:
            raise ValueError("task_stdp_factor must be in [0, 1]")
        if self.teaching_current < 0.0 or self.teaching_steps < 0:
            raise ValueError("teaching_current and teaching_steps must be non-negative")
        if self.cue_population_size <= 1:
            raise ValueError("cue_population_size must exceed 1")
        if self.d1_population_size <= 1 or self.d2_population_size <= 1:
            raise ValueError("d1_population_size and d2_population_size must exceed 1")
        if self.reward_amount <= 0.0:
            raise ValueError("reward_amount must be positive")
        return self


@dataclass(slots=True, frozen=True)
class TaskTopology:
    """Neuron groups that define the benchmark cue and readout pathways."""

    cue_green_ids: tuple[int, ...]
    cue_red_ids: tuple[int, ...]
    d1_ids: tuple[int, ...]
    d2_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class TrialOutcome:
    """Serializable output for one cue presentation."""

    phase: str
    trial_index: int
    cue: CueName
    expected: str
    decision: str
    correct: bool
    d1_activity: float
    d2_activity: float
    d1_spikes: int
    d2_spikes: int
    action_bias: float
    task_consistent_bias: float
    task_feedback: float
    applied_feedback: float


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
    task_consistent_bias_improvement: float
    accuracy_improvement: float
    mechanistic_summary: dict[str, Any]
    trial_records: list[dict[str, Any]]
    wall_time_s: float


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return float(np.mean(np.asarray(data, dtype=np.float64)))


def _sample_std(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    return float(np.std(np.asarray(data, dtype=np.float64), ddof=1))


def _brain_seed(master_seed: int, pair_index: int) -> int:
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, pair_index]))
    return int(rng.integers(0, 2**32, dtype=np.uint32))


def _phase_seed(seed: int, phase: str) -> int:
    return int(seed + PHASE_SEED_OFFSETS[phase])


def _cue_schedule(
    n_trials: int,
    seed: int,
    shuffle_within_blocks: bool,
    block_size: int,
) -> list[CueName]:
    if n_trials <= 0:
        return []

    if not shuffle_within_blocks:
        return ["green" if idx % 2 == 0 else "red" for idx in range(n_trials)]

    rng = np.random.default_rng(seed)
    schedule: list[CueName] = []
    remaining = n_trials
    balanced_block = max(2, block_size)

    while remaining > 0:
        current_block = min(remaining, balanced_block)
        n_green = current_block // 2
        n_red = current_block - n_green
        block = ["green"] * n_green + ["red"] * n_red
        rng.shuffle(block)
        schedule.extend(block)
        remaining -= current_block

    return schedule[:n_trials]


def _add_projection(
    net: MolecularNeuralNetwork,
    source_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    probability: float,
    rng: np.random.Generator,
    nt_name: str = "glutamate",
) -> int:
    created = 0
    for src in source_ids:
        for tgt in target_ids:
            if src == tgt:
                continue
            if rng.random() < probability:
                net.create_synapse(src, tgt, nt_name)
                created += 1
    return created


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

    cue_green_ids: list[int] = []
    cue_red_ids: list[int] = []
    d1_ids: list[int] = []
    d2_ids: list[int] = []

    for idx in range(config.cue_population_size):
        cue_green_ids.append(
            net.create_neuron_at(
                1.5 + rng.uniform(-0.6, 0.6),
                2.0 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )
        cue_red_ids.append(
            net.create_neuron_at(
                1.5 + rng.uniform(-0.6, 0.6),
                5.5 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )

    for _ in range(config.d1_population_size):
        d1_ids.append(
            net.create_neuron_at(
                5.5 + rng.uniform(-0.6, 0.6),
                2.5 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.D1_MSN,
            )
        )
    for _ in range(config.d2_population_size):
        d2_ids.append(
            net.create_neuron_at(
                5.5 + rng.uniform(-0.6, 0.6),
                5.0 + rng.uniform(-0.8, 0.8),
                1.5 + rng.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.D2_MSN,
            )
        )

    topology = TaskTopology(
        cue_green_ids=tuple(cue_green_ids),
        cue_red_ids=tuple(cue_red_ids),
        d1_ids=tuple(d1_ids),
        d2_ids=tuple(d2_ids),
    )

    created = {
        "green_to_d1": _add_projection(
            net,
            topology.cue_green_ids,
            topology.d1_ids,
            config.connection_probability,
            rng,
        ),
        "green_to_d2": _add_projection(
            net,
            topology.cue_green_ids,
            topology.d2_ids,
            config.connection_probability,
            rng,
        ),
        "red_to_d1": _add_projection(
            net,
            topology.cue_red_ids,
            topology.d1_ids,
            config.connection_probability,
            rng,
        ),
        "red_to_d2": _add_projection(
            net,
            topology.cue_red_ids,
            topology.d2_ids,
            config.connection_probability,
            rng,
        ),
        "d1_to_d2": _add_projection(net, topology.d1_ids, topology.d2_ids, 0.35, rng, "gaba"),
        "d2_to_d1": _add_projection(net, topology.d2_ids, topology.d1_ids, 0.35, rng, "gaba"),
        "d1_to_d1": _add_projection(net, topology.d1_ids, topology.d1_ids, 0.15, rng, "gaba"),
        "d2_to_d2": _add_projection(net, topology.d2_ids, topology.d2_ids, 0.15, rng, "gaba"),
    }
    return net, topology, created


def _warmup(net: MolecularNeuralNetwork, config: BenchmarkConfig) -> None:
    cue_union = list(net._molecular_neurons.keys())
    for step_i in range(config.warmup_steps):
        if step_i % 4 == 0:
            for nid in cue_union[::2]:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 8.0
        net.step(config.dt)


def _task_synapse_keys(topology: TaskTopology) -> set[tuple[int, int]]:
    return {
        (pre, post)
        for pre_ids, post_ids in (
            (topology.cue_green_ids, topology.d1_ids),
            (topology.cue_green_ids, topology.d2_ids),
            (topology.cue_red_ids, topology.d1_ids),
            (topology.cue_red_ids, topology.d2_ids),
        )
        for pre in pre_ids
        for post in post_ids
    }


def _freeze_non_task_plasticity(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
    task_stdp_factor: float,
) -> None:
    task_keys = _task_synapse_keys(topology)
    for key, syn in net._molecular_synapses.items():
        syn._plasticity_factor = task_stdp_factor if key in task_keys else 0.0


def _stimulate_population(
    net: MolecularNeuralNetwork,
    neuron_ids: tuple[int, ...],
    intensity: float,
) -> None:
    for nid in neuron_ids:
        net._external_currents[nid] = net._external_currents.get(nid, 0.0) + intensity


def set_nmda_conductance_scale(net: MolecularNeuralNetwork, scale: float) -> dict[int, float]:
    """Set NMDA conductance scale on all neurons that expose the channel."""
    saved: dict[int, float] = {}
    for nid, neuron in net._molecular_neurons.items():
        channel = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
        if channel is None:
            continue
        saved[nid] = channel.conductance_scale
        channel.conductance_scale = scale
    return saved


def restore_nmda_conductance_scale(
    net: MolecularNeuralNetwork,
    saved: dict[int, float],
) -> None:
    """Restore NMDA conductance scale after a temporary ablation."""
    for nid, scale in saved.items():
        neuron = net._molecular_neurons.get(nid)
        if neuron is None:
            continue
        channel = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
        if channel is not None:
            channel.conductance_scale = scale


@contextmanager
def _frozen_plasticity(net: MolecularNeuralNetwork) -> Iterator[None]:
    """Temporarily disable STDP and reward learning for clean probe phases."""
    saved_lr = net.learning_rate
    saved_factors = {
        key: syn._plasticity_factor for key, syn in net._molecular_synapses.items()
    }

    net.learning_rate = 0.0
    for syn in net._molecular_synapses.values():
        syn._plasticity_factor = 0.0

    try:
        yield
    finally:
        net.learning_rate = saved_lr
        for key, factor in saved_factors.items():
            if key in net._molecular_synapses:
                net._molecular_synapses[key]._plasticity_factor = factor


def _population_activity(
    net: MolecularNeuralNetwork,
    neuron_ids: tuple[int, ...],
) -> tuple[float, int]:
    if not neuron_ids:
        return 0.0, 0

    activity_values: list[float] = []
    spikes = 0
    for nid in neuron_ids:
        neuron = net._molecular_neurons.get(nid)
        if neuron is None or not neuron.alive:
            continue
        norm_v = (neuron.membrane_potential + 70.0) / 90.0
        norm_v = max(0.0, min(1.0, norm_v))
        spike_bonus = 0.35 if nid in net.last_fired else 0.0
        activity_values.append(norm_v + spike_bonus)
        if nid in net.last_fired:
            spikes += 1
    if not activity_values:
        return 0.0, 0
    return float(np.mean(activity_values)), spikes


def _cue_feedback(condition: ConditionName, cue: CueName, config: BenchmarkConfig) -> float:
    feedback = config.reward_amount if cue == "green" else -config.reward_amount
    if condition == "no_dopamine":
        return 0.0
    if condition == "reward_shuffle":
        return -feedback
    return feedback


def _apply_task_reward_modulated_plasticity(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
) -> None:
    task_keys = _task_synapse_keys(topology)
    for key, syn in net._molecular_synapses.items():
        if key not in task_keys:
            continue
        syn.apply_reward(
            net.dopamine_level,
            net.learning_rate,
            modulation_factor=net.dopamine_plasticity_factor(key[1]),
        )
    net.dopamine_level *= net.dopamine_decay
    da = net.global_nt_concentrations.get("dopamine", 20.0)
    net.global_nt_concentrations["dopamine"] = 20.0 + (da - 20.0) * 0.95


def _expected_action(cue: CueName) -> str:
    return "Go" if cue == "green" else "NoGo"


def _task_consistent_bias(cue: CueName, action_bias: float) -> float:
    return action_bias if cue == "green" else -action_bias


def _teaching_population(topology: TaskTopology, cue: CueName) -> tuple[int, ...]:
    return topology.d1_ids if cue == "green" else topology.d2_ids


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
        shuffle_within_blocks=config.shuffle_within_blocks,
        block_size=config.trial_balance_block,
    )
    outcomes: list[TrialOutcome] = []

    for trial_index, cue in enumerate(schedule):
        cue_population = (
            topology.cue_green_ids if cue == "green" else topology.cue_red_ids
        )
        d1_activity_trace: list[float] = []
        d2_activity_trace: list[float] = []
        d1_spikes_total = 0
        d2_spikes_total = 0

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
                _stimulate_population(
                    net,
                    _teaching_population(topology, cue),
                    config.teaching_current,
                )
            net.step(config.dt)
            if enable_learning:
                net.update_eligibility_traces(dt=config.dt)
            if step_i >= readout_start:
                d1_activity, d1_spikes = _population_activity(net, topology.d1_ids)
                d2_activity, d2_spikes = _population_activity(net, topology.d2_ids)
                d1_activity_trace.append(d1_activity)
                d2_activity_trace.append(d2_activity)
                d1_spikes_total += d1_spikes
                d2_spikes_total += d2_spikes

        d1_activity = _mean(d1_activity_trace)
        d2_activity = _mean(d2_activity_trace)
        action_bias = d1_activity - d2_activity
        decision = "Go" if action_bias > 0.0 else "NoGo"
        expected = _expected_action(cue)
        correct = decision == expected
        task_feedback = config.reward_amount if cue == "green" else -config.reward_amount
        applied_feedback = 0.0

        if enable_learning:
            applied_feedback = _cue_feedback(condition, cue, config)
            if applied_feedback != 0.0:
                net.release_dopamine(applied_feedback)
                _apply_task_reward_modulated_plasticity(net, topology)
                for _ in range(config.feedback_steps):
                    net.step(config.dt)

        for _ in range(config.inter_trial_steps):
            net.step(config.dt)

        outcomes.append(
            TrialOutcome(
                phase=phase,
                trial_index=trial_index,
                cue=cue,
                expected=expected,
                decision=decision,
                correct=correct,
                d1_activity=d1_activity,
                d2_activity=d2_activity,
                d1_spikes=d1_spikes_total,
                d2_spikes=d2_spikes_total,
                action_bias=action_bias,
                task_consistent_bias=_task_consistent_bias(cue, action_bias),
                task_feedback=task_feedback,
                applied_feedback=applied_feedback,
            )
        )

    return outcomes


def _phase_summary(outcomes: list[TrialOutcome]) -> dict[str, float]:
    if not outcomes:
        return {
            "n_trials": 0.0,
            "accuracy": 0.0,
            "go_rate": 0.0,
            "green_accuracy": 0.0,
            "red_accuracy": 0.0,
            "green_action_bias": 0.0,
            "red_action_bias": 0.0,
            "task_consistent_bias": 0.0,
            "mean_d1_activity": 0.0,
            "mean_d2_activity": 0.0,
            "mean_d1_spikes": 0.0,
            "mean_d2_spikes": 0.0,
        }

    green = [trial for trial in outcomes if trial.cue == "green"]
    red = [trial for trial in outcomes if trial.cue == "red"]

    return {
        "n_trials": float(len(outcomes)),
        "accuracy": _mean(float(trial.correct) for trial in outcomes),
        "go_rate": _mean(float(trial.decision == "Go") for trial in outcomes),
        "green_accuracy": _mean(float(trial.correct) for trial in green),
        "red_accuracy": _mean(float(trial.correct) for trial in red),
        "green_action_bias": _mean(float(trial.action_bias) for trial in green),
        "red_action_bias": _mean(float(trial.action_bias) for trial in red),
        "task_consistent_bias": _mean(
            float(trial.task_consistent_bias) for trial in outcomes
        ),
        "mean_d1_activity": _mean(float(trial.d1_activity) for trial in outcomes),
        "mean_d2_activity": _mean(float(trial.d2_activity) for trial in outcomes),
        "mean_d1_spikes": _mean(float(trial.d1_spikes) for trial in outcomes),
        "mean_d2_spikes": _mean(float(trial.d2_spikes) for trial in outcomes),
    }


def _training_summary(outcomes: list[TrialOutcome]) -> dict[str, float]:
    summary = _phase_summary(outcomes)
    if not outcomes:
        summary["first_block_accuracy"] = 0.0
        summary["last_block_accuracy"] = 0.0
        summary["first_block_bias"] = 0.0
        summary["last_block_bias"] = 0.0
        summary["mean_applied_feedback"] = 0.0
        return summary

    block_size = min(10, len(outcomes))
    summary["first_block_accuracy"] = _mean(
        float(trial.correct) for trial in outcomes[:block_size]
    )
    summary["last_block_accuracy"] = _mean(
        float(trial.correct) for trial in outcomes[-block_size:]
    )
    summary["first_block_bias"] = _mean(
        float(trial.task_consistent_bias) for trial in outcomes[:block_size]
    )
    summary["last_block_bias"] = _mean(
        float(trial.task_consistent_bias) for trial in outcomes[-block_size:]
    )
    summary["mean_applied_feedback"] = _mean(
        float(trial.applied_feedback) for trial in outcomes
    )
    return summary


def _pathway_groups(topology: TaskTopology) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    return {
        "green_to_d1": (topology.cue_green_ids, topology.d1_ids),
        "green_to_d2": (topology.cue_green_ids, topology.d2_ids),
        "red_to_d1": (topology.cue_red_ids, topology.d1_ids),
        "red_to_d2": (topology.cue_red_ids, topology.d2_ids),
    }


def _pathway_summary(
    net: MolecularNeuralNetwork,
    pre_ids: tuple[int, ...],
    post_ids: tuple[int, ...],
) -> dict[str, float]:
    pre_set = set(pre_ids)
    post_set = set(post_ids)
    synapses = [
        syn
        for (pre, post), syn in net._molecular_synapses.items()
        if pre in pre_set and post in post_set
    ]
    if not synapses:
        return {
            "n_synapses": 0.0,
            "mean_weight": 0.0,
            "mean_total_receptors": 0.0,
            "mean_ampa_receptors": 0.0,
            "mean_nmda_receptors": 0.0,
            "mean_eligibility_trace": 0.0,
        }

    ampa_counts = [
        float(syn.receptor_count.get(ReceptorType.AMPA, 0))
        for syn in synapses
    ]
    nmda_counts = [
        float(syn.receptor_count.get(ReceptorType.NMDA, 0))
        for syn in synapses
    ]
    total_counts = [float(sum(syn.receptor_count.values())) for syn in synapses]

    return {
        "n_synapses": float(len(synapses)),
        "mean_weight": _mean(float(syn.weight) for syn in synapses),
        "mean_total_receptors": _mean(total_counts),
        "mean_ampa_receptors": _mean(ampa_counts),
        "mean_nmda_receptors": _mean(nmda_counts),
        "mean_eligibility_trace": _mean(float(syn.eligibility_trace) for syn in synapses),
    }


def _collect_pathways(
    net: MolecularNeuralNetwork,
    topology: TaskTopology,
) -> dict[str, dict[str, float]]:
    return {
        name: _pathway_summary(net, pre_ids, post_ids)
        for name, (pre_ids, post_ids) in _pathway_groups(topology).items()
    }


def _delta_summary(
    initial: dict[str, dict[str, float]],
    final: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    all_group_names = set(initial) | set(final)
    for group_name in sorted(all_group_names):
        initial_metrics = initial.get(group_name, {})
        final_metrics = final.get(group_name, {})
        metric_names = set(initial_metrics) | set(final_metrics)
        result[group_name] = {
            metric_name: float(
                final_metrics.get(metric_name, 0.0)
                - initial_metrics.get(metric_name, 0.0)
            )
            for metric_name in sorted(metric_names)
        }
    return result


def _mechanistic_totals(delta: dict[str, dict[str, float]]) -> dict[str, float]:
    aligned = ("green_to_d1", "red_to_d2")
    cross = ("green_to_d2", "red_to_d1")
    return {
        "aligned_weight_delta": _mean(
            delta.get(name, {}).get("mean_weight", 0.0) for name in aligned
        ),
        "cross_weight_delta": _mean(
            delta.get(name, {}).get("mean_weight", 0.0) for name in cross
        ),
        "aligned_ampa_delta": _mean(
            delta.get(name, {}).get("mean_ampa_receptors", 0.0) for name in aligned
        ),
        "cross_ampa_delta": _mean(
            delta.get(name, {}).get("mean_ampa_receptors", 0.0) for name in cross
        ),
    }


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
    _freeze_non_task_plasticity(net, topology, config.task_stdp_factor)
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
        "cue_green_count": len(topology.cue_green_ids),
        "cue_red_count": len(topology.cue_red_ids),
        "d1_count": len(topology.d1_ids),
        "d2_count": len(topology.d2_ids),
        "created_green_to_d1": created["green_to_d1"],
        "created_green_to_d2": created["green_to_d2"],
        "created_red_to_d1": created["red_to_d1"],
        "created_red_to_d2": created["red_to_d2"],
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
        task_consistent_bias_improvement=float(
            post_summary["task_consistent_bias"] - pre_summary["task_consistent_bias"]
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
        bias_improvements = [record.task_consistent_bias_improvement for record in group]
        acc_improvements = [record.accuracy_improvement for record in group]
        pre_bias = [record.pre_training_summary["task_consistent_bias"] for record in group]
        post_bias = [record.post_training_summary["task_consistent_bias"] for record in group]
        pre_acc = [record.pre_training_summary["accuracy"] for record in group]
        post_acc = [record.post_training_summary["accuracy"] for record in group]
        summary[condition] = {
            "n_runs": float(len(group)),
            "mean_pre_training_bias": _mean(pre_bias),
            "mean_post_training_bias": _mean(post_bias),
            "mean_bias_improvement": _mean(bias_improvements),
            "std_bias_improvement": _sample_std(bias_improvements),
            "mean_pre_training_accuracy": _mean(pre_acc),
            "mean_post_training_accuracy": _mean(post_acc),
            "mean_accuracy_improvement": _mean(acc_improvements),
            "std_accuracy_improvement": _sample_std(acc_improvements),
            "mean_training_last_block_accuracy": _mean(
                record.training_summary["last_block_accuracy"] for record in group
            ),
            "mean_training_last_block_bias": _mean(
                record.training_summary["last_block_bias"] for record in group
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
        bias_diffs: list[float] = []
        accuracy_diffs: list[float] = []
        aligned_weight_diffs: list[float] = []
        for pair_records in by_pair.values():
            anchor = pair_records.get(anchor_condition)
            other = pair_records.get(condition)
            if anchor is None or other is None:
                continue
            bias_diffs.append(
                anchor.task_consistent_bias_improvement
                - other.task_consistent_bias_improvement
            )
            accuracy_diffs.append(anchor.accuracy_improvement - other.accuracy_improvement)
            aligned_weight_diffs.append(
                anchor.mechanistic_summary["delta_summary"]["aligned_weight_delta"]
                - other.mechanistic_summary["delta_summary"]["aligned_weight_delta"]
            )
        contrasts[condition] = {
            "n_pairs": len(bias_diffs),
            "mean_bias_improvement_difference": _mean(bias_diffs),
            "raw_bias_improvement_differences": bias_diffs,
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
    output_path = Path(output_dir) / f"corticostriatal_action_bias_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "corticostriatal_action_bias_benchmark",
        "config": asdict(config),
        "conditions": list(conditions),
        "n_seeds": n_seeds,
        "master_seed": master_seed,
        "summary": summary,
        "paired_differences": paired,
        "results": [asdict(record) for record in records],
        "metadata": {
            "git_commit": _git_commit(),
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
