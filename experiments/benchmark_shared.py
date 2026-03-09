"""Shared utilities for small benchmark-safe microcircuit experiments."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from oneuro.molecular.ion_channels import IonChannelType
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.receptors import ReceptorType

DEFAULT_PHASE_SEED_OFFSETS = {
    "pre_training": 1_001,
    "training": 2_003,
    "post_training": 3_007,
    "topology": 4_009,
}


def git_commit(repo_root: Path) -> str | None:
    """Return the current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def mean(values: Iterable[float]) -> float:
    """Compute a stable float mean."""
    data = list(values)
    if not data:
        return 0.0
    return float(np.mean(np.asarray(data, dtype=np.float64)))


def sample_std(values: Iterable[float]) -> float:
    """Compute a sample standard deviation."""
    data = list(values)
    if len(data) < 2:
        return 0.0
    return float(np.std(np.asarray(data, dtype=np.float64), ddof=1))


def brain_seed(master_seed: int, pair_index: int) -> int:
    """Derive a reproducible per-pair seed."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, pair_index]))
    return int(rng.integers(0, 2**32, dtype=np.uint32))


def phase_seed(
    seed: int,
    phase: str,
    phase_offsets: dict[str, int] | None = None,
) -> int:
    """Derive a reproducible per-phase seed."""
    offsets = DEFAULT_PHASE_SEED_OFFSETS if phase_offsets is None else phase_offsets
    return int(seed + offsets[phase])


def cue_schedule(
    n_trials: int,
    seed: int,
    labels: Sequence[str],
    *,
    shuffle_within_blocks: bool,
    block_size: int,
) -> list[str]:
    """Create a balanced cue schedule for a small benchmark."""
    if n_trials <= 0:
        return []

    labels = tuple(labels)
    if len(labels) != 2:
        raise ValueError("cue_schedule currently supports exactly two labels")

    if not shuffle_within_blocks:
        return [labels[idx % len(labels)] for idx in range(n_trials)]

    rng = np.random.default_rng(seed)
    schedule: list[str] = []
    remaining = n_trials
    balanced_block = max(2, block_size)

    while remaining > 0:
        current_block = min(remaining, balanced_block)
        n_first = current_block // 2
        n_second = current_block - n_first
        block = [labels[0]] * n_first + [labels[1]] * n_second
        rng.shuffle(block)
        schedule.extend(block)
        remaining -= current_block

    return schedule[:n_trials]


def add_projection(
    net: MolecularNeuralNetwork,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    probability: float,
    rng: np.random.Generator,
    nt_name: str = "glutamate",
) -> int:
    """Randomly create synapses between two neuron groups."""
    created = 0
    for src in source_ids:
        for tgt in target_ids:
            if src == tgt:
                continue
            if rng.random() < probability:
                net.create_synapse(src, tgt, nt_name)
                created += 1
    return created


def warmup_network(
    net: MolecularNeuralNetwork,
    *,
    warmup_steps: int,
    dt: float,
    neuron_ids: Sequence[int] | None = None,
    pulse_every: int = 4,
    pulse_current: float = 8.0,
) -> None:
    """Warm a benchmark-safe network into a non-degenerate activity regime."""
    ids = list(net._molecular_neurons.keys() if neuron_ids is None else neuron_ids)
    for step_i in range(warmup_steps):
        if step_i % pulse_every == 0:
            for nid in ids[::2]:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + pulse_current
        net.step(dt)


def task_synapse_keys(
    pathway_groups: dict[str, tuple[Sequence[int], Sequence[int]]],
) -> set[tuple[int, int]]:
    """Return the set of synapse ids considered part of the task surface."""
    return {
        (pre, post)
        for pre_ids, post_ids in pathway_groups.values()
        for pre in pre_ids
        for post in post_ids
    }


def freeze_non_task_plasticity(
    net: MolecularNeuralNetwork,
    pathway_groups: dict[str, tuple[Sequence[int], Sequence[int]]],
    task_stdp_factor: float,
) -> set[tuple[int, int]]:
    """Freeze all non-task synapses and return the task-key set."""
    task_keys = task_synapse_keys(pathway_groups)
    for key, syn in net._molecular_synapses.items():
        syn._plasticity_factor = task_stdp_factor if key in task_keys else 0.0
    return task_keys


def stimulate_population(
    net: MolecularNeuralNetwork,
    neuron_ids: Sequence[int],
    intensity: float,
) -> None:
    """Inject external current into a neuron group."""
    for nid in neuron_ids:
        net._external_currents[nid] = net._external_currents.get(nid, 0.0) + intensity


def set_nmda_conductance_scale(
    net: MolecularNeuralNetwork,
    scale: float,
) -> dict[int, float]:
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
def frozen_plasticity(net: MolecularNeuralNetwork) -> Iterator[None]:
    """Temporarily disable STDP and reward learning for probe phases."""
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


def population_activity(
    net: MolecularNeuralNetwork,
    neuron_ids: Sequence[int],
) -> tuple[float, int]:
    """Compute mean normalized population activity and spike count."""
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


def apply_task_reward_modulated_plasticity(
    net: MolecularNeuralNetwork,
    task_keys: set[tuple[int, int]],
) -> None:
    """Apply reward-modulated plasticity only to task synapses."""
    for key, syn in net._molecular_synapses.items():
        if key not in task_keys:
            continue
        syn.apply_reward(
            net.dopamine_level,
            net.learning_rate,
            modulation_factor=net.dopamine_plasticity_factor(key[1]),
        )
    net.dopamine_level *= net.dopamine_decay
    dopamine = net.global_nt_concentrations.get("dopamine", 20.0)
    net.global_nt_concentrations["dopamine"] = 20.0 + (dopamine - 20.0) * 0.95


def pathway_summary(
    net: MolecularNeuralNetwork,
    pre_ids: Sequence[int],
    post_ids: Sequence[int],
) -> dict[str, float]:
    """Summarize one pathway's synaptic state."""
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

    ampa_counts = [float(syn.receptor_count.get(ReceptorType.AMPA, 0)) for syn in synapses]
    nmda_counts = [float(syn.receptor_count.get(ReceptorType.NMDA, 0)) for syn in synapses]
    total_counts = [float(sum(syn.receptor_count.values())) for syn in synapses]

    return {
        "n_synapses": float(len(synapses)),
        "mean_weight": mean(float(syn.weight) for syn in synapses),
        "mean_total_receptors": mean(total_counts),
        "mean_ampa_receptors": mean(ampa_counts),
        "mean_nmda_receptors": mean(nmda_counts),
        "mean_eligibility_trace": mean(float(syn.eligibility_trace) for syn in synapses),
    }


def collect_pathways(
    net: MolecularNeuralNetwork,
    pathway_groups: dict[str, tuple[Sequence[int], Sequence[int]]],
) -> dict[str, dict[str, float]]:
    """Collect summaries for all named pathways."""
    return {
        name: pathway_summary(net, pre_ids, post_ids)
        for name, (pre_ids, post_ids) in pathway_groups.items()
    }


def delta_summary(
    initial: dict[str, dict[str, float]],
    final: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Subtract pathway summary dictionaries."""
    result: dict[str, dict[str, float]] = {}
    all_group_names = set(initial) | set(final)
    for group_name in sorted(all_group_names):
        initial_metrics = initial.get(group_name, {})
        final_metrics = final.get(group_name, {})
        metric_names = set(initial_metrics) | set(final_metrics)
        result[group_name] = {
            metric_name: float(
                final_metrics.get(metric_name, 0.0) - initial_metrics.get(metric_name, 0.0)
            )
            for metric_name in sorted(metric_names)
        }
    return result


def mechanistic_totals(
    delta: dict[str, dict[str, float]],
    *,
    aligned_groups: Sequence[str],
    cross_groups: Sequence[str],
) -> dict[str, float]:
    """Reduce pathway deltas into aligned-versus-cross totals."""
    return {
        "aligned_weight_delta": mean(
            delta.get(name, {}).get("mean_weight", 0.0) for name in aligned_groups
        ),
        "cross_weight_delta": mean(
            delta.get(name, {}).get("mean_weight", 0.0) for name in cross_groups
        ),
        "aligned_ampa_delta": mean(
            delta.get(name, {}).get("mean_ampa_receptors", 0.0) for name in aligned_groups
        ),
        "cross_ampa_delta": mean(
            delta.get(name, {}).get("mean_ampa_receptors", 0.0) for name in cross_groups
        ),
    }
