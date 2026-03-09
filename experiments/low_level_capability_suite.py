#!/usr/bin/env python3
# ruff: noqa: E402
"""Execute the full low-level capability ladder for oNeuro.

This suite turns the ladder in docs/low_level_capability_tests.md into one
reproducible experiment artifact with matched controls where possible.

It does not assume that every rung is positive. Each benchmark is marked as:
- positive: the intended effect survives its control
- ambiguous: the surface moves, but the control is weak or missing
- negative: the intended effect is not supported in the current repo
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from experiments.benchmark_shared import (
    add_projection as add_projection_shared,
)
from experiments.benchmark_shared import (
    git_commit as shared_git_commit,
)
from experiments.benchmark_shared import (
    mean as _mean,
)
from experiments.benchmark_shared import (
    population_activity as shared_population_activity,
)
from experiments.corticostriatal_action_bias_benchmark import (
    BenchmarkConfig as ActionBiasBenchmarkConfig,
)
from experiments.corticostriatal_action_bias_benchmark import (
    run_experiment as run_action_bias_experiment,
)
from experiments.corticostriatal_mechanism_experiment import (
    ProtocolConfig as MechanismProtocolConfig,
)
from experiments.corticostriatal_mechanism_experiment import (
    run_experiment as run_mechanism_experiment,
)
from experiments.stimulus_discrimination_benchmark import (
    BenchmarkConfig as DiscriminationBenchmarkConfig,
)
from experiments.stimulus_discrimination_benchmark import (
    brain_seed as discrimination_brain_seed,
)
from experiments.stimulus_discrimination_benchmark import (
    build_microcircuit as build_discrimination_microcircuit,
)
from experiments.stimulus_discrimination_benchmark import (
    freeze_non_task_plasticity as freeze_discrimination_non_task_plasticity,
)
from experiments.stimulus_discrimination_benchmark import (
    phase_seed as discrimination_phase_seed,
)
from experiments.stimulus_discrimination_benchmark import (
    population_activity as discrimination_population_activity,
)
from experiments.stimulus_discrimination_benchmark import (
    run_experiment as run_discrimination_experiment,
)
from experiments.stimulus_discrimination_benchmark import (
    run_trials as run_discrimination_trials,
)
from experiments.stimulus_discrimination_benchmark import (
    warmup_microcircuit as warmup_discrimination_microcircuit,
)
from oneuro.molecular.brain_regions import RegionalBrain
from oneuro.molecular.ion_channels import IonChannelType
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import NeuronArchetype
from oneuro.molecular.pharmacology import DRUG_LIBRARY
from oneuro.molecular.synapse import MolecularSynapse


@dataclass(slots=True, frozen=True)
class SuiteConfig:
    """Numerical settings for the low-level capability suite."""

    master_seed: int = 2026
    excitability_currents: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    excitability_steps: int = 1000
    drug_n_seeds: int = 6
    drug_warmup_steps: int = 1000
    drug_measure_steps: int = 500
    conditioning_n_seeds: int = 4
    conditioning_acquisition_trials: int = 20
    conditioning_extinction_trials: int = 10
    conditioning_rest_steps: int = 300
    mechanism_replicates: int = 6
    discrimination_n_seeds: int = 10
    action_bias_n_seeds: int = 6
    working_memory_n_seeds: int = 10
    working_memory_short_delay_steps: int = 40
    working_memory_long_delay_steps: int = 120
    pattern_completion_n_seeds: int = 6

    def normalized(self) -> "SuiteConfig":
        if not self.excitability_currents:
            raise ValueError("excitability_currents must be non-empty")
        if any(current <= 0.0 for current in self.excitability_currents):
            raise ValueError("excitability currents must be positive")
        positive_int_fields = (
            "excitability_steps",
            "drug_n_seeds",
            "drug_warmup_steps",
            "drug_measure_steps",
            "conditioning_n_seeds",
            "conditioning_acquisition_trials",
            "conditioning_extinction_trials",
            "conditioning_rest_steps",
            "mechanism_replicates",
            "discrimination_n_seeds",
            "action_bias_n_seeds",
            "working_memory_n_seeds",
            "working_memory_short_delay_steps",
            "working_memory_long_delay_steps",
            "pattern_completion_n_seeds",
        )
        for field_name in positive_int_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        return self


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = 2026,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> list[float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return [0.0, 0.0]
    if data.size == 1:
        value = float(data[0])
        return [value, value]

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.choice(data, size=data.size, replace=True)
        boots[idx] = np.mean(sample)
    lower = float(np.quantile(boots, alpha / 2.0))
    upper = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return [lower, upper]


def _paired_summary(values: Iterable[float], *, seed: int = 2026) -> dict[str, Any]:
    raw = [float(value) for value in values]
    return {
        "mean_difference": _mean(raw),
        "raw_differences": raw,
        "bootstrap_ci_95": _bootstrap_mean_ci(raw, seed=seed),
    }


def _status(positive: bool, ambiguous: bool = False) -> str:
    if positive:
        return "positive"
    if ambiguous:
        return "ambiguous"
    return "negative"


def _run_single_neuron_injection(
    current: float,
    *,
    nav_add: int = 0,
    kv_add: int = 0,
    steps: int = 1000,
    dt: float = 0.1,
) -> dict[str, float]:
    net = MolecularNeuralNetwork(
        initial_neurons=0,
        size=(2.0, 2.0, 2.0),
        benchmark_safe_mode=True,
    )
    nid = net.create_neuron_at(1.0, 1.0, 1.0, archetype=NeuronArchetype.PYRAMIDAL)
    neuron = net._molecular_neurons[nid]
    if nav_add > 0:
        neuron.membrane.channels.add_channel(IonChannelType.Na_v, count=nav_add)
    if kv_add > 0:
        neuron.membrane.channels.add_channel(IonChannelType.K_v, count=kv_add)

    spikes = 0
    voltage_trace: list[float] = []
    for _ in range(steps):
        net._external_currents[nid] = net._external_currents.get(nid, 0.0) + current
        net.step(dt)
        spikes += int(nid in net.last_fired)
        voltage_trace.append(float(neuron.membrane_potential))
    return {
        "spikes": float(spikes),
        "mean_voltage": _mean(voltage_trace),
    }


def run_excitability_benchmark(config: SuiteConfig) -> dict[str, Any]:
    """Single-neuron F-I comparison under channel perturbations."""
    variants = {
        "baseline": {"nav_add": 0, "kv_add": 0},
        "nav_up": {"nav_add": 2, "kv_add": 0},
        "kv_up": {"nav_add": 0, "kv_add": 2},
    }

    curves: dict[str, list[dict[str, float]]] = {}
    for label, kwargs in variants.items():
        curve = []
        for current in config.excitability_currents:
            metrics = _run_single_neuron_injection(
                current,
                nav_add=kwargs["nav_add"],
                kv_add=kwargs["kv_add"],
                steps=config.excitability_steps,
            )
            curve.append({"current": float(current), **metrics})
        curves[label] = curve

    auc = {
        label: float(sum(point["spikes"] for point in curve))
        for label, curve in curves.items()
    }
    selected_current = float(config.excitability_currents[len(config.excitability_currents) // 2])
    selected_index = len(config.excitability_currents) // 2
    selected_spikes = {
        label: float(curves[label][selected_index]["spikes"])
        for label in curves
    }

    positive = (
        auc["nav_up"] > auc["baseline"] > auc["kv_up"]
        and selected_spikes["nav_up"] > selected_spikes["baseline"] >= selected_spikes["kv_up"]
    )
    return {
        "name": "single_neuron_excitability",
        "status": _status(positive),
        "summary": {
            "auc_spikes": auc,
            "selected_current": selected_current,
            "selected_current_spikes": selected_spikes,
        },
        "raw": curves,
    }


def _make_drug_microcircuit(seed: int) -> MolecularNeuralNetwork:
    np.random.seed(seed)
    net = MolecularNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=15,
        energy_supply=3.0,
        benchmark_safe_mode=True,
    )
    for _ in range(6):
        p = np.random.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 5.0])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.INTERNEURON)
    for _ in range(4):
        p = np.random.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 5.0])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MEDIUM_SPINY)
    for _ in range(3):
        p = np.random.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 5.0])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MOTONEURON)

    ids = list(net._molecular_neurons.keys())
    interneuron_ids = [
        nid
        for nid in ids
        if net._molecular_neurons[nid].archetype == NeuronArchetype.INTERNEURON
    ]
    pyramidal_ids = [
        nid
        for nid in ids
        if net._molecular_neurons[nid].archetype == NeuronArchetype.PYRAMIDAL
    ]

    for idx, inh_id in enumerate(interneuron_ids):
        if not pyramidal_ids:
            break
        target = pyramidal_ids[idx % len(pyramidal_ids)]
        net._molecular_synapses[(inh_id, target)] = MolecularSynapse(
            pre_neuron=inh_id,
            post_neuron=target,
            nt_name="gaba",
        )

    for idx in range(0, min(6, len(ids) - 1), 2):
        net._molecular_synapses[(ids[idx], ids[idx + 1])] = MolecularSynapse(
            pre_neuron=ids[idx],
            post_neuron=ids[idx + 1],
            nt_name="serotonin",
        )
    for idx in range(1, min(7, len(ids) - 1), 2):
        net._molecular_synapses[(ids[idx], ids[idx + 1])] = MolecularSynapse(
            pre_neuron=ids[idx],
            post_neuron=ids[idx + 1],
            nt_name="dopamine",
        )
    for idx in range(0, min(4, len(ids) - 1)):
        net._molecular_synapses[(ids[idx], ids[(idx + 2) % len(ids)])] = MolecularSynapse(
            pre_neuron=ids[idx],
            post_neuron=ids[(idx + 2) % len(ids)],
            nt_name="acetylcholine",
        )
    return net


def _warm_drug_network(net: MolecularNeuralNetwork, steps: int) -> None:
    for step_i in range(steps):
        if (step_i % 100) < 50:
            net.stimulate((5.0, 5.0, 2.5), intensity=15.0, radius=6.0)
        net.step(dt=0.1)


def _measure_drug_network(net: MolecularNeuralNetwork, steps: int) -> int:
    start = net.spike_count
    for step_i in range(steps):
        if (step_i % 100) < 50:
            net.stimulate((5.0, 5.0, 2.5), intensity=15.0, radius=6.0)
        net.step(dt=0.1)
    return int(net.spike_count - start)


def run_drug_response_benchmark(config: SuiteConfig) -> dict[str, Any]:
    """Matched microcircuit pharmacology benchmark."""
    drug_specs = {
        "diazepam": 5.0,
        "caffeine": 400.0,
    }
    per_drug: dict[str, dict[str, Any]] = {}

    for drug_name, dose_mg in drug_specs.items():
        deltas: list[float] = []
        records: list[dict[str, Any]] = []
        for seed in range(config.drug_n_seeds):
            control = _make_drug_microcircuit(seed)
            _warm_drug_network(control, config.drug_warmup_steps)
            control_spikes = _measure_drug_network(control, config.drug_measure_steps)

            treated = _make_drug_microcircuit(seed)
            _warm_drug_network(treated, config.drug_warmup_steps)
            drug = DRUG_LIBRARY[drug_name](dose_mg=dose_mg)
            drug.apply(treated)
            treated_spikes = _measure_drug_network(treated, config.drug_measure_steps)
            drug.remove(treated)
            recovery_spikes = _measure_drug_network(treated, config.drug_measure_steps)

            delta = float(treated_spikes - control_spikes)
            deltas.append(delta)
            records.append(
                {
                    "seed": seed,
                    "control_spikes": float(control_spikes),
                    "treated_spikes": float(treated_spikes),
                    "recovery_spikes": float(recovery_spikes),
                    "delta_spikes": delta,
                }
            )

        per_drug[drug_name] = {
            "dose_mg": dose_mg,
            "mean_delta_spikes": _mean(deltas),
            "bootstrap_ci_95": _bootstrap_mean_ci(
                deltas,
                seed=config.master_seed + (1 if drug_name == "diazepam" else 2),
            ),
            "raw_deltas": deltas,
            "records": records,
        }

    diazepam_ci = per_drug["diazepam"]["bootstrap_ci_95"]
    caffeine_ci = per_drug["caffeine"]["bootstrap_ci_95"]
    positive = diazepam_ci[1] < 0.0 and caffeine_ci[0] > 0.0

    return {
        "name": "drug_response_matched_microcircuits",
        "status": _status(positive),
        "summary": {
            "diazepam": {
                "mean_delta_spikes": per_drug["diazepam"]["mean_delta_spikes"],
                "bootstrap_ci_95": diazepam_ci,
            },
            "caffeine": {
                "mean_delta_spikes": per_drug["caffeine"]["mean_delta_spikes"],
                "bootstrap_ci_95": caffeine_ci,
            },
        },
        "raw": per_drug,
    }


def run_nmda_plasticity_benchmark(
    config: SuiteConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the dedicated mechanistic corticostriatal assay."""
    payload = run_mechanism_experiment(
        MechanismProtocolConfig(
            replicates=config.mechanism_replicates,
            master_seed=config.master_seed,
        ),
        output_dir=output_dir,
    )
    contrasts = payload["selected_contrasts"]
    positive = (
        contrasts["d1_rewarded_vs_no_dopamine"]["delta_weight_diff"] > 0.0
        and contrasts["d1_rewarded_vs_nmda_block"]["delta_weight_diff"] > 0.0
        and contrasts["d1_vs_d2_rewarded"]["delta_weight_diff"] > 0.0
    )
    return {
        "name": "nmda_dependent_plasticity",
        "status": _status(positive),
        "summary": contrasts,
        "raw": {"result_path": payload["result_path"]},
    }


def run_discrimination_benchmark(
    config: SuiteConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the strict stimulus discrimination benchmark."""
    payload = run_discrimination_experiment(
        n_seeds=config.discrimination_n_seeds,
        workers=min(4, config.discrimination_n_seeds),
        master_seed=config.master_seed,
        config=DiscriminationBenchmarkConfig(),
        output_dir=output_dir,
    )
    summary = payload["summary"]
    paired = payload["paired_differences"]
    positive = (
        summary["full_learning"]["mean_accuracy_improvement"] > 0.0
        and paired["no_learning"]["mean_accuracy_improvement_difference"] > 0.0
        and paired["label_shuffle"]["mean_accuracy_improvement_difference"] > 0.0
    )
    return {
        "name": "stimulus_discrimination",
        "status": _status(positive),
        "summary": {
            "condition_summary": summary,
            "paired_differences": paired,
        },
        "raw": {"result_path": payload["result_path"]},
    }


def run_go_no_go_benchmark(
    config: SuiteConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the rebuilt corticostriatal Go/No-Go surface."""
    payload = run_action_bias_experiment(
        n_seeds=config.action_bias_n_seeds,
        workers=min(4, config.action_bias_n_seeds),
        master_seed=config.master_seed,
        config=ActionBiasBenchmarkConfig(),
        output_dir=output_dir,
    )
    summary = payload["summary"]
    paired = payload["paired_differences"]
    positive = (
        summary["full_learning"]["mean_bias_improvement"] > 0.0
        and paired["no_dopamine"]["mean_bias_improvement_difference"] > 0.0
        and paired["reward_shuffle"]["mean_bias_improvement_difference"] > 0.0
    )
    return {
        "name": "go_no_go_learning",
        "status": _status(positive),
        "summary": {
            "condition_summary": summary,
            "paired_differences": paired,
        },
        "raw": {"result_path": payload["result_path"]},
    }


def _conditioning_warmup(brain: RegionalBrain, steps: int = 200) -> None:
    for step_i in range(steps):
        if step_i % 4 == 0:
            brain.stimulate_thalamus(intensity=15.0)
        brain.step(0.1)


def _conditioning_cs_response(
    net: MolecularNeuralNetwork,
    relay_ids: list[int],
    cortex_ids: set[int],
    *,
    cs_intensity: float = 35.0,
    steps: int = 50,
) -> int:
    total = 0
    for step_i in range(steps):
        if step_i % 2 == 0 and step_i < 24:
            for nid in relay_ids:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + cs_intensity
        net.step(0.1)
        total += len(net.last_fired & cortex_ids)
    return int(total)


def _conditioning_paired_trial(
    net: MolecularNeuralNetwork,
    relay_ids: list[int],
    l23_ids: list[int],
    cortex_ids: set[int],
    *,
    cs_intensity: float = 35.0,
    us_intensity: float = 28.0,
    paired: bool = True,
) -> int:
    total = 0
    for step_i in range(50):
        if step_i % 2 == 0 and step_i < 24:
            for nid in relay_ids:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + cs_intensity
        if paired and step_i == 24:
            net.release_dopamine(2.0)
            for nid in l23_ids:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + us_intensity
        if paired and step_i % 2 == 0 and 24 <= step_i < 40:
            for nid in l23_ids[: len(l23_ids) // 2]:
                net._external_currents[nid] = (
                    net._external_currents.get(nid, 0.0) + us_intensity * 0.7
                )
        net.step(0.1)
        total += len(net.last_fired & cortex_ids)
    if paired:
        net.apply_reward_modulated_plasticity()
        net.update_eligibility_traces(dt=1.0)
    return int(total)


def _run_conditioning_seed(
    seed: int,
    *,
    acquisition_trials: int,
    extinction_trials: int,
    rest_steps: int,
    paired: bool,
) -> dict[str, float]:
    brain = RegionalBrain.minimal(seed=seed)
    net = brain.network
    _conditioning_warmup(brain)
    for _ in range(100):
        net.step(0.1)

    relay_ids = brain.thalamus.get_ids("relay")
    l23_ids = brain.cortex.get_ids("L2/3")
    cortex_ids = set(brain.cortex.neuron_ids)

    baseline = _conditioning_cs_response(net, relay_ids, cortex_ids)
    acquisition_trace = [
        _conditioning_paired_trial(net, relay_ids, l23_ids, cortex_ids, paired=paired)
        for _ in range(acquisition_trials)
    ]
    post_acquisition = _conditioning_cs_response(net, relay_ids, cortex_ids)

    extinction_trace: list[int] = []
    for _ in range(extinction_trials):
        extinction_trace.append(_conditioning_cs_response(net, relay_ids, cortex_ids))
        net.update_eligibility_traces(dt=1.0)
    post_extinction = _conditioning_cs_response(net, relay_ids, cortex_ids)

    for _ in range(rest_steps):
        net.step(0.1)
    recovery = _conditioning_cs_response(net, relay_ids, cortex_ids)

    return {
        "baseline_cs_response": float(baseline),
        "mean_late_acquisition_response": _mean(acquisition_trace[-5:]),
        "post_acquisition_response": float(post_acquisition),
        "post_extinction_response": float(post_extinction),
        "recovery_response": float(recovery),
        "recovery_minus_extinction": float(recovery - post_extinction),
    }


def run_conditioning_benchmark(config: SuiteConfig) -> dict[str, Any]:
    """Run a strict but still small classical conditioning check."""
    paired_records = [
        _run_conditioning_seed(
            seed=config.master_seed + seed,
            acquisition_trials=config.conditioning_acquisition_trials,
            extinction_trials=config.conditioning_extinction_trials,
            rest_steps=config.conditioning_rest_steps,
            paired=True,
        )
        for seed in range(config.conditioning_n_seeds)
    ]
    cs_only_records = [
        _run_conditioning_seed(
            seed=config.master_seed + seed,
            acquisition_trials=config.conditioning_acquisition_trials,
            extinction_trials=config.conditioning_extinction_trials,
            rest_steps=config.conditioning_rest_steps,
            paired=False,
        )
        for seed in range(config.conditioning_n_seeds)
    ]

    paired_recovery_minus_ext = [
        record["recovery_minus_extinction"] for record in paired_records
    ]
    control_recovery_minus_ext = [
        record["recovery_minus_extinction"] for record in cs_only_records
    ]
    contrast = [
        paired - control
        for paired, control in zip(paired_recovery_minus_ext, control_recovery_minus_ext)
    ]
    paired_mean = _mean(paired_recovery_minus_ext)
    positive = paired_mean > 0.0 and _mean(contrast) > 0.0
    ambiguous = not positive and paired_mean > 0.0

    return {
        "name": "classical_conditioning",
        "status": _status(positive, ambiguous=ambiguous),
        "summary": {
            "paired_recovery_minus_extinction": _paired_summary(
                paired_recovery_minus_ext,
                seed=config.master_seed + 31,
            ),
            "cs_only_recovery_minus_extinction": _paired_summary(
                control_recovery_minus_ext,
                seed=config.master_seed + 32,
            ),
            "paired_minus_cs_only": _paired_summary(
                contrast,
                seed=config.master_seed + 33,
            ),
        },
        "raw": {
            "paired_records": paired_records,
            "cs_only_records": cs_only_records,
        },
    }


def _memory_population_activity(net: MolecularNeuralNetwork, neuron_ids: list[int]) -> float:
    return shared_population_activity(net, neuron_ids)[0]


def _build_working_memory_microcircuit(
    seed: int,
) -> tuple[MolecularNeuralNetwork, list[int], list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    net = MolecularNeuralNetwork(
        initial_neurons=0,
        size=(8.0, 8.0, 4.0),
        enable_advanced_neurons=True,
        benchmark_safe_mode=True,
        psc_scale=45.0,
    )

    cue_a_ids: list[int] = []
    cue_b_ids: list[int] = []
    memory_a_ids: list[int] = []
    memory_b_ids: list[int] = []

    for _ in range(4):
        cue_a_ids.append(
            net.create_neuron_at(
                1.0 + rng.uniform(),
                2.0 + rng.uniform(),
                1.5,
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )
        cue_b_ids.append(
            net.create_neuron_at(
                1.0 + rng.uniform(),
                5.0 + rng.uniform(),
                1.5,
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )
    for _ in range(6):
        memory_a_ids.append(
            net.create_neuron_at(
                5.0 + rng.uniform(),
                2.0 + rng.uniform(),
                1.5,
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )
        memory_b_ids.append(
            net.create_neuron_at(
                5.0 + rng.uniform(),
                5.0 + rng.uniform(),
                1.5,
                archetype=NeuronArchetype.PYRAMIDAL,
            )
        )

    add_projection_shared(net, cue_a_ids, memory_a_ids, 0.8, np.random.default_rng(seed + 1))
    add_projection_shared(net, cue_b_ids, memory_b_ids, 0.8, np.random.default_rng(seed + 2))
    add_projection_shared(
        net,
        memory_a_ids,
        memory_b_ids,
        0.4,
        np.random.default_rng(seed + 3),
        "gaba",
    )
    add_projection_shared(
        net,
        memory_b_ids,
        memory_a_ids,
        0.4,
        np.random.default_rng(seed + 4),
        "gaba",
    )

    for step_i in range(80):
        for nid in memory_a_ids + memory_b_ids:
            net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 6.0
        if step_i % 4 == 0:
            for nid in cue_a_ids + cue_b_ids:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 6.0
        net.step(0.1)

    return net, cue_a_ids, cue_b_ids, memory_a_ids, memory_b_ids


def _run_working_memory_trial(seed: int, cue_label: str, delay_steps: int) -> dict[str, float]:
    net, cue_a_ids, cue_b_ids, memory_a_ids, memory_b_ids = _build_working_memory_microcircuit(seed)
    active_cue = cue_a_ids if cue_label == "A" else cue_b_ids

    for step_i in range(20):
        if step_i % 2 == 0:
            for nid in active_cue:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 55.0
        for nid in memory_a_ids + memory_b_ids:
            net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 6.0
        net.step(0.1)

    trace: list[tuple[float, float]] = []
    for step_i in range(delay_steps):
        for nid in memory_a_ids + memory_b_ids:
            net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 6.0
        net.step(0.1)
        if step_i >= max(0, delay_steps - 10):
            trace.append(
                (
                    _memory_population_activity(net, memory_a_ids),
                    _memory_population_activity(net, memory_b_ids),
                )
            )

    memory_a = _mean(activity_a for activity_a, _ in trace)
    memory_b = _mean(activity_b for _, activity_b in trace)
    decision = "A" if memory_a > memory_b else "B"
    margin = memory_a - memory_b if cue_label == "A" else memory_b - memory_a
    return {
        "correct": float(decision == cue_label),
        "task_consistent_margin": float(margin),
    }


def run_working_memory_benchmark(config: SuiteConfig) -> dict[str, Any]:
    """Short-delay working-memory assay with long-delay control."""
    short_records: list[dict[str, float]] = []
    long_records: list[dict[str, float]] = []
    for seed in range(config.working_memory_n_seeds):
        for cue_label in ("A", "B"):
            short_records.append(
                _run_working_memory_trial(
                    config.master_seed + seed,
                    cue_label,
                    config.working_memory_short_delay_steps,
                )
            )
            long_records.append(
                _run_working_memory_trial(
                    config.master_seed + seed,
                    cue_label,
                    config.working_memory_long_delay_steps,
                )
            )

    short_acc = [record["correct"] for record in short_records]
    long_acc = [record["correct"] for record in long_records]
    acc_diffs = [short - long for short, long in zip(short_acc, long_acc)]
    positive = _mean(short_acc) > 0.75 and _mean(acc_diffs) > 0.0

    return {
        "name": "short_delay_working_memory",
        "status": _status(positive),
        "summary": {
            "short_delay_accuracy": _paired_summary(short_acc, seed=config.master_seed + 41),
            "long_delay_accuracy": _paired_summary(long_acc, seed=config.master_seed + 42),
            "short_minus_long_accuracy": _paired_summary(acc_diffs, seed=config.master_seed + 43),
            "short_delay_margin": _paired_summary(
                [record["task_consistent_margin"] for record in short_records],
                seed=config.master_seed + 44,
            ),
            "long_delay_margin": _paired_summary(
                [record["task_consistent_margin"] for record in long_records],
                seed=config.master_seed + 45,
            ),
        },
        "raw": {
            "short_records": short_records,
            "long_records": long_records,
        },
    }


def _run_pattern_completion_trial(
    net: MolecularNeuralNetwork,
    topology,
    cue_label: str,
    *,
    fraction: float,
    random_pattern: bool,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    full_cue_ids = list(topology.cue_a_ids if cue_label == "A" else topology.cue_b_ids)
    if random_pattern:
        all_ids = list(topology.cue_a_ids + topology.cue_b_ids)
        selected_ids = [nid for nid in all_ids if rng.random() < fraction]
    else:
        selected_ids = [nid for nid in full_cue_ids if rng.random() < fraction]

    trace: list[tuple[float, float]] = []
    for step_i in range(30 + 20):
        if step_i < 30 and step_i % 2 == 0:
            for nid in selected_ids:
                net._external_currents[nid] = net._external_currents.get(nid, 0.0) + 55.0
        net.step(0.1)
        if step_i >= 10:
            trace.append(
                (
                    discrimination_population_activity(net, topology.output_a_ids)[0],
                    discrimination_population_activity(net, topology.output_b_ids)[0],
                )
            )
    output_a = _mean(value_a for value_a, _ in trace)
    output_b = _mean(value_b for _, value_b in trace)
    decision = "A" if output_a > output_b else "B"
    margin = output_a - output_b if cue_label == "A" else output_b - output_a
    return {
        "correct": float(decision == cue_label),
        "task_consistent_margin": float(margin),
    }


def run_pattern_completion_benchmark(config: SuiteConfig) -> dict[str, Any]:
    """Partial-cue completion benchmark on the trained discrimination circuit."""
    full_records: list[dict[str, float]] = []
    random_records: list[dict[str, float]] = []
    discrimination_config = DiscriminationBenchmarkConfig().normalized()

    for pair_index in range(config.pattern_completion_n_seeds):
        seed = discrimination_brain_seed(config.master_seed, pair_index)
        net, topology, _ = build_discrimination_microcircuit(
            discrimination_config,
            seed=discrimination_phase_seed(seed, "topology"),
        )
        freeze_discrimination_non_task_plasticity(
            net,
            topology,
            discrimination_config.task_stdp_factor,
        )
        warmup_discrimination_microcircuit(net, discrimination_config)
        run_discrimination_trials(
            net,
            topology,
            discrimination_config,
            n_trials=discrimination_config.training_trials,
            phase="training",
            schedule_seed=discrimination_phase_seed(seed, "training"),
            condition="full_learning",
            enable_learning=True,
        )

        for cue_label in ("A", "B"):
            full_records.append(
                _run_pattern_completion_trial(
                    deepcopy(net),
                    topology,
                    cue_label,
                    fraction=0.5,
                    random_pattern=False,
                )
            )
            random_records.append(
                _run_pattern_completion_trial(
                    deepcopy(net),
                    topology,
                    cue_label,
                    fraction=0.5,
                    random_pattern=True,
                )
            )

    full_acc = [record["correct"] for record in full_records]
    random_acc = [record["correct"] for record in random_records]
    diff = [full - random for full, random in zip(full_acc, random_acc)]
    positive = _mean(full_acc) > 0.7 and _mean(diff) > 0.0

    return {
        "name": "pattern_completion_partial_recall",
        "status": _status(positive),
        "summary": {
            "partial_cue_accuracy": _paired_summary(full_acc, seed=config.master_seed + 51),
            "random_cue_accuracy": _paired_summary(random_acc, seed=config.master_seed + 52),
            "partial_minus_random_accuracy": _paired_summary(diff, seed=config.master_seed + 53),
            "partial_cue_margin": _paired_summary(
                [record["task_consistent_margin"] for record in full_records],
                seed=config.master_seed + 54,
            ),
            "random_cue_margin": _paired_summary(
                [record["task_consistent_margin"] for record in random_records],
                seed=config.master_seed + 55,
            ),
        },
        "raw": {
            "partial_records": full_records,
            "random_records": random_records,
        },
    }


def run_suite(
    config: SuiteConfig = SuiteConfig(),
    *,
    output_dir: str | Path = REPO_ROOT / "experiments" / "results",
) -> dict[str, Any]:
    """Run the complete low-level capability ladder."""
    config = config.normalized()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    benchmarks = [
        run_excitability_benchmark(config),
        run_drug_response_benchmark(config),
        run_nmda_plasticity_benchmark(config, output_dir=output_dir),
        run_discrimination_benchmark(config, output_dir=output_dir),
        run_go_no_go_benchmark(config, output_dir=output_dir),
        run_conditioning_benchmark(config),
        run_working_memory_benchmark(config),
        run_pattern_completion_benchmark(config),
    ]

    status_counts: dict[str, int] = {}
    for benchmark in benchmarks:
        status_counts[benchmark["status"]] = status_counts.get(benchmark["status"], 0) + 1

    timestamp = int(time.time())
    output_path = output_dir / f"low_level_capability_suite_{timestamp}.json"
    payload = {
        "experiment": "low_level_capability_suite",
        "config": asdict(config),
        "benchmarks": benchmarks,
        "status_counts": status_counts,
        "metadata": {
            "git_commit": shared_git_commit(REPO_ROOT),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timestamp": timestamp,
            "wall_time_s": float(time.perf_counter() - started_at),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    payload["result_path"] = str(output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument("--drug-n-seeds", type=int, default=6)
    parser.add_argument("--conditioning-n-seeds", type=int, default=4)
    parser.add_argument("--mechanism-replicates", type=int, default=6)
    parser.add_argument("--discrimination-n-seeds", type=int, default=10)
    parser.add_argument("--action-bias-n-seeds", type=int, default=6)
    parser.add_argument("--working-memory-n-seeds", type=int, default=10)
    parser.add_argument("--pattern-completion-n-seeds", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "results"),
    )
    args = parser.parse_args()

    payload = run_suite(
        SuiteConfig(
            master_seed=args.master_seed,
            drug_n_seeds=args.drug_n_seeds,
            conditioning_n_seeds=args.conditioning_n_seeds,
            mechanism_replicates=args.mechanism_replicates,
            discrimination_n_seeds=args.discrimination_n_seeds,
            action_bias_n_seeds=args.action_bias_n_seeds,
            working_memory_n_seeds=args.working_memory_n_seeds,
            pattern_completion_n_seeds=args.pattern_completion_n_seeds,
        ),
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["status_counts"], indent=2))
    print(f"Saved results to {payload['result_path']}")


if __name__ == "__main__":
    main()
