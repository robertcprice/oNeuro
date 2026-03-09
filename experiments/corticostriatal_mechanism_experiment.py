#!/usr/bin/env python3
# ruff: noqa: E402
"""Mechanistic corticostriatal timing assay for oNeuro.

This experiment isolates one claim that the simulator can realistically support:

1. Causal versus anti-causal pairing should change glutamatergic synapses differently.
2. NMDA blockade should suppress causal potentiation.
3. Dopamine should bias plasticity in opposite directions for D1-like and D2-like MSNs.

The protocol is intentionally low-level and benchmark-safe. It uses a tiny
two-neuron preparation with explicit pair timing, dopamine pulse timing, and
pathway-specific reward modulation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from oneuro.molecular.ion_channels import IonChannelType
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import NeuronArchetype
from oneuro.molecular.receptors import ReceptorType

DEFAULT_PAIRING_TIMINGS_MS: tuple[float, ...] = (10.0, -10.0)
DEFAULT_DOPAMINE_DELAYS_MS: tuple[float, ...] = (0.0, 20.0, 80.0)


@dataclass(slots=True, frozen=True)
class ProtocolConfig:
    """Numerical settings for the pair protocol."""

    dt_ms: float = 1.0
    pairings: int = 40
    inter_pair_interval_ms: float = 40.0
    dopamine_pulse_ms: float = 200.0
    dopamine_pulse_nM: float = 5000.0
    dopamine_reward_amount: float = 4.0
    learning_rate: float = 0.25
    replicates: int = 8
    pairing_timings_ms: tuple[float, ...] = DEFAULT_PAIRING_TIMINGS_MS
    dopamine_delays_ms: tuple[float, ...] = DEFAULT_DOPAMINE_DELAYS_MS
    master_seed: int = 2026

    def normalized(self) -> "ProtocolConfig":
        if self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")
        if self.pairings <= 0:
            raise ValueError("pairings must be positive")
        if self.inter_pair_interval_ms < 0.0:
            raise ValueError("inter_pair_interval_ms must be non-negative")
        if self.dopamine_pulse_ms <= 0.0:
            raise ValueError("dopamine_pulse_ms must be positive")
        if self.dopamine_pulse_nM <= 0.0:
            raise ValueError("dopamine_pulse_nM must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.replicates <= 0:
            raise ValueError("replicates must be positive")
        return self


@dataclass(slots=True, frozen=True)
class ProtocolVariant:
    """A mechanistic contrast applied on top of the same base pair protocol."""

    name: str
    reward_enabled: bool
    nmda_scale: float


@dataclass(slots=True)
class ProtocolResult:
    """Serializable output for one condition replicate."""

    condition: str
    variant: str
    postsynaptic_archetype: str
    pair_timing_ms: float
    dopamine_delay_ms: float
    replicate_index: int
    seed: int
    initial_weight: float
    final_weight: float
    delta_weight: float
    initial_ampa_receptors: int
    final_ampa_receptors: int
    delta_ampa_receptors: int
    initial_nmda_receptors: int
    final_nmda_receptors: int
    initial_strength: float
    final_strength: float
    final_eligibility_trace: float
    modulation_factor: float
    nmda_scale: float
    baseline_pka: float
    peak_pka: float
    trough_pka: float
    wall_time_s: float


VARIANTS: tuple[ProtocolVariant, ...] = (
    ProtocolVariant(name="rewarded", reward_enabled=True, nmda_scale=1.0),
    ProtocolVariant(name="no_dopamine", reward_enabled=False, nmda_scale=1.0),
    ProtocolVariant(name="nmda_block", reward_enabled=True, nmda_scale=0.05),
)


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


def _build_pair_network(
    postsynaptic_archetype: NeuronArchetype,
    seed: int,
) -> tuple[MolecularNeuralNetwork, int, int]:
    rng = np.random.default_rng(seed)
    net = MolecularNeuralNetwork(
        initial_neurons=0,
        size=(3.0, 3.0, 3.0),
        enable_advanced_neurons=True,
        benchmark_safe_mode=True,
    )
    pre_id = net.create_neuron_at(0.5, 1.5, 1.5, archetype=NeuronArchetype.PYRAMIDAL)
    post_id = net.create_neuron_at(2.0, 1.5, 1.5, archetype=postsynaptic_archetype)
    net.create_synapse(pre_id, post_id, "glutamate")
    syn = net._molecular_synapses[(pre_id, post_id)]
    syn._postsynaptic_receptor_count[ReceptorType.AMPA] = max(
        20,
        syn._postsynaptic_receptor_count.get(ReceptorType.AMPA, 50)
        + int(rng.integers(-4, 5)),
    )
    syn._postsynaptic_receptor_count[ReceptorType.NMDA] = max(
        8,
        syn._postsynaptic_receptor_count.get(ReceptorType.NMDA, 20)
        + int(rng.integers(-2, 3)),
    )
    syn.strength = float(np.clip(1.0 + rng.normal(0.0, 0.03), 0.85, 1.0))
    return net, pre_id, post_id


def _step_quiet_state(post, syn, dt_ms: float) -> None:
    post.membrane.step(dt_ms, nt_concentrations={}, external_current=0.0)
    syn.update_eligibility(0.0, 0.0, dt_ms)
    syn.update(syn.age + dt_ms, dt_ms)


def _advance_quiet(post, syn, duration_ms: float, dt_ms: float) -> None:
    steps = max(0, int(round(duration_ms / dt_ms)))
    for _ in range(steps):
        _step_quiet_state(post, syn, dt_ms)


def _pre_event(post, syn, time_ms: float, dt_ms: float) -> None:
    post.membrane.step(
        dt_ms,
        nt_concentrations={"glutamate": 3000.0},
        external_current=5.0,
    )
    syn._camkii_level = post.membrane.camkii_activation
    syn.update_stdp(pre_fired=True, post_fired=False, time=time_ms, dt=dt_ms)
    syn.update_eligibility(1.0, 0.0, dt_ms)
    syn.update(time_ms, dt_ms)


def _post_event(post, syn, time_ms: float, dt_ms: float) -> None:
    post.membrane.step(
        dt_ms,
        nt_concentrations={"glutamate": 1000.0},
        external_current=60.0,
    )
    syn._camkii_level = post.membrane.camkii_activation
    syn.update_stdp(pre_fired=False, post_fired=True, time=time_ms, dt=dt_ms)
    syn.update_eligibility(0.0, 1.0, dt_ms)
    syn.update(time_ms, dt_ms)


def _deliver_dopamine_pulse(
    post,
    syn,
    config: ProtocolConfig,
    modulation_sign: float,
) -> tuple[float, float, float, float]:
    baseline_pka = 0.0
    peak_pka = 0.0
    trough_pka = 0.0
    sms = post.second_messenger_system
    if sms is not None:
        baseline_pka = sms.pka_activity
        peak_pka = baseline_pka
        trough_pka = baseline_pka

    steps = max(1, int(round(config.dopamine_pulse_ms / config.dt_ms)))
    current_time = syn.age
    for _ in range(steps):
        post.membrane.step(
            config.dt_ms,
            nt_concentrations={"dopamine": config.dopamine_pulse_nM},
            external_current=0.0,
        )
        syn.update_eligibility(0.0, 0.0, config.dt_ms)
        syn.update(current_time, config.dt_ms)
        current_time += config.dt_ms
        if sms is not None:
            peak_pka = max(peak_pka, sms.pka_activity)
            trough_pka = min(trough_pka, sms.pka_activity)

    pka_delta = max(peak_pka - baseline_pka, baseline_pka - trough_pka)
    modulation_factor = modulation_sign * (1.0 + 5.0 * pka_delta)
    syn.apply_reward(
        config.dopamine_reward_amount,
        learning_rate=config.learning_rate,
        modulation_factor=modulation_factor,
    )
    return baseline_pka, peak_pka, trough_pka, modulation_factor


def _condition_name(
    postsynaptic_archetype: NeuronArchetype,
    pair_timing_ms: float,
    dopamine_delay_ms: float,
    variant: ProtocolVariant,
) -> str:
    timing_name = "causal" if pair_timing_ms > 0 else "anti_causal"
    return (
        f"{postsynaptic_archetype.value}"
        f"__{timing_name}"
        f"__delay_{int(dopamine_delay_ms)}ms"
        f"__{variant.name}"
    )


def run_protocol(
    *,
    postsynaptic_archetype: NeuronArchetype,
    pair_timing_ms: float,
    dopamine_delay_ms: float,
    variant: ProtocolVariant,
    config: ProtocolConfig,
    replicate_index: int,
) -> ProtocolResult:
    start = time.perf_counter()
    seed_seq = np.random.SeedSequence(
        [
            config.master_seed,
            replicate_index,
            1 if postsynaptic_archetype == NeuronArchetype.D1_MSN else 2,
        ]
    )
    seed = int(seed_seq.generate_state(1, dtype=np.uint32)[0])
    net, pre_id, post_id = _build_pair_network(postsynaptic_archetype, seed)
    syn = net._molecular_synapses[(pre_id, post_id)]
    post = net._molecular_neurons[post_id]
    nmda_channel = post.membrane.channels.get_channel(IonChannelType.NMDA)
    if nmda_channel is not None:
        nmda_channel.conductance_scale = variant.nmda_scale
    syn._nmda_scale = variant.nmda_scale

    modulation_sign = net.dopamine_plasticity_factor(post_id)
    initial_weight = syn.weight
    initial_ampa = syn.receptor_count.get(ReceptorType.AMPA, 0)
    initial_nmda = syn.receptor_count.get(ReceptorType.NMDA, 0)
    initial_strength = syn.strength

    baseline_pka = 0.0
    peak_pka = 0.0
    trough_pka = 0.0
    modulation_factor = 0.0

    time_ms = 0.0
    for _ in range(config.pairings):
        if pair_timing_ms >= 0.0:
            _pre_event(post, syn, time_ms, config.dt_ms)
            time_ms += config.dt_ms
            _advance_quiet(post, syn, pair_timing_ms, config.dt_ms)
            time_ms += max(0.0, pair_timing_ms)
            _post_event(post, syn, time_ms, config.dt_ms)
            time_ms += config.dt_ms
        else:
            _post_event(post, syn, time_ms, config.dt_ms)
            time_ms += config.dt_ms
            _advance_quiet(post, syn, abs(pair_timing_ms), config.dt_ms)
            time_ms += abs(pair_timing_ms)
            _pre_event(post, syn, time_ms, config.dt_ms)
            time_ms += config.dt_ms

        _advance_quiet(post, syn, dopamine_delay_ms, config.dt_ms)
        time_ms += max(0.0, dopamine_delay_ms)

        if variant.reward_enabled:
            baseline_pka, peak_pka, trough_pka, modulation_factor = _deliver_dopamine_pulse(
                post,
                syn,
                config,
                modulation_sign=modulation_sign,
            )
            time_ms += config.dopamine_pulse_ms

        remaining_interval = max(
            0.0,
            config.inter_pair_interval_ms - abs(pair_timing_ms) - dopamine_delay_ms,
        )
        _advance_quiet(post, syn, remaining_interval, config.dt_ms)
        time_ms += remaining_interval

    return ProtocolResult(
        condition=_condition_name(
            postsynaptic_archetype,
            pair_timing_ms,
            dopamine_delay_ms,
            variant,
        ),
        variant=variant.name,
        postsynaptic_archetype=postsynaptic_archetype.value,
        pair_timing_ms=pair_timing_ms,
        dopamine_delay_ms=dopamine_delay_ms,
        replicate_index=replicate_index,
        seed=seed,
        initial_weight=initial_weight,
        final_weight=syn.weight,
        delta_weight=syn.weight - initial_weight,
        initial_ampa_receptors=initial_ampa,
        final_ampa_receptors=syn.receptor_count.get(ReceptorType.AMPA, 0),
        delta_ampa_receptors=syn.receptor_count.get(ReceptorType.AMPA, 0) - initial_ampa,
        initial_nmda_receptors=initial_nmda,
        final_nmda_receptors=syn.receptor_count.get(ReceptorType.NMDA, 0),
        initial_strength=initial_strength,
        final_strength=syn.strength,
        final_eligibility_trace=syn.eligibility_trace,
        modulation_factor=modulation_factor,
        nmda_scale=variant.nmda_scale,
        baseline_pka=baseline_pka,
        peak_pka=peak_pka,
        trough_pka=trough_pka,
        wall_time_s=time.perf_counter() - start,
    )


def _summarize(results: list[ProtocolResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[ProtocolResult]] = {}
    for record in results:
        grouped.setdefault(record.condition, []).append(record)

    summary: dict[str, dict[str, float | int]] = {}
    for condition, records in grouped.items():
        summary[condition] = {
            "n": len(records),
            "mean_delta_weight": _mean(r.delta_weight for r in records),
            "std_delta_weight": _sample_std(r.delta_weight for r in records),
            "mean_delta_ampa_receptors": _mean(r.delta_ampa_receptors for r in records),
            "std_delta_ampa_receptors": _sample_std(r.delta_ampa_receptors for r in records),
            "mean_final_strength": _mean(r.final_strength for r in records),
            "mean_final_eligibility_trace": _mean(r.final_eligibility_trace for r in records),
            "mean_peak_pka": _mean(r.peak_pka for r in records),
            "mean_trough_pka": _mean(r.trough_pka for r in records),
            "mean_modulation_factor": _mean(r.modulation_factor for r in records),
        }
    return summary


def _contrast(
    summary: dict[str, dict[str, float | int]],
    left: str,
    right: str,
) -> dict[str, float]:
    left_summary = summary.get(left)
    right_summary = summary.get(right)
    if left_summary is None or right_summary is None:
        return {}
    return {
        "delta_weight_diff": float(left_summary["mean_delta_weight"])
        - float(right_summary["mean_delta_weight"]),
        "delta_ampa_diff": float(left_summary["mean_delta_ampa_receptors"]) - float(
            right_summary["mean_delta_ampa_receptors"]
        ),
        "peak_pka_diff": float(left_summary["mean_peak_pka"])
        - float(right_summary["mean_peak_pka"]),
    }


def _selected_contrasts(
    summary: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    contrasts: dict[str, dict[str, float]] = {}
    contrasts["d1_rewarded_vs_no_dopamine"] = _contrast(
        summary,
        "d1_msn__causal__delay_0ms__rewarded",
        "d1_msn__causal__delay_0ms__no_dopamine",
    )
    contrasts["d1_rewarded_vs_nmda_block"] = _contrast(
        summary,
        "d1_msn__causal__delay_0ms__rewarded",
        "d1_msn__causal__delay_0ms__nmda_block",
    )
    contrasts["d1_vs_d2_rewarded"] = _contrast(
        summary,
        "d1_msn__causal__delay_0ms__rewarded",
        "d2_msn__causal__delay_0ms__rewarded",
    )
    contrasts["d1_immediate_vs_delayed"] = _contrast(
        summary,
        "d1_msn__causal__delay_0ms__rewarded",
        "d1_msn__causal__delay_80ms__rewarded",
    )
    return contrasts


def run_experiment(
    config: ProtocolConfig,
    *,
    output_dir: str | Path = REPO_ROOT / "experiments" / "results",
) -> dict[str, Any]:
    config = config.normalized()
    results: list[ProtocolResult] = []
    archetypes = (NeuronArchetype.D1_MSN, NeuronArchetype.D2_MSN)

    for postsynaptic_archetype in archetypes:
        for pair_timing_ms in config.pairing_timings_ms:
            for dopamine_delay_ms in config.dopamine_delays_ms:
                for variant in VARIANTS:
                    for replicate_index in range(config.replicates):
                        results.append(
                            run_protocol(
                                postsynaptic_archetype=postsynaptic_archetype,
                                pair_timing_ms=pair_timing_ms,
                                dopamine_delay_ms=dopamine_delay_ms,
                                variant=variant,
                                config=config,
                                replicate_index=replicate_index,
                            )
                        )

    summary = _summarize(results)
    selected_contrasts = _selected_contrasts(summary)
    timestamp = int(time.time())
    output_path = Path(output_dir) / f"corticostriatal_mechanism_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "corticostriatal_mechanism_experiment",
        "config": asdict(config),
        "results": [asdict(result) for result in results],
        "summary": summary,
        "selected_contrasts": selected_contrasts,
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


def _parse_tuple_floats(values: list[str] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return default
    return tuple(float(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairings", type=int, default=40)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--inter-pair-interval-ms", type=float, default=40.0)
    parser.add_argument("--dopamine-pulse-ms", type=float, default=200.0)
    parser.add_argument("--dopamine-pulse-nm", type=float, default=5000.0)
    parser.add_argument("--dopamine-reward-amount", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--pairing-timings-ms", nargs="*", default=None)
    parser.add_argument("--dopamine-delays-ms", nargs="*", default=None)
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "results"),
    )
    args = parser.parse_args()

    config = ProtocolConfig(
        dt_ms=args.dt_ms,
        pairings=args.pairings,
        inter_pair_interval_ms=args.inter_pair_interval_ms,
        dopamine_pulse_ms=args.dopamine_pulse_ms,
        dopamine_pulse_nM=args.dopamine_pulse_nm,
        dopamine_reward_amount=args.dopamine_reward_amount,
        learning_rate=args.learning_rate,
        replicates=args.replicates,
        pairing_timings_ms=_parse_tuple_floats(
            args.pairing_timings_ms,
            DEFAULT_PAIRING_TIMINGS_MS,
        ),
        dopamine_delays_ms=_parse_tuple_floats(
            args.dopamine_delays_ms,
            DEFAULT_DOPAMINE_DELAYS_MS,
        ),
        master_seed=args.master_seed,
    )
    payload = run_experiment(config, output_dir=args.output_dir)
    print(json.dumps(payload["selected_contrasts"], indent=2))
    print(f"Saved results to {payload['result_path']}")


if __name__ == "__main__":
    main()
