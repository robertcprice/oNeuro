#!/usr/bin/env python3
"""Mega Benchmark: All experiments + consciousness tracking.

Orchestrates training comparisons and adds consciousness measurement
throughout.  Produces a comprehensive pass/fail table.

Sections:
  A. Continual Learning
  B. Damage Recovery
  C. Sleep Consolidation
  D. Sample Efficiency
  E. Consciousness Comparison (dedicated)

Pass/Fail Criteria:
  1. Molecular avg forgetting < organic (< 15%)
  2. Damage recovery > 50% lost performance within 5000 steps
  3. Sleep advantage > 5% accuracy improvement
  4. Molecular Phi > 1.5x minimal Phi
  5. Branching ratio in [0.8, 1.2] (critical range)
  6. Consciousness drops under anesthesia (>50% reduction)

Output: console tables + JSON at /tmp/oneuro_mega_benchmark_results.json

Usage:
    cd oNeuro && python3 experiments/mega_benchmark.py
"""

from __future__ import annotations

import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.consciousness import ConsciousnessMonitor, ConsciousnessMetrics
from oneuro.molecular.ion_channels import IonChannelType

# Import training experiments
from experiments.training_comparison import (
    experiment_continual_learning,
    experiment_damage_recovery,
    experiment_sleep_consolidation,
    experiment_sample_efficiency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_with_consciousness(
    net: MolecularNeuralNetwork,
    steps: int,
    dt: float = 0.1,
    stimulate: bool = True,
    intensity: float = 12.0,
) -> ConsciousnessMetrics:
    """Run network with consciousness monitoring, return final metrics."""
    monitor = ConsciousnessMonitor(net)
    center = np.array(net.size) / 2.0
    # Radius scales with network volume so stimulation reaches most neurons
    radius = max(4.0, min(net.size))

    for i in range(steps):
        if stimulate and i % 2 == 0:
            net.stimulate(center, intensity=intensity, radius=radius)
        net.step(dt)
        monitor.record_step(net.last_fired)

    return monitor.compute_all()


# ---------------------------------------------------------------------------
# Section E: Consciousness Comparison
# ---------------------------------------------------------------------------

def _apply_anesthesia(net: MolecularNeuralNetwork) -> None:
    """Simulate deep general anesthesia (propofol + sevoflurane level).

    Molecular actions modeled:
    1. GABA-A potentiation (8x) — primary inhibitory mechanism
    2. NMDA block (95%) — excitatory suppression
    3. AMPA partial block (60%) — reduces fast excitatory transmission
    4. Na_v block (50%) — reduces overall excitability
    5. K_leak potentiation (2x) — hyperpolarization (K+ leak enhancement)
    6. PSC reduction (90%) — presynaptic release depression
    7. Orch-OR suppression (95%) — microtubule quantum decoherence
    """
    # Store original PSC scale and reduce it (presynaptic release depression)
    net._pre_anesthesia_psc = net.psc_scale
    net.psc_scale *= 0.1  # 90% reduction in synaptic transmission

    for mol_n in net._molecular_neurons.values():
        # 1. Orch-OR suppression (anesthetics disrupt microtubule coherence)
        if mol_n.cytoskeleton is not None:
            mol_n.cytoskeleton.anesthetic_factor = 0.05
        # 2. Enhance GABA-A conductance (8x potentiation — deep sedation)
        gaba = mol_n.membrane.channels.get_channel(IonChannelType.GABA_A)
        if gaba is not None:
            gaba.conductance_scale = 8.0
        # 3. Block NMDA channels (95% block)
        nmda = mol_n.membrane.channels.get_channel(IonChannelType.NMDA)
        if nmda is not None:
            nmda.conductance_scale = 0.05
        # 4. Partially block AMPA (60% block — reduces fast excitation)
        ampa = mol_n.membrane.channels.get_channel(IonChannelType.AMPA)
        if ampa is not None:
            ampa.conductance_scale = 0.4
        # 5. Block Na_v (50% — reduces overall excitability)
        nav = mol_n.membrane.channels.get_channel(IonChannelType.Na_v)
        if nav is not None:
            nav.conductance_scale = 0.5
        # 6. Potentiate K_leak (2x — hyperpolarization)
        k_leak = mol_n.membrane.channels.get_channel(IonChannelType.K_leak)
        if k_leak is not None:
            k_leak.conductance_scale = 2.0


def _remove_anesthesia(net: MolecularNeuralNetwork) -> None:
    """Remove anesthesia effects, restore normal channel conductances."""
    net.psc_scale = getattr(net, '_pre_anesthesia_psc', 30.0)

    for mol_n in net._molecular_neurons.values():
        if mol_n.cytoskeleton is not None:
            mol_n.cytoskeleton.anesthetic_factor = 1.0
        for ct in (
            IonChannelType.GABA_A, IonChannelType.NMDA, IonChannelType.AMPA,
            IonChannelType.Na_v, IonChannelType.K_leak,
        ):
            ch = mol_n.membrane.channels.get_channel(ct)
            if ch is not None:
                ch.conductance_scale = 1.0


def experiment_consciousness_comparison(n_seeds: int = 3) -> dict:
    """Dedicated consciousness comparison: full_brain vs minimal.

    Uses 50 neurons for meaningful IIT Phi and criticality metrics.
    Tests consciousness state changes: Rest → Training → Sleep → Anesthesia.
    Anesthesia models real pharmacology: GABA-A potentiation + NMDA block + Orch-OR suppression.
    """
    print("\n" + "=" * 70)
    print("  Section E: Consciousness Comparison")
    print("=" * 70)

    N_NEURONS = 75  # Need 50+ for meaningful Phi, 75 for robust Orch-OR

    phi_ratios = []
    branching_ratios = []
    anesthesia_reductions = []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...")
        np.random.seed(seed)

        # Full brain (all subsystems)
        full = MolecularNeuralNetwork(
            initial_neurons=N_NEURONS, full_brain=True, size=(12.0, 12.0, 8.0),
        )
        if full._circadian is not None:
            full._circadian.clock.time_scale = 100000.0  # Minimize circadian effects

        # Minimal brain (no subsystems, just HH neurons + synapses)
        np.random.seed(seed)
        minimal = MolecularNeuralNetwork(
            initial_neurons=N_NEURONS, size=(12.0, 12.0, 8.0),
        )

        # Warmup both with stimulation to build spike history
        center_full = np.array(full.size) / 2.0
        center_min = np.array(minimal.size) / 2.0
        for i in range(500):
            if i % 2 == 0:
                full.stimulate(center_full, intensity=20.0, radius=6.0)
                minimal.stimulate(center_min, intensity=20.0, radius=6.0)
            full.step(0.1)
            minimal.step(0.1)

        # Run both with consciousness monitoring (2000 steps)
        print("    Measuring full brain consciousness...", end=" ", flush=True)
        metrics_full = run_with_consciousness(full, steps=2000, intensity=18.0)
        print(f"Phi={metrics_full.phi_approx:.3f} composite={metrics_full.composite:.3f}")

        print("    Measuring minimal brain consciousness...", end=" ", flush=True)
        metrics_min = run_with_consciousness(minimal, steps=2000, intensity=18.0)
        print(f"Phi={metrics_min.phi_approx:.3f} composite={metrics_min.composite:.3f}")

        # Phi ratio
        phi_ratio = (metrics_full.phi_approx / max(0.001, metrics_min.phi_approx))
        phi_ratios.append(phi_ratio)
        branching_ratios.append(metrics_full.branching_ratio)

        # Consciousness state changes on full brain
        print("    Testing consciousness state changes:")

        # Rest (low stimulation)
        rest_metrics = run_with_consciousness(full, steps=500, stimulate=False)
        print(f"      Rest:       composite={rest_metrics.composite:.3f}")

        # Training (high stimulation)
        train_metrics = run_with_consciousness(full, steps=500, stimulate=True, intensity=25.0)
        print(f"      Training:   composite={train_metrics.composite:.3f}")

        # Sleep (no stimulation, circadian drift)
        sleep_metrics = run_with_consciousness(full, steps=500, stimulate=False)
        print(f"      Sleep:      composite={sleep_metrics.composite:.3f}")

        # Anesthesia: full pharmacological model
        _apply_anesthesia(full)
        anesthesia_metrics = run_with_consciousness(full, steps=500, stimulate=True, intensity=18.0)
        print(f"      Anesthesia: composite={anesthesia_metrics.composite:.3f}")
        _remove_anesthesia(full)

        # Anesthesia reduction
        if train_metrics.composite > 0.001:
            reduction = 1.0 - anesthesia_metrics.composite / train_metrics.composite
        else:
            reduction = 0.0
        anesthesia_reductions.append(reduction)

    # Results
    mean_phi_ratio = np.mean(phi_ratios)
    mean_br = np.mean(branching_ratios)
    mean_anest_reduction = np.mean(anesthesia_reductions)

    phi_pass = mean_phi_ratio > 1.5
    br_pass = 0.8 <= mean_br <= 1.2
    anest_pass = mean_anest_reduction > 0.50

    print(f"\n  Phi ratio (full/minimal): {mean_phi_ratio:.2f} {'PASS' if phi_pass else 'FAIL'} (>1.5)")
    print(f"  Branching ratio: {mean_br:.3f} {'PASS' if br_pass else 'FAIL'} ([0.8, 1.2])")
    print(f"  Anesthesia reduction: {mean_anest_reduction:.1%} {'PASS' if anest_pass else 'FAIL'} (>50%)")

    return {
        "name": "consciousness_comparison",
        "phi_ratio": float(mean_phi_ratio),
        "branching_ratio": float(mean_br),
        "anesthesia_reduction": float(mean_anest_reduction),
        "phi_pass": phi_pass,
        "br_pass": br_pass,
        "anest_pass": anest_pass,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_mega_benchmark(n_seeds: int = 3) -> dict:
    """Run all sections of the mega benchmark."""
    print("\n" + "#" * 70)
    print("  oNeuro MEGA BENCHMARK: Training + Consciousness")
    print("#" * 70)

    t0 = time.time()
    all_results = {}

    # Sections A-D: Training comparisons
    print("\n  === TRAINING COMPARISONS ===")
    r_a = experiment_continual_learning(n_seeds=n_seeds)
    all_results["A_continual_learning"] = r_a

    r_b = experiment_damage_recovery(n_seeds=n_seeds)
    all_results["B_damage_recovery"] = r_b

    r_c = experiment_sleep_consolidation(n_seeds=n_seeds)
    all_results["C_sleep_consolidation"] = r_c

    r_d = experiment_sample_efficiency(n_seeds=n_seeds)
    all_results["D_sample_efficiency"] = r_d

    # Section E: Consciousness
    print("\n  === CONSCIOUSNESS MEASUREMENTS ===")
    r_e = experiment_consciousness_comparison(n_seeds=n_seeds)
    all_results["E_consciousness"] = r_e

    elapsed = time.time() - t0

    # ---- Pass/Fail Summary ----
    criteria = [
        ("Molecular forgetting < 15%", r_a.get("passed", False)),
        ("Damage recovery > 50%", r_b.get("passed", False)),
        ("Sleep advantage > 5%", r_c.get("passed", False)),
        ("Molecular Phi > 1.5x minimal", r_e.get("phi_pass", False)),
        ("Branching ratio in [0.8, 1.2]", r_e.get("br_pass", False)),
        ("Anesthesia drops consciousness >50%", r_e.get("anest_pass", False)),
    ]

    passed = sum(1 for _, p in criteria if p)
    total = len(criteria)

    print("\n" + "=" * 70)
    print(f"  MEGA BENCHMARK RESULTS: {passed}/{total} criteria met ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"  {'Criterion':<45s} {'Result':>10s}")
    print(f"  {'-' * 45} {'-' * 10}")
    for name, p in criteria:
        status = "PASS" if p else "FAIL"
        print(f"  {name:<45s} {status:>10s}")
    print()

    overall_pass = passed >= 4  # Need at least 4 of 6
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'} ({passed}/{total}, need 4+)")

    # Save to JSON
    output_path = "/tmp/oneuro_mega_benchmark_results.json"
    json_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "criteria_passed": passed,
        "criteria_total": total,
        "overall_pass": overall_pass,
        "sections": {},
    }

    for name, r in all_results.items():
        # Convert to JSON-serializable
        serializable = {}
        for k, v in r.items():
            if isinstance(v, (int, float, bool, str)):
                serializable[k] = v
            elif isinstance(v, dict):
                serializable[k] = {
                    str(kk): [float(x) for x in vv] if isinstance(vv, list) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                serializable[k] = [float(x) if isinstance(x, (int, float)) else str(x) for x in v]
        json_results["sections"][name] = serializable

    try:
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n  Results saved to {output_path}")
    except (OSError, IOError) as e:
        print(f"\n  Could not save results: {e}")

    return all_results


if __name__ == "__main__":
    run_mega_benchmark(n_seeds=3)
