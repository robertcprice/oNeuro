#!/usr/bin/env python3
"""Molecular tissue demo — build a 30-neuron network, stimulate, show activity.

Demonstrates:
  - Network construction with molecular neurons
  - External stimulation causing action potentials
  - Voltage traces and ASCII visualisation
  - Synaptic transmission via real NT release/binding

Usage:
    cd oNeuro && PYTHONPATH=src python3 demos/molecular_tissue_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from oneuro.molecular import MolecularNeuralNetwork


def voltage_bar(v: float, width: int = 30) -> str:
    """ASCII bar for membrane voltage [-80, +40] mV."""
    norm = (v + 80.0) / 120.0
    norm = max(0.0, min(1.0, norm))
    filled = int(norm * width)
    return "█" * filled + "░" * (width - filled)


def main():
    np.random.seed(42)

    print("=" * 60)
    print("  oNeuro Molecular Tissue Demo")
    print("=" * 60)
    print()

    # Build network
    net = MolecularNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=30,
        energy_supply=3.0,
    )
    print(f"Network: {len(net._molecular_neurons)} neurons, "
          f"{len(net._molecular_synapses)} synapses")
    print()

    # Warmup: reach steady state before measuring
    print("--- Warmup (200 ms) ---")
    for i in range(2000):
        if (i % 100) < 50:
            net.stimulate((5.0, 5.0, 2.5), intensity=20.0, radius=5.0)
        net.step(dt=0.1)
    print(f"  Done ({net.spike_count} spikes during warmup)")
    print()

    # Phase 1: Baseline (pulsed stimulation, same as measurement)
    print("--- Phase 1: Baseline (100 ms, pulsed stimulation 20 µA/cm²) ---")
    pre_baseline = net.spike_count
    for i in range(1000):
        if (i % 100) < 50:
            net.stimulate((5.0, 5.0, 2.5), intensity=20.0, radius=5.0)
        net.step(dt=0.1)
    baseline_spikes = net.spike_count - pre_baseline
    print(f"  Spikes: {baseline_spikes}")
    print()

    # Phase 2: Strong stimulation
    print("--- Phase 2: Strong pulsed stimulation (100 ms, 30 µA/cm²) ---")
    pre_stim = net.spike_count
    for i in range(1000):
        if (i % 100) < 50:
            net.stimulate((5.0, 5.0, 2.5), intensity=30.0, radius=5.0)
        net.step(dt=0.1)
    stim_spikes = net.spike_count - pre_stim
    print(f"  Spikes during strong stimulation: {stim_spikes}")
    print(f"  Spike rate increase: {stim_spikes / max(1, baseline_spikes):.1f}x")
    print()

    # Phase 3: Recovery (no stimulation)
    print("--- Phase 3: Recovery (100 ms, no stimulation) ---")
    pre_recovery = net.spike_count
    for _ in range(1000):
        net.step(dt=0.1)
    recovery_spikes = net.spike_count - pre_recovery
    print(f"  Spikes during recovery: {recovery_spikes}")
    print()

    # Voltage snapshot
    print("--- Current Voltage Snapshot ---")
    neurons_sorted = sorted(
        net._molecular_neurons.values(),
        key=lambda n: n.membrane_potential,
        reverse=True,
    )
    for n in neurons_sorted[:10]:
        v = n.membrane_potential
        state = "FIRE" if n.membrane.fired else ("act" if n.is_active else "rest")
        print(f"  N{n.id:3d} [{state:4s}] {v:+6.1f} mV {voltage_bar(v)}")
    if len(neurons_sorted) > 10:
        print(f"  ... and {len(neurons_sorted) - 10} more neurons")
    print()

    # ASCII network visualisation
    print("--- Network Topology ---")
    print(net.visualize_ascii())
    print()

    # NT concentration summary
    print("--- Global NT Concentrations ---")
    for nt, conc in sorted(net.global_nt_concentrations.items()):
        print(f"  {nt:18s}: {conc:8.1f} nM")
    print()

    print(f"Total simulation time: {net.time:.1f} ms")
    print(f"Total spikes: {net.spike_count}")
    print(f"Neurogenesis events: {net.neurogenesis_events}")
    print(f"Pruning events: {net.pruning_events}")


if __name__ == "__main__":
    main()
