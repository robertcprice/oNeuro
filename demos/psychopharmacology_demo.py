#!/usr/bin/env python3
"""Computational psychopharmacology demo — THE FLAGSHIP.

Demonstrates drug effects on a molecular neural network:
  1. Baseline activity measurement under pulsed stimulation
  2. Apply drug → observe changes in spike rate, NT concentrations
  3. Remove drug → observe recovery
  4. Dose-response curves for selected drugs

Tests: Fluoxetine (SSRI), Caffeine, Ketamine, Diazepam

Usage:
    cd oNeuro && PYTHONPATH=src python3 demos/psychopharmacology_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from oneuro.molecular import MolecularNeuralNetwork, DRUG_LIBRARY
from oneuro.molecular.neuron import NeuronArchetype
from oneuro.molecular.synapse import MolecularSynapse


STIM_INTENSITY = 15.0
STIM_RADIUS = 6.0
STIM_POS = (5.0, 5.0, 2.5)
MEASURE_STEPS = 1000  # 100ms measurement window for finer resolution


def measure_spikes(net, steps=500, dt=0.1):
    """Run network with pulsed stimulation (5ms on / 5ms off), return spike count.

    Pulsed stimulation avoids depolarization block (a real biophysics effect
    where sustained current inactivates Na+ channels).
    """
    start = net.spike_count
    for i in range(steps):
        if (i % 100) < 50:  # 5ms on, 5ms off at dt=0.1
            net.stimulate(STIM_POS, intensity=STIM_INTENSITY, radius=STIM_RADIUS)
        net.step(dt=dt)
    return net.spike_count - start


def warmup(net, steps=500):
    """Clear initial transients so baseline measurement is stable."""
    for i in range(steps):
        if (i % 100) < 50:
            net.stimulate(STIM_POS, intensity=STIM_INTENSITY, radius=STIM_RADIUS)
        net.step(dt=0.1)


def phase_bar(label, spikes, max_spikes=100):
    """ASCII bar for spike counts."""
    width = 40
    norm = min(1.0, spikes / max(1, max_spikes))
    filled = int(norm * width)
    return f"  {label:12s} {spikes:4d} spikes {'█' * filled}{'░' * (width - filled)}"


def run_drug_test(drug_name, dose_mg, steps=MEASURE_STEPS, dt=0.1, seed=42):
    """Run drug vs control on identical fresh networks.

    Uses paired comparison: both networks get same seed and warmup,
    but only one receives the drug. This eliminates STDP drift confound.
    """
    drug_cls = DRUG_LIBRARY[drug_name]
    drug = drug_cls(dose_mg=dose_mg)

    print(f"\n{'=' * 60}")
    print(f"  {drug.name} ({drug.drug_class}) — {dose_mg} mg")
    cmax = drug.plasma_concentration(drug.tmax_hours)
    emax = drug.effect_strength(cmax)
    print(f"  EC50={drug.EC50_nM} nM, t½={drug.half_life_hours}h, "
          f"Cmax={cmax:.0f} nM")
    print(f"  Effect at Cmax: {emax:.1%}")
    print(f"{'=' * 60}")

    # Control network (no drug)
    ctrl = make_network(seed=seed)
    warmup(ctrl, steps=2000)
    control_spikes = measure_spikes(ctrl, steps=steps, dt=dt)
    control_nt = dict(ctrl.global_nt_concentrations)

    # Drug network (identical setup)
    net = make_network(seed=seed)
    warmup(net, steps=2000)
    drug.apply(net)
    on_drug = measure_spikes(net, steps=steps, dt=dt)
    drug_nt = dict(net.global_nt_concentrations)

    # Recovery: remove drug and measure next window
    drug.remove(net)
    recovery = measure_spikes(net, steps=steps, dt=dt)

    max_bar = max(control_spikes, on_drug, recovery, 1)
    print(phase_bar("Control", control_spikes, max_spikes=max_bar))
    print(phase_bar("On-drug", on_drug, max_spikes=max_bar))
    print(phase_bar("Recovery", recovery, max_spikes=max_bar))

    change = (on_drug - control_spikes) / max(1, control_spikes) * 100
    print(f"\n  Spike rate change vs control: {change:+.1f}%")

    # NT changes
    changed = {}
    for nt in control_nt:
        old, new = control_nt[nt], drug_nt[nt]
        if old > 0 and abs(new - old) / old * 100 > 1.0:
            changed[nt] = (new - old) / old * 100
    if changed:
        print("  NT concentration changes:")
        for nt, pct in sorted(changed.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {nt:18s}: {pct:+.1f}%")

    return {"control": control_spikes, "on_drug": on_drug, "recovery": recovery,
            "spike_change_pct": change}


def dose_response(drug_name, doses, steps=MEASURE_STEPS, dt=0.1):
    """Run same drug at multiple doses — paired comparison vs control."""
    drug_cls = DRUG_LIBRARY[drug_name]

    # Single control network for all doses (same seed, same time window)
    ctrl = make_network(seed=42)
    warmup(ctrl, steps=2000)
    control_spikes = measure_spikes(ctrl, steps=steps, dt=dt)

    print(f"\n{'─' * 65}")
    print(f"  Dose-Response: {drug_cls.__name__} (control={control_spikes} spikes)")
    print(f"{'─' * 65}")
    print(f"  {'Dose':>8s}  {'Effect':>6s}  {'Control':>8s}  {'On-drug':>8s}  {'Change':>8s}  Bar")

    for dose in doses:
        drug = drug_cls(dose_mg=dose)
        net = make_network(seed=42)
        warmup(net, steps=2000)
        drug.apply(net)
        on_drug = measure_spikes(net, steps=steps, dt=dt)
        drug.remove(net)

        eff = drug.effect_strength(drug.plasma_concentration(drug.tmax_hours))
        change = (on_drug - control_spikes) / max(1, control_spikes) * 100
        bar_len = int(min(40, max(0, (change + 100) / 5)))
        bar = "█" * bar_len
        print(f"  {dose:7.1f}mg  {eff:5.1%}  {control_spikes:>8d}  {on_drug:>8d}  {change:>+7.1f}%  {bar}")


def make_network(seed=42):
    """Build a mixed-archetype network with diverse synapse types.

    Includes GABAergic synapses from interneurons, glutamatergic recurrent
    connections, serotonin, dopamine, and ACh synapses — all drug targets.
    """
    np.random.seed(seed)
    net = MolecularNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=15,
        energy_supply=3.0,
    )
    # Add interneurons (have GABA-A channels) and medium spiny (have D1/D2)
    for _ in range(6):
        p = np.random.uniform([0, 0, 0], [10, 10, 5])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.INTERNEURON)
    for _ in range(4):
        p = np.random.uniform([0, 0, 0], [10, 10, 5])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MEDIUM_SPINY)
    for _ in range(3):
        p = np.random.uniform([0, 0, 0], [10, 10, 5])
        net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MOTONEURON)

    ids = list(net._molecular_neurons.keys())

    # GABAergic synapses from interneurons → pyramidal (primary inhibitory circuit)
    interneuron_ids = [nid for nid in ids
                       if net._molecular_neurons[nid].archetype == NeuronArchetype.INTERNEURON]
    pyramidal_ids = [nid for nid in ids
                     if net._molecular_neurons[nid].archetype == NeuronArchetype.PYRAMIDAL]
    for i, inh_id in enumerate(interneuron_ids):
        if pyramidal_ids:
            target = pyramidal_ids[i % len(pyramidal_ids)]
            net._molecular_synapses[(inh_id, target)] = MolecularSynapse(
                pre_neuron=inh_id, post_neuron=target, nt_name="gaba")

    # Serotonin, dopamine, ACh synapses
    for i in range(0, min(6, len(ids) - 1), 2):
        net._molecular_synapses[(ids[i], ids[i + 1])] = MolecularSynapse(
            pre_neuron=ids[i], post_neuron=ids[i + 1], nt_name="serotonin")
    for i in range(1, min(7, len(ids) - 1), 2):
        net._molecular_synapses[(ids[i], ids[i + 1])] = MolecularSynapse(
            pre_neuron=ids[i], post_neuron=ids[i + 1], nt_name="dopamine")
    for i in range(0, min(4, len(ids) - 1)):
        net._molecular_synapses[(ids[i], ids[(i + 2) % len(ids)])] = MolecularSynapse(
            pre_neuron=ids[i], post_neuron=ids[(i + 2) % len(ids)], nt_name="acetylcholine")
    return net


def main():
    print("=" * 60)
    print("  oNeuro Computational Psychopharmacology Demo")
    print("=" * 60)

    # Show network info
    net = make_network()
    archetypes = {}
    for n in net._molecular_neurons.values():
        archetypes[n.archetype.name] = archetypes.get(n.archetype.name, 0) + 1
    synapse_types = {}
    for s in net._molecular_synapses.values():
        synapse_types[s.nt_name] = synapse_types.get(s.nt_name, 0) + 1
    print(f"\n  Neurons: {len(net._molecular_neurons)} ({archetypes})")
    print(f"  Synapses: {len(net._molecular_synapses)} ({synapse_types})")
    print(f"  Stimulation: I={STIM_INTENSITY} µA/cm², r={STIM_RADIUS}mm, pulsed 5ms on/off")

    # Verify baseline stability first
    print(f"\n--- Baseline Stability Check ---")
    warmup(net, steps=2000)
    s1 = measure_spikes(net, steps=MEASURE_STEPS)
    s2 = measure_spikes(net, steps=MEASURE_STEPS)
    s3 = measure_spikes(net, steps=MEASURE_STEPS)
    print(f"  Trial 1: {s1} spikes")
    print(f"  Trial 2: {s2} spikes")
    print(f"  Trial 3: {s3} spikes")
    cv = np.std([s1, s2, s3]) / max(1, np.mean([s1, s2, s3]))
    print(f"  CV = {cv:.2f} (lower = more stable)")

    # Test 4 drugs — each on a fresh network
    results = {}
    for drug_name in ["fluoxetine", "caffeine", "ketamine", "diazepam"]:
        r = run_drug_test(drug_name, DRUG_LIBRARY[drug_name]().dose_mg)
        results[drug_name] = r

    # Dose-response curves
    dose_response("caffeine", [25, 50, 100, 200, 400])
    dose_response("diazepam", [0.1, 0.3, 0.5, 1, 5])

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'Drug':<15s} {'Control':>8s} {'On-drug':>8s} {'Recovery':>8s} {'Change':>8s}")
    print(f"  {'-' * 47}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['control']:>8d} {r['on_drug']:>8d} "
              f"{r['recovery']:>8d} {r['spike_change_pct']:>+7.1f}%")


if __name__ == "__main__":
    main()
