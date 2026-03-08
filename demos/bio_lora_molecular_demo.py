#!/usr/bin/env python3
"""Bio-LoRA ↔ Molecular Network bridge demo.

Demonstrates:
  1. Resting vs arousal BioState generation (numpy only, no torch needed)
  2. Driving a MolecularNeuralNetwork's global NT concentrations from BioState
  3. Measuring the spike-rate difference under identical pulsed stimulation
  4. Reading network state back as a BioState vector

Usage:
    cd oNeuro && PYTHONPATH=src python3 demos/bio_lora_molecular_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from oneuro.molecular import MolecularNeuralNetwork, NeuroBioState, BioLoRABridge


def make_bio_state(name: str) -> np.ndarray:
    """Generate named BioState vectors (no torch dependency)."""
    state = np.zeros(12)
    if name == "resting":
        state[1] = 0.5   # coherence
        state[6] = 0.5   # somatic_comfort
        state[10] = 0.6  # vagal_tone
    elif name == "arousal":
        state[2] = 0.8   # dopamine high
        state[5] = 0.9   # norepinephrine high
        state[7] = 0.9   # somatic_arousal high
        state[8] = 0.6   # neural_stress moderate
        state[9] = 1.5   # encoding_boost high
    elif name == "calm_focus":
        state[1] = 0.8   # high coherence
        state[3] = 0.7   # serotonin elevated
        state[4] = 0.6   # acetylcholine moderate
        state[6] = 0.7   # comfort high
        state[10] = 0.8  # vagal_tone high
    elif name == "stress":
        state[5] = 0.95  # norepinephrine very high
        state[7] = 0.95  # somatic_arousal very high
        state[8] = 0.9   # neural_stress high
        state[9] = 0.5   # encoding_boost moderate
    return state


def measure_activity(net, steps=500, dt=0.1):
    """Run network with pulsed stimulation, return spike count.

    Uses pulsed stimulation (5ms on / 5ms off) to avoid depolarization
    block — a real biophysics effect where sustained current inactivates
    Na+ channels and suppresses firing.
    """
    start = net.spike_count
    for i in range(steps):
        if (i % 100) < 50:  # 5ms on, 5ms off
            net.stimulate((5.0, 5.0, 2.5), intensity=12.0, radius=5.0)
        net.step(dt=dt)
    return net.spike_count - start


def main():
    print("=" * 60)
    print("  Bio-LoRA ↔ Molecular Network Bridge Demo")
    print("=" * 60)

    bridge = BioLoRABridge()

    # Show concentration mapping for each state
    states = ["resting", "arousal", "calm_focus", "stress"]

    print("\n--- BioState → NT Concentration Mapping ---\n")
    print(f"  {'NT':<18s}", end="")
    for s in states:
        print(f"  {s:>12s}", end="")
    print()
    print(f"  {'-' * 18}", end="")
    for _ in states:
        print(f"  {'-' * 12}", end="")
    print()

    concs_by_state = {}
    for s in states:
        bio = make_bio_state(s)
        concs = bridge.torch_to_concentrations(bio)
        concs_by_state[s] = concs

    for nt in ["dopamine", "serotonin", "norepinephrine", "acetylcholine", "gaba", "glutamate"]:
        print(f"  {nt:<18s}", end="")
        for s in states:
            val = concs_by_state[s].get(nt, 0)
            print(f"  {val:>10.1f}nM", end="")
        print()

    # Drive networks with different states and measure spike rates
    print("\n--- Spike Rate Under Different BioStates ---")
    print("  (Each network warmed up 50ms, then 50ms pulsed stimulation)\n")

    from oneuro.molecular.neuron import NeuronArchetype

    spike_results = {}
    for state_name in states:
        np.random.seed(42)
        net = MolecularNeuralNetwork(
            size=(10.0, 10.0, 5.0), initial_neurons=10, energy_supply=3.0
        )
        # Add mixed archetypes so NT changes affect different cell types
        for _ in range(5):
            p = np.random.uniform([0, 0, 0], [10, 10, 5])
            net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.INTERNEURON)
        for _ in range(5):
            p = np.random.uniform([0, 0, 0], [10, 10, 5])
            net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MEDIUM_SPINY)
        for _ in range(5):
            p = np.random.uniform([0, 0, 0], [10, 10, 5])
            net._add_neuron(p[0], p[1], p[2], archetype=NeuronArchetype.MOTONEURON)

        bio = make_bio_state(state_name)
        bridge.drive_network(net, bio, blend=1.0)

        # Warm up to clear transients
        for i in range(500):
            if (i % 100) < 50:
                net.stimulate((5.0, 5.0, 2.5), intensity=12.0, radius=5.0)
            net.step(dt=0.1)

        spikes = measure_activity(net, steps=500, dt=0.1)
        spike_results[state_name] = spikes

        bar_len = min(40, max(1, spikes))
        bar = "█" * bar_len
        print(f"  {state_name:12s}: {spikes:4d} spikes  {bar}")

    # Ratios
    resting = max(1, spike_results["resting"])
    print(f"\n  Arousal/Resting ratio: {spike_results['arousal'] / resting:.1f}x")
    print(f"  Stress/Resting ratio:  {spike_results['stress'] / resting:.1f}x")
    print(f"  Focus/Resting ratio:   {spike_results['calm_focus'] / resting:.1f}x")

    # Round-trip: network → BioState → concentrations
    print("\n--- Round-Trip: Network State → BioState → Concentrations ---\n")
    np.random.seed(42)
    net = MolecularNeuralNetwork(
        size=(10.0, 10.0, 5.0), initial_neurons=25, energy_supply=3.0
    )
    # Drive with arousal
    arousal_bio = make_bio_state("arousal")
    bridge.drive_network(net, arousal_bio, blend=1.0)

    # Read back
    read_back = bridge.read_network_state(net)
    print(f"  Original BioState (arousal):")
    dim_names = [
        "cardiac_phase", "coherence", "dopamine", "serotonin",
        "acetylcholine", "norepinephrine", "somatic_comfort",
        "somatic_arousal", "neural_stress", "encoding_boost",
        "vagal_tone", "reserved",
    ]
    for i, name in enumerate(dim_names):
        orig = arousal_bio[i]
        back = read_back[i]
        match = "✓" if abs(orig - back) < 0.2 else "≈"
        if orig > 0 or back > 0:
            print(f"    {name:20s}: orig={orig:.2f}  read={back:.2f}  {match}")

    # Verify input format compatibility
    print("\n--- Input Format Compatibility ---\n")
    list_input = [0.0, 0.5, 0.8, 0.3, 0.6, 0.7, 0.5, 0.2, 0.1, 1.0, 0.6, 0.0]
    np_input = np.array(list_input)
    concs_list = bridge.torch_to_concentrations(list_input)
    concs_np = bridge.torch_to_concentrations(np_input)
    match = all(abs(concs_list[k] - concs_np[k]) < 1e-10 for k in concs_list)
    print(f"  list input == numpy input: {match}")
    print(f"  (torch.Tensor would also work via .detach().cpu().numpy())")


if __name__ == "__main__":
    main()
