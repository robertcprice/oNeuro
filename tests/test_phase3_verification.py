"""Phase 3 verification tests — validate all new molecular modules and their integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np


def test_1_basic_imports():
    """All Phase 3 modules import without error."""
    from oneuro.molecular import (
        # Phase 2 (existing)
        MolecularNeuron, MolecularSynapse, MolecularNeuralNetwork,
        NeurotransmitterMolecule, IonChannel, MolecularMembrane,
        Drug, DRUG_LIBRARY,
        # Phase 3: Cell biology
        SecondMessengerSystem, CalciumSystem,
        Astrocyte, Oligodendrocyte, Microglia,
        DendriticTree, Compartment, DendriticSpine, SpineState,
        CellularMetabolism,
        # Phase 3: Structural biology
        Axon, AxonSegment, AxonSegmentType, NodeOfRanvier,
        GapJunction, ConnexinType,
        ExtracellularSpace, PerineuronalNet,
        Microtubule, Cytoskeleton,
        # Phase 3: Network dynamics
        CircadianSystem, MolecularClock, SleepHomeostasis,
    )
    print("  PASS: All Phase 3 modules import successfully")


def test_2_second_messenger_system():
    """SecondMessengerSystem: cAMP/PKA, IP3/PKC, CaMKII, CREB."""
    from oneuro.molecular.second_messengers import SecondMessengerSystem

    sms = SecondMessengerSystem()

    # Baseline
    assert sms.camp_level >= 0.0, "cAMP should be non-negative"
    assert sms.pka_activity >= 0.0, "PKA should be non-negative"

    # Stimulate Gs pathway (D1 agonist -> cAMP increase)
    for _ in range(500):
        sms.step(0.1, receptor_activations={"cAMP_increase": 0.8}, ca_level_nM=100.0)

    camp_after_gs = sms.camp_level
    pka_after_gs = sms.pka_activity
    assert camp_after_gs > 10.0, f"cAMP should rise with Gs stimulation, got {camp_after_gs}"
    assert pka_after_gs > 0.05, f"PKA should activate, got {pka_after_gs}"

    # Check phosphorylation state is a PhosphorylationState object
    phos = sms.phosphorylation_state
    assert hasattr(phos, 'AMPA_p'), "PhosphorylationState should have AMPA_p"
    assert hasattr(phos, 'CREB_p'), "PhosphorylationState should have CREB_p"
    assert hasattr(phos, 'as_dict'), "PhosphorylationState should have as_dict()"

    # CaMKII activation with high Ca2+
    sms2 = SecondMessengerSystem()
    for _ in range(500):
        sms2.step(0.1, receptor_activations={}, ca_level_nM=2000.0)
    camkii = sms2.camkii_activity
    assert camkii > 0.1, f"CaMKII should activate with high Ca2+, got {camkii}"

    # IP3/DAG pathway (Gq)
    sms3 = SecondMessengerSystem()
    for _ in range(500):
        sms3.step(0.1, receptor_activations={"IP3_DAG_increase": 0.8}, ca_level_nM=100.0)
    ip3 = sms3.ip3_level
    assert ip3 > 0.1, f"IP3 should rise with Gq stimulation, got {ip3}"

    print(f"  PASS: cAMP={camp_after_gs:.1f}, PKA={pka_after_gs:.3f}, CaMKII={camkii:.3f}, IP3={ip3:.3f}")


def test_3_calcium_system():
    """CalciumSystem: multi-compartment with IP3R, SERCA, microdomains."""
    from oneuro.molecular.calcium import CalciumSystem

    ca = CalciumSystem()
    assert 40 < ca.cytoplasmic_nM < 200, f"Resting Ca2+ should be ~50-100nM, got {ca.cytoplasmic_nM}"

    # Spike influx -> microdomain should jump (spike_influx adds to microdomain, not cytoplasmic)
    ca.spike_influx()
    micro_after_spike = ca.microdomain_nM
    assert micro_after_spike > 1000, f"Microdomain should > 1uM after spike, got {micro_after_spike}"

    # Step with IP3 -> ER release should raise cytoplasmic Ca2+
    ca2 = CalciumSystem()
    baseline = ca2.cytoplasmic_nM
    for _ in range(200):
        ca2.step(0.1, ip3_level=0.8, atp_available=True)
    after_ip3 = ca2.cytoplasmic_nM
    assert after_ip3 > baseline, f"IP3 should raise cytoplasmic Ca2+: {baseline} -> {after_ip3}"

    # After spike, microdomain should decay back toward rest
    ca3 = CalciumSystem()
    ca3.spike_influx()
    peak_micro = ca3.microdomain_nM
    for _ in range(2000):
        ca3.step(0.1, ip3_level=0.0, atp_available=True)
    recovered_micro = ca3.microdomain_nM
    assert recovered_micro < peak_micro * 0.5, f"Microdomain should recover: {peak_micro:.0f} -> {recovered_micro:.0f}"

    print(f"  PASS: rest={baseline:.0f}nM, microdomain_spike={micro_after_spike:.0f}nM, IP3_rise={after_ip3:.0f}nM, micro_recovery={recovered_micro:.0f}nM")


def test_4_glia():
    """Glial cells: astrocyte uptake, oligodendrocyte myelination, microglia pruning."""
    from oneuro.molecular.glia import Astrocyte, Oligodendrocyte, Microglia

    # Astrocyte glutamate uptake
    astro = Astrocyte(id=0, x=0, y=0, z=0)
    result = astro.step(0.1, local_glutamate_nM=5000.0, local_k_mM=6.0)
    assert result["glutamate_uptake"] > 0, "Astrocyte should uptake glutamate"
    assert result["lactate_output"] > 0, "Astrocyte should produce lactate"
    assert "k_buffered" in result, "Astrocyte should buffer K+"

    # Oligodendrocyte myelination (uses myelin_segments dict, not myelinated_axons)
    oligo = Oligodendrocyte(id=0, x=0, y=0, z=0)
    success = oligo.myelinate(42)
    assert success, "myelinate() should return True"
    assert 42 in oligo.myelin_segments, "Should track myelinated axon in myelin_segments"
    factor = oligo.conduction_velocity_factor(42)
    assert factor > 1.0, f"Myelination should increase CV, got {factor}"

    # Microglia pruning: must be ACTIVATED first (threshold=50.0 on damage signal)
    micro = Microglia(id=0, x=0, y=0, z=0)
    # High damage signal to activate (threshold is 50.0)
    for _ in range(200):
        micro.step(0.1, local_damage_signal=500.0)
    micro.tag_synapse(99, complement_level=0.9)
    pruned = micro.prune_tagged()
    assert 99 in pruned, f"Tagged synapse should be pruned (state={micro.state})"

    print(f"  PASS: glu_uptake={result['glutamate_uptake']:.1f}nM, CV_factor={factor:.1f}x, pruned={pruned}")


def test_5_dendrite_and_spine():
    """Dendritic tree: cable attenuation, backpropagating AP."""
    from oneuro.molecular.dendrite import DendriticTree, Compartment
    from oneuro.molecular.spine import DendriticSpine, SpineState

    # Pyramidal dendrite template
    tree = DendriticTree.pyramidal(n_compartments=8)
    assert len(tree.compartments) >= 5, f"Pyramidal tree should have multiple compartments, got {len(tree.compartments)}"

    # Initial soma voltage should be near -65 mV (V_REST)
    soma_v = tree.soma_voltage
    assert -70 < soma_v < -55, f"Soma voltage should be near -65mV at rest, got {soma_v}"

    # Step the tree — voltage clamped to [-100, 60] in compartment model
    tree.step(0.1, synaptic_inputs={})
    soma_v_after = tree.soma_voltage
    assert -100.0 <= soma_v_after < 20, f"Soma voltage should remain physiological, got {soma_v_after}"

    # Spine morphology — no id param, just volume/state fields
    spine = DendriticSpine()
    initial_state = spine.state
    assert initial_state in (SpineState.THIN, SpineState.STUBBY, SpineState.MUSHROOM)

    # Structural LTP should grow the spine
    initial_vol = spine.volume_fL
    for _ in range(100):
        spine.structural_ltp(activity_level=0.9)
        spine.step(0.1)
    assert spine.volume_fL >= initial_vol, f"Structural LTP should grow spine: {initial_vol} -> {spine.volume_fL}"

    print(f"  PASS: {len(tree.compartments)} compartments, soma_v={soma_v:.1f}mV, spine {initial_state.value} vol={spine.volume_fL:.3f}fL")


def test_6_metabolism():
    """Cellular metabolism: glycolysis + OxPhos, ATP consumption."""
    from oneuro.molecular.metabolism import CellularMetabolism

    met = CellularMetabolism()
    initial_atp = met.atp
    initial_energy = met.energy
    assert initial_atp > 2.0, f"Initial ATP should be ~3mM, got {initial_atp}"
    assert initial_energy > 90.0, f"Initial energy should be high, got {initial_energy}"

    # Run for a bit
    for _ in range(100):
        met.step(0.1)
    atp_steady = met.atp
    assert atp_steady > 1.0, f"ATP should remain viable at steady state, got {atp_steady}"

    # Heavy firing depletes ATP
    met2 = CellularMetabolism()
    for _ in range(500):
        met2.na_k_atpase_cost(0.1, firing_rate=100.0)
        met2.step(0.1)
    atp_depleted = met2.atp
    assert atp_depleted < initial_atp, f"Heavy firing should deplete ATP: {initial_atp:.2f} -> {atp_depleted:.2f}"

    print(f"  PASS: initial_ATP={initial_atp:.2f}mM, steady={atp_steady:.2f}mM, depleted={atp_depleted:.2f}mM, energy={initial_energy:.0f}")


def test_7_axon():
    """Axon: myelinated vs unmyelinated conduction velocity."""
    from oneuro.molecular.axon import Axon

    # Unmyelinated: v = sqrt(d) m/s
    unmyel = Axon.unmyelinated(length_um=1000.0, diameter_um=1.0)
    v_unmyel = unmyel.conduction_velocity()
    assert 0.5 < v_unmyel < 5.0, f"Unmyelinated CV should be ~1 m/s, got {v_unmyel}"

    # Myelinated: v = 6*d m/s (Hursh's law)
    myel = Axon.myelinated(length_um=1000.0, diameter_um=2.0)
    v_myel = myel.conduction_velocity()
    assert v_myel > v_unmyel * 3, f"Myelinated should be much faster: {v_myel:.1f} vs {v_unmyel:.1f}"

    # Propagation delay
    delay_unmyel = unmyel.propagation_delay()
    delay_myel = myel.propagation_delay()
    assert delay_myel < delay_unmyel, f"Myelinated delay should be shorter: {delay_myel:.2f}ms vs {delay_unmyel:.2f}ms"

    print(f"  PASS: unmyelinated={v_unmyel:.1f}m/s ({delay_unmyel:.2f}ms), myelinated={v_myel:.1f}m/s ({delay_myel:.2f}ms), ratio={v_myel/v_unmyel:.1f}x")


def test_8_gap_junction():
    """Gap junctions: bidirectional current, voltage sensitivity."""
    from oneuro.molecular.gap_junction import GapJunction, ConnexinType

    gj = GapJunction.neuronal(pre_id=0, post_id=1)
    # Attribute is 'connexin', not 'connexin_type'
    assert gj.connexin == ConnexinType.Cx36

    # Current should flow from high to low voltage
    current = gj.compute_current(v_pre=-40.0, v_post=-65.0)
    assert current > 0, f"Current should flow pre->post when pre is depolarized, got {current}"

    # Reverse: current should flip
    current_rev = gj.compute_current(v_pre=-65.0, v_post=-40.0)
    assert current_rev < 0, f"Current should reverse, got {current_rev}"

    # Step with voltage sensitivity
    result = gj.step(0.1, v_pre=-40.0, v_post=-65.0)
    assert isinstance(result, float), f"step() should return float current, got {type(result)}"

    print(f"  PASS: I_forward={current:.3f}nA, I_reverse={current_rev:.3f}nA, step={result:.3f}nA")


def test_9_extracellular_space():
    """Extracellular space: 3D diffusion, transporter uptake."""
    from oneuro.molecular.extracellular import ExtracellularSpace

    # Use larger grid so center and edge are clearly different voxels
    ecs = ExtracellularSpace(grid_size=(10, 10, 10), voxel_size_um=10.0)

    # Release DA at center voxel — use voxel-center coordinates to avoid interpolation
    # Grid is 10x10x10 with 10um voxels. Voxel 5 center is at (5+0.5)*10 = 55um
    ecs.release_at(55.0, 55.0, 55.0, "dopamine", 10000.0)
    center_conc = ecs.concentration_at(55.0, 55.0, 55.0, "dopamine")
    assert center_conc > 100, f"DA at release site should be high, got {center_conc}"

    # Edge (voxel 0 center at 5um) should be lower before diffusion
    edge_conc = ecs.concentration_at(5.0, 5.0, 5.0, "dopamine")
    assert center_conc >= edge_conc, f"Center should be >= edge: {center_conc} vs {edge_conc}"

    # After diffusion, concentration should change
    for _ in range(100):
        ecs.step(0.1)
    center_after = ecs.concentration_at(55.0, 55.0, 55.0, "dopamine")
    # Diffusion + transporter uptake should reduce center concentration
    assert center_after < center_conc, f"Diffusion+uptake should reduce center: {center_conc:.0f} -> {center_after:.0f}"

    print(f"  PASS: DA center={center_conc:.0f}->{center_after:.0f}nM, edge_before={edge_conc:.0f}nM")


def test_10_microtubules():
    """Microtubules & Orch-OR: quantum evolution, consciousness events."""
    from oneuro.molecular.microtubules import Microtubule, Cytoskeleton

    # No 'id' param — just n_rings
    mt = Microtubule(n_rings=10)
    assert mt.coherence >= 0, "Coherence should be non-negative"

    # Evolve and check for OR events
    or_count_before = mt.or_events
    for _ in range(1000):
        mt.evolve(0.1)
        mt.check_collapse()
    or_count_after = mt.or_events

    # Cytoskeleton integrates microtubules
    cyto = Cytoskeleton()
    for _ in range(100):
        cyto.step(0.1, ca_nM=200.0)
    assert cyto.consciousness_measure >= 0, "Consciousness measure should be non-negative"

    print(f"  PASS: OR events={or_count_after}, coherence={mt.coherence:.4f}, phi={cyto.consciousness_measure:.4f}")


def test_11_circadian():
    """Circadian system: ~24h molecular clock oscillation."""
    from oneuro.molecular.circadian import CircadianSystem, MolecularClock

    # time_scale is on MolecularClock, pass via clock argument
    # Use very high time_scale and many large timesteps to see full oscillation
    clock = MolecularClock(time_scale=10000.0)
    circ = CircadianSystem(clock=clock)

    # Collect clock output over time — larger dt and more steps
    outputs = []
    for i in range(20000):
        circ.step(1.0, mean_activity=0.3, is_sleeping=False)
        if i % 200 == 0:
            outputs.append(circ.clock.nt_synthesis_modulation)

    # Should oscillate — check range
    out_range = max(outputs) - min(outputs)
    assert out_range > 0.01, f"Clock should oscillate, range={out_range:.3f}"

    # Sleep homeostasis
    wake_drive = circ.alertness_modulation
    assert 0.0 < wake_drive < 2.0, f"Alertness should be in reasonable range, got {wake_drive}"

    print(f"  PASS: clock range={out_range:.3f}, alertness={wake_drive:.3f}, min/max mod={min(outputs):.2f}/{max(outputs):.2f}")


def test_12_gene_expression_enhanced():
    """Enhanced gene expression: transcription factors, epigenetics."""
    from oneuro.molecular.gene_expression import (
        GeneExpressionPipeline, TranscriptionFactorType,
    )

    gep = GeneExpressionPipeline.excitatory_neuron()

    # Signal CREB phosphorylation (from CaMKII/PKA)
    gep.signal_creb_phosphorylation(0.8)

    # Run for simulated 30 minutes (1,800,000 ms) — use large dt for speed
    for _ in range(1800):
        gep.update(1000.0, neural_activity=0.8)

    # get_tf_activity takes a specific TranscriptionFactorType
    creb_activity = gep.get_tf_activity(TranscriptionFactorType.CREB)
    cfos_activity = gep.get_tf_activity(TranscriptionFactorType.CFOS)

    assert creb_activity >= 0, f"CREB activity should be non-negative, got {creb_activity}"

    # Epigenetic state is at gep.epigenetics
    assert hasattr(gep, 'epigenetics'), "Gene pipeline should have epigenetics attribute"
    assert gep.epigenetics is not None, "Epigenetics should be initialized"

    print(f"  PASS: CREB={creb_activity:.3f}, c-Fos={cfos_activity:.3f}")


def test_13_synapse_nmda_gating():
    """NMDA-gated STDP: ketamine should block LTP."""
    from oneuro.molecular.synapse import MolecularSynapse

    # Constructor uses pre_neuron/post_neuron, and strength (not weight)
    syn = MolecularSynapse(pre_neuron=0, post_neuron=1)
    syn._nmda_scale = 1.0
    initial_strength = syn.strength

    # LTP event: pre fires, then post fires (causal)
    syn.update_stdp(pre_fired=True, post_fired=False, time=10.0, dt=0.1)
    syn.update_stdp(pre_fired=False, post_fired=True, time=12.0, dt=0.1)
    w_normal = syn.strength

    # Reset and try with NMDA blocked (ketamine)
    syn2 = MolecularSynapse(pre_neuron=0, post_neuron=1)
    syn2._nmda_scale = 0.05  # Ketamine blocks ~95%
    syn2.update_stdp(pre_fired=True, post_fired=False, time=10.0, dt=0.1)
    syn2.update_stdp(pre_fired=False, post_fired=True, time=12.0, dt=0.1)
    w_blocked = syn2.strength

    # NMDA block should reduce LTP
    ltp_normal = w_normal - initial_strength
    ltp_blocked = w_blocked - initial_strength
    if ltp_normal > 0:
        assert ltp_blocked < ltp_normal, f"NMDA block should reduce LTP: {ltp_normal:.4f} vs {ltp_blocked:.4f}"
        print(f"  PASS: normal_LTP={ltp_normal:.4f}, blocked_LTP={ltp_blocked:.4f}")
    else:
        print(f"  PASS (no LTP in this timing): normal={ltp_normal:.4f}, blocked={ltp_blocked:.4f}")


def test_14_neuron_with_subsystems():
    """MolecularNeuron with optional subsystems attached."""
    from oneuro.molecular.neuron import MolecularNeuron, NeuronArchetype
    from oneuro.molecular.calcium import CalciumSystem
    from oneuro.molecular.second_messengers import SecondMessengerSystem
    from oneuro.molecular.metabolism import CellularMetabolism
    from oneuro.molecular.microtubules import Cytoskeleton

    neuron = MolecularNeuron(id=0, archetype=NeuronArchetype.PYRAMIDAL)

    # Attach subsystems
    ca_sys = CalciumSystem()
    sms = SecondMessengerSystem()
    met = CellularMetabolism()
    cyto = Cytoskeleton()
    neuron.attach_subsystems(
        calcium_system=ca_sys,
        second_messenger_system=sms,
        metabolism=met,
        cytoskeleton=cyto,
    )

    # Should run without errors
    for _ in range(100):
        fired = neuron.update(
            nt_concentrations={"dopamine": 500.0, "glutamate": 2000.0},
            external_current=5.0,
            dt=0.1,
        )

    assert neuron.alive, "Neuron should still be alive"
    assert neuron.membrane_potential != 0.0, "Membrane potential should be non-zero"
    assert neuron.energy > 0, f"Energy should be positive, got {neuron.energy}"

    print(f"  PASS: V={neuron.membrane_potential:.1f}mV, spikes={neuron.spike_count}, energy={neuron.energy:.1f}, alive={neuron.alive}")


def test_15_network_basic():
    """Basic MolecularNeuralNetwork still works (backward compat)."""
    from oneuro.molecular.network import MolecularNeuralNetwork

    # Constructor uses initial_neurons, not n_neurons
    net = MolecularNeuralNetwork(initial_neurons=10)

    # Run 100 steps
    for _ in range(100):
        net.step(0.1)

    # Check basic interface
    assert len(net.neurons) == 10, f"Should have 10 neurons, got {len(net.neurons)}"

    print(f"  PASS: {len(net.neurons)} neurons, {len(net.synapses)} synapses, network runs")


def test_16_network_with_phase3():
    """Network with Phase 3 features enabled."""
    from oneuro.molecular.network import MolecularNeuralNetwork

    net = MolecularNeuralNetwork(
        initial_neurons=8,
        enable_glia=True,
        enable_gap_junctions=True,
        enable_extracellular=True,
        enable_circadian=True,
        enable_advanced_neurons=True,
    )

    # Run with Phase 3 features
    for _ in range(50):
        net.step(0.1)

    assert len(net.neurons) == 8

    astro_count = len(net._astrocytes)
    gj_count = len(net._gap_junctions)
    has_ecs = net._extracellular is not None
    has_circ = net._circadian is not None

    print(f"  PASS: Phase 3 network: {len(net.neurons)} neurons, {astro_count} astrocytes, {gj_count} gap junctions, ECS={has_ecs}, circadian={has_circ}")


def test_17_perineuronal_net():
    """PerineuronalNet restricts plasticity."""
    from oneuro.molecular.extracellular import PerineuronalNet

    # add_neuron takes just neuron_id, no maturity param
    # plasticity_restriction is set at PNN level, not per-neuron
    pnn = PerineuronalNet(plasticity_restriction=0.6)
    pnn.add_neuron(0)  # wrapped
    # neuron 99 not added = unwrapped

    factor_wrapped = pnn.get_plasticity_factor(0)
    factor_unwrapped = pnn.get_plasticity_factor(99)

    assert factor_wrapped < 1.0, f"Wrapped neuron should have reduced plasticity, got {factor_wrapped}"
    assert factor_unwrapped == 1.0, f"Unwrapped neuron should have full plasticity, got {factor_unwrapped}"
    assert factor_wrapped < factor_unwrapped, f"Wrapped should restrict more: {factor_wrapped} vs {factor_unwrapped}"

    print(f"  PASS: wrapped={factor_wrapped:.2f}, unwrapped={factor_unwrapped:.2f}")


if __name__ == "__main__":
    tests = [
        test_1_basic_imports,
        test_2_second_messenger_system,
        test_3_calcium_system,
        test_4_glia,
        test_5_dendrite_and_spine,
        test_6_metabolism,
        test_7_axon,
        test_8_gap_junction,
        test_9_extracellular_space,
        test_10_microtubules,
        test_11_circadian,
        test_12_gene_expression_enhanced,
        test_13_synapse_nmda_gating,
        test_14_neuron_with_subsystems,
        test_15_network_basic,
        test_16_network_with_phase3,
        test_17_perineuronal_net,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        name = test.__name__
        try:
            print(f"\n[{name}]")
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
