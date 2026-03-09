"""Tests for the corticostriatal mechanism surface."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.corticostriatal_mechanism_experiment import (
    VARIANTS,
    ProtocolConfig,
    run_experiment,
    run_protocol,
)
from oneuro.molecular.brain_regions import RegionalBrain
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import MolecularNeuron, NeuronArchetype
from oneuro.molecular.receptors import ReceptorType
from oneuro.molecular.synapse import MolecularSynapse


def test_d1_and_d2_msn_archetypes_are_distinct():
    """Direct and indirect pathway MSNs should no longer share the same profile."""
    d1 = MolecularNeuron(id=0, archetype=NeuronArchetype.D1_MSN)
    d2 = MolecularNeuron(id=1, archetype=NeuronArchetype.D2_MSN)

    assert d1.membrane.receptor_count(ReceptorType.D1) > d1.membrane.receptor_count(
        ReceptorType.D2
    )
    assert d2.membrane.receptor_count(ReceptorType.D2) > d2.membrane.receptor_count(
        ReceptorType.D1
    )

    brain = RegionalBrain.minimal(seed=11)
    for nid in brain.basal_ganglia.get_ids("D1"):
        assert brain.network._molecular_neurons[nid].archetype == NeuronArchetype.D1_MSN
    for nid in brain.basal_ganglia.get_ids("D2"):
        assert brain.network._molecular_neurons[nid].archetype == NeuronArchetype.D2_MSN


def test_benchmark_safe_mode_skips_pruning_and_pnn_wrapping():
    """Short assays should not silently mutate structure during cleanup."""
    safe_net = MolecularNeuralNetwork(
        initial_neurons=0,
        enable_extracellular=True,
        benchmark_safe_mode=True,
    )
    pre = safe_net.create_neuron_at(0.0, 0.0, 0.0)
    post = safe_net.create_neuron_at(1.0, 0.0, 0.0)
    safe_net.create_synapse(pre, post, "glutamate")
    safe_syn = safe_net._molecular_synapses[(pre, post)]
    safe_syn.strength = 0.05
    safe_syn._postsynaptic_receptor_count = {ReceptorType.AMPA: 1}
    safe_net._molecular_neurons[post].age = 600.0
    safe_net._molecular_neurons[post].membrane._spike_count = 150
    safe_net._cleanup()

    assert (pre, post) in safe_net._molecular_synapses
    assert safe_net._perineuronal_net is not None
    assert not safe_net._perineuronal_net.is_wrapped(post)

    normal_net = MolecularNeuralNetwork(
        initial_neurons=0,
        enable_extracellular=True,
        benchmark_safe_mode=False,
    )
    pre2 = normal_net.create_neuron_at(0.0, 0.0, 0.0)
    post2 = normal_net.create_neuron_at(1.0, 0.0, 0.0)
    normal_net.create_synapse(pre2, post2, "glutamate")
    normal_syn = normal_net._molecular_synapses[(pre2, post2)]
    normal_syn.strength = 0.05
    normal_syn._postsynaptic_receptor_count = {ReceptorType.AMPA: 1}
    normal_net._molecular_neurons[post2].age = 600.0
    normal_net._molecular_neurons[post2].membrane._spike_count = 150
    normal_net._cleanup()

    assert (pre2, post2) not in normal_net._molecular_synapses
    assert normal_net._perineuronal_net is not None
    assert normal_net._perineuronal_net.is_wrapped(post2)


def test_dopamine_plasticity_factor_tracks_pathway_sign():
    """Reward modulation should now differ by D1 versus D2 pathway identity."""
    net = MolecularNeuralNetwork(initial_neurons=0)
    d1_id = net.create_neuron_at(0.0, 0.0, 0.0, archetype=NeuronArchetype.D1_MSN)
    d2_id = net.create_neuron_at(1.0, 0.0, 0.0, archetype=NeuronArchetype.D2_MSN)
    py_id = net.create_neuron_at(2.0, 0.0, 0.0, archetype=NeuronArchetype.PYRAMIDAL)
    interneuron_id = net.create_neuron_at(
        3.0,
        0.0,
        0.0,
        archetype=NeuronArchetype.INTERNEURON,
    )

    assert net.dopamine_plasticity_factor(d1_id) > 0.0
    assert net.dopamine_plasticity_factor(d2_id) < 0.0
    assert net.dopamine_plasticity_factor(py_id) == 1.0
    assert net.dopamine_plasticity_factor(interneuron_id) == 1.0


def test_zero_plasticity_factor_blocks_stdp_receptor_traffic():
    """A frozen synapse should not still add or remove receptors via STDP."""
    syn = MolecularSynapse(pre_neuron=0, post_neuron=1)
    initial_ampa = syn.receptor_count.get(ReceptorType.AMPA, 0)

    syn._plasticity_factor = 0.0
    syn.update_stdp(pre_fired=True, post_fired=False, time=10.0, dt=0.1)
    syn.update_stdp(pre_fired=False, post_fired=True, time=12.0, dt=0.1)

    assert syn.receptor_count.get(ReceptorType.AMPA, 0) == initial_ampa


def test_reward_capture_respects_nmda_and_tag_state():
    """Reward capture should collapse under NMDA block even with the same reward."""
    tagged = MolecularSynapse(pre_neuron=0, post_neuron=1)
    blocked = MolecularSynapse(pre_neuron=0, post_neuron=1)

    for syn in (tagged, blocked):
        syn.eligibility_trace = 0.35
        syn._tagged = True
        syn._tag_strength = 0.8

    blocked._nmda_scale = 0.05

    tagged_initial = tagged.receptor_count.get(ReceptorType.AMPA, 0)
    blocked_initial = blocked.receptor_count.get(ReceptorType.AMPA, 0)

    tagged.apply_reward(4.0, learning_rate=0.25, modulation_factor=2.5)
    blocked.apply_reward(4.0, learning_rate=0.25, modulation_factor=2.5)

    assert tagged.receptor_count.get(ReceptorType.AMPA, 0) > tagged_initial
    assert blocked.receptor_count.get(ReceptorType.AMPA, 0) == blocked_initial


def test_reward_capture_does_not_drift_strength_without_receptor_traffic():
    """Blocked or ineligible synapses should not still change effective weight."""
    syn = MolecularSynapse(pre_neuron=0, post_neuron=1)
    syn.strength = 0.5
    syn._nmda_scale = 0.0
    syn.eligibility_trace = 0.01

    initial_weight = syn.weight
    initial_strength = syn.strength
    syn.apply_reward(-1.0, learning_rate=0.1, modulation_factor=1.0)

    assert syn.weight == initial_weight
    assert syn.strength == initial_strength


def test_non_glutamatergic_synapses_ignore_reward_modulation():
    """Reward traffic should not mutate inhibitory or modulatory synapses."""
    syn = MolecularSynapse(pre_neuron=0, post_neuron=1, nt_name="gaba")
    syn.strength = 0.5
    syn.eligibility_trace = 0.8

    initial_weight = syn.weight
    initial_strength = syn.strength
    initial_receptors = syn.receptor_count
    syn.apply_reward(4.0, learning_rate=0.25, modulation_factor=1.0)

    assert syn.weight == initial_weight
    assert syn.strength == initial_strength
    assert syn.receptor_count == initial_receptors


def test_run_protocol_smoke_has_opposite_modulation_signs():
    """Rewarded D1 and D2 protocols should expose opposite-sign modulation."""
    config = ProtocolConfig(
        pairings=2,
        replicates=1,
        dopamine_pulse_ms=40.0,
        inter_pair_interval_ms=10.0,
        dopamine_delays_ms=(0.0,),
        pairing_timings_ms=(10.0,),
        master_seed=7,
    )
    rewarded = {variant.name: variant for variant in VARIANTS}["rewarded"]

    d1_result = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=0.0,
        variant=rewarded,
        config=config,
        replicate_index=0,
    )
    d2_result = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D2_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=0.0,
        variant=rewarded,
        config=config,
        replicate_index=0,
    )

    assert d1_result.modulation_factor > 0.0
    assert d2_result.modulation_factor < 0.0


def test_variants_share_initial_conditions_for_pure_protocol_contrasts():
    """Mechanistic variants should start from the same synapse realization."""
    config = ProtocolConfig(
        pairings=2,
        replicates=1,
        dopamine_pulse_ms=40.0,
        inter_pair_interval_ms=10.0,
        dopamine_delays_ms=(0.0,),
        pairing_timings_ms=(10.0,),
        master_seed=7,
    )
    variants = {variant.name: variant for variant in VARIANTS}

    rewarded = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=0.0,
        variant=variants["rewarded"],
        config=config,
        replicate_index=0,
    )
    blocked = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=0.0,
        variant=variants["nmda_block"],
        config=config,
        replicate_index=0,
    )

    assert rewarded.seed == blocked.seed
    assert rewarded.initial_weight == blocked.initial_weight
    assert rewarded.initial_ampa_receptors == blocked.initial_ampa_receptors
    assert rewarded.initial_nmda_receptors == blocked.initial_nmda_receptors


def test_timing_and_delay_contrasts_share_initial_conditions():
    """Protocol timing should not change the initial synapse realization."""
    config = ProtocolConfig(
        pairings=2,
        replicates=1,
        dopamine_pulse_ms=40.0,
        inter_pair_interval_ms=10.0,
        dopamine_delays_ms=(0.0, 80.0),
        pairing_timings_ms=(10.0, -10.0),
        master_seed=7,
    )
    rewarded = {variant.name: variant for variant in VARIANTS}["rewarded"]

    immediate = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=0.0,
        variant=rewarded,
        config=config,
        replicate_index=0,
    )
    delayed = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=10.0,
        dopamine_delay_ms=80.0,
        variant=rewarded,
        config=config,
        replicate_index=0,
    )
    anti_causal = run_protocol(
        postsynaptic_archetype=NeuronArchetype.D1_MSN,
        pair_timing_ms=-10.0,
        dopamine_delay_ms=0.0,
        variant=rewarded,
        config=config,
        replicate_index=0,
    )

    assert immediate.seed == delayed.seed == anti_causal.seed
    assert immediate.initial_weight == delayed.initial_weight == anti_causal.initial_weight
    assert (
        immediate.initial_ampa_receptors
        == delayed.initial_ampa_receptors
        == anti_causal.initial_ampa_receptors
    )
    assert (
        immediate.initial_nmda_receptors
        == delayed.initial_nmda_receptors
        == anti_causal.initial_nmda_receptors
    )


def test_run_experiment_smoke_writes_json():
    """Mechanism runner should emit structured JSON plus summary contrasts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload = run_experiment(
            ProtocolConfig(
                pairings=2,
                replicates=1,
                dopamine_pulse_ms=40.0,
                inter_pair_interval_ms=10.0,
                dopamine_delays_ms=(0.0,),
                pairing_timings_ms=(10.0, -10.0),
                master_seed=7,
            ),
            output_dir=tmp_dir,
        )

        result_path = Path(payload["result_path"])
        assert result_path.exists()
        assert "d1_rewarded_vs_no_dopamine" in payload["selected_contrasts"]

        saved = json.loads(result_path.read_text())
        assert saved["experiment"] == "corticostriatal_mechanism_experiment"
        assert saved["summary"]
        assert saved["results"]
