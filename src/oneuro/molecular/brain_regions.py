"""Multi-region brain architecture built on MolecularNeuralNetwork.

All regions share ONE underlying MolecularNeuralNetwork (shared ECS, circadian,
glia).  Regions are logical groupings with specific connectivity patterns
that create emergent functional specialization.

Regions:
- CorticalColumn: 6-layer canonical microcircuit (L4→L2/3→L5→L6)
- ThalamicNucleus: relay + reticular neurons, burst/tonic from HH dynamics
- Hippocampus: DG→CA3→CA1 for episodic memory (encode, recall, replay)
- BasalGanglia: D1/D2 MSNs for action selection via Go/NoGo pathways
- RegionalBrain: composes all regions with inter-region projections

Usage:
    brain = RegionalBrain.minimal()   # ~120 neurons
    brain = RegionalBrain.standard()  # ~260 neurons
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.neuron import NeuronArchetype


# ---------------------------------------------------------------------------
# Region base
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """Base class for a brain region — a logical group of neurons."""

    name: str
    neuron_ids: List[int] = field(default_factory=list)
    # Subgroups within the region (e.g., layers, pathways)
    subgroups: Dict[str, List[int]] = field(default_factory=dict)
    # Position offset in the shared network space
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def get_ids(self, subgroup: Optional[str] = None) -> List[int]:
        """Get neuron IDs for a subgroup or all neurons."""
        if subgroup is not None:
            return self.subgroups.get(subgroup, [])
        return self.neuron_ids


# ---------------------------------------------------------------------------
# Cortical Column
# ---------------------------------------------------------------------------

@dataclass
class CorticalColumn(Region):
    """Canonical cortical microcircuit with 4 functional layers.

    L4 (input, granule/stellate): receives thalamic input
    L2/3 (processing, pyramidal + interneuron): cortico-cortical
    L5 (output, large pyramidal): projects to subcortical
    L6 (feedback, pyramidal): projects back to thalamus

    ~80% excitatory, ~20% inhibitory per layer (except L4 which is
    predominantly granule cells).
    """

    @staticmethod
    def build(
        network: MolecularNeuralNetwork,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        n_per_layer: int = 15,
        name: str = "cortex",
    ) -> "CorticalColumn":
        """Build a cortical column in the network."""
        col = CorticalColumn(name=name, origin=origin)
        ox, oy, oz = origin

        # L4: input layer (granule cells + some interneurons)
        l4_ids = []
        n_l4 = n_per_layer
        for i in range(n_l4):
            arch = NeuronArchetype.INTERNEURON if i >= int(n_l4 * 0.8) else NeuronArchetype.GRANULE
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 2),
                oz + 0.0 + np.random.uniform(-0.3, 0.3),
                archetype=arch,
            )
            l4_ids.append(nid)

        # L2/3: processing layer (pyramidal + interneurons)
        l23_ids = []
        n_l23 = n_per_layer
        for i in range(n_l23):
            arch = NeuronArchetype.INTERNEURON if i >= int(n_l23 * 0.8) else NeuronArchetype.PYRAMIDAL
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 2),
                oz + 2.0 + np.random.uniform(-0.3, 0.3),
                archetype=arch,
            )
            l23_ids.append(nid)

        # L5: output layer (large pyramidal)
        l5_ids = []
        n_l5 = max(5, n_per_layer // 2)
        for i in range(n_l5):
            arch = NeuronArchetype.INTERNEURON if i >= int(n_l5 * 0.8) else NeuronArchetype.PYRAMIDAL
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 2),
                oz + 4.0 + np.random.uniform(-0.3, 0.3),
                archetype=arch,
            )
            l5_ids.append(nid)

        # L6: feedback layer (pyramidal)
        l6_ids = []
        n_l6 = max(5, n_per_layer // 3)
        for i in range(n_l6):
            arch = NeuronArchetype.PYRAMIDAL
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 2),
                oz + 6.0 + np.random.uniform(-0.3, 0.3),
                archetype=arch,
            )
            l6_ids.append(nid)

        # Intra-layer connectivity: feedforward L4→L2/3→L5→L6
        _connect_layers(network, l4_ids, l23_ids, p=0.3, nt="glutamate")
        _connect_layers(network, l23_ids, l5_ids, p=0.3, nt="glutamate")
        _connect_layers(network, l5_ids, l6_ids, p=0.2, nt="glutamate")

        # Feedback: L6→L4 (weak)
        _connect_layers(network, l6_ids, l4_ids, p=0.15, nt="glutamate")

        # Lateral inhibition within each layer (interneurons → excitatory)
        for layer_ids in [l4_ids, l23_ids, l5_ids]:
            excitatory = [nid for nid in layer_ids
                          if network._molecular_neurons[nid].archetype != NeuronArchetype.INTERNEURON]
            inhibitory = [nid for nid in layer_ids
                          if network._molecular_neurons[nid].archetype == NeuronArchetype.INTERNEURON]
            _connect_layers(network, inhibitory, excitatory, p=0.5, nt="gaba")
            _connect_layers(network, excitatory, inhibitory, p=0.3, nt="glutamate")

        col.neuron_ids = l4_ids + l23_ids + l5_ids + l6_ids
        col.subgroups = {"L4": l4_ids, "L2/3": l23_ids, "L5": l5_ids, "L6": l6_ids}
        return col


# ---------------------------------------------------------------------------
# Thalamic Nucleus
# ---------------------------------------------------------------------------

@dataclass
class ThalamicNucleus(Region):
    """Thalamic relay + reticular nucleus.

    Relay neurons: PYRAMIDAL with extra Ca_v for T-type burst mode.
    Reticular: INTERNEURON (GABAergic), provides rhythmic inhibition.
    Burst/tonic mode emerges naturally from HH ion channel dynamics
    combined with circadian hyperpolarization — no explicit mode switch.
    """

    @staticmethod
    def build(
        network: MolecularNeuralNetwork,
        origin: Tuple[float, float, float] = (5.0, 0.0, 0.0),
        n_relay: int = 10,
        n_reticular: int = 5,
        name: str = "thalamus",
    ) -> "ThalamicNucleus":
        """Build a thalamic nucleus in the network."""
        thal = ThalamicNucleus(name=name, origin=origin)
        ox, oy, oz = origin

        relay_ids = []
        for _ in range(n_relay):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 2),
                oz + np.random.uniform(-0.5, 0.5),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
            # Add extra Ca_v channels for T-type burst capability
            mol_n = network._molecular_neurons[nid]
            from oneuro.molecular.ion_channels import IonChannelType
            mol_n.membrane.channels.add_channel(IonChannelType.Ca_v, count=2)
            relay_ids.append(nid)

        reticular_ids = []
        for _ in range(n_reticular):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(2, 4),
                oz + np.random.uniform(-0.5, 0.5),
                archetype=NeuronArchetype.INTERNEURON,
            )
            reticular_ids.append(nid)

        # Relay → Reticular (glutamatergic collaterals)
        _connect_layers(network, relay_ids, reticular_ids, p=0.4, nt="glutamate")
        # Reticular → Relay (GABAergic inhibition — rhythmic bursting)
        _connect_layers(network, reticular_ids, relay_ids, p=0.5, nt="gaba")
        # Reticular ↔ Reticular (GABAergic, desynchronizing)
        _connect_within(network, reticular_ids, p=0.3, nt="gaba")

        thal.neuron_ids = relay_ids + reticular_ids
        thal.subgroups = {"relay": relay_ids, "reticular": reticular_ids}
        return thal


# ---------------------------------------------------------------------------
# Hippocampus
# ---------------------------------------------------------------------------

@dataclass
class Hippocampus(Region):
    """Hippocampal circuit: DG → CA3 → CA1.

    DG (granule cells): sparse coding, pattern separation.
    CA3 (pyramidal, recurrent p=0.3): pattern completion.
    CA1 (pyramidal, output): episodic memory readout.

    Supports encode_pattern(), recall_from_partial(), and replay_episode().
    """

    @staticmethod
    def build(
        network: MolecularNeuralNetwork,
        origin: Tuple[float, float, float] = (0.0, 5.0, 0.0),
        n_dg: int = 15,
        n_ca3: int = 10,
        n_ca1: int = 8,
        name: str = "hippocampus",
    ) -> "Hippocampus":
        """Build hippocampal circuit."""
        hipp = Hippocampus(name=name, origin=origin)
        ox, oy, oz = origin

        dg_ids = []
        for _ in range(n_dg):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 1),
                oz + np.random.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.GRANULE,
            )
            dg_ids.append(nid)

        ca3_ids = []
        for _ in range(n_ca3):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(1.5, 2.5),
                oz + np.random.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
            ca3_ids.append(nid)

        ca1_ids = []
        for _ in range(n_ca1):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(3.0, 4.0),
                oz + np.random.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.PYRAMIDAL,
            )
            ca1_ids.append(nid)

        # DG → CA3 (mossy fibers — few but strong "detonator" synapses)
        _connect_layers(network, dg_ids, ca3_ids, p=0.4, nt="glutamate")
        # CA3 → CA3 (recurrent collaterals — dense for pattern completion)
        _connect_within(network, ca3_ids, p=0.5, nt="glutamate")
        # CA3 → CA1 (Schaffer collaterals)
        _connect_layers(network, ca3_ids, ca1_ids, p=0.5, nt="glutamate")

        hipp.neuron_ids = dg_ids + ca3_ids + ca1_ids
        hipp.subgroups = {"DG": dg_ids, "CA3": ca3_ids, "CA1": ca1_ids}
        return hipp

    def encode_pattern(
        self, network: MolecularNeuralNetwork, pattern: List[float],
        intensity: float = 25.0, encode_steps: int = 20,
    ) -> None:
        """Encode a pattern by stimulating DG + CA3 over multiple steps.

        Biologically, entorhinal cortex projects to both DG (mossy fibers)
        and CA3 (perforant path), so encoding drives both layers.
        """
        dg_ids = self.subgroups.get("DG", [])
        ca3_ids = self.subgroups.get("CA3", [])
        if not dg_ids or not pattern:
            return
        # Sustained pulsed stimulation (currents are popped each step)
        for s in range(encode_steps):
            if s % 2 == 0:  # Pulsed to avoid depolarization block
                # Primary: DG (sparse coding)
                for i, nid in enumerate(dg_ids):
                    val = pattern[i % len(pattern)]
                    if val > 0.3:
                        network._external_currents[nid] = (
                            network._external_currents.get(nid, 0.0) + val * intensity
                        )
                # Secondary: CA3 direct path (weaker, perforant path)
                for i, nid in enumerate(ca3_ids):
                    val = pattern[i % len(pattern)]
                    if val > 0.3:
                        network._external_currents[nid] = (
                            network._external_currents.get(nid, 0.0) + val * intensity * 0.5
                        )
            network.step(0.1)

    def recall_from_partial(
        self, network: MolecularNeuralNetwork, partial_cue: List[float],
        settle_steps: int = 50, intensity: float = 25.0,
    ) -> List[float]:
        """Cue CA3 with partial pattern, let recurrents settle, read CA1."""
        ca3_ids = self.subgroups.get("CA3", [])
        ca1_ids = self.subgroups.get("CA1", [])
        if not ca3_ids:
            return []

        # Sustained pulsed stimulation of CA3 (currents are popped each step)
        ca1_set = set(ca1_ids)
        ca1_spike_counts = [0] * len(ca1_ids)
        stim_steps = min(settle_steps, 30)  # Stimulate for first 30 steps

        for s in range(settle_steps):
            # Re-apply cue current each step (pulsed: on/off to avoid Na+ inactivation)
            if s < stim_steps and s % 2 == 0:
                for i, nid in enumerate(ca3_ids):
                    if i < len(partial_cue) and partial_cue[i] > 0.3:
                        network._external_currents[nid] = (
                            network._external_currents.get(nid, 0.0)
                            + partial_cue[i] * intensity
                        )
            network.step(0.1)
            # Track CA1 spikes during recall
            for j, nid in enumerate(ca1_ids):
                if nid in network.last_fired:
                    ca1_spike_counts[j] += 1

        # Encode CA1 activity as normalized spike rates
        max_spikes = max(ca1_spike_counts) if ca1_spike_counts else 1
        if max_spikes == 0:
            # Fallback: read voltage
            result = []
            for nid in ca1_ids:
                mol_n = network._molecular_neurons.get(nid)
                if mol_n is not None:
                    norm_v = (mol_n.membrane.voltage + 70.0) / 90.0
                    result.append(max(0.0, min(1.0, norm_v)))
                else:
                    result.append(0.0)
            return result
        return [c / max_spikes for c in ca1_spike_counts]

    def replay_episode(
        self, network: MolecularNeuralNetwork,
        burst_intensity: float = 30.0, replay_steps: int = 100,
    ) -> int:
        """Replay: inject burst into CA3 during sleep → CA1 replay (SWR).

        Returns number of CA1 spikes during replay.
        """
        ca3_ids = self.subgroups.get("CA3", [])
        ca1_ids = self.subgroups.get("CA1", [])

        ca1_spikes = 0
        ca1_set = set(ca1_ids)
        burst_steps = min(replay_steps, 40)  # Burst for first 40 steps

        for s in range(replay_steps):
            # Sustained pulsed burst into CA3 (pulsed to avoid depolarization block)
            if s < burst_steps and s % 2 == 0:
                for nid in ca3_ids:
                    network._external_currents[nid] = (
                        network._external_currents.get(nid, 0.0) + burst_intensity
                    )
            network.step(0.1)
            ca1_spikes += len(network.last_fired & ca1_set)

        return ca1_spikes

    def replay_pattern(
        self, network: MolecularNeuralNetwork, pattern: List[float],
        intensity: float = 40.0, replay_steps: int = 60,
    ) -> int:
        """Replay a SPECIFIC pattern through DG → CA3 → CA1.

        Unlike replay_episode() which bursts all CA3 uniformly, this
        drives the specific DG neurons that were active during encoding,
        letting the learned DG→CA3→CA1 pathway reconstruct the pattern.

        Returns number of CA1 spikes during replay.
        """
        dg_ids = self.subgroups.get("DG", [])
        ca1_ids = self.subgroups.get("CA1", [])
        if not dg_ids or not pattern:
            return 0

        ca1_spikes = 0
        ca1_set = set(ca1_ids)
        stim_steps = min(replay_steps, 40)

        for s in range(replay_steps):
            if s < stim_steps and s % 2 == 0:
                for i, nid in enumerate(dg_ids):
                    val = pattern[i % len(pattern)]
                    if val > 0.3:
                        network._external_currents[nid] = (
                            network._external_currents.get(nid, 0.0) + val * intensity
                        )
            network.step(0.1)
            ca1_spikes += len(network.last_fired & ca1_set)

        return ca1_spikes


# ---------------------------------------------------------------------------
# Basal Ganglia
# ---------------------------------------------------------------------------

@dataclass
class BasalGanglia(Region):
    """Basal ganglia Go/NoGo pathways for action selection.

    D1 MSNs (Go pathway): excitation → action selection.
    D2 MSNs (NoGo pathway): inhibition → action suppression.
    Dopamine modulates: high DA → Go, low DA → NoGo.
    Lateral inhibition between D1 and D2 implements winner-take-all.
    """

    @staticmethod
    def build(
        network: MolecularNeuralNetwork,
        origin: Tuple[float, float, float] = (5.0, 5.0, 0.0),
        n_d1: int = 8,
        n_d2: int = 8,
        name: str = "basal_ganglia",
    ) -> "BasalGanglia":
        """Build basal ganglia circuit."""
        bg = BasalGanglia(name=name, origin=origin)
        ox, oy, oz = origin

        from oneuro.molecular.receptors import ReceptorType

        d1_ids = []
        for _ in range(n_d1):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(0, 1.5),
                oz + np.random.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.MEDIUM_SPINY,
            )
            d1_ids.append(nid)

        d2_ids = []
        for _ in range(n_d2):
            nid = network.create_neuron_at(
                ox + np.random.uniform(0, 2),
                oy + np.random.uniform(2.0, 3.5),
                oz + np.random.uniform(-0.3, 0.3),
                archetype=NeuronArchetype.MEDIUM_SPINY,
            )
            d2_ids.append(nid)

        # D1 ↔ D2 lateral inhibition (GABAergic)
        _connect_layers(network, d1_ids, d2_ids, p=0.3, nt="gaba")
        _connect_layers(network, d2_ids, d1_ids, p=0.3, nt="gaba")
        # Within-pathway weak connections
        _connect_within(network, d1_ids, p=0.15, nt="gaba")
        _connect_within(network, d2_ids, p=0.15, nt="gaba")

        bg.neuron_ids = d1_ids + d2_ids
        bg.subgroups = {"D1": d1_ids, "D2": d2_ids}
        return bg


# ---------------------------------------------------------------------------
# Regional Brain
# ---------------------------------------------------------------------------

@dataclass
class RegionalBrain:
    """Composes all brain regions in a single MolecularNeuralNetwork.

    Inter-region projections:
      - Thalamus relay → Cortex L4 (feedforward sensory)
      - Thalamus relay → Cortex L5 (fast sensorimotor shortcut)
      - Cortex L6 → Thalamus relay (corticothalamic feedback)
      - Cortex L5 → Basal Ganglia D1 (action commands)
      - Cortex L2/3 → Hippocampus DG (episodic encoding)
      - Hippocampus CA1 → Cortex L5 (memory-guided behavior)
    """

    network: MolecularNeuralNetwork = field(init=False)
    cortex: CorticalColumn = field(init=False)
    thalamus: ThalamicNucleus = field(init=False)
    hippocampus: Hippocampus = field(init=False)
    basal_ganglia: BasalGanglia = field(init=False)
    # Additional cortical columns for large brains
    extra_cortices: List[CorticalColumn] = field(init=False, default_factory=list)

    def all_neuron_ids(self) -> List[int]:
        """All neuron IDs across all regions."""
        ids = (
            self.cortex.neuron_ids
            + self.thalamus.neuron_ids
            + self.hippocampus.neuron_ids
            + self.basal_ganglia.neuron_ids
        )
        for col in self.extra_cortices:
            ids += col.neuron_ids
        return ids

    @classmethod
    def minimal(cls, seed: int = 42) -> "RegionalBrain":
        """Build a minimal regional brain (~120 neurons)."""
        np.random.seed(seed)
        brain = cls()
        # Create empty network (no initial random neurons)
        brain.network = MolecularNeuralNetwork(
            initial_neurons=0, full_brain=True,
            size=(10.0, 10.0, 10.0),
        )
        # Override circadian for faster dynamics
        if brain.network._circadian is not None:
            brain.network._circadian.clock.time_scale = 10000.0

        brain.cortex = CorticalColumn.build(
            brain.network, origin=(0.0, 0.0, 0.0), n_per_layer=10,
        )
        brain.thalamus = ThalamicNucleus.build(
            brain.network, origin=(5.0, 0.0, 0.0), n_relay=8, n_reticular=4,
        )
        brain.hippocampus = Hippocampus.build(
            brain.network, origin=(0.0, 5.0, 0.0), n_dg=10, n_ca3=8, n_ca1=6,
        )
        brain.basal_ganglia = BasalGanglia.build(
            brain.network, origin=(5.0, 5.0, 0.0), n_d1=6, n_d2=6,
        )

        brain._wire_inter_region_projections()
        return brain

    @classmethod
    def standard(cls, seed: int = 42) -> "RegionalBrain":
        """Build a standard regional brain (~260 neurons)."""
        np.random.seed(seed)
        brain = cls()
        brain.network = MolecularNeuralNetwork(
            initial_neurons=0, full_brain=True,
            size=(12.0, 12.0, 12.0),
        )
        if brain.network._circadian is not None:
            brain.network._circadian.clock.time_scale = 10000.0

        brain.cortex = CorticalColumn.build(
            brain.network, origin=(0.0, 0.0, 0.0), n_per_layer=20,
        )
        brain.thalamus = ThalamicNucleus.build(
            brain.network, origin=(6.0, 0.0, 0.0), n_relay=15, n_reticular=8,
        )
        brain.hippocampus = Hippocampus.build(
            brain.network, origin=(0.0, 6.0, 0.0), n_dg=20, n_ca3=15, n_ca1=12,
        )
        brain.basal_ganglia = BasalGanglia.build(
            brain.network, origin=(6.0, 6.0, 0.0), n_d1=12, n_d2=12,
        )

        brain._wire_inter_region_projections()
        return brain

    @classmethod
    def large(cls, seed: int = 42) -> "RegionalBrain":
        """Build a large regional brain (~800-1000 neurons).

        4 cortical columns (sensory, motor, association, prefrontal) with
        inter-column corticocortical projections, larger subcortical regions.
        """
        np.random.seed(seed)
        brain = cls()
        brain.network = MolecularNeuralNetwork(
            initial_neurons=0, full_brain=True,
            size=(20.0, 20.0, 15.0),
        )
        if brain.network._circadian is not None:
            brain.network._circadian.clock.time_scale = 10000.0

        # Primary sensory cortex (receives thalamic input)
        brain.cortex = CorticalColumn.build(
            brain.network, origin=(0.0, 0.0, 0.0), n_per_layer=20,
            name="sensory_cortex",
        )

        # Motor cortex
        motor = CorticalColumn.build(
            brain.network, origin=(5.0, 0.0, 0.0), n_per_layer=20,
            name="motor_cortex",
        )

        # Association cortex
        assoc = CorticalColumn.build(
            brain.network, origin=(10.0, 0.0, 0.0), n_per_layer=20,
            name="association_cortex",
        )

        # Prefrontal cortex
        pfc = CorticalColumn.build(
            brain.network, origin=(15.0, 0.0, 0.0), n_per_layer=20,
            name="prefrontal_cortex",
        )

        brain.extra_cortices = [motor, assoc, pfc]

        # Larger thalamus
        brain.thalamus = ThalamicNucleus.build(
            brain.network, origin=(7.0, 8.0, 0.0),
            n_relay=40, n_reticular=20,
        )

        # Larger hippocampus
        brain.hippocampus = Hippocampus.build(
            brain.network, origin=(0.0, 10.0, 0.0),
            n_dg=60, n_ca3=40, n_ca1=30,
        )

        # Larger basal ganglia
        brain.basal_ganglia = BasalGanglia.build(
            brain.network, origin=(14.0, 10.0, 0.0),
            n_d1=30, n_d2=30,
        )

        # Standard inter-region wiring (primary cortex ↔ subcortical)
        brain._wire_inter_region_projections()

        # Inter-column corticocortical projections (sparse, long-range)
        net = brain.network
        all_cols = [brain.cortex, motor, assoc, pfc]
        for i, src_col in enumerate(all_cols):
            for j, tgt_col in enumerate(all_cols):
                if i == j:
                    continue
                # L2/3 → L2/3 (associational) and L5 → L5 (commissural)
                _connect_layers(net, src_col.get_ids("L2/3"),
                                tgt_col.get_ids("L2/3"), p=0.1, nt="glutamate")
                _connect_layers(net, src_col.get_ids("L5"),
                                tgt_col.get_ids("L5"), p=0.08, nt="glutamate")

        # Motor cortex L5 → Basal Ganglia (action commands from motor cortex)
        bg_d1 = brain.basal_ganglia.get_ids("D1")
        bg_d2 = brain.basal_ganglia.get_ids("D2")
        _connect_layers(net, motor.get_ids("L5"), bg_d1, p=0.2, nt="glutamate")
        _connect_layers(net, motor.get_ids("L5"), bg_d2, p=0.1, nt="glutamate")

        # PFC L5 → Hippocampus CA1 (top-down control)
        _connect_layers(net, pfc.get_ids("L5"),
                        brain.hippocampus.get_ids("CA1"), p=0.1, nt="glutamate")

        # Association cortex L2/3 → Hippocampus DG (polymodal encoding)
        _connect_layers(net, assoc.get_ids("L2/3"),
                        brain.hippocampus.get_ids("DG"), p=0.1, nt="glutamate")

        return brain

    @classmethod
    def xlarge(cls, seed: int = 42) -> "RegionalBrain":
        """Build an extra-large regional brain (~1200+ neurons).

        6 cortical columns with 30 neurons per layer, expanded subcortical
        regions. Designed for behavioral demos at meaningful scale.
        """
        np.random.seed(seed)
        brain = cls()
        brain.network = MolecularNeuralNetwork(
            initial_neurons=0, full_brain=True,
            size=(30.0, 30.0, 20.0),
        )
        if brain.network._circadian is not None:
            brain.network._circadian.clock.time_scale = 10000.0

        n_per = 35

        brain.cortex = CorticalColumn.build(
            brain.network, origin=(0.0, 0.0, 0.0), n_per_layer=n_per,
            name="sensory_cortex",
        )
        motor = CorticalColumn.build(
            brain.network, origin=(6.0, 0.0, 0.0), n_per_layer=n_per,
            name="motor_cortex",
        )
        assoc = CorticalColumn.build(
            brain.network, origin=(12.0, 0.0, 0.0), n_per_layer=n_per,
            name="association_cortex",
        )
        pfc = CorticalColumn.build(
            brain.network, origin=(18.0, 0.0, 0.0), n_per_layer=n_per,
            name="prefrontal_cortex",
        )
        somato = CorticalColumn.build(
            brain.network, origin=(24.0, 0.0, 0.0), n_per_layer=n_per,
            name="somatosensory_cortex",
        )
        auditory = CorticalColumn.build(
            brain.network, origin=(0.0, 6.0, 0.0), n_per_layer=n_per,
            name="auditory_cortex",
        )

        brain.extra_cortices = [motor, assoc, pfc, somato, auditory]

        brain.thalamus = ThalamicNucleus.build(
            brain.network, origin=(10.0, 12.0, 0.0),
            n_relay=70, n_reticular=35,
        )
        brain.hippocampus = Hippocampus.build(
            brain.network, origin=(0.0, 15.0, 0.0),
            n_dg=95, n_ca3=65, n_ca1=55,
        )
        brain.basal_ganglia = BasalGanglia.build(
            brain.network, origin=(20.0, 15.0, 0.0),
            n_d1=55, n_d2=55,
        )

        brain._wire_inter_region_projections()

        # Inter-column projections (sparse)
        net = brain.network
        all_cols = [brain.cortex, motor, assoc, pfc, somato, auditory]
        for i, src_col in enumerate(all_cols):
            for j, tgt_col in enumerate(all_cols):
                if i == j:
                    continue
                _connect_layers(net, src_col.get_ids("L2/3"),
                                tgt_col.get_ids("L2/3"), p=0.08, nt="glutamate")
                _connect_layers(net, src_col.get_ids("L5"),
                                tgt_col.get_ids("L5"), p=0.05, nt="glutamate")

        # Motor → BG
        bg_d1 = brain.basal_ganglia.get_ids("D1")
        bg_d2 = brain.basal_ganglia.get_ids("D2")
        _connect_layers(net, motor.get_ids("L5"), bg_d1, p=0.15, nt="glutamate")
        _connect_layers(net, motor.get_ids("L5"), bg_d2, p=0.08, nt="glutamate")

        # PFC → Hippocampus
        _connect_layers(net, pfc.get_ids("L5"),
                        brain.hippocampus.get_ids("CA1"), p=0.08, nt="glutamate")

        # Association → Hippocampus
        _connect_layers(net, assoc.get_ids("L2/3"),
                        brain.hippocampus.get_ids("DG"), p=0.08, nt="glutamate")

        return brain

    def _wire_inter_region_projections(self) -> None:
        """Create inter-region synaptic projections."""
        net = self.network

        # Thalamus relay → Cortex L4 (feedforward sensory drive)
        thal_relay = self.thalamus.get_ids("relay")
        cortex_l4 = self.cortex.get_ids("L4")
        _connect_layers(net, thal_relay, cortex_l4, p=0.3, nt="glutamate")

        # Thalamus relay → Cortex L5 (fast sensorimotor shortcut for motor output)
        # This provides a direct pathway for retina→relay→L5→motor without waiting
        # for the full L4→L2/3→L5 cascade (critical for real-time action)
        _connect_layers(net, thal_relay, cortex_l5, p=0.15, nt="glutamate")

        # Cortex L6 → Thalamus relay (corticothalamic feedback)
        cortex_l6 = self.cortex.get_ids("L6")
        _connect_layers(net, cortex_l6, thal_relay, p=0.2, nt="glutamate")

        # Cortex L5 → Basal Ganglia D1 (action commands)
        cortex_l5 = self.cortex.get_ids("L5")
        bg_d1 = self.basal_ganglia.get_ids("D1")
        _connect_layers(net, cortex_l5, bg_d1, p=0.25, nt="glutamate")

        # Cortex L2/3 → Hippocampus DG (episodic encoding)
        cortex_l23 = self.cortex.get_ids("L2/3")
        hipp_dg = self.hippocampus.get_ids("DG")
        _connect_layers(net, cortex_l23, hipp_dg, p=0.2, nt="glutamate")

        # Hippocampus CA1 → Cortex L5 (memory-guided behavior)
        hipp_ca1 = self.hippocampus.get_ids("CA1")
        _connect_layers(net, hipp_ca1, cortex_l5, p=0.2, nt="glutamate")

    def step(self, dt: float = 0.1) -> None:
        """Step the entire brain."""
        self.network.step(dt)

    def stimulate_thalamus(self, intensity: float = 30.0) -> None:
        """Stimulate thalamic relay (sensory input).

        HH neurons need ~20-30 µA/cm² sustained input to spike reliably.
        """
        for nid in self.thalamus.get_ids("relay"):
            self.network._external_currents[nid] = (
                self.network._external_currents.get(nid, 0.0) + intensity
            )

    def read_cortex_output(self) -> float:
        """Read mean L5 activity (cortical output)."""
        l5_ids = self.cortex.get_ids("L5")
        if not l5_ids:
            return 0.0
        total = 0.0
        for nid in l5_ids:
            mol_n = self.network._molecular_neurons.get(nid)
            if mol_n is not None:
                norm_v = (mol_n.membrane.voltage + 70.0) / 90.0
                total += max(0.0, min(1.0, norm_v))
        return total / len(l5_ids)

    def visualize_regions(self) -> str:
        """ASCII visualization showing region-colored neurons."""
        lines = [f"RegionalBrain: {len(self.network._molecular_neurons)} neurons, "
                 f"{len(self.network._molecular_synapses)} synapses, "
                 f"t={self.network.time:.1f}ms"]

        region_map = {}
        for nid in self.cortex.neuron_ids:
            region_map[nid] = "C"
        for col in self.extra_cortices:
            for nid in col.neuron_ids:
                region_map[nid] = "C"
        for nid in self.thalamus.neuron_ids:
            region_map[nid] = "T"
        for nid in self.hippocampus.neuron_ids:
            region_map[nid] = "H"
        for nid in self.basal_ganglia.neuron_ids:
            region_map[nid] = "B"

        active = sum(1 for n in self.network._molecular_neurons.values() if n.is_active)
        fired = len(self.network.last_fired)
        n_cortex = len(self.cortex.neuron_ids) + sum(
            len(c.neuron_ids) for c in self.extra_cortices
        )
        lines.append(f"Active: {active}  Fired: {fired}")
        lines.append(f"Regions: C=Cortex({n_cortex}) "
                     f"T=Thalamus({len(self.thalamus.neuron_ids)}) "
                     f"H=Hippocampus({len(self.hippocampus.neuron_ids)}) "
                     f"B=BasalGanglia({len(self.basal_ganglia.neuron_ids)})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

def _connect_layers(
    network: MolecularNeuralNetwork,
    source_ids: List[int],
    target_ids: List[int],
    p: float = 0.3,
    nt: str = "glutamate",
) -> None:
    """Probabilistically connect neurons between two groups."""
    for src in source_ids:
        for tgt in target_ids:
            if src != tgt and np.random.random() < p:
                network.create_synapse(src, tgt, nt)


def _connect_within(
    network: MolecularNeuralNetwork,
    ids: List[int],
    p: float = 0.2,
    nt: str = "glutamate",
) -> None:
    """Probabilistically connect neurons within a group."""
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if np.random.random() < p:
                network.create_synapse(a, b, nt)
            if np.random.random() < p:
                network.create_synapse(b, a, nt)
