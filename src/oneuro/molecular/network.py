"""MolecularNeuralNetwork — extends OrganicNeuralNetwork with molecular neurons.

Overrides neuron/synapse creation to use molecular versions.
Tracks global neurotransmitter concentrations.
Existing training tasks (XOR, Pattern, etc.) work unchanged.

Phase 3 additions (all optional, enabled via flags):
- Astrocytes, oligodendrocytes, microglia (glia.py)
- Gap junctions for electrical synapses (gap_junction.py)
- 3D extracellular space with volume transmission (extracellular.py)
- Axonal conduction with myelination (axon.py)
- Circadian rhythms and sleep homeostasis (circadian.py)
- Perineuronal nets restricting plasticity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from oneuro.molecular.neuron import MolecularNeuron, NeuronArchetype
from oneuro.molecular.synapse import MolecularSynapse
from oneuro.molecular.adapters import MolecularNeuronAdapter, MolecularSynapseAdapter
from oneuro.molecular.ion_channels import (
    IonChannelType, BatchIonChannelState,
    _alpha_m_vec, _beta_m_vec, _alpha_h_vec, _beta_h_vec,
    _alpha_n_vec, _beta_n_vec,
)


@dataclass
class MolecularNeuralNetwork:
    """A neural network where neurons are molecular assemblies.

    Mirrors OrganicNeuralNetwork's interface so existing tasks work unchanged.
    Neurons have emergent membrane potentials from ion channel physics.
    Synapses transmit via real neurotransmitter release and receptor binding.
    """

    size: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    initial_neurons: int = 20
    energy_supply: float = 1.0
    default_archetype: NeuronArchetype = NeuronArchetype.PYRAMIDAL

    # Feature flags — enable Phase 3 subsystems
    enable_glia: bool = False
    enable_gap_junctions: bool = False
    enable_extracellular: bool = False
    enable_axons: bool = False
    enable_circadian: bool = False
    enable_advanced_neurons: bool = False  # Calcium, second messengers, metabolism, cytoskeleton

    # Convenience: enable ALL subsystems at once
    full_brain: bool = False

    # Postsynaptic current scale (µA/cm²) per synapse weight unit.
    # Compensates for small-network NT pathway underestimation.
    psc_scale: float = 30.0

    # Internal state
    neurons: Dict[int, MolecularNeuronAdapter] = field(init=False, default_factory=dict)
    synapses: Dict[Tuple[int, int], MolecularSynapseAdapter] = field(init=False, default_factory=dict)
    _molecular_neurons: Dict[int, MolecularNeuron] = field(init=False, default_factory=dict)
    _molecular_synapses: Dict[Tuple[int, int], MolecularSynapse] = field(init=False, default_factory=dict)

    # Pre-indexed outgoing synapse lookup: pre_id → [(post_id, synapse), ...]
    # Turns O(fired × total_synapses) → O(fired × avg_outgoing_degree)
    _outgoing: Dict[int, List[Tuple[int, MolecularSynapse]]] = field(init=False, default_factory=dict)

    # Active synapses: keys of synapses with non-zero cleft concentration.
    # Only these need per-step updates; inactive ones have zero cleft and idle vesicles.
    _active_synapses: Set[Tuple[int, int]] = field(init=False, default_factory=set)

    # Track when each neuron last fired (step count) for lazy subsystem updates
    _neuron_last_fired: Dict[int, int] = field(init=False, default_factory=dict)

    # Per-step ECS concentration cache: voxel (ix,iy,iz) → {nt_name: conc}
    _ecs_cache: Dict[Tuple[int, int, int], Dict[str, float]] = field(init=False, default_factory=dict)

    # Simulation state
    time: float = field(init=False, default=0.0)
    neuron_counter: int = field(init=False, default=0)
    generation: int = field(init=False, default=0)
    spike_count: int = field(init=False, default=0)
    neurogenesis_events: int = field(init=False, default=0)
    pruning_events: int = field(init=False, default=0)
    entanglement_events: int = field(init=False, default=0)

    # Global neurotransmitter concentrations (ambient, in nM)
    global_nt_concentrations: Dict[str, float] = field(init=False, default_factory=lambda: {
        "dopamine": 20.0,
        "serotonin": 10.0,
        "norepinephrine": 15.0,
        "acetylcholine": 50.0,
        "gaba": 200.0,
        "glutamate": 500.0,
    })

    # Learning state (matching OrganicNeuralNetwork)
    dopamine_level: float = field(init=False, default=0.0)
    dopamine_decay: float = 0.9
    learning_rate: float = 0.1
    task_performance: Dict[str, List[float]] = field(init=False, default_factory=dict)

    # Structural plasticity thresholds
    performance_threshold_grow: float = 0.7
    performance_threshold_prune: float = 0.3

    # I/O regions (matching OrganicNeuralNetwork)
    input_regions: Dict[str, Tuple[Tuple[float, float, float], float]] = field(
        init=False, default_factory=dict
    )
    output_regions: Dict[str, Tuple[Tuple[float, float, float], float]] = field(
        init=False, default_factory=dict
    )

    # ---- Phase 3 subsystems ----
    _astrocytes: Dict[int, object] = field(init=False, default_factory=dict)
    _oligodendrocytes: Dict[int, object] = field(init=False, default_factory=dict)
    _microglia: Dict[int, object] = field(init=False, default_factory=dict)
    _gap_junctions: List[object] = field(init=False, default_factory=list)
    _axons: Dict[Tuple[int, int], object] = field(init=False, default_factory=dict)
    _extracellular: Optional[object] = field(init=False, default=None)
    _circadian: Optional[object] = field(init=False, default=None)
    _perineuronal_net: Optional[object] = field(init=False, default=None)
    _glia_counter: int = field(init=False, default=0)

    # Last set of neurons that fired (for ConsciousnessMonitor)
    _last_fired: Set[int] = field(init=False, default_factory=set)

    @property
    def last_fired(self) -> Set[int]:
        """Neurons that fired on the most recent step (read-only)."""
        return self._last_fired

    def __post_init__(self):
        # full_brain=True enables all subsystems
        if self.full_brain:
            self.enable_glia = True
            self.enable_gap_junctions = True
            self.enable_extracellular = True
            self.enable_axons = True
            self.enable_circadian = True
            self.enable_advanced_neurons = True

        self._size = np.array(self.size)
        self._create_initial_neurons()
        self._create_initial_synapses()

        # Initialize Phase 3 subsystems
        if self.enable_extracellular:
            self._init_extracellular()
        if self.enable_glia:
            self._init_glia()
        if self.enable_circadian:
            self._init_circadian()

    def _init_extracellular(self) -> None:
        """Initialize 3D extracellular space."""
        try:
            from oneuro.molecular.extracellular import ExtracellularSpace, PerineuronalNet
            # Grid resolution: ~10 µm per voxel
            grid = tuple(max(3, int(s)) for s in self.size)
            self._extracellular = ExtracellularSpace(grid_size=grid)
            self._perineuronal_net = PerineuronalNet()
        except ImportError:
            self.enable_extracellular = False

    def _init_glia(self) -> None:
        """Create initial glial cells."""
        try:
            from oneuro.molecular.glia import Astrocyte, Oligodendrocyte, Microglia
            n_synapses = len(self._molecular_synapses)

            # ~1 astrocyte per 5 synapses
            n_astrocytes = max(1, n_synapses // 5)
            for i in range(n_astrocytes):
                pos = np.random.uniform([0, 0, 0], self._size)
                aid = self._glia_counter
                self._glia_counter += 1
                astro = Astrocyte(id=aid, x=pos[0], y=pos[1], z=pos[2])
                self._astrocytes[aid] = astro

            # A few oligodendrocytes
            n_oligos = max(1, len(self._molecular_neurons) // 10)
            for i in range(n_oligos):
                pos = np.random.uniform([0, 0, 0], self._size)
                oid = self._glia_counter
                self._glia_counter += 1
                oligo = Oligodendrocyte(id=oid, x=pos[0], y=pos[1], z=pos[2])
                self._oligodendrocytes[oid] = oligo

            # Patrolling microglia
            n_microglia = max(1, len(self._molecular_neurons) // 15)
            for i in range(n_microglia):
                pos = np.random.uniform([0, 0, 0], self._size)
                mid = self._glia_counter
                self._glia_counter += 1
                micro = Microglia(id=mid, x=pos[0], y=pos[1], z=pos[2])
                self._microglia[mid] = micro

        except ImportError:
            self.enable_glia = False

    def _init_circadian(self) -> None:
        """Initialize circadian clock."""
        try:
            from oneuro.molecular.circadian import CircadianSystem
            self._circadian = CircadianSystem()
        except ImportError:
            self.enable_circadian = False

    def _create_initial_neurons(self) -> None:
        for i in range(self.initial_neurons):
            pos = np.random.uniform([0, 0, 0], self._size)
            self._add_neuron(pos[0], pos[1], pos[2])

    def _create_initial_synapses(self) -> None:
        """Connect nearby neurons probabilistically."""
        ids = list(self._molecular_neurons.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                n_a = self._molecular_neurons[id_a]
                n_b = self._molecular_neurons[id_b]
                dist = math.sqrt(
                    (n_a.x - n_b.x) ** 2 + (n_a.y - n_b.y) ** 2 + (n_a.z - n_b.z) ** 2
                )
                max_dist = np.linalg.norm(self._size)
                prob = max(0, 0.3 * (1.0 - dist / (max_dist * 0.5)))
                if np.random.random() < prob:
                    self._add_synapse(id_a, id_b)
                if np.random.random() < prob * 0.5:
                    self._add_synapse(id_b, id_a)

    def _add_neuron(
        self, x: float, y: float, z: float,
        archetype: Optional[NeuronArchetype] = None,
    ) -> int:
        nid = self.neuron_counter
        self.neuron_counter += 1
        mol = MolecularNeuron(
            id=nid, x=x, y=y, z=z,
            archetype=archetype or self.default_archetype,
        )

        # Attach advanced subsystems if enabled
        if self.enable_advanced_neurons:
            self._attach_advanced_subsystems(mol)

        self._molecular_neurons[nid] = mol
        self.neurons[nid] = MolecularNeuronAdapter(_mol=mol)
        return nid

    def _attach_advanced_subsystems(self, neuron: MolecularNeuron) -> None:
        """Attach calcium, second messengers, metabolism, cytoskeleton to a neuron."""
        try:
            from oneuro.molecular.calcium import CalciumSystem
            from oneuro.molecular.second_messengers import SecondMessengerSystem
            from oneuro.molecular.metabolism import CellularMetabolism
            from oneuro.molecular.microtubules import Cytoskeleton, Microtubule

            # Orch-OR network scale: our N neurons represent a cortical column
            # of ~100k neurons.  E_G uses N^2 scaling for coherent superposition.
            # The total column has ~19.2M superposed tubulins (100k × 3 MTs × 64).
            # We distribute the column's total E_G across our network's MTs so
            # OR events fire at biologically relevant rates (theta-alpha band).
            n_column_neurons = 1e5
            n_sup_typical = 64  # superposed tubulins per MT at steady state
            n_column_total = n_column_neurons * 3 * n_sup_typical  # ~19.2M
            n_sim_mts = max(1, (len(self._molecular_neurons) + 1) * 3)
            orch_or_scale = n_column_total ** 2 / (n_sup_typical ** 2 * n_sim_mts)

            cyto = Cytoskeleton()
            for mt in cyto.microtubules:
                mt.network_tubulin_scale = orch_or_scale

            neuron.attach_subsystems(
                calcium_system=CalciumSystem(),
                second_messenger_system=SecondMessengerSystem(),
                metabolism=CellularMetabolism(),
                cytoskeleton=cyto,
            )
        except ImportError:
            pass

    def _add_synapse(self, pre_id: int, post_id: int, nt_name: str = "glutamate") -> None:
        key = (pre_id, post_id)
        if key in self._molecular_synapses:
            return
        mol_syn = MolecularSynapse(pre_neuron=pre_id, post_neuron=post_id, nt_name=nt_name)

        # Attach dendritic spine for structural plasticity
        if self.enable_advanced_neurons:
            try:
                from oneuro.molecular.spine import DendriticSpine
                mol_syn.spine = DendriticSpine()
            except ImportError:
                pass

        self._molecular_synapses[key] = mol_syn
        self.synapses[key] = MolecularSynapseAdapter(_mol=mol_syn)

        # Maintain outgoing index
        if pre_id not in self._outgoing:
            self._outgoing[pre_id] = []
        self._outgoing[pre_id].append((post_id, mol_syn))

        # Create axon if enabled
        if self.enable_axons:
            self._create_axon(pre_id, post_id, mol_syn)

        # Update connectivity
        if pre_id in self._molecular_neurons:
            self._molecular_neurons[pre_id].outputs.add(post_id)
        if post_id in self._molecular_neurons:
            self._molecular_neurons[post_id].inputs.add(pre_id)

    def _create_axon(self, pre_id: int, post_id: int, synapse: MolecularSynapse) -> None:
        """Create an axon for a synapse with distance-based properties."""
        try:
            from oneuro.molecular.axon import Axon
            pre = self._molecular_neurons.get(pre_id)
            post = self._molecular_neurons.get(post_id)
            if pre is None or post is None:
                return
            dist_um = math.sqrt(
                (pre.x - post.x) ** 2 + (pre.y - post.y) ** 2 + (pre.z - post.z) ** 2
            ) * 100.0  # Scale to µm

            # Long-range connections get myelinated
            myelinated = dist_um > 500.0
            axon = Axon.from_distance(dist_um, myelinated=myelinated, diameter_um=1.0)
            self._axons[(pre_id, post_id)] = axon

            # Replace fixed delay with axon propagation delay
            synapse.delay = axon.propagation_delay()
        except ImportError:
            pass

    # ---- Public neuron/synapse creation (for brain regions) ----

    def create_neuron_at(
        self, x: float, y: float, z: float,
        archetype: Optional[NeuronArchetype] = None,
    ) -> int:
        """Create a neuron at a specific position. Returns neuron ID."""
        return self._add_neuron(x, y, z, archetype)

    def create_synapse(
        self, pre_id: int, post_id: int, nt_name: str = "glutamate",
    ) -> None:
        """Create a synapse between two neurons."""
        self._add_synapse(pre_id, post_id, nt_name)

    # Vectorization threshold
    _vectorize_threshold: int = 20
    _step_count: int = field(init=False, default=0)

    # ---- Main simulation step ----

    def step(self, dt: float = 0.1) -> None:
        """Advance the entire network by dt ms.

        Auto-dispatches to step_vectorized() when neuron count exceeds threshold.
        """
        if len(self._molecular_neurons) > self._vectorize_threshold:
            self.step_vectorized(dt)
            return

        self.time += dt
        self._step_count += 1
        self._ecs_cache.clear()

        # 0. Circadian modulation (TTFL ODE → modulation factors)
        circadian_mod = self._circadian_step(dt)

        # 0b. Apply circadian NT synthesis modulation to ambient levels
        # The TTFL protein concentrations directly modulate how much NT is available
        nt_synth_mod = circadian_mod.get("nt_synthesis", 1.0)
        if nt_synth_mod != 1.0:
            for nt_name in self.global_nt_concentrations:
                self.global_nt_concentrations[nt_name] *= nt_synth_mod

        # 1. Distribute energy
        self._distribute_energy()

        # 1b. Extracellular diffusion + transporter uptake (before synapse
        #     updates so astrocytes see current-step NT concentrations)
        if self._extracellular is not None:
            self._extracellular.step(dt)

        # 2. Update active synapses (cleft dynamics, vesicle replenishment)
        # Only iterate synapses with non-zero cleft concentration (+ all for vesicle replenish)
        alertness_mod = circadian_mod.get("alertness", 1.0)
        synapse_nt: Dict[int, Dict[str, float]] = {}
        newly_inactive: List[Tuple[int, int]] = []
        for key in list(self._active_synapses):
            mol_syn = self._molecular_synapses.get(key)
            if mol_syn is None:
                newly_inactive.append(key)
                continue
            pre_id, post_id = key
            conc = mol_syn.update(self.time, dt)
            conc *= alertness_mod
            if conc > 0:
                if post_id not in synapse_nt:
                    synapse_nt[post_id] = {}
                nt = mol_syn.nt_name
                synapse_nt[post_id][nt] = synapse_nt[post_id].get(nt, 0.0) + conc

                if self._extracellular is not None:
                    pre_n = self._molecular_neurons.get(pre_id)
                    if pre_n is not None:
                        self._extracellular.release_at(
                            pre_n.x, pre_n.y, pre_n.z, nt, conc * 0.1
                        )
            else:
                newly_inactive.append(key)
        for key in newly_inactive:
            self._active_synapses.discard(key)

        # Bulk vesicle replenishment for inactive synapses (every 10 steps)
        if self._step_count % 10 == 0:
            for key, mol_syn in self._molecular_synapses.items():
                if key not in self._active_synapses:
                    mol_syn.vesicle_pool.replenish(dt * 10.0)

        # 2b. Gap junction currents
        gap_currents: Dict[int, float] = {}
        if self.enable_gap_junctions:
            gap_currents = self._compute_gap_junction_currents(dt)

        # 3. Update all neurons
        # Circadian excitability → depolarizing/hyperpolarizing bias current
        # from TTFL CLOCK:BMAL1 level (±2.5 µA/cm²)
        excitability_bias = (circadian_mod.get("excitability", 1.0) - 1.0) * 5.0
        # Adenosine A1 receptor inhibition (hyperpolarizing)
        excitability_bias -= circadian_mod.get("adenosine_inhibition", 0.0)

        fired_neurons: Set[int] = set()
        step_count = self._step_count
        for nid, mol_neuron in self._molecular_neurons.items():
            if not mol_neuron.alive:
                continue

            # Pass ATP state to membrane before step (metabolism → channel gating)
            if mol_neuron.metabolism is not None:
                mol_neuron.membrane._atp_ok = mol_neuron.metabolism.atp_available
            else:
                mol_neuron.membrane._atp_ok = True

            # Get NT concentrations from synapses + ambient
            if self._extracellular is not None:
                nt_concs = self._get_local_nt(mol_neuron)
            else:
                nt_concs = dict(self.global_nt_concentrations)

            if nid in synapse_nt:
                for nt, conc in synapse_nt[nid].items():
                    nt_concs[nt] = nt_concs.get(nt, 0.0) + conc

            # External current from stimulation + gap junctions + circadian excitability
            ext_current = self._get_external_current(nid)
            ext_current += gap_currents.get(nid, 0.0)
            ext_current += excitability_bias

            # Skip slow subsystems for neurons inactive >50 steps
            skip_slow = (step_count - self._neuron_last_fired.get(nid, 0)) > 50

            # Cytoskeleton consciousness → small excitability boost
            if not skip_slow and mol_neuron.cytoskeleton is not None:
                ext_current += mol_neuron.cytoskeleton.consciousness_measure * 0.5

            if mol_neuron.update(nt_concentrations=nt_concs, external_current=ext_current, dt=dt,
                                 skip_slow_subsystems=skip_slow, step_count=step_count):
                fired_neurons.add(nid)
                self._neuron_last_fired[nid] = step_count
                self.spike_count += 1

        # Store fired neurons for consciousness monitor
        self._last_fired = fired_neurons

        # 4. Propagate spikes to synapses (using pre-indexed _outgoing)
        for nid in fired_neurons:
            pre_n = self._molecular_neurons[nid]
            ca = pre_n.membrane.ca_internal
            for post, mol_syn in self._outgoing.get(nid, []):
                mol_syn.presynaptic_spike(self.time, ca_level_nM=ca)
                # Mark synapse as active (has cleft NT to process)
                self._active_synapses.add((nid, post))
                # Vesicle recycling ATP cost (1 release event per spike)
                if pre_n.metabolism is not None:
                    pre_n.metabolism.vesicle_recycling_cost(0.1, 1.0)
                # Fast postsynaptic current (models AMPA-mediated EPSC/IPSC).
                psc = mol_syn.weight * self.psc_scale
                if mol_syn.nt_name == "GABA":
                    psc = -psc  # Inhibitory
                self._external_currents[post] = (
                    self._external_currents.get(post, 0.0) + psc
                )

        # 5. STDP (with NMDA gating from postsynaptic neuron)
        self._update_stdp(fired_neurons, dt)

        # 6. Glial cell updates
        if self.enable_glia:
            self._update_glia(dt, fired_neurons)

        # 7. Restore ambient NT levels (undo circadian scaling to prevent drift)
        if nt_synth_mod != 1.0:
            for nt_name in self.global_nt_concentrations:
                self.global_nt_concentrations[nt_name] /= nt_synth_mod

        # 8. Spontaneous synaptogenesis (1% chance)
        if np.random.random() < 0.01:
            self._spontaneous_synaptogenesis()

        # 9. Cleanup dead neurons and prunable synapses
        self._cleanup()

    def step_vectorized(self, dt: float = 0.1) -> None:
        """Vectorised network step — batches HH channel updates across all neurons."""
        self.time += dt
        self._step_count += 1
        self._ecs_cache.clear()

        # 0. Circadian (TTFL ODE → modulation factors)
        circadian_mod = self._circadian_step(dt)

        # 0b. Apply circadian NT synthesis modulation to ambient levels
        nt_synth_mod = circadian_mod.get("nt_synthesis", 1.0)
        if nt_synth_mod != 1.0:
            for nt_name in self.global_nt_concentrations:
                self.global_nt_concentrations[nt_name] *= nt_synth_mod

        # 1. Distribute energy
        self._distribute_energy()

        # 1b. Extracellular diffusion + transporter uptake (before synapse
        #     updates so astrocytes see current-step NT concentrations)
        if self._extracellular is not None:
            self._extracellular.step(dt)

        # 2. Update active synapses
        alertness_mod = circadian_mod.get("alertness", 1.0)
        synapse_nt: Dict[int, Dict[str, float]] = {}
        newly_inactive_v: List[Tuple[int, int]] = []
        for key in list(self._active_synapses):
            mol_syn = self._molecular_synapses.get(key)
            if mol_syn is None:
                newly_inactive_v.append(key)
                continue
            pre_id, post_id = key
            conc = mol_syn.update(self.time, dt)
            conc *= alertness_mod
            if conc > 0:
                if post_id not in synapse_nt:
                    synapse_nt[post_id] = {}
                nt = mol_syn.nt_name
                synapse_nt[post_id][nt] = synapse_nt[post_id].get(nt, 0.0) + conc

                if self._extracellular is not None:
                    pre_n = self._molecular_neurons.get(pre_id)
                    if pre_n is not None:
                        self._extracellular.release_at(
                            pre_n.x, pre_n.y, pre_n.z, nt, conc * 0.1
                        )
            else:
                newly_inactive_v.append(key)
        for key in newly_inactive_v:
            self._active_synapses.discard(key)

        # Bulk vesicle replenishment for inactive synapses (every 10 steps)
        if self._step_count % 10 == 0:
            for key, mol_syn in self._molecular_synapses.items():
                if key not in self._active_synapses:
                    mol_syn.vesicle_pool.replenish(dt * 10.0)

        # 2b. Gap junctions
        gap_currents: Dict[int, float] = {}
        if self.enable_gap_junctions:
            gap_currents = self._compute_gap_junction_currents(dt)

        # 3. Batch-update voltage-gated channels, then per-neuron membrane step
        alive_ids = [nid for nid, n in self._molecular_neurons.items() if n.alive]
        alive_neurons = [self._molecular_neurons[nid] for nid in alive_ids]

        if alive_neurons:
            # Collect voltages
            voltages = np.array([n.membrane.voltage for n in alive_neurons])

            # Batch Na_v channels (m3h gating)
            na_channels = []
            for n in alive_neurons:
                ch = n.membrane.channels.get_channel(IonChannelType.Na_v)
                na_channels.append(ch)
            na_present = [ch for ch in na_channels if ch is not None]
            na_mask = np.array([ch is not None for ch in na_channels])
            if na_present:
                batch_na = BatchIonChannelState.from_channels(na_present)
                batch_na.update(voltages[na_mask], dt)
                batch_na.write_back(na_present)

            # Batch K_v channels (n4 gating)
            kv_channels = []
            for n in alive_neurons:
                ch = n.membrane.channels.get_channel(IonChannelType.K_v)
                kv_channels.append(ch)
            kv_present = [ch for ch in kv_channels if ch is not None]
            kv_mask = np.array([ch is not None for ch in kv_channels])
            if kv_present:
                batch_kv = BatchIonChannelState.from_channels(kv_present)
                batch_kv.update(voltages[kv_mask], dt)
                batch_kv.write_back(kv_present)

            # Batch Ca_v channels (m2h gating)
            ca_channels = []
            for n in alive_neurons:
                ch = n.membrane.channels.get_channel(IonChannelType.Ca_v)
                ca_channels.append(ch)
            ca_present = [ch for ch in ca_channels if ch is not None]
            ca_mask = np.array([ch is not None for ch in ca_channels])
            if ca_present:
                batch_ca = BatchIonChannelState.from_channels(ca_present)
                batch_ca.update(voltages[ca_mask], dt)
                batch_ca.write_back(ca_present)

        # Per-neuron: membrane integration, gene expression
        excitability_bias = (circadian_mod.get("excitability", 1.0) - 1.0) * 5.0
        excitability_bias -= circadian_mod.get("adenosine_inhibition", 0.0)

        fired_neurons: Set[int] = set()
        step_count = self._step_count
        for nid, mol_neuron in zip(alive_ids, alive_neurons):
            # Pass ATP state to membrane
            if mol_neuron.metabolism is not None:
                mol_neuron.membrane._atp_ok = mol_neuron.metabolism.atp_available
            else:
                mol_neuron.membrane._atp_ok = True

            if self._extracellular is not None:
                nt_concs = self._get_local_nt(mol_neuron)
            else:
                nt_concs = dict(self.global_nt_concentrations)

            if nid in synapse_nt:
                for nt, conc in synapse_nt[nid].items():
                    nt_concs[nt] = nt_concs.get(nt, 0.0) + conc

            ext_current = self._get_external_current(nid)
            ext_current += gap_currents.get(nid, 0.0)
            ext_current += excitability_bias

            # Skip slow subsystems for neurons inactive >50 steps
            skip_slow = (step_count - self._neuron_last_fired.get(nid, 0)) > 50

            # Cytoskeleton consciousness → small excitability boost
            if not skip_slow and mol_neuron.cytoskeleton is not None:
                ext_current += mol_neuron.cytoskeleton.consciousness_measure * 0.5

            if mol_neuron.update(
                nt_concentrations=nt_concs, external_current=ext_current, dt=dt,
                skip_slow_subsystems=skip_slow, step_count=step_count,
            ):
                fired_neurons.add(nid)
                self._neuron_last_fired[nid] = step_count
                self.spike_count += 1

        # Store fired neurons for consciousness monitor
        self._last_fired = fired_neurons

        # 4-9: same as scalar step (using pre-indexed _outgoing)
        for nid in fired_neurons:
            pre_n = self._molecular_neurons[nid]
            ca = pre_n.membrane.ca_internal
            for post, mol_syn in self._outgoing.get(nid, []):
                mol_syn.presynaptic_spike(self.time, ca_level_nM=ca)
                self._active_synapses.add((nid, post))
                if pre_n.metabolism is not None:
                    pre_n.metabolism.vesicle_recycling_cost(0.1, 1.0)
                psc = mol_syn.weight * self.psc_scale
                if mol_syn.nt_name == "GABA":
                    psc = -psc
                self._external_currents[post] = (
                    self._external_currents.get(post, 0.0) + psc
                )

        self._update_stdp(fired_neurons, dt)

        if self.enable_glia:
            self._update_glia(dt, fired_neurons)

        # Restore ambient NT levels (undo circadian scaling to prevent drift)
        if nt_synth_mod != 1.0:
            for nt_name in self.global_nt_concentrations:
                self.global_nt_concentrations[nt_name] /= nt_synth_mod

        if np.random.random() < 0.01:
            self._spontaneous_synaptogenesis()
        self._cleanup()

    # ---- Phase 3: Glial updates ----

    def _update_glia(self, dt: float, fired_neurons: Set[int]) -> None:
        """Update all glial cells with full bidirectional wiring."""
        # Collect ATP released by astrocytes (for microglia damage signal + adenosine)
        total_atp_released = 0.0

        # Astrocytes: glutamate uptake, gliotransmitter release
        for astro in self._astrocytes.values():
            # Find local glutamate near this astrocyte
            local_glut = 0.0
            if self._extracellular is not None:
                local_glut = self._extracellular.concentration_at(
                    astro.x, astro.y, astro.z, "glutamate"
                )
            else:
                local_glut = self.global_nt_concentrations.get("glutamate", 0.0)

            result = astro.step(dt, local_glutamate_nM=local_glut, local_k_mM=4.0)

            # Apply glutamate uptake to extracellular space
            if self._extracellular is not None and result.get("glutamate_uptake", 0) > 0:
                self._extracellular.release_at(
                    astro.x, astro.y, astro.z,
                    "glutamate", -result["glutamate_uptake"]
                )

            # Wire D-serine → extracellular space (NMDA co-agonist)
            d_serine = result.get("d_serine_release", 0.0)
            if d_serine > 0 and self._extracellular is not None:
                # D-serine acts like glycine at NMDA glycine site;
                # release as glutamate co-signal (enhances NMDA current)
                self._extracellular.release_at(
                    astro.x, astro.y, astro.z,
                    "glutamate", d_serine * 0.1  # Small co-agonist contribution
                )

            # Wire ATP release → extracellular space (purinergic signaling)
            atp = result.get("atp_release", 0.0)
            if atp > 0:
                total_atp_released += atp
                # ATP is rapidly converted to adenosine by ectoenzymes,
                # contributing to sleep homeostasis (wired via circadian)

            # Wire lactate → nearby neurons' metabolism
            lactate_out = result.get("lactate_output", 0.0)
            if lactate_out > 0 and self.enable_advanced_neurons:
                self._deliver_lactate_to_nearby_neurons(
                    astro.x, astro.y, astro.z, lactate_out
                )

        # Microglia: damage detection from dead neurons + purinergic ATP signal
        dead_positions = [
            (n.x, n.y, n.z) for n in self._molecular_neurons.values() if not n.alive
        ]
        for micro in self._microglia.values():
            # Compute damage signal from: nearby dead neurons + astrocyte ATP
            damage = 0.0
            for dx, dy, dz in dead_positions:
                dist = math.sqrt(
                    (micro.x - dx) ** 2 + (micro.y - dy) ** 2 + (micro.z - dz) ** 2
                )
                if dist < 3.0:
                    damage += 100.0 / max(0.1, dist)  # Closer = stronger signal
            # Purinergic ATP signal from astrocytes
            damage += total_atp_released * 0.01

            micro.step(dt, local_damage_signal=damage)

            # Prune tagged synapses
            pruned = micro.prune_tagged()
            for key in pruned:
                if key in self._molecular_synapses:
                    pre, post = key
                    del self._molecular_synapses[key]
                    if key in self.synapses:
                        del self.synapses[key]
                    self._active_synapses.discard(key)
                    if pre in self._outgoing:
                        self._outgoing[pre] = [
                            (p, s) for p, s in self._outgoing[pre] if p != post
                        ]
                    self.pruning_events += 1

        # Oligodendrocytes: activity-dependent myelination (every 100 steps)
        if self._oligodendrocytes and self._axons and self._step_count % 100 == 0:
            self._update_oligodendrocyte_myelination(dt * 100.0, fired_neurons)

    def _deliver_lactate_to_nearby_neurons(
        self, ax: float, ay: float, az: float, lactate_nM: float
    ) -> None:
        """Deliver astrocyte lactate to nearby neurons' metabolism."""
        radius = 3.0
        recipients = []
        for nid, mol_n in self._molecular_neurons.items():
            if not mol_n.alive or mol_n.metabolism is None:
                continue
            dist = math.sqrt((ax - mol_n.x)**2 + (ay - mol_n.y)**2 + (az - mol_n.z)**2)
            if dist < radius:
                recipients.append(mol_n)
        if recipients:
            per_neuron = lactate_nM / len(recipients) * 0.001  # nM → mM scale
            for mol_n in recipients:
                mol_n.metabolism.supply_lactate(per_neuron)

    def _update_oligodendrocyte_myelination(
        self, dt: float, fired_neurons: Set[int]
    ) -> None:
        """Dynamic myelination: active pathways get faster."""
        # Build firing rate proxy per neuron
        for (pre_id, post_id), axon in list(self._axons.items()):
            mol_syn = self._molecular_synapses.get((pre_id, post_id))
            if mol_syn is None:
                continue

            pre_n = self._molecular_neurons.get(pre_id)
            if pre_n is None:
                continue

            # Activity level: was this neuron recently active?
            activity = 1.0 if pre_id in fired_neurons else (
                0.3 if pre_n.is_active else 0.0
            )

            # Find nearest oligodendrocyte
            best_oligo = None
            best_dist = float('inf')
            for oligo in self._oligodendrocytes.values():
                dist = math.sqrt(
                    (oligo.x - pre_n.x)**2 + (oligo.y - pre_n.y)**2
                    + (oligo.z - pre_n.z)**2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_oligo = oligo

            if best_oligo is None or best_dist > 5.0:
                continue

            # Myelinate unmyelinated axons within range
            if pre_id not in best_oligo.myelin_segments:
                if best_oligo.capacity_remaining > 0:
                    best_oligo.myelinate(pre_id)

            # Mature existing myelin (activity-dependent)
            best_oligo.mature_myelin(pre_id, dt, activity_level=activity)

            # Update axon propagation delay from myelin velocity factor
            vel_factor = best_oligo.conduction_velocity_factor(pre_id)
            if vel_factor > 1.0:
                base_delay = axon.propagation_delay()
                mol_syn.delay = base_delay / vel_factor

        # Step all oligodendrocytes
        for oligo in self._oligodendrocytes.values():
            oligo.step(dt)

    def _compute_gap_junction_currents(self, dt: float) -> Dict[int, float]:
        """Compute currents from gap junctions."""
        currents: Dict[int, float] = {}
        for gj in self._gap_junctions:
            pre_n = self._molecular_neurons.get(gj.pre_id)
            post_n = self._molecular_neurons.get(gj.post_id)
            if pre_n is None or post_n is None:
                continue

            v_pre = pre_n.membrane.voltage
            v_post = post_n.membrane.voltage
            I = gj.compute_current(v_pre, v_post)

            # Bidirectional: current flows into lower-voltage cell
            currents[gj.pre_id] = currents.get(gj.pre_id, 0.0) - I * 0.001  # nA → µA/cm² scaling
            currents[gj.post_id] = currents.get(gj.post_id, 0.0) + I * 0.001

            gj.step(dt, v_pre, v_post)

        return currents

    def _get_local_nt(self, neuron: MolecularNeuron) -> Dict[str, float]:
        """Get local NT concentrations for a neuron from extracellular space.

        Uses per-step voxel cache: neurons in the same voxel share results.
        """
        voxel = self._extracellular.position_to_voxel(neuron.x, neuron.y, neuron.z)
        cached = self._ecs_cache.get(voxel)
        if cached is not None:
            return dict(cached)
        nt_concs = {}
        for nt_name in self.global_nt_concentrations:
            nt_concs[nt_name] = self._extracellular.concentration_at(
                neuron.x, neuron.y, neuron.z, nt_name
            )
        self._ecs_cache[voxel] = nt_concs
        return dict(nt_concs)

    def _circadian_step(self, dt: float) -> Dict[str, float]:
        """Update circadian system and return modulation factors.

        Returns modulation factors derived from the TTFL ODE state
        (CLOCK:BMAL1, Per, Cry, PER_CRY concentrations), NOT from
        hardcoded schedules.  Each factor is in [0.5, 1.5].

        Sleep homeostasis: adenosine accumulates from neural activity,
        feeds back as A1 receptor inhibition (hyperpolarizing current).
        Sleep state is auto-detected from wake_drive threshold.
        """
        if self._circadian is None:
            return {}
        mean_activity = sum(
            1 for n in self._molecular_neurons.values() if n.is_active
        ) / max(1, len(self._molecular_neurons))

        # Auto-detect sleep state from the two-process model
        is_sleeping = self._circadian.wake_drive < 0.3
        self._circadian.step(dt, mean_activity=mean_activity, is_sleeping=is_sleeping)

        # Adenosine A1 receptor inhibition → hyperpolarizing bias current
        a1_inhibition = self._circadian.homeostasis.A1_receptor_activation * 3.0

        return {
            "excitability": self._circadian.excitability_modulation,
            "nt_synthesis": self._circadian.nt_synthesis_modulation,
            "receptor_expr": self._circadian.receptor_expression_modulation,
            "alertness": self._circadian.alertness_modulation,
            "adenosine_inhibition": a1_inhibition,
            "sleep_pressure": self._circadian.sleep_pressure,
            "is_sleeping": 1.0 if is_sleeping else 0.0,
        }

    # ---- Internal helpers ----

    _external_currents: Dict[int, float] = field(init=False, default_factory=dict)

    def _get_external_current(self, nid: int) -> float:
        return self._external_currents.pop(nid, 0.0)

    def _distribute_energy(self) -> None:
        n_alive = sum(1 for n in self._molecular_neurons.values() if n.alive)
        if n_alive == 0:
            return
        per_neuron = self.energy_supply / n_alive

        # Astrocyte lactate shuttle (if glia enabled)
        lactate_bonus = 0.0
        if self.enable_glia and self._astrocytes:
            for astro in self._astrocytes.values():
                lactate_bonus += getattr(astro, 'lactate_pool', 0.0) * 0.001

        for n in self._molecular_neurons.values():
            if n.alive:
                bonus = 0.5 if n.is_active else 0.0
                n.supply_energy((per_neuron + bonus + lactate_bonus) * 0.1)

    def _update_stdp(self, fired: Set[int], dt: float) -> None:
        """STDP with NMDA gating, CaMKII scaling, and spine dynamics.

        Only processes synapses where pre or post neuron fired (STDP requires
        at least one spike). Spine morphology runs every 10 steps for all synapses.
        """
        if not fired:
            # No spikes → no STDP. Periodic spine updates only.
            if self._step_count % 10 == 0:
                for mol_syn in self._molecular_synapses.values():
                    if mol_syn.spine is not None:
                        mol_syn.spine.step(dt * 10.0)
            return

        # Collect synapses involving fired neurons via outgoing index
        relevant_synapses: List[Tuple[Tuple[int, int], MolecularSynapse]] = []
        seen_keys: Set[Tuple[int, int]] = set()

        # Pre-synaptic spikes
        for nid in fired:
            for post_id, mol_syn in self._outgoing.get(nid, []):
                key = (nid, post_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    relevant_synapses.append((key, mol_syn))

        # Post-synaptic spikes: find incoming synapses
        for nid in fired:
            mol_n = self._molecular_neurons.get(nid)
            if mol_n is None:
                continue
            for pre_id in mol_n.inputs:
                key = (pre_id, nid)
                if key not in seen_keys:
                    mol_syn = self._molecular_synapses.get(key)
                    if mol_syn is not None:
                        seen_keys.add(key)
                        relevant_synapses.append((key, mol_syn))

        for (pre, post), mol_syn in relevant_synapses:
            pre_fired = pre in fired
            post_fired = post in fired

            post_n = self._molecular_neurons.get(post)
            if post_n is not None:
                nmda_ch = post_n.membrane.channels.get_channel(IonChannelType.NMDA)
                if nmda_ch is not None:
                    mol_syn._nmda_scale = nmda_ch.conductance_scale
                else:
                    mol_syn._nmda_scale = 1.0

                if post_n.calcium_system is not None:
                    mol_syn._camkii_level = post_n.membrane.camkii_activation
                else:
                    mol_syn._camkii_level = None

            if self._perineuronal_net is not None:
                mol_syn._plasticity_factor = self._perineuronal_net.get_plasticity_factor(post)

            mol_syn.update_stdp(pre_fired, post_fired, self.time, dt)

        # Periodic spine morphology updates (every 10 steps for all synapses)
        if self._step_count % 10 == 0:
            for mol_syn in self._molecular_synapses.values():
                if mol_syn.spine is not None:
                    mol_syn.spine.step(dt * 10.0)

    def _spontaneous_synaptogenesis(self) -> None:
        ids = [nid for nid, n in self._molecular_neurons.items() if n.alive]
        if len(ids) < 2:
            return
        a, b = np.random.choice(ids, 2, replace=False)
        na, nb = self._molecular_neurons[a], self._molecular_neurons[b]
        dist = math.sqrt((na.x - nb.x) ** 2 + (na.y - nb.y) ** 2 + (na.z - nb.z) ** 2)
        if dist < np.linalg.norm(self._size) * 0.4:
            self._add_synapse(a, b)

    def _cleanup(self) -> None:
        # Remove dead neurons
        dead = [nid for nid, n in self._molecular_neurons.items() if not n.alive]
        for nid in dead:
            del self._molecular_neurons[nid]
            del self.neurons[nid]
            # Clean outgoing index for dead neuron
            self._outgoing.pop(nid, None)

        # Remove prunable synapses
        to_prune = [k for k, s in self._molecular_synapses.items() if s.should_prune()]
        for key in to_prune:
            pre, post = key
            del self._molecular_synapses[key]
            del self.synapses[key]
            self._active_synapses.discard(key)
            # Remove from outgoing index
            if pre in self._outgoing:
                self._outgoing[pre] = [
                    (p, s) for p, s in self._outgoing[pre] if p != post
                ]
            if pre in self._molecular_neurons:
                self._molecular_neurons[pre].outputs.discard(post)
            if post in self._molecular_neurons:
                self._molecular_neurons[post].inputs.discard(pre)
            self.pruning_events += 1

        # PNN auto-population: mature, stable neurons get wrapped
        if self._perineuronal_net is not None:
            for nid, mol_n in self._molecular_neurons.items():
                if (mol_n.age > 500.0
                        and mol_n.spike_count > 100
                        and not self._perineuronal_net.is_wrapped(nid)):
                    self._perineuronal_net.add_neuron(nid)

    # ---- Gap junction management ----

    def add_gap_junction(self, pre_id: int, post_id: int,
                         connexin_type: str = "Cx36", conductance_nS: float = 0.5) -> None:
        """Add a gap junction (electrical synapse) between two cells."""
        try:
            from oneuro.molecular.gap_junction import GapJunction, ConnexinType
            cx_map = {"Cx36": ConnexinType.Cx36, "Cx43": ConnexinType.Cx43, "Cx32": ConnexinType.Cx32}
            cx = cx_map.get(connexin_type, ConnexinType.Cx36)
            gj = GapJunction(connexin=cx, pre_id=pre_id, post_id=post_id, conductance_nS=conductance_nS)
            self._gap_junctions.append(gj)
            self.enable_gap_junctions = True
        except ImportError:
            pass

    # ---- I/O interface (matching OrganicNeuralNetwork) ----

    def define_input_region(self, name: str, position, radius: float = 1.5) -> None:
        pos = tuple(np.array(position, dtype=float))
        self.input_regions[name] = (pos, radius)

    def define_output_region(self, name: str, position, radius: float = 1.5) -> None:
        pos = tuple(np.array(position, dtype=float))
        self.output_regions[name] = (pos, radius)

    def stimulate(self, position, intensity: float = 10.0, radius: float = 2.0) -> None:
        pos = np.array(position, dtype=float)
        for nid, neuron in self._molecular_neurons.items():
            if not neuron.alive:
                continue
            npos = np.array([neuron.x, neuron.y, neuron.z])
            dist = np.linalg.norm(pos - npos)
            if dist < radius:
                proximity = 1.0 - dist / radius
                current = intensity * proximity * 1.5
                self._external_currents[nid] = self._external_currents.get(nid, 0.0) + current

    def read_activity(self, position, radius: float = 2.0) -> float:
        pos = np.array(position, dtype=float)
        total_weight = 0.0
        total_activity = 0.0
        for nid, neuron in self._molecular_neurons.items():
            if not neuron.alive:
                continue
            npos = np.array([neuron.x, neuron.y, neuron.z])
            dist = np.linalg.norm(pos - npos)
            if dist < radius:
                proximity = 1.0 - dist / radius
                norm_v = (neuron.membrane_potential + 70.0) / 90.0
                norm_v = max(0.0, min(1.0, norm_v))
                total_activity += norm_v * proximity
                total_weight += proximity
        return total_activity / total_weight if total_weight > 0 else 0.0

    def set_input(self, name: str, value: float) -> None:
        if name in self.input_regions:
            pos, radius = self.input_regions[name]
            self.stimulate(pos, intensity=value * 20.0, radius=radius)

    def read_output(self, name: str) -> float:
        if name in self.output_regions:
            pos, radius = self.output_regions[name]
            return self.read_activity(pos, radius=radius)
        return 0.0

    def set_inputs(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            self.set_input(name, value)

    def read_outputs(self) -> Dict[str, float]:
        return {name: self.read_output(name) for name in self.output_regions}

    # ---- Learning interface (matching OrganicNeuralNetwork) ----

    def release_dopamine(self, amount: float) -> None:
        self.dopamine_level += amount
        self.global_nt_concentrations["dopamine"] = min(
            500.0,
            self.global_nt_concentrations.get("dopamine", 20.0) + amount * 50.0,
        )
        # Release into extracellular space
        if self._extracellular is not None:
            center = np.array(self.size) / 2.0
            self._extracellular.release_at(
                center[0], center[1], center[2], "dopamine", amount * 50.0
            )

    def apply_reward_modulated_plasticity(self) -> None:
        for mol_syn in self._molecular_synapses.values():
            mol_syn.apply_reward(self.dopamine_level, self.learning_rate)
        self.dopamine_level *= self.dopamine_decay
        da = self.global_nt_concentrations.get("dopamine", 20.0)
        self.global_nt_concentrations["dopamine"] = 20.0 + (da - 20.0) * 0.95

    def update_eligibility_traces(self, dt: float = 0.1) -> None:
        for (pre, post), mol_syn in self._molecular_synapses.items():
            pre_n = self._molecular_neurons.get(pre)
            post_n = self._molecular_neurons.get(post)
            if pre_n and post_n:
                pre_act = 1.0 if pre_n.is_active else 0.0
                post_act = 1.0 if post_n.is_active else 0.0
                mol_syn.update_eligibility(pre_act, post_act, dt)

    def structural_adaptation(self, performance: float) -> None:
        if performance > self.performance_threshold_grow:
            if np.random.random() < 0.1:
                for name in self.output_regions:
                    self.grow_neurons_in_region(name, n=1)
        elif performance < self.performance_threshold_prune:
            self._prune_weak_connections(threshold=0.1)

    def grow_neurons_in_region(self, region: str, n: int = 1) -> None:
        if region in self.output_regions:
            pos, radius = self.output_regions[region]
        elif region in self.input_regions:
            pos, radius = self.input_regions[region]
        else:
            return

        for _ in range(n):
            if len(self._molecular_neurons) >= 100:
                return
            offset = np.random.normal(0, radius * 0.5, 3)
            new_pos = np.array(pos) + offset
            new_pos = np.clip(new_pos, [0, 0, 0], self._size)
            nid = self._add_neuron(new_pos[0], new_pos[1], new_pos[2])
            self.neurogenesis_events += 1

            for other_id, other_n in self._molecular_neurons.items():
                if other_id == nid:
                    continue
                dist = math.sqrt(
                    (new_pos[0] - other_n.x) ** 2
                    + (new_pos[1] - other_n.y) ** 2
                    + (new_pos[2] - other_n.z) ** 2
                )
                if dist < radius * 1.5 and np.random.random() < 0.3:
                    self._add_synapse(other_id, nid)
                    if np.random.random() < 0.3:
                        self._add_synapse(nid, other_id)

    def _prune_weak_connections(self, threshold: float = 0.1) -> None:
        to_prune = [
            k for k, s in self._molecular_synapses.items()
            if s.strength < threshold or s.weight < 0.05
        ]
        for key in to_prune:
            pre, post = key
            del self._molecular_synapses[key]
            if key in self.synapses:
                del self.synapses[key]
            self._active_synapses.discard(key)
            if pre in self._outgoing:
                self._outgoing[pre] = [
                    (p, s) for p, s in self._outgoing[pre] if p != post
                ]
            if pre in self._molecular_neurons:
                self._molecular_neurons[pre].outputs.discard(post)
            if post in self._molecular_neurons:
                self._molecular_neurons[post].inputs.discard(pre)
            self.pruning_events += 1

    def give_energy_bonus(self, region: str, amount: float) -> None:
        if region in self.output_regions:
            pos, radius = self.output_regions[region]
        elif region in self.input_regions:
            pos, radius = self.input_regions[region]
        else:
            return
        pos_arr = np.array(pos)
        for neuron in self._molecular_neurons.values():
            npos = np.array([neuron.x, neuron.y, neuron.z])
            if np.linalg.norm(pos_arr - npos) < radius:
                neuron.supply_energy(amount)

    def prune_weak_connections(self, threshold: float = 0.1) -> None:
        self._prune_weak_connections(threshold)

    # ---- Training interface (matching OrganicNeuralNetwork) ----

    def train_episode(self, task, max_steps: int = 100):
        """Run one training episode (matching OrganicNeuralNetwork)."""
        task.reset()
        total_reward = 0.0

        for step_i in range(max_steps):
            inputs = task.get_inputs()
            self.set_inputs(inputs)

            for _ in range(10):
                self.step(dt=0.1)
            self.update_eligibility_traces(dt=1.0)

            outputs = self.read_outputs()
            reward, done = task.evaluate(outputs)
            total_reward += reward

            if reward != 0:
                self.release_dopamine(reward)
            self.apply_reward_modulated_plasticity()

            if done:
                break

        success = task.is_success()
        task_name = task.name
        if task_name not in self.task_performance:
            self.task_performance[task_name] = []
        self.task_performance[task_name].append(total_reward)

        return total_reward, success

    def train_task(self, task, n_episodes: int = 100, report_every: int = 10):
        """Train for multiple episodes (matching OrganicNeuralNetwork)."""
        results = {
            "task": task.name,
            "episodes": n_episodes,
            "rewards": [],
            "successes": [],
        }

        for ep in range(n_episodes):
            reward, success = self.train_episode(task)
            results["rewards"].append(reward)
            results["successes"].append(success)

            if (ep + 1) % 5 == 0:
                recent = results["rewards"][-5:]
                avg_perf = sum(recent) / len(recent)
                self.structural_adaptation(avg_perf / max(1.0, max(recent)))

        final_rewards = results["rewards"][-10:]
        final_successes = results["successes"][-10:]
        results["final_avg_reward"] = sum(final_rewards) / len(final_rewards)
        results["final_success_rate"] = sum(final_successes) / len(final_successes)
        results["total_neurons"] = len(self._molecular_neurons)
        results["total_synapses"] = len(self._molecular_synapses)
        return results

    def evaluate_task(self, task, n_trials: int = 20):
        """Evaluate without learning (matching OrganicNeuralNetwork)."""
        saved_lr = self.learning_rate
        self.learning_rate = 0.0
        rewards = []
        successes = []
        for _ in range(n_trials):
            r, s = self.train_episode(task)
            rewards.append(r)
            successes.append(s)
        self.learning_rate = saved_lr
        return {
            "success_rate": sum(successes) / len(successes),
            "avg_reward": sum(rewards) / len(rewards),
        }

    def get_learning_curve(self, task_name: str) -> List[float]:
        raw = self.task_performance.get(task_name, [])
        if len(raw) < 10:
            return raw
        return [sum(raw[max(0, i - 9):i + 1]) / min(10, i + 1) for i in range(len(raw))]

    # ---- Visualization ----

    def visualize_ascii(self) -> str:
        width, height = 60, 30
        grid = [["." for _ in range(width)] for _ in range(height)]

        for neuron in self._molecular_neurons.values():
            cx = int(neuron.x / self._size[0] * (width - 1))
            cy = int(neuron.y / self._size[1] * (height - 1))
            cx = max(0, min(width - 1, cx))
            cy = max(0, min(height - 1, cy))

            if not neuron.alive:
                sym = "x"
            elif neuron.membrane.fired:
                sym = "@"
            elif neuron.is_active:
                sym = "*"
            else:
                sym = "o"
            grid[cy][cx] = sym

        lines = ["".join(row) for row in grid]
        header = (
            f"Molecular Network t={self.time:.1f}ms | "
            f"{len(self._molecular_neurons)} neurons | "
            f"{len(self._molecular_synapses)} synapses"
        )
        if self._astrocytes:
            header += f" | {len(self._astrocytes)} astrocytes"
        if self._gap_junctions:
            header += f" | {len(self._gap_junctions)} gap junctions"
        return header + "\n" + "\n".join(lines)
