"""MolecularNeuron — a neuron whose membrane potential emerges from molecular physics.

Composes: MolecularMembrane + GeneExpressionPipeline.
Optional: DendriticTree, CellularMetabolism, Cytoskeleton, CalciumSystem,
          SecondMessengerSystem.

Same interface as OrganicNeuron (id, x, y, z, alive, energy) for compatibility.
membrane_potential is read-only — it comes from the membrane's ion channel dynamics.

When optional subsystems are not attached, the neuron behaves as a point neuron
with abstract energy — fully backward compatible with Phase 2 code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

from oneuro.molecular.gene_expression import (
    GeneExpressionPipeline,
)
from oneuro.molecular.ion_channels import IonChannelType
from oneuro.molecular.membrane import MolecularMembrane
from oneuro.molecular.receptors import ReceptorType


class NeuronArchetype(Enum):
    """Neuron types define channel/receptor population profiles."""

    PYRAMIDAL = "pyramidal"  # Excitatory cortical
    INTERNEURON = "interneuron"  # Inhibitory (fast-spiking)
    PURKINJE = "purkinje"  # Cerebellar, complex spikes
    GRANULE = "granule"  # Small excitatory
    MEDIUM_SPINY = "medium_spiny"  # Striatal, dopamine-sensitive
    D1_MSN = "d1_msn"  # Direct-pathway striatal MSN
    D2_MSN = "d2_msn"  # Indirect-pathway striatal MSN
    MOTONEURON = "motoneuron"  # Cholinergic output


def _make_medium_spiny_membrane(
    d1_count: int,
    d2_count: int,
) -> MolecularMembrane:
    """Create a striatal MSN membrane with explicit dopamine-receptor bias."""
    m = MolecularMembrane.excitatory()
    m.add_receptor(ReceptorType.D1, count=d1_count)
    m.add_receptor(ReceptorType.D2, count=d2_count)
    m.add_receptor(ReceptorType.GABA_A, count=35)
    return m


def _make_membrane(archetype: NeuronArchetype) -> MolecularMembrane:
    """Create a membrane configured for a specific neuron archetype."""
    if archetype == NeuronArchetype.PYRAMIDAL:
        m = MolecularMembrane.excitatory()
        m.add_receptor(ReceptorType.GABA_A, count=20)
        m.add_receptor(ReceptorType.D1, count=5)
        m.add_receptor(ReceptorType.HT_2A, count=3)
        return m
    elif archetype == NeuronArchetype.INTERNEURON:
        m = MolecularMembrane.inhibitory()
        m.add_receptor(ReceptorType.AMPA, count=30)
        m.add_receptor(ReceptorType.D2, count=5)
        return m
    elif archetype == NeuronArchetype.PURKINJE:
        m = MolecularMembrane.excitatory()
        m.channels.add_channel(IonChannelType.Ca_v, count=3)
        m.add_receptor(ReceptorType.GABA_A, count=60)
        return m
    elif archetype == NeuronArchetype.GRANULE:
        m = MolecularMembrane.excitatory()
        return m
    elif archetype == NeuronArchetype.MEDIUM_SPINY:
        return _make_medium_spiny_membrane(d1_count=20, d2_count=15)
    elif archetype == NeuronArchetype.D1_MSN:
        return _make_medium_spiny_membrane(d1_count=30, d2_count=4)
    elif archetype == NeuronArchetype.D2_MSN:
        return _make_medium_spiny_membrane(d1_count=4, d2_count=30)
    elif archetype == NeuronArchetype.MOTONEURON:
        m = MolecularMembrane.cholinergic()
        return m
    else:
        return MolecularMembrane.excitatory()


def _make_gene_pipeline(archetype: NeuronArchetype) -> GeneExpressionPipeline:
    """Create gene expression pipeline for a neuron archetype."""
    if archetype in (
        NeuronArchetype.INTERNEURON,
        NeuronArchetype.MEDIUM_SPINY,
        NeuronArchetype.D1_MSN,
        NeuronArchetype.D2_MSN,
    ):
        return GeneExpressionPipeline.inhibitory_neuron()
    return GeneExpressionPipeline.excitatory_neuron()


@dataclass
class MolecularNeuron:
    """A neuron composed of molecular assemblies.

    Membrane potential emerges from ion channel physics.
    Compatible with OrganicNeuron's interface for seamless integration.

    Optional subsystems (all None by default for backward compat):
    - dendrite: DendriticTree for compartmental modeling
    - metabolism: CellularMetabolism for real ATP/glucose
    - cytoskeleton: Cytoskeleton for microtubules/Orch-OR
    - calcium_system: CalciumSystem for multi-compartment Ca²⁺
    - second_messenger_system: SecondMessengerSystem for intracellular cascades
    """

    # Identity (matching OrganicNeuron interface)
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    alive: bool = True
    age: float = 0.0
    generation: int = 0

    # Molecular components
    archetype: NeuronArchetype = NeuronArchetype.PYRAMIDAL
    membrane: MolecularMembrane = field(init=False)
    gene_pipeline: GeneExpressionPipeline = field(init=False)

    # Metabolic (matching OrganicNeuron, backward compat)
    energy: float = 100.0
    energy_consumption: float = 0.1
    plasticity: float = 0.01

    # Optional advanced subsystems
    dendrite: Optional[object] = None  # DendriticTree
    metabolism: Optional[object] = None  # CellularMetabolism
    cytoskeleton: Optional[object] = None  # Cytoskeleton
    calcium_system: Optional[object] = None  # CalciumSystem
    second_messenger_system: Optional[object] = None  # SecondMessengerSystem

    # Connectivity (matching OrganicNeuron)
    inputs: Set[int] = field(default_factory=set)
    outputs: Set[int] = field(default_factory=set)

    # History
    activation_history: List[float] = field(default_factory=list)
    _history_max: int = field(init=False, default=100)

    def __post_init__(self):
        self.membrane = _make_membrane(self.archetype)
        self.gene_pipeline = _make_gene_pipeline(self.archetype)

        # Wire optional subsystems to membrane
        if self.calcium_system is not None:
            self.membrane.set_calcium_system(self.calcium_system)
        if self.second_messenger_system is not None:
            self.membrane.set_second_messenger_system(self.second_messenger_system)

    # ---- Properties matching OrganicNeuron ----

    @property
    def membrane_potential(self) -> float:
        """Membrane potential in mV. Read-only — emerges from channel physics."""
        if self.dendrite is not None:
            return self.dendrite.soma_voltage
        return self.membrane.voltage

    @property
    def threshold(self) -> float:
        """Firing threshold (mV). For molecular neuron this is the AP detection level."""
        return -20.0  # AP threshold in membrane.py

    @property
    def fired(self) -> bool:
        """Whether neuron fired on the last update."""
        return self.membrane.fired

    @property
    def is_active(self) -> bool:
        return self.membrane.voltage > -55.0

    @property
    def spike_count(self) -> int:
        return self.membrane.spike_count

    # ---- Core update ----

    def update(
        self,
        nt_concentrations: Optional[Dict[str, float]] = None,
        external_current: float = 0.0,
        dt: float = 0.1,
        skip_slow_subsystems: bool = False,
        step_count: int = 0,
        runtime_rng: Optional[np.random.Generator] = None,
        microtubule_collapse_jitter_std: float = 0.0,
    ) -> bool:
        """Update neuron for one timestep.

        Args:
            nt_concentrations: NT name → concentration in nM at this neuron.
            external_current: Injected current in uA/cm².
            dt: Timestep in ms.
            skip_slow_subsystems: Skip ALL slow subsystems (for inactive neurons).
            step_count: Global step counter for interval-gated subsystem updates.
                       When >0, gene expression runs every 100 steps, metabolism
                       every 10 steps, cytoskeleton every 10 steps (but always on fire).
            runtime_rng: Optional runtime entropy-backed generator for ongoing dynamics.
            microtubule_collapse_jitter_std: Relative jitter applied to Orch-OR
                       collapse threshold when runtime entropy is enabled.

        Returns:
            True if neuron fired an action potential.
        """
        if not self.alive:
            return False

        # Energy check
        self.consume_energy(dt)
        if self.energy <= 0:
            self.alive = False
            return False

        # Dendrite: if present, delegates to compartmental model
        if self.dendrite is not None:
            # Synaptic inputs go to specific compartments (handled by network)
            # External current applies to soma
            self.dendrite.step(dt, synaptic_inputs={})
            # Soma voltage drives the membrane model
            fired = self.membrane.step(dt, nt_concentrations, external_current)
        else:
            # Point neuron: standard membrane dynamics
            fired = self.membrane.step(dt, nt_concentrations, external_current)

        if skip_slow_subsystems:
            self.age += dt
            return fired

        # Interval gating: when step_count is provided, run slow subsystems
        # at reduced frequency. Always run on fire events (activity-dependent).
        # Intervals chosen to preserve learning dynamics while saving CPU:
        #   gene expression: every 10 steps (~1ms bio time, fine for minute-scale gene regulation)
        #   metabolism: every 5 steps (ATP barely changes in 0.5ms)
        #   cytoskeleton: every 10 steps (quantum decoherence timescale ~ms)
        use_intervals = step_count > 0
        run_cyto = fired or not use_intervals or (step_count % 10 == 0)
        run_metab = fired or not use_intervals or (step_count % 5 == 0)
        run_gene = fired or not use_intervals or (step_count % 10 == 0)

        # Cytoskeleton: microtubule quantum evolution + structural dynamics
        if run_cyto and self.cytoskeleton is not None:
            ca = self.membrane.ca_internal
            self.cytoskeleton.step(
                dt * (10 if use_intervals and not fired else 1),
                ca_nM=ca,
                rng=runtime_rng,
                collapse_jitter_std=microtubule_collapse_jitter_std,
            )

        # Metabolism: ATP production and consumption
        if run_metab and self.metabolism is not None:
            metab_dt = dt * (5 if use_intervals and not fired else 1)
            # Na/K-ATPase cost scales with activity
            firing_rate = 1.0 if fired else 0.0
            self.metabolism.na_k_atpase_cost(metab_dt, firing_rate)

            # Ca²⁺ cycling cost
            ca_cycling = max(0.0, self.membrane.ca_internal - 100.0) / 1000.0
            self.metabolism.serca_cost(metab_dt, ca_cycling)

            # Run metabolic pathways
            self.metabolism.step(metab_dt)

            # Sync abstract energy for backward compat
            self.energy = self.metabolism.energy

        # Gene expression (slow timescale — runs every 10 steps or on fire)
        if run_gene:
            gene_dt = dt * (10 if use_intervals and not fired else 1)
            # Bridge CaMKII → CREB phosphorylation → gene expression
            if self.second_messenger_system is not None:
                phos = getattr(self.second_messenger_system, 'phosphorylation_state', None)
                creb_p = getattr(phos, 'CREB_p', 0.0) if phos is not None else 0.0
                self.gene_pipeline.signal_creb_phosphorylation(creb_p)

            neural_activity = 1.0 if fired else (0.3 if self.is_active else 0.0)
            self.gene_pipeline.update(gene_dt, neural_activity=neural_activity)

            # Activity-dependent gene regulation
            if fired:
                self.gene_pipeline.signal_ltp(self.membrane.ca_internal)
            elif self.membrane.ca_internal > 200.0:
                self.gene_pipeline.signal_ltd(self.membrane.ca_internal)

            # Apply receptor trafficking from gene expression
            receptor_changes = self.gene_pipeline.get_receptor_changes()
            for receptor_type, count_change in receptor_changes.items():
                if count_change > 0:
                    self.membrane.add_receptor(receptor_type, count=count_change)
                elif count_change < 0:
                    self.membrane.remove_receptors(receptor_type, count=abs(count_change))

        # Track activation (every 10 steps or on fire)
        if fired or not use_intervals or (step_count % 10 == 0):
            normalized = (self.membrane.voltage + 70.0) / 90.0  # [-70, 20] → [0, 1]
            self.activation_history.append(max(0.0, min(1.0, normalized)))
            if len(self.activation_history) > self._history_max:
                self.activation_history.pop(0)

        self.age += dt
        return fired

    def consume_energy(self, dt: float = 0.1) -> None:
        """Metabolic cost: base + active bonus + connection cost."""
        if self.metabolism is not None:
            # Real metabolism handles this
            return

        base = self.energy_consumption * dt
        active_bonus = 0.2 * dt if self.is_active else 0.0
        connection_cost = 0.001 * len(self.inputs) * dt
        self.energy -= base + active_bonus + connection_cost
        self.energy = max(0.0, self.energy)

    def supply_energy(self, amount: float) -> None:
        """External energy supply (from tissue metabolism)."""
        if self.metabolism is not None:
            self.metabolism.supply_glucose(amount * 0.01)  # Scale to mM
            self.metabolism.supply_oxygen(amount * 0.001)
        else:
            self.energy = min(200.0, self.energy + amount)

    def attach_subsystems(self, calcium_system=None, second_messenger_system=None,
                          metabolism=None, cytoskeleton=None, dendrite=None) -> None:
        """Attach optional advanced subsystems after construction."""
        if calcium_system is not None:
            self.calcium_system = calcium_system
            self.membrane.set_calcium_system(calcium_system)
        if second_messenger_system is not None:
            self.second_messenger_system = second_messenger_system
            self.membrane.set_second_messenger_system(second_messenger_system)
        if metabolism is not None:
            self.metabolism = metabolism
        if cytoskeleton is not None:
            self.cytoskeleton = cytoskeleton
        if dendrite is not None:
            self.dendrite = dendrite

    def can_divide(self) -> bool:
        """Check neurogenesis conditions (matching OrganicNeuron)."""
        return (
            self.alive
            and self.energy > 80.0
            and self.age > 1000.0
            and len(self.outputs) >= 2
        )

    def divide(self, new_id: int) -> Optional["MolecularNeuron"]:
        """Neurogenesis: create a daughter neuron."""
        if not self.can_divide():
            return None

        # Daughter inherits archetype, position nearby
        offset = np.random.normal(0, 0.5, 3)
        daughter = MolecularNeuron(
            id=new_id,
            x=self.x + offset[0],
            y=self.y + offset[1],
            z=self.z + offset[2],
            archetype=self.archetype,
            energy=self.energy * 0.4,
            generation=self.generation + 1,
        )

        # Parent loses energy
        self.energy *= 0.6
        return daughter
