"""Glial cells -- astrocytes, oligodendrocytes, and microglia.

The three major non-neuronal cell types that constitute ~50% of brain volume.
They are not passive scaffolding: they actively shape neural computation through
metabolic coupling, myelination, synapse pruning, and gliotransmission.

Astrocytes:
  - Perisynaptic processes uptake glutamate (EAAT1/2, Michaelis-Menten Km=20 uM)
  - IP3-mediated calcium waves propagate to neighbors
  - Gliotransmitter release: D-serine (NMDA co-agonist), ATP (paracrine)
  - Lactate shuttle: glucose -> lactate -> neurons (metabolic coupling)

Oligodendrocytes:
  - Myelinate axon segments -> increase conduction velocity
  - Metabolic support via MCT transporters (lactate to axon)

Microglia:
  - Immune surveillance with state transitions (surveying -> activated -> phagocytic)
  - Complement-tagged synapse pruning (C1q/C3 pathway)
  - Cytokine release (TNF-alpha, IL-1beta) modulates neuronal excitability

All concentrations in nM. All times in ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Glutamate transporter Km (Michaelis-Menten) ~20 uM = 20_000 nM
_EAAT_KM_NM: float = 20_000.0
# Max uptake rate per astrocyte process per ms (nM/ms)
_EAAT_VMAX_NM_PER_MS: float = 500.0

# Kir4.1 K+ buffering half-saturation (mM above resting ~3 mM extracellular)
_KIR41_KM_MM: float = 2.0

# IP3 receptor Ca2+ release parameters
_IP3R_EC50: float = 300.0  # nM IP3 for half-max Ca2+ release
_IP3R_HILL_N: float = 2.0
_IP3_DECAY_RATE: float = 0.005  # per ms, enzymatic IP3 hydrolysis

# Calcium dynamics
_CA_RESTING_NM: float = 100.0  # Astrocyte resting [Ca2+]i ~100 nM
_CA_STORE_RELEASE_MAX: float = 5000.0  # Max Ca2+ from ER per ms (nM)
_CA_DECAY_TAU_MS: float = 500.0  # Slower than neurons (~5x)
_CA_MAX_NM: float = 50_000.0  # Physiological ceiling (ER capacity limit)
_CA_WAVE_THRESHOLD: float = 500.0  # nM; above this, IP3 propagates to neighbors

# Gliotransmitter release thresholds
_DSERINE_CA_THRESHOLD: float = 400.0  # nM Ca2+ to trigger D-serine release
_DSERINE_RELEASE_RATE: float = 50.0  # nM per ms when active
_ATP_CA_THRESHOLD: float = 600.0  # nM Ca2+ to trigger ATP release
_ATP_RELEASE_RATE: float = 100.0  # nM per ms when active

# Lactate shuttle parameters
_GLUCOSE_TO_LACTATE_RATE: float = 0.01  # fraction per ms (glycolysis)
_LACTATE_DELIVERY_RATE: float = 0.5  # nM per ms per connected synapse
_GLYCOGEN_SYNTHESIS_RATE: float = 0.001  # fraction per ms
_GLYCOGEN_MOBILIZATION_RATE: float = 0.005  # fraction per ms when demand exceeds supply

# Oligodendrocyte conduction velocity model
# Rushton's ratio: conduction velocity ~ sqrt(myelin_thickness * axon_diameter)
# We normalize so factor=1.0 at zero myelin, up to ~10x at full myelination
_MYELIN_MAX_VELOCITY_FACTOR: float = 10.0

# Microglia parameters
_COMPLEMENT_PRUNE_THRESHOLD: float = 0.7  # Complement level 0-1 for tagging
_SURVEYING_SCAN_RATE: float = 0.001  # Probability per ms of detecting damage
_ACTIVATION_THRESHOLD: float = 50.0  # Damage signal level to activate
_PHAGOCYTIC_THRESHOLD: float = 200.0  # Damage level for full phagocytic mode
_DEACTIVATION_RATE: float = 0.002  # Per ms, exponential decay (half-life ~350 ms)

# Cytokine dynamics
_TNF_RELEASE_RATE: float = 2.0  # nM per ms when activated
_TNF_DECAY_RATE: float = 0.01  # per ms
_IL1B_RELEASE_RATE: float = 1.0  # nM per ms when activated
_IL1B_DECAY_RATE: float = 0.005  # per ms


# ---------------------------------------------------------------------------
# Astrocyte
# ---------------------------------------------------------------------------

@dataclass
class Astrocyte:
    """Protoplasmic astrocyte with perisynaptic processes.

    Implements the tripartite synapse model: astrocyte processes ensheath
    synaptic clefts, uptake excess glutamate, release gliotransmitters
    (D-serine, ATP), buffer extracellular K+, and supply lactate to neurons.

    IP3-mediated calcium waves enable long-range signaling between astrocytes.
    """

    # Identity and position
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Tripartite synapse connections
    synapse_ids: List[int] = field(default_factory=list)

    # Intracellular calcium (nM) -- drives gliotransmitter release
    ca_internal: float = _CA_RESTING_NM

    # IP3 signaling
    ip3_level: float = 0.0

    # Glutamate stored from uptake (nM equivalent, for recycling to neurons)
    glutamate_stored: float = 0.0

    # Metabolic pools
    lactate_pool: float = 5000.0  # nM, available for neuron supply
    glycogen_store: float = 50_000.0  # nM glucose equivalent

    # Transporter parameters (can be modulated by drugs)
    eaat_vmax: float = _EAAT_VMAX_NM_PER_MS
    eaat_km: float = _EAAT_KM_NM

    # Neighbor astrocyte IDs for calcium wave propagation
    neighbor_ids: List[int] = field(default_factory=list)

    # Tracking
    _time: float = field(init=False, default=0.0)
    _d_serine_released_total: float = field(init=False, default=0.0)
    _atp_released_total: float = field(init=False, default=0.0)

    def step(
        self,
        dt: float,
        local_glutamate_nM: float,
        local_k_mM: float = 3.0,
    ) -> Dict[str, float]:
        """Advance astrocyte state by dt milliseconds.

        Args:
            dt: Timestep in ms.
            local_glutamate_nM: Extracellular glutamate at perisynaptic site (nM).
            local_k_mM: Extracellular potassium concentration (mM).

        Returns:
            Dict with changes applied this timestep:
              - glutamate_uptake: nM glutamate removed from cleft
              - k_buffered: mM K+ buffered
              - d_serine_release: nM D-serine released
              - atp_release: nM ATP released
              - lactate_output: nM lactate delivered to neurons
              - ip3_to_neighbors: IP3 level propagated (0 if below threshold)
        """
        self._time += dt

        result: Dict[str, float] = {
            "glutamate_uptake": 0.0,
            "k_buffered": 0.0,
            "d_serine_release": 0.0,
            "atp_release": 0.0,
            "lactate_output": 0.0,
            "ip3_to_neighbors": 0.0,
        }

        # 1. Glutamate uptake via EAAT1/2 (Michaelis-Menten kinetics)
        if local_glutamate_nM > 0:
            uptake = self._glutamate_uptake(local_glutamate_nM, dt)
            result["glutamate_uptake"] = uptake
            self.glutamate_stored += uptake
            # Glutamate binding triggers metabotropic signaling -> IP3 production
            ip3_production = uptake * 0.01  # Small fraction triggers mGluR5 -> IP3
            self.ip3_level += ip3_production

        # 2. K+ buffering via Kir4.1 channels
        k_excess = max(0.0, local_k_mM - 3.0)  # Above resting ~3 mM
        if k_excess > 0:
            buffered = k_excess * dt / (k_excess + _KIR41_KM_MM) * 0.1
            result["k_buffered"] = buffered

        # 3. IP3-mediated calcium release from ER stores
        self._update_calcium(dt)

        # 4. Gliotransmitter release (Ca2+-dependent)
        d_serine = self._release_d_serine(dt)
        atp = self._release_atp(dt)
        result["d_serine_release"] = d_serine
        result["atp_release"] = atp
        self._d_serine_released_total += d_serine
        self._atp_released_total += atp

        # 5. Lactate shuttle (metabolic coupling to neurons)
        lactate_out = self._lactate_shuttle(dt)
        result["lactate_output"] = lactate_out

        # 6. IP3 decay (enzymatic hydrolysis by IP3 phosphatase)
        self.ip3_level *= (1.0 - _IP3_DECAY_RATE * dt)
        self.ip3_level = max(0.0, self.ip3_level)

        # 7. Calcium wave propagation check
        if self.ca_internal > _CA_WAVE_THRESHOLD:
            result["ip3_to_neighbors"] = self.ip3_level * 0.3  # 30% of IP3 diffuses

        return result

    def receive_ip3(self, amount: float) -> None:
        """Receive IP3 from a neighboring astrocyte (calcium wave propagation)."""
        self.ip3_level += amount

    def _glutamate_uptake(self, glutamate_nM: float, dt: float) -> float:
        """Michaelis-Menten glutamate uptake via EAAT1/2 transporters.

        V = Vmax * [Glu] / (Km + [Glu])

        EAAT1 (GLAST) and EAAT2 (GLT-1) are the primary astrocytic glutamate
        transporters, responsible for ~90% of cortical glutamate clearance.
        Km ~20 uM ensures efficient uptake at synaptic concentrations.
        """
        rate = self.eaat_vmax * glutamate_nM / (self.eaat_km + glutamate_nM)
        uptake = rate * dt
        return min(uptake, glutamate_nM)  # Cannot remove more than present

    def _update_calcium(self, dt: float) -> None:
        """IP3-mediated Ca2+ release from endoplasmic reticulum.

        The IP3 receptor (IP3R) on the ER membrane opens when IP3 binds,
        releasing Ca2+ from internal stores. This follows a Hill equation
        with cooperative binding (n=2).
        """
        if self.ip3_level > 0:
            # IP3R open fraction (Hill equation)
            ip3_n = self.ip3_level ** _IP3R_HILL_N
            open_fraction = ip3_n / (_IP3R_EC50 ** _IP3R_HILL_N + ip3_n)
            ca_release = _CA_STORE_RELEASE_MAX * open_fraction * dt
            self.ca_internal += ca_release

        # Ca2+ extrusion and SERCA pump (exponential decay to resting)
        dca = dt * (_CA_RESTING_NM - self.ca_internal) / _CA_DECAY_TAU_MS
        self.ca_internal += dca
        self.ca_internal = max(0.0, min(_CA_MAX_NM, self.ca_internal))

    def _release_d_serine(self, dt: float) -> float:
        """Ca2+-dependent D-serine release.

        D-serine is the primary NMDA receptor co-agonist at the glycine
        binding site in many brain regions. Astrocytic release is triggered
        by elevated intracellular Ca2+ (>400 nM above resting).
        """
        if self.ca_internal > _DSERINE_CA_THRESHOLD:
            excess = self.ca_internal - _DSERINE_CA_THRESHOLD
            fraction = excess / (excess + 200.0)  # Saturating release
            return _DSERINE_RELEASE_RATE * fraction * dt
        return 0.0

    def _release_atp(self, dt: float) -> float:
        """Ca2+-dependent ATP release.

        ATP is released from astrocytes through vesicular exocytosis and
        connexin hemichannels. Once in the extracellular space, ATP acts
        on purinergic receptors (P2Y, P2X) on neurons and other glia.
        Ectoenzymes rapidly degrade ATP to adenosine, which has
        predominantly inhibitory effects.
        """
        if self.ca_internal > _ATP_CA_THRESHOLD:
            excess = self.ca_internal - _ATP_CA_THRESHOLD
            fraction = excess / (excess + 300.0)
            return _ATP_RELEASE_RATE * fraction * dt
        return 0.0

    def _lactate_shuttle(self, dt: float) -> float:
        """Astrocyte-neuron lactate shuttle (ANLS).

        Glutamate uptake stimulates astrocytic glycolysis, converting glucose
        to lactate. Lactate is exported via MCT1/4 transporters and imported
        by neuronal MCT2 to fuel oxidative phosphorylation. This couples
        synaptic activity to metabolic supply.

        Glycogen stores provide a buffer during high-demand periods.
        """
        # Activity-dependent lactate production from glutamate uptake
        production = self.glutamate_stored * _GLUCOSE_TO_LACTATE_RATE * dt
        self.glutamate_stored = max(0.0, self.glutamate_stored - production * 0.1)
        self.lactate_pool += production

        # Glycogen synthesis when lactate is abundant
        if self.lactate_pool > 3000.0:
            to_glycogen = self.lactate_pool * _GLYCOGEN_SYNTHESIS_RATE * dt
            self.lactate_pool -= to_glycogen
            self.glycogen_store += to_glycogen

        # Glycogen mobilization when lactate pool is depleted
        if self.lactate_pool < 1000.0 and self.glycogen_store > 0:
            mobilized = self.glycogen_store * _GLYCOGEN_MOBILIZATION_RATE * dt
            self.glycogen_store -= mobilized
            self.lactate_pool += mobilized

        # Deliver lactate to connected neurons via MCT transporters
        n_synapses = max(1, len(self.synapse_ids))
        delivery = min(
            _LACTATE_DELIVERY_RATE * n_synapses * dt,
            self.lactate_pool * 0.1,  # Never deliver more than 10% per step
        )
        self.lactate_pool -= delivery
        self.lactate_pool = max(0.0, self.lactate_pool)
        return delivery


# ---------------------------------------------------------------------------
# Oligodendrocyte
# ---------------------------------------------------------------------------

@dataclass
class Oligodendrocyte:
    """Myelinating oligodendrocyte.

    Each oligodendrocyte can myelinate up to 30-50 axon segments (internodes).
    Myelin thickness determines conduction velocity via saltatory conduction.
    Also provides metabolic support to ensheathed axons through MCT transporters
    delivering lactate/pyruvate to the periaxonal space.
    """

    # Identity and position
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Health state (0-1): reflects oligodendrocyte viability
    # Reduced by oxidative stress, excitotoxicity, or immune attack
    health: float = 1.0

    # Maximum segments this OL can maintain
    max_segments: int = 40

    # Myelin segments: axon_id -> myelin thickness (relative 0-1)
    # 0.0 = barely myelinated, 1.0 = maximally thick compact myelin
    myelin_segments: Dict[int, float] = field(default_factory=dict)

    # Metabolic state for axonal support
    _lactate_pool: float = field(init=False, default=3000.0)

    def myelinate(self, axon_id: int, initial_thickness: float = 0.1) -> bool:
        """Add an axon segment to this oligodendrocyte's myelin repertoire.

        Args:
            axon_id: ID of the axon (pre-synaptic neuron ID) to myelinate.
            initial_thickness: Starting myelin thickness, 0-1 (default 0.1).

        Returns:
            True if myelination succeeded, False if capacity exceeded.
        """
        if len(self.myelin_segments) >= self.max_segments:
            return False
        if axon_id in self.myelin_segments:
            return True  # Already myelinated
        self.myelin_segments[axon_id] = max(0.0, min(1.0, initial_thickness))
        return True

    def demyelinate(self, axon_id: int) -> bool:
        """Remove myelin from an axon segment.

        Args:
            axon_id: Axon to demyelinate.

        Returns:
            True if the axon was myelinated and is now removed.
        """
        if axon_id in self.myelin_segments:
            del self.myelin_segments[axon_id]
            return True
        return False

    def mature_myelin(self, axon_id: int, dt: float, activity_level: float = 0.5) -> None:
        """Grow myelin thickness over time.

        Activity-dependent myelination: active axons get thicker myelin.
        Maturation rate is proportional to oligodendrocyte health and
        axonal firing activity.

        Args:
            axon_id: Axon segment to mature.
            dt: Timestep in ms.
            activity_level: Normalized firing rate of the axon, 0-1.
        """
        if axon_id not in self.myelin_segments:
            return
        current = self.myelin_segments[axon_id]
        if current >= 1.0:
            return
        # Growth rate: ~0.001/ms at full health and activity -> ~1.0 over ~1000ms
        growth = 0.001 * self.health * (0.5 + 0.5 * activity_level) * dt
        self.myelin_segments[axon_id] = min(1.0, current + growth)

    def conduction_velocity_factor(self, axon_id: int) -> float:
        """Compute conduction velocity multiplier for a myelinated axon.

        Unmyelinated axon: factor = 1.0
        Fully myelinated: factor ~ 10.0 (saltatory conduction)

        The relationship follows: v ~ sqrt(myelin_thickness * diameter)
        normalized so that zero thickness gives 1.0 and max gives ~10.0.

        Args:
            axon_id: The axon to query.

        Returns:
            Velocity multiplier >= 1.0.
        """
        thickness = self.myelin_segments.get(axon_id, 0.0)
        if thickness <= 0.0:
            return 1.0
        # Saturating curve: 1.0 + (max-1) * sqrt(thickness)
        return 1.0 + (_MYELIN_MAX_VELOCITY_FACTOR - 1.0) * math.sqrt(thickness)

    def metabolic_support(self, axon_id: int, dt: float) -> float:
        """Deliver lactate to a myelinated axon via MCT1 transporters.

        Oligodendrocytes supply ~70% of axonal energy needs via monocarboxylate
        transporters (MCT1 on OL, MCT2 on axon). This metabolic coupling is
        essential for long-term axon survival.

        Args:
            axon_id: Axon to supply.
            dt: Timestep in ms.

        Returns:
            Lactate delivered (nM).
        """
        if axon_id not in self.myelin_segments:
            return 0.0
        thickness = self.myelin_segments[axon_id]
        # Thicker myelin = better metabolic coupling
        delivery_rate = 0.2 * thickness * self.health  # nM per ms
        delivery = delivery_rate * dt
        delivery = min(delivery, self._lactate_pool * 0.05)
        self._lactate_pool -= delivery
        self._lactate_pool = max(0.0, self._lactate_pool)
        return delivery

    def supply_lactate(self, amount: float) -> None:
        """Receive lactate from astrocytes or blood-brain barrier."""
        self._lactate_pool += amount

    def step(self, dt: float) -> None:
        """Update oligodendrocyte state per timestep.

        Slow health recovery when not under stress.
        """
        # Gradual health recovery toward 1.0
        if self.health < 1.0:
            self.health += 0.0001 * dt
            self.health = min(1.0, self.health)

        # Degenerate segments if health is critically low
        if self.health < 0.2:
            for axon_id in list(self.myelin_segments.keys()):
                self.myelin_segments[axon_id] *= (1.0 - 0.001 * dt)
                if self.myelin_segments[axon_id] < 0.01:
                    del self.myelin_segments[axon_id]

    @property
    def segment_count(self) -> int:
        """Number of actively myelinated segments."""
        return len(self.myelin_segments)

    @property
    def capacity_remaining(self) -> int:
        """Number of additional axons that can be myelinated."""
        return max(0, self.max_segments - len(self.myelin_segments))


# ---------------------------------------------------------------------------
# Microglia
# ---------------------------------------------------------------------------

class MicrogliaState(Enum):
    """Microglia activation states.

    Surveying: resting ramified morphology, scanning for damage signals.
    Activated: retracted processes, cytokine release, chemo-attraction.
    Phagocytic: amoeboid morphology, actively engulfing debris/synapses.
    """
    SURVEYING = "surveying"
    ACTIVATED = "activated"
    PHAGOCYTIC = "phagocytic"


@dataclass
class Microglia:
    """Resident immune cell of the CNS.

    Microglia survey the brain parenchyma, detect damage signals (ATP, DAMPs),
    transition through activation states, release cytokines, and perform
    complement-mediated synapse pruning. This is critical for developmental
    circuit refinement and pathological neuroinflammation.

    The complement pathway (C1q -> C3 -> CR3 receptor on microglia) tags
    weak or inactive synapses for phagocytic elimination.
    """

    # Identity and position
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Activation state
    state: MicrogliaState = MicrogliaState.SURVEYING

    # Cytokine levels (nM, released into local environment)
    tnf_alpha: float = 0.0
    il_1beta: float = 0.0

    # Complement-tagged synapses: synapse_key -> complement level (0-1)
    tagged_synapses: Dict[int, float] = field(default_factory=dict)

    # Internal state
    _activation_level: float = field(init=False, default=0.0)
    _phagocytosis_cooldown: float = field(init=False, default=0.0)
    _time: float = field(init=False, default=0.0)

    def step(self, dt: float, local_damage_signal: float = 0.0) -> Dict[str, float]:
        """Advance microglia state by dt milliseconds.

        Args:
            dt: Timestep in ms.
            local_damage_signal: Aggregate damage signal from environment
                (ATP from dying cells, DAMPs, etc.). Range 0-1000+.

        Returns:
            Dict with:
              - tnf_alpha_release: nM TNF-alpha released this step
              - il_1beta_release: nM IL-1beta released this step
              - state: current state name string
        """
        self._time += dt

        result: Dict[str, float] = {
            "tnf_alpha_release": 0.0,
            "il_1beta_release": 0.0,
            "state": self.state.value,
        }

        # 1. Integrate damage signal into activation level
        self._activation_level += local_damage_signal * dt * 0.01
        # Multiplicative decay: activation decays proportionally (half-life ~350ms)
        decay = math.exp(-_DEACTIVATION_RATE * dt)
        self._activation_level *= decay
        self._activation_level = max(0.0, self._activation_level)

        # 2. State transitions
        self._transition_state()
        result["state"] = self.state.value

        # 3. Cytokine dynamics
        tnf_released, il1b_released = self._update_cytokines(dt)
        result["tnf_alpha_release"] = tnf_released
        result["il_1beta_release"] = il1b_released

        # 4. Phagocytosis cooldown
        if self._phagocytosis_cooldown > 0:
            self._phagocytosis_cooldown = max(0.0, self._phagocytosis_cooldown - dt)

        return result

    def tag_synapse(self, synapse_key: int, complement_level: float) -> None:
        """Mark a synapse for potential pruning via complement tagging.

        In the complement pathway, weak or inactive synapses accumulate
        C1q and C3 opsonins. When complement levels exceed the pruning
        threshold, microglia recognize C3 via their CR3 receptor and
        phagocytose the tagged synapse.

        Args:
            synapse_key: Unique synapse identifier (e.g., hash of pre/post pair).
            complement_level: Complement opsonization level, 0-1.
                Typically increases with synapse inactivity.
        """
        current = self.tagged_synapses.get(synapse_key, 0.0)
        # Complement accumulates but does not decrease (sticky opsonization)
        self.tagged_synapses[synapse_key] = max(current, complement_level)

    def prune_tagged(self) -> List[int]:
        """Execute complement-mediated synapse pruning.

        Returns synapse keys that exceed the complement threshold and should
        be removed from the network. Only prunes when microglia is in
        ACTIVATED or PHAGOCYTIC state and phagocytosis cooldown has elapsed.

        Returns:
            List of synapse keys to remove from the network.
        """
        if self.state == MicrogliaState.SURVEYING:
            return []
        if self._phagocytosis_cooldown > 0:
            return []

        pruned: List[int] = []
        for synapse_key, level in list(self.tagged_synapses.items()):
            if level >= _COMPLEMENT_PRUNE_THRESHOLD:
                pruned.append(synapse_key)

        # Remove pruned synapses from tracking
        for key in pruned:
            del self.tagged_synapses[key]

        # Phagocytosis takes time -- cooldown prevents instantaneous mass pruning
        if pruned:
            self._phagocytosis_cooldown = 50.0  # ms

        return pruned

    def clear_tag(self, synapse_key: int) -> None:
        """Remove complement tag from a synapse (e.g., if it becomes active).

        Active synapses can shed complement opsonins through complement
        regulatory proteins (CD46, CD55).
        """
        self.tagged_synapses.pop(synapse_key, None)

    def _transition_state(self) -> None:
        """State machine for microglia activation.

        Surveying -> Activated: moderate damage signal
        Activated -> Phagocytic: high sustained damage
        Activated -> Surveying: damage resolved
        Phagocytic -> Activated: damage subsiding
        Phagocytic -> Surveying: full resolution
        """
        if self.state == MicrogliaState.SURVEYING:
            if self._activation_level > _ACTIVATION_THRESHOLD:
                self.state = MicrogliaState.ACTIVATED
        elif self.state == MicrogliaState.ACTIVATED:
            if self._activation_level > _PHAGOCYTIC_THRESHOLD:
                self.state = MicrogliaState.PHAGOCYTIC
            elif self._activation_level < _ACTIVATION_THRESHOLD * 0.5:
                self.state = MicrogliaState.SURVEYING
        elif self.state == MicrogliaState.PHAGOCYTIC:
            if self._activation_level < _PHAGOCYTIC_THRESHOLD * 0.5:
                self.state = MicrogliaState.ACTIVATED
            if self._activation_level < _ACTIVATION_THRESHOLD * 0.3:
                self.state = MicrogliaState.SURVEYING

    def _update_cytokines(self, dt: float) -> Tuple[float, float]:
        """Update cytokine release and decay.

        TNF-alpha and IL-1beta are pro-inflammatory cytokines released by
        activated microglia. They modulate nearby neuronal excitability:
          - TNF-alpha: increases AMPA receptor trafficking -> more excitable
          - IL-1beta: modulates GABA-A conductance -> altered inhibition

        Returns:
            Tuple of (tnf_released, il1b_released) in nM.
        """
        tnf_released = 0.0
        il1b_released = 0.0

        if self.state in (MicrogliaState.ACTIVATED, MicrogliaState.PHAGOCYTIC):
            # Release rate scales with activation level
            activation_factor = min(1.0, self._activation_level / _PHAGOCYTIC_THRESHOLD)

            tnf_released = _TNF_RELEASE_RATE * activation_factor * dt
            il1b_released = _IL1B_RELEASE_RATE * activation_factor * dt

            self.tnf_alpha += tnf_released
            self.il_1beta += il1b_released

        # Exponential decay of cytokines (enzymatic degradation + diffusion)
        self.tnf_alpha *= (1.0 - _TNF_DECAY_RATE * dt)
        self.il_1beta *= (1.0 - _IL1B_DECAY_RATE * dt)
        self.tnf_alpha = max(0.0, self.tnf_alpha)
        self.il_1beta = max(0.0, self.il_1beta)

        return tnf_released, il1b_released

    @property
    def is_active(self) -> bool:
        """Whether microglia is in any non-surveying state."""
        return self.state != MicrogliaState.SURVEYING

    @property
    def activation_level(self) -> float:
        """Current activation level (for diagnostics)."""
        return self._activation_level
