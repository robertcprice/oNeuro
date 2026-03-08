"""MolecularSynapse — full molecular synaptic transmission pipeline.

The complete sequence:
  1. Presynaptic spike → Ca²⁺ influx → vesicle fusion
  2. NT release into cleft (quantity from vesicle pool)
  3. Cleft dynamics: diffusion + enzyme degradation + reuptake
  4. Postsynaptic receptor binding → conductance change

STDP via receptor trafficking: LTP = insert receptors, LTD = remove.
.effective_weight property for backwards compatibility with OrganicSynapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from oneuro.molecular.neurotransmitters import NeurotransmitterMolecule, NEUROTRANSMITTER_LIBRARY
from oneuro.molecular.enzymes import SynapticEnzyme, ENZYME_LIBRARY
from oneuro.molecular.receptors import ReceptorType


@dataclass
class VesiclePool:
    """Presynaptic vesicle pool for neurotransmitter release.

    Three pools as in real synapses:
    - Readily releasable pool (RRP): immediately available
    - Recycling pool: replenishes RRP
    - Reserve pool: slow backup
    """

    nt_name: str = "glutamate"
    nt_per_vesicle_nM: float = 3000.0  # ~3000 molecules per vesicle ≈ 3000 nM burst

    # Pool sizes (in vesicles)
    rrp: float = 10.0  # Readily releasable
    rrp_max: float = 10.0
    recycling: float = 50.0
    recycling_max: float = 50.0
    reserve: float = 200.0
    reserve_max: float = 200.0

    # Replenishment rates (vesicles per ms)
    rrp_refill_rate: float = 0.05  # From recycling → RRP
    recycling_refill_rate: float = 0.01  # From reserve → recycling

    # Release probability
    base_release_prob: float = 0.3  # Per spike, fraction of RRP released

    def release(self, ca_level_nM: float = 1000.0) -> float:
        """Release vesicles. Returns NT concentration (nM) released into cleft.

        Ca²⁺ modulates release probability.
        """
        # Ca-dependent release probability
        ca_factor = min(2.0, ca_level_nM / 500.0)
        release_prob = self.base_release_prob * ca_factor

        vesicles_released = self.rrp * release_prob
        vesicles_released = min(vesicles_released, self.rrp)

        self.rrp -= vesicles_released
        nt_released = vesicles_released * self.nt_per_vesicle_nM
        return nt_released

    def replenish(self, dt: float) -> None:
        """Replenish vesicle pools over dt ms."""
        # Reserve → Recycling
        transfer_to_recycling = min(
            self.recycling_refill_rate * dt,
            self.reserve,
            self.recycling_max - self.recycling,
        )
        self.reserve -= transfer_to_recycling
        self.recycling += transfer_to_recycling

        # Recycling → RRP
        transfer_to_rrp = min(
            self.rrp_refill_rate * dt,
            self.recycling,
            self.rrp_max - self.rrp,
        )
        self.recycling -= transfer_to_rrp
        self.rrp += transfer_to_rrp


@dataclass
class SynapticCleft:
    """The synaptic cleft with diffusion, degradation, and reuptake dynamics."""

    nt_name: str = "glutamate"
    concentration_nM: float = 0.0
    volume_fL: float = 0.01  # ~10 attoliters = typical cleft volume

    # Clearance parameters
    _diffusion_rate: float = 0.5  # Fraction per ms that diffuses away
    _reuptake_rate: float = 0.1  # Fraction per ms reuptaken by transporters

    # Enzymes present in the cleft
    enzymes: List[SynapticEnzyme] = field(default_factory=list)

    def __post_init__(self):
        if not self.enzymes:
            # Auto-populate enzymes for this NT
            for enz_name, enz_info in ENZYME_LIBRARY.items():
                target = enz_info["target_nt"]
                if isinstance(target, list):
                    if self.nt_name in target:
                        self.enzymes.append(SynapticEnzyme(name=enz_name))
                elif target == self.nt_name:
                    self.enzymes.append(SynapticEnzyme(name=enz_name))

    def add_nt(self, amount_nM: float) -> None:
        """Add neurotransmitter to the cleft (from vesicle release)."""
        self.concentration_nM += amount_nM

    def update(self, dt: float) -> float:
        """Update cleft dynamics. Returns current concentration."""
        if self.concentration_nM <= 0:
            return 0.0

        # Enzymatic degradation
        for enzyme in self.enzymes:
            degraded = enzyme.degrade(self.concentration_nM, dt)
            self.concentration_nM -= degraded

        # Diffusion out of cleft
        self.concentration_nM *= (1.0 - self._diffusion_rate * dt)

        # Reuptake by transporters
        self.concentration_nM *= (1.0 - self._reuptake_rate * dt)

        self.concentration_nM = max(0.0, self.concentration_nM)
        return self.concentration_nM


@dataclass
class MolecularSynapse:
    """A synapse with full molecular transmission pipeline.

    Compatible with OrganicSynapse interface via .effective_weight property.

    Enhanced with:
    - NMDA-gated LTP (ketamine blocks it via NMDA conductance_scale)
    - BCM metaplasticity (sliding LTP threshold based on activity history)
    - Synaptic tagging & capture (strong stim → tag, captures PRPs)
    - Homeostatic scaling (global multiplicative adjustment)
    """

    # Identity (matching OrganicSynapse)
    pre_neuron: int
    post_neuron: int

    # Neurotransmitter type
    nt_name: str = "glutamate"

    # Molecular components
    vesicle_pool: VesiclePool = field(init=False)
    cleft: SynapticCleft = field(init=False)

    # Postsynaptic receptor configuration
    _postsynaptic_receptor_count: Dict[ReceptorType, int] = field(default_factory=dict)

    # OrganicSynapse-compatible fields
    delay: float = 1.0  # ms
    strength: float = 1.0  # [0, 1] synaptic health
    age: float = 0.0

    # STDP tracking (matching OrganicSynapse)
    last_pre_spike: float = -1000.0
    last_post_spike: float = -1000.0

    # Eligibility trace
    eligibility_trace: float = 0.0
    eligibility_decay: float = 0.95
    pre_activity_avg: float = 0.0
    post_activity_avg: float = 0.0

    # NMDA gating for LTP (set by network from postsynaptic neuron's NMDA conductance_scale)
    _nmda_scale: float = field(init=False, default=1.0)

    # BCM metaplasticity: sliding threshold
    _bcm_theta: float = field(init=False, default=0.5)
    _post_activity_history: float = field(init=False, default=0.0)

    # Synaptic tagging & capture
    _tagged: bool = field(init=False, default=False)
    _tag_strength: float = field(init=False, default=0.0)
    _tag_decay_tau: float = field(init=False, default=60000.0)  # Tag lasts ~1 hour

    # Homeostatic scaling factor
    _homeostatic_scale: float = field(init=False, default=1.0)

    # Plasticity restriction (from perineuronal net)
    _plasticity_factor: float = field(init=False, default=1.0)

    # CaMKII level from postsynaptic neuron's calcium system (set by network)
    _camkii_level: Optional[float] = field(init=False, default=None)

    # Dendritic spine (structural plasticity coupling)
    spine: Optional[object] = field(init=False, default=None)  # DendriticSpine

    # Internal state
    _pending_release: float = field(init=False, default=0.0)
    _delay_buffer: List[Tuple[float, float]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.vesicle_pool = VesiclePool(nt_name=self.nt_name)
        self.cleft = SynapticCleft(nt_name=self.nt_name)

        # Default receptor configuration based on NT
        if not self._postsynaptic_receptor_count:
            if self.nt_name == "glutamate":
                self._postsynaptic_receptor_count = {
                    ReceptorType.AMPA: 50,
                    ReceptorType.NMDA: 20,
                }
            elif self.nt_name == "gaba":
                self._postsynaptic_receptor_count = {ReceptorType.GABA_A: 40}
            elif self.nt_name == "acetylcholine":
                self._postsynaptic_receptor_count = {ReceptorType.nAChR: 30}
            elif self.nt_name == "dopamine":
                self._postsynaptic_receptor_count = {
                    ReceptorType.D1: 10,
                    ReceptorType.D2: 10,
                }
            elif self.nt_name == "serotonin":
                self._postsynaptic_receptor_count = {
                    ReceptorType.HT_1A: 10,
                    ReceptorType.HT_2A: 5,
                }

    @property
    def weight(self) -> float:
        """Abstract synaptic weight derived from receptor count.

        Maps total receptor count to [0, 2] range for OrganicSynapse compatibility.
        Includes homeostatic scaling factor.
        """
        total = sum(self._postsynaptic_receptor_count.values())
        # Normalize: 50 receptors ≈ weight 1.0
        return min(2.0, total / 50.0) * self.strength * self._homeostatic_scale

    @property
    def effective_weight(self) -> float:
        """Backwards-compatible weight for OrganicSynapse integration."""
        return self.weight

    @property
    def receptor_count(self) -> Dict[ReceptorType, int]:
        return dict(self._postsynaptic_receptor_count)

    def presynaptic_spike(self, time: float, ca_level_nM: float = 1000.0) -> None:
        """Handle presynaptic action potential.

        1. Ca²⁺ influx triggers vesicle fusion
        2. NT released into cleft (after delay)
        """
        self.last_pre_spike = time
        nt_released = self.vesicle_pool.release(ca_level_nM)
        # Buffer the release with synaptic delay
        self._delay_buffer.append((time + self.delay, nt_released))

    def update(self, time: float, dt: float) -> float:
        """Update synapse for one timestep.

        Returns cleft concentration (nM) for postsynaptic neuron to read.
        """
        self.age += dt

        # Check delay buffer — release NT into cleft when delay expires
        remaining = []
        for release_time, amount in self._delay_buffer:
            if time >= release_time:
                self.cleft.add_nt(amount)
            else:
                remaining.append((release_time, amount))
        self._delay_buffer = remaining

        # Update cleft dynamics (degradation, diffusion, reuptake)
        conc = self.cleft.update(dt)

        # Replenish vesicle pools
        self.vesicle_pool.replenish(dt)

        return conc

    def get_postsynaptic_nt_dict(self) -> Dict[str, float]:
        """Get NT concentration dict for the postsynaptic neuron."""
        return {self.nt_name: self.cleft.concentration_nM}

    # ---- STDP via receptor trafficking ----

    def update_stdp(
        self, pre_fired: bool, post_fired: bool, time: float, dt: float = 0.1
    ) -> None:
        """STDP implemented as receptor trafficking with NMDA gating and BCM.

        LTP (pre→post, causal): insert AMPA/NMDA receptors.
          - GATED by postsynaptic NMDA conductance_scale (ketamine blocks LTP)
          - BCM: LTP threshold slides with postsynaptic activity history
        LTD (post→pre, anti-causal): remove AMPA receptors.
        Synaptic tagging: strong LTP sets tag for plasticity-related protein capture.
        """
        if pre_fired:
            self.last_pre_spike = time
        if post_fired:
            self.last_post_spike = time
            # BCM: update postsynaptic activity history (sliding average)
            self._post_activity_history = (
                0.999 * self._post_activity_history + 0.001
            )

        # Decay activity history toward baseline
        self._post_activity_history *= (1.0 - 0.0001 * dt)

        # BCM sliding threshold: theta rises with activity, making LTP harder
        self._bcm_theta = max(0.1, min(0.9, self._post_activity_history * 2.0 + 0.3))

        # Tag decay
        if self._tagged:
            self._tag_strength -= dt / self._tag_decay_tau
            if self._tag_strength <= 0:
                self._tagged = False
                self._tag_strength = 0.0

        # Plasticity factor from perineuronal net
        pf = self._plasticity_factor

        if pre_fired and self.last_post_spike > 0:
            dt_spike = time - self.last_post_spike
            if 0 < dt_spike < 20.0:
                # Anti-causal: post fired recently before pre → LTD
                ltd_strength = math.exp(-dt_spike / 20.0) * 0.5 * pf
                self._remove_receptors(int(max(1, ltd_strength * 3)))
                # Structural LTD: shrink spine
                if self.spine is not None:
                    self.spine.structural_ltd(ltd_strength)

        if post_fired and self.last_pre_spike > 0:
            dt_spike = time - self.last_pre_spike
            if 0 < dt_spike < 20.0:
                # Causal: pre fired recently before post → LTP
                raw_ltp = math.exp(-dt_spike / 20.0) * 0.5

                # NMDA gating: LTP requires NMDA current (ketamine blocks this)
                nmda_gate = self._nmda_scale  # 1.0 normal, 0.0 fully blocked

                # BCM: only potentiate if strength exceeds sliding threshold
                bcm_gate = 1.0 if raw_ltp > self._bcm_theta else 0.3

                # CaMKII-scaled LTP: when calcium system is available, use
                # graded CaMKII activation instead of fixed exponential decay
                if self._camkii_level is not None:
                    ltp_strength = self._camkii_level * nmda_gate * bcm_gate * pf
                else:
                    ltp_strength = raw_ltp * nmda_gate * bcm_gate * pf

                receptors_to_insert = int(max(1, ltp_strength * 3))

                # Clamp to spine AMPA capacity if spine exists
                if self.spine is not None:
                    current_ampa = self._postsynaptic_receptor_count.get(
                        ReceptorType.AMPA, 0
                    )
                    max_insert = max(0, self.spine.ampa_capacity - current_ampa)
                    receptors_to_insert = min(receptors_to_insert, max_insert)

                if receptors_to_insert > 0:
                    self._insert_receptors(receptors_to_insert)

                # Structural LTP: enlarge spine
                if self.spine is not None:
                    self.spine.structural_ltp(ltp_strength)

                # Synaptic tagging: strong LTP sets a tag
                if ltp_strength > 0.3:
                    self._tagged = True
                    self._tag_strength = min(1.0, ltp_strength)

    def capture_prps(self, prp_available: float) -> None:
        """Synaptic tagging & capture: tagged synapses capture PRPs.

        PRPs (plasticity-related proteins) from gene expression convert
        early-LTP (tag-only) to late-LTP (structural change).
        """
        if self._tagged and prp_available > 0.1:
            # Capture PRPs → strengthen synapse permanently
            boost = min(0.1, prp_available * self._tag_strength * 0.05)
            self.strength = min(1.0, self.strength + boost)
            self._insert_receptors(int(max(1, boost * 10)))
            self._tagged = False
            self._tag_strength = 0.0

    def apply_homeostatic_scaling(self, target_activity: float,
                                   actual_activity: float, rate: float = 0.001) -> None:
        """Homeostatic synaptic scaling: maintain target firing rate.

        Multiplicative scaling of all AMPA receptors to keep network activity
        in the target range. Prevents runaway excitation or silence.
        """
        if actual_activity > 0:
            ratio = target_activity / actual_activity
        else:
            ratio = 2.0  # Upscale if silent

        # Slow adjustment toward ratio
        self._homeostatic_scale += rate * (ratio - self._homeostatic_scale)
        self._homeostatic_scale = max(0.5, min(2.0, self._homeostatic_scale))

    def _insert_receptors(self, count: int) -> None:
        """LTP: insert more AMPA receptors (receptor trafficking)."""
        if ReceptorType.AMPA in self._postsynaptic_receptor_count:
            self._postsynaptic_receptor_count[ReceptorType.AMPA] += count
        elif self.nt_name == "glutamate":
            self._postsynaptic_receptor_count[ReceptorType.AMPA] = count

    def _remove_receptors(self, count: int) -> None:
        """LTD: remove AMPA receptors (receptor internalization)."""
        if ReceptorType.AMPA in self._postsynaptic_receptor_count:
            current = self._postsynaptic_receptor_count[ReceptorType.AMPA]
            self._postsynaptic_receptor_count[ReceptorType.AMPA] = max(1, current - count)

    # ---- Eligibility trace (matching OrganicSynapse) ----

    def update_eligibility(
        self, pre_active: float, post_active: float, dt: float = 0.1
    ) -> None:
        """Update eligibility trace for reward-modulated learning."""
        alpha = 1.0 - self.eligibility_decay
        self.pre_activity_avg = self.eligibility_decay * self.pre_activity_avg + alpha * pre_active
        self.post_activity_avg = self.eligibility_decay * self.post_activity_avg + alpha * post_active
        hebbian = self.pre_activity_avg * self.post_activity_avg
        self.eligibility_trace = self.eligibility_decay * self.eligibility_trace + (
            1.0 - self.eligibility_decay
        ) * hebbian

    def apply_reward(self, reward: float, learning_rate: float = 0.1) -> None:
        """Apply reward-modulated plasticity via receptor trafficking."""
        delta_receptors = int(learning_rate * reward * self.eligibility_trace * 10)
        if delta_receptors > 0:
            self._insert_receptors(delta_receptors)
        elif delta_receptors < 0:
            self._remove_receptors(abs(delta_receptors))

        # Update synaptic health
        if reward > 0:
            self.strength = min(1.0, self.strength + 0.01)
        else:
            self.strength = max(0.1, self.strength - 0.01)

    def should_prune(self) -> bool:
        """Check if synapse should be removed (matching OrganicSynapse)."""
        total_receptors = sum(self._postsynaptic_receptor_count.values())
        return self.strength < 0.1 or total_receptors < 5

    @classmethod
    def glutamatergic(cls, pre: int, post: int) -> "MolecularSynapse":
        return cls(pre_neuron=pre, post_neuron=post, nt_name="glutamate")

    @classmethod
    def gabaergic(cls, pre: int, post: int) -> "MolecularSynapse":
        return cls(pre_neuron=pre, post_neuron=post, nt_name="gaba")

    @classmethod
    def dopaminergic(cls, pre: int, post: int) -> "MolecularSynapse":
        return cls(pre_neuron=pre, post_neuron=post, nt_name="dopamine")

    @classmethod
    def cholinergic(cls, pre: int, post: int) -> "MolecularSynapse":
        return cls(pre_neuron=pre, post_neuron=post, nt_name="acetylcholine")
