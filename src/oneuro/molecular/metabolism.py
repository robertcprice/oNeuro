"""Cellular energy metabolism — ATP production from real biochemical pathways.

Replaces the abstract ``energy`` float in MolecularNeuron with proper substrate
pools (glucose, pyruvate, lactate, O2) and energy currency (ATP/ADP/AMP,
NAD+/NADH).

Three ATP-producing pathways:
  1. **Glycolysis**: glucose -> 2 pyruvate + 2 ATP + 2 NADH  (fast, anaerobic)
  2. **Oxidative phosphorylation**: pyruvate + O2 -> 34 ATP   (slow, aerobic)
  3. **Lactate shuttle**: lactate -> pyruvate                  (astrocyte supply)

ATP consumers model the real energy budget of a neuron:
  - Na+/K+-ATPase: 50-70 % of brain ATP, scales with firing rate
  - SERCA pump: ER calcium reuptake
  - Vesicle recycling: synaptic vesicle endocytosis and refilling
  - Protein synthesis: from gene expression pipeline

When nQPU is available, electron transport chain complexes I and III use
quantum tunneling calculations for enhanced OxPhos efficiency.

All concentrations in mM.  Time in ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from oneuro.molecular.backend import HAS_NQPU, quantum_enzyme_tunneling


# ---------------------------------------------------------------------------
# Physical / biochemical constants
# ---------------------------------------------------------------------------

# Rate constants are given in mM/s in the literature; we store them as mM/ms
# by dividing by 1000.

# Glycolysis Vmax ~ 0.5 mM glucose consumed per second in brain tissue
_GLYCOLYSIS_VMAX_PER_MS: float = 0.5 / 1000.0  # mM/ms

# Km for hexokinase (rate-limiting step) ~ 0.05 mM glucose
_GLYCOLYSIS_KM: float = 0.05  # mM

# Oxidative phosphorylation Vmax ~ 0.1 mM pyruvate consumed per second
_OXPHOS_VMAX_PER_MS: float = 0.1 / 1000.0  # mM/ms

# Km for pyruvate dehydrogenase ~ 0.05 mM
_OXPHOS_PYRUVATE_KM: float = 0.05  # mM

# Km for O2 in cytochrome c oxidase ~ 0.001 mM
_OXPHOS_O2_KM: float = 0.001  # mM

# Lactate dehydrogenase (reverse): lactate -> pyruvate, Vmax ~ 0.3 mM/s
_LDH_VMAX_PER_MS: float = 0.3 / 1000.0  # mM/ms
_LDH_KM: float = 1.0  # mM

# Minimum ATP for cellular function
_ATP_MIN_FUNCTIONAL: float = 0.5  # mM

# Hypoxia threshold
_O2_HYPOXIC: float = 0.01  # mM

# Na+/K+-ATPase basal cost at rest: ~0.002 mM ATP/ms (~2 mM/s)
# Brain uses ~20 % of body O2 for ~1400g tissue; Na/K pump is 50-70 % of
# neuronal ATP budget.  Total production capacity ~4-6 mM/s (glycolysis +
# OxPhos with respiratory control), so basal pump at ~2 mM/s ≈ 50 % of
# budget.  Each spike requires ~4e8 Na+ ions pumped back; at typical
# intracellular volume this translates to ~0.00005 mM per spike.
_NAK_BASAL_PER_MS: float = 0.002    # mM/ms at rest (~2 mM/s)
_NAK_FIRING_SCALE: float = 0.00005  # additional mM/ms per Hz of firing

# Physiological ATP ceiling — cannot exceed ~8 mM intracellular
_ATP_CEILING: float = 8.0  # mM

# SERCA cost coefficient
_SERCA_PER_MS: float = 0.001  # mM ATP per mM Ca cycled per ms

# Vesicle recycling cost
_VESICLE_PER_MS: float = 0.0005  # mM ATP per vesicle release event per ms

# Protein synthesis cost
_PROTEIN_SYNTH_PER_MS: float = 0.0001  # mM ATP per unit gene-expression rate per ms

# ETC quantum tunneling barriers (eV) for proton transfer
_COMPLEX_I_BARRIER_EV: float = 0.30
_COMPLEX_III_BARRIER_EV: float = 0.25

# Quantum enhancement coefficient: how much tunneling boosts OxPhos rate
_QUANTUM_ENHANCEMENT: float = 0.15


# ---------------------------------------------------------------------------
# Michaelis-Menten helper
# ---------------------------------------------------------------------------

def _michaelis_menten(substrate: float, vmax: float, km: float) -> float:
    """Michaelis-Menten rate: V = Vmax * [S] / (Km + [S])."""
    if substrate <= 0.0:
        return 0.0
    return vmax * substrate / (km + substrate)


# ---------------------------------------------------------------------------
# CellularMetabolism
# ---------------------------------------------------------------------------

@dataclass
class CellularMetabolism:
    """Biochemical energy metabolism for a single neuron.

    Substrate pools and energy currency are tracked in mM.  The ``step``
    method advances glycolysis and oxidative phosphorylation each timestep,
    and individual ``*_cost`` methods deduct ATP for specific cellular
    processes.

    Backward-compatible ``energy`` property maps to the legacy 0-200 scale
    used by MolecularNeuron.
    """

    # ------------------------------------------------------------------
    # Substrate pools (mM)
    # ------------------------------------------------------------------
    glucose: float = 5.0        # Extracellular supply; normal blood ~5 mM
    pyruvate: float = 0.1       # Glycolysis intermediate
    lactate: float = 1.0        # Astrocyte shuttle supply
    oxygen: float = 0.05        # Dissolved O2

    # ------------------------------------------------------------------
    # Energy currency (mM)
    # ------------------------------------------------------------------
    atp: float = 3.0            # Normal intracellular 2-4 mM
    adp: float = 0.3
    amp: float = 0.05
    nad_plus: float = 0.5       # Oxidized NAD+
    nadh: float = 0.05          # Reduced NADH

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    _consumption_by_purpose: Dict[str, float] = field(
        default_factory=lambda: {}
    )
    _total_atp_produced: float = field(init=False, default=0.0)
    _total_atp_consumed: float = field(init=False, default=0.0)

    # ================================================================
    # Properties
    # ================================================================

    @property
    def energy_ratio(self) -> float:
        """ATP charge: ATP / (ATP + ADP + AMP).  Health indicator in [0, 1].

        A healthy cell maintains energy_ratio > 0.85.  Below 0.5 the cell
        is in severe metabolic crisis.
        """
        total = self.atp + self.adp + self.amp
        if total <= 0.0:
            return 0.0
        return self.atp / total

    @property
    def atp_available(self) -> bool:
        """Whether ATP pool exceeds the minimum required for cellular function."""
        return self.atp > _ATP_MIN_FUNCTIONAL

    @property
    def is_hypoxic(self) -> bool:
        """Whether dissolved O2 is below the hypoxia threshold."""
        return self.oxygen < _O2_HYPOXIC

    @property
    def energy(self) -> float:
        """Backward-compatible energy on a 0-200 scale.

        Maps ``energy_ratio`` linearly: 0.0 -> 0, 1.0 -> 200.
        MolecularNeuron checks ``energy > 0`` for alive and ``energy > 80``
        for division, so this preserves those semantics.
        """
        return self.energy_ratio * 200.0

    # ================================================================
    # Supply methods
    # ================================================================

    def supply_glucose(self, amount: float) -> None:
        """Add extracellular glucose (mM)."""
        self.glucose += amount

    def supply_oxygen(self, amount: float) -> None:
        """Add dissolved O2 (mM)."""
        self.oxygen += amount

    def supply_lactate(self, amount: float) -> None:
        """Add lactate from astrocyte shuttle (mM)."""
        self.lactate += amount

    # ================================================================
    # ATP consumption
    # ================================================================

    def consume_atp(self, amount_mM: float, purpose: str) -> bool:
        """Deduct *amount_mM* from the ATP pool.

        The consumed ATP is converted to ADP (and a small fraction to AMP).
        Tracks cumulative consumption by *purpose* for diagnostics.

        Returns:
            True if sufficient ATP was available and the deduction succeeded.
            False if ATP was insufficient (no deduction performed).
        """
        if amount_mM <= 0.0:
            return True
        if self.atp < amount_mM:
            return False

        self.atp -= amount_mM
        # ~95 % goes to ADP, ~5 % to AMP (adenylate kinase equilibrium)
        self.adp += amount_mM * 0.95
        self.amp += amount_mM * 0.05

        self._consumption_by_purpose[purpose] = (
            self._consumption_by_purpose.get(purpose, 0.0) + amount_mM
        )
        self._total_atp_consumed += amount_mM
        return True

    # ---- Specific consumers ----

    def na_k_atpase_cost(self, dt: float, firing_rate: float) -> bool:
        """Na+/K+-ATPase pump cost.  Dominant brain ATP consumer (50-70 %).

        The pump restores ionic gradients after every action potential.
        Basal rate covers resting leak; firing_rate (Hz) adds proportional
        cost for each spike's Na+/K+ exchange.

        Args:
            dt: Timestep in ms.
            firing_rate: Instantaneous firing rate in Hz.

        Returns:
            True if ATP was available.
        """
        cost = (_NAK_BASAL_PER_MS + _NAK_FIRING_SCALE * firing_rate) * dt
        return self.consume_atp(cost, "Na+/K+-ATPase")

    def serca_cost(self, dt: float, ca_cycling: float) -> bool:
        """SERCA pump cost for ER Ca2+ reuptake.

        Args:
            dt: Timestep in ms.
            ca_cycling: Intracellular Ca2+ flux magnitude (mM equivalent).

        Returns:
            True if ATP was available.
        """
        cost = _SERCA_PER_MS * ca_cycling * dt
        return self.consume_atp(cost, "SERCA")

    def vesicle_recycling_cost(self, dt: float, release_rate: float) -> bool:
        """Synaptic vesicle recycling ATP cost.

        Endocytosis, re-acidification, and neurotransmitter refilling all
        require ATP.

        Args:
            dt: Timestep in ms.
            release_rate: Vesicle release events per ms.

        Returns:
            True if ATP was available.
        """
        cost = _VESICLE_PER_MS * release_rate * dt
        return self.consume_atp(cost, "vesicle_recycling")

    def protein_synthesis_cost(self, dt: float, gene_expression_rate: float) -> bool:
        """ATP cost of ribosomal protein synthesis driven by gene expression.

        Each amino acid addition costs ~4 ATP equivalents (2 GTP + 1 ATP for
        charging tRNA + 1 ATP for proofreading).

        Args:
            dt: Timestep in ms.
            gene_expression_rate: Aggregate transcription/translation activity
                                  (arbitrary units; 1.0 = baseline).

        Returns:
            True if ATP was available.
        """
        cost = _PROTEIN_SYNTH_PER_MS * gene_expression_rate * dt
        return self.consume_atp(cost, "protein_synthesis")

    # ================================================================
    # Metabolic pathways
    # ================================================================

    def glycolysis(self, dt: float) -> float:
        """Glycolysis: glucose -> 2 pyruvate + 2 ATP + 2 NADH.

        Michaelis-Menten kinetics with hexokinase as the rate-limiting step.
        Anaerobic: does not require O2.

        Args:
            dt: Timestep in ms.

        Returns:
            Amount of glucose consumed (mM) this step.
        """
        # Rate-limited by glucose availability AND NAD+ (electron acceptor)
        glucose_rate = _michaelis_menten(self.glucose, _GLYCOLYSIS_VMAX_PER_MS, _GLYCOLYSIS_KM)

        # NAD+ is required as electron acceptor; limit rate if depleted
        nad_factor = self.nad_plus / (self.nad_plus + 0.05) if self.nad_plus > 0 else 0.0
        glucose_consumed = glucose_rate * nad_factor * dt

        # Cannot consume more glucose than available
        glucose_consumed = min(glucose_consumed, self.glucose)

        if glucose_consumed <= 0.0:
            return 0.0

        # Stoichiometry: 1 glucose -> 2 pyruvate + 2 ATP + 2 NADH
        self.glucose -= glucose_consumed
        self.pyruvate += glucose_consumed * 2.0

        atp_produced = glucose_consumed * 2.0
        self.atp += atp_produced
        # ATP comes from ADP phosphorylation
        adp_used = min(self.adp, atp_produced)
        self.adp -= adp_used

        nadh_produced = glucose_consumed * 2.0
        nad_converted = min(self.nad_plus, nadh_produced)
        self.nad_plus -= nad_converted
        self.nadh += nad_converted

        self._total_atp_produced += atp_produced
        return glucose_consumed

    def oxidative_phosphorylation(self, dt: float) -> float:
        """Oxidative phosphorylation: pyruvate + O2 -> ~34 ATP + CO2.

        Combines pyruvate dehydrogenase, TCA cycle, and electron transport
        chain.  Rate is limited by both pyruvate and O2 availability.

        When nQPU is available, ETC complexes I and III use quantum tunneling
        for proton transfer, enhancing the effective rate.

        Args:
            dt: Timestep in ms.

        Returns:
            Amount of pyruvate consumed (mM) this step.
        """
        # Dual-substrate Michaelis-Menten: limited by BOTH pyruvate and O2
        pyruvate_rate = _michaelis_menten(
            self.pyruvate, _OXPHOS_VMAX_PER_MS, _OXPHOS_PYRUVATE_KM
        )
        o2_factor = self.oxygen / (_OXPHOS_O2_KM + self.oxygen) if self.oxygen > 0 else 0.0

        # NADH is the electron donor for ETC; limit if depleted
        nadh_factor = self.nadh / (self.nadh + 0.01) if self.nadh > 0 else 0.0

        # Respiratory control: ADP stimulates OxPhos (State 3 respiration).
        # When ATP is plentiful and ADP is low, OxPhos slows down (State 4).
        # This is the primary feedback mechanism for metabolic homeostasis.
        # ADP_factor ranges from ~0.1 (resting, low ADP) to ~1.0 (active, high ADP).
        adp_factor = self.adp / (self.adp + 0.1)  # Km ~0.1 mM for ADP stimulation

        effective_rate = pyruvate_rate * o2_factor * nadh_factor * adp_factor

        # Quantum tunneling enhancement for ETC complexes I and III.
        # quantum_enzyme_tunneling uses nQPU when available, otherwise WKB fallback.
        tunnel_I = quantum_enzyme_tunneling(
            barrier_eV=_COMPLEX_I_BARRIER_EV, mass_amu=1.008, temperature_K=310.0
        )
        tunnel_III = quantum_enzyme_tunneling(
            barrier_eV=_COMPLEX_III_BARRIER_EV, mass_amu=1.008, temperature_K=310.0
        )
        combined_tunnel = (tunnel_I + tunnel_III) / 2.0
        quantum_factor = 1.0 + combined_tunnel * _QUANTUM_ENHANCEMENT

        effective_rate *= quantum_factor

        pyruvate_consumed = effective_rate * dt
        pyruvate_consumed = min(pyruvate_consumed, self.pyruvate)

        if pyruvate_consumed <= 0.0:
            return 0.0

        # Stoichiometry: 1 pyruvate -> ~34 ATP (via TCA + ETC)
        # O2 consumption: ~2.5 O2 per pyruvate (complete oxidation)
        o2_consumed = pyruvate_consumed * 2.5
        o2_consumed = min(o2_consumed, self.oxygen)

        self.pyruvate -= pyruvate_consumed
        self.oxygen -= o2_consumed

        atp_produced = pyruvate_consumed * 34.0
        self.atp += atp_produced
        adp_used = min(self.adp, atp_produced)
        self.adp -= adp_used

        # NADH is oxidized back to NAD+ in the ETC
        nadh_oxidized = pyruvate_consumed * 10.0  # ~10 NADH per pyruvate in TCA
        nadh_oxidized = min(nadh_oxidized, self.nadh)
        self.nadh -= nadh_oxidized
        self.nad_plus += nadh_oxidized

        self._total_atp_produced += atp_produced
        return pyruvate_consumed

    def lactate_to_pyruvate(self, dt: float) -> float:
        """Lactate shuttle: lactate -> pyruvate (+ NAD+ -> NADH).

        Astrocytes export lactate which neurons convert to pyruvate via
        lactate dehydrogenase (reverse direction).

        Args:
            dt: Timestep in ms.

        Returns:
            Amount of lactate consumed (mM) this step.
        """
        rate = _michaelis_menten(self.lactate, _LDH_VMAX_PER_MS, _LDH_KM)

        # Requires NAD+ as electron acceptor
        nad_factor = self.nad_plus / (self.nad_plus + 0.05) if self.nad_plus > 0 else 0.0

        lactate_consumed = rate * nad_factor * dt
        lactate_consumed = min(lactate_consumed, self.lactate)

        if lactate_consumed <= 0.0:
            return 0.0

        # Stoichiometry: 1 lactate -> 1 pyruvate, 1 NAD+ -> 1 NADH
        self.lactate -= lactate_consumed
        self.pyruvate += lactate_consumed

        nad_converted = min(self.nad_plus, lactate_consumed)
        self.nad_plus -= nad_converted
        self.nadh += nad_converted

        return lactate_consumed

    def _adenylate_kinase(self, dt: float) -> None:
        """Adenylate kinase equilibrium: AMP + ATP -> 2 ADP.

        This near-equilibrium enzyme rapidly interconverts adenine nucleotides.
        When ``consume_atp`` converts ATP -> ADP + AMP, adenylate kinase
        recycles AMP back into ADP, keeping the ADP pool available for
        re-phosphorylation by glycolysis and OxPhos.

        The forward reaction rate is proportional to [AMP][ATP], modeling
        mass-action kinetics for this fast cytoplasmic enzyme.
        """
        if self.amp <= 0.0 or self.atp <= 0.0:
            return

        # AK forward: AMP + ATP -> 2 ADP
        # Rate constant: ~0.5 / (mM * ms), fast enzyme
        ak_rate = 0.5

        forward_flux = ak_rate * self.amp * self.atp * dt
        # Cannot convert more AMP or ATP than available
        forward_flux = min(forward_flux, self.amp, self.atp)

        if forward_flux > 0:
            self.amp -= forward_flux
            self.atp -= forward_flux
            self.adp += 2.0 * forward_flux

    # ================================================================
    # Core simulation step
    # ================================================================

    def step(self, dt: float) -> None:
        """Advance metabolism by *dt* milliseconds.

        Runs all three pathways in sequence:
          1. Lactate shuttle (convert astrocyte lactate to pyruvate)
          2. Glycolysis (glucose -> pyruvate + ATP)
          3. Oxidative phosphorylation (pyruvate + O2 -> ATP)

        Substrates are auto-managed: glucose and O2 are replenished at a
        slow basal rate simulating capillary perfusion, unless the caller
        manages supply explicitly.

        Args:
            dt: Timestep in ms.
        """
        # 1. Adenylate kinase equilibrium: 2 ADP <-> ATP + AMP
        #    This is the crucial recycling reaction that converts AMP back
        #    to ADP so OxPhos can phosphorylate it.  The enzyme is fast
        #    (near-equilibrium), so we drive toward Keq ~ 1.0 each step.
        self._adenylate_kinase(dt)

        # 2. Lactate shuttle feeds pyruvate pool
        self.lactate_to_pyruvate(dt)

        # 3. Glycolysis: fast anaerobic ATP
        self.glycolysis(dt)

        # 4. OxPhos: slow but high-yield aerobic ATP
        self.oxidative_phosphorylation(dt)

        # 5. Basal capillary perfusion — homeostatic delivery from blood.
        #    In vivo, cerebral blood flow maintains steady-state substrate
        #    concentrations.  Delivery is proportional to deficit from
        #    set-point (feedback via neurovascular coupling).
        #
        #    The time constant must be short enough (~50 ms) to replace
        #    substrates at the rate they are consumed.  This models the
        #    continuous capillary blood supply to the neuron.
        #    For ischemia/hypoxia simulations, the caller should override
        #    by setting substrates to zero after each step.
        _glucose_set = 5.0    # mM, normal extracellular
        _o2_set = 0.05        # mM, normal dissolved
        _lactate_set = 1.0    # mM, normal astrocyte supply
        _perfusion_tau = 50.0  # ms, neurovascular coupling response time

        self.glucose += max(0.0, _glucose_set - self.glucose) / _perfusion_tau * dt
        self.oxygen += max(0.0, _o2_set - self.oxygen) / _perfusion_tau * dt
        self.lactate += max(0.0, _lactate_set - self.lactate) / _perfusion_tau * dt

        # 6. Clamp all pools to non-negative; cap ATP at physiological ceiling
        self.glucose = max(0.0, min(self.glucose, 15.0))   # Hyperglycemia cap
        self.pyruvate = max(0.0, min(self.pyruvate, 5.0))  # Reasonable ceiling
        self.lactate = max(0.0, min(self.lactate, 5.0))
        self.oxygen = max(0.0, min(self.oxygen, 0.2))      # Saturation limit
        self.atp = max(0.0, min(self.atp, _ATP_CEILING))
        self.adp = max(0.0, self.adp)
        self.amp = max(0.0, self.amp)
        self.nad_plus = max(0.0, self.nad_plus)
        self.nadh = max(0.0, self.nadh)

    # ================================================================
    # Diagnostics
    # ================================================================

    @property
    def consumption_summary(self) -> Dict[str, float]:
        """Cumulative ATP consumed by each purpose (mM)."""
        return dict(self._consumption_by_purpose)

    @property
    def total_atp_produced(self) -> float:
        """Cumulative ATP produced across all pathways (mM)."""
        return self._total_atp_produced

    @property
    def total_atp_consumed(self) -> float:
        """Cumulative ATP consumed across all purposes (mM)."""
        return self._total_atp_consumed

    def __repr__(self) -> str:
        return (
            f"CellularMetabolism("
            f"ATP={self.atp:.2f} ADP={self.adp:.2f} AMP={self.amp:.2f} "
            f"ratio={self.energy_ratio:.3f} "
            f"glc={self.glucose:.2f} pyr={self.pyruvate:.3f} "
            f"lac={self.lactate:.2f} O2={self.oxygen:.4f}"
            f")"
        )
