"""Cytoskeleton with Penrose-Hameroff Orch-OR quantum consciousness model.

Implements the cellular cytoskeleton -- microtubules, actin filaments,
neurofilaments, and microtubule-associated proteins (MAPs) -- with a
particular focus on the quantum dynamics of tubulin dimers as described
by the Orchestrated Objective Reduction (Orch-OR) theory.

Orch-OR proposes that consciousness arises from quantum computations in
brain microtubules.  Tubulin proteins can exist in quantum superposition
of two conformational states (alpha and beta).  When the gravitational
self-energy of the superposed mass distribution reaches the Diosi-Penrose
threshold (E_G * t >= hbar/2), the quantum state undergoes "objective
reduction" -- a quantum-gravity-induced collapse that constitutes a
moment of proto-conscious experience.

When nQPU is available, quantum evolution is delegated to the
PyOrchORSimulator (Rust-backed, GPU-accelerated).  Otherwise a
semiclassical fallback model is used that captures the essential
decoherence and collapse dynamics without full state-vector simulation.

References:
    - Penrose, R. (1994). Shadows of the Mind.
    - Hameroff, S. & Penrose, R. (2014). Consciousness in the universe.
    - Diosi, L. (1987). A universal master equation for the gravitational
      violation of quantum mechanics.
    - Craddock, T.J.A. et al. (2017). Anesthetic alterations of collective
      tubulin quantum dynamics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from oneuro.molecular.backend import HAS_NQPU, get_nqpu_metal

# Attempt to import the Rust-backed Orch-OR simulator.
# This is behind nqpu_metal's "experimental" feature flag and may not
# be compiled into every build.
PyOrchORSimulator = None
try:
    from nqpu_metal import PyOrchORSimulator  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    pass


# =====================================================================
# Physical constants (SI)
# =====================================================================

HBAR: float = 1.0546e-34          # Reduced Planck constant (J*s)
G_NEWTON: float = 6.674e-11       # Gravitational constant (N*m^2/kg^2)
TUBULIN_MASS: float = 1.1e-22     # Tubulin dimer mass ~110 kDa (kg)
SUPERPOSITION_SEP: float = 2.5e-9  # Conformational displacement (m)
K_BOLTZMANN: float = 1.381e-23    # Boltzmann constant (J/K)
BODY_TEMP_K: float = 310.0        # Human body temperature (K)

# Derived: gravitational self-energy for a single tubulin dimer
# E_G_single = G * m^2 / delta_x
E_G_SINGLE: float = G_NEWTON * TUBULIN_MASS * TUBULIN_MASS / SUPERPOSITION_SEP

# Tubulin dipole moment in Debye (measured: ~1740 D for alpha/beta dimer)
TUBULIN_DIPOLE_DEBYE: float = 1740.0


# =====================================================================
# Tubulin
# =====================================================================

@dataclass
class Tubulin:
    """A single tubulin dimer within a microtubule protofilament.

    In the Orch-OR framework, each tubulin can adopt one of two
    conformational states (alpha or beta), or exist in a quantum
    superposition of both.  The electric dipole moment (~1740 D) couples
    neighbouring tubulins electrostatically, enabling cooperative quantum
    computation along the protofilament lattice.

    Attributes:
        conformation: Current classical conformation -- "alpha" or "beta".
        dipole_moment: Electric dipole moment in Debye units.
        quantum_state: Index of the corresponding qubit in the nQPU
            simulator, or None when running in fallback mode.
        superposition: Whether this tubulin is currently in a quantum
            superposition of alpha and beta conformations.
    """

    conformation: str = "alpha"
    dipole_moment: float = TUBULIN_DIPOLE_DEBYE
    quantum_state: Optional[int] = None
    superposition: bool = False

    def __post_init__(self) -> None:
        if self.conformation not in ("alpha", "beta"):
            raise ValueError(
                f"conformation must be 'alpha' or 'beta', got '{self.conformation}'"
            )


# =====================================================================
# Microtubule
# =====================================================================

@dataclass
class Microtubule:
    """A single microtubule -- hollow cylinder of tubulin protofilaments.

    Standard microtubules have 13 protofilaments arranged in a helical
    lattice.  Each ring of 13 tubulins constitutes one "row" of the
    computational array.  In Orch-OR, quantum superpositions evolve
    under the tubulin Hamiltonian until the Diosi-Penrose gravitational
    self-energy threshold triggers objective reduction.

    When PyOrchORSimulator is available (nQPU with ``experimental``
    feature), quantum evolution uses the full state-vector simulator.
    Otherwise a semiclassical model tracks coherence decay and collapse
    timing using the same physical constants.

    Attributes:
        n_protofilaments: Number of protofilaments (biologically 13).
        n_rings: Number of tubulin rings along the microtubule axis.
        tubulins: Flat list of all tubulin dimers (n_protofilaments * n_rings).
        _simulator: nQPU PyOrchORSimulator instance, or None.
        _coherence_time_ms: Decoherence timescale in milliseconds.
        _superposition_start: Monotonic timestamp (seconds) when the
            current superposition epoch began.
    """

    n_protofilaments: int = 13
    n_rings: int = 20

    # Populated in __post_init__
    tubulins: List[Tubulin] = field(default_factory=list, repr=False)

    # Private state
    _simulator: Optional[object] = field(default=None, repr=False)
    _coherence_time_ms: float = field(default=25.0, repr=False)
    _superposition_start: float = field(default=0.0, repr=False)

    # Network-scale factor for Diosi-Penrose threshold.
    # Real Orch-OR requires ~10^9 simultaneous tubulins across a cortical
    # column.  Our small networks have ~10^4 tubulins.  This factor scales
    # E_G in the collapse criterion so OR events fire on biologically
    # relevant timescales (~25ms at 40 Hz gamma).  Set by the network
    # based on total neuron count.
    network_tubulin_scale: float = field(default=1.0, repr=False)

    # Internal bookkeeping
    _coherence_level: float = field(default=0.0, repr=False, init=False)
    _or_event_count: int = field(default=0, repr=False, init=False)
    _elapsed_since_superposition_ms: float = field(default=0.0, repr=False, init=False)
    _n_superposed_cache: int = field(default=0, repr=False, init=False)

    def __post_init__(self) -> None:
        n_total = self.n_protofilaments * self.n_rings
        if not self.tubulins:
            self.tubulins = [
                Tubulin(
                    conformation="alpha",
                    quantum_state=i if PyOrchORSimulator is not None else None,
                )
                for i in range(n_total)
            ]

        # Attempt to create the nQPU simulator
        if PyOrchORSimulator is not None and self._simulator is None:
            try:
                self._simulator = PyOrchORSimulator(
                    num_tubulins=n_total,
                    coherence_time_ns=self._coherence_time_ms * 1e6,  # ms -> ns
                    temperature_kelvin=BODY_TEMP_K,
                )
            except Exception:
                self._simulator = None

    @property
    def n_tubulins(self) -> int:
        """Total number of tubulin dimers in this microtubule."""
        return len(self.tubulins)

    @property
    def n_superposed(self) -> int:
        """Number of tubulins currently in quantum superposition."""
        return self._n_superposed_cache

    # -----------------------------------------------------------------
    # Quantum evolution
    # -----------------------------------------------------------------

    def evolve(self, dt: float) -> None:
        """Evolve the quantum state of all tubulin superpositions.

        Args:
            dt: Timestep in milliseconds.

        When nQPU PyOrchORSimulator is available, delegates to the
        Rust-backed simulator which performs full Hamiltonian evolution
        with Trotterized ZZ interactions, transverse-field tunneling,
        and amplitude-damping decoherence.

        Fallback: a semiclassical model where coherence decays
        exponentially with time constant ``_coherence_time_ms``, and
        tubulins in superposition are independently evolved.
        """
        if self.n_superposed == 0:
            self._coherence_level = 0.0
            return

        if self._simulator is not None:
            self._evolve_nqpu(dt)
        else:
            self._evolve_semiclassical(dt)

    def _evolve_nqpu(self, dt: float) -> None:
        """Full quantum evolution via nQPU Orch-OR simulator."""
        sim = self._simulator
        try:
            # Convert dt from ms to ns for the Rust simulator
            dt_ns = dt * 1e6
            time_steps = max(1, int(dt_ns))
            sim.evolve(time_steps)  # type: ignore[union-attr]

            # Read back coherence
            self._coherence_level = sim.measure_coherence()  # type: ignore[union-attr]
        except Exception:
            # Fall back to semiclassical if nQPU call fails
            self._evolve_semiclassical(dt)

    def _evolve_semiclassical(self, dt: float) -> None:
        """Semiclassical coherence-decay model.

        Models decoherence as exponential decay of the off-diagonal
        density-matrix elements, with a temperature-dependent
        coherence time:

            C(t) = C_0 * exp(-t / T_coh)

        The coherence time is modulated by the Arrhenius factor for
        thermally activated decoherence.
        """
        self._elapsed_since_superposition_ms += dt

        if self._coherence_time_ms <= 0.0:
            self._coherence_level = 0.0
            return

        # Exponential coherence decay
        decay_factor = math.exp(-dt / self._coherence_time_ms)
        if self._coherence_level <= 0.0:
            # Freshly entered superposition -- start at full coherence
            self._coherence_level = 1.0
        self._coherence_level *= decay_factor

        # Clamp
        self._coherence_level = max(0.0, min(1.0, self._coherence_level))

    # -----------------------------------------------------------------
    # Objective reduction (Diosi-Penrose collapse)
    # -----------------------------------------------------------------

    def check_collapse(self) -> bool:
        """Check whether Diosi-Penrose objective reduction has occurred.

        The collapse criterion is:

            E_G * t >= hbar / 2

        where E_G is the gravitational self-energy of the superposed
        mass distribution and t is the time the superposition has
        persisted.  E_G scales linearly with the number of superposed
        tubulins:

            E_G = N * G * m^2 / delta_x

        Returns:
            True if objective reduction (OR) has occurred this check.
        """
        n_sup = self.n_superposed
        if n_sup == 0:
            return False

        # Gravitational self-energy for N coherently superposed tubulins.
        # E_G scales as N^2 for coherent superpositions (each pair of
        # superposed masses contributes).  The network_tubulin_scale
        # represents entangled tubulins across the cortical column that
        # this microtubule is part of.
        e_g = n_sup * n_sup * E_G_SINGLE * self.network_tubulin_scale

        # Time in superposition (seconds)
        t_s = self._elapsed_since_superposition_ms * 1e-3

        # Diosi-Penrose threshold: E_G * t >= hbar / 2
        if e_g * t_s >= HBAR / 2.0:
            self._collapse()
            return True

        return False

    def _collapse(self) -> None:
        """Collapse all superposed tubulins to definite conformations.

        After objective reduction, each tubulin in superposition is
        resolved to either alpha or beta.  The outcome is determined
        by the current coherence level -- higher coherence biases
        toward the energetically orchestrated outcome; lower coherence
        produces a more random result.

        Uses a simple deterministic hash-based "measurement" to avoid
        requiring a PRNG dependency, while still producing non-trivial
        outcomes that depend on the tubulin index and coherence state.
        """
        for i, t in enumerate(self.tubulins):
            if t.superposition:
                phase = math.sin(i * 0.618033988749895 + self._coherence_level * 3.14159)
                t.conformation = "alpha" if phase >= 0.0 else "beta"
                t.superposition = False

        self._n_superposed_cache = 0
        self._or_event_count += 1
        self._elapsed_since_superposition_ms = 0.0
        self._coherence_level = 0.0

    # -----------------------------------------------------------------
    # Consciousness event
    # -----------------------------------------------------------------

    def consciousness_event(self) -> bool:
        """Check whether an Orch-OR consciousness event has occurred.

        A consciousness event requires BOTH:
        1. Objective reduction (OR) has occurred (Diosi-Penrose threshold met).
        2. The quantum computation was sufficiently "integrated" -- meaning
           the coherence at the moment of collapse was above a meaningful
           threshold, indicating that the quantum state carried non-trivial
           information (not just random noise).

        The integration threshold (0.1) corresponds to requiring that at
        least ~10% of the maximum possible quantum coherence was present
        at collapse time.  This filters out trivially decohered states that
        would collapse with no meaningful computational content.

        Returns:
            True if a genuine Orch-OR consciousness event occurred.
        """
        # Record coherence BEFORE checking collapse (collapse resets it)
        pre_collapse_coherence = self._coherence_level
        integration_threshold = 0.1

        if self.check_collapse():
            return pre_collapse_coherence > integration_threshold

        return False

    # -----------------------------------------------------------------
    # Neural orchestration
    # -----------------------------------------------------------------

    def orchestration(
        self,
        dendritic_ca_nM: float = 50.0,
        map2_state: float = 1.0,
    ) -> None:
        """Neural activity orchestrates tubulin quantum states.

        Classical neural inputs modulate the quantum computation in the
        microtubule lattice, making the reduction "orchestrated" rather
        than random.  Two key modulators:

        1. **MAP2 binding** (dendrites): MAP2 proteins bind to the
           microtubule surface, stabilizing the lattice and extending
           the quantum coherence time.  MAP2 is regulated by
           phosphorylation (CaMKII, PKA).

        2. **Dendritic calcium**: Elevated Ca2+ from synaptic activity
           affects tubulin conformational dynamics.  Above ~200 nM,
           calcium promotes transition to superposition (quantum
           computation begins).  Above ~1000 nM, excessive calcium
           destabilises superpositions (protective decoherence).

        Args:
            dendritic_ca_nM: Dendritic calcium concentration in nM.
                Resting is ~50 nM; synaptic activation reaches 200-1000 nM.
            map2_state: MAP2 binding level in [0, 1].
                1.0 = fully bound (maximal stabilization).
                0.0 = fully unbound (e.g. hyperphosphorylated, Alzheimer's-like).
        """
        map2_state = max(0.0, min(1.0, map2_state))

        # MAP2 stabilization extends coherence time
        # Fully bound MAP2 can extend coherence by up to 2x
        # Fully unbound MAP2 halves it
        map2_factor = 0.5 + 1.5 * map2_state
        effective_coherence = self._coherence_time_ms * map2_factor

        # Ca2+ modulation of tubulin conformational dynamics
        if dendritic_ca_nM > 200.0 and dendritic_ca_nM <= 1000.0:
            # Moderate Ca2+: promotes superposition (synaptic activity)
            fraction_to_superpose = min(
                1.0, (dendritic_ca_nM - 200.0) / 800.0
            )
            n_to_superpose = int(fraction_to_superpose * self.n_tubulins * 0.3)

            for i, t in enumerate(self.tubulins):
                if i >= n_to_superpose:
                    break
                if not t.superposition:
                    t.superposition = True
                    self._n_superposed_cache += 1

            # Start coherence tracking if not already in superposition
            if self._coherence_level <= 0.0:
                self._coherence_level = 1.0
                self._elapsed_since_superposition_ms = 0.0
                self._superposition_start = time.monotonic()

        elif dendritic_ca_nM > 1000.0:
            # High Ca2+: destabilises superpositions (protective decoherence)
            excess_factor = min(1.0, (dendritic_ca_nM - 1000.0) / 5000.0)
            effective_coherence *= (1.0 - 0.8 * excess_factor)

        self._coherence_time_ms = max(0.1, effective_coherence)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def coherence(self) -> float:
        """Current quantum coherence level in [0, 1].

        0.0 = fully classical (no off-diagonal density matrix elements).
        1.0 = maximally coherent superposition.
        """
        return self._coherence_level

    @property
    def or_events(self) -> int:
        """Total number of objective reduction events in this microtubule."""
        return self._or_event_count

    @property
    def gravitational_self_energy(self) -> float:
        """Current E_G in Joules for the superposed tubulin mass."""
        return self.n_superposed * E_G_SINGLE

    @property
    def time_to_reduction_ms(self) -> float:
        """Estimated time until objective reduction, in milliseconds.

        Returns infinity if no tubulins are in superposition.
        Uses tau = hbar / (2 * E_G) from the Diosi-Penrose formula.
        """
        e_g = self.gravitational_self_energy
        if e_g <= 0.0:
            return float("inf")
        # tau = hbar / (2 * E_G), convert seconds to ms
        tau_s = HBAR / (2.0 * e_g)
        return tau_s * 1e3

    def __repr__(self) -> str:
        return (
            f"Microtubule("
            f"proto={self.n_protofilaments}, "
            f"rings={self.n_rings}, "
            f"tubulins={self.n_tubulins}, "
            f"superposed={self.n_superposed}, "
            f"coherence={self._coherence_level:.4f}, "
            f"OR_events={self._or_event_count})"
        )


# =====================================================================
# Cytoskeleton
# =====================================================================

@dataclass
class Cytoskeleton:
    """Complete neuronal cytoskeleton with Orch-OR consciousness dynamics.

    The cytoskeleton is the structural scaffold of the neuron, composed of:

    1. **Microtubules** -- hollow tubes of polymerized tubulin, the
       primary computational substrate in Orch-OR theory.
    2. **Actin filaments** -- dynamic polymers driving dendritic spine
       morphology and synaptic plasticity.
    3. **Neurofilaments** -- intermediate filaments that set axon caliber
       and thus conduction velocity.
    4. **Microtubule-associated proteins** (MAPs):
       - **MAP2** (dendrites): stabilizes microtubules, extends quantum
         coherence time.
       - **Tau** (axons): stabilizes axonal microtubules.  Pathological
         hyperphosphorylation (Alzheimer's disease) causes tau to detach,
         destabilizing microtubules and disrupting quantum coherence.

    The ``anesthetic_factor`` models the effect of anesthetic agents
    (e.g. xenon, sevoflurane) that bind to hydrophobic pockets in
    tubulin, suppressing quantum coherence and thus consciousness
    according to the Orch-OR framework.

    Attributes:
        microtubules: List of Microtubule instances in this neuron.
        actin_polymerization: Fraction of actin in polymerized (F-actin)
            form [0, 1].  Higher values indicate active spine growth.
        neurofilament_density: Relative neurofilament content [0, 1].
            Higher density -> larger axon caliber -> faster conduction.
        map2_level: MAP2 binding level [0, 1].  1.0 = fully bound.
        tau_level: Tau protein binding level [0, 1].
        tau_phosphorylation: Tau phosphorylation level [0, 1].
            High values = pathological (Alzheimer's-like).
        anesthetic_factor: Multiplier on coherence time [0, 1].
            1.0 = normal.  < 1.0 = under anesthesia.
    """

    microtubules: List[Microtubule] = field(default_factory=list)

    # Actin dynamics
    actin_polymerization: float = 0.5

    # Neurofilaments
    neurofilament_density: float = 0.5

    # MAPs
    map2_level: float = 1.0
    tau_level: float = 1.0
    tau_phosphorylation: float = 0.0

    # Anesthesia
    anesthetic_factor: float = 1.0

    # Internal OR event counter (reset by user)
    _or_count: int = field(default=0, repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.microtubules:
            # Default: 3 microtubules per neuron (simplified)
            self.microtubules = [Microtubule() for _ in range(3)]

    # -----------------------------------------------------------------
    # Main evolution
    # -----------------------------------------------------------------

    def step(self, dt: float, ca_nM: float = 50.0) -> None:
        """Evolve the entire cytoskeleton for one timestep.

        This is the main update loop that:
        1. Computes effective MAP2 binding (accounting for tau pathology).
        2. Applies anesthetic modulation to coherence times.
        3. Orchestrates tubulin quantum states via neural Ca2+ signals.
        4. Evolves quantum dynamics of each microtubule.
        5. Checks for Diosi-Penrose objective reduction events.
        6. Updates actin and neurofilament dynamics.

        Args:
            dt: Timestep in milliseconds.
            ca_nM: Dendritic calcium concentration in nanomolar.
        """
        # ---- 1. Tau pathology reduces microtubule stability ----
        # Hyperphosphorylated tau detaches from microtubules, reducing
        # the effective stabilization.  This models the progressive
        # microtubule collapse seen in Alzheimer's disease.
        effective_tau = self.tau_level * (1.0 - self.tau_phosphorylation)

        # MAP2 and tau together determine microtubule stability.
        # In dendrites, MAP2 dominates; in axons, tau dominates.
        # We use MAP2 as the primary modulator for quantum coherence
        # since Orch-OR focuses on dendritic processing.
        effective_map2 = self.map2_level * (0.7 + 0.3 * effective_tau)

        for mt in self.microtubules:
            # ---- 2. Anesthetic modulation ----
            # Anesthetics reduce the coherence time, suppressing
            # quantum computation.  anesthetic_factor < 1 shortens
            # the effective decoherence timescale.
            base_coherence_ms = mt._coherence_time_ms
            mt._coherence_time_ms = base_coherence_ms * max(0.0, min(1.0, self.anesthetic_factor))

            # ---- 3. Neural orchestration ----
            mt.orchestration(
                dendritic_ca_nM=ca_nM,
                map2_state=effective_map2,
            )

            # ---- 4. Quantum evolution ----
            mt.evolve(dt)

            # ---- 5. Check for OR events ----
            if mt.consciousness_event():
                self._or_count += 1

            # Restore base coherence time (orchestration may have modified it)
            mt._coherence_time_ms = base_coherence_ms

        # ---- 6. Actin dynamics ----
        # Ca2+ promotes actin polymerization through CaMKII -> cofilin pathway
        # Above ~200 nM: increase polymerization (spine growth)
        # Below resting: allow depolymerization
        if ca_nM > 200.0:
            actin_drive = min(1.0, (ca_nM - 200.0) / 2000.0) * 0.01
            self.actin_polymerization = min(1.0, self.actin_polymerization + actin_drive * dt)
        else:
            self.actin_polymerization = max(0.0, self.actin_polymerization - 0.001 * dt)

        # ---- 7. Neurofilament slow dynamics ----
        # Neurofilaments are relatively stable; small activity-dependent
        # modulation over long timescales.
        # No fast dynamics -- neurofilament_density stays constant unless
        # explicitly modulated.

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def or_events(self) -> int:
        """Total Orch-OR consciousness events since last reset."""
        return self._or_count

    def reset_or_count(self) -> None:
        """Reset the OR event counter to zero."""
        self._or_count = 0

    @property
    def consciousness_measure(self) -> float:
        """Integrated quantum information measure (phi-like).

        Combines the coherence levels across all microtubules, weighted
        by the number of superposed tubulins, to produce a single
        scalar measure of "quantum integration" analogous to integrated
        information theory's phi.

        Returns a value in [0, 1]:
        - 0.0 = fully classical (no quantum coherence anywhere).
        - 1.0 = maximal quantum integration (all tubulins in maximally
          coherent superposition across all microtubules).
        """
        if not self.microtubules:
            return 0.0

        total_tubulins = sum(mt.n_tubulins for mt in self.microtubules)
        if total_tubulins == 0:
            return 0.0

        # Weighted coherence: each microtubule contributes proportionally
        # to the fraction of its tubulins in superposition times its
        # coherence level.
        weighted_sum = 0.0
        for mt in self.microtubules:
            if mt.n_tubulins > 0:
                superposition_fraction = mt.n_superposed / mt.n_tubulins
                weighted_sum += superposition_fraction * mt.coherence * mt.n_tubulins

        phi = weighted_sum / total_tubulins

        # Anesthetic suppression reduces the effective phi
        phi *= max(0.0, min(1.0, self.anesthetic_factor))

        return max(0.0, min(1.0, phi))

    @property
    def total_coherence(self) -> float:
        """Average coherence across all microtubules."""
        if not self.microtubules:
            return 0.0
        return sum(mt.coherence for mt in self.microtubules) / len(self.microtubules)

    @property
    def total_superposed(self) -> int:
        """Total number of superposed tubulins across all microtubules."""
        return sum(mt.n_superposed for mt in self.microtubules)

    @property
    def conduction_velocity_factor(self) -> float:
        """Relative conduction velocity based on neurofilament density.

        Higher neurofilament density -> larger axon caliber -> faster
        conduction.  Returns a factor in [0.5, 2.0] relative to baseline.
        """
        return 0.5 + 1.5 * self.neurofilament_density

    def __repr__(self) -> str:
        return (
            f"Cytoskeleton("
            f"MTs={len(self.microtubules)}, "
            f"OR_events={self._or_count}, "
            f"phi={self.consciousness_measure:.4f}, "
            f"actin={self.actin_polymerization:.2f}, "
            f"tau_phos={self.tau_phosphorylation:.2f}, "
            f"anesthetic={self.anesthetic_factor:.2f})"
        )
