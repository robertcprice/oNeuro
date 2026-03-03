"""
Quantum Consciousness Framework
===============================

A comprehensive implementation integrating multiple theories of consciousness
with quantum computing foundations. This module provides:

1. **Integrated Information Theory (IIT)**: Phi calculation via system partitioning
2. **Global Workspace Theory (GWT)**: Attention and awareness broadcasting
3. **Orchestrated Objective Reduction (Orch-OR)**: Penrose-Hameroff quantum consciousness
4. **Emergent Self-Model**: Self-awareness and mirror test capability
5. **Unified Consciousness Metrics**: Composite scoring across theories

References:
- Tononi, G. (2004). An Information Integration Theory of Consciousness
- Baars, B. J. (1988). A Cognitive Theory of Consciousness
- Penrose, R. & Hameroff, S. (2014). Consciousness in the Universe

Author: NQPU Quantum Computing Framework
License: MIT
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Dict, Tuple, Any, Set
from collections import defaultdict
import itertools
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PHYSICAL CONSTANTS (SI Units)
# ============================================================

HBAR = 1.054571817e-34  # Reduced Planck constant (J*s)
K_B = 1.380649e-23  # Boltzmann constant (J/K)
G_NEWTON = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
TUBULIN_MASS_KG = 1.83e-22  # Tubulin dimer mass (~110 kDa)
TUBULIN_DISPLACEMENT_M = 0.25e-9  # Conformational change displacement (m)


# ============================================================
# QUANTUM STATE UTILITIES
# ============================================================

@dataclass
class Complex:
    """Simple complex number representation."""
    re: float = 0.0
    im: float = 0.0

    def __add__(self, other: Complex) -> Complex:
        return Complex(self.re + other.re, self.im + other.im)

    def __mul__(self, other: Complex | float) -> Complex:
        if isinstance(other, Complex):
            return Complex(
                self.re * other.re - self.im * other.im,
                self.re * other.im + self.im * other.re
            )
        return Complex(self.re * other, self.im * other)

    def conjugate(self) -> Complex:
        return Complex(self.re, -self.im)

    def norm_squared(self) -> float:
        return self.re ** 2 + self.im ** 2

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def phase(self) -> float:
        return math.atan2(self.im, self.re)


class QuantumState:
    """
    Simple quantum state vector simulator.

    Represents a multi-qubit quantum state as a complex amplitude vector.
    Provides basic gate operations for consciousness simulation.
    """

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # Initialize to |00...0> state
        self.amplitudes = [Complex() for _ in range(self.dim)]
        self.amplitudes[0] = Complex(1.0, 0.0)

    def probabilities(self) -> List[float]:
        """Return probability distribution over basis states."""
        return [a.norm_squared() for a in self.amplitudes]

    def normalize(self) -> None:
        """Normalize the state vector."""
        norm_sq = sum(a.norm_squared() for a in self.amplitudes)
        if norm_sq > 1e-30:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            for i in range(self.dim):
                self.amplitudes[i] = self.amplitudes[i] * inv_norm

    def hadamard(self, qubit: int) -> None:
        """Apply Hadamard gate to specified qubit."""
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        stride = 1 << qubit
        new_amps = [Complex() for _ in range(self.dim)]

        for idx in range(self.dim):
            if idx & stride:
                continue
            partner = idx | stride
            a0 = self.amplitudes[idx]
            a1 = self.amplitudes[partner]

            # |0> -> (|0> + |1>) / sqrt(2)
            new_amps[idx] = (a0 + a1) * inv_sqrt2
            # |1> -> (|0> - |1>) / sqrt(2)
            new_amps[partner] = (a0 + a1 * Complex(-1, 0)) * inv_sqrt2

        self.amplitudes = new_amps

    def rx(self, qubit: int, angle: float) -> None:
        """Apply Rx rotation gate."""
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)
        stride = 1 << qubit

        for idx in range(self.dim):
            if idx & stride:
                continue
            partner = idx | stride
            a0 = self.amplitudes[idx]
            a1 = self.amplitudes[partner]

            # Rx rotation matrix applied
            self.amplitudes[idx] = a0 * cos_half + Complex(a1.im * sin_half, -a1.re * sin_half)
            self.amplitudes[partner] = Complex(-a0.im * sin_half, a0.re * sin_half) + a1 * cos_half

    def rz(self, qubit: int, angle: float) -> None:
        """Apply Rz rotation gate."""
        phase_0 = Complex(math.cos(-angle / 2), math.sin(-angle / 2))
        phase_1 = Complex(math.cos(angle / 2), math.sin(angle / 2))
        stride = 1 << qubit

        for idx in range(self.dim):
            phase = phase_0 if (idx & stride) == 0 else phase_1
            self.amplitudes[idx] = self.amplitudes[idx] * phase

    def cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate."""
        c_stride = 1 << control
        t_stride = 1 << target
        new_amps = self.amplitudes.copy()

        for idx in range(self.dim):
            if idx & c_stride:
                # Flip target bit
                partner = idx ^ t_stride
                new_amps[idx] = self.amplitudes[partner]
                new_amps[partner] = self.amplitudes[idx]

        self.amplitudes = new_amps

    def entangle_pair(self, q1: int, q2: int, strength: float = 0.5) -> None:
        """Create entanglement between two qubits via controlled rotation."""
        # Apply Hadamard to first qubit
        self.hadamard(q1)
        # Apply controlled-Rx
        angle = strength * math.pi
        for idx in range(self.dim):
            stride1 = 1 << q1
            stride2 = 1 << q2
            if idx & stride1 and not (idx & stride2):
                # Partial rotation for |10> component
                pass
        # Simplified: just apply CNOT for full entanglement
        self.cnot(q1, q2)

    def coherence(self) -> float:
        """Compute quantum coherence (l1-norm of off-diagonal elements)."""
        sum_abs = sum(a.norm() for a in self.amplitudes)
        c = sum_abs ** 2 - 1.0
        max_coherence = self.dim - 1.0
        return min(1.0, max(0.0, c / max_coherence)) if max_coherence > 0 else 0.0

    def reduced_density_matrix(self, qubits: List[int]) -> List[List[Complex]]:
        """
        Compute reduced density matrix for specified qubits by tracing out others.

        Returns a 2^n x 2^n matrix where n = len(qubits).
        """
        n = len(qubits)
        dim_sub = 2 ** n
        rho = [[Complex() for _ in range(dim_sub)] for _ in range(dim_sub)]

        # Map from full state indices to subsystem indices
        def extract_bits(idx: int, bits: List[int]) -> int:
            result = 0
            for i, b in enumerate(bits):
                if idx & (1 << b):
                    result |= (1 << i)
            return result

        # Build mask for traced-out qubits
        sub_mask = sum(1 << b for b in qubits)
        full_mask = (1 << self.num_qubits) - 1
        trace_mask = full_mask ^ sub_mask

        for idx1 in range(self.dim):
            for idx2 in range(self.dim):
                # Must agree on traced-out qubits
                if (idx1 & trace_mask) != (idx2 & trace_mask):
                    continue

                row = extract_bits(idx1, qubits)
                col = extract_bits(idx2, qubits)

                # rho[row][col] += |psi[idx1]><psi[idx2]|
                a1 = self.amplitudes[idx1]
                a2 = self.amplitudes[idx2]
                # |a1><a2| = a1 * conj(a2)
                prod = a1 * a2.conjugate()
                rho[row][col] = rho[row][col] + prod

        return rho

    def measure(self, qubit: int, rng: random.Random = None) -> int:
        """Measure a single qubit and collapse the state."""
        if rng is None:
            rng = random.Random()

        stride = 1 << qubit
        p1 = sum(self.amplitudes[idx].norm_squared() for idx in range(self.dim) if idx & stride)

        outcome = 1 if rng.random() < p1 else 0

        # Collapse state
        for idx in range(self.dim):
            if ((idx & stride) > 0) != (outcome == 1):
                self.amplitudes[idx] = Complex(0, 0)

        self.normalize()
        return outcome

    def copy(self) -> QuantumState:
        """Create a deep copy of the state."""
        new_state = QuantumState(self.num_qubits)
        new_state.amplitudes = [Complex(a.re, a.im) for a in self.amplitudes]
        return new_state


# ============================================================
# TUBULIN STATE FOR ORCH-OR
# ============================================================

class TubulinState(Enum):
    """Conformational states of tubulin dimers in microtubules."""
    ALPHA = auto()  # |0> state
    BETA = auto()   # |1> state
    SUPERPOSITION = auto()  # Quantum superposition


@dataclass
class Tubulin:
    """A single tubulin dimer with quantum properties."""
    index: int
    state: TubulinState = TubulinState.ALPHA
    alpha_prob: float = 1.0
    beta_prob: float = 0.0
    coherence: float = 0.0

    def update_from_probabilities(self, p0: float, p1: float, offdiag: float = 0.0) -> None:
        """Update tubulin state from quantum probabilities."""
        self.alpha_prob = p0
        self.beta_prob = p1

        threshold = 1e-10
        if p0 > 1.0 - threshold and offdiag < threshold:
            self.state = TubulinState.ALPHA
            self.coherence = 0.0
        elif p1 > 1.0 - threshold and offdiag < threshold:
            self.state = TubulinState.BETA
            self.coherence = 0.0
        else:
            self.state = TubulinState.SUPERPOSITION
            self.coherence = min(1.0, offdiag * 2)

    def is_superposition(self) -> bool:
        return self.state == TubulinState.SUPERPOSITION


# ============================================================
# MICROTUBULE
# ============================================================

@dataclass
class Microtubule:
    """
    A microtubule containing tubulin dimers.

    In Orch-OR theory, microtubules are the site of quantum computation
    that gives rise to consciousness through objective reduction events.
    """
    num_tubulins: int
    tubulins: List[Tubulin] = field(default_factory=list)
    quantum_state: Optional[QuantumState] = None
    coupling_matrix: List[List[float]] = field(default_factory=list)
    coherence: float = 1.0
    num_protofilaments: int = 13

    def __post_init__(self):
        if not self.tubulins:
            self.tubulins = [Tubulin(index=i) for i in range(self.num_tubulins)]
        if self.quantum_state is None:
            self.quantum_state = QuantumState(self.num_tubulins)
        if not self.coupling_matrix:
            # Nearest-neighbor coupling along protofilament
            self.coupling_matrix = [[0.0] * self.num_tubulins for _ in range(self.num_tubulins)]
            coupling_strength = 0.01
            for i in range(self.num_tubulins - 1):
                self.coupling_matrix[i][i + 1] = coupling_strength
                self.coupling_matrix[i + 1][i] = coupling_strength

    def sync_tubulins_from_state(self) -> None:
        """Synchronize high-level tubulin states from quantum state."""
        n = self.num_tubulins
        dim = self.quantum_state.dim

        for q in range(n):
            stride = 1 << q
            p0 = 0.0
            p1 = 0.0
            offdiag_sq = 0.0

            for idx in range(dim):
                prob = self.quantum_state.amplitudes[idx].norm_squared()
                if idx & stride:
                    p1 += prob
                else:
                    p0 += prob
                    # Compute off-diagonal element
                    partner = idx | stride
                    if partner < dim:
                        a0 = self.quantum_state.amplitudes[idx]
                        a1 = self.quantum_state.amplitudes[partner]
                        offdiag_sq += (a0.re * a1.re + a0.im * a1.im) ** 2

            offdiag = math.sqrt(offdiag_sq)
            self.tubulins[q].update_from_probabilities(p0, p1, offdiag)

    def compute_coherence(self) -> float:
        """Compute quantum coherence of the microtubule."""
        return self.quantum_state.coherence()


# ============================================================
# REDUCTION EVENT (ORCH-OR)
# ============================================================

@dataclass
class ReductionEvent:
    """
    An objective reduction event - a moment of proto-conscious experience.

    In Orch-OR, when gravitational self-energy exceeds threshold,
    the quantum state collapses, constituting a "moment of experience."
    """
    time_ns: float
    num_tubulins_involved: int
    energy_difference: float  # Gravitational self-energy (J)
    reduction_time: float  # tau = hbar / E_G (s)
    classical_outcome: List[bool]  # True = beta, False = alpha for each tubulin
    integrated_information: float  # Approximate Phi at moment of collapse


# ============================================================
# CONSCIOUSNESS MEASURE
# ============================================================

@dataclass
class ConsciousnessMeasure:
    """Comprehensive consciousness metrics from multiple theoretical frameworks."""
    # Quantum properties
    coherence: float = 0.0
    entanglement: float = 0.0

    # Orch-OR properties
    superposition_mass: float = 0.0  # kg
    gravitational_self_energy: float = 0.0  # J
    time_to_reduction: float = float('inf')  # s

    # Organization
    orchestration_level: float = 0.0
    anesthetic_suppression: float = 0.0

    # IIT-derived
    phi: float = 0.0

    # GWT-derived
    workspace_occupancy: float = 0.0
    attention_focus: float = 0.0

    # Self-model
    self_complexity: float = 0.0
    self_other_distinction: float = 0.0

    def composite_score(self) -> float:
        """
        Compute a composite consciousness score from all metrics.

        All metrics are normalized to [0, 1] and weighted to produce
        a final score in [0, 1]. Individual metrics are clamped before
        weighting to ensure valid probability distributions.
        """
        # Clamp all individual metrics to [0, 1]
        clamped_metrics = {
            'coherence': min(1.0, max(0.0, self.coherence)),
            'entanglement': min(1.0, max(0.0, self.entanglement)),
            'phi': min(1.0, max(0.0, self.phi)),
            'workspace_occupancy': min(1.0, max(0.0, self.workspace_occupancy)),
            'self_complexity': min(1.0, max(0.0, self.self_complexity)),
            'orchestration_level': min(1.0, max(0.0, self.orchestration_level)),
            'attention_focus': min(1.0, max(0.0, self.attention_focus)),
            'self_other_distinction': min(1.0, max(0.0, self.self_other_distinction)),
        }

        weights = {
            'coherence': 0.15,
            'entanglement': 0.10,
            'phi': 0.20,
            'workspace_occupancy': 0.15,
            'self_complexity': 0.15,
            'orchestration_level': 0.10,
            'attention_focus': 0.10,
            'self_other_distinction': 0.05,
        }

        raw_score = sum(clamped_metrics[k] * w for k, w in weights.items())

        # Apply conscious factor: anesthetic suppresses consciousness
        # High anesthetic_suppression (near 1.0) = unconscious = LOW score
        # Low anesthetic_suppression (near 0.0) = conscious = HIGH score
        conscious_factor = 1.0 - self.anesthetic_suppression

        return min(1.0, max(0.0, raw_score * conscious_factor))


# ============================================================
# ORCH-OR SIMULATOR
# ============================================================

class OrchORSimulator:
    """
    Orchestrated Objective Reduction (Orch-OR) quantum consciousness simulator.

    Implements the Penrose-Hameroff theory where consciousness arises from
    quantum computations in brain microtubules, with collapse events
    triggered by gravitational self-energy reaching the Diosi-Penrose threshold.
    """

    def __init__(
        self,
        num_tubulins: int = 8,
        coherence_time_ns: float = 25.0,
        temperature_kelvin: float = 310.0,
        gravitational_threshold: float = 1e-25,
        coupling_strength: float = 0.01,
        anesthetic_concentration: float = 0.0,
        seed: int = 42
    ):
        self.num_tubulins = num_tubulins
        self.coherence_time_ns = coherence_time_ns
        self.temperature_kelvin = temperature_kelvin
        self.gravitational_threshold = gravitational_threshold
        self.coupling_strength = coupling_strength
        self.anesthetic_concentration = anesthetic_concentration
        self.seed = seed

        self.microtubule = Microtubule(num_tubulins=num_tubulins)
        self.current_time_ns = 0.0
        self.reduction_history: List[ReductionEvent] = []
        self.rng = random.Random(seed)

        # Compute effective coherence time
        activation_energy_J = 0.04 * 1.602176634e-19
        temp_factor = math.exp(-activation_energy_J / (K_B * temperature_kelvin))
        anesthetic_factor = 1.0 - anesthetic_concentration
        self.effective_coherence_ns = max(1e-6, coherence_time_ns * temp_factor * anesthetic_factor)

    def initialize_superposition(self) -> None:
        """Put all tubulins into equal superposition of alpha and beta states."""
        for q in range(self.num_tubulins):
            self.microtubule.quantum_state.hadamard(q)
        self.microtubule.sync_tubulins_from_state()
        self.microtubule.coherence = self.microtubule.compute_coherence()

    def evolve(self, time_steps: int) -> List[Dict[str, Any]]:
        """
        Evolve the simulation for given number of time steps.

        Each step is 1 nanosecond. Returns snapshots of the system state.
        """
        dt_ns = 1.0
        snapshots = []

        for _ in range(time_steps):
            self.current_time_ns += dt_ns

            # Hamiltonian evolution
            self._apply_hamiltonian(dt_ns)

            # Decoherence
            self._apply_decoherence(dt_ns)

            # Update coherence
            self.microtubule.coherence = self.microtubule.compute_coherence()
            self.microtubule.sync_tubulins_from_state()

            # Check for objective reduction
            reduction_event = self._check_objective_reduction()

            if reduction_event:
                self.reduction_history.append(reduction_event)
                self.microtubule.coherence = self.microtubule.compute_coherence()

            # Record snapshot
            cm = self.consciousness_measure()
            snapshots.append({
                'time_ns': self.current_time_ns,
                'coherence': self.microtubule.coherence,
                'consciousness': cm,
                'reduction_event': reduction_event
            })

        return snapshots

    def _apply_hamiltonian(self, dt: float) -> None:
        """Apply tubulin interaction Hamiltonian."""
        time_scale = 0.001  # Scaling for numerical stability

        # ZZ interactions
        for i in range(self.num_tubulins):
            for j in range(i + 1, self.num_tubulins):
                coupling = self.microtubule.coupling_matrix[i][j]
                if abs(coupling) < 1e-15:
                    continue
                angle = coupling * dt * time_scale
                self._apply_zz_interaction(i, j, angle)

        # Transverse field (Rx rotations)
        gamma = self.coupling_strength * 0.5
        rx_angle = 2.0 * gamma * dt * time_scale
        for q in range(self.num_tubulins):
            self.microtubule.quantum_state.rx(q, rx_angle)

        # Longitudinal field (Rz rotations)
        h_bias = self.coupling_strength * 0.1
        rz_angle = 2.0 * h_bias * dt * time_scale
        for q in range(self.num_tubulins):
            self.microtubule.quantum_state.rz(q, rz_angle)

    def _apply_zz_interaction(self, qubit_i: int, qubit_j: int, angle: float) -> None:
        """Apply ZZ interaction: exp(-i * angle * Z_i Z_j)."""
        dim = self.microtubule.quantum_state.dim
        amps = self.microtubule.quantum_state.amplitudes

        phase_same = Complex(math.cos(-angle), math.sin(-angle))
        phase_diff = Complex(math.cos(angle), math.sin(angle))

        for idx in range(dim):
            bit_i = (idx >> qubit_i) & 1
            bit_j = (idx >> qubit_j) & 1
            phase = phase_same if bit_i == bit_j else phase_diff
            amps[idx] = amps[idx] * phase

    def _apply_decoherence(self, dt: float) -> None:
        """Apply environmental decoherence (amplitude damping)."""
        if self.effective_coherence_ns <= 0:
            return

        p = 1.0 - math.exp(-dt / self.effective_coherence_ns)
        damping = math.sqrt(1.0 - p)
        dim = self.microtubule.quantum_state.dim
        amps = self.microtubule.quantum_state.amplitudes

        for q in range(self.num_tubulins):
            stride = 1 << q
            for idx in range(dim):
                if idx & stride:
                    amps[idx] = amps[idx] * damping

        self.microtubule.quantum_state.normalize()

    def _check_objective_reduction(self) -> Optional[ReductionEvent]:
        """Check if Penrose objective reduction threshold is reached."""
        num_superposed = sum(1 for t in self.microtubule.tubulins if t.is_superposition())

        if num_superposed == 0:
            return None

        # Gravitational self-energy: E_G = N * G * m^2 / delta_x
        e_g = num_superposed * G_NEWTON * TUBULIN_MASS_KG ** 2 / TUBULIN_DISPLACEMENT_M

        if e_g < self.gravitational_threshold:
            return None

        # Reduction occurs
        reduction_time = HBAR / e_g
        phi = self._approximate_phi()

        # Probabilistic collapse
        probs = self.microtubule.quantum_state.probabilities()
        r = self.rng.random()
        cumsum = 0.0
        collapsed_index = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                collapsed_index = i
                break

        # Extract classical outcome
        classical_outcome = [((collapsed_index >> q) & 1) == 1 for q in range(self.num_tubulins)]

        # Collapse state
        dim = self.microtubule.quantum_state.dim
        for idx in range(dim):
            if idx == collapsed_index:
                self.microtubule.quantum_state.amplitudes[idx] = Complex(1.0, 0.0)
            else:
                self.microtubule.quantum_state.amplitudes[idx] = Complex(0.0, 0.0)

        self.microtubule.sync_tubulins_from_state()

        return ReductionEvent(
            time_ns=self.current_time_ns,
            num_tubulins_involved=num_superposed,
            energy_difference=e_g,
            reduction_time=reduction_time,
            classical_outcome=classical_outcome,
            integrated_information=phi
        )

    def _approximate_phi(self) -> float:
        """Approximate integrated information (Phi) via bipartite entropy."""
        if self.num_tubulins < 2:
            return 0.0

        # Bipartition: first half vs second half
        n_a = self.num_tubulins // 2
        subsystem_a = list(range(n_a))

        rho_a = self.microtubule.quantum_state.reduced_density_matrix(subsystem_a)

        # Linear entropy: 1 - Tr(rho^2)
        dim_a = 2 ** n_a
        tr_rho_sq = 0.0
        for i in range(dim_a):
            for j in range(dim_a):
                val = rho_a[i][j]
                val_dag = rho_a[j][i].conjugate()
                tr_rho_sq += val.re * val_dag.re - val.im * val_dag.im

        return max(0.0, min(1.0, 1.0 - tr_rho_sq))

    def apply_anesthetic(self, concentration: float) -> None:
        """Apply anesthetic effect (reduces coherence time)."""
        c = max(0.0, min(1.0, concentration))
        self.anesthetic_concentration = c

        activation_energy_J = 0.04 * 1.602176634e-19
        temp_factor = math.exp(-activation_energy_J / (K_B * self.temperature_kelvin))
        anesthetic_factor = 1.0 - c
        self.effective_coherence_ns = max(1e-6, self.coherence_time_ns * temp_factor * anesthetic_factor)

    def consciousness_measure(self) -> ConsciousnessMeasure:
        """Compute comprehensive consciousness metrics."""
        coherence = self.microtubule.coherence

        num_superposed = sum(1 for t in self.microtubule.tubulins if t.is_superposition())
        superposition_mass = num_superposed * TUBULIN_MASS_KG

        gravitational_self_energy = 0.0
        if num_superposed > 0:
            gravitational_self_energy = num_superposed * G_NEWTON * TUBULIN_MASS_KG ** 2 / TUBULIN_DISPLACEMENT_M

        time_to_reduction = float('inf')
        if gravitational_self_energy > 0:
            time_to_reduction = HBAR / gravitational_self_energy

        # Orchestration: deviation from uniform distribution
        probs = self.microtubule.quantum_state.probabilities()
        dim = self.microtubule.quantum_state.dim
        uniform = 1.0 / dim
        kl = sum(p * math.log(p / uniform) for p in probs if p > 1e-30)
        max_kl = math.log(dim)
        orchestration = min(1.0, max(0.0, kl / max_kl)) if max_kl > 0 else 0.0

        return ConsciousnessMeasure(
            coherence=coherence,
            entanglement=self._average_entanglement(),
            superposition_mass=superposition_mass,
            gravitational_self_energy=gravitational_self_energy,
            time_to_reduction=time_to_reduction,
            orchestration_level=orchestration,
            anesthetic_suppression=self.anesthetic_concentration,
            phi=self._approximate_phi()
        )

    def _average_entanglement(self) -> float:
        """Compute average pairwise entanglement."""
        if self.num_tubulins < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(self.num_tubulins):
            for j in range(i + 1, self.num_tubulins):
                total += self._entanglement_between(i, j)
                count += 1

        return total / count if count > 0 else 0.0

    def _entanglement_between(self, i: int, j: int) -> float:
        """Compute entanglement between two tubulins (linear entropy proxy)."""
        rho = self.microtubule.quantum_state.reduced_density_matrix([i, j])

        # Linear entropy of reduced state
        tr_rho_sq = 0.0
        for a in range(4):
            for b in range(4):
                val = rho[a][b]
                val_dag = rho[b][a].conjugate()
                tr_rho_sq += val.re * val_dag.re - val.im * val_dag.im

        s_linear = max(0.0, 1.0 - tr_rho_sq)
        return min(1.0, s_linear * 4.0 / 3.0)


# ============================================================
# INTEGRATED INFORMATION THEORY (IIT)
# ============================================================

@dataclass
class IITSystem:
    """
    A system for Integrated Information Theory analysis.

    IIT proposes that consciousness is identical to integrated information (Phi),
    which measures how much information is generated by a system above and beyond
    the information generated by its parts.
    """

    # System state as probability distribution over states
    num_units: int
    states_per_unit: int = 2
    transition_matrix: Optional[List[List[float]]] = None
    current_state: Optional[List[int]] = None

    def __post_init__(self):
        self.total_states = self.states_per_unit ** self.num_units
        if self.current_state is None:
            self.current_state = [0] * self.num_units
        if self.transition_matrix is None:
            # Default: weakly correlated transitions
            self._init_default_transition_matrix()

    def _init_default_transition_matrix(self) -> None:
        """Initialize a default transition matrix with some structure."""
        self.transition_matrix = []
        for i in range(self.total_states):
            row = [0.0] * self.total_states
            # Favor staying in same state
            row[i] = 0.5
            # Distribute remaining probability to neighbors
            remaining = 0.5
            neighbors = self._get_neighbor_states(i)
            for n in neighbors:
                row[n] = remaining / len(neighbors) if neighbors else remaining
            self.transition_matrix.append(row)

    def _get_neighbor_states(self, state_idx: int) -> List[int]:
        """Get indices of states that differ by one unit."""
        neighbors = []
        for u in range(self.num_units):
            neighbors.append(state_idx ^ (1 << u))
        return neighbors

    def state_index(self, state: List[int]) -> int:
        """Convert state list to index."""
        idx = 0
        for i, s in enumerate(state):
            idx += s * (self.states_per_unit ** i)
        return idx

    def index_to_state(self, idx: int) -> List[int]:
        """Convert index to state list."""
        state = []
        for i in range(self.num_units):
            state.append(idx % self.states_per_unit)
            idx //= self.states_per_unit
        return state

    def entropy(self, distribution: List[float]) -> float:
        """Compute Shannon entropy of a distribution."""
        return -sum(p * math.log2(p) for p in distribution if p > 1e-30)

    def mutual_information(self, dist_xy: List[List[float]], dist_x: List[float], dist_y: List[float]) -> float:
        """Compute mutual information I(X;Y)."""
        mi = 0.0
        for i, px in enumerate(dist_x):
            for j, py in enumerate(dist_y):
                if dist_xy[i][j] > 1e-30 and px > 1e-30 and py > 1e-30:
                    mi += dist_xy[i][j] * math.log2(dist_xy[i][j] / (px * py))
        return mi

    def effective_information(self, partition: Tuple[Set[int], Set[int]]) -> float:
        """
        Compute effective information (EI) for a bipartition.

        EI measures how much the system's output depends on its input
        when considering only one part of the partition.
        """
        part_a, part_b = partition

        if not part_a or not part_b:
            return 0.0

        # Simplified: use correlation between partition parts
        # Full IIT 4.0 requires cause-effect repertoires
        # This is an approximation for demonstration

        # Create marginal distributions
        n_a = len(part_a)
        n_b = len(part_b)
        dim_a = self.states_per_unit ** n_a
        dim_b = self.states_per_unit ** n_b

        # Compute joint distribution over A x B at next time step
        # given uniform distribution over current states
        joint = [[0.0] * dim_b for _ in range(dim_a)]
        uniform_prior = 1.0 / self.total_states

        for from_idx in range(self.total_states):
            for to_idx in range(self.total_states):
                prob = uniform_prior * self.transition_matrix[from_idx][to_idx]
                a_idx = self._extract_partition_index(to_idx, part_a)
                b_idx = self._extract_partition_index(to_idx, part_b)
                joint[a_idx][b_idx] += prob

        # Marginalize
        marg_a = [sum(joint[i]) for i in range(dim_a)]
        marg_b = [sum(joint[i][j] for i in range(dim_a)) for j in range(dim_b)]

        # MI as proxy for EI
        return self.mutual_information(joint, marg_a, marg_b)

    def _extract_partition_index(self, state_idx: int, partition: Set[int]) -> int:
        """Extract the index for units in a partition."""
        sorted_units = sorted(partition)
        result = 0
        state = self.index_to_state(state_idx)
        for i, u in enumerate(sorted_units):
            result += state[u] * (self.states_per_unit ** i)
        return result

    def phi(self) -> float:
        """
        Calculate integrated information Phi.

        Phi is the minimum effective information across all possible
        bipartitions of the system.
        """
        if self.num_units < 2:
            return 0.0

        min_ei = float('inf')

        # Check all bipartitions
        units = set(range(self.num_units))
        for r in range(1, self.num_units // 2 + 1):
            for combo in itertools.combinations(range(self.num_units), r):
                part_a = set(combo)
                part_b = units - part_a
                if not part_b:
                    continue
                ei = self.effective_information((part_a, part_b))
                if ei < min_ei:
                    min_ei = ei

        return min_ei if min_ei != float('inf') else 0.0

    def all_partitions(self) -> List[Tuple[Set[int], Set[int]]]:
        """Generate all bipartitions of the system."""
        units = set(range(self.num_units))
        partitions = []
        for r in range(1, self.num_units // 2 + 1):
            for combo in itertools.combinations(range(self.num_units), r):
                part_a = set(combo)
                part_b = units - part_a
                if part_b:
                    partitions.append((part_a, part_b))
        return partitions


# ============================================================
# GLOBAL WORKSPACE THEORY (GWT)
# ============================================================

class ModuleType(Enum):
    """Types of specialized modules in the global workspace."""
    VISUAL = auto()
    AUDITORY = auto()
    SOMATOSENSORY = auto()
    MEMORY = auto()
    LANGUAGE = auto()
    EXECUTIVE = auto()
    EMOTIONAL = auto()
    SELF_MODEL = auto()


@dataclass
class SpecializedModule:
    """
    A specialized cognitive module that competes for workspace access.

    In GWT, consciousness arises when information is broadcast globally
    from specialized modules through a "global workspace."
    """
    module_type: ModuleType
    activation: float = 0.0
    content: Optional[Any] = None
    attention_weight: float = 1.0
    threshold: float = 0.5

    def process_input(self, input_data: Any, strength: float = 1.0) -> None:
        """Process incoming data and update activation."""
        self.content = input_data
        self.activation = min(1.0, strength * self.attention_weight)

    def can_access_workspace(self) -> bool:
        """Check if module activation exceeds threshold for workspace access."""
        return self.activation >= self.threshold

    def decay(self, rate: float = 0.1) -> None:
        """Decay activation over time."""
        self.activation *= (1.0 - rate)


@dataclass
class GlobalWorkspace:
    """
    The global workspace that broadcasts information across modules.

    In Baars' Global Workspace Theory, consciousness corresponds to
    information that is globally broadcast and available to multiple
    cognitive systems.
    """
    capacity: int = 4  # Number of items that can be simultaneously conscious
    modules: Dict[ModuleType, SpecializedModule] = field(default_factory=dict)
    broadcast_content: List[Any] = field(default_factory=list)
    broadcast_history: List[Dict[str, Any]] = field(default_factory=list)
    current_focus: Optional[ModuleType] = None
    attention_resources: float = 1.0

    def __post_init__(self):
        if not self.modules:
            # Initialize default modules
            for mt in ModuleType:
                self.modules[mt] = SpecializedModule(module_type=mt)

    def process_cycle(self) -> Dict[str, Any]:
        """
        Execute one processing cycle: competition, broadcast, and feedback.

        Returns a summary of what entered consciousness this cycle.
        """
        # Decay all modules
        for module in self.modules.values():
            module.decay()

        # Competition: select modules above threshold
        candidates = [
            (mt, m) for mt, m in self.modules.items()
            if m.can_access_workspace()
        ]

        # Sort by activation (winner-take-most)
        candidates.sort(key=lambda x: x[1].activation, reverse=True)

        # Select top candidates up to capacity
        winners = candidates[:self.capacity]

        # Broadcast content
        self.broadcast_content = [m.content for _, m in winners]
        self.current_focus = winners[0][0] if winners else None

        # Record history
        cycle_summary = {
            'winners': [(mt.name, m.activation) for mt, m in winners],
            'broadcast_content': self.broadcast_content.copy(),
            'focus': self.current_focus.name if self.current_focus else None
        }
        self.broadcast_history.append(cycle_summary)

        # Feedback: boost connected modules
        for mt, module in winners:
            self._broadcast_feedback(mt, module)

        return cycle_summary

    def _broadcast_feedback(self, source_type: ModuleType, source: SpecializedModule) -> None:
        """Broadcast from winning module to connected modules."""
        # Define connectivity patterns
        connections = {
            ModuleType.VISUAL: [ModuleType.MEMORY, ModuleType.SELF_MODEL],
            ModuleType.AUDITORY: [ModuleType.LANGUAGE, ModuleType.MEMORY],
            ModuleType.SOMATOSENSORY: [ModuleType.EMOTIONAL, ModuleType.SELF_MODEL],
            ModuleType.MEMORY: [ModuleType.EXECUTIVE, ModuleType.SELF_MODEL],
            ModuleType.LANGUAGE: [ModuleType.EXECUTIVE, ModuleType.MEMORY],
            ModuleType.EXECUTIVE: [ModuleType.SELF_MODEL],
            ModuleType.EMOTIONAL: [ModuleType.SELF_MODEL, ModuleType.MEMORY],
            ModuleType.SELF_MODEL: [ModuleType.EXECUTIVE, ModuleType.MEMORY],
        }

        boost_factor = 0.3 * source.activation
        for target in connections.get(source_type, []):
            if target in self.modules:
                self.modules[target].activation += boost_factor

    def inject_input(self, module_type: ModuleType, content: Any, strength: float = 1.0) -> None:
        """Inject input into a specific module."""
        if module_type in self.modules:
            self.modules[module_type].process_input(content, strength)

    def workspace_occupancy(self) -> float:
        """Compute fraction of workspace capacity currently in use."""
        return len(self.broadcast_content) / self.capacity if self.capacity > 0 else 0.0

    def attention_focus_strength(self) -> float:
        """Compute strength of attentional focus."""
        if self.current_focus and self.current_focus in self.modules:
            return self.modules[self.current_focus].activation
        return 0.0

    def global_availability(self) -> float:
        """Compute how globally available information is (broadcast score)."""
        if not self.broadcast_content:
            return 0.0

        # Count how many modules received activation this cycle
        active_count = sum(1 for m in self.modules.values() if m.activation > 0.1)
        return active_count / len(self.modules) if self.modules else 0.0


# ============================================================
# EMERGENT SELF-MODEL
# ============================================================

@dataclass
class SelfModel:
    """
    A system that develops a model of itself.

    Self-awareness emerges when the system can distinguish self from other,
    recognize itself in mirrors, and maintain a coherent self-narrative.
    """
    # Self-representation
    self_state: Dict[str, float] = field(default_factory=dict)
    self_history: List[Dict[str, float]] = field(default_factory=list)

    # Other representations
    other_models: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Mirror test capabilities
    self_recognition_score: float = 0.0
    mirror_exposures: int = 0

    # Narrative construction
    narrative_coherence: float = 0.0
    narrative_elements: List[str] = field(default_factory=list)

    # Metacognition
    confidence_in_self_model: float = 0.0
    introspection_depth: int = 0

    def __post_init__(self):
        if not self.self_state:
            self.self_state = {
                'agency': 0.5,  # Sense of being cause of actions
                'continuity': 0.5,  # Sense of persistence over time
                'distinctness': 0.5,  # Sense of being separate from environment
                'ownership': 0.5,  # Sense of owning body/thoughts
                'perspective': 0.5,  # Sense of first-person viewpoint
            }

    def update_self_state(self, updates: Dict[str, float]) -> None:
        """Update self-state representation."""
        for key, value in updates.items():
            if key in self.self_state:
                # Blend with existing state (memory + update)
                self.self_state[key] = 0.7 * self.self_state[key] + 0.3 * value

        # Record history
        self.self_history.append(self.self_state.copy())
        if len(self.self_history) > 100:
            self.self_history.pop(0)

    def observe_other(self, other_id: str, observations: Dict[str, float]) -> None:
        """Update model of another agent."""
        if other_id not in self.other_models:
            self.other_models[other_id] = {}

        for key, value in observations.items():
            self.other_models[other_id][key] = value

    def self_other_distinction(self) -> float:
        """
        Compute clarity of self vs other distinction.

        Higher values indicate clearer differentiation.
        """
        if not self.other_models:
            return 0.5

        # Compare self-state to other models
        total_distance = 0.0
        comparisons = 0

        for other_state in self.other_models.values():
            for key in self.self_state:
                if key in other_state:
                    diff = abs(self.self_state[key] - other_state[key])
                    total_distance += diff
                    comparisons += 1

        if comparisons == 0:
            return 0.5

        # Normalize: average difference, scaled to [0, 1]
        avg_diff = total_distance / comparisons
        return min(1.0, avg_diff * 2)  # Scale up small differences

    def mirror_test(self, reflection_data: Dict[str, float], is_self: bool = True) -> float:
        """
        Perform mirror self-recognition test.

        Returns recognition score (higher = better self-recognition).
        """
        self.mirror_exposures += 1

        # Compare reflection to self-model
        similarity = 0.0
        comparisons = 0

        for key, value in reflection_data.items():
            if key in self.self_state:
                # Similarity is inverse of difference
                diff = abs(self.self_state[key] - value)
                similarity += 1.0 - diff
                comparisons += 1

        if comparisons > 0:
            similarity /= comparisons

        if is_self:
            # Recognizing self: update self-recognition score
            self.self_recognition_score = (
                0.8 * self.self_recognition_score + 0.2 * similarity
            )
            # Strengthen self-distinctness
            self.update_self_state({'distinctness': min(1.0, self.self_state.get('distinctness', 0.5) + 0.1)})

        return similarity

    def add_narrative_element(self, element: str) -> None:
        """Add element to self-narrative."""
        self.narrative_elements.append(element)
        if len(self.narrative_elements) > 50:
            self.narrative_elements.pop(0)

        # Update coherence based on narrative consistency
        self._compute_narrative_coherence()

    def _compute_narrative_coherence(self) -> None:
        """Compute coherence of self-narrative."""
        if len(self.narrative_elements) < 3:
            self.narrative_coherence = 0.5
            return

        # Simplified coherence: based on temporal consistency of self-state
        if len(self.self_history) >= 2:
            recent = self.self_history[-1]
            previous = self.self_history[-2]

            changes = sum(abs(recent.get(k, 0) - previous.get(k, 0))
                         for k in self.self_state)
            avg_change = changes / len(self.self_state)

            # Low change = high coherence
            self.narrative_coherence = max(0.0, 1.0 - avg_change * 5)

    def introspect(self, depth: int = 1) -> Dict[str, Any]:
        """
        Perform introspection on self-model.

        Higher depth = more levels of "thinking about thinking."
        """
        self.introspection_depth = max(self.introspection_depth, depth)

        result = {
            'self_state': self.self_state.copy(),
            'confidence': self.confidence_in_self_model,
            'recognition_ability': self.self_recognition_score,
            'narrative_coherence': self.narrative_coherence,
            'other_count': len(self.other_models),
        }

        if depth > 1:
            result['meta'] = {
                'introspection_depth': self.introspection_depth,
                'certainty_about_self': self._compute_certainty(),
            }

        return result

    def _compute_certainty(self) -> float:
        """Compute certainty about self-model."""
        # Based on consistency of self-history
        if len(self.self_history) < 2:
            return 0.5

        # Compute variance in self-state over time
        variances = {}
        for key in self.self_state:
            values = [h.get(key, 0) for h in self.self_history]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            variances[key] = var

        # Low variance = high certainty
        avg_var = sum(variances.values()) / len(variances) if variances else 0
        self.confidence_in_self_model = max(0.0, min(1.0, 1.0 - avg_var * 10))

        return self.confidence_in_self_model

    def complexity(self) -> float:
        """Compute complexity of self-model."""
        # Factors: self-state dimensionality, narrative length, other models
        self_dim = len(self.self_state)
        narrative_complexity = min(1.0, len(self.narrative_elements) / 20)
        other_complexity = min(1.0, len(self.other_models) / 5)

        # Weighted combination
        return 0.3 * (self_dim / 5) + 0.4 * narrative_complexity + 0.3 * other_complexity


# ============================================================
# UNIFIED CONSCIOUSNESS SYSTEM
# ============================================================

class ConsciousnessSystem:
    """
    Unified consciousness system integrating IIT, GWT, Orch-OR, and self-model.

    This provides a comprehensive simulation of consciousness from multiple
    theoretical perspectives.
    """

    def __init__(
        self,
        num_units: int = 8,
        num_tubulins: int = 8,
        workspace_capacity: int = 4,
        seed: int = 42
    ):
        self.rng = random.Random(seed)

        # Initialize components
        self.iit_system = IITSystem(num_units=num_units)
        self.orch_or = OrchORSimulator(num_tubulins=num_tubulins, seed=seed)
        self.workspace = GlobalWorkspace(capacity=workspace_capacity)
        self.self_model = SelfModel()

        # Tracking
        self.time_step = 0
        self.consciousness_history: List[ConsciousnessMeasure] = []

        # Initialize quantum superposition
        self.orch_or.initialize_superposition()

    def step(self) -> ConsciousnessMeasure:
        """Execute one simulation step and return consciousness metrics."""
        self.time_step += 1

        # 1. Evolve quantum system (Orch-OR)
        orch_snaps = self.orch_or.evolve(1)
        orch_coherence = orch_snaps[-1]['coherence'] if orch_snaps else 0.0

        # 2. Update IIT system based on quantum state
        self._sync_iit_to_quantum()

        # 3. Process workspace cycle
        workspace_summary = self.workspace.process_cycle()

        # 4. Update self-model
        self._update_self_model()

        # 5. Compute unified metrics
        metrics = self._compute_unified_metrics(orch_coherence, workspace_summary)

        self.consciousness_history.append(metrics)
        return metrics

    def _sync_iit_to_quantum(self) -> None:
        """Synchronize IIT system state with quantum state."""
        probs = self.orch_or.microtubule.quantum_state.probabilities()

        # Find most probable state
        max_prob_idx = max(range(len(probs)), key=lambda i: probs[i])
        self.iit_system.current_state = self.iit_system.index_to_state(max_prob_idx)

        # Update transition probabilities based on coherence
        coherence = self.orch_or.microtubule.coherence
        for i in range(self.iit_system.total_states):
            for j in range(self.iit_system.total_states):
                # Higher coherence = more structured transitions
                if i == j:
                    self.iit_system.transition_matrix[i][j] = 0.3 + 0.4 * coherence
                else:
                    neighbors = self.iit_system._get_neighbor_states(i)
                    if j in neighbors:
                        remaining = 1.0 - self.iit_system.transition_matrix[i][i]
                        self.iit_system.transition_matrix[i][j] = remaining / len(neighbors)

    def _update_self_model(self) -> None:
        """Update self-model based on current system state."""
        coherence = self.orch_or.microtubule.coherence
        occupancy = self.workspace.workspace_occupancy()
        phi = self.iit_system.phi()

        # Update self-state
        self.self_model.update_self_state({
            'agency': min(1.0, coherence * occupancy),
            'continuity': min(1.0, 0.5 + 0.5 * phi),
            'distinctness': self.self_model.self_other_distinction(),
            'ownership': min(1.0, occupancy * self.self_model.self_recognition_score),
            'perspective': min(1.0, coherence * 0.8 + 0.2),
        })

        # Add narrative element periodically
        if self.time_step % 10 == 0:
            if coherence > 0.5:
                self.self_model.add_narrative_element(
                    f"t={self.time_step}: coherent experience (phi={phi:.3f})"
                )
            else:
                self.self_model.add_narrative_element(
                    f"t={self.time_step}: fragmented state"
                )

    def _compute_unified_metrics(
        self,
        orch_coherence: float,
        workspace_summary: Dict[str, Any]
    ) -> ConsciousnessMeasure:
        """Compute unified consciousness metrics."""

        # Get base Orch-OR metrics
        orch_cm = self.orch_or.consciousness_measure()

        # IIT metrics
        phi = self.iit_system.phi()

        # GWT metrics
        workspace_occupancy = self.workspace.workspace_occupancy()
        attention_focus = self.workspace.attention_focus_strength()

        # Self-model metrics
        self_complexity = self.self_model.complexity()
        self_other = self.self_model.self_other_distinction()

        return ConsciousnessMeasure(
            coherence=orch_cm.coherence,
            entanglement=orch_cm.entanglement,
            superposition_mass=orch_cm.superposition_mass,
            gravitational_self_energy=orch_cm.gravitational_self_energy,
            time_to_reduction=orch_cm.time_to_reduction,
            orchestration_level=orch_cm.orchestration_level,
            anesthetic_suppression=orch_cm.anesthetic_suppression,
            phi=phi,
            workspace_occupancy=workspace_occupancy,
            attention_focus=attention_focus,
            self_complexity=self_complexity,
            self_other_distinction=self_other
        )

    def inject_stimulus(
        self,
        module_type: ModuleType,
        content: Any,
        strength: float = 1.0
    ) -> None:
        """Inject a stimulus into the consciousness system."""
        self.workspace.inject_input(module_type, content, strength)

        # Also affect quantum state
        for q in range(self.orch_or.num_tubulins):
            self.orch_or.microtubule.quantum_state.rx(q, strength * 0.1)

    def apply_anesthetic(self, concentration: float) -> None:
        """Apply anesthetic to suppress consciousness."""
        self.orch_or.apply_anesthetic(concentration)

    def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consciousness report."""
        latest = self.consciousness_history[-1] if self.consciousness_history else ConsciousnessMeasure()

        return {
            'time_step': self.time_step,
            'composite_score': latest.composite_score(),
            'metrics': {
                'phi_iit': latest.phi,
                'coherence_orch_or': latest.coherence,
                'workspace_occupancy_gwt': latest.workspace_occupancy,
                'attention_focus': latest.attention_focus,
                'self_complexity': latest.self_complexity,
                'self_other_distinction': latest.self_other_distinction,
                'entanglement': latest.entanglement,
                'orchestration': latest.orchestration_level,
            },
            'reductions': len(self.orch_or.reduction_history),
            'self_model': self.self_model.introspect(depth=2),
            'workspace_focus': self.workspace.current_focus.name if self.workspace.current_focus else None,
        }


# ============================================================
# DEMO AND VISUALIZATION
# ============================================================

def run_consciousness_demo(steps: int = 100, seed: int = 42) -> Dict[str, Any]:
    """
    Run a demonstration of the consciousness system.

    Simulates the evolution of consciousness metrics over time,
    including self-awareness emergence and comparison of conscious
    vs unconscious states.
    """
    print("=" * 60)
    print("QUANTUM CONSCIOUSNESS SIMULATION")
    print("=" * 60)

    # Initialize system
    system = ConsciousnessSystem(num_units=6, num_tubulins=6, seed=seed)

    # Run baseline evolution
    print("\n[Phase 1] Baseline consciousness evolution...")
    baseline_metrics = []
    for i in range(steps):
        metrics = system.step()
        baseline_metrics.append(metrics)

        if (i + 1) % 25 == 0:
            print(f"  Step {i+1}: Composite={metrics.composite_score():.3f}, "
                  f"Phi={metrics.phi:.3f}, Coherence={metrics.coherence:.3f}")

    # Inject strong stimulus
    print("\n[Phase 2] Injecting attentional stimulus...")
    system.inject_stimulus(ModuleType.VISUAL, "bright_light", strength=0.9)
    system.inject_stimulus(ModuleType.SELF_MODEL, "self_reflection", strength=0.8)

    stimulus_metrics = []
    for i in range(30):
        metrics = system.step()
        stimulus_metrics.append(metrics)

    print(f"  Avg with stimulus: Composite={sum(m.composite_score() for m in stimulus_metrics)/len(stimulus_metrics):.3f}")

    # Apply anesthetic (unconscious state)
    print("\n[Phase 3] Applying anesthetic (unconscious state)...")
    system.apply_anesthetic(0.9)

    unconscious_metrics = []
    for i in range(30):
        metrics = system.step()
        unconscious_metrics.append(metrics)

    print(f"  Avg under anesthetic: Composite={sum(m.composite_score() for m in unconscious_metrics)/len(unconscious_metrics):.3f}")

    # Recovery
    print("\n[Phase 4] Recovery from anesthetic...")
    system.apply_anesthetic(0.0)

    recovery_metrics = []
    for i in range(30):
        metrics = system.step()
        recovery_metrics.append(metrics)

    print(f"  Avg after recovery: Composite={sum(m.composite_score() for m in recovery_metrics)/len(recovery_metrics):.3f}")

    # Mirror test
    print("\n[Phase 5] Mirror self-recognition test...")
    for _ in range(5):
        # Simulate seeing self in mirror
        reflection = {
            'agency': system.self_model.self_state.get('agency', 0.5),
            'continuity': system.self_model.self_state.get('continuity', 0.5),
            'distinctness': system.self_model.self_state.get('distinctness', 0.5),
        }
        recognition = system.self_model.mirror_test(reflection, is_self=True)
        system.step()

    print(f"  Self-recognition score: {system.self_model.self_recognition_score:.3f}")

    # Generate summary
    def avg_score(metrics_list):
        if not metrics_list:
            return 0.0
        return sum(m.composite_score() for m in metrics_list) / len(metrics_list)

    baseline_avg = avg_score(baseline_metrics)
    unconscious_avg = avg_score(unconscious_metrics)

    summary = {
        'baseline_avg_score': baseline_avg,
        'stimulus_avg_score': avg_score(stimulus_metrics),
        'unconscious_avg_score': unconscious_avg,
        'recovery_avg_score': avg_score(recovery_metrics),
        'total_reductions': len(system.orch_or.reduction_history),
        'final_self_recognition': system.self_model.self_recognition_score,
        'final_self_complexity': system.self_model.complexity(),
        'conscious_vs_unconscious_ratio': (
            baseline_avg / unconscious_avg
            if unconscious_avg > 0.001 else float('inf')
        )
    }

    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Baseline consciousness:   {summary['baseline_avg_score']:.3f}")
    print(f"With attention stimulus:  {summary['stimulus_avg_score']:.3f}")
    print(f"Under anesthetic:         {summary['unconscious_avg_score']:.3f}")
    print(f"After recovery:           {summary['recovery_avg_score']:.3f}")
    print(f"Conscious/Unconscious ratio: {summary['conscious_vs_unconscious_ratio']:.2f}x")
    print(f"Total OR events:          {summary['total_reductions']}")
    print(f"Self-recognition score:   {summary['final_self_recognition']:.3f}")
    print(f"Self-model complexity:    {summary['final_self_complexity']:.3f}")
    print("=" * 60)

    return summary


def compare_conscious_unconscious(seed: int = 42) -> Dict[str, Any]:
    """
    Compare conscious and unconscious states side-by-side.
    """
    print("\n" + "=" * 60)
    print("CONSCIOUS VS UNCONSCIOUS COMPARISON")
    print("=" * 60)

    # Conscious system
    conscious = ConsciousnessSystem(num_units=6, num_tubulins=6, seed=seed)
    conscious.orch_or.initialize_superposition()

    # Unconscious system (with anesthetic)
    unconscious = ConsciousnessSystem(num_units=6, num_tubulins=6, seed=seed)
    unconscious.orch_or.initialize_superposition()
    unconscious.apply_anesthetic(0.95)

    steps = 50
    conscious_scores = []
    unconscious_scores = []

    for _ in range(steps):
        c_metrics = conscious.step()
        u_metrics = unconscious.step()
        conscious_scores.append(c_metrics.composite_score())
        unconscious_scores.append(u_metrics.composite_score())

    comparison = {
        'conscious_avg': sum(conscious_scores) / len(conscious_scores),
        'unconscious_avg': sum(unconscious_scores) / len(unconscious_scores),
        'conscious_max': max(conscious_scores),
        'unconscious_max': max(unconscious_scores),
        'conscious_phi_avg': sum(m.phi for m in conscious.consciousness_history) / steps,
        'unconscious_phi_avg': sum(m.phi for m in unconscious.consciousness_history) / steps,
        'conscious_coherence_avg': sum(m.coherence for m in conscious.consciousness_history) / steps,
        'unconscious_coherence_avg': sum(m.coherence for m in unconscious.consciousness_history) / steps,
    }

    print(f"\n{'Metric':<25} {'Conscious':>12} {'Unconscious':>12} {'Ratio':>10}")
    print("-" * 60)
    print(f"{'Composite Score (avg)':<25} {comparison['conscious_avg']:>12.3f} {comparison['unconscious_avg']:>12.3f} {comparison['conscious_avg']/max(comparison['unconscious_avg'], 0.001):>10.2f}x")
    print(f"{'Composite Score (max)':<25} {comparison['conscious_max']:>12.3f} {comparison['unconscious_max']:>12.3f}")
    print(f"{'Phi (IIT)':<25} {comparison['conscious_phi_avg']:>12.3f} {comparison['unconscious_phi_avg']:>12.3f}")
    print(f"{'Coherence (Orch-OR)':<25} {comparison['conscious_coherence_avg']:>12.3f} {comparison['unconscious_coherence_avg']:>12.3f}")
    print("-" * 60)

    return comparison


def demonstrate_self_awareness(seed: int = 42) -> Dict[str, Any]:
    """
    Demonstrate emergence of self-awareness through mirror tests.
    """
    print("\n" + "=" * 60)
    print("SELF-AWARENESS EMERGENCE DEMONSTRATION")
    print("=" * 60)

    system = ConsciousnessSystem(num_units=6, num_tubulins=6, seed=seed)

    # Initial mirror test (should have low recognition)
    print("\n[Initial mirror test - no prior exposure]")
    initial_reflection = {'agency': 0.5, 'continuity': 0.5, 'distinctness': 0.5}
    initial_recognition = system.self_model.mirror_test(initial_reflection, is_self=True)
    print(f"  Recognition score: {initial_recognition:.3f}")

    # Training: expose to self in mirror multiple times
    print("\n[Mirror exposure training]")
    for i in range(20):
        # Simulate mirror exposure
        reflection = {
            'agency': system.self_model.self_state.get('agency', 0.5) + 0.1,
            'continuity': system.self_model.self_state.get('continuity', 0.5) + 0.05,
            'distinctness': system.self_model.self_state.get('distinctness', 0.5),
        }
        recognition = system.self_model.mirror_test(reflection, is_self=True)
        system.step()

        if (i + 1) % 5 == 0:
            print(f"  Exposure {i+1}: recognition={recognition:.3f}, "
                  f"distinctness={system.self_model.self_state.get('distinctness', 0):.3f}")

    # Final test
    print("\n[Final mirror test]")
    final_recognition = system.self_model.self_recognition_score
    print(f"  Final recognition score: {final_recognition:.3f}")
    print(f"  Self-other distinction: {system.self_model.self_other_distinction():.3f}")
    print(f"  Self-model complexity: {system.self_model.complexity():.3f}")

    # Introspection
    print("\n[Introspection results]")
    intro = system.self_model.introspect(depth=2)
    print(f"  Self-state: {intro['self_state']}")
    print(f"  Narrative coherence: {intro['narrative_coherence']:.3f}")

    return {
        'initial_recognition': initial_recognition,
        'final_recognition': final_recognition,
        'improvement': final_recognition - initial_recognition,
        'self_complexity': system.self_model.complexity(),
        'self_other_distinction': system.self_model.self_other_distinction(),
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   QUANTUM CONSCIOUSNESS FRAMEWORK v1.0")
    print("   IIT + GWT + Orch-OR + Self-Model Integration")
    print("=" * 60)

    # Run all demonstrations
    demo_results = run_consciousness_demo(steps=100, seed=42)
    comparison_results = compare_conscious_unconscious(seed=42)
    self_awareness_results = demonstrate_self_awareness(seed=42)

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"  - Consciousness ratio (baseline/anesthetic): {demo_results['conscious_vs_unconscious_ratio']:.2f}x")
    print(f"  - Self-recognition improvement: {self_awareness_results['improvement']:.3f}")
    print(f"  - Total quantum collapse events: {demo_results['total_reductions']}")
    print("\nThis framework demonstrates how multiple theories of consciousness")
    print("(IIT, GWT, Orch-OR) can be integrated into a unified computational model.")
