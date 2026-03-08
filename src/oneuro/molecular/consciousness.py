"""Network-level consciousness metrics for MolecularNeuralNetwork.

Implements five information-theoretic and dynamical measures of consciousness:

1. **Approximate IIT Phi**: Integrated Information Theory (Tononi 2004).
   Builds effective connectivity from spike history, samples random bipartitions,
   computes mutual information across cuts.  Minimum = Phi lower bound.

2. **Perturbational Complexity Index (PCI)**: Casali et al. 2013.
   Perturb the network, record response, compute Lempel-Ziv complexity.
   High PCI = conscious, low PCI = unconscious (validated clinically).

3. **Neural Complexity (CN)**: Tononi & Sporns 1994.
   Balance of integration and segregation.  High CN = rich dynamics.

4. **Criticality**: Beggs & Plenz 2003.
   Branching ratio σ and avalanche size power-law exponent.
   σ=1.0 = critical (optimal information processing).

5. **Global Workspace Score**: Baars 1988 / Dehaene & Naccache 2001.
   Ignition events (local → global broadcast) and persistence.

6. **Orch-OR Network Phi**: Aggregate microtubule consciousness_measure
   across all neurons with cytoskeletons.

All metrics are non-invasive except PCI which uses save/restore snapshots.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oneuro.molecular.network import MolecularNeuralNetwork


# ---------------------------------------------------------------------------
# Lempel-Ziv complexity (LZ76)
# ---------------------------------------------------------------------------

def _lempel_ziv_complexity(binary_string: str) -> int:
    """Lempel-Ziv complexity of a binary string (LZ76 algorithm).

    Counts the number of distinct substrings encountered when parsing
    left to right.  Normalized by len/log2(len) gives a value in [0, 1].
    """
    n = len(binary_string)
    if n == 0:
        return 0

    complexity = 1
    l = 1  # current prefix length
    k = 1  # current component length

    while l + k <= n:
        # Check if substring s[l:l+k] appears in s[0:l+k-1]
        if binary_string[l:l + k] in binary_string[:l + k - 1]:
            k += 1
        else:
            complexity += 1
            l += k
            k = 1

    return complexity


def _normalized_lz_complexity(binary_string: str) -> float:
    """Lempel-Ziv complexity normalized to [0, 1]."""
    n = len(binary_string)
    if n <= 1:
        return 0.0
    raw = _lempel_ziv_complexity(binary_string)
    # Theoretical upper bound for random binary string
    upper = n / max(1.0, math.log2(n))
    return min(1.0, raw / upper)


# ---------------------------------------------------------------------------
# Network snapshot for PCI perturbation
# ---------------------------------------------------------------------------

@dataclass
class NetworkSnapshot:
    """Lightweight snapshot of network state for PCI restore."""

    voltages: Dict[int, float] = field(default_factory=dict)
    gating_m: Dict[int, float] = field(default_factory=dict)
    gating_h: Dict[int, float] = field(default_factory=dict)
    gating_n: Dict[int, float] = field(default_factory=dict)
    ca_internal: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def capture(cls, network: "MolecularNeuralNetwork") -> "NetworkSnapshot":
        """Capture minimal state needed for PCI restore."""
        snap = cls()
        for nid, mol_n in network._molecular_neurons.items():
            snap.voltages[nid] = mol_n.membrane._voltage
            snap.ca_internal[nid] = mol_n.membrane._ca_internal_nM
            # Gating variables from Na_v channel
            from oneuro.molecular.ion_channels import IonChannelType
            na_ch = mol_n.membrane.channels.get_channel(IonChannelType.Na_v)
            if na_ch is not None:
                snap.gating_m[nid] = na_ch.m
                snap.gating_h[nid] = na_ch.h
            kv_ch = mol_n.membrane.channels.get_channel(IonChannelType.K_v)
            if kv_ch is not None:
                snap.gating_n[nid] = kv_ch.n
        return snap

    def restore(self, network: "MolecularNeuralNetwork") -> None:
        """Restore network to captured state."""
        from oneuro.molecular.ion_channels import IonChannelType
        for nid, mol_n in network._molecular_neurons.items():
            if nid in self.voltages:
                mol_n.membrane._voltage = self.voltages[nid]
            if nid in self.ca_internal:
                mol_n.membrane._ca_internal_nM = self.ca_internal[nid]
            na_ch = mol_n.membrane.channels.get_channel(IonChannelType.Na_v)
            if na_ch is not None and nid in self.gating_m:
                na_ch.m = self.gating_m[nid]
                na_ch.h = self.gating_h.get(nid, na_ch.h)
            kv_ch = mol_n.membrane.channels.get_channel(IonChannelType.K_v)
            if kv_ch is not None and nid in self.gating_n:
                kv_ch.n = self.gating_n[nid]


# ---------------------------------------------------------------------------
# ConsciousnessMetrics
# ---------------------------------------------------------------------------

@dataclass
class ConsciousnessMetrics:
    """Container for all consciousness measurements."""

    phi_approx: float = 0.0          # IIT Phi (stochastic bipartition approximation)
    pci: float = 0.0                 # Perturbational Complexity Index [0, 1]
    neural_complexity: float = 0.0   # Tononi-Sporns CN
    branching_ratio: float = 0.0     # Criticality sigma (1.0 = critical)
    avalanche_exponent: float = 0.0  # Power-law exponent of avalanche sizes
    global_workspace_score: float = 0.0  # Ignition + broadcasting
    orch_or_phi: float = 0.0         # Aggregated Orch-OR from cytoskeletons
    composite: float = 0.0           # Weighted mean

    def compute_composite(self, n_neurons: int = 100) -> float:
        """Weighted mean of all metrics, each normalized to [0, 1].

        Args:
            n_neurons: Network size for scale-adaptive Phi normalization.
                       Phi grows with N, so we use log-scale normalization
                       to preserve differentiation at all scales.
        """
        # Phi: log-scale normalization (Phi grows with N, raw /5.0 saturates)
        if self.phi_approx > 0 and n_neurons > 1:
            phi_norm = min(1.0, math.log(1.0 + self.phi_approx)
                          / math.log(1.0 + n_neurons ** 1.5))
        else:
            phi_norm = 0.0
        # PCI already in [0, 1]
        pci_norm = self.pci
        # Neural complexity: typical values 0-2, normalize by 2
        cn_norm = min(1.0, self.neural_complexity / 2.0)
        # Branching ratio: 1.0 is critical, score = 1 - |1 - sigma|
        crit_norm = max(0.0, 1.0 - abs(1.0 - self.branching_ratio))
        # Avalanche exponent: 1.5 is critical (mean-field), score by proximity
        aval_norm = max(0.0, 1.0 - abs(1.5 - self.avalanche_exponent) / 1.5)
        # GW score already in [0, 1]
        gw_norm = self.global_workspace_score
        # Orch-OR: typical 0-1
        orch_norm = min(1.0, self.orch_or_phi)

        weights = [0.25, 0.15, 0.10, 0.15, 0.10, 0.15, 0.10]
        values = [phi_norm, pci_norm, cn_norm, crit_norm, aval_norm, gw_norm, orch_norm]
        self.composite = sum(w * v for w, v in zip(weights, values))
        return self.composite


# ---------------------------------------------------------------------------
# ConsciousnessMonitor
# ---------------------------------------------------------------------------

class ConsciousnessMonitor:
    """Monitors a MolecularNeuralNetwork and computes consciousness metrics.

    Usage:
        monitor = ConsciousnessMonitor(network)
        for _ in range(1000):
            network.step(0.1)
            monitor.record_step(network.last_fired)
        metrics = monitor.compute_all()
    """

    def __init__(self, network: "MolecularNeuralNetwork", history_length: int = 1000):
        self.network = network
        self.history_length = history_length
        self._spike_history: List[Set[int]] = []
        self._avalanche_sizes: List[int] = []
        self._avalanche_durations: List[int] = []
        self._current_avalanche_size: int = 0
        self._current_avalanche_duration: int = 0
        self._in_avalanche: bool = False

    def record_step(self, fired_neurons: Set[int]) -> None:
        """Record which neurons fired at this timestep."""
        self._spike_history.append(set(fired_neurons))
        if len(self._spike_history) > self.history_length:
            self._spike_history.pop(0)

        # Avalanche tracking
        if fired_neurons:
            self._current_avalanche_size += len(fired_neurons)
            self._current_avalanche_duration += 1
            self._in_avalanche = True
        elif self._in_avalanche:
            # Avalanche ended
            if self._current_avalanche_size > 0:
                self._avalanche_sizes.append(self._current_avalanche_size)
                self._avalanche_durations.append(self._current_avalanche_duration)
            self._current_avalanche_size = 0
            self._current_avalanche_duration = 0
            self._in_avalanche = False

    def compute_all(self) -> ConsciousnessMetrics:
        """Compute all consciousness metrics from recorded spike history."""
        metrics = ConsciousnessMetrics()

        if len(self._spike_history) < 50:
            return metrics

        metrics.phi_approx = self.phi_approximate()
        metrics.pci = self.perturbational_complexity_index()
        metrics.neural_complexity = self.neural_complexity()
        br, ae = self.criticality_metrics()
        metrics.branching_ratio = br
        metrics.avalanche_exponent = ae
        metrics.global_workspace_score = self.global_workspace_score()
        metrics.orch_or_phi = self.orch_or_network_phi()
        metrics.compute_composite(n_neurons=len(self.network._molecular_neurons))

        return metrics

    # ---- 1. Approximate IIT Phi ----

    def phi_approximate(self, n_partitions: int = 50) -> float:
        """Approximate IIT Phi via stochastic bipartition sampling.

        Builds effective connectivity W[i,j] = P(j fires at t+1 | i fires at t)
        from spike history, samples random bipartitions, computes mutual
        information across each cut.  Returns minimum Phi (lower bound).
        """
        neuron_ids = sorted(self.network._molecular_neurons.keys())
        n = len(neuron_ids)
        if n < 4:
            return 0.0

        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

        # Build effective connectivity matrix
        W = np.zeros((n, n))
        fire_counts = np.zeros(n)

        for t in range(len(self._spike_history) - 1):
            current = self._spike_history[t]
            next_step = self._spike_history[t + 1]
            for nid in current:
                if nid in id_to_idx:
                    i = id_to_idx[nid]
                    fire_counts[i] += 1
                    for nid_next in next_step:
                        if nid_next in id_to_idx:
                            j = id_to_idx[nid_next]
                            W[i, j] += 1

        # Normalize: P(j fires at t+1 | i fires at t)
        for i in range(n):
            if fire_counts[i] > 0:
                W[i, :] /= fire_counts[i]

        # Sample bipartitions and compute MI across cuts
        min_phi = float('inf')
        for _ in range(n_partitions):
            # Random bipartition (each part must have at least 1 neuron)
            perm = np.random.permutation(n)
            split = np.random.randint(1, n)
            part_a = set(perm[:split])
            part_b = set(perm[split:])

            # MI across cut: sum of W[i,j] for i in A, j in B (and vice versa)
            mi_across = 0.0
            for i in part_a:
                for j in part_b:
                    if W[i, j] > 0:
                        mi_across += W[i, j] * math.log2(max(1e-10, W[i, j]) + 1e-10)
            for i in part_b:
                for j in part_a:
                    if W[i, j] > 0:
                        mi_across += W[i, j] * math.log2(max(1e-10, W[i, j]) + 1e-10)

            # Phi for this cut = total mutual information across the partition
            phi_cut = abs(mi_across)
            min_phi = min(min_phi, phi_cut)

        return min_phi if min_phi < float('inf') else 0.0

    # ---- 2. Perturbational Complexity Index ----

    def perturbational_complexity_index(
        self, n_perturbations: int = 10, perturb_neurons: int = 0,
        perturb_current: float = 0.0, response_steps: int = 100,
        perturb_duration: int = 0,
    ) -> float:
        """PCI: perturb the network and measure response complexity.

        Saves network state, injects current, records binary spike matrix,
        computes Lempel-Ziv complexity, restores state.  Repeat and average.

        perturb_neurons=0 means auto-scale: ~5% of network size (min 5).
        perturb_current=0 means auto-scale: 25 µA for small, more for large.
        perturb_duration=0 means auto-scale: 1 step for small, 5 for large.
        """
        neuron_ids = sorted(self.network._molecular_neurons.keys())
        n = len(neuron_ids)
        if n < 5:
            return 0.0
        if perturb_neurons <= 0:
            perturb_neurons = max(5, n // 20)  # ~5% of network
        if perturb_current <= 0:
            # Scale current with network size: 25 at N=75, 50 at N=500, 80 at N=1000+
            perturb_current = min(80.0, 25.0 + 0.06 * n)
        if perturb_duration <= 0:
            # Multi-step pulse for large networks: 1 at N<100, 3 at N=500, 5 at N=1000+
            perturb_duration = max(1, min(5, n // 200))

        pci_values = []
        for _ in range(n_perturbations):
            # Save state
            snapshot = NetworkSnapshot.capture(self.network)

            # Choose random neurons to perturb
            targets = list(np.random.choice(neuron_ids, min(perturb_neurons, n), replace=False))

            # Inject current and record response
            binary_matrix = []
            for step in range(response_steps):
                # Inject pulsed current for perturb_duration steps
                if step < perturb_duration and step % 2 == 0:
                    for nid in targets:
                        self.network._external_currents[nid] = (
                            self.network._external_currents.get(nid, 0.0) + perturb_current
                        )

                self.network.step(0.1)

                # Record binary spike vector
                row = []
                for nid in neuron_ids:
                    mol_n = self.network._molecular_neurons.get(nid)
                    if mol_n is not None and mol_n.membrane.fired:
                        row.append('1')
                    else:
                        row.append('0')
                binary_matrix.append(''.join(row))

            # Flatten to single binary string (time × neurons)
            flat = ''.join(binary_matrix)
            lz = _normalized_lz_complexity(flat)
            pci_values.append(lz)

            # Restore state
            snapshot.restore(self.network)

        return sum(pci_values) / len(pci_values) if pci_values else 0.0

    # ---- 3. Neural Complexity ----

    def neural_complexity(self, n_samples: int = 20) -> float:
        """Tononi-Sporns neural complexity: balance of integration and segregation.

        CN = Σ_{k=1}^{N-1} [H(subset_k) + H(rest_k) - H(all)] averaged over
        random subsets of size k.
        """
        neuron_ids = sorted(self.network._molecular_neurons.keys())
        n = len(neuron_ids)
        if n < 4:
            return 0.0

        # Build binary activation matrix: time × neurons
        T = len(self._spike_history)
        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

        act_matrix = np.zeros((T, n), dtype=np.float32)
        for t, fired in enumerate(self._spike_history):
            for nid in fired:
                if nid in id_to_idx:
                    act_matrix[t, id_to_idx[nid]] = 1.0

        # Entropy of binary vector
        def _entropy(cols: np.ndarray) -> float:
            # cols: (T, k) binary matrix
            if cols.shape[1] == 0:
                return 0.0
            # Estimate entropy from mean firing probability per neuron
            probs = np.mean(cols, axis=0)
            h = 0.0
            for p in probs:
                if 0.0 < p < 1.0:
                    h += -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
            return h

        H_all = _entropy(act_matrix)

        cn = 0.0
        for k in range(1, min(n, 8)):  # Cap at 8 for efficiency
            for _ in range(n_samples):
                subset_idx = np.random.choice(n, k, replace=False)
                rest_idx = np.array([i for i in range(n) if i not in subset_idx])

                H_subset = _entropy(act_matrix[:, subset_idx])
                H_rest = _entropy(act_matrix[:, rest_idx])

                # Mutual information between subset and rest
                mi = H_subset + H_rest - H_all
                cn += max(0.0, mi)

        cn /= max(1, n_samples * min(n - 1, 7))
        return cn

    # ---- 4. Criticality Metrics ----

    def criticality_metrics(self) -> Tuple[float, float]:
        """Branching ratio and avalanche exponent.

        Returns:
            (branching_ratio, avalanche_exponent)
            branching_ratio σ = 1.0 is critical
            avalanche_exponent ~ 1.5 for critical systems (mean-field)
        """
        # Branching ratio from spike history
        branching_ratios = []
        for t in range(len(self._spike_history) - 1):
            ancestors = len(self._spike_history[t])
            descendants = len(self._spike_history[t + 1])
            if ancestors > 0:
                branching_ratios.append(descendants / ancestors)

        sigma = np.mean(branching_ratios) if branching_ratios else 0.0

        # Avalanche exponent via MLE
        sizes = self._avalanche_sizes
        if len(sizes) < 10:
            return float(sigma), 0.0

        sizes_arr = np.array(sizes, dtype=float)
        # MLE for power-law exponent: alpha = 1 + n / Σ ln(x_i / x_min)
        x_min = max(1.0, np.min(sizes_arr))
        valid = sizes_arr[sizes_arr >= x_min]
        if len(valid) < 5:
            return float(sigma), 0.0

        log_sum = np.sum(np.log(valid / x_min))
        if abs(log_sum) < 1e-10:
            return float(sigma), 0.0
        alpha = 1.0 + len(valid) / log_sum
        return float(sigma), float(alpha)

    # ---- 5. Global Workspace Score ----

    def global_workspace_score(self) -> float:
        """Measure ignition events and broadcasting persistence.

        Thresholds scale with network size: large networks rarely have >50%
        simultaneously active, so we use adaptive thresholds.

        Ignition: <low_thresh active → >high_thresh active within 5 steps.
        Broadcasting: sustained >broadcast_thresh activity for 3+ steps.
        """
        n_neurons = len(self.network._molecular_neurons)
        if n_neurons < 5:
            return 0.0

        # Scale-adaptive thresholds: smaller fractions for larger networks
        # At N=75: low=15%, high=40%, broadcast=25% (original-like)
        # At N=1000: low=3%, high=10%, broadcast=5%
        scale_factor = min(1.0, 75.0 / max(1.0, n_neurons))
        low_thresh = max(0.02, 0.15 * scale_factor + 0.02)
        high_thresh = max(0.05, 0.40 * scale_factor + 0.05)
        broadcast_thresh = max(0.03, 0.25 * scale_factor + 0.03)

        ignition_count = 0
        broadcast_duration = 0
        in_broadcast = False

        for t in range(len(self._spike_history) - 5):
            frac_t = len(self._spike_history[t]) / n_neurons

            # Check for ignition: low activity → high activity within 5 steps
            if frac_t < low_thresh:
                for dt in range(1, 6):
                    if t + dt < len(self._spike_history):
                        frac_future = len(self._spike_history[t + dt]) / n_neurons
                        if frac_future > high_thresh:
                            ignition_count += 1
                            break

            # Track broadcasting persistence
            if frac_t > broadcast_thresh:
                if not in_broadcast:
                    in_broadcast = True
                broadcast_duration += 1
            else:
                in_broadcast = False

        # Normalize
        T = len(self._spike_history)
        ignition_rate = ignition_count / max(1, T)
        broadcast_frac = broadcast_duration / max(1, T)

        # Combine: ignition frequency × broadcast persistence
        score = min(1.0, ignition_rate * 50.0 + broadcast_frac * 2.0)
        return score

    # ---- 6. Orch-OR Network Phi ----

    def orch_or_network_phi(self) -> float:
        """Aggregate Orch-OR consciousness_measure across all neurons."""
        total = 0.0
        count = 0
        for mol_n in self.network._molecular_neurons.values():
            if mol_n.cytoskeleton is not None:
                cm = mol_n.cytoskeleton.consciousness_measure
                mt_count = len(mol_n.cytoskeleton.microtubules)
                total += cm * max(1, mt_count)
                count += 1

        if count == 0:
            return 0.0
        # Normalize: average consciousness per neuron weighted by microtubule count
        return total / count
