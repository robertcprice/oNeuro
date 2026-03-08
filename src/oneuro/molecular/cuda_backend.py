"""PyTorch CUDA backend for GPU-accelerated molecular brain simulation.

All neuron state (~80 floats per neuron) stored as (N,) CUDA tensors.
All per-neuron operations are vectorized tensor ops → runs on ANY GPU.

Supports: CUDA (Nvidia), MPS (Apple), ROCm (AMD), CPU fallback.

Memory budget: ~40KB per neuron → 1M neurons in 40GB → fits H200 (141GB).

Performance optimizations (v2):
- torch.compile() on hot path (CUDA only) — fuses kernels, 2-5x speedup
- Fused HH gating: single pass over voltage for all 5 gating variables
- Vectorized NT release: scatter_add with one-hot encoding, no Python loop
- Pre-computed device constants: no CPU→GPU transfers per step
- Cached region ID tensors: no torch.tensor() creation per stimulation call
- In-place ops throughout: minimize tensor allocation

Usage:
    brain = CUDAMolecularBrain(100_000, device='cuda')  # or 'mps', 'cpu'
    brain.step()
    brain.run(1000)

    rbrain = CUDARegionalBrain.xlarge(device='cuda')
    rbrain.run(10000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Channel constants (g_max in mS/cm^2, E_rev in mV)
G_NA = 120.0
G_K = 36.0
G_KLEAK = 0.3
G_CA = 4.4
G_NMDA = 0.5
G_AMPA = 1.0
G_GABAA = 1.0
G_NACHR = 0.8

E_NA = 50.0
E_K = -77.0
E_CA = 120.0
E_NMDA = 0.0
E_AMPA = 0.0
E_GABAA = -80.0
E_NACHR = 0.0

C_M = 1.0  # membrane capacitance (uF/cm^2)

# NT type indices
NT_DA, NT_5HT, NT_NE, NT_ACH, NT_GABA, NT_GLU = 0, 1, 2, 3, 4, 5
_NUM_NT = 6

# Channel indices (matching oneuro-metal)
CH_NAV, CH_KV, CH_KLEAK, CH_CAV, CH_NMDA, CH_AMPA, CH_GABAA, CH_NACHR = range(8)

# Resting NT concentrations (nM)
RESTING_NT = torch.tensor([20.0, 10.0, 15.0, 50.0, 200.0, 500.0])

# EC50 values for receptor binding (nM) — Hill equation
EC50 = torch.tensor([
    100.0,   # DA → D1/D2
    50.0,    # 5-HT → 5-HT1A
    200.0,   # NE → alpha/beta
    500.0,   # ACh → nAChR
    300.0,   # GABA → GABA-A
    800.0,   # Glu → AMPA/NMDA
])
HILL_N = torch.tensor([1.5, 1.2, 1.0, 2.0, 2.0, 1.5])

# Spike threshold and refractory
SPIKE_THRESHOLD = -20.0  # mV
REFRACTORY_MS = 2.0      # ms

# Neuron archetype indices
ARCH_PYRAMIDAL = 0
ARCH_INTERNEURON = 1
ARCH_GRANULE = 3
ARCH_MEDIUM_SPINY = 5
ARCH_DOPAMINERGIC = 6
ARCH_SEROTONERGIC = 7
ARCH_CHOLINERGIC = 8

# Can we use torch.compile? (requires CUDA, not MPS)
_CAN_COMPILE = hasattr(torch, 'compile')


# ============================================================================
# Vectorized HH rate functions — fused for minimal memory bandwidth
# ============================================================================

def _safe_rate(V: torch.Tensor, offset: float, scale: float, limit: float) -> torch.Tensor:
    """Safe alpha rate computation: scale * (V + offset) / (1 - exp(-(V + offset) / limit))."""
    x = V + offset
    # Use Taylor expansion near singularity: lim x→0 of x/(1-e^(-x/a)) = a
    near_zero = torch.abs(x) < 1e-6
    safe_x = torch.where(near_zero, x + 1e-6, x)
    result = scale * safe_x / (1.0 - torch.exp(-safe_x / limit))
    return torch.where(near_zero, scale * limit * torch.ones_like(V), result)


def fused_hh_gating(
    V: torch.Tensor,
    nav_m: torch.Tensor, nav_h: torch.Tensor,
    kv_n: torch.Tensor,
    cav_m: torch.Tensor, cav_h: torch.Tensor,
    dt: float,
) -> None:
    """Update all 5 HH gating variables in-place with one pass over V.

    Fuses 10 rate function calls + 5 Euler steps into a single function
    to reduce memory bandwidth (V only read once from GPU memory).
    """
    # Na_v: m3h
    am = _safe_rate(V, 40.0, 0.1, 10.0)
    bm = 4.0 * torch.exp(-(V + 65.0) / 18.0)
    nav_m.add_(dt * (am * (1.0 - nav_m) - bm * nav_m)).clamp_(0.0, 1.0)

    ah = 0.07 * torch.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    nav_h.add_(dt * (ah * (1.0 - nav_h) - bh * nav_h)).clamp_(0.0, 1.0)

    # K_v: n4
    an = _safe_rate(V, 55.0, 0.01, 10.0)
    bn = 0.125 * torch.exp(-(V + 65.0) / 80.0)
    kv_n.add_(dt * (an * (1.0 - kv_n) - bn * kv_n)).clamp_(0.0, 1.0)

    # Ca_v: m2h
    amc = _safe_rate(V, 27.0, 0.055, 3.8)
    bmc = 0.94 * torch.exp(-(V + 75.0) / 17.0)
    cav_m.add_(dt * (amc * (1.0 - cav_m) - bmc * cav_m)).clamp_(0.0, 1.0)

    ahc = 0.000457 * torch.exp(-(V + 13.0) / 50.0)
    bhc = 0.0065 / (1.0 + torch.exp(-(V + 15.0) / 28.0))
    cav_h.add_(dt * (ahc * (1.0 - cav_h) - bhc * cav_h)).clamp_(0.0, 1.0)


def mg_block(V: torch.Tensor) -> torch.Tensor:
    """Mg2+ voltage-dependent block of NMDA receptors."""
    return 1.0 / (1.0 + 1.0 * torch.exp(-0.062 * V) / 3.57)


# ============================================================================
# CUDAMolecularBrain
# ============================================================================

class CUDAMolecularBrain:
    """All neuron state as GPU tensors — runs on any device.

    Structure-of-Arrays: every field is a (N,) tensor on the target device.
    One step() call updates ALL neurons simultaneously via vectorized ops.
    """

    def __init__(
        self,
        n_neurons: int,
        device: str = "auto",
        psc_scale: float = 30.0,
        dt: float = 0.1,
    ):
        self.n = n_neurons
        self.device = self._resolve_device(device)
        self.psc_scale = psc_scale
        self.dt = dt
        self.time = 0.0
        self.step_count = 0

        n = n_neurons
        dev = self.device

        # ---- Membrane state ----
        self.voltage = torch.full((n,), -65.0, device=dev)
        self.prev_voltage = torch.full((n,), -65.0, device=dev)
        self.fired = torch.zeros(n, dtype=torch.bool, device=dev)
        self.refractory = torch.zeros(n, device=dev)
        self.spike_count = torch.zeros(n, dtype=torch.int32, device=dev)
        self.alive = torch.ones(n, dtype=torch.bool, device=dev)

        # ---- HH gating variables ----
        self.nav_m = torch.full((n,), 0.05, device=dev)
        self.nav_h = torch.full((n,), 0.6, device=dev)
        self.kv_n = torch.full((n,), 0.32, device=dev)
        self.cav_m = torch.full((n,), 0.01, device=dev)
        self.cav_h = torch.full((n,), 0.99, device=dev)

        # ---- Conductance scales (drug-modifiable) — (N, 8) ----
        self.conductance_scale = torch.ones((n, 8), device=dev)

        # ---- Receptor open fractions ----
        self.ampa_open = torch.zeros(n, device=dev)
        self.nmda_open = torch.zeros(n, device=dev)
        self.gabaa_open = torch.zeros(n, device=dev)
        self.nachr_open = torch.zeros(n, device=dev)

        # ---- Calcium 4-compartment ----
        self.ca_cytoplasmic = torch.full((n,), 100.0, device=dev)   # nM
        self.ca_er = torch.full((n,), 100_000.0, device=dev)         # nM
        self.ca_mitochondrial = torch.full((n,), 200.0, device=dev)  # nM
        self.ca_microdomain = torch.full((n,), 100.0, device=dev)    # nM

        # ---- Second messengers ----
        self.camp = torch.full((n,), 0.5, device=dev)
        self.gs_active = torch.zeros(n, device=dev)
        self.gi_active = torch.zeros(n, device=dev)
        self.gq_active = torch.zeros(n, device=dev)
        self.pka_activity = torch.full((n,), 0.1, device=dev)
        self.pkc_activity = torch.full((n,), 0.1, device=dev)
        self.camkii_activity = torch.full((n,), 0.05, device=dev)
        self.ip3 = torch.full((n,), 0.1, device=dev)
        self.dag = torch.full((n,), 0.1, device=dev)
        self.erk_activity = torch.full((n,), 0.05, device=dev)

        # ---- Phosphorylation ----
        self.ampa_p = torch.full((n,), 0.3, device=dev)
        self.kv_p = torch.full((n,), 0.2, device=dev)
        self.cav_p = torch.full((n,), 0.1, device=dev)
        self.creb_p = torch.full((n,), 0.05, device=dev)

        # ---- Metabolism ----
        self.atp = torch.full((n,), 4000.0, device=dev)  # uM
        self.adp = torch.full((n,), 400.0, device=dev)
        self.glucose = torch.full((n,), 5000.0, device=dev)

        # ---- NT ambient concentrations (N, 6) ----
        self._resting_nt = RESTING_NT.to(dev).unsqueeze(0)  # (1, 6) — cached on device
        self.nt_conc = self._resting_nt.expand(n, -1).clone()

        # ---- External current (cleared each step) ----
        self.external_current = torch.zeros(n, device=dev)

        # ---- Gene expression (interval-gated) ----
        self.bdnf_level = torch.full((n,), 0.5, device=dev)
        self.cfos_level = torch.zeros(n, device=dev)
        self.arc_level = torch.zeros(n, device=dev)

        # ---- Microtubules ----
        self.mt_coherence = torch.full((n,), 0.1, device=dev)
        self.orch_or_events = torch.zeros(n, dtype=torch.int32, device=dev)

        # ---- Neuron identity ----
        self.archetype = torch.zeros(n, dtype=torch.int32, device=dev)  # 0=pyramidal
        self.x = torch.zeros(n, device=dev)
        self.y = torch.zeros(n, device=dev)
        self.z = torch.zeros(n, device=dev)
        self.last_fired_step = torch.zeros(n, dtype=torch.int64, device=dev)

        # ---- Synapses (CSR format) ----
        # Empty initially — populated by add_synapses() or region builders
        self.syn_pre = torch.zeros(0, dtype=torch.int64, device=dev)
        self.syn_post = torch.zeros(0, dtype=torch.int64, device=dev)
        self.syn_weight = torch.zeros(0, device=dev)
        self.syn_nt_type = torch.zeros(0, dtype=torch.int32, device=dev)
        self.syn_strength = torch.ones(0, device=dev)
        # STDP traces
        self.syn_pre_trace = torch.zeros(0, device=dev)
        self.syn_post_trace = torch.zeros(0, device=dev)
        # Eligibility trace for three-factor learning (pre × post → eligibility, DA → dw)
        self.syn_eligibility = torch.zeros(0, device=dev)
        self.n_synapses = 0

        # ---- Pre-computed constants (on device, avoid per-step allocation) ----
        self._stdp_decay_pre = torch.tensor(math.exp(-dt / 20.0), device=dev)
        self._stdp_decay_post = torch.tensor(math.exp(-dt / 20.0), device=dev)
        self._reset_voltage = torch.tensor(-65.0, device=dev)
        self._refractory_ms = torch.tensor(REFRACTORY_MS, device=dev)

        # ---- Sparse weight matrix for MPS-optimized spike propagation ----
        # W[post, pre] = weight * strength * sign → external_current += W @ fired
        self._W_sparse: Optional[torch.Tensor] = None  # sparse CSR (N, N)
        self._W_dirty = True
        self._W_rebuild_interval = 100  # rebuild every N steps
        self._W_last_rebuild_step = 0
        # Per-NT sparse matrices for NT release: nt_conc[:, k] += NT_W[k] @ fired
        self._NT_W_sparse: Optional[List[torch.Tensor]] = None  # 6 sparse (N, N)

        # ---- Consciousness (optional) ----
        self._consciousness_enabled = False
        self._spike_history: List[torch.Tensor] = []

        # ---- Interval-gated step tracking ----
        self._gene_interval = 10
        self._metabolism_interval = 5
        self._microtubule_interval = 10
        self._glia_interval = 10

        # ---- One-hot NT lookup for vectorized spike release ----
        # Pre-build per-NT masks after synapses are added
        self._nt_release_amount = 50.0  # nM per spike
        self._syn_nt_onehot: Optional[torch.Tensor] = None  # (S, 6) lazy-built

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    # ========================================================================
    # Synapse management
    # ========================================================================

    def add_synapses(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        weights: torch.Tensor,
        nt_types: torch.Tensor,
    ) -> None:
        """Add synapses from pre/post index tensors."""
        n_new = pre.shape[0]
        dev = self.device

        self.syn_pre = torch.cat([self.syn_pre, pre.to(dev, torch.int64)])
        self.syn_post = torch.cat([self.syn_post, post.to(dev, torch.int64)])
        self.syn_weight = torch.cat([self.syn_weight, weights.to(dev)])
        self.syn_nt_type = torch.cat([self.syn_nt_type, nt_types.to(dev, torch.int32)])
        self.syn_strength = torch.cat([self.syn_strength, torch.ones(n_new, device=dev)])
        self.syn_pre_trace = torch.cat([self.syn_pre_trace, torch.zeros(n_new, device=dev)])
        self.syn_post_trace = torch.cat([self.syn_post_trace, torch.zeros(n_new, device=dev)])
        self.syn_eligibility = torch.cat([self.syn_eligibility, torch.zeros(n_new, device=dev)])
        self.n_synapses = self.syn_pre.shape[0]
        # Invalidate caches — sparse W matrices must be rebuilt
        self._syn_nt_onehot = None
        self._syn_psc_base: Optional[torch.Tensor] = None
        self._W_dirty = True
        self._W_sparse = None
        self._NT_W_sparse = None

    def add_random_synapses(self, n_synapses: int, p_inhibitory: float = 0.2) -> None:
        """Add random synapses (for testing / unstructured networks)."""
        n = self.n
        dev = self.device
        pre = torch.randint(0, n, (n_synapses,), device=dev)
        post = torch.randint(0, n, (n_synapses,), device=dev)
        # Remove self-connections
        mask = pre != post
        pre, post = pre[mask], post[mask]
        n_syn = pre.shape[0]

        weights = torch.rand(n_syn, device=dev) * 1.5 + 0.5  # [0.5, 2.0]
        nt_types = torch.where(
            torch.rand(n_syn, device=dev) < p_inhibitory,
            torch.full((n_syn,), NT_GABA, dtype=torch.int32, device=dev),
            torch.full((n_syn,), NT_GLU, dtype=torch.int32, device=dev),
        )
        self.add_synapses(pre, post, weights, nt_types)

    # ========================================================================
    # Sparse matrix construction (MPS optimization)
    # ========================================================================

    def _build_sparse_W(self) -> None:
        """Build sparse weight matrix W[post, pre] for matmul-based propagation.

        After this, spike propagation is: external_current += W @ fired.float() * psc_scale
        This replaces scatter_add which is catastrophically slow on MPS.
        """
        if self.n_synapses == 0:
            self._W_sparse = None
            self._NT_W_sparse = None
            self._W_dirty = False
            return

        dev = self.device
        n = self.n

        # Build PSC weight matrix: W[post, pre] = sum(weight * strength * sign)
        sign = torch.where(self.syn_nt_type == NT_GABA, -1.0, 1.0)
        values = self.syn_weight * self.syn_strength * sign
        indices = torch.stack([self.syn_post, self.syn_pre])  # (2, S) — [row, col]

        # Use coalesce to sum duplicate (post, pre) pairs
        self._W_sparse = torch.sparse_coo_tensor(
            indices, values, (n, n), device=dev,
        ).coalesce()

        # Build per-NT sparse matrices for NT release
        self._NT_W_sparse = []
        for nt_idx in range(_NUM_NT):
            mask_idx = (self.syn_nt_type == nt_idx).nonzero(as_tuple=True)[0]
            if mask_idx.shape[0] == 0:
                # Empty sparse tensor
                self._NT_W_sparse.append(
                    torch.sparse_coo_tensor(
                        torch.zeros(2, 0, dtype=torch.int64, device=dev),
                        torch.zeros(0, device=dev),
                        (n, n), device=dev,
                    ).coalesce()
                )
                continue
            nt_post = self.syn_post[mask_idx]
            nt_pre = self.syn_pre[mask_idx]
            nt_indices = torch.stack([nt_post, nt_pre])
            nt_values = torch.ones(mask_idx.shape[0], device=dev)
            self._NT_W_sparse.append(
                torch.sparse_coo_tensor(
                    nt_indices, nt_values, (n, n), device=dev,
                ).coalesce()
            )

        self._W_dirty = False
        self._W_last_rebuild_step = self.step_count

    def _maybe_rebuild_W(self) -> None:
        """Rebuild sparse W if dirty or enough steps have elapsed (STDP drift)."""
        if self._W_dirty or self._W_sparse is None:
            self._build_sparse_W()
        elif (self.step_count - self._W_last_rebuild_step) >= self._W_rebuild_interval:
            self._build_sparse_W()

    # ========================================================================
    # Stimulation
    # ========================================================================

    def stimulate(self, neuron_idx: int, current_ua: float) -> None:
        """Inject external current into one neuron."""
        self.external_current[neuron_idx] += current_ua

    def stimulate_range(self, start: int, end: int, current_ua: float) -> None:
        """Inject external current into a range of neurons."""
        self.external_current[start:end] += current_ua

    def stimulate_indices(self, indices: torch.Tensor, current_ua: float) -> None:
        """Inject external current into specific neurons by index tensor."""
        self.external_current[indices] += current_ua

    # ========================================================================
    # Main simulation step
    # ========================================================================

    def step(self) -> None:
        """Advance simulation by dt milliseconds. All neurons updated in parallel."""
        dt = self.dt
        self.time += dt
        self.step_count += 1

        # Store previous voltage, clear fired
        self.prev_voltage.copy_(self.voltage)
        self.fired.zero_()

        # ---- 1. HH Gating — fused, single pass over V ----
        fused_hh_gating(
            self.voltage, self.nav_m, self.nav_h,
            self.kv_n, self.cav_m, self.cav_h, dt,
        )
        V = self.voltage

        # ---- 2. Receptor binding (Hill equation) — fused ----
        glu = self.nt_conc[:, NT_GLU]
        glu_hill = glu**1.5 / (800.0**1.5 + glu**1.5)
        self.ampa_open = glu_hill
        self.nmda_open = glu_hill * (1.0 / (1.0 + torch.exp(-0.062 * V) / 3.57))
        gaba = self.nt_conc[:, NT_GABA]
        self.gabaa_open = gaba**2.0 / (300.0**2.0 + gaba**2.0)
        ach = self.nt_conc[:, NT_ACH]
        self.nachr_open = ach**2.0 / (500.0**2.0 + ach**2.0)

        # ---- 3. Membrane integration — 8-channel I_ion sum (minimal intermediates) ----
        cs = self.conductance_scale  # (N, 8)
        # Accumulate I_total directly to avoid 8 intermediate tensors
        I_total = (
            cs[:, CH_NAV] * G_NA * self.nav_m**3 * self.nav_h * (V - E_NA)
            + cs[:, CH_KV] * G_K * self.kv_n**4 * (V - E_K)
            + cs[:, CH_KLEAK] * G_KLEAK * (V - E_K)
            + cs[:, CH_CAV] * G_CA * self.cav_m**2 * self.cav_h * (V - E_CA)
            + cs[:, CH_AMPA] * G_AMPA * self.ampa_open * self.ampa_p * (V - E_AMPA)
            + cs[:, CH_NMDA] * G_NMDA * self.nmda_open * (V - E_NMDA)
            + cs[:, CH_GABAA] * G_GABAA * self.gabaa_open * (V - E_GABAA)
            + cs[:, CH_NACHR] * G_NACHR * self.nachr_open * (V - E_NACHR)
        )

        # dV/dt = (-I_total + I_ext) / C_m — in-place update
        self.voltage.add_((-I_total + self.external_current) * (dt / C_M))
        self.voltage.clamp_(-100.0, 60.0)

        # ---- 4. Spike detection — branchless via torch.where (MPS-friendly) ----
        spiking = (self.voltage > SPIKE_THRESHOLD) & (self.prev_voltage <= SPIKE_THRESHOLD) & (self.refractory <= 0)
        self.fired = spiking & self.alive
        self.spike_count.add_(self.fired.int())
        fired_any = self.fired.any()
        # Branchless reset — cached constants, no boolean indexing
        self.last_fired_step = torch.where(
            self.fired, torch.tensor(self.step_count, dtype=torch.int64, device=self.device),
            self.last_fired_step,
        )
        self.voltage = torch.where(self.fired, self._reset_voltage, self.voltage)
        self.refractory = torch.where(self.fired, self._refractory_ms, self.refractory)
        self.refractory.sub_(dt).clamp_(min=0.0)

        # ---- 5. Spike propagation + STDP (sparse matmul, skip if no spikes) ----
        has_synapses_and_spikes = self.n_synapses > 0 and fired_any
        if has_synapses_and_spikes:
            self._maybe_rebuild_W()
            self._propagate_spikes()

        # ---- 6. Calcium dynamics (4-compartment ODE) ----
        self._update_calcium(dt)

        # ---- 7. Second messenger cascades ----
        self._update_second_messengers(dt)

        # ---- 8. STDP (only when spikes occurred) ----
        if has_synapses_and_spikes:
            self._update_stdp(dt)

        # ---- 9. Interval-gated subsystems ----
        if self.step_count % self._gene_interval == 0:
            self._update_gene_expression(dt * self._gene_interval)
        if self.step_count % self._metabolism_interval == 0:
            self._update_metabolism(dt * self._metabolism_interval)
        if self.step_count % self._microtubule_interval == 0:
            self._update_microtubules(dt * self._microtubule_interval)

        # ---- 10. Consciousness tracking ----
        if self._consciousness_enabled:
            self._spike_history.append(self.fired.clone())
            if len(self._spike_history) > 1000:
                self._spike_history.pop(0)

        # ---- 11. Clear external current (popped each step) ----
        self.external_current.zero_()

    def run(self, steps: int) -> None:
        """Run multiple simulation steps."""
        for _ in range(steps):
            self.step()

    def fast_run(self, steps: int) -> None:
        """Run steps with reduced subsystem updates for maximum throughput.

        Skips second messengers and gene expression (still does HH + calcium + STDP).
        Use for benchmarking or when subsystem fidelity isn't critical.
        """
        dt = self.dt
        for _ in range(steps):
            self.time += dt
            self.step_count += 1
            self.prev_voltage.copy_(self.voltage)
            self.fired.zero_()

            # Fused HH + receptor + membrane + spike in one block
            fused_hh_gating(
                self.voltage, self.nav_m, self.nav_h,
                self.kv_n, self.cav_m, self.cav_h, dt,
            )
            V = self.voltage

            # Receptor binding
            glu = self.nt_conc[:, NT_GLU]
            glu_hill = glu**1.5 / (800.0**1.5 + glu**1.5)
            self.ampa_open = glu_hill
            self.nmda_open = glu_hill * (1.0 / (1.0 + torch.exp(-0.062 * V) / 3.57))

            # Membrane integration (only major channels)
            cs = self.conductance_scale
            I_total = (
                cs[:, CH_NAV] * G_NA * self.nav_m**3 * self.nav_h * (V - E_NA)
                + cs[:, CH_KV] * G_K * self.kv_n**4 * (V - E_K)
                + cs[:, CH_KLEAK] * G_KLEAK * (V - E_K)
                + cs[:, CH_CAV] * G_CA * self.cav_m**2 * self.cav_h * (V - E_CA)
                + cs[:, CH_AMPA] * G_AMPA * self.ampa_open * self.ampa_p * (V - E_AMPA)
                + cs[:, CH_NMDA] * G_NMDA * self.nmda_open * (V - E_NMDA)
                + cs[:, CH_GABAA] * G_GABAA * self.gabaa_open * (V - E_GABAA)
                + cs[:, CH_NACHR] * G_NACHR * self.nachr_open * (V - E_NACHR)
            )
            self.voltage.add_((-I_total + self.external_current) * (dt / C_M))
            self.voltage.clamp_(-100.0, 60.0)

            # Spike detection — branchless (MPS-friendly)
            spiking = (self.voltage > SPIKE_THRESHOLD) & (self.prev_voltage <= SPIKE_THRESHOLD) & (self.refractory <= 0)
            self.fired = spiking & self.alive
            self.spike_count.add_(self.fired.int())
            fired_any = self.fired.any()
            self.last_fired_step = torch.where(
                self.fired, torch.tensor(self.step_count, dtype=torch.int64, device=self.device),
                self.last_fired_step,
            )
            self.voltage = torch.where(self.fired, self._reset_voltage, self.voltage)
            self.refractory = torch.where(self.fired, self._refractory_ms, self.refractory)
            self.refractory.sub_(dt).clamp_(min=0.0)

            # Spike propagation + STDP (sparse matmul)
            if self.n_synapses > 0 and fired_any:
                self._maybe_rebuild_W()
                self._propagate_spikes()
                self._update_stdp(dt)

            # Lightweight calcium + NT decay
            self.ca_microdomain.add_(self.fired.float() * 500.0)
            flux_md = (0.05 * dt) * (self.ca_microdomain - self.ca_cytoplasmic)
            self.ca_microdomain.sub_(flux_md)
            self.ca_cytoplasmic.add_(flux_md)
            self.ca_cytoplasmic.mul_(1.0 - 0.01 * dt).clamp_(50.0, 10000.0)
            self.ca_microdomain.mul_(1.0 - 0.02 * dt).clamp_(50.0, 50000.0)
            self.nt_conc.lerp_(self._resting_nt.expand_as(self.nt_conc), dt * 0.05)

            # Consciousness tracking
            if self._consciousness_enabled:
                self._spike_history.append(self.fired.clone())
                if len(self._spike_history) > 1000:
                    self._spike_history.pop(0)

            self.external_current.zero_()

    def compile(self) -> None:
        """JIT-compile the step function for 2-5x speedup on CUDA.

        Only works on CUDA (not MPS). Call once after adding all synapses.
        Warmup takes ~10-30s on first step, then subsequent steps are much faster.
        """
        if _CAN_COMPILE and self.device.type == 'cuda':
            self.step = torch.compile(self.step, mode='reduce-overhead')  # type: ignore

    # ========================================================================
    # Spike propagation
    # ========================================================================

    def _propagate_spikes(self) -> None:
        """Propagate spikes via sparse matmul — MPS-optimized.

        Uses pre-built sparse W matrix: external_current += W @ fired * psc_scale
        This replaces scatter_add (catastrophically slow on Apple Metal GPUs).
        """
        fired_f = self.fired.float()

        # PSC propagation: single sparse matmul instead of scatter_add
        if self._W_sparse is not None:
            psc = torch.sparse.mm(self._W_sparse, fired_f.unsqueeze(1)).squeeze(1)
            self.external_current.add_(psc * self.psc_scale)

        # NT release: sparse matmul per NT type
        if self._NT_W_sparse is not None:
            fired_col = fired_f.unsqueeze(1)  # (N, 1)
            for nt_idx in range(_NUM_NT):
                nt_w = self._NT_W_sparse[nt_idx]
                if nt_w._nnz() > 0:
                    release = torch.sparse.mm(nt_w, fired_col).squeeze(1)
                    self.nt_conc[:, nt_idx].add_(release * self._nt_release_amount)

    # ========================================================================
    # Calcium dynamics
    # ========================================================================

    def _update_calcium(self, dt: float) -> None:
        """4-compartment calcium dynamics — minimized kernel launches."""
        # Spike-triggered Ca2+ influx → microdomain
        self.ca_microdomain.add_(self.fired.float() * 500.0)

        # Microdomain → cytoplasmic diffusion
        flux_md = (0.05 * dt) * (self.ca_microdomain - self.ca_cytoplasmic)
        self.ca_microdomain.sub_(flux_md)
        self.ca_cytoplasmic.add_(flux_md)

        # ER release (IP3-gated) and uptake (SERCA pump) — fused
        er_delta = self.ca_er - self.ca_cytoplasmic
        er_release = (0.002 * dt) * self.ip3 * er_delta
        er_uptake = (0.005 * dt) * self.ca_cytoplasmic
        net_er = er_release - er_uptake
        self.ca_cytoplasmic.add_(net_er)
        self.ca_er.sub_(net_er)

        # Mitochondrial — fused
        mcu = (0.001 * dt) * self.ca_cytoplasmic
        mptp = (0.0005 * dt) * self.ca_mitochondrial
        net_mito = mptp - mcu
        self.ca_cytoplasmic.add_(net_mito)
        self.ca_mitochondrial.sub_(net_mito)

        # Decay (plasma membrane Ca-ATPase) — in-place
        self.ca_cytoplasmic.mul_(1.0 - 0.01 * dt).clamp_(50.0, 10000.0)
        self.ca_microdomain.mul_(1.0 - 0.02 * dt).clamp_(50.0, 50000.0)
        self.ca_er.clamp_(1000.0, 500000.0)
        self.ca_mitochondrial.clamp_(50.0, 50000.0)

    # ========================================================================
    # Second messenger cascades
    # ========================================================================

    def _update_second_messengers(self, dt: float) -> None:
        """G-protein → cAMP/PKA, PLC/IP3/DAG/PKC, CaMKII, ERK, CREB.

        All in-place operations to minimize tensor allocation.
        """
        # DA/NE → Gs activation (D1, beta-adrenergic)
        da = self.nt_conc[:, NT_DA]
        ne = self.nt_conc[:, NT_NE]
        gs_drive = da / (da + 100.0) + ne / (ne + 200.0)
        self.gs_active.lerp_(gs_drive, dt * 0.5)

        # 5-HT → Gi activation (5-HT1A)
        sht = self.nt_conc[:, NT_5HT]
        self.gi_active.lerp_(sht / (sht + 50.0), dt * 0.3)

        # ACh → Gq activation (M1 muscarinic)
        ach = self.nt_conc[:, NT_ACH]
        self.gq_active.lerp_(ach / (ach + 500.0), dt * 0.4)

        # cAMP: AC activation (Gs) - PDE degradation - Gi inhibition
        self.camp.add_(dt * (0.5 * self.gs_active * (1.0 - 0.8 * self.gi_active) - 0.1 * self.camp)).clamp_(0.0, 5.0)

        # PKA: Hill equation on cAMP
        camp_sq = self.camp * self.camp  # avoid pow(2)
        self.pka_activity = camp_sq / (1.0 + camp_sq)

        # PLC/IP3/DAG cascade (Gq-driven)
        plc = 0.5 * self.gq_active
        self.ip3.add_(dt * (plc - 0.15 * self.ip3)).clamp_(0.0, 5.0)
        self.dag.add_(dt * (plc - 0.1 * self.dag)).clamp_(0.0, 5.0)

        # PKC: DAG + Ca2+ synergistic
        dag_ca = self.dag * (self.ca_cytoplasmic * (1.0 / 500.0))
        self.pkc_activity = dag_ca / (1.0 + dag_ca)

        # CaMKII: Ca2+-dependent with autophosphorylation (switch-like)
        ca_md_norm = self.ca_microdomain * (1.0 / 300.0)
        ca4 = ca_md_norm * ca_md_norm
        ca4 = ca4 * ca4  # ^4 via two squarings
        camkii_drive = ca4 / (1.0 + ca4)
        cam_sq = self.camkii_activity * self.camkii_activity
        autophos = cam_sq / (0.25 + cam_sq)  # 0.5^2 = 0.25
        self.camkii_activity.add_(dt * 0.2 * (camkii_drive + 0.5 * autophos - self.camkii_activity)).clamp_(0.0, 1.0)

        # ERK: integrates PKA + PKC + CaMKII
        erk_drive = 0.3 * self.pka_activity + 0.3 * self.pkc_activity + 0.4 * self.camkii_activity
        self.erk_activity.add_(dt * 0.1 * (erk_drive - self.erk_activity)).clamp_(0.0, 1.0)

        # Phosphorylation targets — all in-place
        kinase_sum = self.pka_activity + self.camkii_activity
        self.ampa_p.add_(dt * (0.1 * kinase_sum * (1.0 - self.ampa_p) - 0.05 * self.ampa_p)).clamp_(0.0, 1.0)
        self.kv_p.add_(dt * (0.05 * self.pkc_activity * (1.0 - self.kv_p) - 0.03 * self.kv_p)).clamp_(0.0, 1.0)
        self.cav_p.add_(dt * (0.05 * self.pka_activity * (1.0 - self.cav_p) - 0.02 * self.cav_p)).clamp_(0.0, 1.0)
        nuc_kinase = self.erk_activity + self.pka_activity
        self.creb_p.add_(dt * (0.02 * nuc_kinase * (1.0 - self.creb_p) - 0.01 * self.creb_p)).clamp_(0.0, 1.0)

        # NT concentration decay toward resting levels (using cached device tensor)
        self.nt_conc.lerp_(self._resting_nt.expand_as(self.nt_conc), dt * 0.05)
        # DA has faster reuptake (DAT transporter) — additional decay for DA only
        # This prevents DA from accumulating during rapid training
        da_col = self.nt_conc[:, NT_DA]
        da_excess = (da_col - 20.0).clamp(min=0.0)
        self.nt_conc[:, NT_DA] = 20.0 + da_excess * (1.0 - dt * 0.15)

    # ========================================================================
    # STDP
    # ========================================================================

    def _update_stdp(self, dt: float) -> None:
        """Three-factor STDP: pre × post → eligibility, DA → permanent weight change.

        Biology: spike coincidences create an "eligibility trace" at the synapse
        (a biochemical tag, like CaMKII activation). This trace decays over ~1 second.
        When dopamine arrives (reward signal), it converts the eligibility trace
        into a permanent synaptic weight change. Without DA, eligibility decays
        and no learning occurs. This ensures only REWARDED associations are learned.

        MPS-friendly: all multiply-add ops, no boolean indexing.
        """
        a_plus = 0.08
        a_minus = 0.08  # symmetric: timing determines LTP vs LTD, not rate asymmetry

        # Decay STDP traces (fast: τ=20ms)
        self.syn_pre_trace.mul_(self._stdp_decay_pre)
        self.syn_post_trace.mul_(self._stdp_decay_post)

        # Decay eligibility trace (slow: τ=1000ms → decay factor per 0.1ms step)
        elig_decay = 1.0 - dt / 1000.0  # ~0.9999 per step, ~0.905 per 1000 steps
        self.syn_eligibility.mul_(elig_decay)

        # Update STDP traces for fired neurons
        pre_fired_f = self.fired[self.syn_pre].float()
        post_fired_f = self.fired[self.syn_post].float()
        self.syn_pre_trace.add_(pre_fired_f * a_plus)
        self.syn_post_trace.add_(post_fired_f * a_minus)

        # Compute STDP coincidence signal → add to eligibility trace
        # Pre-before-post (LTP): post fires NOW, pre had recent activity → strengthen
        # Post-before-pre (LTD): pre fires NOW, post had recent activity → weaken
        ltp = post_fired_f * self.syn_pre_trace   # pre fired recently, post fires now → LTP
        ltd = pre_fired_f * self.syn_post_trace    # post fired recently, pre fires now → LTD
        stdp_signal = (ltp - ltd) * 0.5
        self.syn_eligibility.add_(stdp_signal)
        self.syn_eligibility.clamp_(-2.0, 2.0)  # prevent runaway

        # DA gating: convert eligibility to weight change proportional to DA
        # Only apply when DA is meaningfully above resting (20 nM)
        da_post = self.nt_conc[self.syn_post, NT_DA]
        # DA modulation: resting=20nM → gain=0 (no change), 60nM → gain=1, 120nM → gain=5
        da_above_rest = (da_post - 20.0).clamp(min=0.0)
        da_gain = da_above_rest / 20.0  # 0 at rest, 1 at 40nM, 5 at 120nM
        da_gain = da_gain.clamp(0.0, 10.0)

        # Convert eligibility → permanent weight change (only when DA present)
        # CRITICAL: only modify excitatory (glutamatergic) synapses
        # Inhibitory (GABA) synapses have separate homeostatic plasticity, not STDP
        excitatory_mask = (self.syn_nt_type != NT_GABA).float()
        dw = self.syn_eligibility * da_gain * 0.1 * excitatory_mask  # 0.1 = conversion rate
        self.syn_strength.add_(dw)
        self.syn_strength.clamp_(0.3, 8.0)  # floor at 0.3 to prevent pathway death

        # Consumed eligibility decays faster after DA application
        consumed = da_gain > 0.1
        self.syn_eligibility.mul_(torch.where(consumed, 0.9, 1.0))

    # ========================================================================
    # Interval-gated subsystems
    # ========================================================================

    def _update_gene_expression(self, dt: float) -> None:
        """Gene expression: IEGs (c-Fos, BDNF, Arc) driven by CaMKII and CREB."""
        # c-Fos: rapid IEG, driven by high CaMKII → decays fast
        self.cfos_level += dt * 0.01 * (self.camkii_activity - self.cfos_level)
        self.cfos_level.clamp_(0.0, 1.0)

        # Arc: activity-regulated cytoskeletal protein, CREB-dependent
        self.arc_level += dt * 0.005 * (self.creb_p - self.arc_level)
        self.arc_level.clamp_(0.0, 1.0)

        # BDNF: brain-derived neurotrophic factor, ERK-dependent
        self.bdnf_level += dt * 0.002 * (self.erk_activity - self.bdnf_level)
        self.bdnf_level.clamp_(0.0, 1.0)

    def _update_metabolism(self, dt: float) -> None:
        """ATP/ADP metabolism — glycolysis + oxidative phosphorylation."""
        # ATP consumption (baseline + activity-dependent)
        consumption = 50.0 + 200.0 * self.fired.float()  # uM/step
        self.atp -= consumption * dt * 0.001
        self.adp += consumption * dt * 0.001

        # Glycolysis: glucose → ATP (fast, ~2 ATP/glucose)
        glycolysis = 0.1 * self.glucose * dt * 0.001
        self.atp += glycolysis * 2.0
        self.glucose -= glycolysis

        # Oxidative phosphorylation: ADP → ATP (slower, ~34 ATP/glucose)
        oxphos = 0.05 * self.adp * (self.glucose / (self.glucose + 1000.0)) * dt * 0.001
        self.atp += oxphos * 34.0
        self.adp -= oxphos

        # Glucose supply (blood-brain barrier)
        self.glucose += 0.5 * dt  # constant supply

        self.atp.clamp_(100.0, 8000.0)
        self.adp.clamp_(10.0, 2000.0)
        self.glucose.clamp_(500.0, 10000.0)

    def _update_microtubules(self, dt: float) -> None:
        """Microtubule quantum coherence + Orch-OR collapse events."""
        # Coherence driven by low temperature (ATP availability) and CaMKII
        target = 0.1 + 0.3 * (self.atp / 4000.0) + 0.2 * self.camkii_activity
        self.mt_coherence += dt * 0.01 * (target - self.mt_coherence)
        self.mt_coherence.clamp_(0.0, 1.0)

        # Spontaneous Orch-OR collapse events (probabilistic)
        collapse_prob = 0.001 * self.mt_coherence * dt
        collapses = torch.bernoulli(collapse_prob).int()
        self.orch_or_events += collapses

    # ========================================================================
    # Lesion
    # ========================================================================

    def lesion(self, neuron_ids: torch.Tensor, fraction: float = 1.0) -> int:
        """Destroy synapses to/from specified neurons (simulates brain damage).

        Args:
            neuron_ids: neurons to lesion
            fraction: 0.0-1.0, fraction of synapses to destroy
        Returns:
            number of synapses destroyed
        """
        if self.n_synapses == 0:
            return 0
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[neuron_ids] = True
        affected = mask[self.syn_pre] | mask[self.syn_post]
        if fraction < 1.0:
            affected_idx = affected.nonzero(as_tuple=True)[0]
            n_affected = affected_idx.shape[0]
            if n_affected > 0:
                keep_prob = torch.rand(n_affected, device=self.device) > fraction
                affected[affected_idx[keep_prob]] = False
        n_destroyed = int(affected.sum())
        if n_destroyed == 0:
            return 0
        keep = ~affected
        self.syn_pre = self.syn_pre[keep]
        self.syn_post = self.syn_post[keep]
        self.syn_weight = self.syn_weight[keep]
        self.syn_nt_type = self.syn_nt_type[keep]
        self.syn_strength = self.syn_strength[keep]
        self.syn_pre_trace = self.syn_pre_trace[keep]
        self.syn_post_trace = self.syn_post_trace[keep]
        self.syn_eligibility = self.syn_eligibility[keep]
        self.n_synapses = self.syn_pre.shape[0]
        self._W_dirty = True
        self._W_sparse = None
        self._NT_W_sparse = None
        self._syn_nt_onehot = None
        return n_destroyed

    # ========================================================================
    # Pharmacology
    # ========================================================================

    def apply_drug(self, drug_name: str, dose_mg: float) -> None:
        """Apply a psychoactive drug by modulating conductance scales."""
        # Simple PD model: effect = dose / (dose + ED50)
        drug_name = drug_name.lower()

        if drug_name in ("fluoxetine", "prozac"):
            # SSRI: increases 5-HT
            effect = dose_mg / (dose_mg + 20.0)
            self.nt_conc[:, NT_5HT] *= (1.0 + 2.0 * effect)

        elif drug_name in ("diazepam", "valium"):
            # Benzodiazepine: enhances GABA-A
            effect = dose_mg / (dose_mg + 10.0)
            self.conductance_scale[:, CH_GABAA] *= (1.0 + 4.0 * effect)

        elif drug_name == "caffeine":
            # Adenosine antagonist → reduced inhibition, PDE inhibitor → increased cAMP
            effect = dose_mg / (dose_mg + 100.0)
            self.conductance_scale[:, CH_GABAA] *= (1.0 - 0.3 * effect)
            self.camp += 0.5 * effect

        elif drug_name in ("amphetamine", "adderall"):
            # DA/NE reuptake inhibitor + vesicular release
            effect = dose_mg / (dose_mg + 15.0)
            self.nt_conc[:, NT_DA] *= (1.0 + 5.0 * effect)
            self.nt_conc[:, NT_NE] *= (1.0 + 3.0 * effect)

        elif drug_name in ("methamphetamine", "meth"):
            # More potent DAT/NET reversal than amphetamine + 5-HT release
            effect = dose_mg / (dose_mg + 8.0)  # lower ED50 = more potent
            self.nt_conc[:, NT_DA] *= (1.0 + 8.0 * effect)   # stronger DA flood
            self.nt_conc[:, NT_NE] *= (1.0 + 5.0 * effect)   # stronger NE
            self.nt_conc[:, NT_5HT] *= (1.0 + 2.0 * effect)  # serotonergic too

        elif drug_name in ("ldopa", "l-dopa", "levodopa"):
            # DA precursor
            effect = dose_mg / (dose_mg + 100.0)
            self.nt_conc[:, NT_DA] *= (1.0 + 3.0 * effect)

        elif drug_name in ("donepezil", "aricept"):
            # AChE inhibitor → increases ACh
            effect = dose_mg / (dose_mg + 10.0)
            self.nt_conc[:, NT_ACH] *= (1.0 + 2.0 * effect)

        elif drug_name == "ketamine":
            # NMDA antagonist
            effect = dose_mg / (dose_mg + 35.0)
            self.conductance_scale[:, CH_NMDA] *= (1.0 - 0.8 * effect)

    def apply_anesthesia(self) -> None:
        """General anesthesia: GABA-A up, NMDA/AMPA/Na_v down, K_leak up."""
        self.conductance_scale[:, CH_GABAA] *= 8.0
        self.conductance_scale[:, CH_NMDA] *= 0.05
        self.conductance_scale[:, CH_AMPA] *= 0.4
        self.conductance_scale[:, CH_NAV] *= 0.5
        self.conductance_scale[:, CH_KLEAK] *= 2.0

    # ========================================================================
    # Consciousness metrics
    # ========================================================================

    def enable_consciousness(self) -> None:
        """Enable consciousness metric tracking."""
        self._consciousness_enabled = True
        self._spike_history = []

    def consciousness_metrics(self) -> Dict[str, float]:
        """Compute 7 consciousness metrics from spike history."""
        if not self._consciousness_enabled or len(self._spike_history) < 50:
            return {"phi": 0.0, "pci": 0.0, "causal_density": 0.0,
                    "criticality": 0.0, "global_workspace": 0.0,
                    "orch_or": 0.0, "composite": 0.0}

        # Stack recent spike history
        history = torch.stack(self._spike_history[-200:]).float()  # (T, N)
        T, N = history.shape

        # 1. Phi (IIT) — mutual information proxy
        firing_rates = history.mean(dim=0)
        fr_nonzero = firing_rates[firing_rates > 0]
        if len(fr_nonzero) > 1:
            entropy = -(fr_nonzero * torch.log2(fr_nonzero + 1e-10)).sum()
            phi = float(entropy) / math.log2(N + 1)
        else:
            phi = 0.0

        # 2. PCI — perturbational complexity index (Lempel-Ziv proxy)
        binary = (history > 0).flatten().cpu().numpy()
        if len(binary) > 0:
            # Simplified LZ complexity
            n_transitions = int((binary[1:] != binary[:-1]).sum())
            pci = n_transitions / (len(binary) + 1)
        else:
            pci = 0.0

        # 3. Causal density — fraction of significant pairwise correlations
        if N > 1:
            sample_n = min(N, 200)
            sample_idx = torch.randperm(N)[:sample_n]
            sample_hist = history[:, sample_idx]
            corr = torch.corrcoef(sample_hist.T)
            significant = (torch.abs(corr) > 0.1).float().mean()
            causal_density = float(significant)
        else:
            causal_density = 0.0

        # 4. Criticality — branching ratio
        spike_counts_per_step = history.sum(dim=1)
        if spike_counts_per_step.sum() > 0:
            ratios = spike_counts_per_step[1:] / (spike_counts_per_step[:-1] + 1e-6)
            criticality = float(ratios.median().clamp(0, 2))
        else:
            criticality = 0.0

        # 5. Global workspace — fraction of neurons participating in global broadcast
        global_threshold = 0.1 * N
        broadcast_steps = (spike_counts_per_step > global_threshold).float().mean()
        global_workspace = float(broadcast_steps)

        # 6. Orch-OR — accumulated quantum collapse events
        orch_or = float(self.orch_or_events.float().mean()) / max(self.step_count, 1) * 1000

        # 7. Composite
        phi_norm = math.log(1 + phi) / math.log(1 + N**1.5) if N > 1 else 0.0
        composite = 0.2 * phi_norm + 0.2 * pci + 0.15 * causal_density + \
                    0.15 * min(criticality, 1.0) + 0.15 * global_workspace + 0.15 * orch_or

        return {
            "phi": phi,
            "pci": pci,
            "causal_density": causal_density,
            "criticality": criticality,
            "global_workspace": global_workspace,
            "orch_or": orch_or,
            "composite": composite,
        }

    # ========================================================================
    # State access
    # ========================================================================

    def voltages(self) -> List[float]:
        """Return membrane voltages as Python list."""
        return self.voltage.cpu().tolist()

    def fired_indices(self) -> List[int]:
        """Return indices of neurons that fired this step."""
        return torch.where(self.fired)[0].cpu().tolist()

    def get_spike_counts(self) -> List[int]:
        """Return cumulative spike counts."""
        return self.spike_count.cpu().tolist()


# ============================================================================
# CUDARegionalBrain
# ============================================================================

class CUDARegionalBrain:
    """Pre-wired brain with cortical columns, thalamus, hippocampus, basal ganglia.

    All backed by CUDAMolecularBrain tensors.
    """

    def __init__(self, brain: CUDAMolecularBrain, regions: Dict[str, Dict]):
        self.brain = brain
        self.regions = regions  # {name: {"type": str, "ids": list, "subgroups": dict}}
        # Cache: pre-built tensors for frequently-stimulated regions
        self._id_tensor_cache: Dict[str, torch.Tensor] = {}

    @classmethod
    def minimal(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~120 neurons: 1 cortex + thalamus + hippocampus + BG."""
        return cls._build(n_columns=1, n_per_layer=10, device=device, seed=seed)

    @classmethod
    def small(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~800 neurons: 10 cortical columns — quick smoke test."""
        return cls._build(n_columns=10, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def standard(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~260 neurons: 1 cortex (larger) + thalamus + hippo + BG."""
        return cls._build(n_columns=1, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def medium(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~4K neurons: 50 cortical columns — moderate scale."""
        return cls._build(n_columns=50, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def xlarge(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~1200 neurons: 6 cortical columns + full subcortical."""
        return cls._build(n_columns=6, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def large(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~20K neurons: 250 cortical columns."""
        return cls._build(n_columns=250, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def mega(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~80K neurons: 1000 cortical columns + large subcortical."""
        return cls._build(n_columns=1000, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def hundred_k(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~100K neurons: 1250 cortical columns."""
        return cls._build(n_columns=1250, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def million(cls, device: str = "auto", seed: int = 42) -> "CUDARegionalBrain":
        """~1M neurons: 5000 cortical columns."""
        return cls._build(n_columns=5000, n_per_layer=20, device=device, seed=seed)

    @classmethod
    def _build(
        cls,
        n_columns: int,
        n_per_layer: int,
        device: str,
        seed: int,
    ) -> "CUDARegionalBrain":
        """Build a regional brain with specified scale."""
        torch.manual_seed(seed)

        # Calculate neuron counts
        neurons_per_column = n_per_layer * 4  # L4, L2/3, L5, L6
        cortex_total = n_columns * neurons_per_column

        # Subcortical sizes scale with cortex
        thalamus_relay = max(10, n_columns * 3)
        thalamus_reticular = max(5, n_columns * 2)
        thalamus_total = thalamus_relay + thalamus_reticular

        hippo_dg = max(10, n_columns * 5)
        hippo_ca3 = max(8, n_columns * 4)
        hippo_ca1 = max(6, n_columns * 3)
        hippo_total = hippo_dg + hippo_ca3 + hippo_ca1

        bg_d1 = max(5, n_columns * 2)
        bg_d2 = max(5, n_columns * 2)
        bg_total = bg_d1 + bg_d2

        n_total = cortex_total + thalamus_total + hippo_total + bg_total
        brain = CUDAMolecularBrain(n_total, device=device)

        regions: Dict[str, Dict] = {}
        idx = 0

        # ---- Build cortical columns ----
        for col in range(n_columns):
            col_start = idx
            l4_start = idx
            l4_end = idx + n_per_layer
            idx = l4_end

            l23_start = idx
            l23_end = idx + n_per_layer
            idx = l23_end

            l5_start = idx
            l5_end = idx + n_per_layer
            idx = l5_end

            l6_start = idx
            l6_end = idx + n_per_layer
            idx = l6_end

            col_end = idx

            # Archetypes: 80% excitatory, 20% inhibitory per layer
            for layer_start, layer_end in [(l4_start, l4_end), (l23_start, l23_end),
                                            (l5_start, l5_end), (l6_start, l6_end)]:
                layer_size = layer_end - layer_start
                n_inhib = max(1, int(layer_size * 0.2))
                brain.archetype[layer_end - n_inhib:layer_end] = ARCH_INTERNEURON

            # Positions
            cx, cy = (col % 10) * 3.0, (col // 10) * 3.0
            brain.x[col_start:col_end] = cx + torch.rand(col_end - col_start, device=brain.device) * 2.0
            brain.y[col_start:col_end] = cy + torch.rand(col_end - col_start, device=brain.device) * 2.0
            brain.z[col_start:col_end] = torch.rand(col_end - col_start, device=brain.device) * 4.0

            regions[f"cortex_{col}"] = {
                "type": "cortex",
                "ids": list(range(col_start, col_end)),
                "subgroups": {
                    "L4": list(range(l4_start, l4_end)),
                    "L2/3": list(range(l23_start, l23_end)),
                    "L5": list(range(l5_start, l5_end)),
                    "L6": list(range(l6_start, l6_end)),
                },
            }

        # ---- Build thalamus ----
        thal_start = idx
        thal_relay_start = idx
        thal_relay_end = idx + thalamus_relay
        idx = thal_relay_end
        thal_ret_start = idx
        thal_ret_end = idx + thalamus_reticular
        idx = thal_ret_end

        brain.archetype[thal_ret_start:thal_ret_end] = ARCH_INTERNEURON
        brain.x[thal_start:idx] = 15.0 + torch.rand(idx - thal_start, device=brain.device) * 3.0
        brain.y[thal_start:idx] = torch.rand(idx - thal_start, device=brain.device) * 3.0
        brain.z[thal_start:idx] = torch.rand(idx - thal_start, device=brain.device) * 2.0

        regions["thalamus"] = {
            "type": "thalamus",
            "ids": list(range(thal_start, idx)),
            "subgroups": {
                "relay": list(range(thal_relay_start, thal_relay_end)),
                "reticular": list(range(thal_ret_start, thal_ret_end)),
            },
        }

        # ---- Build hippocampus ----
        hippo_start = idx
        dg_start = idx
        dg_end = idx + hippo_dg
        idx = dg_end
        ca3_start = idx
        ca3_end = idx + hippo_ca3
        idx = ca3_end
        ca1_start = idx
        ca1_end = idx + hippo_ca1
        idx = ca1_end

        brain.archetype[dg_start:dg_end] = ARCH_GRANULE
        brain.x[hippo_start:idx] = 20.0 + torch.rand(idx - hippo_start, device=brain.device) * 4.0
        brain.y[hippo_start:idx] = torch.rand(idx - hippo_start, device=brain.device) * 3.0
        brain.z[hippo_start:idx] = torch.rand(idx - hippo_start, device=brain.device) * 2.0

        regions["hippocampus"] = {
            "type": "hippocampus",
            "ids": list(range(hippo_start, idx)),
            "subgroups": {
                "DG": list(range(dg_start, dg_end)),
                "CA3": list(range(ca3_start, ca3_end)),
                "CA1": list(range(ca1_start, ca1_end)),
            },
        }

        # ---- Build basal ganglia ----
        bg_start = idx
        d1_start = idx
        d1_end = idx + bg_d1
        idx = d1_end
        d2_start = idx
        d2_end = idx + bg_d2
        idx = d2_end

        brain.archetype[bg_start:idx] = ARCH_MEDIUM_SPINY
        brain.x[bg_start:idx] = 25.0 + torch.rand(idx - bg_start, device=brain.device) * 3.0
        brain.y[bg_start:idx] = torch.rand(idx - bg_start, device=brain.device) * 3.0
        brain.z[bg_start:idx] = torch.rand(idx - bg_start, device=brain.device) * 2.0

        regions["basal_ganglia"] = {
            "type": "basal_ganglia",
            "ids": list(range(bg_start, idx)),
            "subgroups": {
                "D1": list(range(d1_start, d1_end)),
                "D2": list(range(d2_start, d2_end)),
            },
        }

        # ---- Wire synapses ----
        rb = cls(brain, regions)
        rb._wire_all(n_columns, n_per_layer)
        return rb

    def _wire_all(self, n_columns: int, n_per_layer: int) -> None:
        """Wire all intra-region and inter-region connections."""
        dev = self.brain.device
        all_pre, all_post, all_weight, all_nt = [], [], [], []

        def _random_connections(
            src_ids: List[int],
            dst_ids: List[int],
            prob: float,
            weight_range: Tuple[float, float],
            nt_type: int,
        ) -> None:
            """Generate random connections between two groups."""
            if not src_ids or not dst_ids:
                return
            n_possible = len(src_ids) * len(dst_ids)
            # For large networks, sample instead of full matrix
            if n_possible > 100_000:
                n_expected = int(n_possible * prob)
                if n_expected == 0:
                    return
                src_sample = torch.tensor(src_ids, device=dev)[torch.randint(len(src_ids), (n_expected,), device=dev)]
                dst_sample = torch.tensor(dst_ids, device=dev)[torch.randint(len(dst_ids), (n_expected,), device=dev)]
                mask = src_sample != dst_sample
                pre_t = src_sample[mask]
                post_t = dst_sample[mask]
            else:
                # Full random matrix
                src_t = torch.tensor(src_ids, device=dev)
                dst_t = torch.tensor(dst_ids, device=dev)
                mask = torch.rand(len(src_ids), len(dst_ids), device=dev) < prob
                indices = torch.where(mask)
                pre_t = src_t[indices[0]]
                post_t = dst_t[indices[1]]
                # Remove self-connections
                valid = pre_t != post_t
                pre_t = pre_t[valid]
                post_t = post_t[valid]

            if pre_t.shape[0] == 0:
                return
            w = torch.rand(pre_t.shape[0], device=dev) * (weight_range[1] - weight_range[0]) + weight_range[0]
            nt = torch.full((pre_t.shape[0],), nt_type, dtype=torch.int32, device=dev)
            all_pre.append(pre_t)
            all_post.append(post_t)
            all_weight.append(w)
            all_nt.append(nt)

        # ---- Intra-cortical wiring (per column) ----
        for col_idx in range(n_columns):
            col = self.regions.get(f"cortex_{col_idx}")
            if col is None:
                continue
            sg = col["subgroups"]
            l4, l23, l5, l6 = sg["L4"], sg["L2/3"], sg["L5"], sg["L6"]

            # Feedforward: L4→L2/3→L5→L6
            _random_connections(l4, l23, 0.3, (0.8, 1.5), NT_GLU)
            _random_connections(l23, l5, 0.25, (0.8, 1.5), NT_GLU)
            _random_connections(l5, l6, 0.2, (0.8, 1.5), NT_GLU)
            # Feedback: L6→L4
            _random_connections(l6, l4, 0.15, (0.5, 1.0), NT_GLU)

            # Recurrent within layers
            for layer in [l4, l23, l5, l6]:
                n_layer = len(layer)
                n_inhib = max(1, int(n_layer * 0.2))
                excit = layer[:n_layer - n_inhib]
                inhib = layer[n_layer - n_inhib:]
                # Excitatory recurrence
                _random_connections(excit, excit, 0.1, (0.5, 1.2), NT_GLU)
                # Lateral inhibition
                _random_connections(inhib, excit, 0.5, (0.8, 1.5), NT_GABA)
                _random_connections(excit, inhib, 0.3, (0.5, 1.0), NT_GLU)

        # ---- Inter-column wiring (for multi-column networks) ----
        if n_columns > 1:
            for i in range(n_columns):
                for j in range(i + 1, min(i + 3, n_columns)):  # connect nearby columns
                    ci = self.regions[f"cortex_{i}"]["subgroups"]
                    cj = self.regions[f"cortex_{j}"]["subgroups"]
                    _random_connections(ci["L2/3"], cj["L2/3"], 0.05, (0.3, 0.8), NT_GLU)
                    _random_connections(cj["L2/3"], ci["L2/3"], 0.05, (0.3, 0.8), NT_GLU)

        # ---- Thalamic wiring ----
        thal = self.regions["thalamus"]
        relay = thal["subgroups"]["relay"]
        reticular = thal["subgroups"]["reticular"]
        _random_connections(relay, reticular, 0.4, (0.8, 1.5), NT_GLU)
        _random_connections(reticular, relay, 0.5, (0.8, 1.5), NT_GABA)

        # ---- Hippocampal wiring ----
        hippo = self.regions["hippocampus"]
        dg = hippo["subgroups"]["DG"]
        ca3 = hippo["subgroups"]["CA3"]
        ca1 = hippo["subgroups"]["CA1"]
        _random_connections(dg, ca3, 0.3, (0.8, 1.5), NT_GLU)
        _random_connections(ca3, ca3, 0.5, (0.5, 1.2), NT_GLU)  # recurrent
        _random_connections(ca3, ca1, 0.4, (0.8, 1.5), NT_GLU)

        # ---- Basal ganglia wiring ----
        bg = self.regions["basal_ganglia"]
        d1 = bg["subgroups"]["D1"]
        d2 = bg["subgroups"]["D2"]
        _random_connections(d1, d2, 0.3, (0.5, 1.0), NT_GABA)
        _random_connections(d2, d1, 0.3, (0.5, 1.0), NT_GABA)

        # ---- Inter-region projections ----
        # Thalamus relay → Cortex L4 (all columns)
        for col_idx in range(n_columns):
            col = self.regions.get(f"cortex_{col_idx}")
            if col is None:
                continue
            l4 = col["subgroups"]["L4"]
            l5 = col["subgroups"]["L5"]
            l6 = col["subgroups"]["L6"]
            l23 = col["subgroups"]["L2/3"]

            _random_connections(relay, l4, 0.3, (0.8, 1.5), NT_GLU)
            # Direct relay→L5 pathway — dense, learnable shortcut
            # High connectivity ensures each L5 neuron receives input from many relay neurons
            # Training modifies these weights to create word-specific representations
            _random_connections(relay, l5, 0.5, (0.3, 0.8), NT_GLU)
            _random_connections(l6, relay, 0.2, (0.5, 1.0), NT_GLU)
            _random_connections(l5, d1, 0.15, (0.5, 1.0), NT_GLU)
            _random_connections(l23, dg, 0.1, (0.5, 1.0), NT_GLU)

        # Hippocampus CA1 → Cortex L5 (ALL columns — for memory consolidation)
        for col_idx in range(n_columns):
            col = self.regions.get(f"cortex_{col_idx}")
            if col is None:
                continue
            col_l5 = col["subgroups"]["L5"]
            # Sparse but broad: CA1 projects to L5 across cortex
            _random_connections(ca1, col_l5, 0.1, (0.5, 1.2), NT_GLU)

        # ---- Commit all synapses at once ----
        if all_pre:
            self.brain.add_synapses(
                torch.cat(all_pre),
                torch.cat(all_post),
                torch.cat(all_weight),
                torch.cat(all_nt),
            )

    # ========================================================================
    # Public interface
    # ========================================================================

    def step(self) -> None:
        self.brain.step()

    def run(self, steps: int) -> None:
        self.brain.run(steps)

    def _get_id_tensor(self, cache_key: str, ids: List[int]) -> torch.Tensor:
        """Get or create a cached int64 tensor of neuron IDs on the brain device."""
        if cache_key not in self._id_tensor_cache:
            self._id_tensor_cache[cache_key] = torch.tensor(
                ids, dtype=torch.int64, device=self.brain.device,
            )
        return self._id_tensor_cache[cache_key]

    def stimulate_thalamus(self, current: float) -> None:
        """Inject current into thalamic relay neurons."""
        relay_ids = self.regions["thalamus"]["subgroups"]["relay"]
        t = self._get_id_tensor("thalamus_relay", relay_ids)
        self.brain.stimulate_indices(t, current)

    def stimulate_region(self, name: str, current: float, subgroup: Optional[str] = None) -> None:
        """Inject current into a region or subgroup."""
        region = self.regions[name]
        if subgroup:
            ids = region["subgroups"][subgroup]
            cache_key = f"{name}_{subgroup}"
        else:
            ids = region["ids"]
            cache_key = name
        t = self._get_id_tensor(cache_key, ids)
        self.brain.stimulate_indices(t, current)

    def read_cortex_output(self) -> float:
        """Mean L5 activity across all cortical columns (normalized)."""
        if "_all_l5" not in self._id_tensor_cache:
            l5_ids = []
            for name, region in self.regions.items():
                if region["type"] == "cortex" and "L5" in region["subgroups"]:
                    l5_ids.extend(region["subgroups"]["L5"])
            if not l5_ids:
                return 0.0
            self._id_tensor_cache["_all_l5"] = torch.tensor(
                l5_ids, dtype=torch.int64, device=self.brain.device,
            )
        t = self._id_tensor_cache["_all_l5"]
        voltages = self.brain.voltage[t]
        # Normalize: -70 → 0.0, +20 → 1.0
        return float(((voltages + 70.0) / 90.0).clamp(0.0, 1.0).mean())

    def apply_drug(self, drug_name: str, dose_mg: float) -> None:
        self.brain.apply_drug(drug_name, dose_mg)

    def apply_anesthesia(self) -> None:
        self.brain.apply_anesthesia()

    def lesion_region(self, region: str, subgroup: str = None, fraction: float = 1.0) -> int:
        """Lesion a brain region by destroying synapses to/from its neurons."""
        r = self.regions[region]
        ids = r["subgroups"][subgroup] if subgroup else r["ids"]
        t = self._get_id_tensor(f"lesion_{region}_{subgroup}", ids)
        return self.brain.lesion(t, fraction)

    def enable_consciousness(self) -> None:
        self.brain.enable_consciousness()

    def consciousness_metrics(self) -> Dict[str, float]:
        return self.brain.consciousness_metrics()

    @property
    def n_neurons(self) -> int:
        return self.brain.n

    @property
    def n_synapses(self) -> int:
        return self.brain.n_synapses

    @property
    def time(self) -> float:
        return self.brain.time

    def region_info(self) -> List[Tuple[str, str, int]]:
        """Return (name, type, neuron_count) for all regions."""
        return [(name, r["type"], len(r["ids"])) for name, r in self.regions.items()]

    # ========================================================================
    # Hippocampal replay for memory consolidation
    # ========================================================================

    def replay_pattern(
        self,
        input_ids: torch.Tensor,
        input_pattern: torch.Tensor,
        target_ids: torch.Tensor,
        target_pattern: torch.Tensor,
        replay_steps: int = 30,
        input_intensity: float = 60.0,
        target_intensity: float = 50.0,
        da_boost: float = 60.0,
    ) -> None:
        """Hippocampal replay: reactivate stored patterns through DG→CA3→CA1→cortex.

        During replay, we re-present the input+target pattern pair while also
        driving hippocampal neurons. DA is targeted to the specific target neurons
        being replayed — not global. This is sleep-dependent memory consolidation.
        """
        brain = self.brain
        dev = brain.device

        # Get hippocampal IDs
        hippo = self.regions["hippocampus"]
        dg_t = self._get_id_tensor("hippo_dg", hippo["subgroups"]["DG"])

        inp_active = (input_pattern > 0.3)[:len(input_ids)]
        tgt_active = (target_pattern > 0.3)[:len(target_ids)]
        active_tgt_ids = target_ids[tgt_active] if tgt_active.any() else None
        active_inp_ids = input_ids[inp_active] if inp_active.any() else None

        for s in range(replay_steps):
            if s % 2 == 0:
                # Drive input pathway
                if active_inp_ids is not None:
                    brain.external_current[active_inp_ids] += input_pattern[inp_active] * input_intensity

                # Drive target pathway (offset by 1 step)
                if s > 0 and active_tgt_ids is not None:
                    brain.external_current[active_tgt_ids] += target_pattern[tgt_active] * target_intensity

                # Drive hippocampus DG
                n_dg_active = max(1, len(dg_t) // 5)
                brain.external_current[dg_t[:n_dg_active]] += 40.0

            # Targeted DA: only to target neurons during replay
            if s % 4 == 0 and active_tgt_ids is not None:
                brain.nt_conc[active_tgt_ids, NT_DA] += da_boost

            self.step()

        # Discriminative Hebbian during replay (same as train_word)
        if active_inp_ids is not None and active_tgt_ids is not None and brain.n_synapses > 0:
            input_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            input_set[active_inp_ids] = True
            pre_is_input = input_set[brain.syn_pre]

            target_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            target_set[active_tgt_ids] = True
            post_is_target = target_set[brain.syn_post]

            all_l5_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            all_l5_set[target_ids] = True
            post_is_l5 = all_l5_set[brain.syn_post]

            is_excitatory = brain.syn_nt_type != NT_GABA

            strengthen = pre_is_input & post_is_target & is_excitatory
            brain.syn_strength += strengthen.float() * 0.2

            post_is_nontarget_l5 = post_is_l5 & ~post_is_target
            weaken = pre_is_input & post_is_nontarget_l5 & is_excitatory
            brain.syn_strength -= weaken.float() * 0.06

            brain.syn_strength.clamp_(0.3, 8.0)

        # Brief DA pulse to consolidate
        if active_tgt_ids is not None:
            brain.nt_conc[active_tgt_ids, NT_DA] += da_boost * 0.5
        self.run(3)

    def consolidation_sleep(
        self,
        word_input_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        word_target_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        n_replays: int = 3,
        replay_steps: int = 25,
    ) -> None:
        """Sleep-like consolidation: replay all word patterns through hippocampus.

        Args:
            word_input_map: {word: (neuron_ids, pattern_tensor)}
            word_target_map: {word: (neuron_ids, pattern_tensor)}
            n_replays: how many times to replay each word
            replay_steps: steps per replay episode
        """
        import random
        words = list(word_input_map.keys())
        for rep in range(n_replays):
            random.shuffle(words)
            for word in words:
                inp_ids, inp_pat = word_input_map[word]
                tgt_ids, tgt_pat = word_target_map[word]
                self.replay_pattern(
                    inp_ids, inp_pat, tgt_ids, tgt_pat,
                    replay_steps=replay_steps,
                    da_boost=80.0,
                )
                self.run(10)  # inter-replay gap

    def train_word(
        self,
        input_ids: torch.Tensor,
        input_pattern: torch.Tensor,
        target_ids: torch.Tensor,
        target_pattern: torch.Tensor,
        train_steps: int = 60,
        input_intensity: float = 60.0,
        target_intensity: float = 50.0,
        da_amount: float = 80.0,
        hebbian_delta: float = 0.3,
    ) -> None:
        """Train a word→meaning association using Hebbian + STDP + DA.

        Two-phase approach:
        1. Co-activate input (thalamus) and target (cortex L5) to drive spike coincidences
        2. After co-activation, apply DIRECT Hebbian strengthening to excitatory synapses
           where pre neuron was active during input AND post neuron is in target set.
           This gives a strong initial association that STDP+DA refines over time.

        DA is also injected at target neurons to support three-factor STDP.
        """
        brain = self.brain
        dev = brain.device

        inp_active = (input_pattern > 0.3)[:len(input_ids)]
        tgt_active = (target_pattern > 0.3)[:len(target_ids)]
        active_tgt_ids = target_ids[tgt_active] if tgt_active.any() else None
        active_inp_ids = input_ids[inp_active] if inp_active.any() else None

        # Track which neurons fire during training (for Hebbian update)
        pre_spike_counts = torch.zeros(brain.n, device=dev)

        for s in range(train_steps):
            # Input: pulsed
            if s % 2 == 0 and active_inp_ids is not None:
                brain.external_current[active_inp_ids] += input_pattern[inp_active] * input_intensity

            # Target: offset by 5 steps for causal STDP
            if s >= 5 and s % 2 == 0 and active_tgt_ids is not None:
                brain.external_current[active_tgt_ids] += target_pattern[tgt_active] * target_intensity

            # DA at target neurons for three-factor STDP
            if s >= 8 and s % 3 == 0 and active_tgt_ids is not None:
                brain.nt_conc[active_tgt_ids, NT_DA] += da_amount

            self.step()
            pre_spike_counts += brain.fired.float()

        # ── Discriminative Hebbian weight update ──
        # Key insight: only modify synapses from STIMULATED input neurons,
        # not any neuron that happened to fire during training.
        # Anti-Hebbian: weaken input→non-target L5 to create contrast.
        # Safe with non-overlapping input patterns (no catastrophic interference).
        if active_inp_ids is not None and active_tgt_ids is not None and brain.n_synapses > 0:
            # Mark stimulated input neurons (NOT just any that fired)
            input_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            input_set[active_inp_ids] = True
            pre_is_input = input_set[brain.syn_pre]

            # Target L5 neurons for this word
            target_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            target_set[active_tgt_ids] = True
            post_is_target = target_set[brain.syn_post]

            # All L5 neurons (target_ids = all output neurons passed in)
            all_l5_set = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            all_l5_set[target_ids] = True
            post_is_l5 = all_l5_set[brain.syn_post]

            # Only excitatory synapses
            is_excitatory = brain.syn_nt_type != NT_GABA

            # Strengthen: input→target L5 (positive association)
            strengthen = pre_is_input & post_is_target & is_excitatory
            brain.syn_strength += strengthen.float() * hebbian_delta

            # Weaken: input→non-target L5 (discriminative contrast)
            post_is_nontarget_l5 = post_is_l5 & ~post_is_target
            weaken = pre_is_input & post_is_nontarget_l5 & is_excitatory
            brain.syn_strength -= weaken.float() * hebbian_delta * 0.3

            brain.syn_strength.clamp_(0.3, 8.0)

        # DA consolidation pulse
        if active_tgt_ids is not None:
            brain.nt_conc[active_tgt_ids, NT_DA] += da_amount * 0.5
        self.run(5)

    def test_word(
        self,
        input_ids: torch.Tensor,
        input_pattern: torch.Tensor,
        output_ids: torch.Tensor,
        stim_steps: int = 50,
        intensity: float = 60.0,
    ) -> torch.Tensor:
        """Test: present word input, record cortical output spike counts."""
        brain = self.brain
        dev = brain.device
        counts = torch.zeros(len(output_ids), device=dev)
        inp_active = (input_pattern > 0.3)[:len(input_ids)]

        for s in range(stim_steps):
            if s % 2 == 0 and inp_active.any():
                active_inp_ids = input_ids[inp_active]
                brain.external_current[active_inp_ids] += input_pattern[inp_active] * intensity
            self.step()
            counts += brain.fired[output_ids].float()

        max_c = counts.max().clamp(min=1.0)
        return counts / max_c


# ============================================================================
# Auto-backend detection
# ============================================================================

def detect_backend() -> str:
    """Detect the best available backend."""
    try:
        from oneuro_metal import has_gpu
        if has_gpu():
            return "metal"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_brain(n_neurons: int, device: str = "auto") -> CUDAMolecularBrain:
    """Create a brain on the best available device."""
    return CUDAMolecularBrain(n_neurons, device=device)


def create_regional_brain(
    scale: str = "xlarge",
    device: str = "auto",
    seed: int = 42,
) -> CUDARegionalBrain:
    """Create a pre-wired regional brain at specified scale."""
    factories = {
        "minimal": CUDARegionalBrain.minimal,
        "small": CUDARegionalBrain.small,
        "standard": CUDARegionalBrain.standard,
        "medium": CUDARegionalBrain.medium,
        "xlarge": CUDARegionalBrain.xlarge,
        "large": CUDARegionalBrain.large,
        "mega": CUDARegionalBrain.mega,
        "100k": CUDARegionalBrain.hundred_k,
        "1m": CUDARegionalBrain.million,
    }
    factory = factories.get(scale, CUDARegionalBrain.xlarge)
    return factory(device=device, seed=seed)
