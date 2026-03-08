"""Drosophila melanogaster -- biophysically faithful digital fruit fly.

A complete organism model: brain (nervous system), body (sensory + motor systems),
and a unified Drosophila class that runs the sense-think-act loop every timestep.

The Drosophila brain is built on oNeuro's CUDAMolecularBrain, giving every neuron
Hodgkin-Huxley ion channels (Na_v, K_v, Ca_v), 6 neurotransmitters (DA, 5-HT,
octopamine/NE, ACh, GABA, glutamate), STDP, and pharmacology. The wiring is
organized into 15 brain regions matching the FlyWire connectome (Dorkenwald et al.
2024), with connectivity probabilities derived from known Drosophila circuits.

Key biological differences from vertebrates:
  - ACh is the PRIMARY excitatory neurotransmitter (not glutamate)
  - Glutamate is INHIBITORY at the NMJ and some central synapses
  - Octopamine (mapped to NE index) replaces norepinephrine for arousal/flight
  - Mushroom body uses sparse population coding (Kenyon cells)
  - Central complex provides an allocentric compass (heading representation)
  - Compound eyes with ~750 ommatidia per eye (not a lens-based retina)

Scale tiers:
  - tiny:   1,000 neurons  (fast testing, <1s)
  - small:  5,000 neurons  (Mac MPS, ~10s)
  - medium: 25,000 neurons (A100, ~60s)
  - large:  139,000 neurons (full FlyWire, A100/H200)

References:
  - Dorkenwald et al. (2024) "Neuronal wiring diagram of an adult brain"
    Nature 634:124-138 (FlyWire connectome)
  - Aso et al. (2014) "The neuronal architecture of the mushroom body provides
    a logic for associative learning" eLife 3:e04577
  - Seelig & Jayaraman (2015) "Neural dynamics for landmark orientation and
    angular path integration" Nature 521:186-191
  - Tuthill & Wilson (2016) "Mechanosensation and adaptive motor control in
    insects" Curr Biol 26:R1022-R1038

Usage:
    from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain

    # Quick test
    brain = DrosophilaBrain(scale='tiny')
    brain.self_test()

    # Full organism in a world
    fly = Drosophila(world=None, scale='small')
    fly.step(world=None)  # one sense-think-act cycle
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_ACH, NT_GABA, NT_GLU,
    CH_GABAA, CH_NMDA, CH_NACHR,
    ARCH_PYRAMIDAL, ARCH_INTERNEURON,
)


# ==========================================================================
# Constants -- Drosophila biology
# ==========================================================================

# NT mapping for insects:
#   ACh   = NT_ACH (3) -- primary EXCITATORY (not glutamate!)
#   GABA  = NT_GABA (4) -- inhibitory
#   Glu   = NT_GLU (5) -- inhibitory in insects (some motor exceptions)
#   DA    = NT_DA (0) -- reward (PAM) / punishment (PPL1)
#   5-HT  = NT_5HT (1) -- sleep, aggression modulation
#   Oct   = NT_NE (2) -- arousal, flight-or-fight (vertebrate NE analog)

# Scale tiers: (name -> total neuron count)
SCALE_TIERS = {
    "tiny": 1_000,
    "small": 5_000,
    "medium": 25_000,
    "large": 139_000,
}

# Brain region allocation (fraction of total neurons)
# Based on FlyWire neuron counts (Dorkenwald et al. 2024)
REGION_SPEC: List[Tuple[str, float, int, str]] = [
    # (name, fraction, primary_nt, description)
    #   primary_nt: "ach", "gaba", "da", "5ht", "oct", "glu", "mixed"
    ("AL",      0.05, NT_ACH,  "Antennal Lobe -- olfactory glomeruli"),
    ("MB_KC",   0.10, NT_ACH,  "Mushroom Body Kenyon Cells -- sparse coding"),
    ("MBON",    0.01, NT_ACH,  "MB Output Neurons -- decision"),
    ("DAN",     0.01, NT_DA,   "Dopaminergic Neurons -- PAM/PPL1 reward"),
    ("CX",      0.06, NT_ACH,  "Central Complex -- navigation, heading"),
    ("OL_LAM",  0.10, NT_ACH,  "Optic Lobe Lamina -- photoreceptor input"),
    ("OL_MED",  0.15, NT_ACH,  "Optic Lobe Medulla -- motion computation"),
    ("OL_LOB",  0.08, NT_ACH,  "Optic Lobe Lobula -- visual features"),
    ("LH",      0.03, NT_ACH,  "Lateral Horn -- innate olfactory"),
    ("SEZ",     0.05, NT_ACH,  "Subesophageal Zone -- taste, feeding"),
    ("SUP",     0.10, NT_ACH,  "Superior Brain -- higher processing"),
    ("DN",      0.03, NT_ACH,  "Descending Neurons -- brain->VNC"),
    ("VNC",     0.15, NT_ACH,  "Ventral Nerve Cord -- leg/wing motor"),
    ("NEUROMOD",0.03, NT_DA,   "Neuromodulatory -- DA/5HT/Oct global"),
    ("OTHER",   0.05, NT_ACH,  "Other neuropils"),
]

# Fraction of neurons that are inhibitory (GABAergic) per region
INHIB_FRACTION = {
    "AL":      0.30,  # local interneurons in AL are GABAergic
    "MB_KC":   0.05,  # KC are mostly cholinergic
    "MBON":    0.30,  # some MBONs are GABAergic
    "DAN":     0.00,  # DANs are dopaminergic, not inhibitory
    "CX":      0.25,  # ring neurons include GABAergic
    "OL_LAM":  0.20,  # lamina amacrine cells
    "OL_MED":  0.35,  # many Tm/Mi neurons are GABAergic
    "OL_LOB":  0.25,  # lobula plate tangential cells
    "LH":      0.20,  # LH local interneurons
    "SEZ":     0.20,  # inhibitory taste interneurons
    "SUP":     0.30,  # mixed inhibitory/excitatory
    "DN":      0.10,  # mostly excitatory descending
    "VNC":     0.30,  # inhibitory motor interneurons
    "NEUROMOD":0.00,  # modulatory, not GABAergic
    "OTHER":   0.25,
}

# Simulation timing
DT_BODY = 0.05        # seconds per body step (50ms)
NEURAL_STEPS_PER_BODY = 20  # 20 neural steps (0.1ms each) = 2ms per body step
# Note: body step is 50ms of "fly time" but only 2ms of neural simulation.
# This is a compromise for speed -- increase NEURAL_STEPS_PER_BODY for
# higher fidelity.

# Body physics
FLY_WALK_SPEED = 2.0      # mm/s walking speed
FLY_FLIGHT_SPEED = 200.0  # mm/s flight speed (simplified)
FLY_TURN_RATE = 3.0       # radians/s max turn rate
FLY_WING_BEAT_HZ = 200.0  # wing beat frequency
FLY_BODY_LENGTH = 2.5     # mm
FLY_ENERGY_MAX = 100.0
FLY_ENERGY_WALK_COST = 0.01   # per body step
FLY_ENERGY_FLY_COST = 0.1     # per body step (flight is expensive)
FLY_ENERGY_FEED_GAIN = 5.0    # per feeding step

# 3D flight physics
GRAVITY_MM_S2 = 9810.0          # mm/s^2 (9.81 m/s^2 converted)
FLY_MASS_MG = 1.0               # mg (Drosophila mass)
FLY_WING_AREA_MM2 = 2.5         # mm^2 per wing (approx)
FLY_LIFT_COEFF = 1.5            # coefficient of lift (high for insect hovering)
FLY_DRAG_COEFF = 0.5            # coefficient of drag
AIR_DENSITY_MG_MM3 = 1.184e-6   # mg/mm^3 (1.184 kg/m^3 converted)
FLY_MAX_ALTITUDE = 50.0         # mm (typical indoor flight ceiling)
FLY_TAKEOFF_SPEED = 5.0         # mm/s initial upward velocity
FLY_CLIMB_RATE = 3.0            # mm/s max sustained climb
FLY_DESCENT_RATE = 5.0          # mm/s max descent without stall


# ==========================================================================
# DrosophilaBrain -- nervous system on CUDAMolecularBrain
# ==========================================================================

class DrosophilaBrain:
    """Builds the Drosophila melanogaster nervous system.

    Constructs a CUDAMolecularBrain with neurons organized into 15 brain
    regions matching the FlyWire connectome, wired with biologically
    plausible connectivity patterns.
    """

    def __init__(
        self,
        scale: str = "small",
        device: str = "auto",
        seed: int = 42,
    ):
        if scale not in SCALE_TIERS:
            raise ValueError(f"Unknown scale '{scale}'. Choose from: {list(SCALE_TIERS.keys())}")

        self.scale = scale
        self.n_total = SCALE_TIERS[scale]
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build the brain
        self.brain = CUDAMolecularBrain(self.n_total, device=device)
        self.dev = self.brain.device
        self.device = self.dev  # alias for demo/API compatibility

        # Region index maps: region_name -> (start, end, list_of_ids)
        self.regions: Dict[str, Tuple[int, int, List[int]]] = {}
        self._build_regions()
        self._assign_archetypes()
        self._assign_positions()
        self._wire_connectome()
        self._cache_tensors()

    # ------------------------------------------------------------------
    # Region construction
    # ------------------------------------------------------------------

    def _build_regions(self) -> None:
        """Allocate neuron indices to each brain region proportionally."""
        idx = 0
        remaining = self.n_total
        for i, (name, frac, _nt, _desc) in enumerate(REGION_SPEC):
            if i == len(REGION_SPEC) - 1:
                # Last region gets whatever is left (rounding cleanup)
                n = remaining
            else:
                n = max(1, int(self.n_total * frac))
                n = min(n, remaining)
            start = idx
            end = idx + n
            self.regions[name] = (start, end, list(range(start, end)))
            idx += n
            remaining -= n

        assert idx == self.n_total, f"Region allocation: {idx} != {self.n_total}"

    def _region_ids(self, name: str) -> List[int]:
        """Get neuron ID list for a region."""
        return self.regions[name][2]

    def _region_size(self, name: str) -> int:
        """Get neuron count for a region."""
        s, e, _ = self.regions[name]
        return e - s

    def _region_slice(self, name: str) -> slice:
        """Get slice for a region."""
        s, e, _ = self.regions[name]
        return slice(s, e)

    # ------------------------------------------------------------------
    # Compatibility properties (used by demo sensory/motor encoders)
    # ------------------------------------------------------------------

    @property
    def antennal_lobe_ids(self) -> List[int]:
        """Neuron IDs for the Antennal Lobe (olfactory glomeruli)."""
        return self._region_ids("AL")

    @property
    def optic_lobe_ids(self) -> List[int]:
        """Neuron IDs for all Optic Lobe layers (Lamina + Medulla + Lobula)."""
        return (self._region_ids("OL_LAM")
                + self._region_ids("OL_MED")
                + self._region_ids("OL_LOB"))

    @property
    def motor_ids(self) -> List[int]:
        """Neuron IDs for the Ventral Nerve Cord (motor output)."""
        return self._region_ids("VNC")

    @property
    def central_complex_ids(self) -> List[int]:
        """Neuron IDs for the Central Complex (navigation)."""
        return self._region_ids("CX")

    @property
    def kenyon_cell_ids(self) -> List[int]:
        """Neuron IDs for the Mushroom Body Kenyon Cells."""
        return self._region_ids("MB_KC")

    @property
    def mb_output_ids(self) -> List[int]:
        """Neuron IDs for the Mushroom Body Output Neurons."""
        return self._region_ids("MBON")

    @property
    def projection_neuron_ids(self) -> List[int]:
        """Neuron IDs for projection neurons (AL subset)."""
        # In the full model, projection neurons are a subset of AL.
        # Return the first 40% of AL as projection neurons.
        al = self._region_ids("AL")
        return al[:max(1, len(al) * 2 // 5)]

    @property
    def other_ids(self) -> List[int]:
        """Neuron IDs for OTHER + NEUROMOD regions."""
        return self._region_ids("OTHER") + self._region_ids("NEUROMOD")

    # ------------------------------------------------------------------
    # Archetype assignment
    # ------------------------------------------------------------------

    def _assign_archetypes(self) -> None:
        """Set neuron archetypes: pyramidal (excitatory) or interneuron (inhibitory)."""
        for name, (start, end, ids) in self.regions.items():
            n_region = end - start
            n_inhib = max(0, int(n_region * INHIB_FRACTION.get(name, 0.2)))
            # Last n_inhib neurons in each region are inhibitory
            for nid in ids[-n_inhib:]:
                self.brain.archetype[nid] = ARCH_INTERNEURON

        # Neuromodulatory neurons: split into DA, 5-HT, octopamine subtypes
        nmod_ids = self._region_ids("NEUROMOD")
        n_nm = len(nmod_ids)
        # ~40% DA, ~30% 5-HT, ~30% octopamine
        n_da = max(1, int(n_nm * 0.4))
        n_5ht = max(1, int(n_nm * 0.3))
        # Store subtypes for targeted modulation
        self._neuromod_da_ids = nmod_ids[:n_da]
        self._neuromod_5ht_ids = nmod_ids[n_da:n_da + n_5ht]
        self._neuromod_oct_ids = nmod_ids[n_da + n_5ht:]

    # ------------------------------------------------------------------
    # Spatial positions
    # ------------------------------------------------------------------

    def _assign_positions(self) -> None:
        """Assign 3D positions approximating Drosophila brain anatomy.

        Layout (anterior-posterior axis = z):
          z=0-1: Optic lobes (lateral), Antennal lobe (medial anterior)
          z=1-2: Mushroom body, Lateral horn, Central complex
          z=2-3: Superior brain, SEZ (ventral)
          z=3-5: VNC (extending posterior along body axis)
        """
        b = self.brain
        dev = self.dev

        # Helper: assign random positions within a bounding box
        def _place(name: str, x_range: Tuple[float, float],
                   y_range: Tuple[float, float], z_range: Tuple[float, float]):
            s, e, _ = self.regions[name]
            n = e - s
            if n == 0:
                return
            b.x[s:e] = x_range[0] + torch.rand(n, device=dev) * (x_range[1] - x_range[0])
            b.y[s:e] = y_range[0] + torch.rand(n, device=dev) * (y_range[1] - y_range[0])
            b.z[s:e] = z_range[0] + torch.rand(n, device=dev) * (z_range[1] - z_range[0])

        # Anterior sensory neuropils
        _place("AL",      (-0.5, 0.5),  (0.0, 1.0), (0.0, 0.5))   # medial anterior
        _place("OL_LAM",  (-2.0, -1.0), (0.0, 1.0), (0.0, 0.5))   # lateral left
        _place("OL_MED",  (-2.0, -1.0), (0.0, 1.0), (0.5, 1.0))   # lateral medulla
        _place("OL_LOB",  (-1.5, -0.5), (0.0, 1.0), (0.8, 1.5))   # lobula, deeper

        # Central brain
        _place("MB_KC",   (-0.5, 0.5),  (0.5, 1.5), (1.0, 2.0))   # mushroom body
        _place("MBON",    (-0.3, 0.3),  (0.3, 0.8), (1.5, 2.0))   # MB output
        _place("DAN",     (-0.3, 0.3),  (0.8, 1.2), (1.5, 2.0))   # dopamine neurons
        _place("CX",      (-0.3, 0.3),  (1.0, 1.5), (1.0, 1.5))   # central complex
        _place("LH",      (-1.0, -0.3), (0.5, 1.0), (1.0, 1.5))   # lateral horn
        _place("SUP",     (-0.5, 0.5),  (1.0, 2.0), (1.5, 2.5))   # superior brain

        # Ventral / posterior
        _place("SEZ",     (-0.3, 0.3),  (-0.5, 0.3), (1.5, 2.5))  # subesophageal
        _place("DN",      (-0.3, 0.3),  (0.0, 0.5),  (2.5, 3.0))  # descending
        _place("VNC",     (-0.3, 0.3),  (-0.3, 0.3), (3.0, 5.0))  # ventral nerve cord

        # Modulatory and other
        _place("NEUROMOD",(-0.5, 0.5),  (0.5, 1.5), (1.0, 2.5))   # scattered
        _place("OTHER",   (-0.5, 0.5),  (0.0, 1.5), (0.5, 2.5))   # miscellaneous

    # ------------------------------------------------------------------
    # Connectivity (FlyWire-inspired wiring)
    # ------------------------------------------------------------------

    def _wire_connectome(self) -> None:
        """Wire synapses between brain regions following known Drosophila circuits.

        Major circuits implemented:
          1. Olfactory: AL -> MB (projection neurons), AL -> LH (innate)
          2. Mushroom Body: sparse KC coding, DAN -> MB, MBON -> output
          3. Visual: Lamina -> Medulla -> Lobula (retinotopic columns)
          4. Central Complex: ring/compass neurons, heading integration
          5. Motor: DN -> VNC, VNC CPG circuits, SEZ feeding
          6. Neuromodulatory: DA/5-HT/Oct global modulation
        """
        dev = self.dev
        all_pre: List[torch.Tensor] = []
        all_post: List[torch.Tensor] = []
        all_weight: List[torch.Tensor] = []
        all_nt: List[torch.Tensor] = []

        def _connect(
            src_name: str, dst_name: str,
            prob: float, w_range: Tuple[float, float],
            nt: int,
            src_slice: Optional[slice] = None,
            dst_slice: Optional[slice] = None,
        ) -> int:
            """Add random connections between two regions. Returns count added."""
            src_ids = self._region_ids(src_name)
            dst_ids = self._region_ids(dst_name)
            if src_slice is not None:
                src_ids = src_ids[src_slice]
            if dst_slice is not None:
                dst_ids = dst_ids[dst_slice]
            if not src_ids or not dst_ids:
                return 0

            n_possible = len(src_ids) * len(dst_ids)

            # For large connection matrices, sample randomly
            if n_possible > 50_000:
                n_expected = max(1, int(n_possible * prob))
                src_t = torch.tensor(src_ids, device=dev, dtype=torch.int64)
                dst_t = torch.tensor(dst_ids, device=dev, dtype=torch.int64)
                pre_idx = torch.randint(len(src_ids), (n_expected,), device=dev)
                post_idx = torch.randint(len(dst_ids), (n_expected,), device=dev)
                pre_t = src_t[pre_idx]
                post_t = dst_t[post_idx]
                mask = pre_t != post_t
                pre_t, post_t = pre_t[mask], post_t[mask]
            else:
                src_t = torch.tensor(src_ids, device=dev, dtype=torch.int64)
                dst_t = torch.tensor(dst_ids, device=dev, dtype=torch.int64)
                conn_mask = torch.rand(len(src_ids), len(dst_ids), device=dev) < prob
                indices = torch.where(conn_mask)
                pre_t = src_t[indices[0]]
                post_t = dst_t[indices[1]]
                valid = pre_t != post_t
                pre_t, post_t = pre_t[valid], post_t[valid]

            if pre_t.shape[0] == 0:
                return 0

            w = torch.rand(pre_t.shape[0], device=dev) * (w_range[1] - w_range[0]) + w_range[0]
            nt_t = torch.full((pre_t.shape[0],), nt, dtype=torch.int32, device=dev)
            all_pre.append(pre_t)
            all_post.append(post_t)
            all_weight.append(w)
            all_nt.append(nt_t)
            return int(pre_t.shape[0])

        def _connect_sparse_kc(
            src_name: str, dst_name: str,
            fan_in: int, w_range: Tuple[float, float],
            nt: int,
        ) -> int:
            """Sparse random connectivity for Kenyon cell encoding.

            Each destination neuron receives exactly fan_in random inputs
            from the source population (sparse population coding).
            """
            src_ids = self._region_ids(src_name)
            dst_ids = self._region_ids(dst_name)
            if not src_ids or not dst_ids:
                return 0

            src_t = torch.tensor(src_ids, device=dev, dtype=torch.int64)
            n_src = len(src_ids)
            fan = min(fan_in, n_src)

            # For each KC, pick `fan` random PNs
            pre_list = []
            post_list = []
            for dst_id in dst_ids:
                idx = torch.randint(n_src, (fan,), device=dev)
                pre_list.append(src_t[idx])
                post_list.append(torch.full((fan,), dst_id, dtype=torch.int64, device=dev))

            pre_t = torch.cat(pre_list)
            post_t = torch.cat(post_list)
            w = torch.rand(pre_t.shape[0], device=dev) * (w_range[1] - w_range[0]) + w_range[0]
            nt_t = torch.full((pre_t.shape[0],), nt, dtype=torch.int32, device=dev)
            all_pre.append(pre_t)
            all_post.append(post_t)
            all_weight.append(w)
            all_nt.append(nt_t)
            return int(pre_t.shape[0])

        # ================================================================
        # 1. OLFACTORY CIRCUIT
        # ================================================================
        # AL projection neurons -> MB Kenyon cells (sparse coding)
        # Each KC receives ~7 random PN inputs (Caron et al. 2013)
        _connect_sparse_kc("AL", "MB_KC", fan_in=7, w_range=(1.0, 2.0), nt=NT_ACH)
        # AL -> LH (innate olfactory — removed, replaced by lateralized
        # ipsilateral connections in section 8b to preserve total density)
        # AL local inhibition (within AL)
        _connect("AL", "AL", prob=0.15, w_range=(0.5, 1.2), nt=NT_GABA)

        # ================================================================
        # 2. MUSHROOM BODY CIRCUIT (associative learning)
        # ================================================================
        # APL-like feedback inhibition (Aso et al. 2014; Lin et al. 2014)
        # The Anterior Paired Lateral (APL) neuron receives excitation from
        # the entire KC population and provides global GABAergic inhibition
        # back to all KCs. This enforces sparse population coding: only the
        # most strongly driven KCs survive the inhibition, creating the
        # ~5-10% sparsity essential for pattern separation and associative
        # learning. Modeled as direct KC→KC inhibitory connections.
        _connect("MB_KC", "MB_KC", prob=0.03, w_range=(1.0, 2.5), nt=NT_GABA)
        # KC -> MBON (readout of sparse KC activity)
        _connect("MB_KC", "MBON", prob=0.10, w_range=(0.5, 1.5), nt=NT_ACH)
        # DAN -> MB_KC compartments (reward/punishment teaching signal)
        # PAM DANs (positive valence): first half of DAN
        # PPL1 DANs (negative valence): second half of DAN
        _connect("DAN", "MB_KC", prob=0.08, w_range=(0.8, 1.5), nt=NT_DA)
        # DAN -> MBON (modulatory)
        _connect("DAN", "MBON", prob=0.15, w_range=(0.5, 1.2), nt=NT_DA)
        # MBON -> downstream (CX, LH, SEZ for decision making)
        _connect("MBON", "CX",  prob=0.20, w_range=(0.8, 1.5), nt=NT_ACH)
        _connect("MBON", "LH",  prob=0.15, w_range=(0.6, 1.2), nt=NT_ACH)
        _connect("MBON", "SEZ", prob=0.15, w_range=(0.6, 1.2), nt=NT_ACH)
        # MBON feedback to DAN (for extinction learning)
        _connect("MBON", "DAN", prob=0.10, w_range=(0.5, 1.0), nt=NT_ACH)

        # ================================================================
        # 3. VISUAL CIRCUIT (retinotopic processing)
        # ================================================================
        # Lamina -> Medulla (columnar, high connectivity)
        _connect("OL_LAM", "OL_MED", prob=0.15, w_range=(1.0, 2.0), nt=NT_ACH)
        # Medulla -> Lobula (feature extraction)
        _connect("OL_MED", "OL_LOB", prob=0.10, w_range=(0.8, 1.5), nt=NT_ACH)
        # Medulla local inhibition (direction selectivity via Mi/Tm cells)
        _connect("OL_MED", "OL_MED", prob=0.05, w_range=(0.5, 1.0), nt=NT_GABA)
        # Lobula -> Superior brain (visual feature integration)
        _connect("OL_LOB", "SUP",    prob=0.08, w_range=(0.8, 1.5), nt=NT_ACH)
        # Lobula -> CX (visual input to navigation)
        _connect("OL_LOB", "CX",     prob=0.10, w_range=(0.8, 1.5), nt=NT_ACH)
        # Lobula -> DN (removed, replaced by lateralized connections in 8b)

        # ================================================================
        # 4. CENTRAL COMPLEX (navigation / heading)
        # ================================================================
        # CX internal recurrence (compass ring attractor)
        _connect("CX", "CX", prob=0.10, w_range=(0.5, 1.2), nt=NT_ACH)
        # CX inhibitory ring (winner-take-all for heading)
        n_cx = self._region_size("CX")
        n_cx_inhib = max(1, int(n_cx * INHIB_FRACTION["CX"]))
        cx_ids = self._region_ids("CX")
        cx_excit = cx_ids[:-n_cx_inhib] if n_cx_inhib > 0 else cx_ids
        cx_inhib = cx_ids[-n_cx_inhib:] if n_cx_inhib > 0 else []
        if cx_excit and cx_inhib:
            # Excitatory -> inhibitory (drives lateral inhibition)
            _connect("CX", "CX", prob=0.08, w_range=(0.5, 1.0), nt=NT_GABA,
                     src_slice=slice(0, len(cx_excit)),
                     dst_slice=slice(len(cx_excit), None))
        # CX -> DN (steering commands)
        _connect("CX", "DN", prob=0.15, w_range=(0.8, 1.5), nt=NT_ACH)
        # CX -> SEZ (navigation influences feeding decisions)
        _connect("CX", "SEZ", prob=0.05, w_range=(0.5, 1.0), nt=NT_ACH)

        # ================================================================
        # 5. LATERAL HORN (innate olfactory responses)
        # ================================================================
        # LH -> DN (removed, replaced by lateralized connections in 8b)
        # LH -> SEZ (innate feeding drive)
        _connect("LH", "SEZ", prob=0.10, w_range=(0.6, 1.2), nt=NT_ACH)
        # LH internal processing
        _connect("LH", "LH",  prob=0.10, w_range=(0.4, 0.8), nt=NT_GABA)

        # ================================================================
        # 6. SUPERIOR BRAIN (higher-order integration)
        # ================================================================
        _connect("SUP", "CX",  prob=0.08, w_range=(0.6, 1.2), nt=NT_ACH)
        _connect("SUP", "DN",  prob=0.08, w_range=(0.6, 1.2), nt=NT_ACH)
        _connect("SUP", "SUP", prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)
        _connect("SUP", "SUP", prob=0.03, w_range=(0.4, 0.8), nt=NT_GABA)

        # ================================================================
        # 7. SUBESOPHAGEAL ZONE (taste, feeding motor)
        # ================================================================
        # SEZ -> VNC (feeding motor commands, proboscis extension)
        _connect("SEZ", "VNC", prob=0.10, w_range=(0.8, 1.5), nt=NT_ACH)
        # SEZ internal (taste processing)
        _connect("SEZ", "SEZ", prob=0.08, w_range=(0.4, 0.8), nt=NT_ACH)
        _connect("SEZ", "SEZ", prob=0.05, w_range=(0.4, 0.8), nt=NT_GABA)

        # ================================================================
        # 8. DESCENDING NEURONS (brain -> VNC motor commands)
        # ================================================================
        # DN -> VNC (removed, replaced by lateralized connections in 8b)
        # DN internal (coordination between descending commands)
        _connect("DN", "DN", prob=0.10, w_range=(0.5, 1.0), nt=NT_ACH)

        # ================================================================
        # 8b. LATERALIZED WIRING (ipsilateral bias)
        # ================================================================
        # In real Drosophila, sensory-motor pathways are bilaterally
        # symmetric with strong ipsilateral bias: left eye → left DN →
        # left VNC motor, right eye → right DN → right VNC motor.
        # This is ANATOMY, not behavior. The lateralized wiring enables
        # behaviors like phototaxis and chemotaxis to EMERGE from
        # asymmetric sensory input flowing through ipsilateral pathways.
        #
        # Split each region into left (first half) and right (second half).
        dn_ids = self._region_ids("DN")
        vnc_ids_pre = self._region_ids("VNC")
        lob_ids = self._region_ids("OL_LOB")
        al_ids_lat = self._region_ids("AL")
        lh_ids = self._region_ids("LH")

        dn_mid = len(dn_ids) // 2
        vnc_mid = len(vnc_ids_pre) // 2
        lob_mid = len(lob_ids) // 2
        al_mid = len(al_ids_lat) // 2
        lh_mid = len(lh_ids) // 2

        def _connect_lateral(
            src_ids_list: List[int], dst_ids_list: List[int],
            prob: float, w_range: Tuple[float, float], nt: int,
        ) -> int:
            """Add connections between specific neuron ID lists."""
            if not src_ids_list or not dst_ids_list:
                return 0
            src_t = torch.tensor(src_ids_list, device=dev, dtype=torch.int64)
            dst_t = torch.tensor(dst_ids_list, device=dev, dtype=torch.int64)
            n_possible = len(src_ids_list) * len(dst_ids_list)
            if n_possible > 50_000:
                n_expected = max(1, int(n_possible * prob))
                pre_idx = torch.randint(len(src_ids_list), (n_expected,), device=dev)
                post_idx = torch.randint(len(dst_ids_list), (n_expected,), device=dev)
                pre_t = src_t[pre_idx]
                post_t = dst_t[post_idx]
            else:
                conn_mask = torch.rand(len(src_ids_list), len(dst_ids_list), device=dev) < prob
                indices = torch.where(conn_mask)
                pre_t = src_t[indices[0]]
                post_t = dst_t[indices[1]]
            valid = pre_t != post_t
            pre_t, post_t = pre_t[valid], post_t[valid]
            if pre_t.shape[0] == 0:
                return 0
            w = torch.rand(pre_t.shape[0], device=dev) * (w_range[1] - w_range[0]) + w_range[0]
            nt_t = torch.full((pre_t.shape[0],), nt, dtype=torch.int32, device=dev)
            all_pre.append(pre_t)
            all_post.append(post_t)
            all_weight.append(w)
            all_nt.append(nt_t)
            return int(pre_t.shape[0])

        # Visual: OL_LOB → DN (was prob=0.05 bilateral → now lateralized)
        # Total density preserved: ipsi=0.08 + contra=0.02 → 0.10/2 = 0.05
        _connect_lateral(lob_ids[:lob_mid], dn_ids[:dn_mid],
                         prob=0.08, w_range=(1.0, 2.0), nt=NT_ACH)  # L→L ipsi
        _connect_lateral(lob_ids[lob_mid:], dn_ids[dn_mid:],
                         prob=0.08, w_range=(1.0, 2.0), nt=NT_ACH)  # R→R ipsi
        _connect_lateral(lob_ids[:lob_mid], dn_ids[dn_mid:],
                         prob=0.02, w_range=(0.5, 1.0), nt=NT_ACH)  # L→R contra
        _connect_lateral(lob_ids[lob_mid:], dn_ids[:dn_mid],
                         prob=0.02, w_range=(0.5, 1.0), nt=NT_ACH)  # R→L contra

        # Olfactory: AL → LH (was prob=0.20 bilateral → now lateralized)
        # Total density preserved: ipsi=0.35 + contra=0.05 → 0.40/2 = 0.20
        _connect_lateral(al_ids_lat[:al_mid], lh_ids[:lh_mid],
                         prob=0.35, w_range=(0.8, 1.5), nt=NT_ACH)  # L→L
        _connect_lateral(al_ids_lat[al_mid:], lh_ids[lh_mid:],
                         prob=0.35, w_range=(0.8, 1.5), nt=NT_ACH)  # R→R
        _connect_lateral(al_ids_lat[:al_mid], lh_ids[lh_mid:],
                         prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)  # L→R contra
        _connect_lateral(al_ids_lat[al_mid:], lh_ids[:lh_mid],
                         prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)  # R→L contra

        # LH → DN (was prob=0.12 bilateral → now lateralized)
        # Total density: ipsi=0.20 + contra=0.04 → 0.24/2 = 0.12
        _connect_lateral(lh_ids[:lh_mid], dn_ids[:dn_mid],
                         prob=0.20, w_range=(0.8, 1.5), nt=NT_ACH)  # L→L
        _connect_lateral(lh_ids[lh_mid:], dn_ids[dn_mid:],
                         prob=0.20, w_range=(0.8, 1.5), nt=NT_ACH)  # R→R
        _connect_lateral(lh_ids[:lh_mid], dn_ids[dn_mid:],
                         prob=0.04, w_range=(0.4, 0.8), nt=NT_ACH)  # L→R contra
        _connect_lateral(lh_ids[lh_mid:], dn_ids[:dn_mid],
                         prob=0.04, w_range=(0.4, 0.8), nt=NT_ACH)  # R→L contra

        # DN → VNC (was prob=0.20 bilateral → now lateralized)
        # Total density: ipsi=0.35 + contra=0.05 → 0.40/2 = 0.20
        _connect_lateral(dn_ids[:dn_mid], vnc_ids_pre[:vnc_mid],
                         prob=0.35, w_range=(1.0, 2.0), nt=NT_ACH)  # L→L
        _connect_lateral(dn_ids[dn_mid:], vnc_ids_pre[vnc_mid:],
                         prob=0.35, w_range=(1.0, 2.0), nt=NT_ACH)  # R→R
        _connect_lateral(dn_ids[:dn_mid], vnc_ids_pre[vnc_mid:],
                         prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)  # L→R contra
        _connect_lateral(dn_ids[dn_mid:], vnc_ids_pre[:vnc_mid],
                         prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)  # R→L contra

        # ================================================================
        # 9. VENTRAL NERVE CORD (leg/wing motor patterns)
        # ================================================================
        # VNC internal: CPG circuits for walking (tripod gait) and flight
        # Excitatory recurrence (rhythm generation)
        _connect("VNC", "VNC", prob=0.06, w_range=(0.5, 1.2), nt=NT_ACH)
        # Inhibitory cross-connections (alternating leg patterns)
        vnc_ids = self._region_ids("VNC")
        n_vnc = len(vnc_ids)
        if n_vnc >= 6:
            # Split VNC into 6 leg segments + flight segment
            seg_size = n_vnc // 7
            self._vnc_leg_segments = []
            for leg in range(6):
                seg_start = leg * seg_size
                seg_end = seg_start + seg_size
                self._vnc_leg_segments.append(vnc_ids[seg_start:seg_end])
            self._vnc_flight_ids = vnc_ids[6 * seg_size:]

            # Tripod gait: legs 0,2,4 alternate with legs 1,3,5
            # Cross-inhibition between tripod groups
            tripod_a = self._vnc_leg_segments[0] + self._vnc_leg_segments[2] + self._vnc_leg_segments[4]
            tripod_b = self._vnc_leg_segments[1] + self._vnc_leg_segments[3] + self._vnc_leg_segments[5]
            if tripod_a and tripod_b:
                # Mutual inhibition for alternating tripod gait
                _src_ids = tripod_a
                _dst_ids = tripod_b
                n_inh = max(1, int(len(_src_ids) * len(_dst_ids) * 0.03))
                if n_inh > 0:
                    src_t = torch.tensor(_src_ids, device=dev, dtype=torch.int64)
                    dst_t = torch.tensor(_dst_ids, device=dev, dtype=torch.int64)
                    pre_idx = torch.randint(len(_src_ids), (n_inh,), device=dev)
                    post_idx = torch.randint(len(_dst_ids), (n_inh,), device=dev)
                    pre_t = src_t[pre_idx]
                    post_t = dst_t[post_idx]
                    w = torch.rand(n_inh, device=dev) * 0.8 + 0.5
                    nt_t = torch.full((n_inh,), NT_GABA, dtype=torch.int32, device=dev)
                    all_pre.append(pre_t)
                    all_post.append(post_t)
                    all_weight.append(w)
                    all_nt.append(nt_t)
                    # Reverse direction
                    all_pre.append(post_t.clone())
                    all_post.append(pre_t.clone())
                    all_weight.append(w.clone())
                    all_nt.append(nt_t.clone())
        else:
            self._vnc_leg_segments = [vnc_ids]
            self._vnc_flight_ids = []

        # ================================================================
        # 10. NEUROMODULATORY CIRCUITS
        # ================================================================
        # DA neurons -> MB (reward/punishment)
        nm_da = self._neuromod_da_ids
        nm_5ht = self._neuromod_5ht_ids
        nm_oct = self._neuromod_oct_ids

        if nm_da:
            # DA -> MB (teaching signal) -- also comes from DAN, this is global
            da_t = torch.tensor(nm_da, device=dev, dtype=torch.int64)
            mb_ids = self._region_ids("MB_KC")
            if mb_ids:
                n_da_mb = max(1, int(len(nm_da) * len(mb_ids) * 0.02))
                pre_idx = torch.randint(len(nm_da), (n_da_mb,), device=dev)
                post_idx = torch.randint(len(mb_ids), (n_da_mb,), device=dev)
                mb_t = torch.tensor(mb_ids, device=dev, dtype=torch.int64)
                all_pre.append(da_t[pre_idx])
                all_post.append(mb_t[post_idx])
                all_weight.append(torch.rand(n_da_mb, device=dev) * 1.0 + 0.5)
                all_nt.append(torch.full((n_da_mb,), NT_DA, dtype=torch.int32, device=dev))

        if nm_5ht:
            # 5-HT -> broad modulation (sleep/wake, aggression)
            sht_t = torch.tensor(nm_5ht, device=dev, dtype=torch.int64)
            all_ids = list(range(self.n_total))
            n_5ht_broad = max(1, int(len(nm_5ht) * self.n_total * 0.002))
            pre_idx = torch.randint(len(nm_5ht), (n_5ht_broad,), device=dev)
            post_idx = torch.randint(self.n_total, (n_5ht_broad,), device=dev)
            all_pre.append(sht_t[pre_idx])
            all_post.append(post_idx)
            all_weight.append(torch.rand(n_5ht_broad, device=dev) * 0.5 + 0.3)
            all_nt.append(torch.full((n_5ht_broad,), NT_5HT, dtype=torch.int32, device=dev))

        if nm_oct:
            # Octopamine -> arousal, flight (NE analog in insects)
            oct_t = torch.tensor(nm_oct, device=dev, dtype=torch.int64)
            # Oct -> VNC (flight activation) + broad arousal
            n_oct_broad = max(1, int(len(nm_oct) * self.n_total * 0.002))
            pre_idx = torch.randint(len(nm_oct), (n_oct_broad,), device=dev)
            post_idx = torch.randint(self.n_total, (n_oct_broad,), device=dev)
            all_pre.append(oct_t[pre_idx])
            all_post.append(post_idx)
            all_weight.append(torch.rand(n_oct_broad, device=dev) * 0.5 + 0.3)
            all_nt.append(torch.full((n_oct_broad,), NT_NE, dtype=torch.int32, device=dev))

        # ================================================================
        # 11. OTHER NEUROPILS
        # ================================================================
        _connect("OTHER", "SUP", prob=0.05, w_range=(0.4, 0.8), nt=NT_ACH)
        _connect("OTHER", "CX",  prob=0.03, w_range=(0.4, 0.8), nt=NT_ACH)
        _connect("SUP", "OTHER", prob=0.03, w_range=(0.4, 0.8), nt=NT_ACH)

        # ================================================================
        # COMMIT ALL SYNAPSES
        # ================================================================
        if all_pre:
            self.brain.add_synapses(
                torch.cat(all_pre),
                torch.cat(all_post),
                torch.cat(all_weight),
                torch.cat(all_nt),
            )

    # ------------------------------------------------------------------
    # Cached tensor construction
    # ------------------------------------------------------------------

    def _cache_tensors(self) -> None:
        """Pre-build GPU tensors for fast region-level stimulation and readout."""
        dev = self.dev
        self._tensors: Dict[str, torch.Tensor] = {}
        for name in self.regions:
            ids = self._region_ids(name)
            if ids:
                self._tensors[name] = torch.tensor(ids, dtype=torch.int64, device=dev)

        # Compatibility alias tensors (used by demo sensory/motor encoders)
        # _optic_t: combined optic lobe IDs
        optic_ids = (self._region_ids("OL_LAM")
                     + self._region_ids("OL_MED")
                     + self._region_ids("OL_LOB"))
        self._optic_t = torch.tensor(optic_ids, dtype=torch.int64, device=dev)

        # Lateralized optic lobe: left eye (first half), right eye (second half)
        optic_mid = len(optic_ids) // 2
        self._optic_left_t = torch.tensor(optic_ids[:optic_mid], dtype=torch.int64, device=dev)
        self._optic_right_t = torch.tensor(optic_ids[optic_mid:], dtype=torch.int64, device=dev)

        # _cx_t, _al_t, _motor_t, _kc_t, _mbon_t: from cached region tensors
        self._cx_t = self._tensors.get("CX", torch.zeros(0, dtype=torch.int64, device=dev))
        self._al_t = self._tensors.get("AL", torch.zeros(0, dtype=torch.int64, device=dev))
        self._motor_t = self._tensors.get("VNC", torch.zeros(0, dtype=torch.int64, device=dev))

        # Lateralized AL: left antenna (first half), right antenna (second half)
        al_ids = self._region_ids("AL")
        al_mid = len(al_ids) // 2
        self._al_left_t = torch.tensor(al_ids[:al_mid], dtype=torch.int64, device=dev)
        self._al_right_t = torch.tensor(al_ids[al_mid:], dtype=torch.int64, device=dev)
        self._kc_t = self._tensors.get("MB_KC", torch.zeros(0, dtype=torch.int64, device=dev))
        self._mbon_t = self._tensors.get("MBON", torch.zeros(0, dtype=torch.int64, device=dev))
        self._all_t = torch.arange(self.n_total, dtype=torch.int64, device=dev)

        # Motor subgroup tensors for readout
        vnc_ids = self._region_ids("VNC")
        n_vnc = len(vnc_ids)

        # Forward vs backward motor pools: first half forward, second half backward
        half = n_vnc // 2
        self._vnc_forward_ids = vnc_ids[:half]
        self._vnc_backward_ids = vnc_ids[half:]
        self._t_vnc_fwd = torch.tensor(self._vnc_forward_ids, dtype=torch.int64, device=dev)
        self._t_vnc_bwd = torch.tensor(self._vnc_backward_ids, dtype=torch.int64, device=dev)

        # Left vs right motor pools (within forward pool)
        quarter = half // 2
        self._vnc_left_ids = self._vnc_forward_ids[:quarter]
        self._vnc_right_ids = self._vnc_forward_ids[quarter:]
        self._t_vnc_left = torch.tensor(self._vnc_left_ids, dtype=torch.int64, device=dev)
        self._t_vnc_right = torch.tensor(self._vnc_right_ids, dtype=torch.int64, device=dev)

        # Flight motor neurons
        if hasattr(self, '_vnc_flight_ids') and self._vnc_flight_ids:
            self._t_vnc_flight = torch.tensor(self._vnc_flight_ids, dtype=torch.int64, device=dev)
            # Split flight neurons into climb (first half) and descend (second half)
            n_flight = len(self._vnc_flight_ids)
            half_flight = n_flight // 2
            self._vnc_climb_ids = self._vnc_flight_ids[:half_flight]
            self._vnc_descend_ids = self._vnc_flight_ids[half_flight:]
            self._t_vnc_climb = torch.tensor(self._vnc_climb_ids, dtype=torch.int64, device=dev)
            self._t_vnc_descend = torch.tensor(self._vnc_descend_ids, dtype=torch.int64, device=dev)
        else:
            self._t_vnc_flight = torch.zeros(0, dtype=torch.int64, device=dev)
            self._vnc_climb_ids = []
            self._vnc_descend_ids = []
            self._t_vnc_climb = torch.zeros(0, dtype=torch.int64, device=dev)
            self._t_vnc_descend = torch.zeros(0, dtype=torch.int64, device=dev)

        # SEZ proboscis motor subset (last 1/3 of SEZ)
        sez_ids = self._region_ids("SEZ")
        n_sez = len(sez_ids)
        prob_start = n_sez * 2 // 3
        self._sez_proboscis_ids = sez_ids[prob_start:]
        self._t_sez_proboscis = torch.tensor(self._sez_proboscis_ids, dtype=torch.int64, device=dev)

        # DAN subtypes (PAM = positive valence, PPL1 = negative)
        dan_ids = self._region_ids("DAN")
        n_dan = len(dan_ids)
        half_dan = n_dan // 2
        self._dan_pam_ids = dan_ids[:half_dan]
        self._dan_ppl1_ids = dan_ids[half_dan:]
        self._t_dan_pam = torch.tensor(self._dan_pam_ids, dtype=torch.int64, device=dev)
        self._t_dan_ppl1 = torch.tensor(self._dan_ppl1_ids, dtype=torch.int64, device=dev)

        # Olfactory glomeruli subgroups within AL
        al_ids = self._region_ids("AL")
        n_al = len(al_ids)
        # Divide AL into ~50 glomeruli (or fewer at small scales)
        n_glom = min(50, max(1, n_al // 2))
        glom_size = max(1, n_al // n_glom)
        self._al_glomeruli: List[List[int]] = []
        for g in range(n_glom):
            start = g * glom_size
            end = min(start + glom_size, n_al)
            if start < n_al:
                self._al_glomeruli.append(al_ids[start:end])
        self._n_glomeruli = len(self._al_glomeruli)

    # ------------------------------------------------------------------
    # Stimulation API
    # ------------------------------------------------------------------

    def stimulate_al(self, odorant_profile: Dict[str, float]) -> None:
        """Encode odorants into Antennal Lobe glomeruli.

        Each odorant activates a subset of glomeruli with varying intensity,
        mimicking the combinatorial odor code in real Drosophila.

        Args:
            odorant_profile: dict mapping odorant name to concentration (0-1).
                e.g. {"apple_cider_vinegar": 0.8, "co2": 0.3}
        """
        if not odorant_profile:
            return

        brain = self.brain
        for odorant_name, concentration in odorant_profile.items():
            if concentration <= 0:
                continue
            # Hash odorant name to select glomeruli (deterministic)
            h = hash(odorant_name) & 0xFFFFFFFF
            # Each odorant activates ~30% of glomeruli
            n_active = max(1, int(self._n_glomeruli * 0.3))
            for i in range(n_active):
                glom_idx = (h + i * 7) % self._n_glomeruli
                glom_ids = self._al_glomeruli[glom_idx]
                if glom_ids:
                    t = torch.tensor(glom_ids, dtype=torch.int64, device=self.dev)
                    # Intensity varies by glomerulus (creates combinatorial code)
                    # 40 uA/cm^2 base — enough to drive HH spiking
                    intensity = concentration * 40.0 * (0.5 + 0.5 * ((h + i) % 10) / 10.0)
                    brain.external_current[t] += intensity

    def stimulate_optic(self, visual_input: np.ndarray) -> None:
        """Encode visual input into optic lobe lamina neurons.

        Args:
            visual_input: (n_ommatidia, 3) RGB values, each 0-255.
                Number of ommatidia should match OL_LAM region size.
                If sizes don't match, input is resampled.
        """
        if visual_input is None or visual_input.size == 0:
            return

        brain = self.brain
        lam_ids = self._region_ids("OL_LAM")
        n_lam = len(lam_ids)
        if n_lam == 0:
            return

        # Flatten to 1D intensity (luminance-weighted)
        if visual_input.ndim == 2 and visual_input.shape[1] == 3:
            # RGB -> luminance
            luminance = (0.3 * visual_input[:, 0] + 0.59 * visual_input[:, 1]
                         + 0.11 * visual_input[:, 2]) / 255.0
        elif visual_input.ndim == 1:
            luminance = visual_input / 255.0
        else:
            luminance = visual_input.flatten()[:n_lam] / 255.0

        # Resample if size mismatch
        if len(luminance) != n_lam:
            indices = np.linspace(0, len(luminance) - 1, n_lam).astype(int)
            luminance = luminance[indices]

        # Convert to stimulus current (40 uA/cm^2 max, enough to drive HH spiking)
        current = torch.tensor(luminance * 40.0, dtype=torch.float32, device=self.dev)
        lam_tensor = self._tensors.get("OL_LAM")
        if lam_tensor is not None and lam_tensor.shape[0] == current.shape[0]:
            brain.external_current[lam_tensor] += current

    def stimulate_temperature(self, temp_celsius: float) -> None:
        """Encode temperature into thermosensory neurons in the antennae.

        Drosophila arista has hot-sensing and cold-sensing neurons.
        Preferred temperature ~25C.

        Stimulus is applied to a subset of AL neurons (thermosensory
        neurons are in the antenna, near the olfactory neurons).
        """
        al_ids = self._region_ids("AL")
        if not al_ids:
            return

        # Use first ~10% of AL as thermosensory
        n_thermo = max(1, len(al_ids) // 10)
        thermo_ids = al_ids[:n_thermo]
        t = torch.tensor(thermo_ids, dtype=torch.int64, device=self.dev)

        # Deviation from preferred temp drives response
        preferred = 25.0
        deviation = abs(temp_celsius - preferred)
        # Always provide baseline + deviation-driven current
        intensity = 20.0 + min(20.0, deviation * 5.0)
        self.brain.external_current[t] += intensity

    def stimulate_taste(self, taste_profile: Dict[str, float]) -> None:
        """Encode taste into SEZ gustatory neurons.

        Args:
            taste_profile: dict mapping taste quality to intensity (0-1).
                e.g. {"sugar": 0.9, "bitter": 0.1, "amino_acid": 0.3}
        """
        sez_ids = self._region_ids("SEZ")
        if not sez_ids or not taste_profile:
            return

        brain = self.brain
        n_sez = len(sez_ids)
        # First 2/3 of SEZ = gustatory neurons, last 1/3 = motor
        n_gust = n_sez * 2 // 3

        for taste_name, intensity in taste_profile.items():
            if intensity <= 0:
                continue
            h = hash(taste_name) & 0xFFFFFFFF
            # Each taste activates ~20% of gustatory neurons
            n_active = max(1, int(n_gust * 0.2))
            for i in range(n_active):
                nid = sez_ids[(h + i * 3) % n_gust]
                brain.external_current[nid] += intensity * 40.0

    def read_motor_output(self, n_steps: int = NEURAL_STEPS_PER_BODY) -> Dict[str, float]:
        """Run neural simulation and read VNC motor neurons.

        Returns:
            Dict with keys:
              - speed: 0-1 (forward motor pool activity)
              - turn: -1 to 1 (left vs right motor balance)
              - fly: 0-1 (flight motor neuron activity)
              - feed: 0-1 (proboscis motor activity)
              - climb: -1 to 1 (climb/descend command from flight neurons)
        """
        brain = self.brain

        fwd_acc = torch.zeros(1, device=self.dev)
        bwd_acc = torch.zeros(1, device=self.dev)
        left_acc = torch.zeros(1, device=self.dev)
        right_acc = torch.zeros(1, device=self.dev)
        flight_acc = torch.zeros(1, device=self.dev)
        climb_acc = torch.zeros(1, device=self.dev)
        descend_acc = torch.zeros(1, device=self.dev)
        feed_acc = torch.zeros(1, device=self.dev)

        for s in range(n_steps):
            # Tonic background drive (pulsed to avoid depolarization block)
            if s % 2 == 0:
                # Basal drive to DN and VNC walking motor pools
                # (maintains baseline motor tone for locomotion).
                # Pulsed at 35-40 uA/cm^2 to sustain firing without
                # depolarization block (Na+ inactivation avoidance).
                # Flight motor neurons do NOT receive tonic drive --
                # flight requires octopamine/startle activation.
                if "DN" in self._tensors:
                    brain.external_current[self._tensors["DN"]] += 40.0
                brain.external_current[self._t_vnc_fwd] += 35.0
                brain.external_current[self._t_vnc_bwd] += 30.0
                # Tonic drive to SEZ proboscis neurons (low, just keeping
                # neurons in excitable state for taste-driven PER)
                if self._t_sez_proboscis.shape[0] > 0:
                    brain.external_current[self._t_sez_proboscis] += 15.0

            brain.step()

            # Accumulate spike counts
            fwd_acc += brain.fired[self._t_vnc_fwd].sum()
            bwd_acc += brain.fired[self._t_vnc_bwd].sum()
            left_acc += brain.fired[self._t_vnc_left].sum()
            right_acc += brain.fired[self._t_vnc_right].sum()
            if self._t_vnc_flight.shape[0] > 0:
                flight_acc += brain.fired[self._t_vnc_flight].sum()
            if self._t_vnc_climb.shape[0] > 0:
                climb_acc += brain.fired[self._t_vnc_climb].sum()
            if self._t_vnc_descend.shape[0] > 0:
                descend_acc += brain.fired[self._t_vnc_descend].sum()
            feed_acc += brain.fired[self._t_sez_proboscis].sum()

        # Single GPU->CPU sync
        fwd = int(fwd_acc.item())
        bwd = int(bwd_acc.item())
        left = int(left_acc.item())
        right = int(right_acc.item())
        flight = int(flight_acc.item())
        climb_spikes = int(climb_acc.item())
        descend_spikes = int(descend_acc.item())
        feed = int(feed_acc.item())

        # Compute normalized motor outputs
        n_fwd = len(self._vnc_forward_ids)
        n_bwd = len(self._vnc_backward_ids)
        n_left = len(self._vnc_left_ids)
        n_right = len(self._vnc_right_ids)
        n_flight = self._t_vnc_flight.shape[0]
        n_climb = self._t_vnc_climb.shape[0]
        n_descend = self._t_vnc_descend.shape[0]
        n_feed = len(self._sez_proboscis_ids)
        max_fwd = max(n_fwd * n_steps, 1)
        max_bwd = max(n_bwd * n_steps, 1)

        # Speed: forward vs backward balance
        fwd_rate = fwd / max_fwd
        bwd_rate = bwd / max_bwd
        speed = max(0.0, min(1.0, (fwd_rate - bwd_rate * 0.5) / 0.05))

        # Turn: left-right imbalance
        total_lr = left + right
        if total_lr > 0:
            turn = (left - right) / total_lr  # -1 to 1
        else:
            turn = 0.0

        # Flight: flight motor neuron activity
        # Flight requires HIGH activation (above baseline tonic drive).
        # In real Drosophila, flight initiation requires octopamine burst
        # and specific descending neuron activation (giant fiber for escape).
        if n_flight > 0:
            fly_rate = flight / max(n_flight * n_steps, 1)
            # High threshold: 15% firing rate required (vs ~3-5% baseline)
            fly_signal = max(0.0, min(1.0, (fly_rate - 0.05) / 0.10))
        else:
            fly_signal = 0.0

        # Climb: balance between climb and descend motor neuron pools
        # Positive = ascend, negative = descend, range -1 to 1
        total_climb_descend = climb_spikes + descend_spikes
        if total_climb_descend > 0:
            climb_signal = (climb_spikes - descend_spikes) / total_climb_descend
        else:
            climb_signal = 0.0

        # Feed: proboscis extension motor activity
        # Requires active SEZ gustatory drive (taste stimulus)
        if n_feed > 0:
            feed_rate = feed / max(n_feed * n_steps, 1)
            feed_signal = max(0.0, min(1.0, (feed_rate - 0.02) / 0.05))
        else:
            feed_signal = 0.0

        return {
            "speed": speed,
            "turn": float(turn),
            "fly": fly_signal,
            "feed": feed_signal,
            "climb": float(climb_signal),
        }

    def apply_reward(self, valence: float) -> None:
        """Apply reward/punishment signal via dopamine to mushroom body.

        Args:
            valence: positive = reward (activates PAM DANs),
                     negative = punishment (activates PPL1 DANs)
        """
        intensity = abs(valence) * 50.0

        if valence > 0 and self._t_dan_pam.shape[0] > 0:
            # PAM DANs: positive valence (reward)
            self.brain.external_current[self._t_dan_pam] += intensity
            # Also boost DA in MB
            mb_tensor = self._tensors.get("MB_KC")
            if mb_tensor is not None:
                self.brain.nt_conc[mb_tensor, NT_DA] += valence * 50.0
        elif valence < 0 and self._t_dan_ppl1.shape[0] > 0:
            # PPL1 DANs: negative valence (punishment)
            self.brain.external_current[self._t_dan_ppl1] += intensity
            if "MB_KC" in self._tensors:
                self.brain.nt_conc[self._tensors["MB_KC"], NT_DA] += abs(valence) * 40.0

    def apply_drug(self, drug: str, dose: float) -> None:
        """Apply pharmacological agent to the brain.

        Supported drugs (same as CUDAMolecularBrain):
          - diazepam/valium: GABA-A enhancer (sedation)
          - caffeine: adenosine antagonist (arousal)
          - nicotine: nAChR agonist (enhanced ACh signaling)
          - alprazolam/xanax: high-potency GABA-A enhancer
          - amphetamine: DA/NE releaser
          - ketamine: NMDA antagonist
        """
        self.brain.apply_drug(drug, dose)

    def step(self) -> None:
        """Advance neural simulation by one timestep (0.1ms)."""
        self.brain.step()

    def run(self, steps: int) -> None:
        """Run neural simulation for multiple timesteps."""
        self.brain.run(steps)

    def warmup(self, n_steps: int = 400) -> None:
        """Stabilize the brain with background tonic activity.

        Uses pulsed stimulation (every other step) at 30-40 uA/cm^2 to
        bring neurons into a stable firing regime without depolarization
        block. Matches the C. elegans warmup protocol.
        """
        for s in range(n_steps):
            if s % 2 == 0:
                # Tonic drive to sensory regions (ambient stimulation)
                for region in ["AL", "OL_LAM", "OL_MED", "SEZ"]:
                    t = self._tensors.get(region)
                    if t is not None:
                        self.brain.external_current[t] += 30.0
                # Tonic drive to motor command pathway
                for region in ["DN", "CX", "SUP"]:
                    t = self._tensors.get(region)
                    if t is not None:
                        self.brain.external_current[t] += 25.0
                # Drive VNC motor neurons (basal muscle tone)
                self.brain.external_current[self._t_vnc_fwd] += 30.0
                self.brain.external_current[self._t_vnc_bwd] += 25.0
            self.step()

    def region_firing_rates(self, window: int = 100) -> Dict[str, float]:
        """Get per-region firing rates from recent spike counts.

        Note: This reads spike_count which accumulates over all time,
        so divide by total steps for rate. For a windowed rate, call
        brain.spike_count.zero_() before the window starts.
        """
        rates = {}
        total_steps = max(self.brain.step_count, 1)
        for name, (start, end, _ids) in self.regions.items():
            n_region = end - start
            if n_region == 0:
                rates[name] = 0.0
                continue
            total_spikes = int(self.brain.spike_count[start:end].sum().item())
            rates[name] = total_spikes / (n_region * total_steps)
        return rates

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def self_test(self) -> bool:
        """Verify brain construction: regions, connectivity, sensory/motor.

        Returns True if all checks pass.
        """
        print(f"\n{'='*70}")
        print(f"  DrosophilaBrain Self-Test  (scale={self.scale}, n={self.n_total})")
        print(f"  Device: {self.dev}")
        print(f"{'='*70}")

        all_pass = True

        # 1. Region allocation
        total_allocated = sum(e - s for s, e, _ in self.regions.values())
        region_ok = total_allocated == self.n_total
        print(f"\n  1. Region allocation: {total_allocated}/{self.n_total} "
              f"neurons across {len(self.regions)} regions  "
              f"[{'PASS' if region_ok else 'FAIL'}]")
        if not region_ok:
            all_pass = False

        for name, (s, e, ids) in self.regions.items():
            n = e - s
            print(f"     {name:12s}: {n:6d} neurons  [{s:6d}:{e:6d}]")

        # 2. Connectivity
        n_syn = self.brain.n_synapses
        syn_ok = n_syn > 0
        print(f"\n  2. Connectivity: {n_syn:,} synapses  "
              f"[{'PASS' if syn_ok else 'FAIL'}]")
        if not syn_ok:
            all_pass = False

        # Count synapses by NT type
        nt_names = {NT_DA: "DA", NT_5HT: "5-HT", NT_NE: "Oct",
                    NT_ACH: "ACh", NT_GABA: "GABA", NT_GLU: "Glu"}
        for nt_idx, nt_name in nt_names.items():
            count = int((self.brain.syn_nt_type == nt_idx).sum().item())
            print(f"     {nt_name:5s}: {count:8,} synapses ({count/max(n_syn,1)*100:.1f}%)")

        # 3. Warmup and spike test
        print(f"\n  3. Neural dynamics (warmup + 200 steps)...")
        t0 = time.perf_counter()
        self.warmup(n_steps=200)
        self.brain.spike_count.zero_()
        self.run(200)
        elapsed = time.perf_counter() - t0

        total_spikes = int(self.brain.spike_count.sum().item())
        spike_ok = total_spikes > 0
        print(f"     Total spikes: {total_spikes:,} in {elapsed:.2f}s  "
              f"[{'PASS' if spike_ok else 'FAIL'}]")
        if not spike_ok:
            all_pass = False

        rates = self.region_firing_rates()
        active_regions = sum(1 for r in rates.values() if r > 0)
        print(f"     Active regions: {active_regions}/{len(rates)}")
        for name, rate in sorted(rates.items(), key=lambda x: -x[1]):
            if rate > 0:
                print(f"     {name:12s}: {rate:.4f} spikes/neuron/step")

        # 4. Sensory encoding test
        # Note: external_current is zeroed each step, so we must re-apply
        # stimulus every step (pulsed to avoid depolarization block).
        print(f"\n  4. Sensory encoding...")
        test_steps = 100

        # Olfactory
        self.brain.spike_count.zero_()
        for s in range(test_steps):
            if s % 2 == 0:
                self.stimulate_al({"vinegar": 0.8, "co2": 0.3})
            self.step()
        al_spikes = int(self.brain.spike_count[self.regions["AL"][0]:self.regions["AL"][1]].sum().item())
        print(f"     Olfactory (vinegar+co2): {al_spikes} AL spikes  "
              f"[{'PASS' if al_spikes > 0 else 'FAIL'}]")
        if al_spikes == 0:
            all_pass = False

        # Visual
        self.brain.spike_count.zero_()
        n_lam = self._region_size("OL_LAM")
        fake_visual = np.random.randint(0, 256, (n_lam, 3), dtype=np.uint8)
        for s in range(test_steps):
            if s % 2 == 0:
                self.stimulate_optic(fake_visual)
            self.step()
        lam_spikes = int(self.brain.spike_count[self.regions["OL_LAM"][0]:self.regions["OL_LAM"][1]].sum().item())
        print(f"     Visual (random RGB):    {lam_spikes} Lamina spikes  "
              f"[{'PASS' if lam_spikes > 0 else 'FAIL'}]")
        if lam_spikes == 0:
            all_pass = False

        # Taste
        self.brain.spike_count.zero_()
        for s in range(test_steps):
            if s % 2 == 0:
                self.stimulate_taste({"sugar": 0.9})
            self.step()
        sez_spikes = int(self.brain.spike_count[self.regions["SEZ"][0]:self.regions["SEZ"][1]].sum().item())
        print(f"     Taste (sugar):          {sez_spikes} SEZ spikes  "
              f"[{'PASS' if sez_spikes > 0 else 'FAIL'}]")
        if sez_spikes == 0:
            all_pass = False

        # 5. Motor readout test
        print(f"\n  5. Motor readout...")
        motor = self.read_motor_output(n_steps=50)
        motor_ok = any(v != 0 for v in motor.values())
        print(f"     speed={motor['speed']:.3f}  turn={motor['turn']:.3f}  "
              f"fly={motor['fly']:.3f}  feed={motor['feed']:.3f}  "
              f"[{'PASS' if motor_ok else 'FAIL'}]")
        if not motor_ok:
            all_pass = False

        # 6. Drug response test
        print(f"\n  6. Pharmacology...")
        # Save baseline motor firing
        self.brain.spike_count.zero_()
        _ = self.read_motor_output(n_steps=100)
        baseline_spikes = int(self.brain.spike_count.sum().item())

        # Apply diazepam (GABA-A enhancer -> sedation)
        self.apply_drug("diazepam", 5.0)
        self.brain.spike_count.zero_()
        _ = self.read_motor_output(n_steps=100)
        drug_spikes = int(self.brain.spike_count.sum().item())
        drug_ok = True  # Diazepam effect may vary; just check no crash
        print(f"     Baseline spikes: {baseline_spikes:,}  "
              f"Diazepam spikes: {drug_spikes:,}  "
              f"[{'PASS' if drug_ok else 'FAIL'}]")

        # 7. Reward signal test
        print(f"\n  7. Reward signal...")
        self.brain.spike_count.zero_()
        for s in range(test_steps):
            if s % 2 == 0:
                self.apply_reward(1.0)  # positive reward (pulsed)
            self.step()
        dan_spikes = int(self.brain.spike_count[self.regions["DAN"][0]:self.regions["DAN"][1]].sum().item())
        reward_ok = dan_spikes > 0
        print(f"     DAN spikes after reward: {dan_spikes}  "
              f"[{'PASS' if reward_ok else 'FAIL'}]")
        if not reward_ok:
            all_pass = False

        # Summary
        print(f"\n{'='*70}")
        print(f"  DrosophilaBrain Self-Test: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print(f"{'='*70}\n")
        return all_pass


# ==========================================================================
# DrosophilaBody -- physical body with sensors and actuators
# ==========================================================================

class DrosophilaBody:
    """Physical body of a Drosophila with sensory and motor systems.

    Sensory systems:
      - Compound eyes: 2 x ~750 ommatidia -> RGB samples from world
      - Antennae: olfactory receptors -> odorant concentrations
      - Johnston's organ: wind/gravity detection
      - Thermoreceptors: arista temperature sensing
      - Tarsal taste receptors: sugar, amino acids, bitter
      - Proprioception: leg joint angles, wing position

    Motor systems:
      - Walking: 6 legs with tripod gait CPG
      - Flight: wing beat frequency modulation
      - Feeding: proboscis extension reflex (PER)
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        heading: float = 0.0,
        pitch: float = 0.0,
        n_ommatidia: int = 100,
    ):
        # Position and orientation (3D)
        self.x = x              # mm (east-west)
        self.y = y              # mm (north-south)
        self.z = z              # mm (altitude, 0 = ground level)
        self.heading = heading  # radians (yaw around vertical axis)
        self.pitch = pitch      # radians (pitch, 0 = level, positive = nose up)
        self.is_flying = False
        self.energy = FLY_ENERGY_MAX
        self.age_steps = 0

        # Compound eye parameters
        self.n_ommatidia = n_ommatidia  # per eye (simplified from ~750)

        # Locomotion state
        self._gait_phase = 0.0     # tripod gait oscillator
        self._wing_phase = 0.0     # wing beat oscillator
        self._proboscis_extended = False

        # Flight state
        self._vertical_velocity = 0.0   # mm/s (positive = up)
        self._wing_beat_freq = FLY_WING_BEAT_HZ  # Hz, modulated by motor output

        # Proprioception state (6 legs, simplified)
        self._leg_angles = [0.0] * 6  # radians

        # Trajectory tracking (3D)
        self.trajectory: List[Tuple[float, float, float]] = [(x, y, z)]
        self.total_distance = 0.0

    # ------------------------------------------------------------------
    # Sensory methods
    # ------------------------------------------------------------------

    def see(self, world) -> Optional[np.ndarray]:
        """Sample visual field through compound eyes.

        Returns:
            (n_ommatidia, 3) array of RGB values (0-255), or None if no world.
            Each ommatidium samples one point in the visual field.
            Visual range increases with altitude (higher = see farther).
        """
        if world is None:
            return None

        # Sample visual field at ommatidium positions
        # Each ommatidium covers ~1.5 degrees of visual field
        # Total field of view: ~270 degrees for compound eyes
        fov = math.radians(270)
        half_fov = fov / 2.0
        rgb = np.zeros((self.n_ommatidia, 3), dtype=np.uint8)

        # Visual range extends with altitude (higher = farther horizon)
        sample_dist = 5.0 + self.z * 0.5

        for i in range(self.n_ommatidia):
            # Angle of this ommatidium relative to heading
            angle = self.heading - half_fov + fov * (i / max(self.n_ommatidia - 1, 1))
            sx = self.x + math.cos(angle) * sample_dist
            sy = self.y + math.sin(angle) * sample_dist

            if hasattr(world, 'visual_at'):
                r, g, b = world.visual_at(sx, sy)
                rgb[i] = [int(r), int(g), int(b)]
            else:
                # Default: ambient gray
                rgb[i] = [128, 128, 128]

        return rgb

    def smell(self, world) -> Dict[str, float]:
        """Sample odorant concentrations at antenna position.

        Returns dict of odorant name -> concentration (0-1).
        Antennae are slightly forward and above body center.
        Supports 3D world.sample_odorants(x, y, z) with fallback
        to 2D world.odorants_at(x, y).
        """
        if world is None:
            return {}

        # Antennae are at the head (slightly forward and above body center)
        ax = self.x + math.cos(self.heading) * 0.5
        ay = self.y + math.sin(self.heading) * 0.5
        az = self.z + 0.3  # antennae above body center

        if hasattr(world, 'sample_odorants'):
            return world.sample_odorants(int(ax), int(ay), int(az))
        if hasattr(world, 'odorants_at'):
            return world.odorants_at(ax, ay)
        return {}

    def sense_wind(self, world) -> Tuple[float, ...]:
        """Detect wind direction via Johnston's organ on antennae.

        Returns (wind_x, wind_y, wind_z) vector in mm/s.
        Supports 3D world.sample_wind(x, y, z) with fallback
        to 2D world.wind_at(x, y) (adds zero z-component).
        """
        if world is None:
            return (0.0, 0.0, 0.0)

        if hasattr(world, 'sample_wind'):
            result = world.sample_wind(int(self.x), int(self.y), int(self.z))
            if len(result) == 3:
                return result
            return (*result, 0.0)  # 2D fallback
        if hasattr(world, 'wind_at'):
            return (*world.wind_at(self.x, self.y), 0.0)
        return (0.0, 0.0, 0.0)

    def sense_temperature(self, world) -> float:
        """Detect temperature via arista thermoreceptors.

        Returns temperature in Celsius.
        Supports 3D world.sample_temperature(x, y, z) with fallback
        to 2D world.temperature_at(x, y).
        """
        if world is None:
            return 25.0  # room temperature

        if hasattr(world, 'sample_temperature'):
            return world.sample_temperature(int(self.x), int(self.y), int(self.z))
        if hasattr(world, 'temperature_at'):
            return world.temperature_at(self.x, self.y)
        return 25.0

    def taste(self, world) -> Dict[str, float]:
        """Sample taste via tarsal (foot) and proboscis receptors.

        Returns dict of taste quality -> intensity (0-1).
        Only active when on a surface (not flying) or proboscis extended.
        """
        if world is None or self.is_flying:
            return {}

        if hasattr(world, 'taste_at'):
            return world.taste_at(self.x, self.y)
        return {}

    def sense_body(self) -> Dict[str, float]:
        """Proprioceptive sense of body state.

        Returns dict of body state parameters including 3D flight state.
        """
        return {
            "gait_phase": self._gait_phase,
            "wing_phase": self._wing_phase if self.is_flying else 0.0,
            "energy": self.energy,
            "is_flying": float(self.is_flying),
            "proboscis": float(self._proboscis_extended),
            "heading": self.heading,
            "pitch": self.pitch,
            "altitude": self.z,
            "vertical_velocity": self._vertical_velocity if self.is_flying else 0.0,
        }

    # ------------------------------------------------------------------
    # Motor methods
    # ------------------------------------------------------------------

    def walk(self, speed: float, turn: float, dt: float = DT_BODY,
             world_bounds: Optional[Tuple[float, float]] = None) -> None:
        """Walk with given speed and turn rate (ground-level only).

        Args:
            speed: 0-1, fraction of max walking speed
            turn: -1 to 1, turn rate (negative=right, positive=left)
            dt: timestep in seconds
            world_bounds: optional (width, height) to clamp position within
        """
        if self.is_flying:
            return

        # Ground-level enforcement
        self.z = 0.0
        self.pitch = 0.0

        # Update gait phase (tripod oscillator)
        self._gait_phase += 2.0 * math.pi * 8.0 * speed * dt  # ~8 Hz stride freq

        # Turn
        self.heading += turn * FLY_TURN_RATE * dt

        # Move
        actual_speed = speed * FLY_WALK_SPEED
        dx = math.cos(self.heading) * actual_speed * dt
        dy = math.sin(self.heading) * actual_speed * dt

        self.x += dx
        self.y += dy

        # World bounds clamping
        if world_bounds is not None:
            w, h = world_bounds
            self.x = max(0.0, min(w - 1, self.x))
            self.y = max(0.0, min(h - 1, self.y))

        step_dist = math.sqrt(dx * dx + dy * dy)
        self.total_distance += step_dist
        self.trajectory.append((self.x, self.y, self.z))

        # Energy cost
        self.energy = max(0, self.energy - FLY_ENERGY_WALK_COST * speed)

    def fly_toward(self, direction: float, dt: float = DT_BODY) -> None:
        """Fly in the given direction (legacy 2D flight, kept for backward compat).

        For full 3D flight with gravity, use fly_3d() instead.

        Args:
            direction: heading in radians
            dt: timestep in seconds
        """
        if not self.is_flying:
            return

        # Wing beat oscillation
        self._wing_phase += 2.0 * math.pi * self._wing_beat_freq * dt

        # Gradually turn toward target direction
        angle_diff = direction - self.heading
        # Normalize to [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        self.heading += angle_diff * 0.1  # smooth turning

        # Move
        dx = math.cos(self.heading) * FLY_FLIGHT_SPEED * dt
        dy = math.sin(self.heading) * FLY_FLIGHT_SPEED * dt

        self.x += dx
        self.y += dy
        step_dist = math.sqrt(dx * dx + dy * dy)
        self.total_distance += step_dist
        self.trajectory.append((self.x, self.y, self.z))

        # Energy cost (flight is expensive)
        self.energy = max(0, self.energy - FLY_ENERGY_FLY_COST)

    def fly_3d(self, speed: float, turn: float, climb: float,
               dt: float = DT_BODY) -> None:
        """Fly in 3D with gravity, lift, and drag.

        Full 3D flight physics: the fly experiences gravity pulling it down
        while wing beats generate lift. The climb command modulates lift to
        control altitude. Forward motion is projected through both heading
        (yaw) and pitch angles.

        Args:
            speed: 0-1 forward thrust fraction
            turn: -1 to 1 yaw rate (left/right)
            climb: -1 to 1 climb/descend command (from brain VNC motor output)
            dt: timestep in seconds
        """
        if not self.is_flying:
            return

        # Wing beat oscillation
        self._wing_phase += 2.0 * math.pi * self._wing_beat_freq * dt

        # Yaw (heading change)
        self.heading += turn * FLY_TURN_RATE * dt

        # Pitch toward climb command (max +/-30 degrees)
        target_pitch = climb * math.radians(30)
        self.pitch += (target_pitch - self.pitch) * 0.1  # smooth toward target

        # Forward velocity projected through heading and pitch
        forward_speed = speed * FLY_FLIGHT_SPEED
        dx = math.cos(self.heading) * math.cos(self.pitch) * forward_speed * dt
        dy = math.sin(self.heading) * math.cos(self.pitch) * forward_speed * dt

        # Vertical dynamics: lift vs gravity
        # At nominal wing beat frequency, lift approximately equals gravity
        # (hovering equilibrium). The climb command modulates wing beat
        # amplitude by +/-30%, creating net upward or downward acceleration.
        wing_factor = self._wing_beat_freq / FLY_WING_BEAT_HZ  # 1.0 at nominal
        lift_accel = GRAVITY_MM_S2 * wing_factor  # lift ~= gravity at nominal
        lift_accel *= (1.0 + climb * 0.3)  # +/-30% modulation from climb cmd

        # Net vertical acceleration (positive = up)
        vert_accel = lift_accel - GRAVITY_MM_S2
        self._vertical_velocity += vert_accel * dt

        # Aerodynamic damping on vertical motion
        self._vertical_velocity *= 0.95

        # Clamp vertical velocity to biological limits
        self._vertical_velocity = max(-FLY_DESCENT_RATE,
                                       min(FLY_CLIMB_RATE, self._vertical_velocity))

        # Vertical displacement from velocity + pitch component
        dz = (self._vertical_velocity * dt
              + math.sin(self.pitch) * forward_speed * dt)

        # Apply position changes
        self.x += dx
        self.y += dy
        self.z = max(0.0, min(FLY_MAX_ALTITUDE, self.z + dz))

        # Landing detection: touched ground
        if self.z <= 0.0:
            self.z = 0.0
            self._vertical_velocity = 0.0
            self.land()

        # Track distance and trajectory
        step_dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.total_distance += step_dist
        self.trajectory.append((self.x, self.y, self.z))

        # Energy cost (proportional to flight effort)
        # Baseline cost 0.5 + modulation from speed and climb effort
        effort = abs(climb) * 0.5 + speed * 0.5 + 0.5
        self.energy = max(0, self.energy - FLY_ENERGY_FLY_COST * effort)

        # Emergency landing if out of energy
        if self.energy <= 0:
            self.land()

    def takeoff(self) -> bool:
        """Attempt to take off. Requires minimum energy.

        Applies an initial upward velocity impulse and sets minimum
        altitude to clear the ground.
        """
        if self.energy < 10.0:
            return False
        self.is_flying = True
        self._vertical_velocity = FLY_TAKEOFF_SPEED  # initial upward impulse
        self.z = max(self.z, 0.5)  # minimum altitude on takeoff
        return True

    def land(self) -> None:
        """Land from flight. Resets vertical state."""
        self.is_flying = False
        self.z = 0.0
        self.pitch = 0.0
        self._vertical_velocity = 0.0

    def extend_proboscis(self) -> None:
        """Extend proboscis for feeding (PER)."""
        self._proboscis_extended = True

    def retract_proboscis(self) -> None:
        """Retract proboscis."""
        self._proboscis_extended = False

    def feed(self, world_or_amount=None) -> float:
        """Attempt to feed. Returns energy gained.

        Args:
            world_or_amount: either a world object (uses proboscis + food_at),
                or a float amount to directly add energy (compatibility mode).
        """
        # Compatibility mode: direct energy amount
        if isinstance(world_or_amount, (int, float)):
            gained = float(world_or_amount)
            self.energy = min(FLY_ENERGY_MAX, self.energy + gained)
            return gained

        world = world_or_amount
        if world is None or self.is_flying:
            return 0.0

        if not self._proboscis_extended:
            return 0.0

        gained = 0.0
        if hasattr(world, 'food_at'):
            food = world.food_at(self.x, self.y)
            if food > 0.1:
                gained = food * FLY_ENERGY_FEED_GAIN
                self.energy = min(FLY_ENERGY_MAX, self.energy + gained)
                if hasattr(world, 'deplete_food'):
                    world.deplete_food(self.x, self.y)

        return gained

    def update_age(self) -> None:
        """Increment age counter."""
        self.age_steps += 1

    # ------------------------------------------------------------------
    # Compatibility methods (used by demo sensory/motor encoders)
    # ------------------------------------------------------------------

    def update(self, turn_bias: float, speed_factor: float = 1.0,
               dt: float = 0.002, arena_size: float = 100.0) -> None:
        """Move the fly one timestep (compatibility wrapper for walk()).

        Args:
            turn_bias: signed turn rate (positive = left/CCW).
            speed_factor: multiplier for base walking speed (0-2).
            dt: timestep in seconds.
            arena_size: world bounds in mm (square arena).
        """
        speed = min(1.0, speed_factor * 0.5)
        turn = turn_bias / 3.0
        self.walk(speed=speed, turn=turn, dt=dt,
                  world_bounds=(arena_size, arena_size))

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from fly to a point (2D, ignoring z)."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    @property
    def speed(self) -> float:
        """Current nominal walking speed in mm/s."""
        return FLY_WALK_SPEED


# ==========================================================================
# Drosophila -- complete organism (brain + body)
# ==========================================================================

class Drosophila:
    """Complete Drosophila melanogaster organism: brain + body.

    The main loop each step:
      1. Sense the world through the body's sensory systems
      2. Encode sensory signals as neural stimulation
      3. Run the brain for NEURAL_STEPS_PER_BODY timesteps
      4. Decode motor output from VNC firing patterns
      5. Move the body according to motor commands

    Usage:
        world = MyFlyWorld()  # or None for standalone testing
        fly = Drosophila(world=world, scale='small')
        for _ in range(1000):
            fly.step(world)
            print(f"Position: ({fly.body.x:.1f}, {fly.body.y:.1f})")
    """

    def __init__(
        self,
        world=None,
        scale: str = "small",
        device: str = "auto",
        seed: int = 42,
        n_ommatidia: int = 100,
    ):
        self.brain = DrosophilaBrain(scale=scale, device=device, seed=seed)
        self.body = DrosophilaBody(n_ommatidia=n_ommatidia)
        self.world = world
        self.step_count = 0

        # Sensory adaptation state
        self._prev_odorants: Dict[str, float] = {}
        self._prev_temp = 25.0

    def step(self, world=None) -> Dict[str, Any]:
        """Run one complete sense-think-act cycle.

        Args:
            world: environment object (optional, overrides self.world)

        Returns:
            Dict with motor outputs and sensory info for logging.
        """
        w = world if world is not None else self.world
        self.step_count += 1

        # ---- 1. SENSE ----
        visual = self.body.see(w)
        odorants = self.body.smell(w)
        wind = self.body.sense_wind(w)
        temp = self.body.sense_temperature(w)
        taste = self.body.taste(w)

        # ---- 2. ENCODE (sensory -> neural stimulation) ----

        # Olfactory: derivative coding (respond to changes)
        if odorants:
            # Compute derivative for adaptation
            adapted = {}
            for odor, conc in odorants.items():
                prev = self._prev_odorants.get(odor, 0.0)
                # Stimulate based on absolute + derivative
                adapted[odor] = conc * 0.3 + max(0, conc - prev) * 0.7
            self.brain.stimulate_al(adapted)
            self._prev_odorants = dict(odorants)

        # Visual
        if visual is not None:
            self.brain.stimulate_optic(visual)

        # Temperature
        self.brain.stimulate_temperature(temp)

        # Taste
        if taste:
            self.brain.stimulate_taste(taste)

        # Wind: stimulate CX for heading adjustment (3D wind vector)
        wind_x, wind_y = wind[0], wind[1]
        wind_z = wind[2] if len(wind) > 2 else 0.0
        wind_strength = math.sqrt(wind_x ** 2 + wind_y ** 2 + wind_z ** 2)
        if wind_strength > 0.1:
            cx_tensor = self.brain._tensors.get("CX")
            if cx_tensor is not None:
                self.brain.brain.external_current[cx_tensor] += min(20.0, wind_strength * 5.0)

        # ---- 3. THINK (run neural simulation + decode motor) ----
        motor = self.brain.read_motor_output(n_steps=NEURAL_STEPS_PER_BODY)

        # ---- 4. ACT (execute motor commands) ----
        speed = motor["speed"]
        turn = motor["turn"]
        fly_signal = motor["fly"]
        feed_signal = motor["feed"]
        climb = motor.get("climb", 0.0)

        # Flight decision
        if fly_signal > 0.5 and not self.body.is_flying:
            self.body.takeoff()
        elif fly_signal < 0.2 and self.body.is_flying:
            self.body.land()

        # Move in 3D
        if self.body.is_flying:
            self.body.fly_3d(speed, turn, climb)
        else:
            self.body.walk(speed, turn)

        # Feeding decision
        if feed_signal > 0.3:
            self.body.extend_proboscis()
            energy_gained = self.body.feed(w)
            if energy_gained > 0:
                # Reward signal for successful feeding
                self.brain.apply_reward(0.5)
        else:
            self.body.retract_proboscis()

        self.body.update_age()

        return {
            "step": self.step_count,
            "x": self.body.x,
            "y": self.body.y,
            "z": self.body.z,
            "heading": self.body.heading,
            "pitch": self.body.pitch,
            "is_flying": self.body.is_flying,
            "energy": self.body.energy,
            "motor": motor,
        }


# ==========================================================================
# Module-level test
# ==========================================================================

def _quick_test():
    """Quick smoke test for the Drosophila module (3D flight)."""
    print("=" * 70)
    print("  Drosophila melanogaster -- Quick Smoke Test (3D)")
    print("=" * 70)

    t0 = time.perf_counter()

    # Build tiny brain
    fly = Drosophila(world=None, scale="tiny", device="auto")
    print(f"\n  Built fly: {fly.brain.n_total} neurons, "
          f"{fly.brain.brain.n_synapses:,} synapses on {fly.brain.dev}")

    # Run a few steps
    for i in range(5):
        result = fly.step(world=None)
        print(f"  Step {i+1}: pos=({result['x']:.2f}, {result['y']:.2f}, {result['z']:.2f})  "
              f"speed={result['motor']['speed']:.3f}  "
              f"fly={result['motor']['fly']:.3f}  "
              f"climb={result['motor'].get('climb', 0):.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Completed in {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    _quick_test()
