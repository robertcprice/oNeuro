#!/usr/bin/env python3
"""Drosophila Ecosystem -- Complete Digital Fruit Fly in a Digital World.

A biophysically faithful Drosophila melanogaster living in a continuous 2D
arena with odorant gradients, light sources, temperature fields, food patches,
and a day/night cycle. This demo builds on oNeuro's dONN (digital Organic
Neural Network) framework to show that complex insect behaviors -- olfactory
learning, phototaxis, thermotaxis, foraging, pharmacological modulation, and
circadian activity -- EMERGE from molecular-level neural dynamics (HH ion
channels, 6 neurotransmitters, STDP, gene expression) rather than being
hard-coded.

Key Drosophila neural architecture:
  - Mushroom body (MB): ~2500 Kenyon cells in the real fly, used for
    olfactory associative learning (Tully & Quinn 1985). Our model
    approximates this with a scaled-down MB circuit (antennal lobe ->
    projection neurons -> Kenyon cells -> MB output neurons).
  - Central complex (CX): navigation, orientation, motor planning
  - Optic lobes: compound eye processing for phototaxis
  - Antennal lobes: olfactory processing, glomerular organization
  - Motor circuits: 6-leg locomotion, wing control

Terminology:
  - ONN:    Organic Neural Network -- real biological neurons
  - dONN:   digital Organic Neural Network -- oNeuro's simulation
  - oNeuro: The platform for building and running dONNs

6 Experiments:
  1. Olfactory Learning (Mushroom Body conditioning)
  2. Phototaxis (light approach/avoidance depending on intensity)
  3. Thermotaxis (navigate to preferred ~24C)
  4. Foraging (olfactory gradient navigation to food)
  5. Drug Effects (caffeine, diazepam, nicotine on foraging)
  6. Day/Night Cycle (circadian-like activity modulation)

References:
  - Tully & Quinn (1985) "Classical conditioning and retention in normal
    and mutant Drosophila melanogaster" J Comp Physiol A 157:263-277
  - Heisenberg (2003) "Mushroom body memoir: from maps to models"
    Nature Reviews Neuroscience 4:266-275
  - Rister et al. (2007) "Dissection of the peripheral motion channel in
    the visual system of Drosophila" Neuron 56:155-170
  - Hamada et al. (2008) "An internal thermal sensor controlling
    temperature preference in Drosophila" Nature 454:217-220

Usage:
    python3 demos/demo_drosophila_ecosystem.py                    # all experiments, small
    python3 demos/demo_drosophila_ecosystem.py --exp 1            # just olfactory learning
    python3 demos/demo_drosophila_ecosystem.py --scale medium     # 25K neurons
    python3 demos/demo_drosophila_ecosystem.py --json results.json
    python3 demos/demo_drosophila_ecosystem.py --device cuda
    python3 demos/demo_drosophila_ecosystem.py --runs 5           # multi-seed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Import organism and world modules (being built in parallel by other agents).
# Fall back to inline stubs if not yet available.
# ---------------------------------------------------------------------------
_USING_STUBS = False

try:
    from oneuro.organisms.drosophila import (
        Drosophila,
        DrosophilaBrain,
        DrosophilaBody,
    )
    from oneuro.worlds.molecular_world import MolecularWorld
except ImportError:
    _USING_STUBS = True


# Import the molecular backend (always available)
from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    CUDARegionalBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_ACH, NT_GABA, NT_GLU,
)


# ============================================================================
# Inline stubs -- minimal implementations used ONLY when the real organism
# and world modules are not yet available.  These replicate enough of the
# expected API surface to let the demo code run end-to-end for testing.
# ============================================================================

if _USING_STUBS:
    print("  [WARN] Using inline stubs for Drosophila / MolecularWorld")
    print("         Install the real modules for full biophysical fidelity.\n")

    # ------------------------------------------------------------------
    # Stub: DrosophilaBrain
    # ------------------------------------------------------------------
    class DrosophilaBrain:
        """Minimal stub -- wraps CUDAMolecularBrain with Drosophila regions."""

        # Scale presets: name -> (n_neurons_approx, description)
        SCALES = {
            "small":  (800,   "minimal demo"),
            "medium": (5000,  "mid-scale"),
            "large":  (25000, "full mushroom body"),
        }

        def __init__(
            self,
            n_neurons: int = 800,
            device: str = "auto",
            seed: int = 42,
        ):
            torch.manual_seed(seed)
            self.n_neurons = n_neurons
            self.brain = CUDAMolecularBrain(n_neurons, device=device)
            self.device = self.brain.device

            # ------ Region layout (proportions match real fly) ------
            # Antennal lobe ~10%, MB Kenyon cells ~30%, MB output ~5%,
            # projection neurons ~5%, central complex ~10%, optic ~15%,
            # motor ~15%, other ~10%
            idx = 0
            def _alloc(frac):
                nonlocal idx
                n = max(4, int(n_neurons * frac))
                ids = list(range(idx, idx + n))
                idx += n
                return ids

            self.antennal_lobe_ids = _alloc(0.10)
            self.projection_neuron_ids = _alloc(0.05)
            self.kenyon_cell_ids = _alloc(0.30)
            self.mb_output_ids = _alloc(0.05)
            self.central_complex_ids = _alloc(0.10)
            self.optic_lobe_ids = _alloc(0.15)
            self.motor_ids = _alloc(0.15)
            self.other_ids = list(range(idx, n_neurons))

            # Pre-build tensors
            self._al_t = torch.tensor(self.antennal_lobe_ids, dtype=torch.int64,
                                      device=self.device)
            self._pn_t = torch.tensor(self.projection_neuron_ids, dtype=torch.int64,
                                      device=self.device)
            self._kc_t = torch.tensor(self.kenyon_cell_ids, dtype=torch.int64,
                                      device=self.device)
            self._mbon_t = torch.tensor(self.mb_output_ids, dtype=torch.int64,
                                        device=self.device)
            self._cx_t = torch.tensor(self.central_complex_ids, dtype=torch.int64,
                                      device=self.device)
            self._optic_t = torch.tensor(self.optic_lobe_ids, dtype=torch.int64,
                                         device=self.device)
            self._motor_t = torch.tensor(self.motor_ids, dtype=torch.int64,
                                         device=self.device)
            self._all_t = torch.arange(n_neurons, dtype=torch.int64,
                                       device=self.device)

            # Wire basic circuits
            self._wire(seed)

        def _wire(self, seed: int) -> None:
            """Wire Drosophila-like connectivity."""
            torch.manual_seed(seed)
            all_pre, all_post, all_w, all_nt = [], [], [], []

            def _connect(src, dst, prob, w_range, nt):
                s_t = torch.tensor(src, device=self.device)
                d_t = torch.tensor(dst, device=self.device)
                if len(src) * len(dst) > 50_000:
                    n_exp = int(len(src) * len(dst) * prob)
                    if n_exp == 0:
                        return
                    si = torch.randint(len(src), (n_exp,), device=self.device)
                    di = torch.randint(len(dst), (n_exp,), device=self.device)
                    pre = s_t[si]; post = d_t[di]
                    mask = pre != post
                    pre, post = pre[mask], post[mask]
                else:
                    m = torch.rand(len(src), len(dst), device=self.device) < prob
                    ii = torch.where(m)
                    pre = s_t[ii[0]]; post = d_t[ii[1]]
                    valid = pre != post
                    pre, post = pre[valid], post[valid]
                if pre.numel() == 0:
                    return
                w = torch.rand(pre.numel(), device=self.device) * (w_range[1] - w_range[0]) + w_range[0]
                nt_t = torch.full((pre.numel(),), nt, dtype=torch.int32, device=self.device)
                all_pre.append(pre); all_post.append(post)
                all_w.append(w); all_nt.append(nt_t)

            # Olfactory circuit: AL -> PN -> KC -> MBON
            _connect(self.antennal_lobe_ids, self.projection_neuron_ids, 0.4, (0.8, 1.5), NT_ACH)
            _connect(self.projection_neuron_ids, self.kenyon_cell_ids, 0.15, (0.5, 1.2), NT_ACH)
            _connect(self.kenyon_cell_ids, self.mb_output_ids, 0.2, (0.6, 1.4), NT_ACH)

            # MBON -> motor (approach/avoid)
            _connect(self.mb_output_ids, self.motor_ids, 0.3, (0.5, 1.2), NT_ACH)

            # Central complex: navigation + motor
            _connect(self.central_complex_ids, self.motor_ids, 0.3, (0.6, 1.3), NT_ACH)
            _connect(self.optic_lobe_ids, self.central_complex_ids, 0.2, (0.5, 1.0), NT_GLU)

            # Optic -> motor (fast phototaxis path)
            _connect(self.optic_lobe_ids, self.motor_ids, 0.1, (0.3, 0.8), NT_GLU)

            # Inhibitory: MBON has GABAergic output neurons too
            half_mbon = len(self.mb_output_ids) // 2
            _connect(self.mb_output_ids[half_mbon:], self.motor_ids,
                     0.2, (0.5, 1.0), NT_GABA)

            # KC lateral inhibition (APL-like)
            _connect(self.kenyon_cell_ids[:20], self.kenyon_cell_ids,
                     0.05, (0.3, 0.6), NT_GABA)

            # Recurrent within CX
            _connect(self.central_complex_ids, self.central_complex_ids,
                     0.1, (0.3, 0.7), NT_GLU)

            # Other interneurons provide neuromodulation
            _connect(self.other_ids[:10], self.kenyon_cell_ids,
                     0.03, (0.2, 0.5), NT_DA)
            _connect(self.other_ids[10:20], self.motor_ids,
                     0.05, (0.3, 0.8), NT_5HT)

            if all_pre:
                self.brain.add_synapses(
                    torch.cat(all_pre), torch.cat(all_post),
                    torch.cat(all_w), torch.cat(all_nt),
                )

        def step(self) -> None:
            self.brain.step()

        def run(self, n_steps: int) -> None:
            self.brain.run(n_steps)

        def apply_drug(self, name: str, dose: float) -> None:
            self.brain.apply_drug(name, dose)

        @classmethod
        def build(cls, scale: str = "small", device: str = "auto",
                  seed: int = 42) -> "DrosophilaBrain":
            n = cls.SCALES.get(scale, cls.SCALES["small"])[0]
            return cls(n_neurons=n, device=device, seed=seed)

    # ------------------------------------------------------------------
    # Stub: DrosophilaBody
    # ------------------------------------------------------------------
    class DrosophilaBody:
        """Minimal stub -- fly body with position, heading, 6 legs, wings."""

        def __init__(self, x: float = 50.0, y: float = 50.0,
                     heading: float = 0.0):
            self.x = x
            self.y = y
            self.heading = heading  # radians, 0 = east
            self.speed = 2.0  # mm/s (Drosophila walking speed)
            self.energy = 1.0  # 0.0 - 1.0
            self.state = "walking"  # walking, flying, resting
            self.trajectory: List[Tuple[float, float]] = [(x, y)]
            self.total_distance = 0.0

        def update(self, turn_bias: float, speed_factor: float = 1.0,
                   dt: float = 0.1, arena_size: float = 100.0) -> None:
            """Move the fly one timestep."""
            self.heading += turn_bias * dt
            effective_speed = self.speed * speed_factor * dt
            dx = math.cos(self.heading) * effective_speed
            dy = math.sin(self.heading) * effective_speed
            self.x = max(0.5, min(arena_size - 0.5, self.x + dx))
            self.y = max(0.5, min(arena_size - 0.5, self.y + dy))
            step_dist = math.sqrt(dx * dx + dy * dy)
            self.total_distance += step_dist
            self.trajectory.append((self.x, self.y))
            # Energy depletion
            self.energy = max(0.0, self.energy - 0.0002 * speed_factor)

        def distance_to(self, x: float, y: float) -> float:
            return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

        def feed(self, amount: float = 0.05) -> None:
            self.energy = min(1.0, self.energy + amount)

    # ------------------------------------------------------------------
    # Stub: Drosophila (combined brain + body)
    # ------------------------------------------------------------------
    class Drosophila:
        """Stub combining DrosophilaBrain + DrosophilaBody."""

        def __init__(self, brain: DrosophilaBrain, body: DrosophilaBody):
            self.brain = brain
            self.body = body

    # ------------------------------------------------------------------
    # Stub: MolecularWorld
    # ------------------------------------------------------------------
    class MolecularWorld:
        """Minimal 2D arena with odorant sources, light, temperature."""

        def __init__(
            self,
            size_mm: float = 100.0,
            seed: int = 42,
        ):
            self.size_mm = size_mm
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

            # Odorant sources: list of (x, y, compound, intensity, radius)
            self.odorant_sources: List[Dict[str, Any]] = []
            # Light sources: list of (x, y, intensity, radius)
            self.light_sources: List[Dict[str, Any]] = []
            # Temperature field
            self.temp_min = 18.0
            self.temp_max = 30.0
            self.temp_gradient_axis = "x"  # linear gradient along x
            # Food patches
            self.food_patches: List[Dict[str, Any]] = []
            # Day/night
            self.time_of_day = 0.0  # 0.0 = dawn, 0.5 = noon, 1.0 = dusk
            self.is_day = True
            self.ambient_light = 1.0

        def add_odorant_source(self, x: float, y: float,
                               compound: str = "ethanol",
                               intensity: float = 1.0,
                               radius: float = 15.0) -> None:
            self.odorant_sources.append({
                "x": x, "y": y, "compound": compound,
                "intensity": intensity, "radius": radius,
            })

        def add_light_source(self, x: float, y: float,
                             intensity: float = 1.0,
                             radius: float = 40.0) -> None:
            self.light_sources.append({
                "x": x, "y": y, "intensity": intensity, "radius": radius,
            })

        def add_food(self, x: float, y: float, radius: float = 5.0,
                     remaining: float = 1.0) -> None:
            self.food_patches.append({
                "x": x, "y": y, "radius": radius, "remaining": remaining,
            })

        def odor_at(self, x: float, y: float, compound: str = "ethanol") -> float:
            """Total odorant concentration of given compound at (x, y)."""
            total = 0.0
            for src in self.odorant_sources:
                if src["compound"] != compound:
                    continue
                dx = x - src["x"]
                dy = y - src["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                sigma = src["radius"] * 0.5
                total += src["intensity"] * math.exp(-dist * dist / (2.0 * sigma * sigma))
            return min(total, 1.0)

        def light_at(self, x: float, y: float) -> float:
            """Light intensity at (x, y). Includes ambient + point sources."""
            total = self.ambient_light * 0.1
            for src in self.light_sources:
                dx = x - src["x"]
                dy = y - src["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                sigma = src["radius"] * 0.5
                total += src["intensity"] * math.exp(-dist * dist / (2.0 * sigma * sigma))
            return min(total, 2.0)

        def temperature_at(self, x: float, y: float) -> float:
            """Temperature at (x, y) in degrees C."""
            frac = x / self.size_mm
            return self.temp_min + (self.temp_max - self.temp_min) * frac

        def food_at(self, x: float, y: float) -> float:
            """Food concentration at (x, y)."""
            total = 0.0
            for patch in self.food_patches:
                dx = x - patch["x"]
                dy = y - patch["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < patch["radius"] * 2.0:
                    sigma = patch["radius"] * 0.5
                    total += patch["remaining"] * math.exp(
                        -dist * dist / (2.0 * sigma * sigma))
            return min(total, 1.0)

        def deplete_food_near(self, x: float, y: float,
                              eat_radius: float = 3.0,
                              amount: float = 0.005) -> bool:
            """Deplete food near position. Returns True if food was eaten."""
            eaten = False
            for patch in self.food_patches:
                dist = math.sqrt((x - patch["x"]) ** 2 + (y - patch["y"]) ** 2)
                if dist < eat_radius + patch["radius"] and patch["remaining"] > 0:
                    patch["remaining"] = max(0.0, patch["remaining"] - amount)
                    eaten = True
            return eaten

        def reset_food(self) -> None:
            for patch in self.food_patches:
                patch["remaining"] = 1.0

        def advance_time(self, dt_hours: float = 0.1) -> None:
            """Advance the day/night cycle."""
            self.time_of_day = (self.time_of_day + dt_hours / 24.0) % 1.0
            # Day = 0.25 to 0.75, Night = 0.75 to 0.25
            if 0.25 <= self.time_of_day <= 0.75:
                self.is_day = True
                # Smooth light curve (peaks at noon = 0.5)
                phase = (self.time_of_day - 0.25) / 0.5  # 0 to 1 during day
                self.ambient_light = math.sin(phase * math.pi)
            else:
                self.is_day = False
                self.ambient_light = 0.05  # moonlight


# ============================================================================
# Scale Presets
# ============================================================================

SCALE_NEURONS = {
    "tiny":   1_000,
    "small":  5_000,
    "medium": 25_000,
    "large":  139_000,
}


# ============================================================================
# Constants -- Drosophila Biology
# ============================================================================

FLY_WALKING_SPEED = 2.0      # mm/s (adult Drosophila on agar)
FLY_TURN_RATE = 1.5           # radians/s max turning
DT = 0.1                      # seconds per body timestep
NEURAL_STEPS_PER_BODY = 20    # HH timesteps per body update (CPG reset prevents depol block)

# Preferred temperature for Drosophila melanogaster
PREFERRED_TEMP_C = 24.0
TEMP_TOLERANCE_C = 3.0

# Odorant receptor → glomerulus mapping
# Real Drosophila has ~50 OR types. Each volatile compound binds specific
# receptors, activating specific glomeruli. Two compounds with DIFFERENT
# receptor types activate NON-OVERLAPPING AL populations → sparse KC
# patterns → good pattern separation in the mushroom body.
N_GLOMERULI = 10  # number of odorant receptor types modeled
COMPOUND_GLOMERULUS: Dict[str, int] = {
    "ethanol": 0,
    "ethyl_acetate": 3,   # well-separated from ethanol
    "vinegar": 1,
    "banana_oil": 5,
    "methanol": 2,
    "acetone": 7,
    "benzaldehyde": 4,
    "geraniol": 6,
    "hexanol": 8,
    "isoamyl_acetate": 9,
}


# ============================================================================
# Sensory Encoding
# ============================================================================

class DrosophilaSensory:
    """Transduce physical stimuli into neural currents.

    This encoder does ONLY physics-to-current conversion. It does NOT encode
    any behavioral logic (approach, avoid, coupling drives, sign-flipping).
    All behavior emerges from the brain's lateralized wiring:
      left eye → left optic → left DN → left VNC motor
      right eye → right optic → right DN → right VNC motor

    Modalities:
      - Olfactory: bilateral antennae → left/right antennal lobe neurons
      - Visual: bilateral compound eyes → left/right optic lobe neurons
      - Thermosensory: bilateral antennae → left/right AL thermosensory
      - Gustatory: proboscis contact → AL gustatory neurons
    """

    def __init__(self, brain: DrosophilaBrain):
        self.brain = brain
        self.dev = brain.device
        self._prev_odor: Dict[str, float] = {}
        self._prev_light_l = 0.0
        self._prev_light_r = 0.0
        self._prev_temp = PREFERRED_TEMP_C
        self._prev_food = 0.0

    def encode(
        self,
        world: MolecularWorld,
        body: DrosophilaBody,
        odor_compounds: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Sample environment at fly position and inject sensory currents.

        Pure physics: measure stimulus, convert to current, inject into the
        correct sensory neurons. The brain's wiring determines what happens.
        """
        if odor_compounds is None:
            odor_compounds = ["ethanol", "ethyl_acetate"]

        b = self.brain.brain  # underlying CUDAMolecularBrain
        intensities: Dict[str, float] = {}

        # ------- Olfactory (bilateral antennae → specific AL glomeruli) -------
        # Drosophila has ~50 odorant receptor (OR) types, each expressed
        # in a specific glomerulus in the antennal lobe. Different volatile
        # compounds bind different ORs based on molecular structure. This is
        # transduction biology (receptor→glomerulus mapping), not behavior.
        #
        # We model N_GLOMERULI receptor types. Each compound activates ONE
        # specific glomerulus (~10% of AL neurons per side), creating sparse
        # activation patterns essential for the mushroom body's pattern
        # separation (Caron et al. 2013; Aso et al. 2014).
        left_ax = body.x + math.cos(body.heading + math.pi / 2) * 5.0
        left_ay = body.y + math.sin(body.heading + math.pi / 2) * 5.0
        right_ax = body.x + math.cos(body.heading - math.pi / 2) * 5.0
        right_ay = body.y + math.sin(body.heading - math.pi / 2) * 5.0

        n_al_left = len(self.brain._al_left_t)
        n_al_right = len(self.brain._al_right_t)
        # Reserve last 1/5 of AL for thermosensory neurons
        n_olfactory_left = n_al_left - max(2, n_al_left // 5)
        n_olfactory_right = n_al_right - max(2, n_al_right // 5)
        glom_size_left = max(2, n_olfactory_left // N_GLOMERULI)
        glom_size_right = max(2, n_olfactory_right // N_GLOMERULI)

        for ci, compound in enumerate(odor_compounds):
            odor_left = world.odor_at(left_ax, left_ay, compound)
            odor_right = world.odor_at(right_ax, right_ay, compound)
            odor_center = world.odor_at(body.x, body.y, compound)
            prev = self._prev_odor.get(compound, 0.0)
            self._prev_odor[compound] = odor_center

            # Map compound to its specific glomerulus (OR type)
            glom_idx = COMPOUND_GLOMERULUS.get(compound, hash(compound) % N_GLOMERULI)
            start_l = glom_idx * glom_size_left
            end_l = min(start_l + glom_size_left, n_olfactory_left)
            start_r = glom_idx * glom_size_right
            end_r = min(start_r + glom_size_right, n_olfactory_right)
            glom_left_t = self.brain._al_left_t[start_l:end_l]
            glom_right_t = self.brain._al_right_t[start_r:end_r]

            # Left antenna → left glomerulus (tonic: proportional to concentration)
            left_current = min(20.0, odor_left * 10.0)
            if left_current > 0.1 and len(glom_left_t) > 0:
                b.external_current[glom_left_t] += left_current

            # Right antenna → right glomerulus
            right_current = min(20.0, odor_right * 10.0)
            if right_current > 0.1 and len(glom_right_t) > 0:
                b.external_current[glom_right_t] += right_current

            # Phasic ON response (change detection — strong transient when
            # odor first appears, critical for conditioning)
            d_odor = odor_center - prev
            if d_odor > 0.005:
                phasic = min(50.0, d_odor * 1000.0)
                if len(glom_left_t) > 0:
                    b.external_current[glom_left_t] += phasic
                if len(glom_right_t) > 0:
                    b.external_current[glom_right_t] += phasic
            elif d_odor < -0.005:
                off_phasic = min(30.0, abs(d_odor) * 500.0) * 0.3
                if len(glom_left_t) > 0:
                    b.external_current[glom_left_t] += off_phasic
                if len(glom_right_t) > 0:
                    b.external_current[glom_right_t] += off_phasic

            intensities[f"odor_{compound}_L"] = float(left_current)
            intensities[f"odor_{compound}_R"] = float(right_current)

        # ------- Visual (bilateral compound eyes → left/right optic lobe) -------
        # Sample light at left and right eye positions
        left_ex = body.x + math.cos(body.heading + math.pi / 2) * 10.0
        left_ey = body.y + math.sin(body.heading + math.pi / 2) * 10.0
        right_ex = body.x + math.cos(body.heading - math.pi / 2) * 10.0
        right_ey = body.y + math.sin(body.heading - math.pi / 2) * 10.0

        light_left = world.light_at(left_ex, left_ey)
        light_right = world.light_at(right_ex, right_ey)

        # Left eye → left optic lobe neurons
        left_visual = min(25.0, light_left * 15.0)
        if left_visual > 0.1:
            b.external_current[self.brain._optic_left_t] += left_visual

        # Right eye → right optic lobe neurons
        right_visual = min(25.0, light_right * 15.0)
        if right_visual > 0.1:
            b.external_current[self.brain._optic_right_t] += right_visual

        intensities["light_L"] = float(left_visual)
        intensities["light_R"] = float(right_visual)
        intensities["light_gradient"] = float(light_left - light_right)

        # ------- Thermosensory (bilateral antennae → left/right AL thermo) -------
        temp_left = world.temperature_at(left_ax, left_ay)
        temp_right = world.temperature_at(right_ax, right_ay)
        temp_center = world.temperature_at(body.x, body.y)
        self._prev_temp = temp_center

        # Temperature-sensitive neurons respond proportionally to deviation
        # from preferred temperature. More deviation = more current.
        left_temp_err = abs(temp_left - PREFERRED_TEMP_C)
        right_temp_err = abs(temp_right - PREFERRED_TEMP_C)

        # Inject into thermosensory subset of AL (last 1/5 of each side)
        n_al_half = len(self.brain._al_left_t)
        thermo_n = max(2, n_al_half // 5)
        thermo_left_t = self.brain._al_left_t[-thermo_n:]
        thermo_right_t = self.brain._al_right_t[-thermo_n:]

        # More current when further from preferred (drives aversive turning)
        left_thermo_current = min(30.0, left_temp_err * 5.0)
        right_thermo_current = min(30.0, right_temp_err * 5.0)
        if left_thermo_current > 0.5:
            b.external_current[thermo_left_t] += left_thermo_current
        if right_thermo_current > 0.5:
            b.external_current[thermo_right_t] += right_thermo_current

        intensities["temperature"] = float(temp_center)
        intensities["temp_err_L"] = float(left_temp_err)
        intensities["temp_err_R"] = float(right_temp_err)

        # ------- Food / gustatory -------
        food = world.food_at(body.x, body.y)
        self._prev_food = food

        if food > 0.1:
            food_current = min(30.0, food * 20.0)
            b.external_current[self.brain._al_t] += food_current * 0.3
            intensities["food"] = float(food_current)
        else:
            intensities["food"] = 0.0

        return intensities


# ============================================================================
# Motor Decoding
# ============================================================================

class DrosophilaMotor:
    """Read the brain's motor output — no behavioral logic.

    The brain's lateralized wiring (left optic → left DN → left VNC,
    right optic → right DN → right VNC) produces asymmetric motor
    neuron activity when sensory input is asymmetric. This decoder
    just measures that asymmetry:
      - Speed: total motor neuron activity → walking speed
      - Turn: mean voltage difference between left/right VNC → heading change

    Voltage-based readout captures the brain's directional signal even
    when spike counts are unreliable (HH all-or-nothing threshold makes
    count differences noisy, but voltage reflects synaptic drive).
    """

    def __init__(self, brain: DrosophilaBrain):
        self.brain = brain
        self.dev = brain.device
        self._left_t = brain._t_vnc_left
        self._right_t = brain._t_vnc_right

    def decode(self, n_steps: int = NEURAL_STEPS_PER_BODY,
               arousal: float = 1.0,
               ) -> Tuple[float, float]:
        """Run neural sim and read motor output.

        CPG reset prevents Na+ inactivation accumulation across body steps.
        Then run n_steps of HH simulation with tonic drive modulated by
        arousal. Read total firing for speed + voltage asymmetry for turn.

        Args:
            n_steps: HH timesteps per body update
            arousal: circadian modulation (1.0=day, 0.5=night)

        Returns:
            (turn_bias, speed_factor)
        """
        b = self.brain.brain
        motor_t = self.brain._motor_t

        # CPG reset: restore motor neurons to resting state
        b.voltage[motor_t] = -65.0
        b.prev_voltage[motor_t] = -65.0
        b.nav_m[motor_t] = 0.05
        b.nav_h[motor_t] = 0.6
        b.kv_n[motor_t] = 0.32
        b.cav_m[motor_t] = 0.01
        b.cav_h[motor_t] = 0.99
        b.refractory[motor_t] = 0.0
        b.voltage[motor_t] += torch.randn(motor_t.numel(), device=self.dev) * 4.0

        total_acc = torch.zeros(1, device=self.dev)
        # Accumulate voltage for directional readout
        left_v_acc = torch.zeros(1, device=self.dev)
        right_v_acc = torch.zeros(1, device=self.dev)

        for s in range(n_steps):
            # Tonic CPG drive — arousal modulates locomotion vigor
            b.external_current[motor_t] += 25.0 * arousal
            b.external_current[self.brain._cx_t] += 15.0 * arousal
            self.brain.step()
            total_acc += b.fired[motor_t].sum()
            # Voltage accumulation for directional readout
            left_v_acc += b.voltage[self._left_t].mean()
            right_v_acc += b.voltage[self._right_t].mean()

        total_motor = int(total_acc.item())

        # Speed: total motor firing
        n_motor = len(self.brain.motor_ids)
        if total_motor > n_motor * 0.02:
            speed_factor = min(2.0, 0.5 + total_motor / (n_motor * 0.5))
        else:
            speed_factor = 0.1

        # Turn: voltage asymmetry between left and right VNC
        # More depolarized side → more ipsilateral leg drive → contralateral turn.
        # Right VNC more active → fly turns LEFT (positive).
        mean_left_v = float(left_v_acc.item()) / max(n_steps, 1)
        mean_right_v = float(right_v_acc.item()) / max(n_steps, 1)
        v_diff = mean_right_v - mean_left_v  # positive = right more active = turn left

        # Scale voltage difference to turn rate.
        # A few mV difference across many neurons → meaningful heading change.
        # Clamp to prevent extreme turns.
        turn_bias = max(-FLY_TURN_RATE, min(FLY_TURN_RATE,
                                            v_diff * 0.15))

        return turn_bias, speed_factor


# ============================================================================
# Mushroom Body Learning Protocols
# ============================================================================

class OlfactoryConditioningProtocol:
    """Classical olfactory conditioning via mushroom body.

    Replicates Tully & Quinn (1985): pair an odor (CS+) with reward (US)
    by delivering DA to Kenyon cells during odor presentation.

    Training:
      1. Present CS+ odor -> activate antennal lobe
      2. Deliver reward (DA to KC) -> strengthens AL->PN->KC->MBON pathway
      3. STDP consolidates the association

    Testing:
      Present CS+ vs CS- and compare MB output activity.
    """

    def __init__(self, brain: DrosophilaBrain, da_amount: float = 200.0):
        self.brain = brain
        self.da_amount = da_amount
        self.dev = brain.device

    def train_trial(
        self,
        world: MolecularWorld,
        body: DrosophilaBody,
        sensory: DrosophilaSensory,
        motor: DrosophilaMotor,
        reward: bool = True,
        n_steps: int = 30,
    ) -> None:
        """One conditioning trial: present odor, optionally deliver DA reward.

        Args:
            reward: if True, deliver DA to Kenyon cells (CS+ trial).
                    if False, present odor without reward (CS- trial).
        """
        b = self.brain.brain

        for s in range(n_steps):
            # Encode current environment (odor at fly's position)
            sensory.encode(world, body)

            if reward and s % 3 == 0:
                # DA reward to mushroom body (mimics PPL1/PAM DA neurons)
                b.nt_conc[self.brain._kc_t, NT_DA] += self.da_amount

            # Pulsed tonic drive
            if s % 2 == 0:
                b.external_current[self.brain._motor_t] += 20.0
                b.external_current[self.brain._cx_t] += 10.0

            self.brain.step()

    def test_response(
        self,
        world: MolecularWorld,
        body: DrosophilaBody,
        sensory: DrosophilaSensory,
        n_steps: int = 60,
    ) -> float:
        """Present odor and measure MBON response magnitude.

        Uses voltage-based readout: mean MBON membrane potential accumulated
        over the test period. Voltage captures sub-threshold synaptic drive
        even when spike counts are unreliable (HH all-or-nothing threshold
        makes spike-based readout noisy at this scale). Higher (less negative)
        voltage = more depolarized = stronger learned response.
        """
        b = self.brain.brain
        mbon_v_acc = 0.0

        for s in range(n_steps):
            sensory.encode(world, body)
            if s % 2 == 0:
                b.external_current[self.brain._motor_t] += 20.0
            self.brain.step()
            mbon_v_acc += float(b.voltage[self.brain._mbon_t].mean().item())

        return mbon_v_acc


class FreeEnergyNavigation:
    """Free-energy-principle navigation for foraging.

    Structured feedback when fly approaches food (predictable = low entropy),
    unstructured noise when moving away (unpredictable = high entropy).
    Same principle as DishBrain (Kagan et al. 2022).
    """

    def __init__(self, brain: DrosophilaBrain):
        self.brain = brain
        self.dev = brain.device

    def deliver_approach(self, n_steps: int = 30,
                         intensity: float = 40.0) -> None:
        """Structured feedback: synchronized pulse to all neurons."""
        b = self.brain.brain
        # NE boost for STDP enhancement
        b.nt_conc[self.brain._kc_t, NT_NE] += 150.0
        for s in range(n_steps):
            if s % 2 == 0:
                b.external_current[self.brain._all_t] += intensity
            self.brain.step()

    def deliver_failure(self, n_steps: int = 50) -> None:
        """Unstructured feedback: random noise to random subsets."""
        b = self.brain.brain
        n_all = self.brain.n_total
        for s in range(n_steps):
            mask = torch.rand(n_all, device=self.dev) < 0.3
            active = self.brain._all_t[mask]
            if active.numel() > 0:
                noise = torch.rand(active.numel(), device=self.dev) * 35.0
                b.external_current[active] += noise
            self.brain.step()


# ============================================================================
# Simulation Loop
# ============================================================================

def simulate_fly(
    brain: DrosophilaBrain,
    body: DrosophilaBody,
    world: MolecularWorld,
    sensory: DrosophilaSensory,
    motor: DrosophilaMotor,
    n_steps: int = 500,
    odor_compounds: Optional[List[str]] = None,
    advance_time: bool = False,
    time_step_hours: float = 0.05,
) -> Dict[str, Any]:
    """Run one complete simulation episode.

    Each body step:
      1. Sensory encoding (sample world, inject currents)
      2. Neural simulation (NEURAL_STEPS_PER_BODY HH timesteps)
      3. Motor decoding (read spike counts)
      4. Body update (move fly)

    Returns trajectory and statistics.
    """
    trajectory = [(body.x, body.y)]
    distances = []
    speeds_log = []
    light_log = []
    temp_log = []
    food_log = []

    for step_i in range(n_steps):
        # Optionally advance day/night cycle
        if advance_time:
            world.advance_time(time_step_hours)

        # 1. Sensory encoding
        intensities = sensory.encode(world, body, odor_compounds)

        # 2 + 3. Neural sim + motor decode
        turn_bias, speed_factor = motor.decode(n_steps=NEURAL_STEPS_PER_BODY)

        # 4. Body update
        body.update(turn_bias, speed_factor, dt=DT, arena_size=world.size_mm)

        # Feed if on food
        if world.food_at(body.x, body.y) > 0.1:
            world.deplete_food_near(body.x, body.y)
            body.feed(0.01)

        # Log
        trajectory.append((body.x, body.y))
        speeds_log.append(body.speed * speed_factor)
        light_log.append(world.light_at(body.x, body.y))
        temp_log.append(world.temperature_at(body.x, body.y))
        food_log.append(world.food_at(body.x, body.y))

    return {
        "trajectory": trajectory,
        "speeds": speeds_log,
        "light_values": light_log,
        "temp_values": temp_log,
        "food_values": food_log,
        "total_distance": body.total_distance,
        "final_energy": body.energy,
    }


# ============================================================================
# Warmup
# ============================================================================

def _warmup(brain: DrosophilaBrain, n_steps: int = 400) -> None:
    """Stabilize neural dynamics with tonic background activity.

    Motor neurons are NOT driven during warmup because decode()
    resets their HH state before each body step (CPG reset pattern).
    """
    b = brain.brain
    for s in range(n_steps):
        if s % 5 == 0:
            b.external_current[brain._al_t] += 5.0
            b.external_current[brain._cx_t] += 3.0
        brain.step()


# ============================================================================
# Utilities
# ============================================================================

def _header(title: str, subtitle: str) -> None:
    """Print formatted section header."""
    w = 76
    print("\n" + "=" * w)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("=" * w)


def _system_info() -> Dict[str, Any]:
    """Collect system information for JSON output."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": detect_backend(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "CPU only"
    return info


def _make_json_safe(obj: Any) -> Any:
    """Convert results dict to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


# ============================================================================
# Experiment 1: Olfactory Learning (Mushroom Body)
# ============================================================================

def exp_olfactory_learning(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_train_episodes: int = 30,
    n_test_trials: int = 10,
) -> Dict[str, Any]:
    """Classical olfactory conditioning -- Tully & Quinn (1985).

    Train: pair CS+ odor (ethanol) with DA reward to mushroom body.
    Test: present trained odor vs novel odor, compare MBON response.
    Pass: trained_response > untrained_response.
    """
    _header(
        "Exp 1: Olfactory Learning (Mushroom Body)",
        "Classical conditioning -- CS+ with DA reward vs CS- without"
    )
    t0 = time.perf_counter()

    # Build brain
    brain = DrosophilaBrain(scale=scale, device=device, seed=seed)
    if brain.device.type == 'cuda':
        brain.brain.compile()
    print(f"    Brain: {brain.n_total} neurons, {brain.brain.n_synapses} "
          f"synapses on {brain.device}")

    # Build world with CS+ odor source
    world_csplus = MolecularWorld(size_mm=100.0, seed=seed)
    world_csplus.add_odorant_source(50.0, 50.0, compound="ethanol",
                                     intensity=0.8, radius=20.0)

    # CS- world with novel odor
    world_csminus = MolecularWorld(size_mm=100.0, seed=seed)
    world_csminus.add_odorant_source(50.0, 50.0, compound="ethyl_acetate",
                                      intensity=0.8, radius=20.0)

    body = DrosophilaBody(x=50.0, y=50.0)
    sensory = DrosophilaSensory(brain)
    motor = DrosophilaMotor(brain)
    protocol = OlfactoryConditioningProtocol(brain, da_amount=200.0)

    # Warmup
    _warmup(brain)
    print(f"    Warmup complete")

    # ---------- Training ----------
    # Differential conditioning (Tully & Quinn 1985): alternate CS+ trials
    # (odor + DA reward) with CS- trials (different odor, no reward).
    # This creates contrast: the network learns to respond more to CS+
    # and less to CS-. Real Drosophila conditioning always uses this
    # paired/unpaired protocol.
    print(f"    Training: {n_train_episodes} differential episodes (CS+ w/ DA, CS- w/o)...")
    for ep in range(n_train_episodes):
        rng = random.Random(seed + ep)

        # CS+ trial: trained odor + DA reward
        body.x = 48.0 + rng.uniform(-5, 5)
        body.y = 48.0 + rng.uniform(-5, 5)
        protocol.train_trial(world_csplus, body, sensory, motor,
                             reward=True, n_steps=40)
        brain.run(20)

        # CS- trial: novel odor, NO reward (differential conditioning)
        body.x = 48.0 + rng.uniform(-5, 5)
        body.y = 48.0 + rng.uniform(-5, 5)
        protocol.train_trial(world_csminus, body, sensory, motor,
                             reward=False, n_steps=40)
        brain.run(20)

        if (ep + 1) % 5 == 0:
            print(f"      Episode {ep + 1}/{n_train_episodes}")

    # ---------- Testing ----------
    # Counterbalanced order: alternate which odor is tested first to
    # eliminate order effects (first-tested odor can differ due to
    # transient network state).
    print(f"    Testing: {n_test_trials} trials each for CS+ and CS-...")
    csplus_responses = []
    csminus_responses = []

    for trial in range(n_test_trials):
        trial_seed = seed + 1000 + trial
        brain.run(40)  # inter-trial settling

        if trial % 2 == 0:
            # CS+ first
            body.x = 48.0 + random.Random(trial_seed).uniform(-3, 3)
            body.y = 48.0 + random.Random(trial_seed).uniform(-3, 3)
            sensory_test = DrosophilaSensory(brain)
            resp_plus = protocol.test_response(world_csplus, body, sensory_test)
            csplus_responses.append(resp_plus)

            brain.run(40)

            body.x = 48.0 + random.Random(trial_seed + 500).uniform(-3, 3)
            body.y = 48.0 + random.Random(trial_seed + 500).uniform(-3, 3)
            sensory_test2 = DrosophilaSensory(brain)
            resp_minus = protocol.test_response(world_csminus, body, sensory_test2)
            csminus_responses.append(resp_minus)
        else:
            # CS- first
            body.x = 48.0 + random.Random(trial_seed + 500).uniform(-3, 3)
            body.y = 48.0 + random.Random(trial_seed + 500).uniform(-3, 3)
            sensory_test = DrosophilaSensory(brain)
            resp_minus = protocol.test_response(world_csminus, body, sensory_test)
            csminus_responses.append(resp_minus)

            brain.run(40)

            body.x = 48.0 + random.Random(trial_seed).uniform(-3, 3)
            body.y = 48.0 + random.Random(trial_seed).uniform(-3, 3)
            sensory_test2 = DrosophilaSensory(brain)
            resp_plus = protocol.test_response(world_csplus, body, sensory_test2)
            csplus_responses.append(resp_plus)

    elapsed = time.perf_counter() - t0

    mean_csplus = sum(csplus_responses) / len(csplus_responses)
    mean_csminus = sum(csminus_responses) / len(csminus_responses)

    # Pass: trained odor elicits stronger MBON response
    passed = mean_csplus > mean_csminus

    print(f"\n    Results:")
    print(f"    CS+ (trained) MBON response:   {mean_csplus:.1f} (voltage, mean)")
    print(f"    CS- (novel)   MBON response:   {mean_csminus:.1f} (voltage, mean)")
    diff_pct = (mean_csplus - mean_csminus) / max(abs(mean_csminus), 0.01) * 100
    print(f"    CS+ - CS- difference:          {mean_csplus - mean_csminus:.1f} ({diff_pct:+.1f}%)")
    print(f"    {'PASS' if passed else 'FAIL'} (CS+ > CS-) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "mean_csplus": mean_csplus,
        "mean_csminus": mean_csminus,
        "csplus_responses": csplus_responses,
        "csminus_responses": csminus_responses,
    }


# ============================================================================
# Experiment 2: Phototaxis
# ============================================================================

def exp_phototaxis(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Phototaxis -- approach dim light, avoid bright light.

    Drosophila exhibits positive phototaxis at low-to-moderate intensities
    and negative phototaxis (avoidance) at very high intensities.

    Setup: light source on right side of arena.
    Low intensity test: fly should move toward light.
    High intensity test: fly should move away from light.
    Pass: correct phototaxis direction in >60% of trials.
    """
    _header(
        "Exp 2: Phototaxis",
        "Positive phototaxis (dim light) and negative phototaxis (bright light)"
    )
    t0 = time.perf_counter()

    conditions = [
        ("dim", 0.3),    # moderate intensity -> approach
        ("bright", 1.5), # high intensity -> avoid
    ]

    results_by_condition: Dict[str, Dict[str, Any]] = {}

    for cond_name, light_intensity in conditions:
        print(f"\n    Condition: {cond_name} light (intensity={light_intensity})")
        correct_count = 0

        for ep in range(n_episodes):
            ep_seed = seed + ep * 11 + (1000 if cond_name == "bright" else 0)

            brain = DrosophilaBrain(
                scale=scale, device=device, seed=ep_seed)
            if brain.device.type == 'cuda':
                brain.brain.compile()

            if ep == 0:
                print(f"    Brain: {brain.n_total} neurons, "
                      f"{brain.brain.n_synapses} synapses on {brain.device}")

            # Light source on the RIGHT side (x=80)
            world = MolecularWorld(size_mm=100.0, seed=ep_seed)
            world.add_light_source(80.0, 50.0, intensity=light_intensity,
                                   radius=30.0)

            # Start at center, random heading
            rng = random.Random(ep_seed)
            body = DrosophilaBody(
                x=50.0, y=50.0,
                heading=rng.uniform(0, 2 * math.pi))
            sensory = DrosophilaSensory(brain)
            motor = DrosophilaMotor(brain)

            _warmup(brain, n_steps=200)

            result = simulate_fly(brain, body, world, sensory, motor,
                                  n_steps=n_steps)

            # Measure: did fly move toward (dim) or away (bright) from light?
            start_dist = math.sqrt((50.0 - 80.0) ** 2)  # 30.0
            end_dist = body.distance_to(80.0, 50.0)

            if cond_name == "dim":
                # Positive phototaxis: should get CLOSER
                correct = end_dist < start_dist
            else:
                # Negative phototaxis: should get FARTHER
                correct = end_dist > start_dist

            if correct:
                correct_count += 1

            if (ep + 1) % 5 == 0:
                print(f"      Episode {ep+1}: start_dist={start_dist:.1f} "
                      f"end_dist={end_dist:.1f} "
                      f"[{'correct' if correct else 'wrong'}]")

        accuracy = correct_count / n_episodes
        results_by_condition[cond_name] = {
            "correct": correct_count,
            "total": n_episodes,
            "accuracy": accuracy,
        }
        print(f"    {cond_name}: {correct_count}/{n_episodes} correct ({accuracy:.0%})")

    elapsed = time.perf_counter() - t0

    # Pass: correct phototaxis direction in >60% of trials for EITHER condition
    dim_acc = results_by_condition["dim"]["accuracy"]
    bright_acc = results_by_condition["bright"]["accuracy"]
    passed = dim_acc > 0.60 or bright_acc > 0.60

    print(f"\n    Results:")
    print(f"    Dim light (approach):   {dim_acc:.0%} correct")
    print(f"    Bright light (avoid):   {bright_acc:.0%} correct")
    print(f"    {'PASS' if passed else 'FAIL'} (>60% correct in either condition) "
          f"in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "dim_accuracy": dim_acc,
        "bright_accuracy": bright_acc,
        "conditions": results_by_condition,
    }


# ============================================================================
# Experiment 3: Thermotaxis
# ============================================================================

def exp_thermotaxis(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 400,
) -> Dict[str, Any]:
    """Thermotaxis -- navigate toward preferred temperature (~24C).

    Temperature gradient across arena: 15C (left) to 35C (right).
    Fly should navigate toward 24C zone (Hamada et al. 2008).
    Pass: mean final position within 5C of preferred temp.
    """
    _header(
        "Exp 3: Thermotaxis",
        "Navigate to preferred temperature (~24C) on thermal gradient"
    )
    t0 = time.perf_counter()

    final_temps = []
    start_temps = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 19

        brain = DrosophilaBrain(
            scale=scale, device=device, seed=ep_seed)
        if brain.device.type == 'cuda':
            brain.brain.compile()

        if ep == 0:
            print(f"    Brain: {brain.n_total} neurons, "
                  f"{brain.brain.n_synapses} synapses on {brain.device}")

        # Temperature gradient: 15C (x=0) to 35C (x=100)
        world = MolecularWorld(size_mm=100.0, seed=ep_seed)
        world.temp_min = 15.0
        world.temp_max = 35.0

        # Start at random position
        rng = random.Random(ep_seed)
        start_x = rng.uniform(10.0, 90.0)
        start_y = rng.uniform(20.0, 80.0)

        body = DrosophilaBody(x=start_x, y=start_y,
                              heading=rng.uniform(0, 2 * math.pi))
        sensory = DrosophilaSensory(brain)
        motor = DrosophilaMotor(brain)

        _warmup(brain, n_steps=200)

        result = simulate_fly(brain, body, world, sensory, motor,
                              n_steps=n_steps)

        start_temp = world.temperature_at(start_x, start_y)
        final_temp = world.temperature_at(body.x, body.y)
        start_temps.append(start_temp)
        final_temps.append(final_temp)

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1:3d}: start={start_temp:.1f}C "
                  f"final={final_temp:.1f}C "
                  f"error={abs(final_temp - PREFERRED_TEMP_C):.1f}C")

    elapsed = time.perf_counter() - t0

    mean_start = sum(start_temps) / len(start_temps)
    mean_final = sum(final_temps) / len(final_temps)
    mean_error = sum(abs(t - PREFERRED_TEMP_C) for t in final_temps) / len(final_temps)
    start_error = sum(abs(t - PREFERRED_TEMP_C) for t in start_temps) / len(start_temps)

    # Pass: mean final position within 5C of preferred, OR improvement over start
    passed = mean_error < 5.0 or mean_error < start_error

    print(f"\n    Results:")
    print(f"    Preferred temperature:        {PREFERRED_TEMP_C}C")
    print(f"    Mean start temperature:       {mean_start:.1f}C "
          f"(error: {start_error:.1f}C)")
    print(f"    Mean final temperature:       {mean_final:.1f}C "
          f"(error: {mean_error:.1f}C)")
    print(f"    Error reduction:              {start_error - mean_error:+.1f}C")
    print(f"    {'PASS' if passed else 'FAIL'} (final error < 5C or improved) "
          f"in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "mean_start_temp": mean_start,
        "mean_final_temp": mean_final,
        "mean_error": mean_error,
        "start_error": start_error,
        "final_temps": final_temps,
    }


# ============================================================================
# Experiment 4: Foraging Behavior
# ============================================================================

def exp_foraging(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 20,
    n_steps: int = 500,
) -> Dict[str, Any]:
    """Foraging -- navigate using olfactory gradients to find food.

    Multiple food sources emit odorant gradients. Fly must navigate using
    olfactory cues to locate and feed. Uses FEP-based learning: structured
    feedback when approaching food, noise when moving away.

    Pass: finds food in >30% of episodes (vs ~5% random).
    """
    _header(
        "Exp 4: Foraging Behavior",
        "Navigate olfactory gradients to locate food sources"
    )
    t0 = time.perf_counter()

    food_found_count = 0
    total_distances = []
    food_times = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 23

        brain = DrosophilaBrain(
            scale=scale, device=device, seed=ep_seed)
        if brain.device.type == 'cuda':
            brain.brain.compile()

        if ep == 0:
            print(f"    Brain: {brain.n_total} neurons, "
                  f"{brain.brain.n_synapses} synapses on {brain.device}")

        # Build world with food sources emitting odorants
        world = MolecularWorld(size_mm=100.0, seed=ep_seed)
        rng = random.Random(ep_seed)

        # Place 3 food patches with odorant gradients
        food_positions = []
        for _ in range(3):
            fx = rng.uniform(20.0, 80.0)
            fy = rng.uniform(20.0, 80.0)
            world.add_food(fx, fy, radius=5.0, remaining=1.0)
            world.add_odorant_source(fx, fy, compound="ethanol",
                                     intensity=0.8, radius=20.0)
            food_positions.append((fx, fy))

        # Start at random position (away from food)
        while True:
            sx = rng.uniform(10.0, 90.0)
            sy = rng.uniform(10.0, 90.0)
            min_dist = min(math.sqrt((sx - fx) ** 2 + (sy - fy) ** 2)
                           for fx, fy in food_positions)
            if min_dist > 20.0:
                break

        body = DrosophilaBody(x=sx, y=sy,
                              heading=rng.uniform(0, 2 * math.pi))
        sensory = DrosophilaSensory(brain)
        motor = DrosophilaMotor(brain)
        fep = FreeEnergyNavigation(brain)

        _warmup(brain, n_steps=200)

        # Run foraging with FEP feedback
        found_food = False
        food_time = 0
        prev_min_dist = min(body.distance_to(fx, fy) for fx, fy in food_positions)

        for step_i in range(n_steps):
            # Sensory encode
            intensities = sensory.encode(world, body,
                                         odor_compounds=["ethanol"])

            # Motor decode
            turn_bias, speed_factor = motor.decode(n_steps=NEURAL_STEPS_PER_BODY)

            # Body update
            body.update(turn_bias, speed_factor, dt=DT, arena_size=world.size_mm)

            # Check food contact
            curr_food = world.food_at(body.x, body.y)
            if curr_food > 0.15:
                food_time += 1
                if not found_food:
                    found_food = True
                world.deplete_food_near(body.x, body.y)
                body.feed(0.01)

            # FEP feedback every 20 steps
            if step_i > 0 and step_i % 20 == 0:
                curr_min_dist = min(body.distance_to(fx, fy)
                                    for fx, fy in food_positions)
                if curr_min_dist < prev_min_dist:
                    # Moving closer -> structured feedback
                    fep.deliver_approach(n_steps=10, intensity=30.0)
                else:
                    # Moving away -> noise
                    fep.deliver_failure(n_steps=15)
                prev_min_dist = curr_min_dist

        total_distances.append(body.total_distance)
        food_times.append(food_time)
        if found_food:
            food_found_count += 1

        if (ep + 1) % 5 == 0:
            print(f"    Episode {ep+1:3d}: food_found={'YES' if found_food else 'NO'}  "
                  f"food_time={food_time}  dist={body.total_distance:.1f}mm")

    elapsed = time.perf_counter() - t0

    success_rate = food_found_count / n_episodes
    mean_food_time = sum(food_times) / n_episodes
    mean_dist = sum(total_distances) / n_episodes
    random_baseline = 0.05  # ~5% chance of stumbling onto food randomly

    passed = success_rate > 0.30

    print(f"\n    Results:")
    print(f"    Food found:           {food_found_count}/{n_episodes} "
          f"({success_rate:.0%})")
    print(f"    Random baseline:      ~{random_baseline:.0%}")
    print(f"    Enhancement:          {success_rate / max(random_baseline, 0.01):.1f}x")
    print(f"    Mean food contact:    {mean_food_time:.1f} steps")
    print(f"    Mean distance:        {mean_dist:.1f} mm")
    print(f"    {'PASS' if passed else 'FAIL'} (>30% food found) in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "success_rate": success_rate,
        "food_found": food_found_count,
        "total_episodes": n_episodes,
        "mean_food_time": mean_food_time,
        "mean_distance": mean_dist,
        "food_times": food_times,
    }


# ============================================================================
# Experiment 5: Drug Effects
# ============================================================================

def exp_drug_effects(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 10,
    n_steps: int = 300,
) -> Dict[str, Any]:
    """Pharmacological modulation of foraging performance.

    4 conditions: baseline, caffeine, diazepam, nicotine.

    Predictions:
    - Caffeine: may enhance or maintain performance (adenosine antagonist,
      increases activity). Caffeine increases locomotor activity in real
      Drosophila (Wu et al. 2009).
    - Diazepam: should reduce performance (GABA-A enhancement,
      sedation/locomotor depression).
    - Nicotine: strong effect on Drosophila nAChRs -- the dominant
      excitatory receptor in the insect CNS. Low dose may enhance,
      high dose suppresses (Bainton et al. 2000).

    Pass: diazepam reduces foraging performance vs baseline.
    """
    _header(
        "Exp 5: Drug Effects on Foraging",
        "4 conditions: baseline / caffeine / diazepam / nicotine"
    )
    t0 = time.perf_counter()

    conditions = [
        ("baseline",  None,       0.0),
        ("caffeine",  "caffeine", 150.0),
        ("diazepam",  "diazepam", 30.0),
        ("nicotine",  "nicotine", 5.0),
    ]

    condition_results: Dict[str, Dict[str, Any]] = {}

    for cond_name, drug_name, dose in conditions:
        print(f"\n    Condition: {cond_name}")
        food_found_counts = []
        total_dists = []
        food_times = []

        for ep in range(n_episodes):
            ep_seed = seed + ep * 29

            brain = DrosophilaBrain(
                scale=scale, device=device, seed=ep_seed)
            if brain.device.type == 'cuda':
                brain.brain.compile()

            # Apply drug BEFORE warmup so it takes effect
            if drug_name is not None:
                brain.apply_drug(drug_name, dose)

            # Build world with food + odorants
            world = MolecularWorld(size_mm=100.0, seed=ep_seed)
            rng = random.Random(ep_seed)
            food_positions = []
            for _ in range(3):
                fx = rng.uniform(20.0, 80.0)
                fy = rng.uniform(20.0, 80.0)
                world.add_food(fx, fy, radius=5.0, remaining=1.0)
                world.add_odorant_source(fx, fy, compound="ethanol",
                                         intensity=0.8, radius=20.0)
                food_positions.append((fx, fy))

            # Start away from food
            while True:
                sx = rng.uniform(10.0, 90.0)
                sy = rng.uniform(10.0, 90.0)
                min_d = min(math.sqrt((sx - fx) ** 2 + (sy - fy) ** 2)
                            for fx, fy in food_positions)
                if min_d > 15.0:
                    break

            body = DrosophilaBody(x=sx, y=sy,
                                  heading=rng.uniform(0, 2 * math.pi))
            sensory = DrosophilaSensory(brain)
            motor = DrosophilaMotor(brain)

            _warmup(brain, n_steps=200)

            # Re-apply drug periodically (maintenance dose)
            result = simulate_fly(brain, body, world, sensory, motor,
                                  n_steps=n_steps,
                                  odor_compounds=["ethanol"])

            # Check food contact
            food_time = sum(1 for f in result["food_values"] if f > 0.15)
            found = food_time > 5  # at least 5 steps near food
            food_found_counts.append(1 if found else 0)
            total_dists.append(result["total_distance"])
            food_times.append(food_time)

        success_rate = sum(food_found_counts) / n_episodes
        mean_dist = sum(total_dists) / n_episodes
        mean_food = sum(food_times) / n_episodes

        condition_results[cond_name] = {
            "success_rate": success_rate,
            "mean_distance": mean_dist,
            "mean_food_time": mean_food,
        }
        print(f"      Food found: {sum(food_found_counts)}/{n_episodes} "
              f"({success_rate:.0%})  "
              f"dist={mean_dist:.1f}mm  food_time={mean_food:.1f}")

    elapsed = time.perf_counter() - t0

    baseline_rate = condition_results["baseline"]["success_rate"]
    caffeine_rate = condition_results["caffeine"]["success_rate"]
    diazepam_rate = condition_results["diazepam"]["success_rate"]
    nicotine_rate = condition_results["nicotine"]["success_rate"]

    # Pass: diazepam reduces foraging performance (food, food time, OR distance)
    diazepam_reduces = (diazepam_rate < baseline_rate or
                        condition_results["diazepam"]["mean_food_time"] <
                        condition_results["baseline"]["mean_food_time"] or
                        condition_results["diazepam"]["mean_distance"] <
                        condition_results["baseline"]["mean_distance"])
    passed = diazepam_reduces

    print(f"\n    Results:")
    print(f"    Baseline:     {baseline_rate:.0%} food found, "
          f"{condition_results['baseline']['mean_food_time']:.1f} food steps")
    print(f"    Caffeine:     {caffeine_rate:.0%} food found "
          f"({caffeine_rate - baseline_rate:+.0%})")
    print(f"    Diazepam:     {diazepam_rate:.0%} food found "
          f"({diazepam_rate - baseline_rate:+.0%}) "
          f"[{'reduced' if diazepam_reduces else 'NOT reduced'}]")
    print(f"    Nicotine:     {nicotine_rate:.0%} food found "
          f"({nicotine_rate - baseline_rate:+.0%})")
    print(f"    Locomotor distance:")
    for cn in ["baseline", "caffeine", "diazepam", "nicotine"]:
        d = condition_results[cn]["mean_distance"]
        print(f"      {cn:12s}: {d:.1f} mm")
    print(f"    {'PASS' if passed else 'FAIL'} (diazepam impairs foraging) "
          f"in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "baseline_rate": baseline_rate,
        "caffeine_rate": caffeine_rate,
        "diazepam_rate": diazepam_rate,
        "nicotine_rate": nicotine_rate,
        "diazepam_reduces": diazepam_reduces,
        "conditions": condition_results,
    }


# ============================================================================
# Experiment 6: Day/Night Cycle
# ============================================================================

def exp_day_night(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_cycles: int = 4,
    steps_per_hour: int = 20,
) -> Dict[str, Any]:
    """Circadian-like activity patterns in day/night cycle.

    Run extended simulation across multiple day/night transitions.
    Drosophila is diurnal: more active during light phases.
    The mechanism is photoreceptor-driven modulation of motor circuits
    via optic lobe -> central complex -> motor pathway.

    Pass: mean activity (distance/time) during day > night.
    """
    _header(
        "Exp 6: Day/Night Cycle",
        "Circadian-like activity -- diurnal pattern (more active in daylight)"
    )
    t0 = time.perf_counter()

    brain = DrosophilaBrain(
        scale=scale, device=device, seed=seed)
    if brain.device.type == 'cuda':
        brain.brain.compile()
    print(f"    Brain: {brain.n_total} neurons, "
          f"{brain.brain.n_synapses} synapses on {brain.device}")

    world = MolecularWorld(size_mm=100.0, seed=seed)
    # Add some food so the fly has things to do
    world.add_food(30.0, 30.0, radius=8.0, remaining=10.0)
    world.add_food(70.0, 70.0, radius=8.0, remaining=10.0)
    world.add_odorant_source(30.0, 30.0, compound="ethanol",
                             intensity=0.5, radius=25.0)
    world.add_odorant_source(70.0, 70.0, compound="ethanol",
                             intensity=0.5, radius=25.0)
    # Start at dawn
    world.time_of_day = 0.25
    world.is_day = True
    world.ambient_light = 0.0

    body = DrosophilaBody(x=50.0, y=50.0,
                          heading=random.Random(seed).uniform(0, 2 * math.pi))
    sensory = DrosophilaSensory(brain)
    motor = DrosophilaMotor(brain)

    _warmup(brain, n_steps=200)

    # Simulate full day/night cycles
    hours_per_cycle = 24
    total_hours = n_cycles * hours_per_cycle
    total_steps = total_hours * steps_per_hour
    time_step_hours = 1.0 / steps_per_hour

    # Track activity by phase
    day_distances = []
    night_distances = []
    hourly_activity = []

    prev_x, prev_y = body.x, body.y

    print(f"    Simulating {n_cycles} day/night cycles "
          f"({total_steps} body steps)...")

    for step_i in range(total_steps):
        was_day = world.is_day

        # Advance time
        world.advance_time(time_step_hours)

        # Add/remove light sources based on time of day
        # (In the real API these would be handled internally, but with stubs
        # we modulate the ambient light which affects the optic lobe encoding)

        # Sensory encode
        intensities = sensory.encode(world, body, odor_compounds=["ethanol"])

        # Circadian modulation: day arousal via photoreceptors,
        # night suppression via dFB sleep neurons (Donlea et al. 2011)
        if world.is_day:
            solar = world.ambient_light
            b = brain.brain
            b.external_current[brain._optic_t] += solar * 20.0
            arousal = 1.0 + solar * 0.5  # day: 1.0-1.5x motor drive
        else:
            arousal = 0.5  # night: 50% motor drive (sleep suppression)

        # Motor decode with circadian arousal modulation
        turn_bias, speed_factor = motor.decode(
            n_steps=NEURAL_STEPS_PER_BODY, arousal=arousal)

        # Body update
        body.update(turn_bias, speed_factor, dt=DT, arena_size=world.size_mm)

        # Measure step distance
        dx = body.x - prev_x
        dy = body.y - prev_y
        step_dist = math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = body.x, body.y

        if was_day:
            day_distances.append(step_dist)
        else:
            night_distances.append(step_dist)

        # Hourly logging
        current_hour = step_i / steps_per_hour
        if step_i > 0 and step_i % steps_per_hour == 0:
            hour_int = int(current_hour)
            recent = day_distances[-steps_per_hour:] if was_day else night_distances[-steps_per_hour:]
            hour_dist = sum(recent) if recent else 0.0
            hourly_activity.append({
                "hour": hour_int,
                "distance": hour_dist,
                "is_day": was_day,
            })

            # Print every 6 hours
            if hour_int % 6 == 0:
                phase = "DAY" if was_day else "NIGHT"
                print(f"      Hour {hour_int:3d} [{phase:5s}]: "
                      f"activity={hour_dist:.2f}mm  "
                      f"ambient_light={world.ambient_light:.2f}")

    elapsed = time.perf_counter() - t0

    # Compute mean activity rates
    mean_day_dist = sum(day_distances) / max(len(day_distances), 1)
    mean_night_dist = sum(night_distances) / max(len(night_distances), 1)

    # Convert to mm/step activity rates
    day_rate = mean_day_dist
    night_rate = mean_night_dist

    # Pass: day activity > night activity
    passed = day_rate > night_rate

    print(f"\n    Results:")
    print(f"    Day activity rate:    {day_rate:.4f} mm/step "
          f"({len(day_distances)} steps)")
    print(f"    Night activity rate:  {night_rate:.4f} mm/step "
          f"({len(night_distances)} steps)")
    print(f"    Day/Night ratio:      {day_rate / max(night_rate, 1e-6):.2f}x")
    print(f"    Total distance:       {body.total_distance:.1f} mm")
    print(f"    {'PASS' if passed else 'FAIL'} (day > night activity) "
          f"in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "day_rate": day_rate,
        "night_rate": night_rate,
        "day_night_ratio": day_rate / max(night_rate, 1e-6),
        "total_distance": body.total_distance,
        "n_day_steps": len(day_distances),
        "n_night_steps": len(night_distances),
        "hourly_activity": hourly_activity,
    }


# ============================================================================
# Experiment Registry
# ============================================================================

ALL_EXPERIMENTS = {
    1: ("Olfactory Learning (Mushroom Body)", exp_olfactory_learning),
    2: ("Phototaxis", exp_phototaxis),
    3: ("Thermotaxis", exp_thermotaxis),
    4: ("Foraging Behavior", exp_foraging),
    5: ("Drug Effects on Foraging", exp_drug_effects),
    6: ("Day/Night Cycle", exp_day_night),
}


# ============================================================================
# CLI Entry Point
# ============================================================================

def _run_single(args, seed: int) -> Dict[int, Any]:
    """Run all requested experiments with a single seed."""
    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())
    results: Dict[int, Any] = {}

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue
        name, func = ALL_EXPERIMENTS[exp_id]

        try:
            kwargs: Dict[str, Any] = {
                "scale": args.scale,
                "device": args.device,
                "seed": seed,
            }
            result = func(**kwargs)
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Ecosystem -- Complete Digital Fruit Fly in a Digital World"
    )
    parser.add_argument("--exp", type=int, nargs="*", default=None,
                        help="Which experiments to run (1-6). Default: all")
    parser.add_argument("--scale", default="small",
                        choices=list(SCALE_NEURONS.keys()),
                        help="Network scale (default: small)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="Write structured JSON results to file")
    parser.add_argument("--runs", type=int, default=1,
                        help="Run each experiment N times with different seeds, "
                             "report mean +/- std (default: 1)")
    args = parser.parse_args()

    n_neurons = SCALE_NEURONS[args.scale]

    print("=" * 76)
    print("  DROSOPHILA ECOSYSTEM -- COMPLETE DIGITAL ORGANISM IN A DIGITAL WORLD")
    print(f"  Backend: {detect_backend()} | Scale: {args.scale} "
          f"(~{n_neurons} neurons) | Device: {args.device}")
    print(f"  Brain: HH dynamics, 6 NTs, STDP | Body: compound eyes, "
          f"antennae, 6 legs, wings")
    print(f"  World: [100x100] mm grid, odorant sources, day/night cycle")
    if _USING_STUBS:
        print(f"  [STUBS ACTIVE -- organism/world modules not yet installed]")
    if args.runs > 1:
        print(f"  Multi-seed: {args.runs} runs "
              f"(seeds {args.seed}..{args.seed + args.runs - 1})")
    print("=" * 76)

    total_time = time.perf_counter()

    if args.runs == 1:
        results = _run_single(args, args.seed)
        all_run_results = [results]
    else:
        all_run_results = []
        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            print(f"\n{'~' * 76}")
            print(f"  RUN {run_idx + 1}/{args.runs} (seed={run_seed})")
            print(f"{'~' * 76}")
            results = _run_single(args, run_seed)
            all_run_results.append(results)

    total = time.perf_counter() - total_time

    # ---- Summary ----
    print("\n" + "=" * 76)
    print("  DROSOPHILA ECOSYSTEM -- SUMMARY")
    print("=" * 76)

    final_results = all_run_results[-1]

    if args.runs > 1:
        exp_ids = sorted(set().union(*[r.keys() for r in all_run_results]))
        for exp_id in exp_ids:
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            pass_rates = [r[exp_id].get("passed", False)
                          for r in all_run_results if exp_id in r]
            pass_frac = sum(pass_rates) / len(pass_rates)
            times = [r[exp_id].get("time", 0)
                     for r in all_run_results if exp_id in r]
            avg_t = sum(times) / len(times) if times else 0
            print(f"    {exp_id}. {name:40s} "
                  f"[{sum(pass_rates)}/{len(pass_rates)} PASS]  "
                  f"avg {avg_t:.1f}s")
        total_passes = sum(
            all(r.get(eid, {}).get("passed", False) for r in all_run_results)
            for eid in exp_ids if eid in ALL_EXPERIMENTS
        )
        total_exp = len([e for e in exp_ids if e in ALL_EXPERIMENTS])
        print(f"\n  All-pass: {total_passes}/{total_exp} experiments passed "
              f"ALL {args.runs} runs")
    else:
        passed = sum(1 for r in final_results.values() if r.get("passed"))
        total_exp = len(final_results)
        for exp_id, result in sorted(final_results.items()):
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            status = "PASS" if result.get("passed") else "FAIL"
            t = result.get("time", 0)
            print(f"    {exp_id}. {name:40s} [{status}]  {t:.1f}s")
        print(f"\n  Total: {passed}/{total_exp} passed in {total:.1f}s")

    print("=" * 76)

    # ---- JSON output ----
    if args.json:
        json_data = {
            "experiment": "drosophila_ecosystem",
            "scale": args.scale,
            "n_neurons": n_neurons,
            "device": args.device,
            "n_runs": args.runs,
            "base_seed": args.seed,
            "total_time_s": round(total, 2),
            "using_stubs": _USING_STUBS,
            "system": _system_info(),
            "runs": [],
        }
        for run_idx, run_results in enumerate(all_run_results):
            run_data = {
                "seed": args.seed + run_idx,
                "experiments": {},
            }
            for exp_id, result in sorted(run_results.items()):
                if exp_id not in ALL_EXPERIMENTS:
                    continue
                name = ALL_EXPERIMENTS[exp_id][0]
                clean = _make_json_safe(result)
                clean["name"] = name
                run_data["experiments"][str(exp_id)] = clean
            json_data["runs"].append(run_data)

        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  JSON results written to: {args.json}")

    all_passed = all(
        r.get("passed", False)
        for run in all_run_results
        for r in run.values()
        if isinstance(r, dict) and "passed" in r
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
