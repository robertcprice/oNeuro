"""Dendritic spine morphology and structural plasticity.

Dendritic spines are tiny protrusions from the dendritic shaft that house
excitatory postsynaptic machinery.  Their morphology is tightly coupled to
synaptic strength: larger spines contain more AMPA receptors and a larger
post-synaptic density (PSD).

Three canonical morphological states exist on a continuum:

    THIN  ->  STUBBY  ->  MUSHROOM
      |         |            |
    weak     moderate     strong  (synaptic weight)
    high     moderate      low   (structural plasticity)

Structural LTP/LTD is driven by actin polymerisation/depolymerisation in the
spine head.  This module models the slow (seconds-to-minutes) morphological
dynamics that consolidate or weaken synaptic connections, complementing the
fast (millisecond) receptor trafficking in the synapse module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SpineState(Enum):
    """Canonical spine morphological types.

    THIN:     Filopodia-like, small head, long thin neck.  High motility,
              few AMPA receptors, learning-ready.
    STUBBY:   Short wide neck, moderate head.  Transitional form.
    MUSHROOM: Large bulbous head, constricted neck.  Stable, memory spine,
              maximal AMPA capacity.
    """

    THIN = "thin"
    STUBBY = "stubby"
    MUSHROOM = "mushroom"


# ---------------------------------------------------------------------------
# Morphological parameters by state
# ---------------------------------------------------------------------------

_STATE_DEFAULTS = {
    SpineState.THIN: {
        "volume_fL": 0.01,
        "psd_area_um2": 0.02,
        "neck_resistance_MOhm": 500.0,
    },
    SpineState.STUBBY: {
        "volume_fL": 0.05,
        "psd_area_um2": 0.06,
        "neck_resistance_MOhm": 100.0,
    },
    SpineState.MUSHROOM: {
        "volume_fL": 0.10,
        "psd_area_um2": 0.12,
        "neck_resistance_MOhm": 300.0,
    },
}

# PSD area to AMPA capacity conversion factor.
# Empirical: ~300 AMPA receptors per 0.1 um^2 PSD area (Nusser et al. 1998).
AMPA_PER_PSD_UM2 = 3000.0

# Volume thresholds for state transitions (femtolitres)
_THIN_TO_STUBBY_VOLUME = 0.03
_STUBBY_TO_MUSHROOM_VOLUME = 0.08
_MUSHROOM_TO_STUBBY_VOLUME = 0.06
_STUBBY_TO_THIN_VOLUME = 0.02

# Actin dynamics time constant (ms).
# Real structural plasticity unfolds over minutes, but the model operates at
# per-event granularity where each ``structural_ltp`` call represents an
# integrated burst of CaMKII activity, not a single millisecond.
ACTIN_TAU_MS = 500.0  # convergence half-life ~350 ms of model time


@dataclass
class DendriticSpine:
    """A single dendritic spine with morphology-dependent synaptic capacity.

    The spine's volume determines how many AMPA receptors it can hold.
    Structural LTP (driven by CaMKII / actin polymerisation) enlarges the
    spine; structural LTD (driven by calcineurin / actin depolymerisation)
    shrinks it.  Morphological state transitions (thin <-> stubby <-> mushroom)
    happen when volume crosses biophysical thresholds.

    Attributes:
        volume_fL: Spine head volume in femtolitres.
        psd_area_um2: Post-synaptic density area in um^2.
        neck_resistance_MOhm: Spine neck electrical resistance.
        ampa_count: Current number of AMPA receptors inserted.
        actin_polymerization: Fraction of actin in polymerised (F-actin) form.
        state: Current morphological classification.
        age_ms: Time since spine formation.
    """

    volume_fL: float = 0.01
    psd_area_um2: float = 0.02
    neck_resistance_MOhm: float = 500.0
    ampa_count: int = 0
    actin_polymerization: float = 0.3
    state: SpineState = SpineState.THIN
    age_ms: float = 0.0

    # Internal dynamics
    _actin_target: float = field(init=False, default=0.3)
    _volume_target: float = field(init=False, default=0.01)

    def __post_init__(self) -> None:
        self._actin_target = self.actin_polymerization
        self._volume_target = self.volume_fL

    # ---- Properties --------------------------------------------------------

    @property
    def ampa_capacity(self) -> int:
        """Maximum AMPA receptors this spine can hold, given its PSD area."""
        return max(1, int(self.psd_area_um2 * AMPA_PER_PSD_UM2))

    @property
    def max_ampa_receptors(self) -> int:
        """Alias: maximum AMPA receptors based on current PSD area."""
        return self.ampa_capacity

    @property
    def occupancy(self) -> float:
        """Fraction of AMPA slots filled (0-1)."""
        cap = self.ampa_capacity
        if cap <= 0:
            return 0.0
        return min(1.0, self.ampa_count / cap)

    @property
    def synaptic_weight(self) -> float:
        """Effective synaptic weight (0-1) combining spine size and AMPA count.

        Accounts for both the number of receptors and the neck resistance
        (which electrically isolates the spine from the dendrite).
        """
        # Conductance contribution: proportional to AMPA count
        g_ampa = self.ampa_count * 0.01  # ~10 pS per AMPA receptor
        # Neck attenuation: larger neck resistance = more isolation
        attenuation = 1.0 / (1.0 + self.neck_resistance_MOhm * 0.001)
        return min(1.0, g_ampa * attenuation)

    @property
    def is_mature(self) -> bool:
        """Whether the spine is considered structurally stable (mushroom)."""
        return self.state == SpineState.MUSHROOM

    # ---- Structural plasticity ---------------------------------------------

    def structural_ltp(self, activity_level: float) -> None:
        """Activity-driven spine enlargement (structural long-term potentiation).

        High synaptic activity (via CaMKII) promotes actin polymerisation,
        which increases spine head volume, PSD area, and AMPA receptor capacity.
        This is the morphological consolidation of Hebbian learning.

        Args:
            activity_level: Normalised activity intensity (0-1).
                0 = no activity, 1 = maximal (e.g. strong tetanic stimulation).
        """
        activity = max(0.0, min(1.0, activity_level))
        if activity < 0.1:
            return

        # Actin polymerisation drive: proportional to activity
        # Mushroom spines have a ceiling -- they resist further growth
        saturation = 1.0 - (self.volume_fL / 0.15)  # diminishing returns past 0.15 fL
        saturation = max(0.0, saturation)

        delta_actin = activity * 0.2 * saturation
        self._actin_target = min(1.0, self.actin_polymerization + delta_actin)

        # Volume growth: additive increment proportional to actin drive.
        # Real spine enlargement is ~2-3x over 30-60 min of strong stimulation.
        # Additive (not multiplicative on current volume) so small spines can
        # actually reach the transition thresholds.
        growth_fL = delta_actin * 0.02  # ~0.004 fL per strong LTP event
        self._volume_target = self.volume_fL + growth_fL

        # Cap volume at biophysical maximum
        self._volume_target = min(0.15, self._volume_target)

    def structural_ltd(self, activity_level: float) -> None:
        """Activity-driven spine shrinkage (structural long-term depression).

        Low-frequency stimulation or calcineurin activation promotes actin
        depolymerisation, shrinking the spine head and reducing AMPA capacity.

        Args:
            activity_level: Normalised activity intensity (0-1).
                Higher values = stronger depression.
        """
        activity = max(0.0, min(1.0, activity_level))
        if activity < 0.1:
            return

        # Actin depolymerisation: proportional to activity
        # Thin spines resist further shrinkage
        floor_factor = self.volume_fL / 0.01  # more to shrink = easier to shrink
        floor_factor = min(1.0, floor_factor)

        delta_actin = activity * 0.15 * floor_factor
        self._actin_target = max(0.05, self.actin_polymerization - delta_actin)

        # Volume shrinkage: additive decrement
        shrink_fL = delta_actin * 0.015
        self._volume_target = self.volume_fL - shrink_fL

        # Floor at minimal spine volume
        self._volume_target = max(0.005, self._volume_target)

    # ---- Time-step integration ---------------------------------------------

    def step(self, dt: float) -> None:
        """Advance slow morphological dynamics by *dt* ms.

        Actin polymerisation and volume converge toward their target values
        with an exponential time constant.  PSD area and neck resistance are
        updated from volume, and AMPA count is clamped to the new capacity.

        Args:
            dt: Timestep in ms.
        """
        self.age_ms += dt

        # --- Actin dynamics: exponential approach to target
        alpha = 1.0 - math.exp(-dt / ACTIN_TAU_MS)
        self.actin_polymerization += alpha * (self._actin_target - self.actin_polymerization)
        self.actin_polymerization = max(0.0, min(1.0, self.actin_polymerization))

        # --- Volume dynamics: exponential approach to target
        vol_tau = ACTIN_TAU_MS * 2.0  # volume lags behind actin
        beta = 1.0 - math.exp(-dt / vol_tau)
        self.volume_fL += beta * (self._volume_target - self.volume_fL)
        self.volume_fL = max(0.005, min(0.15, self.volume_fL))

        # --- PSD area scales with volume^(2/3) (surface-to-volume scaling)
        # Reference: mushroom at 0.10 fL has PSD 0.12 um^2
        ref_volume = 0.10
        ref_psd = 0.12
        self.psd_area_um2 = ref_psd * (self.volume_fL / ref_volume) ** (2.0 / 3.0)
        self.psd_area_um2 = max(0.005, self.psd_area_um2)

        # --- Neck resistance: inversely related to spine width
        # Thin spines: high neck resistance (electrical isolation)
        # Mushroom spines: intermediate (large head, moderate neck)
        # Stubby spines: low (essentially no neck)
        if self.volume_fL < _THIN_TO_STUBBY_VOLUME:
            # Thin: long narrow neck
            self.neck_resistance_MOhm = 500.0 * (0.03 / max(0.005, self.volume_fL))
            self.neck_resistance_MOhm = min(1500.0, self.neck_resistance_MOhm)
        elif self.volume_fL < _STUBBY_TO_MUSHROOM_VOLUME:
            # Stubby: minimal neck
            self.neck_resistance_MOhm = 80.0 + 120.0 * (
                (_STUBBY_TO_MUSHROOM_VOLUME - self.volume_fL)
                / (_STUBBY_TO_MUSHROOM_VOLUME - _THIN_TO_STUBBY_VOLUME)
            )
        else:
            # Mushroom: constricted neck with large head
            self.neck_resistance_MOhm = 200.0 + 200.0 * (
                self.volume_fL / 0.15
            )

        # --- Clamp AMPA count to new capacity
        cap = self.ampa_capacity
        if self.ampa_count > cap:
            self.ampa_count = cap

        # --- State transitions based on volume thresholds
        self._update_state()

    def _update_state(self) -> None:
        """Reclassify spine morphology based on current volume."""
        if self.state == SpineState.THIN:
            if self.volume_fL >= _THIN_TO_STUBBY_VOLUME:
                self.state = SpineState.STUBBY
        elif self.state == SpineState.STUBBY:
            if self.volume_fL >= _STUBBY_TO_MUSHROOM_VOLUME:
                self.state = SpineState.MUSHROOM
            elif self.volume_fL <= _STUBBY_TO_THIN_VOLUME:
                self.state = SpineState.THIN
        elif self.state == SpineState.MUSHROOM:
            if self.volume_fL <= _MUSHROOM_TO_STUBBY_VOLUME:
                self.state = SpineState.STUBBY

    # ---- Convenience constructors ------------------------------------------

    @classmethod
    def thin(cls) -> "DendriticSpine":
        """Create a thin (learning-ready) spine."""
        defaults = _STATE_DEFAULTS[SpineState.THIN]
        return cls(
            volume_fL=defaults["volume_fL"],
            psd_area_um2=defaults["psd_area_um2"],
            neck_resistance_MOhm=defaults["neck_resistance_MOhm"],
            ampa_count=5,
            actin_polymerization=0.2,
            state=SpineState.THIN,
        )

    @classmethod
    def stubby(cls) -> "DendriticSpine":
        """Create a stubby (transitional) spine."""
        defaults = _STATE_DEFAULTS[SpineState.STUBBY]
        return cls(
            volume_fL=defaults["volume_fL"],
            psd_area_um2=defaults["psd_area_um2"],
            neck_resistance_MOhm=defaults["neck_resistance_MOhm"],
            ampa_count=30,
            actin_polymerization=0.5,
            state=SpineState.STUBBY,
        )

    @classmethod
    def mushroom(cls) -> "DendriticSpine":
        """Create a mushroom (memory) spine."""
        defaults = _STATE_DEFAULTS[SpineState.MUSHROOM]
        return cls(
            volume_fL=defaults["volume_fL"],
            psd_area_um2=defaults["psd_area_um2"],
            neck_resistance_MOhm=defaults["neck_resistance_MOhm"],
            ampa_count=80,
            actin_polymerization=0.8,
            state=SpineState.MUSHROOM,
        )
