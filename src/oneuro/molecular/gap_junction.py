"""Electrical synapses (gap junctions) for direct intercellular coupling.

Gap junctions are clusters of connexin hemichannels that form direct
electrical connections between cells. Unlike chemical synapses, they are:
  - Bidirectional (current flows down the voltage gradient)
  - Fast (no synaptic delay)
  - Modulatable by voltage, pH, and intracellular Ca2+

Key connexin types:
  - Cx36: Neuronal gap junctions (interneuron networks, synchronization)
  - Cx43: Astrocytic gap junctions (glial syncytium, K+ buffering)
  - Cx32: Oligodendrocytic (myelin maintenance, metabolic support)

Current: I = g * p_open * (V_pre - V_post)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ConnexinType(Enum):
    """Major connexin families in neural tissue."""

    Cx36 = "neuronal"
    Cx43 = "astrocytic"
    Cx32 = "oligodendrocytic"


# Default conductance ranges (nS) from electrophysiology literature
_CONNEXIN_PARAMS: dict = {
    ConnexinType.Cx36: {
        "g_default_nS": 0.5,
        "g_range_nS": (0.1, 1.0),
        "description": "Neuronal gap junctions (interneuron synchronization)",
        "voltage_half_inactivation_mV": 60.0,  # Relatively voltage-insensitive
        "voltage_steepness": 15.0,
    },
    ConnexinType.Cx43: {
        "g_default_nS": 5.0,
        "g_range_nS": (1.0, 10.0),
        "description": "Astrocytic gap junctions (glial syncytium)",
        "voltage_half_inactivation_mV": 50.0,
        "voltage_steepness": 12.0,
    },
    ConnexinType.Cx32: {
        "g_default_nS": 2.0,
        "g_range_nS": (0.5, 5.0),
        "description": "Oligodendrocytic (myelin metabolic coupling)",
        "voltage_half_inactivation_mV": 55.0,
        "voltage_steepness": 14.0,
    },
}


@dataclass
class GapJunction:
    """An electrical synapse formed by connexin hemichannel pairs.

    Provides direct, bidirectional electrical coupling between two cells.
    Current magnitude and direction are determined by the voltage difference
    between coupled cells, modulated by open probability.

    Open probability is dynamically regulated by:
      - Transjunctional voltage (voltage sensitivity)
      - Intracellular pH (acidosis closes junctions)
      - Intracellular Ca2+ (high calcium closes junctions)

    Examples:
        >>> gj = GapJunction(connexin=ConnexinType.Cx36, pre_id=0, post_id=1)
        >>> current = gj.step(dt=0.1, v_pre=-65.0, v_post=-70.0)
        >>> current  # Positive: current flows pre -> post (down gradient)
    """

    connexin: ConnexinType
    pre_id: int
    post_id: int

    # Conductance in nanosiemens
    conductance_nS: float = 0.0  # 0 triggers auto-set from connexin type

    # Gating state
    open_probability: float = 0.8

    # Internal modulation state
    _voltage_gate: float = field(init=False, default=1.0)
    _ph_gate: float = field(init=False, default=1.0)
    _ca_gate: float = field(init=False, default=1.0)

    # Connexin-specific parameters (loaded from library)
    _params: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._params = _CONNEXIN_PARAMS[self.connexin]

        # Auto-set conductance from connexin type if not explicitly provided
        if self.conductance_nS == 0.0:
            self.conductance_nS = self._params["g_default_nS"]

        # Validate conductance is within physiological range
        g_min, g_max = self._params["g_range_nS"]
        self.conductance_nS = max(g_min, min(g_max, self.conductance_nS))

        # Initialize gating to match open_probability
        self._voltage_gate = 1.0
        self._ph_gate = 1.0
        self._ca_gate = 1.0

    def compute_current(self, v_pre: float, v_post: float) -> float:
        """Compute instantaneous gap junction current in nanoamperes.

        I = g * p_open * (V_pre - V_post)

        Convention: positive current flows from pre to post (i.e., when
        V_pre > V_post, current flows pre -> post, depolarizing post).

        Args:
            v_pre: Membrane potential of first cell (mV).
            v_post: Membrane potential of second cell (mV).

        Returns:
            Current in nA. Positive = flows from pre to post.
        """
        return self.conductance_nS * self.open_probability * (v_pre - v_post)

    def voltage_sensitivity(self, v_diff: float) -> float:
        """Compute voltage-dependent gating factor.

        Transjunctional voltage gating: open probability decreases when
        the voltage difference exceeds ~40 mV. This is a fundamental
        property of all connexins, varying in sensitivity by type.

        Uses a Boltzmann function:
            gate = 1 / (1 + exp((|V_diff| - V_half) / steepness))

        Args:
            v_diff: Transjunctional voltage V_pre - V_post (mV).

        Returns:
            Voltage gating factor in [0, 1].
        """
        v_half = self._params["voltage_half_inactivation_mV"]
        steepness = self._params["voltage_steepness"]

        abs_diff = abs(v_diff)
        return 1.0 / (1.0 + math.exp((abs_diff - v_half) / steepness))

    def ph_sensitivity(self, ph: float) -> float:
        """Compute pH-dependent gating factor.

        Gap junctions close during intracellular acidosis (pH < 6.5).
        This protects healthy cells from damaged neighbors that release
        acid during ischemia or injury.

        Uses a Hill-like function centered at pH 6.5:
            gate = pH^4 / (pH_half^4 + pH^4)

        Args:
            ph: Intracellular pH (normal ~7.2, acidotic < 6.5).

        Returns:
            pH gating factor in [0, 1].
        """
        ph_half = 6.5
        hill_n = 4.0

        if ph <= 0:
            return 0.0

        return ph**hill_n / (ph_half**hill_n + ph**hill_n)

    def ca_sensitivity(self, ca_nM: float) -> float:
        """Compute calcium-dependent gating factor.

        High intracellular Ca2+ (> 1000 nM) closes gap junctions.
        This is a protective mechanism: cells with pathologically high
        calcium (dying, ischemic) are decoupled from the network.

        Uses an inverted Hill function:
            gate = IC50^2 / (IC50^2 + [Ca]^2)

        Args:
            ca_nM: Intracellular calcium concentration in nM.
                Normal resting: 50-100 nM. Pathological: > 1000 nM.

        Returns:
            Calcium gating factor in [0, 1].
        """
        ic50 = 1000.0  # nM, half-inactivation concentration
        hill_n = 2.0

        if ca_nM <= 0:
            return 1.0

        return ic50**hill_n / (ic50**hill_n + ca_nM**hill_n)

    def step(
        self,
        dt: float,
        v_pre: float,
        v_post: float,
        ph: float = 7.2,
        ca_nM: float = 100.0,
    ) -> float:
        """Advance gap junction state by dt and return current.

        Updates open probability based on all modulatory inputs
        (voltage, pH, Ca2+), then computes the resulting current.

        The open probability adjusts toward the target with a time
        constant of ~5 ms, preventing instantaneous gating changes.

        Args:
            dt: Timestep in milliseconds.
            v_pre: Membrane potential of pre cell (mV).
            v_post: Membrane potential of post cell (mV).
            ph: Intracellular pH (default 7.2, normal).
            ca_nM: Intracellular calcium in nM (default 100, resting).

        Returns:
            Gap junction current in nA. Positive = pre to post.
        """
        # Compute individual gating factors
        v_diff = v_pre - v_post
        self._voltage_gate = self.voltage_sensitivity(v_diff)
        self._ph_gate = self.ph_sensitivity(ph)
        self._ca_gate = self.ca_sensitivity(ca_nM)

        # Target open probability is the product of all gates
        target_p = self._voltage_gate * self._ph_gate * self._ca_gate

        # Smooth transition with ~5 ms time constant
        tau = 5.0  # ms
        alpha = 1.0 - math.exp(-dt / tau)
        self.open_probability += alpha * (target_p - self.open_probability)
        self.open_probability = max(0.0, min(1.0, self.open_probability))

        return self.compute_current(v_pre, v_post)

    @property
    def is_open(self) -> bool:
        """Whether the junction is functionally open (p > 0.1)."""
        return self.open_probability > 0.1

    @property
    def effective_conductance_nS(self) -> float:
        """Actual conductance accounting for open probability."""
        return self.conductance_nS * self.open_probability

    def reverse_current(self, v_pre: float, v_post: float) -> float:
        """Current from post to pre perspective (convenience method).

        Returns the negative of compute_current, i.e., the current
        experienced by the pre cell due to this junction.
        """
        return -self.compute_current(v_pre, v_post)

    # ---- Factory class methods ----

    @classmethod
    def neuronal(cls, pre_id: int, post_id: int, conductance_nS: float = 0.5) -> GapJunction:
        """Create a Cx36 neuronal gap junction.

        Found between inhibitory interneurons, enabling fast synchronization
        of gamma oscillations and sharp-wave ripples.

        Args:
            pre_id: First neuron ID.
            post_id: Second neuron ID.
            conductance_nS: Conductance in nS (default 0.5, range 0.1-1.0).
        """
        return cls(
            connexin=ConnexinType.Cx36,
            pre_id=pre_id,
            post_id=post_id,
            conductance_nS=conductance_nS,
        )

    @classmethod
    def astrocytic(cls, pre_id: int, post_id: int, conductance_nS: float = 5.0) -> GapJunction:
        """Create a Cx43 astrocytic gap junction.

        Forms the glial syncytium, enabling K+ spatial buffering,
        Ca2+ wave propagation, and metabolite sharing across astrocytes.

        Args:
            pre_id: First astrocyte ID.
            post_id: Second astrocyte ID.
            conductance_nS: Conductance in nS (default 5.0, range 1.0-10.0).
        """
        return cls(
            connexin=ConnexinType.Cx43,
            pre_id=pre_id,
            post_id=post_id,
            conductance_nS=conductance_nS,
        )

    @classmethod
    def oligodendrocytic(
        cls, pre_id: int, post_id: int, conductance_nS: float = 2.0
    ) -> GapJunction:
        """Create a Cx32 oligodendrocytic gap junction.

        Couples oligodendrocytes for metabolic support of myelin sheaths
        and coordinated myelination responses.

        Args:
            pre_id: First oligodendrocyte ID.
            post_id: Second oligodendrocyte ID.
            conductance_nS: Conductance in nS (default 2.0, range 0.5-5.0).
        """
        return cls(
            connexin=ConnexinType.Cx32,
            pre_id=pre_id,
            post_id=post_id,
            conductance_nS=conductance_nS,
        )

    def __repr__(self) -> str:
        return (
            f"GapJunction({self.connexin.name}, "
            f"{self.pre_id}<->{self.post_id}, "
            f"g={self.conductance_nS:.1f} nS, "
            f"p_open={self.open_probability:.2f})"
        )
