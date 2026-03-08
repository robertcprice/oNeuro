"""Compartmental dendritic tree model.

Each neuron's dendrite is modelled as a tree of cylindrical compartments with
independent ion channel populations.  Voltage in every compartment evolves via
the cable equation:

    C_m * dV/dt  =  -I_ion  +  I_axial  +  I_syn

Axial (inter-compartment) current is computed from cytoplasmic resistivity and
compartment geometry.  Processing order is distal-to-proximal so that children's
axial contributions are available when the parent integrates.

This module does NOT import from ``membrane.py`` to avoid circular
dependencies.  Channel conductances, reversal potentials, and HH rate constants
are inlined from the same biophysical literature values used in
``ion_channels.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Cytoplasmic resistivity (Ohm * cm)
RHO_CYTOPLASM = 150.0

# Membrane capacitance (uF / cm^2)
C_M = 1.0

# Resting potential (mV)
V_REST = -65.0

# Action potential detection threshold (mV)
AP_THRESHOLD = -20.0


# ---------------------------------------------------------------------------
# Inline channel biophysics (avoids membrane.py import)
# ---------------------------------------------------------------------------

class _ChannelKind(Enum):
    Na_v = "Na_v"
    K_leak = "K_leak"
    Ca_v = "Ca_v"


_CHANNEL_DEFAULTS: Dict[_ChannelKind, Dict[str, float]] = {
    _ChannelKind.Na_v: {"g_max": 120.0, "E_rev": 50.0},
    _ChannelKind.K_leak: {"g_max": 0.3, "E_rev": -77.0},
    _ChannelKind.Ca_v: {"g_max": 4.4, "E_rev": 120.0},
}


# --- HH rate functions (scalar) -------------------------------------------

def _alpha_m(V: float) -> float:
    if abs(V + 40.0) < 1e-6:
        return 1.0
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))


def _beta_m(V: float) -> float:
    return 4.0 * math.exp(-(V + 65.0) / 18.0)


def _alpha_h(V: float) -> float:
    return 0.07 * math.exp(-(V + 65.0) / 20.0)


def _beta_h(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))


def _alpha_m_ca(V: float) -> float:
    if abs(V + 27.0) < 1e-6:
        return 0.5
    return 0.055 * (V + 27.0) / (1.0 - math.exp(-(V + 27.0) / 3.8))


def _beta_m_ca(V: float) -> float:
    return 0.94 * math.exp(-(V + 75.0) / 17.0)


def _alpha_h_ca(V: float) -> float:
    return 0.000457 * math.exp(-(V + 13.0) / 50.0)


def _beta_h_ca(V: float) -> float:
    return 0.0065 / (1.0 + math.exp(-(V + 15.0) / 28.0))


# --- Vectorized HH rate functions -----------------------------------------

def _alpha_m_vec(V: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(V + 40.0) < 1e-6, -40.0 + 1e-6, V)
    return np.where(
        np.abs(V + 40.0) < 1e-6,
        1.0,
        0.1 * (safe + 40.0) / (1.0 - np.exp(-(safe + 40.0) / 10.0)),
    )


def _beta_m_vec(V: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h_vec(V: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h_vec(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def _alpha_m_ca_vec(V: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(V + 27.0) < 1e-6, -27.0 + 1e-6, V)
    return np.where(
        np.abs(V + 27.0) < 1e-6,
        0.5,
        0.055 * (safe + 27.0) / (1.0 - np.exp(-(safe + 27.0) / 3.8)),
    )


def _beta_m_ca_vec(V: np.ndarray) -> np.ndarray:
    return 0.94 * np.exp(-(V + 75.0) / 17.0)


def _alpha_h_ca_vec(V: np.ndarray) -> np.ndarray:
    return 0.000457 * np.exp(-(V + 13.0) / 50.0)


def _beta_h_ca_vec(V: np.ndarray) -> np.ndarray:
    return 0.0065 / (1.0 + np.exp(-(V + 15.0) / 28.0))


# ---------------------------------------------------------------------------
# Compartment channel state
# ---------------------------------------------------------------------------

@dataclass
class _CompartmentChannels:
    """Lightweight channel population for a single compartment.

    Stores per-channel-type conductance scaling and gating variables.
    Much simpler than a full ``IonChannelEnsemble`` -- just what is needed
    for the cable equation in each dendritic segment.
    """

    kinds: List[_ChannelKind] = field(default_factory=list)
    g_scale: Dict[_ChannelKind, float] = field(default_factory=dict)

    # Na_v gating (m^3 h)
    m_na: float = 0.0
    h_na: float = 1.0

    # Ca_v gating (m^2 h)
    m_ca: float = 0.0
    h_ca: float = 1.0

    def __post_init__(self) -> None:
        # Initialise gating at resting potential
        am = _alpha_m(V_REST)
        bm = _beta_m(V_REST)
        ah = _alpha_h(V_REST)
        bh = _beta_h(V_REST)
        self.m_na = am / (am + bm)
        self.h_na = ah / (ah + bh)

        amc = _alpha_m_ca(V_REST)
        bmc = _beta_m_ca(V_REST)
        ahc = _alpha_h_ca(V_REST)
        bhc = _beta_h_ca(V_REST)
        self.m_ca = amc / (amc + bmc)
        self.h_ca = ahc / (ahc + bhc)

    def update(self, V: float, dt: float) -> None:
        """Advance gating variables at voltage *V* for *dt* ms."""
        if _ChannelKind.Na_v in self.g_scale:
            am, bm = _alpha_m(V), _beta_m(V)
            ah, bh = _alpha_h(V), _beta_h(V)
            self.m_na += dt * (am * (1.0 - self.m_na) - bm * self.m_na)
            self.h_na += dt * (ah * (1.0 - self.h_na) - bh * self.h_na)
            self.m_na = max(0.0, min(1.0, self.m_na))
            self.h_na = max(0.0, min(1.0, self.h_na))

        if _ChannelKind.Ca_v in self.g_scale:
            amc, bmc = _alpha_m_ca(V), _beta_m_ca(V)
            ahc, bhc = _alpha_h_ca(V), _beta_h_ca(V)
            self.m_ca += dt * (amc * (1.0 - self.m_ca) - bmc * self.m_ca)
            self.h_ca += dt * (ahc * (1.0 - self.h_ca) - bhc * self.h_ca)
            self.m_ca = max(0.0, min(1.0, self.m_ca))
            self.h_ca = max(0.0, min(1.0, self.h_ca))

    def total_current(self, V: float) -> float:
        """Total ionic current I_ion (uA/cm^2) through this channel set."""
        I = 0.0
        for kind in self.g_scale:
            params = _CHANNEL_DEFAULTS[kind]
            g = params["g_max"] * self.g_scale[kind]
            E = params["E_rev"]

            if kind == _ChannelKind.Na_v:
                gating = (self.m_na ** 3) * self.h_na
            elif kind == _ChannelKind.Ca_v:
                gating = (self.m_ca ** 2) * self.h_ca
            elif kind == _ChannelKind.K_leak:
                gating = 1.0
            else:
                gating = 0.0

            I += g * gating * (V - E)
        return I


def _default_channels(distal: bool = False) -> _CompartmentChannels:
    """Standard channel population for a dendritic compartment.

    Most compartments carry Na_v (at reduced density compared to the soma)
    and K_leak.  Distal compartments are enriched with Ca_v channels which
    are critical for dendritic calcium spikes and synaptic plasticity.
    """
    chs = _CompartmentChannels()
    # Na_v at 30 % of somatic density to support back-propagating APs
    chs.g_scale[_ChannelKind.Na_v] = 0.3
    chs.g_scale[_ChannelKind.K_leak] = 1.0
    if distal:
        chs.g_scale[_ChannelKind.Ca_v] = 1.0
    return chs


def _soma_channels() -> _CompartmentChannels:
    """Full-density channel population for the soma compartment."""
    chs = _CompartmentChannels()
    chs.g_scale[_ChannelKind.Na_v] = 1.0
    chs.g_scale[_ChannelKind.K_leak] = 1.0
    chs.g_scale[_ChannelKind.Ca_v] = 0.3
    return chs


# ---------------------------------------------------------------------------
# Compartment
# ---------------------------------------------------------------------------

@dataclass
class Compartment:
    """A single cylindrical cable compartment.

    Attributes:
        id: Unique integer identifier within the tree.
        voltage: Membrane potential (mV).  Evolves via the cable equation.
        diameter_um: Cylinder diameter in micrometres.
        length_um: Cylinder length in micrometres.
        parent_id: Index of the parent compartment (``None`` for the soma).
        children_ids: Indices of child compartments.
        channels: Local ion channel population with conductance scaling.
        ca_concentration_nM: Local intracellular Ca^{2+} concentration (nM).
    """

    id: int
    voltage: float = V_REST
    diameter_um: float = 2.0
    length_um: float = 50.0
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    channels: _CompartmentChannels = field(default_factory=_default_channels)
    ca_concentration_nM: float = 50.0

    # ---- Geometry-derived quantities --------------------------------------

    @property
    def radius_cm(self) -> float:
        """Radius converted to centimetres."""
        return self.diameter_um * 1e-4 / 2.0

    @property
    def length_cm(self) -> float:
        """Length converted to centimetres."""
        return self.length_um * 1e-4

    @property
    def cross_section_area_cm2(self) -> float:
        """Cross-sectional area (pi * r^2) in cm^2."""
        return math.pi * self.radius_cm ** 2

    @property
    def surface_area_cm2(self) -> float:
        """Lateral membrane surface area (pi * d * L) in cm^2."""
        return math.pi * self.diameter_um * 1e-4 * self.length_cm

    @property
    def R_axial(self) -> float:
        """Axial resistance (MOhm) from cytoplasmic geometry.

        R = 4 * rho * L / (pi * d^2)

        rho is in Ohm-cm, d and L must be in cm.  Result is in Ohm, converted
        to MOhm for compatibility with current units (uA).
        """
        d_cm = self.diameter_um * 1e-4
        L_cm = self.length_cm
        if d_cm <= 0:
            return 1e12  # effectively infinite
        R_ohm = 4.0 * RHO_CYTOPLASM * L_cm / (math.pi * d_cm ** 2)
        return R_ohm * 1e-6  # Ohm -> MOhm

    # ---- Electrical coupling -----------------------------------------------

    def compute_axial_current(self, parent_voltage: float) -> float:
        """Axial current flowing from parent into this compartment (uA/cm^2).

        I_axial = (V_parent - V_self) / R_axial, then normalised by the
        compartment's membrane surface area so the cable equation works in
        current-density units.
        """
        R = self.R_axial
        if R <= 0:
            return 0.0
        I_raw_uA = (parent_voltage - self.voltage) / R  # uA (R in MOhm, V in mV)
        area = self.surface_area_cm2
        if area <= 0:
            return 0.0
        return I_raw_uA / area  # uA/cm^2

    # ---- Calcium dynamics --------------------------------------------------

    def _update_calcium(self, dt: float) -> None:
        """Exponential decay of Ca^{2+} toward resting level."""
        ca_rest = 50.0   # nM
        ca_tau = 50.0     # ms
        self.ca_concentration_nM += dt * (ca_rest - self.ca_concentration_nM) / ca_tau
        self.ca_concentration_nM = max(0.0, self.ca_concentration_nM)

        # Voltage-dependent Ca influx through Ca_v channels
        if _ChannelKind.Ca_v in self.channels.g_scale and self.voltage > -30.0:
            g_ca = self.channels.g_scale[_ChannelKind.Ca_v]
            influx = g_ca * max(0.0, self.voltage + 30.0) * 0.5 * dt
            self.ca_concentration_nM += influx


# ---------------------------------------------------------------------------
# Dendritic tree
# ---------------------------------------------------------------------------

@dataclass
class DendriticTree:
    """Tree of :class:`Compartment` objects with the soma at the root.

    The cable equation is solved per-compartment every ``step()``.  Processing
    order is distal-to-proximal (leaves first) so that children contribute
    their axial currents before the parent integrates.
    """

    compartments: List[Compartment] = field(default_factory=list)
    _topo_order: List[int] = field(init=False, default_factory=list, repr=False)

    # ---- Construction API -------------------------------------------------

    @property
    def soma(self) -> Compartment:
        """Root compartment (the soma).  Always at index 0."""
        return self.compartments[0]

    @property
    def soma_voltage(self) -> float:
        """Convenience accessor for the somatic membrane potential."""
        return self.compartments[0].voltage

    @property
    def total_compartments(self) -> int:
        return len(self.compartments)

    def add_compartment(
        self,
        parent_id: int,
        diameter: float = 2.0,
        length: float = 50.0,
        distal: bool = False,
    ) -> int:
        """Append a child compartment and return its id.

        Args:
            parent_id: Id of the parent compartment.
            diameter: Cylinder diameter in um.
            length: Cylinder length in um.
            distal: If ``True``, enrich the compartment with Ca_v channels.

        Returns:
            Integer id of the new compartment.
        """
        new_id = len(self.compartments)
        comp = Compartment(
            id=new_id,
            diameter_um=diameter,
            length_um=length,
            parent_id=parent_id,
            channels=_default_channels(distal=distal),
        )
        self.compartments.append(comp)
        self.compartments[parent_id].children_ids.append(new_id)
        self._rebuild_topo()
        return new_id

    # ---- Topological sort (distal first) -----------------------------------

    def _rebuild_topo(self) -> None:
        """Rebuild reverse BFS order: leaves processed before parents."""
        if not self.compartments:
            self._topo_order = []
            return

        # BFS from soma
        visited: List[int] = []
        queue: List[int] = [0]
        while queue:
            cid = queue.pop(0)
            visited.append(cid)
            queue.extend(self.compartments[cid].children_ids)

        # Reverse gives distal-first order
        self._topo_order = list(reversed(visited))

    # ---- Simulation step ---------------------------------------------------

    def step(self, dt: float, synaptic_inputs: Optional[Dict[int, float]] = None) -> None:
        """Advance all compartments by *dt* ms.

        Args:
            dt: Timestep in ms.
            synaptic_inputs: Mapping from compartment id to injected current
                density (uA/cm^2) representing synaptic drive.

        The cable equation for each compartment is:

            C_m * dV/dt = -I_ion + I_axial_from_parent
                          + sum(I_axial_from_children) + I_syn
        """
        if not self.compartments:
            return
        if not self._topo_order:
            self._rebuild_topo()

        syn = synaptic_inputs or {}

        # Pre-compute child axial contributions (child -> parent current density
        # as seen by the parent's membrane area).
        child_axial_to_parent: Dict[int, float] = {}

        for cid in self._topo_order:
            comp = self.compartments[cid]

            # --- 1. Update channel gating at current voltage
            comp.channels.update(comp.voltage, dt)

            # --- 2. Ionic current
            I_ion = comp.channels.total_current(comp.voltage)

            # --- 3. Axial current from parent
            I_axial_parent = 0.0
            if comp.parent_id is not None:
                parent_v = self.compartments[comp.parent_id].voltage
                I_axial_parent = comp.compute_axial_current(parent_v)

            # --- 4. Axial current from children (summed)
            I_axial_children = child_axial_to_parent.get(cid, 0.0)

            # --- 5. Synaptic input
            I_syn = syn.get(cid, 0.0)

            # --- 6. Integrate: C_m * dV/dt = -I_ion + I_axial + I_syn
            dV = (-I_ion + I_axial_parent + I_axial_children + I_syn) / C_M * dt
            comp.voltage += dV
            comp.voltage = max(-100.0, min(60.0, comp.voltage))

            # --- 7. Calcium dynamics
            comp._update_calcium(dt)

            # --- 8. Record this compartment's axial contribution TO its parent
            if comp.parent_id is not None:
                # Current flowing from child into parent (flip sign for parent's perspective)
                R = comp.R_axial
                if R > 0:
                    I_raw_uA = (comp.voltage - self.compartments[comp.parent_id].voltage) / R
                    parent_area = self.compartments[comp.parent_id].surface_area_cm2
                    if parent_area > 0:
                        I_density = I_raw_uA / parent_area
                        child_axial_to_parent[comp.parent_id] = (
                            child_axial_to_parent.get(comp.parent_id, 0.0) + I_density
                        )

    # ---- Back-propagating action potential ----------------------------------

    def backpropagating_ap(self, soma_voltage: float) -> None:
        """Propagate an action potential from soma outward.

        Sets the soma voltage and attenuates it with distance.  In real
        neurons, bAPs lose ~50 % amplitude over 200-300 um due to decreasing
        Na_v density.  Attenuation factor: exp(-distance / lambda) where
        lambda = 200 um.

        This is a one-shot injection (call once per spike).  The subsequent
        ``step()`` calls will handle the cable dynamics.
        """
        LAMBDA_UM = 200.0  # length constant for bAP attenuation
        self.compartments[0].voltage = soma_voltage

        # Somatic Ca influx from the spike itself
        if soma_voltage > AP_THRESHOLD:
            self.compartments[0].ca_concentration_nM += 500.0

        # BFS from soma, accumulating distance
        distance: Dict[int, float] = {0: 0.0}
        queue: List[int] = list(self.compartments[0].children_ids)
        for cid in self.compartments[0].children_ids:
            distance[cid] = self.compartments[cid].length_um

        while queue:
            cid = queue.pop(0)
            comp = self.compartments[cid]
            d = distance[cid]
            attenuation = math.exp(-d / LAMBDA_UM)
            bap_voltage = V_REST + (soma_voltage - V_REST) * attenuation
            comp.voltage = bap_voltage

            # Calcium influx from bAP (activity-dependent plasticity signal)
            if bap_voltage > AP_THRESHOLD:
                comp.ca_concentration_nM += 200.0 * attenuation

            for child_id in comp.children_ids:
                distance[child_id] = d + self.compartments[child_id].length_um
                queue.append(child_id)

    # ---- Archetype templates (class methods) --------------------------------

    @classmethod
    def pyramidal(cls, n_compartments: int = 10) -> "DendriticTree":
        """Cortical pyramidal neuron dendritic tree.

        Layout (default 10 compartments):
          - Soma (1): large diameter, full channel density
          - Apical trunk (5): long ascending dendrite, progressively thinner
          - Basal dendrites (3): short, thick
          - Oblique branches (2): off the apical trunk

        Distal apical tuft has enriched Ca_v channels for dendritic calcium
        spikes, a hallmark of pyramidal neuron computation.
        """
        tree = cls()

        # Soma
        soma = Compartment(
            id=0,
            diameter_um=20.0,
            length_um=20.0,
            parent_id=None,
            channels=_soma_channels(),
        )
        tree.compartments.append(soma)

        # Budget: n_compartments total.  Distribute as ~50 % apical, 30 %
        # basal, 20 % oblique (minimum 1 each).
        n_apical = max(1, int(n_compartments * 0.5))
        n_basal = max(1, int(n_compartments * 0.3))
        n_oblique = max(1, n_compartments - 1 - n_apical - n_basal)

        # Apical dendrite: progressively thinner, distal is Ca_v-enriched
        parent = 0
        for i in range(n_apical):
            diameter = max(0.5, 4.0 - i * (3.0 / max(1, n_apical - 1)))
            is_distal = (i >= n_apical - 2)
            cid = tree.add_compartment(
                parent_id=parent,
                diameter=diameter,
                length=80.0,
                distal=is_distal,
            )
            parent = cid

        # Basal dendrites: branch from soma, shorter and thicker
        for i in range(n_basal):
            tree.add_compartment(
                parent_id=0,
                diameter=3.0,
                length=40.0,
                distal=(i == n_basal - 1),
            )

        # Oblique branches: off the middle of the apical trunk
        mid_apical = min(n_apical // 2 + 1, len(tree.compartments) - 1)
        for i in range(n_oblique):
            tree.add_compartment(
                parent_id=mid_apical,
                diameter=1.5,
                length=60.0,
                distal=True,
            )

        return tree

    @classmethod
    def interneuron(cls, n_compartments: int = 5) -> "DendriticTree":
        """Fast-spiking interneuron dendritic tree.

        Short, relatively uniform dendrites with high Na_v density for fast
        conduction.  Aspiny (no spines) -- synaptic input arrives directly
        on the dendritic shaft.
        """
        tree = cls()

        # Soma
        soma = Compartment(
            id=0,
            diameter_um=15.0,
            length_um=15.0,
            parent_id=None,
            channels=_soma_channels(),
        )
        tree.compartments.append(soma)

        # Short symmetric dendrites radiating from soma
        n_dendrites = max(1, n_compartments - 1)
        for i in range(n_dendrites):
            chs = _default_channels(distal=False)
            # Interneurons have higher Na_v density for fast propagation
            chs.g_scale[_ChannelKind.Na_v] = 0.6
            tree.add_compartment(
                parent_id=0,
                diameter=2.0,
                length=30.0,
                distal=False,
            )
            # Override the channels that add_compartment created
            tree.compartments[-1].channels = chs

        return tree

    @classmethod
    def purkinje(cls, n_compartments: int = 15) -> "DendriticTree":
        """Cerebellar Purkinje cell dendritic tree.

        Massive planar dendritic arbour with extremely high Ca_v density.
        Purkinje cells have the most elaborate dendritic tree of any neuron,
        receiving ~200,000 parallel fibre synapses.

        Layout:
          - Soma (1)
          - Primary trunk (2)
          - Secondary branches (4-6): fan out in a plane
          - Tertiary branches: dense terminal arborisation
        """
        tree = cls()

        # Soma -- large
        soma = Compartment(
            id=0,
            diameter_um=25.0,
            length_um=25.0,
            parent_id=None,
            channels=_soma_channels(),
        )
        tree.compartments.append(soma)

        # Primary trunk
        n_primary = 2
        parent = 0
        for _ in range(n_primary):
            parent = tree.add_compartment(
                parent_id=parent,
                diameter=6.0,
                length=40.0,
                distal=False,
            )

        trunk_tip = parent

        # Distribute remaining compartments across secondary and tertiary
        remaining = n_compartments - 1 - n_primary
        n_secondary = max(2, remaining // 2)
        n_tertiary = max(0, remaining - n_secondary)

        # Secondary branches fan out from trunk tip
        secondary_tips: List[int] = []
        for i in range(n_secondary):
            diameter = max(1.0, 3.0 - i * 0.5)
            cid = tree.add_compartment(
                parent_id=trunk_tip,
                diameter=diameter,
                length=50.0,
                distal=True,
            )
            secondary_tips.append(cid)

        # Tertiary branches off secondary tips (round-robin)
        for i in range(n_tertiary):
            tip_parent = secondary_tips[i % len(secondary_tips)] if secondary_tips else trunk_tip
            tree.add_compartment(
                parent_id=tip_parent,
                diameter=0.8,
                length=30.0,
                distal=True,
            )

        return tree
