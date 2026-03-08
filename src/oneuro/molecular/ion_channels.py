"""Ion channels as folded proteins with Hodgkin-Huxley gating.

Each channel type (Na_v, K_v, Ca_v, NMDA, AMPA, etc.) has real conductance
values, reversal potentials, and gating kinetics. When nQPU is available,
channel protein structure is computed via quantum protein folding; otherwise
classical HH rate constants are used directly.

Voltage emerges from: dV/dt = -I_ion / C_m
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from oneuro.molecular.backend import HAS_NQPU, get_nqpu_chem, quantum_channel_gating


class IonChannelType(Enum):
    """Major ion channel families in neural tissue."""

    Na_v = "voltage-gated sodium"
    K_v = "voltage-gated potassium"
    K_leak = "potassium leak"
    Ca_v = "voltage-gated calcium"
    NMDA = "NMDA receptor channel"
    AMPA = "AMPA receptor channel"
    GABA_A = "GABA-A receptor channel"
    nAChR = "nicotinic acetylcholine receptor"


# Biophysical parameters from Hodgkin-Huxley and literature.
# g_max in mS/cm², E_rev in mV.
_CHANNEL_PARAMS: Dict[IonChannelType, dict] = {
    IonChannelType.Na_v: {
        "g_max": 120.0,
        "E_rev": 50.0,
        "ion": "Na+",
        "gates": "m3h",
        "protein_sequence": "SCN1A",
    },
    IonChannelType.K_v: {
        "g_max": 36.0,
        "E_rev": -77.0,
        "ion": "K+",
        "gates": "n4",
        "protein_sequence": "KCNA1",
    },
    IonChannelType.K_leak: {
        "g_max": 0.3,
        "E_rev": -77.0,
        "ion": "K+",
        "gates": "always_open",
        "protein_sequence": "KCNK2",
    },
    IonChannelType.Ca_v: {
        "g_max": 4.4,
        "E_rev": 120.0,
        "ion": "Ca2+",
        "gates": "m2h",
        "protein_sequence": "CACNA1A",
    },
    IonChannelType.NMDA: {
        "g_max": 0.5,
        "E_rev": 0.0,
        "ion": "Ca2+/Na+",
        "gates": "ligand_voltage",
        "protein_sequence": "GRIN1/GRIN2A",
    },
    IonChannelType.AMPA: {
        "g_max": 1.0,
        "E_rev": 0.0,
        "ion": "Na+/K+",
        "gates": "ligand",
        "protein_sequence": "GRIA1",
    },
    IonChannelType.GABA_A: {
        "g_max": 1.0,
        "E_rev": -80.0,
        "ion": "Cl-",
        "gates": "ligand",
        "protein_sequence": "GABRA1",
    },
    IonChannelType.nAChR: {
        "g_max": 0.8,
        "E_rev": 0.0,
        "ion": "Na+/K+/Ca2+",
        "gates": "ligand",
        "protein_sequence": "CHRNA4",
    },
}


def _alpha_m(V: float) -> float:
    """Na+ activation rate (HH)."""
    if abs(V + 40.0) < 1e-6:
        return 1.0
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))


def _beta_m(V: float) -> float:
    return 4.0 * math.exp(-(V + 65.0) / 18.0)


def _alpha_h(V: float) -> float:
    """Na+ inactivation rate (HH)."""
    return 0.07 * math.exp(-(V + 65.0) / 20.0)


def _beta_h(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))


def _alpha_n(V: float) -> float:
    """K+ activation rate (HH)."""
    if abs(V + 55.0) < 1e-6:
        return 0.1
    return 0.01 * (V + 55.0) / (1.0 - math.exp(-(V + 55.0) / 10.0))


def _beta_n(V: float) -> float:
    return 0.125 * math.exp(-(V + 65.0) / 80.0)


def _alpha_m_ca(V: float) -> float:
    """Ca2+ activation rate."""
    if abs(V + 27.0) < 1e-6:
        return 0.5
    return 0.055 * (V + 27.0) / (1.0 - math.exp(-(V + 27.0) / 3.8))


def _beta_m_ca(V: float) -> float:
    return 0.94 * math.exp(-(V + 75.0) / 17.0)


def _alpha_h_ca(V: float) -> float:
    return 0.000457 * math.exp(-(V + 13.0) / 50.0)


def _beta_h_ca(V: float) -> float:
    return 0.0065 / (1.0 + math.exp(-(V + 15.0) / 28.0))


# ---------------------------------------------------------------------------
# Vectorized HH rate functions for batch computation
# ---------------------------------------------------------------------------

def _alpha_m_vec(V: np.ndarray) -> np.ndarray:
    """Na+ activation rate — vectorized."""
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


def _alpha_n_vec(V: np.ndarray) -> np.ndarray:
    """K+ activation rate — vectorized."""
    safe = np.where(np.abs(V + 55.0) < 1e-6, -55.0 + 1e-6, V)
    return np.where(
        np.abs(V + 55.0) < 1e-6,
        0.1,
        0.01 * (safe + 55.0) / (1.0 - np.exp(-(safe + 55.0) / 10.0)),
    )


def _beta_n_vec(V: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


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


@dataclass
class BatchIonChannelState:
    """Vectorised gating state across N neurons for a single channel type.

    Maintains m, h, n as numpy arrays. update() applies HH dynamics in bulk.
    """

    gate_type: str
    m: np.ndarray = field(default_factory=lambda: np.array([]))
    h: np.ndarray = field(default_factory=lambda: np.array([]))
    n: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_channels(cls, channels: List["IonChannel"]) -> "BatchIonChannelState":
        """Collect gating variables from a list of IonChannel objects."""
        if not channels:
            return cls(gate_type="none")
        gate_type = channels[0]._params["gates"]
        m = np.array([ch.m for ch in channels])
        h = np.array([ch.h for ch in channels])
        n = np.array([ch.n for ch in channels])
        return cls(gate_type=gate_type, m=m, h=h, n=n)

    def update(self, V: np.ndarray, dt: float) -> None:
        """Update all gating variables in batch."""
        if self.gate_type == "m3h":
            am, bm = _alpha_m_vec(V), _beta_m_vec(V)
            ah, bh = _alpha_h_vec(V), _beta_h_vec(V)
            self.m += dt * (am * (1.0 - self.m) - bm * self.m)
            self.h += dt * (ah * (1.0 - self.h) - bh * self.h)
            np.clip(self.m, 0.0, 1.0, out=self.m)
            np.clip(self.h, 0.0, 1.0, out=self.h)
        elif self.gate_type == "n4":
            an, bn = _alpha_n_vec(V), _beta_n_vec(V)
            self.n += dt * (an * (1.0 - self.n) - bn * self.n)
            np.clip(self.n, 0.0, 1.0, out=self.n)
        elif self.gate_type == "m2h":
            amc, bmc = _alpha_m_ca_vec(V), _beta_m_ca_vec(V)
            ahc, bhc = _alpha_h_ca_vec(V), _beta_h_ca_vec(V)
            self.m += dt * (amc * (1.0 - self.m) - bmc * self.m)
            self.h += dt * (ahc * (1.0 - self.h) - bhc * self.h)
            np.clip(self.m, 0.0, 1.0, out=self.m)
            np.clip(self.h, 0.0, 1.0, out=self.h)

    def write_back(self, channels: List["IonChannel"]) -> None:
        """Write updated gating variables back to individual channels."""
        for i, ch in enumerate(channels):
            ch.m = float(self.m[i]) if len(self.m) > i else ch.m
            ch.h = float(self.h[i]) if len(self.h) > i else ch.h
            ch.n = float(self.n[i]) if len(self.n) > i else ch.n


@dataclass
class IonChannel:
    """A single ion channel type with Hodgkin-Huxley gating variables.

    Computes ionic current as: I = g_max * gating_product * (V - E_rev)
    """

    channel_type: IonChannelType
    count: int = 1  # Number of channels of this type in patch
    _params: dict = field(init=False, repr=False)

    # Gating variables
    m: float = field(init=False, default=0.0)
    h: float = field(init=False, default=1.0)
    n: float = field(init=False, default=0.0)

    # Drug-modifiable conductance multiplier (float, unlike integer count)
    conductance_scale: float = field(init=False, default=1.0)

    # Ligand-gated state
    ligand_open_fraction: float = field(init=False, default=0.0)

    # NMDA Mg2+ block
    _mg_conc_mM: float = field(init=False, default=1.0)

    def __post_init__(self):
        self._params = _CHANNEL_PARAMS[self.channel_type]
        # Initialize gating variables at resting potential (-65 mV)
        V_rest = -65.0
        if self._params["gates"] == "m3h":
            am, bm = _alpha_m(V_rest), _beta_m(V_rest)
            ah, bh = _alpha_h(V_rest), _beta_h(V_rest)
            self.m = am / (am + bm)
            self.h = ah / (ah + bh)
        elif self._params["gates"] == "n4":
            an, bn = _alpha_n(V_rest), _beta_n(V_rest)
            self.n = an / (an + bn)
        elif self._params["gates"] == "m2h":
            amc, bmc = _alpha_m_ca(V_rest), _beta_m_ca(V_rest)
            ahc, bhc = _alpha_h_ca(V_rest), _beta_h_ca(V_rest)
            self.m = amc / (amc + bmc)
            self.h = ahc / (ahc + bhc)

    @property
    def g_max(self) -> float:
        return self._params["g_max"]

    @property
    def E_rev(self) -> float:
        return self._params["E_rev"]

    @property
    def ion(self) -> str:
        return self._params["ion"]

    def _quantum_gating_correction(self, V: float, temperature_K: float = 310.0) -> float:
        """Quantum correction factor for voltage-gated channel transitions.

        Conformational changes in voltage sensor S4 helices involve charge
        transfer through energy barriers where quantum tunneling is relevant.
        Returns multiplicative correction >= 1.0 for transition rates.
        """
        gate_type = self._params["gates"]
        if gate_type not in ("m3h", "n4", "m2h"):
            return 1.0
        channel_name_map = {"m3h": "Na_v", "n4": "K_v", "m2h": "Ca_v"}
        return quantum_channel_gating(channel_name_map[gate_type], V, temperature_K)

    def update(self, V: float, dt: float) -> None:
        """Update gating variables for one timestep at voltage V (mV)."""
        gate_type = self._params["gates"]

        # Quantum correction enhances transition rates for voltage-gated channels
        qcorr = self._quantum_gating_correction(V) if HAS_NQPU else 1.0
        dt_eff = dt * qcorr

        if gate_type == "m3h":
            am, bm = _alpha_m(V), _beta_m(V)
            ah, bh = _alpha_h(V), _beta_h(V)
            self.m += dt_eff * (am * (1.0 - self.m) - bm * self.m)
            self.h += dt_eff * (ah * (1.0 - self.h) - bh * self.h)
            self.m = max(0.0, min(1.0, self.m))
            self.h = max(0.0, min(1.0, self.h))

        elif gate_type == "n4":
            an, bn = _alpha_n(V), _beta_n(V)
            self.n += dt_eff * (an * (1.0 - self.n) - bn * self.n)
            self.n = max(0.0, min(1.0, self.n))

        elif gate_type == "m2h":
            amc, bmc = _alpha_m_ca(V), _beta_m_ca(V)
            ahc, bhc = _alpha_h_ca(V), _beta_h_ca(V)
            self.m += dt_eff * (amc * (1.0 - self.m) - bmc * self.m)
            self.h += dt_eff * (ahc * (1.0 - self.h) - bhc * self.h)
            self.m = max(0.0, min(1.0, self.m))
            self.h = max(0.0, min(1.0, self.h))

        elif gate_type == "ligand":
            # Ligand-gated: open fraction set externally via set_ligand_concentration
            pass

        elif gate_type == "ligand_voltage":
            # NMDA: both ligand and voltage dependent (Mg2+ block)
            pass

    def set_ligand_concentration(self, concentration_nM: float, EC50_nM: float = 500.0,
                                  hill_n: float = 1.5) -> None:
        """Set open fraction from ligand concentration via Hill equation."""
        if concentration_nM <= 0:
            self.ligand_open_fraction = 0.0
        else:
            self.ligand_open_fraction = (
                concentration_nM ** hill_n
                / (EC50_nM ** hill_n + concentration_nM ** hill_n)
            )

    def current(self, V: float) -> float:
        """Compute ionic current I (uA/cm²) at voltage V.

        I = g_max * gating * (V - E_rev) * count_factor
        Positive current = outward (depolarizing for reversal > V).
        """
        gate_type = self._params["gates"]
        g = self.g_max * self.count * self.conductance_scale

        if gate_type == "m3h":
            gating = self.m ** 3 * self.h
        elif gate_type == "n4":
            gating = self.n ** 4
        elif gate_type == "m2h":
            gating = self.m ** 2 * self.h
        elif gate_type == "always_open":
            gating = 1.0
        elif gate_type == "ligand":
            gating = self.ligand_open_fraction
        elif gate_type == "ligand_voltage":
            # NMDA: ligand gating * Mg2+ block factor
            mg_block = 1.0 / (1.0 + self._mg_conc_mM * math.exp(-0.062 * V) / 3.57)
            gating = self.ligand_open_fraction * mg_block
        else:
            gating = 0.0

        return g * gating * (V - self.E_rev)


@dataclass
class IonChannelEnsemble:
    """Collection of ion channels in a membrane patch.

    Computes total ionic current from all channel types.
    This is the biophysical heart of the membrane model.
    """

    channels: Dict[IonChannelType, IonChannel] = field(default_factory=dict)

    def add_channel(self, channel_type: IonChannelType, count: int = 1) -> None:
        if channel_type in self.channels:
            self.channels[channel_type].count += count
        else:
            ch = IonChannel(channel_type=channel_type, count=count)
            self.channels[channel_type] = ch

    def update_all(self, V: float, dt: float) -> None:
        """Update all gating variables at current voltage."""
        for ch in self.channels.values():
            ch.update(V, dt)

    def total_current(self, V: float) -> float:
        """Sum of ionic currents from all channels (uA/cm²)."""
        return sum(ch.current(V) for ch in self.channels.values())

    def set_synaptic_concentration(
        self, channel_type: IonChannelType, concentration_nM: float
    ) -> None:
        """Set ligand concentration for a receptor channel."""
        if channel_type in self.channels:
            ch = self.channels[channel_type]
            # EC50 values from literature
            ec50_map = {
                IonChannelType.NMDA: 3000.0,
                IonChannelType.AMPA: 500.0,
                IonChannelType.GABA_A: 200.0,
                IonChannelType.nAChR: 30.0,
            }
            ec50 = ec50_map.get(channel_type, 500.0)
            ch.set_ligand_concentration(concentration_nM, EC50_nM=ec50)

    def get_channel(self, channel_type: IonChannelType) -> Optional[IonChannel]:
        return self.channels.get(channel_type)

    @classmethod
    def standard_hh(cls) -> "IonChannelEnsemble":
        """Create a standard Hodgkin-Huxley channel ensemble (Na, K, leak)."""
        ens = cls()
        ens.add_channel(IonChannelType.Na_v, count=1)
        ens.add_channel(IonChannelType.K_v, count=1)
        ens.add_channel(IonChannelType.K_leak, count=1)
        return ens

    @classmethod
    def excitatory_postsynaptic(cls) -> "IonChannelEnsemble":
        """Excitatory postsynaptic membrane: HH + AMPA + NMDA."""
        ens = cls.standard_hh()
        ens.add_channel(IonChannelType.AMPA, count=1)
        ens.add_channel(IonChannelType.NMDA, count=1)
        ens.add_channel(IonChannelType.Ca_v, count=1)
        return ens

    @classmethod
    def inhibitory_postsynaptic(cls) -> "IonChannelEnsemble":
        """Inhibitory postsynaptic membrane: HH + GABA-A."""
        ens = cls.standard_hh()
        ens.add_channel(IonChannelType.GABA_A, count=1)
        return ens
