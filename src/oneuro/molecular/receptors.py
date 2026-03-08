"""Synaptic receptors as protein structures.

Two major families:
- Ionotropic (fast): ligand opens an ion channel directly
- Metabotropic (slow): ligand triggers G-protein cascade

Binding uses Hill equation or nQPU quantum docking when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oneuro.molecular.backend import HAS_NQPU, get_nqpu_chem
from oneuro.molecular.ion_channels import IonChannelType


class ReceptorType(Enum):
    """Receptor classification."""

    # Ionotropic (fast, direct channel opening)
    AMPA = "AMPA"
    NMDA = "NMDA"
    KAINATE = "Kainate"
    GABA_A = "GABA-A"
    nAChR = "nAChR"
    GLYCINE = "Glycine"

    # Metabotropic (slow, G-protein cascade)
    D1 = "D1"
    D2 = "D2"
    HT_1A = "5-HT1A"
    HT_2A = "5-HT2A"
    mAChR_M1 = "mAChR-M1"
    mGluR1 = "mGluR1"
    GABA_B = "GABA-B"
    ALPHA_1 = "alpha-1"
    ALPHA_2 = "alpha-2"
    BETA_1 = "beta-1"


# Which NT binds which receptor, and the properties
_RECEPTOR_PROPERTIES = {
    ReceptorType.AMPA: {
        "neurotransmitter": "glutamate",
        "ionotropic": True,
        "channel_type": IonChannelType.AMPA,
        "EC50_nM": 480.0,
        "hill_n": 1.3,
        "rise_time_ms": 0.5,
        "decay_time_ms": 5.0,
        "desensitization_ms": 15.0,
    },
    ReceptorType.NMDA: {
        "neurotransmitter": "glutamate",
        "ionotropic": True,
        "channel_type": IonChannelType.NMDA,
        "EC50_nM": 2400.0,
        "hill_n": 1.5,
        "rise_time_ms": 5.0,
        "decay_time_ms": 150.0,
        "desensitization_ms": 500.0,
    },
    ReceptorType.GABA_A: {
        "neurotransmitter": "gaba",
        "ionotropic": True,
        "channel_type": IonChannelType.GABA_A,
        "EC50_nM": 200.0,
        "hill_n": 2.0,
        "rise_time_ms": 0.5,
        "decay_time_ms": 30.0,
        "desensitization_ms": 200.0,
    },
    ReceptorType.nAChR: {
        "neurotransmitter": "acetylcholine",
        "ionotropic": True,
        "channel_type": IonChannelType.nAChR,
        "EC50_nM": 30.0,
        "hill_n": 1.8,
        "rise_time_ms": 0.2,
        "decay_time_ms": 3.0,
        "desensitization_ms": 50.0,
    },
    ReceptorType.D1: {
        "neurotransmitter": "dopamine",
        "ionotropic": False,
        "EC50_nM": 2340.0,
        "hill_n": 1.0,
        "rise_time_ms": 50.0,
        "decay_time_ms": 500.0,
        "cascade_effect": "cAMP_increase",
    },
    ReceptorType.D2: {
        "neurotransmitter": "dopamine",
        "ionotropic": False,
        "EC50_nM": 2.8,
        "hill_n": 1.0,
        "rise_time_ms": 50.0,
        "decay_time_ms": 500.0,
        "cascade_effect": "cAMP_decrease",
    },
    ReceptorType.HT_1A: {
        "neurotransmitter": "serotonin",
        "ionotropic": False,
        "EC50_nM": 3.2,
        "hill_n": 1.0,
        "rise_time_ms": 100.0,
        "decay_time_ms": 1000.0,
        "cascade_effect": "cAMP_decrease",
    },
    ReceptorType.HT_2A: {
        "neurotransmitter": "serotonin",
        "ionotropic": False,
        "EC50_nM": 54.0,
        "hill_n": 1.2,
        "rise_time_ms": 100.0,
        "decay_time_ms": 800.0,
        "cascade_effect": "IP3_DAG_increase",
    },
    ReceptorType.mAChR_M1: {
        "neurotransmitter": "acetylcholine",
        "ionotropic": False,
        "EC50_nM": 7900.0,
        "hill_n": 1.0,
        "rise_time_ms": 50.0,
        "decay_time_ms": 300.0,
        "cascade_effect": "IP3_DAG_increase",
    },
    ReceptorType.GABA_B: {
        "neurotransmitter": "gaba",
        "ionotropic": False,
        "EC50_nM": 35.0,
        "hill_n": 1.5,
        "rise_time_ms": 50.0,
        "decay_time_ms": 300.0,
        "cascade_effect": "K_channel_open",
    },
    ReceptorType.ALPHA_1: {
        "neurotransmitter": "norepinephrine",
        "ionotropic": False,
        "EC50_nM": 330.0,
        "hill_n": 1.0,
        "rise_time_ms": 50.0,
        "decay_time_ms": 500.0,
        "cascade_effect": "IP3_DAG_increase",
    },
    ReceptorType.ALPHA_2: {
        "neurotransmitter": "norepinephrine",
        "ionotropic": False,
        "EC50_nM": 56.0,
        "hill_n": 1.0,
        "rise_time_ms": 50.0,
        "decay_time_ms": 500.0,
        "cascade_effect": "cAMP_decrease",
    },
}


@dataclass
class SynapticReceptor:
    """A synaptic receptor that responds to neurotransmitter concentration.

    Ionotropic receptors directly modulate ion channel conductance.
    Metabotropic receptors produce a slower modulatory signal.
    """

    receptor_type: ReceptorType
    count: int = 1  # Number of receptors at this synapse
    _props: dict = field(init=False, repr=False)

    # State
    activation: float = field(init=False, default=0.0)  # [0, 1]
    desensitization: float = field(init=False, default=0.0)  # [0, 1]
    _cascade_signal: float = field(init=False, default=0.0)

    def __post_init__(self):
        self._props = _RECEPTOR_PROPERTIES.get(self.receptor_type, {})

    @property
    def is_ionotropic(self) -> bool:
        return self._props.get("ionotropic", False)

    @property
    def neurotransmitter(self) -> str:
        return self._props.get("neurotransmitter", "unknown")

    @property
    def channel_type(self) -> Optional[IonChannelType]:
        return self._props.get("channel_type")

    @property
    def EC50_nM(self) -> float:
        return self._props.get("EC50_nM", 500.0)

    def bind(self, concentration_nM: float) -> float:
        """Compute receptor activation from NT concentration (Hill equation).

        Returns fractional activation [0, 1] accounting for desensitization.
        Receptor count does NOT scale activation — it represents trafficking
        state for STDP (insert = LTP, remove = LTD) and synaptic weight.
        The binding fraction is a per-receptor property.
        """
        ec50 = self._props.get("EC50_nM", 500.0)
        hill_n = self._props.get("hill_n", 1.0)

        if concentration_nM <= 0:
            raw = 0.0
        else:
            raw = concentration_nM ** hill_n / (ec50 ** hill_n + concentration_nM ** hill_n)

        # Apply desensitization (count-independent)
        effective = raw * (1.0 - self.desensitization)
        self.activation = min(effective, 1.0)
        return self.activation

    def update(self, concentration_nM: float, dt: float) -> None:
        """Update receptor state over dt milliseconds."""
        self.bind(concentration_nM)

        # Desensitization accumulates with sustained activation
        desens_rate = self.activation / max(self._props.get("desensitization_ms", 200.0), 0.01)
        recovery_rate = (1.0 - self.activation) * 0.01  # Slow recovery
        self.desensitization += dt * (desens_rate - recovery_rate)
        self.desensitization = max(0.0, min(1.0, self.desensitization))

        # Metabotropic cascade signal
        if not self.is_ionotropic:
            rise_tau = self._props.get("rise_time_ms", 50.0)
            decay_tau = self._props.get("decay_time_ms", 500.0)
            target = self.activation
            if self._cascade_signal < target:
                self._cascade_signal += dt * (target - self._cascade_signal) / rise_tau
            else:
                self._cascade_signal += dt * (target - self._cascade_signal) / decay_tau
            self._cascade_signal = max(0.0, min(1.0, self._cascade_signal))

    @property
    def cascade_signal(self) -> float:
        """Metabotropic cascade output [0, 1]. Always 0 for ionotropic."""
        if self.is_ionotropic:
            return 0.0
        return self._cascade_signal

    def reset_desensitization(self) -> None:
        """Reset desensitization (e.g., after prolonged silence)."""
        self.desensitization = 0.0
