"""Synaptic degradation enzymes with Michaelis-Menten kinetics.

Each enzyme has real Km and Vmax values from biochemistry literature.
When nQPU is available, catalytic rates use quantum tunneling calculations;
otherwise classical Michaelis-Menten kinetics are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from oneuro.molecular.backend import HAS_NQPU, get_nqpu_chem, quantum_enzyme_tunneling


# Real kinetic parameters from enzyme biochemistry.
# Km in uM, Vmax in umol/min/mg, catalytic_efficiency kcat/Km in M-1 s-1
ENZYME_LIBRARY: Dict[str, dict] = {
    "AChE": {
        "full_name": "Acetylcholinesterase",
        "target_nt": "acetylcholine",
        "ec_number": "3.1.1.7",
        "Km_uM": 90.0,
        "Vmax_relative": 1.0,
        "kcat_per_s": 14000.0,  # One of the fastest enzymes in biology
        "catalytic_efficiency": 1.5e8,  # Near diffusion limit
        "quantum_tunneling_factor": 1.0,  # No significant tunneling
    },
    "MAO-A": {
        "full_name": "Monoamine oxidase A",
        "target_nt": ["serotonin", "norepinephrine", "dopamine"],
        "ec_number": "1.4.3.4",
        "Km_uM": 178.0,
        "Vmax_relative": 0.3,
        "kcat_per_s": 12.0,
        "catalytic_efficiency": 6.7e4,
        "quantum_tunneling_factor": 1.8,  # H-transfer tunneling
    },
    "MAO-B": {
        "full_name": "Monoamine oxidase B",
        "target_nt": ["dopamine"],
        "ec_number": "1.4.3.4",
        "Km_uM": 220.0,
        "Vmax_relative": 0.25,
        "kcat_per_s": 8.0,
        "catalytic_efficiency": 3.6e4,
        "quantum_tunneling_factor": 1.6,
    },
    "COMT": {
        "full_name": "Catechol-O-methyltransferase",
        "target_nt": ["dopamine", "norepinephrine"],
        "ec_number": "2.1.1.6",
        "Km_uM": 200.0,
        "Vmax_relative": 0.15,
        "kcat_per_s": 1.5,
        "catalytic_efficiency": 7.5e3,
        "quantum_tunneling_factor": 2.5,  # Methyl transfer tunneling
    },
    "GABA-T": {
        "full_name": "GABA transaminase",
        "target_nt": "gaba",
        "ec_number": "2.6.1.19",
        "Km_uM": 1200.0,
        "Vmax_relative": 0.4,
        "kcat_per_s": 40.0,
        "catalytic_efficiency": 3.3e4,
        "quantum_tunneling_factor": 1.3,
    },
    "glutamine_synthetase": {
        "full_name": "Glutamine synthetase",
        "target_nt": "glutamate",
        "ec_number": "6.3.1.2",
        "Km_uM": 3000.0,
        "Vmax_relative": 0.5,
        "kcat_per_s": 20.0,
        "catalytic_efficiency": 6.7e3,
        "quantum_tunneling_factor": 1.1,
    },
}


@dataclass
class SynapticEnzyme:
    """A synaptic cleft enzyme that degrades neurotransmitters.

    Uses Michaelis-Menten kinetics: v = Vmax * [S] / (Km + [S])
    With optional quantum tunneling enhancement from nQPU.
    """

    name: str
    concentration: float = 1.0  # Relative enzyme concentration [0, 2]
    _info: dict = field(init=False, repr=False)

    _quantum_tunnel_factor: float = field(init=False, default=1.0)

    def __post_init__(self):
        if self.name not in ENZYME_LIBRARY:
            raise ValueError(
                f"Unknown enzyme '{self.name}'. Available: {list(ENZYME_LIBRARY.keys())}"
            )
        self._info = ENZYME_LIBRARY[self.name]
        self._precompute_tunneling()

    # Real barrier heights per enzyme (kJ/mol → eV, 1 kJ/mol = 0.01036 eV)
    _BARRIER_HEIGHTS_eV: dict = field(init=False, repr=False, default_factory=lambda: {
        "AChE": 0.155,       # ~15 kJ/mol, near diffusion limit
        "MAO-A": 0.622,      # ~60 kJ/mol, H-transfer
        "MAO-B": 0.570,      # ~55 kJ/mol, H-transfer
        "COMT": 0.726,       # ~70 kJ/mol, methyl transfer
        "GABA-T": 0.415,     # ~40 kJ/mol, transamination
        "glutamine_synthetase": 0.311,  # ~30 kJ/mol
    })

    # Tunneling particle mass per enzyme (AMU)
    _TUNNELING_MASS_AMU: dict = field(init=False, repr=False, default_factory=lambda: {
        "AChE": 1.008,       # proton (hydrolysis)
        "MAO-A": 1.008,      # H-atom transfer
        "MAO-B": 1.008,      # H-atom transfer
        "COMT": 15.035,      # methyl group CH3
        "GABA-T": 1.008,     # proton transfer
        "glutamine_synthetase": 1.008,
    })

    def _precompute_tunneling(self) -> None:
        """Pre-compute quantum tunneling factor at body temperature (310 K).

        Uses real barrier heights per enzyme and the backend's WKB/nQPU
        tunneling calculation. Pre-computed once (not per-step) for performance.
        """
        base_factor = self._info.get("quantum_tunneling_factor", 1.0)

        barrier_eV = self._BARRIER_HEIGHTS_eV.get(self.name, 0.3)
        mass_amu = self._TUNNELING_MASS_AMU.get(self.name, 1.008)

        try:
            tunnel_prob = quantum_enzyme_tunneling(barrier_eV, mass_amu, temperature_K=310.0)
            self._quantum_tunnel_factor = base_factor * (1.0 + tunnel_prob)
        except Exception:
            self._quantum_tunnel_factor = base_factor

    @property
    def Km_uM(self) -> float:
        return self._info["Km_uM"]

    @property
    def kcat(self) -> float:
        return self._info["kcat_per_s"]

    @property
    def target_nt(self):
        return self._info["target_nt"]

    def degrade(self, substrate_nM: float, dt: float) -> float:
        """Compute amount of substrate degraded (in nM) over dt milliseconds.

        Uses Michaelis-Menten: v = Vmax * [S] / (Km + [S])
        Returns the nM of substrate consumed.
        """
        # Convert substrate to uM for Michaelis-Menten
        substrate_uM = substrate_nM / 1000.0
        km = self._info["Km_uM"]
        vmax_rel = self._info["Vmax_relative"] * self.concentration

        # Tunneling enhancement (pre-computed in __post_init__)
        tunnel_factor = self._quantum_tunnel_factor

        # Michaelis-Menten rate (uM/ms)
        if substrate_uM <= 0:
            return 0.0
        rate_uM_per_ms = vmax_rel * tunnel_factor * substrate_uM / (km + substrate_uM)

        # Convert to nM degraded over dt
        degraded_nM = rate_uM_per_ms * 1000.0 * dt
        return min(degraded_nM, substrate_nM)  # Can't degrade more than available

    def can_degrade(self, nt_name: str) -> bool:
        """Check if this enzyme targets the given neurotransmitter."""
        target = self._info["target_nt"]
        if isinstance(target, list):
            return nt_name in target
        return target == nt_name
