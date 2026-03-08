"""nQPU detection and fallback machinery.

Every molecular class checks HAS_NQPU and uses quantum chemistry when
available, otherwise falls back to numpy-only approximations.

The actual Python module is ``nqpu_metal`` (a PyO3/maturin extension).
Previous code tried ``from nqpu import chem`` which doesn't exist.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Optional

import numpy as np

HAS_NQPU = False
_nqpu_metal = None
_nqpu_simulator = None

try:
    import nqpu_metal as _nqpu_metal_module

    _nqpu_metal = _nqpu_metal_module
    HAS_NQPU = True
except (ImportError, ModuleNotFoundError):
    pass


def get_nqpu_metal():
    """Return the nqpu_metal module or None."""
    return _nqpu_metal


def get_nqpu_simulator():
    """Return a configured PyQuantumSimulator, or None."""
    global _nqpu_simulator
    if not HAS_NQPU:
        return None
    if _nqpu_simulator is None:
        try:
            _nqpu_simulator = _nqpu_metal.QuantumSimulator(num_qubits=8)
        except Exception:
            return None
    return _nqpu_simulator


# ---- Legacy accessors for backward compat ----

def get_nqpu_chem():
    """Legacy accessor — returns nqpu_metal module (has Molecule, etc.)."""
    return _nqpu_metal


def get_nqpu_bio():
    """Legacy accessor — returns nqpu_metal module."""
    return _nqpu_metal


def require_nqpu(func):
    """Decorator that raises ImportError if nQPU is not available."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_NQPU:
            raise ImportError(
                f"{func.__name__} requires nQPU. "
                "Install with: cd quantum/nqpu-metal && maturin develop --release"
            )
        return func(*args, **kwargs)

    return wrapper


class FallbackMolecule:
    """Minimal molecule stand-in when nQPU is unavailable."""

    def __init__(self, smiles: str, name: str = ""):
        self.smiles = smiles
        self.name = name
        self.atoms: list = []
        self.molecular_weight_val = 0.0

    def molecular_weight(self) -> float:
        return self.molecular_weight_val

    @classmethod
    def from_smiles(cls, smiles: str, name: str = "") -> "FallbackMolecule":
        return cls(smiles, name)


def make_molecule(smiles: str, name: str = "") -> Any:
    """Create a Molecule from SMILES, using nQPU if available."""
    if HAS_NQPU:
        try:
            mol = _nqpu_metal.Molecule(name or "mol")
            mol.set_smiles(smiles)
            return mol
        except Exception:
            pass
    return FallbackMolecule.from_smiles(smiles, name)


# ---- Quantum helper functions ----

def quantum_enzyme_tunneling(barrier_eV: float, mass_amu: float = 1.008,
                              temperature_K: float = 310.0) -> float:
    """Compute quantum tunneling probability through an enzyme barrier.

    Uses WKB approximation: T = exp(-2 * sqrt(2*m*V) * d / hbar)
    When nQPU is available, uses its quantum simulator for a more
    accurate result. Falls back to semiclassical WKB otherwise.

    Args:
        barrier_eV: Barrier height in electron-volts.
        mass_amu: Tunneling particle mass in AMU (default: proton).
        temperature_K: Temperature in Kelvin.

    Returns:
        Tunneling probability [0, 1].
    """
    if HAS_NQPU:
        try:
            sim = get_nqpu_simulator()
            if sim is not None and hasattr(sim, 'tunneling_probability'):
                return sim.tunneling_probability(barrier_eV, mass_amu, temperature_K)
        except Exception:
            pass

    # WKB semiclassical approximation
    hbar = 1.0546e-34  # J·s
    eV_to_J = 1.602e-19
    amu_to_kg = 1.661e-27
    kB = 1.381e-23  # J/K

    V = barrier_eV * eV_to_J
    m = mass_amu * amu_to_kg
    # Typical enzyme barrier width ~0.5 Angstrom for H-transfer
    d = 0.5e-10  # meters

    kappa = math.sqrt(2.0 * m * V) / hbar
    transmission = math.exp(-2.0 * kappa * d)

    # Thermal enhancement factor
    thermal_energy = kB * temperature_K
    thermal_factor = 1.0 + thermal_energy / V if V > 0 else 1.0

    return min(1.0, transmission * thermal_factor)


def quantum_channel_gating(channel_type: str, voltage_mV: float,
                            temperature_K: float = 310.0) -> float:
    """Quantum correction factor for voltage-gated ion channel transitions.

    Conformational changes in voltage sensors involve proton/charge transfer
    through energy barriers where quantum tunneling is relevant at biological
    temperatures. Returns a multiplicative correction to classical HH rates.

    Args:
        channel_type: "Na_v", "K_v", or "Ca_v".
        voltage_mV: Membrane voltage.
        temperature_K: Temperature.

    Returns:
        Correction factor >= 1.0 (1.0 = no quantum effect).
    """
    if HAS_NQPU:
        try:
            sim = get_nqpu_simulator()
            if sim is not None and hasattr(sim, 'channel_gating_correction'):
                return sim.channel_gating_correction(channel_type, voltage_mV, temperature_K)
        except Exception:
            pass

    # Semiclassical estimate: voltage sensor S4 helix charge transfer
    # Barrier heights from MD simulations (Bhatt & Bhatt 2018)
    barrier_map = {
        "Na_v": 0.25,   # eV, S4 helix charge transfer
        "K_v": 0.30,    # eV
        "Ca_v": 0.28,   # eV
    }
    barrier_eV = barrier_map.get(channel_type, 0.0)
    if barrier_eV <= 0:
        return 1.0

    # Proton mass tunneling through gating charge barrier
    prob = quantum_enzyme_tunneling(barrier_eV, mass_amu=1.008, temperature_K=temperature_K)
    # Small correction: tunneling enhances transition rate by probability factor
    return 1.0 + prob * 0.1  # Conservative 10% of tunneling probability


def quantum_protein_fold(sequence: str) -> Optional[dict]:
    """Use nQPU Grover-based protein folding to predict structure.

    Args:
        sequence: Amino acid sequence (1-letter codes).

    Returns:
        Dict with fold_energy, secondary_structure, etc., or None.
    """
    if not HAS_NQPU:
        return None
    try:
        if hasattr(_nqpu_metal, 'protein_fold'):
            return _nqpu_metal.protein_fold(sequence)
    except Exception:
        pass
    return None
