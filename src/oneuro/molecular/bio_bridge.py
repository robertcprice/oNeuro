"""BioState → molecular neurotransmitter concentrations bridge.

Maps entropy project's 12-dim BioState vector to actual NT concentrations
using physiological concentration ranges from neuroscience literature.

BioState dimensions (from entropy project):
  0: cardiac_phase (0-2pi)
  1: coherence (0-1)
  2: dopamine (0-1)
  3: serotonin (0-1)
  4: acetylcholine (0-1)
  5: norepinephrine (0-1)
  6: somatic_comfort (0-1)
  7: somatic_arousal (0-1)
  8: neural_stress (0-1)
  9: encoding_boost (0-2)
  10: vagal_tone (0-1)
  11: reserved (0-1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from typing import Union

from oneuro.molecular.neurotransmitters import NEUROTRANSMITTER_LIBRARY


# BioState dimension indices
_BIO_CARDIAC_PHASE = 0
_BIO_COHERENCE = 1
_BIO_DOPAMINE = 2
_BIO_SEROTONIN = 3
_BIO_ACETYLCHOLINE = 4
_BIO_NOREPINEPHRINE = 5
_BIO_SOMATIC_COMFORT = 6
_BIO_SOMATIC_AROUSAL = 7
_BIO_NEURAL_STRESS = 8
_BIO_ENCODING_BOOST = 9
_BIO_VAGAL_TONE = 10


def _lerp(t: float, low: float, high: float) -> float:
    """Linear interpolation: t=0 → low, t=1 → high."""
    return low + t * (high - low)


@dataclass
class NeuroBioState:
    """Maps a 12-dim BioState vector to physiological NT concentrations.

    The mapping uses the bio signal as a normalized position within
    each NT's physiological concentration range (from literature).

    Example:
        bio_state[2] = 0.8 (dopamine dimension)
        → dopamine conc = lerp(0.8, 5.0, 500.0) = 401 nM
    """

    # Mapping configuration: which BioState dim controls which NT
    _dim_map: Dict[str, int] = field(default_factory=lambda: {
        "dopamine": _BIO_DOPAMINE,
        "serotonin": _BIO_SEROTONIN,
        "acetylcholine": _BIO_ACETYLCHOLINE,
        "norepinephrine": _BIO_NOREPINEPHRINE,
    })

    # Additional mappings from composite signals
    _composite_maps: Dict[str, list] = field(default_factory=lambda: {
        # GABA: high vagal tone + comfort → more inhibition
        "gaba": [
            (_BIO_VAGAL_TONE, 0.5),
            (_BIO_SOMATIC_COMFORT, 0.3),
            (_BIO_COHERENCE, 0.2),
        ],
        # Glutamate: arousal + stress + encoding → more excitation
        "glutamate": [
            (_BIO_SOMATIC_AROUSAL, 0.4),
            (_BIO_NEURAL_STRESS, 0.3),
            (_BIO_ENCODING_BOOST, 0.3),
        ],
    })

    def to_concentrations(self, bio_state: np.ndarray) -> Dict[str, float]:
        """Convert 12-dim BioState vector to NT concentrations in nM.

        Args:
            bio_state: numpy array of shape (12,) with values in [0, 1]
                       (except encoding_boost which is [0, 2]).

        Returns:
            Dict mapping NT name to concentration in nM.
        """
        assert len(bio_state) >= 11, f"BioState must have >= 11 dims, got {len(bio_state)}"

        concentrations: Dict[str, float] = {}

        # Direct mappings (1:1 bio dim → NT)
        for nt_name, dim_idx in self._dim_map.items():
            info = NEUROTRANSMITTER_LIBRARY[nt_name]
            low, high = info["conc_range_nM"]
            signal = float(np.clip(bio_state[dim_idx], 0.0, 1.0))
            concentrations[nt_name] = _lerp(signal, low, high)

        # Composite mappings (weighted sum of bio dims → NT)
        for nt_name, components in self._composite_maps.items():
            info = NEUROTRANSMITTER_LIBRARY[nt_name]
            low, high = info["conc_range_nM"]
            signal = 0.0
            for dim_idx, weight in components:
                val = float(bio_state[dim_idx]) if dim_idx < len(bio_state) else 0.0
                signal += np.clip(val, 0.0, 1.0) * weight
            signal = min(1.0, signal)
            concentrations[nt_name] = _lerp(signal, low, high)

        return concentrations

    def from_concentrations(self, concentrations: Dict[str, float]) -> np.ndarray:
        """Inverse mapping: NT concentrations → approximate BioState vector.

        Useful for reading network state back into BioState representation.
        """
        bio_state = np.zeros(12)

        # Direct mappings (inverse)
        for nt_name, dim_idx in self._dim_map.items():
            info = NEUROTRANSMITTER_LIBRARY[nt_name]
            low, high = info["conc_range_nM"]
            conc = concentrations.get(nt_name, info["resting_conc_nM"])
            bio_state[dim_idx] = np.clip((conc - low) / (high - low), 0.0, 1.0)

        # Composite mappings: approximate inverse (use dominant component)
        for nt_name, components in self._composite_maps.items():
            info = NEUROTRANSMITTER_LIBRARY[nt_name]
            low, high = info["conc_range_nM"]
            conc = concentrations.get(nt_name, info["resting_conc_nM"])
            signal = np.clip((conc - low) / (high - low), 0.0, 1.0)
            # Distribute signal back to dimensions by weight
            for dim_idx, weight in components:
                if dim_idx < len(bio_state):
                    bio_state[dim_idx] = max(bio_state[dim_idx], signal * weight / max(w for _, w in components))

        return bio_state

    @staticmethod
    def resting_concentrations() -> Dict[str, float]:
        """Resting-state NT concentrations from literature."""
        return {
            name: info["resting_conc_nM"]
            for name, info in NEUROTRANSMITTER_LIBRARY.items()
        }

    @staticmethod
    def resting_bio_state() -> np.ndarray:
        """Default resting BioState vector."""
        state = np.zeros(12)
        state[_BIO_COHERENCE] = 0.5
        state[_BIO_VAGAL_TONE] = 0.6
        state[_BIO_SOMATIC_COMFORT] = 0.5
        return state


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert torch.Tensor or numpy array to numpy, no torch dependency."""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    # Quacks like a torch tensor
    if hasattr(tensor_or_array, "detach") and hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


@dataclass
class BioLoRABridge:
    """Bridge between Bio-LoRA's 12-dim bio state and MolecularNeuralNetwork.

    Accepts torch.Tensor OR numpy array (no torch dependency required).
    Uses NeuroBioState for the actual mapping.

    Usage:
        bridge = BioLoRABridge()
        bridge.drive_network(network, bio_state_tensor, blend=0.5)
        state_back = bridge.read_network_state(network)
    """

    _neuro_bridge: NeuroBioState = field(default_factory=NeuroBioState)

    def torch_to_concentrations(self, bio_state) -> Dict[str, float]:
        """Convert a bio state (torch.Tensor or np.ndarray) to NT concentrations.

        Args:
            bio_state: 12-dim vector (torch.Tensor, np.ndarray, or list).

        Returns:
            Dict mapping NT name to concentration in nM.
        """
        arr = _to_numpy(bio_state).flatten()
        return self._neuro_bridge.to_concentrations(arr)

    def drive_network(self, network, bio_state, blend: float = 0.5) -> None:
        """Drive a MolecularNeuralNetwork's global NT concentrations from bio state.

        Args:
            network: MolecularNeuralNetwork instance.
            bio_state: 12-dim vector (torch.Tensor, np.ndarray, or list).
            blend: Blending factor [0, 1]. 0 = keep existing, 1 = full override.
        """
        new_concs = self.torch_to_concentrations(bio_state)
        blend = max(0.0, min(1.0, blend))
        for nt_name, new_val in new_concs.items():
            old_val = network.global_nt_concentrations.get(nt_name, new_val)
            network.global_nt_concentrations[nt_name] = (
                (1.0 - blend) * old_val + blend * new_val
            )

    def read_network_state(self, network) -> np.ndarray:
        """Read a network's current global NT state as a 12-dim BioState vector.

        Args:
            network: MolecularNeuralNetwork instance.

        Returns:
            np.ndarray of shape (12,) — approximate BioState.
        """
        return self._neuro_bridge.from_concentrations(
            dict(network.global_nt_concentrations)
        )
