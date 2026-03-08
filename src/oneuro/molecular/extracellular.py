"""3D extracellular space with neurotransmitter diffusion and volume transmission.

Neurotransmitters released at synapses do not just act locally -- they diffuse
through the extracellular space (ECS), enabling volume transmission: a mode of
signalling where neuromodulators like dopamine and serotonin influence entire
brain regions rather than single synapses.

This module discretises a 3D tissue volume into a voxel grid.  Each voxel
tracks concentrations of 6 major NTs.  Fick's law governs diffusion (discrete
Laplacian via np.roll, no per-voxel Python loops), and Michaelis-Menten
transporter uptake clears NTs from the ECS.

Also includes PerineuronalNet, the extracellular matrix (ECM) that wraps
mature neurons and restricts plasticity -- a key mechanism in critical-period
closure and memory stabilisation.

No other oneuro imports required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Literature diffusion coefficients (um^2/ms)
# ---------------------------------------------------------------------------
# Sources:
#   Nicholson & Sykova 1998 (tortuosity-corrected free diffusion)
#   Bhatt et al. 2005 (DA), Bhatt et al. 2009 (5-HT)
#   Rusakov & Bhatt 2011 (glutamate, GABA)

DIFFUSION_COEFFICIENTS: Dict[str, float] = {
    "dopamine": 0.763,
    "serotonin": 0.54,
    "norepinephrine": 0.72,
    "acetylcholine": 0.40,
    "gaba": 0.76,
    "glutamate": 0.76,
}


# ---------------------------------------------------------------------------
# Transporter uptake parameters (Michaelis-Menten per voxel per ms)
# ---------------------------------------------------------------------------
# Sources:
#   Jones et al. 1998 (DAT), Blakely et al. 1991 (SERT)
#   Daws et al. 2005 (NET), Danbolt 2001 (EAAT)
#   Borden 1996 (GAT)

@dataclass(frozen=True)
class TransporterKinetics:
    """Michaelis-Menten parameters for a neurotransmitter transporter."""
    transporter_name: str
    nt_name: str
    Km_nM: float   # Michaelis constant (nM)
    Vmax_nM_per_ms: float  # Maximum uptake rate (nM/ms)


TRANSPORTER_TABLE: Dict[str, TransporterKinetics] = {
    "dopamine": TransporterKinetics("DAT", "dopamine", Km_nM=4700.0, Vmax_nM_per_ms=0.4),
    "serotonin": TransporterKinetics("SERT", "serotonin", Km_nM=170.0, Vmax_nM_per_ms=0.1),
    "norepinephrine": TransporterKinetics("NET", "norepinephrine", Km_nM=330.0, Vmax_nM_per_ms=0.2),
    "glutamate": TransporterKinetics("EAAT", "glutamate", Km_nM=20000.0, Vmax_nM_per_ms=2.0),
    "gaba": TransporterKinetics("GAT", "gaba", Km_nM=2500.0, Vmax_nM_per_ms=0.5),
}

# Default NT names matching the rest of the molecular layer
DEFAULT_NT_NAMES: List[str] = [
    "dopamine", "serotonin", "norepinephrine",
    "acetylcholine", "gaba", "glutamate",
]


# ---------------------------------------------------------------------------
# ExtracellularSpace
# ---------------------------------------------------------------------------

@dataclass
class ExtracellularSpace:
    """Discretised 3D extracellular space with NT diffusion and uptake.

    The tissue volume is divided into a regular grid of cubic voxels.  Each
    voxel stores NT concentrations in nM.  At every simulation step:

      1. **Diffusion** -- 3D discrete Fick's law via finite differences with
         Neumann (no-flux) boundary conditions.
      2. **Transporter uptake** -- Michaelis-Menten kinetics per voxel removes
         NT from the ECS (reuptake into presynaptic terminals and glia).

    Attributes:
        grid_size: Number of voxels along each axis (nx, ny, nz).
        voxel_size_um: Edge length of each voxel in micrometers.
        nt_names: Names of tracked neurotransmitters.
        nt_field: Concentration array, shape (nx, ny, nz, n_nts), in nM.
    """

    grid_size: Tuple[int, int, int] = (10, 10, 10)
    voxel_size_um: float = 10.0
    nt_names: List[str] = field(default_factory=lambda: list(DEFAULT_NT_NAMES))

    # Concentration field: shape (nx, ny, nz, n_nts), nM
    nt_field: np.ndarray = field(init=False, repr=False)

    # Fast NT name -> index lookup
    _nt_index: Dict[str, int] = field(init=False, repr=False)

    # Precomputed diffusion coefficient vector, shape (n_nts,)
    _diff_coeffs: np.ndarray = field(init=False, repr=False)

    # Precomputed transporter Km and Vmax vectors, shape (n_nts,)
    _km_vec: np.ndarray = field(init=False, repr=False)
    _vmax_vec: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nx, ny, nz = self.grid_size
        n_nts = len(self.nt_names)

        # Build name-to-index mapping
        self._nt_index = {name: i for i, name in enumerate(self.nt_names)}

        # Initialise concentration field to zero
        self.nt_field = np.zeros((nx, ny, nz, n_nts), dtype=np.float64)

        # Vectorise diffusion coefficients
        self._diff_coeffs = np.array(
            [DIFFUSION_COEFFICIENTS.get(name, 0.5) for name in self.nt_names],
            dtype=np.float64,
        )

        # Vectorise transporter kinetics (NTs without a known transporter
        # get Vmax=0 so uptake is effectively disabled for them)
        self._km_vec = np.ones(n_nts, dtype=np.float64)
        self._vmax_vec = np.zeros(n_nts, dtype=np.float64)
        for i, name in enumerate(self.nt_names):
            tk = TRANSPORTER_TABLE.get(name)
            if tk is not None:
                self._km_vec[i] = tk.Km_nM
                self._vmax_vec[i] = tk.Vmax_nM_per_ms

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def position_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert continuous position (um) to nearest voxel indices.

        Coordinates are clamped to the grid boundaries.

        Args:
            x, y, z: Position in micrometers.

        Returns:
            (ix, iy, iz) integer voxel indices.
        """
        nx, ny, nz = self.grid_size
        ix = max(0, min(nx - 1, int(x / self.voxel_size_um)))
        iy = max(0, min(ny - 1, int(y / self.voxel_size_um)))
        iz = max(0, min(nz - 1, int(z / self.voxel_size_um)))
        return ix, iy, iz

    # ------------------------------------------------------------------
    # Release and read
    # ------------------------------------------------------------------

    def release_at(
        self, x: float, y: float, z: float, nt_name: str, amount_nM: float
    ) -> None:
        """Release neurotransmitter at a spatial position.

        The amount (in nM) is added to the nearest voxel.

        Args:
            x, y, z: Release site in micrometers.
            nt_name: Neurotransmitter name (must be in self.nt_names).
            amount_nM: Concentration increment in nM.

        Raises:
            KeyError: If nt_name is not tracked.
        """
        idx = self._nt_index[nt_name]
        ix, iy, iz = self.position_to_voxel(x, y, z)
        self.nt_field[ix, iy, iz, idx] += amount_nM

    def concentration_at(self, x: float, y: float, z: float, nt_name: str) -> float:
        """Read local NT concentration, trilinearly interpolated from the grid.

        For positions exactly at voxel centres this returns the voxel value.
        For positions between voxels, trilinear interpolation provides a smooth
        continuous concentration field.

        Args:
            x, y, z: Query position in micrometers.
            nt_name: Neurotransmitter name.

        Returns:
            Interpolated concentration in nM.

        Raises:
            KeyError: If nt_name is not tracked.
        """
        idx = self._nt_index[nt_name]
        nx, ny, nz = self.grid_size

        # Continuous voxel coordinates (centre of voxel i is at (i + 0.5) * voxel_size)
        fx = x / self.voxel_size_um - 0.5
        fy = y / self.voxel_size_um - 0.5
        fz = z / self.voxel_size_um - 0.5

        # Floor indices (clamped)
        x0 = max(0, min(nx - 2, int(math.floor(fx))))
        y0 = max(0, min(ny - 2, int(math.floor(fy))))
        z0 = max(0, min(nz - 2, int(math.floor(fz))))
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # Fractional distances [0, 1]
        xd = max(0.0, min(1.0, fx - x0))
        yd = max(0.0, min(1.0, fy - y0))
        zd = max(0.0, min(1.0, fz - z0))

        # Trilinear interpolation
        c = self.nt_field
        c000 = c[x0, y0, z0, idx]
        c100 = c[x1, y0, z0, idx]
        c010 = c[x0, y1, z0, idx]
        c110 = c[x1, y1, z0, idx]
        c001 = c[x0, y0, z1, idx]
        c101 = c[x1, y0, z1, idx]
        c011 = c[x0, y1, z1, idx]
        c111 = c[x1, y1, z1, idx]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return float(c0 * (1 - zd) + c1 * zd)

    # ------------------------------------------------------------------
    # Diffusion (Fick's law, finite differences)
    # ------------------------------------------------------------------

    def diffusion_step(self, dt: float) -> None:
        """Advance diffusion by dt milliseconds using discrete Fick's law.

        Uses the 7-point stencil Laplacian on the 3D grid with Neumann
        (no-flux) boundary conditions implemented via np.roll with edge
        duplication.

        The update is:
            dc/dt = D * laplacian(c)
            c_new = c + D * dt / dx^2 * (sum_neighbours - 6*c)

        Stability requires dt <= dx^2 / (6*D_max).  This method does NOT
        subdivide internally -- the caller is responsible for choosing a
        stable dt, or using step() which handles both diffusion and uptake.

        Args:
            dt: Timestep in milliseconds.
        """
        c = self.nt_field
        dx2 = self.voxel_size_um ** 2

        # Compute Laplacian via shifted copies.
        # For Neumann BCs, pad by replicating the boundary slice before rolling.
        # np.roll wraps around, so we overwrite the wrapped boundary with the
        # adjacent interior value (equivalent to zero-gradient BC).

        # Shifted +1 and -1 along each axis
        # axis 0 (x)
        cp_x = np.roll(c, -1, axis=0)
        cp_x[-1, :, :, :] = c[-1, :, :, :]  # Neumann: no flux at +x boundary
        cm_x = np.roll(c, 1, axis=0)
        cm_x[0, :, :, :] = c[0, :, :, :]    # Neumann: no flux at -x boundary

        # axis 1 (y)
        cp_y = np.roll(c, -1, axis=1)
        cp_y[:, -1, :, :] = c[:, -1, :, :]
        cm_y = np.roll(c, 1, axis=1)
        cm_y[:, 0, :, :] = c[:, 0, :, :]

        # axis 2 (z)
        cp_z = np.roll(c, -1, axis=2)
        cp_z[:, :, -1, :] = c[:, :, -1, :]
        cm_z = np.roll(c, 1, axis=2)
        cm_z[:, :, 0, :] = c[:, :, 0, :]

        # 7-point stencil Laplacian: sum of 6 neighbours minus 6 * centre
        laplacian = cp_x + cm_x + cp_y + cm_y + cp_z + cm_z - 6.0 * c

        # D is per-NT: broadcast (n_nts,) over last axis
        # shape of _diff_coeffs is (n_nts,), broadcasts to (nx, ny, nz, n_nts)
        self.nt_field = c + (self._diff_coeffs * dt / dx2) * laplacian

        # Concentrations cannot go negative (numerical artefact at steep gradients)
        np.clip(self.nt_field, 0.0, None, out=self.nt_field)

    # ------------------------------------------------------------------
    # Transporter uptake (Michaelis-Menten)
    # ------------------------------------------------------------------

    def transporter_uptake(self, dt: float) -> None:
        """Remove NT from each voxel via Michaelis-Menten transporter kinetics.

        v = Vmax * [C] / (Km + [C])
        dC = -v * dt

        NTs without a known transporter (e.g. acetylcholine, which is degraded
        enzymatically by AChE rather than reuptake) have Vmax=0 and are
        unaffected.

        Args:
            dt: Timestep in milliseconds.
        """
        c = self.nt_field

        # Michaelis-Menten rate: shape broadcasts (n_nts,) over (nx,ny,nz,n_nts)
        rate = self._vmax_vec * c / (self._km_vec + c)

        self.nt_field = c - rate * dt
        np.clip(self.nt_field, 0.0, None, out=self.nt_field)

    # ------------------------------------------------------------------
    # Combined step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the extracellular space by dt milliseconds.

        Applies diffusion followed by transporter uptake.

        Args:
            dt: Timestep in milliseconds.
        """
        self.diffusion_step(dt)
        self.transporter_uptake(dt)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def global_average(self, nt_name: str) -> float:
        """Return the spatial mean concentration (nM) for a neurotransmitter.

        Useful for backward-compatible interfaces that treat the whole tissue
        volume as a single compartment.

        Args:
            nt_name: Neurotransmitter name.

        Returns:
            Mean concentration across all voxels in nM.

        Raises:
            KeyError: If nt_name is not tracked.
        """
        idx = self._nt_index[nt_name]
        return float(np.mean(self.nt_field[:, :, :, idx]))

    def total_mass(self, nt_name: str) -> float:
        """Return the total amount of NT across all voxels (sum of nM values).

        This is proportional to total moles in the volume (each voxel has the
        same volume, so sum of concentrations * voxel_volume = total).

        Args:
            nt_name: Neurotransmitter name.

        Returns:
            Sum of concentrations across all voxels.
        """
        idx = self._nt_index[nt_name]
        return float(np.sum(self.nt_field[:, :, :, idx]))

    def max_concentration(self, nt_name: str) -> float:
        """Return the peak voxel concentration (nM) for a neurotransmitter.

        Useful for detecting hotspots / release sites.

        Args:
            nt_name: Neurotransmitter name.

        Returns:
            Maximum concentration across all voxels in nM.
        """
        idx = self._nt_index[nt_name]
        return float(np.max(self.nt_field[:, :, :, idx]))

    @property
    def physical_size_um(self) -> Tuple[float, float, float]:
        """Total physical dimensions of the tissue volume in micrometers."""
        nx, ny, nz = self.grid_size
        s = self.voxel_size_um
        return (nx * s, ny * s, nz * s)

    @property
    def n_voxels(self) -> int:
        """Total number of voxels in the grid."""
        nx, ny, nz = self.grid_size
        return nx * ny * nz

    def stability_limit(self) -> float:
        """Return maximum stable dt (ms) for explicit Euler diffusion.

        For a 3D grid with spacing dx:
            dt_max = dx^2 / (6 * D_max)

        Using a dt larger than this will cause numerical instability.
        """
        d_max = float(np.max(self._diff_coeffs))
        if d_max <= 0:
            return float("inf")
        return self.voxel_size_um ** 2 / (6.0 * d_max)


# ---------------------------------------------------------------------------
# Perineuronal Net (ECM)
# ---------------------------------------------------------------------------

@dataclass
class PerineuronalNet:
    """Perineuronal net -- extracellular matrix ensheathing mature neurons.

    PNNs are chondroitin-sulphate proteoglycan (CSPG) lattices that
    preferentially wrap fast-spiking parvalbumin+ interneurons.  They are
    assembled during postnatal critical periods and, once in place, restrict
    synaptic plasticity -- stabilising the existing connectivity pattern.

    Enzymatic degradation (e.g. chondroitinase ABC) or experience-dependent
    remodelling can reopen plasticity windows.

    Attributes:
        neuron_ids: Set of neuron IDs currently wrapped in PNN.
        plasticity_restriction: Fraction [0, 1] by which PNN reduces
            plasticity.  0 = no restriction, 1 = full block.
    """

    neuron_ids: Set[int] = field(default_factory=set)
    plasticity_restriction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.plasticity_restriction <= 1.0:
            raise ValueError(
                f"plasticity_restriction must be in [0, 1], "
                f"got {self.plasticity_restriction}"
            )

    def add_neuron(self, neuron_id: int) -> None:
        """Wrap a neuron in the perineuronal net.

        Args:
            neuron_id: Integer ID of the neuron to ensheathe.
        """
        self.neuron_ids.add(neuron_id)

    def remove_neuron(self, neuron_id: int) -> None:
        """Remove PNN from a neuron (e.g. chondroitinase ABC digestion).

        Args:
            neuron_id: Integer ID of the neuron to unwrap.
        """
        self.neuron_ids.discard(neuron_id)

    def get_plasticity_factor(self, neuron_id: int) -> float:
        """Return the plasticity scaling factor for a neuron.

        Neurons NOT in the PNN have full plasticity (factor = 1.0).
        Neurons IN the PNN have reduced plasticity:
            factor = 1.0 - plasticity_restriction

        This factor should be multiplied into STDP weight updates.

        Args:
            neuron_id: Integer ID of the neuron.

        Returns:
            Plasticity scaling factor in [0, 1].
        """
        if neuron_id in self.neuron_ids:
            return 1.0 - self.plasticity_restriction
        return 1.0

    def is_wrapped(self, neuron_id: int) -> bool:
        """Check whether a neuron is ensheathed in PNN.

        Args:
            neuron_id: Integer ID of the neuron.

        Returns:
            True if the neuron is wrapped.
        """
        return neuron_id in self.neuron_ids

    @property
    def count(self) -> int:
        """Number of neurons currently wrapped in PNN."""
        return len(self.neuron_ids)
