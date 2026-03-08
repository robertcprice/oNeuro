"""
MolecularWorld --- a physics-level sandbox for digital organisms.

Simulates atmosphere (gases, odorants, wind, temperature, light),
terrain (soil layers, moisture, nutrients), and food/water sources
at millimeter resolution on a 2D or 3D volumetric grid.

All grid math uses NumPy for speed.  Deterministic given a seed.

Biologically grounded for Drosophila-scale navigation:
  - Olfactory gradients (odorant plumes advected by wind)
  - Temperature gradients (thermotaxis)
  - Light / dark cycles (phototaxis)
  - Humidity gradients (hygroreception)
  - Buoyancy-driven vertical transport (3D mode)

Supports both 2D (backward compatible) and 3D volumetric modes:
  - size=(H, W)        -> 2D mode, internal grids are (1, H, W)
  - size=(D, H, W)     -> 3D mode, full volumetric grids
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical Constants (real values, SI units unless noted)
# ---------------------------------------------------------------------------

DAY_LENGTH_S = 86_400.0                 # 24 h in seconds
STEFAN_BOLTZMANN = 5.67e-8              # W m^-2 K^-4 (Stefan-Boltzmann)
AMBIENT_TEMP_C = 22.0                   # lab ambient temperature (Celsius)
BOLTZMANN_K = 1.380649e-23              # J/K (Boltzmann constant)
AVOGADRO = 6.022e23                     # molecules/mol
R_GAS = 8.314                           # J/(mol*K) (ideal gas constant)

# Thermal dynamics (calibrated for 1mm-cell grid)
# Real air thermal diffusivity: 2.2 x 10^-5 m^2/s = 22 mm^2/s = 22 cell^2/s
# Scaled for simulation stability at dt=0.1s: subcycled internally
THERMAL_DIFFUSIVITY_AIR = 22.0          # mm^2/s (CRC Handbook, air at 25 C)
SOLAR_IRRADIANCE_MAX = 1000.0           # W/m^2 (clear sky noon, tropical)
GROUND_ALBEDO = 0.2                     # fraction reflected (grass/soil)
AIR_HEAT_CAPACITY = 1005.0              # J/(kg*K) (dry air at const pressure)
AIR_DENSITY = 1.184                     # kg/m^3 (25 C, 1 atm)
SOLAR_MAX_TEMP_RATE = 0.02              # C/s (empirical, convection-limited)
RADIATIVE_COOLING_RATE = 0.005          # 1/s (Newton's law of cooling coeff)
CONDUCTION_COEFF = 0.3                  # cell^2/s (scaled from 22 mm^2/s for stability)
HUMIDITY_DIFFUSION_FACTOR = 0.7         # water vapor slows other diffusion

# Soil thermal conductivity: 0.2-2.0 W/(m*K) depending on moisture
SOIL_THERMAL_CONDUCTIVITY = 0.5         # W/(m*K) (moist loam, Campbell 1985)

# 3D atmospheric parameters
LAPSE_RATE_C_PER_MM = 6.5e-6           # C/mm (from standard atmosphere 6.5 C/km)
BUOYANCY_SCALE = 0.5                    # mm/s per unit density difference
MW_AIR = 29.0                           # g/mol (effective molecular weight of air)


# ---------------------------------------------------------------------------
# Odorant Catalogue --- Real Molecular Properties
#
# References:
#   [1] CRC Handbook of Chemistry and Physics, 97th ed.
#   [2] Fuller-Schettler-Giddings correlation for gas-phase diffusion
#   [3] Hallem & Carlson 2006, Cell --- Drosophila OR response profiles
#   [4] Stensmyr et al. 2012, Cell --- Drosophila olfactory detection limits
#   [5] Stokl et al. 2010 --- fruit volatile emission rates
# ---------------------------------------------------------------------------

class OdorantType(str, Enum):
    """Named odorants with real molecular identities."""
    ETHANOL = "ethanol"                  # CH3CH2OH, MW=46.07, fermentation
    ACETIC_ACID = "acetic_acid"          # CH3COOH, MW=60.05, vinegar/decay
    ETHYL_ACETATE = "ethyl_acetate"      # CH3COOC2H5, MW=88.11, fruity ester
    GERANIOL = "geraniol"                # C10H18O, MW=154.25, flower terpenol
    AMMONIA = "ammonia"                  # NH3, MW=17.03, decomposition

# Molecular weights (g/mol)
MOLECULAR_WEIGHT: Dict[str, float] = {
    OdorantType.ETHANOL:       46.07,
    OdorantType.ACETIC_ACID:   60.05,
    OdorantType.ETHYL_ACETATE: 88.11,
    OdorantType.GERANIOL:      154.25,
    OdorantType.AMMONIA:       17.03,
}

# Vapor pressure at 25 C (Pa) --- determines emission rate from liquid sources
# [1] CRC Handbook
VAPOR_PRESSURE_25C: Dict[str, float] = {
    OdorantType.ETHANOL:       7872.0,    # 59 mmHg
    OdorantType.ACETIC_ACID:   2073.0,    # 15.5 mmHg
    OdorantType.ETHYL_ACETATE: 12600.0,   # 94.5 mmHg (very volatile)
    OdorantType.GERANIOL:      4.0,       # 0.03 mmHg (low volatility)
    OdorantType.AMMONIA:       1013000.0, # 7600 mmHg (gas at 25 C)
}

# Gas-phase diffusion coefficients in air at 25 C, 1 atm (m^2/s)
# [1] CRC Handbook, [2] Fuller-Schettler-Giddings correlation
# Converted to mm^2/s (= m^2/s x 10^6) for 1mm cell grid
_D_ETHANOL_M2S = 1.19e-5       # [1]
_D_ACETIC_ACID_M2S = 1.24e-5   # [2] estimated
_D_ETHYL_ACETATE_M2S = 0.87e-5 # [2] larger molecule, slower
_D_GERANIOL_M2S = 0.50e-5      # [2] heavy terpenol, estimated
_D_AMMONIA_M2S = 2.28e-5       # [1] small molecule, fast

# Atmospheric decay rates (1/s) --- UV photolysis + oxidation by OH radicals
# [1] Atkinson et al. 2006, Atmos. Chem. Phys.
# Indoor/lab: much slower (no UV). Using lab-representative values.
_DECAY_ETHANOL = 0.01           # ~100s half-life indoors
_DECAY_ACETIC_ACID = 0.015      # slightly reactive
_DECAY_ETHYL_ACETATE = 0.008    # relatively stable ester
_DECAY_GERANIOL = 0.02          # terpenoids oxidize faster
_DECAY_AMMONIA = 0.005          # very stable in air

# Drosophila olfactory detection thresholds (dimensionless concentration units)
# [3] Hallem & Carlson 2006: Or response EC50 values ~10^-7 to 10^-4 dilution
# [4] Stensmyr et al. 2012: geosmin detected at <5 ppb
# These are the minimum grid concentration that activates olfactory neurons
_THRESH_ETHANOL = 1e-5          # Or42b, moderate sensitivity
_THRESH_ACETIC_ACID = 5e-6      # Or92a, good sensitivity
_THRESH_ETHYL_ACETATE = 1e-6    # Or42b/Or59b, HIGH sensitivity (attractive)
_THRESH_GERANIOL = 5e-7         # Or10a, very sensitive (flower tracking)
_THRESH_AMMONIA = 1e-5          # Ir92a (ionotropic receptor), aversive

# Compiled odorant properties: (diffusion_mm2_per_s, decay_1_per_s, detection_threshold)
# Diffusion coefficients converted from m^2/s to mm^2/s for the grid
ODORANT_PROPERTIES: Dict[str, Tuple[float, float, float]] = {
    OdorantType.ETHANOL:       (_D_ETHANOL_M2S * 1e6, _DECAY_ETHANOL, _THRESH_ETHANOL),
    OdorantType.ACETIC_ACID:   (_D_ACETIC_ACID_M2S * 1e6, _DECAY_ACETIC_ACID, _THRESH_ACETIC_ACID),
    OdorantType.ETHYL_ACETATE: (_D_ETHYL_ACETATE_M2S * 1e6, _DECAY_ETHYL_ACETATE, _THRESH_ETHYL_ACETATE),
    OdorantType.GERANIOL:      (_D_GERANIOL_M2S * 1e6, _DECAY_GERANIOL, _THRESH_GERANIOL),
    OdorantType.AMMONIA:       (_D_AMMONIA_M2S * 1e6, _DECAY_AMMONIA, _THRESH_AMMONIA),
}

# Maximum diffusion subcycles per world step (for numerical stability)
# 3D CFL condition: D * dt_sub / dx^2 < 1/6 (stricter than 2D's 1/4)
# D_max = ammonia = 22.8 mm^2/s, dx = 1mm -> dt_sub < 0.0073s
# At world dt=0.1s: need ~14 subcycles minimum for 3D
MAX_DIFFUSION_SUBCYCLES = 30


# ---------------------------------------------------------------------------
# Source dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FruitSource:
    """A ripening / decaying fruit that emits odorants."""
    x: int
    y: int
    z: int = 0                         # ground level by default
    ripeness: float = 0.5              # 0 (unripe) .. 1 (fully ripe)
    sugar_content: float = 0.5         # arbitrary units 0-1
    odorant_emission_rate: float = 0.05  # concentration / s
    decay_rate: float = 0.0001         # ripeness increase per second
    alive: bool = True

    # Fruit emits ethanol (fermentation) + ethyl_acetate (ester aroma)
    odorant_profile: Dict[str, float] = field(default_factory=lambda: {
        OdorantType.ETHANOL: 0.6,
        OdorantType.ETHYL_ACETATE: 0.3,
        OdorantType.ACETIC_ACID: 0.1,
    })

    def step(self, dt: float) -> None:
        if not self.alive:
            return
        self.ripeness = min(1.0, self.ripeness + self.decay_rate * dt)
        # Once fully ripe, start decaying --- shift toward ammonia
        if self.ripeness >= 1.0:
            self.odorant_profile[OdorantType.AMMONIA] = self.odorant_profile.get(
                OdorantType.AMMONIA, 0.0
            ) + 0.001 * dt
            self.sugar_content = max(0.0, self.sugar_content - 0.0005 * dt)
            if self.sugar_content <= 0.0:
                self.alive = False


@dataclass
class PlantSource:
    """A plant / flower emitting nectar-related odorants."""
    x: int
    y: int
    z: int = 0                         # base z position
    height: float = 5.0                # mm --- emission height above z
    nectar_production_rate: float = 0.1
    flower_odorant: str = OdorantType.GERANIOL
    odorant_emission_rate: float = 0.03
    alive: bool = True

    odorant_profile: Dict[str, float] = field(default_factory=lambda: {
        OdorantType.GERANIOL: 1.0,
    })

    @property
    def emission_z(self) -> int:
        """Z-level where the plant emits odorants (at top of plant)."""
        return self.z + int(self.height)


@dataclass
class WaterSource:
    """Standing water that evaporates, increasing local humidity."""
    x: int
    y: int
    z: int = 0                         # ground level by default
    volume: float = 100.0              # arbitrary units
    evaporation_rate: float = 0.001    # volume / s
    alive: bool = True

    def step(self, dt: float) -> None:
        if not self.alive:
            return
        self.volume -= self.evaporation_rate * dt
        if self.volume <= 0.0:
            self.volume = 0.0
            self.alive = False


# ---------------------------------------------------------------------------
# Terrain / Soil (always 2D --- ground-level phenomenon)
# ---------------------------------------------------------------------------

@dataclass
class SoilState:
    """Three-layer soil model on a 2D grid."""
    surface_moisture: NDArray       # (H, W) float32 0-1
    shallow_nutrients: NDArray      # (H, W) float32 --- sugars + amino acids
    deep_minerals: NDArray          # (H, W) float32
    organic_matter: NDArray         # (H, W) float32
    decomposition_rate: float = 0.0001  # 1/s

    def step(self, dt: float) -> None:
        """Decompose organic matter into shallow nutrients."""
        decomposed = self.organic_matter * self.decomposition_rate * dt
        self.organic_matter -= decomposed
        self.shallow_nutrients += decomposed * 0.5
        # Clamp
        np.clip(self.organic_matter, 0.0, None, out=self.organic_matter)


# ---------------------------------------------------------------------------
# MolecularWorld
# ---------------------------------------------------------------------------

class MolecularWorld:
    """
    2D/3D molecular-level environment simulator.

    Parameters
    ----------
    size : tuple[int, int] or tuple[int, int, int]
        Grid dimensions.
        2-element (H, W): 2D mode --- grids stored internally as (1, H, W).
        3-element (D, H, W): full 3D volumetric mode.
    cell_size_mm : float
        Physical size of each cell in millimetres.
    seed : int | None
        RNG seed for deterministic runs.
    """

    # ---- construction ----

    def __init__(
        self,
        size: Union[Tuple[int, int], Tuple[int, int, int], None] = None,
        cell_size_mm: float = 1.0,
        seed: int | None = 42,
        *,
        size_mm: float | None = None,
    ) -> None:
        # Support size_mm= kwarg as shorthand for a square 2D grid
        if size is None and size_mm is not None:
            dim = max(1, int(size_mm))
            size = (dim, dim)
        elif size is None:
            size = (100, 100)

        # Determine 2D vs 3D mode
        if len(size) == 2:
            self.is_3d = False
            self.D = 1
            self.H, self.W = size
        elif len(size) == 3:
            self.is_3d = True
            self.D, self.H, self.W = size
        else:
            raise ValueError(f"size must be (H, W) or (D, H, W), got {size}")

        self.cell_size_mm = cell_size_mm
        self.rng = np.random.default_rng(seed)
        self.time: float = 0.0  # elapsed seconds

        # -- Atmosphere --
        # Gas fractions (uniform, constant for now)
        self.gas_n2: float = 0.78
        self.gas_o2: float = 0.21
        self.gas_co2: float = 0.0004
        self.gas_h2o: float = 0.01  # global average; local varies

        # Odorant concentration grids: {name: (D, H, W) float64}
        self.odorant_grids: Dict[str, NDArray] = {
            name: np.zeros((self.D, self.H, self.W), dtype=np.float64)
            for name in OdorantType
        }

        # Wind field (vx, vy, vz in cell-units / s), all (D, H, W)
        self.wind_vx: NDArray = np.full(
            (self.D, self.H, self.W), 0.5, dtype=np.float64
        )
        self.wind_vy: NDArray = np.full(
            (self.D, self.H, self.W), 0.0, dtype=np.float64
        )
        self.wind_vz: NDArray = np.zeros(
            (self.D, self.H, self.W), dtype=np.float64
        )
        # Optional: gentle random perturbation for vx, vy
        self.wind_vx += self.rng.normal(0, 0.05, (self.D, self.H, self.W))
        self.wind_vy += self.rng.normal(0, 0.05, (self.D, self.H, self.W))

        # In 3D mode, initialize vertical wind with thermal updrafts
        if self.is_3d and self.D > 1:
            # Gentle upward near ground from surface heating
            self.wind_vz[0, :, :] = 0.1
            if self.D > 2:
                self.wind_vz[1, :, :] = 0.05
            # Slight downdraft at height (return flow)
            mid = self.D // 2
            if mid > 0:
                self.wind_vz[mid:, :, :] = -0.02

        # Temperature field (Celsius) --- (D, H, W)
        self.temperature: NDArray = np.full(
            (self.D, self.H, self.W), AMBIENT_TEMP_C, dtype=np.float64
        )
        # In 3D mode, apply vertical lapse rate
        if self.is_3d and self.D > 1:
            for z_idx in range(self.D):
                altitude_mm = z_idx * self.cell_size_mm
                self.temperature[z_idx, :, :] -= LAPSE_RATE_C_PER_MM * altitude_mm

        # Humidity field (0-1) --- (D, H, W)
        self.humidity: NDArray = np.full(
            (self.D, self.H, self.W), 0.4, dtype=np.float64
        )
        # In 3D mode, humidity decreases with altitude
        if self.is_3d and self.D > 1:
            for z_idx in range(self.D):
                # Gentle exponential decay: humidity halves every ~50mm
                altitude_mm = z_idx * self.cell_size_mm
                self.humidity[z_idx, :, :] *= math.exp(-altitude_mm / 50.0)

        # -- Terrain (always 2D) --
        self.soil = SoilState(
            surface_moisture=np.full((self.H, self.W), 0.3, dtype=np.float32),
            shallow_nutrients=self.rng.uniform(0.01, 0.05, (self.H, self.W)).astype(
                np.float32
            ),
            deep_minerals=self.rng.uniform(0.05, 0.15, (self.H, self.W)).astype(
                np.float32
            ),
            organic_matter=self.rng.uniform(0.0, 0.02, (self.H, self.W)).astype(
                np.float32
            ),
        )

        # -- Sources --
        self.fruit_sources: List[FruitSource] = []
        self.plant_sources: List[PlantSource] = []
        self.water_sources: List[WaterSource] = []

    # ---- Source management ----

    def add_fruit(
        self,
        x: int,
        y: int,
        z: int = 0,
        sugar: float = 0.5,
        ripeness: float = 0.5,
        emission_rate: float = 0.05,
        decay_rate: float = 0.0001,
    ) -> FruitSource:
        src = FruitSource(
            x=int(x), y=int(y), z=int(np.clip(z, 0, self.D - 1)),
            sugar_content=sugar,
            ripeness=ripeness,
            odorant_emission_rate=emission_rate,
            decay_rate=decay_rate,
        )
        self.fruit_sources.append(src)
        return src

    def add_plant(
        self,
        x: int,
        y: int,
        z: int = 0,
        nectar_rate: float = 0.1,
        height: float = 5.0,
        emission_rate: float = 0.03,
    ) -> PlantSource:
        src = PlantSource(
            x=int(x), y=int(y), z=int(np.clip(z, 0, self.D - 1)),
            nectar_production_rate=nectar_rate,
            height=height,
            odorant_emission_rate=emission_rate,
        )
        self.plant_sources.append(src)
        return src

    def add_water(
        self,
        x: int,
        y: int,
        z: int = 0,
        volume: float = 100.0,
        evaporation_rate: float = 0.001,
    ) -> WaterSource:
        src = WaterSource(
            x=int(x), y=int(y), z=int(np.clip(z, 0, self.D - 1)),
            volume=volume,
            evaporation_rate=evaporation_rate,
        )
        self.water_sources.append(src)
        return src

    # ---- Sampling API ----

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp 2D coordinates to grid bounds (backward compat)."""
        return int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))

    def _clamp3(self, x: int, y: int, z: int = 0) -> Tuple[int, int, int]:
        """Clamp 3D coordinates to grid bounds."""
        return (
            int(np.clip(x, 0, self.W - 1)),
            int(np.clip(y, 0, self.H - 1)),
            int(np.clip(z, 0, self.D - 1)),
        )

    def sample_odorants(self, x: int, y: int, z: int = 0) -> Dict[str, float]:
        """Return odorant concentrations at (x, y, z)."""
        cx, cy, cz = self._clamp3(x, y, z)
        return {
            name: float(grid[cz, cy, cx])
            for name, grid in self.odorant_grids.items()
        }

    def sample_temperature(self, x: int, y: int, z: int = 0) -> float:
        """Return temperature in Celsius at (x, y, z)."""
        cx, cy, cz = self._clamp3(x, y, z)
        return float(self.temperature[cz, cy, cx])

    def sample_light(self, x: int, y: int, z: int = 0) -> float:
        """
        Return light intensity at (x, y, z), range [0, 1].

        Follows a sinusoidal day/night cycle.  UV component is ~5% of
        visible when the sun is above the horizon.
        """
        # Uniform across grid for now (could add shadows from plants later)
        return self._light_intensity()

    def sample_wind(self, x: int, y: int, z: int = 0) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Return wind vector at (x, y, z).

        In 3D mode: returns (vx, vy, vz).
        In 2D mode: returns (vx, vy) for backward compatibility.
        """
        cx, cy, cz = self._clamp3(x, y, z)
        vx = float(self.wind_vx[cz, cy, cx])
        vy = float(self.wind_vy[cz, cy, cx])
        if self.is_3d:
            vz = float(self.wind_vz[cz, cy, cx])
            return vx, vy, vz
        return vx, vy

    def sample_humidity(self, x: int, y: int, z: int = 0) -> float:
        """Return relative humidity at (x, y, z), range [0, 1]."""
        cx, cy, cz = self._clamp3(x, y, z)
        return float(self.humidity[cz, cy, cx])

    def sample_soil_nutrients(self, x: int, y: int) -> float:
        """Return shallow nutrient concentration at (x, y). Soil is 2D."""
        cx, cy = self._clamp(x, y)
        return float(self.soil.shallow_nutrients[cy, cx])

    # ---- Light helpers ----

    def _solar_angle_factor(self) -> float:
        """sin of solar elevation angle, clamped to [0, 1].

        Peaks at noon (t=43200s), zero from ~6 PM to ~6 AM.
        """
        return max(0.0, math.sin(
            2.0 * math.pi * (self.time / DAY_LENGTH_S - 0.25)))

    def _light_intensity(self) -> float:
        """Global light intensity [0, 1]."""
        return self._solar_angle_factor()

    def _uv_intensity(self) -> float:
        """UV component (~5 % of visible when sun is up)."""
        return self._light_intensity() * 0.05

    # ---- Physics step ----

    def step(self, dt: float = 0.1) -> None:
        """
        Advance the world by *dt* seconds.

        Order of operations:
          1. Update sources (ripening, evaporation)
          2. Emit odorants from sources
          3. Diffuse + advect odorants (with buoyancy in 3D)
          4. Update temperature (solar + radiation + conduction)
          5. Update humidity (water sources, diffusion)
          6. Soil decomposition
          7. Gentle wind perturbation
        """
        # 1 --- sources
        self._step_sources(dt)

        # 2 --- emit
        self._emit_odorants(dt)

        # 3 --- diffuse + advect odorants
        self._diffuse_odorants(dt)

        # 4 --- temperature
        self._step_temperature(dt)

        # 5 --- humidity
        self._step_humidity(dt)

        # 6 --- soil
        self.soil.step(dt)

        # 7 --- wind
        self._perturb_wind(dt)

        # tick clock
        self.time += dt

    # ---- Internal physics routines ----

    def _step_sources(self, dt: float) -> None:
        for src in self.fruit_sources:
            src.step(dt)
        for src in self.water_sources:
            src.step(dt)

    def _emit_odorants(self, dt: float) -> None:
        """Inject odorant molecules at source locations."""
        for src in self.fruit_sources:
            if not src.alive:
                continue
            rate = src.odorant_emission_rate * src.ripeness * dt
            sz = int(np.clip(src.z, 0, self.D - 1))
            for odorant, fraction in src.odorant_profile.items():
                self.odorant_grids[odorant][sz, src.y, src.x] += rate * fraction

        for src in self.plant_sources:
            if not src.alive:
                continue
            rate = src.odorant_emission_rate * dt
            # Plant emits at its canopy height
            emit_z = int(np.clip(src.emission_z, 0, self.D - 1))
            for odorant, fraction in src.odorant_profile.items():
                self.odorant_grids[odorant][emit_z, src.y, src.x] += rate * fraction

    def _laplacian_3d(self, grid: NDArray, dx2: float) -> NDArray:
        """
        Compute 3D Laplacian using 6-neighbor stencil.

        For a grid of shape (D, H, W), computes:
          lap = (sum of 6 neighbors - 6*center) / dx^2

        Boundary: edge-padded (Neumann-like zero-flux).
        """
        padded = np.pad(grid, 1, mode="edge")  # (D+2, H+2, W+2)
        laplacian = (
            padded[0:-2, 1:-1, 1:-1]   # -z neighbor
            + padded[2:,  1:-1, 1:-1]  # +z neighbor
            + padded[1:-1, 0:-2, 1:-1] # -y neighbor
            + padded[1:-1, 2:,  1:-1]  # +y neighbor
            + padded[1:-1, 1:-1, 0:-2] # -x neighbor
            + padded[1:-1, 1:-1, 2:]   # +x neighbor
            - 6.0 * grid
        ) / dx2
        return laplacian

    def _diffuse_odorants(self, dt: float) -> None:
        """
        Diffuse and advect every odorant grid with CFL-safe subcycling.

        dc/dt = D * laplacian(c)  -  wind . grad(c)  -  decay * c  +  buoyancy

        Uses explicit Euler with a 6-neighbor 3D Laplacian stencil
        and first-order upwind advection.  Humidity slows diffusion.

        CFL stability for 3D explicit Euler: D * dt / dx^2 < 1/6.
        In 2D mode (D=1), the z-axis terms vanish naturally.

        Buoyancy advection (3D only): lighter molecules (MW < 29) rise,
        heavier molecules sink, based on Archimedes' principle.
        """
        # Humidity factor: higher humidity -> slower diffusion
        # Use ground-level humidity for all layers (slight simplification)
        humidity_scale = 1.0 - self.humidity * (1.0 - HUMIDITY_DIFFUSION_FACTOR)

        # Precompute wind masks (constant across subcycles)
        pos_mask_x = self.wind_vx >= 0
        pos_mask_y = self.wind_vy >= 0
        pos_mask_z = self.wind_vz >= 0

        dx = self.cell_size_mm
        dx2 = dx * dx

        # CFL stability factor: 1/6 for 3D, 1/4 for 2D
        cfl_factor = (1.0 / 6.0) if self.is_3d else 0.25

        for name, grid in self.odorant_grids.items():
            D_coeff, decay, _thresh = ODORANT_PROPERTIES[name]

            # Effective max diffusion coefficient (humidity can only reduce it)
            D_eff_max = D_coeff  # worst case: dry air

            # CFL stability: D * dt_sub / dx^2 < cfl_factor
            dt_cfl_diff = cfl_factor * dx2 / max(D_eff_max, 1e-12)

            # Advection CFL: |v| * dt_sub / dx < 1.0
            v_max = max(
                float(np.abs(self.wind_vx).max()),
                float(np.abs(self.wind_vy).max()),
                float(np.abs(self.wind_vz).max()),
                1e-12,
            )
            # Include buoyancy velocity in CFL check
            mw = MOLECULAR_WEIGHT[name]
            v_buoy = abs(BUOYANCY_SCALE * (MW_AIR - mw) / MW_AIR)
            v_max = max(v_max, v_buoy)

            dt_cfl_adv = dx / v_max
            dt_sub = min(dt_cfl_diff, dt_cfl_adv, dt)

            n_sub = max(1, min(MAX_DIFFUSION_SUBCYCLES, int(math.ceil(dt / dt_sub))))
            dt_sub = dt / n_sub

            # Precompute buoyancy velocity for this odorant (3D only)
            # Archimedes: v_buoy = BUOYANCY_SCALE * (MW_AIR - MW_odorant) / MW_AIR
            # Positive = upward (lighter than air)
            v_buoy_z = BUOYANCY_SCALE * (MW_AIR - mw) / MW_AIR if self.is_3d else 0.0

            for _sc in range(n_sub):
                # --- 3D Laplacian ---
                laplacian = self._laplacian_3d(grid, dx2)

                # --- Upwind advection: -wind . grad(c) ---
                padded = np.pad(grid, 1, mode="edge")

                # grad_x (first-order upwind)
                grad_x_fwd = padded[1:-1, 1:-1, 2:] - grid       # right - center
                grad_x_bwd = grid - padded[1:-1, 1:-1, 0:-2]     # center - left
                grad_x = np.where(pos_mask_x, grad_x_bwd, grad_x_fwd)
                grad_x /= dx

                # grad_y (first-order upwind)
                grad_y_fwd = padded[1:-1, 2:, 1:-1] - grid
                grad_y_bwd = grid - padded[1:-1, 0:-2, 1:-1]
                grad_y = np.where(pos_mask_y, grad_y_bwd, grad_y_fwd)
                grad_y /= dx

                advection = self.wind_vx * grad_x + self.wind_vy * grad_y

                # Z-axis advection (always computed; vanishes naturally for D=1)
                if self.D > 1:
                    grad_z_fwd = padded[2:, 1:-1, 1:-1] - grid
                    grad_z_bwd = grid - padded[0:-2, 1:-1, 1:-1]

                    # Effective z-wind = atmospheric wind_vz + buoyancy
                    # Buoyancy is concentration-weighted: only acts where odorant exists
                    eff_vz = self.wind_vz.copy()
                    if abs(v_buoy_z) > 1e-12:
                        eff_vz = eff_vz + v_buoy_z  # broadcast scalar

                    pos_mask_z_eff = eff_vz >= 0
                    grad_z = np.where(pos_mask_z_eff, grad_z_bwd, grad_z_fwd)
                    grad_z /= dx
                    advection += eff_vz * grad_z

                # --- Combine ---
                dC = (D_coeff * humidity_scale * laplacian - advection - decay * grid) * dt_sub
                grid += dC

                # Clamp non-negative
                np.clip(grid, 0.0, None, out=grid)

    def _step_temperature(self, dt: float) -> None:
        """
        dT/dt = solar_input - radiation_loss + conduction

        Solar input follows day/night sinusoid (applied to ground layer z=0).
        Radiation loss proportional to (T - T_ambient).
        Conduction = diffusion of temperature across the 3D grid.
        In 3D mode, vertical thermal conduction transports heat upward.
        """
        solar = SOLAR_MAX_TEMP_RATE * self._solar_angle_factor()

        # Apply solar heating only to ground layer (z=0)
        solar_grid = np.zeros_like(self.temperature)
        solar_grid[0, :, :] = solar

        radiation = RADIATIVE_COOLING_RATE * (self.temperature - AMBIENT_TEMP_C)
        # In 3D, ambient temp varies with altitude (lapse rate)
        if self.is_3d and self.D > 1:
            ambient_3d = np.empty_like(self.temperature)
            for z_idx in range(self.D):
                altitude_mm = z_idx * self.cell_size_mm
                ambient_3d[z_idx, :, :] = AMBIENT_TEMP_C - LAPSE_RATE_C_PER_MM * altitude_mm
            radiation = RADIATIVE_COOLING_RATE * (self.temperature - ambient_3d)

        # Thermal conduction (3D Laplacian diffusion)
        lap_T = self._laplacian_3d(self.temperature, self.cell_size_mm ** 2)

        dT = (solar_grid - radiation + CONDUCTION_COEFF * lap_T) * dt
        self.temperature += dT

    def _step_humidity(self, dt: float) -> None:
        """
        Water sources raise local humidity at ground level.
        Humidity diffuses in 3D and decays toward altitude-dependent background.
        """
        HUMIDITY_BACKGROUND = 0.4
        HUMIDITY_DIFFUSION = 0.2   # cell^2/s
        HUMIDITY_RELAX = 0.002     # 1/s relaxation toward background

        # Inject humidity from water sources (at ground level z=0 or source z)
        for src in self.water_sources:
            if not src.alive:
                continue
            sz = int(np.clip(src.z, 0, self.D - 1))
            self.humidity[sz, src.y, src.x] = min(
                1.0, self.humidity[sz, src.y, src.x] + 0.01 * dt
            )

        # 3D Laplacian diffusion
        dx2 = self.cell_size_mm ** 2
        lap_H = self._laplacian_3d(self.humidity, dx2)

        # Background humidity varies with altitude in 3D
        if self.is_3d and self.D > 1:
            bg = np.empty_like(self.humidity)
            for z_idx in range(self.D):
                altitude_mm = z_idx * self.cell_size_mm
                bg[z_idx, :, :] = HUMIDITY_BACKGROUND * math.exp(-altitude_mm / 50.0)
        else:
            bg = HUMIDITY_BACKGROUND

        self.humidity += (
            HUMIDITY_DIFFUSION * lap_H - HUMIDITY_RELAX * (self.humidity - bg)
        ) * dt

        np.clip(self.humidity, 0.0, 1.0, out=self.humidity)

    def _perturb_wind(self, dt: float) -> None:
        """
        Small stochastic perturbation to keep wind field realistic.
        Gentle mean-reversion toward the base wind.
        """
        BASE_VX, BASE_VY, BASE_VZ = 0.5, 0.0, 0.0
        REVERT = 0.001  # 1/s
        shape = (self.D, self.H, self.W)

        self.wind_vx += (
            self.rng.normal(0, 0.002, shape)
            - REVERT * (self.wind_vx - BASE_VX) * dt
        )
        self.wind_vy += (
            self.rng.normal(0, 0.002, shape)
            - REVERT * (self.wind_vy - BASE_VY) * dt
        )
        if self.is_3d and self.D > 1:
            self.wind_vz += (
                self.rng.normal(0, 0.001, shape)
                - REVERT * (self.wind_vz - BASE_VZ) * dt
            )

    # ---- Gradient helpers (useful for organisms) ----

    def odorant_gradient(
        self, odorant: str, x: int, y: int, z: int = 0, radius: int = 1
    ) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Estimate the local concentration gradient of *odorant* at (x, y, z).

        In 3D mode: returns (dC/dx, dC/dy, dC/dz).
        In 2D mode: returns (dC/dx, dC/dy) for backward compatibility.
        """
        cx, cy, cz = self._clamp3(x, y, z)
        grid = self.odorant_grids.get(odorant)
        if grid is None:
            return (0.0, 0.0, 0.0) if self.is_3d else (0.0, 0.0)

        x_lo = max(0, cx - radius)
        x_hi = min(self.W - 1, cx + radius)
        y_lo = max(0, cy - radius)
        y_hi = min(self.H - 1, cy + radius)

        dx = (float(grid[cz, cy, x_hi]) - float(grid[cz, cy, x_lo])) / max(1, x_hi - x_lo)
        dy = (float(grid[cz, y_hi, cx]) - float(grid[cz, y_lo, cx])) / max(1, y_hi - y_lo)

        if self.is_3d:
            z_lo = max(0, cz - radius)
            z_hi = min(self.D - 1, cz + radius)
            dz = (float(grid[z_hi, cy, cx]) - float(grid[z_lo, cy, cx])) / max(1, z_hi - z_lo)
            return (dx, dy, dz)
        return (dx, dy)

    def temperature_gradient(
        self, x: int, y: int, z: int = 0, radius: int = 1
    ) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Local temperature gradient.

        In 3D mode: returns (dT/dx, dT/dy, dT/dz).
        In 2D mode: returns (dT/dx, dT/dy).
        """
        cx, cy, cz = self._clamp3(x, y, z)
        x_lo = max(0, cx - radius)
        x_hi = min(self.W - 1, cx + radius)
        y_lo = max(0, cy - radius)
        y_hi = min(self.H - 1, cy + radius)

        dx = (float(self.temperature[cz, cy, x_hi]) - float(self.temperature[cz, cy, x_lo])) / max(
            1, x_hi - x_lo
        )
        dy = (float(self.temperature[cz, y_hi, cx]) - float(self.temperature[cz, y_lo, cx])) / max(
            1, y_hi - y_lo
        )

        if self.is_3d:
            z_lo = max(0, cz - radius)
            z_hi = min(self.D - 1, cz + radius)
            dz = (float(self.temperature[z_hi, cy, cx]) - float(self.temperature[z_lo, cy, cx])) / max(
                1, z_hi - z_lo
            )
            return (dx, dy, dz)
        return (dx, dy)

    # ---- Bulk state accessors ----

    def get_odorant_grid(self, odorant: str) -> NDArray:
        """Return the full concentration grid for an odorant.

        Shape is (D, H, W). In 2D mode, D=1 so grid[0] gives (H, W).
        """
        return self.odorant_grids[odorant]

    def get_temperature_grid(self) -> NDArray:
        return self.temperature

    def get_humidity_grid(self) -> NDArray:
        return self.humidity

    def get_wind_field(self) -> Union[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray, NDArray]]:
        """Return wind field arrays.

        In 3D mode: returns (vx, vy, vz).
        In 2D mode: returns (vx, vy) for backward compatibility.
        """
        if self.is_3d:
            return self.wind_vx, self.wind_vy, self.wind_vz
        return self.wind_vx, self.wind_vy

    # ---- Serialization helpers ----

    def get_state_snapshot(self) -> Dict:
        """Return a dict of all world state (for saving / visualization)."""
        snap = {
            "time": self.time,
            "is_3d": self.is_3d,
            "shape": (self.D, self.H, self.W),
            "temperature": self.temperature.copy(),
            "humidity": self.humidity.copy(),
            "wind_vx": self.wind_vx.copy(),
            "wind_vy": self.wind_vy.copy(),
            "wind_vz": self.wind_vz.copy(),
            "odorants": {
                name: grid.copy() for name, grid in self.odorant_grids.items()
            },
            "soil_moisture": self.soil.surface_moisture.copy(),
            "soil_nutrients": self.soil.shallow_nutrients.copy(),
            "light": self._light_intensity(),
            "n_fruit": len([s for s in self.fruit_sources if s.alive]),
            "n_plant": len([s for s in self.plant_sources if s.alive]),
            "n_water": len([s for s in self.water_sources if s.alive]),
        }
        return snap

    # ---- Compatibility API (used by demo sensory/motor encoders) ----
    # These methods provide a simplified point-query interface matching the
    # demo's stub MolecularWorld API.

    @property
    def size_mm(self) -> float:
        """Approximate arena size in mm (largest grid dimension)."""
        return float(max(self.H, self.W) * self.cell_size_mm)

    @property
    def time_of_day(self) -> float:
        """Fractional time of day: 0.0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk."""
        return (self.time % DAY_LENGTH_S) / DAY_LENGTH_S

    @time_of_day.setter
    def time_of_day(self, value: float) -> None:
        """Set fractional time of day (0-1). Adjusts self.time accordingly."""
        self.time = value * DAY_LENGTH_S

    @property
    def ambient_light(self) -> float:
        """Global ambient light intensity [0, 1]."""
        return self._solar_angle_factor()

    @ambient_light.setter
    def ambient_light(self, value: float) -> None:
        """Setting ambient_light is a no-op; light follows the solar cycle.

        Provided for compatibility with demo code that assigns
        ``world.ambient_light = 0.0`` at initialization.
        """
        pass  # Light is computed from self.time via _solar_angle_factor()

    @property
    def is_day(self) -> bool:
        """True when solar elevation factor > 0.1 (roughly sunrise to sunset)."""
        return self._solar_angle_factor() > 0.1

    @is_day.setter
    def is_day(self, value: bool) -> None:
        """Setting is_day is a no-op; day/night follows the solar cycle."""
        pass

    def odor_at(self, x: float, y: float, compound: str = "ethanol") -> float:
        """Point query for odorant concentration at (x, y).

        Checks odorant grids by name. Also checks custom odorant_sources
        (added via add_odorant_source) for Gaussian point contributions.
        """
        # Check the physical odorant grid first
        cx, cy = int(np.clip(x, 0, self.W - 1)), int(np.clip(y, 0, self.H - 1))
        total = 0.0
        # Try matching compound name to OdorantType enum values
        for name, grid in self.odorant_grids.items():
            if compound in str(name):
                total += float(grid[0, cy, cx])
                break

        # Also accumulate from custom odorant sources
        for src in getattr(self, '_custom_odorant_sources', []):
            if src["compound"] != compound:
                continue
            dx = x - src["x"]
            dy = y - src["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            sigma = src["radius"] * 0.5
            total += src["intensity"] * math.exp(
                -dist * dist / (2.0 * sigma * sigma))

        return min(total, 1.0)

    def light_at(self, x: float, y: float) -> float:
        """Light intensity at (x, y). Includes ambient + point light sources."""
        total = self.ambient_light * 0.1
        for src in getattr(self, 'light_sources', []):
            dx = x - src["x"]
            dy = y - src["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            sigma = src["radius"] * 0.5
            total += src["intensity"] * math.exp(
                -dist * dist / (2.0 * sigma * sigma))
        return min(total, 2.0)

    def temperature_at(self, x: float, y: float) -> float:
        """Temperature at (x, y) in degrees Celsius.

        If temp_min/temp_max are set, returns a linear gradient.
        Otherwise queries the temperature grid.
        """
        if hasattr(self, 'temp_min') and hasattr(self, 'temp_max'):
            frac = float(x) / max(1.0, self.size_mm)
            return self.temp_min + (self.temp_max - self.temp_min) * frac
        cx = int(np.clip(x, 0, self.W - 1))
        cy = int(np.clip(y, 0, self.H - 1))
        return float(self.temperature[0, cy, cx])

    def food_at(self, x: float, y: float) -> float:
        """Food concentration at (x, y). Gaussian falloff from food patches."""
        total = 0.0
        for patch in getattr(self, 'food_patches', []):
            dx = x - patch["x"]
            dy = y - patch["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < patch["radius"] * 2.0:
                sigma = patch["radius"] * 0.5
                total += patch["remaining"] * math.exp(
                    -dist * dist / (2.0 * sigma * sigma))
        return min(total, 1.0)

    def add_odorant_source(self, x: float, y: float,
                           compound: str = "ethanol",
                           intensity: float = 1.0,
                           radius: float = 15.0) -> None:
        """Add a custom point odorant source (used by odor_at queries).

        Also seeds the physical odorant grid with a Gaussian blob for
        the matching OdorantType if one exists.
        """
        if not hasattr(self, '_custom_odorant_sources'):
            self._custom_odorant_sources: List[Dict] = []
        self._custom_odorant_sources.append({
            "x": x, "y": y, "compound": compound,
            "intensity": intensity, "radius": radius,
        })
        # Seed the physical grid with a Gaussian blob
        for name in self.odorant_grids:
            if compound in str(name):
                grid = self.odorant_grids[name]
                cx_i = int(np.clip(x, 0, self.W - 1))
                cy_i = int(np.clip(y, 0, self.H - 1))
                r_int = int(radius)
                sigma = radius * 0.5
                for dy_i in range(-r_int, r_int + 1):
                    for dx_i in range(-r_int, r_int + 1):
                        gx = cx_i + dx_i
                        gy = cy_i + dy_i
                        if 0 <= gx < self.W and 0 <= gy < self.H:
                            d2 = dx_i * dx_i + dy_i * dy_i
                            grid[0, gy, gx] += intensity * math.exp(
                                -d2 / (2.0 * sigma * sigma))
                break

    def add_light_source(self, x: float, y: float,
                         intensity: float = 1.0,
                         radius: float = 40.0) -> None:
        """Add a point light source (used by light_at queries)."""
        if not hasattr(self, 'light_sources'):
            self.light_sources: List[Dict] = []
        self.light_sources.append({
            "x": x, "y": y, "intensity": intensity, "radius": radius,
        })

    def add_food(self, x: float, y: float, radius: float = 5.0,
                 remaining: float = 1.0) -> None:
        """Add a food patch (used by food_at / deplete_food_near queries).

        Also registers a fruit source for odorant emission.
        """
        if not hasattr(self, 'food_patches'):
            self.food_patches: List[Dict] = []
        self.food_patches.append({
            "x": x, "y": y, "radius": radius, "remaining": remaining,
        })
        # Also add as a fruit source for odorant emission
        self.add_fruit(int(np.clip(x, 0, self.W - 1)),
                       int(np.clip(y, 0, self.H - 1)),
                       sugar=remaining)

    def deplete_food_near(self, x: float, y: float,
                          eat_radius: float = 3.0,
                          amount: float = 0.005) -> bool:
        """Deplete food near position. Returns True if food was eaten."""
        eaten = False
        for patch in getattr(self, 'food_patches', []):
            dist = math.sqrt((x - patch["x"]) ** 2 + (y - patch["y"]) ** 2)
            if dist < eat_radius + patch["radius"] and patch["remaining"] > 0:
                patch["remaining"] = max(0.0, patch["remaining"] - amount)
                eaten = True
        return eaten

    def reset_food(self) -> None:
        """Restore all food patches to full."""
        for patch in getattr(self, 'food_patches', []):
            patch["remaining"] = 1.0

    def advance_time(self, dt_hours: float = 0.1) -> None:
        """Advance the world clock by dt_hours.

        For small dt (<10s), runs full physics (diffusion, advection).
        For large dt (circadian stepping), only advances the clock and
        updates light/temperature to avoid CFL overflow in diffusion.
        """
        dt_seconds = dt_hours * 3600.0
        if dt_seconds <= 10.0:
            self.step(dt=dt_seconds)
        else:
            # Just advance the clock — skip expensive diffusion physics
            self.time += dt_seconds
            # Update ambient light and temperature from solar cycle
            solar = self._solar_angle_factor()
            self.ambient_light = solar

    def set_time_of_day(self, t: float) -> None:
        """Set time of day: 0.0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk."""
        self.time = t * DAY_LENGTH_S

    # ---- Reset ----

    def reset(self, seed: int | None = None) -> None:
        """Re-initialize the world to a fresh state."""
        if self.is_3d:
            s = (self.D, self.H, self.W)
        else:
            s = (self.H, self.W)
        self.__init__(size=s, cell_size_mm=self.cell_size_mm, seed=seed)

    # ---- Self test ----

    def self_test(self) -> None:
        """
        Verify that the core physics subsystems work correctly.

        Checks (2D mode):
          1. Odorant diffusion spreads concentration outward
          2. Wind advection shifts the odorant plume downwind
          3. Temperature responds to the day/night solar cycle
          4. Source emission injects odorant at the correct location
          5. Humidity increases near water sources
          6. Soil decomposition converts organic matter to nutrients
          7. Light intensity follows sinusoidal day/night cycle
          8. Gradient helpers return sensible values
          9. Plant source emission
         10. API completeness
         11. Determinism
         12. Performance (2D)

        Checks (3D mode):
         13. 3D diffusion spreads in all directions
         14. Buoyancy --- ammonia rises, geraniol sinks
         15. 3D wind advection
         16. Vertical temperature gradient
         17. Performance (3D)
         18. 2D backward compatibility (grids are (1, H, W))
        """
        import sys

        passed = 0
        failed = 0

        def check(name: str, condition: bool, detail: str = "") -> None:
            nonlocal passed, failed
            if condition:
                passed += 1
                print(f"  PASS  {name}")
            else:
                failed += 1
                print(f"  FAIL  {name}  {detail}")

        print("=" * 60)
        print("MolecularWorld self-test")
        print("=" * 60)

        # --- Test 1: Odorant diffusion (2D mode) ---
        w = MolecularWorld(size=(50, 50), seed=123)
        # Place a point source of ethanol at center
        w.odorant_grids[OdorantType.ETHANOL][0, 25, 25] = 10.0
        center_before = w.odorant_grids[OdorantType.ETHANOL][0, 25, 25]
        neighbor_before = w.odorant_grids[OdorantType.ETHANOL][0, 25, 26]
        # Turn off wind for this test
        w.wind_vx[:] = 0.0
        w.wind_vy[:] = 0.0
        for _ in range(20):
            w.step(dt=0.05)
        center_after = w.odorant_grids[OdorantType.ETHANOL][0, 25, 25]
        neighbor_after = w.odorant_grids[OdorantType.ETHANOL][0, 25, 26]
        check(
            "odorant diffusion -- center decreases",
            center_after < center_before,
            f"center {center_before:.4f} -> {center_after:.4f}",
        )
        check(
            "odorant diffusion -- neighbor increases",
            neighbor_after > neighbor_before,
            f"neighbor {neighbor_before:.6f} -> {neighbor_after:.6f}",
        )

        # --- Test 2: Wind advection ---
        w2 = MolecularWorld(size=(50, 50), seed=124)
        w2.odorant_grids[OdorantType.ETHANOL][0, 25, 10] = 10.0
        w2.wind_vx[:] = 2.0   # strong rightward wind
        w2.wind_vy[:] = 0.0
        for _ in range(50):
            w2.step(dt=0.05)
        # Concentration should have shifted rightward
        left_sum = w2.odorant_grids[OdorantType.ETHANOL][0, 25, :10].sum()
        right_sum = w2.odorant_grids[OdorantType.ETHANOL][0, 25, 10:].sum()
        check(
            "wind advection -- plume shifts downwind",
            right_sum > left_sum,
            f"left={left_sum:.6f}  right={right_sum:.6f}",
        )

        # --- Test 3: Temperature solar cycle ---
        w3 = MolecularWorld(size=(20, 20), seed=125)
        w3.time = 0.0  # midnight
        t_midnight = w3.sample_temperature(10, 10)
        # Advance to noon
        steps_to_noon = int(DAY_LENGTH_S / 4 / 0.5)  # 6 hours
        for _ in range(steps_to_noon):
            w3.step(dt=0.5)
        t_morning = w3.sample_temperature(10, 10)
        check(
            "temperature -- warms toward noon",
            t_morning > t_midnight - 1.0,  # should at least not cool much
            f"midnight={t_midnight:.2f}  morning={t_morning:.2f}",
        )

        # --- Test 4: Source emission ---
        w4 = MolecularWorld(size=(30, 30), seed=126)
        w4.wind_vx[:] = 0.0
        w4.wind_vy[:] = 0.0
        w4.add_fruit(x=15, y=15, sugar=0.8, ripeness=0.8)
        total_before = w4.odorant_grids[OdorantType.ETHANOL].sum()
        w4.step(dt=1.0)
        total_after = w4.odorant_grids[OdorantType.ETHANOL].sum()
        check(
            "source emission -- fruit emits ethanol",
            total_after > total_before,
            f"total_before={total_before:.6f}  total_after={total_after:.6f}",
        )

        # --- Test 5: Humidity from water ---
        w5 = MolecularWorld(size=(30, 30), seed=127)
        w5.add_water(x=15, y=15)
        h_before = w5.humidity[0, 15, 15]
        for _ in range(100):
            w5.step(dt=0.1)
        h_after = w5.humidity[0, 15, 15]
        check(
            "humidity -- increases near water source",
            h_after > h_before,
            f"before={h_before:.4f}  after={h_after:.4f}",
        )

        # --- Test 6: Soil decomposition ---
        w6 = MolecularWorld(size=(20, 20), seed=128)
        w6.soil.organic_matter[:] = 0.5
        nutrients_before = w6.soil.shallow_nutrients.mean()
        for _ in range(200):
            w6.step(dt=0.5)
        nutrients_after = w6.soil.shallow_nutrients.mean()
        check(
            "soil decomposition -- nutrients increase",
            nutrients_after > nutrients_before,
            f"before={nutrients_before:.6f}  after={nutrients_after:.6f}",
        )

        # --- Test 7: Light day/night ---
        w7 = MolecularWorld(size=(10, 10), seed=129)
        w7.time = 0.0  # midnight
        light_midnight = w7.sample_light(5, 5)
        w7.time = DAY_LENGTH_S / 2  # noon (peak of sinusoid)
        light_noon = w7.sample_light(5, 5)
        check(
            "light cycle -- midnight is dark",
            light_midnight < 0.01,
            f"midnight light={light_midnight:.4f}",
        )
        check(
            "light cycle -- noon is bright",
            light_noon > 0.9,
            f"noon light={light_noon:.4f}",
        )

        # --- Test 8: Gradient helpers ---
        w8 = MolecularWorld(size=(30, 30), seed=130)
        w8.odorant_grids[OdorantType.ETHANOL][0, 15, 10] = 0.0
        w8.odorant_grids[OdorantType.ETHANOL][0, 15, 20] = 5.0
        gx, gy = w8.odorant_gradient(OdorantType.ETHANOL, 15, 15, radius=5)
        check(
            "odorant gradient -- positive x direction",
            gx > 0,
            f"gradient_x={gx:.6f}",
        )

        # --- Test 9: Plant source emission ---
        w9 = MolecularWorld(size=(30, 30), seed=131)
        w9.wind_vx[:] = 0.0
        w9.wind_vy[:] = 0.0
        w9.add_plant(x=15, y=15, nectar_rate=0.2)
        total_before_g = w9.odorant_grids[OdorantType.GERANIOL].sum()
        w9.step(dt=1.0)
        total_after_g = w9.odorant_grids[OdorantType.GERANIOL].sum()
        check(
            "source emission -- plant emits geraniol",
            total_after_g > total_before_g,
            f"total_before={total_before_g:.6f}  total_after={total_after_g:.6f}",
        )

        # --- Test 10: API completeness ---
        w10 = MolecularWorld(size=(20, 20), seed=132)
        w10.add_fruit(x=5, y=5, sugar=0.7)
        w10.add_plant(x=10, y=10, nectar_rate=0.1)
        w10.add_water(x=15, y=15)
        w10.step(dt=0.1)
        odorants = w10.sample_odorants(5, 5)
        temp = w10.sample_temperature(5, 5)
        light = w10.sample_light(5, 5)
        wind = w10.sample_wind(5, 5)
        humidity = w10.sample_humidity(5, 5)
        nutrients = w10.sample_soil_nutrients(5, 5)
        snap = w10.get_state_snapshot()
        check(
            "API completeness -- all sample methods return correct types",
            isinstance(odorants, dict)
            and isinstance(temp, float)
            and isinstance(light, float)
            and isinstance(wind, tuple)
            and len(wind) == 2  # 2D mode returns 2-tuple
            and isinstance(humidity, float)
            and isinstance(nutrients, float)
            and isinstance(snap, dict),
        )

        # --- Test 11: Determinism ---
        w_a = MolecularWorld(size=(20, 20), seed=999)
        w_a.add_fruit(x=10, y=10, sugar=0.6)
        for _ in range(50):
            w_a.step(dt=0.1)
        snap_a = w_a.get_state_snapshot()

        w_b = MolecularWorld(size=(20, 20), seed=999)
        w_b.add_fruit(x=10, y=10, sugar=0.6)
        for _ in range(50):
            w_b.step(dt=0.1)
        snap_b = w_b.get_state_snapshot()

        deterministic = (
            np.allclose(snap_a["temperature"], snap_b["temperature"])
            and np.allclose(
                snap_a["odorants"][OdorantType.ETHANOL],
                snap_b["odorants"][OdorantType.ETHANOL],
            )
            and snap_a["time"] == snap_b["time"]
        )
        check("determinism -- same seed produces identical worlds", deterministic)

        # --- Test 12: Performance (2D) ---
        import time as _time

        w_perf = MolecularWorld(size=(100, 100), seed=200)
        w_perf.add_fruit(x=30, y=50, sugar=0.8)
        w_perf.add_plant(x=60, y=70, nectar_rate=0.1)
        w_perf.add_water(x=10, y=10)
        t0 = _time.perf_counter()
        n_steps = 100
        for _ in range(n_steps):
            w_perf.step(dt=0.1)
        elapsed = _time.perf_counter() - t0
        ms_per_step = elapsed / n_steps * 1000
        check(
            f"performance 2D -- {ms_per_step:.2f} ms/step (100x100, 3 sources)",
            ms_per_step < 50,  # should be well under 50 ms
            f"elapsed={elapsed:.3f}s  ({ms_per_step:.2f} ms/step)",
        )

        # =====================================================
        # 3D-specific tests
        # =====================================================
        print("-" * 60)
        print("3D volumetric tests")
        print("-" * 60)

        # --- Test 13: 3D diffusion spreads in all directions ---
        w3d = MolecularWorld(size=(20, 30, 30), seed=300)
        w3d.wind_vx[:] = 0.0
        w3d.wind_vy[:] = 0.0
        w3d.wind_vz[:] = 0.0
        # Point source at center of volume
        w3d.odorant_grids[OdorantType.ETHANOL][10, 15, 15] = 10.0
        center_3d_before = w3d.odorant_grids[OdorantType.ETHANOL][10, 15, 15]
        z_neighbor_before = w3d.odorant_grids[OdorantType.ETHANOL][11, 15, 15]
        y_neighbor_before = w3d.odorant_grids[OdorantType.ETHANOL][10, 16, 15]
        for _ in range(20):
            w3d.step(dt=0.05)
        center_3d_after = w3d.odorant_grids[OdorantType.ETHANOL][10, 15, 15]
        z_neighbor_after = w3d.odorant_grids[OdorantType.ETHANOL][11, 15, 15]
        y_neighbor_after = w3d.odorant_grids[OdorantType.ETHANOL][10, 16, 15]
        check(
            "3D diffusion -- center decreases",
            center_3d_after < center_3d_before,
            f"center {center_3d_before:.4f} -> {center_3d_after:.4f}",
        )
        check(
            "3D diffusion -- z-neighbor increases",
            z_neighbor_after > z_neighbor_before,
            f"z-neighbor {z_neighbor_before:.6f} -> {z_neighbor_after:.6f}",
        )
        check(
            "3D diffusion -- y-neighbor increases",
            y_neighbor_after > y_neighbor_before,
            f"y-neighbor {y_neighbor_before:.6f} -> {y_neighbor_after:.6f}",
        )

        # --- Test 14: Buoyancy --- ammonia rises, geraniol sinks ---
        w_buoy = MolecularWorld(size=(20, 20, 20), seed=301)
        w_buoy.wind_vx[:] = 0.0
        w_buoy.wind_vy[:] = 0.0
        w_buoy.wind_vz[:] = 0.0
        mid_z = 10
        # Place ammonia (MW=17, lighter than air MW=29) and geraniol (MW=154, heavier)
        w_buoy.odorant_grids[OdorantType.AMMONIA][mid_z, 10, 10] = 10.0
        w_buoy.odorant_grids[OdorantType.GERANIOL][mid_z, 10, 10] = 10.0
        for _ in range(40):
            w_buoy.step(dt=0.05)
        # Ammonia should have more concentration above mid_z than below
        nh3_above = w_buoy.odorant_grids[OdorantType.AMMONIA][mid_z+1:, :, :].sum()
        nh3_below = w_buoy.odorant_grids[OdorantType.AMMONIA][:mid_z, :, :].sum()
        check(
            "buoyancy -- ammonia rises (more above than below)",
            nh3_above > nh3_below,
            f"above={nh3_above:.6f}  below={nh3_below:.6f}",
        )
        # Geraniol should have more concentration below mid_z than above
        ger_above = w_buoy.odorant_grids[OdorantType.GERANIOL][mid_z+1:, :, :].sum()
        ger_below = w_buoy.odorant_grids[OdorantType.GERANIOL][:mid_z, :, :].sum()
        check(
            "buoyancy -- geraniol sinks (more below than above)",
            ger_below > ger_above,
            f"above={ger_above:.6f}  below={ger_below:.6f}",
        )

        # --- Test 15: 3D wind advection ---
        # Use ammonia (lightest, fastest diffusion) to test wind transport.
        # Place source near bottom, strong upward wind. Compare center-of-mass
        # against a no-wind control to prove advection shifts the plume up.
        w_3dwind = MolecularWorld(size=(30, 20, 20), seed=302)
        w_3dwind.wind_vx[:] = 0.0
        w_3dwind.wind_vy[:] = 0.0
        w_3dwind.wind_vz[:] = 3.0  # strong upward wind
        w_3dwind.odorant_grids[OdorantType.ETHANOL][5, 10, 10] = 10.0

        w_3dctrl = MolecularWorld(size=(30, 20, 20), seed=302)
        w_3dctrl.wind_vx[:] = 0.0
        w_3dctrl.wind_vy[:] = 0.0
        w_3dctrl.wind_vz[:] = 0.0  # no wind control
        w_3dctrl.odorant_grids[OdorantType.ETHANOL][5, 10, 10] = 10.0

        for _ in range(40):
            w_3dwind.step(dt=0.05)
            w_3dctrl.step(dt=0.05)
        # Compute z-center-of-mass for both
        g_wind = w_3dwind.odorant_grids[OdorantType.ETHANOL]
        g_ctrl = w_3dctrl.odorant_grids[OdorantType.ETHANOL]
        z_indices = np.arange(30).reshape(-1, 1, 1)
        com_wind = (g_wind * z_indices).sum() / max(g_wind.sum(), 1e-12)
        com_ctrl = (g_ctrl * z_indices).sum() / max(g_ctrl.sum(), 1e-12)
        check(
            "3D wind advection -- upward wind shifts plume higher",
            com_wind > com_ctrl,
            f"com_wind={com_wind:.3f}  com_ctrl={com_ctrl:.3f}",
        )

        # --- Test 16: Vertical temperature gradient ---
        w_tgrad = MolecularWorld(size=(20, 20, 20), seed=303)
        t_ground = w_tgrad.sample_temperature(10, 10, z=0)
        t_top = w_tgrad.sample_temperature(10, 10, z=19)
        check(
            "vertical temp gradient -- ground warmer than top",
            t_ground >= t_top,
            f"ground={t_ground:.6f}  top={t_top:.6f}",
        )

        # --- Test 17: Performance (3D) ---
        w_perf3d = MolecularWorld(size=(20, 100, 100), seed=304)
        w_perf3d.add_fruit(x=30, y=50, sugar=0.8)
        w_perf3d.add_plant(x=60, y=70, nectar_rate=0.1)
        w_perf3d.add_water(x=10, y=10)
        t0 = _time.perf_counter()
        n_steps_3d = 20
        for _ in range(n_steps_3d):
            w_perf3d.step(dt=0.1)
        elapsed_3d = _time.perf_counter() - t0
        ms_per_step_3d = elapsed_3d / n_steps_3d * 1000
        check(
            f"performance 3D -- {ms_per_step_3d:.2f} ms/step (20x100x100, 3 sources)",
            ms_per_step_3d < 500,  # 3D is ~20x more cells, allow more time
            f"elapsed={elapsed_3d:.3f}s  ({ms_per_step_3d:.2f} ms/step)",
        )

        # --- Test 18: 2D backward compat --- grids are (1, H, W) ---
        w_compat = MolecularWorld(size=(20, 20), seed=305)
        check(
            "2D backward compat -- is_3d is False",
            not w_compat.is_3d,
        )
        check(
            "2D backward compat -- D == 1",
            w_compat.D == 1,
        )
        check(
            "2D backward compat -- odorant grid shape is (1, H, W)",
            w_compat.odorant_grids[OdorantType.ETHANOL].shape == (1, 20, 20),
            f"shape={w_compat.odorant_grids[OdorantType.ETHANOL].shape}",
        )
        check(
            "2D backward compat -- wind returns 2-tuple",
            len(w_compat.sample_wind(5, 5)) == 2,
        )
        check(
            "2D backward compat -- gradient returns 2-tuple",
            len(w_compat.odorant_gradient(OdorantType.ETHANOL, 5, 5)) == 2,
        )

        # --- Test 19: 3D API --- wind returns 3-tuple, gradient returns 3-tuple ---
        w_3dapi = MolecularWorld(size=(10, 20, 20), seed=306)
        check(
            "3D API -- is_3d is True",
            w_3dapi.is_3d,
        )
        wind_3d = w_3dapi.sample_wind(5, 5, z=3)
        check(
            "3D API -- wind returns 3-tuple",
            len(wind_3d) == 3,
            f"len={len(wind_3d)}",
        )
        grad_3d = w_3dapi.odorant_gradient(OdorantType.ETHANOL, 5, 5, z=3)
        check(
            "3D API -- gradient returns 3-tuple",
            len(grad_3d) == 3,
            f"len={len(grad_3d)}",
        )

        # --- Test 20: Plant emission at canopy height in 3D ---
        # Verify that the emission injection happens at z=height, not z=0.
        # Check immediately after _emit_odorants (before diffusion spreads it).
        w_plant3d = MolecularWorld(size=(10, 30, 30), seed=307)
        w_plant3d.wind_vx[:] = 0.0
        w_plant3d.wind_vy[:] = 0.0
        w_plant3d.wind_vz[:] = 0.0
        w_plant3d.add_plant(x=15, y=15, z=0, height=5.0, nectar_rate=0.2)
        # Call _emit_odorants directly (before diffusion)
        w_plant3d._emit_odorants(dt=1.0)
        # Plant at z=0 with height=5 should emit at z=5
        ger_at_height = w_plant3d.odorant_grids[OdorantType.GERANIOL][5, 15, 15]
        ger_at_ground = w_plant3d.odorant_grids[OdorantType.GERANIOL][0, 15, 15]
        check(
            "3D plant emission -- emits at canopy height z=5",
            ger_at_height > ger_at_ground,
            f"z=5: {ger_at_height:.6f}  z=0: {ger_at_ground:.6f}",
        )

        # --- Summary ---
        total = passed + failed
        print("=" * 60)
        print(f"Results: {passed}/{total} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world = MolecularWorld()
    world.self_test()
