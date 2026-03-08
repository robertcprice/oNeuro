"""Biophysically faithful retinal processing: pixel images to spike trains.

Three-layer retina (photoreceptors, bipolar cells, RGCs) converting RGB images
into spike trains using HH-based membrane dynamics.

Biology:
    - Photoreceptors HYPERPOLARIZE in light (Baylor et al. 1979)
    - ON/OFF bipolar pathways via mGluR6 / iGluR (Werblin & Dowling 1969)
    - Center-surround antagonism for edge detection (Kuffler 1953)
    - Only RGCs produce action potentials (optic nerve output, Masland 2001)
    - Fovea: cones only, high density. Periphery: rods ~20:1 (Curcio et al. 1990)
    - Spectral peaks: S=420nm, M=530nm, L=560nm, Rod=498nm (Govardovskii 2000)

Usage:
    retina = MolecularRetina(resolution=(32, 32))
    spikes = retina.process_frame(rgb_array)  # (H,W,3) uint8 -> spike indices
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# -- Biophysical constants --------------------------------------------------
C_M = 1.0                  # membrane capacitance (uF/cm^2)
V_REST_PHOTORECEPTOR = -40.0  # depolarized in darkness (mV)
V_REST_BIPOLAR = -60.0     # bipolar resting potential (mV)
V_REST_RGC = -65.0         # RGC resting potential (mV)
V_HYPER = -70.0            # fully hyperpolarized photoreceptor (mV)
SPIKE_THRESHOLD = -20.0    # RGC spike threshold (mV)
REFRACTORY_MS = 2.0        # RGC refractory period (ms)
G_NA, G_K, G_LEAK = 120.0, 36.0, 0.3       # HH conductances (mS/cm^2)
E_NA, E_K, E_LEAK = 50.0, -77.0, -54.387   # reversal potentials (mV)
LAMBDA_S, LAMBDA_M, LAMBDA_L, LAMBDA_ROD = 420.0, 530.0, 560.0, 498.0
SPECTRAL_BW = 30.0         # Govardovskii Gaussian half-width (nm)
W_CENTER = 1.0             # center RF weight
W_SURROUND = -0.3          # surround RF weight (inhibitory)
W_BIP_TO_RGC = 5.0         # bipolar->RGC gain
TAU_ADAPT = 200.0          # Weber adaptation time constant (ms)
DT = 0.5                   # default integration step (ms)
BIPOLAR_THRESHOLD = 8.0    # ribbon synapse release threshold (mV above rest)

# -- Spectral sensitivity ---------------------------------------------------

def _govardovskii_sensitivity(wavelength_nm: float, peak_nm: float) -> float:
    """Gaussian approximation of Govardovskii nomogram (2000)."""
    d = (wavelength_nm - peak_nm) / SPECTRAL_BW
    return math.exp(-0.5 * d * d)

def _rgb_to_spectral_activation(r: float, g: float, b: float,
                                peak_nm: float) -> float:
    """Convert normalized RGB to photoreceptor activation for given peak."""
    # Monitor RGB ~ 600/540/450 nm dominant wavelengths
    s = [_govardovskii_sensitivity(wl, peak_nm) for wl in (600, 540, 450)]
    norm = sum(s)
    return min((r * s[0] + g * s[1] + b * s[2]) / max(norm, 1e-9), 1.0)

def _precompute_spectral_weights(peak_nm: float) -> np.ndarray:
    """Precomputed (3,) RGB->activation weight vector."""
    s = np.array([_govardovskii_sensitivity(wl, peak_nm)
                  for wl in (600, 540, 450)], dtype=np.float32)
    n = s.sum()
    return s / max(n, 1e-9)

_SPECTRAL_WEIGHTS = {
    "S_cone": _precompute_spectral_weights(LAMBDA_S),
    "M_cone": _precompute_spectral_weights(LAMBDA_M),
    "L_cone": _precompute_spectral_weights(LAMBDA_L),
    "rod": _precompute_spectral_weights(LAMBDA_ROD),
}

# -- HH gating (vectorized, clipped to prevent overflow) --------------------

def _alpha_m(V: np.ndarray) -> np.ndarray:
    Vc = np.clip(V, -150, 100); x = Vc + 40.0
    s = np.where(np.abs(x) < 1e-6, 1e-6, x)
    return 0.1 * s / (1.0 - np.exp(-s / 10.0))

def _beta_m(V: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(np.clip(V, -150, 100) + 65.0) / 18.0)

def _alpha_h(V: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(np.clip(V, -150, 100) + 65.0) / 20.0)

def _beta_h(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.clip(V, -150, 100) + 35.0) / 10.0))

def _alpha_n(V: np.ndarray) -> np.ndarray:
    Vc = np.clip(V, -150, 100); x = Vc + 55.0
    s = np.where(np.abs(x) < 1e-6, 1e-6, x)
    return 0.01 * s / (1.0 - np.exp(-s / 10.0))

def _beta_n(V: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(np.clip(V, -150, 100) + 65.0) / 80.0)

# -- Enums -------------------------------------------------------------------

class PhotoreceptorType(Enum):
    S_CONE = "S_cone"   # 420 nm
    M_CONE = "M_cone"   # 530 nm
    L_CONE = "L_cone"   # 560 nm
    ROD = "rod"         # 498 nm

class BipolarPolarity(Enum):
    ON = "ON"     # sign-inverting (mGluR6): depolarize in light
    OFF = "OFF"   # sign-preserving (iGluR): depolarize in dark

class RGCType(Enum):
    ON_CENTER = "ON_center"
    OFF_CENTER = "OFF_center"

# -- Retinal neuron dataclasses ----------------------------------------------

@dataclass
class RetinalNeuron:
    """Base retinal neuron with position in visual field [0,1]x[0,1]."""
    neuron_id: int
    x: float
    y: float
    voltage: float = -65.0

@dataclass
class Photoreceptor(RetinalNeuron):
    """Rod or cone. Hyperpolarizes in light (Baylor 1979). Graded potentials only."""
    cell_type: PhotoreceptorType = PhotoreceptorType.L_CONE
    spectral_peak: float = LAMBDA_L
    spectral_weights: np.ndarray = field(
        default_factory=lambda: _SPECTRAL_WEIGHTS["L_cone"])

    def __post_init__(self) -> None:
        self.voltage = V_REST_PHOTORECEPTOR
        cfg = {PhotoreceptorType.S_CONE: (LAMBDA_S, "S_cone"),
               PhotoreceptorType.M_CONE: (LAMBDA_M, "M_cone"),
               PhotoreceptorType.L_CONE: (LAMBDA_L, "L_cone"),
               PhotoreceptorType.ROD: (LAMBDA_ROD, "rod")}
        self.spectral_peak, key = cfg[self.cell_type]
        self.spectral_weights = _SPECTRAL_WEIGHTS[key]

@dataclass
class BipolarCell(RetinalNeuron):
    """ON or OFF bipolar cell with center-surround RF. Graded potentials only."""
    polarity: BipolarPolarity = BipolarPolarity.ON
    center_inputs: List[int] = field(default_factory=list)
    surround_inputs: List[int] = field(default_factory=list)
    center_weights: List[float] = field(default_factory=list)
    surround_weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.voltage = V_REST_BIPOLAR

@dataclass
class RetinalGanglionCell(RetinalNeuron):
    """Spiking output neuron — axons form the optic nerve (Masland 2001)."""
    rgc_type: RGCType = RGCType.ON_CENTER
    bipolar_inputs: List[int] = field(default_factory=list)
    input_weights: List[float] = field(default_factory=list)
    m: float = 0.05; h: float = 0.6; n: float = 0.32  # HH gating
    fired: bool = False; refractory_timer: float = 0.0; spike_count: int = 0

    def __post_init__(self) -> None:
        self.voltage = V_REST_RGC

# -- Mosaic generation -------------------------------------------------------

def _assign_cone_type(rng: np.random.Generator) -> PhotoreceptorType:
    """Random cone type with L:M:S ~ 10:5:1 (Roorda & Williams 1999)."""
    r = rng.random()
    if r < 10.0 / 16.0: return PhotoreceptorType.L_CONE
    if r < 15.0 / 16.0: return PhotoreceptorType.M_CONE
    return PhotoreceptorType.S_CONE

def _build_photoreceptor_mosaic(
    resolution: Tuple[int, int], fovea_ratio: float, rng: np.random.Generator,
) -> List[Photoreceptor]:
    """Create mosaic: fovea = cones only, periphery = rods ~20:1 (Curcio 1990)."""
    w, h = resolution
    photoreceptors: List[Photoreceptor] = []
    pid = 0
    for iy in range(h):
        for ix in range(w):
            xn, yn = (ix + 0.5) / w, (iy + 0.5) / h
            dist = math.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
            if dist < fovea_ratio:
                photoreceptors.append(Photoreceptor(
                    neuron_id=pid, x=xn, y=yn, cell_type=_assign_cone_type(rng)))
                pid += 1
            elif rng.random() < max(0.15, 1.0 - dist):
                ct = _assign_cone_type(rng) if rng.random() < 1/21 else PhotoreceptorType.ROD
                photoreceptors.append(Photoreceptor(neuron_id=pid, x=xn, y=yn, cell_type=ct))
                pid += 1
    return photoreceptors

def _find_neighbors(x: float, y: float, neurons: Sequence[RetinalNeuron],
                    radius: float) -> List[int]:
    r2 = radius * radius
    return [i for i, n in enumerate(neurons)
            if (n.x - x)**2 + (n.y - y)**2 <= r2]

# -- MolecularRetina ---------------------------------------------------------

class MolecularRetina:
    """Complete biophysical retina: pixel images -> spike trains.

    Three-layer circuit with photoreceptors (graded hyperpolarization),
    bipolar cells (ON/OFF pathways with center-surround), and RGCs
    (HH spiking neurons forming the optic nerve output).

    Args:
        resolution: (width, height) of visual field in pixels.
        fovea_ratio: Fovea radius as fraction of field size.
        device: "cpu", "cuda", or "mps" for torch acceleration.
        seed: Random seed for reproducible mosaic.
        center_radius: RF center radius in normalized coords.
        surround_radius: RF surround radius in normalized coords.
    """

    def __init__(self, resolution: Tuple[int, int] = (32, 32),
                 fovea_ratio: float = 0.3, device: str = "cpu",
                 seed: int = 42, center_radius: float = 0.06,
                 surround_radius: float = 0.15) -> None:
        self.resolution = resolution
        self.fovea_ratio = fovea_ratio
        self.device = device if HAS_TORCH else "cpu"
        self._use_torch = HAS_TORCH and device != "cpu"
        self.center_radius = center_radius
        self.surround_radius = surround_radius
        rng = np.random.default_rng(seed)

        # Layer 1: Photoreceptor mosaic
        self.photoreceptors = _build_photoreceptor_mosaic(resolution, fovea_ratio, rng)
        self.n_photo = len(self.photoreceptors)
        self._spectral_matrix = np.stack(
            [p.spectral_weights for p in self.photoreceptors], axis=0)
        self._adaptation = np.full(self.n_photo, 0.5, dtype=np.float32)
        self._photo_voltage = np.full(self.n_photo, V_REST_PHOTORECEPTOR, dtype=np.float32)

        # Layer 2: Bipolar cells
        self.bipolar_cells = self._wire_bipolar_cells(rng)
        self.n_bipolar = len(self.bipolar_cells)
        self._bipolar_voltage = np.full(self.n_bipolar, V_REST_BIPOLAR, dtype=np.float32)

        # Layer 3: Retinal ganglion cells
        self.rgc_cells = self._wire_rgc_cells(rng)
        self.n_rgc = len(self.rgc_cells)
        self._rgc_voltage = np.full(self.n_rgc, V_REST_RGC, dtype=np.float32)
        self._rgc_m = np.full(self.n_rgc, 0.05, dtype=np.float32)
        self._rgc_h = np.full(self.n_rgc, 0.6, dtype=np.float32)
        self._rgc_n = np.full(self.n_rgc, 0.32, dtype=np.float32)
        self._rgc_refractory = np.zeros(self.n_rgc, dtype=np.float32)

        if self._use_torch:
            self._spectral_matrix_t = torch.tensor(
                self._spectral_matrix, device=self.device, dtype=torch.float32)

        self._total_frames = 0
        self._total_spikes = 0

    # -- Wiring --------------------------------------------------------------

    def _wire_bipolar_cells(self, rng: np.random.Generator) -> List[BipolarCell]:
        """Create ON+OFF bipolar cells on a grid with center-surround RF."""
        cells: List[BipolarCell] = []
        bid = 0
        spacing = max(self.center_radius * 2.0, 0.04)
        nx, ny = max(1, int(1.0 / spacing)), max(1, int(1.0 / spacing))
        for iy in range(ny):
            for ix in range(nx):
                bx = np.clip((ix + 0.5) / nx + rng.uniform(-0.01, 0.01), 0, 1)
                by = np.clip((iy + 0.5) / ny + rng.uniform(-0.01, 0.01), 0, 1)
                center_ids = _find_neighbors(bx, by, self.photoreceptors, self.center_radius)
                if not center_ids:
                    continue
                surround_ids = [s for s in _find_neighbors(
                    bx, by, self.photoreceptors, self.surround_radius) if s not in center_ids]
                cw = [W_CENTER / len(center_ids)] * len(center_ids)
                sw = [W_SURROUND / len(surround_ids)] * len(surround_ids) if surround_ids else []
                for pol in (BipolarPolarity.ON, BipolarPolarity.OFF):
                    cells.append(BipolarCell(
                        neuron_id=bid, x=bx, y=by, polarity=pol,
                        center_inputs=list(center_ids), surround_inputs=list(surround_ids),
                        center_weights=list(cw), surround_weights=list(sw)))
                    bid += 1
        return cells

    def _wire_rgc_cells(self, rng: np.random.Generator) -> List[RetinalGanglionCell]:
        """Create ON-center and OFF-center RGCs connected to matching bipolars."""
        cells: List[RetinalGanglionCell] = []
        rid = 0
        on_bip = [(i, b) for i, b in enumerate(self.bipolar_cells)
                  if b.polarity == BipolarPolarity.ON]
        off_bip = [(i, b) for i, b in enumerate(self.bipolar_cells)
                   if b.polarity == BipolarPolarity.OFF]
        spacing = max(self.center_radius * 3.0, 0.06)
        nx, ny = max(1, int(1.0 / spacing)), max(1, int(1.0 / spacing))
        r2 = (self.center_radius * 2) ** 2
        for iy in range(ny):
            for ix in range(nx):
                rx = np.clip((ix + 0.5) / nx + rng.uniform(-0.01, 0.01), 0, 1)
                ry = np.clip((iy + 0.5) / ny + rng.uniform(-0.01, 0.01), 0, 1)
                for bip_list, rgc_type in ((on_bip, RGCType.ON_CENTER),
                                           (off_bip, RGCType.OFF_CENTER)):
                    inputs = [bi for bi, b in bip_list
                              if (b.x - rx)**2 + (b.y - ry)**2 <= r2]
                    if inputs:
                        w = [W_BIP_TO_RGC / len(inputs)] * len(inputs)
                        cells.append(RetinalGanglionCell(
                            neuron_id=rid, x=rx, y=ry, rgc_type=rgc_type,
                            bipolar_inputs=inputs, input_weights=w))
                        rid += 1
        return cells

    # -- Frame processing ----------------------------------------------------

    def process_frame(self, frame: np.ndarray, n_steps: int = 10,
                      dt: float = DT) -> List[int]:
        """Process one RGB frame (H,W,3 uint8) -> list of fired RGC neuron_ids."""
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            raise ValueError(
                f"Frame shape ({w},{h}) != retina resolution {self.resolution}")
        frame_f = frame.astype(np.float32) / 255.0
        activation = self._compute_activation(frame_f)
        fired_rgcs: set = set()
        for _ in range(n_steps):
            self._update_photoreceptors(activation, dt)
            self._update_bipolar_cells(dt)
            fired_rgcs.update(self._update_rgc_cells(dt))
        self._total_frames += 1
        self._total_spikes += len(fired_rgcs)
        return sorted(fired_rgcs)

    def process_sequence(self, frames: Sequence[np.ndarray],
                         dt_ms: float = 33.0, n_steps_per_frame: int = 10,
                         ) -> List[List[int]]:
        """Process video frames -> list of spike lists per frame."""
        step_dt = dt_ms / n_steps_per_frame
        return [self.process_frame(f, n_steps=n_steps_per_frame, dt=step_dt)
                for f in frames]

    # -- Layer updates -------------------------------------------------------

    def _compute_activation(self, frame_f: np.ndarray) -> np.ndarray:
        """Spectral activation for each photoreceptor from normalized RGB frame."""
        h, w = frame_f.shape[:2]
        pixels = np.zeros((self.n_photo, 3), dtype=np.float32)
        for i, p in enumerate(self.photoreceptors):
            pixels[i] = frame_f[min(int(p.y * h), h - 1), min(int(p.x * w), w - 1)]
        return np.clip(np.sum(pixels * self._spectral_matrix, axis=1), 0, 1)

    def _update_photoreceptors(self, activation: np.ndarray, dt: float) -> None:
        """Graded hyperpolarization: dark=-40mV, bright=-70mV (Baylor 1979)."""
        alpha = dt / 3.0  # tau_photo ~3 ms (fast phototransduction)
        weber = activation / (activation + self._adaptation + 1e-9)
        v_target = V_REST_PHOTORECEPTOR + (V_HYPER - V_REST_PHOTORECEPTOR) * weber
        self._photo_voltage += alpha * (v_target - self._photo_voltage)
        np.clip(self._photo_voltage, -80, -30, out=self._photo_voltage)
        adapt_alpha = 1.0 - np.exp(-dt / TAU_ADAPT)
        self._adaptation += adapt_alpha * (activation - self._adaptation)

    def _update_bipolar_cells(self, dt: float) -> None:
        """ON/OFF pathways via glutamate-gated center-surround computation."""
        alpha = dt / 10.0  # tau_bipolar ~10 ms
        # Normalized light level: 0=dark, 1=bright
        photo_light = np.clip(
            (V_REST_PHOTORECEPTOR - self._photo_voltage) / 30.0, 0, 1)
        photo_glut = 1.0 - photo_light  # glutamate: 1=dark (max), 0=bright (min)

        for i, bc in enumerate(self.bipolar_cells):
            cg = sum(w * photo_glut[pi] for pi, w in
                     zip(bc.center_inputs, bc.center_weights))
            sg = sum(w * photo_glut[pi] for pi, w in
                     zip(bc.surround_inputs, bc.surround_weights))
            net = max(0.0, min(1.0, cg + sg))
            # ON: depolarize when glut LOW (light). OFF: depolarize when glut HIGH (dark).
            drive = ((1.0 - net) if bc.polarity == BipolarPolarity.ON else net) * 20.0
            self._bipolar_voltage[i] += alpha * (V_REST_BIPOLAR + drive - self._bipolar_voltage[i])
        np.clip(self._bipolar_voltage, -80, -20, out=self._bipolar_voltage)

    def _update_rgc_cells(self, dt: float) -> List[int]:
        """Full HH dynamics for RGC spiking (the only spiking layer)."""
        V, m, h, n = self._rgc_voltage, self._rgc_m, self._rgc_h, self._rgc_n
        refr = self._rgc_refractory

        # Synaptic current from bipolar cells (ribbon synapse threshold)
        I_syn = np.zeros(self.n_rgc, dtype=np.float32)
        for i, rgc in enumerate(self.rgc_cells):
            for bi, w in zip(rgc.bipolar_inputs, rgc.input_weights):
                depol = self._bipolar_voltage[bi] - V_REST_BIPOLAR - BIPOLAR_THRESHOLD
                if depol > 0:
                    I_syn[i] += w * depol

        # HH gating variable updates (Euler)
        am, bm = _alpha_m(V), _beta_m(V)
        ah, bh = _alpha_h(V), _beta_h(V)
        an, bn = _alpha_n(V), _beta_n(V)
        m += dt * (am * (1 - m) - bm * m)
        h += dt * (ah * (1 - h) - bh * h)
        n += dt * (an * (1 - n) - bn * n)
        np.clip(m, 0, 1, out=m); np.clip(h, 0, 1, out=h); np.clip(n, 0, 1, out=n)

        # Ionic currents and voltage update
        I_ion = (G_NA * m**3 * h * (V - E_NA)
                 + G_K * n**4 * (V - E_K) + G_LEAK * (V - E_LEAK))
        V += dt * (-I_ion + I_syn) / C_M

        # Refractory timer
        refr -= dt
        np.clip(refr, 0, None, out=refr)

        # Spike detection
        fired_mask = (V >= SPIKE_THRESHOLD) & (refr <= 0)
        fired_ids = []
        for idx in np.where(fired_mask)[0]:
            fired_ids.append(self.rgc_cells[idx].neuron_id)
            self.rgc_cells[idx].spike_count += 1

        # Reset fired neurons
        V[fired_mask] = V_REST_RGC
        refr[fired_mask] = REFRACTORY_MS
        m[fired_mask] = 0.05; h[fired_mask] = 0.6; n[fired_mask] = 0.32
        return fired_ids

    # -- Analysis helpers ----------------------------------------------------

    def get_receptive_field_map(self, rgc_index: int) -> np.ndarray:
        """Trace RGC->bipolar->photoreceptor to get 2D spatial RF map."""
        w, h = self.resolution
        rf = np.zeros((h, w), dtype=np.float32)
        rgc = self.rgc_cells[rgc_index]
        for bi, bw in zip(rgc.bipolar_inputs, rgc.input_weights):
            bc = self.bipolar_cells[bi]
            sign = -1.0 if bc.polarity == BipolarPolarity.ON else 1.0
            for pi, cw in zip(bc.center_inputs, bc.center_weights):
                p = self.photoreceptors[pi]
                rf[min(int(p.y * h), h-1), min(int(p.x * w), w-1)] += bw * sign * cw
            for pi, sw in zip(bc.surround_inputs, bc.surround_weights):
                p = self.photoreceptors[pi]
                rf[min(int(p.y * h), h-1), min(int(p.x * w), w-1)] += bw * sign * sw
        return rf

    def get_spike_rates(self) -> Dict[str, float]:
        """Mean spike counts for ON-center and OFF-center RGCs."""
        on = [r.spike_count for r in self.rgc_cells if r.rgc_type == RGCType.ON_CENTER]
        off = [r.spike_count for r in self.rgc_cells if r.rgc_type == RGCType.OFF_CENTER]
        return {"ON_center": float(np.mean(on)) if on else 0.0,
                "OFF_center": float(np.mean(off)) if off else 0.0}

    def get_photoreceptor_voltages(self) -> np.ndarray:
        return self._photo_voltage.copy()

    def get_bipolar_voltages(self) -> np.ndarray:
        return self._bipolar_voltage.copy()

    def get_rgc_voltages(self) -> np.ndarray:
        return self._rgc_voltage.copy()

    def reset(self) -> None:
        """Reset all state to initial conditions."""
        self._photo_voltage[:] = V_REST_PHOTORECEPTOR
        self._adaptation[:] = 0.5
        self._bipolar_voltage[:] = V_REST_BIPOLAR
        self._rgc_voltage[:] = V_REST_RGC
        self._rgc_m[:] = 0.05; self._rgc_h[:] = 0.6; self._rgc_n[:] = 0.32
        self._rgc_refractory[:] = 0.0
        for rgc in self.rgc_cells:
            rgc.spike_count = 0; rgc.fired = False
        self._total_frames = 0; self._total_spikes = 0

    @property
    def total_neurons(self) -> int:
        return self.n_photo + self.n_bipolar + self.n_rgc

    def summary(self) -> str:
        """Human-readable architecture summary."""
        tc: Dict[str, int] = {}
        for p in self.photoreceptors:
            tc[p.cell_type.value] = tc.get(p.cell_type.value, 0) + 1
        on_b = sum(1 for b in self.bipolar_cells if b.polarity == BipolarPolarity.ON)
        on_r = sum(1 for r in self.rgc_cells if r.rgc_type == RGCType.ON_CENTER)
        ac = np.mean([len(b.center_inputs) for b in self.bipolar_cells]) if self.bipolar_cells else 0
        ar = np.mean([len(r.bipolar_inputs) for r in self.rgc_cells]) if self.rgc_cells else 0
        lines = [f"MolecularRetina (resolution={self.resolution}, fovea_ratio={self.fovea_ratio})",
                 f"  Total neurons: {self.total_neurons}", ""]
        lines.append(f"  Layer 1 — Photoreceptors ({self.n_photo}):")
        for t, c in sorted(tc.items()):
            lines.append(f"    {t}: {c}")
        lines.extend(["", f"  Layer 2 — Bipolar cells ({self.n_bipolar}):",
                       f"    ON: {on_b}  OFF: {self.n_bipolar - on_b}  Avg center: {ac:.1f}",
                       "", f"  Layer 3 — RGCs ({self.n_rgc}):",
                       f"    ON-center: {on_r}  OFF-center: {self.n_rgc - on_r}  Avg inputs: {ar:.1f}",
                       "", f"  Stats: {self._total_frames} frames, {self._total_spikes} spikes"])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MolecularRetina(resolution={self.resolution}, n_photo={self.n_photo}, "
                f"n_bipolar={self.n_bipolar}, n_rgc={self.n_rgc}, total={self.total_neurons})")
