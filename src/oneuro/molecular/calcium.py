"""Multi-compartment calcium signaling system.

Replaces the single-float ``_ca_internal_nM`` in membrane.py with a
biophysically grounded model of intracellular calcium handling.

Four compartments:
  1. **Cytoplasmic** -- bulk cytoplasm (~50-100 nM at rest).
  2. **ER store** -- endoplasmic reticulum lumen (400-600 uM).
  3. **Mitochondrial** -- mitochondrial matrix buffering.
  4. **Microdomain** -- sub-membrane nanodomain near VGCCs; transiently
     reaches 10-100 uM during an action potential and decays within ~1 ms.

Inter-compartment fluxes use Michaelis-Menten or Hill kinetics:
  - IP3R:  ER -> cytoplasm (IP3-gated, bell-shaped Ca dependence).
  - RyR:   ER -> cytoplasm (CICR, activated by cytoplasmic Ca > ~500 nM).
  - SERCA: cytoplasm -> ER (ATP-dependent reuptake, Km ~200 nM).
  - MCU:   cytoplasm -> mitochondria (low affinity, high capacity).
  - PMCA:  cytoplasm -> extracellular (plasma membrane Ca ATPase).
  - NCX:   cytoplasm -> extracellular (3 Na+:1 Ca2+ exchanger).

Concentrations are stored in **nM** and time is in **ms** throughout.

CaMKII activation is derived from microdomain calcium via a
calmodulin binding model (Hill coefficient = 4, Kd ~1000 nM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical / physiological constants
# ---------------------------------------------------------------------------

# Resting concentrations (nM)
REST_CYTOPLASMIC_NM: float = 50.0
REST_ER_NM: float = 500_000.0        # 500 uM = 500,000 nM
REST_MITOCHONDRIAL_NM: float = 100.0  # matrix buffered equivalent
REST_MICRODOMAIN_NM: float = 50.0     # same as cytoplasm at rest

# AP-triggered influx into microdomain (nM bolus per spike)
SPIKE_INFLUX_NM: float = 50_000.0     # ~50 uM peak in nanodomain

# Microdomain -> cytoplasm diffusion time constant (ms)
MICRODOMAIN_DIFFUSION_TAU: float = 0.5  # sub-ms equilibration

# Extracellular calcium (effectively infinite reservoir, ~2 mM)
EXTRACELLULAR_CA_NM: float = 2_000_000.0


# ---------------------------------------------------------------------------
# Flux parameter containers
# ---------------------------------------------------------------------------

@dataclass
class IP3RParams:
    """IP3 receptor: ER -> cytoplasm release channel.

    Bell-shaped cytoplasmic Ca dependence (activation then inhibition)
    combined with IP3 gating.
    """

    v_max: float = 6000.0       # nM/ms  max flux
    k_ip3: float = 300.0        # nM     IP3 half-activation
    k_act: float = 200.0        # nM     Ca activation Kd
    k_inh: float = 500.0        # nM     Ca inhibition Kd
    hill_ip3: float = 2.0       # IP3 cooperativity
    hill_act: float = 2.0       # Ca activation cooperativity
    hill_inh: float = 2.0       # Ca inhibition cooperativity

    def flux(self, ca_cyt: float, ca_er: float, ip3: float) -> float:
        """Compute ER -> cytoplasm flux (nM/ms). Positive = release."""
        if ip3 <= 0.0 or ca_er <= 0.0:
            return 0.0

        # IP3 gating
        ip3_gate = _hill(ip3, self.k_ip3, self.hill_ip3)

        # Bell-shaped Ca dependence: activation * (1 - inhibition)
        ca_act = _hill(ca_cyt, self.k_act, self.hill_act)
        ca_inh = _hill(ca_cyt, self.k_inh, self.hill_inh)

        open_prob = ip3_gate * ca_act * (1.0 - ca_inh)

        # Driving force proportional to ER-cytoplasm gradient
        gradient = (ca_er - ca_cyt) / ca_er  # normalised, avoids huge raw diff
        gradient = max(0.0, gradient)

        return self.v_max * open_prob * gradient


@dataclass
class RyRParams:
    """Ryanodine receptor: Ca-induced Ca release (CICR) from ER.

    Activated when cytoplasmic Ca exceeds ~500 nM.  Steep Hill curve
    gives the regenerative positive-feedback character of CICR.
    """

    v_max: float = 2000.0       # nM/ms  (moderate -- CICR is a transient burst)
    k_act: float = 500.0        # nM     Ca activation Kd
    hill: float = 3.0           # steep cooperativity
    k_inh: float = 1500.0       # nM     Ca inactivation Kd (must be low enough
                                #        for CICR to self-terminate before ER drains)
    hill_inh: float = 3.0       # steep inactivation for sharp cutoff

    def flux(self, ca_cyt: float, ca_er: float) -> float:
        """Compute ER -> cytoplasm flux (nM/ms). Positive = release."""
        if ca_er <= 0.0:
            return 0.0

        activation = _hill(ca_cyt, self.k_act, self.hill)
        inactivation = 1.0 - _hill(ca_cyt, self.k_inh, self.hill_inh)
        inactivation = max(0.0, inactivation)

        gradient = (ca_er - ca_cyt) / ca_er
        gradient = max(0.0, gradient)

        return self.v_max * activation * inactivation * gradient


@dataclass
class SERCAParams:
    """Sarco/endoplasmic reticulum Ca ATPase: cytoplasm -> ER pump.

    Michaelis-Menten with Hill coefficient ~2. Requires ATP.
    """

    v_max: float = 1500.0       # nM/ms  max pump rate
    km: float = 200.0           # nM     half-maximal Ca
    hill: float = 2.0           # cooperativity

    def flux(self, ca_cyt: float, atp_available: bool = True) -> float:
        """Compute cytoplasm -> ER flux (nM/ms). Positive = uptake.

        Returns 0 when ATP is unavailable.
        """
        if not atp_available or ca_cyt <= 0.0:
            return 0.0
        return self.v_max * _hill(ca_cyt, self.km, self.hill)


@dataclass
class MCUParams:
    """Mitochondrial Ca uniporter: cytoplasm -> mitochondria.

    Low affinity (Kd ~10 uM), high capacity.  Only significant when
    cytoplasmic Ca is elevated well above resting.
    """

    v_max: float = 800.0        # nM/ms
    km: float = 10_000.0        # nM  (~10 uM -- low affinity)
    hill: float = 2.5           # steep voltage-dependent activation

    def flux(self, ca_cyt: float) -> float:
        """Compute cytoplasm -> mitochondria flux (nM/ms)."""
        if ca_cyt <= 0.0:
            return 0.0
        return self.v_max * _hill(ca_cyt, self.km, self.hill)


@dataclass
class PMCAParams:
    """Plasma membrane Ca ATPase: cytoplasm -> extracellular export.

    High affinity, low capacity.  Primary mechanism keeping resting
    Ca low. Requires ATP.
    """

    v_max: float = 300.0        # nM/ms
    km: float = 100.0           # nM  high affinity
    hill: float = 1.0

    def flux(self, ca_cyt: float, atp_available: bool = True) -> float:
        """Compute cytoplasm -> extracellular flux (nM/ms)."""
        if not atp_available or ca_cyt <= 0.0:
            return 0.0
        return self.v_max * _hill(ca_cyt, self.km, self.hill)


@dataclass
class NCXParams:
    """Na/Ca exchanger: 3 Na+ in, 1 Ca2+ out.

    Electrogenic, voltage-sensitive.  In forward mode it exports Ca.
    Operates at lower affinity than PMCA but higher capacity.
    """

    v_max: float = 1500.0       # nM/ms  forward mode max (NCX is the
                                #        primary high-capacity exporter)
    km: float = 1000.0          # nM
    hill: float = 1.0

    def flux(self, ca_cyt: float) -> float:
        """Compute cytoplasm -> extracellular flux (nM/ms).

        Simplified forward-mode only (no reverse-mode at resting Na/Ca).
        """
        if ca_cyt <= 0.0:
            return 0.0
        return self.v_max * _hill(ca_cyt, self.km, self.hill)


# ---------------------------------------------------------------------------
# CaMKII / calmodulin model
# ---------------------------------------------------------------------------

@dataclass
class CaMKIIParams:
    """CaMKII activation via Ca/calmodulin binding.

    Calmodulin binds 4 Ca ions cooperatively (Hill ~4), then activates
    CaMKII.  The microdomain concentration is the relevant signal
    because CaMKII is tethered to the PSD near channels.
    """

    kd: float = 1000.0          # nM  half-max microdomain Ca for CaM binding
    hill: float = 4.0           # 4 Ca ions per calmodulin
    autophosphorylation_tau: float = 50.0  # ms  decay of autonomous activity

    def activation(self, ca_microdomain: float) -> float:
        """Instantaneous CaMKII activation fraction [0, 1]."""
        return _hill(ca_microdomain, self.kd, self.hill)


# ---------------------------------------------------------------------------
# Main CalciumSystem
# ---------------------------------------------------------------------------

@dataclass
class CalciumSystem:
    """Multi-compartment intracellular calcium signaling system.

    Maintains four calcium pools and computes inter-compartment fluxes
    each timestep. Designed to be composed into MolecularMembrane as a
    drop-in replacement for the single ``_ca_internal_nM`` float.

    All concentrations are in nM. Time is in ms.
    """

    # ----- Compartment concentrations (nM) -----
    cytoplasmic: float = REST_CYTOPLASMIC_NM
    er_store: float = REST_ER_NM
    mitochondrial: float = REST_MITOCHONDRIAL_NM
    microdomain: float = REST_MICRODOMAIN_NM

    # ----- Flux channel / pump parameters -----
    ip3r: IP3RParams = field(default_factory=IP3RParams)
    ryr: RyRParams = field(default_factory=RyRParams)
    serca: SERCAParams = field(default_factory=SERCAParams)
    mcu: MCUParams = field(default_factory=MCUParams)
    pmca: PMCAParams = field(default_factory=PMCAParams)
    ncx: NCXParams = field(default_factory=NCXParams)
    camkii: CaMKIIParams = field(default_factory=CaMKIIParams)

    # ----- Mitochondrial slow release back to cytoplasm -----
    _mito_release_rate: float = 0.02  # fraction per ms

    # ----- ER leak (passive, small) -----
    _er_leak_rate: float = 0.0001  # fraction of gradient per ms

    # ----- Cytoplasmic buffering (endogenous Ca-binding proteins) -----
    _buffer_capacity: float = 20.0  # effective buffering ratio
    # Real neurons buffer ~95-99% of free Ca.  A buffering ratio of 20
    # means only 1/20 of net flux appears as free [Ca].

    # ----- Passive leak from extracellular space -----
    # At rest, a small inward Ca leak through non-specific channels
    # balances PMCA/NCX export to maintain ~50 nM cytoplasmic [Ca].
    _passive_leak_rate: float = 210.0  # nM/ms -- calibrated so that at
    # resting [Ca]=50 nM the net flux (leak_in - PMCA - NCX - SERCA + ER_leak)
    # is approximately zero, maintaining the 50 nM resting level.

    def step(
        self,
        dt: float,
        channel_ca_influx_nM: float = 0.0,
        ip3_level: float = 0.0,
        atp_available: bool = True,
    ) -> None:
        """Advance all compartments by *dt* milliseconds.

        Args:
            dt: Timestep in ms.
            channel_ca_influx_nM: Voltage-gated Ca channel influx this step
                (added to microdomain).
            ip3_level: IP3 concentration in nM (from metabotropic signaling).
            atp_available: Whether ATP is available for SERCA and PMCA.
        """
        # ---- 1. Channel influx goes to microdomain ----
        if channel_ca_influx_nM > 0.0:
            self.microdomain += channel_ca_influx_nM

        # ---- 2. Microdomain -> cytoplasm diffusion ----
        # Exponential equilibration toward cytoplasmic level.
        micro_diff = (self.microdomain - self.cytoplasmic)
        diffusion_flux = micro_diff * (1.0 - math.exp(-dt / MICRODOMAIN_DIFFUSION_TAU))
        self.microdomain -= diffusion_flux
        self.cytoplasmic += diffusion_flux / self._buffer_capacity

        # ---- 3. ER <-> cytoplasm fluxes ----
        # IP3R release
        ip3r_flux = self.ip3r.flux(self.cytoplasmic, self.er_store, ip3_level) * dt
        # RyR (CICR) release
        ryr_flux = self.ryr.flux(self.cytoplasmic, self.er_store) * dt
        # SERCA pump (reuptake)
        serca_flux = self.serca.flux(self.cytoplasmic, atp_available) * dt
        # Passive ER leak
        er_leak = self._er_leak_rate * (self.er_store - self.cytoplasmic) * dt

        net_er_release = ip3r_flux + ryr_flux + er_leak - serca_flux

        self.cytoplasmic += net_er_release / self._buffer_capacity
        self.er_store -= net_er_release

        # ---- 4. Cytoplasm -> mitochondria (MCU) ----
        mcu_flux = self.mcu.flux(self.cytoplasmic) * dt
        self.cytoplasmic -= mcu_flux / self._buffer_capacity
        self.mitochondrial += mcu_flux

        # Mitochondrial slow release back to cytoplasm
        mito_release = self.mitochondrial * self._mito_release_rate * dt
        self.mitochondrial -= mito_release
        self.cytoplasmic += mito_release / self._buffer_capacity

        # ---- 5. Extracellular <-> cytoplasm (PMCA + NCX export, passive leak in) ----
        pmca_flux = self.pmca.flux(self.cytoplasmic, atp_available) * dt
        ncx_flux = self.ncx.flux(self.cytoplasmic) * dt
        passive_leak_in = self._passive_leak_rate * dt
        net_export = pmca_flux + ncx_flux - passive_leak_in
        self.cytoplasmic -= net_export / self._buffer_capacity

        # ---- 6. Clamp all compartments to non-negative ----
        self.cytoplasmic = max(0.0, self.cytoplasmic)
        self.er_store = max(0.0, self.er_store)
        self.mitochondrial = max(0.0, self.mitochondrial)
        self.microdomain = max(0.0, self.microdomain)

    def add_channel_influx(self, amount_nM: float) -> None:
        """Add Ca influx to the microdomain (not bulk cytoplasm).

        This is the correct target for voltage-gated Ca channel currents
        because VGCCs create local nanodomains of high [Ca] at the
        channel mouth before diffusing into the bulk cytoplasm.
        """
        self.microdomain += max(0.0, amount_nM)

    def spike_influx(self) -> None:
        """Standard action-potential-triggered Ca entry into microdomain.

        A typical AP opens ~100 VGCCs for ~0.5 ms, admitting enough Ca
        to transiently raise the nanodomain to 10-100 uM.
        """
        self.microdomain += SPIKE_INFLUX_NM

    # ---- Read-only properties ----

    @property
    def cytoplasmic_nM(self) -> float:
        """Bulk cytoplasmic free [Ca] in nM."""
        return self.cytoplasmic

    @property
    def microdomain_nM(self) -> float:
        """Sub-membrane nanodomain [Ca] in nM."""
        return self.microdomain

    @property
    def er_nM(self) -> float:
        """Endoplasmic reticulum lumen [Ca] in nM."""
        return self.er_store

    @property
    def mitochondrial_nM(self) -> float:
        """Mitochondrial matrix [Ca] in nM."""
        return self.mitochondrial

    @property
    def camkii_activation(self) -> float:
        """CaMKII activation level [0, 1] from microdomain Ca.

        CaMKII is tethered to the post-synaptic density adjacent to
        Ca channels, so it senses the *microdomain* concentration
        through calmodulin (Hill n=4, Kd ~1 uM).
        """
        return self.camkii.activation(self.microdomain)

    @property
    def ca_internal(self) -> float:
        """Backward-compatible property returning cytoplasmic [Ca] in nM.

        Drop-in replacement for the old ``_ca_internal_nM`` float in
        MolecularMembrane.
        """
        return self.cytoplasmic

    # ---- Utilities ----

    def reset(self) -> None:
        """Reset all compartments to resting concentrations."""
        self.cytoplasmic = REST_CYTOPLASMIC_NM
        self.er_store = REST_ER_NM
        self.mitochondrial = REST_MITOCHONDRIAL_NM
        self.microdomain = REST_MICRODOMAIN_NM

    def total_ca(self) -> float:
        """Total calcium across all compartments (conservation check)."""
        return self.cytoplasmic + self.er_store + self.mitochondrial + self.microdomain

    def __repr__(self) -> str:
        return (
            f"CalciumSystem("
            f"cyt={self.cytoplasmic:.1f} nM, "
            f"micro={self.microdomain:.1f} nM, "
            f"ER={self.er_store:.0f} nM, "
            f"mito={self.mitochondrial:.1f} nM, "
            f"CaMKII={self.camkii_activation:.3f})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hill(concentration: float, kd: float, n: float) -> float:
    """Hill equation: ``c^n / (Kd^n + c^n)``.

    Returns a value in [0, 1].  Safe for concentration <= 0.
    """
    if concentration <= 0.0 or kd <= 0.0:
        return 0.0
    cn = concentration ** n
    kn = kd ** n
    return cn / (kn + cn)
