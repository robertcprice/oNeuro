"""Intracellular second messenger cascades.

The missing link between metabotropic receptor activation and downstream
effects on ion channels, receptor sensitivity, and gene expression.

Three major G-protein-coupled signaling pathways are modeled:

1. **Gs/Gi -> cAMP -> PKA** pathway:
   - Gs (D1, beta-1 adrenergic) activates adenylyl cyclase -> cAMP rises
   - Gi (D2, 5-HT1A, alpha-2) inhibits adenylyl cyclase -> cAMP falls
   - cAMP activates PKA, which phosphorylates AMPA, K_v, Ca_v, CREB

2. **Gq -> PLC -> IP3/DAG** pathway:
   - Gq (5-HT2A, mAChR-M1, alpha-1) activates phospholipase C
   - PLC cleaves PIP2 into IP3 + DAG
   - IP3 triggers Ca2+ release from ER
   - DAG activates PKC
   - PKC phosphorylates AMPA (trafficking), NMDA, and ion channels

3. **CaMKII** pathway:
   - Ca2+/calmodulin-dependent protein kinase II
   - Activated when intracellular Ca2+ exceeds ~500 nM
   - THE critical kinase for LTP induction and memory consolidation
   - Autophosphorylation creates a molecular memory switch

4. **MAPK/ERK** cascade:
   - Activated by growth factors AND by cross-talk from PKA/PKC
   - Drives long-term gene expression via CREB phosphorylation
   - Slow timescale (~minutes) for structural plasticity

Output is a PhosphorylationState dict that modifies channel conductances
and receptor sensitivities in the membrane.

Rate constants from: Bhalla & Iyengar (1999) Science 283:381-387,
Bhatt et al. (2005) Biophys J, and Castellani et al. (2005) J Physiol.
All concentrations in nM. Time in ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Rate constants (per ms) from literature
# ---------------------------------------------------------------------------

# Adenylyl cyclase: Gs activates, Gi inhibits
# Bhalla & Iyengar 1999, scaled to ms from s
_AC_BASAL_RATE: float = 0.05            # nM/ms basal cAMP production
_AC_GS_VMAX: float = 1.5                # nM/ms max Gs-stimulated production
_AC_GS_KM: float = 0.5                  # Gs activation for half-max
_AC_GI_INHIBITION_MAX: float = 0.85     # Max fractional inhibition by Gi
_AC_GI_KM: float = 0.4                  # Gi activation for half-max inhibition

# cAMP degradation by phosphodiesterase (PDE)
_PDE_VMAX: float = 2.0                  # nM/ms
_PDE_KM: float = 500.0                  # nM (Michaelis constant)

# PKA activation by cAMP
# PKA holoenzyme: 2 cAMP bind each regulatory subunit (Hill n~1.7)
_PKA_HILL_N: float = 1.7
_PKA_KA: float = 200.0                  # nM cAMP for half-max PKA activation
_PKA_MAX: float = 1.0                   # Normalized max activity

# Phospholipase C (PLC) activation by Gq
_PLC_VMAX: float = 1.0                  # nM/ms
_PLC_KM: float = 0.4                    # Gq activation for half-max

# IP3 dynamics
_IP3_DEGRADATION_RATE: float = 0.003    # per ms (tau ~330 ms)
_IP3_BASAL: float = 10.0                # nM resting IP3

# DAG dynamics
_DAG_DEGRADATION_RATE: float = 0.002    # per ms (tau ~500 ms)
_DAG_BASAL: float = 5.0                 # nM resting DAG

# ER Ca2+ release via IP3 receptor
# De Young-Keizer model simplified: Hill equation for IP3-gated release
_ER_CA_STORE: float = 200_000.0         # nM total ER Ca2+ store
_IP3R_EC50: float = 300.0               # nM IP3 for half-max release
_IP3R_HILL_N: float = 2.5               # Cooperativity
_ER_RELEASE_RATE: float = 0.001         # Fractional release per ms at max
_SERCA_VMAX: float = 0.4               # nM/ms SERCA pump reuptake
_SERCA_KM: float = 200.0               # nM Ca2+ for half-max SERCA

# PKC activation by DAG + Ca2+
_PKC_DAG_EC50: float = 100.0            # nM DAG for half-max
_PKC_CA_EC50: float = 600.0             # nM Ca2+ for half-max
_PKC_HILL_N: float = 1.5

# CaMKII activation
# Lisman (1989, 1994): bistable switch with autophosphorylation
_CAMKII_CA_THRESHOLD: float = 500.0     # nM Ca2+ activation threshold
_CAMKII_CA_EC50: float = 800.0          # nM for half-max activation
_CAMKII_HILL_N: float = 4.0             # High cooperativity (12-mer ring)
_CAMKII_AUTOPHOSPHO_RATE: float = 0.001 # per ms positive feedback
_CAMKII_PHOSPHATASE_RATE: float = 0.0003  # per ms PP1 dephosphorylation
_CAMKII_MAX: float = 1.0

# MAPK/ERK cascade
# Slow cascade: Ras -> Raf -> MEK -> ERK
_ERK_ACTIVATION_RATE: float = 0.0002    # per ms (slow: tau ~5 s)
_ERK_DEACTIVATION_RATE: float = 0.0005  # per ms
_ERK_PKA_WEIGHT: float = 0.3            # PKA cross-talk to ERK
_ERK_PKC_WEIGHT: float = 0.5            # PKC cross-talk to ERK
_ERK_CAMKII_WEIGHT: float = 0.2         # CaMKII cross-talk to ERK

# CREB phosphorylation
# PKA and ERK both phosphorylate CREB at Ser133
_CREB_PHOSPHO_RATE: float = 0.0005      # per ms
_CREB_DEPHOSPHO_RATE: float = 0.0002    # per ms (PP1)

# Phosphorylation rates for downstream targets (per ms)
_PKA_PHOSPHO_RATE: float = 0.002        # Rate PKA phosphorylates targets
_PKA_DEPHOSPHO_RATE: float = 0.001      # Basal phosphatase rate
_PKC_PHOSPHO_RATE: float = 0.002
_PKC_DEPHOSPHO_RATE: float = 0.001
_CAMKII_PHOSPHO_RATE: float = 0.003     # CaMKII is potent phosphorylator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhosphorylationState:
    """Phosphorylation levels of key substrates, all in [0, 1].

    These values directly modulate channel conductances and receptor
    properties when read by the membrane/neuron integration layer.

    AMPA_p: Phosphorylation of AMPA GluA1 at Ser831 (CaMKII) and Ser845 (PKA).
             Increases single-channel conductance and membrane insertion.
    Kv_p:   Phosphorylation of Kv channels. PKA phosphorylation shifts
             activation curve, generally reducing delayed rectifier current.
    Cav_p:  Phosphorylation of L-type Ca_v channels (Ser1928 by PKA).
             Increases Ca2+ current, enhancing excitability and Ca2+ signaling.
    CREB_p: Phosphorylation of CREB at Ser133. Activates CRE-dependent
             gene transcription (BDNF, Arc, c-Fos) for long-term plasticity.
    """

    AMPA_p: float = 0.0
    Kv_p: float = 0.0
    Cav_p: float = 0.0
    CREB_p: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Return phosphorylation state as a dictionary."""
        return {
            "AMPA_p": self.AMPA_p,
            "Kv_p": self.Kv_p,
            "Cav_p": self.Cav_p,
            "CREB_p": self.CREB_p,
        }

    def clamp(self) -> None:
        """Clamp all values to [0, 1]."""
        self.AMPA_p = max(0.0, min(1.0, self.AMPA_p))
        self.Kv_p = max(0.0, min(1.0, self.Kv_p))
        self.Cav_p = max(0.0, min(1.0, self.Cav_p))
        self.CREB_p = max(0.0, min(1.0, self.CREB_p))


@dataclass
class CascadeState:
    """Internal state of all second messenger cascades.

    Concentrations are in nM unless noted. Kinase activities are
    normalized to [0, 1].
    """

    # G-protein activation levels [0, 1] (set from receptor_activations)
    gs_active: float = 0.0
    gi_active: float = 0.0
    gq_active: float = 0.0

    # cAMP / PKA pathway
    camp: float = 50.0           # Basal cAMP ~50-100 nM
    pka_activity: float = 0.0    # [0, 1]

    # PLC / IP3 / DAG pathway
    plc_activity: float = 0.0    # [0, 1]
    ip3: float = _IP3_BASAL      # nM
    dag: float = _DAG_BASAL      # nM
    er_ca_store: float = _ER_CA_STORE  # nM Ca2+ in ER lumen
    pkc_activity: float = 0.0    # [0, 1]

    # Ca2+ released from ER into cytoplasm (added to ca_level_nM)
    er_ca_released: float = 0.0  # nM released this step

    # CaMKII
    camkii_activity: float = 0.0  # [0, 1]

    # MAPK/ERK
    erk_activity: float = 0.0     # [0, 1]

    # Time tracking
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _hill(x: float, ec50: float, n: float) -> float:
    """Hill equation: x^n / (ec50^n + x^n). Returns [0, 1]."""
    if x <= 0.0 or ec50 <= 0.0:
        return 0.0
    xn = x ** n
    return xn / (ec50 ** n + xn)


def _michaelis_menten(substrate: float, vmax: float, km: float) -> float:
    """Michaelis-Menten kinetics: rate = Vmax * S / (Km + S)."""
    if substrate <= 0.0:
        return 0.0
    return vmax * substrate / (km + substrate)


# ---------------------------------------------------------------------------
# Main system
# ---------------------------------------------------------------------------

@dataclass
class SecondMessengerSystem:
    """ODE-based intracellular signaling cascades.

    Bridges the gap between slow metabotropic receptor activation and
    downstream effects on channels, receptors, and gene expression.

    The ``receptor_activations`` dict uses cascade_effect strings as keys
    (matching the metabotropic receptor properties in receptors.py):

        - ``"cAMP_increase"``  : Gs-coupled receptors (D1, beta-1)
        - ``"cAMP_decrease"``  : Gi-coupled receptors (D2, 5-HT1A, alpha-2)
        - ``"IP3_DAG_increase"``: Gq-coupled receptors (5-HT2A, mAChR-M1, alpha-1)

    Values are activation levels in [0, 1].

    Usage::

        system = SecondMessengerSystem()
        activations = {"cAMP_increase": 0.7, "IP3_DAG_increase": 0.3}
        system.step(dt=0.1, receptor_activations=activations, ca_level_nM=120.0)
        pstate = system.phosphorylation_state
        # pstate.AMPA_p, pstate.Kv_p etc. now reflect kinase activity
    """

    state: CascadeState = field(default_factory=CascadeState)
    phosphorylation: PhosphorylationState = field(
        default_factory=PhosphorylationState
    )

    # Configuration
    enable_camkii_bistability: bool = True  # Lisman bistable switch
    enable_erk_crosstalk: bool = True       # Allow PKA/PKC -> ERK

    def step(
        self,
        dt: float,
        receptor_activations: Optional[Dict[str, float]] = None,
        ca_level_nM: float = 50.0,
    ) -> Dict[str, float]:
        """Advance all second messenger cascades by dt milliseconds.

        Args:
            dt: Timestep in ms. Typical: 0.01-0.1 ms.
            receptor_activations: Maps cascade_effect strings to activation
                levels [0, 1]. Keys are ``"cAMP_increase"``,
                ``"cAMP_decrease"``, ``"IP3_DAG_increase"``.
                Missing keys default to 0.
            ca_level_nM: Intracellular calcium concentration in nM.
                This comes from the membrane model (spike-triggered influx,
                NMDA, Ca_v channels).

        Returns:
            Phosphorylation state as dict with keys:
            ``AMPA_p``, ``Kv_p``, ``Cav_p``, ``CREB_p`` (all [0, 1]).
        """
        s = self.state
        s.total_time_ms += dt

        if receptor_activations is None:
            receptor_activations = {}

        # -- Phase 1: G-protein activation from receptor signals ----------

        # Gs: D1, beta-1 adrenergic -> cAMP up
        gs_input = receptor_activations.get("cAMP_increase", 0.0)
        s.gs_active += dt * (gs_input - s.gs_active) / 20.0  # tau ~20 ms

        # Gi: D2, 5-HT1A, alpha-2 -> cAMP down
        gi_input = receptor_activations.get("cAMP_decrease", 0.0)
        s.gi_active += dt * (gi_input - s.gi_active) / 20.0

        # Gq: 5-HT2A, mAChR-M1, alpha-1 -> PLC
        gq_input = receptor_activations.get("IP3_DAG_increase", 0.0)
        s.gq_active += dt * (gq_input - s.gq_active) / 20.0

        # Clamp G-protein activities
        s.gs_active = max(0.0, min(1.0, s.gs_active))
        s.gi_active = max(0.0, min(1.0, s.gi_active))
        s.gq_active = max(0.0, min(1.0, s.gq_active))

        # -- Phase 2: cAMP / PKA pathway ---------------------------------

        self._update_camp_pka(dt)

        # -- Phase 3: PLC / IP3 / DAG pathway -----------------------------

        self._update_plc_ip3_dag(dt, ca_level_nM)

        # -- Phase 4: CaMKII ----------------------------------------------

        # Total cytoplasmic Ca2+ = membrane Ca2+ + ER-released Ca2+
        total_ca = ca_level_nM + s.er_ca_released
        self._update_camkii(dt, total_ca)

        # -- Phase 5: MAPK/ERK cascade ------------------------------------

        self._update_erk(dt)

        # -- Phase 6: Phosphorylation of downstream targets ----------------

        self._update_phosphorylation(dt, total_ca)

        # -- Phase 7: CREB -------------------------------------------------

        self._update_creb(dt)

        s.total_time_ms = s.total_time_ms  # no-op, for clarity

        return self.phosphorylation.as_dict()

    # -------------------------------------------------------------------
    # Pathway update methods
    # -------------------------------------------------------------------

    def _update_camp_pka(self, dt: float) -> None:
        """Update cAMP production/degradation and PKA activation."""
        s = self.state

        # Adenylyl cyclase production (Gs stimulation)
        ac_gs = _AC_GS_VMAX * _hill(s.gs_active, _AC_GS_KM, 1.0)

        # Gi inhibition (reduces total AC activity)
        gi_inhibition = _AC_GI_INHIBITION_MAX * _hill(
            s.gi_active, _AC_GI_KM, 1.0
        )
        ac_total = (_AC_BASAL_RATE + ac_gs) * (1.0 - gi_inhibition)

        # PDE degradation (Michaelis-Menten)
        pde_rate = _michaelis_menten(s.camp, _PDE_VMAX, _PDE_KM)

        # dcamp/dt = production - degradation
        d_camp = ac_total - pde_rate
        s.camp += d_camp * dt
        s.camp = max(0.0, s.camp)

        # PKA activation by cAMP (cooperative Hill)
        s.pka_activity = _PKA_MAX * _hill(s.camp, _PKA_KA, _PKA_HILL_N)

    def _update_plc_ip3_dag(self, dt: float, ca_level_nM: float) -> None:
        """Update PLC -> IP3 + DAG cascade and ER Ca2+ release."""
        s = self.state

        # PLC activation by Gq
        s.plc_activity = _hill(s.gq_active, _PLC_KM, 1.0)

        # PLC cleaves PIP2 -> IP3 + DAG (equimolar)
        plc_production = _PLC_VMAX * s.plc_activity

        # IP3 dynamics: production - degradation (IP3 5-phosphatase)
        d_ip3 = plc_production - _IP3_DEGRADATION_RATE * (s.ip3 - _IP3_BASAL)
        s.ip3 += d_ip3 * dt
        s.ip3 = max(0.0, s.ip3)

        # DAG dynamics: production - degradation (DAG lipase + DAG kinase)
        d_dag = plc_production - _DAG_DEGRADATION_RATE * (s.dag - _DAG_BASAL)
        s.dag += d_dag * dt
        s.dag = max(0.0, s.dag)

        # ER Ca2+ release via IP3 receptor
        # IP3R has bell-shaped Ca2+ dependence (activation at low Ca2+,
        # inhibition at high Ca2+). Simplified to Hill on IP3 alone.
        ip3r_open = _hill(s.ip3, _IP3R_EC50, _IP3R_HILL_N)
        ca_release = _ER_RELEASE_RATE * ip3r_open * s.er_ca_store * dt

        # SERCA pump pushes Ca2+ back into ER
        total_ca = ca_level_nM + s.er_ca_released
        serca_uptake = _michaelis_menten(total_ca, _SERCA_VMAX, _SERCA_KM) * dt

        s.er_ca_released += ca_release - serca_uptake
        s.er_ca_released = max(0.0, s.er_ca_released)

        # ER store depletes and refills
        s.er_ca_store -= ca_release
        s.er_ca_store += serca_uptake
        s.er_ca_store = max(0.0, min(_ER_CA_STORE, s.er_ca_store))

        # PKC activation: requires BOTH DAG and Ca2+
        # PKC = classical isoform, needs DAG + Ca2+ + phosphatidylserine
        dag_factor = _hill(s.dag, _PKC_DAG_EC50, _PKC_HILL_N)
        ca_factor = _hill(total_ca, _PKC_CA_EC50, 1.0)
        pkc_target = dag_factor * ca_factor  # Multiplicative AND gate
        # Smooth approach to target
        tau_pkc = 50.0  # ms activation time constant
        s.pkc_activity += dt * (pkc_target - s.pkc_activity) / tau_pkc
        s.pkc_activity = max(0.0, min(1.0, s.pkc_activity))

    def _update_camkii(self, dt: float, total_ca_nM: float) -> None:
        """Update CaMKII activation with optional bistable switch.

        CaMKII is a 12-subunit holoenzyme that autophosphorylates when
        Ca2+/calmodulin is bound. Once autophosphorylated, it remains
        active even after Ca2+ drops (Lisman's molecular memory switch).

        Activation requires Ca2+ > 500 nM (Bhalla & Iyengar 1999).
        """
        s = self.state

        # Ca2+/calmodulin activation (steep Hill function)
        ca_activation = _hill(
            total_ca_nM, _CAMKII_CA_EC50, _CAMKII_HILL_N
        )

        if self.enable_camkii_bistability:
            # Bistable dynamics: autophosphorylation creates positive feedback
            # d(CaMKII*)/dt = activation * (1 - CaMKII*) * (autophospho + ca_drive)
            #                 - phosphatase * CaMKII*
            autophospho = _CAMKII_AUTOPHOSPHO_RATE * s.camkii_activity
            drive = ca_activation * 0.005  # Ca2+-driven activation rate per ms
            activation_rate = (drive + autophospho) * (1.0 - s.camkii_activity)
            deactivation_rate = _CAMKII_PHOSPHATASE_RATE * s.camkii_activity

            # Only allow activation above Ca2+ threshold
            if total_ca_nM < _CAMKII_CA_THRESHOLD:
                activation_rate = 0.0

            d_camkii = activation_rate - deactivation_rate
        else:
            # Simple Hill activation without bistability
            target = ca_activation if total_ca_nM >= _CAMKII_CA_THRESHOLD else 0.0
            d_camkii = (target - s.camkii_activity) * 0.001  # tau ~1000 ms

        s.camkii_activity += d_camkii * dt
        s.camkii_activity = max(0.0, min(_CAMKII_MAX, s.camkii_activity))

    def _update_erk(self, dt: float) -> None:
        """Update MAPK/ERK cascade.

        ERK is activated by cross-talk from PKA, PKC, and CaMKII via the
        Ras-Raf-MEK pathway. This is the slow cascade (~minutes) that
        drives gene expression changes.
        """
        s = self.state

        if not self.enable_erk_crosstalk:
            return

        # Weighted input from upstream kinases
        erk_drive = (
            _ERK_PKA_WEIGHT * s.pka_activity
            + _ERK_PKC_WEIGHT * s.pkc_activity
            + _ERK_CAMKII_WEIGHT * s.camkii_activity
        )

        # Activation and deactivation
        activation = _ERK_ACTIVATION_RATE * erk_drive * (1.0 - s.erk_activity)
        deactivation = _ERK_DEACTIVATION_RATE * s.erk_activity

        s.erk_activity += (activation - deactivation) * dt
        s.erk_activity = max(0.0, min(1.0, s.erk_activity))

    def _update_phosphorylation(self, dt: float, total_ca_nM: float) -> None:
        """Update phosphorylation state of downstream targets.

        Each target integrates signals from multiple kinases:

        - AMPA GluA1: PKA (Ser845) + CaMKII (Ser831) + PKC
          Increases conductance and promotes membrane insertion.

        - K_v channels: PKA phosphorylation shifts V1/2 of activation.
          Net effect: reduced K+ current -> increased excitability.

        - Ca_v (L-type): PKA (Ser1928) increases open probability.
          Enhances Ca2+ influx during depolarization.
        """
        p = self.phosphorylation
        s = self.state

        # --- AMPA_p: PKA + CaMKII + PKC all contribute ---
        ampa_phospho = (
            _PKA_PHOSPHO_RATE * s.pka_activity * 0.4
            + _CAMKII_PHOSPHO_RATE * s.camkii_activity * 0.4
            + _PKC_PHOSPHO_RATE * s.pkc_activity * 0.2
        )
        ampa_dephospho = _PKA_DEPHOSPHO_RATE * p.AMPA_p
        p.AMPA_p += (ampa_phospho * (1.0 - p.AMPA_p) - ampa_dephospho) * dt

        # --- Kv_p: primarily PKA ---
        kv_phospho = _PKA_PHOSPHO_RATE * s.pka_activity * 0.7
        kv_phospho += _PKC_PHOSPHO_RATE * s.pkc_activity * 0.3
        kv_dephospho = _PKA_DEPHOSPHO_RATE * p.Kv_p
        p.Kv_p += (kv_phospho * (1.0 - p.Kv_p) - kv_dephospho) * dt

        # --- Cav_p: PKA is dominant (Ser1928 on Cav1.2) ---
        cav_phospho = _PKA_PHOSPHO_RATE * s.pka_activity * 0.8
        cav_phospho += _CAMKII_PHOSPHO_RATE * s.camkii_activity * 0.2
        cav_dephospho = _PKA_DEPHOSPHO_RATE * p.Cav_p
        p.Cav_p += (cav_phospho * (1.0 - p.Cav_p) - cav_dephospho) * dt

        p.clamp()

    def _update_creb(self, dt: float) -> None:
        """Update CREB phosphorylation at Ser133.

        CREB is phosphorylated by PKA (directly) and ERK (via RSK).
        Phospho-CREB recruits CBP and drives transcription of
        plasticity genes (BDNF, Arc, c-Fos, Zif268).
        """
        s = self.state
        p = self.phosphorylation

        # Both PKA and ERK drive CREB phosphorylation
        creb_drive = (
            0.6 * s.pka_activity
            + 0.3 * s.erk_activity
            + 0.1 * s.camkii_activity  # Minor direct contribution
        )

        phospho = _CREB_PHOSPHO_RATE * creb_drive * (1.0 - p.CREB_p)
        dephospho = _CREB_DEPHOSPHO_RATE * p.CREB_p

        p.CREB_p += (phospho - dephospho) * dt
        p.CREB_p = max(0.0, min(1.0, p.CREB_p))

    # -------------------------------------------------------------------
    # Public accessors
    # -------------------------------------------------------------------

    @property
    def phosphorylation_state(self) -> PhosphorylationState:
        """Current phosphorylation state (read-only reference)."""
        return self.phosphorylation

    @property
    def camp_level(self) -> float:
        """Current cAMP concentration in nM."""
        return self.state.camp

    @property
    def ip3_level(self) -> float:
        """Current IP3 concentration in nM."""
        return self.state.ip3

    @property
    def dag_level(self) -> float:
        """Current DAG concentration in nM."""
        return self.state.dag

    @property
    def pka_activity(self) -> float:
        """Current PKA activity [0, 1]."""
        return self.state.pka_activity

    @property
    def pkc_activity(self) -> float:
        """Current PKC activity [0, 1]."""
        return self.state.pkc_activity

    @property
    def camkii_activity(self) -> float:
        """Current CaMKII activity [0, 1]."""
        return self.state.camkii_activity

    @property
    def erk_activity(self) -> float:
        """Current ERK activity [0, 1]."""
        return self.state.erk_activity

    @property
    def er_calcium_released(self) -> float:
        """Calcium released from ER stores this step (nM)."""
        return self.state.er_ca_released

    def get_kinase_summary(self) -> Dict[str, float]:
        """Return all kinase activities as a dict."""
        return {
            "PKA": self.state.pka_activity,
            "PKC": self.state.pkc_activity,
            "CaMKII": self.state.camkii_activity,
            "ERK": self.state.erk_activity,
        }

    def get_messenger_summary(self) -> Dict[str, float]:
        """Return all second messenger concentrations."""
        return {
            "cAMP_nM": self.state.camp,
            "IP3_nM": self.state.ip3,
            "DAG_nM": self.state.dag,
            "ER_Ca_released_nM": self.state.er_ca_released,
            "ER_Ca_store_nM": self.state.er_ca_store,
        }

    def reset(self) -> None:
        """Reset all cascades to resting state."""
        self.state = CascadeState()
        self.phosphorylation = PhosphorylationState()
