"""Molecular circadian clock and sleep homeostasis.

Implements the Transcription-Translation Feedback Loop (TTFL) that drives
~24-hour rhythms in neurotransmitter synthesis, receptor expression, and
neuronal excitability.  Paired with an adenosine-based sleep homeostasis
model that tracks sleep pressure from neural activity.

The TTFL core:
    CLOCK:BMAL1 (activator) drives Per and Cry transcription.
    PER:CRY complex (inhibitor) represses CLOCK:BMAL1 production.
    Differential degradation rates of Per vs Cry create the delay
    required for sustained oscillation (Goodwin oscillator topology).

All concentrations in nM.  All times in ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_PERIOD_REAL_MS = 86_400_000.0  # 24 hours in milliseconds

# Base angular frequency for a 24h oscillator (rad/ms).
_OMEGA_BASE = _TWO_PI / _PERIOD_REAL_MS  # ~7.27e-8

# TTFL rate constants (per ms at time_scale=1.0, i.e. real-time 24h cycle).
# Tuned via Goodwin oscillator theory: for 4-variable negative feedback,
# the period T ~ 2*pi * sqrt(tau_1 * tau_2 * tau_3 * tau_4) where tau_i
# are the characteristic lifetimes.  We target T = 86.4e6 ms.
#
# With time_scale > 1, all rates are multiplied by time_scale, compressing
# the period proportionally: effective period = 86.4e6 / time_scale ms.

_K_PROD = 9.6e-8           # CLOCK:BMAL1 production rate (/ms)
_K_DEG = 4.0e-8            # CLOCK:BMAL1 degradation rate (/ms)
_K_TRANSCRIPTION = 6.4e-8  # Per/Cry transcription rate (/ms)
_K_DEG_PER = 4.8e-8        # Per degradation rate (/ms) — faster than Cry
_K_DEG_CRY = 2.8e-8        # Cry degradation rate (/ms) — slower creates phase lag
_K_COMPLEX = 1.2e-7        # PER:CRY complex formation rate (/ms)
_K_DISSOC = 3.2e-8         # PER:CRY complex dissociation rate (/ms)


# ---------------------------------------------------------------------------
# MolecularClock — TTFL circadian oscillator
# ---------------------------------------------------------------------------

@dataclass
class MolecularClock:
    """Transcription-Translation Feedback Loop producing ~24h oscillations.

    The system consists of four coupled molecular species:
      - CLOCK_BMAL1: transcriptional activator complex
      - Per: Period protein
      - Cry: Cryptochrome protein
      - PER_CRY: inhibitor complex that represses CLOCK:BMAL1

    ODE system:
      d[CLOCK_BMAL1]/dt = k_prod * (1 - PER_CRY) - k_deg * CLOCK_BMAL1
      d[Per]/dt         = k_transcription * CLOCK_BMAL1 - k_deg_per * Per
      d[Cry]/dt         = k_transcription * CLOCK_BMAL1 - k_deg_cry * Cry
      d[PER_CRY]/dt     = k_complex * Per * Cry - k_dissoc * PER_CRY

    All variables are normalized to [0, 1].

    Args:
        time_scale: Compression factor for the circadian period.
            1.0 = real-time 24h.  1000.0 = period of ~86.4 seconds.
            For practical neural simulations, use 1000-10000.
    """

    # TTFL state variables (normalised 0-1)
    CLOCK_BMAL1: float = 0.8
    Per: float = 0.2
    Cry: float = 0.3
    PER_CRY: float = 0.1

    # Time compression
    time_scale: float = 1.0

    # Rate constants — exposed for experimentation but default-tuned
    k_prod: float = _K_PROD
    k_deg: float = _K_DEG
    k_transcription: float = _K_TRANSCRIPTION
    k_deg_per: float = _K_DEG_PER
    k_deg_cry: float = _K_DEG_CRY
    k_complex: float = _K_COMPLEX
    k_dissoc: float = _K_DISSOC

    # Internal tracking
    _elapsed_ms: float = field(init=False, default=0.0)

    def step(self, dt: float) -> None:
        """Advance the molecular clock by dt milliseconds.

        Uses RK4 integration for stability at large timesteps.
        All rate constants are scaled by self.time_scale.

        Args:
            dt: Timestep in milliseconds.
        """
        self._elapsed_ms += dt

        # Scale rates by time compression factor
        s = self.time_scale

        def derivatives(
            cb: float, per: float, cry: float, pc: float,
        ) -> Tuple[float, float, float, float]:
            d_cb = s * (self.k_prod * (1.0 - pc) - self.k_deg * cb)
            d_per = s * (self.k_transcription * cb - self.k_deg_per * per)
            d_cry = s * (self.k_transcription * cb - self.k_deg_cry * cry)
            d_pc = s * (self.k_complex * per * cry - self.k_dissoc * pc)
            return d_cb, d_per, d_cry, d_pc

        # RK4 integration
        cb, per, cry, pc = self.CLOCK_BMAL1, self.Per, self.Cry, self.PER_CRY

        k1 = derivatives(cb, per, cry, pc)
        cb2 = cb + 0.5 * dt * k1[0]
        per2 = per + 0.5 * dt * k1[1]
        cry2 = cry + 0.5 * dt * k1[2]
        pc2 = pc + 0.5 * dt * k1[3]

        k2 = derivatives(cb2, per2, cry2, pc2)
        cb3 = cb + 0.5 * dt * k2[0]
        per3 = per + 0.5 * dt * k2[1]
        cry3 = cry + 0.5 * dt * k2[2]
        pc3 = pc + 0.5 * dt * k2[3]

        k3 = derivatives(cb3, per3, cry3, pc3)
        cb4 = cb + dt * k3[0]
        per4 = per + dt * k3[1]
        cry4 = cry + dt * k3[2]
        pc4 = pc + dt * k3[3]

        k4 = derivatives(cb4, per4, cry4, pc4)

        self.CLOCK_BMAL1 = cb + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.Per = per + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.Cry = cry + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        self.PER_CRY = pc + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        # Clamp to [0, 1]
        self.CLOCK_BMAL1 = max(0.0, min(1.0, self.CLOCK_BMAL1))
        self.Per = max(0.0, min(1.0, self.Per))
        self.Cry = max(0.0, min(1.0, self.Cry))
        self.PER_CRY = max(0.0, min(1.0, self.PER_CRY))

    @property
    def elapsed_ms(self) -> float:
        """Total elapsed simulation time in milliseconds."""
        return self._elapsed_ms

    @property
    def phase(self) -> float:
        """Current circadian phase in radians [0, 2*pi).

        Derived from the CLOCK:BMAL1 oscillation via atan2 against
        its rate of change.  Phase 0 = peak CLOCK:BMAL1 (subjective noon),
        pi = trough (subjective midnight).
        """
        # Use the activator and inhibitor as a 2D phase portrait
        # Normalise to unit circle: x = CLOCK_BMAL1 - 0.5, y = PER_CRY - 0.5
        x = self.CLOCK_BMAL1 - 0.5
        y = self.PER_CRY - 0.5
        angle = math.atan2(y, x)
        if angle < 0:
            angle += _TWO_PI
        return angle

    @property
    def nt_synthesis_modulation(self) -> float:
        """Circadian modulation factor for neurotransmitter synthesis rates.

        Peaks during subjective day (high CLOCK:BMAL1), troughs at night.
        Range: [0.5, 1.5] centered on 1.0.
        """
        return 0.5 + self.CLOCK_BMAL1  # CLOCK_BMAL1 in [0,1] -> [0.5, 1.5]

    @property
    def receptor_expression_modulation(self) -> float:
        """Circadian modulation factor for gene expression / receptor trafficking.

        Driven by CLOCK:BMAL1 activation of E-box promoter elements.
        Range: [0.5, 1.5] centered on 1.0.
        """
        # Slightly smoothed version — responds to activator but with inertia
        # from the inhibitor complex
        activation = self.CLOCK_BMAL1 * (1.0 - 0.5 * self.PER_CRY)
        return 0.5 + activation  # activation in [0, ~1] -> [0.5, ~1.5]

    @property
    def excitability_modulation(self) -> float:
        """Circadian modulation factor for neuronal firing threshold.

        During subjective day: lower threshold (more excitable).
        During subjective night: higher threshold (less excitable).
        Range: [0.5, 1.5] centered on 1.0.

        This reflects SCN-driven modulation of cortical excitability
        through orexin, histamine, and other wake-promoting signals.
        """
        # High CLOCK:BMAL1 = day = high excitability (modulation > 1)
        # Low CLOCK:BMAL1 = night = low excitability (modulation < 1)
        return 0.5 + self.CLOCK_BMAL1


# ---------------------------------------------------------------------------
# SleepHomeostasis — adenosine-based sleep pressure
# ---------------------------------------------------------------------------

@dataclass
class SleepHomeostasis:
    """Adenosine-mediated sleep homeostasis (Process S).

    During wakefulness, neural activity generates adenosine from ATP
    hydrolysis.  Adenosine accumulates and activates A1 receptors
    (inhibitory, promoting sleep) and A2A receptors (inhibiting
    wake-promoting regions in the basal forebrain).

    During sleep, adenosine is cleared by adenosine deaminase and
    kinases, reducing sleep pressure.

    All concentrations in nM.  All times in ms.
    """

    # Adenosine state
    adenosine_nM: float = 50.0

    # Baseline parameters
    adenosine_baseline: float = 50.0  # nM — minimum/resting level
    adenosine_threshold: float = 200.0  # nM — strong sleep pressure

    # Kinetics
    adenosine_accumulation_rate: float = 0.0002  # nM per ms per unit activity
    adenosine_clearance_rate: float = 0.00005  # first-order decay constant (/ms)
    adenosine_wake_clearance_rate: float = 0.00001  # slower clearance while awake

    # Pharmacological modulation
    caffeine_block: float = 0.0  # [0, 1] — fraction of A2A blocked by caffeine

    @property
    def sleep_pressure(self) -> float:
        """Normalised sleep pressure: adenosine / threshold.

        0.0 = fully rested (adenosine at baseline).
        1.0 = strong sleep pressure (at threshold).
        Can exceed 1.0 for sleep deprivation.
        """
        return max(0.0, (self.adenosine_nM - self.adenosine_baseline)
                   / (self.adenosine_threshold - self.adenosine_baseline))

    @property
    def A1_receptor_activation(self) -> float:
        """A1 adenosine receptor activation via Hill equation.

        A1 receptors mediate direct inhibition of wake-active neurons.
        EC50 ~70 nM, Hill coefficient 1.5 (moderate cooperativity).

        Returns:
            Fractional activation [0, 1].
        """
        ec50 = 70.0  # nM
        n = 1.5
        conc = self.adenosine_nM
        if conc <= 0.0:
            return 0.0
        cn = conc ** n
        return cn / (ec50 ** n + cn)

    @property
    def A2A_receptor_activation(self) -> float:
        """A2A adenosine receptor activation via Hill equation.

        A2A receptors in the basal forebrain inhibit wake-promoting
        cholinergic neurons.  Caffeine acts primarily here.
        EC50 ~150 nM, Hill coefficient 1.2.

        Returns:
            Fractional activation [0, 1], reduced by caffeine_block.
        """
        ec50 = 150.0  # nM
        n = 1.2
        conc = self.adenosine_nM
        if conc <= 0.0:
            return 0.0
        cn = conc ** n
        raw = cn / (ec50 ** n + cn)
        # Caffeine competitively antagonises A2A
        return raw * (1.0 - self.caffeine_block)

    def step(
        self,
        dt: float,
        mean_neural_activity: float = 0.0,
        is_sleeping: bool = False,
    ) -> None:
        """Advance adenosine dynamics by dt milliseconds.

        Args:
            dt: Timestep in ms.
            mean_neural_activity: Average firing rate or activity metric
                (arbitrary units, typically 0-10).  Drives adenosine
                accumulation from ATP hydrolysis.
            is_sleeping: If True, adenosine clears at the faster sleep
                clearance rate.
        """
        if is_sleeping:
            # Sleep: exponential clearance toward baseline
            decay = self.adenosine_clearance_rate
            self.adenosine_nM += dt * (
                -decay * (self.adenosine_nM - self.adenosine_baseline)
            )
        else:
            # Wake: accumulation from activity + slow clearance
            accumulation = self.adenosine_accumulation_rate * mean_neural_activity
            slow_decay = self.adenosine_wake_clearance_rate
            self.adenosine_nM += dt * (
                accumulation
                - slow_decay * (self.adenosine_nM - self.adenosine_baseline)
            )

        # Floor at zero (physically cannot have negative concentration)
        self.adenosine_nM = max(0.0, self.adenosine_nM)


# ---------------------------------------------------------------------------
# CircadianSystem — integrated clock + homeostasis
# ---------------------------------------------------------------------------

@dataclass
class CircadianSystem:
    """Combined circadian clock and sleep homeostasis.

    Integrates the TTFL molecular clock (Process C) with adenosine-based
    sleep homeostasis (Process S) to produce the two-process model of
    sleep-wake regulation (Borbely, 1982).

    Output properties combine both processes to modulate neural network
    behaviour at the molecular level.
    """

    clock: MolecularClock = field(default_factory=MolecularClock)
    homeostasis: SleepHomeostasis = field(default_factory=SleepHomeostasis)

    def step(
        self,
        dt: float,
        mean_activity: float = 0.0,
        is_sleeping: bool = False,
    ) -> None:
        """Advance both circadian clock and sleep homeostasis.

        Args:
            dt: Timestep in milliseconds.
            mean_activity: Mean neural activity level for adenosine
                accumulation (arbitrary units, 0-10).
            is_sleeping: Whether the organism is in a sleep state.
        """
        self.clock.step(dt)
        self.homeostasis.step(dt, mean_activity, is_sleeping)

    # -- Combined output properties --

    @property
    def wake_drive(self) -> float:
        """Net wake-promoting drive combining circadian and homeostatic.

        High circadian activation + low sleep pressure = strong wake drive.
        Low circadian activation + high sleep pressure = sleep onset.

        Returns:
            Value in [0, 1] where 1 = maximal wakefulness.
        """
        circadian = self.clock.CLOCK_BMAL1  # 0-1, peaks during day
        pressure = min(1.0, self.homeostasis.sleep_pressure)
        # Multiplicative interaction: both must cooperate
        return circadian * (1.0 - 0.7 * pressure)

    @property
    def alertness_modulation(self) -> float:
        """Combined alertness factor for neural processing speed.

        Modulates synaptic transmission efficacy and response latency.
        Range: [0.5, 1.5] centered on 1.0.
        """
        return 0.5 + self.wake_drive  # wake_drive in [0,1] -> [0.5, 1.5]

    @property
    def nt_synthesis_modulation(self) -> float:
        """Combined NT synthesis modulation from clock + homeostasis.

        Circadian rhythm modulates baseline synthesis.  High sleep pressure
        slightly reduces synthesis efficiency.
        Range: [0.5, 1.5].
        """
        circadian = self.clock.nt_synthesis_modulation  # [0.5, 1.5]
        # Sleep pressure dampens synthesis slightly (up to 20% at max pressure)
        pressure_factor = 1.0 - 0.2 * min(1.0, self.homeostasis.sleep_pressure)
        return max(0.5, min(1.5, circadian * pressure_factor))

    @property
    def receptor_expression_modulation(self) -> float:
        """Combined receptor expression modulation.

        Delegates to clock — receptor trafficking is primarily circadian.
        Range: [0.5, 1.5].
        """
        return self.clock.receptor_expression_modulation

    @property
    def excitability_modulation(self) -> float:
        """Combined excitability modulation from clock + homeostasis.

        Circadian excitability is reduced by adenosine-mediated inhibition.
        Range: [0.5, 1.5].
        """
        circadian = self.clock.excitability_modulation  # [0.5, 1.5]
        # A1 receptor activation inhibits excitability
        a1_inhibition = self.homeostasis.A1_receptor_activation  # [0, 1]
        inhibition_factor = 1.0 - 0.4 * a1_inhibition  # up to 40% reduction
        return max(0.5, min(1.5, circadian * inhibition_factor))

    @property
    def sleep_pressure(self) -> float:
        """Normalised sleep pressure from homeostasis."""
        return self.homeostasis.sleep_pressure

    @property
    def circadian_phase(self) -> float:
        """Current circadian phase in radians [0, 2*pi)."""
        return self.clock.phase
