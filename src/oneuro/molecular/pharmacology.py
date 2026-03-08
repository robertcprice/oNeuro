"""Computational psychopharmacology engine.

Seven drugs modeled with real pharmacokinetic (1-compartment) and
pharmacodynamic (Hill equation) parameters from clinical literature.
Each drug targets specific molecular components that already exist in the
oNeuro molecular layer — no modifications to existing files needed.

Usage:
    from oneuro.molecular.pharmacology import DRUG_LIBRARY, DrugCocktail

    ssri = DRUG_LIBRARY["fluoxetine"](dose_mg=20.0)
    ssri.apply(network)
    # ... run simulation ...
    ssri.remove(network)

The save/restore pattern means drugs are fully reversible: apply() caches
original values, remove() restores them exactly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from oneuro.molecular.ion_channels import IonChannelType


# ---------------------------------------------------------------------------
# Pharmacokinetic helpers
# ---------------------------------------------------------------------------

def _one_compartment_concentration(
    dose_mg: float,
    bioavailability: float,
    volume_of_distribution_L: float,
    tmax_hours: float,
    half_life_hours: float,
    time_hours: float,
) -> float:
    """1-compartment oral PK model → plasma concentration in nM.

    Uses Bateman function: C(t) = F·D·ka / (V·(ka-ke)) · (e^{-ke·t} - e^{-ka·t})
    Simplified to peak-normalised form for robustness.
    """
    if time_hours <= 0:
        return 0.0
    ke = math.log(2) / half_life_hours
    # Absorption rate estimated from tmax: ka ≈ 1 / tmax (simplified)
    ka = 1.0 / max(tmax_hours, 0.01)
    if abs(ka - ke) < 1e-10:
        ka += 0.01

    raw = (math.exp(-ke * time_hours) - math.exp(-ka * time_hours)) / (ka - ke)
    # Peak value of Bateman function
    t_peak = math.log(ka / ke) / (ka - ke)
    peak_raw = (math.exp(-ke * t_peak) - math.exp(-ka * t_peak)) / (ka - ke)
    if peak_raw <= 0:
        return 0.0

    # Cmax estimation: dose (mg) → rough nM via molecular weight ~300 Da, Vd
    cmax_nM = (dose_mg * bioavailability * 1e6) / (volume_of_distribution_L * 300.0)
    return cmax_nM * (raw / peak_raw)


def _hill_equation(concentration: float, EC50: float, hill_n: float) -> float:
    """Sigmoidal dose-response: E = C^n / (EC50^n + C^n)."""
    if concentration <= 0 or EC50 <= 0:
        return 0.0
    cn = concentration ** hill_n
    return cn / (EC50 ** hill_n + cn)


# ---------------------------------------------------------------------------
# Drug base class
# ---------------------------------------------------------------------------

@dataclass
class Drug(ABC):
    """Abstract pharmacological agent with PK/PD modelling.

    Subclasses implement apply() and remove() to target specific molecular
    components in a MolecularNeuralNetwork.
    """

    name: str
    generic_name: str
    drug_class: str
    dose_mg: float
    EC50_nM: float
    hill_coefficient: float = 1.0
    half_life_hours: float = 24.0
    tmax_hours: float = 2.0
    bioavailability: float = 0.7
    volume_of_distribution_L: float = 30.0

    # Runtime state
    _time_hours: float = field(init=False, default=0.0)
    _applied: bool = field(init=False, default=False)
    _saved_state: Dict[int, Any] = field(init=False, default_factory=dict)

    def plasma_concentration(self, time_hours: Optional[float] = None) -> float:
        """Current plasma concentration in nM."""
        t = time_hours if time_hours is not None else self._time_hours
        return _one_compartment_concentration(
            self.dose_mg, self.bioavailability,
            self.volume_of_distribution_L, self.tmax_hours,
            self.half_life_hours, t,
        )

    def effect_strength(self, concentration_nM: Optional[float] = None) -> float:
        """Fractional effect [0, 1] via Hill equation."""
        conc = concentration_nM if concentration_nM is not None else self.plasma_concentration()
        return _hill_equation(conc, self.EC50_nM, self.hill_coefficient)

    def update_pk(self, dt_hours: float) -> float:
        """Advance pharmacokinetics by dt_hours. Returns new effect strength."""
        self._time_hours += dt_hours
        return self.effect_strength()

    @abstractmethod
    def apply(self, network) -> None:
        """Apply drug effects to network. Saves original values for restoration."""
        ...

    @abstractmethod
    def remove(self, network) -> None:
        """Remove drug effects, restoring original values."""
        ...

    @property
    def is_applied(self) -> bool:
        return self._applied


# ---------------------------------------------------------------------------
# 7 Drug implementations
# ---------------------------------------------------------------------------

@dataclass
class Fluoxetine(Drug):
    """SSRI — blocks serotonin reuptake transporter (SERT).

    Mechanism: Reduces _reuptake_rate on serotonin synapses.
    Clinical: Prozac, first-line antidepressant.
    """

    name: str = "Fluoxetine"
    generic_name: str = "fluoxetine"
    drug_class: str = "SSRI"
    dose_mg: float = 20.0
    EC50_nM: float = 0.8  # Ki = 0.8 nM (very potent SERT blocker)
    hill_coefficient: float = 1.0
    half_life_hours: float = 72.0  # 1-3 days (longest SSRI)
    tmax_hours: float = 6.0
    bioavailability: float = 0.72
    volume_of_distribution_L: float = 35.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for key, syn in network._molecular_synapses.items():
            if syn.nt_name == "serotonin":
                obj_id = id(syn.cleft)
                self._saved_state[obj_id] = syn.cleft._reuptake_rate
                syn.cleft._reuptake_rate *= (1.0 - 0.85 * effect)
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "serotonin":
                obj_id = id(syn.cleft)
                if obj_id in self._saved_state:
                    syn.cleft._reuptake_rate = self._saved_state[obj_id]
        self._saved_state.clear()
        self._applied = False


@dataclass
class Diazepam(Drug):
    """Benzodiazepine — positive allosteric modulator of GABA-A.

    Mechanism: Enhances GABA-A conductance via conductance_scale (PAM).
    Clinical: Valium, anxiolytic/sedative.
    """

    name: str = "Diazepam"
    generic_name: str = "diazepam"
    drug_class: str = "Benzodiazepine"
    dose_mg: float = 5.0
    EC50_nM: float = 20.0
    hill_coefficient: float = 1.5
    half_life_hours: float = 40.0
    tmax_hours: float = 1.0
    bioavailability: float = 0.95
    volume_of_distribution_L: float = 70.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.GABA_A)
            if ch is not None:
                obj_id = id(ch)
                self._saved_state[obj_id] = ch.conductance_scale
                # PAM: enhance existing GABA-A conductance up to 2x at full effect
                ch.conductance_scale *= (1.0 + 1.0 * effect)
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.GABA_A)
            if ch is not None:
                obj_id = id(ch)
                if obj_id in self._saved_state:
                    ch.conductance_scale = self._saved_state[obj_id]
        self._saved_state.clear()
        self._applied = False


@dataclass
class Caffeine(Drug):
    """Adenosine antagonist — reduces inhibition, increases arousal.

    Mechanism: Reduces GABA-A conductance + increases global norepinephrine.
    Clinical: Most widely consumed psychoactive substance.
    """

    name: str = "Caffeine"
    generic_name: str = "caffeine"
    drug_class: str = "Adenosine antagonist"
    dose_mg: float = 100.0
    EC50_nM: float = 40000.0  # ~40 µM (low potency, high dose)
    hill_coefficient: float = 1.0
    half_life_hours: float = 5.0
    tmax_hours: float = 0.75
    bioavailability: float = 0.99
    volume_of_distribution_L: float = 37.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        # Reduce GABA-A conductance (adenosine normally enhances inhibition)
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.GABA_A)
            if ch is not None:
                obj_id = id(ch)
                self._saved_state[obj_id] = ("gaba_scale", ch.conductance_scale)
                ch.conductance_scale *= (1.0 - 0.4 * effect)
        # Increase global NE
        self._saved_state["global_ne"] = network.global_nt_concentrations.get(
            "norepinephrine", 15.0
        )
        network.global_nt_concentrations["norepinephrine"] = (
            self._saved_state["global_ne"] * (1.0 + 2.0 * effect)
        )
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.GABA_A)
            if ch is not None:
                obj_id = id(ch)
                if obj_id in self._saved_state:
                    _, scale = self._saved_state[obj_id]
                    ch.conductance_scale = scale
        if "global_ne" in self._saved_state:
            network.global_nt_concentrations["norepinephrine"] = self._saved_state["global_ne"]
        self._saved_state.clear()
        self._applied = False


@dataclass
class Amphetamine(Drug):
    """Stimulant — reverses DAT (dopamine transporter) + enhances release.

    Mechanism: Reduces DA reuptake rate + increases vesicle release probability.
    Clinical: Adderall component, ADHD treatment.
    """

    name: str = "Amphetamine"
    generic_name: str = "amphetamine"
    drug_class: str = "Stimulant"
    dose_mg: float = 10.0
    EC50_nM: float = 70.0
    hill_coefficient: float = 1.2
    half_life_hours: float = 10.0
    tmax_hours: float = 3.0
    bioavailability: float = 0.75
    volume_of_distribution_L: float = 20.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for key, syn in network._molecular_synapses.items():
            if syn.nt_name == "dopamine":
                obj_id = id(syn)
                self._saved_state[obj_id] = (
                    syn.cleft._reuptake_rate,
                    syn.vesicle_pool.base_release_prob,
                )
                syn.cleft._reuptake_rate *= (1.0 - 0.7 * effect)
                syn.vesicle_pool.base_release_prob = min(
                    0.9, syn.vesicle_pool.base_release_prob * (1.0 + 1.5 * effect)
                )
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "dopamine":
                obj_id = id(syn)
                if obj_id in self._saved_state:
                    reuptake, release = self._saved_state[obj_id]
                    syn.cleft._reuptake_rate = reuptake
                    syn.vesicle_pool.base_release_prob = release
        self._saved_state.clear()
        self._applied = False


@dataclass
class LDOPA(Drug):
    """Dopamine precursor — increases DA synthesis.

    Mechanism: Increases vesicle NT content on DA synapses.
    Clinical: Gold standard for Parkinson's disease.
    """

    name: str = "L-DOPA"
    generic_name: str = "levodopa"
    drug_class: str = "DA precursor"
    dose_mg: float = 100.0
    EC50_nM: float = 1000.0
    hill_coefficient: float = 1.0
    half_life_hours: float = 1.5
    tmax_hours: float = 1.0
    bioavailability: float = 0.30  # Low without carbidopa
    volume_of_distribution_L: float = 50.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "dopamine":
                obj_id = id(syn.vesicle_pool)
                self._saved_state[obj_id] = syn.vesicle_pool.nt_per_vesicle_nM
                syn.vesicle_pool.nt_per_vesicle_nM *= (1.0 + 3.0 * effect)
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "dopamine":
                obj_id = id(syn.vesicle_pool)
                if obj_id in self._saved_state:
                    syn.vesicle_pool.nt_per_vesicle_nM = self._saved_state[obj_id]
        self._saved_state.clear()
        self._applied = False


@dataclass
class Donepezil(Drug):
    """AChE inhibitor — blocks acetylcholinesterase.

    Mechanism: Reduces AChE enzyme concentration in ACh synapses.
    Clinical: Aricept, Alzheimer's disease treatment.
    """

    name: str = "Donepezil"
    generic_name: str = "donepezil"
    drug_class: str = "AChE inhibitor"
    dose_mg: float = 10.0
    EC50_nM: float = 6.7  # IC50 = 6.7 nM
    hill_coefficient: float = 1.0
    half_life_hours: float = 70.0
    tmax_hours: float = 4.0
    bioavailability: float = 1.0  # ~100% oral
    volume_of_distribution_L: float = 800.0  # Very large Vd

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "acetylcholine":
                for enzyme in syn.cleft.enzymes:
                    if enzyme.name == "AChE":
                        obj_id = id(enzyme)
                        self._saved_state[obj_id] = enzyme.concentration
                        enzyme.concentration *= (1.0 - 0.8 * effect)
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for syn in network._molecular_synapses.values():
            if syn.nt_name == "acetylcholine":
                for enzyme in syn.cleft.enzymes:
                    if enzyme.name == "AChE":
                        obj_id = id(enzyme)
                        if obj_id in self._saved_state:
                            enzyme.concentration = self._saved_state[obj_id]
        self._saved_state.clear()
        self._applied = False


@dataclass
class Ketamine(Drug):
    """NMDA antagonist — blocks NMDA receptor channels.

    Mechanism: Reduces NMDA conductance via open-channel block.
    Clinical: Anaesthetic, rapid-acting antidepressant at sub-anaesthetic doses.
    """

    name: str = "Ketamine"
    generic_name: str = "ketamine"
    drug_class: str = "NMDA antagonist"
    dose_mg: float = 35.0  # Sub-anaesthetic ~0.5 mg/kg for 70kg person
    EC50_nM: float = 500.0  # Ki ≈ 500 nM
    hill_coefficient: float = 1.2
    half_life_hours: float = 2.5
    tmax_hours: float = 0.25  # IV: minutes; IM: ~15 min
    bioavailability: float = 0.93  # IM
    volume_of_distribution_L: float = 200.0

    def apply(self, network) -> None:
        if self._applied:
            return
        effect = self.effect_strength(self.plasma_concentration(self.tmax_hours))
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
            if ch is not None:
                obj_id = id(ch)
                self._saved_state[obj_id] = ch.conductance_scale
                # Open-channel block: reduce conductance up to 80% at full effect
                ch.conductance_scale *= (1.0 - 0.8 * effect)
        self._applied = True

    def remove(self, network) -> None:
        if not self._applied:
            return
        for neuron in network._molecular_neurons.values():
            ch = neuron.membrane.channels.get_channel(IonChannelType.NMDA)
            if ch is not None:
                obj_id = id(ch)
                if obj_id in self._saved_state:
                    ch.conductance_scale = self._saved_state[obj_id]
        self._saved_state.clear()
        self._applied = False


# ---------------------------------------------------------------------------
# Drug Cocktail (polypharmacy)
# ---------------------------------------------------------------------------

@dataclass
class DrugCocktail:
    """Multiple drugs applied simultaneously.

    Handles apply/remove ordering and interaction reporting.
    """

    drugs: List[Drug] = field(default_factory=list)

    def add(self, drug: Drug) -> None:
        self.drugs.append(drug)

    def apply(self, network) -> None:
        for drug in self.drugs:
            drug.apply(network)

    def remove(self, network) -> None:
        for drug in reversed(self.drugs):
            drug.remove(network)

    def summary(self) -> str:
        lines = [f"DrugCocktail ({len(self.drugs)} drugs):"]
        for d in self.drugs:
            conc = d.plasma_concentration(d.tmax_hours)
            eff = d.effect_strength(conc)
            lines.append(f"  {d.name} {d.dose_mg}mg — Cmax={conc:.1f}nM, Emax={eff:.2%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drug library
# ---------------------------------------------------------------------------

DRUG_LIBRARY: Dict[str, Type[Drug]] = {
    "fluoxetine": Fluoxetine,
    "diazepam": Diazepam,
    "caffeine": Caffeine,
    "amphetamine": Amphetamine,
    "l-dopa": LDOPA,
    "donepezil": Donepezil,
    "ketamine": Ketamine,
}
