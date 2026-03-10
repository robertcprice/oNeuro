"""Digital Virus Models for oNeuro Pharma Platform.

Biophysically-realistic viral infections that interact with:
- Host cell receptors (ACE2, CD4, ICAM-1, etc.)
- Ion channels and membrane dynamics
- Neurotransmitter systems
- Immune response pathways

Each virus has realistic:
- Binding kinetics (kon, koff, KD)
- Replication rates
- Cytopathic effects
- Drug targets and resistance mutations
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ViralFamily(Enum):
    """Virus family classification."""
    ORTHOMYXOVIRIDAE = "orthomyxoviridae"  # Influenza
    PICORNAVIRIDAE = "picornaviridae"       # Rhinovirus, Poliovirus
    CORONAVIRIDAE = "coronaviridae"         # SARS-CoV-2, MERS, Common cold
    RETROVIRIDAE = "retroviridae"           # HIV
    HERPESVIRIDAE = "herpesviridae"         # HSV-1, HSV-2, VZV
    FLAVIVIRIDAE = "flaviviridae"           # Dengue, Zika, HCV
    PARAMYXOVIRIDAE = "paramyxoviridae"     # Measles, Mumps, RSV
    RHABDOVIRIDAE = "rhabdoviridae"         # Rabies


class ViralTarget(Enum):
    """Molecular targets for antiviral drugs."""
    SPIKE_PROTEIN = "spike"
    NEURAMINIDASE = "neuraminidase"
    PROTEASE = "protease"
    POLYMERASE = "polymerase"
    INTEGRASE = "integrase"
    REVERSE_TRANSCRIPTASE = "rt"
    M2_ION_CHANNEL = "m2_channel"
    RECEPTOR_BINDING = "receptor_binding"
    FUSION_PROTEIN = "fusion"


@dataclass
class ViralLoad:
    """Quantitative viral load tracking."""
    copies_per_ml: float = 0.0
    infected_cells: int = 0
    total_cells: int = 1_000_000  # Default 1M cells
    time_hours: float = 0.0

    @property
    def infection_fraction(self) -> float:
        """Fraction of cells infected."""
        return min(1.0, self.infected_cells / max(1, self.total_cells))

    @property
    def log_copies(self) -> float:
        """Log10 viral copies per ml."""
        return math.log10(max(1, self.copies_per_ml))


@dataclass
class ViralKinetics:
    """Viral replication kinetics parameters."""
    doubling_time_hours: float = 6.0
    burst_size: int = 1000  # Virions per infected cell
    eclipse_period_hours: float = 6.0  # Time before virus production
    infectious_period_hours: float = 48.0  # How long cell produces virus
    clearance_rate_per_hour: float = 0.05  # Immune clearance

    # Binding kinetics
    KD_nM: float = 10.0  # Receptor binding affinity
    kon_per_nM_per_hour: float = 0.001  # Association rate
    koff_per_hour: float = 0.01  # Dissociation rate


@dataclass
class DrugResistance:
    """Drug resistance mutations and profiles."""
    drug_name: str
    mutations: List[str]
    fold_resistance: float = 1.0  # 1.0 = no resistance
    cross_resistance: List[str] = field(default_factory=list)


@dataclass
class DigitalVirus(ABC):
    """Abstract base class for digital viral pathogens."""

    name: str
    family: ViralFamily
    genome_type: str  # RNA, DNA, ss, ds
    envelope: bool

    # Viral parameters
    kinetics: ViralKinetics = field(default_factory=ViralKinetics)
    load: ViralLoad = field(default_factory=ViralLoad)

    # Receptor targets (virus: receptor pairs)
    receptors: Dict[str, float] = field(default_factory=dict)  # receptor_name: affinity

    # Drug targets and resistance
    drug_targets: List[ViralTarget] = field(default_factory=list)
    resistance_profile: List[DrugResistance] = field(default_factory=list)

    # State
    _time_infected_hours: float = field(init=False, default=0.0)
    _applied_drugs: Dict[str, Any] = field(init=False, default_factory=dict)

    def infect(self, n_cells: int, initial_copies: float = 1000.0) -> None:
        """Initialize infection."""
        self.load.infected_cells = n_cells
        self.load.copies_per_ml = initial_copies
        self.load.total_cells = max(self.load.total_cells, n_cells * 100)

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance viral dynamics by dt_hours.

        Returns dict with:
        - copies_per_ml: current viral load
        - infected_cells: number of infected cells
        - infection_fraction: fraction of cells infected
        - new_infections: cells newly infected this step
        """
        # Viral production from infected cells
        if self._time_infected_hours > self.kinetics.eclipse_period_hours:
            production_rate = (
                self.load.infected_cells *
                self.kinetics.burst_size /
                self.kinetics.infectious_period_hours
            )
            self.load.copies_per_ml += production_rate * dt_hours

        # New cell infections
        infection_rate = (
            self.load.copies_per_ml *
            self.kinetics.kon_per_nM_per_hour *
            sum(self.receptors.values()) /
            self.kinetics.KD_nM
        )
        new_infections = int(infection_rate * dt_hours)
        available_cells = self.load.total_cells - self.load.infected_cells
        new_infections = min(new_infections, available_cells)
        self.load.infected_cells += new_infections

        # Viral clearance
        clearance = (
            self.load.copies_per_ml *
            self.kinetics.clearance_rate_per_hour *
            dt_hours
        )
        self.load.copies_per_ml = max(0, self.load.copies_per_ml - clearance)

        # Cell death / recovery
        cell_clearance = int(self.load.infected_cells * 0.01 * dt_hours)
        self.load.infected_cells = max(0, self.load.infected_cells - cell_clearance)

        self._time_infected_hours += dt_hours
        self.load.time_hours = self._time_infected_hours

        return {
            "copies_per_ml": self.load.copies_per_ml,
            "log_copies": self.load.log_copies,
            "infected_cells": self.load.infected_cells,
            "infection_fraction": self.load.infection_fraction,
            "new_infections": new_infections,
        }

    def apply_antiviral(self, drug_name: str, efficacy: float) -> None:
        """Apply antiviral drug effect (0-1 scale)."""
        # Reduce replication rate
        self.kinetics.burst_size = int(
            self.kinetics.burst_size * (1.0 - efficacy)
        )
        self._applied_drugs[drug_name] = efficacy

    def check_resistance(self, drug_name: str) -> float:
        """Check fold-resistance to a drug."""
        for res in self.resistance_profile:
            if res.drug_name == drug_name:
                return res.fold_resistance
        return 1.0

    @abstractmethod
    def get_neurological_effects(self) -> Dict[str, float]:
        """Return effects on neural tissue."""
        pass

    @abstractmethod
    def get_cytokine_response(self) -> Dict[str, float]:
        """Return cytokine storm potential."""
        pass


# ============================================================================
# Specific Virus Implementations
# ============================================================================

@dataclass
class InfluenzaA(DigitalVirus):
    """Influenza A virus - seasonal flu and pandemic strains.

    Targets: Neuraminidase, M2 ion channel
    Receptors: Sialic acid (alpha-2,6 in humans, alpha-2,3 in birds)
    """

    name: str = "Influenza A"
    family: ViralFamily = ViralFamily.ORTHOMYXOVIRIDAE
    genome_type: str = "ssRNA(-)"
    envelope: bool = True

    # Strain-specific
    hemagglutinin: str = "H1"  # H1, H3, H5, H7, etc.
    neuraminidase: str = "N1"  # N1, N2, etc.

    kinetics: ViralKinetics = field(default_factory=lambda: ViralKinetics(
        doubling_time_hours=6.0,
        burst_size=10000,
        eclipse_period_hours=8.0,
        KD_nM=50.0,
    ))

    receptors: Dict[str, float] = field(default_factory=lambda: {
        "sialic_acid_a2_6": 1.0,  # Human airway
        "sialic_acid_a2_3": 0.1,  # Avian (rare in humans)
    })

    drug_targets: List[ViralTarget] = field(default_factory=lambda: [
        ViralTarget.NEURAMINIDASE,
        ViralTarget.M2_ION_CHANNEL,
    ])

    def get_neurological_effects(self) -> Dict[str, float]:
        """Influenza neurological complications."""
        base_effects = {
            "fever_celsius": 2.0 + self.load.infection_fraction * 1.5,
            "fatigue": 0.4 + self.load.infection_fraction * 0.4,
            "headache": 0.3 + self.load.infection_fraction * 0.3,
            "myalgia": 0.5 + self.load.infection_fraction * 0.3,
        }
        # Rare encephalitis in severe cases
        if self.load.infection_fraction > 0.3:
            base_effects["encephalitis_risk"] = 0.01
        return base_effects

    def get_cytokine_response(self) -> Dict[str, float]:
        """Influenza cytokine storm potential."""
        frac = self.load.infection_fraction
        return {
            "ifn_alpha": 10.0 + frac * 100.0,
            "ifn_beta": 5.0 + frac * 50.0,
            "il_6": 2.0 + frac * 80.0,
            "tnf_alpha": 1.0 + frac * 30.0,
            "il_1beta": 1.0 + frac * 20.0,
        }


@dataclass
class Rhinovirus(DigitalVirus):
    """Rhinovirus - common cold virus.

    Targets: ICAM-1 receptor, 3C protease
    Most common viral infection in humans.
    """

    name: str = "Rhinovirus"
    family: ViralFamily = ViralFamily.PICORNAVIRIDAE
    genome_type: str = "ssRNA(+)"
    envelope: bool = False  # Naked virus

    serotype: int = 1  # 1-100+ serotypes

    kinetics: ViralKinetics = field(default_factory=lambda: ViralKinetics(
        doubling_time_hours=8.0,
        burst_size=50000,  # High burst for naked virus
        eclipse_period_hours=6.0,
        KD_nM=5.0,  # High affinity for ICAM-1
    ))

    receptors: Dict[str, float] = field(default_factory=lambda: {
        "ICAM_1": 1.0,  # Major group
        "LDLR": 0.3,    # Minor group
    })

    drug_targets: List[ViralTarget] = field(default_factory=lambda: [
        ViralTarget.PROTEASE,  # 3C protease
        ViralTarget.RECEPTOR_BINDING,
    ])

    def get_neurological_effects(self) -> Dict[str, float]:
        """Rhinovirus typically mild, but can exacerbate asthma."""
        frac = self.load.infection_fraction
        return {
            "nasal_congestion": 0.3 + frac * 0.5,
            "sneezing": 0.4 + frac * 0.4,
            "sore_throat": 0.2 + frac * 0.3,
            "cough": 0.2 + frac * 0.4,
            "mild_fatigue": frac * 0.3,
        }

    def get_cytokine_response(self) -> Dict[str, float]:
        """Mild cytokine response."""
        frac = self.load.infection_fraction
        return {
            "ifn_lambda": 5.0 + frac * 30.0,  # Type III IFN
            "il_8": 2.0 + frac * 20.0,
            "il_6": 0.5 + frac * 5.0,
        }


@dataclass
class SARSCoV2(DigitalVirus):
    """SARS-CoV-2 - COVID-19 virus.

    Targets: Spike protein, ACE2 receptor, TMPRSS2, 3CL protease
    Notable for neurological involvement (loss of smell, brain fog)
    """

    name: str = "SARS-CoV-2"
    family: ViralFamily = ViralFamily.CORONAVIRIDAE
    genome_type: str = "ssRNA(+)"
    envelope: bool = True

    variant: str = "Omicron"  # Wuhan, Alpha, Delta, Omicron, etc.

    kinetics: ViralKinetics = field(default_factory=lambda: ViralKinetics(
        doubling_time_hours=12.0,  # Slower than flu
        burst_size=1000,
        eclipse_period_hours=24.0,
        infectious_period_hours=120.0,  # 5 days
        clearance_rate_per_hour=0.02,
        KD_nM=15.0,  # ACE2 affinity
    ))

    receptors: Dict[str, float] = field(default_factory=lambda: {
        "ACE2": 1.0,
        "TMPRSS2": 0.8,  # Protease for spike activation
        "NRP1": 0.3,     # Neuropilin co-receptor
    })

    drug_targets: List[ViralTarget] = field(default_factory=lambda: [
        ViralTarget.SPIKE_PROTEIN,
        ViralTarget.PROTEASE,  # 3CL/Mpro
        ViralTarget.POLYMERASE,  # RdRp
    ])

    def get_neurological_effects(self) -> Dict[str, float]:
        """COVID-19 neurological manifestations."""
        frac = self.load.infection_fraction
        effects = {
            "anosmia": 0.4 + frac * 0.4,  # Loss of smell
            "ageusia": 0.3 + frac * 0.3,  # Loss of taste
            "brain_fog": frac * 0.5,
            "headache": 0.2 + frac * 0.4,
            "fatigue": 0.3 + frac * 0.5,
        }
        # Long COVID risk
        if frac > 0.2:
            effects["long_covid_risk"] = 0.05 + frac * 0.15
        # Severe cases
        if frac > 0.5:
            effects["encephalopathy_risk"] = 0.1
            effects["stroke_risk"] = 0.02
        return effects

    def get_cytokine_response(self) -> Dict[str, float]:
        """COVID-19 cytokine storm potential."""
        frac = self.load.infection_fraction
        return {
            "il_6": 5.0 + frac * 200.0,  # Major cytokine storm driver
            "tnf_alpha": 2.0 + frac * 100.0,
            "il_1beta": 1.0 + frac * 50.0,
            "ifn_gamma": 1.0 + frac * 80.0,
            "il_10": 1.0 + frac * 30.0,  # Anti-inflammatory
            "gm_csf": frac * 40.0,
            "il_8": 2.0 + frac * 60.0,
        }


@dataclass
class HIV1(DigitalVirus):
    """HIV-1 - Human Immunodeficiency Virus.

    Targets: CD4 receptor, CCR5/CXCR4 co-receptors, reverse transcriptase
    Chronic infection with neurological complications (HAND)
    """

    name: str = "HIV-1"
    family: ViralFamily = ViralFamily.RETROVIRIDAE
    genome_type: str = "ssRNA(+)x2"  # Diploid
    envelope: bool = True

    tropism: str = "R5"  # R5 (CCR5), X4 (CXCR4), or R5X4

    kinetics: ViralKinetics = field(default_factory=lambda: ViralKinetics(
        doubling_time_hours=24.0,
        burst_size=10000,
        eclipse_period_hours=24.0,
        infectious_period_hours=72.0,
        clearance_rate_per_hour=0.01,
        KD_nM=5.0,  # High CD4 affinity
    ))

    receptors: Dict[str, float] = field(default_factory=lambda: {
        "CD4": 1.0,
        "CCR5": 0.8,  # R5 tropic
        "CXCR4": 0.2,  # Some X4
    })

    drug_targets: List[ViralTarget] = field(default_factory=lambda: [
        ViralTarget.REVERSE_TRANSCRIPTASE,
        ViralTarget.INTEGRASE,
        ViralTarget.PROTEASE,
        ViralTarget.FUSION_PROTEIN,
        ViralTarget.RECEPTOR_BINDING,  # Entry inhibitors
    ])

    # Common resistance mutations
    resistance_profile: List[DrugResistance] = field(default_factory=lambda: [
        DrugResistance("efavirenz", ["K103N"], fold_resistance=20.0),
        DrugResistance("nevirapine", ["Y181C"], fold_resistance=10.0),
        DrugResistance("azidothymidine", ["M184V"], fold_resistance=100.0),
        DrugResistance("raltegravir", ["Q148H"], fold_resistance=50.0),
    ])

    # Latent reservoir
    _latent_cells: int = field(init=False, default=0)
    _years_infected: float = field(init=False, default=0.0)

    def get_neurological_effects(self) -> Dict[str, float]:
        """HIV-associated neurocognitive disorders (HAND)."""
        frac = self.load.infection_fraction
        years = self._years_infected
        return {
            "cognitive_impairment": min(0.8, frac * 0.3 + years * 0.05),
            "motor_slowing": frac * 0.2 + years * 0.02,
            "peripheral_neuropathy": years * 0.05,
            "cd4_count_reduction": frac * 500.0,  # cells/uL
        }

    def get_cytokine_response(self) -> Dict[str, float]:
        """Chronic immune activation."""
        frac = self.load.infection_fraction
        return {
            "il_6": 2.0 + frac * 20.0,
            "tnf_alpha": 1.0 + frac * 15.0,
            "il_1beta": 0.5 + frac * 10.0,
            "ifn_gamma": 1.0 + frac * 25.0,
            "il_10": 2.0 + frac * 15.0,  # Immunosuppressive
        }


@dataclass
class HerpesSimplex(DigitalVirus):
    """Herpes Simplex Virus - HSV-1 and HSV-2.

    Targets: Nectin-1, HVEM receptors
    Notable for latent infection in sensory ganglia.
    """

    name: str = "Herpes Simplex"
    family: ViralFamily = ViralFamily.HERPESVIRIDAE
    genome_type: str = "dsDNA"
    envelope: bool = True

    hsv_type: int = 1  # 1 = oral, 2 = genital

    kinetics: ViralKinetics = field(default_factory=lambda: ViralKinetics(
        doubling_time_hours=18.0,
        burst_size=1000,
        eclipse_period_hours=12.0,
        KD_nM=10.0,
    ))

    receptors: Dict[str, float] = field(default_factory=lambda: {
        "nectin_1": 1.0,
        "HVEM": 0.7,
        "3_os_hs": 0.5,  # 3-O-sulfated heparan sulfate
    })

    drug_targets: List[ViralTarget] = field(default_factory=lambda: [
        ViralTarget.POLYMERASE,  # Viral DNA polymerase
    ])

    # Latency state
    _latent_in_ganglia: bool = field(init=False, default=False)
    _reactivation_threshold: float = field(init=False, default=0.3)

    def establish_latency(self) -> None:
        """Establish latent infection in sensory ganglia."""
        self._latent_in_ganglia = True
        self.load.infected_cells = max(1, self.load.infected_cells // 1000)

    def reactivate(self, trigger_strength: float = 0.5) -> bool:
        """Attempt reactivation from latency."""
        if not self._latent_in_ganglia:
            return False
        if trigger_strength > self._reactivation_threshold:
            self.load.infected_cells *= 100
            self._latent_in_ganglia = False
            return True
        return False

    def get_neurological_effects(self) -> Dict[str, float]:
        """HSV neurological manifestations."""
        frac = self.load.infection_fraction
        effects = {
            "pain": 0.2 + frac * 0.6,
            "tingling": 0.3 + frac * 0.4,
            "lesion_formation": frac * 0.8,
        }
        # Herpes encephalitis (rare but severe)
        if frac > 0.7:
            effects["encephalitis_risk"] = 0.001
        return effects

    def get_cytokine_response(self) -> Dict[str, float]:
        """HSV cytokine response."""
        frac = self.load.infection_fraction
        return {
            "ifn_alpha": 10.0 + frac * 100.0,
            "ifn_gamma": 5.0 + frac * 50.0,
            "tnf_alpha": 1.0 + frac * 20.0,
            "il_6": 0.5 + frac * 15.0,
        }
