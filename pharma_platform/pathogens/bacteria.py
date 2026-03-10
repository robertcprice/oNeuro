"""Digital Bacterial Pathogen Models for oNeuro Pharma Platform.

Biophysically-realistic bacterial infections that interact with:
- Host immune response
- Antibiotic drug targets
- Quorum sensing and biofilm formation
- Toxin production

Each bacterium has realistic:
- Growth kinetics (doubling time, carrying capacity)
- Antibiotic resistance mechanisms
- Virulence factors
- Quorum sensing thresholds
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BacterialClass(Enum):
    """Bacterial classification."""
    GRAM_POSITIVE = "gram_positive"
    GRAM_NEGATIVE = "gram_negative"
    ACID_FAST = "acid_fast"  # Mycobacteria
    SPIROCHETE = "spirochete"
    MYCOPLASMA = "mycoplasma"


class AntibioticTarget(Enum):
    """Molecular targets for antibiotics."""
    CELL_WALL_SYNTHESIS = "cell_wall"
    PROTEIN_SYNTHESIS_30S = "30s_ribosome"
    PROTEIN_SYNTHESIS_50S = "50s_ribosome"
    DNA_GYRASE = "dna_gyrase"
    RNA_POLYMERASE = "rna_polymerase"
    FOLATE_SYNTHESIS = "folate"
    CELL_MEMBRANE = "cell_membrane"
    MYCOLIC_ACID_SYNTHESIS = "mycolic_acid"  # TB specific


class ResistanceMechanism(Enum):
    """Antibiotic resistance mechanisms."""
    BETA_LACTAMASE = "beta_lactamase"
    EFFLUX_PUMP = "efflux_pump"
    TARGET_MODIFICATION = "target_mod"
    TARGET_PROTECTION = "target_protection"
    ENZYMATIC_INACTIVATION = "enzymatic_inactivation"
    DECREASED_PERMEABILITY = "decreased_permeability"
    BIOFILM_FORMATION = "biofilm"


@dataclass
class BacterialLoad:
    """Quantitative bacterial load tracking."""
    cfu_per_ml: float = 0.0  # Colony forming units
    biofilm_fraction: float = 0.0  # 0-1, fraction in biofilm
    time_hours: float = 0.0

    @property
    def log_cfu(self) -> float:
        """Log10 CFU per ml."""
        return math.log10(max(1, self.cfu_per_ml))

    @property
    def planktonic_cfu(self) -> float:
        """Free-floating bacteria."""
        return self.cfu_per_ml * (1 - self.biofilm_fraction)

    @property
    def biofilm_cfu(self) -> float:
        """Bacteria in biofilm."""
        return self.cfu_per_ml * self.biofilm_fraction


@dataclass
class BacterialKinetics:
    """Bacterial growth parameters."""
    doubling_time_minutes: float = 30.0
    lag_phase_hours: float = 2.0
    carrying_capacity_log_cfu: float = 10.0  # Max log10 CFU/ml
    death_rate_per_hour: float = 0.01

    # Biofilm parameters
    biofilm_formation_rate: float = 0.01
    quorum_threshold_log_cfu: float = 7.0  # Quorum sensing threshold


@dataclass
class AntibioticResistance:
    """Antibiotic resistance profile."""
    antibiotic_class: str
    mic_ug_ml: float  # Minimum inhibitory concentration
    mic_breakpoint_susceptible: float
    mic_breakpoint_resistant: float
    mechanism: ResistanceMechanism
    fold_resistance: float = 1.0

    @property
    def resistance_category(self) -> str:
        """SIR categorization."""
        if self.mic_ug_ml <= self.mic_breakpoint_susceptible:
            return "Susceptible"
        elif self.mic_ug_ml >= self.mic_breakpoint_resistant:
            return "Resistant"
        return "Intermediate"


@dataclass
class Toxin:
    """Bacterial toxin."""
    name: str
    toxin_type: str  # exotoxin, endotoxin, enterotoxin, neurotoxin
    production_rate_per_hour: float = 0.1
    target: str = "host_cells"
    potency_ec50: float = 1.0  # nM


@dataclass
class DigitalBacteria(ABC):
    """Abstract base class for digital bacterial pathogens."""

    name: str
    species: str
    bacterial_class: BacterialClass

    # Growth parameters
    kinetics: BacterialKinetics = field(default_factory=BacterialKinetics)
    load: BacterialLoad = field(default_factory=BacterialLoad)

    # Antibiotic profile
    resistance_profile: List[AntibioticResistance] = field(default_factory=list)
    drug_targets: List[AntibioticTarget] = field(default_factory=list)

    # Virulence
    toxins: List[Toxin] = field(default_factory=list)
    virulence_factors: List[str] = field(default_factory=list)

    # State
    _in_lag_phase: bool = field(init=False, default=True)
    _quorum_activated: bool = field(init=False, default=False)

    def inoculate(self, initial_cfu: float = 1000.0) -> None:
        """Initialize infection."""
        self.load.cfu_per_ml = initial_cfu
        self._in_lag_phase = True

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance bacterial dynamics by dt_hours."""
        self.load.time_hours += dt_hours

        # Exit lag phase
        if self._in_lag_phase and self.load.time_hours > self.kinetics.lag_phase_hours:
            self._in_lag_phase = False

        # Growth (logistic)
        if not self._in_lag_phase:
            growth_rate = math.log(2) / (self.kinetics.doubling_time_minutes / 60)
            current_log = self.load.log_cfu
            capacity_log = self.kinetics.kinetics.carrying_capacity_log_cfu

            # Logistic growth
            if current_log < capacity_log:
                carrying_factor = 1 - (current_log / capacity_log)
                growth = growth_rate * carrying_factor * dt_hours
                self.load.cfu_per_ml *= math.exp(growth)

        # Death rate
        death = math.exp(-self.kinetics.death_rate_per_hour * dt_hours)
        self.load.cfu_per_ml *= death

        # Quorum sensing activation
        if not self._quorum_activated and self.load.log_cfu >= self.kinetics.quorum_threshold_log_cfu:
            self._quorum_activated = True

        # Biofilm formation
        if self._quorum_activated:
            biofilm_increase = self.kinetics.biofilm_formation_rate * dt_hours
            self.load.biofilm_fraction = min(0.9, self.load.biofilm_fraction + biofilm_increase)

        return {
            "cfu_per_ml": self.load.cfu_per_ml,
            "log_cfu": self.load.log_cfu,
            "biofilm_fraction": self.load.biofilm_fraction,
            "quorum_activated": self._quorum_activated,
            "in_lag_phase": self._in_lag_phase,
        }

    def apply_antibiotic(self, antibiotic_class: str, concentration_ug_ml: float) -> float:
        """Apply antibiotic and return kill fraction.

        Returns fraction of bacteria killed (0-1).
        """
        # Find resistance profile
        resistance = None
        for r in self.resistance_profile:
            if r.antibiotic_class == antibiotic_class:
                resistance = r
                break

        if resistance is None:
            # Susceptible - assume MIC = 1 ug/ml
            mic = 1.0
        else:
            mic = resistance.mic_ug_ml

        # Concentration-dependent killing (Hill equation)
        effect = concentration_ug_ml ** 2 / (mic ** 2 + concentration_ug_ml ** 2)

        # Biofilm protection
        if self.load.biofilm_fraction > 0.5:
            effect *= 0.3  # Biofilm reduces antibiotic efficacy by 70%

        # Apply killing
        kill_fraction = effect * 0.99  # Max 99% kill per application
        self.load.cfu_per_ml *= (1 - kill_fraction)

        return kill_fraction

    def get_cytokine_response(self) -> Dict[str, float]:
        """Host cytokine response to infection."""
        log_cfu = self.load.log_cfu
        frac = min(1.0, log_cfu / 10.0)

        # Base inflammatory response
        cytokines = {
            "il_6": 1.0 + frac * 100.0,
            "tnf_alpha": 0.5 + frac * 80.0,
            "il_1beta": 0.5 + frac * 50.0,
            "il_8": 1.0 + frac * 200.0,  # Neutrophil chemoattractant
            "il_10": 0.5 + frac * 20.0,  # Anti-inflammatory
        }

        # Gram-negative adds endotoxin response
        if self.bacterial_class == BacterialClass.GRAM_NEGATIVE:
            cytokines["il_6"] *= 2.0
            cytokines["tnf_alpha"] *= 2.5

        return cytokines


@dataclass
class EColi(DigitalBacteria):
    """Escherichia coli - Gram-negative, common UTI/pathogen.

    Notable for: Beta-lactamase production, ESBL strains.
    """

    name: str = "E. coli"
    species: str = "Escherichia coli"
    bacterial_class: BacterialClass = BacterialClass.GRAM_NEGATIVE

    kinetics: BacterialKinetics = field(default_factory=lambda: BacterialKinetics(
        doubling_time_minutes=20.0,
        lag_phase_hours=1.0,
        carrying_capacity_log_cfu=10.0,
        quorum_threshold_log_cfu=7.0,
        biofilm_formation_rate=0.005,
    ))

    drug_targets: List[AntibioticTarget] = field(default_factory=lambda: [
        AntibioticTarget.CELL_WALL_SYNTHESIS,
        AntibioticTarget.PROTEIN_SYNTHESIS_30S,
        AntibioticTarget.PROTEIN_SYNTHESIS_50S,
        AntibioticTarget.DNA_GYRASE,
        AntibioticTarget.FOLATE_SYNTHESIS,
    ])

    # Common resistance (ESBL strain)
    resistance_profile: List[AntibioticResistance] = field(default_factory=lambda: [
        AntibioticResistance(
            "beta_lactam", mic_ug_ml=64.0,
            mic_breakpoint_susceptible=8.0, mic_breakpoint_resistant=16.0,
            mechanism=ResistanceMechanism.BETA_LACTAMASE, fold_resistance=8.0
        ),
        AntibioticResistance(
            "fluoroquinolone", mic_ug_ml=4.0,
            mic_breakpoint_susceptible=0.5, mic_breakpoint_resistant=2.0,
            mechanism=ResistanceMechanism.TARGET_MODIFICATION, fold_resistance=8.0
        ),
    ])

    toxins: List[Toxin] = field(default_factory=lambda: [
        Toxin("Shiga toxin", "exotoxin", 0.1, "endothelial_cells", 0.01),
        Toxin("LPS", "endotoxin", 1.0, "immune_cells", 10.0),
    ])


@dataclass
class Staphylococcus(DigitalBacteria):
    """Staphylococcus aureus - Gram-positive, major pathogen.

    Notable for: MRSA, multiple resistance mechanisms.
    """

    name: str = "S. aureus"
    species: str = "Staphylococcus aureus"
    bacterial_class: BacterialClass = BacterialClass.GRAM_POSITIVE

    is_mrsa: bool = True

    kinetics: BacterialKinetics = field(default_factory=lambda: BacterialKinetics(
        doubling_time_minutes=30.0,
        lag_phase_hours=2.0,
        carrying_capacity_log_cfu=9.5,
        quorum_threshold_log_cfu=6.0,
        biofilm_formation_rate=0.02,
    ))

    drug_targets: List[AntibioticTarget] = field(default_factory=lambda: [
        AntibioticTarget.CELL_WALL_SYNTHESIS,
        AntibioticTarget.PROTEIN_SYNTHESIS_50S,
        AntibioticTarget.DNA_GYRASE,
        AntibioticTarget.FOLATE_SYNTHESIS,
    ])

    resistance_profile: List[AntibioticResistance] = field(default_factory=lambda: [
        # MRSA: methicillin/oxacillin resistance via PBP2a
        AntibioticResistance(
            "beta_lactam", mic_ug_ml=32.0,
            mic_breakpoint_susceptible=2.0, mic_breakpoint_resistant=4.0,
            mechanism=ResistanceMechanism.TARGET_PROTECTION, fold_resistance=16.0
        ),
        AntibioticResistance(
            "macrolide", mic_ug_ml=16.0,
            mic_breakpoint_susceptible=2.0, mic_breakpoint_resistant=8.0,
            mechanism=ResistanceMechanism.TARGET_MODIFICATION, fold_resistance=8.0
        ),
    ])

    toxins: List[Toxin] = field(default_factory=lambda: [
        Toxin("TSST-1", "exotoxin", 0.05, "t_cells", 0.001),  # Toxic shock
        Toxin("alpha-hemolysin", "exotoxin", 0.2, "erythrocytes", 0.1),
        Toxin("PVL", "exotoxin", 0.1, "neutrophils", 0.05),
    ])

    virulence_factors: List[str] = field(default_factory=lambda: [
        "protein_a",  # Binds Fc region of IgG
        "coagulase",  # Clot formation
        "clumping_factor",  # Fibrinogen binding
        "capsule",  # Anti-phagocytic
    ])


@dataclass
class Streptococcus(DigitalBacteria):
    """Streptococcus pneumoniae - Gram-positive, respiratory pathogen.

    Notable for: Capsule serotypes, penicillin resistance.
    """

    name: str = "S. pneumoniae"
    species: str = "Streptococcus pneumoniae"
    bacterial_class: BacterialClass = BacterialClass.GRAM_POSITIVE

    serotype: int = 19  # Common serotype

    kinetics: BacterialKinetics = field(default_factory=lambda: BacterialKinetics(
        doubling_time_minutes=40.0,
        lag_phase_hours=3.0,
        carrying_capacity_log_cfu=8.5,
        quorum_threshold_log_cfu=5.0,
        biofilm_formation_rate=0.01,
    ))

    drug_targets: List[AntibioticTarget] = field(default_factory=lambda: [
        AntibioticTarget.CELL_WALL_SYNTHESIS,
        AntibioticTarget.PROTEIN_SYNTHESIS_50S,
        AntibioticTarget.FOLATE_SYNTHESIS,
    ])

    resistance_profile: List[AntibioticResistance] = field(default_factory=lambda: [
        AntibioticResistance(
            "beta_lactam", mic_ug_ml=4.0,
            mic_breakpoint_susceptible=0.06, mic_breakpoint_resistant=2.0,
            mechanism=ResistanceMechanism.TARGET_MODIFICATION, fold_resistance=67.0
        ),
        AntibioticResistance(
            "macrolide", mic_ug_ml=8.0,
            mic_breakpoint_susceptible=0.25, mic_breakpoint_resistant=1.0,
            mechanism=ResistanceMechanism.EFFLUX_PUMP, fold_resistance=32.0
        ),
    ])

    toxins: List[Toxin] = field(default_factory=lambda: [
        Toxin("pneumolysin", "exotoxin", 0.3, "host_cells", 0.5),
    ])

    virulence_factors: List[str] = field(default_factory=lambda: [
        "capsule",
        "pneumolysin",
        "iga_protease",
        "neuraminidase",
    ])


@dataclass
class Pseudomonas(DigitalBacteria):
    """Pseudomonas aeruginosa - Gram-negative, opportunistic pathogen.

    Notable for: Intrinsic resistance, biofilm, CF lung infections.
    """

    name: str = "P. aeruginosa"
    species: str = "Pseudomonas aeruginosa"
    bacterial_class: BacterialClass = BacterialClass.GRAM_NEGATIVE

    kinetics: BacterialKinetics = field(default_factory=lambda: BacterialKinetics(
        doubling_time_minutes=45.0,
        lag_phase_hours=2.0,
        carrying_capacity_log_cfu=9.0,
        quorum_threshold_log_cfu=6.5,
        biofilm_formation_rate=0.05,  # Strong biofilm former
    ))

    drug_targets: List[AntibioticTarget] = field(default_factory=lambda: [
        AntibioticTarget.CELL_WALL_SYNTHESIS,
        AntibioticTarget.PROTEIN_SYNTHESIS_30S,
        AntibioticTarget.PROTEIN_SYNTHESIS_50S,
        AntibioticTarget.DNA_GYRASE,
    ])

    resistance_profile: List[AntibioticResistance] = field(default_factory=lambda: [
        AntibioticResistance(
            "beta_lactam", mic_ug_ml=16.0,
            mic_breakpoint_susceptible=8.0, mic_breakpoint_resistant=16.0,
            mechanism=ResistanceMechanism.EFFLUX_PUMP, fold_resistance=2.0
        ),
        AntibioticResistance(
            "aminoglycoside", mic_ug_ml=8.0,
            mic_breakpoint_susceptible=4.0, mic_breakpoint_resistant=16.0,
            mechanism=ResistanceMechanism.ENZYMATIC_INACTIVATION, fold_resistance=2.0
        ),
        # Intrinsic: decreased permeability
        AntibioticResistance(
            "multiple", mic_ug_ml=32.0,
            mic_breakpoint_susceptible=4.0, mic_breakpoint_resistant=16.0,
            mechanism=ResistanceMechanism.DECREASED_PERMEABILITY, fold_resistance=8.0
        ),
    ])

    toxins: List[Toxin] = field(default_factory=lambda: [
        Toxin("exotoxin_a", "exotoxin", 0.1, "host_cells", 0.01),
        Toxin("pyocyanin", "exotoxin", 0.2, "host_cells", 1.0),
        Toxin("elastase", "exotoxin", 0.3, "tissue", 5.0),
    ])

    virulence_factors: List[str] = field(default_factory=lambda: [
        "flagellum",
        "pili",
        "alginate",  # Mucoid biofilm in CF
        "type_iii_secretion",
        "quorum_sensing_las_rhl",
    ])


@dataclass
class MycobacteriumTB(DigitalBacteria):
    """Mycobacterium tuberculosis - Acid-fast, TB pathogen.

    Notable for: Slow growth, cell wall mycolic acids, persistence.
    """

    name: str = "M. tuberculosis"
    species: str = "Mycobacterium tuberculosis"
    bacterial_class: BacterialClass = BacterialClass.ACID_FAST

    is_mdr: bool = False  # Multi-drug resistant
    is_xdr: bool = False  # Extensively drug resistant

    kinetics: BacterialKinetics = field(default_factory=lambda: BacterialKinetics(
        doubling_time_hours=24.0,  # Very slow growth!
        lag_phase_hours=72.0,
        carrying_capacity_log_cfu=8.0,
        quorum_threshold_log_cfu=5.0,
        biofilm_formation_rate=0.001,
        death_rate_per_hour=0.001,  # Very persistent
    ))

    drug_targets: List[AntibioticTarget] = field(default_factory=lambda: [
        AntibioticTarget.MYCOLIC_ACID_SYNTHESIS,
        AntibioticTarget.RNA_POLYMERASE,
        AntibioticTarget.DNA_GYRASE,
        AntibioticTarget.PROTEIN_SYNTHESIS_30S,
    ])

    resistance_profile: List[AntibioticResistance] = field(default_factory=lambda: [
        AntibioticResistance(
            "rifampin", mic_ug_ml=1.0,
            mic_breakpoint_susceptible=1.0, mic_breakpoint_resistant=1.0,
            mechanism=ResistanceMechanism.TARGET_MODIFICATION, fold_resistance=1.0
        ),
        AntibioticResistance(
            "isoniazid", mic_ug_ml=0.1,
            mic_breakpoint_susceptible=0.1, mic_breakpoint_resistant=0.2,
            mechanism=ResistanceMechanism.ENZYMATIC_INACTIVATION, fold_resistance=1.0
        ),
    ])

    # Latent/persistent state
    _latent: bool = field(init=False, default=False)
    _granuloma_formation: float = field(init=False, default=0.0)

    def establish_latency(self) -> None:
        """Enter latent/persistent state."""
        self._latent = True
        self.kinetics.doubling_time_minutes *= 1000  # Extremely slow

    def get_granuloma_response(self) -> Dict[str, float]:
        """Granuloma formation response."""
        log_cfu = self.load.log_cfu
        return {
            "granuloma_size_mm": log_cfu * 0.5,
            "macrophage_infiltration": min(1.0, log_cfu / 8.0),
            "caseation_necrosis": min(0.8, (log_cfu - 5.0) * 0.1) if log_cfu > 5.0 else 0.0,
            "fibrosis": min(0.6, self._granuloma_formation),
        }
