"""Extended Drug Library for oNeuro Pharma Platform.

This module provides pharmacological agents beyond the base oNeuro
pharmacology system, specifically targeting:
- Bacterial infections (antibiotics)
- Viral infections (antivirals)
- Parasitic infections (antiparasitics)
- Fungal infections (antifungals)
- Inflammatory conditions (anti-inflammatory)

- Prion disease (experimental)

Each drug has:
- Mechanism of action
- PK/PD parameters
- Drug interactions
- Resistance mechanisms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import math
import numpy as np


class DrugCategory(Enum):
    ANTIBIOTIC = "antibiotic"
    ANTIVIRAL = "antiviral"
    ANTIPARASITIC = "antiparasitic"
    ANTIFUNGAL = "antifungal"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    IMMUNOMODULATOR = "immunomodulator"
    CHEMOTHERAPEUTIC = "chemotherapy"
    EXPERIMENTAL = "experimental"


class BactericidalType(Enum):
    TIME_DEPENDENT = "time_dependent"
    CONCENTRATION_DEPENDENT = "concentration_dependent"


# =============================================================================
# Antibiotics
# =============================================================================

@dataclass
class Antibiotic:
    """Base class for antibiotics."""
    name: str
    generic_name: str
    drug_class: str
    category: DrugCategory = DrugCategory.ANTIBIOTIC

    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=list)
    mic_ug_ml: float = 1.0
    protein_binding_percent: float = 80.0
    half_life_hours: float = 1.0
    protein_binding: bool = True
    dose_mg: float = 500.0
    _bound_state: Dict[str, Any] = field(init=False, default_factory=dict)

    def get_kill_fraction(self, concentration: float) -> float:
        """Calculate bacterial kill fraction (0-1)."""
        if self.bactericidal_type == BactericidalType.TIME_DEPENDENT:
            # Time-dependent: fraction of dosing interval
            # AUC/MIC ratio determines effect
            ratio = concentration / self.mic_ug_ml
            return min(1.0, ratio / (ratio + 2))
        else:  # Concentration-dependent
            hill_n = 2.0
            effect = concentration ** hill_n / (
                (self.mic_ug_ml ** hill_n) + concentration ** hill_n
            )
            return effect / (1 + effect ** hill_n)


# Beta-lactams
@dataclass
class Amoxicillin(Antibiotic):
    """Penicillin-class antibiotic (beta-lactam)."""
    name: str = "Amoxicillin"
    generic_name: str = "amoxicillin"
    drug_class: str = "penicillin"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["PBPs", "transpeptidoglycan"])
    mic_ug_ml: float = 2.0
    half_life_hours: float = 1.0
    dose_mg: float = 500.0


@dataclass
class Piperacillin(Antibiotic):
    """Extended-spectrum penicillin."""
    name: str = "Piperacillin"
    generic_name: str = "piperacillin/tazobactam"
    drug_class: str = "penicillin"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["PBPs", "transpeptidoglycan"])
    mic_ug_ml: float = 4.0
    half_life_hours: float = 0.7


@dataclass
class Ceftriaxone(Antibiotic):
    """Third-generation cephalosporin."""
    name: str = "Ceftriaxone"
    generic_name: str = "ceftriaxone"
    drug_class: str = "cephalosporin"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["PBPs"])
    mic_ug_ml: float = 0.0
    half_life_hours: float = 2.0
    protein_binding: bool = True
    dose_mg: float = 1000.0


@dataclass
class Meropenem(Antibiotic):
    """Carbapenem - beta-lactamase resistant."""
    name: str = "Meropenem"
    generic_name: str = "meropenem"
    drug_class: str = "carbapenem"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["PBPs"])
    mic_ug_ml: float = 2.0
    half_life_hours: float = 0.0
    protein_binding: bool = False
    dose_mg: float = 1000.0


# Glycopeptides / Lipopeptides
@dataclass
class Vancomycin(Antibiotic):
    """Glycopeptide antibiotic."""
    name: str = "Vancomycin"
    generic_name: str = "vancomycin"
    drug_class: str = "glycopeptide"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["cell_wall_synthesis"])
    mic_ug_ml: float = 5.0
    half_life_hours: float = 8.0
    protein_binding: bool = True
    dose_mg: float = 1000.0


@dataclass
class Linezolid(Antibiotic):
    """Oxazolidinone antibiotic often used for MRSA."""
    name: str = "Linezolid"
    generic_name: str = "linezolid"
    drug_class: str = "oxazolidinone"
    bactericidal_type: BactericidalType = BactericidalType.TIME_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["50s_ribosome"])
    mic_ug_ml: float = 3.0
    half_life_hours: float = 6.0
    protein_binding: bool = True
    dose_mg: float = 600.0


# Fluoroquinolones
@dataclass
class Ciprofloxacin(Antibiotic):
    """Fluoroquinolone antibiotic."""
    name: str = "Ciprofloxacin"
    generic_name: str = "ciprofloxacin"
    drug_class: str = "fluoroquinolone"
    bactericidal_type: BactericidalType = BactericidalType.CONCENTRATION_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["DNA_gyrase"])
    mic_ug_ml: float = 1.0
    half_life_hours: float = 4.0
    dose_mg: float = 500.0


# Aminoglycosides
@dataclass
class Gentamicin(Antibiotic):
    """Aminoglycoside antibiotic."""
    name: str = "Gentamicin"
    generic_name: str = "gentamicin"
    drug_class: str = "aminoglycoside"
    bactericidal_type: BactericidalType = BactericidalType.CONCENTRATION_DEPENDENT
    targets: List[str] = field(default_factory=lambda: ["30s_ribosome"])
    mic_ug_ml: float = 4.0
    half_life_hours: float = 2.0
    protein_binding: bool = True
    dose_mg: float = 5.0  # mg/kg dosing


# =============================================================================
# Antivirals
# =============================================================================
@dataclass
class Antiviral:
    """Base class for antivirals."""
    name: str
    generic_name: str
    drug_class: str
    targets: List[str] = field(default_factory=list)
    ec50_um: float = 100.0
    half_life_hours: float = 12.0
    dose_mg: float = 100.0
    _bound_state: Dict[str, Any] = field(init=False, default_factory=dict)


@dataclass
class Oseltamivir(Antiviral):
    """Neuraminidase inhibitor (Tamiflu)."""
    name: str = "Oseltamivir"
    generic_name: str = "oseltamivir"
    drug_class: str = "neuraminidase_inhibitor"
    targets: List[str] = field(default_factory=lambda: ["neuraminidase"])
    ec50_um: float = 50.0
    half_life_hours: float = 8.0
    dose_mg: float = 75.0


@dataclass
class Remdesivir(Antiviral):
    """RNA polymerase inhibitor."""
    name: str = "Remdesivir"
    generic_name: str = "remdesivir"
    drug_class: str = "nucleoside_analogue"
    targets: List[str] = field(default_factory=lambda: ["RNA_polymerase_RdRp"])
    ec50_um: float = 100.0
    half_life_hours: float = 24.0
    dose_mg: float = 100.0


@dataclass
class Paxlovid(Antiviral):
    """Nirmatrelvir + ritonavir COVID combination."""
    name: str = "Paxlovid"
    generic_name: str = "nirmatrelvir/ritonavir"
    drug_class: str = "protease_inhibitor"
    targets: List[str] = field(default_factory=lambda: ["3CL_protease"])
    ec50_um: float = 100.0
    half_life_hours: float = 6.0
    dose_mg: float = 300.0


# =============================================================================
# Antiparasitics
# =============================================================================
@dataclass
class Antimalarial:
    """Base class for antimalarials."""
    name: str
    generic_name: str
    targets: List[str] = field(default_factory=list)
    stage_specific: bool = True
    resistance_prevalence: float = 0.0


@dataclass
class Artemisinin(Antimalarial):
    """Artemisinin-based combination therapy (ACT)."""
    name: str = "Artemisinin"
    generic_name: str = "artemether-lumefantrine"
    targets: List[str] = field(default_factory=lambda: ["heme_polymerization", "calcium_homeostasis"])
    stage_specific: bool = True
    resistance_prevalence: float = 0.05  # 5% in SE Asia


    half_life_hours: float = 2.0
    dose_mg: float = 80.0


@dataclass
class Chloroquine(Antimalarial):
    """Chloroquine - 4-aminoquinoline."""
    name: str = "Chloroquine"
    generic_name: str = "chloroquine"
    targets: List[str] = field(default_factory=lambda: ["heme_polymerization"])
    stage_specific: bool = False
    resistance_prevalence: float = 0.7  # 70% resistance in endemic areas
    half_life_hours: float = 720.0  # 30-60 days
    dose_mg: float = 600.0


# =============================================================================
# Anti-inflammatory
# =============================================================================
@dataclass
class Dexamethasone:
    """Corticosteroid anti-inflammatory."""
    name: str = "Dexamethasone"
    generic_name: str = "dexamethasone"
    drug_class: str = "corticosteroid"
    potency: float = 25.0  # 25x cortisol potency
    anti_inflammatory_effect: float = 0.8
    immunosuppressive: bool = True
    half_life_hours: float = 4.0
    dose_mg: float = 8.0


@dataclass
class Ibuprofen:
    """NSAID anti-inflammatory."""
    name: str = "Ibuprofen"
    generic_name: str = "ibuprofen"
    drug_class: str = "NSAID"
    cox1_inhibition_ic50_um: float = 50.0
    cox2_inhibition_ic50_um: float = 10.0
    anti_inflammatory_effect: float = 0.6
    half_life_hours: float = 2.0
    dose_mg: float = 400.0


# =============================================================================
# Drug Library Registry
# =============================================================================
EXTENDED_DRUG_LIBRARY: Dict[str, Type] = {
    # Antibiotics
    "amoxicillin": Amoxicillin,
    "piperacillin": Piperacillin,
    "ceftriaxone": Ceftriaxone,
    "meropenem": Meropenem,
    "vancomycin": Vancomycin,
    "linezolid": Linezolid,
    "ciprofloxacin": Ciprofloxacin,
    "gentamicin": Gentamicin,
    # Antivirals
    "oseltamivir": Oseltamivir,
    "remdesivir": Remdesivir,
    "paxlovid": Paxlovid,
    # Antiparasitics
    "artemisinin": Artemisinin,
    "chloroquine": Chloroquine,
    # Anti-inflammatory
    "dexamethasone": Dexamethasone,
    "ibuprofen": Ibuprofen,
}

# Backward-compatible lowercase aliases for the current tests and earlier local code.
amoxicillin = Amoxicillin
piperacillin = Piperacillin
ceftriaxone = Ceftriaxone
meropenem = Meropenem
vancomycin = Vancomycin
linezolid = Linezolid
ciprofloxacin = Ciprofloxacin
gentamicin = Gentamicin
oseltamivir = Oseltamivir
remdesivir = Remdesivir
paxlovid = Paxlovid
artemisinin = Artemisinin
chloroquine = Chloroquine

__all__ = [
    "DrugCategory",
    "BactericidalType",
    "Antibiotic",
    "Antiviral",
    "Antimalarial",
    "Amoxicillin",
    "Piperacillin",
    "Ceftriaxone",
    "Meropenem",
    "Vancomycin",
    "Linezolid",
    "Ciprofloxacin",
    "Gentamicin",
    "Oseltamivir",
    "Remdesivir",
    "Paxlovid",
    "Artemisinin",
    "Chloroquine",
    "Dexamethasone",
    "Ibuprofen",
    "EXTENDED_DRUG_LIBRARY",
]
