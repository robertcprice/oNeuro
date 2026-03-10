"""Digital Pathogen Library for oNeuro Pharma Platform.

This module provides biophysically-realistic digital pathogens (viruses, bacteria,
parasites) that can infect digital organisms and respond to pharmacological treatments.

Each pathogen has:
- Molecular targets (receptors, enzymes, ion channels)
- Replication dynamics
- Immune evasion strategies
- Drug susceptibility profiles
"""

from .virus import (
    DigitalVirus,
    InfluenzaA,
    Rhinovirus,
    SARSCoV2,
    HIV1,
    HerpesSimplex,
)
from .bacteria import (
    DigitalBacteria,
    EColi,
    Staphylococcus,
    Streptococcus,
    Pseudomonas,
    MycobacteriumTB,
)
from .parasite import (
    DigitalParasite,
    PlasmodiumFalciparum,  # Malaria
    ToxoplasmaGondii,
    Giardia,
)
from .prion import (
    DigitalPrion,
    PrionP,
)

__all__ = [
    # Viruses
    "DigitalVirus",
    "InfluenzaA",
    "Rhinovirus",
    "SARSCoV2",
    "HIV1",
    "HerpesSimplex",
    # Bacteria
    "DigitalBacteria",
    "EColi",
    "Staphylococcus",
    "Streptococcus",
    "Pseudomonas",
    "MycobacteriumTB",
    # Parasites
    "DigitalParasite",
    "PlasmodiumFalciparum",
    "ToxoplasmaGondii",
    "Giardia",
    # Prions
    "DigitalPrion",
    "PrionP",
]
