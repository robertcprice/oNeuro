"""Digital Parasite Models for oNeuro Pharma Platform.

Biophysically-realistic parasitic infections including:
- Protozoa (malaria, toxoplasma, giardia)
- Helminths (schistosomes, nematodes)

Each parasite has realistic:
- Life cycle stages
- Host-parasite interactions
- Drug targets by stage
- Immunomodulation strategies
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ParasiteType(Enum):
    PROTOZOAN = "protozoan"
    HELMINTH = "helminth"
    ECTOPARASITE = "ectoparasite"


class LifeCycleStage(Enum):
    SPOROZOITE = "sporozoite"
    MEROZOITE = "merozoite"
    TROPHOZOITE = "trophozoite"
    SCHIZONT = "schizont"
    GAMETOCYTE = "gametocyte"
    TACHYZOITE = "tachyzoite"
    BRADYZOITE = "bradyzoite"
    CYST = "cyst"
    TROPHOZOITE = "trophozoite"
    EGG = "egg"
    LARVA = "larva"
    ADULT = "adult"


class AntiparasiticTarget(Enum):
    HEME_POLYMERIZATION = "heme_polymerization"
    FOLATE_SYNTHESIS = "folate_synthesis"
    MITOCHONDRIAL_ELECTRON_TRANSPORT = "mito_et"
    CALCIUM_HOMEOSTASIS = "calcium"
    TUBULIN = "tubulin"
    NEUROMUSCULAR = "neuromuscular"
    GLUTAMATE_GATED_CHANNELS = "glutamate_channel"


@dataclass
class ParasiteLoad:
    """Parasite burden tracking."""
    count: int = 0
    stage: LifeCycleStage = LifeCycleStage.TROPHOZOITE
    parasitemia_percent: float = 0.0  # % infected cells
    time_hours: float = 0.0


@dataclass
class ParasiteKinetics:
    """Parasite growth parameters."""
    replication_rate_per_hour: float = 0.1
    stage_duration_hours: Dict[LifeCycleStage, float] = field(default_factory=dict)
    host_cell_burst_size: int = 16  # Merozoites per burst
    immune_evasion_factor: float = 0.5


@dataclass
class DigitalParasite:
    """Base class for digital parasites."""
    name: str
    species: str
    parasite_type: ParasiteType

    kinetics: ParasiteKinetics = field(default_factory=ParasiteKinetics)
    load: ParasiteLoad = field(default_factory=ParasiteLoad)
    drug_targets: List[AntiparasiticTarget] = field(default_factory=list)

    _current_stage_idx: int = field(init=False, default=0)
    _time_in_stage: float = field(init=False, default=0.0)

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance parasite dynamics."""
        self.load.time_hours += dt_hours
        self._time_in_stage += dt_hours
        return {"count": self.load.count, "stage": self.load.stage}


@dataclass
class PlasmodiumFalciparum(DigitalParasite):
    """Malaria parasite - Plasmodium falciparum.

    Life cycle in human:
    1. Sporozoite (mosquito bite) → liver
    2. Merozoite (liver release) → RBC invasion
    3. Ring stage → Trophozoite → Schizont
    4. Merozoite burst → reinvasion
    5. Gametocyte (for mosquito transmission)

    Severe malaria: cerebral involvement, severe anemia
    """

    name: str = "P. falciparum"
    species: str = "Plasmodium falciparum"
    parasite_type: ParasiteType = ParasiteType.PROTOZOAN

    kinetics: ParasiteKinetics = field(default_factory=lambda: ParasiteKinetics(
        replication_rate_per_hour=0.5,  # 48h cycle
        host_cell_burst_size=16,
        immune_evasion_factor=0.7,
        stage_duration_hours={
            LifeCycleStage.SPOROZOITE: 0.5,
            LifeCycleStage.MEROZOITE: 0.1,
            LifeCycleStage.TROPHOZOITE: 24.0,
            LifeCycleStage.SCHIZONT: 24.0,
            LifeCycleStage.GAMETOCYTE: 288.0,  # 12 days
        }
    ))

    drug_targets: List[AntiparasiticTarget] = field(default_factory=lambda: [
        AntiparasiticTarget.HEME_POLYMERIZATION,  # Chloroquine
        AntiparasiticTarget.FOLATE_SYNTHESIS,     # Pyrimethamine
        AntiparasiticTarget.MITOCHONDRIAL_ELECTRON_TRANSPORT,  # Atovaquone
        AntiparasiticTarget.CALCIUM_HOMEOSTASIS,  # Artemisinin
    ])

    # Drug resistance
    chloroquine_resistant: bool = True
    artemisinin_resistant: bool = False

    # Severe malaria tracking
    _cerebral_involvement: float = field(init=False, default=0.0)
    _parasite_sequestration: float = field(init=False, default=0.0)

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance malaria cycle."""
        result = super().step(dt_hours)

        # Parasitemia calculation (simplified)
        total_rbc = 5_000_000_000_000  # 5 trillion RBCs
        infected = self.load.count
        self.load.parasitemia_percent = (infected / total_rbc) * 100

        # Stage progression
        stage_duration = self.kinetics.stage_duration_hours.get(self.load.stage, 24.0)
        if self._time_in_stage >= stage_duration:
            self._advance_stage()

        # Schizont burst
        if self.load.stage == LifeCycleStage.SCHIZONT and self._time_in_stage >= 24.0:
            self.load.count *= self.kinetics.host_cell_burst_size
            self.load.stage = LifeCycleStage.MEROZOITE
            self._time_in_stage = 0.0

        # Cerebral involvement (severe malaria)
        if self.load.parasitemia_percent > 2.0:
            self._cerebral_involvement = min(1.0, self.load.parasitemia_percent / 10.0)

        result.update({
            "parasitemia_percent": self.load.parasitemia_percent,
            "cerebral_involvement": self._cerebral_involvement,
            "stage": self.load.stage.value,
        })
        return result

    def _advance_stage(self) -> None:
        """Progress to next life cycle stage."""
        stage_order = [
            LifeCycleStage.SPOROZOITE,
            LifeCycleStage.MEROZOITE,
            LifeCycleStage.TROPHOZOITE,
            LifeCycleStage.SCHIZONT,
        ]
        try:
            idx = stage_order.index(self.load.stage)
            if idx < len(stage_order) - 1:
                self.load.stage = stage_order[idx + 1]
                self._time_in_stage = 0.0
        except ValueError:
            pass

    def get_cytokine_response(self) -> Dict[str, float]:
        """Malaria cytokine storm."""
        para = self.load.parasitemia_percent
        return {
            "tnf_alpha": 10.0 + para * 200.0,
            "il_6": 20.0 + para * 150.0,
            "ifn_gamma": 50.0 + para * 300.0,
            "il_10": 10.0 + para * 50.0,  # Anti-inflammatory
            "il_1ra": 5.0 + para * 30.0,  # IL-1 receptor antagonist
        }

    def get_severe_malaria_risk(self) -> Dict[str, float]:
        """Risk factors for severe malaria."""
        para = self.load.parasitemia_percent
        return {
            "cerebral_malaria": min(0.8, para / 5.0) * (1 - self.kinetics.immune_evasion_factor),
            "severe_anemia": min(0.6, para / 8.0),
            "acute_kidney_injury": min(0.4, para / 10.0),
            "ards": min(0.3, para / 15.0),
            "hypoglycemia": min(0.5, para / 6.0),
        }


@dataclass
class ToxoplasmaGondii(DigitalParasite):
    """Toxoplasma gondii - obligate intracellular parasite.

    Notable for: CNS cysts, behavioral effects, congenital infection.
    """

    name: str = "T. gondii"
    species: str = "Toxoplasma gondii"
    parasite_type: ParasiteType = ParasiteType.PROTOZOAN

    kinetics: ParasiteKinetics = field(default_factory=lambda: ParasiteKinetics(
        replication_rate_per_hour=0.2,
        stage_duration_hours={
            LifeCycleStage.TACHYZOITE: 24.0,
            LifeCycleStage.BRADYZOITE: 720.0,  # 30 days to form cyst
            LifeCycleStage.CYST: 86400.0,  # Years
        }
    ))

    drug_targets: List[AntiparasiticTarget] = field(default_factory=lambda: [
        AntiparasiticTarget.FOLATE_SYNTHESIS,  # Pyrimethamine + sulfadiazine
    ])

    _cyst_count: int = field(init=False, default=0)
    _brain_cysts: int = field(init=False, default=0)
    _behavioral_changes: float = field(init=False, default=0.0)

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance Toxoplasma dynamics."""
        result = super().step(dt_hours)

        # Stage transition: tachyzoite → bradyzoite (under immune pressure)
        if self.load.stage == LifeCycleStage.TACHYZOITE:
            # Transition to chronic stage
            if self.load.time_hours > 72.0:
                self.load.stage = LifeCycleStage.BRADYZOITE

        # Cyst formation in brain
        if self.load.stage == LifeCycleStage.BRADYZOITE:
            if self._time_in_stage > 720.0:  # 30 days
                self._brain_cysts += 1
                self.load.count = max(1, self.load.count - 10)

        # Behavioral effects (controversial but documented)
        self._behavioral_changes = min(0.3, self._brain_cysts * 0.01)

        result.update({
            "brain_cysts": self._brain_cysts,
            "behavioral_changes": self._behavioral_changes,
            "stage": self.load.stage.value,
        })
        return result


@dataclass
class Giardia(DigitalParasite):
    """Giardia lamblia - intestinal protozoan.

    Causes giardiasis: diarrhea, malabsorption.
    """

    name: str = "Giardia"
    species: str = "Giardia lamblia"
    parasite_type: ParasiteType = ParasiteType.PROTOZOAN

    kinetics: ParasiteKinetics = field(default_factory=lambda: ParasiteKinetics(
        replication_rate_per_hour=0.15,
        stage_duration_hours={
            LifeCycleStage.TROPHOZOITE: 120.0,  # 5 days
            LifeCycleStage.CYST: 720.0,
        }
    ))

    drug_targets: List[AntiparasiticTarget] = field(default_factory=lambda: [
        AntiparasiticTarget.TUBULIN,  # Metronidazole
    ])

    _gi_count: int = field(init=False, default=0)  # GI tract load

    def step(self, dt_hours: float = 0.1) -> Dict[str, Any]:
        """Advance Giardia dynamics."""
        result = super().step(dt_hours)

        # Gut colonization
        self._gi_count = self.load.count

        result.update({
            "gi_count": self._gi_count,
            "stage": self.load.stage.value,
        })
        return result

    def get_gi_symptoms(self) -> Dict[str, float]:
        """Gastrointestinal symptoms."""
        burden = min(1.0, self._gi_count / 1000000)
        return {
            "diarrhea": burden * 0.8,
            "steatorrhea": burden * 0.4,
            "bloating": burden * 0.6,
            "malabsorption": burden * 0.3,
        }
