"""Digital Prion Models for oNeuro Pharma Platform.

Prions are misfolded proteins that propagate by inducing conformational
changes in normal proteins. They cause neurodegenerative diseases:

- Creutzfeldt-Jakob Disease (CJD)
- Bovine Spongiform Encephalopathy (BSE)
- Scrapie
- Fatal Familial Insomnia (FFI)
- Kuru

Key features:
- No nucleic acid (protein-only infectious agent)
- Resistant to standard sterilization
- Long incubation periods (years)
- 100% fatal once symptomatic
- No effective treatment
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class PrionStrain(Enum):
    """Prion strain classification."""
    CJD_SPORADIC = "cjd_sporadic"
    CJD_IATROGENIC = "cjd_iatrogenic"
    CJD_FAMILIAL = "cjd_familial"
    VCJD = "vcjd"  # Variant CJD (BSE-derived)
    SCRAPIE = "scrapie"
    BSE = "bse"
    FFI = "ffi"  # Fatal Familial Insomnia
    GSS = "gss"  # Gerstmann-Straussler-Scheinker
    KURU = "kuru"


class BrainRegion(Enum):
    """Affected brain regions."""
    CEREBRAL_CORTEX = "cortex"
    CEREBELLUM = "cerebellum"
    THALAMUS = "thalamus"
    BASAL_GANGLIA = "basal_ganglia"
    HIPPOCAMPUS = "hippocampus"
    BRAINSTEM = "brainstem"


@dataclass
class PrionLoad:
    """Prion burden tracking."""
    prion_concentration_pm: float = 0.0  # pM concentration
    infected_neurons: int = 0
    total_neurons: int = 86_000_000_000  # Human brain
    incubation_progress: float = 0.0  # 0-1
    time_years: float = 0.0

    @property
    def fraction_infected(self) -> float:
        return min(1.0, self.infected_neurons / max(1, self.total_neurons))


@dataclass
class PrionKinetics:
    """Prion propagation parameters."""
    conversion_rate_per_hour: float = 0.0001
    aggregation_threshold_nm: float = 100.0
    incubation_years: float = 10.0  # Average incubation
    clearance_rate_per_hour: float = 0.00001  # Very slow


@dataclass
class NeurodegenerationProfile:
    """Brain region-specific degeneration."""
    region: BrainRegion
    damage_fraction: float = 0.0
    spongiform_changes: float = 0.0
    neuronal_loss: float = 0.0
    gliosis: float = 0.0


@dataclass
class DigitalPrion:
    """Digital prion infection model.

    Models protein misfolding cascade:
    PrP^C (normal) + PrP^Sc (scrapie) → 2 PrP^Sc

    No nucleic acid - pure protein propagation.
    """

    name: str
    strain: PrionStrain

    kinetics: PrionKinetics = field(default_factory=PrionKinetics)
    load: PrionLoad = field(default_factory=PrionLoad)

    # Regional damage
    brain_damage: Dict[BrainRegion, NeurodegenerationProfile] = field(default_factory=dict)

    # State
    _symptomatic: bool = field(init=False, default=False)
    _time_since_exposure_years: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Initialize brain regions."""
        for region in BrainRegion:
            if region not in self.brain_damage:
                self.brain_damage[region] = NeurodegenerationProfile(region=region)

    def expose(self, initial_dose_pm: float = 1.0) -> None:
        """Initial prion exposure."""
        self.load.prion_concentration_pm = initial_dose_pm
        self._time_since_exposure_years = 0.0

    def step(self, dt_years: float = 0.01) -> Dict[str, Any]:
        """Advance prion propagation by dt_years.

        Returns dict with:
        - prion_concentration: current concentration
        - fraction_infected: fraction of neurons infected
        - symptomatic: whether symptomatic phase reached
        - incubation_progress: progress through incubation
        """
        dt_hours = dt_years * 365.25 * 24
        self.load.time_years += dt_years
        self._time_since_exposure_years += dt_years

        # Prion conversion: PrP^C + PrP^Sc → 2 PrP^Sc (autocatalytic)
        # d[PrP^Sc]/dt = k * [PrP^Sc] * [PrP^C] ≈ k * [PrP^Sc] (PrP^C abundant)
        conversion = self.kinetics.conversion_rate_per_hour * self.load.prion_concentration_pm * dt_hours
        clearance = self.kinetics.clearance_rate_per_hour * self.load.prion_concentration_pm * dt_hours

        self.load.prion_concentration_pm += conversion - clearance
        self.load.prion_concentration_pm = max(0, self.load.prion_concentration_pm)

        # Neuronal infection
        infection_prob = min(0.1, self.load.prion_concentration_pm / 1000.0)
        new_infections = int(self.load.total_neurons * infection_prob * dt_years)
        self.load.infected_neurons += new_infections

        # Incubation progress
        self.load.incubation_progress = min(1.0, self._time_since_exposure_years / self.kinetics.incubation_years)

        # Symptomatic phase
        if self.load.incubation_progress >= 0.9 and not self._symptomatic:
            self._symptomatic = True

        # Regional damage propagation
        self._update_regional_damage(dt_years)

        return {
            "prion_concentration_pm": self.load.prion_concentration_pm,
            "fraction_infected": self.load.fraction_infected,
            "symptomatic": self._symptomatic,
            "incubation_progress": self.load.incubation_progress,
            "time_years": self.load.time_years,
        }

    def _update_regional_damage(self, dt_years: float) -> None:
        """Update regional brain damage."""
        frac = self.load.fraction_infected

        # Strain-specific tropism
        tropism = self._get_strain_tropism()

        for region in BrainRegion:
            profile = self.brain_damage[region]
            weight = tropism.get(region, 1.0)

            # Spongiform changes
            profile.spongiform_changes = min(1.0, frac * weight * 2.0)

            # Neuronal loss (delayed)
            if profile.spongiform_changes > 0.3:
                profile.neuronal_loss = min(1.0, frac * weight * 1.5)

            # Gliosis (reactive)
            profile.gliosis = min(1.0, profile.neuronal_loss * 0.8)

            # Total damage
            profile.damage_fraction = (
                profile.spongiform_changes * 0.3 +
                profile.neuronal_loss * 0.5 +
                profile.gliosis * 0.2
            )

    def _get_strain_tropism(self) -> Dict[BrainRegion, float]:
        """Strain-specific brain region tropism."""
        if self.strain == PrionStrain.FFI:
            return {BrainRegion.THALAMUS: 3.0}
        elif self.strain == PrionStrain.GSS:
            return {BrainRegion.CEREBELLUM: 2.5}
        elif self.strain == PrionStrain.VCJD:
            return {
                BrainRegion.CEREBELLUM: 1.5,
                BrainRegion.BASAL_GANGLIA: 1.5,
            }
        else:
            # Sporadic CJD and others
            return {
                BrainRegion.CEREBRAL_CORTEX: 1.5,
                BrainRegion.CEREBELLUM: 1.2,
                BrainRegion.BASAL_GANGLIA: 1.0,
            }

    def get_clinical_symptoms(self) -> Dict[str, float]:
        """Clinical symptom severity."""
        if not self._symptomatic:
            return {"asymptomatic": 1.0}

        damage = self.brain_damage
        cortex = damage.get(BrainRegion.CEREBRAL_CORTEX, NeurodegenerationProfile(BrainRegion.CEREBRAL_CORTEX))
        cerebellum = damage.get(BrainRegion.CEREBELLUM, NeurodegenerationProfile(BrainRegion.CEREBELLUM))
        thalamus = damage.get(BrainRegion.THALAMUS, NeurodegenerationProfile(BrainRegion.THALAMUS))
        basal = damage.get(BrainRegion.BASAL_GANGLIA, NeurodegenerationProfile(BrainRegion.BASAL_GANGLIA))

        return {
            "dementia": cortex.damage_fraction * 0.9,
            "ataxia": cerebellum.damage_fraction * 0.8,
            "myoclonus": cortex.damage_fraction * 0.6 + basal.damage_fraction * 0.3,
            "visual_disturbances": cortex.damage_fraction * 0.5,
            "insomnia": thalamus.damage_fraction * 0.7,  # FFI specific
            "pyramidal_signs": 0.3 * cortex.damage_fraction,
            "extrapyramidal_signs": basal.damage_fraction * 0.5,
            "akinetic_mutism": cortex.damage_fraction * 0.8,
        }

    def get_eeg_pattern(self) -> Dict[str, Any]:
        """EEG findings in prion disease."""
        if not self._symptomatic:
            return {"normal": True}

        cortex = self.brain_damage.get(BrainRegion.CEREBRAL_CORTEX,
                                       NeurodegenerationProfile(BrainRegion.CEREBRAL_CORTEX))

        return {
            "normal": False,
            "periodic_sharp_wave_complexes": cortex.damage_fraction > 0.5,
            "background_slowing": cortex.damage_fraction > 0.3,
            "pswc_frequency_hz": 1.0 + cortex.damage_fraction * 0.5,
        }

    def get_mri_findings(self) -> Dict[str, Any]:
        """MRI signal changes."""
        findings = {}
        for region, profile in self.brain_damage.items():
            if profile.damage_fraction > 0.2:
                findings[region.value] = {
                    "hyperintensity_t2": profile.spongiform_changes,
                    "cortical_ribboning": profile.spongiform_changes > 0.5,
                    "atrophy": profile.neuronal_loss > 0.3,
                }
        return findings


@dataclass
class PrionP(DigitalPrion):
    """Generic PrP^Sc prion (default CJD model)."""

    name: str = "PrP^Sc"
    strain: PrionStrain = PrionStrain.CJD_SPORADIC

    kinetics: PrionKinetics = field(default_factory=lambda: PrionKinetics(
        conversion_rate_per_hour=0.00015,
        incubation_years=5.0,  # Sporadic onset
        clearance_rate_per_hour=0.00001,
    ))


@dataclass
class VariantCJD(DigitalPrion):
    """Variant CJD (vCJD) - BSE-derived human prion disease.

    Younger onset, longer duration, prominent psychiatric symptoms.
    Transmitted via contaminated beef products.
    """

    name: str = "vCJD"
    strain: PrionStrain = PrionStrain.VCJD

    kinetics: PrionKinetics = field(default_factory=lambda: PrionKinetics(
        conversion_rate_per_hour=0.00008,
        incubation_years=12.0,  # Longer incubation
        clearance_rate_per_hour=0.000008,
    ))


@dataclass
class FatalFamilialInsomnia(DigitalPrion):
    """Fatal Familial Insomnia - genetic prion disease.

    Mutation in PRNP gene (D178N, M129V).
    Selective thalamic degeneration.
    """

    name: str = "FFI"
    strain: PrionStrain = PrionStrain.FFI

    kinetics: PrionKinetics = field(default_factory=lambda: PrionKinetics(
        conversion_rate_per_hour=0.0002,
        incubation_years=0.0,  # Genetic - always present
        clearance_rate_per_hour=0.00002,
    ))

    def get_sleep_architecture(self) -> Dict[str, float]:
        """Sleep disruption profile."""
        thalamus = self.brain_damage.get(BrainRegion.THALAMUS,
                                         NeurodegenerationProfile(BrainRegion.THALAMUS))
        damage = thalamus.damage_fraction

        return {
            "total_sleep_time_hours": max(0, 8 - damage * 8),
            "rem_sleep_minutes": max(0, 90 - damage * 90),
            "slow_wave_sleep_minutes": max(0, 60 - damage * 60),
            "sleep_efficiency": max(0, 0.95 - damage * 0.9),
            "hypnagogic_hallucinations": damage * 0.7,
        }
