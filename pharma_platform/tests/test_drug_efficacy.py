"""oNeuro Pharma Platform - Drug Efficacy Tests.

Comprehensive testing framework for validating drug-pathogen interactions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pharma_platform.pathogens.bacteria import (
    DigitalBacteria,
    EColi,
    Staphylococcus,
    Streptococcus,
    Pseudomonas,
    MycobacteriumTB,
    BacterialClass,
    AntibioticTarget,
)
from pharma_platform.pathogens.virus import (
    DigitalVirus,
    InfluenzaA,
    Rhinovirus,
    SARSCoV2,
    HIV1,
    HerpesSimplex,
    ViralTarget,
)
from pharma_platform.pathogens.parasite import (
    DigitalParasite,
    PlasmodiumFalciparum,
    ToxoplasmaGondii,
    Giardia,
    AntiparasiticTarget,
)

# Import drugs
from pharma_platform.drugs.extended_drug_library import (
    Antibiotic,
    amoxicillin,
    piperacillin,
    ceftriaxone,
    meropenem,
    vancomycin,
    linezolid,
    ciprofloxacin,
    gentamicin,
    artemisinin,
    chloroquine,
    Dexamethasone,
    Ibuprofen,
)


@dataclass
class TestResult:
    """Result of a drug efficacy test."""
    test_name: str
    pathogen_name: str
    drug_name: str
    dose_mg: float
    initial_burden: float
    final_burden: float
    log_reduction: float
    percent_killed: float
    mic_ratio: float
    bactericidal: bool
    time_hours: float
    passed: bool
    notes: str = ""


class DrugEfficacyTester:
    """Test drug efficacy against pathogens."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("test_results")
        self.results: List[TestResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_antibiotic(
        self,
        bacteria: DigitalBacteria,
        antibiotic_class: str,
        concentrations: List[float] = None,
        duration_hours: float = 24.0,
    ) -> List[TestResult]:
        """Test antibiotic efficacy against bacteria."""
        if concentrations is None:
            concentrations = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

        bacteria.inoculate(initial_cfu=1e6)
        initial_burden = bacteria.load.log_cfu

        results = []
        for conc in concentrations:
            # Reset bacteria
            bacteria.load.cfu_per_ml = 1e6
            bacteria._in_lag_phase = True

            # Run simulation
            for _ in range(int(duration_hours * 10)):
                bacteria.step(dt_hours=0.0)
                bacteria.apply_antibiotic(antibiotic_class, conc)

            final_burden = bacteria.load.log_cfu
            log_reduction = initial_burden - final_burden
            percent_killed = (1 - 10**final_burden) * 100

            # Determine bactericidal vs bacteriostatic
            bactericidal = percent_killed > 99.0

            # Get MIC
            mic = self._estimate_mic(bacteria, antibiotic_class)

            result = TestResult(
                test_name=f"{bacteria.name}_{antibiotic_class}_{conc}ug_ml",
                pathogen_name=bacteria.name,
                drug_name=antibiotic_class,
                dose_mg=conc,
                initial_burden=initial_burden,
                final_burden=final_burden,
                log_reduction=log_reduction,
                percent_killed=percent_killed,
                mic_ratio=conc / mic if mic > 0e-10 else conc,
                bactericidal=bactericidal,
                time_hours=duration_hours,
                passed=log_reduction >= 2.0,
            )
            results.append(result)

        return results

    def _estimate_mic(
        self,
        bacteria: DigitalBacteria,
        antibiotic_class: str,
    ) -> float:
        """Estimate MIC from resistance profile."""
        for resistance in bacteria.resistance_profile:
            if resistance.antibiotic_class == antibiotic_class:
                return resistance.mic_ug_ml
        # Susceptible default
        return 1.0

    def run_mrsa_test(
        self,
        drug_classes: List[str] = None,
    ) -> Dict[str, List[TestResult]]:
        """Test MRSA against multiple antibiotic classes."""
        mrsa = Staphylococcus(is_mrsa=True)
        mrsa.inoculate(initial_cfu=1e6)

        if drug_classes is None:
            drug_classes = ["beta_lactam", "macrolide", "fluoroquinolone", "glycopeptide"]

        results = {}
        for drug_class in drug_classes:
            results[drug_class] = self.test_antibiotic(mrsa, drug_class)

        return results
    def run_malaria_drug_test(
        self,
        parasite: PlasmodiumFalciparum,
        drugs: List[str] = None,
    ) -> Dict[str, List[TestResult]]:
        """Test antimalarial drug efficacy."""
        if drugs is None:
            drugs = ["artemisinin", "chloroquine"]

        results = {}
        for drug_name in drugs:
            # Reset parasite
            parasite.load.count = 1e8
            parasite.load.parasitemia_percent = 0.001

            initial = parasite.load.parasitemia_percent

            # Simulate treatment
            if drug_name == "artemisinin":
                # Artemisinin: rapid parasite clearance
                for _ in range(48):  # 48 hours
                    result = parasite.step(dt_hours=1.0)
                    parasite.load.count *= 0.9  # 90% reduction per step

            elif drug_name == "chloroquine":
                # Chloroquine: effective only if sensitive
                if not parasite.chloroquine_resistant:
                    for _ in range(48):
                        result = parasite.step(dt_hours=1.0)
                        parasite.load.count *= 0.95
                else:
                    # Resistant: no effect
                    for _ in range(48):
                        result = parasite.step(dt_hours=1.0)

            final = parasite.load.parasitemia_percent
            reduction = initial - final

            passed = reduction > initial * 0.8

            results[drug_name] = TestResult(
                test_name=f"malaria_{drug_name}",
                pathogen_name="P_falciparum",
                drug_name=drug_name,
                dose_mg=80.0 if drug_name == "artemisinin" else 600.0,
                initial_burden=initial,
                final_burden=final,
                log_reduction=-math.log10(max(1, final)) + math.log10(max(1, initial)),
                percent_killed=reduction * 100,
                mic_ratio=1.0,
                bactericidal=reduction > 0.99,
                time_hours=48.0,
                passed=passed,
            )
        return results

    def generate_dose_response_curve(
        self,
        bacteria: DigitalBacteria,
        antibiotic_class: str,
        concentrations: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Generate dose-response curve data."""
        if concentrations is None:
            concentrations = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

        results = self.test_antibiotic(bacteria, antibiotic_class, concentrations)

        # Calculate EC50, EC90
        doses = np.array([r.dose_mg for r in results])
        kills = np.array([r.percent_killed for r in results])
        ec50 = self._interpolate_ec(doses, kills, 50)
        ec90 = self._interpolate_ec(doses, kills, 90)
        mic = self._estimate_mic(bacteria, antibiotic_class)
        return {
            "bacteria": bacteria.name,
            "antibiotic": antibiotic_class,
            "concentrations": doses.tolist(),
            "kill_fractions": kills.tolist(),
            "ec50": ec50,
            "ec90": ec90,
            "mic": mic,
            "susceptibility": "S" if ec50 < mic * 2 else "r" if ec50 > mic * 4 else "i",
        }
    def _interpolate_ec(self, doses: np.ndarray, kills: np.ndarray, target_percent: float) -> float:
        """Interpolate to find ECx value."""
        for i in range(len(kills) - 1):
            if kills[i] <= target_percent <= kills[i + 1]:
                # Linear interpolation
                t = (target_percent - kills[i]) / (kills[i + 1] - kills[i])
                return doses[i] + t * (doses[i + 1] - doses[i])
        return doses[-1]
    def save_results(self, filename: str = "test_results.json") -> None:
        """Save all results to JSON file."""
        data = {
            "results": [
                {
                    "test_name": r.test_name,
                    "pathogen": r.pathogen_name,
                    "drug": r.drug_name,
                    "dose_mg": r.dose_mg,
                    "log_reduction": r.log_reduction,
                    "percent_killed": r.percent_killed,
                    "mic_ratio": r.mic_ratio,
                    "bactericidal": r.bactericidal,
                    "passed": r.passed,
                    "notes": r.notes,
                }
                for r in self.results
            ]
        }
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


def run_standard_battery():
    """Run a standard battery of drug-pathogen tests."""
    tester = DrugEfficacyTester()
    all_results = []
    # Test common bacteria
    bacteria_tests = [
        (EColi(), "beta_lactam"),
        (EColi(), "fluoroquinolone"),
        (Staphylococcus(is_mrsa=True), "beta_lactam"),
        (Staphylococcus(is_mrsa=True), "glycopeptide"),
        (Staphylococcus(is_mrsa=True), "oxazolidinone"),
        (Pseudomonas(), "fluoroquinolone"),
        (Pseudomonas(), "beta_lactam"),
    ]
    for bacteria, drug_class in bacteria_tests:
        results = tester.test_antibiotic(bacteria, drug_class)
        all_results.extend(results)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"{r.pathogen_name} + {r.drug_name} @ {r.dose_mg}ug/ml: {status} ({r.log_reduction:.2f} log kill)")
    # Test malaria
    malaria = PlasmodiumFalciparum(chloroquine_resistant=True)
    malaria_results = tester.run_malaria_drug_test(malaria)
    print("\nMalaria Drug Tests:")
    for drug_name, result in malaria_results.items():
        if isinstance(result, TestResult):
            status = "PASS" if result.passed else "FAIL"
            print(f"  {drug_name}: {status} (parasitemia reduction: {result.percent_killed:.1f}%)")
    # Save results
    output_path = tester.save_results()
    print(f"\nResults saved to: {output_path}")
    return all_results
if __name__ == "__main__":
    print("=" * 60)
    print("oNeuro Pharma Platform - Drug Efficacy Test Suite")
    print("=" * 60)
    results = run_standard_battery()
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print(f"{'=' * 60}")
