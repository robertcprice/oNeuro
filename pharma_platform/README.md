# oNeuro Pharma Platform

`pharma_platform/` is the digital pathogen and pharmacology sidecar for oNeuro.
It is a working subproject, not a frozen release, and it is now organized as a
real Python package rather than a loose set of scripts.

## Scope

- digital pathogens under `pathogens/`
- drug definitions under `drugs/`
- test/regression coverage under `tests/`
- future runnable studies under `experiments/`
- future performance measurements under `benchmarks/`
- explanatory material under `docs/`

## Layout

```text
pharma_platform/
├── __init__.py
├── README.md
├── pathogens/
├── drugs/
├── tests/
├── experiments/
├── benchmarks/
└── docs/
```

## Quick Start

From the repo root:

```bash
PYTHONPATH=src:. python3 -m py_compile pharma_platform/pathogens/*.py pharma_platform/drugs/*.py pharma_platform/tests/*.py
```

Example:

```python
from pharma_platform.pathogens.bacteria import EColi, Staphylococcus
from pharma_platform.tests.test_drug_efficacy import DrugEfficacyTester

e_coli = EColi()
e_coli.inoculate(initial_cfu=1e6)

tester = DrugEfficacyTester()
results = tester.test_antibiotic(e_coli, "beta_lactam")

for r in results:
    print(f"{r.drug_name}: {r.percent_killed:.1f}% killed")
```

## Notes

- `drugs/extended_drug_library.py` is the current consolidated drug definition file.
- `tests/test_drug_efficacy.py` is the current regression-style entrypoint.
- `docs/`, `benchmarks/`, and `experiments/` are now explicit directories so future work lands in predictable places.
