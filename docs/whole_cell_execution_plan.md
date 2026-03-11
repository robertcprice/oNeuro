# Whole-Cell Execution Plan

## Purpose

This is the execution-grade plan from the current `oNeuro` state to a genome-explicit, chemically explicit, multiscale microbial simulator with native local atomistic refinement.

This document is meant to stop the stop-and-go loop.

- Work should proceed through these phases in order.
- Do not stop after each substep.
- Stop only for a real blocker:
  - missing licensed data
  - impossible hardware assumption
  - contradictory biological target
  - architecture decision that would permanently narrow the system

## Current Starting Point

What already exists:

- Rust whole-cell runtime in [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- Whole-cell program/state data model in [oneuro-metal/src/whole_cell_data.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_data.rs)
- Whole-cell chemistry/assembly/subsystem bridge in [oneuro-metal/src/whole_cell_submodels.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_submodels.rs)
- Local atomistic topology templates in [oneuro-metal/src/atomistic_topology.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/atomistic_topology.rs)
- Bundled Syn3A program and organism specs in:
  - [whole_cell_syn3a_reference.json](/Users/bobbyprice/projects/oNeuro/oneuro-metal/specs/whole_cell_syn3a_reference.json)
  - [whole_cell_syn3a_organism.json](/Users/bobbyprice/projects/oNeuro/oneuro-metal/specs/whole_cell_syn3a_organism.json)
- Existing strategic docs in:
  - [whole_cell_strategy.md](/Users/bobbyprice/projects/oNeuro/docs/whole_cell_strategy.md)
  - [whole_cell_atomic_roadmap.md](/Users/bobbyprice/projects/oNeuro/docs/whole_cell_atomic_roadmap.md)
  - [whole_cell_mc4d_gap.md](/Users/bobbyprice/projects/oNeuro/docs/whole_cell_mc4d_gap.md)

What does not exist yet:

- full genome-compiled organism asset pipeline from real source annotations
- explicit global species/reaction registry driving the full cell
- explicit chromosome polymer mechanics at the fidelity needed for parity
- native whole-cell multirate scheduler with explicit solver clocks
- real atomistic subsystem extraction from live whole-cell state
- calibration and validation stack strong enough for MC4D-level claims

## Final Target

The target is not “simulate every atom in the whole cell at all times.”

The target is:

- explicit microbial genome assets as the organism source of truth
- explicit RNAs, proteins, complexes, metabolites, ions, lipids, and membrane species as runtime state
- explicit assembly and chromosome state
- explicit reaction registries and solver schedules
- local atomistic refinement for the highest-value sites and transitions
- compiled multiscale reductions from atomistic and mesoscale truth back into the whole-cell runtime
- restartable, reproducible, benchmarked runs that match and then exceed MC4D observables

## Non-Negotiable Architecture Rules

1. Keep the authoritative hot path in Rust.
2. Keep the performance-critical kernels on GPU when possible.
3. Do not move the whole-cell engine back into Python.
4. Use Python for orchestration, experiment driving, reporting, and adapters.
5. Keep all major biology layers serializable and restartable.
6. Every abstraction must compile down to explicit runtime state, not just metadata.
7. Every phase must add tests, artifacts, and benchmark outputs.

## Master Execution Sequence

### Phase A: Freeze The Runtime Contract

Goal:
- lock the data contracts so later work does not thrash the architecture

Build steps:
1. Freeze units for counts, concentrations, lattice geometry, energies, forces, and clocks.
2. Freeze canonical runtime layers:
   - genome features
   - transcripts
   - proteins
   - complexes
   - reactions
   - chromosomes
   - membranes
   - geometry
   - atomistic local domains
3. Freeze canonical IR layers:
   - genome/transcription IR
   - species/reaction IR
   - assembly IR
   - chromosome/polymer IR
   - multirate schedule IR
4. Freeze restart schema and provenance metadata.
5. Add a progress ledger mapping each milestone to tests and datasets.

Primary files:
- [oneuro-metal/src/whole_cell_data.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_data.rs)
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- [docs/whole_cell_atomic_roadmap.md](/Users/bobbyprice/projects/oNeuro/docs/whole_cell_atomic_roadmap.md)

Exit criteria:
- no core state object is ambiguous
- save/restore covers every authoritative layer
- all later phases can add content without changing the contract

### Phase B: Compile Real Organism Assets

Goal:
- stop hand-maintaining the organism as flat JSON blobs

Build steps:
6. Add parsers for `FASTA`, `GenBank`, `GFF`, operon tables, protein annotations, and essentiality tables.
7. Build a compiler that turns those inputs into organism asset packages:
   - genes
   - operons
   - transcription units
   - RNAs
   - proteins
   - complexes
   - composition priors
   - geometry priors
8. Extend the current Syn3A package generation path to come from that compiler.
9. Add a second organism target to keep the pipeline generic.
10. Add validation that compiled assets round-trip and preserve identifiers and coordinates.

Primary files/modules:
- [oneuro-metal/src/whole_cell_data.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_data.rs)
- new `oneuro.whole_cell.assets` Python-side compiler surface
- bundled specs in [oneuro-metal/specs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/specs)

Exit criteria:
- Syn3A assets are compiler-produced, not manually curated blobs
- second-organism assets compile through the same path

### Phase C: Make Species And Reactions Explicit

Goal:
- replace coarse process surrogates with explicit molecular state

Build steps:
11. Add a canonical species registry covering:
   - metabolites
   - ions
   - cofactors
   - lipids
   - RNAs
   - proteins
   - complexes
   - membrane species
12. Add a canonical reaction registry with stoichiometry and compartment scope.
13. Compile expression, metabolism, degradation, transport, membrane synthesis, repair, and division chemistry into that registry.
14. Add explicit molecule-count bookkeeping for every compiled species.
15. Replace remaining hand-authored process-rate branches with compiled reaction execution where feasible.
16. Make enzyme abundance and complex state gate reaction capacity.

Primary files/modules:
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- [oneuro-metal/src/whole_cell_data.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_data.rs)
- [oneuro-metal/src/whole_cell_submodels.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_submodels.rs)

Exit criteria:
- every important process has explicit species and reactions behind it
- reaction execution is driven by compiled registries, not scattered heuristics

### Phase D: Make Assembly Explicit

Goal:
- move from aggregate capacities to named complexes and intermediates

Build steps:
17. Extend complex state into a first-class assembly graph runtime.
18. Add named assembly intermediates for:
   - ribosomes
   - RNAP
   - replisomes
   - ATP synthase
   - transporters
   - membrane enzymes
   - FtsZ/divisome
19. Add recruitment, completion, failure, degradation, and repair paths.
20. Tie protein inventories directly to assembly demand and sequestration.
21. Export concrete assembly-aware capacities to higher solver layers.

Primary files/modules:
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- [oneuro-metal/src/whole_cell_submodels.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_submodels.rs)

Exit criteria:
- no major subsystem depends on a fake unnamed capacity pool

### Phase E: Make Chromosomes Explicit

Goal:
- move from scalar replication state to actual chromosome mechanics

Build steps:
22. Add explicit circular chromosome feature indexing over the compiled genome.
23. Add fork state, origin state, terminus state, replication initiation, and fork progression.
24. Add collision-aware transcription/replication bookkeeping.
25. Add polymer-level chromosome state:
   - loci
   - domains
   - tethering
   - compaction
   - segregation
26. Add DNA topology state where it materially affects expression and replication.
27. Add restartable chromosome observables and artifact outputs.

Primary files/modules:
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- new chromosome/polymer module in `oneuro-metal/src`

Exit criteria:
- chromosome behavior is an explicit runtime subsystem, not just a progress scalar

### Phase F: Make Membranes And Division Explicit

Goal:
- tie shape and division to actual molecular state

Build steps:
28. Add explicit membrane composition and inserted-protein state.
29. Add curvature and septum-local remodeling state.
30. Add divisome assembly order and occupancy.
31. Couple constriction to assembled divisome mechanics.
32. Add geometry updates driven by membrane synthesis, osmotic load, and chromosome occlusion.

Primary files/modules:
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- membrane/division submodules in `oneuro-metal/src`

Exit criteria:
- division and geometry no longer run as only scalar progress curves

### Phase G: Turn Atomistic Chemistry Into A Native Truth Source

Goal:
- move from template-only local MD probes to real subsystem extraction and feedback

Build steps:
33. Extend the atomistic topology pipeline to ingest real subsystem structures and parameter sets.
34. Add force-field ingestion and validation for bonded, electrostatic, and nonbonded terms.
35. Build live local-domain extractors from whole-cell state around:
   - ribosomes
   - replisomes
   - ATP synthase bands
   - membrane insertion sites
   - divisome zones
36. Add uncertainty or rare-event triggers that invoke atomistic refinement.
37. Feed atomistic outputs back as:
   - rate corrections
   - energies
   - conformational states
   - diffusion changes
   - assembly/degradation propensities
38. Build atomistic microbenchmarks against known structures and local observables.

Primary files/modules:
- [oneuro-metal/src/atomistic_topology.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/atomistic_topology.rs)
- [oneuro-metal/src/whole_cell_submodels.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell_submodels.rs)
- MD engine code in `oneuro-metal/src/molecular_dynamics.rs`

Exit criteria:
- atomistic refinement is driven from live cell state
- whole-cell rates and assemblies materially depend on it in selected hotspots

### Phase H: Replace The Scheduler With A True Multirate Solver Stack

Goal:
- upgrade the orchestration from staged updates to explicit multiscale clocks

Build steps:
39. Add solver schedule IR with separate clocks for:
   - RDME
   - CME
   - ODE
   - chromosome/polymer BD
   - membrane mechanics
   - atomistic refinement
40. Add event-driven scheduling for rare transitions and checkpoints.
41. Add deterministic replay and solver-layer checkpointing.
42. Add distributed execution hooks for larger atomistic workloads.

Primary files/modules:
- [oneuro-metal/src/whole_cell.rs](/Users/bobbyprice/projects/oNeuro/oneuro-metal/src/whole_cell.rs)
- new scheduler modules in `oneuro-metal/src`

Exit criteria:
- each solver layer has an explicit cadence and restart surface

### Phase I: Build The Calibration And Validation System

Goal:
- make the biology measurable and fit for regression

Build steps:
43. Build the Syn3A reference dataset bundle:
   - genome
   - transcript measurements
   - protein measurements
   - metabolite concentrations
   - growth curves
   - replication timing
   - division timing
   - perturbation data
44. Add calibration pipelines for:
   - expression
   - metabolism
   - assembly
   - chromosome behavior
   - membrane/division
   - atomistic local refinement
45. Add held-out validation and sensitivity analysis.
46. Add per-module residual reports and uncertainty bands.
47. Add regression suites against overlapping MC4D observables.

Primary files/modules:
- new `oneuro.whole_cell.validation`
- experiment/reporting paths under `results/` and `docs/`

Exit criteria:
- every major module has a benchmark, fit target, and held-out validation surface

### Phase J: Reach MC4D Parity

Goal:
- match the published baseline on robustness and observables

Build steps:
48. Reproduce the published Syn3A observables that matter:
   - cell-cycle timing
   - expression distributions
   - metabolic support
   - chromosome behavior
   - division progression
49. Match restartability, artifact completeness, and reproducibility.
50. Match solver coverage at the observable level.
51. Remove remaining heuristics that are not grounded in lower-scale or data-driven logic.

Exit criteria:
- parity reports show overlap with MC4D on the key targets

### Phase K: Surpass MC4D

Goal:
- use the native Rust/GPU stack plus atomistic coupling to go beyond the published reference

Build steps:
52. Expand explicit operon/polycistronic mechanics and named assembly intermediates.
53. Improve membrane, transport, and division biophysics.
54. Add genotype-to-phenotype perturbation workflows.
55. Add cross-organism generalization.
56. Add high-throughput design loops for genome edits, media changes, and drugs.
57. Add adaptive promotion/demotion between coarse and atomistic regions during live runs.
58. Add learned physical surrogates compiled from repeated atomistic neighborhoods.

Exit criteria:
- the same stack supports more organisms, more explicit assemblies, and better local physical grounding than the MC4D baseline

## Immediate Build Order From Today

This is the exact order to execute from the current repository state:

1. Freeze the whole-cell runtime contract and serialization contract.
2. Build compiler-driven organism asset ingestion for Syn3A.
3. Replace bundled hand-maintained Syn3A descriptors with compiled assets.
4. Add explicit species and reaction registries.
5. Convert the current coarse process logic to compiled explicit reaction execution where possible.
6. Upgrade complex state into full named assembly graphs and intermediates.
7. Add explicit chromosome/fork/polymer state.
8. Add explicit membrane composition and divisome mechanics.
9. Replace the staged scheduler with explicit multirate orchestration.
10. Upgrade atomistic topology/parameter ingestion from templates to real subsystem extraction.
11. Build live atomistic refinement triggers and feedback into the whole-cell runtime.
12. Build the Syn3A calibration/validation bundle and regression suites.
13. Reproduce MC4D-equivalent observables.
14. Generalize to a second microbial organism.
15. Expand atomistic coverage and adaptive partitioning.

## Rust/Metal vs Python Responsibilities

Rust/Metal owns:

- authoritative runtime state
- reaction execution
- assembly state
- chromosome state
- membrane/division state
- multirate scheduler
- local atomistic extraction and feedback
- GPU kernels and restart state

Python owns:

- data ingestion and compilation tooling
- experiment orchestration
- reporting and artifact management
- external solver adapters where still needed
- calibration pipelines
- benchmark and regression harnesses

## Validation Gates

Each phase is not complete until all three are true:

1. state is explicit
2. restart/save/restore covers it
3. there is at least one test plus one benchmark or artifact proving it works

## Program Exit Criteria

This plan is complete only when:

- the organism is compiled from explicit microbial genome assets
- whole-cell state contains explicit named molecules and complexes
- chromosome and membrane/division mechanics are explicit subsystems
- key high-value subsystems are grounded by native atomistic refinement
- the runtime can reproduce and checkpoint a validated microbial cell cycle
- the system matches or exceeds MC4D on robustness and observable fidelity
