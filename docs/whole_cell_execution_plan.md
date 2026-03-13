# Whole-Cell Execution Plan

## Purpose

This is the single execution document for taking `oNeuro` from its current Rust-first whole-cell runtime to a genome-explicit, chemistry-explicit, atomistically grounded microbial simulator.

This document exists to end the stop-and-go loop.

- Work should proceed through the phases and work packages below without stopping after every substep.
- Stop only for a real blocker:
  - missing licensed or legally redistributable data
  - impossible hardware assumption
  - contradictory biological target
  - architecture decision that would permanently narrow the system

## Target End State

The target is not a naive full-cell all-atom trajectory for an entire cell cycle.

The target is:

- explicit microbial genome assets as the organism source of truth
- explicit runtime state for transcripts, proteins, complexes, metabolites, ions, lipids, chromosome state, membrane state, and geometry
- explicit reaction registries and assembly graphs driving the whole-cell runtime
- explicit multirate scheduling across RDME, CME, ODE, chromosome BD, membrane mechanics, and local atomistic refinement
- local atomistic extraction and feedback as a native source of truth for the highest-value subsystems
- restartable, benchmarked, calibrated runs that reproduce a validated microbial cell cycle
- MC4D parity first, then broader coverage and stronger atomistic grounding

## Current Achieved State

What is already built on the active path:

- strict structured-bundle organism compilation for bundled Syn3A plus a second demo organism
- native Rust ingestion of structured organism bundles and compiled process registries
- explicit runtime state for transcription units, RNA/protein execution pools, named complexes, chromosome state, membrane/division state, spatial fields, scheduler clocks, and restart payloads
- native registry writeback into authoritative pools, expression state, complex state, stress response, and repair
- registry-aware multirate scheduling across RDME, CME, ODE, chromosome BD, geometry, and atomistic-refinement stages
- explicit complex channel ownership from subsystem targets, family, and asset class
- direct process-capacity computation from explicit channel inventory, chemistry support, local pool state, and quantum efficiency instead of generic scalar-rule capacity wrappers
- direct CME, ODE, BD, and geometry stage flux or drive computation from explicit capacities and local signals instead of generic scalar-rule stage wrappers
- active stage execution now reads base local chemistry and explicit inventory directly, without an intermediate stage-rule context carrying derived channel surrogates
- base whole-cell resource signals now come directly from local pools, local chemistry, support, and pressure instead of generic resource-estimator rules
- diagnostic RNAP, ribosome, DnaA, and FtsZ pools no longer drive active replication or expression execution when explicit complex inventory is present
- explicit asset bundles no longer fall back to derived complex-target inventories on the live assembly inventory path
- explicit asset bundles now refresh diagnostic RNAP, ribosome, DnaA, and FtsZ summaries directly from explicit inventory instead of flux-blended surrogate rollups
- legacy-derived complex targets now prefer persisted complex assembly state before falling back to scalar-rule priors
- active scheduler, rule-context, assembly, BD, geometry, and spatial-field hot paths now read replicated fraction, division progress, surface area, radius, and chromosome separation from explicit chromosome or membrane state accessors instead of synchronized summary scalars
- snapshot, save-state, and public whole-cell progress or diagnostic getters now derive core chromosome, membrane, and diagnostic summary values from explicit state accessors on the active explicit-asset path instead of reusing stale synchronized scalar summaries
- explicit saved-state restore now uses synchronized scalar core fields as legacy or missing-state seeds, while the explicit-asset restore path rehydrates chromosome, membrane, and diagnostic state from explicit saved biology instead of stale core summaries
- bundle-less saved-state restore now also preserves explicit saved chromosome and membrane state when present, instead of always reseeding non-explicit restores from coarse core summary scalars
- bundle-less snapshot, save-state, and public diagnostic getters now also derive RNAP, ribosome, DnaA, and FtsZ summaries from persisted explicit `complex_assembly` state instead of stale surrogate pool scalars when no bundle assets are present
- program-spec bootstrap can now carry and preserve explicit `complex_assembly` or per-complex `named_complexes`, so non-saved-state initialization no longer has to reseed assembly and diagnostic state when explicit inventory is already provided
- program-spec bootstrap can now also carry explicit organism-expression state, so transcription-unit execution state and cached process-scale support no longer have to be regenerated from organism descriptors when a caller already has explicit expression state
- program-spec bootstrap can now also carry explicit runtime species, runtime reactions, and scheduler clocks, so non-saved-state initialization no longer has to regenerate runtime chemistry state or multirate clock state when those layers are already available
- program-spec bootstrap can now also carry explicit local-chemistry reports, site reports, subsystem probe state, and MD coupling scales, so non-saved-state initialization no longer has to reset local-chemistry support signals before expression, assembly, runtime chemistry, and scheduler fallback stages consume them
- program-spec bootstrap can now also carry explicit spatial fields, so non-saved-state initialization no longer has to regenerate membrane, septum, nucleoid, membrane-band, and pole locality when compiled or persisted field state is already available
- public snapshot and getter boundaries now expose explicit persisted local-chemistry state even when no live chemistry bridge is attached, so persisted chemistry support is no longer hidden behind an implementation detail of the current backend attachment

What is still not at the target:

- the remaining compatibility-only summary payloads and legacy serialization bridges still preserve synchronized scalar state for non-explicit paths that do not yet carry richer explicit persisted state for every biology layer, instead of deriving everything from explicit state at the boundary
- reaction/species coverage is still narrow relative to a full microbial cell
- membrane chemistry, chromosome mechanics, and local chemistry are still too coarse for parity
- atomistic refinement exists as infrastructure, but not yet as an authoritative live feedback service
- calibration, validation, and parity reporting are still incomplete

## Current Starting Position

What already exists in this repo:

- Rust whole-cell runtime in `oneuro-metal/src/whole_cell.rs`
- serialized whole-cell program and asset contracts in `oneuro-metal/src/whole_cell_data.rs`
- local chemistry, assembly-bridge, and local MD probe surfaces in `oneuro-metal/src/whole_cell_submodels.rs`
- bundled Syn3A native reference compiled from structured source files in `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/`
- atomistic template support in `oneuro-metal/specs/whole_cell_atomistic_templates.json` and `oneuro-metal/src/atomistic_topology.rs`
- Python orchestration surfaces in `src/oneuro/whole_cell`
- prior strategy and gap docs in:
  - `docs/whole_cell_strategy.md`
  - `docs/whole_cell_atomic_roadmap.md`
  - `docs/whole_cell_mc4d_gap.md`

What is partially present but not complete:

- explicit species, process-registry, expression, assembly, chromosome, membrane/division, and spatial runtime state
- strict structured-bundle compilation and native ingestion for bundled organisms
- native multirate scheduling and restart surfaces
- local chemistry and local atomistic refinement stubs that are not yet authoritative enough
- direct process-capacity ownership, but not yet direct stage-drive ownership end to end

What still does not exist:

- broader external organism ingestion beyond the current structured-bundle and demo-source set
- canonical full-cell species registry and canonical full-cell reaction registry at parity breadth
- explicit chromosome polymer mechanics at parity depth
- explicit membrane composition, membrane patch chemistry, and division mechanics at parity depth
- event-driven multirate solver scheduler with full checkpointing
- live atomistic extraction from runtime state with calibrated feedback loops
- full Syn3A dataset, calibration, validation, and parity reporting stack

## Non-Negotiable Architecture Rules

1. Rust remains the authoritative runtime for whole-cell execution.
2. GPU kernels stay in Metal or CUDA when they are part of the hot path.
3. Python owns ingestion, compilation, orchestration, reporting, validation, and adapters.
4. Every major biology layer must be explicit runtime state, not just metadata.
5. Every major biology layer must be restartable and serializable.
6. Every phase must add tests, artifacts, and benchmark outputs.
7. Every abstraction must compile down to explicit species, explicit state, or explicit schedule entries.
8. No phase is done until state, restart, and validation are all in place.

## Critical Path

This is the dependency chain that controls program completion:

1. Freeze runtime and IR contracts.
2. Build compiler-driven organism asset ingestion.
3. Build explicit species and reaction registries.
4. Drive expression, metabolism, transport, degradation, and repair from compiled registries.
5. Make assembly explicit.
6. Make chromosome and membrane/division explicit.
7. Replace staged stepping with a real multirate scheduler.
8. Make local atomistic refinement a live runtime service instead of a template-only side probe.
9. Build calibration and validation against Syn3A data and MC4D observables.
10. Reach parity.
11. Expand atomistic coverage and organism generality.

Everything else is either support work or parallelizable side work.

## Execution Rules

These rules are operational, not aspirational.

- Do not ask for permission for each work package.
- Finish a work package end to end:
  - schema or API
  - runtime integration
  - tests
  - artifact or benchmark
- Do not leave new biology layers only on the Python side if they belong in the runtime.
- Do not keep hand-authored heuristics once the compiled IR path exists for the same domain.
- Do not add atomistic machinery that cannot feed calibrated information back into the whole-cell runtime.
- Keep a running progress ledger in the repo as phases complete.

## Master Phase Table

| Phase | Name | Main Output | Depends On | Status |
|---|---|---|---|---|
| 0 | Contract Freeze | stable runtime, IR, and restart schemas | current repo | complete |
| 1 | Organism Compiler | compiler-produced Syn3A asset package | 0 | active on strict structured-bundle path |
| 2 | Species/Reaction IR | canonical species and reaction registries | 0, 1 | active, partial breadth |
| 3 | Expression Runtime | explicit transcription/translation/degradation execution | 1, 2 | active |
| 4 | Assembly Runtime | named complexes and assembly intermediates | 1, 2, 3 | active |
| 5 | Chromosome Runtime | explicit forks, loci, polymer state, topology | 1, 2, 3, 4 | active |
| 6 | Membrane/Division Runtime | explicit membrane composition, divisome, geometry | 2, 4, 5 | active |
| 7 | Spatial Chemistry Runtime | authoritative intracellular spatial chemistry | 2, 3, 4, 5, 6 | active |
| 8 | Atomistic Refinement Runtime | live local extraction and feedback | 2, 4, 5, 6, 7 | scaffold only |
| 9 | Multirate Scheduler | event-driven solver orchestration and checkpoints | 2, 3, 4, 5, 6, 7, 8 | active, needs deeper ownership |
| 10 | Calibration/Validation | dataset bundle, fitting, held-out tests | 1 through 9 | not complete |
| 11 | MC4D Parity | matched observables and restart quality | 10 | not started |
| 12 | Beyond Parity | broader atomistic coverage and extra organisms | 11 | not started |

## Active Todo Ladder

This is the direct task ladder from the current runtime to the full vision. The detailed work packages below remain authoritative; this section is the execution-facing todo.

1. Remove the remaining compatibility-only summary payload rollups and legacy serialization bridges, so synchronized scalar summaries become boundary serialization or diagnostics only and not live execution state even on compatibility paths without bundle assets or saved-state payloads. Program-spec bootstrap now preserves explicit spatial fields, local chemistry, expression, assembly, runtime chemistry, and scheduler state, and boundary getters now expose explicit persisted chemistry state without a live bridge; the remaining work is the narrower legacy compatibility surface.
2. Replace any remaining uses of scalar-rule inventory priors in non-asset compatibility code with persisted explicit state or explicit local inventories wherever possible.
3. Expand compiled species/reaction coverage for metabolites, ions, cofactors, lipids, damage states, repair states, and membrane-local species beyond the current narrow bundle set.
4. Move more reaction execution from generic process scales into explicit registry-driven reaction families with direct ownership of pools, transcripts, proteins, complexes, and local fields.
5. Make chromosome-domain chemistry fully compiled and local, not just domain-weighted over generic nucleoid pools.
6. Make membrane patch reactions explicit: synthesis, insertion, remodeling, turnover, stress, constriction support, and pole/septum differentiation.
7. Deepen membrane/division mechanics so geometry, patch composition, divisome occupancy, and chromosome occlusion are driven by explicit local state instead of mixed summary signals.
8. Deepen chromosome mechanics toward polymer-level behavior: fork barriers, supercoiling, topological strain, macrodomain constraints, segregation forces, and locus-local transcription/replication conflicts.
9. Expand spatial chemistry so RDME fields are authoritative for more resource and damage flows, with fewer well-mixed fallback shortcuts.
10. Turn atomistic refinement from a side probe into a live service that extracts local systems from runtime state, runs calibrated local chemistry or MD, and writes validated corrections back into the cell state.
11. Add organism-calibration bundles and benchmark datasets for Syn3A covering growth, ATP usage, transcript/protein abundance, division timing, chromosome progression, membrane composition, and perturbation responses.
12. Build parity reporting against MC4D-style observables and close each mismatch by improving explicit state and reaction ownership, not by reintroducing heuristic shortcuts.
13. After parity, generalize the same compiler/runtime path to additional organisms and broaden atomistic coverage to the highest-value subsystems.

## Phase 0: Freeze The Contract

### Goal

Stop architecture churn before deeper biology is added.

### Work Packages

1. Freeze authoritative units for:
   - molecule counts
   - concentrations
   - lattice coordinates
   - time
   - energy
   - force
   - geometry
2. Freeze the authoritative runtime layer boundaries:
   - organism assets
   - species registry
   - reaction registry
   - expression state
   - assembly state
   - chromosome state
   - membrane/division state
   - spatial state
   - atomistic local domains
   - observables and artifacts
3. Freeze canonical IR boundaries:
   - genome/transcription IR
   - species/reaction IR
   - assembly graph IR
   - chromosome/polymer IR
   - solver schedule IR
   - atomistic extraction IR
4. Freeze restart and provenance schema:
   - organism asset hash
   - compiled IR hash
   - calibration bundle hash
   - random seeds
   - backend and hardware info
   - run manifest
5. Split authoritative state structs from convenience summaries where that is still mixed.
6. Add schema tests for every serialized contract.
7. Add a progress ledger mapping each later phase to tests and artifacts.

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/whole_cell_data.rs`
- `src/oneuro/whole_cell/state.py`
- `src/oneuro/whole_cell/manifest.py`

### Exit Criteria

- no ambiguous unit surfaces remain
- restart payloads cover every authoritative layer
- later phases can add content without redefining the contract

## Phase 1: Build The Organism Compiler

### Goal

Replace hand-maintained bundled descriptors with compiler-produced organism assets.

### Work Packages

8. Add source ingestion for:
   - FASTA
   - GenBank
   - GFF or GTF
   - operon tables
   - promoter and terminator tables
   - protein annotation tables
   - complex composition tables
   - essentiality tables
   - media and composition priors
9. Define a normalized Python-side intermediate organism asset model.
10. Add sequence normalization and ID canonicalization.
11. Build feature extraction for:
   - coding genes
   - ncRNAs
   - promoters
   - terminators
   - origins
   - termini
   - ribosome-binding sites
   - motifs and structural tags
12. Build compiler passes for:
   - transcription units
   - operons
   - RNA products
   - protein products
   - complexes
   - pool priors
   - geometry priors
13. Make the compiler emit a single organism package with strict schema versioning.
14. Add package hashing and provenance capture.
15. Replace bundled Syn3A specs with compiler-emitted Syn3A assets.
16. Add a second organism target so the compiler is not Syn3A-hardcoded.
17. Add round-trip tests between source annotations and compiled assets.
18. Add compiler reports for:
   - missing genes
   - duplicated IDs
   - unresolved products
   - unresolved complex components
   - feature-coordinate conflicts
19. Add artifact generation:
   - organism summary
   - operon summary
   - RNA/protein/complex inventory summary

### Primary Code Targets

- `src/oneuro/whole_cell/artifacts.py`
- `src/oneuro/whole_cell/manifest.py`
- new `src/oneuro/whole_cell/assets/`
- `oneuro-metal/src/whole_cell_data.rs`
- `oneuro-metal/specs/`

### Exit Criteria

- Syn3A assets are compiler-produced
- second organism compiles through the same path
- no bundled organism file is maintained by hand beyond source-data patches

## Phase 2: Build Canonical Species And Reaction IR

### Goal

Turn chemistry from scattered heuristics into explicit registries.

### Work Packages

20. Define the canonical species registry:
   - metabolites
   - ions
   - cofactors
   - nucleotides
   - amino acids
   - lipids
   - RNAs
   - proteins
   - complexes
   - membrane species
   - damaged species
21. Define compartment and localization scopes for every species class.
22. Define the canonical reaction registry:
   - stoichiometry
   - compartments
   - catalysts
   - modifiers
   - reversibility
   - rate-law family
   - schedule affinity
23. Build compiler passes from organism assets into expression and assembly reactions.
24. Build compiler passes from curated metabolism and membrane chemistry into canonical reactions.
25. Build transport and degradation reaction compilers.
26. Build repair and stress chemistry compilers.
27. Add canonical IDs and name mapping across all reaction inputs.
28. Add runtime molecule-count state for all compiled species.
29. Replace any remaining coarse pool-only logic where explicit species now exist.
30. Add reaction validation:
   - mass-balance checks where possible
   - compartment legality
   - unknown species detection
   - catalyst existence checks
31. Add reaction summaries and inventory reports.

### Primary Code Targets

- `oneuro-metal/src/whole_cell_data.rs`
- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/whole_cell_submodels.rs`
- new species/reaction IR modules in `oneuro-metal/src`
- new compiler modules under `src/oneuro/whole_cell`

### Exit Criteria

- major chemistry is registry-driven
- explicit species counts exist for all compiled biology
- reaction execution no longer depends on scattered hard-coded branches for covered domains

## Phase 3: Build Explicit Expression Execution

### Goal

Move from shallow activity scaling to explicit transcription, translation, maturation, and degradation.

### Work Packages

32. Add promoter-level and transcription-unit-level initiation logic.
33. Add polymerase occupancy and elongation state.
34. Add RNA synthesis, maturation, and degradation bookkeeping.
35. Add ribosome occupancy and translation elongation state.
36. Add protein synthesis, folding, maturation, targeting, and degradation bookkeeping.
37. Add translation resource competition for amino acids, charged tRNAs, ATP, and GTP equivalents.
38. Add transcript and protein damage channels.
39. Add growth-rate and stress-sensitive transcription modifiers.
40. Add collision bookkeeping between transcription and replication interfaces.
41. Replace current coarse expression-rate scaling with explicit expression execution for compiled units.
42. Add restart payloads for RNAP occupancy, ribosome occupancy, elongation state, and partially completed products.
43. Add artifacts:
   - transcript counts over time
   - protein counts over time
   - occupancy histograms
   - degradation fluxes

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/whole_cell_data.rs`
- `oneuro-metal/src/gene_expression.rs`
- `oneuro-metal/src/gpu/gene_expression.rs`
- `oneuro-metal/src/metal/gene_expression.metal`

### Exit Criteria

- explicit transcript and protein dynamics drive expression
- expression restart state is complete
- expression observables can be benchmarked independently

## Phase 4: Build Explicit Assembly Runtime

### Goal

Replace aggregate capacities with named complexes and assembly intermediates.

### Work Packages

44. Define an assembly graph IR for each major complex family.
45. Add named intermediates for:
   - ribosomes
   - RNAP
   - replisomes
   - ATP synthase
   - transporters
   - membrane enzymes
   - chaperone clients
   - FtsZ and divisome modules
46. Add subunit recruitment and completion logic.
47. Add failed assembly, stalled assembly, and disassembly logic.
48. Add damage, repair, and degradation pathways for assembled complexes.
49. Add sequestration and competition across complexes sharing the same subunits.
50. Couple explicit assemblies to reaction capacity.
51. Couple explicit assemblies to membrane insertion state where applicable.
52. Couple explicit assemblies to chromosome and division state where applicable.
53. Remove fake unnamed capacity pools for covered systems.
54. Add artifacts:
   - assembly-state occupancy
   - limiting-subunit reports
   - failure-state counts

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/whole_cell_submodels.rs`
- new assembly modules in `oneuro-metal/src`

### Exit Criteria

- no major subsystem depends on anonymous aggregate capacity where an explicit assembly graph exists

## Phase 5: Build Explicit Chromosome Runtime

### Goal

Move from replication progress scalars to a live chromosome subsystem.

### Work Packages

55. Add explicit circular chromosome indexing over the compiled genome.
56. Add origin, terminus, replication forks, and replication initiation state.
57. Add fork progression, pausing, and completion bookkeeping.
58. Add collision-aware transcription-replication bookkeeping.
59. Add locus-level occupancy and accessibility state.
60. Add polymer-level chromosome representation:
   - beads
   - loci
   - domains
   - tether points
   - segregation tags
61. Add compaction and segregation state.
62. Add topology state where it materially affects behavior:
   - supercoiling
   - torsional stress
   - strand separation cost
63. Add chromosome restart payloads and artifacts.
64. Add chromosome-specific validation:
   - fork timing
   - locus separation
   - replication duration
   - collision statistics

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- new chromosome/polymer modules in `oneuro-metal/src`
- `src/oneuro/whole_cell/runner.py`

### Exit Criteria

- chromosome behavior is an explicit subsystem
- replication and segregation are no longer scalar placeholders

## Phase 6: Build Explicit Membrane And Division Runtime

### Goal

Tie shape and cytokinesis to actual molecular state.

### Work Packages

65. Add explicit lipid classes and membrane species inventories.
66. Add membrane-protein insertion state.
67. Add curvature-related state and septum-local composition state.
68. Add divisome assembly order and occupancy.
69. Add constriction mechanics driven by assembled divisome state.
70. Add membrane-growth coupling to geometry updates.
71. Add chromosome occlusion constraints on septation.
72. Add osmotic and volume constraints where they materially change geometry.
73. Add membrane and division restart payloads and artifacts.
74. Add validation on:
   - radius and surface area trajectories
   - constriction timing
   - division failure modes

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- membrane/division modules in `oneuro-metal/src`
- geometry-related kernels if promoted to GPU

### Exit Criteria

- geometry and division are driven by explicit molecular state

## Phase 7: Build Authoritative Spatial Chemistry

### Goal

Make intracellular spatial chemistry a real coupled subsystem instead of only a supporting approximation.

### Work Packages

75. Expand the intracellular lattice to carry authoritative species fields for the compiled chemistry subset that needs spatial treatment.
76. Define which chemistry remains well-mixed and which is spatial.
77. Add compartment and patch-local constraints:
   - membrane adjacency
   - nucleoid exclusion or occupancy
   - septum-local zones
   - ribosome or enzyme clusters
78. Couple reaction execution to spatial availability where needed.
79. Couple transport and membrane reactions to local membrane patches.
80. Couple chromosome-local processes to chromosome spatial state.
81. Add GPU kernels for the spatial hot path where needed.
82. Add spatial restart payloads and artifacts:
   - field slices
   - local concentration traces
   - crowding maps
   - compartment exchange rates

### Primary Code Targets

- `oneuro-metal/src/gpu/whole_cell_rdme.rs`
- `oneuro-metal/src/metal/whole_cell_rdme.metal`
- `oneuro-metal/src/whole_cell.rs`
- `oneuro-metal/src/whole_cell_submodels.rs`

### Exit Criteria

- spatial chemistry materially affects runtime outcomes for the domains assigned to spatial treatment

## Phase 8: Make Atomistic Refinement Native

### Goal

Promote atomistic local refinement from template-only probes to live runtime support.

### Work Packages

83. Expand atomistic topology ingestion beyond templates:
   - protein structures
   - RNA segments
   - protein-RNA complexes
   - membrane patches
   - metabolite and ion neighborhoods
84. Add force-field and parameter ingestion with validation.
85. Add local-domain builders that carve live subsystems out of runtime state.
86. Add trigger policies:
   - high uncertainty
   - rare events
   - assembly transitions
   - transport bottlenecks
   - division-zone instability
87. Add atomistic simulation metadata capture:
   - topology source
   - parameter source
   - local boundary conditions
   - extracted state hash
88. Add reducers that feed atomistic outputs back into the runtime as:
   - effective rates
   - energy penalties
   - state transitions
   - diffusion or localization corrections
   - assembly or degradation propensities
89. Add atomistic microbenchmarks against known structures and known local observables.
90. Add caches and surrogate compilation so repeated neighborhoods do not rerun full atomistic refinement unnecessarily.

### Primary Code Targets

- `oneuro-metal/src/atomistic_topology.rs`
- `oneuro-metal/src/molecular_dynamics.rs`
- `oneuro-metal/src/cuda/`
- `oneuro-metal/src/gpu/md_gpu.rs`
- `oneuro-metal/src/whole_cell_submodels.rs`

### Exit Criteria

- live whole-cell state can trigger, build, run, and consume local atomistic refinement

## Phase 9: Replace Staged Stepping With A True Multirate Scheduler

### Goal

Turn the scheduler into the real orchestration layer for the full multiscale cell.

### Work Packages

91. Define schedule IR with explicit clocks for:
   - RDME
   - CME
   - ODE
   - chromosome BD
   - membrane mechanics
   - atomistic refinement
92. Add event queues for:
   - replication initiation
   - fork completion
   - collision events
   - divisome transitions
   - rare chemistry triggers
   - atomistic trigger requests
93. Add deterministic ordering rules and seed handling.
94. Add partial checkpoints for every solver layer.
95. Add replayable manifests that capture the exact compiled state and calibration context.
96. Add long-run resume support without cross-solver drift.
97. Add distributed execution hooks for expensive atomistic jobs while keeping Rust authoritative.
98. Add scheduler benchmarks for:
   - throughput
   - checkpoint latency
   - replay fidelity

### Primary Code Targets

- `oneuro-metal/src/whole_cell.rs`
- new scheduler modules in `oneuro-metal/src`
- `src/oneuro/whole_cell/scheduler.py`
- `src/oneuro/whole_cell/runner.py`

### Exit Criteria

- every solver layer has its own cadence, checkpoint surface, and deterministic replay path

## Phase 10: Build Calibration And Validation

### Goal

Make the simulator measurable, fit, and defensible.

### Work Packages

99. Build the Syn3A reference dataset bundle:
   - genome
   - transcript data
   - protein data
   - metabolite data
   - growth data
   - replication timing
   - division timing
   - perturbation data
100. Build dataset versioning, hashing, and provenance.
101. Add per-module calibration pipelines for:
   - expression
   - metabolism
   - assembly
   - chromosome dynamics
   - membrane/division
   - local atomistic reducers
102. Add held-out validation sets for every calibration target.
103. Add sensitivity and ablation analysis.
104. Add per-module residual reports and uncertainty bands.
105. Add regression suites against overlapping MC4D observables.
106. Add calibration artifacts and dashboards to the repo reporting flow.

### Primary Code Targets

- new `src/oneuro/whole_cell/validation/`
- `src/oneuro/whole_cell/artifacts.py`
- `src/oneuro/whole_cell/runner.py`
- `docs/benchmarks/`
- `results/`

### Exit Criteria

- every major subsystem has a benchmark, a fit target, and a held-out validation report

## Phase 11: Reach MC4D Parity

### Goal

Match the published Syn3A baseline on observable behavior and run quality.

### Work Packages

107. Reproduce key observables:
   - cell-cycle timing
   - transcript distributions
   - protein distributions
   - metabolite support
   - replication timing
   - chromosome behavior
   - division progression
108. Match restartability and artifact completeness.
109. Match or exceed reproducibility and provenance quality.
110. Match solver coverage at the observable level.
111. Remove remaining heuristics that do not have explicit or fitted grounding.
112. Publish parity reports in-repo.

### Exit Criteria

- parity reports show overlap with the MC4D baseline on the chosen observable set

## Phase 12: Go Beyond Parity

### Goal

Use the native runtime to do things the reference stack does not do as well.

### Work Packages

113. Expand explicit assembly intermediates where the reference remains coarse.
114. Deepen membrane and transport biophysics.
115. Deepen division-zone mechanics and failure-state handling.
116. Add genotype-to-phenotype perturbation workflows.
117. Add media-perturbation and drug-perturbation workflows.
118. Compile at least two additional organisms through the same compiler path.
119. Expand atomistic coverage to a larger fraction of the proteome and membrane machinery.
120. Add adaptive promotion and demotion between coarse and atomistic treatment during live runs.
121. Add learned physical surrogates compiled from repeated atomistic neighborhoods.

### Exit Criteria

- the same stack supports multiple organisms and broader atomistic grounding than the MC4D baseline

## Workstream Mapping

These workstreams can run in parallel once their dependencies are met.

### Workstream A: Contracts And Schemas

- Phase 0
- restart payloads
- provenance
- artifact manifests

### Workstream B: Organism Compiler

- Phase 1
- source ingestion
- normalization
- package emission

### Workstream C: Runtime Chemistry

- Phase 2
- Phase 3
- Phase 4

### Workstream D: Physical Cell State

- Phase 5
- Phase 6
- Phase 7

### Workstream E: Atomistic Integration

- Phase 8
- Phase 9

### Workstream F: Calibration And Parity

- Phase 10
- Phase 11
- Phase 12

## Exact Immediate Build Order From Today

This is the concrete near-term order to execute from the current repo state.

1. Freeze the whole-cell runtime contract.
2. Freeze the restart and provenance contract.
3. Add the progress ledger.
4. Build Python-side organism ingestion modules.
5. Compile Syn3A assets from source data instead of hand-maintained bundled descriptors.
6. Compile a second organism through the same pipeline.
7. Introduce the canonical species registry.
8. Introduce the canonical reaction registry.
9. Migrate compiled expression reactions onto the registry.
10. Migrate transport and degradation reactions onto the registry.
11. Migrate membrane and repair chemistry onto the registry.
12. Promote transcript and protein execution to explicit elongation and degradation state.
13. Promote complex state to named assembly graphs and intermediates.
14. Add chromosome fork and locus state.
15. Add chromosome polymer state.
16. Add membrane composition and divisome state.
17. Promote spatial chemistry to authoritative status for the chosen species subset.
18. Add live atomistic domain extraction from runtime state.
19. Add atomistic feedback reducers that update the runtime.
20. Replace staged stepping with a true multirate scheduler.
21. Build the Syn3A reference dataset bundle.
22. Build calibration and held-out validation.
23. Run parity reports.
24. Generalize to more organisms and deeper atomistic coverage.

## Phase Completion Gate

No phase is complete until all of these are true:

1. state is explicit
2. restart/save/restore covers it
3. one test proves the contract
4. one benchmark or artifact proves it runs
5. one validation report proves it is not just internally consistent

## Progress Ledger Format

Every completed work package should add a short ledger entry with:

- date
- phase and work package number
- files changed
- tests run
- artifacts produced
- remaining dependency blockers

## Program Exit Criteria

This program is complete only when all of the following are true:

- organism state is compiled from explicit microbial genome assets
- runtime state contains explicit named molecules, RNAs, proteins, complexes, chromosome state, and membrane/division state
- chemistry is driven by compiled species and reaction registries
- the scheduler is multirate, checkpointable, and replayable
- atomistic refinement is a live calibrated truth source for selected subsystems
- a validated microbial cell cycle can be reproduced and restarted
- parity with the published Syn3A reference has been demonstrated
- the same stack can then extend beyond that reference to additional organisms and broader atomistic coverage
