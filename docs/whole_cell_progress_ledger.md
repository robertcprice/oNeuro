# Whole-Cell Progress Ledger

Use this ledger to record completed work packages from `docs/whole_cell_execution_plan.md`.

## Entry Template

### YYYY-MM-DD - Phase X / Work Package N

- Summary:
- Files changed:
  - `path/to/file`
- Tests run:
  - `command`
- Artifacts produced:
  - `path/to/artifact`
- Remaining blockers:
  - blocker or `none`

## Entries

### 2026-03-11 - Phase 0 / Work Packages 1-7

- Summary:
  - froze the first explicit whole-cell contract slice in code by adding schema/provenance metadata to Python whole-cell program, manifest, and state surfaces plus Rust whole-cell program and saved-state payloads
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/__init__.py`
  - `src/oneuro/whole_cell/architecture.py`
  - `src/oneuro/whole_cell/contracts.py`
  - `src/oneuro/whole_cell/manifest.py`
  - `src/oneuro/whole_cell/state.py`
  - `tests/test_whole_cell.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/contracts.py src/oneuro/whole_cell/architecture.py src/oneuro/whole_cell/manifest.py src/oneuro/whole_cell/state.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `none for Phase 0 contract freeze; next execution slice is Phase 1 organism compiler ingestion`

### 2026-03-11 - Phase 1 / Work Packages 8-19

- Summary:
  - added a manifest-driven Python organism bundle compiler that emits whole-cell organism specs and derived genome asset packages against the frozen Rust contract, with Syn3A compiling through a declared source bundle and a second demo organism exercising FASTA and GFF ingestion
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/__init__.py`
  - `src/oneuro/whole_cell/assets/__init__.py`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/metadata.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/genome.fasta`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/features.gff3`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/gene_products.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/transcription_units.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/pools.json`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py tests/test_whole_cell.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `native Rust ingestion of these source bundles is still the next step; the current compiler is a POC that targets the frozen Rust contract`

### 2026-03-11 - Phase 1 / Native Bundle Ingestion Slice

- Summary:
  - added native Rust ingestion for source-bundle manifests so the whole-cell runtime can compile organism specs, program specs, and genome asset packages directly from declared bundle sources, and exposed that path through the PyO3 simulator bindings
  - tracked the bundled Syn3A native reference/spec JSON assets required by clean Rust builds
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_atomistic_templates.json`
  - `oneuro-metal/specs/whole_cell_derivation_calibration.json`
  - `oneuro-metal/specs/whole_cell_subsystems.json`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `oneuro-metal/specs/whole_cell_syn3a_reference.json`
  - `oneuro-metal/src/python.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `native source-bundle ingestion is in place; the next execution slice is compiling richer explicit species and reaction registries from these bundle manifests`

### 2026-03-11 - Phase 2 / Canonical Registry Persistence Slice

- Summary:
  - made the canonical whole-cell process registry explicit program and saved-state data instead of a purely derived runtime view, and propagated its hash into provenance as the compiled IR identity
  - added native registry JSON compilation for bundle manifests and exposed the new registry surfaces through the Rust and PyO3 APIs
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/python.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-registry && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-registry && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-registry && PYTHONPATH=src python - <<'PY' ...`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the process registry is now persisted and hashed, but the execution kernels still need deeper migration from registry-informed runtime state into fully registry-driven transport, degradation, repair, and multirate scheduling`

### 2026-03-11 - Phase 2 / Transport And Degradation Bridge Slice

- Summary:
  - extended the native whole-cell process registry with explicit bulk-field bindings for pool species plus compiled `pool_transport`, `rna_degradation`, and `protein_degradation` reactions
  - wired registry pool deltas back into the authoritative scalar and lattice-backed metabolite fields so registry chemistry now changes execution state instead of only mutating shadow species counts
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-transport.C9ElA8 && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-transport.C9ElA8 && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `registry-driven repair chemistry and the multirate scheduler still need to move onto this same authoritative execution path`

### 2026-03-11 - Phase 2 / Repair And Stress Writeback Slice

- Summary:
  - extended the native whole-cell process registry with explicit `stress_response` and `complex_repair` reactions keyed to operons, and propagated those classes through the runtime summary and JSON surfaces
  - bridged registry RNA, protein, and complex-species deltas back into the authoritative expression and named-complex state so transcription, degradation, stress response, and repair now persist past the next species-sync pass
  - wired stress-response extent into real metabolic-load relief and unit-level stress/support updates, and wired complex-repair extent into real named-complex abundance and subunit-pool changes
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-repair.kZAVlw && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-repair.kZAVlw && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase2-repair.kZAVlw && PYTHONPATH=src python - <<'PY' ...`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the registry now reaches metabolism, expression, degradation, stress, and repair; the next execution slice is moving scheduler cadence from fixed staged intervals to registry-aware multirate orchestration`

### 2026-03-11 - Phase 3 / Registry-Aware Multirate Scheduler Slice

- Summary:
  - replaced fixed `% interval` stage dispatch in the native whole-cell runtime with explicit per-stage scheduler clocks for atomistic refinement, RDME, CME, ODE, chromosome BD, and geometry
  - persisted scheduler state through snapshots and saved-state JSON, normalized clock restoration across restart boundaries, and removed the CME double-count path so accumulated stage `dt` now flows through the runtime once
  - added restart and stress-driven cadence tests that verify the new scheduler can preserve clock state and re-fire CME or ODE earlier than the static config interval under load
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `oneuro-metal/src/whole_cell_submodels.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase3-scheduler.4uKnA5 && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase3-scheduler.4uKnA5 && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the scheduler is now explicit and restartable, but local chemistry probes still use simple interval checks inside the atomistic stage and the next execution slice is deeper registry-driven solver ownership plus finer multiscale coupling`

### 2026-03-11 - Phase 3 / Explicit Expression Execution Slice

- Summary:
  - extended transcription-unit runtime state with explicit promoter openness, RNAP occupancy, transcription progress, ribosome occupancy, translation progress, and mature/nascent/damaged RNA and protein pools
  - replaced the old direct abundance-only update path with occupancy-driven transcription and translation execution at the compiled transcription-unit layer while keeping restart compatibility through the existing organism-expression payload
  - added progression and restart tests that verify the explicit execution state advances during simulation and round-trips through saved-state JSON without drift
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase4-expression.12904 && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase4-expression.12904 && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `expression now carries explicit unit execution state, but assembly, chromosome, and membrane layers still depend on aggregate runtime channels and need the same treatment in later slices`

### 2026-03-11 - Phase 4 / Explicit Assembly Traits Slice

- Summary:
  - extended compiled complex specs with assembly-family traits and coupling flags so membrane, replication, transport, ribosome, RNAP, and divisome assemblies carry semantic runtime constraints instead of being treated as uniform channels
  - extended named complex runtime state with stall, damage, limiting-component, shared-component-pressure, insertion-progress, and failure bookkeeping and fed those signals into the native assembly update path
  - added family-aware assembly gating and shared-subunit competition so aggregate capacity now reflects more local assembly rules before it rolls up into process capacity
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase5-assembly.8470 && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase5-assembly.8470 && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `named assemblies now carry richer local state, but chromosome runtime, membrane mechanics, and explicit solver ownership still need to take over more of the remaining aggregate process channels`

### 2026-03-11 - Phase 7 / Metadata-First Pool Classification Slice

- Summary:
  - switched pool-species registry classification to prefer explicit `bulk_field` metadata for asset-class and compartment selection before any legacy string heuristic fallback
  - added an opaque-pool regression so pools with non-semantic names still compile into the correct membrane/cytosol execution path when their metadata is explicit
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy pool-field inference still exists as a compatibility fallback when bundle metadata is absent; the next slice is removing more of that fallback chain from registry compilation and runtime anchoring`

### 2026-03-11 - Phase 7 / Pool Metadata Ingress Normalization Slice

- Summary:
  - moved legacy pool-field inference to organism and asset-package ingress so compiler and program-spec construction operate on normalized explicit pool metadata instead of re-inferring fields repeatedly
  - added a regression proving legacy asset-package JSON without `bulk_field` still reparses into normalized explicit pool metadata at the boundary
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `runtime pool seeding and species anchoring still retain a compatibility name-based fallback for legacy callers that construct pool specs without explicit metadata`

### 2026-03-11 - Phase 7 / Runtime Pool Metadata Boundary Slice

- Summary:
  - removed opportunistic pool-field name inference from normal runtime seeding and runtime-species synchronization, so execution now follows explicit `bulk_field` metadata in the hot path
  - kept legacy compatibility only in one-time boundary backfills for organism pool initialization and old runtime species payloads
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `diagnostic and subsystem inventory seeding still uses legacy name checks for ribosome/RNAP/DnaA/FtsZ pools because those pools do not yet carry compiled role metadata`

### 2026-03-11 - Phase 7 / Explicit Pool Role Metadata Slice

- Summary:
  - added explicit pool-role metadata for ribosome, RNAP, DnaA, and FtsZ diagnostic pools and normalized that metadata at the same asset ingress boundary as bulk fields
  - removed runtime diagnostic seeding from string matching so opaque pool names can seed those subsystems when their compiled role metadata is explicit
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/whole_cell.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `broader asset-class, subsystem-target, and patch-domain inference for genes, operons, and complexes still falls back to name heuristics when bundle metadata is absent`

### 2026-03-11 - Phase 7 / Explicit Operon Metadata Slice

- Summary:
  - added explicit transcription-unit and operon metadata for subsystem targets, asset class, and complex family so compiled complexes can inherit those semantics directly instead of re-inferring them from names
  - updated the bundled Syn3A and demo bundle transcription-unit assets to carry that metadata and kept the Python bundle compiler aligned with the same explicit metadata path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/transcription_units.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `gene-level asset and family inference still falls back to names when source annotations do not carry explicit semantic metadata`

### 2026-03-11 - Phase 7 / Explicit Gene Product Metadata Slice

- Summary:
  - added explicit gene-product asset-class and complex-family metadata so protein compilation and singleton operons can inherit semantics directly from annotations instead of recovering them from names
  - extended the bundled Syn3A quality-control gene and the demo gene-product annotations to carry that metadata, and aligned the Python bundle compiler and Rust registry compiler with the same explicit gene path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/gene_products.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `some fallback family and asset inference still remains for completely unannotated legacy genes and bundles; the next slice is either explicit source coverage expansion or replacing those last heuristics with compiled semantic maps`

### 2026-03-11 - Phase 7 / Organism Semantic Boundary Normalization Slice

- Summary:
  - moved legacy gene and transcription-unit semantic inference to organism parse/ingress normalization so asset class, complex family, and aggregated subsystem targets are made explicit before later compilation stages
  - added a regression proving a sparse legacy division operon/gene spec reparses into explicit semantics at the boundary instead of relying on later compile-time name heuristics
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `asset-package and higher-level compatibility paths still retain some fallback semantic inference for legacy payloads that arrive without explicit compiled metadata`

### 2026-03-11 - Phase 7 / Asset-Package Semantic Boundary Slice

- Summary:
  - added compatibility normalization for compiled asset packages so sparse legacy operon semantics are rebuilt from protein/complex structure and carried forward explicitly before registry compilation
  - backfilled operon subsystem targets plus operon/complex asset class, family, and coupling flags for legacy package payloads at parse time
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `fully unannotated legacy genes and operons still require fallback semantic recovery; remaining work is narrowing or replacing those last heuristic defaults with explicit compiled semantic maps`

### 2026-03-11 - Phase 5 / Explicit Chromosome Runtime Slice

- Summary:
  - added explicit restartable chromosome state with live fork, locus, initiation, pause, collision, torsional-stress, compaction, and segregation bookkeeping in the native Rust whole-cell contract
  - moved whole-cell replication progress and gene copy/accessibility control onto that chromosome subsystem so expression now reads explicit chromosome state instead of only the old replicated-fraction heuristic
  - added chromosome restart and collision tests and propagated chromosome state through native snapshots, saved-state JSON, and bundled calibration fixtures
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `oneuro-metal/src/whole_cell_submodels.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase6-chromosome.18395 && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase6-chromosome.18395 && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `membrane/division runtime and deeper native solver ownership still need to replace the remaining aggregate geometry and atomistic orchestration channels`

### 2026-03-11 - Phase 6 / Explicit Membrane and Division Runtime Slice

- Summary:
  - added explicit restartable membrane/division state with lipid inventories, insertion debt, curvature stress, septum localization, divisome occupancy/order, ring mechanics, envelope integrity, osmotic balance, chromosome occlusion, and scission bookkeeping
  - moved geometry and division progression onto that membrane/division subsystem so surface area, volume, and division progress are now synchronized summaries of explicit local membrane state instead of the old scalar geometry channel
  - coupled membrane growth and constriction to local primitive signals from membrane precursors, membrane complexes, FtsZ/divisome assemblies, crowding, chromosome occlusion, and envelope stress, and added restart and occlusion tests around the new subsystem
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `oneuro-metal/src/whole_cell_submodels.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `membrane and division now flow through explicit local state, but envelope turnover, cell-wall synthesis, and division completion still need deeper native ownership in the next slice`

### 2026-03-11 - Phase 7 / Spatial Field Coupling Slice

- Summary:
  - added explicit membrane-adjacency, septum-zone, and nucleoid-occupancy spatial fields on the native intracellular lattice and persisted them through saved-state JSON as restartable spatial artifacts
  - upgraded the Rust and Metal RDME path so diffusion, source, and sink terms now depend on those spatial fields plus local membrane/chromosome demand instead of only uniform whole-cell means
  - coupled chromosome and membrane rule inputs to localized nucleotide, membrane-precursor, and ATP availability so spatial chemistry now changes replication and constriction support through local fields rather than ad hoc direct overrides
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/gpu/whole_cell_rdme.rs`
  - `oneuro-metal/src/metal/whole_cell_rdme.metal`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `spatial chemistry now carries authoritative local fields for the current lattice species, but membrane patch reactions, richer compartment scopes, and chromosome-local execution still need broader compiled-species coverage in later Phase 7 slices`

### 2026-03-11 - Phase 7 / Compiled Spatial Scope Registry Slice

- Summary:
  - added explicit `spatial_scope` metadata to compiled whole-cell species and reactions so locality is now declared by organism assets and carried through registry, runtime-state, and restart payloads
  - replaced late bulk-anchor and reaction-locality branching with generic spatial-scope caches, overlap coefficients, and weighted lattice deltas so runtime localization comes from compiled scope plus spatial fields instead of stage-specific special cases
  - extended registry validation so nucleoid-local, membrane-adjacent, and septum-local entities are asserted directly in the compiled Syn3A process registry
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/gpu/whole_cell_rdme.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled locality now flows through registry and runtime reaction ownership, but membrane patch reactions, richer compartment scopes, and broader chromosome-local chemistry still need to be pushed onto the same primitive-driven path`

### 2026-03-11 - Phase 7 / Patch-Local RDME Drive Slice

- Summary:
  - replaced the coarse RDME sink/source scalar handoff with dynamic local drive fields for energy source, ATP demand, amino demand, nucleotide demand, membrane source, membrane demand, and crowding
  - built those drive fields from compiled species/reaction ownership plus patch-local chemistry site reports, so membrane and chromosome chemistry now feed the lattice through localized primitive fields instead of a few global knobs
  - extended the native Rust and Metal RDME paths to consume those per-voxel drive fields and added scope-aware validation for the new compiled-source / compiled-demand behavior
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/gpu/whole_cell_rdme.rs`
  - `oneuro-metal/src/metal/whole_cell_rdme.metal`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `RDME now sees compiled and patch-local drive fields, but richer compartment types, explicit membrane patch turnover chemistry, and broader chromosome-local species ownership still need to be moved off the remaining coarse rule channels`

### 2026-03-11 - Phase 7 / Compiled Patch Domain And Membrane Patch Turnover Slice

- Summary:
  - added explicit compiled `patch_domain` metadata for whole-cell species and reactions, with native runtime-state carry-through so membrane bands, poles, septum patches, and nucleoid tracks are declared by compiled biology rather than inferred ad hoc inside solver stages
  - extended restartable spatial state with membrane-band and pole fields, and used the same generic locality weighting path to drive RDME demand/source fields and bulk-pool lattice writes from `spatial_scope + patch_domain` instead of scope alone
  - added explicit membrane patch turnover state for membrane bands, poles, and septum, then coupled the membrane runtime to localized precursor, ATP, demand, source, and crowding signals so patch inventories and turnover pressure evolve from local chemistry rather than a single aggregate envelope pool
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/gpu/whole_cell_rdme.rs`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled patch domains now drive localized membrane behavior, but membrane patch reactions, richer compartment scopes, and chromosome-local execution still need broader compiled-species coverage and tighter multiscale coupling in later slices`

### 2026-03-11 - Phase 7 / Compiled Membrane Patch Precursor Reactions Slice

- Summary:
  - compiled explicit membrane patch precursor pool species for membrane bands, poles, and septum patches, so localized precursor ownership now lives in the organism process registry instead of being inferred only inside the membrane updater
  - added compiled `membrane_patch_transfer` and `membrane_patch_turnover` reactions, then extended the native spatial coupling cache so bulk-field concentration and species availability are resolved from both `spatial_scope` and `patch_domain`
  - fixed registry bulk-field delta application to use participant locality rather than reaction locality, so patch transfer now moves precursor mass into patch-local domains instead of subtracting and re-adding inside the same locality bucket
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q bundled_syn3a_process_registry_compiles_from_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_membrane_patch_transfer_moves_precursors_into_band_zone --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && cd /tmp/oNeuro-phase7-membrane.5ezW2U && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `membrane precursor ownership now runs through compiled patch-local reactions, but ATP, nucleotide, amino, and broader compartment-local chemistry still need to be pushed onto the same generic compiled-locality path`

### 2026-03-11 - Phase 7 / Compiled Localized Cofactor Pools Slice

- Summary:
  - added compiled localized ATP, amino-acid, and nucleotide pool species plus localized transfer/turnover reactions, so compartment-local support chemistry is now emitted from the organism process registry instead of staying implicit behind only global bulk pools
  - rewired localized transcription, translation, degradation, stress-response, and repair reactions to consume or release those localized support pools directly when their compiled `spatial_scope` or `patch_domain` is not global, so local chemistry now follows the same generic compiled-locality path as the reactions that need it
  - extended the native runtime with generic localized-pool hinting and a nucleoid-local transfer regression test, so localized support pools pull mass into active local domains through shared drive fields and shared locality primitives instead of another membrane-only special case
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q bundled_syn3a_process_registry_compiles_from_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_localized_pool_transfer_moves_nucleotides_into_nucleoid_track --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `localized support pools now exist for ATP, amino acids, and nucleotides, but membrane patch chemistry still has a dedicated path and broader chromosome-local execution still needs more reactions to consume compiled local pools instead of global fallbacks`

### 2026-03-11 - Phase 7 / Unified Membrane Localized Pool Slice

- Summary:
  - removed the dedicated compiler-only membrane precursor transfer block and moved membrane precursor localization onto the same compiled localized-pool species and reaction machinery already used for ATP, amino acids, and nucleotides
  - added generic membrane-local request accumulation from compiled species and reactions, so membrane-band and septum-local precursor pools now emerge from compiled locality and membrane association instead of a bespoke patch-only code path
  - updated the native regression coverage so membrane precursor transfer is now tested through the generic localized-pool runtime path rather than the old membrane-only reaction class
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q bundled_syn3a_process_registry_compiles_from_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_localized_pool_transfer_moves_membrane_precursors_into_band_zone --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `membrane precursor localization now uses the generic localized-pool path, but membrane-only runtime hints/classes still exist for backward compatibility and broader chromosome-local execution still needs more compiled reactions to consume local pools instead of global fallbacks`

### 2026-03-11 - Phase 7 / Generic Legacy Membrane Runtime Hint Slice

- Summary:
  - removed the dedicated membrane-only runtime hint calculation from the reaction update path and routed legacy `membrane_patch_transfer` / `membrane_patch_turnover` reactions through the same generic localized-pool hinting used by the compiled localized support reactions
  - kept the legacy reaction classes available for backward-compatible state loading, but collapsed their control logic onto the shared locality machinery so the runtime no longer needs a second membrane-specific controller to keep old payloads working
  - revalidated the Rust whole-cell suites and Python whole-cell tests after the runtime unification so the compatibility path now rides on the same generic locality behavior as the new compiled registry
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q gpu::whole_cell_rdme --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the runtime now shares one locality-control path for new and legacy localized support reactions, but the remaining Phase 7 work is broader chromosome-local execution and eventually dropping the legacy membrane-only reaction enums once saved-state compatibility is formally versioned`

### 2026-03-11 - Phase 7 / Localized Assembly Energy Slice

- Summary:
  - added explicit ATP consumption to compiled subunit-pool formation, nucleation, elongation, maturation, and turnover reactions so complex assembly/disassembly no longer bypasses the localized support chemistry that was added in the earlier Phase 7 slices
  - routed those new ATP reactants through the localized pool compiler, which makes chromosome-local assembly in the bundled Syn3A registry consume `pool_nucleoid_track_atp` rather than falling back to the global ATP pool
  - revalidated the registry, Rust runtime, and Python whole-cell bindings after the assembly-energy change so the new local-energy dependency is now part of the tested compiled execution path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q bundled_syn3a_process_registry_compiles_from_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled assembly now consumes localized ATP, but broader chromosome-local execution still needs more chromosome-coupled reactions and solver updates to depend on compiled local pools rather than global scalar fallbacks`

### 2026-03-11 - Phase 7 / Chromosome-Local Expression Support Slice

- Summary:
  - added `localized_nucleoid_atp_pool_mm()` and blended nucleoid-local ATP into the rule-context energy signal so chromosome-coupled control signals respond to local energetic state instead of leaning only on global ATP and membrane-adjacent support
  - updated organism expression support to derive chromosome-localized supply, energy support, and nucleotide support from nucleoid-local ATP and nucleotide pools before scoring transcription units, which makes replication-cycle and other chromosome-coupled units track compiled local chemistry more directly
  - added a regression that biases ATP into nucleoid voxels versus pole voxels and verifies higher support and effective activity for the `replication_cycle_operon` under the nucleoid-loaded state
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_nucleoid_atp_localization_biases_replication_unit_support --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_nucleoid_localization_biases_nucleotide_signal --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_organism_expression_state_responds_to_energy_and_load_stress --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `chromosome-coupled expression now reads nucleoid-local ATP and nucleotide state, but more chromosome-linked reactions still need to move onto compiled local pools and away from residual global scalar support paths`

### 2026-03-11 - Phase 7 / Chromosome Runtime Local Support Slice

- Summary:
  - moved chromosome initiation and fork-progression support further onto local chemistry by adding reusable `chromosome_local_energy_support()` and `chromosome_local_nucleotide_support()` helpers and blending them directly into `advance_chromosome_state()`
  - removed the unconditional minimum 1 bp fork crawl when `replication_drive > 0`, so fork motion can now actually stall under poor local nucleoid ATP/nucleotide support instead of advancing through starvation by hardcoded fallback
  - added a chromosome-runtime regression that redistributes ATP and nucleotides into nucleoid versus pole voxels and verifies that the nucleoid-loaded state receives stronger chromosome-local support and a stronger fork-initiation response
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_nucleoid_localization_biases_chromosome_progress --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_chromosome_state_tracks_forks_loci_and_restart --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_head_on_transcription_increases_chromosome_collision_pressure --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `chromosome runtime now responds more directly to local nucleoid chemistry, but fork and locus behavior still use coarse whole-nucleoid averages rather than per-domain compiled local pools and domain-resolved chromosome chemistry`

### 2026-03-11 - Phase 7 / Domain-Resolved Chromosome Locality Slice

- Summary:
  - introduced sequence-domain-aware chromosome locality by deriving four dynamic axial domain weight maps from the existing nucleoid spatial field instead of treating the whole nucleoid as one chemistry bucket
  - added domain-resolved ATP, nucleotide, and localized-supply helpers and cached those per domain inside both `advance_chromosome_state()` and `refresh_organism_expression_state()`, so fork dynamics and per-unit expression support now respond to chromosome position rather than only a whole-nucleoid mean
  - added regressions for domain-loaded local pools and for left-vs-right domain loading flipping support in low-midpoint versus high-midpoint transcription units
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_chromosome_domain_support_tracks_domain_loaded_pools --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_chromosome_domain_loading_biases_expression_by_midpoint --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_nucleoid_localization_biases_chromosome_progress --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `chromosome locality is now domain-resolved, but the domains are still inferred from dynamic axial weights rather than explicit compiled chromosome-domain chemistry assets and domain-specific reaction ownership`

### 2026-03-11 - Phase 7 / Compiled Chromosome Domain Ownership Slice

- Summary:
  - added explicit `chromosome_domains` to the native organism spec, compiled asset package, and process registry, plus `chromosome_domain` ownership on compiled species and reactions so chromosome locality is now part of the Rust IR instead of only runtime inference
  - taught the Rust bundle compiler and hydration path to normalize or derive chromosome-domain assets, backfill older program and saved-state payloads, and propagate domain ownership into operon-, gene-, and complex-linked reactions without requiring a separate handwritten runtime map
  - switched the whole-cell runtime to read compiled domain intervals and axial centers when assigning loci, fork support, and expression support, while keeping the old fixed-slice logic only as a fallback when compiled domains are absent
  - updated the Python bundle POC compiler to emit the same compiled chromosome-domain metadata so the bundle artifacts used for debugging and inspection stay aligned with the native Rust/Metal execution path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo fmt --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_compiled_chromosome_domain_centers_bias_weight_peaks --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `chromosome domain ownership is now compiled into the IR, but the domain chemistry itself is still derived from generic source geometry rather than richer organism-specific domain assets and domain-resolved reaction datasets`

### 2026-03-11 - Phase 7 / Chromosome Domain RDME Coupling Slice

- Summary:
  - wired compiled `chromosome_domain` ownership into `locality_weights()`, so nucleoid-local species and reactions can now shape RDME drive fields through their explicit compiled domain IDs instead of only broad `NucleoidLocal` or `NucleoidTrack` scopes
  - added a regression that assigns the same replication-driving reaction to the first versus last compiled chromosome domain and verifies that nucleotide demand shifts into the corresponding domain-weighted chemistry field
  - kept the new coupling local to the native Rust runtime so the compiled chromosome domains now affect actual chemistry placement without introducing another hardcoded side channel
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs`
  - `cargo test -q test_rdme_drive_fields_follow_compiled_chromosome_domains --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled chromosome domains now bias RDME chemistry fields, but bulk-delta application and broader solver ownership still need to carry domain IDs through the remaining localized pool and reaction execution paths`

### 2026-03-11 - Phase 7 / Chromosome Domain Bulk Transfer Slice

- Summary:
  - carried compiled `chromosome_domain` ownership through localized pool hinting, runtime reactant availability, and bulk-field delta application so reaction execution now preserves the domain that requested nucleoid-local chemistry instead of collapsing back to generic nucleoid-wide transfer
  - added domain-aware bulk concentration and drive-field readers that reuse the same compiled domain weight maps, keeping reaction satisfaction, localized transfer hints, and weighted lattice deltas on one shared locality basis
  - added a regression that runs the same localized nucleotide-transfer reaction against the first versus last compiled chromosome domain and verifies that the nucleotide pool shifts into the matching compiled domain after execution
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs`
  - `cargo test -q test_rdme_drive_fields_follow_compiled_chromosome_domains --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_localized_pool_transfer_preserves_compiled_chromosome_domain --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled chromosome domains now survive localized pool execution, but the compiler still emits generic nucleoid-track pool species and reactions rather than explicitly domain-scoped localized pool assets`

### 2026-03-11 - Phase 7 / Domain-Scoped Localized Pool Compiler Slice

- Summary:
  - widened localized-pool compiler identity to carry optional compiled `chromosome_domain` ownership and emit explicit domain-scoped nucleoid-local pool species plus localized transfer/turnover reactions instead of collapsing chromosome-local chemistry into one anonymous nucleoid-track pool
  - constrained domain-scoping to chromosome-local and nucleoid-local pool requests so membrane and septum pool assets stay generic while chromosome-local chemistry now preserves the compiled domain boundary all the way from registry generation into runtime execution
  - added regressions that require bundled Syn3A to compile domain-scoped nucleoid-track nucleotide pools and matching localized transfer/turnover reactions, keeping the new ownership path under native test coverage
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_rdme_drive_fields_follow_compiled_chromosome_domains --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiled chromosome domains now shape localized pool asset generation, but broader localized support-pool chemistry and per-domain reaction ownership still need to move off generic fallback pools and onto richer compiled chromosome-domain datasets`

### 2026-03-11 - Phase 7 / Localized Support-Pool Turnover Slice

- Summary:
  - extended the generic localized-pool primitive to support explicit ADP localization requests, including transfer/turnover kinetics and request-weight inference from the same asset-class rule set used for ATP, so chromosome-local energy turnover no longer has to collapse back to a global product pool when the source bundle exposes ADP
  - switched stress-response and chromosome-local assembly/repair/turnover product generation onto localized ADP participants, keeping ATP consumption and ADP return on the same domain-aware locality path instead of mixing local reactants with global products
  - added a focused regression that clones the bundled Syn3A asset package, injects an explicit ADP pool, and verifies that domain-scoped nucleoid-track ADP pools plus matching chromosome-local ADP-producing reactions compile without changing the bundled fixture contract
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `localized support-pool turnover is now generic, but the bundled organism datasets still under-specify several counterpools and reversible metabolite states, so richer compiled chemistry inputs are still needed before chromosome-local support chemistry can fully replace the remaining global fallback pools`

### 2026-03-11 - Phase 7 / Explicit Pool Metadata Slice

- Summary:
  - added explicit `bulk_field` metadata to shipped molecule-pool specs and taught the native compiler/runtime to prefer declared pool identity over name-based substring inference, while keeping the old inference path as a compatibility fallback for older bundles
  - replaced the core pool lookups in the native registry compiler with field-based resolution so ATP, amino-acid, nucleotide, membrane-precursor, oxygen, and future ADP pools can be bound from explicit chemistry metadata instead of string hints
  - updated shipped Syn3A and demo bundle pool JSON plus Python bundle tests so explicit pool identity is part of the checked asset contract rather than an internal assumption
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/pools.json`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `python3 -m py_compile tests/test_whole_cell_assets.py`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `pool identity is now explicit, but the bundle chemistry still lacks many declared counterpools, redox pairs, and reversible-state relationships, so several metabolite classes still depend on partial rather than fully enumerated chemical state datasets`

### 2026-03-11 - Phase 7 / Metadata-First Pool Seeding Slice

- Summary:
  - switched program-spec initialization and runtime pool seeding to prefer explicit `bulk_field` metadata over raw pool-name parsing, so the initial lattice and scalar metabolite state now come from declared chemistry identity whenever the bundle provides it
  - kept the old name-based matching only as a compatibility fallback behind the explicit metadata path, reducing one more class of hardcoded name heuristics without breaking older bundle payloads
  - added a direct runtime regression that seeds ATP, ADP, glucose, and oxygen from non-informative pool names carrying explicit `bulk_field` metadata and verifies that the simulator updates the correct chemistry channels
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `runtime seeding now honors explicit pool identity, but several later execution paths still infer chemistry from pool/species names when compiled field metadata is absent or not yet threaded all the way through the runtime`

### 2026-03-11 - Phase 7 / Runtime Pool Field Normalization Slice

- Summary:
  - moved missing pool `bulk_field` recovery out of the per-step pool anchoring path and into runtime-species normalization, so compiled registry metadata now backfills pool identity once during initialization, restore, and species sync instead of forcing the hot loop to parse pool names every step
  - reduced the pool anchor logic to metadata-driven concentration anchoring plus basal fallback, with legacy name inference retained only inside the one-time normalization helper for older or manually constructed runtime species that still lack explicit fields
  - added a regression that injects a pool runtime species with an opaque name but a matching registry entry and verifies that the runtime backfills `bulk_field` from compiled registry metadata
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_runtime_pool_bulk_fields_backfill_from_registry_metadata --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `pool anchoring no longer parses names in the hot loop, but later runtime paths still contain species-name-to-field fallbacks for manually injected or partially compiled chemistry outside the normalized pool-state path`

### 2026-03-11 - Phase 7 / Operon Semantic Map Slice

- Summary:
  - compiled an explicit `operon_semantics` map into genome asset packages so operon-level asset class, assembly family, and subsystem targets are carried as first-class metadata instead of being repeatedly re-derived from downstream proteins or names
  - made native asset-package normalization treat that semantic map as authoritative when present, while still backfilling it from normalized operon records for legacy payloads that predate the field
  - aligned the Python bundle compiler and bundle-contract tests with the new field, and added a regression showing that sparse legacy packages with opaque operon names still recover their operon/complex semantics from the explicit map
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `operon semantics are now explicit at the asset-package layer, but older bundle sources still rely on ingress-time semantic recovery when annotations are missing entirely, so the next de-hardcoding step is expanding explicit source coverage for gene/operon semantics and removing more of those compatibility fills`

### 2026-03-11 - Phase 7 / Complex Semantic Map Slice

- Summary:
  - compiled an explicit `complex_semantics` map into genome asset packages so complex family, targeting, and coupling flags are carried as first-class metadata instead of being recovered indirectly from operons during asset normalization
  - made native asset-package normalization treat that complex map as authoritative for complex semantic restoration while still backfilling it from existing compiled complexes for legacy payloads that do not yet include the field
  - aligned the Python bundle compiler and bundle-contract tests with the new complex semantic metadata, and added a regression showing that a sparse package with an opaque complex operon still restores complex behavior from the explicit semantic map
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `complex semantics are now explicit at the asset-package layer, but fully unannotated source bundles still require ingress-time semantic recovery for genes and transcription units before those compiled maps exist, so expanding explicit upstream source annotations remains the next removal target`

### 2026-03-11 - Phase 7 / Protein Semantic Map Slice

- Summary:
  - compiled an explicit `protein_semantics` map into genome asset packages so per-protein asset class and subsystem targeting are carried directly in package metadata instead of being restored only through operon-level semantics
  - made native asset-package normalization treat that protein map as authoritative for sparse protein restoration while still backfilling it from existing compiled proteins for legacy payloads that do not yet include the field
  - aligned the Python bundle compiler and bundle-contract tests with the new protein semantic metadata, and added a regression showing that a sparse package with an opaque protein operon still restores protein semantics from the explicit product map
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `protein, operon, and complex semantics are now explicit at the asset-package layer, but the upstream structured bundle sources can still omit semantic annotations entirely, so the next removal target remains richer explicit source-side semantic coverage for genes and transcription units before ingress normalization has to infer anything`

### 2026-03-11 - Phase 7 / Source Semantic Overlay Slice

- Summary:
  - extended the structured bundle manifest schema with explicit `gene_semantics_json` and `transcription_unit_semantics_json` sources, and taught both the Rust and Python bundle compilers to merge those semantic overlays before runtime normalization
  - moved the demo bundle’s gene and transcription-unit semantic fields out of the mixed-purpose `gene_products.json` and `transcription_units.json` files into dedicated source-side semantic overlays, so source coverage rather than ingress inference now carries the asset class, family, and subsystem intent for that organism
  - added stronger bundle-compiler regressions that verify the demo bundle still compiles with fully populated gene/transcription-unit semantics and that the new semantic overlay files are part of the source-hash contract
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/gene_products.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/gene_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/transcription_unit_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/transcription_units.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the structured bundle path now supports explicit source-side semantic overlays, but the richer upstream datasets still need broader semantic annotation coverage so ingress normalization can eventually become a compatibility-only path rather than part of the normal bundle workflow`

### 2026-03-11 - Phase 7 / Strict Explicit-Semantic Bundle Slice

- Summary:
  - added manifest-level `require_explicit_gene_semantics` and `require_explicit_transcription_unit_semantics` controls so structured bundles can opt out of silent semantic recovery and fail fast when source-side semantic coverage is incomplete
  - taught both the Rust and Python bundle compilers to validate explicit semantic completeness before continuing, and normalized missing overlay-file handling into a consistent bundle-compiler error on the Python path
  - enabled strict explicit semantics on the demo structured bundle and added a regression proving compilation fails when a required gene-semantic overlay is omitted
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/manifest.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict explicit semantics now exists for structured bundles, but the richer upstream organism sources still need more direct semantic annotation coverage before the inference path can be demoted to compatibility-only status across the broader pipeline`

### 2026-03-11 - Phase 7 / Monolith-to-Structured Export Slice

- Summary:
  - added a structured-bundle export path in the Python asset compiler so an existing organism spec can be decomposed into explicit source files (`metadata`, `gene_features`, `gene_products`, `gene_semantics`, `transcription_units`, `transcription_unit_semantics`, `chromosome_domains`, `pools`, and `manifest`) instead of staying trapped in the monolithic `organism_spec_json` format
  - normalized sparse legacy organism semantics during export so partially annotated monoliths like the bundled Syn3A reference can still be migrated onto the strict structured-source workflow with explicit semantic overlays
  - added round-trip coverage showing the bundled Syn3A organism can be exported to a strict structured bundle and then recompiled through the normal bundle compiler and Rust manifest path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/__init__.py`
  - `src/oneuro/whole_cell/assets/__init__.py`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the export path now makes monolithic organism specs convertible into strict structured bundles, but the upstream reference assets still need to be actually migrated and checked into source-bundle form before the monolithic path can stop being the default for richer organisms`

### 2026-03-11 - Phase 7 / Structured Syn3A Source Migration Slice

- Summary:
  - migrated the bundled Syn3A reference from the monolithic `organism_spec_json` path to an explicit structured source bundle with dedicated `metadata`, `gene_features`, `gene_products`, `gene_semantics`, `transcription_units`, `transcription_unit_semantics`, `chromosome_domains`, and `pools` sources
  - fixed the Python structured-bundle compiler so declared `chromosome_domains_json` is part of the authoritative source contract instead of being silently recomputed and omitted from source-hash provenance
  - updated Python and Rust validation to assert the structured-source provenance shape directly, so the checked-in Syn3A bundle now exercises the same explicit-source workflow as the newer bundles
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/chromosome_domains.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_features.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_products.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/metadata.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/pools.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/transcription_unit_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/transcription_units.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `Syn3A now compiles through explicit structured sources, but the remaining monolithic compatibility path still exists for older organism specs and should keep shrinking until explicit bundles are the default ingestion form`

### 2026-03-11 - Phase 7 / Embedded Structured Syn3A Native Slice

- Summary:
  - replaced the native Rust bundled Syn3A organism source path so it now compiles from embedded structured bundle files (`manifest`, `metadata`, `gene_features`, `gene_products`, `gene_semantics`, `transcription_units`, `transcription_unit_semantics`, `chromosome_domains`, and `pools`) instead of a separate monolithic embedded organism JSON
  - kept `bundled_syn3a_organism_spec_json()` as a convenience surface, but made it a serialized view of the structured native compilation result rather than an independent source of truth
  - added a native regression proving the embedded structured Syn3A build matches the checked-in structured manifest compilation exactly, so the bundled Rust path and the source-bundle path stay locked together
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the bundled Syn3A native path now uses explicit structured sources, but the broader compatibility layer still supports legacy monolithic organism ingestion and should continue shrinking as more organisms move onto explicit bundle contracts`

### 2026-03-11 - Phase 7 / Remove Dead Syn3A Monolith Artifact Slice

- Summary:
  - removed the unused `oneuro-metal/specs/whole_cell_syn3a_organism.json` file after the native Rust bundled Syn3A path stopped depending on it
  - updated the top-level execution plan to point at the structured Syn3A bundle directory as the authoritative bundled organism source path instead of the deleted monolithic file
  - kept the bundled program reference spec intact, so only the dead organism monolith was removed
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the dead bundled Syn3A monolith is gone, but the broader bundle compiler still supports legacy organism_spec_json ingestion for compatibility and should keep shrinking as more organisms live entirely on structured-source contracts`

### 2026-03-11 - Phase 7 / Explicit Asset-Semantic Bundle Slice

- Summary:
  - extended the structured bundle contract so operon, protein, and complex semantics can be carried as explicit source files (`operon_semantics.json`, `protein_semantics.json`, and `complex_semantics.json`) instead of only being regenerated during asset compilation
  - updated both the Python bundle compiler and the native Rust bundle-manifest path to consume those asset-semantic overlays and re-normalize the compiled asset package, so explicit source semantics can override or complete the derived package state before registry compilation
  - regenerated the checked-in Syn3A structured bundle with the new asset-semantic files and aligned the embedded native bundled Syn3A asset package path to those same files, keeping the source-tree bundle and the native bundled path on one contract
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/complex_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/operon_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/protein_semantics.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `asset semantics can now come from explicit source files, but the remaining legacy organism_spec_json compatibility path still exists and more downstream entity descriptions still need to migrate onto explicit structured-source contracts to eliminate inference as the default path`

### 2026-03-11 - Phase 7 / Explicit Asset-Entity Bundle Slice

- Summary:
  - extended the structured bundle contract so downstream asset entities themselves (`operons`, `rnas`, `proteins`, and `complexes`) can be carried as explicit source files instead of only being regenerated from organism-level descriptions during every compile
  - updated both the Python bundle compiler and the native Rust manifest compiler to consume those entity files before applying asset-semantic overlays, with re-normalization so explicit source entities become the default package state
  - regenerated the checked-in Syn3A structured bundle with explicit `operons.json`, `rnas.json`, `proteins.json`, and `complexes.json`, and aligned the embedded native bundled Syn3A asset path to the same files with a regression proving manifest and embedded asset packages match
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/complexes.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/operons.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/proteins.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/rnas.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `asset entities and asset semantics can now come from explicit source files, but the remaining legacy organism_spec_json compatibility path still exists and more downstream state descriptions still need to move onto explicit structured-source contracts to make inference a compatibility-only path`

### 2026-03-11 - Phase 7 / Strict Structured Bundle Enforcement Slice

- Summary:
  - added explicit structured-bundle enforcement flags so a bundle can reject the legacy `organism_spec_json` path and require explicit asset-entity and asset-semantic source files instead of silently falling back to inferred package generation
  - wired those checks through both the Python bundle compiler and the native Rust manifest compiler, so strict bundles fail fast when they drift back toward monolithic or inferred ingestion
  - enabled the stricter contract on the checked-in Syn3A structured bundle and added Python regressions covering both the `organism_spec_json` rejection path and missing explicit asset-entity sources
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict structured bundles now exist for Syn3A, but the broader compatibility layer still supports legacy organism_spec_json ingestion for older bundles and more state surfaces still need explicit structured contracts before inference stops being the normal fallback`

### 2026-03-12 - Phase 7 / Explicit Program-Default Bundle Slice

- Summary:
  - extended the structured bundle contract so a bundle can carry explicit runtime program defaults in `program_defaults.json` instead of relying only on `build_program_spec_from_organism` defaults during manifest compilation
  - wired those defaults through both the Python structured-bundle exporter and the native Rust manifest compiler, so program-name, config, lattice, state, and quantum defaults can come from declared source files before hydration
  - enabled strict explicit program-default enforcement on the checked-in Syn3A bundle and added regressions covering both the rejection path for missing defaults and the positive native hydration path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/program_defaults.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict structured bundles now carry explicit program defaults, but the broader compatibility layer still supports legacy organism_spec_json ingestion for older bundles and more downstream state surfaces still need explicit structured-source contracts before inference becomes compatibility-only`

### 2026-03-12 - Phase 7 / Explicit Organism-Source Contract Slice

- Summary:
  - extended the structured bundle contract so strict bundles can require the core organism-level source files (`metadata`, gene features, gene products, transcription units, chromosome domains, and pools) instead of only checking semantic overlays and downstream asset files
  - wired that stricter contract through both the Python bundle compiler and the native Rust manifest validator, so strict bundles fail fast when any of those core structured-source entries are omitted from the manifest
  - enabled the stricter contract on the checked-in Syn3A bundle and added a regression covering the missing transcription-unit source rejection path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict structured bundles now require explicit core organism sources, but the broader compatibility layer still supports legacy organism_spec_json ingestion for older bundles and some downstream state surfaces still default to inferred descriptions when explicit structured sources are absent`

### 2026-03-12 - Phase 7 / Remove Legacy Monolithic Bundle Path

- Summary:
  - removed the active `organism_spec_json` manifest compile path from both the Python bundle compiler and the native Rust manifest compiler, so bundle compilation always flows through explicit structured sources instead of silently accepting monolithic organism-spec payloads
  - kept the manifest field only as a migration error surface, with direct validation messages pointing callers at structured bundle sources
  - updated the regression coverage so manifest bundles that still try to declare `organism_spec_json` fail with the new migration error
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the in-repo monolithic manifest path is gone, but explicit structured-source coverage still needs to expand across more downstream state descriptions so fewer semantics and runtime defaults depend on compatibility-only inference`

### 2026-03-12 - Phase 7 / Move Legacy Pool Backfill To Parse Boundaries

- Summary:
  - removed name-based pool metadata and runtime-species bulk-field backfill from the live simulator initialization and restore paths, so active whole-cell execution now relies on explicit metadata or registry wiring instead of runtime name heuristics
  - moved the remaining saved-state compatibility fill into `parse_saved_state_json`, where organism pool metadata and pool runtime-species bulk fields are normalized once at the data boundary before the simulator sees them
  - updated Rust regressions so legacy saved-state payloads still regain pool metadata at parse time while simulator initialization now only uses explicit pool fields
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell.rs oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `runtime pool inference is now out of the active whole-cell path, but compatibility-only semantic and pool backfills still exist at JSON/data ingress and more downstream descriptions still need fully explicit structured-source coverage`

### 2026-03-12 - Phase 7 / Enforce Explicit Pool Metadata In Strict Bundles

- Summary:
  - added explicit pool-metadata validation to the strict structured-bundle path in both the Python and Rust compilers, so strict bundles now fail if any pool omits its declared `bulk_field`
  - updated the native bundled-organism compile path to trust explicit organism-level pool and semantic metadata when strict manifest flags are enabled instead of renormalizing those fields after load
  - removed one more active-path normalization layer from the Rust bundle program-spec compiler by making it build from the already compiled organism payload rather than re-normalizing the organism again
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict bundle compilation now enforces explicit pool metadata, but explicit asset-package validation still shares some normalization logic and compatibility-only semantic backfills still remain at JSON/data ingress`

### 2026-03-12 - Phase 7 / Remove Strict Asset Package Self-Healing

- Summary:
  - removed the strict-bundle asset-package self-healing path in both the Python and Rust compilers, so explicit asset entities and semantic overlays are no longer silently re-normalized after load
  - added strict validation for explicit operon, protein, and complex entity metadata plus full semantic-map coverage, making strict bundles fail when those source contracts are incomplete instead of regenerating missing asset meaning downstream
  - changed the native Rust strict path to merge explicit semantic overlays directly onto entity records and otherwise return explicit asset packages as-is, keeping normalization as a compatibility-only path for non-strict bundles
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict bundle compilation now trusts explicit asset packages end to end, but JSON/state ingress still contains compatibility-only semantic backfills and broader source coverage is still needed so more bundles can run without any inferred fallback metadata`

### 2026-03-12 - Phase 7 / Stop Strict Bundles From Bootstrapping Derived Assets

- Summary:
  - changed the strict bundle compile path in both Python and Rust to start from an explicit empty asset package seeded only with organism/pool/domain state instead of first deriving operons, RNAs, proteins, and complexes from the organism spec and then overlaying explicit files on top
  - added strict asset-entity coverage validation against the organism source contract, so strict bundles now fail if any expected operon, RNA, protein, or operon-linked complex is missing from the explicit asset files
  - kept derived asset-package compilation as a compatibility path for non-strict bundles only, further separating explicit bottom-up execution from fallback synthesis
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `strict bundle compilation now requires explicit asset coverage, but compatibility-only normalization and semantic backfills still exist at JSON/state ingress and the remaining active parser/resolver boundaries still need to be split into explicit-vs-legacy paths`

### 2026-03-12 - Phase 7 / Split Explicit And Legacy Organism Asset JSON Parsers

- Summary:
  - changed `parse_organism_spec_json` and `parse_genome_asset_package_json` to behave as explicit round-trip loaders instead of boundary repair loaders, so compiled organism and asset JSON now reparse without hidden normalization
  - moved the old organism and asset repair behavior behind `parse_legacy_organism_spec_json` and `parse_legacy_genome_asset_package_json`, preserving compatibility-only semantic and pool backfills without keeping them on the main explicit path
  - updated the Rust public exports and regressions so explicit JSON round-trips are checked directly while the legacy backfill tests now exercise the legacy parser entrypoints
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/lib.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `saved-state ingress still defaults to compatibility backfills and the remaining explicit-vs-legacy parser boundaries need the same split so simulator-emitted state is not repaired on load`

### 2026-03-12 - Phase 7 / Split Explicit And Legacy Saved-State Parsers

- Summary:
  - changed `parse_saved_state_json` to keep the explicit simulator-emitted restore path free of pool and runtime-species name backfills, while still sharing registry/contract/provenance hydration that belongs to current-state restoration
  - moved the legacy saved-state repair behavior behind `parse_legacy_saved_state_json`, so old saved states still regain pool and runtime-species bulk-field metadata at a dedicated compatibility boundary instead of on the main restore path
  - added explicit saved-state round-trip coverage and updated the legacy saved-state regressions to target the new legacy parser entrypoint
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/lib.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the live parser path is now cleaner, but bundled/reference resolvers and any remaining compatibility loaders still need the same explicit-vs-legacy separation so no current-state path depends on inferred metadata`

### 2026-03-12 - Phase 7 / Split Explicit And Legacy Program-Spec Loading

- Summary:
  - changed `parse_program_spec_json` to preserve explicit program specs as-authored, so inline organism-only specs no longer silently derive genome assets or process registries during parse
  - moved the old program-spec repair behavior behind `parse_legacy_program_spec_json`, preserving compatibility for older specs that still need chromosome-domain compilation, asset compilation, and registry regeneration from organism data
  - removed simulator-side asset and registry derivation from `WholeCellSimulator::from_program_spec`, making the live constructor trust explicit program-spec state instead of rebuilding missing runtime inputs behind the caller's back
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/lib.rs oneuro-metal/src/whole_cell.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `saved-state restore and simulator bootstrap still contain later compatibility fallbacks for missing assets and registries, and those remaining live-path repair points need the same explicit-vs-legacy split`

### 2026-03-12 - Phase 7 / Split Explicit And Legacy Restore-Time Registry Rebuilds

- Summary:
  - removed live-path registry derivation from `ensure_process_registry`, `organism_process_registry()`, and `restore_saved_state`, so explicit simulators and explicit saved-state restores no longer compile process registries or assets behind the caller's back
  - added explicit legacy constructor entrypoints on `WholeCellSimulator` for program-spec and saved-state JSON, keeping compatibility-only rebuild behavior opt-in at the runtime boundary instead of implicit in the default constructors
  - expanded `parse_legacy_saved_state_json` to rebuild missing asset packages and process registries from inline organism data, aligning legacy saved-state restore behavior with the legacy program-spec path while keeping the main explicit restore path strict
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/whole_cell.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `runtime rebuild helpers and any remaining bootstrap paths that still compile registries from assets need the same explicit-vs-legacy split so the live simulator never repairs missing whole-cell metadata implicitly`

### 2026-03-12 - Phase 7 / Remove Explicit Parser Registry And Species Repair

- Summary:
  - changed `parse_program_spec_json` and `parse_saved_state_json` to stop regenerating process registries from inline asset packages, so explicit JSON now preserves missing registry state instead of silently compiling one from assets
  - kept bundled-reference hydration on the explicit path by resolving `organism_data_ref` to the matching bundled process registry, while moving inline-asset registry regeneration fully behind the legacy parser entrypoints
  - removed runtime-species bulk-field normalization from the explicit saved-state parser and kept it only on the legacy saved-state path, so explicit saved-state JSON now round-trips species metadata without silent repair
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && maturin develop -m oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the remaining explicit compatibility hook is the manual registry rebuild helper in the simulator, and any other live-path repair utilities should be pushed behind legacy-only or explicitly named migration paths`

### 2026-03-12 - Phase 1 / Promote Demo Bundle To Explicit Asset Sources

- Summary:
  - added explicit chromosome-domain, operon, RNA, protein, complex, and asset-semantic source files to `mgen_minimal_demo`, so both shipped organism bundles now carry declared asset entities and semantics instead of one demo bundle still relying on derived asset metadata
  - tightened the demo manifest to require structured bundle sources, explicit organism sources, explicit asset entities, explicit asset semantics, and explicit program defaults while still preserving the FASTA/GFF ingestion path for gene features
  - updated asset/compiler tests so the demo bundle now asserts the full explicit source-hash contract on both the Python compiler path and the native Rust bundle-ingestion path
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/manifest.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/chromosome_domains.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/operons.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/rnas.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/proteins.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/complexes.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/operon_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/protein_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/complex_semantics.json`
  - `src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/program_defaults.json`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the remaining compiler shortcut is the generic Python bundle compiler fallback that still derives asset entities when a source bundle omits them, so the next step is either eliminating that fallback or confining it to an explicitly legacy bundle mode`

### 2026-03-12 - Phase 1 / Gate Derived Asset Compilation Behind Legacy Opt-In

- Summary:
  - added `allow_legacy_derived_assets` as the explicit manifest switch for source bundles that still need derived operons/RNAs/proteins/complexes or derived asset semantics, and made the normal structured-bundle path reject missing explicit asset entities or semantics by default
  - updated both the Python bundle compiler and the native Rust manifest compiler so derived asset/entity compilation is only reachable when a manifest explicitly opts into that legacy mode
  - added reject-by-default and opt-in legacy coverage in both Python and Rust tests, proving that derived asset compilation is no longer part of the standard structured-bundle contract
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler-time normalization helpers still exist for legacy parsers and explicit migration/export tooling, so the next cleanup target is deciding which of those should move into a dedicated migration module instead of staying beside the live compiler`

### 2026-03-12 - Phase 1 / Split Explicit And Legacy Bundle Compiler Entrypoints

- Summary:
  - added explicit legacy bundle compiler entrypoints in Python (`compile_legacy_bundle_manifest`, `compile_legacy_named_bundle`) and in Rust (`compile_legacy_*_from_bundle_manifest_path`, `from_legacy_bundle_manifest_path`) so legacy-derived asset compilation no longer shares the same top-level API surface as the standard structured-bundle compiler
  - changed the standard bundle compile entrypoints to reject manifests with `allow_legacy_derived_assets`, forcing callers to be explicit at both the manifest level and the API level when they want legacy-derived asset compilation
  - added Python and Rust regressions proving that standard compile entrypoints reject legacy-derived manifests, while the legacy compile entrypoints reject explicit manifests and only accept manifests that explicitly opt into legacy-derived assets
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/__init__.py`
  - `src/oneuro/whole_cell/__init__.py`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `oneuro-metal/src/lib.rs`
  - `oneuro-metal/src/whole_cell.rs`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `rustfmt oneuro-metal/src/whole_cell_data.rs oneuro-metal/src/lib.rs oneuro-metal/src/whole_cell.rs`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `source /Users/bobbyprice/projects/oNeuro/.venv-codex/bin/activate && PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy normalization and export/migration helpers still live beside the main compiler code, so the next cleanup target is isolating those migration-only transforms from the active explicit compiler implementation`

### 2026-03-12 - Phase 1 / Export Boundary And Bundled Syn3A Spec Asset

- Summary:
  - added a dedicated Python exporter module as the public home for whole-cell bundle export helpers and checked in the bundled native Syn3A organism spec JSON that the Rust whole-cell code already compiles with `include_str!`, so the structured bundle path and bundled native reference both resolve directly from tracked repository assets
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `src/oneuro/whole_cell/assets/__init__.py`
  - `src/oneuro/whole_cell/assets/exporter.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/assets/exporter.py`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
- Remaining blockers:
  - `the exporter implementation still lives in compiler.py internally; the next cleanup target is moving that implementation behind the exporter module without regressing the newer explicit-bundle compiler path on origin/main`

### 2026-03-12 - Phase 1 / Embedded Structured Syn3A Reference Program

- Summary:
  - switched the bundled native Syn3A reference program to compile from the embedded structured bundle plus explicit `program_defaults.json` instead of a standalone reference-spec JSON, preserved the public `jcvi_syn3a_reference` alias on the bundled helper, and removed the dead bundled Syn3A organism/reference monolith files from `oneuro-metal/specs`
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `oneuro-metal/specs/whole_cell_syn3a_organism.json`
  - `oneuro-metal/specs/whole_cell_syn3a_reference.json`
  - `src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/program_defaults.json`
  - `tests/test_whole_cell.py`
- Tests run:
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `bundled export helpers still share implementation with compiler.py; the next cleanup target is moving those implementation details fully behind the exporter boundary`

### 2026-03-12 - Phase 1 / Exporter Implementation Split

- Summary:
  - moved whole-cell bundle export and structured-bundle emission implementation out of `compiler.py` and into `assets/exporter.py`, leaving the compiler module focused on manifest ingestion, validation, and compilation while keeping the public Python API unchanged
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/exporter.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the exporter now owns the implementation, but the next cleanup target is isolating any remaining migration-only normalization helpers that still live in compiler.py for explicit/legacy compatibility`

### 2026-03-12 - Phase 1 / Derived Asset Helper Split

- Summary:
  - moved legacy-derived genome asset construction and semantic inference helpers out of `assets/compiler.py` and into `assets/derived_assets.py`, so the explicit structured-bundle compiler keeps ownership of manifest parsing and explicit source validation while export and legacy-derived paths share a separate compatibility helper layer
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/derived_assets.py`
  - `src/oneuro/whole_cell/assets/exporter.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy-normalizing bundle migration helpers still live in compiler.py, so the next cleanup target is pushing the remaining compatibility transforms behind explicit migration-only entrypoints`

### 2026-03-12 - Phase 1 / Source Normalization Helper Split

- Summary:
  - moved explicit source merge, source-level semantic validation, and chromosome-domain compilation helpers out of `assets/compiler.py` and into `assets/source_normalization.py`, so compiler ownership is narrower and exporter no longer reaches back into compiler internals for structured-source normalization
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/exporter.py`
  - `src/oneuro/whole_cell/assets/source_normalization.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/source_normalization.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler.py still owns explicit-vs-legacy manifest dispatch and asset overlay application, so the next cleanup target is isolating the remaining compatibility/overlay transforms behind migration-only helpers`

### 2026-03-12 - Phase 1 / Asset Overlay Helper Split

- Summary:
  - moved asset overlay application, explicit asset coverage validation, and empty asset-package construction out of `assets/compiler.py` and into `assets/asset_overlays.py`, so compiler orchestration no longer owns the compatibility overlay path directly
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/asset_overlays.py`
  - `src/oneuro/whole_cell/assets/compiler.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/asset_overlays.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/source_normalization.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler.py still combines explicit manifest dispatch with the legacy-derived asset entrypoint, so the next cleanup target is isolating the explicit-vs-legacy bundle orchestration boundary itself`

### 2026-03-12 - Phase 1 / Explicit And Legacy Compiler Split

- Summary:
  - removed the boolean-gated manifest compiler path in `assets/compiler.py` and split it into explicit and legacy compile flows, so the normal structured-bundle compiler no longer branches through the legacy-derived asset mode at runtime
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/asset_overlays.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/source_normalization.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler.py still owns low-level manifest/source loading and file format readers, so the next cleanup target is separating raw source ingress from the explicit compiler entrypoint without widening the active runtime path again`

### 2026-03-12 - Phase 1 / Source Ingress Helper Split

- Summary:
  - moved raw manifest/source loading, FASTA and GFF readers, hash/load helpers, and structured-source ingress into `assets/source_ingress.py`, so `assets/compiler.py` now focuses on compile orchestration instead of file parsing
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/source_ingress.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/asset_overlays.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/source_ingress.py src/oneuro/whole_cell/assets/source_normalization.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler.py still owns manifest contract validation, so the next cleanup target is splitting manifest contract checking from compile orchestration if we keep narrowing the active compiler surface`

### 2026-03-12 - Phase 1 / Manifest Contract Helper Split

- Summary:
  - moved manifest contract checks for explicit asset requirements and legacy-entrypoint gating out of `assets/compiler.py` and into `assets/manifest_contracts.py`, leaving compiler closer to a pure orchestration layer
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `src/oneuro/whole_cell/assets/compiler.py`
  - `src/oneuro/whole_cell/assets/manifest_contracts.py`
- Tests run:
  - `python3 -m py_compile src/oneuro/whole_cell/assets/compiler.py src/oneuro/whole_cell/assets/asset_overlays.py src/oneuro/whole_cell/assets/derived_assets.py src/oneuro/whole_cell/assets/manifest_contracts.py src/oneuro/whole_cell/assets/source_ingress.py src/oneuro/whole_cell/assets/source_normalization.py src/oneuro/whole_cell/assets/exporter.py src/oneuro/whole_cell/assets/__init__.py src/oneuro/whole_cell/__init__.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell.py tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `compiler.py is now mostly orchestration, so the next cleanup target is optional further consolidation of shared type/contract helpers rather than untangling mixed explicit-vs-legacy logic`

### 2026-03-12 - Phase 7 / Runtime Chromosome Domain Authority

- Summary:
  - made compiled chromosome domains from the native process registry participate directly in runtime domain indexing and weighting, and removed the old fixed quartile fallback from the active chromosome-domain path in the native simulator
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_compiled_chromosome_domain_centers_bias_weight_peaks --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_registry_chromosome_domains_bias_weight_peaks_without_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `domain-local runtime behavior now prefers compiled domains over the fixed quartile fallback, but other coarse fallback scales remain in the native runtime and should be replaced with compiled metadata or direct local state as we keep deepening the bottom-up path`

### 2026-03-12 - Phase 7 / Feature-Driven Legacy Chromosome Domains

- Summary:
  - replaced the fixed four-quartile chromosome-domain fallback in the native data compiler with feature-gap-driven domain recovery, so sparse legacy organism specs derive implicit domains from actual operon or gene spacing instead of a hardcoded count
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q compiled_chromosome_domains_follow_feature_gaps_for_sparse_specs --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q compiled_chromosome_domains_use_single_domain_without_features --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit structured bundles already own chromosome domains, but the remaining legacy recovery path still needs more compiled/local ownership in other organism subsystems so sparse fallback behavior keeps shrinking`

### 2026-03-12 - Phase 7 / Explicit Domain Source Authority

- Summary:
  - aligned the Rust and Python chromosome-domain compilers so explicit domain membership stays source-authoritative, while implicit legacy domains are still recovered from feature gaps instead of a fixed count
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell_data.rs`
  - `src/oneuro/whole_cell/assets/source_normalization.py`
  - `tests/test_whole_cell_assets.py`
- Tests run:
  - `cargo test -q compiled_chromosome_domains_preserve_explicit_membership_without_backfill --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `python3 -m py_compile src/oneuro/whole_cell/assets/source_normalization.py tests/test_whole_cell_assets.py`
  - `PYTHONPATH=src pytest -q tests/test_whole_cell_assets.py`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit chromosome domains are now normalized instead of inferred, but more downstream bundle and runtime surfaces still need the same treatment so explicit sources fully own the active path`

### 2026-03-12 - Phase 7 / Component-Limited Complex Assembly Targets

- Summary:
  - shifted the active named-complex assembly path away from static `basal_abundance` targets and toward component-limited assembly capacity, so strict-bundle complex targets now respond more directly to available subunits and mature protein output
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_named_complex_target_tracks_component_capacity_over_static_prior --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the active assembly path now uses component capacity instead of a static abundance prior, but other runtime channels still collapse explicit inventories into heuristic subsystem shares and should be pushed toward direct local species ownership next`

### 2026-03-12 - Phase 7 / Named-Complex Inventory Authority

- Summary:
  - made assembly inventory prefer live named-complex state when compiled assets are present, so the active runtime no longer falls back to derived aggregate targets while explicit complex state already exists
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_assembly_inventory_prefers_named_complex_state_when_assets_exist --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `named-complex inventory now owns the active assembly rollup, but the downstream process-capacity rules still compress explicit inventories into coarse subsystem scalars and remain a good next target for bottom-up replacement`

### 2026-03-12 - Phase 7 / Explicit Complex Channel Ownership

- Summary:
  - replaced the active named-complex aggregation share mix with explicit channel ownership from subsystem targets, family, and asset class, leaving the old process-weight blend only as a compatibility fallback for unmapped legacy cases
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_named_complex_aggregation_prefers_explicit_family_channels --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `named-complex aggregation now follows explicit biological ownership, but the downstream process-capacity rules still collapse those explicit inventories into scalar rule surrogates and remain the next bottom-up target`

### 2026-03-12 - Phase 7 / Direct Process Capacity From Explicit Channels

- Summary:
  - removed the generic scalar-rule layer for whole-cell process capacities and now compute energy, transcription, translation, replication, segregation, membrane, and constriction capacities directly from explicit channel inventory, chemistry support, local pool state, and quantum efficiency
- Files changed:
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_process_fluxes_follow_explicit_channel_inventory --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `process capacities now come straight from explicit runtime state, but several downstream stage drive mixes and fallback inventory targets still depend on generic scalar-rule surrogates instead of explicit local channel execution`

### 2026-03-12 - Phase 7 / Direct Stage Flux And Drive Execution

- Summary:
  - removed the generic scalar-rule wrappers from the active CME, ODE, BD, and geometry stage flux or drive path, so transcription, translation, membrane flux, replication drive, segregation drive, membrane growth, and constriction now execute directly from explicit capacities and local runtime signals
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_process_fluxes_follow_explicit_channel_inventory --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_cme_stage_follows_explicit_inventory_channels --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the active stage drives are now direct, but energy/resource rules and fallback inventory targets still retain generic scalar-rule or derived-surrogate behavior and are the next bottom-up removal target`

### 2026-03-12 - Phase 7 / Direct Resource Signals And Diagnostic Pool Isolation

- Summary:
  - removed the generic resource-estimator rule layer from `base_rule_context()` so glucose, oxygen, amino-acid, nucleotide, membrane, and energy signals now come directly from local pools, local chemistry, support, and pressure
  - moved active replication and expression execution off the diagnostic `active_rnap`, `active_ribosomes`, and `dnaa` pools and back onto explicit complex inventory, so those compatibility pools no longer steer the live stage path
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_surrogate_pools_are_diagnostics_not_stage_drivers --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_cme_stage_follows_explicit_inventory_channels --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `base resource signals and diagnostic pools are now off the active path, but fallback inventory targets and several derived summary rollups still rely on surrogate or scalar-rule behavior instead of explicit local channel ownership`

### 2026-03-12 - Phase 7 / Explicit Assembly Inventory Fallback Narrowing

- Summary:
  - removed the live derived-target fallback from the explicit asset assembly inventory path, so structured bundles now use named-complex aggregation or persisted complex state instead of regenerating heuristic complex targets
  - kept the derived complex-target path as a legacy-only fallback for non-asset runtimes
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_explicit_asset_inventory_does_not_fall_back_to_derived_targets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the explicit assembly path no longer regenerates derived targets, but legacy fallback inventory targets and several derived summary rollups still need to be pushed behind compatibility-only boundaries`

### 2026-03-12 - Phase 7 / Explicit Diagnostic Summary Narrowing

- Summary:
  - removed the flux-blended surrogate refresh from the explicit asset path, so RNAP, ribosome, DnaA, and FtsZ diagnostic summaries now mirror explicit complex inventory instead of derived stage rollups
  - kept the blended surrogate refresh only for legacy non-asset runtimes where explicit complex inventory is not available
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_explicit_asset_diagnostics_follow_inventory_not_flux_surrogates --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_surrogate_pools_are_diagnostics_not_stage_drivers --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit diagnostic summaries now follow explicit inventory, but legacy fallback inventory targets and the remaining compatibility-only derived summary rollups still need to be collapsed or isolated behind legacy-only boundaries`

### 2026-03-12 - Phase 7 / Legacy Assembly Seed Narrowing

- Summary:
  - made the legacy-derived complex target path prefer persisted `complex_assembly` inventory before falling back to scalar-rule priors, so non-asset runtimes stop rebuilding targets from heuristic inventory whenever explicit assembly state already exists
  - updated the quantum-growth regression to assert on explicit complex assembly state instead of the compatibility `ftsz` summary pool
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_legacy_derived_complex_targets_prefer_persistent_complex_inventory --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_quantum_profile_accelerates_growth --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy derived targets now prefer persisted explicit assembly state, but the remaining compatibility-only derived summary rollups still need to be collapsed or isolated behind legacy-only boundaries`

### 2026-03-12 - Phase 7 / Explicit State Hot-Path Accessors

- Summary:
  - moved active scheduler, rule-context, assembly, BD, geometry, and spatial-field reads of replicated fraction, division progress, surface area, radius, and chromosome separation off synchronized summary scalars and onto explicit chromosome or membrane-state accessors
  - kept synchronized summary scalars for compatibility and snapshot boundaries, but removed them from more execution-time decisions on the active path
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_hot_path_accessors_prefer_explicit_chromosome_and_membrane_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `execution-time consumers now prefer explicit chromosome and membrane state, but compatibility-only summary payloads and boundary-facing rollups still need to be pushed fully to serialization and diagnostics boundaries`

### 2026-03-12 - Phase 7 / Boundary State Derivation Narrowing

- Summary:
  - moved snapshot, save-state, and public whole-cell progress or diagnostic getters onto explicit chromosome, membrane, and complex-inventory accessors so boundary payloads on the explicit-asset path no longer reuse stale synchronized scalar summaries
  - kept the synchronized scalar summaries as compatibility state for legacy restore and non-asset paths, but narrowed their role further toward boundary-only behavior
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_boundary_snapshots_and_save_state_prefer_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit-asset boundary payloads now derive from explicit state, but legacy restore or serialization bridges still preserve synchronized scalar summaries and need to be pushed further behind compatibility-only boundaries`

### 2026-03-12 - Phase 7 / Explicit Restore Boundary Narrowing

- Summary:
  - changed saved-state restore so explicit chromosome, membrane, and diagnostic state is rehydrated from explicit saved biology on the explicit-asset path, with synchronized scalar core fields used only as legacy or missing-state seeds
  - added a regression proving that stale `saved.core` progress and diagnostic summaries no longer override explicit saved chromosome, membrane, and complex state during restore
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_restore_saved_state_prefers_explicit_state_over_stale_core_summary --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit-asset restore now ignores stale synchronized core summaries, but the remaining non-explicit compatibility paths still preserve synchronized scalar summaries and need to be pushed behind legacy-only boundaries`

### 2026-03-12 - Phase 7 / Bundle-Less Restore State Preservation

- Summary:
  - changed the non-explicit, bundle-less restore path to preserve explicit saved chromosome and membrane state when present instead of always reseeding those subsystems from coarse core summary scalars
  - added a regression proving that a saved state with no organism bundle still restores from explicit saved chromosome, membrane, and complex state even when the legacy core summaries are stale
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_from_saved_state_json_without_organism_prefers_explicit_saved_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `bundle-less restore now preserves explicit chromosome and membrane state, but the remaining compatibility-only serialization payloads still carry synchronized scalar summaries and need further narrowing`

### 2026-03-12 - Phase 7 / Bundle-Less Diagnostic Boundary Narrowing

- Summary:
  - changed bundle-less snapshot, save-state, and public diagnostic getters to derive RNAP, ribosome, DnaA, and FtsZ summaries from persisted explicit `complex_assembly` state when present instead of reusing stale surrogate pool scalars
  - added a regression proving that boundary diagnostics now follow explicit complex assembly state even without bundle assets
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_bundleless_boundary_diagnostics_prefer_explicit_complex_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `bundle-less diagnostic boundaries now prefer explicit complex state, but the remaining compatibility-only serialization payloads still need further narrowing where no richer persisted explicit state exists yet`

### 2026-03-12 - Phase 7 / Program-Spec Explicit Assembly Bootstrap

- Summary:
  - extended `WholeCellProgramSpec` so bootstrap payloads can carry explicit `complex_assembly` totals or richer `named_complexes` state instead of forcing non-saved-state initialization to reseed assembly from defaults
  - updated `WholeCellSimulator::from_program_spec()` to preserve explicit assembly payloads after expression refresh, so descriptor-driven scaling still shapes fallback targets while explicit inventory wins when provided
  - added comments in the bootstrap path documenting the explicit-state precedence and why it has to run after expression refresh
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q test_from_program_spec_preserves_explicit_complex_assembly_without_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_from_program_spec_preserves_explicit_named_complexes_with_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_organism_descriptor_drives_division_and_replication_scales --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `program-spec bootstrap now preserves explicit assembly state, but the remaining compatibility-only serialization payloads still need further narrowing where richer explicit state is still absent or only represented by synchronized summaries`

### 2026-03-12 - Phase 7 / Program-Spec Explicit Expression Bootstrap

- Summary:
  - extended `WholeCellProgramSpec` so bootstrap payloads can carry explicit `organism_expression` state rather than always regenerating transcription-unit execution state and process-support scales from organism descriptors
  - updated `WholeCellSimulator::from_program_spec()` to preserve explicit expression payloads first and only fall back to descriptor-driven refresh when no transcription-unit state is supplied
  - added comments in the bootstrap path documenting why explicit expression state is applied before the explicit assembly-preservation step
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q test_from_program_spec_preserves_explicit_expression_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `program-spec bootstrap now preserves explicit expression and assembly state, but remaining compatibility-only serialization paths still need richer explicit payloads for other layers before synchronized scalar summaries can disappear completely`

### 2026-03-12 - Phase 7 / Program-Spec Explicit Runtime Chemistry And Scheduler Bootstrap

- Summary:
  - extended `WholeCellProgramSpec` so bootstrap payloads can also carry explicit runtime species, runtime reactions, and scheduler clocks rather than always regenerating runtime chemistry state and multirate clock state from the compiled registry
  - updated `WholeCellSimulator::from_program_spec()` to preserve explicit runtime chemistry state when both species and reactions are supplied, while still falling back to registry-driven initialization when either side is missing
  - updated `WholeCellSimulator::from_program_spec()` to preserve explicit multirate scheduler clocks when supplied, and only recompute adaptive intervals when the program spec omits a scheduler payload entirely
  - added comments in the bootstrap path documenting why explicit runtime chemistry and scheduler state sit after explicit expression and assembly precedence
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q test_from_program_spec_preserves_explicit_runtime_process_and_scheduler_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `program-spec bootstrap now preserves explicit expression, assembly, runtime chemistry, and scheduler state, but the remaining compatibility-only serialization paths still need richer explicit payloads for the last legacy-only biology layers before synchronized scalar summaries can disappear completely`

### 2026-03-12 - Phase 7 / Program-Spec Explicit Local Chemistry Bootstrap

- Summary:
  - extended `WholeCellProgramSpec` so bootstrap payloads can also carry explicit local-chemistry reports, site reports, probe schedules, subsystem coupling state, MD probe payloads, and MD coupling scales instead of always resetting those layers during non-saved-state initialization
  - updated `WholeCellSimulator::from_program_spec()` to preserve explicit local-chemistry runtime state immediately after local-chemistry bridge configuration, so later expression refresh, assembly fallback, runtime chemistry bootstrap, and scheduler adaptation can consume the supplied support signals
  - added comments in the bootstrap path documenting why explicit local-chemistry state has to land before expression, assembly, runtime chemistry, and scheduler precedence
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q test_from_program_spec_preserves_explicit_local_chemistry_runtime_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `program-spec bootstrap now preserves explicit local chemistry, expression, assembly, runtime chemistry, and scheduler state, but the remaining compatibility-only serialization paths still need richer explicit payloads for the last legacy-only biology layers before synchronized scalar summaries can disappear completely`

### 2026-03-12 - Phase 7 / Program-Spec Explicit Spatial Field Bootstrap

- Summary:
  - extended `WholeCellProgramSpec` so bootstrap payloads can also carry explicit spatial fields rather than always regenerating membrane, septum, nucleoid, membrane-band, and pole locality from chromosome and membrane summaries
  - added a dedicated spatial-field application helper and updated both restore-time and program-spec bootstrap paths to use it, so explicit field payloads land after chromosome and membrane normalization but before RDME drive refresh and downstream chemistry-aware bootstrap stages consume locality
  - added comments in the bootstrap path documenting why explicit spatial fields have to land before RDME and the later chemistry, expression, assembly, runtime-chemistry, and scheduler precedence chain
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q test_from_program_spec_preserves_explicit_spatial_fields --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `program-spec bootstrap now preserves explicit spatial fields, local chemistry, expression, assembly, runtime chemistry, and scheduler state, but the remaining compatibility-only serialization paths still need richer explicit payloads for the last legacy-only biology layers before synchronized scalar summaries can disappear completely`

### 2026-03-12 - Phase 7 / Explicit Local Chemistry Boundary Visibility

- Summary:
  - added a boundary helper so persisted explicit local-chemistry state is treated as real state even when no live chemistry bridge is attached
  - updated `local_chemistry_report()`, `local_chemistry_sites()`, and snapshot export so explicit chemistry reports, site reports, subsystem coupling state, and MD scales supplied through program-spec or saved-state bootstrap are no longer hidden behind `chemistry_bridge.is_some()`
  - added a regression proving a program-spec can carry explicit chemistry state without a live bridge and still expose that state through the public getters and snapshot boundary
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_local_chemistry_getters_expose_explicit_state_without_bridge --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `explicit persisted chemistry state is now visible at public boundaries without a live bridge, but the remaining compatibility-only serialization paths still need richer explicit payloads for the last legacy-only biology layers before synchronized scalar summaries can disappear completely`

### 2026-03-12 - Phase 7 / Bundle-Less Explicit Biology Restore And Boundary Preservation

- Summary:
  - added an explicit-expression-state helper and updated the bundle-less execution path so `refresh_organism_expression_state()`, process-scale access, and assembly-target derivation preserve persisted expression state instead of wiping it when bundled organism descriptors are absent
  - added asset-free named-complex aggregation from explicit family, subsystem, and asset-class metadata, and routed bundle-less assembly inventory, diagnostics, save-state export, and per-step complex refresh through that explicit inventory path instead of clearing named complexes or falling back to stale scalar summaries
  - updated bundle-less saved-state restore so explicit organism-expression state, process registries, runtime species, runtime reactions, and named-complex state are preserved when present, while legacy scalar core summaries remain only as fallback seeds for layers that are actually missing
  - updated public bundle-less expression boundaries so `organism_expression_state()` exposes persisted explicit expression state even without bundled organism metadata
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_bundleless_boundary_diagnostics_prefer_explicit_named_complex_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_bundleless_restore_preserves_explicit_expression_runtime_and_named_complex_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_from_saved_state_json_without_organism_prefers_explicit_saved_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the remaining compatibility-only serialization surface is now narrower: legacy-only restore or snapshot paths still keep synchronized scalar summaries where richer explicit persisted payloads have not been defined yet`

### 2026-03-12 - Phase 7 / Bundle-Less Runtime Chemistry Persistence Without Assets

- Summary:
  - updated bundle-less runtime-species sync so explicit runtime species are no longer cleared just because `organism_assets` is absent
  - updated runtime chemistry bootstrap so registry-only bundle-less paths can initialize runtime species and reactions on demand during `step()`, instead of stalling with empty chemistry state until a bundle asset package is present
  - added an asset-free operon-total path for runtime RNA and protein synchronization, so bundle-less explicit runtime chemistry can still use persisted species metadata and compiled registries as the live source of truth
  - extended the bundle-less restore regression so a post-restore `step()` now proves runtime species and reactions survive execution and remain visible through save-state boundaries
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_bundleless_restore_preserves_explicit_expression_runtime_and_named_complex_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_bundleless_registry_bootstraps_runtime_process_state_without_assets --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `the remaining compatibility-only surface is narrower again: legacy-only restore, snapshot, or migration paths still keep synchronized scalar summaries where richer explicit persisted payloads still do not exist`

### 2026-03-12 - Phase 7 / Legacy Saved-State Core Summary Promotion

- Summary:
  - moved the legacy saved-state compatibility bridge farther down to the parser boundary by promoting coarse `saved.core` summary payloads into explicit chromosome, membrane, and complex-assembly state before restore-time runtime logic runs
  - added legacy synthesis helpers for fork state, chromosome loci, membrane geometry, and coarse assembly totals so older saved states regain explicit biology layers without adding more restore-time branching in the live simulator
  - added both data-layer and runtime regressions proving legacy saved states with stale or missing explicit biology still restore into explicit chromosome, membrane, and complex state that drives later snapshot and diagnostic boundaries
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q parse_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_from_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy compatibility is narrower again, but expression, local-chemistry, scheduler, and named-complex detail still lack equally rich promotion on the remaining non-explicit migration and serialization paths`

### 2026-03-12 - Phase 7 / Legacy Scheduler Clock Promotion

- Summary:
  - extended the legacy saved-state parser so older payloads that only persisted coarse step and time counters now regain an explicit multirate scheduler state before runtime restore runs
  - added a legacy scheduler synthesis path keyed off saved config intervals, step count, and elapsed time, so parser repair now carries stage interval, run-count, and due-step history at the boundary instead of leaving runtime restore to rebuild clocks from scratch
  - extended both data-layer and runtime regressions so legacy restore now proves chromosome, membrane, complex, and scheduler state are all promoted into explicit persisted layers on the compatibility path
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q parse_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_from_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy compatibility is narrower again, but expression, local-chemistry, and named-complex detail still lack equally rich promotion on the remaining non-explicit migration and serialization paths`

### 2026-03-12 - Phase 7 / Legacy Named-Complex Promotion

- Summary:
  - extended the legacy saved-state parser so coarse aggregate assembly channels now promote into explicit `named_complexes`, distributing through asset-backed complex specs when bundles are present and through generic family-level complex carriers when they are not
  - updated the bundle-less assembly inventory path so explicit named-complex aggregation remains preferred, but any persisted aggregate channels that the current named-complex semantics still cannot represent exactly are preserved instead of being silently lost during restore
  - added data-layer and runtime regressions proving legacy restore now yields explicit named-complex state alongside explicit chromosome, membrane, complex, and scheduler state on the compatibility path
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
  - `oneuro-metal/src/whole_cell_data.rs`
- Tests run:
  - `cargo test -q parse_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_from_legacy_saved_state_json_promotes_core_summary_to_explicit_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy compatibility is narrower again, but expression and local-chemistry detail still lack equally rich promotion on the remaining non-explicit migration and serialization paths, and some assembly channels still need richer direct named-complex carriers before persisted aggregate assembly can disappear completely`

### 2026-03-12 - Phase 7 / Bundle-Less Expression Promotion

- Summary:
  - added a native bundle-less expression synthesis path so restore and execution can rebuild explicit operon expression state directly from persisted runtime species and reactions when bundled organism descriptors are absent
  - wired the bundle-less expression refresh and restore paths to prefer that synthesized operon state over empty defaults, keeping runtime chemistry and expression coupled even on stripped compatibility payloads
  - added a regression that clears explicit expression from a stripped saved state, keeps operon-tagged runtime species and reactions, and proves restore plus a subsequent `step()` still expose explicit operon expression through save-state and getter boundaries
- Files changed:
  - `docs/whole_cell_execution_plan.md`
  - `docs/whole_cell_progress_ledger.md`
  - `oneuro-metal/src/whole_cell.rs`
- Tests run:
  - `cargo test -q test_bundleless_restore_synthesizes_expression_from_runtime_process_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q test_bundleless_restore_preserves_explicit_expression_runtime_and_named_complex_state --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell_data --manifest-path oneuro-metal/Cargo.toml`
  - `cargo test -q whole_cell --manifest-path oneuro-metal/Cargo.toml`
- Artifacts produced:
  - `none`
- Remaining blockers:
  - `legacy compatibility is narrower again, but local-chemistry detail and a few remaining fine-grained expression or assembly carriers still lack equally rich promotion on the remaining non-explicit migration and serialization paths`
