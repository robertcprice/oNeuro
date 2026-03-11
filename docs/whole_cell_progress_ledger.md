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
