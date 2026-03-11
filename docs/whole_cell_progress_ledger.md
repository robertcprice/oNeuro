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
