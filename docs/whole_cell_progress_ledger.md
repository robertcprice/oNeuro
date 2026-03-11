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
