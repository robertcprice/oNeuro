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
