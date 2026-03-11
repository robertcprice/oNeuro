import json
import tarfile
from pathlib import Path

import pytest

from oneuro.whole_cell import (
    CellCompartment,
    CouplingStage,
    ExternalTool,
    MC4DRunner,
    NQPUWholeCellProfile,
    WHOLE_CELL_CONTRACT_VERSION,
    WholeCellScheduler,
    WholeCellStageResult,
    WholeCellArtifactIngestor,
    WholeCellContract,
    WholeCellState,
    WholeCellConfig,
    WholeCellProvenance,
    apply_nqpu_whole_cell_profile,
    build_nqpu_whole_cell_profile,
    WholeCellProgramSpec,
    build_syn3a_skeleton_scheduler,
    syn3a_minimal_state,
    syn3a_reference_program,
    RustWholeCellSimulator,
)
from oneuro.whole_cell.adapters import MC4DAdapter, MC4DRunConfig
from oneuro.whole_cell.manifest import syn3a_reference_manifest


def test_syn3a_reference_program_shape():
    spec = syn3a_reference_program()

    assert isinstance(spec, WholeCellProgramSpec)
    assert isinstance(spec.config, WholeCellConfig)
    assert spec.organism == "JCVI-syn3A"
    assert CouplingStage.RDME in spec.coupling_stages
    assert CouplingStage.BD in spec.coupling_stages
    assert ExternalTool.LAMMPS in spec.external_tools
    assert isinstance(spec.contract, WholeCellContract)
    assert spec.contract.contract_version == WHOLE_CELL_CONTRACT_VERSION
    assert isinstance(spec.provenance, WholeCellProvenance)


def test_syn3a_minimal_state_shape():
    state = syn3a_minimal_state()

    assert isinstance(state, WholeCellState)
    assert state.organism == "JCVI-syn3A"
    assert state.compartments[CellCompartment.CYTOPLASM]["ATP"] > 0
    assert state.chromosome.genome_bp == 543_000
    assert state.geometry.radius_nm == 200.0


def test_whole_cell_state_apply_deltas_and_snapshot():
    state = syn3a_minimal_state()
    state.advance_time(5.0)
    state.apply_deltas(
        compartment_deltas={"cytoplasm": {"ATP": -100.0}},
        metabolite_deltas={"ATP": -0.1},
        protein_deltas={"DnaA": 2.0},
        chromosome_updates={"replicated_bp": 1000},
        geometry_updates={"division_progress": 0.25},
    )

    snapshot = state.snapshot()
    assert snapshot["time_ms"] == 5.0
    assert snapshot["compartments"]["cytoplasm"]["ATP"] == 7900.0
    assert snapshot["metabolites_mM"]["ATP"] == pytest.approx(1.1)
    assert snapshot["proteins"]["DnaA"] == 82.0
    assert snapshot["chromosome"]["replicated_bp"] == 1000
    assert snapshot["geometry"]["division_progress"] == 0.25
    assert snapshot["contract"]["contract_version"] == WHOLE_CELL_CONTRACT_VERSION
    assert snapshot["provenance"]["source_dataset"] == "JCVI-syn3A minimal native skeleton"


def test_syn3a_reference_manifest_shape():
    manifest = syn3a_reference_manifest()

    assert manifest.program.name == "syn3a_reference_program"
    assert manifest.entrypoint == Path("Whole_Cell_Minimal_Cell.py")
    assert manifest.restart_entrypoint == Path("Restart_Whole_Cell_Minimal_Cell.py")
    assert any(dep.tool == ExternalTool.BTREE_CHROMO_GPU for dep in manifest.dependencies)
    assert any(path == Path("input_data") for path in manifest.expected_repo_paths)
    assert manifest.contract.contract_version == WHOLE_CELL_CONTRACT_VERSION
    assert manifest.provenance.run_manifest_hash == "syn3a_reference_manifest_v1"


def test_scheduler_runs_stage_handlers_in_manifest_order():
    manifest = syn3a_reference_manifest()
    state = syn3a_minimal_state()
    scheduler = WholeCellScheduler.from_manifest(state, manifest)
    observed = []

    def make_handler(label, delta):
        def handler(current_state, stage_dt, stage):
            observed.append((label, stage_dt, stage.value))
            return WholeCellStageResult(
                stage=stage,
                advanced_ms=stage_dt,
                metabolite_deltas={"ATP": delta},
                notes=label,
            )

        return handler

    scheduler.register_handler(CouplingStage.RDME, make_handler("rdme", 0.00))
    scheduler.register_handler(CouplingStage.CME, make_handler("cme", 0.00))
    scheduler.register_handler(CouplingStage.ODE, make_handler("ode", 0.05))
    scheduler.register_handler(CouplingStage.BD, make_handler("bd", 0.00))
    scheduler.register_handler(CouplingStage.GEOMETRY, make_handler("geometry", 0.00))

    results = scheduler.step(12.5)

    assert observed[0][0] == "rdme"
    assert observed[-1][0] == "geometry"
    assert any(result.stage == CouplingStage.ODE for result in results)
    assert state.metabolites_mM["ATP"] > 1.2
    assert state.stage_history


def test_scheduler_run_for_accumulates_multiple_small_steps():
    manifest = syn3a_reference_manifest()
    state = syn3a_minimal_state()
    scheduler = WholeCellScheduler.from_manifest(state, manifest)
    scheduler.register_handler(
        CouplingStage.ODE,
        lambda current_state, stage_dt, stage: WholeCellStageResult(
            stage=stage,
            advanced_ms=stage_dt,
            protein_deltas={"FtsZ": 1.0},
        ),
    )

    results = scheduler.run_for(duration_ms=25.0, dt_ms=2.5)

    assert state.time_ms == 25.0
    assert sum(1 for result in results if result.stage == CouplingStage.ODE) == 2
    assert state.proteins["FtsZ"] == 92.0


def test_syn3a_skeleton_scheduler_progresses_native_state():
    scheduler = build_syn3a_skeleton_scheduler()
    state = scheduler.state
    start_atp = state.metabolites_mM["ATP"]
    start_replication = state.chromosome.replicated_bp
    start_surface = state.geometry.surface_area_nm2
    start_ftsz = state.proteins["FtsZ"]

    results = scheduler.run_for(duration_ms=25.0, dt_ms=2.5)

    assert results
    assert state.time_ms == 25.0
    assert state.metabolites_mM["ATP"] > start_atp
    assert state.chromosome.replicated_bp > start_replication
    assert state.geometry.surface_area_nm2 > start_surface
    assert state.proteins["FtsZ"] > start_ftsz


def test_mc4d_adapter_validate_repo_reports_missing_paths(tmp_path):
    adapter = MC4DAdapter(repo_root=tmp_path)

    statuses = adapter.validate_repo()

    assert statuses
    assert any(not status.ok for status in statuses)
    assert any(status.name == "Whole_Cell_Minimal_Cell.py" for status in statuses)


def test_mc4d_adapter_builds_main_command(tmp_path):
    repo_root = tmp_path / "mc4d"
    repo_root.mkdir()
    (repo_root / "Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")
    (repo_root / "Restart_Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")

    adapter = MC4DAdapter(repo_root=repo_root)
    command = adapter.build_command(
        MC4DRunConfig(
            output_dir="replicate1",
            sim_time_seconds=1200,
            cuda_device=1,
            dna_rng_seed=13,
            dna_software_directory=tmp_path / "dna",
            working_directory=tmp_path / "work",
        )
    )

    assert command[0] == "python3"
    assert command[1].endswith("Whole_Cell_Minimal_Cell.py")
    assert "-od" in command and "replicate1" in command
    assert "-wd" in command


def test_mc4d_adapter_builds_restart_command(tmp_path):
    repo_root = tmp_path / "mc4d"
    repo_root.mkdir()
    (repo_root / "Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")
    (repo_root / "Restart_Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")

    adapter = MC4DAdapter(repo_root=repo_root)
    command = adapter.build_command(
        MC4DRunConfig(
            output_dir="replicate1",
            sim_time_seconds=2400,
            cuda_device=0,
            dna_rng_seed=42,
            dna_software_directory=tmp_path / "dna",
            restart=True,
        )
    )

    assert command[1].endswith("Restart_Whole_Cell_Minimal_Cell.py")
    assert "2400" in command


def test_mc4d_adapter_from_environment_reads_paths(tmp_path):
    repo_root = tmp_path / "repo"
    dna_root = tmp_path / "dna_software"
    repo_root.mkdir()
    dna_root.mkdir()

    adapter = MC4DAdapter.from_environment(
        env={
            "ONEURO_MC4D_REPO": str(repo_root),
            "ONEURO_MC4D_DNA_SOFTWARE_DIR": str(dna_root),
            "ONEURO_MC4D_CONDA_ENV": "mc4d-env",
        }
    )

    assert adapter.repo_root == repo_root.resolve()
    assert adapter.dna_software_directory == dna_root.resolve()
    assert adapter.conda_env_name == "mc4d-env"


def test_mc4d_adapter_validate_environment_uses_env_paths(tmp_path):
    repo_root = tmp_path / "repo"
    dna_root = tmp_path / "dna_software"
    (dna_root / "btree_chromo_gpu").mkdir(parents=True)
    (dna_root / "sc_chain_generation").mkdir(parents=True)
    repo_root.mkdir()

    adapter = MC4DAdapter(repo_root=repo_root)
    statuses = adapter.validate_environment(
        env={
            "ONEURO_MC4D_DNA_SOFTWARE_DIR": str(dna_root),
        }
    )

    by_name = {status.name: status for status in statuses}
    assert by_name["btree_chromo_gpu"].ok
    assert by_name["sc_chain_generation"].ok


def test_mc4d_build_environment_sets_expected_vars(tmp_path):
    repo_root = tmp_path / "repo"
    dna_root = tmp_path / "dna"
    repo_root.mkdir()
    dna_root.mkdir()

    adapter = MC4DAdapter(
        repo_root=repo_root,
        dna_software_directory=dna_root,
        conda_env_name="mc4d-env",
    )
    env = adapter.build_environment(base_env={"PATH": "/usr/bin"})

    assert env["ONEURO_MC4D_REPO"] == str(repo_root.resolve())
    assert env["ONEURO_MC4D_DNA_SOFTWARE_DIR"] == str(dna_root.resolve())
    assert env["ONEURO_MC4D_CONDA_ENV"] == "mc4d-env"


def test_mc4d_runner_writes_launch_manifest(tmp_path):
    repo_root = tmp_path / "mc4d"
    repo_root.mkdir()
    (repo_root / "Whole_Cell_Minimal_Cell.py").write_text("print('ok')\n", encoding="ascii")
    (repo_root / "Restart_Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")
    (repo_root / "README.md").write_text("", encoding="ascii")
    (repo_root / "Hook.py").write_text("", encoding="ascii")
    (repo_root / "MC_RDME_initialization.py").write_text("", encoding="ascii")
    (repo_root / "SpatialDnaDynamics.py").write_text("", encoding="ascii")
    (repo_root / "input_data").mkdir()

    runner = MC4DRunner(MC4DAdapter(repo_root=repo_root), artifact_root=tmp_path / "artifacts")
    plan = runner.plan_run(
        MC4DRunConfig(
            output_dir="replicate1",
            sim_time_seconds=10,
            cuda_device=0,
            dna_rng_seed=1,
            dna_software_directory=tmp_path / "dna",
        ),
        env={"PATH": "/usr/bin"},
        job_name="mc4d_test_job",
    )

    manifest_path = Path(plan.manifest_path)
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    assert payload["job_name"] == "mc4d_test_job"
    assert payload["runtime"] == "mc4d"


def test_mc4d_runner_launches_fake_python_job(tmp_path):
    repo_root = tmp_path / "mc4d"
    repo_root.mkdir()
    (repo_root / "Whole_Cell_Minimal_Cell.py").write_text(
        "import sys\nprint('launch', sys.argv[1:])\n",
        encoding="ascii",
    )
    (repo_root / "Restart_Whole_Cell_Minimal_Cell.py").write_text("", encoding="ascii")
    (repo_root / "README.md").write_text("", encoding="ascii")
    (repo_root / "Hook.py").write_text("", encoding="ascii")
    (repo_root / "MC_RDME_initialization.py").write_text("", encoding="ascii")
    (repo_root / "SpatialDnaDynamics.py").write_text("", encoding="ascii")
    (repo_root / "input_data").mkdir()

    runner = MC4DRunner(MC4DAdapter(repo_root=repo_root), artifact_root=tmp_path / "artifacts")
    result = runner.launch(
        MC4DRunConfig(
            output_dir="replicate1",
            sim_time_seconds=10,
            cuda_device=0,
            dna_rng_seed=1,
            dna_software_directory=tmp_path / "dna",
        ),
        env={"PATH": "/usr/bin"},
        job_name="mc4d_launch_job",
    )

    assert result.returncode == 0
    assert Path(result.stdout_path).read_text(encoding="utf-8").startswith("launch")


def test_artifact_ingestor_summarizes_directory_and_writes_json(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "counts.csv").write_text("time,ATP\n0,1\n1,2\n", encoding="ascii")
    (source / "traj.lm").write_text("placeholder", encoding="ascii")

    ingestor = WholeCellArtifactIngestor(artifact_root=tmp_path / "artifacts")
    summary = ingestor.ingest(source, name="directory_bundle")

    assert summary.source_kind == "directory"
    assert summary.file_count == 2
    assert summary.extension_counts[".csv"] == 1
    assert summary.csv_summaries[0]["row_count"] == 2
    assert list((tmp_path / "artifacts").glob("directory_bundle_whole_cell_artifact_*.json"))


def test_artifact_ingestor_summarizes_tar_gz(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    csv_path = source_dir / "counts.csv"
    csv_path.write_text("time,ATP\n0,1\n", encoding="ascii")
    lm_path = source_dir / "traj.lm"
    lm_path.write_text("placeholder", encoding="ascii")
    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(csv_path, arcname="bundle/counts.csv")
        archive.add(lm_path, arcname="bundle/traj.lm")

    ingestor = WholeCellArtifactIngestor(artifact_root=tmp_path / "artifacts")
    summary = ingestor.ingest(tar_path, name="tar_bundle")

    assert summary.source_kind == "tar.gz"
    assert summary.file_count == 2
    assert summary.extension_counts[".lm"] == 1
    assert summary.csv_summaries[0]["column_count"] == 2


def test_build_nqpu_whole_cell_profile_uses_fallbacks_cleanly():
    profile = build_nqpu_whole_cell_profile()

    assert isinstance(profile, NQPUWholeCellProfile)
    assert profile.oxphos_efficiency >= 1.0
    assert profile.translation_efficiency >= 1.0
    assert profile.nucleotide_polymerization_efficiency >= 1.0
    assert profile.membrane_synthesis_efficiency >= 1.0
    assert profile.chromosome_segregation_efficiency >= 1.0
    assert profile.source in {"nqpu", "wkb_fallback"}


def test_apply_nqpu_whole_cell_profile_calls_simulator():
    class DummySimulator:
        def __init__(self):
            self.last_profile = None

        def set_quantum_profile(self, **kwargs):
            self.last_profile = kwargs

    simulator = DummySimulator()
    profile = apply_nqpu_whole_cell_profile(simulator)

    assert simulator.last_profile is not None
    assert simulator.last_profile["oxphos_efficiency"] == pytest.approx(profile.oxphos_efficiency)
    assert simulator.last_profile["translation_efficiency"] == pytest.approx(
        profile.translation_efficiency
    )


def test_rust_whole_cell_subsystem_states_if_available():
    if RustWholeCellSimulator is None:
        pytest.skip("oneuro_metal extension not available")

    sim = RustWholeCellSimulator(use_gpu=False, dt_ms=0.25)
    sim.enable_default_syn3a_subsystems()
    sim.run(12)

    states = sim.subsystem_states()
    assert len(states) == 4
    assert {state[0] for state in states} == {
        "atp_synthase_membrane_band",
        "ribosome_polysome_cluster",
        "replisome_track",
        "ftsz_septum_ring",
    }
    assert all(len(state[10]) == 4 for state in states)
    assert all(state[10][1] > 0.0 for state in states)
    assert all(len(state[12]) == 4 for state in states)
    assert all(state[12][3] != 0.0 for state in states)


def test_rust_whole_cell_local_chemistry_sites_if_available():
    if RustWholeCellSimulator is None:
        pytest.skip("oneuro_metal extension not available")

    sim = RustWholeCellSimulator(use_gpu=False, dt_ms=0.25)
    sim.enable_default_syn3a_subsystems()
    sim.run(8)

    site_reports = sim.local_chemistry_sites()
    assert len(site_reports) == 4
    assert {report[0] for report in site_reports} == {
        "atp_synthase_membrane_band",
        "ribosome_polysome_cluster",
        "replisome_track",
        "ftsz_septum_ring",
    }
    assert all(len(report[5]) == 4 for report in site_reports)
    assert all(report[5][1] > 0.0 for report in site_reports)
    assert all(len(report[6]) == 4 for report in site_reports)
    assert all(report[6][3] != 0.0 for report in site_reports)


def test_rust_whole_cell_derivation_calibration_if_available():
    if RustWholeCellSimulator is None:
        pytest.skip("oneuro_metal extension not available")

    sim = RustWholeCellSimulator(use_gpu=False, dt_ms=0.25)
    sim.enable_local_chemistry(12, 12, 6, 0.5, False)

    samples = sim.derivation_calibration_samples(0.25, 2)
    assert len(samples) == 4
    assert {sample["preset"] for sample in samples} == {
        "atp_synthase_membrane_band",
        "ribosome_polysome_cluster",
        "replisome_track",
        "ftsz_septum_ring",
    }
    assert all(sample["site_report"]["mean_nitrate"] >= 0.0 for sample in samples)
    assert all(sample["md_report"]["structural_order"] > 0.0 for sample in samples)
    assert all(sample["md_report"]["bond_density"] > 0.0 for sample in samples)
    assert all(sample["md_report"]["thermal_stability"] > 0.0 for sample in samples)

    fit = sim.fit_derivation_calibration(0.25, 2)
    assert fit is not None
    assert fit["sample_count"] == 4
    assert fit["fitted_loss"] < fit["baseline_loss"]
    assert fit["calibration"]["reaction_focus_gain"] > 0.0


def test_rust_whole_cell_bundled_spec_and_restart_if_available():
    if RustWholeCellSimulator is None:
        pytest.skip("oneuro_metal extension not available")

    spec_json = RustWholeCellSimulator.bundled_syn3a_reference_spec_json()
    organism_json = RustWholeCellSimulator.bundled_syn3a_organism_spec_json()
    asset_json = RustWholeCellSimulator.bundled_syn3a_genome_asset_package_json()
    assert "jcvi_syn3a_reference_native" in spec_json
    assert "JCVI-syn3A" in organism_json
    assert "ribosome_biogenesis_operon_complex" in asset_json

    sim = RustWholeCellSimulator.from_program_spec_json(spec_json)
    summary = sim.organism_summary()
    asset_summary = sim.organism_asset_summary()
    process_registry_summary = sim.organism_process_registry_summary()
    process_registry = sim.organism_process_registry()
    profile = sim.organism_profile()
    expression = sim.organism_expression_state()
    complex_assembly = sim.complex_assembly_state()
    named_complexes = sim.named_complexes_state()
    sim.run(6)
    saved = sim.save_state_json()
    restored = RustWholeCellSimulator.from_saved_state_json(saved)
    saved_payload = json.loads(saved)

    sim_snapshot = sim.snapshot()
    restored_snapshot = restored.snapshot()
    restored_complex = restored.complex_assembly_state()
    restored_named_complexes = restored.named_complexes_state()

    assert summary is not None
    assert summary["organism"] == "JCVI-syn3A"
    assert summary["gene_count"] >= 10
    assert asset_summary is not None
    assert asset_summary["operon_count"] >= summary["transcription_unit_count"]
    assert asset_summary["protein_count"] == summary["gene_count"]
    assert asset_summary["targeted_complex_count"] >= 4
    assert process_registry_summary is not None
    assert process_registry_summary["species_count"] > asset_summary["protein_count"]
    assert process_registry_summary["rna_species_count"] == asset_summary["rna_count"]
    assert process_registry_summary["protein_species_count"] == asset_summary["protein_count"]
    assert process_registry_summary["complex_species_count"] == asset_summary["complex_count"]
    assert (
        process_registry_summary["assembly_intermediate_species_count"]
        >= asset_summary["complex_count"] * 3
    )
    assert process_registry_summary["translation_reaction_count"] >= asset_summary["protein_count"]
    assert process_registry is not None
    assert any(
        species["id"] == "ribosome_biogenesis_operon_complex_mature"
        for species in process_registry["species"]
    )
    assert any(
        reaction["id"] == "ribosome_biogenesis_operon_complex_maturation"
        for reaction in process_registry["reactions"]
    )
    assert profile is not None
    assert profile["process_scales"]["translation"] > 0.9
    assert profile["metabolic_burden_scale"] > 0.9
    assert expression is not None
    assert expression["global_activity"] > 0.0
    assert expression["total_transcript_abundance"] > 0.0
    assert expression["total_protein_abundance"] > 0.0
    assert "ribosome_biogenesis_operon" in expression["transcription_units"]
    assert (
        expression["transcription_units"]["ribosome_biogenesis_operon"]["protein_abundance"] > 0.0
    )
    assert complex_assembly["total_complexes"] > 0.0
    assert complex_assembly["ribosome_target"] > 0.0
    assert len(named_complexes) == asset_summary["complex_count"]
    assert any(
        state["id"] == "ribosome_biogenesis_operon_complex"
        and state["subunit_pool"] > 0.0
        and state["nucleation_intermediate"] > 0.0
        and state["elongation_intermediate"] > 0.0
        and state["abundance"] > 0.0
        and state["component_satisfaction"] > 0.0
        and state["assembly_progress"] > 0.0
        for state in named_complexes
    )
    assert restored_complex["ribosome_complexes"] == pytest.approx(
        sim.complex_assembly_state()["ribosome_complexes"]
    )
    assert len(restored_named_complexes) == len(named_complexes)
    assert next(
        state["abundance"]
        for state in restored_named_complexes
        if state["id"] == "ribosome_biogenesis_operon_complex"
    ) == pytest.approx(
        next(
            state["abundance"]
            for state in sim.named_complexes_state()
            if state["id"] == "ribosome_biogenesis_operon_complex"
        )
    )
    assert next(
        state["subunit_pool"]
        for state in restored_named_complexes
        if state["id"] == "ribosome_biogenesis_operon_complex"
    ) == pytest.approx(
        next(
            state["subunit_pool"]
            for state in sim.named_complexes_state()
            if state["id"] == "ribosome_biogenesis_operon_complex"
        )
    )
    assert restored_snapshot[2] == sim_snapshot[2]
    assert restored_snapshot[7] == sim_snapshot[7]
    assert restored_snapshot[3] == pytest.approx(sim_snapshot[3])
    assert saved_payload["contract"]["contract_version"] == WHOLE_CELL_CONTRACT_VERSION
    assert saved_payload["provenance"]["backend"] in {"cpu", "metal"}
    assert saved_payload["provenance"]["organism_asset_hash"]
    assert saved_payload["provenance"]["run_manifest_hash"]
