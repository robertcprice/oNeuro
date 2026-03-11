from pathlib import Path

from oneuro.whole_cell import (
    RustWholeCellSimulator,
    available_bundles,
    compile_named_bundle,
    write_compiled_bundle,
)


def test_available_bundles_include_syn3a_and_demo_bundle():
    bundles = available_bundles()

    assert "jcvi_syn3a" in bundles
    assert "mgen_minimal_demo" in bundles


def test_compile_syn3a_bundle_matches_current_runtime_shape():
    bundle = compile_named_bundle("jcvi_syn3a")
    summary = bundle.summary()

    assert bundle.organism == "JCVI-syn3A"
    assert summary["gene_count"] >= 10
    assert summary["transcription_unit_count"] >= 4
    assert summary["operon_count"] >= summary["transcription_unit_count"]
    assert summary["protein_count"] == summary["gene_count"]
    assert summary["targeted_complex_count"] >= 4
    assert "organism_spec_json" in bundle.source_hashes


def test_compile_demo_bundle_from_fasta_and_gff_sources(tmp_path):
    bundle = compile_named_bundle("mgen_minimal_demo")
    summary = bundle.summary()
    written = write_compiled_bundle(bundle, tmp_path)

    assert bundle.organism == "Mgen-minimal-demo"
    assert bundle.organism_spec["chromosome_length_bp"] > 1000
    assert summary["gene_count"] == 4
    assert summary["transcription_unit_count"] == 3
    assert summary["polycistronic_operon_count"] == 1
    assert summary["protein_count"] == 4
    assert summary["complex_count"] >= 3
    assert "genome_fasta" in bundle.source_hashes
    assert "gene_features_gff" in bundle.source_hashes
    assert Path(written["organism_spec"]).exists()
    assert Path(written["genome_assets"]).exists()
    assert Path(written["summary"]).exists()


def test_rust_bundle_manifest_ingestion_if_available():
    if RustWholeCellSimulator is None:
        return

    manifest_path = Path(
        "src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo/manifest.json"
    ).resolve()
    compiled_spec = RustWholeCellSimulator.compile_bundle_manifest_program_spec_json(
        str(manifest_path)
    )
    sim = RustWholeCellSimulator.from_bundle_manifest_path(str(manifest_path))
    summary = sim.organism_summary()

    assert "\"Mgen-minimal-demo\"" in compiled_spec
    assert summary is not None
    assert summary["organism"] == "Mgen-minimal-demo"
