import json
from pathlib import Path

import pytest

from oneuro.whole_cell import (
    RustWholeCellSimulator,
    available_bundles,
    compile_bundle_manifest,
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
    bulk_fields = {pool.get("bulk_field") for pool in bundle.organism_spec["pools"]}
    transcription_units = bundle.organism_spec["transcription_units"]
    genes = bundle.organism_spec["genes"]

    assert bundle.organism == "JCVI-syn3A"
    assert summary["gene_count"] >= 10
    assert summary["transcription_unit_count"] >= 4
    assert summary["operon_count"] >= summary["transcription_unit_count"]
    assert summary["protein_count"] == summary["gene_count"]
    assert summary["targeted_complex_count"] >= 4
    assert len(bundle.organism_spec["chromosome_domains"]) >= 4
    assert len(bundle.genome_asset_package["chromosome_domains"]) == len(
        bundle.organism_spec["chromosome_domains"]
    )
    assert len(bundle.genome_asset_package["operon_semantics"]) == len(
        bundle.genome_asset_package["operons"]
    )
    assert len(bundle.genome_asset_package["protein_semantics"]) == len(
        bundle.genome_asset_package["proteins"]
    )
    assert len(bundle.genome_asset_package["complex_semantics"]) == len(
        bundle.genome_asset_package["complexes"]
    )
    assert {"a_t_p", "amino_acids", "nucleotides", "membrane_precursors"} <= bulk_fields
    assert any(unit.get("asset_class") for unit in transcription_units)
    assert any(unit.get("complex_family") for unit in transcription_units)
    assert any(gene.get("asset_class") for gene in genes)
    assert any(gene.get("complex_family") for gene in genes)
    assert all(
        semantic.get("asset_class") and semantic.get("complex_family")
        for semantic in bundle.genome_asset_package["operon_semantics"]
    )
    assert all(
        semantic.get("asset_class")
        for semantic in bundle.genome_asset_package["protein_semantics"]
    )
    assert all(
        semantic.get("asset_class") and semantic.get("family")
        for semantic in bundle.genome_asset_package["complex_semantics"]
    )
    assert "organism_spec_json" in bundle.source_hashes


def test_compile_demo_bundle_from_fasta_and_gff_sources(tmp_path):
    bundle = compile_named_bundle("mgen_minimal_demo")
    summary = bundle.summary()
    written = write_compiled_bundle(bundle, tmp_path)
    bulk_fields = {pool.get("bulk_field") for pool in bundle.organism_spec["pools"]}
    transcription_units = bundle.organism_spec["transcription_units"]
    genes = bundle.organism_spec["genes"]

    assert bundle.organism == "Mgen-minimal-demo"
    assert bundle.organism_spec["chromosome_length_bp"] > 1000
    assert summary["gene_count"] == 4
    assert summary["transcription_unit_count"] == 3
    assert summary["polycistronic_operon_count"] == 1
    assert summary["protein_count"] == 4
    assert summary["complex_count"] >= 3
    assert len(bundle.organism_spec["chromosome_domains"]) >= 4
    assert len(bundle.genome_asset_package["operon_semantics"]) == len(
        bundle.genome_asset_package["operons"]
    )
    assert len(bundle.genome_asset_package["protein_semantics"]) == len(
        bundle.genome_asset_package["proteins"]
    )
    assert len(bundle.genome_asset_package["complex_semantics"]) == len(
        bundle.genome_asset_package["complexes"]
    )
    assert {"a_t_p", "amino_acids", "nucleotides", "membrane_precursors"} <= bulk_fields
    assert all(unit.get("asset_class") for unit in transcription_units)
    assert all(unit.get("complex_family") for unit in transcription_units)
    assert all(gene.get("asset_class") for gene in genes)
    assert all(gene.get("complex_family") for gene in genes)
    assert all(
        semantic.get("asset_class") and semantic.get("complex_family")
        for semantic in bundle.genome_asset_package["operon_semantics"]
    )
    assert all(
        semantic.get("asset_class")
        for semantic in bundle.genome_asset_package["protein_semantics"]
    )
    assert all(
        semantic.get("asset_class") and semantic.get("family")
        for semantic in bundle.genome_asset_package["complex_semantics"]
    )
    assert "genome_fasta" in bundle.source_hashes
    assert "gene_features_gff" in bundle.source_hashes
    assert "gene_semantics_json" in bundle.source_hashes
    assert "transcription_unit_semantics_json" in bundle.source_hashes
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
    compiled_registry = RustWholeCellSimulator.compile_bundle_manifest_process_registry_json(
        str(manifest_path)
    )
    sim = RustWholeCellSimulator.from_bundle_manifest_path(str(manifest_path))
    summary = sim.organism_summary()

    assert "\"Mgen-minimal-demo\"" in compiled_spec
    assert "\"compiled_ir_hash\"" in compiled_spec
    assert "\"complex_maturation\"" in compiled_registry
    assert "\"pool_transport\"" in compiled_registry
    assert "\"rna_degradation\"" in compiled_registry
    assert "\"protein_degradation\"" in compiled_registry
    assert "\"stress_response\"" in compiled_registry
    assert "\"complex_repair\"" in compiled_registry
    assert "\"chromosome_domains\"" in compiled_spec
    assert "\"chromosome_domain\"" in compiled_registry
    assert "\"bulk_field\"" in compiled_spec
    assert summary is not None
    assert summary["organism"] == "Mgen-minimal-demo"


def test_explicit_semantic_bundle_rejects_missing_gene_overlay(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo").resolve()
    for name in [
        "metadata.json",
        "genome.fasta",
        "features.gff3",
        "gene_products.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "pools.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(ValueError, match="missing gene_semantics_json"):
        compile_bundle_manifest(bundle_dir / "manifest.json")
