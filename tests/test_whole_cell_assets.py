import json
from pathlib import Path

import pytest

from oneuro.whole_cell import (
    RustWholeCellSimulator,
    available_bundles,
    compile_bundle_manifest,
    compile_named_bundle,
    write_structured_bundle_sources,
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
    expected_source_keys = {
        "manifest.json",
        "metadata_json",
        "gene_features_json",
        "gene_products_json",
        "gene_semantics_json",
        "transcription_units_json",
        "transcription_unit_semantics_json",
        "chromosome_domains_json",
        "pools_json",
        "operons_json",
        "rnas_json",
        "proteins_json",
        "complexes_json",
        "operon_semantics_json",
        "protein_semantics_json",
        "complex_semantics_json",
        "program_defaults_json",
    }

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
    assert set(bundle.source_hashes) == expected_source_keys


def test_compile_demo_bundle_from_fasta_and_gff_sources(tmp_path):
    bundle = compile_named_bundle("mgen_minimal_demo")
    summary = bundle.summary()
    written = write_compiled_bundle(bundle, tmp_path)
    bulk_fields = {pool.get("bulk_field") for pool in bundle.organism_spec["pools"]}
    transcription_units = bundle.organism_spec["transcription_units"]
    genes = bundle.organism_spec["genes"]
    expected_source_keys = {
        "manifest.json",
        "metadata_json",
        "genome_fasta",
        "gene_features_gff",
        "gene_products_json",
        "gene_semantics_json",
        "transcription_units_json",
        "transcription_unit_semantics_json",
        "chromosome_domains_json",
        "pools_json",
        "operons_json",
        "rnas_json",
        "proteins_json",
        "complexes_json",
        "operon_semantics_json",
        "protein_semantics_json",
        "complex_semantics_json",
        "program_defaults_json",
    }

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
    assert set(bundle.source_hashes) == expected_source_keys
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


def test_structured_bundle_export_round_trips_syn3a_python(tmp_path):
    bundle = compile_named_bundle("jcvi_syn3a")
    written = write_structured_bundle_sources(
        bundle.organism_spec,
        tmp_path / "syn3a_structured",
        source_dataset="jcvi_syn3a_structured_export",
    )
    round_tripped = compile_bundle_manifest(written["manifest"])

    assert round_tripped.organism == bundle.organism
    assert round_tripped.summary()["gene_count"] == bundle.summary()["gene_count"]
    assert (
        round_tripped.summary()["transcription_unit_count"]
        == bundle.summary()["transcription_unit_count"]
    )
    assert (
        round_tripped.summary()["protein_count"] == bundle.summary()["protein_count"]
    )
    assert round_tripped.organism_spec["chromosome_length_bp"] == bundle.organism_spec[
        "chromosome_length_bp"
    ]
    assert "gene_semantics_json" in round_tripped.source_hashes
    assert "transcription_unit_semantics_json" in round_tripped.source_hashes
    assert "chromosome_domains_json" in round_tripped.source_hashes
    assert "operons_json" in round_tripped.source_hashes
    assert "rnas_json" in round_tripped.source_hashes
    assert "proteins_json" in round_tripped.source_hashes
    assert "complexes_json" in round_tripped.source_hashes
    assert "operon_semantics_json" in round_tripped.source_hashes
    assert "protein_semantics_json" in round_tripped.source_hashes
    assert "complex_semantics_json" in round_tripped.source_hashes
    assert "program_defaults_json" in round_tripped.source_hashes


def test_structured_bundle_export_round_trips_syn3a_rust_if_available(tmp_path):
    if RustWholeCellSimulator is None:
        return

    bundle = compile_named_bundle("jcvi_syn3a")
    written = write_structured_bundle_sources(
        bundle.organism_spec,
        tmp_path / "syn3a_structured",
        source_dataset="jcvi_syn3a_structured_export",
    )
    compiled_spec = RustWholeCellSimulator.compile_bundle_manifest_organism_spec_json(
        written["manifest"]
    )
    sim = RustWholeCellSimulator.from_bundle_manifest_path(written["manifest"])
    summary = sim.organism_summary()

    assert "\"JCVI-syn3A\"" in compiled_spec
    assert summary["organism"] == "JCVI-syn3A"
    assert summary["gene_count"] >= bundle.summary()["gene_count"]


def test_explicit_semantic_bundle_rejects_missing_gene_overlay(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/mgen_minimal_demo").resolve()
    for name in [
        "metadata.json",
        "genome.fasta",
        "features.gff3",
        "gene_products.json",
        "chromosome_domains.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "pools.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(ValueError, match="missing gene_semantics_json"):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_bundle_rejects_organism_spec_json(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    organism_spec = {
        "organism": "Bad-bundle",
        "chromosome_length_bp": 1000,
        "origin_bp": 0,
        "terminus_bp": 500,
        "geometry": {"radius_nm": 100.0, "chromosome_radius_fraction": 0.5, "membrane_fraction": 0.2},
        "composition": {"dry_mass_fg": 1.0, "gc_fraction": 0.3, "protein_fraction": 0.5, "rna_fraction": 0.2, "lipid_fraction": 0.1},
        "chromosome_domains": [],
        "pools": [],
        "genes": [],
        "transcription_units": [],
    }
    (bundle_dir / "organism.json").write_text(
        json.dumps(organism_spec, indent=2), encoding="ascii"
    )
    manifest = {
        "organism": "Bad-bundle",
        "require_structured_bundle": True,
        "organism_spec_json": "organism.json",
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match="bundle manifests may not define organism_spec_json; use explicit structured bundle sources",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_explicit_asset_bundle_rejects_missing_operon_source(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    manifest.pop("operons_json", None)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError, match="requires explicit asset entities but is missing operons_json"
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_missing_transcription_unit_source(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    manifest.pop("transcription_units_json", None)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match="requires explicit organism sources but is missing transcription_units_json",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_missing_pool_bulk_field(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    pools = json.loads((source_dir / "pools.json").read_text(encoding="ascii"))
    for pool in pools:
        if pool["species"] == "ATP":
            pool.pop("bulk_field", None)
            break
    (bundle_dir / "pools.json").write_text(json.dumps(pools, indent=2), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match="requires explicit pool metadata but 1 pool\\(s\\) are incomplete: ATP",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_incomplete_operon_asset_entity(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    operons = json.loads((source_dir / "operons.json").read_text(encoding="ascii"))
    target = operons[0]["name"]
    operons[0].pop("asset_class", None)
    (bundle_dir / "operons.json").write_text(
        json.dumps(operons, indent=2), encoding="ascii"
    )
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match=rf"requires explicit asset entities but 1 operon\(s\) are incomplete: {target}",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_missing_operon_semantic_coverage(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    operon_semantics = json.loads(
        (source_dir / "operon_semantics.json").read_text(encoding="ascii")
    )
    removed = operon_semantics.pop(0)["name"]
    (bundle_dir / "operon_semantics.json").write_text(
        json.dumps(operon_semantics, indent=2), encoding="ascii"
    )
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match=rf"requires explicit asset semantics but 1 operon semantic entry\(s\) are incomplete: {removed}",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_missing_operon_entity_coverage(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
        "program_defaults.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    operons = json.loads((source_dir / "operons.json").read_text(encoding="ascii"))
    removed = operons.pop(0)["name"]
    (bundle_dir / "operons.json").write_text(
        json.dumps(operons, indent=2), encoding="ascii"
    )
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match=rf"requires explicit asset entity coverage but 1 operon\(s\) are missing: {removed}",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")


def test_strict_structured_bundle_rejects_missing_program_defaults(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source_dir = Path("src/oneuro/whole_cell/assets/bundles/jcvi_syn3a").resolve()
    for name in [
        "metadata.json",
        "gene_features.json",
        "gene_products.json",
        "gene_semantics.json",
        "transcription_units.json",
        "transcription_unit_semantics.json",
        "chromosome_domains.json",
        "pools.json",
        "operons.json",
        "rnas.json",
        "proteins.json",
        "complexes.json",
        "operon_semantics.json",
        "protein_semantics.json",
        "complex_semantics.json",
    ]:
        (bundle_dir / name).write_text((source_dir / name).read_text(), encoding="ascii")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="ascii"))
    manifest.pop("program_defaults_json", None)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )

    with pytest.raises(
        ValueError,
        match="requires explicit program defaults but is missing program_defaults_json",
    ):
        compile_bundle_manifest(bundle_dir / "manifest.json")
