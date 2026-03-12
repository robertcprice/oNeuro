"""Manifest-driven compiler for whole-cell organism source bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from .derived_assets import (
    _compile_genome_asset_package,
    _derive_complex_semantics,
    _derive_operon_semantics,
    _derive_protein_semantics,
    _operon_bounds,
)

_BUNDLES_ROOT = Path(__file__).resolve().parent / "bundles"


@dataclass(frozen=True)
class CompiledOrganismBundle:
    """Compiled organism spec plus derived runtime asset package."""

    manifest_path: str
    organism: str
    organism_spec: Dict[str, Any]
    genome_asset_package: Dict[str, Any]
    source_hashes: Dict[str, str]

    def summary(self) -> Dict[str, Any]:
        operons = self.genome_asset_package.get("operons", [])
        complexes = self.genome_asset_package.get("complexes", [])
        return {
            "organism": self.organism,
            "gene_count": len(self.organism_spec.get("genes", [])),
            "transcription_unit_count": len(
                self.organism_spec.get("transcription_units", [])
            ),
            "operon_count": len(operons),
            "polycistronic_operon_count": sum(
                1 for operon in operons if operon.get("polycistronic")
            ),
            "rna_count": len(self.genome_asset_package.get("rnas", [])),
            "protein_count": len(self.genome_asset_package.get("proteins", [])),
            "complex_count": len(complexes),
            "targeted_complex_count": sum(
                1
                for complex_spec in complexes
                if complex_spec.get("subsystem_targets")
            ),
        }


def available_bundles() -> tuple[str, ...]:
    """Return bundle names shipped with the package."""

    if not _BUNDLES_ROOT.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in _BUNDLES_ROOT.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        )
    )


def compile_named_bundle(name: str) -> CompiledOrganismBundle:
    """Compile one of the packaged organism bundles by name."""

    manifest_path = (_BUNDLES_ROOT / name / "manifest.json").resolve()
    return compile_bundle_manifest(manifest_path)


def compile_legacy_named_bundle(name: str) -> CompiledOrganismBundle:
    """Compile a packaged organism bundle through the legacy derived-asset path."""

    manifest_path = (_BUNDLES_ROOT / name / "manifest.json").resolve()
    return compile_legacy_bundle_manifest(manifest_path)


def compile_bundle_manifest(manifest_path: Path | str) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest into runtime-ready JSON payloads."""

    return _compile_bundle_manifest_impl(manifest_path, allow_legacy_derived_assets=False)


def compile_legacy_bundle_manifest(manifest_path: Path | str) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest through the legacy derived-asset path."""

    return _compile_bundle_manifest_impl(manifest_path, allow_legacy_derived_assets=True)


def _compile_bundle_manifest_impl(
    manifest_path: Path | str, *, allow_legacy_derived_assets: bool
) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest into runtime-ready JSON payloads."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path)
    source_hashes: Dict[str, str] = {"manifest.json": _sha256_path(path)}
    _validate_manifest_mode(manifest)
    _validate_bundle_compile_entrypoint(
        manifest, allow_legacy_derived_assets=allow_legacy_derived_assets
    )
    manifest_allows_legacy_derived_assets = bool(
        manifest.get("allow_legacy_derived_assets")
    )
    require_explicit_asset_entities = bool(manifest.get("require_explicit_asset_entities"))
    require_explicit_asset_semantics = bool(
        manifest.get("require_explicit_asset_semantics")
    )
    organism_spec = _compile_structured_bundle(path, manifest, source_hashes)
    operons_overlay = _load_optional_json(path, manifest, "operons_json", source_hashes) or []
    rnas_overlay = _load_optional_json(path, manifest, "rnas_json", source_hashes) or []
    proteins_overlay = (
        _load_optional_json(path, manifest, "proteins_json", source_hashes) or []
    )
    complexes_overlay = (
        _load_optional_json(path, manifest, "complexes_json", source_hashes) or []
    )
    operon_semantics_overlay = (
        _load_optional_json(path, manifest, "operon_semantics_json", source_hashes) or []
    )
    protein_semantics_overlay = (
        _load_optional_json(path, manifest, "protein_semantics_json", source_hashes) or []
    )
    complex_semantics_overlay = (
        _load_optional_json(path, manifest, "complex_semantics_json", source_hashes) or []
    )
    _load_optional_json(path, manifest, "program_defaults_json", source_hashes)
    _validate_explicit_asset_contracts(manifest, source_hashes)

    organism_spec = _with_compiled_chromosome_domains(organism_spec)
    if require_explicit_asset_entities:
        asset_package = _empty_genome_asset_package(organism_spec)
    else:
        assert manifest_allows_legacy_derived_assets
        asset_package = _compile_genome_asset_package(organism_spec)
    asset_package = _apply_asset_entity_overlays(
        asset_package,
        operons_overlay,
        rnas_overlay,
        proteins_overlay,
        complexes_overlay,
        derive_semantics=not require_explicit_asset_semantics,
    )
    if require_explicit_asset_entities:
        _validate_explicit_asset_entities(asset_package)
        _validate_explicit_asset_entity_coverage(organism_spec, asset_package)
    asset_package = _apply_asset_semantic_overlays(
        asset_package,
        operon_semantics_overlay,
        protein_semantics_overlay,
        complex_semantics_overlay,
    )
    if require_explicit_asset_semantics:
        _validate_explicit_asset_semantics(asset_package)
    return CompiledOrganismBundle(
        manifest_path=str(path),
        organism=organism_spec["organism"],
        organism_spec=organism_spec,
        genome_asset_package=asset_package,
        source_hashes=source_hashes,
    )


def _validate_manifest_mode(manifest: Dict[str, Any]) -> None:
    if "organism_spec_json" in manifest:
        raise ValueError(
            "bundle manifests may not define organism_spec_json; "
            "use explicit structured bundle sources"
        )
    if manifest.get("require_explicit_organism_sources"):
        missing = []
        if "metadata_json" not in manifest:
            missing.append("metadata_json")
        if "gene_features_json" not in manifest and "gene_features_gff" not in manifest:
            missing.append("gene_features_json|gene_features_gff")
        if "gene_products_json" not in manifest:
            missing.append("gene_products_json")
        if "transcription_units_json" not in manifest:
            missing.append("transcription_units_json")
        if "chromosome_domains_json" not in manifest:
            missing.append("chromosome_domains_json")
        if "pools_json" not in manifest:
            missing.append("pools_json")
        if missing:
            raise ValueError(
                "bundle requires explicit organism sources but is missing "
                + ", ".join(missing)
            )


def _validate_explicit_asset_contracts(
    manifest: Dict[str, Any],
    source_hashes: Dict[str, str],
) -> None:
    allow_legacy_derived_assets = bool(manifest.get("allow_legacy_derived_assets"))
    require_explicit_asset_entities = bool(
        manifest.get("require_explicit_asset_entities")
    )
    require_explicit_asset_semantics = bool(
        manifest.get("require_explicit_asset_semantics")
    )

    if allow_legacy_derived_assets and (
        require_explicit_asset_entities or require_explicit_asset_semantics
    ):
        raise ValueError(
            "allow_legacy_derived_assets is incompatible with explicit asset entity or semantic requirements"
        )
    if not require_explicit_asset_entities and not allow_legacy_derived_assets:
        raise ValueError(
            "bundle must declare explicit asset entities or set allow_legacy_derived_assets"
        )
    if not require_explicit_asset_semantics and not allow_legacy_derived_assets:
        raise ValueError(
            "bundle must declare explicit asset semantics or set allow_legacy_derived_assets"
        )
    if require_explicit_asset_entities:
        required_entity_keys = {
            "operons_json",
            "rnas_json",
            "proteins_json",
            "complexes_json",
        }
        missing = sorted(key for key in required_entity_keys if key not in source_hashes)
        if missing:
            raise ValueError(
                "bundle requires explicit asset entities but is missing "
                + ", ".join(missing)
            )
    if require_explicit_asset_semantics:
        required_semantic_keys = {
            "operon_semantics_json",
            "protein_semantics_json",
            "complex_semantics_json",
        }
        missing = sorted(key for key in required_semantic_keys if key not in source_hashes)
        if missing:
            raise ValueError(
                "bundle requires explicit asset semantics but is missing "
                + ", ".join(missing)
            )
    if manifest.get("require_explicit_program_defaults"):
        if "program_defaults_json" not in source_hashes:
            raise ValueError(
                "bundle requires explicit program defaults but is missing "
                "program_defaults_json"
            )


def _validate_bundle_compile_entrypoint(
    manifest: Dict[str, Any], *, allow_legacy_derived_assets: bool
) -> None:
    manifest_allows_legacy_derived_assets = bool(
        manifest.get("allow_legacy_derived_assets")
    )
    if manifest_allows_legacy_derived_assets and not allow_legacy_derived_assets:
        raise ValueError(
            "legacy-derived-asset bundles must use compile_legacy_bundle_manifest"
        )
    if allow_legacy_derived_assets and not manifest_allows_legacy_derived_assets:
        raise ValueError(
            "compile_legacy_bundle_manifest requires allow_legacy_derived_assets in the manifest"
        )


def _apply_asset_semantic_overlays(
    asset_package: Dict[str, Any],
    operon_semantics_overlay: list[Dict[str, Any]],
    protein_semantics_overlay: list[Dict[str, Any]],
    complex_semantics_overlay: list[Dict[str, Any]],
) -> Dict[str, Any]:
    compiled = dict(asset_package)
    operons = [dict(operon) for operon in compiled.get("operons", [])]
    proteins = [dict(protein) for protein in compiled.get("proteins", [])]
    complexes = [dict(complex_spec) for complex_spec in compiled.get("complexes", [])]

    operon_semantics = {
        semantic["name"]: dict(semantic)
        for semantic in compiled.get("operon_semantics", [])
    }
    for semantic in operon_semantics_overlay:
        merged = operon_semantics.setdefault(semantic["name"], {"name": semantic["name"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        if semantic.get("complex_family"):
            merged["complex_family"] = semantic["complex_family"]
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for operon in operons:
        semantic = operon_semantics.get(operon["name"])
        if semantic is None:
            continue
        operon["asset_class"] = semantic.get("asset_class", operon.get("asset_class"))
        operon["complex_family"] = semantic.get(
            "complex_family", operon.get("complex_family")
        )
        targets = list(operon.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        operon["subsystem_targets"] = targets

    protein_semantics = {
        semantic["id"]: dict(semantic)
        for semantic in compiled.get("protein_semantics", [])
    }
    for semantic in protein_semantics_overlay:
        merged = protein_semantics.setdefault(semantic["id"], {"id": semantic["id"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for protein in proteins:
        semantic = protein_semantics.get(protein["id"])
        if semantic is None:
            continue
        protein["asset_class"] = semantic.get("asset_class", protein.get("asset_class"))
        targets = list(protein.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        protein["subsystem_targets"] = targets

    complex_semantics = {
        semantic["id"]: dict(semantic)
        for semantic in compiled.get("complex_semantics", [])
    }
    for semantic in complex_semantics_overlay:
        merged = complex_semantics.setdefault(semantic["id"], {"id": semantic["id"]})
        if semantic.get("asset_class"):
            merged["asset_class"] = semantic["asset_class"]
        if semantic.get("family"):
            merged["family"] = semantic["family"]
        for key in ("membrane_inserted", "chromosome_coupled", "division_coupled"):
            if key in semantic:
                merged[key] = bool(semantic[key])
        merged_targets = list(merged.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in merged_targets:
                merged_targets.append(target)
        merged["subsystem_targets"] = merged_targets
    for complex_spec in complexes:
        semantic = complex_semantics.get(complex_spec["id"])
        if semantic is None:
            continue
        complex_spec["asset_class"] = semantic.get(
            "asset_class", complex_spec.get("asset_class")
        )
        complex_spec["family"] = semantic.get("family", complex_spec.get("family"))
        for key in ("membrane_inserted", "chromosome_coupled", "division_coupled"):
            if key in semantic:
                complex_spec[key] = bool(semantic[key])
        targets = list(complex_spec.get("subsystem_targets", []))
        for target in semantic.get("subsystem_targets", []):
            if target not in targets:
                targets.append(target)
        complex_spec["subsystem_targets"] = targets

    compiled["operons"] = operons
    compiled["proteins"] = proteins
    compiled["complexes"] = complexes
    compiled["operon_semantics"] = sorted(
        operon_semantics.values(), key=lambda semantic: semantic["name"]
    )
    compiled["protein_semantics"] = sorted(
        protein_semantics.values(), key=lambda semantic: semantic["id"]
    )
    compiled["complex_semantics"] = sorted(
        complex_semantics.values(), key=lambda semantic: semantic["id"]
    )
    return compiled


def _apply_asset_entity_overlays(
    asset_package: Dict[str, Any],
    operons_overlay: list[Dict[str, Any]],
    rnas_overlay: list[Dict[str, Any]],
    proteins_overlay: list[Dict[str, Any]],
    complexes_overlay: list[Dict[str, Any]],
    *,
    derive_semantics: bool = True,
) -> Dict[str, Any]:
    compiled = dict(asset_package)
    entity_overrides = False
    if operons_overlay:
        compiled["operons"] = [dict(operon) for operon in operons_overlay]
        entity_overrides = True
    if rnas_overlay:
        compiled["rnas"] = [dict(rna) for rna in rnas_overlay]
        entity_overrides = True
    if proteins_overlay:
        compiled["proteins"] = [dict(protein) for protein in proteins_overlay]
        entity_overrides = True
    if complexes_overlay:
        compiled["complexes"] = [dict(complex_spec) for complex_spec in complexes_overlay]
        entity_overrides = True
    if entity_overrides:
        if derive_semantics:
            compiled["operon_semantics"] = _derive_operon_semantics(compiled)
            compiled["protein_semantics"] = _derive_protein_semantics(compiled)
            compiled["complex_semantics"] = _derive_complex_semantics(compiled)
        else:
            compiled["operon_semantics"] = []
            compiled["protein_semantics"] = []
            compiled["complex_semantics"] = []
    return compiled


def _validate_explicit_asset_entities(asset_package: Dict[str, Any]) -> None:
    operons = asset_package.get("operons", [])
    proteins = asset_package.get("proteins", [])
    complexes = asset_package.get("complexes", [])

    missing_operons = [
        operon["name"]
        for operon in operons
        if not operon.get("asset_class")
        or not operon.get("complex_family")
        or not operon.get("subsystem_targets")
    ]
    if missing_operons:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_operons)} operon(s) are incomplete: "
            + ", ".join(missing_operons)
        )

    missing_proteins = [
        protein["id"]
        for protein in proteins
        if not protein.get("asset_class") or not protein.get("subsystem_targets")
    ]
    if missing_proteins:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_proteins)} protein(s) are incomplete: "
            + ", ".join(missing_proteins)
        )

    missing_complexes = [
        complex_spec["id"]
        for complex_spec in complexes
        if not complex_spec.get("asset_class")
        or not complex_spec.get("family")
        or not complex_spec.get("subsystem_targets")
    ]
    if missing_complexes:
        raise ValueError(
            "bundle requires explicit asset entities but "
            f"{len(missing_complexes)} complex(es) are incomplete: "
            + ", ".join(missing_complexes)
        )


def _validate_explicit_asset_semantics(asset_package: Dict[str, Any]) -> None:
    operon_semantics = {
        semantic["name"]: semantic for semantic in asset_package.get("operon_semantics", [])
    }
    missing_operon_semantics = [
        operon["name"]
        for operon in asset_package.get("operons", [])
        if operon["name"] not in operon_semantics
        or not operon_semantics[operon["name"]].get("subsystem_targets")
    ]
    if missing_operon_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_operon_semantics)} operon semantic entry(s) are incomplete: "
            + ", ".join(missing_operon_semantics)
        )

    protein_semantics = {
        semantic["id"]: semantic for semantic in asset_package.get("protein_semantics", [])
    }
    missing_protein_semantics = [
        protein["id"]
        for protein in asset_package.get("proteins", [])
        if protein["id"] not in protein_semantics
        or not protein_semantics[protein["id"]].get("subsystem_targets")
    ]
    if missing_protein_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_protein_semantics)} protein semantic entry(s) are incomplete: "
            + ", ".join(missing_protein_semantics)
        )

    complex_semantics = {
        semantic["id"]: semantic for semantic in asset_package.get("complex_semantics", [])
    }
    missing_complex_semantics = [
        complex_spec["id"]
        for complex_spec in asset_package.get("complexes", [])
        if complex_spec["id"] not in complex_semantics
        or not complex_semantics[complex_spec["id"]].get("subsystem_targets")
    ]
    if missing_complex_semantics:
        raise ValueError(
            "bundle requires explicit asset semantics but "
            f"{len(missing_complex_semantics)} complex semantic entry(s) are incomplete: "
            + ", ".join(missing_complex_semantics)
        )


def _validate_explicit_asset_entity_coverage(
    organism_spec: Dict[str, Any], asset_package: Dict[str, Any]
) -> None:
    genes = list(organism_spec.get("genes", []))
    transcription_units = list(organism_spec.get("transcription_units", []))
    operons = list(asset_package.get("operons", []))
    rnas = list(asset_package.get("rnas", []))
    proteins = list(asset_package.get("proteins", []))
    complexes = list(asset_package.get("complexes", []))

    genes_in_units = {
        gene_name
        for unit in transcription_units
        for gene_name in unit.get("genes", [])
    }
    expected_operons = {
        unit["name"] for unit in transcription_units
    } | {gene["gene"] for gene in genes if gene["gene"] not in genes_in_units}
    operon_names = {operon["name"] for operon in operons}
    missing_operons = sorted(expected_operons - operon_names)
    if missing_operons:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_operons)} operon(s) are missing: "
            + ", ".join(missing_operons)
        )

    rna_genes = {rna["gene"] for rna in rnas}
    missing_rnas = sorted(gene["gene"] for gene in genes if gene["gene"] not in rna_genes)
    if missing_rnas:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_rnas)} RNA gene(s) are missing: "
            + ", ".join(missing_rnas)
        )

    protein_genes = {protein["gene"] for protein in proteins}
    missing_proteins = sorted(
        gene["gene"] for gene in genes if gene["gene"] not in protein_genes
    )
    if missing_proteins:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_proteins)} protein gene(s) are missing: "
            + ", ".join(missing_proteins)
        )

    complex_operons = {complex_spec["operon"] for complex_spec in complexes}
    missing_complexes = sorted(expected_operons - complex_operons)
    if missing_complexes:
        raise ValueError(
            "bundle requires explicit asset entity coverage but "
            f"{len(missing_complexes)} complex operon(s) are missing: "
            + ", ".join(missing_complexes)
        )


def _compile_structured_bundle(
    manifest_path: Path,
    manifest: Dict[str, Any],
    source_hashes: Dict[str, str],
) -> Dict[str, Any]:
    metadata = _load_optional_json(manifest_path, manifest, "metadata_json", source_hashes)
    pools = _load_optional_json(manifest_path, manifest, "pools_json", source_hashes) or []
    transcription_units = (
        _load_optional_json(
            manifest_path, manifest, "transcription_units_json", source_hashes
        )
        or []
    )
    transcription_unit_semantics = {
        entry["name"]: entry
        for entry in (
            _load_optional_json(
                manifest_path,
                manifest,
                "transcription_unit_semantics_json",
                source_hashes,
            )
            or []
        )
    }
    gene_products = {
        entry["gene"]: entry
        for entry in (
            _load_optional_json(
                manifest_path, manifest, "gene_products_json", source_hashes
            )
            or []
        )
    }
    gene_semantics = {
        entry["gene"]: entry
        for entry in (
            _load_optional_json(
                manifest_path, manifest, "gene_semantics_json", source_hashes
            )
            or []
        )
    }

    chromosome_length_bp = metadata.get("chromosome_length_bp")
    if "genome_fasta" in manifest:
        fasta_path = _resolve_manifest_path(manifest_path, manifest["genome_fasta"])
        source_hashes["genome_fasta"] = _sha256_path(fasta_path)
        fasta = _read_fasta(fasta_path)
        if chromosome_length_bp is None:
            chromosome_length_bp = len(fasta["sequence"])
    if chromosome_length_bp is None:
        raise ValueError("bundle metadata must define chromosome_length_bp or genome_fasta")

    if "gene_features_json" in manifest:
        genes = _load_optional_json(
            manifest_path, manifest, "gene_features_json", source_hashes
        ) or []
    elif "gene_features_gff" in manifest:
        gff_path = _resolve_manifest_path(manifest_path, manifest["gene_features_gff"])
        source_hashes["gene_features_gff"] = _sha256_path(gff_path)
        genes = _read_gff_features(gff_path)
    else:
        raise ValueError("bundle manifest must define gene_features_json or gene_features_gff")

    compiled_genes = [
        _merge_gene_annotation(
            gene,
            gene_products.get(gene["gene"], {}),
            gene_semantics.get(gene["gene"], {}),
        )
        for gene in genes
    ]
    compiled_transcription_units = [
        _merge_transcription_unit_semantics(
            unit,
            transcription_unit_semantics.get(unit["name"], {}),
        )
        for unit in transcription_units
    ]
    chromosome_domains = (
        _load_optional_json(
            manifest_path, manifest, "chromosome_domains_json", source_hashes
        )
        or []
    )
    if manifest.get("require_explicit_organism_sources"):
        _validate_explicit_pool_metadata(pools)
    if manifest.get("require_explicit_gene_semantics"):
        _validate_explicit_gene_semantics(compiled_genes)
    if manifest.get("require_explicit_transcription_unit_semantics"):
        _validate_explicit_transcription_unit_semantics(compiled_transcription_units)

    return {
        "organism": manifest.get("organism") or metadata["organism"],
        "chromosome_length_bp": int(chromosome_length_bp),
        "origin_bp": int(metadata.get("origin_bp", 0)),
        "terminus_bp": int(metadata.get("terminus_bp", chromosome_length_bp // 2)),
        "geometry": metadata["geometry"],
        "composition": metadata["composition"],
        "chromosome_domains": chromosome_domains,
        "pools": pools,
        "genes": compiled_genes,
        "transcription_units": compiled_transcription_units,
    }


def _merge_gene_annotation(
    gene_feature: Dict[str, Any],
    gene_product: Dict[str, Any],
    gene_semantic: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {
        "gene": gene_feature["gene"],
        "start_bp": int(gene_feature["start_bp"]),
        "end_bp": int(gene_feature["end_bp"]),
        "strand": int(gene_feature.get("strand", 1)),
        "essential": bool(gene_product.get("essential", gene_feature.get("essential", False))),
        "basal_expression": float(gene_product.get("basal_expression", 1.0)),
        "translation_cost": float(gene_product.get("translation_cost", 1.0)),
        "nucleotide_cost": float(gene_product.get("nucleotide_cost", 1.0)),
        "process_weights": gene_product.get("process_weights", {}),
        "subsystem_targets": gene_semantic.get(
            "subsystem_targets", gene_product.get("subsystem_targets", [])
        ),
        "asset_class": gene_semantic.get("asset_class", gene_product.get("asset_class")),
        "complex_family": gene_semantic.get(
            "complex_family", gene_product.get("complex_family")
        ),
    }
    return merged


def _merge_transcription_unit_semantics(
    transcription_unit: Dict[str, Any],
    semantic_annotation: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(transcription_unit)
    if semantic_annotation:
        merged["subsystem_targets"] = list(semantic_annotation.get("subsystem_targets", []))
        merged["asset_class"] = semantic_annotation.get("asset_class")
        merged["complex_family"] = semantic_annotation.get("complex_family")
    return merged


def _validate_explicit_gene_semantics(genes: list[Dict[str, Any]]) -> None:
    missing = [
        gene["gene"]
        for gene in genes
        if not gene.get("asset_class")
        or not gene.get("complex_family")
        or not gene.get("subsystem_targets")
    ]
    if missing:
        raise ValueError(
            "bundle requires explicit gene semantics but "
            f"{len(missing)} gene(s) are incomplete: {', '.join(missing)}"
        )


def _validate_explicit_pool_metadata(pools: list[Dict[str, Any]]) -> None:
    missing = [
        pool.get("species", f"pool_{index}")
        for index, pool in enumerate(pools)
        if not pool.get("bulk_field")
    ]
    if missing:
        raise ValueError(
            "bundle requires explicit pool metadata but "
            f"{len(missing)} pool(s) are incomplete: {', '.join(missing)}"
        )


def _validate_explicit_transcription_unit_semantics(
    transcription_units: list[Dict[str, Any]],
) -> None:
    missing = [
        unit["name"]
        for unit in transcription_units
        if not unit.get("asset_class")
        or not unit.get("complex_family")
        or not unit.get("subsystem_targets")
    ]
    if missing:
        raise ValueError(
            "bundle requires explicit transcription unit semantics but "
            f"{len(missing)} unit(s) are incomplete: {', '.join(missing)}"
        )


def _with_compiled_chromosome_domains(spec: Dict[str, Any]) -> Dict[str, Any]:
    compiled = dict(spec)
    compiled["chromosome_domains"] = _compile_chromosome_domains(compiled)
    return compiled


def _empty_genome_asset_package(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "organism": spec["organism"],
        "chromosome_length_bp": int(spec["chromosome_length_bp"]),
        "origin_bp": int(spec["origin_bp"]),
        "terminus_bp": int(spec["terminus_bp"]),
        "chromosome_domains": list(spec.get("chromosome_domains", [])),
        "operons": [],
        "operon_semantics": [],
        "rnas": [],
        "proteins": [],
        "protein_semantics": [],
        "complex_semantics": [],
        "complexes": [],
        "pools": list(spec.get("pools", [])),
    }


def _compile_chromosome_domains(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
    chromosome_length_bp = max(1, int(spec["chromosome_length_bp"]))
    genes = list(spec.get("genes", []))
    transcription_units = list(spec.get("transcription_units", []))
    operons = []
    for unit in transcription_units:
        promoter_bp, terminator_bp = _operon_bounds(genes, unit.get("genes", []))
        operons.append(
            {
                "name": unit["name"],
                "promoter_bp": promoter_bp,
                "terminator_bp": terminator_bp,
            }
        )
    for gene in genes:
        if any(gene["gene"] in unit.get("genes", []) for unit in transcription_units):
            continue
        operons.append(
            {
                "name": gene["gene"],
                "promoter_bp": min(int(gene["start_bp"]), int(gene["end_bp"])),
                "terminator_bp": max(int(gene["start_bp"]), int(gene["end_bp"])),
            }
        )

    existing_domains = list(spec.get("chromosome_domains", []))
    if existing_domains:
        domains = []
        for index, domain in enumerate(existing_domains):
            start_bp = min(
                chromosome_length_bp - 1, max(0, int(domain.get("start_bp", 0)))
            )
            end_bp = min(
                chromosome_length_bp - 1,
                max(start_bp, int(domain.get("end_bp", start_bp))),
            )
            center_fraction = float(domain.get("axial_center_fraction") or 0.0)
            if center_fraction <= 0.0:
                center_fraction = ((_midpoint_bp(start_bp, end_bp) + 0.5) / chromosome_length_bp)
            domains.append(
                {
                    "id": domain.get("id") or f"chromosome_domain_{index}",
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "axial_center_fraction": max(0.02, min(0.98, center_fraction)),
                    "axial_spread_fraction": max(
                        0.05,
                        min(0.28, float(domain.get("axial_spread_fraction", 0.16))),
                    ),
                    "genes": list(domain.get("genes", [])),
                    "transcription_units": list(domain.get("transcription_units", [])),
                    "operons": list(domain.get("operons", [])),
                }
            )
    else:
        domains = []
        for index in range(4):
            start_bp = int(index * chromosome_length_bp / 4)
            end_bp = max(start_bp, int((index + 1) * chromosome_length_bp / 4) - 1)
            domains.append(
                {
                    "id": f"chromosome_domain_{index}",
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "axial_center_fraction": max(
                        0.02,
                        min(0.98, (_midpoint_bp(start_bp, end_bp) + 0.5) / chromosome_length_bp),
                    ),
                    "axial_spread_fraction": max(
                        0.08,
                        min(0.24, ((end_bp - start_bp + 1) / chromosome_length_bp) * 0.75),
                    ),
                    "genes": [],
                    "transcription_units": [],
                    "operons": [],
                }
            )

    domains.sort(key=lambda domain: (int(domain["start_bp"]), int(domain["end_bp"]), domain["id"]))
    for domain in domains:
        start_bp = int(domain["start_bp"])
        end_bp = int(domain["end_bp"])
        for gene in genes:
            if _interval_contains_bp(start_bp, end_bp, _gene_midpoint_bp(gene)):
                if gene["gene"] not in domain["genes"]:
                    domain["genes"].append(gene["gene"])
        for unit in transcription_units:
            if _interval_contains_bp(
                start_bp,
                end_bp,
                _midpoint_bp(*_operon_bounds(genes, unit.get("genes", []))),
            ):
                if unit["name"] not in domain["transcription_units"]:
                    domain["transcription_units"].append(unit["name"])
        for operon in operons:
            if _interval_contains_bp(
                start_bp,
                end_bp,
                _midpoint_bp(int(operon["promoter_bp"]), int(operon["terminator_bp"])),
            ):
                if operon["name"] not in domain["operons"]:
                    domain["operons"].append(operon["name"])
        domain["genes"].sort()
        domain["transcription_units"].sort()
        domain["operons"].sort()
    return domains


def _gene_midpoint_bp(gene: Dict[str, Any]) -> int:
    return _midpoint_bp(int(gene["start_bp"]), int(gene["end_bp"]))


def _midpoint_bp(start_bp: int, end_bp: int) -> int:
    return (int(start_bp) + int(end_bp)) // 2


def _interval_contains_bp(start_bp: int, end_bp: int, position_bp: int) -> bool:
    left = min(int(start_bp), int(end_bp))
    right = max(int(start_bp), int(end_bp))
    position_bp = int(position_bp)
    return left <= position_bp <= right


def _read_fasta(path: Path) -> Dict[str, Any]:
    header = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].strip()
            continue
        sequence_parts.append(line)
    sequence = "".join(sequence_parts).upper()
    if header is None or not sequence:
        raise ValueError(f"invalid FASTA file: {path}")
    return {"id": header, "sequence": sequence}


def _read_gff_features(path: Path) -> list[Dict[str, Any]]:
    genes: list[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            raise ValueError(f"invalid GFF3 row in {path}: {line}")
        _seqid, _source, feature_type, start, end, _score, strand, _phase, attrs = parts
        if feature_type.lower() not in {"gene", "cds"}:
            continue
        attr_map = _parse_gff_attributes(attrs)
        gene_name = attr_map.get("gene") or attr_map.get("Name") or attr_map.get("ID")
        if not gene_name:
            raise ValueError(f"missing gene identifier in {path}: {line}")
        genes.append(
            {
                "gene": gene_name,
                "start_bp": int(start),
                "end_bp": int(end),
                "strand": -1 if strand == "-" else 1,
            }
        )
    return genes


def _parse_gff_attributes(raw_attributes: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for entry in raw_attributes.split(";"):
        if not entry:
            continue
        if "=" in entry:
            key, value = entry.split("=", 1)
        elif " " in entry:
            key, value = entry.split(" ", 1)
        else:
            key, value = entry, ""
        attributes[key.strip()] = value.strip()
    return attributes


def _load_optional_json(
    manifest_path: Path,
    manifest: Dict[str, Any],
    key: str,
    source_hashes: Dict[str, str],
) -> Any:
    if key not in manifest:
        return None
    path = _resolve_manifest_path(manifest_path, manifest[key])
    if not path.exists():
        raise ValueError(f"bundle manifest references missing {key}: {path}")
    source_hashes[key] = _sha256_path(path)
    return _load_json(path)


def _resolve_manifest_path(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
