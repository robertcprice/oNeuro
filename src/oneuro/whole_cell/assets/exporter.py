"""Export and migration helpers for whole-cell organism bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .compiler import (
    CompiledOrganismBundle,
)
from .derived_assets import (
    _compile_genome_asset_package,
    _default_subsystem_targets_for_asset_class,
    _infer_asset_class,
    _infer_complex_family,
)
from .source_normalization import (
    _compile_chromosome_domains,
    _merge_transcription_unit_semantics,
)


def write_compiled_bundle(
    bundle: CompiledOrganismBundle,
    output_dir: Path | str,
) -> Dict[str, str]:
    """Write compiled organism spec, derived assets, and summary JSON files."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = bundle.organism.lower().replace(" ", "_").replace("-", "_")
    organism_path = out_dir / f"{slug}_organism.json"
    assets_path = out_dir / f"{slug}_assets.json"
    summary_path = out_dir / f"{slug}_summary.json"
    organism_path.write_text(json.dumps(bundle.organism_spec, indent=2), encoding="ascii")
    assets_path.write_text(
        json.dumps(bundle.genome_asset_package, indent=2), encoding="ascii"
    )
    summary_payload = {
        "manifest_path": bundle.manifest_path,
        "organism": bundle.organism,
        "source_hashes": bundle.source_hashes,
        "summary": bundle.summary(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="ascii")
    return {
        "organism_spec": str(organism_path),
        "genome_assets": str(assets_path),
        "summary": str(summary_path),
    }


def write_structured_bundle_sources(
    organism_spec: Dict[str, Any],
    output_dir: Path | str,
    *,
    source_dataset: str | None = None,
    require_structured_bundle: bool = True,
    require_explicit_organism_sources: bool = True,
    require_explicit_gene_semantics: bool = True,
    require_explicit_transcription_unit_semantics: bool = True,
    require_explicit_asset_entities: bool = True,
    require_explicit_asset_semantics: bool = True,
    require_explicit_program_defaults: bool = True,
) -> Dict[str, str]:
    """Write a structured source bundle from a compiled organism spec."""

    organism_spec = _normalize_organism_spec_semantics(organism_spec)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_package = _compile_genome_asset_package(organism_spec)
    metadata = {
        "organism": organism_spec["organism"],
        "chromosome_length_bp": int(organism_spec["chromosome_length_bp"]),
        "origin_bp": int(organism_spec["origin_bp"]),
        "terminus_bp": int(organism_spec["terminus_bp"]),
        "geometry": organism_spec["geometry"],
        "composition": organism_spec["composition"],
    }
    gene_features = [
        {
            "gene": gene["gene"],
            "start_bp": int(gene["start_bp"]),
            "end_bp": int(gene["end_bp"]),
            "strand": int(gene.get("strand", 1)),
        }
        for gene in organism_spec.get("genes", [])
    ]
    gene_products = [
        {
            "gene": gene["gene"],
            "essential": bool(gene.get("essential", False)),
            "basal_expression": float(gene.get("basal_expression", 1.0)),
            "translation_cost": float(gene.get("translation_cost", 1.0)),
            "nucleotide_cost": float(gene.get("nucleotide_cost", 1.0)),
            "process_weights": gene.get("process_weights", {}),
        }
        for gene in organism_spec.get("genes", [])
    ]
    gene_semantics = [
        {
            "gene": gene["gene"],
            "subsystem_targets": list(gene.get("subsystem_targets", [])),
            "asset_class": gene.get("asset_class"),
            "complex_family": gene.get("complex_family"),
        }
        for gene in organism_spec.get("genes", [])
    ]
    transcription_units = [
        {
            "name": unit["name"],
            "genes": list(unit.get("genes", [])),
            "basal_activity": float(unit.get("basal_activity", 1.0)),
            "process_weights": unit.get("process_weights", {}),
        }
        for unit in organism_spec.get("transcription_units", [])
    ]
    transcription_unit_semantics = [
        {
            "name": unit["name"],
            "subsystem_targets": list(unit.get("subsystem_targets", [])),
            "asset_class": unit.get("asset_class"),
            "complex_family": unit.get("complex_family"),
        }
        for unit in organism_spec.get("transcription_units", [])
    ]
    chromosome_domains = list(organism_spec.get("chromosome_domains", []))
    pools = list(organism_spec.get("pools", []))
    operons = list(asset_package.get("operons", []))
    rnas = list(asset_package.get("rnas", []))
    proteins = list(asset_package.get("proteins", []))
    complexes = list(asset_package.get("complexes", []))
    operon_semantics = list(asset_package.get("operon_semantics", []))
    protein_semantics = list(asset_package.get("protein_semantics", []))
    complex_semantics = list(asset_package.get("complex_semantics", []))
    program_defaults = _default_program_defaults(organism_spec)
    manifest = {
        "organism": organism_spec["organism"],
        "source_dataset": source_dataset
        or f"{organism_spec['organism'].lower().replace(' ', '_').replace('-', '_')}_structured",
        "require_structured_bundle": require_structured_bundle,
        "require_explicit_organism_sources": require_explicit_organism_sources,
        "require_explicit_gene_semantics": require_explicit_gene_semantics,
        "require_explicit_transcription_unit_semantics": require_explicit_transcription_unit_semantics,
        "require_explicit_asset_entities": require_explicit_asset_entities,
        "require_explicit_asset_semantics": require_explicit_asset_semantics,
        "require_explicit_program_defaults": require_explicit_program_defaults,
        "metadata_json": "metadata.json",
        "gene_features_json": "gene_features.json",
        "gene_products_json": "gene_products.json",
        "gene_semantics_json": "gene_semantics.json",
        "transcription_units_json": "transcription_units.json",
        "transcription_unit_semantics_json": "transcription_unit_semantics.json",
        "chromosome_domains_json": "chromosome_domains.json",
        "pools_json": "pools.json",
        "operons_json": "operons.json",
        "rnas_json": "rnas.json",
        "proteins_json": "proteins.json",
        "complexes_json": "complexes.json",
        "operon_semantics_json": "operon_semantics.json",
        "protein_semantics_json": "protein_semantics.json",
        "complex_semantics_json": "complex_semantics.json",
        "program_defaults_json": "program_defaults.json",
    }

    written = {
        "manifest": out_dir / "manifest.json",
        "metadata": out_dir / "metadata.json",
        "gene_features": out_dir / "gene_features.json",
        "gene_products": out_dir / "gene_products.json",
        "gene_semantics": out_dir / "gene_semantics.json",
        "transcription_units": out_dir / "transcription_units.json",
        "transcription_unit_semantics": out_dir / "transcription_unit_semantics.json",
        "chromosome_domains": out_dir / "chromosome_domains.json",
        "pools": out_dir / "pools.json",
        "operons": out_dir / "operons.json",
        "rnas": out_dir / "rnas.json",
        "proteins": out_dir / "proteins.json",
        "complexes": out_dir / "complexes.json",
        "operon_semantics": out_dir / "operon_semantics.json",
        "protein_semantics": out_dir / "protein_semantics.json",
        "complex_semantics": out_dir / "complex_semantics.json",
        "program_defaults": out_dir / "program_defaults.json",
    }

    payloads = {
        "manifest": manifest,
        "metadata": metadata,
        "gene_features": gene_features,
        "gene_products": gene_products,
        "gene_semantics": gene_semantics,
        "transcription_units": transcription_units,
        "transcription_unit_semantics": transcription_unit_semantics,
        "chromosome_domains": chromosome_domains,
        "pools": pools,
        "operons": operons,
        "rnas": rnas,
        "proteins": proteins,
        "complexes": complexes,
        "operon_semantics": operon_semantics,
        "protein_semantics": protein_semantics,
        "complex_semantics": complex_semantics,
        "program_defaults": program_defaults,
    }
    for key, path in written.items():
        path.write_text(json.dumps(payloads[key], indent=2), encoding="ascii")
    return {key: str(path) for key, path in written.items()}


def _normalize_organism_spec_semantics(spec: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(spec)
    genes = [dict(gene) for gene in normalized.get("genes", [])]
    for gene in genes:
        subsystem_targets = list(gene.get("subsystem_targets", []))
        asset_class = gene.get("asset_class") or _infer_asset_class(
            gene.get("process_weights", {}),
            subsystem_targets,
            gene["gene"],
        )
        if not subsystem_targets:
            subsystem_targets = _default_subsystem_targets_for_asset_class(asset_class)
        gene["subsystem_targets"] = subsystem_targets
        gene["asset_class"] = asset_class
        gene["complex_family"] = gene.get("complex_family") or _infer_complex_family(
            asset_class,
            subsystem_targets,
            gene["gene"],
        )
    gene_map = {gene["gene"]: gene for gene in genes}

    transcription_units = [
        _merge_transcription_unit_semantics(unit, {})
        for unit in normalized.get("transcription_units", [])
    ]
    for unit in transcription_units:
        subsystem_targets = list(unit.get("subsystem_targets", []))
        for gene_name in unit.get("genes", []):
            gene = gene_map.get(gene_name)
            if gene is None:
                continue
            for target in gene.get("subsystem_targets", []):
                if target not in subsystem_targets:
                    subsystem_targets.append(target)
        unit["subsystem_targets"] = subsystem_targets
        asset_class = unit.get("asset_class") or _infer_asset_class(
            unit.get("process_weights", {}),
            subsystem_targets,
            unit["name"],
        )
        if not subsystem_targets:
            subsystem_targets = _default_subsystem_targets_for_asset_class(asset_class)
            unit["subsystem_targets"] = subsystem_targets
        unit["asset_class"] = asset_class
        unit["complex_family"] = unit.get("complex_family") or _infer_complex_family(
            asset_class,
            subsystem_targets,
            unit["name"],
        )

    normalized["genes"] = genes
    normalized["transcription_units"] = transcription_units
    normalized["chromosome_domains"] = _compile_chromosome_domains(normalized)
    return normalized


def _default_program_defaults(organism_spec: Dict[str, Any]) -> Dict[str, Any]:
    slug = organism_spec["organism"].lower().replace(" ", "_").replace("-", "_")
    geometry = organism_spec["geometry"]
    radius_nm = max(50.0, float(geometry["radius_nm"]))
    chromosome_radius_fraction = max(0.1, float(geometry["chromosome_radius_fraction"]))
    chromosome_separation_nm = round(
        max(10.0, radius_nm * chromosome_radius_fraction), 6
    )
    return {
        "program_name": f"{slug}_structured_bundle_native",
        "config": {
            "x_dim": 24,
            "y_dim": 24,
            "z_dim": 12,
            "voxel_size_nm": 20.0,
            "dt_ms": 0.25,
            "cme_interval": 4,
            "ode_interval": 1,
            "bd_interval": 2,
            "geometry_interval": 4,
            "use_gpu": True,
        },
        "initial_lattice": {
            "atp": _pool_concentration_for_field(organism_spec, "a_t_p", 1.2),
            "amino_acids": _pool_concentration_for_field(organism_spec, "amino_acids", 0.95),
            "nucleotides": _pool_concentration_for_field(organism_spec, "nucleotides", 0.80),
            "membrane_precursors": _pool_concentration_for_field(
                organism_spec, "membrane_precursors", 0.35
            ),
        },
        "initial_state": {
            "adp_mm": _pool_concentration_for_field(organism_spec, "adp", 0.2),
            "glucose_mm": _pool_concentration_for_field(organism_spec, "glucose", 1.0),
            "oxygen_mm": _pool_concentration_for_field(organism_spec, "oxygen", 0.85),
            "genome_bp": int(organism_spec["chromosome_length_bp"]),
            "replicated_bp": 0,
            "chromosome_separation_nm": chromosome_separation_nm,
            "radius_nm": radius_nm,
            "division_progress": 0.0,
            "metabolic_load": 1.0,
        },
        "quantum_profile": {
            "oxphos_efficiency": 1.0,
            "translation_efficiency": 1.0,
            "nucleotide_polymerization_efficiency": 1.0,
            "membrane_synthesis_efficiency": 1.0,
            "chromosome_segregation_efficiency": 1.0,
        },
    }


def _pool_concentration_for_field(
    organism_spec: Dict[str, Any], bulk_field: str, fallback: float
) -> float:
    for pool in organism_spec.get("pools", []):
        if pool.get("bulk_field") == bulk_field:
            return float(pool.get("concentration_mm", fallback))
    return float(fallback)


__all__ = [
    "CompiledOrganismBundle",
    "write_compiled_bundle",
    "write_structured_bundle_sources",
]
