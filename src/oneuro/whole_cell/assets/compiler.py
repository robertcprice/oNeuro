"""Manifest-driven compiler for whole-cell organism source bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

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


def compile_bundle_manifest(manifest_path: Path | str) -> CompiledOrganismBundle:
    """Compile an organism bundle manifest into runtime-ready JSON payloads."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(path)
    source_hashes: Dict[str, str] = {"manifest.json": _sha256_path(path)}

    if "organism_spec_json" in manifest:
        spec_path = _resolve_manifest_path(path, manifest["organism_spec_json"])
        source_hashes["organism_spec_json"] = _sha256_path(spec_path)
        organism_spec = _load_json(spec_path)
    else:
        organism_spec = _compile_structured_bundle(path, manifest, source_hashes)

    organism_spec = _with_compiled_chromosome_domains(organism_spec)
    asset_package = _compile_genome_asset_package(organism_spec)
    return CompiledOrganismBundle(
        manifest_path=str(path),
        organism=organism_spec["organism"],
        organism_spec=organism_spec,
        genome_asset_package=asset_package,
        source_hashes=source_hashes,
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
        "chromosome_domains": [],
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


def _compile_genome_asset_package(spec: Dict[str, Any]) -> Dict[str, Any]:
    genes = list(spec.get("genes", []))
    transcription_units = list(spec.get("transcription_units", []))
    gene_to_operon: Dict[str, str] = {}
    operons = []

    for unit in transcription_units:
        genes_in_unit = list(unit.get("genes", []))
        for gene_name in genes_in_unit:
            gene_to_operon[gene_name] = unit["name"]
        promoter_bp, terminator_bp = _operon_bounds(genes, genes_in_unit)
        subsystem_targets: list[str] = list(unit.get("subsystem_targets", []))
        for gene_name in genes_in_unit:
            gene = next((entry for entry in genes if entry["gene"] == gene_name), None)
            if gene is None:
                continue
            for target in gene.get("subsystem_targets", []):
                if target not in subsystem_targets:
                    subsystem_targets.append(target)
        asset_class = unit.get("asset_class") or _infer_asset_class(
            unit.get("process_weights", {}),
            subsystem_targets,
            unit["name"],
        )
        complex_family = unit.get("complex_family") or _infer_complex_family(
            asset_class,
            subsystem_targets,
            unit["name"],
        )
        operons.append(
            {
                "name": unit["name"],
                "genes": genes_in_unit,
                "promoter_bp": promoter_bp,
                "terminator_bp": terminator_bp,
                "basal_activity": float(unit.get("basal_activity", 1.0)),
                "polycistronic": len(genes_in_unit) > 1,
                "process_weights": _clamp_process_weights(unit.get("process_weights", {})),
                "subsystem_targets": subsystem_targets,
                "asset_class": asset_class,
                "complex_family": complex_family,
            }
        )

    for gene in genes:
        if gene["gene"] in gene_to_operon:
            continue
        gene_to_operon[gene["gene"]] = gene["gene"]
        asset_class = gene.get("asset_class") or _infer_asset_class(
            gene.get("process_weights", {}),
            gene.get("subsystem_targets", []),
            gene["gene"],
        )
        operons.append(
            {
                "name": gene["gene"],
                "genes": [gene["gene"]],
                "promoter_bp": min(int(gene["start_bp"]), int(gene["end_bp"])),
                "terminator_bp": max(int(gene["start_bp"]), int(gene["end_bp"])),
                "basal_activity": float(gene.get("basal_expression", 1.0)),
                "polycistronic": False,
                "process_weights": _clamp_process_weights(gene.get("process_weights", {})),
                "subsystem_targets": list(gene.get("subsystem_targets", [])),
                "asset_class": asset_class,
                "complex_family": gene.get("complex_family")
                or _infer_complex_family(
                    asset_class,
                    gene.get("subsystem_targets", []),
                    gene["gene"],
                ),
            }
        )

    operon_semantics = [
        {
            "name": operon["name"],
            "asset_class": operon["asset_class"],
            "complex_family": operon["complex_family"],
            "subsystem_targets": list(operon.get("subsystem_targets", [])),
        }
        for operon in operons
        if operon.get("asset_class") and operon.get("complex_family")
    ]

    rnas = []
    proteins = []
    for gene in genes:
        length_nt = max(1, _gene_length_bp(gene))
        operon_name = gene_to_operon.get(gene["gene"], gene["gene"])
        asset_class = gene.get("asset_class") or _infer_asset_class(
            gene.get("process_weights", {}),
            gene.get("subsystem_targets", []),
            gene["gene"],
        )
        rna_id = f"{gene['gene']}_rna"
        rnas.append(
            {
                "id": rna_id,
                "gene": gene["gene"],
                "operon": operon_name,
                "length_nt": length_nt,
                "basal_abundance": max(
                    0.5, min(256.0, 4.0 + 6.0 * max(0.05, float(gene["basal_expression"])))
                ),
                "asset_class": asset_class,
                "process_weights": _clamp_process_weights(gene.get("process_weights", {})),
            }
        )
        proteins.append(
            {
                "id": f"{gene['gene']}_protein",
                "gene": gene["gene"],
                "operon": operon_name,
                "rna_id": rna_id,
                "aa_length": max(1, length_nt // 3),
                "basal_abundance": max(
                    0.5, min(512.0, 8.0 + 10.0 * max(0.05, float(gene["basal_expression"])))
                ),
                "translation_cost": float(gene["translation_cost"]),
                "nucleotide_cost": float(gene["nucleotide_cost"]),
                "asset_class": asset_class,
                "process_weights": _clamp_process_weights(gene.get("process_weights", {})),
                "subsystem_targets": list(gene.get("subsystem_targets", [])),
            }
        )

    protein_semantics = [
        {
            "id": protein["id"],
            "asset_class": protein["asset_class"],
            "subsystem_targets": list(protein.get("subsystem_targets", [])),
        }
        for protein in proteins
        if protein.get("asset_class")
    ]

    complexes = []
    for operon in operons:
        components = []
        process_weights = _clamp_process_weights(operon.get("process_weights", {}))
        subsystem_targets: list[str] = list(operon.get("subsystem_targets", []))
        for gene_name in operon["genes"]:
            components.append({"protein_id": f"{gene_name}_protein", "stoichiometry": 1})
            gene = next((entry for entry in genes if entry["gene"] == gene_name), None)
            if gene is None:
                continue
            process_weights = _add_weighted_process_weights(
                process_weights,
                gene.get("process_weights", {}),
                0.35,
            )
            for target in gene.get("subsystem_targets", []):
                if target not in subsystem_targets:
                    subsystem_targets.append(target)
        asset_class = operon.get("asset_class") or _infer_asset_class(
            process_weights, subsystem_targets, operon["name"]
        )
        complex_family = operon.get("complex_family") or _infer_complex_family(
            asset_class, subsystem_targets, operon["name"]
        )
        complexes.append(
            {
                "id": f"{operon['name']}_complex",
                "name": f"{operon['name']} complex",
                "operon": operon["name"],
                "components": components,
                "basal_abundance": max(
                    0.5,
                    min(
                        256.0,
                        3.0
                        + 7.0
                        * max(0.05, float(operon["basal_activity"]))
                        * max(1.0, len(operon["genes"]) ** 0.5),
                    ),
                ),
                "asset_class": asset_class,
                "family": complex_family,
                "process_weights": process_weights,
                "subsystem_targets": subsystem_targets,
                "membrane_inserted": complex_family
                in {"atp_synthase", "transporter", "membrane_enzyme", "divisome"},
                "chromosome_coupled": complex_family in {"replisome", "rna_polymerase"},
                "division_coupled": complex_family == "divisome",
            }
        )

    complex_semantics = [
        {
            "id": complex["id"],
            "asset_class": complex["asset_class"],
            "family": complex["family"],
            "subsystem_targets": list(complex.get("subsystem_targets", [])),
            "membrane_inserted": bool(complex.get("membrane_inserted", False)),
            "chromosome_coupled": bool(complex.get("chromosome_coupled", False)),
            "division_coupled": bool(complex.get("division_coupled", False)),
        }
        for complex in complexes
        if complex.get("asset_class") and complex.get("family")
    ]

    return {
        "organism": spec["organism"],
        "chromosome_length_bp": int(spec["chromosome_length_bp"]),
        "origin_bp": int(spec["origin_bp"]),
        "terminus_bp": int(spec["terminus_bp"]),
        "chromosome_domains": spec.get("chromosome_domains", []),
        "operons": operons,
        "operon_semantics": operon_semantics,
        "rnas": rnas,
        "proteins": proteins,
        "protein_semantics": protein_semantics,
        "complex_semantics": complex_semantics,
        "complexes": complexes,
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


def _gene_length_bp(gene: Dict[str, Any]) -> int:
    start = int(gene["start_bp"])
    end = int(gene["end_bp"])
    return max(0, end - start + 1) if end >= start else 0


def _gene_midpoint_bp(gene: Dict[str, Any]) -> int:
    return _midpoint_bp(int(gene["start_bp"]), int(gene["end_bp"]))


def _midpoint_bp(start_bp: int, end_bp: int) -> int:
    return (int(start_bp) + int(end_bp)) // 2


def _interval_contains_bp(start_bp: int, end_bp: int, position_bp: int) -> bool:
    left = min(int(start_bp), int(end_bp))
    right = max(int(start_bp), int(end_bp))
    position_bp = int(position_bp)
    return left <= position_bp <= right


def _operon_bounds(
    genes: Iterable[Dict[str, Any]],
    genes_in_unit: Iterable[str],
) -> tuple[int, int]:
    promoter_bp = None
    terminator_bp = None
    gene_map = {gene["gene"]: gene for gene in genes}
    for gene_name in genes_in_unit:
        gene = gene_map.get(gene_name)
        if gene is None:
            continue
        start = min(int(gene["start_bp"]), int(gene["end_bp"]))
        end = max(int(gene["start_bp"]), int(gene["end_bp"]))
        promoter_bp = start if promoter_bp is None else min(promoter_bp, start)
        terminator_bp = end if terminator_bp is None else max(terminator_bp, end)
    return promoter_bp or 0, terminator_bp or 0


def _clamp_process_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    return {
        "energy": max(0.0, float(weights.get("energy", 0.0))),
        "transcription": max(0.0, float(weights.get("transcription", 0.0))),
        "translation": max(0.0, float(weights.get("translation", 0.0))),
        "replication": max(0.0, float(weights.get("replication", 0.0))),
        "segregation": max(0.0, float(weights.get("segregation", 0.0))),
        "membrane": max(0.0, float(weights.get("membrane", 0.0))),
        "constriction": max(0.0, float(weights.get("constriction", 0.0))),
    }


def _add_weighted_process_weights(
    base: Dict[str, float],
    other: Dict[str, Any],
    scale: float,
) -> Dict[str, float]:
    clamped = _clamp_process_weights(other)
    return {
        key: float(base.get(key, 0.0)) + scale * float(clamped.get(key, 0.0))
        for key in base
    }


def _infer_asset_class(
    weights: Dict[str, Any],
    subsystem_targets: Iterable[str],
    name: str,
) -> str:
    targets = set(subsystem_targets)
    if "AtpSynthaseMembraneBand" in targets:
        return "energy"
    if "RibosomePolysomeCluster" in targets:
        return "translation"
    if "ReplisomeTrack" in targets:
        return "replication"
    if "FtsZSeptumRing" in targets:
        return "constriction"
    lowered = name.lower()
    if "chaperone" in lowered or "quality_control" in lowered:
        return "quality_control"
    if "transport" in lowered or "homeostasis" in lowered:
        return "homeostasis"
    clamped = _clamp_process_weights(weights)
    ranked = [
        ("energy", clamped["energy"]),
        ("translation", clamped["translation"]),
        ("replication", clamped["replication"]),
        ("segregation", clamped["segregation"]),
        ("membrane", clamped["membrane"]),
        ("constriction", clamped["constriction"]),
        ("homeostasis", clamped["transcription"]),
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0][0] if ranked and ranked[0][1] > 0.0 else "generic"


def _infer_complex_family(
    asset_class: str,
    subsystem_targets: Iterable[str],
    operon_name: str,
) -> str:
    targets = set(subsystem_targets)
    lowered = operon_name.lower()
    if "RibosomePolysomeCluster" in targets or "ribosome" in lowered:
        return "ribosome"
    if (
        "ReplisomeTrack" in targets
        or "replisome" in lowered
        or "replication" in lowered
        or "dna" in lowered
    ):
        return "replisome"
    if (
        "AtpSynthaseMembraneBand" in targets
        or "atp_synthase" in lowered
        or "respir" in lowered
    ):
        return "atp_synthase"
    if (
        "FtsZSeptumRing" in targets
        or "ftsz" in lowered
        or "division" in lowered
        or "sept" in lowered
        or "divisome" in lowered
    ):
        return "divisome"
    if "rnap" in lowered or "rna_polymerase" in lowered or "sigma" in lowered:
        return "rna_polymerase"
    if "chaperone" in lowered or "fold" in lowered or "client" in lowered:
        return "chaperone_client"
    if "transport" in lowered or "porin" in lowered or "pump" in lowered:
        return "transporter"
    if asset_class in {"membrane", "constriction"}:
        return "membrane_enzyme"
    return "generic"


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
