"""Explicit structured-source normalization helpers for whole-cell bundles."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .derived_assets import _operon_bounds


def _merge_gene_annotation(
    gene_feature: Dict[str, Any],
    gene_product: Dict[str, Any],
    gene_semantic: Dict[str, Any],
) -> Dict[str, Any]:
    return {
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


def _normalized_interval(
    start_bp: int, end_bp: int, chromosome_length_bp: int
) -> tuple[int, int]:
    right_bound = max(0, chromosome_length_bp - 1)
    left = min(right_bound, max(0, int(start_bp)))
    right = min(right_bound, max(0, int(end_bp)))
    return (left, right) if left <= right else (right, left)


def _implicit_chromosome_domain_seeds(spec: Dict[str, Any]) -> list[Dict[str, int]]:
    chromosome_length_bp = max(1, int(spec["chromosome_length_bp"]))
    genes = list(spec.get("genes", []))
    transcription_units = list(spec.get("transcription_units", []))
    operons = []
    for unit in transcription_units:
        promoter_bp, terminator_bp = _operon_bounds(genes, unit.get("genes", []))
        start_bp, end_bp = _normalized_interval(
            promoter_bp, terminator_bp, chromosome_length_bp
        )
        operons.append(
            {
                "name": unit["name"],
                "start_bp": start_bp,
                "end_bp": end_bp,
                "midpoint_bp": _midpoint_bp(start_bp, end_bp),
                "span_bp": end_bp - start_bp + 1,
            }
        )
    if operons:
        operons.sort(
            key=lambda operon: (
                operon["midpoint_bp"],
                operon["start_bp"],
                operon["end_bp"],
            )
        )
        return operons

    genes_only = []
    for gene in genes:
        start_bp, end_bp = _normalized_interval(
            int(gene["start_bp"]), int(gene["end_bp"]), chromosome_length_bp
        )
        genes_only.append(
            {
                "name": gene["gene"],
                "start_bp": start_bp,
                "end_bp": end_bp,
                "midpoint_bp": _midpoint_bp(start_bp, end_bp),
                "span_bp": end_bp - start_bp + 1,
            }
        )
    genes_only.sort(
        key=lambda gene: (gene["midpoint_bp"], gene["start_bp"], gene["end_bp"])
    )
    return genes_only


def _compile_implicit_chromosome_domains(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
    chromosome_length_bp = max(1, int(spec["chromosome_length_bp"]))
    seeds = _implicit_chromosome_domain_seeds(spec)
    if not seeds:
        return [
            {
                "id": "chromosome_domain_0",
                "start_bp": 0,
                "end_bp": chromosome_length_bp - 1,
                "axial_center_fraction": 0.5,
                "axial_spread_fraction": 0.24,
                "genes": [],
                "transcription_units": [],
                "operons": [],
            }
        ]

    split_points = []
    for previous, current in zip(seeds, seeds[1:]):
        gap_start = min(chromosome_length_bp - 1, previous["end_bp"] + 1)
        gap_end = current["start_bp"] - 1
        if gap_start > gap_end:
            continue
        gap_bp = gap_end - gap_start + 1
        local_feature_span = max(1, (previous["span_bp"] + current["span_bp"]) // 2)
        if gap_bp > local_feature_span:
            split_points.append(_midpoint_bp(gap_start, gap_end))

    domains = []
    domain_start = 0
    for split_bp in split_points:
        domain_end = min(chromosome_length_bp - 1, int(split_bp))
        if domain_end < domain_start:
            continue
        domains.append(
            {
                "id": f"chromosome_domain_{len(domains)}",
                "start_bp": domain_start,
                "end_bp": domain_end,
                "axial_center_fraction": max(
                    0.02,
                    min(
                        0.98,
                        (_midpoint_bp(domain_start, domain_end) + 0.5)
                        / chromosome_length_bp,
                    ),
                ),
                "axial_spread_fraction": max(
                    0.08,
                    min(
                        0.24,
                        ((domain_end - domain_start + 1) / chromosome_length_bp) * 0.75,
                    ),
                ),
                "genes": [],
                "transcription_units": [],
                "operons": [],
            }
        )
        domain_start = min(chromosome_length_bp - 1, domain_end + 1)
    if not domains or domain_start <= chromosome_length_bp - 1:
        domain_end = chromosome_length_bp - 1
        domains.append(
            {
                "id": f"chromosome_domain_{len(domains)}",
                "start_bp": domain_start,
                "end_bp": domain_end,
                "axial_center_fraction": max(
                    0.02,
                    min(
                        0.98,
                        (_midpoint_bp(domain_start, domain_end) + 0.5)
                        / chromosome_length_bp,
                    ),
                ),
                "axial_spread_fraction": max(
                    0.08,
                    min(
                        0.24,
                        ((domain_end - domain_start + 1) / chromosome_length_bp) * 0.75,
                    ),
                ),
                "genes": [],
                "transcription_units": [],
                "operons": [],
            }
        )
    return domains


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
    implicit_domains = not existing_domains
    if existing_domains:
        domains = []
        for index, domain in enumerate(existing_domains):
            start_bp, end_bp = _normalized_interval(
                int(domain.get("start_bp", 0)),
                int(domain.get("end_bp", domain.get("start_bp", 0))),
                chromosome_length_bp,
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
        domains = _compile_implicit_chromosome_domains(spec)

    domains.sort(key=lambda domain: (int(domain["start_bp"]), int(domain["end_bp"]), domain["id"]))
    for domain in domains:
        start_bp = int(domain["start_bp"])
        end_bp = int(domain["end_bp"])
        if implicit_domains:
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


def _with_compiled_chromosome_domains(spec: Dict[str, Any]) -> Dict[str, Any]:
    compiled = dict(spec)
    compiled["chromosome_domains"] = _compile_chromosome_domains(compiled)
    return compiled
