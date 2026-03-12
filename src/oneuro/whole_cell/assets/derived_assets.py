"""Legacy-derived genome asset construction helpers.

These helpers exist for compatibility and export/migration workflows.
The explicit structured-bundle compiler should only call into this module
when a manifest opts into derived asset generation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


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


def _derive_operon_semantics(asset_package: Dict[str, Any]) -> list[Dict[str, Any]]:
    semantics = []
    for operon in asset_package.get("operons", []):
        asset_class = operon.get("asset_class")
        complex_family = operon.get("complex_family")
        if not asset_class or not complex_family:
            continue
        semantics.append(
            {
                "name": operon["name"],
                "asset_class": asset_class,
                "complex_family": complex_family,
                "subsystem_targets": list(operon.get("subsystem_targets", [])),
            }
        )
    return sorted(semantics, key=lambda semantic: semantic["name"])


def _derive_protein_semantics(asset_package: Dict[str, Any]) -> list[Dict[str, Any]]:
    semantics = []
    for protein in asset_package.get("proteins", []):
        asset_class = protein.get("asset_class")
        if not asset_class:
            continue
        semantics.append(
            {
                "id": protein["id"],
                "asset_class": asset_class,
                "subsystem_targets": list(protein.get("subsystem_targets", [])),
            }
        )
    return sorted(semantics, key=lambda semantic: semantic["id"])


def _derive_complex_semantics(asset_package: Dict[str, Any]) -> list[Dict[str, Any]]:
    semantics = []
    for complex_spec in asset_package.get("complexes", []):
        asset_class = complex_spec.get("asset_class")
        family = complex_spec.get("family")
        if not asset_class or not family:
            continue
        semantics.append(
            {
                "id": complex_spec["id"],
                "asset_class": asset_class,
                "family": family,
                "subsystem_targets": list(complex_spec.get("subsystem_targets", [])),
                "membrane_inserted": bool(complex_spec.get("membrane_inserted", False)),
                "chromosome_coupled": bool(complex_spec.get("chromosome_coupled", False)),
                "division_coupled": bool(complex_spec.get("division_coupled", False)),
            }
        )
    return sorted(semantics, key=lambda semantic: semantic["id"])


def _gene_length_bp(gene: Dict[str, Any]) -> int:
    start = int(gene["start_bp"])
    end = int(gene["end_bp"])
    return max(0, end - start + 1) if end >= start else 0


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


def _default_subsystem_targets_for_asset_class(asset_class: str) -> list[str]:
    if asset_class in {"energy", "membrane", "homeostasis"}:
        return ["AtpSynthaseMembraneBand"]
    if asset_class in {"translation", "quality_control"}:
        return ["RibosomePolysomeCluster"]
    if asset_class in {"replication", "segregation"}:
        return ["ReplisomeTrack"]
    if asset_class == "constriction":
        return ["FtsZSeptumRing"]
    return []


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

