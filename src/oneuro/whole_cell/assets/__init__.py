"""Compiler-driven organism asset ingestion for the whole-cell runtime."""

from .compiler import (
    CompiledOrganismBundle,
    available_bundles,
    compile_bundle_manifest,
    compile_legacy_bundle_manifest,
    compile_legacy_named_bundle,
    compile_named_bundle,
)
from .exporter import write_compiled_bundle, write_structured_bundle_sources

__all__ = [
    "CompiledOrganismBundle",
    "available_bundles",
    "compile_bundle_manifest",
    "compile_legacy_bundle_manifest",
    "compile_legacy_named_bundle",
    "compile_named_bundle",
    "write_structured_bundle_sources",
    "write_compiled_bundle",
]
