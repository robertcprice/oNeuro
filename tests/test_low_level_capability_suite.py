"""Smoke tests for the low-level capability suite."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.low_level_capability_suite import SuiteConfig, run_suite


def test_run_suite_smoke_writes_json():
    """The consolidated suite should emit one JSON artifact with all ladder rungs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload = run_suite(
            SuiteConfig(
                drug_n_seeds=2,
                conditioning_n_seeds=1,
                mechanism_replicates=1,
                discrimination_n_seeds=1,
                action_bias_n_seeds=1,
                working_memory_n_seeds=2,
                pattern_completion_n_seeds=1,
            ),
            output_dir=tmp_dir,
        )

        result_path = Path(payload["result_path"])
        assert result_path.exists()
        assert payload["experiment"] == "low_level_capability_suite"
        assert len(payload["benchmarks"]) == 8
        assert "positive" in payload["status_counts"] or "ambiguous" in payload["status_counts"]

        saved = json.loads(result_path.read_text())
        assert saved["experiment"] == "low_level_capability_suite"
        assert len(saved["benchmarks"]) == 8
