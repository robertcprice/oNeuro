"""Tests for the cue-conditioned corticostriatal action-bias benchmark."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.corticostriatal_action_bias_benchmark import (
    BenchmarkConfig,
    run_condition,
    run_experiment,
)


def test_run_condition_smoke_and_feedback_hooks():
    """Single-condition smoke run should record all benchmark phases."""
    config = BenchmarkConfig(
        warmup_steps=10,
        baseline_trials=2,
        training_trials=4,
        evaluation_trials=2,
        cue_steps=8,
        response_steps=6,
        feedback_steps=2,
        inter_trial_steps=2,
        cue_intensity=45.0,
        connection_probability=0.5,
        reward_amount=1.5,
        learning_rate=0.2,
    )

    record = run_condition(
        pair_index=0,
        condition="no_dopamine",
        config=config,
        master_seed=2026,
    )

    assert record.condition == "no_dopamine"
    assert record.pre_training_summary["n_trials"] == 2.0
    assert record.training_summary["n_trials"] == 4.0
    assert record.post_training_summary["n_trials"] == 2.0
    assert len(record.trial_records) == 8
    assert record.topology_summary["created_green_to_d1"] > 0
    assert record.topology_summary["created_green_to_d2"] > 0
    assert record.topology_summary["created_red_to_d1"] > 0
    assert record.topology_summary["created_red_to_d2"] > 0
    assert all(
        trial["applied_feedback"] == 0.0
        for trial in record.trial_records
        if trial["phase"] == "training"
    )


def test_run_experiment_smoke_writes_json():
    """Experiment runner should save JSON with summary and paired contrasts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload = run_experiment(
            conditions=["full_learning", "reward_shuffle"],
            n_seeds=1,
            config=BenchmarkConfig(
                warmup_steps=10,
                baseline_trials=2,
                training_trials=4,
                evaluation_trials=2,
                cue_steps=8,
                response_steps=6,
                feedback_steps=2,
                inter_trial_steps=2,
                cue_intensity=45.0,
                connection_probability=0.5,
                reward_amount=1.5,
                learning_rate=0.2,
            ),
            output_dir=tmp_dir,
            workers=1,
            master_seed=2026,
        )

        result_path = Path(payload["result_path"])
        assert result_path.exists()
        assert "full_learning" in payload["summary"]
        assert "reward_shuffle" in payload["summary"]
        assert "reward_shuffle" in payload["paired_differences"]

        saved = json.loads(result_path.read_text())
        assert saved["experiment"] == "corticostriatal_action_bias_benchmark"
        assert len(saved["results"]) == 2
