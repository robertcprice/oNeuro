"""Tests for the minimal stimulus discrimination benchmark."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.stimulus_discrimination_benchmark import (
    BenchmarkConfig,
    run_condition,
    run_experiment,
)


def test_run_condition_smoke_and_control_hooks():
    """Single-condition smoke run should record all phases and control behavior."""
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
        teaching_current=12.0,
        teaching_steps=4,
        connection_probability=0.5,
        reward_amount=1.0,
        learning_rate=0.1,
        task_stdp_factor=0.2,
    )

    record = run_condition(
        pair_index=0,
        condition="no_learning",
        config=config,
        master_seed=2026,
    )

    assert record.condition == "no_learning"
    assert record.pre_training_summary["n_trials"] == 2.0
    assert record.training_summary["n_trials"] == 4.0
    assert record.post_training_summary["n_trials"] == 2.0
    assert len(record.trial_records) == 8
    assert record.topology_summary["created_cue_a_to_output_a"] > 0
    assert record.topology_summary["created_cue_a_to_output_b"] > 0
    assert record.topology_summary["created_cue_b_to_output_a"] > 0
    assert record.topology_summary["created_cue_b_to_output_b"] > 0
    assert all(
        trial["applied_reward"] == 0.0
        for trial in record.trial_records
        if trial["phase"] == "training"
    )


def test_run_experiment_smoke_writes_json():
    """Experiment runner should save JSON with summary and paired contrasts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload = run_experiment(
            conditions=["full_learning", "label_shuffle"],
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
                teaching_current=12.0,
                teaching_steps=4,
                connection_probability=0.5,
                reward_amount=1.0,
                learning_rate=0.1,
                task_stdp_factor=0.2,
            ),
            output_dir=tmp_dir,
            workers=1,
            master_seed=2026,
        )

        result_path = Path(payload["result_path"])
        assert result_path.exists()
        assert "full_learning" in payload["summary"]
        assert "label_shuffle" in payload["summary"]
        assert "label_shuffle" in payload["paired_differences"]

        saved = json.loads(result_path.read_text())
        assert saved["experiment"] == "stimulus_discrimination_benchmark"
        assert len(saved["results"]) == 2
