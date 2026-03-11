import math

from oneuro.molecular.validation import (
    measure_current_clamp,
    measure_dopamine_gated_plasticity,
    measure_synaptic_response,
    run_validation_suite,
)


def test_current_clamp_metrics_capture_basic_excitable_cell_behavior():
    metrics = measure_current_clamp(
        device="cpu",
        current_candidates_ua=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
    )

    assert -80.0 <= metrics.resting_potential_mv <= -55.0
    assert metrics.subthreshold_current_ua is not None
    assert metrics.subthreshold_delta_mv > 0.1
    assert metrics.input_gain_mv_per_ua is not None
    assert metrics.input_gain_mv_per_ua > 0.0
    assert metrics.rheobase_current_ua is not None
    assert 0.5 <= metrics.rheobase_current_ua <= 8.0
    assert metrics.first_spike_threshold_mv is not None
    assert -50.0 <= metrics.first_spike_threshold_mv <= -15.0
    assert metrics.absolute_refractory_ms is not None
    assert 1.0 <= metrics.absolute_refractory_ms <= 5.0
    assert any(rate > 0.0 for rate in metrics.fi_curve_hz.values())


def test_synaptic_validation_distinguishes_excitatory_and_inhibitory_signs():
    excit = measure_synaptic_response(inhibitory=False, device="cpu")
    inhib = measure_synaptic_response(inhibitory=True, device="cpu")

    assert excit.pre_spike_count > 0
    assert inhib.pre_spike_count > 0
    assert excit.peak_delta_mv > 0.1
    assert inhib.peak_delta_mv < -0.1
    assert excit.peak_time_ms is not None and excit.peak_time_ms > 0.0
    assert inhib.peak_time_ms is not None and inhib.peak_time_ms > 0.0


def test_dopamine_gated_pairing_rewards_pre_before_post():
    metrics = measure_dopamine_gated_plasticity(device="cpu", pairing_trials=12)

    assert abs(metrics.pre_before_post_no_da_delta) < 0.1
    assert metrics.pre_before_post_da_delta > 0.25
    assert metrics.pre_before_post_da_delta > metrics.post_before_pre_da_delta
    assert metrics.rewarded_pairing_final_strength > metrics.unrewarded_pairing_final_strength


def test_validation_suite_exposes_serializable_checks():
    report = run_validation_suite(device="cpu")
    payload = report.to_dict()

    assert payload["checks"]
    assert any(check["passed"] for check in payload["checks"])
    assert math.isfinite(payload["current_clamp"]["resting_potential_mv"])
