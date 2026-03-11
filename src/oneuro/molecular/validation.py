"""Electrophysiology-style validation protocols for the molecular GPU backend.

The goal here is not to assert that the simulator already matches a specific
cell class perfectly. Instead, this module provides repeatable measurements for
core single-neuron and synaptic behaviors so the model can be benchmarked and
tuned against reference physiology over time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .cuda_backend import CUDAMolecularBrain, NT_DA, NT_GABA, NT_GLU


@dataclass
class ReferenceRange:
    """Inclusive reference window for a scalar benchmark metric."""

    lower: float
    upper: float
    unit: str = ""

    def contains(self, value: Optional[float]) -> Optional[bool]:
        if value is None:
            return None
        return self.lower <= float(value) <= self.upper


@dataclass
class ValidationCheck:
    """One physiology-style benchmark check."""

    name: str
    value: Optional[float]
    unit: str
    target: Optional[ReferenceRange]
    passed: Optional[bool]
    note: str = ""


@dataclass
class CurrentClampMetrics:
    """Current-clamp style measurements for a single model neuron."""

    resting_potential_mv: float
    subthreshold_current_ua: Optional[float]
    subthreshold_delta_mv: float
    input_gain_mv_per_ua: Optional[float]
    membrane_tau_ms: Optional[float]
    rheobase_current_ua: Optional[float]
    fi_curve_hz: Dict[float, float] = field(default_factory=dict)
    first_spike_threshold_mv: Optional[float] = None
    first_spike_peak_mv: Optional[float] = None
    absolute_refractory_ms: Optional[float] = None


@dataclass
class SynapticResponseMetrics:
    """Postsynaptic response to a brief presynaptic spike train."""

    synapse_type: str
    baseline_post_mv: float
    peak_delta_mv: float
    peak_time_ms: Optional[float]
    half_decay_ms: Optional[float]
    pre_spike_count: int
    post_spike_count: int


@dataclass
class PlasticityMetrics:
    """Three-factor plasticity measurements from paired pre/post activation."""

    pre_before_post_no_da_delta: float
    pre_before_post_da_delta: float
    post_before_pre_da_delta: float
    rewarded_pairing_final_strength: float
    unrewarded_pairing_final_strength: float
    reverse_pairing_final_strength: float


@dataclass
class ValidationReport:
    """Aggregate neuron validation report."""

    current_clamp: CurrentClampMetrics
    excitatory_synapse: SynapticResponseMetrics
    inhibitory_synapse: SynapticResponseMetrics
    plasticity: PlasticityMetrics
    checks: List[ValidationCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for check in data["checks"]:
            target = check.get("target")
            if target is None:
                continue
            check["target"] = dict(target)
        return data


DEFAULT_REFERENCE_WINDOWS: Dict[str, ReferenceRange] = {
    "resting_potential_mv": ReferenceRange(-80.0, -55.0, "mV"),
    "first_spike_threshold_mv": ReferenceRange(-50.0, -15.0, "mV"),
    "absolute_refractory_ms": ReferenceRange(1.0, 5.0, "ms"),
}


def _make_validation_brain(
    n_neurons: int,
    device: str = "cpu",
    psc_scale: float = 300.0,
) -> CUDAMolecularBrain:
    """Create a deterministic validation brain on the requested device."""
    brain = CUDAMolecularBrain(n_neurons, device=device, psc_scale=psc_scale)
    brain.disable_interval_biology()
    brain.set_triton_enabled(False)
    return brain


def _pulse_neuron(
    brain: CUDAMolecularBrain,
    neuron_idx: int,
    current_ua: float,
    pulse_steps: int,
) -> int:
    """Inject repeated current steps and return the number of emitted spikes."""
    spikes = 0
    for _ in range(max(1, int(pulse_steps))):
        brain.stimulate(neuron_idx, current_ua)
        brain.step()
        spikes += int(bool(brain.fired[neuron_idx]))
    return spikes


def _mean_tail(values: Sequence[float], window: int = 200) -> float:
    """Return the mean of the trailing window, or of the full list if shorter."""
    if not values:
        return 0.0
    tail = values[-min(len(values), int(window)) :]
    return float(sum(tail) / max(1, len(tail)))


def _first_crossing_time_ms(
    trace: Sequence[float],
    target: float,
    dt_ms: float,
) -> Optional[float]:
    """Return first time the trace crosses target."""
    for idx, value in enumerate(trace):
        if value >= target:
            return float((idx + 1) * dt_ms)
    return None


def measure_current_clamp(
    device: str = "cpu",
    settle_steps: int = 800,
    pulse_steps: int = 1000,
    recovery_steps: int = 400,
    current_candidates_ua: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0),
    spike_pulse_current_ua: float = 50.0,
    spike_pulse_steps: int = 10,
) -> CurrentClampMetrics:
    """Measure basic current-clamp metrics on a single neuron."""
    traces: Dict[float, Dict[str, Any]] = {}
    subthreshold_current = None
    rheobase_current = None

    for current in current_candidates_ua:
        brain = _make_validation_brain(1, device=device, psc_scale=300.0)
        baseline: List[float] = []
        pulse_trace: List[float] = []
        pulse_prev: List[float] = []
        pulse_fired: List[bool] = []

        for step_idx in range(settle_steps + pulse_steps + recovery_steps):
            if settle_steps <= step_idx < (settle_steps + pulse_steps):
                brain.stimulate(0, float(current))
            brain.step()
            voltage = float(brain.voltage[0])
            if step_idx < settle_steps:
                baseline.append(voltage)
            elif step_idx < (settle_steps + pulse_steps):
                pulse_trace.append(voltage)
                pulse_prev.append(float(brain.prev_voltage[0]))
                pulse_fired.append(bool(brain.fired[0]))

        resting_mv = _mean_tail(baseline)
        steady_mv = _mean_tail(pulse_trace)
        delta_mv = steady_mv - resting_mv
        spike_count = int(sum(int(v) for v in pulse_fired))

        traces[float(current)] = {
            "resting_mv": resting_mv,
            "steady_mv": steady_mv,
            "delta_mv": delta_mv,
            "pulse_trace": pulse_trace,
            "pulse_prev": pulse_prev,
            "pulse_fired": pulse_fired,
            "spike_count": spike_count,
        }

        if spike_count == 0:
            subthreshold_current = float(current)
        elif rheobase_current is None:
            rheobase_current = float(current)

    if subthreshold_current is None:
        subthreshold_current = float(current_candidates_ua[0])
    sub = traces[subthreshold_current]

    membrane_tau_ms = None
    steady_mv = float(sub["steady_mv"])
    resting_mv = float(sub["resting_mv"])
    if steady_mv > resting_mv:
        target_mv = resting_mv + 0.632 * (steady_mv - resting_mv)
        membrane_tau_ms = _first_crossing_time_ms(
            sub["pulse_trace"], target_mv, dt_ms=0.1
        )

    fi_curve_hz: Dict[float, float] = {}
    for current, payload in traces.items():
        pulse_seconds = pulse_steps * 0.0001
        fi_curve_hz[float(current)] = float(payload["spike_count"]) / pulse_seconds

    first_spike_threshold_mv = None
    first_spike_peak_mv = None
    if rheobase_current is not None:
        rheo = traces[rheobase_current]
        for idx, fired in enumerate(rheo["pulse_fired"]):
            if fired:
                first_spike_threshold_mv = float(rheo["pulse_prev"][idx])
                peak_window = rheo["pulse_trace"][idx : idx + 20]
                first_spike_peak_mv = float(max(peak_window)) if peak_window else None
                break

    # Double-pulse refractory probe.
    refractory_ms = None
    for gap_steps in range(0, 25):
        brain = _make_validation_brain(1, device=device, psc_scale=300.0)
        spike_steps: List[int] = []
        for _ in range(20):
            brain.step()
        cursor = 20
        for _ in range(spike_pulse_steps):
            brain.stimulate(0, spike_pulse_current_ua)
            brain.step()
            if bool(brain.fired[0]):
                spike_steps.append(cursor)
            cursor += 1
        for _ in range(gap_steps):
            brain.step()
            if bool(brain.fired[0]):
                spike_steps.append(cursor)
            cursor += 1
        for _ in range(spike_pulse_steps):
            brain.stimulate(0, spike_pulse_current_ua)
            brain.step()
            if bool(brain.fired[0]):
                spike_steps.append(cursor)
            cursor += 1
        if len(spike_steps) >= 2:
            refractory_ms = float((spike_steps[1] - spike_steps[0]) * brain.dt)
            break

    input_gain = None
    if subthreshold_current and subthreshold_current > 0.0:
        input_gain = float(sub["delta_mv"]) / float(subthreshold_current)

    return CurrentClampMetrics(
        resting_potential_mv=float(sub["resting_mv"]),
        subthreshold_current_ua=float(subthreshold_current),
        subthreshold_delta_mv=float(sub["delta_mv"]),
        input_gain_mv_per_ua=input_gain,
        membrane_tau_ms=membrane_tau_ms,
        rheobase_current_ua=rheobase_current,
        fi_curve_hz=fi_curve_hz,
        first_spike_threshold_mv=first_spike_threshold_mv,
        first_spike_peak_mv=first_spike_peak_mv,
        absolute_refractory_ms=refractory_ms,
    )


def measure_synaptic_response(
    inhibitory: bool = False,
    device: str = "cpu",
    syn_weight: float = 5.0,
    pulse_current_ua: float = 50.0,
    pulse_steps: int = 10,
) -> SynapticResponseMetrics:
    """Measure the sign and decay of a one-synapse postsynaptic response."""
    brain = _make_validation_brain(2, device=device, psc_scale=300.0)
    nt_type = NT_GABA if inhibitory else NT_GLU
    brain.add_synapses(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([float(syn_weight)]),
        torch.tensor([nt_type], dtype=torch.int32),
    )

    baseline_trace: List[float] = []
    response_trace: List[float] = []
    for _ in range(40):
        brain.step()
        baseline_trace.append(float(brain.voltage[1]))
    for _ in range(max(1, int(pulse_steps))):
        brain.stimulate(0, pulse_current_ua)
        brain.step()
        response_trace.append(float(brain.voltage[1]))
    for _ in range(120):
        brain.step()
        response_trace.append(float(brain.voltage[1]))

    baseline_mv = _mean_tail(baseline_trace, window=20)
    if inhibitory:
        peak_value = float(min(response_trace))
        peak_index = int(min(range(len(response_trace)), key=lambda i: response_trace[i]))
    else:
        peak_value = float(max(response_trace))
        peak_index = int(max(range(len(response_trace)), key=lambda i: response_trace[i]))
    peak_delta = peak_value - baseline_mv

    half_decay_ms = None
    if peak_delta != 0.0:
        half_value = baseline_mv + 0.5 * peak_delta
        for idx in range(peak_index + 1, len(response_trace)):
            value = response_trace[idx]
            if inhibitory:
                if value >= half_value:
                    half_decay_ms = float((idx - peak_index) * brain.dt)
                    break
            else:
                if value <= half_value:
                    half_decay_ms = float((idx - peak_index) * brain.dt)
                    break

    return SynapticResponseMetrics(
        synapse_type="inhibitory" if inhibitory else "excitatory",
        baseline_post_mv=baseline_mv,
        peak_delta_mv=peak_delta,
        peak_time_ms=float((peak_index + 1) * brain.dt),
        half_decay_ms=half_decay_ms,
        pre_spike_count=int(brain.spike_count[0]),
        post_spike_count=int(brain.spike_count[1]),
    )


def measure_dopamine_gated_plasticity(
    device: str = "cpu",
    pairing_trials: int = 12,
    inter_trial_steps: int = 20,
    post_pair_steps: int = 40,
    pair_delay_steps: int = 3,
    pulse_current_ua: float = 50.0,
    pulse_steps: int = 10,
    dopamine_amount: float = 200.0,
) -> PlasticityMetrics:
    """Measure eligibility-trace learning with and without dopamine."""

    def _run_pairing(pre_before_post: bool, dopamine: bool) -> float:
        brain = _make_validation_brain(2, device=device, psc_scale=300.0)
        brain.add_synapses(
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2.0]),
            torch.tensor([NT_GLU], dtype=torch.int32),
        )
        base_strength = float(brain.syn_strength[0])
        for _ in range(max(1, int(pairing_trials))):
            for _ in range(max(0, int(inter_trial_steps))):
                brain.step()
            if pre_before_post:
                _pulse_neuron(brain, 0, pulse_current_ua, pulse_steps)
                for _ in range(max(0, int(pair_delay_steps))):
                    brain.step()
                _pulse_neuron(brain, 1, pulse_current_ua, pulse_steps)
            else:
                _pulse_neuron(brain, 1, pulse_current_ua, pulse_steps)
                for _ in range(max(0, int(pair_delay_steps))):
                    brain.step()
                _pulse_neuron(brain, 0, pulse_current_ua, pulse_steps)
            if dopamine:
                brain.nt_conc[1, NT_DA] += dopamine_amount
            for _ in range(max(0, int(post_pair_steps))):
                brain.step()
        return float(brain.syn_strength[0] - base_strength), float(brain.syn_strength[0])

    pre_no_da, pre_no_da_final = _run_pairing(pre_before_post=True, dopamine=False)
    pre_da, pre_da_final = _run_pairing(pre_before_post=True, dopamine=True)
    post_da, post_da_final = _run_pairing(pre_before_post=False, dopamine=True)

    return PlasticityMetrics(
        pre_before_post_no_da_delta=pre_no_da,
        pre_before_post_da_delta=pre_da,
        post_before_pre_da_delta=post_da,
        rewarded_pairing_final_strength=pre_da_final,
        unrewarded_pairing_final_strength=pre_no_da_final,
        reverse_pairing_final_strength=post_da_final,
    )


def _build_validation_checks(report: ValidationReport) -> List[ValidationCheck]:
    """Attach broad physiology-style checks to the measured report."""
    checks = [
        ValidationCheck(
            name="resting_potential_mv",
            value=report.current_clamp.resting_potential_mv,
            unit="mV",
            target=DEFAULT_REFERENCE_WINDOWS["resting_potential_mv"],
            passed=DEFAULT_REFERENCE_WINDOWS["resting_potential_mv"].contains(
                report.current_clamp.resting_potential_mv
            ),
        ),
        ValidationCheck(
            name="first_spike_threshold_mv",
            value=report.current_clamp.first_spike_threshold_mv,
            unit="mV",
            target=DEFAULT_REFERENCE_WINDOWS["first_spike_threshold_mv"],
            passed=DEFAULT_REFERENCE_WINDOWS["first_spike_threshold_mv"].contains(
                report.current_clamp.first_spike_threshold_mv
            ),
        ),
        ValidationCheck(
            name="absolute_refractory_ms",
            value=report.current_clamp.absolute_refractory_ms,
            unit="ms",
            target=DEFAULT_REFERENCE_WINDOWS["absolute_refractory_ms"],
            passed=DEFAULT_REFERENCE_WINDOWS["absolute_refractory_ms"].contains(
                report.current_clamp.absolute_refractory_ms
            ),
        ),
        ValidationCheck(
            name="excitatory_peak_delta_mv",
            value=report.excitatory_synapse.peak_delta_mv,
            unit="mV",
            target=None,
            passed=report.excitatory_synapse.peak_delta_mv > 0.1,
            note="Excitatory synapse should depolarize the postsynaptic cell.",
        ),
        ValidationCheck(
            name="inhibitory_peak_delta_mv",
            value=report.inhibitory_synapse.peak_delta_mv,
            unit="mV",
            target=None,
            passed=report.inhibitory_synapse.peak_delta_mv < -0.1,
            note="Inhibitory synapse should hyperpolarize the postsynaptic cell.",
        ),
        ValidationCheck(
            name="dopamine_gated_ltp",
            value=report.plasticity.pre_before_post_da_delta,
            unit="syn_strength",
            target=None,
            passed=report.plasticity.pre_before_post_da_delta > 0.25,
            note="Rewarded pre-before-post pairing should strengthen the synapse.",
        ),
        ValidationCheck(
            name="unrewarded_pairing_stability",
            value=report.plasticity.pre_before_post_no_da_delta,
            unit="syn_strength",
            target=None,
            passed=abs(report.plasticity.pre_before_post_no_da_delta) < 0.1,
            note="Pre-before-post pairing without dopamine should remain near baseline.",
        ),
        ValidationCheck(
            name="reverse_pairing_weaker_than_rewarded",
            value=report.plasticity.post_before_pre_da_delta,
            unit="syn_strength",
            target=None,
            passed=(
                report.plasticity.post_before_pre_da_delta
                < report.plasticity.pre_before_post_da_delta
            ),
            note="Rewarded post-before-pre should not outperform rewarded pre-before-post.",
        ),
    ]
    return checks


def run_validation_suite(device: str = "cpu") -> ValidationReport:
    """Run the full validation suite on the selected device."""
    report = ValidationReport(
        current_clamp=measure_current_clamp(device=device),
        excitatory_synapse=measure_synaptic_response(inhibitory=False, device=device),
        inhibitory_synapse=measure_synaptic_response(inhibitory=True, device=device),
        plasticity=measure_dopamine_gated_plasticity(device=device),
    )
    report.checks = _build_validation_checks(report)
    return report

