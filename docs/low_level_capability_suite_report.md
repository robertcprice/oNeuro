# Low-level capability suite

This document explains the benchmark ladder added in this PR and how it fits
into the repo.

## Purpose

The repo has many demos, but those are hard to evaluate in isolation. The
low-level capability suite adds a smaller benchmark ladder that checks whether
the molecular simulator can support basic mechanisms and simple learned
behaviors in a controlled way.

The goal is not to claim full biological validation. The goal is to make these
smaller claims easier to test and review:

- membrane excitability changes with channel composition
- drugs shift microcircuit activity in expected directions
- rewarded corticostriatal plasticity depends on NMDA / dopamine context
- tiny circuits can learn simple cue mappings and action biases
- the simulator can support short-delay retention and partial-cue recovery

## Main pieces

- `experiments/corticostriatal_mechanism_experiment.py`
  - Low-level mechanistic pair protocol for reward-modulated corticostriatal
    plasticity.
- `experiments/corticostriatal_action_bias_benchmark.py`
  - Small cue-conditioned D1-vs-D2 action-bias benchmark.
- `experiments/stimulus_discrimination_benchmark.py`
  - Simpler two-cue discrimination benchmark.
- `experiments/low_level_capability_suite.py`
  - One entrypoint that runs the whole ladder and writes one JSON result.
- `experiments/benchmark_shared.py`
  - Shared microcircuit benchmark utilities.

## Library changes this suite depends on

- `src/oneuro/molecular/neuron.py`
  - Adds distinct `D1_MSN` and `D2_MSN` archetypes.
- `src/oneuro/molecular/brain_regions.py`
  - Uses those archetypes in basal ganglia construction.
- `src/oneuro/molecular/network.py`
  - Adds `benchmark_safe_mode`.
  - Adds pathway-aware dopamine plasticity sign handling.
  - Preserves recent `main` connectivity fixes such as thalamus relay -> L5.
- `src/oneuro/molecular/synapse.py`
  - Hardens reward-modulated AMPA trafficking so blocked or ineligible synapses
    do not still drift through `strength`.
- `src/oneuro/molecular/microtubules.py`
  - Keeps runtime-entropy-compatible cytoskeleton stepping aligned with the
    current neuron/network interfaces.
- `src/oneuro/molecular/runtime_entropy.py`
  - Provides the runtime entropy controller used by the current network stack.

## Why these fixes matter

The suite only makes sense if the short assays are not distorted by unrelated
structural changes and if reward-modulated plasticity behaves in a way that is
internally consistent.

The most important correctness points are:

- short benchmark runs use `benchmark_safe_mode` to avoid pruning, PNN wrapping,
  and spontaneous structure changes during the assay
- reward-modulated plasticity only changes effective synaptic state when there
  is actual AMPA trafficking
- non-glutamatergic synapses are not mutated by the AMPA-trafficking reward rule
- the mechanism assay uses matched initial conditions across its core contrasts

## What the suite covers

`experiments/low_level_capability_suite.py` runs eight benchmark rungs:

1. single-neuron excitability
2. drug response on matched microcircuits
3. NMDA-dependent corticostriatal plasticity
4. simple stimulus discrimination
5. small Go / No-Go-style action bias
6. classical conditioning
7. short-delay working memory
8. pattern completion

Each rung emits a status such as `positive`, `ambiguous`, or `negative` based
on the benchmark's own control logic.

## How to run

Run the full suite:

```bash
PYTHONPATH=src python3 experiments/low_level_capability_suite.py
```

Run the strongest individual surfaces:

```bash
PYTHONPATH=src python3 experiments/corticostriatal_mechanism_experiment.py
PYTHONPATH=src python3 experiments/stimulus_discrimination_benchmark.py
PYTHONPATH=src python3 experiments/corticostriatal_action_bias_benchmark.py
```

Run the regression surface for this PR:

```bash
PYTHONPATH=src pytest -q \
  tests/test_corticostriatal_mechanism.py \
  tests/test_corticostriatal_action_bias_benchmark.py \
  tests/test_stimulus_discrimination_benchmark.py \
  tests/test_low_level_capability_suite.py
```

## Reviewer guidance

If you want to review the smallest high-signal path first, start here:

1. `src/oneuro/molecular/synapse.py`
2. `experiments/corticostriatal_mechanism_experiment.py`
3. `tests/test_corticostriatal_mechanism.py`
4. `experiments/stimulus_discrimination_benchmark.py`
5. `experiments/low_level_capability_suite.py`

That path covers the main mechanism fix, the cleanest assay, and the new
consolidated benchmark entrypoint.
