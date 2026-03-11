# Hybrid PPO Feedback Comparison

Date: March 10, 2026

Target:
- `demos/demo_doom_hybrid_ppo.py`
- scenario: `deadly_corridor`
- scale: `small`
- device: `cpu`

Shared policy settings:
- `--split-attack-head`
- `--dagger-episodes 6`
- `--dagger-epochs 2`
- `--dagger-coef 0.75`

Batch setup:
- train episodes: `8`
- held-out eval episodes: `4`
- seeds: `42, 43, 44`

Conditions:
- `dishbrain`
- `eligibility_da`
- `no_bio`

Aggregate means from `summary.json`:

| condition | train last-q survival | train last-q return | held-out survival | held-out return | held-out damage | held-out kills |
|---|---:|---:|---:|---:|---:|---:|
| dishbrain | 0.500 | -8.070 | 0.750 | -2.468 | 69.0 | 0.917 |
| eligibility_da | 0.667 | -7.558 | 0.667 | -4.786 | 75.0 | 0.667 |
| no_bio | 0.167 | -11.077 | 0.333 | -11.080 | 101.0 | 0.250 |

Readout:
- Both biological-feedback variants beat `no_bio`.
- `eligibility_da` improved short-train metrics over `dishbrain`, but generalized worse on held-out eval.
- `dishbrain` remained the best overall held-out setting in this short batch.

Conclusion:
- The biological loop is helping again after the decoder + DAgger rewrite.
- The new eligibility-trace dopamine branch is viable, but not yet better than the simpler DishBrain signal.

Artifacts:
- aggregate summary: [summary.json](./summary.json)
