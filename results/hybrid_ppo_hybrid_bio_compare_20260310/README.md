# Hybrid PPO Hybrid-Bio Comparison

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
  - predictable/disruptive sensory feedback
- `hybrid_bio`
  - `dishbrain` plus targeted dopamine / 5-HT eligibility-trace feedback
- `no_bio`
  - biological feedback disabled

Aggregate means from `summary.json`:

| condition | train last-q survival | train last-q return | held-out survival | held-out return | held-out damage | held-out kills |
|---|---:|---:|---:|---:|---:|---:|
| dishbrain | 0.500 | -8.070 | 0.750 | -2.468 | 69.0 | 0.917 |
| hybrid_bio | 0.667 | -7.473 | 0.500 | -6.266 | 83.0 | 0.667 |
| no_bio | 0.167 | -11.077 | 0.333 | -11.080 | 101.0 | 0.250 |

Readout:
- `dishbrain` remains the best held-out feedback mode in this short corridor batch.
- `hybrid_bio` improved short-train metrics over `dishbrain`, but generalized worse on held-out eval.
- `no_bio` stayed clearly below both biological-feedback variants.

Conclusion:
- Keep `dishbrain` as the current default feedback style for the split+DAgger policy.
- Keep `eligibility_da` and `hybrid_bio` as experimental branches for longer tuning.

Artifacts:
- aggregate summary: [summary.json](./summary.json)
- per-run JSON:
  - `dishbrain_seed42.json`
  - `dishbrain_seed43.json`
  - `dishbrain_seed44.json`
  - `hybrid_bio_seed42.json`
  - `hybrid_bio_seed43.json`
  - `hybrid_bio_seed44.json`
  - `no_bio_seed42.json`
  - `no_bio_seed43.json`
  - `no_bio_seed44.json`
