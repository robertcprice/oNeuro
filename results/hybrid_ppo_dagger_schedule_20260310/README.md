# Hybrid PPO DAgger Schedule Sweep

Date: March 10, 2026

Target:
- `demos/demo_doom_hybrid_ppo.py`
- scenario: `deadly_corridor`
- scale: `small`
- device: `cpu`

Shared settings:
- `--split-attack-head`
- `--feedback-style dishbrain`
- `--dagger-replay-capacity 0`

Batch setup:
- train episodes: `24`
- held-out eval episodes: `8`
- seeds: `42, 43, 44`

Conditions:
- `dagger8_c075`
  - `--dagger-episodes 8`
  - `--dagger-coef 0.75`
- `dagger24_c075`
  - `--dagger-episodes 24`
  - `--dagger-coef 0.75`
- `dagger24_c050`
  - `--dagger-episodes 24`
  - `--dagger-coef 0.50`

Aggregate means from `summary.json`:

| condition | train last-q survival | train last-q return | held-out survival | held-out return | held-out damage | held-out kills |
|---|---:|---:|---:|---:|---:|---:|
| dagger8_c075 | 0.722 | -7.268 | 0.458 | -8.717 | 88.75 | 0.333 |
| dagger24_c075 | 0.722 | -7.251 | 0.250 | -12.130 | 99.50 | 0.125 |
| dagger24_c050 | 0.667 | -8.136 | 0.333 | -11.035 | 95.50 | 0.167 |

Readout:
- Extending teacher labeling across the full training horizon did not improve held-out corridor behavior.
- Full-horizon DAgger appears to overfit the teacher policy and generalize worse than the shorter schedule.
- The original shorter schedule remains the best current setting.

Conclusion:
- Keep `--dagger-episodes 8 --dagger-coef 0.75` as the current best DAgger schedule.
- Do not promote full-horizon DAgger as the default.

Artifacts:
- aggregate summary: [summary.json](./summary.json)
