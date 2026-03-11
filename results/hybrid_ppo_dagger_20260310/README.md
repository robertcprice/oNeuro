# Hybrid PPO DAgger Corridor Batch

Date: March 10, 2026

Target:
- `demos/demo_doom_hybrid_ppo.py`
- scenario: `deadly_corridor`
- scale: `small`
- device: `cpu`

Batch setup:
- train episodes: `8`
- held-out eval episodes: `4`
- seeds: `42, 43, 44`
- conditions:
  - `baseline`
  - `split_attack`
  - `split_dagger`

Condition details:
- `baseline`
  - flat 6-way decoder
  - no DAgger
- `split_attack`
  - `--split-attack-head`
  - no DAgger
- `split_dagger`
  - `--split-attack-head`
  - `--dagger-episodes 6`
  - `--dagger-epochs 2`
  - `--dagger-coef 0.75`

Aggregate means from `summary.json`:

| condition | train last-q survival | train last-q return | held-out survival | held-out return | held-out damage | held-out kills |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.500 | -8.663 | 0.417 | -10.545 | 96.0 | 0.250 |
| split_attack | 0.667 | -7.337 | 0.583 | -6.883 | 82.5 | 0.500 |
| split_dagger | 0.500 | -8.070 | 0.750 | -2.468 | 69.0 | 0.917 |

Readout:
- Split movement vs attack decoding improved corridor generalization over the flat decoder.
- Adding teacher-labeled DAgger updates improved held-out survival, return, damage, and kills in this 3-seed batch.
- The benefit is clearer in held-out eval than in short training metrics, which fits the goal: reduce bad attack selection and improve corridor behavior under distribution shift.

Artifacts:
- aggregate summary: [summary.json](./summary.json)
- per-run JSON:
  - `baseline_seed42.json`
  - `baseline_seed43.json`
  - `baseline_seed44.json`
  - `split_attack_seed42.json`
  - `split_attack_seed43.json`
  - `split_attack_seed44.json`
  - `split_dagger_seed42.json`
  - `split_dagger_seed43.json`
  - `split_dagger_seed44.json`

Suggested next batch:
- rerun `split_attack` vs `split_dagger` on `12-24` train episodes
- keep held-out eval enabled
- test whether DAgger still wins once PPO has more time to improve on-policy
