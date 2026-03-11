# Hybrid PPO Hard-State DAgger Replay

Date: March 10, 2026

Target:
- `demos/demo_doom_hybrid_ppo.py`
- scenario: `deadly_corridor`
- scale: `small`
- device: `cpu`

Shared settings:
- `--split-attack-head`
- `--feedback-style dishbrain`
- `--dagger-episodes 8`
- `--dagger-epochs 2`
- `--dagger-coef 0.75`

Batch setup:
- train episodes: `24`
- held-out eval episodes: `8`
- seeds: `42, 43, 44`

Conditions:
- `plain_dagger`
  - `--dagger-replay-capacity 0`
- `hard_dagger`
  - `--dagger-replay-capacity 4096`
  - `--dagger-replay-ratio 1.5`

Aggregate means from `summary.json`:

| condition | train last-q survival | train last-q return | held-out survival | held-out return | held-out damage | held-out kills |
|---|---:|---:|---:|---:|---:|---:|
| plain_dagger | 0.722 | -7.268 | 0.458 | -8.717 | 88.75 | 0.333 |
| hard_dagger | 0.667 | -6.917 | 0.250 | -11.721 | 100.75 | 0.208 |

Replay stats:
- `plain_dagger`: replay size `0`
- `hard_dagger`: mean replay size `572.3`

Readout:
- Hard-state replay did not improve held-out corridor performance.
- It slightly improved short-train return, but generalized worse than the simpler DAgger baseline.
- The replay branch appears to overfit hard teacher states instead of learning a broader stable policy.

Conclusion:
- Keep `plain_dagger + split_attack_head + dishbrain` as the best current Doom setting.
- Keep hard-state replay disabled by default.

Artifacts:
- aggregate summary: [summary.json](./summary.json)
