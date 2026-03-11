# Hybrid PPO Auxiliary Combat Heads

Date: 2026-03-10

Goal:
- Test whether auxiliary supervision for `enemy_visible` and `attack_window` improves held-out combat in `deadly_corridor`.

Code changes:
- Added optional auxiliary heads and losses in [demo_doom_hybrid_ppo.py](/Users/bobbyprice/projects/oNeuro/demos/demo_doom_hybrid_ppo.py).
- Logged:
  - `enemy_visible`
  - `attack_window`
  - `attack_gate`

Experiment:
- Scenario: `deadly_corridor`
- Scale: `small`
- Train episodes: `24`
- Held-out eval episodes: `8`
- Seeds: `42, 43, 44`
- Shared config:
  - `--split-attack-head`
  - `--feedback-style dishbrain`
  - `--dagger-episodes 8`
  - `--dagger-epochs 2`
  - `--dagger-coef 0.75`
  - `--dagger-replay-capacity 0`

Conditions:
- `baseline`
- `aux_default`
  - `--aux-combat-heads --aux-head-coef 0.15 --aux-attack-gate-coef 0.25`
- `aux_soft`
  - `--aux-combat-heads --aux-head-coef 0.08 --aux-attack-gate-coef 0.12`

Aggregate results:

| condition | train survival | train return | train kills | eval survival | eval return | eval damage | eval kills |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.722 | -7.268 | 6.33 | 0.458 | -8.717 | 88.75 | 0.333 |
| aux_default | 0.556 | -8.772 | 6.33 | 0.375 | -10.124 | 96.00 | 0.292 |
| aux_soft | 0.667 | -7.629 | 5.33 | 0.333 | -10.687 | 95.50 | 0.208 |

Interpretation:
- The auxiliary combat heads did not improve held-out `deadly_corridor`.
- Both auxiliary settings underperformed the simpler baseline on survival, return, and kills.
- Current best config remained the plain split attack head with DishBrain feedback and short-horizon DAgger.

Artifacts:
- [summary.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_aux_heads_20260310/summary.json)
