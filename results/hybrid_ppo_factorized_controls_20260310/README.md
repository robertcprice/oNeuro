# Hybrid PPO Factorized Controls

Date: 2026-03-10

Goal:
- Test whether allowing simultaneous movement and attack improves held-out Doom combat in `deadly_corridor`.

Code changes:
- Added optional factorized controls in [demo_doom_hybrid_ppo.py](/Users/bobbyprice/projects/oNeuro/demos/demo_doom_hybrid_ppo.py).
- The hybrid policy can now emit:
  - a movement choice (`none`, `forward`, `turn_left`, `turn_right`, `strafe_left`, `strafe_right`)
  - an independent attack button
- The Doom wrapper now accepts a full multi-button action vector in [demo_doom_vizdoom.py](/Users/bobbyprice/projects/oNeuro/demos/demo_doom_vizdoom.py).

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
- `factorized`
  - adds `--factorized-attack-controls`

Aggregate results:

| condition | train survival | train return | train damage | train kills | eval survival | eval return | eval damage | eval kills | pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.722 | -7.268 | 85.50 | 6.33 | 0.458 | -8.717 | 88.75 | 0.333 | 0.667 |
| factorized | 0.556 | -9.253 | 92.92 | 2.67 | 0.292 | -11.342 | 100.75 | 0.208 | 0.667 |

Interpretation:
- The simultaneous movement+attack path worked mechanically but underperformed the current best scalar-action decoder.
- Teacher agreement was also slightly lower in the factorized runs, which suggests the new action space needs better teacher labels or stronger factor-specific imitation, not just a wider control space.
- Current best config remains:
  - `split_attack_head`
  - `feedback_style=dishbrain`
  - `dagger_episodes=8`
  - `dagger_coef=0.75`
  - no replay
  - no factorized attack controls

Artifacts:
- [summary.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/summary.json)
- [baseline_seed42.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/baseline_seed42.json)
- [baseline_seed43.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/baseline_seed43.json)
- [baseline_seed44.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/baseline_seed44.json)
- [factorized_seed42.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/factorized_seed42.json)
- [factorized_seed43.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/factorized_seed43.json)
- [factorized_seed44.json](/Users/bobbyprice/projects/oNeuro/results/hybrid_ppo_factorized_controls_20260310/factorized_seed44.json)
