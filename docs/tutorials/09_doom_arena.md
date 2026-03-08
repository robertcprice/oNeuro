# Tutorial 9: Doom Arena — Spatial Navigation with dONNs

## Overview

This tutorial walks through oNeuro's Doom Arena demo, which extends the DishBrain Pong paradigm (Tutorial 8, Application 5) from 1D tracking to 2D spatial navigation in a procedurally generated dungeon environment. The dONN (digital Organic Neural Network) must navigate rooms and corridors to reach a goal while avoiding enemies and collecting health pickups — all learning is driven by the **Free Energy Principle**, not reward signals.

This tutorial covers:
- The Free Energy Principle (FEP) in spatial environments
- Egocentric local-view sensory encoding (retinotopic V1 analogy)
- 8-directional motor decoding from L5 spike populations
- BSP dungeon generation
- Biological grounding and pharmacological experiments
- Running experiments and interpreting results

## Prerequisites

- Completed Tutorial 1 (Getting Started) and Tutorial 4 (Learning)
- Familiarity with the DishBrain demo (`demos/demo_dishbrain_pong.py`)
- PyTorch installed (`pip install torch`)

## Quick Start

```bash
cd oNeuro

# Run all 3 experiments at small scale
PYTHONPATH=src python3 demos/demo_doom_arena.py

# Run just navigation (Experiment 1)
PYTHONPATH=src python3 demos/demo_doom_arena.py --exp 1

# Run at medium scale with JSON output
PYTHONPATH=src python3 demos/demo_doom_arena.py --scale medium --json results.json

# Multi-seed statistical analysis
PYTHONPATH=src python3 demos/demo_doom_arena.py --runs 5 --json multi_seed.json
```

## The Free Energy Principle in Spatial Environments

### From Pong to Doom

In the DishBrain Pong demo, the FEP protocol is simple:
- **Hit** (correct paddle position): Structured pulse → low entropy → STDP strengthens active pathways
- **Miss**: Random noise → high entropy → no systematic STDP

The Doom Arena extends this to a richer set of spatial outcomes:

| Event | FEP Response | Entropy | Biological Analogy |
|-------|-------------|---------|-------------------|
| **Reach goal** | Strong structured pulse + NE boost | Very low | Successful foraging → locus coeruleus activation |
| **Survive near enemy** | Mild structured pulse | Low | Vigilance state → norepinephrine-mediated attention |
| **Pick up health** | Mild structured feedback | Low | Positive consummatory signal |
| **Take damage** | Random noise to 30% of cortex | High | Pain/stress → cortisol-mediated disruption |
| **Death** | Strong random noise | Very high | Catastrophic allostatic failure |

The key insight: the network doesn't receive an explicit "reward" signal. Instead, it self-organizes via STDP to prefer neural states that produce *predictable* sensory feedback (structured pulses) over *unpredictable* feedback (random noise). This is the free energy principle at work — the network minimizes surprise.

### Why Not Reward?

Traditional reinforcement learning uses an explicit reward signal (e.g., dopamine). The FEP protocol is fundamentally different:

1. **No scalar reward**: There is no number telling the network "how good" an outcome was.
2. **No credit assignment**: There is no mechanism attributing the reward to specific actions.
3. **Information-theoretic**: The distinction between hit and miss is purely about the *predictability* of the sensory feedback pattern.

The Hebbian weight nudge (direct synaptic strengthening toward the optimal action) accelerates learning but isn't the FEP itself — it's a biological acceleration mechanism analogous to hippocampal replay.

## Neural Architecture

### Sensory Encoding: Egocentric Local View

The agent perceives a **5×5 egocentric local view** centered on its position. This 25-cell grid captures immediate surroundings, analogous to the limited visual field processed by retinal ganglion cells → LGN → V1.

```
Agent's 5×5 View (example):
┌───┬───┬───┬───┬───┐
│ W │ . │ . │ . │ W │   W = Wall
│ . │ . │ E │ . │ . │   E = Enemy
│ . │ . │ @ │ . │ . │   @ = Agent (center)
│ . │ H │ . │ . │ . │   H = Health pickup
│ W │ W │ . │ G │ W │   G = Goal
└───┴───┴───┴───┴───┘   . = Empty floor
```

**Encoding**: The relay neurons (thalamus) are split into 25 groups, one per cell in the view. Each group receives an intensity proportional to the cell content:

| Cell Content | Intensity | Biological Motivation |
|-------------|-----------|----------------------|
| Empty | 15.0 | Baseline spatial encoding |
| Wall | 45.0 | Strong boundary signal (like contrast edges in V1) |
| Enemy | 70.0 | High-salience threat (amygdala activation in biology) |
| Health | 55.0 | Positive valence stimulus |
| Goal | 65.0 | Goal-related spatial signal |

This is a **rate-place code** — the position in the view determines *which* neurons fire, and the content type determines *how strongly* they fire. This mimics how V1 neurons have receptive fields: each responds to stimuli in a specific location of the visual field.

### Motor Decoding: 8-Directional Population Code

The L5 (output layer) neurons are split into 8 populations, one per cardinal/ordinal direction:

```
Direction mapping:
   NW  N  NE      7  0  1
    W  ·  E   →   6  ·  2
   SW  S  SE      5  4  3
```

**Decoding**: Zero-threshold spike-count majority voting. Whichever directional population has the most spikes in the stimulus window determines the action. If all counts are zero, a random action is chosen (exploration).

The zero-threshold decoder is essential — learned from the DishBrain experiments, where even a 1-spike difference must drive action to break initial symmetry. Without this, the network starts in a symmetric equilibrium and never bootstraps learning.

### Hebbian Weight Nudge

On each step, the system computes the **optimal action** (move toward goal while avoiding enemies) and applies a direct Hebbian weight nudge:
- **Correct motor population**: Strengthen relay→motor synapses by `+delta`
- **Wrong motor populations**: Weaken relay→motor synapses by `-delta × 0.15`

The delta is scale-adaptive: `0.8 × max(1.0, (n_l5 / 200)^0.3)`. Larger networks need bigger nudges to overcome their larger noise floor.

## Dungeon Generation: BSP Algorithm

The arena uses **Binary Space Partitioning** to generate a deterministic dungeon layout:

1. Start with the full grid as one room
2. Recursively split rooms along the longer axis (with some randomness)
3. Stop when rooms reach minimum size (5×5)
4. Connect rooms with corridors (1-cell-wide passages between room centers)
5. Place agent and goal in opposite rooms
6. Place enemies and health pickups in non-wall cells

The BSP algorithm is seeded, so the same seed always produces the same dungeon. This is critical for reproducibility — when comparing drug conditions, all three brains see the same maze layouts.

## Three Experiments

### Experiment 1: Doom Navigation

**Question**: Can the dONN navigate rooms to reach a goal?

**Setup**: 50 episodes in a procedurally generated dungeon (15×15 at small scale, 25×25 at medium+). 2 enemies, 3 health pickups. The agent has 60-100 steps per episode.

**Pass criteria**: Any evidence of learning:
- Goal rate above random baseline (5%), OR
- Score improvement from first to last quarter, OR
- Goal rate improvement from first to last quarter

**Biological analogy**: Morris water maze — rodents learn to navigate to a hidden platform using spatial cues. Hippocampal lesions impair this task.

### Experiment 2: Doom Threat Avoidance

**Question**: Does the dONN learn to avoid enemies?

**Setup**: 80 episodes with 3 enemies and 2 health pickups. Primary metrics: survival rate (ending with HP > 0) and total damage taken.

**Pass criteria**: Survival rate improves OR damage decreases from first to last quarter.

**Biological analogy**: Conditioned place avoidance — rodents learn to avoid locations associated with aversive stimuli. The FEP noise on enemy contact creates a free-energy gradient away from enemy positions.

### Experiment 3: Doom Drug Effects

**Question**: Does diazepam impair spatial navigation?

**Setup**: Train 3 identical brains (same seed = same wiring), apply drug AFTER training, test with same arena sequence. Drugs: caffeine (200mg, adenosine antagonist), diazepam (40mg, GABA-A potentiator).

**Pass criteria**: Diazepam test score < baseline test score (or diazepam causes more damage).

**Biological analogy**: Morris (1984) demonstrated that benzodiazepines impair hippocampal-dependent spatial learning. This is a key advantage of the dONN approach — drug application is reversible, repeatable, and doesn't require animal subjects.

## Running on GPU

For publication-grade results, run at medium or large scale on a CUDA GPU:

```bash
# On a Vast.ai GPU instance
bash scripts/vast_deploy.sh doom <instance_id> medium --runs 5

# Or directly on a CUDA machine
PYTHONPATH=src python3 demos/demo_doom_arena.py \
    --scale medium --device cuda --runs 5 --json doom_medium.json
```

## Interpreting Results

### Score Components

The episode score captures multiple aspects of performance:

```
Score = (reached_goal × 100) + remaining_HP + (pickups × 10) - steps_taken
```

- **Goal bonus (100)**: Large reward for reaching the target
- **HP retention**: Surviving with high HP indicates enemy avoidance
- **Pickup bonus (10 each)**: Collecting health shows spatial exploration
- **Step penalty (-1 per step)**: Efficient navigation is better

### Learning Curves

A typical learning curve shows:
1. **Early episodes**: Random exploration, frequent deaths, low scores (~−50 to +10)
2. **Mid episodes**: Developing directional bias, some goal reaches, less damage (~+10 to +50)
3. **Late episodes**: Consistent navigation toward goals, enemy avoidance (~+50 to +120)

At small scale (1K neurons), expect noisy learning with many failures. At medium scale (5K neurons), learning curves are smoother and goal rates are higher.

## Comparison to Original DishBrain

| Aspect | DishBrain (Kagan 2022) | oNeuro Pong | oNeuro Doom Arena |
|--------|----------------------|-------------|-------------------|
| **Substrate** | 800K living neurons (ONN) | 1K-100K simulated (dONN) | 1K-100K simulated (dONN) |
| **Task** | 1D Pong | 1D Pong | 25×25 dungeon navigation |
| **Actions** | 2 (left/right) | 3 (up/down/hold) | 8 (cardinal + ordinal) |
| **Sensory** | MEA electrodes | Gaussian population code | 5×5 egocentric view |
| **Learning** | FEP (structured/random) | FEP + Hebbian nudge | FEP + Hebbian nudge |
| **Drugs** | Not reversible | Reversible | Reversible |
| **Environment** | Fixed | Fixed | Procedurally generated |

## References

- Kagan, B.J., et al. (2022). "In vitro neurons learn and exhibit sentience when embodied in a simulated game-world." *Neuron*, 110(23), 3952-3969.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11, 127-138.
- Morris, R.G.M. (1984). "Developments of a water-maze procedure for studying spatial learning in the rat." *J. Neurosci. Methods*, 11(1), 47-60.
- O'Keefe, J. & Dostrovsky, J. (1971). "The hippocampus as a spatial map." *Brain Research*, 34(1), 171-175.

## Summary

In this tutorial, you learned:

1. **FEP Extension**: How the Free Energy Principle extends from 1D tracking to 2D spatial navigation
2. **Sensory Encoding**: Egocentric 5×5 local view as retinotopic V1 population code
3. **Motor Decoding**: 8-directional zero-threshold spike-count decoder
4. **BSP Dungeon Generation**: Deterministic procedural environments for reproducibility
5. **Drug Effects**: In-silico Morris water maze pharmacology (diazepam impairs spatial learning)
6. **Scale Considerations**: Small networks need smaller grids; GPU scale enables 25×25 mazes

The Doom Arena demonstrates that dONNs can handle significantly more complex spatial tasks than 1D Pong, approaching the complexity of real rodent navigation experiments — but with the ability to apply reversible drugs, control every parameter, and run thousands of trials in hours rather than months.
