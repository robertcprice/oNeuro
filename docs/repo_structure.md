# Repository Guide

This repository is an active working research project. The top level is broader
than the current benchmark surface, so this guide separates the canonical paths
from exploratory tracks.

## Start Here

If you want the current measured numbers and the least ambiguous entrypoints,
use these first:

- [`../README.md`](../README.md): project overview and current status
- [`../results/pong_compare_20260310/README.md`](../results/pong_compare_20260310/README.md): current measured 25K Pong comparison across Mac, A100, and H200
- [`../demos/demo_dishbrain_pong.py`](../demos/demo_dishbrain_pong.py): canonical Python/CUDA DishBrain Pong benchmark runner
- [`../src/oneuro/molecular/cuda_backend.py`](../src/oneuro/molecular/cuda_backend.py): main PyTorch/CUDA/MPS backend for the benchmark track
- [`../oneuro-metal/README.md`](../oneuro-metal/README.md): native Rust/Metal backend overview

## Canonical Top-Level Areas

- `src/oneuro/`
  The main Python package. The current benchmark-critical code lives under `src/oneuro/molecular/`.
- `demos/`
  Main experiment entrypoints. Not every demo is equally mature; use [`../demos/README.md`](../demos/README.md) for the current map.
- `oneuro-metal/`
  Native Rust/Metal backend and native Pong / whole-cell work. This is the current Apple-native performance track.
- `results/`
  Checked-in measured artifacts. When README text and result artifacts disagree, treat `results/` as the source of truth.
- `docs/`
  High-level index, repo guide, benchmark notes, and engineering status docs.
- `papers/`
  Draft papers and supporting figures/data. These are useful context, but some claims lag behind the live benchmark artifacts.
- `scripts/`
  Benchmark helpers, analyzers, profiling scripts, telemetry capture, and cloud deployment helpers.

## Sidecar Subprojects

- `oneuro-3d/`
  Native Bevy-based 3D visualization track. Use this for fly-world and terrarium rendering, not for the canonical benchmark path.
- `oneuro-wasm/`
  Browser/WASM demo surface. Use this for lightweight web demos and visualization prototypes.
- `pharma_platform/`
  Python sidecar for pathogen/drug modeling experiments. This is organized as its own package with `drugs/`, `pathogens/`, `tests/`, `experiments/`, `benchmarks/`, and `docs/`.

## Generated or Artifact-Heavy Areas

- `results/`
  Measured JSON, logs, videos, and derived summaries.
- `doom_videos/`
  Gameplay artifacts and videos.
- `tmp_demo_artifacts/`, `tmp_demo_bundle/`, `tmp_demo_mc4d/`
  Generated demo outputs and temporary bundles.
- top-level `*.png`, `*.onnx`
  Figures and exported model artifacts checked into the working tree.

## Practical Rules

- Use `results/` for measured performance claims.
- Use `README.md` and `docs/README.md` for the current project map.
- Treat demos and papers as working surfaces unless backed by a result artifact.
- Expect APIs, CLI flags, and recommended entrypoints to change; this is not a frozen release.
