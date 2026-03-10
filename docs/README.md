# Documentation Index

oNeuro is moving quickly. This index points to the docs that are most useful right now.

## Start Here

- Project overview and current status: [`../README.md`](../README.md)
- Repository guide: [`repo_structure.md`](repo_structure.md)
- Current 25K Pong measured comparison: [`../results/pong_compare_20260310/README.md`](../results/pong_compare_20260310/README.md)
- Rust/Metal backend overview: [`../oneuro-metal/README.md`](../oneuro-metal/README.md)

## Recommended Entry Points

- Python/CUDA/MPS Pong benchmark: [`../demos/demo_dishbrain_pong.py`](../demos/demo_dishbrain_pong.py)
- Native Rust/Metal Pong runner: [`../oneuro-metal/src/bin/dishbrain_pong.rs`](../oneuro-metal/src/bin/dishbrain_pong.rs)
- CUDA/MPS backend implementation: [`../src/oneuro/molecular/cuda_backend.py`](../src/oneuro/molecular/cuda_backend.py)

## Current Engineering Tracks

- Pong latency and real-time path: [`pong_realtime_path.md`](pong_realtime_path.md)
- Whole-cell / minimal-cell strategy: [`whole_cell_strategy.md`](whole_cell_strategy.md)
- Terrarium Rust status: [`terrarium_rust_status.md`](terrarium_rust_status.md)

## Sidecar Projects

- Native 3D visualization track: [`../oneuro-3d/README.md`](../oneuro-3d/README.md)
- Browser/WASM demo surface: [`../oneuro-wasm/README.md`](../oneuro-wasm/README.md)
- Pharma/pathogen package: [`../pharma_platform/README.md`](../pharma_platform/README.md)

## Papers and Research Notes

- Beyond ANN white paper: [`../papers/beyond_ann_white_paper.md`](../papers/beyond_ann_white_paper.md)
- DishBrain replication paper draft: [`../papers/dishbrain_replication_paper.md`](../papers/dishbrain_replication_paper.md)

## Practical Notes

- Treat benchmark/result files under `results/` as the source of truth for measured numbers.
- Treat paper drafts as working documents unless a result is also backed by a checked-in artifact.
- Use `docs/repo_structure.md` if you need to orient yourself to the current top-level layout quickly.
- Expect interfaces and backend behavior to change; this is still a working project.
