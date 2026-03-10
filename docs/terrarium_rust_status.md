# Terrarium Rust Status

## Answer

Yes, the terrarium can become a pure Rust program. The current blocker is no longer missing low-level kernels, and it is no longer the lack of a native world owner either. The remaining blocker is feature-parity ownership: Python still owns the richer demo/runtime shell and some higher-level integration surfaces.

## Native Today

- `oneuro-metal/src/terrarium.rs`
  - Batched atom substrate with CPU and Metal backends.
- `oneuro-metal/src/molecular_atmosphere.rs`
  - Molecular world stepping for odorants, temperature, humidity, source updates, source emission, and wind perturbation.
- `oneuro-metal/src/soil_broad.rs`
  - Broad soil turnover and hydrology update.
- `oneuro-metal/src/soil_uptake.rs`
  - Root-zone resource extraction.
- `oneuro-metal/src/plant_cellular.rs`
  - Native plant cellular state stepping.
- `oneuro-metal/src/cellular_metabolism.rs`
  - Native plant cell metabolism.
- `oneuro-metal/src/plant_organism.rs`
  - Native whole-plant physiology.
- `oneuro-metal/src/ecology_fields.rs`
  - Native canopy and root field rebuilds.
- `oneuro-metal/src/ecology_events.rs`
  - Native food and seed event stepping.
- `oneuro-metal/src/terrarium_field.rs`
  - Native sensory lattice for fly/world coupling.
- `oneuro-metal/src/drosophila.rs`
  - Native fly brain/body step for terrarium sensory input.
- `oneuro-metal/src/terrarium_world.rs`
  - Native terrarium owner for broad soil pools, substrate control sync, plant stepping, seed/food bookkeeping, atmosphere stepping, plant/atmosphere gas exchange, and fly stepping.
- `oneuro-metal/examples/native_terrarium_stack.rs`
  - Headless Rust-only example running the native terrarium owner without Python.
- `oneuro-metal/src/bin/terrarium_native.rs`
  - Rust-native terminal loop for stepping and viewing the terrarium without Python.
- `oneuro-metal/src/bin/terrarium_viewer.rs`
  - Rust-native graphical viewer with a live window, top-down field rendering, keyboard controls, and an in-window stats panel driven directly by `TerrariumWorld`.
- `oneuro-3d/src/bin/terrarium_gpu.rs`
  - Rust-native GPU viewer built on Bevy/wgpu that renders the terrarium through the platform graphics backend while driving the same native `TerrariumWorld`, now with packed raw field textures for terrain/soil/canopy/chemistry/odor/gas layers plus live overlay point data for water, plants, fruit, and flies, all colored on-GPU in WGSL.

## Still Python-Owned

- `src/oneuro/worlds/molecular_world.py`
  - Still owns the Python world object used by the current Python demo path.
- `src/oneuro/ecology/terrarium.py`
  - Still owns the richer Python ecology surface used by the current Python demo path.
- `src/oneuro/organisms/rust_drosophila.py`
  - Still acts as the adapter from the Python world into the native fly sim.
- `demos/demo_actual_molecular_terrarium.py`
  - Still owns the richer graphical demo shell and rendering loop.

## What “Pure Rust” Actually Means Here

To remove Python entirely, the project needs a native terrarium owner, not just native kernels:

1. Keep extending the Rust `TerrariumWorld` so it fully covers the behaviors the Python demo still expects.
2. Move the remaining orchestration in `src/oneuro/ecology/terrarium.py` and the current Python demo loop onto that Rust type.
3. Add Rust-native app entry points:
   - headless CLI
   - terminal viewer
   - graphical viewer
4. Treat PyO3 bindings as optional adapters instead of the main runtime path.

## Practical Next Step

The clean path is:

1. Keep the Python demo as a thin viewer while the Rust `TerrariumWorld` becomes authoritative.
2. Keep `terrarium_native` as the Python-free terminal/runtime path while feature parity improves.
3. Keep `terrarium_viewer` as the lightweight native graphical runtime path for inspecting the world without the Python demo shell.
4. Keep `terrarium_gpu` as the current GPU-backed viewer path when a richer render surface is needed.
5. Deepen the GPU render path further with more simulation-aware shader passes once the current visual contract is stable.

That is how the project gets to “no Python at all” honestly, without pretending the current Python shell is gone when it is not.
