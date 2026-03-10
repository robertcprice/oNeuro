# oNeuro 3D

`oneuro-3d` is the native 3D visualization track for oNeuro. This directory is
smaller and more experimental than the main Python and Rust/Metal benchmark
paths, so the goal here is clarity:

- `src/bin/fly_world.rs`
  Fly-world scene entrypoint.
- `src/bin/terrarium_gpu.rs`
  Terrarium-focused GPU viewer entrypoint.
- `src/shaders/`
  Custom WGSL shaders used by the Bevy renderer.

## Status

This is a working subproject, not a stable public API. It is intended for:

- native 3D world rendering
- interactive scene inspection
- visual debugging of the terrarium/fly stack

## Commands

From `oneuro-3d/`:

```bash
cargo check
cargo run --bin fly_world
cargo run --bin terrarium_gpu
```

## Relationship To The Main Repo

This crate is a visualization/runtime sidecar. The benchmark-critical neural
backend still lives in:

- `src/oneuro/`
- `oneuro-metal/`

Use this directory when you want native 3D rendering, not when you want the
canonical 25K Pong benchmark path.
