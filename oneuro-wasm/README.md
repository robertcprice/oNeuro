# oNeuro WASM

`oneuro-wasm` is the browser-facing sidecar for oNeuro. It keeps the lightweight
WASM library and the browser demos together, but the directory is organized
around two layers:

- `src/lib.rs`
  Rust/WASM library surface.
- `web/`
  Browser demos and demo documentation.

## Status

This is a working web demo surface. It is useful for:

- browser-based neural / molecular visualizations
- fast interactive prototypes
- lightweight demo sharing without the full native stack

It is not the canonical benchmark path for the 25K Pong comparison.

## Layout

- `web/index.html`
  Simple demo index.
- `web/full_integration.html`
  Main combined demo.
- `web/md_visualization.html`
  Molecular dynamics visualization.
- `web/prototypes/`
  Older or more experimental browser demos.

## Commands

From `oneuro-wasm/`:

```bash
cargo check
cd web
python3 -m http.server 8080
```

Then open `http://localhost:8080/index.html`.
