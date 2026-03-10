# oNeuro Web - Browser-Based Neural Molecular Dynamics

This module provides browser-based neural-molecular simulation using WebGPU and WebGL.

## Quick Start

### Full Integration Demo (Recommended)

Open `index.html` first for the demo index, or go directly to
`full_integration.html` for the complete experience:

```bash
# Serve the files
cd oneuro-wasm/web
python3 -m http.server 8080
# Open http://localhost:8080/index.html
```

### Features

- **Neural Brain Simulation**: Hodgkin-Huxley neurons with realistic dynamics
- **Molecular Dynamics**: 8,000 particles with Lennard-Jones interactions
- **Full Integration**: Neural activity drives particle behavior
- **Real-time Visualization**: 60 FPS with Three.js WebGL

## Demos

### 1. full_integration.html
Complete neural-molecular integration demo with:
- 64 Hodgkin-Huxley neurons across 5 brain regions
- 8,000 particles with physics
- Fly agent with wing animation
- Food sources that emit odorants
- Emergent navigation behavior

### 2. md_visualization.html
Pure molecular dynamics visualization:
- 15,000 particles
- Temperature control
- Attraction to food sources
- Real-time energy monitoring

### Prototypes

Additional browser demos now live under `prototypes/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (HTML/JS)                        │
├─────────────────────────────────────────────────────────────┤
│  Three.js WebGL          │  Neural Brain (JS)              │
│  - Particle rendering     │  - HH neurons                   │
│  - Fly visualization     │  - Region dynamics              │
│  - UI controls           │  - Motor output                 │
├─────────────────────────────────────────────────────────────┤
│                   JavaScript Simulation                      │
│  - MD forces (LJ)        │  - Odorant field                │
│  - Integration           │  - Neural processing             │
└─────────────────────────────────────────────────────────────┘
```

## Controls

| Control | Description |
|---------|-------------|
| Run/Pause | Toggle simulation |
| Reset | Reset particles and brain |
| Food | Add new food source |
| Speed | Simulation speed (0.1x - 5x) |
| Temperature | System temperature (100-500K) |
| Neurons | Number of neurons (16-256) |

## Neural Processing Pipeline

1. **Olfactory Input**: Food distance → odorant concentration
2. **Antennal Lobe (AL)**: Receives odorant signal
3. **Mushroom Body (MB)**: Learning/memory (simplified)
4. **Central Complex (CX)**: Navigation/heading
5. **VNC Motor**: Speed and turn commands

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome 113+ | Full WebGPU |
| Edge 113+ | Full WebGPU |
| Firefox  nightly | WebGPU (experimental) |
| Safari 17+ | WebGL fallback |

The demo gracefully falls back to WebGL if WebGPU is unavailable.

## Files

- `index.html` - Demo index / entrypoint
- `full_integration.html` - Main demo with neural-MD integration
- `md_visualization.html` - Pure MD visualization
- `prototypes/` - Archived or experimental pages
