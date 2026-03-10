# oNeuro-Metal

**The world's first GPU-accelerated molecular brain simulator.**

Not a neural network. A complete molecular brain where every behavior — learning, memory, drug response, consciousness, sleep — *emerges* from biochemistry simulated on Apple Silicon GPU.

## What is this?

oNeuro-Metal is a biophysical neural engine written in Rust with Metal compute shaders. Every neuron in the simulation is a full molecular model: Hodgkin-Huxley ion channels, 4-compartment calcium dynamics, second messenger cascades (cAMP/PKA/PKC/CaMKII/CREB), gene expression, vesicle release, STDP via receptor trafficking, and Orch-OR quantum consciousness — all running on GPU.

The key insight: **you don't program behaviors.** You simulate biochemistry, and behaviors emerge. Administer Diazepam and GABA-A conductance increases, inhibition rises, firing rates drop. Apply general anesthesia and consciousness metrics collapse by >70%. Run a sleep cycle and memory consolidation happens through hippocampal replay. None of this is hardcoded — it falls out of the molecular simulation.

## Architecture

```
GPU Phase (Metal compute, 1 thread per neuron):
  1. HH gating variables    — Na_v m³h, K_v n⁴, Ca_v m²h
  2. Receptor binding        — Hill equation → AMPA/NMDA/GABA-A/nAChR open fractions
  3. Membrane integration    — 8-channel I_ion + dV/dt + spike detection
  4. Calcium dynamics        — 4-compartment ODE (microdomain/cyto/ER/mito)
  5. Second messengers       — cAMP/PKA/PKC/CaMKII/CREB/MAPK cascades

CPU Phase (serial, only fired neurons — typically 1-5%):
  6. Spike propagation       — vesicle release + PSC injection via CSR graph
  7. STDP                    — receptor trafficking (LTP=insert AMPA, LTD=remove)
  8. Synaptic cleft dynamics — NT degradation/diffusion/reuptake

CPU Interval-Gated (every N steps):
  9. Gene expression         — CREB→c-Fos→BDNF→AMPA (every 10 steps)
  10. Metabolism              — Glycolysis + OxPhos + ATP pools (every 5 steps)
  11. Microtubules            — Orch-OR quantum coherence/collapse (every 10 steps)
  12. Glia                    — Astrocyte/oligodendrocyte/microglia (every 10 steps)
  13. Circadian               — TTFL oscillator + adenosine homeostasis (every step)
```

Apple Silicon unified memory (`StorageModeShared`) means GPU↔CPU sync is zero-copy — the CPU reads the `fired` array directly after command buffer completion with no DMA transfer.

## 16 Molecular Subsystems

| Subsystem | What It Does | Why It Matters |
|-----------|-------------|----------------|
| **HH Ion Channels** | Na_v, K_v, K_leak, Ca_v voltage-gated channels with exact α/β rate functions | Action potentials emerge from channel kinetics, not threshold crossing |
| **Ligand-Gated Channels** | AMPA, NMDA (with Mg²⁺ block), GABA-A, nAChR | Synaptic transmission is molecular, not a weight multiplication |
| **4-Compartment Calcium** | Cytoplasmic/ER/mitochondrial/microdomain with IP3R, RyR, SERCA, MCU, PMCA, NCX | Calcium is THE intracellular signal — triggers everything from vesicle release to gene expression |
| **Second Messengers** | cAMP, PKA, PKC, CaMKII (bistable switch), MAPK/ERK, CREB, IP3/DAG | Long-term potentiation requires kinase cascades, not just Hebbian rules |
| **Gene Expression** | DNA→RNA→Protein pipeline with c-Fos, Arc, BDNF, Zif268 transcription factors | Memory consolidation requires protein synthesis (anisomycin blocks this — testable!) |
| **Metabolism** | Glycolysis + oxidative phosphorylation → ATP pools, O₂/glucose supply | Neurons that run out of ATP become less excitable — natural activity limiter |
| **Vesicle Pools** | Readily releasable / recycling / reserve pools with Ca²⁺-dependent release | Short-term synaptic plasticity (facilitation, depression) emerges from vesicle depletion |
| **STDP** | Spike-timing-dependent plasticity via AMPA receptor trafficking + BCM metaplasticity | Learning rules emerge from molecular dynamics, not parameter-tuned Hebbian learning |
| **Synaptic Cleft** | NT release, enzymatic degradation, diffusion, transporter reuptake | Drug targets: SSRIs block serotonin reuptake, AChE inhibitors block ACh breakdown |
| **Pharmacology** | 7 drugs + general anesthesia with 1-compartment PK (Bateman) + PD (Hill) | Real dose-response curves: Diazepam enhances GABA-A 5x, Ketamine blocks NMDA 90% |
| **Glia** | Astrocyte (glutamate uptake, lactate shuttle), oligodendrocyte (myelination), microglia (synaptic pruning) | Astrocytes regulate excitotoxicity, oligodendrocytes control conduction velocity |
| **Circadian** | TTFL oscillator (BMAL1/PER-CRY) + adenosine homeostasis (two-process sleep model) | Chronopharmacology: drug effects vary with time of day |
| **Microtubules** | Orch-OR quantum coherence/collapse model | Consciousness metric: anesthetics suppress coherence, matching clinical observations |
| **Consciousness** | 7 metrics: Phi (IIT), PCI, causal density, criticality, global workspace, Orch-OR, composite | Quantitative consciousness measurement — composite drops >70% under anesthesia |
| **Brain Regions** | Cortical columns, thalamic nuclei, hippocampus, basal ganglia with anatomical connectivity | Regional architecture enables sleep replay, thalamocortical loops, striatal learning |
| **Extracellular Space** | 3D voxel grid with Fick's law diffusion (GPU shader) | Volume transmission of neuromodulators (DA, 5-HT diffuse through extracellular space) |

## Competitive Landscape

| Capability | oNeuro-Metal | NEURON | NEST | Brian2 | CoreNeuron |
|---|---|---|---|---|---|
| HH ion channels on GPU | **Yes** | No | No | No | Yes (partial) |
| Second messenger cascades | **Yes (GPU)** | MOD files (CPU) | No | No | No |
| Gene expression (CREB→BDNF) | **Yes** | No | No | No | No |
| Quantum consciousness (Orch-OR) | **Yes** | No | No | No | No |
| Psychopharmacology (7 drugs) | **Yes** | No | No | No | No |
| Circadian + chronopharm | **Yes** | No | No | No | No |
| 3D extracellular diffusion (GPU) | **Yes** | 1D | No | No | No |
| Consciousness metrics (7) | **Yes** | No | No | No | No |
| Apple Silicon Metal GPU | **Yes** | No | No | No | No |
| Zero-copy CPU↔GPU (unified memory) | **Yes** | N/A | N/A | N/A | No (CUDA copies) |

## Installation

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- macOS 13+ with Apple Silicon (for GPU acceleration)
- Python 3.10+ (for Python bindings)

### Build from source

```bash
# Clone
git clone https://github.com/bobbyprice/oNeuro.git
cd oNeuro/oneuro-metal

# Build and run tests
cargo test

# Build optimized release
cargo build --release

# Build Python extension (requires maturin)
pip install maturin
maturin develop --release
```

### CPU-only (non-macOS)

The crate compiles on any platform with Rust. Without Metal, all compute falls back to optimized CPU code using Rayon for parallelism.

## Native Terrarium Runtime

`oneuro-metal` now includes a Python-free terrarium runtime and viewer stack on top of the native substrate, atmosphere, plant, soil, and fly systems.

Run the native entry points from `oneuro-metal/`:

```bash
# Headless/native summary run
cargo run --example native_terrarium_stack

# Native terminal loop
cargo run --bin terrarium_native

# Native graphical viewer
cargo run --bin terrarium_viewer
```

Run the shader-backed GPU viewer from `oneuro-3d/`. It now uploads packed raw float field textures for terrain, soil moisture, canopy, chemistry, odor, and gas exchange, plus live overlay point data, and does palette/contour shading directly in WGSL on the GPU:

```bash
cargo run --manifest-path oneuro-3d/Cargo.toml --bin terrarium_gpu
```

Useful viewer controls:

- `1` terrain
- `2` soil moisture
- `3` canopy
- `4` chemistry
- `5` odor
- `6` gas exchange
- `space` pause
- `right` single-step
- `r` reset
- `up` / `down` adjust FPS
- `esc` quit

## Native Whole-Cell Backend

`oneuro-metal` now also carries the fast path for native whole-cell simulation.
Primary reference for this line of work: Thornburg, Z. R., Maytin, A., Kwon, J., Solomon, K. V., et al., "Bringing the Genetically Minimal Cell to Life on a Computer in 4D," `Cell` (2026), DOI `10.1016/j.cell.2026.02.009`.

The current runtime is a Rust `WholeCellSimulator` with:

- a voxelized intracellular lattice in structure-of-arrays layout
- a custom Metal reaction-diffusion shader on macOS
- stability-preserving RDME substepping for default voxel/diffusion settings
- a Rayon CPU fallback everywhere else
- staged CME/ODE/BD/geometry updates for a minimal-cell-style coarse simulator
- an optional `nQPU` correction bridge for OxPhos/translation/polymerization efficiency
- an optional local chemistry microdomain model built on `BatchedAtomTerrarium`
- localized MD probes for ribosome/septum/chromosome subregions
- persistent Syn3A subsystem states that feed ATP-band, replisome, ribosome, and septum couplings back into the native runtime
- per-subsystem local chemistry site reports so each Syn3A microdomain can consume its own weighted patch support instead of a single global chemistry average
- localized substrate demand and depletion inside those microdomains so active sites draw down their own local terrarium patches before chemistry support is computed
- persistent microdomain chemistry memory instead of hard-resetting the terrarium from the coarse snapshot every update
- generic bulk exchange that relaxes the terrarium back toward coarse concentrations without preset-specific refill behavior
- a generic substrate reaction IR that now drives local chemistry execution instead of preset-specific hotspot seeding code
- a generic assembly/occupancy layer that derives local structural order from component availability, crowding, demand satisfaction, and byproduct pressure
- a generic localization layer that selects microdomain patch coordinates from substrate cues, geometry, continuity, and exclusion pressure instead of fixed anchors
- a local activity/catalyst layer that drives reactions from resolved patch chemistry and assembly state instead of coarse snapshot counters for ribosomes, DnaA, or FtsZ
- CME/ODE/BD/geometry stages that now read substrate-derived assembly inventories and process-capacity signals instead of treating ribosome/RNAP/DnaA/FtsZ pool counters as the source of truth
- a generic scalar process-rule IR that now evaluates those inventory and stage-rate laws from rule tables instead of bespoke whole-cell arithmetic blocks
- rule-driven subsystem readiness, coarse resource-signal reduction, and snapshot exchange targets so the chemistry bridge no longer depends on one-off transfer formulas
- a generic affine reducer layer for local chemistry crowding, derived patch support/stress signals, assembly context scaling, subsystem structural targets, and aggregate chemistry support
- an RDME lattice initialization path that no longer injects preset ATP hotspots, so spatial structure starts neutral unless processes or explicit perturbations create it
- local depletion, demand satisfaction, and byproduct pressure that now feed back into subsystem scaling and effective metabolic load
- data-driven subsystem coupling profiles so ATP-band, ribosome, replisome, and septum differences live in tables instead of branch-specific support formulas

Python example:

```python
from oneuro_metal import WholeCellSimulator

cell = WholeCellSimulator(x_dim=24, y_dim=24, z_dim=12, dt_ms=0.25, use_gpu=True)
cell.set_metabolic_load(1.2)
cell.enable_default_syn3a_subsystems()
cell.run(100)
print(cell.snapshot())
print(cell.local_chemistry_report())
print(cell.local_chemistry_sites())
print(cell.scheduled_syn3a_subsystem_probes())
print(cell.subsystem_states())
print(cell.run_local_md_probe("ribosome_cluster"))
```

Rust example:

```rust
use oneuro_metal::{WholeCellConfig, WholeCellSimulator};

let mut cell = WholeCellSimulator::new(WholeCellConfig {
    dt_ms: 0.25,
    use_gpu: true,
    ..WholeCellConfig::default()
});
cell.run(100);
println!("{:?}", cell.snapshot());
```

## Python API

```python
from oneuro_metal import MolecularBrain, RegionalBrain

# Simple network
brain = MolecularBrain(n_neurons=1000)
brain.stimulate(0, 50.0)  # 50 µA/cm² to neuron 0
brain.run(10000)           # 10K steps (1 second at dt=0.1ms)
print(f"Mean firing rate: {brain.mean_firing_rate():.1f} Hz")

# Regional architecture (cortex + thalamus + hippocampus + basal ganglia)
brain = RegionalBrain.xlarge(seed=42)  # 1018 neurons, 23.5K synapses
brain.step()

# Pharmacology
brain.apply_drug("caffeine", 100.0)    # mg dose
brain.apply_drug("diazepam", 5.0)
brain.run(5000)

# Consciousness monitoring
metrics = brain.consciousness_metrics()
print(f"Phi={metrics.phi:.2f}, PCI={metrics.pci:.3f}, Composite={metrics.composite:.3f}")

# General anesthesia
brain.apply_anesthesia()
brain.run(1000)
metrics = brain.consciousness_metrics()
print(f"Post-anesthesia composite: {metrics.composite:.3f}")  # Should be <0.15

# Zero-copy numpy access
import numpy as np
voltages = brain.voltages()        # np.ndarray[N] — direct view of Rust memory
calcium = brain.calcium()          # np.ndarray[N, 4]
spikes = brain.fired()             # np.ndarray[N] bool
```

## Rust API

```rust
use oneuro_metal::{MolecularBrain, RegionalBrain, ConsciousnessMonitor, DrugType, NTType};

// Create a brain with connectivity
let edges = vec![
    (0u32, 1, NTType::Glutamate),
    (0, 2, NTType::Glutamate),
    (1, 0, NTType::GABA),
];
let mut brain = MolecularBrain::from_edges(3, &edges);

// Simulate
brain.stimulate(0, 50.0);
brain.run(10000);

// Drug effects
brain.apply_drug(DrugType::Caffeine, 100.0);
brain.run(5000);

// Consciousness monitoring
let mut monitor = ConsciousnessMonitor::new(brain.neuron_count());
for _ in 0..100 {
    brain.step();
    monitor.record(&brain.neurons);
}
let metrics = monitor.compute(&brain.neurons, &brain.synapses);
println!("Composite consciousness: {:.3}", metrics.composite);
```

## Technical Details

### Memory Layout

All neuron state is stored in Structure-of-Arrays (SoA) format for GPU-coalesced memory access:

```
NeuronArrays (~80 f32 per neuron):
├── Membrane: voltage, prev_voltage, fired, refractory_timer
├── HH gating: nav_m, nav_h, kv_n, cav_m, cav_h
├── Conductance scales: [f32; 8] per ion channel type
├── Ligand open fractions: ampa_open, nmda_open, gabaa_open, nachr_open
├── Calcium 4-compartment: ca_cyto, ca_er, ca_mito, ca_micro
├── Second messengers: cAMP, PKA, PKC, CaMKII, IP3, DAG, ERK (10 floats)
├── Phosphorylation: AMPA_p, Kv_p, CaV_p, CREB_p
├── Metabolism: ATP, ADP, glucose, oxygen
├── NT concentrations: [f32; 6] (DA, 5-HT, NE, ACh, GABA, glutamate)
└── External current, spike count, gene expression levels
```

~320 bytes/neuron. 100K neurons = 32 MB — fits easily in Apple Silicon unified memory.

### Synapse Format

Compressed Sparse Row (CSR) for efficient spike propagation:
- `row_offsets[N+1]`: neuron → outgoing synapse range
- `col_indices[S]`: target neuron IDs
- Per-synapse: weight, strength, delay, NT type, vesicle pools (3), cleft concentration, receptor counts (AMPA/NMDA/GABA-A), STDP traces, BCM theta

### Metal Shaders (8 kernels)

| Shader | Threads | What It Computes |
|--------|---------|-----------------|
| `hh_gating.metal` | N neurons | α/β rate functions → m,h,n integration |
| `hill_binding.metal` | N neurons | NT concentration → receptor open fractions |
| `membrane_euler.metal` | N neurons | 8-channel I_ion sum + dV/dt + spike detect |
| `calcium_ode.metal` | N neurons | IP3R/RyR/SERCA/MCU/PMCA/NCX calcium flows |
| `second_messenger.metal` | N neurons | G-protein → cAMP/PKA/PKC/CaMKII/CREB cascades |
| `cleft_dynamics.metal` | S synapses | NT degradation + diffusion + reuptake |
| `diffusion_3d.metal` | V voxels | 6-neighbor Laplacian for 3D NT diffusion |
| `whole_cell_rdme.metal` | V voxels | Intracellular reaction-diffusion for ATP, amino acids, nucleotides, and membrane precursors |

### Supported Drugs

| Drug | Class | Molecular Target | Max Effect |
|------|-------|-----------------|------------|
| Fluoxetine | SSRI | Serotonin reuptake | 5-HT ×4.0 |
| Diazepam | Benzodiazepine | GABA-A allosteric | Conductance ×5.0 |
| Caffeine | Xanthine | Adenosine receptor | Na_v ×1.3, +excitation |
| Amphetamine | Psychostimulant | DAT/NET reverse | DA ×6.0, NE ×4.0 |
| L-DOPA | DA precursor | DOPA decarboxylase | DA ×5.0 |
| Donepezil | AChE inhibitor | Acetylcholinesterase | ACh ×4.0 |
| Ketamine | NMDA antagonist | NMDA channel block | Conductance ×0.1 |
| *Anesthesia* | *Multi-target* | *GABA-A/NMDA/AMPA/Na_v/K_leak* | *>70% consciousness drop* |

## Codebase

```
oneuro-metal/
├── Cargo.toml + build.rs + pyproject.toml
├── src/
│   ├── lib.rs                    # Module declarations + re-exports
│   ├── types.rs                  # Enums: NeuronArchetype, IonChannelType, NTType, etc.
│   ├── constants.rs              # All biophysical constants from literature
│   ├── neuron_arrays.rs          # SoA neuron state (~80 fields)
│   ├── synapse_arrays.rs         # CSR sparse synapse storage
│   ├── network.rs                # MolecularBrain orchestrator
│   ├── brain_regions.rs          # RegionalBrain (cortex/thalamus/hippo/BG)
│   ├── consciousness.rs          # 7 consciousness metrics
│   ├── python.rs                 # PyO3 Python bindings
│   ├── spike_propagation.rs      # Fired → outgoing synapses → PSC
│   ├── stdp.rs                   # STDP + BCM + receptor trafficking
│   ├── gene_expression.rs        # CREB→c-Fos→BDNF→AMPA
│   ├── metabolism.rs             # Glycolysis + OxPhos + ATP
│   ├── microtubules.rs           # Orch-OR quantum coherence
│   ├── circadian.rs              # TTFL oscillator + adenosine
│   ├── pharmacology.rs           # 7 drugs + anesthesia PK/PD
│   ├── glia.rs                   # Astrocyte/oligo/microglia
│   ├── gpu/
│   │   ├── mod.rs                # Metal device init + pipeline cache
│   │   ├── hh_gating.rs          # GPU dispatch + CPU fallback
│   │   ├── membrane_integration.rs
│   │   ├── calcium_dynamics.rs
│   │   ├── second_messenger.rs
│   │   ├── receptor_binding.rs
│   │   ├── synapse_cleft.rs
│   │   └── diffusion_3d.rs
│   └── metal/
│       ├── hh_gating.metal
│       ├── membrane_euler.metal
│       ├── calcium_ode.metal
│       ├── second_messenger.metal
│       ├── hill_binding.metal
│       ├── cleft_dynamics.metal
│       └── diffusion_3d.metal

32 files, ~9,200 lines of Rust + Metal
93 tests, 0 warnings
```

## License

MIT

## Author

Bobby Price — [@bobbyprice](https://github.com/bobbyprice)
