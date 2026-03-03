# oNeuro

<div align="center">

![oNeuro Logo](docs/assets/logo.png)

**Biologically-inspired neural networks with neurogenesis, quantum effects, and reward-modulated learning.**

</div>

A novel fusion of four paradigms:
- **Liquid Neural Networks** - Continuous-time dynamics with ODE-based state evolution
- **Neural Cellular Automata** - Self-organization and emergent computation
- **Digital Evolution** - Neurogenesis (birth), pruning (death), and natural selection
- **Quantum Effects** - Superposition states, entanglement, and quantum coherence

This is the **first implementation** combining all four paradigms into a unified neural architecture.

## Installation

```bash
pip install oNeuro

# With visualization support
pip install oNeuro[viz]
```

## Quick Start

```python
from organic_neural_network import OrganicNeuralNetwork, XORTask

# Create neural tissue (3D space)
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),    # 3D dimensions
    initial_neurons=30,         # Starting neurons
    energy_supply=2.0           # Metabolic energy
)

# Define input/output regions (3D zones)
tissue.define_input_region("input_a", (2.5, 5.0, 2.5), radius=1.5)
tissue.define_input_region("input_b", (7.5, 5.0, 2.5), radius=1.5)
tissue.define_output_region("output", (5.0, 5.0, 2.5), radius=1.5)

# Create and train on XOR task
task = XORTask(tissue)
stats = tissue.train_task(task, n_episodes=100)

# Evaluate
result = tissue.evaluate_task(task, n_trials=25)
print(f"Success rate: {result['success_rate']:.1%}")
```

## Architecture

### Core Concepts

**1. Neural Tissue**
- Neurons exist in continuous 3D space (not layers)
- Connections form based on spatial proximity and Hebbian learning
- Metabolic energy system constrains growth

**2. Neurogenesis**
- New neurons can spawn from existing ones (mitosis)
- Stem cells differentiate based on local activity patterns
- Network literally grows its own architecture

**3. Synaptic Plasticity**
- Hebbian learning: "Neurons that fire together wire together"
- Eligibility traces record co-activity patterns
- Dopamine modulates consolidation of traces

**4. Quantum Effects**
- Neurons can enter superposition states
- Entangled pairs share correlated behavior
- Quantum coherence affects computation

### Training System

```python
# Reward-modulated plasticity
tissue.release_dopamine(1.0)  # Global reward signal
tissue.apply_reward_modulated_plasticity()  # Consolidate learning

# Structural adaptation based on performance
tissue.structural_adaptation(performance=0.7)  # Grow if > 0.7, prune if < 0.3
```

## Built-in Tasks

| Task | Description | Tests |
|------|-------------|-------|
| `XORTask` | Classic XOR problem | Non-linear computation |
| `PatternRecognitionTask` | 2x2 pattern classification | Feature detection |
| `MemoryTask` | Delayed pattern recall | Working memory |
| `DecisionMakingTask` | Evidence accumulation | Temporal integration |

## Training Results

| Task | Before Training | After Training | Improvement |
|------|-----------------|----------------|-------------|
| XOR | 46.7% | 56.0% | +9.3% |
| Pattern Recognition | 6.7% | 40.0% | +33.3% |
| Memory | 46.7% | 36.0% | -10.7% |
| Decision Making | 53.3% | 52.0% | -1.3% |

## Multi-Tissue Networks

For brain-like architectures with specialized regions:

```python
from multi_tissue_network import (
    MultiTissueNetwork,
    CorticalTissue,
    HippocampalTissue,
    ThalamicTissue
)

brain = MultiTissueNetwork()

# Add specialized tissues
brain.add_tissue("cortex", CorticalTissue(size=(20, 20, 5)))
brain.add_tissue("hippocampus", HippocampalTissue(size=(10, 10, 3)))
brain.add_tissue("thalamus", ThalamicTissue(size=(5, 5, 5)))

# Connect regions
brain.connect("thalamus", "cortex", bidirectional=True)
brain.connect("cortex", "hippocampus", bidirectional=True)
brain.connect("hippocampus", "thalamus")

# Run unified simulation
brain.step()
```

## Quantum Consciousness Integration

The framework includes a quantum consciousness module implementing:

- **IIT (Integrated Information Theory)** - Phi calculation
- **GWT (Global Workspace Theory)** - Attention and broadcasting
- **Orch-OR (Orchestrated Objective Reduction)** - Penrose-Hameroff quantum consciousness
- **Self-Model** - Mirror self-recognition tests

```python
from quantum_consciousness import ConsciousnessSystem

# Create consciousness system
consciousness = ConsciousnessSystem(num_units=8, num_tubulins=8)

# Run simulation
for _ in range(100):
    metrics = consciousness.step()
    print(f"Phi={metrics.phi:.3f}, Coherence={metrics.coherence:.3f}")

# Test self-awareness
consciousness.self_model.mirror_test(reflection_data, is_self=True)
```

## Key Differences from Traditional Neural Networks

| Feature | Traditional NN | Organic Neural |
|---------|---------------|----------------|
| Architecture | Fixed layers | Dynamic 3D tissue |
| Learning | Backpropagation | Hebbian + dopamine |
| Growth | Pre-defined | Neurogenesis |
| Topology | Layer-wise | Spatial/proximity |
| Dynamics | Discrete steps | Continuous ODEs |
| Quantum | No | Superposition + entanglement |

## API Reference

### OrganicNeuralNetwork

```python
class OrganicNeuralNetwork:
    def __init__(self, size, initial_neurons=50, energy_supply=1.0): ...

    # Regions
    def define_input_region(self, name, position, radius): ...
    def define_output_region(self, name, position, radius): ...
    def set_input(self, name, value): ...
    def read_output(self, name) -> float: ...

    # Training
    def train_task(self, task, n_episodes=100) -> dict: ...
    def evaluate_task(self, task, n_trials=25) -> dict: ...
    def release_dopamine(self, amount): ...

    # Structure
    def structural_adaptation(self, performance): ...
    def grow_neurons_in_region(self, region): ...
    def prune_weak_connections(self, threshold=0.1): ...

    # Simulation
    def step(self, dt=0.001): ...
    def run(self, duration_ms=10.0): ...
```

## Project Structure

```
organic_neural/
├── src/
│   ├── organic_neural_network.py   # Core neural tissue
│   ├── multi_tissue_network.py     # Brain-like architectures
│   └── quantum_consciousness.py    # IIT/GWT/Orch-OR
├── experiments/
│   └── proven_experiments.py       # Demonstrations
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## Citation

If you use this in research, please cite:

```bibtex
@software{organic_neural_2025,
  title = {Organic Neural Network: Biologically-Inspired Neural Tissue},
  author = {Entropy Research},
  year = {2025},
  url = {https://github.com/entropy-research/organic-neural}
}
```

## License

MIT License - See [LICENSE](LICENSE)

## Contributing

Contributions welcome! Areas of interest:
- New training tasks
- Visualization tools
- Hardware acceleration (GPU/NPU)
- Integration with other frameworks (PyTorch, JAX)
