# Organic Neural Networks

Biologically-inspired neural networks with emergent capabilities that standard architectures cannot replicate.

## What Makes This Different

Standard neural networks (PyTorch, TensorFlow) have:
- Fixed architecture (cannot grow)
- Permanent weights (damage is irreversible)
- No self-monitoring capability
- Uniform processing dynamics

**Organic Neural Networks have:**
- **Neurogenesis**: Grow new neurons during operation
- **Damage Recovery**: Regrow and redistribute after failures
- **Emergence Measurement**: Quantifiable consciousness-like metrics
- **Multi-Tissue Processing**: Specialized regions with different dynamics

## Project Structure

```
organic_neural/
├── src/                          # Core library
│   ├── organic_neural_network.py # Main organic network class
│   ├── multi_tissue_network.py   # Connected brain regions
│   ├── quantum_consciousness.py  # IIT/GWT/Orch-OR implementations
│   └── quantum_terrarium.py      # Living quantum ecosystem
│
├── experiments/                  # Research experiments
│   ├── proven_experiments.py     # Demonstrates unique capabilities
│   ├── publishable_experiments.py # Publication-ready experiments
│   ├── neurogenesis_benchmark.py # Growth/adaptation tests
│   └── continual_learning_benchmark.py # Sequential learning tests
│
├── tests/                        # Test suite
└── docs/                         # Documentation
```

## Quick Start

```python
from src.organic_neural_network import OrganicNeuralNetwork

# Create a network that can grow
network = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=20,
    energy_supply=2.0
)

# Define input/output regions
network.define_input_region('input', position=(2, 5, 2.5), radius=2.0)
network.define_output_region('output', position=(8, 5, 2.5), radius=2.0)

# Run the network
network.set_inputs({'input': 0.5})
for _ in range(10):
    network.step(dt=0.3)
output = network.read_output('output')

# Train with dopamine-modulated plasticity
if reward > 0.5:
    network.release_dopamine(0.5)
    network.give_energy_bonus('output', 3.0)
network.apply_reward_modulated_plasticity()

# Grow when needed
network.structural_adaptation(accuracy)
```

## Key Experiments

### 1. Growth During Operation
Demonstrates that the network can grow to handle increasing task complexity, unlike standard NNs which require stopping and retraining.

### 2. Damage Recovery
Shows that the network maintains performance after losing neurons, demonstrating resilience for long-running systems.

### 3. Emergence Measurement
Quantifies emergent properties (integrated information, synchronization, criticality) that standard networks lack.

### 4. Multi-Tissue Temporal Processing
Demonstrates specialized tissues with different time constants processing different timescales simultaneously.

## Run Experiments

```bash
cd /Users/bobbyprice/projects/entropy/organic_neural
python3 experiments/proven_experiments.py
```

## Key Findings (From Latest Run)

| Capability | Standard NN | Organic NN |
|------------|-------------|------------|
| Grow during operation | Must stop & retrain | Automatic neurogenesis |
| Recover from damage | Must restore/retrain | Local adaptation |
| Measure emergence | No metrics | Quantifiable scores |
| Multi-timescale processing | Uniform dynamics | Specialized tissues |

## Applications

- **Space/Remote Systems**: Self-repairing systems for environments where maintenance is impossible
- **Autonomous Agents**: Long-running systems that adapt without cloud retraining
- **Edge Computing**: Adaptive capacity without requiring larger models
- **Safe AI**: Self-monitoring through emergence metrics

## Next Steps

1. Strengthen neurogenesis triggers for more visible growth
2. Implement hippocampus-cortex memory consolidation
3. Benchmark against continual learning methods (EWC, PackNet)
4. Publish research findings

## License

Part of the Entropy project.
