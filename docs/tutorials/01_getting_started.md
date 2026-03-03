# Tutorial 1: Getting Started with Organic Neural Networks

## Overview

Organic Neural Networks represent a paradigm shift from traditional artificial neural networks. Instead of fixed layers and backpropagation, this framework implements neural **tissue** - a living, growing, evolving substrate for computation that mirrors biological brain development.

This tutorial will guide you through:
- Installing the framework
- Understanding core concepts
- Running your first simulation
- Observing emergent behavior

## Installation

### Basic Installation

```bash
pip install organic-neural
```

### With Visualization Support

```bash
pip install organic-neural[viz]
```

### From Source

```bash
git clone https://github.com/entropy-research/organic-neural
cd organic-neural
pip install -e .
```

### Requirements

- Python 3.11 or higher
- NumPy 1.24.0 or higher
- (Optional) Matplotlib 3.7.0+ for visualization

## Core Concepts

### What Makes This Different?

Traditional neural networks have:
- Fixed architecture (layers, neurons per layer)
- Backpropagation learning
- Discrete time steps
- Deterministic weights

**Organic Neural Networks** have:
- Dynamic 3D spatial architecture
- Hebbian + reward-modulated learning
- Continuous-time (liquid) dynamics
- Quantum superposition and entanglement
- Neurogenesis (birth) and apoptosis (death)
- Synaptic pruning

### Key Abstractions

#### 1. Neural Tissue

The fundamental unit is not a "network" but **tissue** - a 3D space where neurons exist at continuous positions:

```python
from organic_neural_network import OrganicNeuralNetwork

# Create a 10x10x5 unit volume of neural tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),  # 3D dimensions
    initial_neurons=30,       # Starting population
    energy_supply=2.0         # Metabolic energy rate
)
```

#### 2. Liquid Neurons

Each neuron uses continuous-time dynamics based on membrane potential:

```
tau * dV/dt = -(V - V_rest) + R * I_input
```

Where:
- `tau` = time constant (10ms default)
- `V` = membrane potential
- `V_rest` = resting potential (-70mV)
- `R` = membrane resistance
- `I_input` = input current

This is a simplified Hodgkin-Huxley model allowing neurons to exhibit biologically realistic dynamics.

#### 3. Neurogenesis

Neurons can divide (mitosis) when conditions are favorable:

```python
# Conditions for neurogenesis:
# - energy > 200 (well-fed)
# - age > 500ms (mature)
# - outputs >= 2 (integrated)
# - 1% random chance per check
```

#### 4. Quantum Effects

Neurons can enter quantum states:
- **Superposition**: Exploring multiple activation patterns simultaneously
- **Entanglement**: Correlated behavior between paired neurons

#### 5. Metabolic Cost

Neurons consume energy and die if starved:
- Base consumption: 0.1 units/ms
- Active neurons: 3x consumption
- More connections = more cost

This creates natural selection pressure - only efficient, useful neurons survive.

## Your First Simulation

### Basic Simulation Loop

```python
from organic_neural_network import OrganicNeuralNetwork

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30,
    energy_supply=2.0
)

# Run for 100ms of simulated time
for _ in range(1000):
    tissue.step(dt=0.1)  # 0.1ms time step

# Check results
print(tissue.statistics())
```

### Adding Stimulation

To input data, stimulate neurons at specific 3D positions:

```python
# Stimulate neurons near position (2, 5, 2.5)
tissue.stimulate(
    position=(2.0, 5.0, 2.5),
    intensity=10.0,  # Current injection
    radius=2.0       # Affect neurons within 2 units
)
```

### Reading Output

Read activity from regions of the tissue:

```python
# Read average activity near position (8, 5, 2.5)
activity = tissue.read_activity(
    position=(8.0, 5.0, 2.5),
    radius=2.0
)
print(f"Output activity: {activity:.3f}")  # Value in [0, 1]
```

### Complete Example

```python
from organic_neural_network import OrganicNeuralNetwork, EmergenceTracker
import time

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30,
    energy_supply=2.0
)

# Track emergent behavior
tracker = EmergenceTracker(tissue)

print("Starting simulation...")
print(f"Initial: {len(tissue.neurons)} neurons, {len(tissue.synapses)} synapses")

# Run for 200ms
for i in range(400):
    tissue.step(dt=0.5)

    # Apply periodic stimulation
    if i % 20 == 0:
        tissue.stimulate(
            position=(2.0, 5.0, 2.5),
            intensity=15.0,
            radius=3.0
        )

    # Check for emergent patterns
    if i % 40 == 0:
        emergence = tracker.detect_emergence()
        print(f"\nTime: {tissue.time:.1f}ms")
        print(f"  Neurons: {len([n for n in tissue.neurons.values() if n.alive])}")
        print(f"  Patterns: {emergence['patterns_detected']:.2f}")
        print(f"  Quantum coherence: {emergence['quantum_coherence']:.1%}")

print("\nSimulation complete!")
print(tissue.statistics())
```

## Expected Output

When you run the simulation, you will see something like:

```
Starting simulation...
Initial: 30 neurons, 85 synapses

Time: 20.0ms
  Neurons: 28
  Patterns: 0.00
  Quantum coherence: 0.0%

Time: 40.0ms
  Neurons: 31
  Patterns: 0.15
  Quantum coherence: 3.2%

...

Simulation complete!
+-----------------------------------------------------------------------------+
| ORGANIC NEURAL NETWORK - Neural Tissue Statistics                          |
+-----------------------------------------------------------------------------+
| Time:    200.0 ms    Generation:   2    Age:  198.5 ms                     |
|                                                                             |
| Neurons:   35 alive    Synapses:  102    Avg Connections: 5.8              |
| Active:      4 (11.4%)    Quantum:   2    Entangled:  1                    |
|                                                                             |
| Avg Energy:   85.3    Avg Weight: 0.512    Avg Age:  142.3 ms              |
|                                                                             |
| Events: Neurogenesis=5  Pruning=12  Entanglement=3                          |
+-----------------------------------------------------------------------------+
```

## Visualizing the Tissue

### ASCII Visualization

The framework includes built-in ASCII visualization for quick inspection:

```python
print(tissue.visualize_ascii())
```

Output shows a 2D slice of the 3D tissue:
- `@` = firing neuron (red)
- `?` = quantum superposition (magenta)
- `&` = entangled neuron (cyan)
- `o` = normal neuron (green)
- `.` = dying neuron (gray)

```
+----------------------------------------------------------+
|                 o     o        .                         |
|        o                  @          o                   |
|              o        &        o                         |
|   o                o              o          o           |
|        ?        o        o                 o             |
|                o    o         o        o                 |
|        o              o     o          o                 |
+----------------------------------------------------------+
```

### Using Matplotlib (Optional)

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Collect neuron positions
positions = []
colors = []
for neuron in tissue.neurons.values():
    if neuron.alive:
        positions.append([neuron.x, neuron.y, neuron.z])
        # Color by state
        if neuron.state.name == 'ACTIVE':
            colors.append('red')
        elif neuron.state.name == 'SUPERPOSITION':
            colors.append('purple')
        elif neuron.state.name == 'ENTANGLED':
            colors.append('cyan')
        else:
            colors.append('green')

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs = zip(*positions)
ax.scatter(xs, ys, zs, c=colors, s=50, alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Neural Tissue Structure')
plt.savefig('tissue_visualization.png')
```

## Understanding the Statistics

The `statistics()` method provides key metrics:

| Metric | Meaning | Typical Range |
|--------|---------|---------------|
| Neurons | Living neurons | 20-100 |
| Synapses | Connections | 50-300 |
| Active | Currently firing | 10-30% |
| Quantum | In superposition/entangled | 0-10% |
| Avg Energy | Metabolic reserve | 50-150 |
| Generation | Highest cell division generation | 0-5 |
| Neurogenesis events | Birth events | 0-20 |
| Pruning events | Connection removals | 0-50 |

## Common Patterns

### Stimulus-Response Loop

```python
for _ in range(100):
    # Input
    tissue.stimulate(input_position, intensity=signal_value * 10)

    # Process
    tissue.step(dt=1.0)

    # Output
    output = tissue.read_activity(output_position)
    print(f"Response: {output:.3f}")
```

### Growth Period

```python
# Let the network self-organize without input
for _ in range(500):
    tissue.step(dt=0.5)

# Now it has developed structure
print(f"Self-organized into {len(tissue.synapses)} connections")
```

### Activity-Dependent Observation

```python
# Watch for population bursts
while True:
    active_count = sum(1 for n in tissue.neurons.values()
                       if n.alive and n.state.name == 'ACTIVE')

    if active_count > len(tissue.neurons) * 0.3:
        print(f"Population burst: {active_count} neurons active!")

    tissue.step(dt=0.1)
```

## Troubleshooting

### All Neurons Died

If energy supply is too low, neurons starve:

```python
# Increase energy supply
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30,
    energy_supply=5.0  # Higher value
)
```

### No Activity

If stimulation is too weak, neurons never reach threshold:

```python
# Increase stimulation intensity
tissue.stimulate(position, intensity=20.0, radius=3.0)
```

### Too Many/Few Connections

Adjust initial connectivity during seeding:

```python
# In _seed_network(), modify n_connections:
n_connections = np.random.randint(3, 8)  # More connections
```

## Next Steps

Now that you understand the basics, proceed to:

- **Tutorial 2**: Creating specialized brain regions
- **Tutorial 3**: Neurotransmitter systems
- **Tutorial 4**: Learning mechanisms

## References

- Hasani, R. et al. (2021). "Liquid Time-Constant Networks" - AAAI
- Mordvintsev, A. et al. (2020). "Neural Cellular Automata" - Distill
- Hodgkin, A. L. & Huxley, A. F. (1952). "A quantitative description of membrane current" - J. Physiol.

## Summary

In this tutorial, you learned:

1. **Installation**: How to install and set up the framework
2. **Core Concepts**: Neural tissue, liquid dynamics, neurogenesis, quantum effects
3. **Basic Operations**: Creating tissue, running simulations, stimulation, reading output
4. **Observation**: Tracking emergence, visualizing structure, interpreting statistics

The key insight is that Organic Neural Networks are not programmed - they are **grown**. The network develops its own architecture through biological-like processes, creating emergent computation that traditional neural networks cannot achieve.
