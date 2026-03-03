# Tutorial 2: Brain Regions and Multi-Tissue Networks

## Overview

The brain is not a homogeneous mass of neurons - it consists of specialized regions with distinct properties and functions. The cortex handles general computation, the thalamus relays signals, the hippocampus forms memories, and so on.

This tutorial covers:
- Creating specialized tissue types
- Connecting brain regions
- Information flow between tissues
- Emergent behavior from distributed processing

## Neuroscience Background

### Real Brain Organization

The vertebrate brain has several major divisions:

| Region | Function | Properties |
|--------|----------|------------|
| **Cortex** | Higher cognition, perception | Large, high plasticity, layered |
| **Thalamus** | Sensory relay, attention | Small, fast switching, hub-like |
| **Hippocampus** | Memory formation, spatial navigation | High plasticity, pattern completion |
| **Brainstem** | Vital functions, arousal | Stable, low plasticity, essential |
| **Cerebellum** | Motor control, timing | Dense connections, precise timing |

### Why Specialization Matters

Specialized regions enable:
1. **Efficiency**: Different computations use optimized hardware
2. **Robustness**: Damage to one region doesn't destroy all function
3. **Emergence**: Connected simple systems can exceed their parts

## Tissue Types in Organic Neural

### TissueConfig Class

```python
from multi_tissue_network import (
    MultiTissueNetwork,
    TissueConfig,
    TissueType
)
```

### Available Tissue Types

#### Cortex

```python
# Large, plastic tissue for general computation
cortex_config = TissueConfig.cortex(
    size=8.0,      # Creates 8x8x4 volume
    neurons=40      # Number of initial neurons
)

# Properties:
# - High plasticity (0.02)
# - Wide time constant range (8-20ms)
# - Moderate threshold (-58 to -52mV)
```

#### Thalamus

```python
# Small, fast relay station
thalamus_config = TissueConfig.thalamus(
    neurons=20
)

# Properties:
# - Low plasticity (0.005) - stable routing
# - Fast time constants (3-8ms)
# - Compact size (4x4x4)
```

#### Hippocampus

```python
# Memory and pattern completion
hippocampus_config = TissueConfig.hippocampus(
    neurons=25
)

# Properties:
# - Very high plasticity (0.03)
# - Slow time constants (10-25ms)
# - High threshold range (-60 to -55mV)
```

#### Brainstem

```python
# Stable, essential functions
brainstem_config = TissueConfig.brainstem(
    neurons=15
)

# Properties:
# - Very low plasticity (0.001) - stable
# - High energy supply (2.0) - priority
# - Moderate time constants (5-10ms)
```

## Creating a Multi-Tissue Network

### Basic Setup

```python
from multi_tissue_network import MultiTissueNetwork, TissueConfig

# Create the brain network
brain = MultiTissueNetwork()

# Add tissues with specific configurations
cortex_id = brain.add_tissue(TissueConfig.cortex(neurons=30))
thalamus_id = brain.add_tissue(TissueConfig.thalamus(neurons=15))
hippocampus_id = brain.add_tissue(TissueConfig.hippocampus(neurons=20))
brainstem_id = brain.add_tissue(TissueConfig.brainstem(neurons=10))

print(f"Created brain with {len(brain.tissues)} regions")
```

### Connecting Regions

Regions are connected via "white matter" tracts - bundles of axons between tissues:

```python
# Connect thalamus <-> cortex (bidirectional sensory pathway)
brain.connect_tissues(
    thalamus_id,
    cortex_id,
    connection_prob=0.15,      # Probability of neuron-neuron connection
    weight_range=(0.3, 0.7)    # Synaptic weight range
)

# Connect cortex <-> hippocampus (memory pathway)
brain.connect_tissues(
    cortex_id,
    hippocampus_id,
    connection_prob=0.10
)

# Connect brainstem -> thalamus (ascending arousal)
brain.connect_tissues(
    brainstem_id,
    thalamus_id,
    connection_prob=0.20
)

print(f"Created {len(brain.inter_connections)} inter-tissue connections")
```

### Connection Patterns

```python
# Define a thalamocortical loop
brain.connect_tissues(thalamus_id, cortex_id, connection_prob=0.15)
brain.connect_tissues(cortex_id, thalamus_id, connection_prob=0.12)

# Hippocampal-cortical loop for memory
brain.connect_tissues(cortex_id, hippocampus_id, connection_prob=0.10)
brain.connect_tissues(hippocampus_id, cortex_id, connection_prob=0.10)

# Ascending arousal pathway
brain.connect_tissues(brainstem_id, thalamus_id, connection_prob=0.20)
brain.connect_tissues(thalamus_id, brainstem_id, connection_prob=0.08)
```

## Running Multi-Tissue Simulations

### Basic Simulation Loop

```python
import time

print("Running brain simulation...")

for step in range(150):
    # Advance all tissues
    brain.step(dt=0.5)

    # Apply sensory-like stimulation to brainstem
    if step % 15 == 0:
        brain.stimulate_region(
            brainstem_id,
            position=(1.5, 1.5, 2.0),
            intensity=15.0,
            radius=1.5
        )

    # Periodic status report
    if step % 30 == 0:
        state = brain.get_global_state()
        print(f"\nTime: {brain.time:.1f}ms")
        for tid, info in state['tissues'].items():
            print(f"  [{tid}] {info['type']}: {info['neurons']} neurons, "
                  f"{info['active_fraction']*100:.0f}% active")
```

### Expected Output

```
Running brain simulation...

Time: 15.0ms
  [0] cortex: 30 neurons, 8% active
  [1] thalamus: 15 neurons, 12% active
  [2] hippocampus: 20 neurons, 5% active
  [3] brainstem: 10 neurons, 20% active

Time: 30.0ms
  [0] cortex: 31 neurons, 15% active
  [1] thalamus: 15 neurons, 18% active
  [2] hippocampus: 21 neurons, 8% active
  [3] brainstem: 10 neurons, 25% active
```

## Information Flow Between Regions

### Reading Cross-Tissue Activity

```python
def observe_information_flow(brain):
    """Track how activity spreads across tissues."""

    # Get activity levels for each region
    activities = {}
    for tid, tissue in brain.tissues.items():
        alive = [n for n in tissue.neurons.values() if n.alive]
        if alive:
            active = sum(1 for n in alive if n.state.name == 'ACTIVE')
            activities[tid] = active / len(alive)
        else:
            activities[tid] = 0.0

    # Check inter-tissue spike count
    cross_tissue_ratio = (brain.inter_tissue_spikes /
                          max(1, brain.total_spikes))

    return {
        'activities': activities,
        'cross_tissue_ratio': cross_tissue_ratio,
        'workspace_size': len(brain.workspace_activity)
    }
```

### Triggering Cascade Events

```python
# Strong stimulus can trigger information cascades
brain.stimulate_region(
    brainstem_id,
    position=(1.5, 1.5, 2.0),
    intensity=25.0,  # Strong stimulus
    radius=2.0
)

# Run for cascade propagation
for _ in range(50):
    brain.step(dt=0.2)
    state = brain.get_global_state()

    # Check for cascade (high cross-tissue activity)
    if len(state['workspace_tissues']) >= 3:
        print(f"CASCADE at {brain.time:.1f}ms!")
        print(f"  Active tissues: {state['workspace_tissues']}")
```

## Global Workspace Theory Implementation

### The Workspace Mechanism

The multi-tissue network implements Baars' Global Workspace Theory:

1. **Competition**: Tissues compete for workspace access
2. **Broadcasting**: Most active tissues broadcast to all others
3. **Integration**: Information becomes globally available

```python
# The workspace contains tissue IDs that are currently "conscious"
print(f"Workspace tissues: {brain.workspace_activity}")
print(f"Broadcast events: {brain.workspace_broadcasts}")
```

### Checking Workspace State

```python
def check_consciousness_state(brain):
    """Assess global workspace (consciousness-like) state."""

    state = brain.get_global_state()

    # Workspace occupancy
    occupancy = len(state['workspace_tissues']) / len(state['tissues'])

    # Cross-tissue integration
    integration = brain.inter_tissue_spikes / max(1, brain.total_spikes)

    return {
        'workspace_occupancy': occupancy,
        'integration': integration,
        'is_integrated': len(state['workspace_tissues']) >= 2
    }
```

## Emergence Analysis

### Using the EmergenceAnalyzer

```python
from multi_tissue_network import EmergenceAnalyzer

analyzer = EmergenceAnalyzer(brain)

# Run simulation
for _ in range(100):
    brain.step(dt=0.5)
    analysis = analyzer.analyze()

    if analysis['emergence_score'] > 0.4:
        print(f"EMERGENCE at t={brain.time:.1f}ms!")
        print(f"  Score: {analysis['emergence_score']:.3f}")
        print(f"  Integration: {analysis['information_integration']:.3f}")
        print(f"  Synchronization: {analysis['synchronization_index']:.3f}")
```

### Emergence Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `information_integration` | How irreducible information is across tissues | 0-1 |
| `specialization_index` | How different tissues have become | 0-1 |
| `synchronization_index` | Phase-locking between tissues | 0-1 |
| `cascade_frequency` | Cross-tissue / total spike ratio | 0-1 |
| `workspace_efficiency` | Global broadcast frequency | 0-1 |
| `emergence_score` | Weighted composite | 0-1 |

### Generating Reports

```python
# Get detailed emergence report
report = analyzer.get_emergence_report()
print(report)
```

## Complete Example: Thalamocortical System

```python
from multi_tissue_network import (
    MultiTissueNetwork,
    TissueConfig,
    EmergenceAnalyzer
)

def build_thalamocortical_system():
    """Build a simplified thalamocortical system."""

    brain = MultiTissueNetwork()

    # Add tissues
    cortex = brain.add_tissue(TissueConfig.cortex(neurons=40))
    thalamus = brain.add_tissue(TissueConfig.thalamus(neurons=20))

    # Create thalamocortical loops
    brain.connect_tissues(thalamus, cortex, connection_prob=0.15)
    brain.connect_tissues(cortex, thalamus, connection_prob=0.12)

    return brain, cortex, thalamus

def run_sensory_processing(brain, sensory_tissue_id, output_tissue_id):
    """Simulate sensory input and observe processing."""

    analyzer = EmergenceAnalyzer(brain)

    print("Sensory Processing Simulation")
    print("=" * 50)

    for trial in range(5):
        # Reset network state
        brain.total_spikes = 0
        brain.inter_tissue_spikes = 0

        # Apply sensory stimulus
        brain.stimulate_region(
            sensory_tissue_id,
            position=(2.0, 2.0, 2.0),
            intensity=20.0,
            radius=2.0
        )

        # Record processing
        activities = []
        for _ in range(50):
            brain.step(dt=0.3)
            state = brain.get_global_state()
            activities.append(state['tissues'][output_tissue_id]['active_fraction'])

        # Analyze
        analysis = analyzer.analyze()
        peak_activity = max(activities)
        latency = activities.index(peak_activity) * 0.3

        print(f"\nTrial {trial + 1}:")
        print(f"  Peak activity: {peak_activity*100:.1f}%")
        print(f"  Response latency: {latency:.1f}ms")
        print(f"  Integration: {analysis['information_integration']:.3f}")

    return analyzer

# Run the system
brain, cortex, thalamus = build_thalamocortical_system()
analyzer = run_sensory_processing(brain, thalamus, cortex)
print(analyzer.get_emergence_report())
```

## Visualizing Multi-Tissue State

### ASCII Visualization

```python
print(brain.visualize_global())
```

Output:
```
================================================================================
MULTI-TISSUE NEURAL NETWORK - Global State
================================================================================
Time: 75.0ms  |  Step: 150  |  Workspace: [0, 1]
--------------------------------------------------------------------------------
   [0] CORTEX        | Neurons:  40 | Active:   6 (15.0%) | Quantum:  2 | Energy: 95.3
   [1] THALAMUS      | Neurons:  20 | Active:   4 (20.0%) | Quantum:  1 | Energy: 87.2
   [2] HIPPOCAMPUS   | Neurons:  21 | Active:   2 ( 9.5%) | Quantum:  0 | Energy: 92.1
   [3] BRAINSTEM     | Neurons:  10 | Active:   3 (30.0%) | Quantum:  0 | Energy: 98.5
--------------------------------------------------------------------------------
Inter-tissue connections: 156 | Total spikes: 2847 | Cross-tissue: 423
--------------------------------------------------------------------------------
RECENT EMERGENCE EVENTS:
  t=45.2: synchronization - {'mean_activity': 0.18, 'std_activity': 0.03}
  t=62.1: information_cascade - {'cascade_ratio': 0.52}
================================================================================
```

## Inter-Tissue Connection Properties

### White Matter Tract Properties

```python
@dataclass
class InterTissueConnection:
    source_tissue: int       # Origin tissue ID
    source_neuron: int       # Origin neuron ID
    target_tissue: int       # Target tissue ID
    target_neuron: int       # Target neuron ID
    weight: float = 0.5      # Synaptic weight
    delay: float = 2.0       # Transmission delay (ms)
    myelination: float = 0.5 # Affects speed (0-1)
```

### Modifying Connections

```python
# Access inter-tissue connections
for conn in brain.inter_connections[:5]:
    print(f"Tissue {conn.source_tissue} -> {conn.target_tissue}")
    print(f"  Weight: {conn.weight:.3f}")
    print(f"  Delay: {conn.delay:.1f}ms")
    print(f"  Activations: {conn.activation_count}")
```

## Best Practices

### 1. Start Simple

Begin with 2-3 tissues before building complex systems:

```python
# Good for learning
brain = MultiTissueNetwork()
cortex = brain.add_tissue(TissueConfig.cortex(neurons=20))
thalamus = brain.add_tissue(TissueConfig.thalamus(neurons=10))
brain.connect_tissues(thalamus, cortex)
```

### 2. Match Connection Probability to Tissue Size

Larger tissues need lower connection probability:

```python
# Large -> Small: lower probability
brain.connect_tissues(cortex_id, thalamus_id, connection_prob=0.08)

# Small -> Large: higher probability
brain.connect_tissues(thalamus_id, cortex_id, connection_prob=0.15)
```

### 3. Create Bidirectional Loops

Information typically flows in loops:

```python
# Thalamocortical loop
brain.connect_tissues(thalamus, cortex, connection_prob=0.15)
brain.connect_tissues(cortex, thalamus, connection_prob=0.12)
```

### 4. Use the Analyzer

Always track emergence to understand what's happening:

```python
analyzer = EmergenceAnalyzer(brain)
# Check periodically
if step % 50 == 0:
    results = analyzer.analyze()
    if results['emergence_score'] > 0.3:
        print("Emergent behavior detected!")
```

## Next Steps

Now that you understand multi-tissue networks, explore:

- **Tutorial 3**: Neurotransmitter systems and neuromodulation
- **Tutorial 4**: Learning mechanisms in connected tissues
- **Tutorial 7**: Consciousness measures and the global workspace

## References

- Baars, B. J. (1988). "A Cognitive Theory of Consciousness" - Cambridge University Press
- Sherman, S. M. (2016). "Thalamus plays a central role in ongoing cortical functioning" - PNAS
- Buzsaki, G. (2006). "Rhythms of the Brain" - Oxford University Press

## Summary

In this tutorial, you learned:

1. **Tissue Types**: Cortex, thalamus, hippocampus, brainstem with distinct properties
2. **Connecting Regions**: How to create white matter tracts between tissues
3. **Information Flow**: Tracking activity across regions
4. **Global Workspace**: The broadcasting mechanism for consciousness-like behavior
5. **Emergence Analysis**: Measuring when distributed processing exceeds local computation

The key insight is that **intelligence emerges from the interaction of specialized regions**, not from any single region alone. By connecting simple tissues with appropriate patterns, complex behavior emerges.
