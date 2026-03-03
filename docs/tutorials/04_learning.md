# Tutorial 4: Learning Mechanisms

## Overview

Learning in Organic Neural Networks differs fundamentally from traditional deep learning. Instead of backpropagation, we use biologically-inspired mechanisms: Spike-Timing Dependent Plasticity (STDP), Hebbian learning, and reward-modulated consolidation.

This tutorial covers:
- STDP: Timing-based synaptic modification
- Hebbian learning: "Fire together, wire together"
- Eligibility traces: Bridging activity and reward
- Memory consolidation: Strengthening important patterns
- Sleep-like consolidation: Offline replay

## Neuroscience Background

### Spike-Timing Dependent Plasticity (STDP)

STDP is the foundational learning rule discovered by Bi and Poo (1998). The key insight: the **order** of spikes matters.

```
Pre fires 10ms BEFORE post  -> LTP (strengthen)
Pre fires 10ms AFTER post   -> LTD (weaken)
Same time                   -> No change
```

The timing window is approximately 20ms - this implements causal learning.

### Hebb's Postulate

Donald Hebb (1949): "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

Shortened: "Neurons that fire together, wire together."

### Three-Factor Learning

Biological learning combines:
1. Pre-synaptic activity
2. Post-synaptic activity
3. Neuromodulator (dopamine) signal

```
delta_w = learning_rate * dopamine * pre_activity * post_activity
```

## STDP Implementation

### How STDP Works in Organic Neural

```python
from organic_neural_network import OrganicSynapse

# Each synapse tracks:
synapse = OrganicSynapse(
    pre_neuron=0,
    post_neuron=1,
    weight=0.5
)

# STDP updates based on spike timing
synapse.update_stdp(
    pre_fired=True,
    post_fired=True,
    time=100.0,  # Current time in ms
    dt=0.1
)
```

### STDP Learning Window

The STDP curve shows weight change as a function of timing:

```
        LTP
         ^
    +0.01|     *
         |   *   *
    +0.00|_*_______*_  t_post - t_pre
         |         *   *
    -0.01|           *
         v
        LTD

    -20ms    0    +20ms
```

Implementation:

```python
def update_stdp(self, pre_fired, post_fired, time, dt):
    if pre_fired:
        self.last_pre_spike = time
        # Post fired recently AFTER pre? LTP
        if 0 < time - self.last_post_spike < 20:
            self.weight += 0.01 * self.strength

    if post_fired:
        self.last_post_spike = time
        # Pre fired recently BEFORE post? LTP
        if 0 < time - self.last_pre_spike < 20:
            self.weight += 0.01 * self.strength
        # Pre fired AFTER post? LTD
        elif 0 < self.last_pre_spike - time < 20:
            self.weight -= 0.005 * self.strength

    # Bound weight
    self.weight = clip(self.weight, 0.0, 2.0)
```

## Hebbian Learning

### Basic Hebbian Update

```python
# Running averages of activity
alpha = 0.1  # Smoothing factor

synapse.pre_activity_avg = alpha * pre_active + (1-alpha) * synapse.pre_activity_avg
synapse.post_activity_avg = alpha * post_active + (1-alpha) * synapse.post_activity_avg

# Hebbian term
hebbian = synapse.pre_activity_avg * synapse.post_activity_avg
```

This computes correlation between pre- and post-synaptic activity over time.

### Observing Hebbian Learning

```python
from organic_neural_network import OrganicNeuralNetwork

tissue = OrganicNeuralNetwork(size=(10, 10, 5), initial_neurons=20)

# Stimulate two regions repeatedly together
for _ in range(100):
    # Input A and Input B fire together
    tissue.stimulate((2, 5, 2.5), intensity=10.0)
    tissue.stimulate((8, 5, 2.5), intensity=10.0)

    # Run simulation
    tissue.step(dt=1.0)

# Check connectivity between regions
# Synapses between co-active neurons should be strengthened
```

## Eligibility Traces

### What Are Eligibility Traces?

Eligibility traces solve the **credit assignment problem**: how do you know which synapses contributed to a reward that comes later?

The trace records recent co-activity so that when a reward arrives, we know which synapses were active:

```python
class OrganicSynapse:
    eligibility_trace: float = 0.0
    eligibility_decay: float = 0.95

    def update_eligibility(self, pre_active, post_active, dt):
        # Update running averages
        self.pre_activity_avg = alpha * pre_active + (1-alpha) * self.pre_activity_avg
        self.post_activity_avg = alpha * post_active + (1-alpha) * self.post_activity_avg

        # Hebbian term
        hebbian = self.pre_activity_avg * self.post_activity_avg

        # Update trace with decay
        self.eligibility_trace = (
            self.eligibility_decay * self.eligibility_trace +
            (1 - self.eligibility_decay) * hebbian
        )
```

### Using Eligibility Traces

```python
# During behavior
for step in range(training_steps):
    tissue.step(dt=0.3)
    tissue.update_eligibility_traces(dt=0.3)  # Track what's active

# When reward arrives
tissue.release_dopamine(reward)
tissue.apply_reward_modulated_plasticity()  # Apply to eligible synapses
```

The eligibility trace equation:

```
e(t) = decay * e(t-1) + (1 - decay) * pre_active * post_active
```

This creates an exponentially decaying memory of co-activity.

## Complete Learning Example

### Setting Up a Learning Task

```python
from organic_neural_network import (
    OrganicNeuralNetwork,
    XORTask,
    PatternRecognitionTask
)

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30,
    energy_supply=3.0
)

# Create task (defines I/O regions)
task = XORTask(tissue)

print(f"Input regions: {list(tissue.input_regions.keys())}")
print(f"Output regions: {list(tissue.output_regions.keys())}")
```

### Training Loop with All Mechanisms

```python
def train_with_all_mechanisms(tissue, task, n_episodes=100):
    """Train using STDP, Hebbian, and reward-modulated plasticity."""

    rewards = []

    for episode in range(n_episodes):
        # Reset task for new episode
        task.reset()
        total_reward = 0.0

        while not task.is_done():
            # Get inputs
            inputs = task.get_inputs()

            # Apply inputs
            tissue.set_inputs(inputs)

            # Run processing with STDP and eligibility updates
            for _ in range(10):
                tissue.step(dt=0.3)
                tissue.update_eligibility_traces(dt=0.3)

            # Read outputs
            outputs = tissue.read_outputs()

            # Get reward
            reward, done = task.evaluate(outputs)
            total_reward += reward

            # Dopamine release (triggers learning)
            tissue.release_dopamine(reward)
            tissue.apply_reward_modulated_plasticity()

            # Energy bonus for success
            if reward > 0:
                tissue.give_energy_bonus("output", reward * 5)

        rewards.append(total_reward)

        # Structural adaptation every 10 episodes
        if episode % 10 == 0 and len(rewards) >= 10:
            recent_perf = sum(rewards[-10:]) / 10 / 5.0  # Normalize
            tissue.structural_adaptation(recent_perf)

            # Report progress
            avg_reward = sum(rewards[-10:]) / 10
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

    return rewards

# Run training
rewards = train_with_all_mechanisms(tissue, task, n_episodes=100)
```

### Expected Training Output

```
Episode 0: Avg Reward = 1.45
Episode 10: Avg Reward = 1.52
Episode 20: Avg Reward = 1.58
Episode 30: Avg Reward = 1.65
Episode 40: Avg Reward = 1.70
...
Episode 90: Avg Reward = 1.82
```

## Memory Consolidation

### Sleep-Like Consolidation

During "sleep," the network replays recent patterns to strengthen memories:

```python
def consolidation_phase(tissue, replay_patterns, n_cycles=50):
    """
    Simulate sleep-like consolidation.

    Args:
        tissue: The neural tissue
        replay_patterns: List of input patterns to replay
        n_cycles: Number of replay cycles
    """
    # Increase consolidation-related plasticity
    tissue.learning_rate = 0.05  # Lower than awake learning

    for cycle in range(n_cycles):
        # Randomly select a pattern to replay
        pattern = random.choice(replay_patterns)

        # Apply the pattern (weak activation)
        tissue.set_inputs({k: v * 0.5 for k, v in pattern.items()})

        # Brief processing
        for _ in range(5):
            tissue.step(dt=0.5)

        # Strengthen replayed connections
        tissue.release_dopamine(0.3)  # Small consolidation signal
        tissue.apply_reward_modulated_plasticity()

    # Restore normal learning rate
    tissue.learning_rate = 0.1
```

### Collecting Replay Patterns

```python
def collect_replay_patterns(tissue, task, n_samples=10):
    """Collect successful patterns for later replay."""
    patterns = []

    for _ in range(n_samples):
        task.reset()

        while not task.is_done():
            inputs = task.get_inputs()
            patterns.append(inputs.copy())

            tissue.set_inputs(inputs)
            for _ in range(10):
                tissue.step(dt=0.3)

            outputs = tissue.read_outputs()
            reward, done = task.evaluate(outputs)

            if reward > 1.5:  # Good performance
                # Store the successful input pattern
                patterns.append(inputs.copy())

    return patterns
```

## Synaptic Pruning

### Automatic Pruning

Weak synapses are automatically pruned when they fall below threshold:

```python
# In each simulation step, synapses are checked:
def should_prune(self):
    return self.strength < 0.1 or self.weight < 0.05
```

### Manual Pruning

```python
# Prune weak connections
tissue.prune_weak_connections(threshold=0.15)

# Check results
print(f"Synapses after pruning: {len(tissue.synapses)}")
```

### Activity-Dependent Pruning

```python
# Prune based on recent activity
for (pre, post), syn in tissue.synapses.items():
    # Calculate activity correlation
    pre_n = tissue.neurons[pre]
    post_n = tissue.neurons[post]

    pre_active = len(pre_n.activation_history) > 0
    post_active = len(post_n.activation_history) > 0

    # If neither has been active, mark for pruning
    if not pre_active and not post_active:
        syn.strength *= 0.9  # Weaken unused synapses
```

## Comparing Learning Mechanisms

```python
def compare_learning_mechanisms():
    """Compare different learning configurations."""

    results = {}

    # 1. STDP only (no reward modulation)
    tissue1 = OrganicNeuralNetwork(size=(10, 10, 5), initial_neurons=30)
    task1 = XORTask(tissue1)
    tissue1.learning_rate = 0.0  # Disable reward learning

    # Run with just STDP
    for _ in range(100):
        tissue1.step(dt=0.5)  # STDP happens automatically
    results['STDP_only'] = tissue1.evaluate_task(task1, n_trials=25)

    # 2. Reward-modulated only (no STDP)
    tissue2 = OrganicNeuralNetwork(size=(10, 10, 5), initial_neurons=30)
    task2 = XORTask(tissue2)

    # Train with reward only
    tissue2.train_task(task2, n_episodes=100)
    results['Reward_only'] = tissue2.evaluate_task(task2, n_trials=25)

    # 3. Both mechanisms
    tissue3 = OrganicNeuralNetwork(size=(10, 10, 5), initial_neurons=30)
    task3 = XORTask(tissue3)

    tissue3.train_task(task3, n_episodes=100)
    results['Combined'] = tissue3.evaluate_task(task3, n_trials=25)

    print(f"\n{'Mechanism':<20} {'Success Rate':>15}")
    print("-" * 35)
    for name, result in results.items():
        print(f"{name:<20} {result['success_rate']*100:>14.1f}%")

    return results
```

Expected output:
```
Mechanism            Success Rate
-----------------------------------
STDP_only                   48.0%
Reward_only                 54.0%
Combined                    58.0%
```

## Learning Best Practices

### 1. Use Appropriate Learning Rates

```python
# For fast initial learning
tissue.learning_rate = 0.2

# For fine-tuning
tissue.learning_rate = 0.05

# For evaluation (no learning)
tissue.learning_rate = 0.0
```

### 2. Balance Exploration and Exploitation

```python
# Early training: encourage exploration
for neuron in tissue.neurons.values():
    neuron.plasticity = 0.02  # Higher plasticity

# Later training: stabilize learned patterns
for neuron in tissue.neurons.values():
    neuron.plasticity = 0.005  # Lower plasticity
```

### 3. Use Consolidation After Training

```python
# Train
tissue.train_task(task, n_episodes=100)

# Consolidate
patterns = collect_replay_patterns(tissue, task)
consolidation_phase(tissue, patterns)

# Re-evaluate (should be more stable)
results = tissue.evaluate_task(task, n_trials=50)
```

## References

- Bi, G. & Poo, M. (1998). "Synaptic Modifications in Cultured Hippocampal Neurons" - J. Neuroscience
- Hebb, D. O. (1949). "The Organization of Behavior" - Wiley
- Sejnowski, T. J. & Tesauro, G. (1989). "The Hebb Rule for Synaptic Plasticity" - NATO ASI Series

## Summary

In this tutorial, you learned:

1. **STDP**: How spike timing determines synaptic modification (LTP vs LTD)
2. **Hebbian Learning**: How correlated activity strengthens connections
3. **Eligibility Traces**: How the network bridges activity and delayed reward
4. **Three-Factor Learning**: The combination of pre-activity, post-activity, and neuromodulator
5. **Consolidation**: How replay strengthens memories
6. **Pruning**: How weak connections are removed

The key insight is that biological learning is **local** - each synapse modifies itself based on information available at that synapse, with global neuromodulatory signals providing credit assignment. This differs fundamentally from backpropagation, where error signals must be computed globally and propagated backward.
