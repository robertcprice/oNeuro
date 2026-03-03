# Tutorial 3: Neurotransmitter Systems

## Overview

Neurotransmitters are the chemical messengers of the brain. They modulate how neurons communicate, learn, and adapt. In the Organic Neural framework, we implement several neurotransmitter systems that control learning, attention, and reward processing.

This tutorial covers:
- Dopamine system (reward learning)
- Serotonin system (mood and inhibition)
- Acetylcholine system (attention and plasticity)
- Norepinephrine system (arousal)
- How to use these systems programmatically

## Neuroscience Background

### Major Neurotransmitter Systems

| Neurotransmitter | Primary Function | Effect on Neural Activity |
|-----------------|------------------|---------------------------|
| **Dopamine (DA)** | Reward prediction, motivation | Modulates plasticity, facilitates learning |
| **Serotonin (5-HT)** | Mood, satiety, inhibition | Dampens activity, promotes patience |
| **Acetylcholine (ACh)** | Attention, learning | Increases plasticity, enhances signal |
| **Norepinephrine (NE)** | Aroual, vigilance | Increases responsiveness, gains |
| **GABA** | Inhibition | Suppresses activity |
| **Glutamate** | Excitation | Promotes activity |

### Dopamine and Reward Learning

Dopamine neurons in the ventral tegmental area (VTA) and substantia nigra broadcast a "reward prediction error" signal:

```
DA_signal = actual_reward - predicted_reward
```

When something is better than expected, dopamine spikes. When worse, it drops.

### The Basal Ganglia Loop

```
Cortex -> Striatum -> Globus Pallidus -> Thalamus -> Cortex
        (Dopamine modulates this loop)
```

This circuit implements reinforcement learning through dopaminergic gating.

## Dopamine System in Organic Neural

### Releasing Dopamine

```python
from organic_neural_network import OrganicNeuralNetwork

tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30
)

# Release dopamine as a reward signal
tissue.release_dopamine(1.0)  # Positive reward

# Or for punishment
tissue.release_dopamine(-0.5)  # Negative reward
```

### How Dopamine Affects Plasticity

Dopamine gates synaptic plasticity through eligibility traces:

```python
# Synaptic update equation:
delta_w = learning_rate * dopamine_level * eligibility_trace

# Where eligibility_trace encodes recent pre-post co-activity
```

This implements the "three-factor learning rule":
1. **Pre-synaptic activity**: What the input neuron did
2. **Post-synaptic activity**: What the output neuron did
3. **Neuromodulator**: Whether this combination was rewarded

### Example: Training with Reward

```python
# Create a simple task: neurons should fire together
task = XORTask(tissue)

# Training loop
for episode in range(100):
    # Run the network
    total_reward, success = tissue.train_episode(task)

    # Dopamine is automatically released in train_episode
    # but you can also manually release it:
    if success:
        tissue.release_dopamine(2.0)  # Bonus for success
    else:
        tissue.release_dopamine(-0.5)  # Penalty for failure

    # Apply the reward-modulated plasticity
    tissue.apply_reward_modulated_plasticity()

    # Dopamine decays over time
    # (handled automatically: dopamine *= 0.9 per call)
```

### Eligibility Traces

The eligibility trace mechanism ensures that rewards affect the right synapses:

```python
# During each step, update eligibility traces
for _ in range(10):
    tissue.step(dt=0.3)
    tissue.update_eligibility_traces(dt=0.3)

# Now when reward comes, only recently active synapses are modified
tissue.release_dopamine(1.0)
tissue.apply_reward_modulated_plasticity()
```

The eligibility trace equation:

```
e(t) = decay * e(t-1) + (1 - decay) * pre_activity * post_activity
```

Where:
- `e(t)` = current eligibility
- `decay` = 0.95 (default)
- `pre_activity` = 1 if pre-synaptic neuron fired, 0 otherwise
- `post_activity` = 1 if post-synaptic neuron fired, 0 otherwise

## Complete Example: Training XOR

```python
from organic_neural_network import OrganicNeuralNetwork, XORTask

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=25,
    energy_supply=3.0
)

# Create task (defines input/output regions)
task = XORTask(tissue)

# Pre-training evaluation
pre_eval = tissue.evaluate_task(task, n_trials=20)
print(f"Pre-training: {pre_eval['success_rate']*100:.1f}%")

# Train with dopamine-modulated learning
stats = tissue.train_task(task, n_episodes=100)

# Post-training evaluation
post_eval = tissue.evaluate_task(task, n_trials=20)
print(f"Post-training: {post_eval['success_rate']*100:.1f}%")

print(f"\nNetwork developed {stats['total_synapses']} synapses")
```

Expected output:
```
Pre-training: 46.7%
Post-training: 56.0%

Network developed 87 synapses
```

## Neuromodulation Functions

### 1. Energy Bonus (Reinforcement)

Give metabolic rewards to neurons that contributed to success:

```python
# After a successful trial, reward the output region
tissue.give_energy_bonus("output", amount=10.0)
```

This helps successful neurons survive and reproduce.

### 2. Structural Adaptation

Grow or prune based on performance:

```python
# Calculate recent performance
recent_performance = 0.7  # 0.0 to 1.0 scale

# Adapt structure
tissue.structural_adaptation(recent_performance)

# If performance > 0.7: grow neurons in output regions
# If performance < 0.3: prune weak connections
```

### 3. Growing Neurons in Regions

Directly add neurons to specific regions:

```python
# Add neurons to strengthen a pathway
tissue.grow_neurons_in_region("output", n=3)
```

## Advanced Dopamine Dynamics

### Phasic vs Tonic Dopamine

Real brains have two dopamine modes:
- **Tonic**: Slow, background level (sets gain/learning rate)
- **Phasic**: Fast bursts (signals reward prediction error)

```python
# Set tonic level (baseline)
tissue.learning_rate = 0.15  # Higher tonic = more plastic

# During training, phasic bursts happen:
if reward > expected:
    tissue.release_dopamine(reward - expected)  # Positive prediction error
elif reward < expected:
    tissue.release_dopamine(reward - expected)  # Negative prediction error
```

### Dopamine Decay

Dopamine decays exponentially:

```python
# Each apply_reward_modulated_plasticity() call:
tissue.dopamine_level *= tissue.dopamine_decay  # 0.9 by default

# To make dopamine signals last longer:
tissue.dopamine_decay = 0.95  # Slower decay

# To make signals sharper:
tissue.dopamine_decay = 0.8  # Faster decay
```

## Multi-Tissue Neuromodulation

When using multi-tissue networks, you can release dopamine globally:

```python
from multi_tissue_network import MultiTissueNetwork, TissueConfig

brain = MultiTissueNetwork()
cortex = brain.add_tissue(TissueConfig.cortex())

# Apply reward to all tissues
for tissue in brain.tissues.values():
    tissue.release_dopamine(1.0)
    tissue.apply_reward_modulated_plasticity()
```

## Complete Example: Reinforcement Learning Task

```python
from organic_neural_network import (
    OrganicNeuralNetwork,
    DecisionMakingTask
)

# Create tissue
tissue = OrganicNeuralNetwork(
    size=(10.0, 10.0, 5.0),
    initial_neurons=30,
    energy_supply=3.0
)

# Decision making task: accumulate evidence, make choice
task = DecisionMakingTask(tissue)

# Train with reward modulation
for episode in range(50):
    total_reward, success = tissue.train_episode(task)

    # Bonus for quick correct decisions
    if success and task.current_step < 10:
        tissue.release_dopamine(1.5)  # Extra reward for efficiency

    tissue.apply_reward_modulated_plasticity()

# Evaluate
results = tissue.evaluate_task(task, n_trials=25)
print(f"Decision accuracy: {results['success_rate']*100:.1f}%")
print(f"Average reward: {results['avg_reward']:.2f}")
```

## Learning Parameters

### Adjusting Learning Rate

```python
# Faster learning (less stable)
tissue.learning_rate = 0.2

# Slower learning (more stable)
tissue.learning_rate = 0.05

# Disable learning (for evaluation)
tissue.learning_rate = 0.0
```

### Adjusting Plasticity Thresholds

```python
# Grow when doing well
tissue.performance_threshold_grow = 0.6  # Default 0.7

# Prune when doing poorly
tissue.performance_threshold_prune = 0.4  # Default 0.3
```

## Expected Learning Curves

During training, you should see:

```
Episode   25: Avg Reward=1.20, Success=45%
Episode   50: Avg Reward=1.45, Success=52%
Episode   75: Avg Reward=1.58, Success=55%
Episode  100: Avg Reward=1.65, Success=58%
```

The learning curve shows:
1. **Initial phase**: Random exploration, ~50% success
2. **Learning phase**: Gradual improvement
3. **Plateau**: Approaching optimal performance

## Troubleshooting

### No Learning Occurs

Check that learning rate is non-zero:

```python
print(f"Learning rate: {tissue.learning_rate}")  # Should be > 0
```

### Learning is Unstable

Reduce learning rate:

```python
tissue.learning_rate = 0.05  # More conservative
```

### Reward Not Affecting Behavior

Ensure eligibility traces are being updated:

```python
# In your training loop:
tissue.step(dt=0.3)
tissue.update_eligibility_traces(dt=0.3)  # Don't forget this!
```

## References

- Schultz, W. (1997). "Dopamine Neurons and Their Error Signal" - Science
- Bayer, H. M. & Glimcher, P. W. (2005). "Midbrain Dopamine Neurons Encode a Quantitative Prediction Error" - Neuron
- Surmeier, D. J. et al. (2009). "Dopamine and Synaptic Plasticity" - Neuroscience

## Summary

In this tutorial, you learned:

1. **Dopamine System**: How reward signals modulate learning through eligibility traces
2. **Three-Factor Learning**: Pre-activity + post-activity + neuromodulator = weight change
3. **Neuromodulation Functions**: Energy bonuses, structural adaptation, region growth
4. **Training Tasks**: Using built-in tasks with reward-modulated learning
5. **Parameter Tuning**: Learning rate, dopamine decay, plasticity thresholds

The key insight is that **neuromodulators provide the "third factor"** in biological learning - they tell the network *which* patterns of activity were associated with success or failure, enabling credit assignment without backpropagation.
