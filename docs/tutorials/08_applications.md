# Tutorial 8: Applications and Research Directions

## Overview

digital Organic Neural Networks (dONNs) — built with oNeuro — open new possibilities for research and applications that are impossible with standard artificial neural networks. Because dONNs simulate the full molecular machinery of biological neurons (HH ion channels, neurotransmitters, gene expression, STDP), they can model development, adapt to damage, respond to drugs, and exhibit emergent consciousness metrics.

This tutorial explores practical applications and cutting-edge research directions, with emphasis on capabilities unique to the dONN paradigm.

This tutorial covers:
- Developmental neuroscience modeling
- Neurodegenerative disease simulation
- Brain-Computer Interfaces (demonstrated in DishBrain replication)
- Neuromorphic engineering
- Cognitive robotics
- Game learning (DishBrain Pong, Spatial Arena)
- Open research questions

## Application Areas

### 1. Developmental Neuroscience

Model how brains grow and develop:

```python
from organic_neural_network import OrganicNeuralNetwork
import numpy as np

class DevelopmentalSimulation:
    """Simulate brain development from birth to maturity."""

    def __init__(self, initial_neurons=20):
        self.tissue = OrganicNeuralNetwork(
            size=(20.0, 20.0, 10.0),
            initial_neurons=initial_neurons,
            energy_supply=2.0
        )
        self.development_log = []

    def simulate_development(self, n_days=100, stimuli_per_day=10):
        """
        Simulate development over time.

        Args:
            n_days: Number of simulated days
            stimuli_per_day: Average daily stimuli events

        Returns:
            Development trajectory
        """
        for day in range(n_days):
            # Daily statistics
            stats = {
                'day': day,
                'neurons': len([n for n in self.tissue.neurons.values() if n.alive]),
                'synapses': len(self.tissue.synapses),
                'generation': self.tissue.generation,
                'neurogenesis': self.tissue.neurogenesis_events
            }

            # Daily activity/stimulation
            for _ in range(stimuli_per_day):
                # Random sensory stimulation
                pos = (
                    np.random.uniform(1, 19),
                    np.random.uniform(1, 19),
                    np.random.uniform(1, 9)
                )
                self.tissue.stimulate(pos, intensity=10.0, radius=3.0)

                # Process
                for _ in range(50):
                    self.tissue.step(dt=1.0)

            self.development_log.append(stats)

        return self.development_log

    def analyze_critical_periods(self):
        """Identify critical periods in development."""
        log = self.development_log

        if len(log) < 10:
            return []

        # Find periods of rapid growth
        neuron_counts = [s['neurons'] for s in log]
        growth_rate = np.diff(neuron_counts)

        critical_periods = []
        for i, range(len(growth_rate)):
            if growth_rate[i] > np.mean(growth_rate) + np.std(growth_rate):
                critical_periods.append({
                    'day': log[i]['day'],
                    'growth': growth_rate[i]
                })

        return critical_periods

# Run simulation
dev_sim = DevelopmentalSimulation(initial_neurons=15)
trajectory = dev_sim.simulate_development(n_days=50)

critical = dev_sim.analyze_critical_periods()

print(f"Development: {trajectory[0]['neurons']} -> {trajectory[-1]['neurons']} neurons")
print(f"Critical periods: {len(critical)}")
```

### 2. Neurodegenerative Disease Modeling

Model Alzheimer's, Parkinson's, and other diseases:

```python
class NeurodegenerativeSimulation:
    """Simulate neurodegenerative disease progression."""

    def __init__(self, tissue):
        self.tissue = tissue
        self.initial_neurons = len([n for n in tissue.neurons.values() if n.alive])
        self.progression_log = []

    def apply_alzheimers pathology(self, severity=0.01):
        """
        Simulate Alzheimer's-like pathology.

        Effects:
        - Reduce energy supply (metabolic deficit)
        - Increase synaptic pruning (tau accumulation)
        - Selective vulnerability of certain neurons
        """
        # Reduce energy (metabolic deficit)
        self.tissue.energy_supply *= (1 - severity)

        # Increase pruning threshold (synapse loss)
        for synapse in self.tissue.synapses.values():
            synapse.strength *= (1 - severity * 0.5)

        # Selective neuronal vulnerability (random loss)
        vulnerable = np.random.choice(
            list(self.tissue.neurons.keys()),
            size=max(1, int(len(self.tissue.neurons) * severity))
        )
        for nid in vulnerable:
            if nid in self.tissue.neurons:
                self.tissue.neurons[nid].energy = 0

    def apply_parkinsons_pathology(self, severity=0.01):
        """
        Simulate Parkinson's-like pathology.

        Effects:
        - Selective loss of specific neuron types
        - Motor region degeneration
        """
        # Identify "motor" neurons (simplified: those in certain positions)
        motor_neurons = [
            nid for nid, n in self.tissue.neurons.items()
            if n.alive and n.z < self.tissue.size[2] * 0.3
        ]

        # Selective loss
        n_loss = max(1, int(len(motor_neurons) * severity))
        for nid in np.random.choice(motor_neurons, size=n_loss, replace=False):
            self.tissue.neurons[nid].energy = 0

    def track_progression(self, n_steps=100, pathology='alzheimers', severity=0.005):
        """Track disease progression over time."""
        for step in range(n_steps):
            # Apply pathology
            if pathology == 'alzheimers':
                self.apply_alzheimers_pathology(severity)
            elif pathology == 'parkinsons':
                self.apply_parkinsons_pathology(severity)

            # Run simulation
            self.tissue.step(dt=1.0)

            # Record
            alive = len([n for n in self.tissue.neurons.values() if n.alive])
            active = sum(1 for n in self.tissue.neurons.values()
                        if n.alive and n.state.name == 'ACTIVE')

            self.progression_log.append({
                'step': step,
                'alive': alive,
                'active': active,
                'synapses': len(self.tissue.synapses),
                'retention': alive / self.initial_neurons
            })

        return self.progression_log

# Run simulation
from organic_neural_network import OrganicNeuralNetwork

tissue = OrganicNeuralNetwork(size=(15, 15, 8), initial_neurons=50)
disease_sim = NeurodegenerativeSimulation(tissue)
progression = disease_sim.track_progression(n_steps=200, pathology='alzheimers')

print(f"Neuron retention: {progression[0]['retention']*100:.1f}% -> {progression[-1]['retention']*100:.1f}%")
```

### 3. Brain-Computer Interfaces

Model neural recording and stimulation:

```python
class BrainComputerInterface:
    """Simulate a BCI system."""

    def __init__(self, tissue):
        self.tissue = tissue
        self.recording_history = []
        self.stimulation_patterns = []

    def record_activity(self, region_name, duration_ms=100):
        """
        Record neural activity from a region.

        Args:
            region_name: Name of output region to record
            duration_ms: Recording duration

        Returns:
            Activity trace
        """
        trace = []
        for _ in range(int(duration_ms / 0.5)):
            self.tissue.step(dt=0.0)
            activity = self.tissue.read_output(region_name)
            trace.append(activity)

        self.recording_history.append(trace)
        return np.array(trace)

    def decode_intention(self, activity_trace):
        """
        Decode intended action from neural activity.

        Simplified decoder based on threshold crossing.
        """
        # Compute features
        mean_activity = np.mean(activity_trace)
        peak_activity = np.max(activity_trace)
        latency = np.argmax(activity_trace > 0.5) * 0.5  # ms

        # Decode intention
        if mean_activity > 0.7:
            return 'move_right'
        elif mean_activity > 0.4:
            return 'move_left'
        else:
            return 'no_move'

    def provide_feedback(self, success):
        """
        Provide feedback to the neural tissue.

        Args:
            success: Whether the decoded intention was correct
        """
        if success:
            # Reward: dopamine and energy bonus
            self.tissue.release_dopamine(1.0)
            for neuron in self.tissue.neurons.values():
                neuron.energy += 10
        else:
            # Error signal
            self.tissue.release_dopamine(-0.5)

    def calibrate(self, n_trials=50):
        """
        Calibrate the BCI decoder.
        """
        accuracies = []

        for trial in range(n_trials):
            # Record activity
            trace = self.record_activity('output', duration_ms=50)

            # Decode
            intention = self.decode_intention(trace)

            # In real system, would compare to actual intention
            # Here we use a simulated target
            target = np.random.choice(['move_right', 'move_left', 'no_move'])

            # Provide feedback
            self.provide_feedback(intention == target)
            accuracies.append(intention == target)

        return np.mean(accuracies)

# Use BCI
from organic_neural_network import OrganicNeuralNetwork

tissue = OrganicNeuralNetwork(size=(10, 10, 5), initial_neurons=30)
tissue.define_output_region('output', (5.0, 5.0, 2.5), radius=2.0)

bci = BrainComputerInterface(tissue)
accuracy = bci.calibrate(n_trials=100)
print(f"BCI calibration accuracy: {accuracy*100:.1f}%")
```

### 4. Cognitive Robotics

Simulate embodied cognitive systems:

```python
class CognitiveRobot:
    """A robot controlled by organic neural tissue."""

    def __init__(self, tissue_size=(10, 10, 5)):
        self.tissue = OrganicNeuralNetwork(
            size=tissue_size,
            initial_neurons=40,
            energy_supply=3.0
        )

        # Define sensor regions
        self.tissue.define_input_region('left_sensor', (2.0, 5.0, 2.5), radius=1.5)
        self.tissue.define_input_region('right_sensor', (8.0, 5.0, 2.5), radius=1.5)
        self.tissue.define_input_region('front_sensor', (5.0, 2.0, 2.5), radius=1.5)

        # Define motor regions
        self.tissue.define_output_region('left_motor', (2.0, 5.0, 2.5), radius=1.5)
        self.tissue.define_output_region('right_motor', (8.0, 5.0, 2.5), radius=1.5)

        self.position = [0.0, 0.0]
        self.target = None

    def sense_environment(self, environment):
        """Process sensory input."""
        # Distance to walls/obstacles
        left_dist = environment.get_distance(self.position, 'left')
        right_dist = environment.get_distance(self.position, 'right')
        front_dist = environment.get_distance(self.position, 'front')

        # Normalize to 0-1
        self.tissue.set_input('left_sensor', 1 - min(left_dist / 10, 1))
        self.tissue.set_input('right_sensor', 1 - min(right_dist / 10, 1))
        self.tissue.set_input('front_sensor', 1 - min(front_dist / 10, 1))

    def decide_action(self):
        """Decide on motor action based on neural output."""
        # Process neural activity
        for _ in range(10):
            self.tissue.step(dt=0.5)

        # Read motor outputs
        left_motor = self.tissue.read_output('left_motor')
        right_motor = self.tissue.read_output('right_motor')

        # Convert to movement
        if left_motor > 0.5 and right_motor < 0.5:
            return 'turn_left'
        elif right_motor > 0.5 and left_motor < 0.5:
            return 'turn_right'
        elif left_motor > 0.5 and right_motor > 0.5:
            return 'forward'
        else:
            return 'stop'

    def execute_action(self, action):
        """Execute the decided action."""
        if action == 'turn_left':
            self.position[0] -= 0.5
        elif action == 'turn_right':
            self.position[0] += 0.5
        elif action == 'forward':
            self.position[1] += 0.5

    def navigate(self, environment, n_steps=100):
        """Navigate through environment."""
        trajectory = [self.position.copy()]

        for step in range(n_steps):
            # Sense
            self.sense_environment(environment)

            # Decide
            action = self.decide_action()

            # Act
            self.execute_action(action)

            # Record
            trajectory.append(self.position.copy())

            # Reward if approaching target
            if self.target:
                dist = np.linalg.norm(np.array(self.position) - np.array(self.target))
                if dist < 2:
                    self.tissue.release_dopamine(1.0)

        return np.array(trajectory)

# Simple environment
class SimpleEnvironment:
    def __init__(self):
        self.walls = []

    def get_distance(self, position, direction):
        """Get distance to nearest obstacle in direction."""
        # Simplified: random distances
        return np.random.uniform(2, 10)

# Run robot
robot = CognitiveRobot()
robot.target = [5.0, 10.0]
env = SimpleEnvironment()
trajectory = robot.navigate(env, n_steps=50)
print(f"Robot navigated from {trajectory[0]} to {trajectory[-1]}")
```

## Research Experiments

### Continual Learning

Test if organic networks resist catastrophic forgetting:

```python
from organic_neural_network import (
    OrganicNeuralNetwork,
    XORTask,
    PatternRecognitionTask
)

def test_continual_learning():
    """Test if network can learn multiple tasks sequentially."""
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=40,
        energy_supply=3.0
    )

    # Task 1: XOR
    xor_task = XORTask(tissue)
    tissue.train_task(xor_task, n_episodes=50)
    xor_performance = tissue.evaluate_task(xor_task, n_trials=25)
    print(f"Task 1 (XOR) after training: {xor_performance['success_rate']*100:.1f}%")

    # Task 2: Pattern Recognition (different regions)
    pattern_task = PatternRecognitionTask(tissue)
    tissue.train_task(pattern_task, n_episodes=50)
    pattern_performance = tissue.evaluate_task(pattern_task, n_trials=25)
    print(f"Task 2 (Pattern) after training: {pattern_performance['success_rate']*100:.1f}%")

    # Re-test Task 1 (catastrophic forgetting?)
    xor_retention = tissue.evaluate_task(xor_task, n_trials=25)
    print(f"Task 1 (XOR) retention: {xor_retention['success_rate']*100:.1f}%")

    forgetting = xor_performance['success_rate'] - xor_retention['success_rate']
    print(f"Forgetting: {forgeting*100:.1f}%")

    return {
        'xor_initial': xor_performance['success_rate'],
        'pattern': pattern_performance['success_rate'],
        'xor_retention': xor_retention['success_rate'],
        'forgetting': forgetting
    }

results = test_continual_learning()
```

### Neural Resilience

Test robustness to damage:

```python
def test_neural_resilience():
    """Test how network handles damage."""
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=50,
        energy_supply=3.0
    )

    task = XORTask(tissue)
    tissue.train_task(task, n_episodes=100)

    baseline = tissue.evaluate_task(task, n_trials=25)
    print(f"Baseline: {baseline['success_rate']*100:.1f}%")

    # Apply damage: remove random neurons
    damage_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for damage in damage_levels:
        # Create damaged copy
        n_remove = int(len(tissue.neurons) * damage)
        to_remove = np.random.choice(
            list(tissue.neurons.keys()),
            size=n_remove,
            replace=False
        )

        for nid in to_remove:
            if nid in tissue.neurons:
                tissue.neurons[nid].alive = False

        # Test performance
        perf = tissue.evaluate_task(task, n_trials=25)
        retention = perf['success_rate'] / baseline['success_rate']
        results.append((damage, retention))

        print(f"Damage {damage*100:.0f}%: Retention {retention*100:.1f}%")

        # Restore for next test
        for nid in tissue.neurons:
            tissue.neurons[nid].alive = True

    return results

results = test_neural_resilience()
```

### 5. Game Learning (DishBrain Replication)

oNeuro's dONN replicates Cortical Labs' DishBrain (Kagan et al. 2022) — the first demonstration that an ONN (Organic Neural Network) can learn to play Pong via the Free Energy Principle. The dONN extends this with pharmacological experiments impossible on real tissue:

```bash
# Replicate DishBrain Pong (5 experiments)
python3 demos/demo_dishbrain_pong.py

# Spatial Arena navigation (3 experiments, inspired by Doom's BSP level generation)
python3 demos/demo_doom_arena.py

# Run at GPU scale with multi-seed statistical analysis
python3 demos/demo_dishbrain_pong.py --scale medium --json results.json --runs 5
```

Key capabilities unique to dONN game learning:
- **Reversible drug application**: Train a brain, apply caffeine or diazepam, test — then remove the drug. Impossible on real neurons where drugs are irreversible.
- **Scale invariance**: Test learning at 1K, 5K, 25K, 100K neurons — far beyond what's feasible with biological tissue.
- **Protocol comparison**: Run identical brains with FEP, DA reward, or random feedback to isolate the learning mechanism.
- **Spatial navigation**: Extend 1D Pong to 25×25 Spatial Arenas (inspired by Doom's BSP level generation) with enemies, health pickups, and room navigation.

See `demos/demo_dishbrain_pong.py` and `demos/demo_doom_arena.py` for full implementations, and Tutorial 9 for the Spatial Arena walkthrough.

## Open Research Questions

1. **Consciousness Thresholds**: What Phi/complexity levels correspond to conscious experience?

2. **Quantum Effects**: Do quantum effects measurably improve computation?

3. **Optimal Growth**: What regulates neurogenesis timing and location?

4. **Sleep Function**: How do sleep-like states affect consolidation?

5. **Individual Differences**: Why do identical networks develop differently?

## Publishing and Reproducibility

### Recommended Benchmarks

```python
# benchmarks/standard_suite.py
def run_standard_benchmark():
    """Standard benchmark for reproducibility."""
    np.random.seed(42)

    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=30,
        energy_supply=2.0
    )

    task = XORTask(tissue)

    # Training
    stats = tissue.train_task(task, n_episodes=100)

    # Evaluation
    eval = tissue.evaluate_task(task, n_trials=50)

    return {
        'final_success_rate': eval['success_rate'],
        'neurons': len([n for n in tissue.neurons.values() if n.alive]),
        'synapses': len(tissue.synapses),
        'neurogenesis': tissue.neurogenesis_events
    }
```

## References

- Zador, A. (2019). "A critique of pure learning" - Nature Neuroscience
- Hassabis, D. et al. (2017). "Neuroscience-Inspired AI" - Neuron
- Marblestone, A. H. et al. (2016). "Toward an integration of deep learning and neuroscience" - Frontiers

## Summary

In this tutorial, you learned:

1. **Developmental Modeling**: Simulating brain development and critical periods
2. **Disease Modeling**: Alzheimer's, Parkinson's, and neurodegeneration
3. **Brain-Computer Interfaces**: Recording, decoding, and feedback
4. **Cognitive Robotics**: Embodied neural control
5. **Continual Learning**: Testing for catastrophic forgetting
6. **Neural Resilience**: Robustness to damage

The key insight is that organic neural networks are **research tools** as much as practical systems - they let us ask questions about neural computation that traditional networks cannot address. By modeling development, disease, and consciousness, we gain insights applicable to both neuroscience and AI.
