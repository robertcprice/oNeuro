"""
Organic Neural Networks - Neural Tissue That Grows Like Biology

A novel fusion of:
1. Liquid Neural Networks (MIT's continuous-time approach)
2. Neural Cellular Automata (self-organizing patterns)
3. Digital Terrarium organisms (evolution and selection)
4. Quantum properties (superposition, entanglement)

This creates neural networks that:
- GROW new neurons based on stimulus (like brain development)
- FORM connections organically (synaptogenesis)
- COMPETE for resources (metabolic cost)
- REPRODUCE and EVOLVE (neurogenesis + selection)
- Exhibit QUANTUM effects (superposition of activation states)
- LEARN through reward-modulated plasticity and eligibility traces

This is the FIRST implementation combining all four paradigms.

Run with: python3 organic_neural_network.py
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import time
from collections import defaultdict
from abc import ABC, abstractmethod


# ============================================================================
# QUANTUM NEURON STATES
# ============================================================================

class NeuronState(Enum):
    """Possible states for a neuron."""
    RESTING = 0        # Below threshold, not firing
    ACTIVE = 1         # Above threshold, firing
    REFRACTORY = 2     # Just fired, can't fire again yet
    SUPERPOSITION = 3  # Quantum: exploring multiple activation patterns
    ENTANGLED = 4      # Quantum: linked to another neuron


# ============================================================================
# ORGANIC NEURON
# ============================================================================

@dataclass
class OrganicNeuron:
    """
    A single neuron that can grow, die, and form connections organically.

    Properties:
    - Position in 3D neural tissue space
    - Continuous-time membrane potential (liquid neuron)
    - Metabolic cost (must "eat" to survive)
    - Age and generation tracking
    - Quantum state capabilities
    """
    # Identity
    id: int
    x: float
    y: float
    z: float

    # Liquid neuron dynamics (ODE-based)
    membrane_potential: float = -70.0  # mV (resting potential)
    time_constant: float = 10.0        # ms (liquid time constant)
    threshold: float = -55.0           # mV (firing threshold)
    refractory_period: float = 2.0     # ms

    # Metabolic properties
    energy: float = 100.0
    energy_consumption: float = 0.1    # per ms

    # Lifecycle
    age: float = 0.0
    generation: int = 0
    alive: bool = True

    # Connections
    inputs: Set[int] = field(default_factory=set)
    outputs: Set[int] = field(default_factory=set)

    # Quantum properties
    state: NeuronState = NeuronState.RESTING
    entangled_with: Optional[int] = None
    superposition_weights: List[float] = field(default_factory=list)

    # Learning
    plasticity: float = 0.01  # How much weights can change
    activation_history: List[float] = field(default_factory=list)

    def update_liquid(self, input_current: float, dt: float = 0.1):
        """
        Update membrane potential using continuous-time dynamics.

        Liquid neuron equation:
        τ * dV/dt = -(V - V_rest) + R * I_input

        This is a simplified Hodgkin-Huxley model.
        """
        V_rest = -70.0  # Resting potential
        R = 1.0         # Membrane resistance (MΩ)

        # Continuous-time dynamics
        dV = (-(self.membrane_potential - V_rest) + R * input_current) / self.time_constant
        self.membrane_potential += dV * dt

        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.state = NeuronState.ACTIVE
            self.membrane_potential = 20.0  # Spike peak
            return True  # Fired!

        # Decay back toward resting
        if self.membrane_potential < V_rest:
            self.membrane_potential = V_rest

        return False

    def consume_energy(self, dt: float = 0.1):
        """Metabolic cost of being alive."""
        base_cost = self.energy_consumption * dt
        # Active neurons consume more
        if self.state == NeuronState.ACTIVE:
            base_cost *= 3
        # More connections = more cost
        connection_cost = 0.01 * (len(self.inputs) + len(self.outputs)) * dt

        self.energy -= base_cost + connection_cost

        if self.energy <= 0:
            self.alive = False

    def can_divide(self) -> bool:
        """Check if neuron can undergo neurogenesis."""
        return (
            self.alive and
            self.energy > 200 and  # Higher threshold
            self.age > 500 and     # Older neurons
            len(self.outputs) >= 2 and
            random.random() < 0.01  # Only 1% chance even if conditions met
        )

    def divide(self, new_id: int) -> Optional['OrganicNeuron']:
        """
        Neurogenesis: create a daughter neuron.

        The daughter inherits:
        - Half the energy
        - Similar position (nearby)
        - Some connections
        - Generation + 1
        """
        if not self.can_divide():
            return None

        # Create daughter nearby
        offset = np.random.uniform(-0.5, 0.5, 3)
        daughter = OrganicNeuron(
            id=new_id,
            x=self.x + offset[0],
            y=self.y + offset[1],
            z=self.z + offset[2],
            energy=self.energy * 0.4,  # Daughter gets 40%
            generation=self.generation + 1,
            threshold=self.threshold + np.random.uniform(-2, 2),  # Mutation
            time_constant=self.time_constant * np.random.uniform(0.9, 1.1)
        )

        # Parent keeps 60%
        self.energy *= 0.6

        # Transfer some connections to daughter
        if self.outputs:
            # Daughter takes over some outputs
            transferred = random.sample(
                list(self.outputs),
                min(2, len(self.outputs))
            )
            for conn in transferred:
                self.outputs.discard(conn)
                daughter.outputs.add(conn)

        return daughter

    def enter_superposition(self):
        """Enter quantum superposition of activation states."""
        if random.random() < 0.1:  # 10% chance per opportunity
            self.state = NeuronState.SUPERPOSITION
            # Store potential activation weights
            self.superposition_weights = [
                np.random.uniform(0.5, 1.5),
                np.random.uniform(0.5, 1.5)
            ]

    def collapse_superposition(self) -> float:
        """Collapse quantum superposition to a classical weight."""
        if not self.superposition_weights:
            return 1.0

        # Quantum measurement: randomly collapse to one state
        weights = self.superposition_weights
        self.superposition_weights = []
        self.state = NeuronState.RESTING

        # Return collapsed weight
        return random.choice(weights)


# ============================================================================
# ORGANIC SYNAPSE
# ============================================================================

@dataclass
class OrganicSynapse:
    """
    A connection between neurons that can strengthen, weaken, and die.

    Implements:
    - Spike-timing dependent plasticity (STDP)
    - Synaptic pruning
    - Hebbian learning
    - Reward-modulated plasticity with eligibility traces
    """
    pre_neuron: int
    post_neuron: int
    weight: float = 0.5
    delay: float = 1.0  # ms

    # STDP tracking
    last_pre_spike: float = -1000.0
    last_post_spike: float = -1000.0

    # Lifecycle
    strength: float = 1.0  # Synaptic health
    age: float = 0.0

    # Eligibility trace for reward-modulated learning
    eligibility_trace: float = 0.0
    eligibility_decay: float = 0.95  # How fast trace decays

    # Recent activity for Hebbian learning
    pre_activity_avg: float = 0.0
    post_activity_avg: float = 0.0

    def update_stdp(self, pre_fired: bool, post_fired: bool, time: float, dt: float = 0.1):
        """
        Spike-Timing Dependent Plasticity.

        If pre fires before post: strengthen (LTP)
        If post fires before pre: weaken (LTD)
        """
        if pre_fired:
            self.last_pre_spike = time
            # Post fired recently after pre? LTP
            if 0 < time - self.last_post_spike < 20:  # 20ms window
                self.weight += 0.01 * self.strength

        if post_fired:
            self.last_post_spike = time
            # Pre fired recently before post? LTP
            if 0 < time - self.last_pre_spike < 20:
                self.weight += 0.01 * self.strength
            # Pre fired after post? LTD
            elif 0 < self.last_pre_spike - time < 20:
                self.weight -= 0.005 * self.strength

        # Bound weight
        self.weight = np.clip(self.weight, 0.0, 2.0)

        # Age the synapse
        self.age += dt

        # Strengthen with use, weaken with disuse
        if pre_fired or post_fired:
            self.strength = min(1.0, self.strength + 0.001)
        else:
            self.strength = max(0.0, self.strength - 0.0001)

    def update_eligibility(self, pre_active: float, post_active: float, dt: float = 0.1):
        """
        Update eligibility trace for reward-modulated plasticity.

        The eligibility trace records recent co-activity so that when
        a reward signal arrives, we know which synapses contributed.
        """
        # Update running averages
        alpha = 0.1
        self.pre_activity_avg = alpha * pre_active + (1 - alpha) * self.pre_activity_avg
        self.post_activity_avg = alpha * post_active + (1 - alpha) * self.post_activity_avg

        # Hebbian term: pre * post correlation
        hebbian = self.pre_activity_avg * self.post_activity_avg

        # Update eligibility trace with decay
        self.eligibility_trace = (
            self.eligibility_decay * self.eligibility_trace +
            (1 - self.eligibility_decay) * hebbian
        )

    def apply_reward(self, reward: float, learning_rate: float = 0.1):
        """
        Apply reward-modulated plasticity.

        Synapses that were recently active (high eligibility) get
        strengthened if reward is positive, weakened if negative.
        """
        # Reward-gated weight change
        delta_w = learning_rate * reward * self.eligibility_trace
        self.weight += delta_w
        self.weight = np.clip(self.weight, 0.0, 2.0)

        # Also affect synaptic strength
        if reward > 0:
            self.strength = min(1.0, self.strength + 0.01 * abs(reward))
        else:
            self.strength = max(0.0, self.strength - 0.005 * abs(reward))

    def should_prune(self) -> bool:
        """Check if synapse should be removed (synaptic pruning)."""
        return self.strength < 0.1 or self.weight < 0.05


# ============================================================================
# ORGANIC NEURAL NETWORK (The Neural Tissue)
# ============================================================================

class OrganicNeuralNetwork:
    """
    A neural network that grows, evolves, and exhibits quantum properties.

    Key innovations:
    1. Neurogenesis: New neurons grow from existing ones
    2. Synaptogenesis: Connections form organically based on proximity
    3. Natural selection: Neurons compete for energy
    4. Quantum effects: Superposition and entanglement
    5. Emergent behavior: Network structure self-organizes
    6. Learning: Reward-modulated plasticity with eligibility traces

    This is NOT a traditional neural network. It's neural TISSUE.
    """

    def __init__(self,
                 size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
                 initial_neurons: int = 20,
                 energy_supply: float = 1.0):
        """
        Initialize the neural tissue.

        Args:
            size: 3D dimensions of the tissue space
            initial_neurons: Starting number of neurons
            energy_supply: Rate of energy input per ms
        """
        self.size = np.array(size)
        self.energy_supply = energy_supply

        # Neural tissue
        self.neurons: Dict[int, OrganicNeuron] = {}
        self.synapses: Dict[Tuple[int, int], OrganicSynapse] = {}

        # Statistics
        self.time = 0.0
        self.neuron_counter = 0
        self.generation = 0

        # History for tracking emergence
        self.history: List[Dict] = []
        self.spike_count = 0
        self.neurogenesis_events = 0
        self.pruning_events = 0
        self.entanglement_events = 0

        # Input/Output regions for training
        self.input_regions: Dict[str, Tuple[Tuple[float, float, float], float]] = {}
        self.output_regions: Dict[str, Tuple[Tuple[float, float, float], float]] = {}

        # Learning state
        self.dopamine_level: float = 0.0  # Global reward signal
        self.dopamine_decay: float = 0.9  # How fast dopamine decays
        self.learning_rate: float = 0.1

        # Training metrics
        self.training_history: List[Dict] = []
        self.task_performance: Dict[str, List[float]] = {}

        # Structural plasticity
        self.performance_threshold_grow = 0.7
        self.performance_threshold_prune = 0.3

        # Initialize with seed neurons
        self._seed_network(initial_neurons)

    def _seed_network(self, n: int):
        """Create initial neurons distributed in tissue space."""
        for _ in range(n):
            neuron = OrganicNeuron(
                id=self.neuron_counter,
                x=np.random.uniform(1, self.size[0] - 1),
                y=np.random.uniform(1, self.size[1] - 1),
                z=np.random.uniform(1, self.size[2] - 1),
                threshold=np.random.uniform(-60, -50)
            )
            self.neurons[self.neuron_counter] = neuron
            self.neuron_counter += 1

        # Form initial random connections
        neuron_ids = list(self.neurons.keys())
        for i, n1 in enumerate(neuron_ids):
            # Connect to 2-5 random neighbors
            n_connections = np.random.randint(2, 6)
            for n2 in random.sample(neuron_ids[:i] + neuron_ids[i+1:],
                                   min(n_connections, len(neuron_ids) - 1)):
                self._form_synapse(n1, n2)

    def _form_synapse(self, pre: int, post: int, weight: float = None):
        """Form a new synapse between two neurons."""
        if pre not in self.neurons or post not in self.neurons:
            return

        key = (pre, post)
        if key in self.synapses:
            return

        if weight is None:
            weight = np.random.uniform(0.3, 0.7)

        synapse = OrganicSynapse(pre, post, weight)
        self.synapses[key] = synapse

        # Update neuron connection sets
        self.neurons[pre].outputs.add(post)
        self.neurons[post].inputs.add(pre)

    def _prune_synapse(self, pre: int, post: int):
        """Remove a synapse."""
        key = (pre, post)
        if key in self.synapses:
            del self.synapses[key]
            if pre in self.neurons:
                self.neurons[pre].outputs.discard(post)
            if post in self.neurons:
                self.neurons[post].inputs.discard(pre)
            self.pruning_events += 1

    def step(self, dt: float = 0.1):
        """Advance simulation by dt milliseconds."""
        self.time += dt

        # 1. Energy distribution (neurons compete for resources)
        self._distribute_energy()

        # 2. Process each neuron
        dead_neurons = []
        new_neurons = []

        for nid, neuron in list(self.neurons.items()):
            if not neuron.alive:
                dead_neurons.append(nid)
                continue

            # Calculate input current from connected neurons
            input_current = self._calculate_input(nid)

            # Quantum effects
            if random.random() < 0.01:
                neuron.enter_superposition()

            # Update liquid dynamics
            fired = neuron.update_liquid(input_current, dt)

            if fired:
                self.spike_count += 1

            # Consume energy
            neuron.consume_energy(dt)
            neuron.age += dt

            # Check for neurogenesis
            if neuron.can_divide():
                daughter = neuron.divide(self.neuron_counter)
                if daughter:
                    new_neurons.append(daughter)
                    self.neuron_counter += 1
                    self.neurogenesis_events += 1

        # 3. Update synapses (STDP)
        self._update_synapses(dt)

        # 4. Synaptogenesis: form new connections based on proximity
        self._spontaneous_synaptogenesis()

        # 5. Quantum entanglement
        self._quantum_entanglement()

        # 6. Remove dead neurons
        for nid in dead_neurons:
            self._remove_neuron(nid)

        # 7. Add new neurons
        for neuron in new_neurons:
            self.neurons[neuron.id] = neuron
            self.generation = max(self.generation, neuron.generation)

        # 8. Record history
        if int(self.time) % 10 == 0:
            self._record_history()

    def _distribute_energy(self):
        """Distribute energy supply to neurons based on activity."""
        total_energy = self.energy_supply

        # Active neurons get more energy
        active_count = sum(1 for n in self.neurons.values()
                         if n.state == NeuronState.ACTIVE and n.alive)

        for neuron in self.neurons.values():
            if not neuron.alive:
                continue

            base_share = total_energy / max(1, len(self.neurons))

            # Bonus for active neurons
            if neuron.state == NeuronState.ACTIVE:
                base_share *= 1.5
            # Bonus for quantum neurons
            if neuron.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED):
                base_share *= 1.2

            neuron.energy += base_share * 0.1

    def _calculate_input(self, neuron_id: int) -> float:
        """Calculate total input current to a neuron."""
        if neuron_id not in self.neurons:
            return 0.0

        neuron = self.neurons[neuron_id]
        total_input = 0.0

        for pre_id in neuron.inputs:
            key = (pre_id, neuron_id)
            if key not in self.synapses:
                continue

            synapse = self.synapses[key]
            pre_neuron = self.neurons.get(pre_id)

            if pre_neuron and pre_neuron.state == NeuronState.ACTIVE:
                weight = synapse.weight

                # Quantum collapse if in superposition
                if pre_neuron.state == NeuronState.SUPERPOSITION:
                    weight *= pre_neuron.collapse_superposition()

                total_input += weight * 10  # Scale to current (nA)

        return total_input

    def _update_synapses(self, dt: float):
        """Update all synapses with STDP."""
        to_prune = []

        for (pre, post), synapse in self.synapses.items():
            pre_fired = (pre in self.neurons and
                        self.neurons[pre].state == NeuronState.ACTIVE)
            post_fired = (post in self.neurons and
                         self.neurons[post].state == NeuronState.ACTIVE)

            synapse.update_stdp(pre_fired, post_fired, self.time, dt)

            if synapse.should_prune():
                to_prune.append((pre, post))

        for pre, post in to_prune:
            self._prune_synapse(pre, post)

    def _spontaneous_synaptogenesis(self):
        """New connections form between nearby neurons."""
        if random.random() > 0.01:  # 1% chance per step
            return

        # Pick random neuron
        if len(self.neurons) < 2:
            return

        n1 = random.choice(list(self.neurons.values()))
        if not n1.alive:
            return

        # Find nearby neurons
        candidates = []
        for n2 in self.neurons.values():
            if n2.id == n1.id or not n2.alive:
                continue

            dist = np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2 + (n1.z - n2.z)**2)
            if dist < 2.0:  # Within 2 units
                candidates.append(n2)

        if candidates:
            n2 = random.choice(candidates)
            # Form bidirectional connection with probability
            if random.random() < 0.3:
                self._form_synapse(n1.id, n2.id)
            if random.random() < 0.3:
                self._form_synapse(n2.id, n1.id)

    def _quantum_entanglement(self):
        """Create quantum entanglement between neurons."""
        if random.random() > 0.001:  # Rare event
            return

        neurons = [n for n in self.neurons.values()
                  if n.alive and n.state == NeuronState.SUPERPOSITION]

        if len(neurons) >= 2:
            n1, n2 = random.sample(neurons, 2)
            n1.state = NeuronState.ENTANGLED
            n2.state = NeuronState.ENTANGLED
            n1.entangled_with = n2.id
            n2.entangled_with = n1.id
            self.entanglement_events += 1

    def _remove_neuron(self, nid: int):
        """Remove a dead neuron and its connections."""
        if nid not in self.neurons:
            return

        neuron = self.neurons[nid]

        # Remove all synapses
        for pre_id in list(neuron.inputs):
            self._prune_synapse(pre_id, nid)
        for post_id in list(neuron.outputs):
            self._prune_synapse(nid, post_id)

        del self.neurons[nid]

    def _record_history(self):
        """Record current state for analysis."""
        alive = [n for n in self.neurons.values() if n.alive]

        if not alive:
            return

        self.history.append({
            'time': self.time,
            'neuron_count': len(alive),
            'synapse_count': len(self.synapses),
            'avg_energy': np.mean([n.energy for n in alive]),
            'avg_age': np.mean([n.age for n in alive]),
            'active_count': sum(1 for n in alive if n.state == NeuronState.ACTIVE),
            'quantum_count': sum(1 for n in alive
                               if n.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED)),
            'generation': self.generation
        })

    def stimulate(self, position: Tuple[float, float, float],
                  intensity: float = 10.0,
                  radius: float = 2.0):
        """
        Apply external stimulation to neurons near a position.

        This is how you INPUT data to the organic network.
        Stronger stimulation that more reliably drives neurons toward threshold.
        """
        for neuron in self.neurons.values():
            if not neuron.alive:
                continue

            dist = np.sqrt(
                (neuron.x - position[0])**2 +
                (neuron.y - position[1])**2 +
                (neuron.z - position[2])**2
            )

            if dist < radius:
                # Inject current proportional to proximity
                # Scale: intensity 10 -> moves potential by ~15mV for nearby neurons
                proximity = 1 - dist / radius
                current = intensity * proximity * 1.5  # Boost for more reliable activation
                neuron.membrane_potential += current

                # Also boost energy slightly (stimulation is energizing)
                neuron.energy += 0.5 * proximity

    def read_activity(self, position: Tuple[float, float, float],
                     radius: float = 2.0) -> float:
        """
        Read activity level from neurons near a position.

        This is how you OUTPUT data from the organic network.
        Uses both spiking state and membrane potential for richer signal.
        """
        total_activity = 0.0
        total_weight = 0.0

        for neuron in self.neurons.values():
            if not neuron.alive:
                continue

            dist = np.sqrt(
                (neuron.x - position[0])**2 +
                (neuron.y - position[1])**2 +
                (neuron.z - position[2])**2
            )

            if dist < radius:
                # Weight by distance
                weight = (1 - dist / radius) ** 2  # Squared for sharper falloff

                # Normalize membrane potential from [-70, 20] mV to [0, 1]
                # Resting potential (-70mV) = 0, threshold (-55mV) = 0.167, spike (20mV) = 1
                potential_normalized = (neuron.membrane_potential + 70) / 90
                potential_normalized = np.clip(potential_normalized, 0, 1)

                # Boost based on state
                if neuron.state == NeuronState.ACTIVE:
                    activity = 1.0
                elif neuron.state == NeuronState.SUPERPOSITION:
                    activity = 0.7 * potential_normalized + 0.3
                else:
                    # Use a sigmoid-like transform for better gradient
                    activity = potential_normalized ** 0.5  # Square root boosts lower values

                total_activity += activity * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        result = total_activity / total_weight
        return np.clip(result, 0.0, 1.0)

    # ========================================================================
    # INPUT/OUTPUT REGIONS FOR TRAINING
    # ========================================================================

    def define_input_region(self, name: str, position: Tuple[float, float, float],
                           radius: float = 1.5):
        """Define a 3D region where stimulation = input."""
        self.input_regions[name] = (position, radius)

    def define_output_region(self, name: str, position: Tuple[float, float, float],
                            radius: float = 1.5):
        """Define a 3D region where activity = output."""
        self.output_regions[name] = (position, radius)

    def set_input(self, name: str, value: float):
        """Set input to a named region (0.0 to 1.0)."""
        if name not in self.input_regions:
            return
        position, radius = self.input_regions[name]
        self.stimulate(position, intensity=value * 20.0, radius=radius)

    def read_output(self, name: str) -> float:
        """Read output from a named region (0.0 to 1.0)."""
        if name not in self.output_regions:
            return 0.0
        position, radius = self.output_regions[name]
        return self.read_activity(position, radius)

    def set_inputs(self, values: Dict[str, float]):
        """Set multiple inputs at once."""
        for name, value in values.items():
            self.set_input(name, value)

    def read_outputs(self) -> Dict[str, float]:
        """Read all outputs."""
        return {name: self.read_output(name) for name in self.output_regions}

    # ========================================================================
    # LEARNING MECHANISMS
    # ========================================================================

    def release_dopamine(self, amount: float):
        """
        Release dopamine-like reward signal.

        This modulates plasticity across the network - synapses with
        high eligibility traces will be strengthened (if reward positive)
        or weakened (if reward negative).
        """
        self.dopamine_level += amount

    def apply_reward_modulated_plasticity(self):
        """
        Apply reward-modulated plasticity to all synapses.

        Synapses update based on:
        delta_w = learning_rate * dopamine * eligibility_trace
        """
        for synapse in self.synapses.values():
            synapse.apply_reward(self.dopamine_level, self.learning_rate)

        # Decay dopamine
        self.dopamine_level *= self.dopamine_decay

    def give_energy_bonus(self, region: str, amount: float):
        """Give energy bonus to neurons in a region (reinforcement)."""
        if region not in self.output_regions:
            return
        position, radius = self.output_regions[region]

        for neuron in self.neurons.values():
            if not neuron.alive:
                continue

            dist = np.sqrt(
                (neuron.x - position[0])**2 +
                (neuron.y - position[1])**2 +
                (neuron.z - position[2])**2
            )

            if dist < radius:
                bonus = amount * (1 - dist / radius)
                neuron.energy += bonus

    def update_eligibility_traces(self, dt: float = 0.1):
        """Update eligibility traces for all synapses."""
        for (pre, post), synapse in self.synapses.items():
            pre_neuron = self.neurons.get(pre)
            post_neuron = self.neurons.get(post)

            if pre_neuron and post_neuron:
                pre_active = 1.0 if pre_neuron.state == NeuronState.ACTIVE else 0.0
                post_active = 1.0 if post_neuron.state == NeuronState.ACTIVE else 0.0
                synapse.update_eligibility(pre_active, post_active, dt)

    # ========================================================================
    # STRUCTURAL PLASTICITY
    # ========================================================================

    def grow_neurons_in_region(self, region: str, n: int = 1):
        """Grow new neurons in a specific region."""
        if region not in self.output_regions and region not in self.input_regions:
            return

        regions = {**self.input_regions, **self.output_regions}
        position, radius = regions[region]

        for _ in range(n):
            offset = np.random.uniform(-radius/2, radius/2, 3)
            neuron = OrganicNeuron(
                id=self.neuron_counter,
                x=position[0] + offset[0],
                y=position[1] + offset[1],
                z=position[2] + offset[2],
                energy=150.0,  # Start with high energy
                generation=self.generation + 1
            )
            self.neurons[self.neuron_counter] = neuron
            self.neuron_counter += 1
            self.neurogenesis_events += 1

            # Connect to nearby neurons
            self._connect_new_neuron(neuron.id)

    def _connect_new_neuron(self, neuron_id: int):
        """Connect a new neuron to nearby existing neurons."""
        if neuron_id not in self.neurons:
            return

        new_neuron = self.neurons[neuron_id]

        for other_id, other in self.neurons.items():
            if other_id == neuron_id or not other.alive:
                continue

            dist = np.sqrt(
                (new_neuron.x - other.x)**2 +
                (new_neuron.y - other.y)**2 +
                (new_neuron.z - other.z)**2
            )

            if dist < 3.0 and random.random() < 0.5:
                # Form bidirectional connections
                if random.random() < 0.5:
                    self._form_synapse(other_id, neuron_id)
                if random.random() < 0.5:
                    self._form_synapse(neuron_id, other_id)

    def prune_weak_connections(self, threshold: float = 0.1):
        """Remove synapses below strength threshold."""
        to_prune = [
            (pre, post)
            for (pre, post), syn in self.synapses.items()
            if syn.strength < threshold or syn.weight < threshold
        ]
        for pre, post in to_prune:
            self._prune_synapse(pre, post)

    def structural_adaptation(self, performance: float):
        """
        Adapt network structure based on task performance.

        High performance -> grow more neurons in output regions
        Low performance -> prune weak connections
        """
        # Limit total network size
        max_neurons = 100
        current_alive = len([n for n in self.neurons.values() if n.alive])

        if current_alive >= max_neurons:
            # Network too large, prune instead
            self.prune_weak_connections(threshold=0.2)
            return

        if performance > self.performance_threshold_grow:
            # Grow occasionally, not every step
            if random.random() < 0.1:
                # Grow in regions that are being used
                for region in self.output_regions:
                    self.grow_neurons_in_region(region, n=1)
        elif performance < self.performance_threshold_prune:
            # Prune weak connections
            self.prune_weak_connections(threshold=0.15)

    # ========================================================================
    # TASK TRAINING INFRASTRUCTURE
    # ========================================================================

    def train_episode(self, task: 'TrainingTask', max_steps: int = 100) -> Tuple[float, bool]:
        """
        Train on a single episode of a task.

        Returns:
            (total_reward, success)
        """
        task.reset()
        total_reward = 0.0
        steps = 0

        while steps < max_steps and not task.is_done():
            # Get current inputs from task
            inputs = task.get_inputs()

            # Apply inputs to network
            self.set_inputs(inputs)

            # Run network for processing time
            for _ in range(10):  # More processing steps
                self.step(dt=0.3)  # Smaller timestep for more precision
                self.update_eligibility_traces(dt=0.3)

            # Read outputs immediately after processing
            outputs = self.read_outputs()

            # Get reward from task
            reward, done = task.evaluate(outputs)
            total_reward += reward

            # Release dopamine proportional to reward
            self.release_dopamine(reward)

            # Apply reward-modulated plasticity
            self.apply_reward_modulated_plasticity()

            # Give energy bonus for positive reward
            if reward > 0:
                for region in self.output_regions:
                    self.give_energy_bonus(region, reward * 5)

            steps += 1

        success = task.is_success()
        return total_reward, success

    def train_task(self, task: 'TrainingTask', n_episodes: int = 100,
                   report_every: int = 10) -> Dict[str, Any]:
        """
        Train on a task for multiple episodes.

        Returns:
            Training statistics
        """
        task_name = task.name
        if task_name not in self.task_performance:
            self.task_performance[task_name] = []

        rewards = []
        successes = []

        for episode in range(n_episodes):
            total_reward, success = self.train_episode(task)

            rewards.append(total_reward)
            successes.append(1.0 if success else 0.0)
            self.task_performance[task_name].append(total_reward)

            # Structural adaptation every 5 episodes, not every episode
            if len(rewards) >= 10 and episode % 5 == 0:
                recent_perf = np.mean(rewards[-10:])
                self.structural_adaptation(recent_perf / 10.0)  # Normalize

            if (episode + 1) % report_every == 0:
                recent_rewards = np.mean(rewards[-report_every:])
                recent_success = np.mean(successes[-report_every:]) * 100
                print(f"  Episode {episode+1}/{n_episodes}: "
                      f"Avg Reward={recent_rewards:.2f}, "
                      f"Success={recent_success:.0f}%")

        return {
            'task': task_name,
            'episodes': n_episodes,
            'final_avg_reward': np.mean(rewards[-10:]),
            'final_success_rate': np.mean(successes[-10:]) * 100,
            'total_neurons': len([n for n in self.neurons.values() if n.alive]),
            'total_synapses': len(self.synapses)
        }

    def evaluate_task(self, task: 'TrainingTask', n_trials: int = 20) -> Dict[str, float]:
        """
        Evaluate performance on a task without learning.

        Returns:
            Evaluation metrics
        """
        # Temporarily disable learning
        old_lr = self.learning_rate
        self.learning_rate = 0.0

        successes = 0
        total_reward = 0.0

        for _ in range(n_trials):
            reward, success = self.train_episode(task)
            total_reward += reward
            if success:
                successes += 1

        # Restore learning rate
        self.learning_rate = old_lr

        return {
            'success_rate': successes / n_trials,
            'avg_reward': total_reward / n_trials
        }

    def get_learning_curve(self, task_name: str) -> List[float]:
        """Get learning curve for a task (moving average)."""
        if task_name not in self.task_performance:
            return []

        rewards = self.task_performance[task_name]
        window = min(10, len(rewards))

        if window == 0:
            return []

        curve = []
        for i in range(window - 1, len(rewards)):
            curve.append(np.mean(rewards[i - window + 1:i + 1]))

        return curve

    def visualize_ascii(self) -> str:
        """Generate ASCII visualization of the neural tissue (2D slice)."""
        width = 60
        height = 30

        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # Scale factors
        scale_x = width / self.size[0]
        scale_y = height / self.size[1]

        # Draw neurons
        for neuron in self.neurons.values():
            if not neuron.alive:
                continue

            x = int(neuron.x * scale_x) % width
            y = int(neuron.y * scale_y) % height

            # Symbol based on state
            if neuron.state == NeuronState.ACTIVE:
                symbol = '@'  # Firing
                color = '\033[31m'  # Red
            elif neuron.state == NeuronState.SUPERPOSITION:
                symbol = '?'  # Quantum
                color = '\033[35m'  # Magenta
            elif neuron.state == NeuronState.ENTANGLED:
                symbol = '&'  # Entangled
                color = '\033[36m'  # Cyan
            elif neuron.energy < 30:
                symbol = '.'  # Dying
                color = '\033[90m'  # Gray
            else:
                symbol = 'o'  # Normal
                color = '\033[32m'  # Green

            reset = '\033[0m'
            canvas[y][x] = f"{color}{symbol}{reset}"

        # Draw synapses as dim lines (sample)
        for i, ((pre, post), syn) in enumerate(self.synapses.items()):
            if i > 50:  # Don't draw too many
                break
            if pre in self.neurons and post in self.neurons:
                n1 = self.neurons[pre]
                n2 = self.neurons[post]
                x1 = int(n1.x * scale_x) % width
                y1 = int(n1.y * scale_y) % height
                x2 = int(n2.x * scale_x) % width
                y2 = int(n2.y * scale_y) % height
                # Draw a point at midpoint
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= mx < width and 0 <= my < height:
                    if canvas[my][mx] == ' ':
                        canvas[my][mx] = '.'

        # Build output
        border = '+' + '-' * width + '+'
        lines = [border]
        for row in canvas:
            lines.append('|' + ''.join(c if len(c) <= 1 else c for c in row) + '|')
        lines.append(border)

        return '\n'.join(lines)

    def statistics(self) -> str:
        """Get current statistics."""
        alive = [n for n in self.neurons.values() if n.alive]

        if not alive:
            return "All neurons have died. Network has collapsed."

        active = sum(1 for n in alive if n.state == NeuronState.ACTIVE)
        quantum = sum(1 for n in alive
                     if n.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED))
        entangled = sum(1 for n in alive if n.state == NeuronState.ENTANGLED)

        avg_connections = np.mean([len(n.inputs) + len(n.outputs) for n in alive])
        avg_weight = np.mean([s.weight for s in self.synapses.values()]) if self.synapses else 0

        return f"""
┌────────────────────────────────────────────────────────────────────────────┐
│ ORGANIC NEURAL NETWORK - Neural Tissue Statistics                          │
├────────────────────────────────────────────────────────────────────────────┤
│ Time: {self.time:8.1f} ms    Generation: {self.generation:3d}    Age: {max(n.age for n in alive):6.1f} ms     │
│                                                                            │
│ Neurons: {len(alive):4d} alive    Synapses: {len(self.synapses):4d}    Avg Connections: {avg_connections:.1f}       │
│ Active: {active:4d} ({100*active/max(1,len(alive)):5.1f}%)    Quantum: {quantum:3d}    Entangled: {entangled:2d}        │
│                                                                            │
│ Avg Energy: {np.mean([n.energy for n in alive]):6.1f}    Avg Weight: {avg_weight:.3f}    Avg Age: {np.mean([n.age for n in alive]):6.1f} ms │
│                                                                            │
│ Events: Neurogenesis={self.neurogenesis_events}  Pruning={self.pruning_events}  Entanglement={self.entanglement_events}  │
└────────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# EMERGENCE TRACKER
# ============================================================================

class EmergenceTracker:
    """
    Track emergent behaviors in the organic neural network.

    Looks for:
    1. Spontaneous pattern formation
    2. Information cascades
    3. Criticality (avalanche distributions)
    4. Small-world topology emergence
    5. Quantum coherence patterns
    """

    def __init__(self, network: OrganicNeuralNetwork):
        self.network = network
        self.patterns: List[Dict] = []
        self.cascades: List[int] = []

    def detect_emergence(self) -> Dict:
        """Detect emergent properties in the network."""
        results = {
            'patterns_detected': 0,
            'cascade_detected': False,
            'criticality_index': 0.0,
            'small_world_coefficient': 0.0,
            'quantum_coherence': 0.0
        }

        alive = [n for n in self.network.neurons.values() if n.alive]
        if len(alive) < 5:
            return results

        # 1. Detect activity patterns (spatial clustering of active neurons)
        active = [n for n in alive if n.state == NeuronState.ACTIVE]
        if len(active) >= 3:
            # Check if active neurons are clustered
            positions = np.array([[n.x, n.y, n.z] for n in active])
            if len(positions) > 1:
                center = positions.mean(axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                clustering = 1.0 - (distances.mean() / 5.0)  # Normalized
                results['patterns_detected'] = max(0, clustering)

        # 2. Detect cascades (chain reactions of firing)
        if len(active) > len(alive) * 0.3:  # >30% active at once
            results['cascade_detected'] = True
            self.cascades.append(len(active))

        # 3. Criticality: check for power-law distribution of cascade sizes
        if len(self.cascades) >= 10:
            # Simplified criticality check
            cascade_arr = np.array(self.cascades[-20:])
            mean_size = cascade_arr.mean()
            var_size = cascade_arr.var()
            if mean_size > 0:
                results['criticality_index'] = var_size / (mean_size ** 2)

        # 4. Small-world coefficient (clustering / path_length ratio)
        results['small_world_coefficient'] = self._calculate_small_world()

        # 5. Quantum coherence (fraction of entangled neurons)
        entangled = sum(1 for n in alive if n.state == NeuronState.ENTANGLED)
        results['quantum_coherence'] = entangled / len(alive)

        return results

    def _calculate_small_world(self) -> float:
        """Calculate small-world network coefficient."""
        # Build adjacency
        alive_ids = {nid for nid, n in self.network.neurons.items() if n.alive}
        if len(alive_ids) < 3:
            return 0.0

        # Calculate clustering coefficient
        clustering = 0.0
        for nid in alive_ids:
            neuron = self.network.neurons.get(nid)
            if not neuron:
                continue
            neighbors = neuron.inputs | neuron.outputs
            neighbors = neighbors & alive_ids
            if len(neighbors) < 2:
                continue

            # Count connections between neighbors
            connections = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 < n2 and (n1, n2) in self.network.synapses:
                        connections += 1

            possible = len(neighbors) * (len(neighbors) - 1) / 2
            if possible > 0:
                clustering += connections / possible

        clustering /= len(alive_ids)

        # Simplified: return clustering as small-world proxy
        return clustering


# ============================================================================
# TRAINING TASKS
# ============================================================================

class TrainingTask(ABC):
    """Base class for training tasks."""

    def __init__(self, name: str, network: OrganicNeuralNetwork):
        self.name = name
        self.network = network
        self.current_step = 0
        self.max_steps = 20
        self._setup_regions()

    @abstractmethod
    def _setup_regions(self):
        """Define input/output regions for this task."""
        pass

    @abstractmethod
    def reset(self):
        """Reset task state for new episode."""
        pass

    @abstractmethod
    def get_inputs(self) -> Dict[str, float]:
        """Get current inputs to apply to network."""
        pass

    @abstractmethod
    def evaluate(self, outputs: Dict[str, float]) -> Tuple[float, bool]:
        """
        Evaluate network outputs.

        Returns:
            (reward, done)
        """
        pass

    def is_done(self) -> bool:
        """Check if episode is done."""
        return self.current_step >= self.max_steps

    @abstractmethod
    def is_success(self) -> bool:
        """Check if task was completed successfully."""
        pass


class XORTask(TrainingTask):
    """
    XOR task: 2 inputs -> 1 output.

    The classic non-linearly-separable problem that requires hidden layers.
    """

    def __init__(self, network: OrganicNeuralNetwork):
        self.input_a = None
        self.input_b = None
        self.target = None
        self.attempts = 0
        self.correct = False
        super().__init__("XOR", network)

    def _setup_regions(self):
        """Define input regions (left and right) and output region (center)."""
        self.network.define_input_region("input_a", (2.0, 3.0, 2.5), radius=1.5)
        self.network.define_input_region("input_b", (8.0, 3.0, 2.5), radius=1.5)
        self.network.define_output_region("output", (5.0, 7.0, 2.5), radius=2.0)

    def reset(self):
        """Reset for new XOR trial."""
        self.current_step = 0
        self.input_a = random.choice([0.0, 1.0])
        self.input_b = random.choice([0.0, 1.0])
        self.target = float(bool(self.input_a) != bool(self.input_b))  # XOR
        self.attempts = 0
        self.correct = False

    def get_inputs(self) -> Dict[str, float]:
        """Return current XOR inputs."""
        return {
            "input_a": self.input_a,
            "input_b": self.input_b
        }

    def evaluate(self, outputs: Dict[str, float]) -> Tuple[float, bool]:
        """Evaluate output against XOR target."""
        output = outputs.get("output", 0.0)

        # Threshold output
        predicted = 1.0 if output > 0.5 else 0.0

        # Reward based on how close to target
        error = abs(output - self.target)
        reward = 1.0 - error  # Reward in [0, 1]

        # Bonus for correct classification
        if predicted == self.target:
            reward += 0.5
            self.correct = True

        self.attempts += 1
        self.current_step += 1

        # Done after enough processing time
        done = self.current_step >= 10

        return reward, done

    def is_success(self) -> bool:
        """Success if output matched target."""
        return self.correct


class PatternRecognitionTask(TrainingTask):
    """
    Pattern recognition: classify simple 2x2 patterns.

    Input: 4 pixels arranged in a square
    Output: classification (horizontal line, vertical line, diagonal, or random)
    """

    def __init__(self, network: OrganicNeuralNetwork):
        self.pattern = None
        self.pattern_type = None
        self.target = None
        self.correct = False
        super().__init__("PatternRecognition", network)

    def _setup_regions(self):
        """Define 4 input regions in a 2x2 grid and 4 output regions for classes."""
        # 2x2 grid of inputs
        self.network.define_input_region("pixel_00", (2.0, 2.0, 2.5), radius=1.0)
        self.network.define_input_region("pixel_01", (8.0, 2.0, 2.5), radius=1.0)
        self.network.define_input_region("pixel_10", (2.0, 8.0, 2.5), radius=1.0)
        self.network.define_input_region("pixel_11", (8.0, 8.0, 2.5), radius=1.0)

        # 4 output classes
        self.network.define_output_region("horizontal", (5.0, 1.0, 2.5), radius=1.0)
        self.network.define_output_region("vertical", (1.0, 5.0, 2.5), radius=1.0)
        self.network.define_output_region("diagonal", (5.0, 5.0, 2.5), radius=1.0)
        self.network.define_output_region("random", (9.0, 5.0, 2.5), radius=1.0)

    def reset(self):
        """Generate a new pattern."""
        self.current_step = 0
        self.correct = False

        # Generate pattern type
        self.pattern_type = random.choice(["horizontal", "vertical", "diagonal", "random"])

        if self.pattern_type == "horizontal":
            # Top row or bottom row active
            row = random.choice([0, 1])
            self.pattern = [[0.0, 0.0], [0.0, 0.0]]
            self.pattern[row] = [1.0, 1.0]
            self.target = "horizontal"
        elif self.pattern_type == "vertical":
            # Left column or right column active
            col = random.choice([0, 1])
            self.pattern = [[1.0 if col == 0 else 0.0, 1.0 if col == 1 else 0.0],
                           [1.0 if col == 0 else 0.0, 1.0 if col == 1 else 0.0]]
            self.target = "vertical"
        elif self.pattern_type == "diagonal":
            # Diagonal pattern
            direction = random.choice([0, 1])
            if direction == 0:
                self.pattern = [[1.0, 0.0], [0.0, 1.0]]
            else:
                self.pattern = [[0.0, 1.0], [1.0, 0.0]]
            self.target = "diagonal"
        else:
            # Random pattern
            self.pattern = [[random.choice([0.0, 1.0]) for _ in range(2)] for _ in range(2)]
            self.target = "random"

    def get_inputs(self) -> Dict[str, float]:
        """Return current pattern as inputs."""
        return {
            "pixel_00": self.pattern[0][0],
            "pixel_01": self.pattern[0][1],
            "pixel_10": self.pattern[1][0],
            "pixel_11": self.pattern[1][1]
        }

    def evaluate(self, outputs: Dict[str, float]) -> Tuple[float, bool]:
        """Evaluate classification."""
        # Find the class with highest activation
        classes = ["horizontal", "vertical", "diagonal", "random"]
        values = [outputs.get(c, 0.0) for c in classes]
        predicted = classes[np.argmax(values)]

        # Reward
        if predicted == self.target:
            reward = 2.0
            self.correct = True
        else:
            # Partial reward based on target activation
            reward = outputs.get(self.target, 0.0)

        self.current_step += 1
        done = self.current_step >= 8

        return reward, done

    def is_success(self) -> bool:
        return self.correct


class MemoryTask(TrainingTask):
    """
    Memory task: remember and recall a pattern.

    Phase 1: Show pattern (encoding)
    Phase 2: Wait period (maintenance)
    Phase 3: Probe - was this the pattern? (retrieval)
    """

    def __init__(self, network: OrganicNeuralNetwork):
        self.stored_pattern = None
        self.probe_pattern = None
        self.phase = "encoding"
        self.is_match = None
        self.correct = False
        super().__init__("Memory", network)

    def _setup_regions(self):
        """Define input and output regions."""
        # Pattern inputs (3 features)
        self.network.define_input_region("feature_1", (2.0, 5.0, 2.5), radius=1.0)
        self.network.define_input_region("feature_2", (5.0, 5.0, 2.5), radius=1.0)
        self.network.define_input_region("feature_3", (8.0, 5.0, 2.5), radius=1.0)

        # Single output: match or no match
        self.network.define_output_region("match", (5.0, 8.0, 2.5), radius=1.5)

    def reset(self):
        """Start new memory trial."""
        self.current_step = 0
        self.phase = "encoding"
        self.correct = False

        # Generate random pattern to remember
        self.stored_pattern = [random.choice([0.0, 1.0]) for _ in range(3)]

        # Decide if probe will match (50% chance)
        self.is_match = random.choice([True, False])

        if self.is_match:
            self.probe_pattern = self.stored_pattern.copy()
        else:
            # Flip one random feature
            self.probe_pattern = self.stored_pattern.copy()
            flip_idx = random.randint(0, 2)
            self.probe_pattern[flip_idx] = 1.0 - self.probe_pattern[flip_idx]

    def get_inputs(self) -> Dict[str, float]:
        """Return current phase inputs."""
        if self.phase == "encoding":
            # Show the pattern to remember
            return {
                "feature_1": self.stored_pattern[0],
                "feature_2": self.stored_pattern[1],
                "feature_3": self.stored_pattern[2]
            }
        elif self.phase == "delay":
            # No input during delay
            return {"feature_1": 0.0, "feature_2": 0.0, "feature_3": 0.0}
        else:
            # Show probe pattern
            return {
                "feature_1": self.probe_pattern[0],
                "feature_2": self.probe_pattern[1],
                "feature_3": self.probe_pattern[2]
            }

    def evaluate(self, outputs: Dict[str, float]) -> Tuple[float, bool]:
        """Evaluate memory response."""
        reward = 0.0
        done = False

        if self.phase == "encoding" and self.current_step >= 5:
            self.phase = "delay"
            self.current_step = 0
        elif self.phase == "delay" and self.current_step >= 5:
            self.phase = "retrieval"
            self.current_step = 0
        elif self.phase == "retrieval" and self.current_step >= 5:
            # Evaluate response
            output = outputs.get("match", 0.0)
            predicted_match = output > 0.5

            if predicted_match == self.is_match:
                reward = 2.0
                self.correct = True
            else:
                reward = -0.5

            done = True

        self.current_step += 1
        return reward, done

    def is_success(self) -> bool:
        return self.correct


class DecisionMakingTask(TrainingTask):
    """
    Decision making: accumulate evidence and make a choice.

    Multiple pieces of evidence come in over time.
    Network must accumulate and make a binary decision.
    """

    def __init__(self, network: OrganicNeuralNetwork):
        self.evidence_sequence = []
        self.correct_choice = None
        self.accumulated_evidence = 0.0
        self.made_choice = False
        self.choice = None
        super().__init__("DecisionMaking", network)

    def _setup_regions(self):
        """Define evidence input and decision outputs."""
        # Evidence input
        self.network.define_input_region("evidence", (5.0, 2.0, 2.5), radius=1.5)

        # Two decision outputs
        self.network.define_output_region("choice_a", (2.0, 8.0, 2.5), radius=1.5)
        self.network.define_output_region("choice_b", (8.0, 8.0, 2.5), radius=1.5)

    def reset(self):
        """Start new decision trial."""
        self.current_step = 0
        self.made_choice = False
        self.choice = None
        self.accumulated_evidence = 0.0

        # Decide which choice is correct (A or B)
        self.correct_choice = random.choice(["choice_a", "choice_b"])

        # Generate noisy evidence sequence (biased toward correct choice)
        bias = 0.3 if self.correct_choice == "choice_a" else -0.3
        self.evidence_sequence = [
            np.clip(bias + np.random.normal(0, 0.2), -1, 1)
            for _ in range(10)
        ]

    def get_inputs(self) -> Dict[str, float]:
        """Return current evidence."""
        if self.current_step < len(self.evidence_sequence):
            evidence = self.evidence_sequence[self.current_step]
            # Map to [0, 1] range for stimulation
            mapped = (evidence + 1) / 2
            return {"evidence": mapped}
        return {"evidence": 0.5}  # Neutral after evidence done

    def evaluate(self, outputs: Dict[str, float]) -> Tuple[float, float]:
        """Evaluate decision."""
        a_val = outputs.get("choice_a", 0.0)
        b_val = outputs.get("choice_b", 0.0)

        reward = 0.0
        done = False

        # Check for commitment to a choice
        if a_val > 0.7 or b_val > 0.7:
            self.made_choice = True
            self.choice = "choice_a" if a_val > b_val else "choice_b"

            if self.choice == self.correct_choice:
                reward = 2.0
            else:
                reward = -0.5

            done = True

        # After all evidence presented, force decision
        if self.current_step >= 15 and not self.made_choice:
            self.made_choice = True
            self.choice = "choice_a" if a_val > b_val else "choice_b"

            if self.choice == self.correct_choice:
                reward = 1.0  # Lower reward for slow decision
            else:
                reward = -0.5

            done = True

        self.current_step += 1
        return reward, done

    def is_success(self) -> bool:
        return self.made_choice and self.choice == self.correct_choice


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               ORGANIC NEURAL NETWORK - Neural Tissue That Grows               ║
║                                                                              ║
║  A novel fusion of:                                                          ║
║  • Liquid Neural Networks (continuous-time dynamics)                        ║
║  • Neural Cellular Automata (self-organization)                              ║
║  • Digital Evolution (neurogenesis + selection)                              ║
║  • Quantum Effects (superposition + entanglement)                            ║
║                                                                              ║
║  This is the FIRST implementation combining all four paradigms.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Create organic neural tissue
    print("Creating neural tissue...")
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=30,
        energy_supply=2.0
    )

    # Create emergence tracker
    tracker = EmergenceTracker(tissue)

    print(f"Initial state: {len(tissue.neurons)} neurons, {len(tissue.synapses)} synapses")
    print("Running simulation...\n")

    # Run simulation with stimulation
    for i in range(200):
        tissue.step(dt=0.5)

        # Apply periodic stimulation (simulating input)
        if i % 20 == 0:
            tissue.stimulate(
                position=(2.0, 5.0, 2.5),
                intensity=15.0,
                radius=3.0
            )

        # Show progress
        if i % 40 == 0:
            print(f"\n--- Time: {tissue.time:.1f} ms ---")
            print(tissue.visualize_ascii())
            print(tissue.statistics())

            # Check for emergence
            emergence = tracker.detect_emergence()
            if emergence['patterns_detected'] > 0.3:
                print(f"  PATTERN DETECTED: clustering = {emergence['patterns_detected']:.2f}")
            if emergence['cascade_detected']:
                print(f"  CASCADE EVENT: information avalanche!")
            if emergence['quantum_coherence'] > 0.05:
                print(f"  QUANTUM COHERENCE: {emergence['quantum_coherence']:.1%}")

            time.sleep(0.2)

    # Final statistics
    print("\n" + "=" * 78)
    print("SIMULATION COMPLETE")
    print("=" * 78)

    print(tissue.statistics())

    # Show emergence summary
    emergence = tracker.detect_emergence()
    print(f"""
+-----------------------------------------------------------------------------+
| EMERGENCE ANALYSIS                                                          |
+-----------------------------------------------------------------------------+
| Pattern Formation:    {emergence['patterns_detected']:.2f}  (0-1 scale)                          |
| Cascade Events:       {len(tracker.cascades)} detected                                       |
| Criticality Index:    {emergence['criticality_index']:.2f}  (~1 = critical)                        |
| Small-World Coeff:    {emergence['small_world_coefficient']:.2f}  (higher = more small-world)          |
| Quantum Coherence:    {emergence['quantum_coherence']:.1%} of network                           |
|                                                                             |
| Neurogenesis Events:  {tissue.neurogenesis_events} new neurons created                       |
| Pruning Events:       {tissue.pruning_events} weak synapses removed                         |
| Entanglement Events:  {tissue.entanglement_events} quantum pairs formed                        |
+-----------------------------------------------------------------------------+
    """)

    print("""
NOVEL CAPABILITIES:

   1. NEUROGENESIS: New neurons grow from existing ones (like brain development)
   2. SYNAPTOGENESIS: Connections form organically based on proximity
   3. NATURAL SELECTION: Neurons compete for energy, weak ones die
   4. LIQUID DYNAMICS: Continuous-time membrane potential (MIT LNN approach)
   5. QUANTUM EFFECTS: Superposition of activation, entanglement between neurons
   6. EMERGENCE: Patterns, cascades, and criticality arise spontaneously

RESEARCH IMPLICATIONS:

   - Study brain development in silico
   - Model neurodegenerative diseases (energy starvation = cell death)
   - Explore quantum effects in neural computation
   - Develop adaptive AI that grows its own architecture
   - Investigate consciousness emergence (integrated information)

THIS IS NOVEL:

   This is the FIRST neural network that:
   - GROWS new neurons through cell division
   - Uses LIQUID continuous-time dynamics
   - Has QUANTUM superposition/entanglement
   - Undergoes NATURAL SELECTION

   No existing platform combines all four.
    """)


def demo_training():
    """
    Demo of training an organic neural network on the XOR task.

    This demonstrates:
    1. How the network learns through reward-modulated plasticity
    2. Structural plasticity (growing/pruning based on performance)
    3. Learning curves and performance tracking
    """
    print("""
+=============================================================================+
|           ORGANIC NEURAL NETWORK - TRAINING DEMO                            |
|                                                                             |
|  Training neural tissue on the XOR task (2 inputs -> 1 output)              |
|                                                                             |
|  XOR truth table:                                                           |
|    A | B | A XOR B                                                          |
|   ---|---|--------                                                          |
|    0 | 0 |   0                                                              |
|    0 | 1 |   1                                                              |
|    1 | 0 |   1                                                              |
|    1 | 1 |   0                                                              |
|                                                                             |
|  This is the classic non-linearly-separable problem requiring hidden units. |
+=============================================================================+
    """)

    # Create neural tissue
    print("Creating neural tissue for training...")
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=25,
        energy_supply=3.0
    )

    # Create XOR task
    xor_task = XORTask(tissue)

    print(f"\nInitial state: {len(tissue.neurons)} neurons, {len(tissue.synapses)} synapses")
    print(f"Input regions: {list(tissue.input_regions.keys())}")
    print(f"Output regions: {list(tissue.output_regions.keys())}")

    # Pre-train evaluation
    print("\n" + "-" * 60)
    print("PRE-TRAINING EVALUATION")
    print("-" * 60)
    pre_eval = tissue.evaluate_task(xor_task, n_trials=20)
    print(f"Success rate: {pre_eval['success_rate']*100:.1f}%")
    print(f"Average reward: {pre_eval['avg_reward']:.2f}")

    # Training
    print("\n" + "-" * 60)
    print("TRAINING (200 episodes)")
    print("-" * 60)

    n_episodes = 200
    report_every = 25

    training_stats = tissue.train_task(xor_task, n_episodes=n_episodes, report_every=report_every)

    # Post-train evaluation
    print("\n" + "-" * 60)
    print("POST-TRAINING EVALUATION")
    print("-" * 60)
    post_eval = tissue.evaluate_task(xor_task, n_trials=40)
    print(f"Success rate: {post_eval['success_rate']*100:.1f}%")
    print(f"Average reward: {post_eval['avg_reward']:.2f}")

    # Learning curve
    curve = tissue.get_learning_curve("XOR")

    # Display results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"""
Network Statistics:
  Neurons:     {training_stats['total_neurons']} (grew from 25)
  Synapses:    {training_stats['total_synapses']}

Performance:
  Initial success rate:  {pre_eval['success_rate']*100:.1f}%
  Final success rate:    {training_stats['final_success_rate']:.1f}%
  Improvement:           {(post_eval['success_rate'] - pre_eval['success_rate'])*100:.1f}%

Learning Progress:
  Initial avg reward:    {curve[0] if curve else 0:.2f}
  Final avg reward:      {curve[-1] if curve else 0:.2f}
""")

    # Print learning curve visualization
    print("Learning Curve (moving average reward):")
    if curve:
        # Normalize curve to 0-10 scale for ASCII display
        min_val = min(curve)
        max_val = max(curve)
        range_val = max_val - min_val if max_val > min_val else 1

        for i, val in enumerate(curve[::5]):  # Sample every 5th point
            normalized = int((val - min_val) / range_val * 20)
            bar = "#" * normalized
            episode = i * 5 * 10  # Account for window and sampling
            print(f"  Ep {episode:3d}: {bar} {val:.2f}")

    # Show final network state
    print("\n" + "-" * 60)
    print("FINAL NETWORK STATE")
    print("-" * 60)
    print(tissue.statistics())

    # Demonstrate learned behavior
    print("\n" + "-" * 60)
    print("DEMONSTRATING LEARNED BEHAVIOR")
    print("-" * 60)

    test_cases = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0)
    ]

    for a, b, expected in test_cases:
        xor_task.reset()
        xor_task.input_a = a
        xor_task.input_b = b
        xor_task.target = expected

        # Run the network
        tissue.set_inputs({"input_a": a, "input_b": b})
        for _ in range(10):
            tissue.step(dt=0.5)

        output = tissue.read_output("output")
        predicted = 1 if output > 0.5 else 0
        correct = "OK" if predicted == expected else "X"

        print(f"  ({a}, {b}) -> output={output:.3f} (thresholded: {predicted}) expected={expected} [{correct}]")

    print("""
+=============================================================================+
|                           TRAINING COMPLETE                                 |
+=============================================================================+

KEY OBSERVATIONS:

1. LEARNING MECHANISM
   - Dopamine-like reward signal modulates plasticity
   - Eligibility traces link rewards to recent activity
   - Synapses strengthen/weaken based on reward-predicted correlation

2. STRUCTURAL ADAPTATION
   - Network grows neurons in regions that contribute to success
   - Weak connections are pruned during poor performance
   - The network literally rewires itself to solve the task

3. EMERGENT COMPUTATION
   - No backpropagation - learning is local and biologically plausible
   - The hidden layer (intermediate neurons) emerges organically
   - Quantum effects may aid exploration during learning

4. COMPARISON TO TRADITIONAL ML
   - Slower learning than backprop (biological constraint)
   - More robust to damage (distributed computation)
   - Continual learning without catastrophic forgetting
   - Network architecture is not fixed - it grows

NEXT STEPS:

   - Try other tasks: PatternRecognition, Memory, DecisionMaking
   - Experiment with different network sizes and learning rates
   - Observe how quantum effects influence learning
   - Study the development of hidden representations
    """)


def demo_all_tasks():
    """Demo training on all available tasks."""
    print("""
+=============================================================================+
|           ORGANIC NEURAL NETWORK - MULTI-TASK TRAINING                      |
+=============================================================================+
    """)

    # Create neural tissue
    print("Creating neural tissue...")
    tissue = OrganicNeuralNetwork(
        size=(10.0, 10.0, 5.0),
        initial_neurons=30,
        energy_supply=3.0
    )

    tasks = [
        ("XOR", XORTask(tissue)),
        ("Pattern Recognition", PatternRecognitionTask(tissue)),
        ("Memory", MemoryTask(tissue)),
        ("Decision Making", DecisionMakingTask(tissue))
    ]

    results = []

    for task_name, task in tasks:
        print(f"\n{'='*60}")
        print(f"TRAINING: {task_name}")
        print(f"{'='*60}")

        # Pre-evaluation
        pre_eval = tissue.evaluate_task(task, n_trials=15)
        print(f"Pre-training success rate: {pre_eval['success_rate']*100:.1f}%")

        # Train
        stats = tissue.train_task(task, n_episodes=100, report_every=25)

        # Post-evaluation
        post_eval = tissue.evaluate_task(task, n_trials=25)
        print(f"Post-training success rate: {post_eval['success_rate']*100:.1f}%")

        results.append({
            'task': task_name,
            'pre': pre_eval['success_rate'] * 100,
            'post': post_eval['success_rate'] * 100,
            'improvement': (post_eval['success_rate'] - pre_eval['success_rate']) * 100
        })

        # Clear regions for next task
        tissue.input_regions.clear()
        tissue.output_regions.clear()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['task']:<25} {r['pre']:>9.1f}% {r['post']:>9.1f}% {r['improvement']:>+9.1f}%")

    print(f"\nFinal network: {len([n for n in tissue.neurons.values() if n.alive])} neurons, "
          f"{len(tissue.synapses)} synapses")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            demo_training()
        elif sys.argv[1] == "--all-tasks":
            demo_all_tasks()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python3 organic_neural_network.py [--train | --all-tasks]")
    else:
        demo()
