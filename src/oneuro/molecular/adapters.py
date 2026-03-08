"""Adapters to plug molecular neurons/synapses into existing oNeuro infrastructure.

MolecularNeuronAdapter wraps MolecularNeuron with OrganicNeuron's full interface.
MolecularSynapseAdapter wraps MolecularSynapse with OrganicSynapse's full interface.
Existing OrganicNeuralNetwork and MultiTissueNetwork work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from oneuro.molecular.neuron import MolecularNeuron, NeuronArchetype
from oneuro.molecular.synapse import MolecularSynapse


class NeuronState:
    """Matches the NeuronState enum from organic_neural_network.py."""

    RESTING = 0
    ACTIVE = 1
    REFRACTORY = 2
    SUPERPOSITION = 3
    ENTANGLED = 4


@dataclass
class MolecularNeuronAdapter:
    """Wraps MolecularNeuron to present OrganicNeuron's interface.

    All properties/methods that OrganicNeuralNetwork expects are implemented.
    The molecular neuron's emergent voltage is translated to the OrganicNeuron API.
    """

    _mol: MolecularNeuron

    # ---- OrganicNeuron identity fields ----

    @property
    def id(self) -> int:
        return self._mol.id

    @property
    def x(self) -> float:
        return self._mol.x

    @x.setter
    def x(self, v: float):
        self._mol.x = v

    @property
    def y(self) -> float:
        return self._mol.y

    @y.setter
    def y(self, v: float):
        self._mol.y = v

    @property
    def z(self) -> float:
        return self._mol.z

    @z.setter
    def z(self, v: float):
        self._mol.z = v

    @property
    def alive(self) -> bool:
        return self._mol.alive

    @alive.setter
    def alive(self, v: bool):
        self._mol.alive = v

    @property
    def age(self) -> float:
        return self._mol.age

    @age.setter
    def age(self, v: float):
        self._mol.age = v

    @property
    def generation(self) -> int:
        return self._mol.generation

    @property
    def energy(self) -> float:
        return self._mol.energy

    @energy.setter
    def energy(self, v: float):
        self._mol.energy = v

    @property
    def energy_consumption(self) -> float:
        return self._mol.energy_consumption

    @property
    def inputs(self) -> Set[int]:
        return self._mol.inputs

    @inputs.setter
    def inputs(self, v: Set[int]):
        self._mol.inputs = v

    @property
    def outputs(self) -> Set[int]:
        return self._mol.outputs

    @outputs.setter
    def outputs(self, v: Set[int]):
        self._mol.outputs = v

    @property
    def plasticity(self) -> float:
        return self._mol.plasticity

    @plasticity.setter
    def plasticity(self, v: float):
        self._mol.plasticity = v

    @property
    def activation_history(self) -> List[float]:
        return self._mol.activation_history

    # ---- OrganicNeuron electrical fields ----

    @property
    def membrane_potential(self) -> float:
        """Emergent from molecular membrane — NOT settable."""
        return self._mol.membrane_potential

    @membrane_potential.setter
    def membrane_potential(self, v: float):
        # Silently ignore — voltage is emergent in molecular neurons.
        # This prevents crashes when OrganicNeuralNetwork tries to set it.
        pass

    @property
    def threshold(self) -> float:
        return self._mol.threshold

    @threshold.setter
    def threshold(self, v: float):
        pass  # Fixed in molecular model

    @property
    def time_constant(self) -> float:
        return 10.0  # Approximate

    @time_constant.setter
    def time_constant(self, v: float):
        pass

    @property
    def refractory_period(self) -> float:
        return self._mol.membrane._refractory_period

    @refractory_period.setter
    def refractory_period(self, v: float):
        self._mol.membrane._refractory_period = v

    # ---- OrganicNeuron quantum fields (minimal) ----

    @property
    def state(self):
        if self._mol.membrane.is_refractory:
            return NeuronState.REFRACTORY
        elif self._mol.is_active:
            return NeuronState.ACTIVE
        return NeuronState.RESTING

    @state.setter
    def state(self, v):
        pass

    @property
    def entangled_with(self) -> Optional[int]:
        return None

    @entangled_with.setter
    def entangled_with(self, v):
        pass

    @property
    def superposition_weights(self) -> List[float]:
        return [1.0, 0.0]

    @superposition_weights.setter
    def superposition_weights(self, v):
        pass

    # ---- OrganicNeuron methods ----

    def update_liquid(self, input_current: float, dt: float = 0.1) -> bool:
        """Drive the molecular membrane with external current."""
        return self._mol.update(external_current=input_current, dt=dt)

    def consume_energy(self, dt: float = 0.1) -> None:
        self._mol.consume_energy(dt)

    def can_divide(self) -> bool:
        return self._mol.can_divide()

    def divide(self, new_id: int) -> Optional["MolecularNeuronAdapter"]:
        daughter = self._mol.divide(new_id)
        if daughter is None:
            return None
        return MolecularNeuronAdapter(_mol=daughter)

    def enter_superposition(self) -> None:
        pass  # Molecular neurons don't use quantum superposition states

    def collapse_superposition(self) -> float:
        return 1.0


@dataclass
class MolecularSynapseAdapter:
    """Wraps MolecularSynapse to present OrganicSynapse's interface."""

    _mol: MolecularSynapse

    @property
    def pre_neuron(self) -> int:
        return self._mol.pre_neuron

    @property
    def post_neuron(self) -> int:
        return self._mol.post_neuron

    @property
    def weight(self) -> float:
        return self._mol.weight

    @weight.setter
    def weight(self, v: float):
        # Adjust receptor count to approximate desired weight
        target_receptors = int(v * 50.0 / max(0.01, self._mol.strength))
        from oneuro.molecular.receptors import ReceptorType
        if ReceptorType.AMPA in self._mol._postsynaptic_receptor_count:
            self._mol._postsynaptic_receptor_count[ReceptorType.AMPA] = max(1, target_receptors)

    @property
    def delay(self) -> float:
        return self._mol.delay

    @delay.setter
    def delay(self, v: float):
        self._mol.delay = v

    @property
    def strength(self) -> float:
        return self._mol.strength

    @strength.setter
    def strength(self, v: float):
        self._mol.strength = v

    @property
    def age(self) -> float:
        return self._mol.age

    @age.setter
    def age(self, v: float):
        self._mol.age = v

    @property
    def last_pre_spike(self) -> float:
        return self._mol.last_pre_spike

    @last_pre_spike.setter
    def last_pre_spike(self, v: float):
        self._mol.last_pre_spike = v

    @property
    def last_post_spike(self) -> float:
        return self._mol.last_post_spike

    @last_post_spike.setter
    def last_post_spike(self, v: float):
        self._mol.last_post_spike = v

    @property
    def eligibility_trace(self) -> float:
        return self._mol.eligibility_trace

    @eligibility_trace.setter
    def eligibility_trace(self, v: float):
        self._mol.eligibility_trace = v

    @property
    def eligibility_decay(self) -> float:
        return self._mol.eligibility_decay

    @eligibility_decay.setter
    def eligibility_decay(self, v: float):
        self._mol.eligibility_decay = v

    @property
    def pre_activity_avg(self) -> float:
        return self._mol.pre_activity_avg

    @pre_activity_avg.setter
    def pre_activity_avg(self, v: float):
        self._mol.pre_activity_avg = v

    @property
    def post_activity_avg(self) -> float:
        return self._mol.post_activity_avg

    @post_activity_avg.setter
    def post_activity_avg(self, v: float):
        self._mol.post_activity_avg = v

    def update_stdp(self, pre_fired: bool, post_fired: bool, time: float, dt: float = 0.1):
        self._mol.update_stdp(pre_fired, post_fired, time, dt)

    def update_eligibility(self, pre_active: float, post_active: float, dt: float = 0.1):
        self._mol.update_eligibility(pre_active, post_active, dt)

    def apply_reward(self, reward: float, learning_rate: float = 0.1):
        self._mol.apply_reward(reward, learning_rate)

    def should_prune(self) -> bool:
        return self._mol.should_prune()
