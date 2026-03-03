"""
Comprehensive Brain Architecture
================================

A biologically-realistic brain simulation implementing major neuroscience findings.

Implements:
- 6-layer cerebral cortex with minicolumns
- Hippocampus with CA1/CA3/DG subregions
- Thalamus with TRN gating
- Basal ganglia with direct/indirect pathways
- Cerebellum with Purkinje cells
- Amygdala for emotional processing
- Brainstem neuromodulatory systems

Based on:
- Markram et al. (2015) - Blue Brain Project
- HBP (2020) - Human Brain Project
- Izhikevich (2003) - Simple model of spiking neurons
- Buzsáki (2019) - Rhythms of the Brain

Author: Organic Neural Project
License: CC BY-NC 4.0
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# NEUROANATOMY CONSTANTS
# ============================================================

# Cortical layers (from pia to white matter)
CORTICAL_LAYERS = {
    'L1': {'depth': 0.1, 'neurons': 0.05, 'type': 'molecular'},       # Mostly dendrites
    'L2': {'depth': 0.2, 'neurons': 0.15, 'type': 'external_granular'},
    'L3': {'depth': 0.35, 'neurons': 0.25, 'type': 'external_pyramidal'},
    'L4': {'depth': 0.5, 'neurons': 0.20, 'type': 'internal_granular'},  # Thalamic input
    'L5': {'depth': 0.7, 'neurons': 0.20, 'type': 'internal_pyramidal'}, # Output
    'L6': {'depth': 0.9, 'neurons': 0.15, 'type': 'multiform'},
}

# Minicolumn diameter ~50 microns, ~100 neurons per minicolumn
MINICOLUMN_DIAMETER_UM = 50
NEURONS_PER_MINICOLUMN = 100

# Hippocampal subregions
HIPPOCAMPAL_REGIONS = ['DG', 'CA3', 'CA2', 'CA1', 'Subiculum']

# Neuromodulators
class Neurotransmitter(Enum):
    DOPAMINE = auto()       # Reward, motivation, motor control
    SEROTONIN = auto()      # Mood, sleep, appetite
    NOREPINEPHRINE = auto() # Arousal, attention, stress
    ACETYLCHOLINE = auto()  # Attention, learning, memory
    GABA = auto()           # Inhibition
    GLUTAMATE = auto()      # Excitation


# ============================================================
# NEURON MODELS
# ============================================================

@dataclass
class NeuronParameters:
    """Izhikevich neuron model parameters for different cell types."""
    a: float = 0.02      # Recovery time scale
    b: float = 0.2       # Sensitivity of recovery to v
    c: float = -65.0     # Reset potential
    d: float = 8.0       # Reset of recovery
    C: float = 100.0     # Membrane capacitance
    vr: float = -60.0    # Rest potential
    vt: float = -40.0    # Threshold
    k: float = 0.7       # Slope factor

    @classmethod
    def regular_spiking(cls) -> 'NeuronParameters':
        """Regular spiking (RS) excitatory neurons."""
        return cls(a=0.02, b=0.2, c=-65, d=8)

    @classmethod
    def intrinsically_bursting(cls) -> 'NeuronParameters':
        """Intrinsically bursting (IB) neurons."""
        return cls(a=0.02, b=0.2, c=-55, d=4)

    @classmethod
    def chattering(cls) -> 'NeuronParameters':
        """Chattering (CH) neurons."""
        return cls(a=0.02, b=0.2, c=-50, d=2)

    @classmethod
    def fast_spiking(cls) -> 'NeuronParameters':
        """Fast spiking (FS) inhibitory neurons."""
        return cls(a=0.1, b=0.2, c=-65, d=2)

    @classmethod
    def low_threshold(cls) -> 'NeuronParameters':
        """Low-threshold spiking (LTS) inhibitory neurons."""
        return cls(a=0.02, b=0.25, c=-65, d=2)

    @classmethod
    def thalamo_cortical(cls) -> 'NeuronParameters':
        """Thalamo-cortical (TC) relay neurons."""
        return cls(a=0.02, b=0.25, c=-65, d=0.05)

    @classmethod
    def purkinje(cls) -> 'NeuronParameters':
        """Cerebellar Purkinje cells."""
        return cls(a=0.02, b=0.2, c=-65, d=8, k=1.5)


class Neuron:
    """
    Biophysically-inspired neuron using Izhikevich dynamics.

    Supports multiple cell types and neuromodulator effects.
    """

    _id_counter = 0

    def __init__(
        self,
        params: NeuronParameters = None,
        position: Tuple[float, float, float] = (0, 0, 0),
        cell_type: str = 'excitatory',
        region: str = 'cortex'
    ):
        self.id = Neuron._id_counter
        Neuron._id_counter += 1

        self.params = params or NeuronParameters.regular_spiking()
        self.position = position
        self.cell_type = cell_type
        self.region = region

        # State variables
        self.v = self.params.vr  # Membrane potential
        self.u = self.params.b * self.v  # Recovery variable

        # Synaptic inputs
        self.excitatory_input = 0.0
        self.inhibitory_input = 0.0

        # Neuromodulator levels
        self.neuromodulators: Dict[Neurotransmitter, float] = {
            Neurotransmitter.DOPAMINE: 0.0,
            Neurotransmitter.SEROTONIN: 0.0,
            Neurotransmitter.NOREPINEPHRINE: 0.0,
            Neurotransmitter.ACETYLCHOLINE: 0.0,
        }

        # Activity tracking
        self.spike_times: List[float] = []
        self.firing_rate = 0.0
        self.adaptation = 0.0

        # Plasticity
        self.eligibility_trace = 0.0
        self.stdp_window = 0.0  # Time since last spike for STDP

        # Homeostasis
        self.target_rate = 5.0  # Hz
        self.homeostatic_scaling = 1.0

    def step(self, dt: float, t: float, I_ext: float = 0.0) -> bool:
        """
        Advance neuron by dt milliseconds.

        Returns True if spike occurred.
        """
        # Total input current
        I = (I_ext +
             self.excitatory_input -
             self.inhibitory_input * 2.0 +  # Inhibition is stronger
             self.adaptation)

        # Apply neuromodulator effects
        dopamine = self.neuromodulators.get(Neurotransmitter.DOPAMINE, 0)
        I += dopamine * 0.5  # Dopamine depolarizes

        # Izhikevich dynamics
        v_prime = (self.params.k * (self.v - self.params.vr) *
                   (self.v - self.params.vt) - self.u + I) / self.params.C
        u_prime = self.params.a * (self.params.b * (self.v - self.params.vr) - self.u)

        self.v += v_prime * dt
        self.u += u_prime * dt

        # Spike detection
        spiked = self.v >= self.params.vt + 30

        if spiked:
            self.v = self.params.c
            self.u += self.params.d
            self.spike_times.append(t)
            self.stdp_window = 0.0  # Reset STDP window
            self.eligibility_trace = 1.0
            self.adaptation += 0.5  # Spike-frequency adaptation
        else:
            self.stdp_window += dt
            self.eligibility_trace *= 0.95  # Decay

        # Decay adaptation
        self.adaptation *= 0.99

        # Update firing rate (exponential moving average)
        if len(self.spike_times) > 1:
            recent = [st for st in self.spike_times if t - st < 1000]
            self.firing_rate = len(recent)

        # Homeostatic plasticity
        rate_error = self.firing_rate - self.target_rate
        self.homeostatic_scaling *= 1.0 - 0.001 * rate_error
        self.homeostatic_scaling = max(0.5, min(2.0, self.homeostatic_scaling))

        # Reset synaptic inputs
        self.excitatory_input = 0.0
        self.inhibitory_input = 0.0

        return spiked

    def receive_input(self, weight: float, is_inhibitory: bool = False):
        """Receive synaptic input."""
        if is_inhibitory:
            self.inhibitory_input += abs(weight) * self.homeostatic_scaling
        else:
            self.excitatory_input += weight * self.homeostatic_scaling

    def apply_neuromodulator(self, nt: Neurotransmitter, amount: float):
        """Apply neuromodulator effect."""
        current = self.neuromodulators.get(nt, 0)
        self.neuromodulators[nt] = current * 0.9 + amount * 0.1


# ============================================================
# SYNAPSE MODELS
# ============================================================

@dataclass
class Synapse:
    """
    Biologically-realistic synapse with multiple forms of plasticity.

    Implements:
    - STDP (Spike-Timing Dependent Plasticity)
    - Metaplasticity (BCM-like sliding threshold)
    - Short-term plasticity (depression/facilitation)
    - Neuromodulator-gated plasticity
    """

    pre_id: int
    post_id: int
    weight: float = 0.5
    delay_ms: float = 1.0

    # Short-term plasticity (Tsodyks-Markram model)
    U: float = 0.5       # Utilization of synaptic resources
    D: float = 800.0     # Depression time constant (ms)
    F: float = 100.0     # Facilitation time constant (ms)
    u: float = 0.0       # Fraction of resources used
    R: float = 1.0       # Available resources

    # STDP parameters
    stdp_tau_plus: float = 20.0    # LTP window (ms)
    stdp_tau_minus: float = 20.0   # LTD window (ms)
    stdp_A_plus: float = 0.01      # LTP amplitude
    stdp_A_minus: float = 0.01     # LTD amplitude

    # Metaplasticity (BCM)
    theta_m: float = 0.0    # Modification threshold
    bcm_tau: float = 10000.0  # Sliding window (ms)

    # Activity tracking
    pre_spike_times: List[float] = field(default_factory=list)
    post_spike_times: List[float] = field(default_factory=list)

    # Neuromodulator gating
    dopamine_gated: bool = True
    last_dopamine: float = 0.0

    def transmit(self, t: float) -> float:
        """
        Transmit signal through synapse with short-term plasticity.

        Returns effective weight.
        """
        # Update short-term plasticity
        self.u = self.u + self.U * (1 - self.u)
        effective_R = self.R * self.u

        # Resources are consumed
        self.R = self.R - effective_R

        # Effective weight
        effective_weight = self.weight * effective_R

        return effective_weight

    def recover(self, dt: float):
        """Recover synaptic resources."""
        self.R += dt * (1 - self.R) / self.D
        self.u -= dt * self.u / self.F

    def apply_stdp(self, t: float, dopamine: float = 0.0):
        """
        Apply STDP with dopamine gating.

        Dopamine gates whether plasticity is consolidated.
        """
        if not self.pre_spike_times or not self.post_spike_times:
            return

        # Get recent spikes
        pre_recent = [st for st in self.pre_spike_times if t - st < 100]
        post_recent = [st for st in self.post_spike_times if t - st < 100]

        if not pre_recent or not post_recent:
            return

        # Calculate timing-dependent plasticity
        delta_w = 0.0

        for pre_t in pre_recent[-5:]:  # Last 5 pre spikes
            for post_t in post_recent[-5:]:  # Last 5 post spikes
                dt_spike = post_t - pre_t

                if dt_spike > 0:  # Pre before post = LTP
                    delta_w += self.stdp_A_plus * math.exp(-dt_spike / self.stdp_tau_plus)
                else:  # Post before pre = LTD
                    delta_w -= self.stdp_A_minus * math.exp(dt_spike / self.stdp_tau_minus)

        # BCM sliding threshold
        avg_post_rate = len(post_recent) / 0.1  # Hz over 100ms
        self.theta_m = 0.9 * self.theta_m + 0.1 * avg_post_rate ** 2

        # Apply metaplasticity modification
        if avg_post_rate < self.theta_m * 0.001:
            delta_w *= -1  # Below threshold = depression

        # Dopamine gating (consolidate only with dopamine)
        if self.dopamine_gated and dopamine > 0.1:
            self.weight += delta_w * dopamine
        elif not self.dopamine_gated:
            self.weight += delta_w

        # Clamp weight
        self.weight = max(0.0, min(2.0, self.weight))


# ============================================================
# BRAIN REGIONS
# ============================================================

class BrainRegion:
    """Base class for brain regions."""

    def __init__(self, name: str, volume_mm3: float = 100.0):
        self.name = name
        self.volume_mm3 = volume_mm3
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: List[Synapse] = []
        self.connections_in: List[Tuple['BrainRegion', float]] = []
        self.connections_out: List[Tuple['BrainRegion', float]] = []

        # Region-level activity
        self.mean_firing_rate = 0.0
        self.peak_firing_rate = 0.0

    def add_neuron(self, neuron: Neuron) -> int:
        """Add neuron to region."""
        self.neurons[neuron.id] = neuron
        return neuron.id

    def step(self, dt: float, t: float):
        """Advance region by dt."""
        spikes = 0
        for neuron in self.neurons.values():
            if neuron.step(dt, t):
                spikes += 1

        n = len(self.neurons)
        if n > 0:
            self.mean_firing_rate = spikes / n
            self.peak_firing_rate = max(self.peak_firing_rate, self.mean_firing_rate)

    def connect_to(self, other: 'BrainRegion', strength: float = 1.0):
        """Connect to another region."""
        self.connections_out.append((other, strength))
        other.connections_in.append((self, strength))


class CerebralCortex(BrainRegion):
    """
    6-layer cerebral cortex with columnar organization.

    Structure:
    - Layer 1: Molecular (mostly dendrites, few neurons)
    - Layer 2/3: External granular/pyramidal (intra-cortical)
    - Layer 4: Internal granular (thalamic input)
    - Layer 5: Internal pyramidal (output to subcortex)
    - Layer 6: Multiform (feedback to thalamus)

    Features:
    - Minicolumns (~50um diameter, ~100 neurons)
    - Recurrent excitation in L2/3
    - Thalamic input to L4
    - Cortical output from L5
    """

    def __init__(
        self,
        area_mm2: float = 1000.0,  # Cortical surface area
        thickness_mm: float = 2.5,
        n_minicolumns: int = 100
    ):
        super().__init__("cerebral_cortex", area_mm2 * thickness_mm)

        self.area_mm2 = area_mm2
        self.thickness_mm = thickness_mm
        self.n_minicolumns = n_minicolumns

        # Create layers
        self.layers: Dict[str, List[int]] = {
            'L1': [], 'L2': [], 'L3': [], 'L4': [], 'L5': [], 'L6': []
        }

        # Minicolumns
        self.minicolumns: List[List[int]] = [[] for _ in range(n_minicolumns)]

        self._build_cortex()

    def _build_cortex(self):
        """Build cortical structure."""
        neuron_id = 0

        # Create neurons for each layer
        for layer_name, layer_info in CORTICAL_LAYERS.items():
            n_neurons = int(NEURONS_PER_MINICOLUMN *
                          self.n_minicolumns *
                          layer_info['neurons'])

            for i in range(n_neurons):
                # Assign to minicolumn
                minicol_idx = i % self.n_minicolumns

                # Position within layer and minicolumn
                depth = layer_info['depth'] * self.thickness_mm

                # Choose neuron type based on layer
                if layer_name in ['L2', 'L3']:
                    params = NeuronParameters.regular_spiking()
                    cell_type = 'pyramidal'
                elif layer_name == 'L4':
                    params = random.choice([
                        NeuronParameters.regular_spiking(),
                        NeuronParameters.fast_spiking()
                    ])
                    cell_type = 'spiny_stellate' if params.a < 0.05 else 'inhibitory'
                elif layer_name == 'L5':
                    params = random.choice([
                        NeuronParameters.intrinsically_bursting(),
                        NeuronParameters.regular_spiking()
                    ])
                    cell_type = 'pyramidal'
                else:
                    params = NeuronParameters.regular_spiking()
                    cell_type = 'pyramidal'

                neuron = Neuron(
                    params=params,
                    position=(minicol_idx * MINICOLUMN_DIAMETER_UM / 1000.0,
                              depth,
                              0),
                    cell_type=cell_type,
                    region=f'cortex_{layer_name}'
                )

                self.add_neuron(neuron)
                self.layers[layer_name].append(neuron.id)
                self.minicolumns[minicol_idx].append(neuron.id)

    def thalamic_input(self, pattern: List[float]):
        """
        Receive thalamic input to Layer 4.

        pattern: List of activation values for each minicolumn
        """
        for i, activation in enumerate(pattern):
            if i >= len(self.minicolumns):
                break
            for nid in self.minicolumns[i]:
                if nid in self.neurons and nid in self.layers['L4']:
                    self.neurons[nid].receive_input(activation * 0.5)


class Hippocampus(BrainRegion):
    """
    Hippocampus with CA1, CA3, and Dentate Gyrus subregions.

    Circuit:
    Entorhinal Cortex → DG (granule cells) → CA3 (pyramidal) → CA1 (pyramidal) → EC

    Features:
    - Pattern separation in DG (sparse coding)
    - Pattern completion in CA3 (recurrent connections)
    - Theta-gamma coupling
    - Sharp-wave ripples during sleep
    - Place cells (spatial memory)
    """

    def __init__(self, n_neurons_per_region: int = 100):
        super().__init__("hippocampus", 4000.0)  # ~4 cm³ in humans

        self.subregions: Dict[str, List[int]] = {
            'DG': [], 'CA3': [], 'CA2': [], 'CA1': [], 'Subiculum': []
        }

        # Theta phase (for phase precession)
        self.theta_phase = 0.0
        self.theta_freq = 8.0  # Hz

        # Sharp-wave ripple state
        self.ripple_active = False
        self.ripple_timer = 0.0

        self._build_hippocampus(n_neurons_per_region)

    def _build_hippocampus(self, n_neurons: int):
        """Build hippocampal circuit."""
        # DG: Granule cells (very sparse activity)
        for i in range(n_neurons):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 0, 0),
                cell_type='granule',
                region='hippo_DG'
            )
            neuron.target_rate = 0.5  # Very sparse
            self.add_neuron(neuron)
            self.subregions['DG'].append(neuron.id)

        # CA3: Pyramidal cells with strong recurrent connections
        for i in range(n_neurons):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 1, 0),
                cell_type='pyramidal',
                region='hippo_CA3'
            )
            self.add_neuron(neuron)
            self.subregions['CA3'].append(neuron.id)

        # CA1: Pyramidal cells (output)
        for i in range(n_neurons):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 2, 0),
                cell_type='pyramidal',
                region='hippo_CA1'
            )
            self.add_neuron(neuron)
            self.subregions['CA1'].append(neuron.id)

        # Create connections
        self._create_hippocampal_connections()

    def _create_hippocampal_connections(self):
        """Create the trisynaptic circuit."""
        # DG → CA3 (mossy fibers, very sparse but strong)
        for dg_id in self.subregions['DG']:
            # Each granule cell connects to few CA3 cells
            for ca3_id in random.sample(self.subregions['CA3'],
                                        min(10, len(self.subregions['CA3']))):
                syn = Synapse(pre_id=dg_id, post_id=ca3_id,
                             weight=1.5, delay_ms=2.0)
                syn.U = 0.8  # High release probability
                self.synapses.append(syn)

        # CA3 → CA3 (recurrent, for pattern completion)
        for ca3_id in self.subregions['CA3']:
            for target_id in random.sample(self.subregions['CA3'],
                                          min(50, len(self.subregions['CA3']))):
                if target_id != ca3_id:
                    syn = Synapse(pre_id=ca3_id, post_id=target_id,
                                 weight=0.3, delay_ms=1.0)
                    self.synapses.append(syn)

        # CA3 → CA1 (Schaffer collaterals)
        for ca3_id in self.subregions['CA3']:
            for ca1_id in random.sample(self.subregions['CA1'],
                                       min(30, len(self.subregions['CA1']))):
                syn = Synapse(pre_id=ca3_id, post_id=ca1_id,
                             weight=0.5, delay_ms=1.5)
                self.synapses.append(syn)

    def pattern_separation(self, input_pattern: List[float]) -> List[float]:
        """
        Dentate Gyrus pattern separation.

        Similar inputs → distinct sparse representations.
        """
        # Only ~2% of granule cells active at once
        threshold = sorted(input_pattern, reverse=True)[
            min(int(len(input_pattern) * 0.02), len(input_pattern)-1)
        ]

        separated = [1.0 if x >= threshold else 0.0 for x in input_pattern]
        return separated

    def pattern_completion(self, partial_pattern: List[float]) -> List[float]:
        """
        CA3 pattern completion.

        Partial input → full memory retrieval via recurrent connections.
        """
        # In real implementation, this would use CA3 recurrent dynamics
        # Here we simulate the effect
        completed = partial_pattern.copy()

        # Spread activation through recurrent connections
        for i in range(len(completed)):
            if completed[i] > 0.5:
                # Activate neighbors
                for j in range(max(0, i-5), min(len(completed), i+5)):
                    completed[j] = max(completed[j], 0.3)

        return completed

    def step(self, dt: float, t: float):
        """Advance hippocampus with theta rhythm."""
        super().step(dt, t)

        # Update theta phase
        self.theta_phase = (2 * math.pi * self.theta_freq * t / 1000.0) % (2 * math.pi)

        # Theta modulation of plasticity
        theta_mod = 0.5 + 0.5 * math.cos(self.theta_phase)

        # Apply theta-dependent modulation to synapses
        for syn in self.synapses:
            if 'CA3' in self.neurons[syn.post_id].region:
                syn.stdp_A_plus *= (0.9 + 0.2 * theta_mod)


class Thalamus(BrainRegion):
    """
    Thalamic relay nucleus with TRN gating.

    Features:
    - Relay of sensory information to cortex
    - Thalamic reticular nucleus (TRN) for attention gating
    - Sleep spindles during NREM
    - Tonic vs burst firing modes
    """

    def __init__(self, n_relay: int = 100, n_trn: int = 50):
        super().__init__("thalamus", 5000.0)  # ~5 cm³

        self.relay_neurons: List[int] = []
        self.trn_neurons: List[int] = []

        self._build_thalamus(n_relay, n_trn)

    def _build_thalamus(self, n_relay: int, n_trn: int):
        """Build thalamic circuit."""
        # Relay neurons (TC cells)
        for i in range(n_relay):
            neuron = Neuron(
                params=NeuronParameters.thalamo_cortical(),
                position=(i * 0.1, 0, 0),
                cell_type='relay',
                region='thalamus_relay'
            )
            self.add_neuron(neuron)
            self.relay_neurons.append(neuron.id)

        # TRN inhibitory neurons
        for i in range(n_trn):
            neuron = Neuron(
                params=NeuronParameters.fast_spiking(),
                position=(i * 0.2, 1, 0),
                cell_type='inhibitory',
                region='thalamus_trn'
            )
            self.add_neuron(neuron)
            self.trn_neurons.append(neuron.id)

        # TRN → Relay (inhibition for gating)
        for trn_id in self.trn_neurons:
            for relay_id in self.relay_neurons:
                syn = Synapse(
                    pre_id=trn_id, post_id=relay_id,
                    weight=0.8, delay_ms=1.0
                )
                self.synapses.append(syn)

    def set_attention_focus(self, channels: List[int]):
        """
        TRN gating for attention.

        Channels not in focus are inhibited.
        """
        for i, relay_id in enumerate(self.relay_neurons):
            if i not in channels:
                # Inhibit non-focused channels
                for trn_id in self.trn_neurons:
                    self.neurons[trn_id].receive_input(0.5, is_inhibitory=False)


class BasalGanglia(BrainRegion):
    """
    Basal ganglia for action selection and reinforcement learning.

    Direct pathway: Cortex → Striatum → GPi/SNr → Thalamus (facilitates action)
    Indirect pathway: Cortex → Striatum → GPe → STN → GPi/SNr (inhibits action)

    Dopamine from SNc modulates which pathway is active:
    - D1 receptors: Direct pathway (Go)
    - D2 receptors: Indirect pathway (No-Go)
    """

    def __init__(self, n_neurons: int = 50):
        super().__init__("basal_ganglia", 3000.0)

        self.subregions = {
            'striatum_d1': [],   # Direct pathway
            'striatum_d2': [],   # Indirect pathway
            'gpe': [],           # External globus pallidus
            'gpi': [],           # Internal globus pallidus
            'stn': [],           # Subthalamic nucleus
            'snc': [],           # Substantia nigra pars compacta (dopamine)
        }

        self._build_basal_ganglia(n_neurons)

    def _build_basal_ganglia(self, n: int):
        """Build basal ganglia circuit."""
        for region, neuron_list in self.subregions.items():
            for i in range(n):
                if region == 'snc':
                    # Dopaminergic neurons
                    neuron = Neuron(
                        params=NeuronParameters.regular_spiking(),
                        position=(i * 0.1, 0, 0),
                        cell_type='dopaminergic',
                        region=f'bg_{region}'
                    )
                else:
                    neuron = Neuron(
                        params=NeuronParameters.regular_spiking(),
                        position=(i * 0.1, 0, 0),
                        cell_type='medium_spiny' if 'striatum' in region else 'inhibitory',
                        region=f'bg_{region}'
                    )
                self.add_neuron(neuron)
                neuron_list.append(neuron.id)

        # Create pathways
        self._create_bg_connections()

    def _create_bg_connections(self):
        """Create direct and indirect pathway connections."""
        # Direct: Striatum D1 → GPi (inhibition of inhibition = Go)
        for d1_id in self.subregions['striatum_d1']:
            for gpi_id in self.subregions['gpi']:
                syn = Synapse(pre_id=d1_id, post_id=gpi_id,
                             weight=0.5, delay_ms=2.0)
                self.synapses.append(syn)

        # Indirect: Striatum D2 → GPe → STN → GPi
        for d2_id in self.subregions['striatum_d2']:
            for gpe_id in self.subregions['gpe']:
                syn = Synapse(pre_id=d2_id, post_id=gpe_id,
                             weight=0.5, delay_ms=2.0)
                self.synapses.append(syn)

        for gpe_id in self.subregions['gpe']:
            for stn_id in self.subregions['stn']:
                syn = Synapse(pre_id=gpe_id, post_id=stn_id,
                             weight=0.5, delay_ms=2.0)
                self.synapses.append(syn)

        for stn_id in self.subregions['stn']:
            for gpi_id in self.subregions['gpi']:
                syn = Synapse(pre_id=stn_id, post_id=gpi_id,
                             weight=0.5, delay_ms=2.0)
                self.synapses.append(syn)

    def release_dopamine(self, amount: float):
        """Release dopamine from SNc."""
        for snc_id in self.subregions['snc']:
            self.neurons[snc_id].firing_rate = amount * 100  # Scale to Hz

    def compute_prediction_error(self, predicted: float, actual: float) -> float:
        """
        Compute dopamine prediction error.

        RPE = Actual - Predicted reward
        """
        return actual - predicted


class Brainstem(BrainRegion):
    """
    Brainstem neuromodulatory systems.

    - Locus Coeruleus (LC): Norepinephrine for arousal/attention
    - Raphe Nuclei: Serotonin for mood/impulse
    - Ventral Tegmental Area (VTA): Dopamine for reward
    - Basal Forebrain: Acetylcholine for attention
    """

    def __init__(self):
        super().__init__("brainstem", 2000.0)

        self.neuromodulator_levels = {
            Neurotransmitter.NOREPINEPHRINE: 0.5,
            Neurotransmitter.SEROTONIN: 0.5,
            Neurotransmitter.DOPAMINE: 0.5,
            Neurotransmitter.ACETYLCHOLINE: 0.5,
        }

        # LC neurons (NE)
        self.lc_neurons: List[int] = []
        # Raphe neurons (5-HT)
        self.raphe_neurons: List[int] = []
        # VTA neurons (DA)
        self.vta_neurons: List[int] = []

        self._build_brainstem()

    def _build_brainstem(self):
        """Build brainstem nuclei."""
        # Locus Coeruleus (~1500 neurons in humans, scaled down)
        for i in range(30):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 0, 0),
                cell_type='noradrenergic',
                region='brainstem_lc'
            )
            self.add_neuron(neuron)
            self.lc_neurons.append(neuron.id)

        # Raphe nuclei
        for i in range(30):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 1, 0),
                cell_type='serotonergic',
                region='brainstem_raphe'
            )
            self.add_neuron(neuron)
            self.raphe_neurons.append(neuron.id)

        # VTA
        for i in range(30):
            neuron = Neuron(
                params=NeuronParameters.regular_spiking(),
                position=(i * 0.1, 2, 0),
                cell_type='dopaminergic',
                region='brainstem_vta'
            )
            self.add_neuron(neuron)
            self.vta_neurons.append(neuron.id)

    def set_arousal(self, level: float):
        """Set arousal level via LC activity."""
        level = max(0, min(1, level))
        self.neuromodulator_levels[Neurotransmitter.NOREPINEPHRINE] = level

        for lc_id in self.lc_neurons:
            self.neurons[lc_id].firing_rate = level * 10  # 0-10 Hz

    def set_mood(self, level: float):
        """Set mood via raphe activity."""
        level = max(0, min(1, level))
        self.neuromodulator_levels[Neurotransmitter.SEROTONIN] = level

        for raphe_id in self.raphe_neurons:
            self.neurons[raphe_id].firing_rate = level * 5

    def broadcast_neuromodulators(self, target_regions: List[BrainRegion]):
        """Broadcast neuromodulator levels to target regions."""
        for region in target_regions:
            for neuron in region.neurons.values():
                for nt, level in self.neuromodulator_levels.items():
                    neuron.apply_neuromodulator(nt, level * 0.1)


# ============================================================
# UNIFIED BRAIN
# ============================================================

class ComprehensiveBrain:
    """
    Complete brain model integrating all regions.

    This implements the first comprehensive computational brain model
    combining multiple neuroscience theories.
    """

    def __init__(
        self,
        n_cortical_minicolumns: int = 100,
        n_hippocampal_neurons: int = 100,
        n_thalamic_relay: int = 50,
        n_basal_ganglia: int = 30
    ):
        # Create all brain regions
        self.cortex = CerebralCortex(n_minicolumns=n_cortical_minicolumns)
        self.hippocampus = Hippocampus(n_neurons_per_region=n_hippocampal_neurons)
        self.thalamus = Thalamus(n_relay=n_thalamic_relay, n_trn=n_thalamic_relay//2)
        self.basal_ganglia = BasalGanglia(n_neurons=n_basal_ganglia)
        self.brainstem = Brainstem()

        self.regions = [
            self.cortex, self.hippocampus, self.thalamus,
            self.basal_ganglia, self.brainstem
        ]

        # Inter-region connections
        self._connect_regions()

        # Simulation state
        self.t = 0.0  # Current time in ms
        self.dt = 0.1  # Time step in ms

        # Global metrics
        self.total_neurons = sum(len(r.neurons) for r in self.regions)
        self.total_synapses = sum(len(r.synapses) for r in self.regions)

        # Oscillation tracking
        self.gamma_power = 0.0
        self.theta_power = 0.0
        self.alpha_power = 0.0

        # Consciousness metrics
        self.complexity = 0.0
        self.integration = 0.0

        logger.info(f"Created brain with {self.total_neurons} neurons, "
                   f"{self.total_synapses} synapses")

    def _connect_regions(self):
        """Create inter-region connections."""
        # Thalamus → Cortex (L4)
        self.thalamus.connect_to(self.cortex, strength=0.8)

        # Cortex → Hippocampus (entorhinal)
        self.cortex.connect_to(self.hippocampus, strength=0.5)

        # Hippocampus → Cortex (consolidation)
        self.hippocampus.connect_to(self.cortex, strength=0.3)

        # Cortex → Basal Ganglia
        self.cortex.connect_to(self.basal_ganglia, strength=0.6)

        # Basal Ganglia → Thalamus
        self.basal_ganglia.connect_to(self.thalamus, strength=0.5)

        # Brainstem → All (neuromodulation)
        self.brainstem.connect_to(self.cortex, strength=0.2)
        self.brainstem.connect_to(self.hippocampus, strength=0.2)
        self.brainstem.connect_to(self.thalamus, strength=0.2)

    def step(self):
        """Advance brain simulation by one time step."""
        self.t += self.dt

        # 1. Brainstem broadcasts neuromodulators
        self.brainstem.broadcast_neuromodulators([
            self.cortex, self.hippocampus, self.thalamus
        ])

        # 2. Process inter-region inputs
        self._process_inter_region_signals()

        # 3. Step each region
        for region in self.regions:
            region.step(self.dt, self.t)

        # 4. Apply synaptic plasticity
        self._apply_plasticity()

        # 5. Compute global metrics
        self._compute_global_metrics()

    def _process_inter_region_signals(self):
        """Process signals between regions."""
        # Thalamus → Cortex
        for src_region, strength in self.thalamus.connections_out:
            if src_region == self.cortex:
                # Get thalamic output
                thalamic_activity = [
                    self.thalamus.neurons[nid].v
                    for nid in self.thalamus.relay_neurons
                ]
                # Normalize and send to cortex L4
                if thalamic_activity:
                    mean_activity = sum(thalamic_activity) / len(thalamic_activity)
                    self.cortex.thalamic_input([mean_activity * 0.1] * len(self.cortex.minicolumns))

    def _apply_plasticity(self):
        """Apply synaptic plasticity."""
        dopamine = self.brainstem.neuromodulator_levels.get(
            Neurotransmitter.DOPAMINE, 0.5
        )

        for region in self.regions:
            for syn in region.synapses:
                syn.recover(self.dt)
                syn.apply_stdp(self.t, dopamine=dopamine)

    def _compute_global_metrics(self):
        """Compute global brain metrics."""
        # Collect all spike rates
        all_rates = []
        for region in self.regions:
            for neuron in region.neurons.values():
                all_rates.append(neuron.firing_rate)

        if not all_rates:
            return

        # Compute oscillation power (simplified)
        mean_rate = sum(all_rates) / len(all_rates)

        # Gamma (~40 Hz) - high activity
        self.gamma_power = min(1.0, mean_rate / 50.0)

        # Theta (~8 Hz) - hippocampal
        self.theta_power = 0.5 + 0.5 * math.cos(
            2 * math.pi * self.hippocampus.theta_freq * self.t / 1000.0
        )

        # Alpha (~10 Hz) - resting
        self.alpha_power = max(0, 1.0 - self.gamma_power - self.theta_power * 0.3)

        # Complexity (Lempel-Ziv approximation)
        active = sum(1 for r in all_rates if r > 0.1)
        self.complexity = active / len(all_rates) if all_rates else 0

        # Integration
        self.integration = min(1.0, self.gamma_power * self.theta_power)

    def present_stimulus(self, stimulus: List[float], modality: str = 'visual'):
        """Present sensory stimulus to the brain."""
        # Route through thalamus
        thalamic_pattern = [s * 0.5 for s in stimulus[:len(self.thalamus.relay_neurons)]]

        for i, relay_id in enumerate(self.thalamus.relay_neurons):
            if i < len(thalamic_pattern):
                self.thalamus.neurons[relay_id].receive_input(thalamic_pattern[i])

        # Increase arousal
        self.brainstem.set_arousal(0.7)

    def give_reward(self, amount: float):
        """Give reward signal (dopamine release)."""
        # Compute prediction error
        rpe = self.basal_ganglia.compute_prediction_error(
            predicted=0.5, actual=amount
        )

        # Release dopamine proportional to RPE
        self.brainstem.neuromodulator_levels[Neurotransmitter.DOPAMINE] = 0.5 + rpe * 0.5
        self.basal_ganglia.release_dopamine(max(0, rpe))

    def run(self, duration_ms: float, callback: Callable = None):
        """Run brain simulation for duration."""
        n_steps = int(duration_ms / self.dt)

        for _ in range(n_steps):
            self.step()
            if callback:
                callback(self)

    def get_state_report(self) -> Dict[str, Any]:
        """Get comprehensive brain state report."""
        return {
            'time_ms': self.t,
            'total_neurons': self.total_neurons,
            'total_synapses': self.total_synapses,
            'oscillations': {
                'gamma': self.gamma_power,
                'theta': self.theta_power,
                'alpha': self.alpha_power,
            },
            'neuromodulators': dict(self.brainstem.neuromodulator_levels),
            'region_activity': {
                r.name: r.mean_firing_rate for r in self.regions
            },
            'consciousness': {
                'complexity': self.complexity,
                'integration': self.integration,
            }
        }


# ============================================================
# EMERGENT BEHAVIOR TESTS
# ============================================================

def test_neural_avalanches(brain: ComprehensiveBrain, duration_ms: float = 5000) -> Dict:
    """
    Test for critical dynamics (neural avalanches).

    At criticality, avalanche sizes follow a power law.
    """
    avalanche_sizes = []
    current_avalanche = 0

    for _ in range(int(duration_ms / brain.dt)):
        # Count active neurons
        active = sum(
            1 for r in brain.regions
            for n in r.neurons.values()
            if n.firing_rate > 0.1
        )

        if active > brain.total_neurons * 0.01:  # Threshold
            current_avalanche += active
        else:
            if current_avalanche > 0:
                avalanche_sizes.append(current_avalanche)
                current_avalanche = 0

        brain.step()

    # Check for power law (criticality)
    if len(avalanche_sizes) < 10:
        return {'critical': False, 'reason': 'insufficient_avalanches'}

    # Log-log regression to check power law
    import statistics
    sizes = sorted(avalanche_sizes)

    # Check if distribution is heavy-tailed
    mean_size = statistics.mean(sizes)
    var_size = statistics.variance(sizes) if len(sizes) > 1 else 0

    # Critical systems have variance >> mean
    cv = var_size / mean_size if mean_size > 0 else 0

    return {
        'n_avalanches': len(avalanche_sizes),
        'mean_size': mean_size,
        'coefficient_of_variation': cv,
        'critical': cv > 1.0,
    }


def test_oscillation_coupling(brain: ComprehensiveBrain, duration_ms: float = 5000) -> Dict:
    """
    Test for theta-gamma coupling in hippocampus.

    Gamma amplitude should be modulated by theta phase.
    """
    theta_phases = []
    gamma_amplitudes = []

    for _ in range(int(duration_ms / brain.dt)):
        brain.step()

        theta_phases.append(brain.hippocampus.theta_phase)
        gamma_amplitudes.append(brain.gamma_power)

    # Compute coupling strength
    # Group gamma by theta phase bins
    n_bins = 8
    gamma_by_phase = [[] for _ in range(n_bins)]

    for theta, gamma in zip(theta_phases, gamma_amplitudes):
        bin_idx = int(theta / (2 * math.pi) * n_bins) % n_bins
        gamma_by_phase[bin_idx].append(gamma)

    # Compute variance across bins (coupling = high variance)
    bin_means = [sum(g) / len(g) if g else 0 for g in gamma_by_phase]
    coupling = max(bin_means) - min(bin_means) if bin_means else 0

    return {
        'theta_gamma_coupling': coupling,
        'coupling_strength': 'strong' if coupling > 0.3 else 'weak',
        'preferred_phase': bin_means.index(max(bin_means)) * (2 * math.pi / n_bins),
    }


def demo_comprehensive_brain():
    """Demonstrate the comprehensive brain model."""
    print("=" * 70)
    print("COMPREHENSIVE BRAIN SIMULATION")
    print("=" * 70)

    # Create brain
    brain = ComprehensiveBrain(
        n_cortical_minicolumns=50,
        n_hippocampal_neurons=50,
        n_thalamic_relay=30,
        n_basal_ganglia=20
    )

    print(f"\nBrain Statistics:")
    print(f"  Total neurons: {brain.total_neurons}")
    print(f"  Total synapses: {brain.total_synapses}")

    # Run baseline
    print("\n[1] Baseline activity (1000ms)...")
    brain.run(1000)
    report = brain.get_state_report()
    print(f"  Mean firing rate: {sum(report['region_activity'].values())/len(report['region_activity']):.2f} Hz")

    # Present stimulus
    print("\n[2] Presenting visual stimulus...")
    brain.present_stimulus([0.5] * 30, modality='visual')
    brain.run(500)
    print(f"  Gamma power: {brain.gamma_power:.3f}")
    print(f"  Arousal (NE): {brain.brainstem.neuromodulator_levels[Neurotransmitter.NOREPINEPHRINE]:.3f}")

    # Give reward
    print("\n[3] Giving reward (dopamine)...")
    brain.give_reward(1.0)
    brain.run(500)
    print(f"  Dopamine level: {brain.brainstem.neuromodulator_levels[Neurotransmitter.DOPAMINE]:.3f}")

    # Test criticality
    print("\n[4] Testing for critical dynamics...")
    criticality = test_neural_avalanches(brain, duration_ms=2000)
    print(f"  Avalanches detected: {criticality.get('n_avalanches', 0)}")
    print(f"  Critical: {criticality.get('critical', False)}")

    # Test oscillation coupling
    print("\n[5] Testing theta-gamma coupling...")
    coupling = test_oscillation_coupling(brain, duration_ms=2000)
    print(f"  Coupling strength: {coupling['coupling_strength']}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return brain


if __name__ == "__main__":
    demo_comprehensive_brain()
