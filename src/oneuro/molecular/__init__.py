"""Molecular neural tissue — neurons composed of quantum-simulated molecules.

Ion channels are folded proteins, neurotransmitters are SMILES molecules,
synaptic enzymes use quantum tunneling catalysis, and membrane potential
EMERGES from ion channel physics rather than being a hand-set float.

Phase 3 additions: second messengers, multi-compartment calcium, glia,
dendrites, spines, metabolism, axons, gap junctions, extracellular space,
microtubules/Orch-OR, circadian rhythms, enhanced gene expression, and
advanced neuroplasticity (NMDA-gated STDP, BCM, synaptic tagging).

Requires nQPU for quantum chemistry features; falls back to numpy otherwise.
"""

from oneuro.molecular.backend import HAS_NQPU, require_nqpu
from oneuro.molecular.neurotransmitters import (
    NeurotransmitterMolecule,
    NEUROTRANSMITTER_LIBRARY,
)
from oneuro.molecular.ion_channels import (
    IonChannelType,
    IonChannel,
    IonChannelEnsemble,
    BatchIonChannelState,
)
from oneuro.molecular.receptors import SynapticReceptor, ReceptorType
from oneuro.molecular.enzymes import SynapticEnzyme, ENZYME_LIBRARY
from oneuro.molecular.membrane import MolecularMembrane
from oneuro.molecular.gene_expression import (
    GeneExpressionPipeline,
    GeneID,
    GeneState,
    TranscriptionFactor,
    TranscriptionFactorType,
    EpigeneticState,
)
from oneuro.molecular.neuron import MolecularNeuron, NeuronArchetype
from oneuro.molecular.synapse import MolecularSynapse
from oneuro.molecular.adapters import MolecularNeuronAdapter, MolecularSynapseAdapter
from oneuro.molecular.network import MolecularNeuralNetwork
from oneuro.molecular.bio_bridge import NeuroBioState, BioLoRABridge
from oneuro.molecular.pharmacology import Drug, DrugCocktail, DRUG_LIBRARY

# Phase 3: Cell biology
from oneuro.molecular.second_messengers import SecondMessengerSystem
from oneuro.molecular.calcium import CalciumSystem
from oneuro.molecular.glia import Astrocyte, Oligodendrocyte, Microglia
from oneuro.molecular.dendrite import DendriticTree, Compartment
from oneuro.molecular.spine import DendriticSpine, SpineState
from oneuro.molecular.metabolism import CellularMetabolism
try:
    from oneuro_metal import CellularMetabolism as RustCellularMetabolism
except ImportError:
    RustCellularMetabolism = None

# Phase 3: Structural biology
from oneuro.molecular.axon import Axon, AxonSegment, AxonSegmentType, NodeOfRanvier
from oneuro.molecular.gap_junction import GapJunction, ConnexinType
from oneuro.molecular.extracellular import ExtracellularSpace, PerineuronalNet
from oneuro.molecular.microtubules import Microtubule, Cytoskeleton

# Phase 3: Network dynamics
from oneuro.molecular.circadian import CircadianSystem, MolecularClock, SleepHomeostasis

# Phase 4: Consciousness & brain architecture
from oneuro.molecular.consciousness import ConsciousnessMonitor, ConsciousnessMetrics, NetworkSnapshot
from oneuro.molecular.brain_regions import (
    Region, CorticalColumn, ThalamicNucleus, Hippocampus, BasalGanglia, RegionalBrain,
)
from oneuro.molecular.validation import (
    CurrentClampMetrics,
    PlasticityMetrics,
    ReferenceRange,
    SynapticResponseMetrics,
    ValidationCheck,
    ValidationReport,
    measure_current_clamp,
    measure_dopamine_gated_plasticity,
    measure_synaptic_response,
    run_validation_suite,
)

# Phase 5: GPU backends
try:
    from oneuro.molecular.cuda_backend import (
        CUDAMolecularBrain,
        CUDARegionalBrain,
        detect_backend,
        create_brain,
        create_regional_brain,
    )
    HAS_CUDA_BACKEND = True
except ImportError:
    HAS_CUDA_BACKEND = False

__all__ = [
    # Backend
    "HAS_NQPU",
    "require_nqpu",
    # Core molecular
    "NeurotransmitterMolecule",
    "NEUROTRANSMITTER_LIBRARY",
    "IonChannelType",
    "IonChannel",
    "IonChannelEnsemble",
    "BatchIonChannelState",
    "SynapticReceptor",
    "ReceptorType",
    "SynapticEnzyme",
    "ENZYME_LIBRARY",
    "MolecularMembrane",
    "GeneExpressionPipeline",
    "GeneID",
    "GeneState",
    "TranscriptionFactor",
    "TranscriptionFactorType",
    "EpigeneticState",
    "MolecularNeuron",
    "NeuronArchetype",
    "MolecularSynapse",
    "MolecularNeuronAdapter",
    "MolecularSynapseAdapter",
    "MolecularNeuralNetwork",
    "NeuroBioState",
    "BioLoRABridge",
    "Drug",
    "DrugCocktail",
    "DRUG_LIBRARY",
    # Phase 3: Cell biology
    "SecondMessengerSystem",
    "CalciumSystem",
    "Astrocyte",
    "Oligodendrocyte",
    "Microglia",
    "DendriticTree",
    "Compartment",
    "DendriticSpine",
    "SpineState",
    "CellularMetabolism",
    "RustCellularMetabolism",
    # Phase 3: Structural biology
    "Axon",
    "AxonSegment",
    "AxonSegmentType",
    "NodeOfRanvier",
    "GapJunction",
    "ConnexinType",
    "ExtracellularSpace",
    "PerineuronalNet",
    "Microtubule",
    "Cytoskeleton",
    # Phase 3: Network dynamics
    "CircadianSystem",
    "MolecularClock",
    "SleepHomeostasis",
    # Phase 4: Consciousness & brain architecture
    "ConsciousnessMonitor",
    "ConsciousnessMetrics",
    "NetworkSnapshot",
    "Region",
    "CorticalColumn",
    "ThalamicNucleus",
    "Hippocampus",
    "BasalGanglia",
    "RegionalBrain",
    "ReferenceRange",
    "ValidationCheck",
    "CurrentClampMetrics",
    "SynapticResponseMetrics",
    "PlasticityMetrics",
    "ValidationReport",
    "measure_current_clamp",
    "measure_synaptic_response",
    "measure_dopamine_gated_plasticity",
    "run_validation_suite",
    # Phase 5: GPU backends
    "CUDAMolecularBrain",
    "CUDARegionalBrain",
    "detect_backend",
    "create_brain",
    "create_regional_brain",
    "HAS_CUDA_BACKEND",
]
