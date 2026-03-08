"""oNeuro — Biologically-inspired neural networks.

Core:
    from oneuro.organic_neural_network import OrganicNeuron, OrganicSynapse, OrganicNeuralNetwork
    from oneuro.multi_tissue_network import MultiTissueNetwork

Molecular (requires nqpu for quantum features):
    from oneuro.molecular import MolecularNeuron, MolecularSynapse, MolecularNeuralNetwork
"""

from oneuro.organic_neural_network import (
    OrganicNeuron,
    OrganicSynapse,
    OrganicNeuralNetwork,
    NeuronState,
    TrainingTask,
    XORTask,
    PatternRecognitionTask,
    EmergenceTracker,
)
from oneuro.multi_tissue_network import MultiTissueNetwork, TissueType, TissueConfig
