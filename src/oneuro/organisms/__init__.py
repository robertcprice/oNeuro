"""oNeuro organism models -- biophysically faithful digital organisms.

Each organism consists of:
  - A nervous system built on CUDAMolecularBrain (HH neurons, NTs, STDP)
  - A body with sensory and motor systems
  - A combined Organism class that runs the sense-think-act loop

Available organisms:
  - Drosophila melanogaster (fruit fly): ~139K neurons, compound eyes, olfaction
"""

from oneuro.organisms.drosophila import Drosophila, DrosophilaBrain, DrosophilaBody
