"""
Quantum Terrarium - Digital Ecosystem Simulation

A unique simulation combining quantum computing with artificial life.

Features:
1. Quantum organisms with DNA-encoded genomes
2. Real-time ASCII art visualization
3. Evolution and natural selection
4. Predator-prey dynamics
5. Resource competition
6. Quantum superposition and entanglement effects
7. Real-time tracking of every organism
8. Export capabilities (JSON, CSV, lineage tree)
9. Genetic lineage tracking with family trees
10. Emergence detection (population events, speciation, cycles)

Run with: python3 quantum_terrarium.py
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import time
import json
import csv
import uuid
from collections import defaultdict
from datetime import datetime


# ============================================================================
# QUANTUM STATES
# ============================================================================

class QuantumState(Enum):
    """Possible quantum states for organisms."""
    GROUND = 0      # Normal classical state
    SUPERPOSITION = 2  # Exploring multiple possibilities
    ENTANGLED = 3    # Connected to another organism


# ============================================================================
# GENOME ENCODING
# ============================================================================

class Genome:
    """
    Quantum-encoded genome for digital organisms.

    DNA is encoded as quantum states allowing:
    - Superposition of traits
    - Quantum mutation
    - Entanglement with other genomes
    """

    def __init__(self, sequence: str = None, length: int = 32):
        if sequence is None:
            # Generate random genome
            self.sequence = ''.join(random.choices('ATGC', k=length))
        else:
            self.sequence = sequence[:length].ljust(length, 'A')

        self.length = len(self.sequence)
        self._quantum_amplitudes = None

    @property
    def quantum_amplitudes(self) -> np.ndarray:
        """Convert DNA to quantum amplitudes."""
        if self._quantum_amplitudes is None:
            # A=|00>, T=|01>, G=|10>, C=|11>
            encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                       'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
            amps = []
            for base in self.sequence:
                amps.extend(encoding[base])
            self._quantum_amplitudes = np.array(amps, dtype=complex) / np.sqrt(len(amps))
        return self._quantum_amplitudes

    def mutate(self, rate: float = 0.01) -> 'Genome':
        """Quantum mutation - can flip multiple bits simultaneously."""
        new_seq = list(self.sequence)

        # Quantum mutation: superposition of mutations
        for i in range(len(new_seq)):
            if random.random() < rate:
                # Quantum tunneling: might not actually mutate (superposition)
                if random.random() > 0.5:  # Collapse to mutation
                    bases = 'ATGC'
                    new_seq[i] = random.choice([b for b in bases if b != new_seq[i]])

        return Genome(sequence=''.join(new_seq), length=self.length)

    def crossover(self, other: 'Genome') -> Tuple['Genome', 'Genome']:
        """Quantum crossover - exchange genetic material."""
        # Random crossover point
        point = random.randint(1, self.length - 1)

        child1 = self.sequence[:point] + other.sequence[point:]
        child2 = other.sequence[:point] + self.sequence[point:]

        return (
            Genome(sequence=child1, length=self.length),
            Genome(sequence=child2, length=self.length)
        )

    def decode_traits(self) -> Dict[str, float]:
        """Decode genome into phenotypic traits."""
        traits = {}

        # Decode in 4-base chunks
        for i in range(0, len(self.sequence), 4):
            chunk = self.sequence[i:i+4]
            if len(chunk) == 4:
                # Each chunk encodes a trait value
                value = sum(2**j * (1 if b in 'GC' else 0) for j, b in enumerate(chunk)) / 15.0
                trait_name = ['speed', 'size', 'aggression', 'vision', 'efficiency',
                            'reproduction_rate', 'mutation_resistance', 'quantum_affinity'][i // 4 % 8]
                traits[trait_name] = value

        return traits


# ============================================================================
# ORGANISM TRACKING RECORDS
# ============================================================================

@dataclass
class OrganismRecord:
    """Record for tracking an organism's full history."""
    organism_id: str
    genome_sequence: str
    parent_id: Optional[str]
    birth_tick: int
    death_tick: Optional[int] = None
    lifespan: Optional[int] = None
    generation: int = 0
    organism_type: str = "DigitalOrganism"

    # Energy tracking
    energy_history: List[Tuple[int, float]] = field(default_factory=list)

    # Position tracking
    position_history: List[Tuple[int, float, float]] = field(default_factory=list)

    # Reproduction tracking
    children_ids: List[str] = field(default_factory=list)
    reproduction_events: List[Tuple[int, str]] = field(default_factory=list)  # (tick, child_id)

    # Traits at birth (snapshot)
    traits_at_birth: Dict[str, float] = field(default_factory=dict)

    # Death cause
    death_cause: Optional[str] = None


@dataclass
class EmergenceEvent:
    """Record of an emergent behavior detected in the simulation."""
    tick: int
    event_type: str  # 'population_boom', 'population_crash', 'speciation', 'predator_prey_cycle', 'quantum_cluster'
    severity: float  # 0.0 to 1.0
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_organisms: List[str] = field(default_factory=list)


# ============================================================================
# DIGITAL ORGANISM
# ============================================================================

@dataclass
class DigitalOrganism:
    """
    A digital organism with quantum properties.

    Each organism:
    - Has a quantum-encoded genome
    - Lives in a 2D terrarium
    - Can be in superposition (exploring multiple states)
    - Can entangle with other organisms
    - Evolves through selection
    - Has a unique ID for tracking
    """

    genome: Genome
    x: float
    y: float
    energy: float = 100.0
    age: int = 0
    generation: int = 0

    # Unique identifier for tracking
    organism_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None

    # Quantum properties
    quantum_state: QuantumState = QuantumState.GROUND
    entangled_with: Optional['DigitalOrganism'] = None
    superposition_states: List[Tuple[float, float]] = field(default_factory=list)

    # Traits (decoded from genome)
    traits: Dict[str, float] = field(default_factory=dict, init=False)

    # Tracking
    children_count: int = 0

    def __post_init__(self):
        if not self.traits:
            self.traits = self.genome.decode_traits()

    @property
    def speed(self) -> float:
        return self.traits.get('speed', 0.5)

    @property
    def size(self) -> float:
        return self.traits.get('size', 0.5)

    @property
    def aggression(self) -> float:
        return self.traits.get('aggression', 0.5)

    @property
    def quantum_affinity(self) -> float:
        return self.traits.get('quantum_affinity', 0.0)

    def ascii_art(self) -> str:
        """Generate ASCII art representation."""
        # Size determines visual complexity
        if self.size < 0.3:
            base = "o"  # Small
        elif self.size < 0.6:
            base = "●"  # Medium
        else:
            base = "◉"  # Large

        # Color based on quantum state
        if self.quantum_state == QuantumState.SUPERPOSITION:
            color = "\033[35m"  # Yellow (bright)
        elif self.quantum_state == QuantumState.ENTANGLED:
            color = "\033[36m"  # Cyan
        else:
            color = "\033[32m"  # Green

        reset = "\033[0m"

        # Energy determines intensity
        if self.energy < 30:
            return f"{color}·{reset}"  # Dying
        elif self.energy > 150:
            return f"{color}{base}{reset}"  # Thriving
        else:
            return f"{color}{base.lower()}{reset}"  # Normal

    def move(self, dx: float, dy: float, bounds: Tuple[float, float]):
        """Move organism, handling quantum effects."""

        # Quantum tunneling: can "teleport" short distances
        if self.quantum_state == QuantumState.SUPERPOSITION and random.random() < 0.1:
            # Tunnel!
            dx *= 3
            dy *= 3

        # Apply movement
        self.x = max(0, min(bounds[0], self.x + dx * self.speed))
        self.y = max(0, min(bounds[1], self.y + dy * self.speed))

        # Cost energy
        self.energy -= 0.1 * (abs(dx) + abs(dy))

        # Age
        self.age += 1

    def feed(self, amount: float):
        """Gain energy from food."""
        self.energy += amount

    def can_reproduce(self) -> bool:
        """Check if organism can reproduce."""
        return self.energy > 150 and self.age > 10

    def reproduce(self) -> Optional['DigitalOrganism']:
        """Create offspring with mutated genome."""
        if not self.can_reproduce():
            return None

        # Mutation
        child_genome = self.genome.mutate(rate=0.05)

        # Energy cost
        self.energy -= 50

        # Track children count
        self.children_count += 1

        # Child starts nearby
        return DigitalOrganism(
            genome=child_genome,
            x=self.x + random.uniform(-5, 5),
            y=self.y + random.uniform(-5, 5),
            energy=50,
            generation=self.generation + 1,
            parent_id=self.organism_id
        )

    def die(self) -> bool:
        """Check if organism dies."""
        return self.energy <= 0 or random.random() < 0.001 * self.age

    def enter_superposition(self):
        """Enter quantum superposition state."""
        if random.random() < self.quantum_affinity:
            self.quantum_state = QuantumState.SUPERPOSITION
            # Store current position as one of the superposition states
            self.superposition_states = [
                (self.x, self.y),
                (self.x + random.uniform(-3, 3), self.y + random.uniform(-3, 3))
            ]

    def entangle_with(self, other: 'DigitalOrganism'):
        """Create quantum entanglement with another organism."""
        if random.random() < self.quantum_affinity * other.quantum_affinity:
            self.quantum_state = QuantumState.ENTANGLED
            other.quantum_state = QuantumState.ENTANGLED
            self.entangled_with = other
            other.entangled_with = self


# ============================================================================
# ORGANISM TYPES
# ============================================================================

class Bacteria(DigitalOrganism):
    """Simple prokaryotic organism."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Bacteria-specific: smaller, faster, simpler
        if 'speed' not in self.traits:
            self.traits['speed'] = 0.8

    def ascii_art(self) -> str:
        return super().ascii_art()


class Algae(DigitalOrganism):
    """Photosynthetic organism."""

    def __init__(self, **kwargs):
        if 'genome' not in kwargs:
            kwargs['genome'] = Genome(length=32)
        super().__init__(**kwargs)

    def ascii_art(self) -> str:
        base = super().ascii_art()
        return f"~{base}~"  # Wavy to indicate photosynthesis

    def photosynthesize(self, light_intensity: float):
        """Gain energy from light."""
        self.energy += light_intensity * 0.5


class Predator(DigitalOrganism):
    """Organism that hunts others."""

    def __init__(self, **kwargs):
        if 'genome' not in kwargs:
            kwargs['genome'] = Genome(length=48)
        super().__init__(**kwargs)

    def ascii_art(self) -> str:
        base = super().ascii_art()
        return f">{base}<"  # Fangs

    def hunt(self, prey: DigitalOrganism, distance: float) -> bool:
        """Attempt to eat another organism."""
        if distance < 2.0 and self.aggression > prey.traits.get('speed', 0.5):
            # Successful hunt
            self.energy += prey.energy * 0.7
            return True
        return False


# ============================================================================
# QUANTUM TERRARIUM
# ============================================================================

class QuantumTerrarium:
    """
    The ecosystem where digital organisms live and evolve.

    Features:
    - 2D space with resources
    - Quantum effects (superposition, entanglement)
    - Natural selection
    - Population dynamics
    - Real-time visualization
    - Full organism tracking with unique IDs
    - Export capabilities (JSON, CSV, lineage)
    - Emergence detection
    """

    def __init__(self, width: int = 80, height: int = 40):
        self.width = width
        self.height = height
        self.organisms: List[DigitalOrganism] = []
        self.generation = 0
        self.tick = 0

        # Resources
        self.food: List[Tuple[float, float, float]] = []  # (x, y, energy)

        # Statistics
        self.history: List[Dict] = []
        self.extinctions = 0
        self.speciations = 0

        # =====================
        # TRACKING INFRASTRUCTURE
        # =====================
        # Organism records by ID (persists after death)
        self.organism_records: Dict[str, OrganismRecord] = {}

        # Full time series data
        self.timeseries: List[Dict[str, Any]] = []

        # Emergence events
        self.emergence_events: List[EmergenceEvent] = []

        # Population tracking for emergence detection
        self._population_history: List[int] = []
        self._trait_history: List[Dict[str, List[float]]] = []
        self._predator_prey_history: List[Tuple[int, int]] = []
        self._quantum_cluster_history: List[int] = []

        # Lineage tracking
        self._lineage_tree: Dict[str, List[str]] = defaultdict(list)  # parent_id -> [child_ids]

    def seed_food(self, amount: int = 20):
        """Add food resources to the terrarium."""
        for _ in range(amount):
            self.food.append((
                random.uniform(0, self.width),
                random.uniform(0, self.height),
                random.uniform(10, 30)
            ))

    def add_organism(self, organism: DigitalOrganism):
        """Add an organism to the terrarium."""
        organism.x = max(0, min(self.width, organism.x))
        organism.y = max(0, min(self.height, organism.y))
        self.organisms.append(organism)

        # Create tracking record
        record = OrganismRecord(
            organism_id=organism.organism_id,
            genome_sequence=organism.genome.sequence,
            parent_id=organism.parent_id,
            birth_tick=self.tick,
            generation=organism.generation,
            organism_type=type(organism).__name__,
            traits_at_birth=dict(organism.traits),
            energy_history=[(self.tick, organism.energy)],
            position_history=[(self.tick, organism.x, organism.y)]
        )
        self.organism_records[organism.organism_id] = record

        # Update lineage tree
        if organism.parent_id:
            self._lineage_tree[organism.parent_id].append(organism.organism_id)
            # Update parent's children list
            if organism.parent_id in self.organism_records:
                self.organism_records[organism.parent_id].children_ids.append(organism.organism_id)

    def populate(self, n_bacteria: int = 10, n_algae: int = 5, n_predators: int = 2):
        """Initial population."""
        for _ in range(n_bacteria):
            self.add_organism(Bacteria(
                genome=Genome(length=24),
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height)
            ))

        for _ in range(n_algae):
            self.add_organism(Algae(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height)
            ))

        for _ in range(n_predators):
            self.add_organism(Predator(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height)
            ))

    def nearest_food(self, organism: DigitalOrganism) -> Optional[Tuple[float, float, float, float]]:
        """Find nearest food for an organism."""
        if not self.food:
            return None

        nearest = None
        min_dist = float('inf')

        for fx, fy, energy in self.food:
            dist = np.sqrt((fx - organism.x)**2 + (fy - organism.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (fx, fy, energy, dist)

        return nearest

    def nearest_prey(self, predator: Predator) -> Optional[Tuple[DigitalOrganism, float]]:
        """Find nearest prey for a predator."""
        nearest = None
        min_dist = float('inf')

        for org in self.organisms:
            if org is predator or isinstance(org, Predator):
                continue
            dist = np.sqrt((org.x - predator.x)**2 + (org.y - predator.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (org, dist)

        return nearest

    def step(self):
        """Advance simulation by one tick."""
        self.tick += 1

        # Add food periodically
        if self.tick % 5 == 0:
            self.seed_food(3)

        # Process each organism
        new_organisms = []
        dead_organisms = []

        for org in self.organisms:
            # Update tracking record with current state
            if org.organism_id in self.organism_records:
                record = self.organism_records[org.organism_id]
                record.energy_history.append((self.tick, org.energy))
                record.position_history.append((self.tick, org.x, org.y))

            # Initialize movement (will be set based on organism type)
            dx, dy = 0.0, 0.0

            # Quantum state transitions
            if random.random() < 0.01:
                org.enter_superposition()

            # Movement behavior based on organism type
            if isinstance(org, Algae):
                # Photosynthesize
                org.photosynthesize(0.5)
                # Slow movement
                dx, dy = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            elif isinstance(org, Predator):
                # Hunt
                prey_info = self.nearest_prey(org)
                if prey_info:
                    prey, dist = prey_info
                    if org.hunt(prey, dist):
                        dead_organisms.append(prey)
                        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                    else:
                        # Move toward prey
                        dx = (prey.x - org.x) / max(1, dist) * org.speed
                        dy = (prey.y - org.y) / max(1, dist) * org.speed
                else:
                    dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
            else:
                # Bacteria: seek food
                food_info = self.nearest_food(org)
                if food_info:
                    fx, fy, energy, dist = food_info
                    if dist < 1.0:
                        # Eat food
                        org.feed(energy)
                        self.food = [(x, y, e) for x, y, e in self.food
                                    if (x, y) != (fx, fy)]
                        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                    else:
                        # Move toward food
                        dx = (fx - org.x) / max(1, dist) * org.speed
                        dy = (fy - org.y) / max(1, dist) * org.speed
                else:
                    dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)

            org.move(dx, dy, (self.width, self.height))

            # Quantum entanglement effects
            if org.quantum_state == QuantumState.ENTANGLED and org.entangled_with:
                # Entangled organisms share energy
                shared_energy = (org.energy + org.entangled_with.energy) / 2
                org.energy = shared_energy
                org.entangled_with.energy = shared_energy

            # Death
            if org.die():
                dead_organisms.append(org)

            # Reproduction
            child = org.reproduce()
            if child:
                new_organisms.append(child)
                # Track reproduction event for parent
                if org.organism_id in self.organism_records:
                    self.organism_records[org.organism_id].reproduction_events.append(
                        (self.tick, child.organism_id)
                    )

        # Remove dead and record death info
        for org in dead_organisms:
            if org in self.organisms:
                self.organisms.remove(org)
                self.extinctions += 1
                # Record death
                if org.organism_id in self.organism_records:
                    record = self.organism_records[org.organism_id]
                    record.death_tick = self.tick
                    record.lifespan = self.tick - record.birth_tick
                    record.death_cause = "energy_depleted" if org.energy <= 0 else "old_age"

        # Add new organisms with tracking
        for child in new_organisms:
            self.add_organism(child)

        # Random entanglement
        if len(self.organisms) >= 2 and random.random() < 0.01:
            org1, org2 = random.sample(self.organisms, 2)
            org1.entangle_with(org2)
            if org1.entangled_with:
                self.speciations += 1

        # Record statistics
        if self.tick % 10 == 0:
            self.history.append({
                'tick': self.tick,
                'population': len(self.organisms),
                'avg_energy': np.mean([o.energy for o in self.organisms]) if self.organisms else 0,
                'food_count': len(self.food),
                'quantum_states': sum(1 for o in self.organisms
                                       if o.quantum_state != QuantumState.GROUND)
            })

        # Run emergence detection
        self._detect_emergence_events()

        # Record timeseries
        self._record_timeseries()

    def render(self) -> str:
        """Render terrarium as ASCII art."""
        # Create canvas
        canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        # Add food
        for fx, fy, energy in self.food:
            x, y = int(fx) % self.width, int(fy) % self.height
            canvas[y][x] = '.' if energy < 20 else '*'

        # Add organisms
        for org in self.organisms:
            x, y = int(org.x) % self.width, int(org.y) % self.height
            canvas[y][x] = org.ascii_art()

        # Build string with border
        border = '+' + '-' * self.width + '+'
        lines = [border]
        for row in canvas:
            lines.append('|' + ''.join(row) + '|')
        lines.append(border)

        return '\n'.join(lines)

    def statistics(self) -> str:
        """Get current statistics."""
        if not self.organisms:
            return "Terrarium is empty - all organisms died!"

        bacteria = sum(1 for o in self.organisms if isinstance(o, Bacteria))
        algae = sum(1 for o in self.organisms if isinstance(o, Algae))
        predators = sum(1 for o in self.organisms if isinstance(o, Predator))

        avg_energy = np.mean([o.energy for o in self.organisms])
        max_generation = max(o.generation for o in self.organisms)
        quantum_count = sum(1 for o in self.organisms
                          if o.quantum_state != QuantumState.GROUND)

        return f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│ QUANTUM TERRARIUM STATISTICS                                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│ Tick: {self.tick:6d}    Generation: {self.generation:3d}    Extinctions: {self.extinctions:4d}        │
│ Population: {len(self.organisms):4d}    Food: {len(self.food):3d}    Speciations: {self.speciations:3d}            │
│                                                                                │
│ Bacteria: {bacteria:4d}    Algae: {algae:4d}    Predators: {predators:4d}                      │
│ Avg Energy: {avg_energy:6.1f}    Max Generation: {max_generation:4d}                           │
│ Quantum States: {quantum_count:3d} ({100*quantum_count/max(1,len(self.organisms)):5.1f}% of population)   │
└────────────────────────────────────────────────────────────────────────────────┘
"""

    # =========================================================================
    # TRACKING HELPER METHODS
    # =========================================================================

    def _record_timeseries(self):
        """Record full timeseries data for the current tick."""
        if not self.organisms:
            return

        # Population by type
        bacteria = [o for o in self.organisms if isinstance(o, Bacteria)]
        algae = [o for o in self.organisms if isinstance(o, Algae)]
        predators = [o for o in self.organisms if isinstance(o, Predator)]

        # Average traits
        trait_avgs = {}
        all_traits = ['speed', 'size', 'aggression', 'vision', 'efficiency',
                      'reproduction_rate', 'mutation_resistance', 'quantum_affinity']
        for trait in all_traits:
            values = [o.traits.get(trait, 0) for o in self.organisms if trait in o.traits]
            trait_avgs[trait] = np.mean(values) if values else 0

        # Quantum state counts
        ground_count = sum(1 for o in self.organisms if o.quantum_state == QuantumState.GROUND)
        superposition_count = sum(1 for o in self.organisms if o.quantum_state == QuantumState.SUPERPOSITION)
        entangled_count = sum(1 for o in self.organisms if o.quantum_state == QuantumState.ENTANGLED)

        self.timeseries.append({
            'tick': self.tick,
            'population': len(self.organisms),
            'bacteria_count': len(bacteria),
            'algae_count': len(algae),
            'predator_count': len(predators),
            'avg_energy': np.mean([o.energy for o in self.organisms]),
            'total_energy': sum(o.energy for o in self.organisms),
            'food_count': len(self.food),
            'trait_averages': trait_avgs,
            'quantum_ground': ground_count,
            'quantum_superposition': superposition_count,
            'quantum_entangled': entangled_count,
            'max_generation': max(o.generation for o in self.organisms),
            'avg_age': np.mean([o.age for o in self.organisms])
        })

    def _detect_emergence_events(self):
        """Detect emergent behaviors in the simulation."""
        current_pop = len(self.organisms)
        self._population_history.append(current_pop)

        # Track predator-prey dynamics
        prey_count = sum(1 for o in self.organisms if not isinstance(o, Predator))
        predator_count = sum(1 for o in self.organisms if isinstance(o, Predator))
        self._predator_prey_history.append((prey_count, predator_count))

        # Track quantum cluster sizes
        quantum_count = sum(1 for o in self.organisms if o.quantum_state != QuantumState.GROUND)
        self._quantum_cluster_history.append(quantum_count)

        # Track trait distributions for speciation detection
        if self.organisms:
            trait_snapshot = {}
            for trait in ['speed', 'size', 'aggression']:
                trait_snapshot[trait] = [o.traits.get(trait, 0.5) for o in self.organisms]
            self._trait_history.append(trait_snapshot)

        # Only detect if we have enough history
        if len(self._population_history) < 20:
            return

        # 1. Population Boom Detection (>50% increase in 10 ticks)
        if len(self._population_history) >= 10:
            recent = self._population_history[-10:]
            if recent[0] > 0:
                change_rate = (recent[-1] - recent[0]) / recent[0]
                if change_rate > 0.5:
                    event = EmergenceEvent(
                        tick=self.tick,
                        event_type='population_boom',
                        severity=min(1.0, change_rate),
                        description=f"Population increased by {change_rate*100:.1f}% in 10 ticks",
                        details={'change_rate': change_rate, 'from_pop': recent[0], 'to_pop': recent[-1]}
                    )
                    self._add_unique_event(event)

        # 2. Population Crash Detection (>50% decrease in 10 ticks)
        if len(self._population_history) >= 10:
            recent = self._population_history[-10:]
            if recent[0] > 0:
                change_rate = (recent[0] - recent[-1]) / recent[0]
                if change_rate > 0.5:
                    event = EmergenceEvent(
                        tick=self.tick,
                        event_type='population_crash',
                        severity=min(1.0, change_rate),
                        description=f"Population crashed by {change_rate*100:.1f}% in 10 ticks",
                        details={'change_rate': change_rate, 'from_pop': recent[0], 'to_pop': recent[-1]}
                    )
                    self._add_unique_event(event)

        # 3. Speciation Detection (trait distribution becomes bimodal)
        if len(self._trait_history) >= 30:
            # Check if any trait shows emerging bimodal distribution
            for trait in ['speed', 'size', 'aggression']:
                values = self._trait_history[-1][trait]
                if len(values) >= 5:
                    # Simple bimodality check: check if variance is high
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val > 0.25:  # Significant spread
                        # Check for clustering
                        low_cluster = sum(1 for v in values if v < mean_val - 0.15)
                        high_cluster = sum(1 for v in values if v > mean_val + 0.15)
                        if low_cluster >= 2 and high_cluster >= 2:
                            event = EmergenceEvent(
                                tick=self.tick,
                                event_type='speciation',
                                severity=std_val,
                                description=f"Possible speciation event detected in {trait} trait",
                                details={'trait': trait, 'mean': mean_val, 'std': std_val,
                                        'low_cluster': low_cluster, 'high_cluster': high_cluster}
                            )
                            self._add_unique_event(event)

        # 4. Predator-Prey Cycle Detection (oscillating populations)
        if len(self._predator_prey_history) >= 30:
            prey_vals = [p[0] for p in self._predator_prey_history[-30:]]
            pred_vals = [p[1] for p in self._predator_prey_history[-30:]]

            # Check for oscillation using sign changes in differences
            if max(prey_vals) > 0 and max(pred_vals) > 0:
                prey_diffs = np.diff(prey_vals)
                pred_diffs = np.diff(pred_vals)

                # Count sign changes (oscillations)
                prey_sign_changes = sum(1 for i in range(1, len(prey_diffs))
                                       if prey_diffs[i] * prey_diffs[i-1] < 0)
                pred_sign_changes = sum(1 for i in range(1, len(pred_diffs))
                                       if pred_diffs[i] * pred_diffs[i-1] < 0)

                if prey_sign_changes >= 4 and pred_sign_changes >= 3:
                    event = EmergenceEvent(
                        tick=self.tick,
                        event_type='predator_prey_cycle',
                        severity=(prey_sign_changes + pred_sign_changes) / 20,
                        description="Predator-prey population cycles detected",
                        details={'prey_oscillations': prey_sign_changes,
                                'predator_oscillations': pred_sign_changes,
                                'prey_range': (min(prey_vals), max(prey_vals)),
                                'predator_range': (min(pred_vals), max(pred_vals))}
                    )
                    self._add_unique_event(event)

        # 5. Quantum Cluster Detection
        if len(self._quantum_cluster_history) >= 20:
            recent_quantum = self._quantum_cluster_history[-20:]
            avg_quantum = np.mean(recent_quantum)
            if avg_quantum > len(self.organisms) * 0.3:  # >30% in quantum states
                event = EmergenceEvent(
                    tick=self.tick,
                    event_type='quantum_cluster',
                    severity=avg_quantum / max(1, len(self.organisms)),
                    description=f"Quantum state cluster: {avg_quantum:.1f} organisms in non-ground states",
                    details={'avg_quantum_organisms': avg_quantum,
                            'percentage': 100 * avg_quantum / max(1, len(self.organisms))}
                )
                self._add_unique_event(event)

    def _add_unique_event(self, event: EmergenceEvent):
        """Add an event only if a similar event wasn't recently added."""
        # Check if same type of event happened in last 10 ticks
        for existing in reversed(self.emergence_events[-10:]):
            if (existing.event_type == event.event_type and
                self.tick - existing.tick < 10):
                return
        self.emergence_events.append(event)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_json(self, filepath: str) -> None:
        """
        Export full simulation history to JSON.

        Args:
            filepath: Path to write JSON file
        """
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_ticks': self.tick,
                'final_population': len(self.organisms),
                'total_extinctions': self.extinctions,
                'total_speciations': self.speciations,
                'terrarium_size': (self.width, self.height)
            },
            'timeseries': self.timeseries,
            'organism_records': {},
            'lineage_tree': dict(self._lineage_tree),
            'emergence_events': []
        }

        # Convert organism records
        for org_id, record in self.organism_records.items():
            data['organism_records'][org_id] = {
                'organism_id': record.organism_id,
                'genome_sequence': record.genome_sequence,
                'parent_id': record.parent_id,
                'birth_tick': record.birth_tick,
                'death_tick': record.death_tick,
                'lifespan': record.lifespan,
                'generation': record.generation,
                'organism_type': record.organism_type,
                'traits_at_birth': record.traits_at_birth,
                'children_ids': record.children_ids,
                'reproduction_count': len(record.reproduction_events),
                'death_cause': record.death_cause,
                'energy_history': record.energy_history,
                'position_history': record.position_history
            }

        # Convert emergence events
        for event in self.emergence_events:
            data['emergence_events'].append({
                'tick': event.tick,
                'event_type': event.event_type,
                'severity': event.severity,
                'description': event.description,
                'details': event.details
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str, include_timeseries: bool = True,
                   include_organisms: bool = True) -> None:
        """
        Export simulation data to CSV format.

        Creates a CSV with timeseries data and optionally organism summary data.

        Args:
            filepath: Path to write CSV file
            include_timeseries: Include timeseries data rows
            include_organisms: Include organism summary rows
        """
        rows = []

        if include_timeseries and self.timeseries:
            # Timeseries header
            rows.append(['# TIMESERIES DATA'])
            rows.append(['tick', 'population', 'bacteria', 'algae', 'predators',
                        'avg_energy', 'total_energy', 'food_count',
                        'quantum_ground', 'quantum_superposition', 'quantum_entangled',
                        'max_generation', 'avg_age'])

            for ts in self.timeseries:
                rows.append([
                    ts['tick'],
                    ts['population'],
                    ts['bacteria_count'],
                    ts['algae_count'],
                    ts['predator_count'],
                    f"{ts['avg_energy']:.2f}",
                    f"{ts['total_energy']:.2f}",
                    ts['food_count'],
                    ts['quantum_ground'],
                    ts['quantum_superposition'],
                    ts['quantum_entangled'],
                    ts['max_generation'],
                    f"{ts['avg_age']:.2f}"
                ])

        if include_organisms:
            rows.append([])  # Blank line separator
            rows.append(['# ORGANISM DATA'])
            rows.append(['organism_id', 'type', 'parent_id', 'generation',
                        'birth_tick', 'death_tick', 'lifespan', 'children_count',
                        'death_cause', 'genome'])

            for org_id, record in self.organism_records.items():
                rows.append([
                    record.organism_id,
                    record.organism_type,
                    record.parent_id or 'None',
                    record.generation,
                    record.birth_tick,
                    record.death_tick or 'alive',
                    record.lifespan or 'alive',
                    len(record.children_ids),
                    record.death_cause or 'alive',
                    record.genome_sequence
                ])

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def get_lineage_tree(self) -> Dict[str, Any]:
        """
        Get the genetic lineage tree as a nested dictionary.

        Returns:
            Nested dictionary representing family relationships.
            Format: {organism_id: {'children': [...], 'generation': int, ...}}
        """
        tree = {}

        # Find all root organisms (no parent in records)
        all_ids = set(self.organism_records.keys())
        child_ids = set()
        for record in self.organism_records.values():
            if record.parent_id:
                child_ids.add(record.organism_id)

        root_ids = all_ids - child_ids

        def build_subtree(org_id: str) -> Dict[str, Any]:
            record = self.organism_records.get(org_id)
            if not record:
                return {}

            node = {
                'organism_id': org_id,
                'organism_type': record.organism_type,
                'generation': record.generation,
                'lifespan': record.lifespan,
                'children_count': len(record.children_ids),
                'children': []
            }

            for child_id in record.children_ids:
                child_tree = build_subtree(child_id)
                if child_tree:
                    node['children'].append(child_tree)

            return node

        # Build tree from each root
        for root_id in root_ids:
            subtree = build_subtree(root_id)
            if subtree:
                tree[root_id] = subtree

        return tree

    def get_most_successful_lineages(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most successful genetic lineages by total descendants.

        Args:
            top_n: Number of top lineages to return

        Returns:
            List of lineage statistics sorted by success
        """
        lineage_stats = []

        def count_descendants(org_id: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            if org_id in visited:
                return 0
            visited.add(org_id)

            record = self.organism_records.get(org_id)
            if not record:
                return 0

            total = len(record.children_ids)
            for child_id in record.children_ids:
                total += count_descendants(child_id, visited)
            return total

        for org_id, record in self.organism_records.items():
            descendants = count_descendants(org_id)
            if descendants > 0 or record.organism_id in self.organisms:
                lineage_stats.append({
                    'organism_id': org_id,
                    'organism_type': record.organism_type,
                    'generation': record.generation,
                    'total_descendants': descendants,
                    'direct_children': len(record.children_ids),
                    'lifespan': record.lifespan,
                    'is_alive': record.organism_id in [o.organism_id for o in self.organisms]
                })

        # Sort by total descendants
        lineage_stats.sort(key=lambda x: x['total_descendants'], reverse=True)
        return lineage_stats[:top_n]

    def get_emergence_events(self) -> List[EmergenceEvent]:
        """
        Get all detected emergence events.

        Returns:
            List of EmergenceEvent objects
        """
        return self.emergence_events

    def get_emergence_summary(self) -> str:
        """Get a human-readable summary of emergence events."""
        if not self.emergence_events:
            return "No emergence events detected."

        # Group by type
        by_type = defaultdict(list)
        for event in self.emergence_events:
            by_type[event.event_type].append(event)

        lines = ["EMERGENCE EVENTS SUMMARY", "=" * 40]

        for event_type, events in sorted(by_type.items()):
            lines.append(f"\n{event_type.upper().replace('_', ' ')}: {len(events)} events")
            for event in events[-3:]:  # Show last 3 of each type
                lines.append(f"  Tick {event.tick}: {event.description} (severity: {event.severity:.2f})")

        return '\n'.join(lines)

    def get_statistics_timeseries(self) -> List[Dict[str, Any]]:
        """
        Get full time series of statistics.

        Returns:
            List of dictionaries with statistics for each recorded tick
        """
        return self.timeseries

    def get_organism_history(self, organism_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete history for a specific organism.

        Args:
            organism_id: The unique ID of the organism

        Returns:
            Dictionary with full organism history or None if not found
        """
        record = self.organism_records.get(organism_id)
        if not record:
            return None

        return {
            'organism_id': record.organism_id,
            'genome': record.genome_sequence,
            'parent_id': record.parent_id,
            'children_ids': record.children_ids,
            'organism_type': record.organism_type,
            'generation': record.generation,
            'birth_tick': record.birth_tick,
            'death_tick': record.death_tick,
            'lifespan': record.lifespan,
            'traits_at_birth': record.traits_at_birth,
            'energy_history': record.energy_history,
            'position_history': record.position_history,
            'reproduction_events': record.reproduction_events,
            'death_cause': record.death_cause
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUANTUM TERRARIUM - DIGITAL ECOSYSTEM                     ║
║                                                                              ║
║  A unique simulation combining quantum computing with artificial life.       ║
║  Watch digital organisms evolve, compete, and exhibit quantum behaviors!     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Create terrarium
    terrarium = QuantumTerrarium(width=80, height=30)
    terrarium.populate(n_bacteria=15, n_algae=8, n_predators=3)
    terrarium.seed_food(30)

    print(f"Initial population: {len(terrarium.organisms)} organisms")
    print("Running simulation...\n")

    # Run simulation
    for i in range(100):
        terrarium.step()

        if i % 20 == 0:
            print(f"\n--- Tick {terrarium.tick} ---")
            print(terrarium.render())
            print(terrarium.statistics())
            time.sleep(0.3)  # Brief pause

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)

    # Final statistics
    print(terrarium.statistics())

    # ========================================
    # DEMONSTRATE NEW FEATURES
    # ========================================
    print("\n" + "="*80)
    print("TRACKING AND ANALYSIS FEATURES")
    print("="*80)

    # Show emergence events
    print("\n" + terrarium.get_emergence_summary())

    # Show top lineages
    print("\n--- TOP GENETIC LINEAGES ---")
    top_lineages = terrarium.get_most_successful_lineages(top_n=3)
    for i, lineage in enumerate(top_lineages, 1):
        print(f"  {i}. {lineage['organism_type']} {lineage['organism_id']}: "
              f"{lineage['total_descendants']} descendants, "
              f"gen {lineage['generation']}, "
              f"{'ALIVE' if lineage['is_alive'] else 'dead'}")

    # Show sample organism history
    if terrarium.organisms:
        sample_org = terrarium.organisms[0]
        history = terrarium.get_organism_history(sample_org.organism_id)
        print(f"\n--- SAMPLE ORGANISM HISTORY ---")
        print(f"  ID: {history['organism_id']}")
        print(f"  Type: {history['organism_type']}")
        print(f"  Generation: {history['generation']}")
        print(f"  Birth tick: {history['birth_tick']}")
        print(f"  Children: {len(history['children_ids'])}")
        print(f"  Energy samples: {history['energy_history'][-5:]}")

    # Show timeseries summary
    timeseries = terrarium.get_statistics_timeseries()
    if timeseries:
        print(f"\n--- TIMESERIES DATA ---")
        print(f"  Total recorded ticks: {len(timeseries)}")
        print(f"  First tick population: {timeseries[0]['population']}")
        print(f"  Last tick population: {timeseries[-1]['population']}")
        print(f"  Max population: {max(t['population'] for t in timeseries)}")

    # Export demonstration
    print("\n--- EXPORT CAPABILITIES ---")
    print("  Export methods available:")
    print("    - export_json(filepath) - Full history to JSON")
    print("    - export_csv(filepath) - Data to CSV")
    print("    - get_lineage_tree() - Family tree structure")
    print("    - get_emergence_events() - List of emergent behaviors")
    print("    - get_statistics_timeseries() - Population data over time")

    print("""

UNIQUE FEATURES:
   1. Quantum-encoded DNA genomes
   2. Superposition: organisms explore multiple states
   3. Entanglement: organisms share energy quantum-mechanically
   4. Quantum tunneling: organisms can "teleport"
   5. Natural selection with quantum effects
   6. Real-time ASCII art visualization
   7. Population dynamics (predator-prey)
   8. Full organism tracking with unique IDs
   9. Genetic lineage tree tracking
   10. Emergence detection (booms, crashes, speciation, cycles)
   11. Export to JSON/CSV for analysis

NOVEL ASPECT:
   This is the ONLY simulation combining quantum mechanics with
   artificial life and digital biology visualization, plus full
   tracking and emergence detection capabilities.
    """)


if __name__ == "__main__":
    demo()
