"""Axonal conduction model with myelination and saltatory propagation.

Models signal propagation along axons with biophysically accurate conduction
velocities. Myelinated axons use saltatory conduction (jumping between Nodes
of Ranvier), while unmyelinated axons conduct continuously.

Key relationships:
  - Myelinated: v = 6.0 * diameter_um  (Hursh's law)
  - Unmyelinated: v = sqrt(diameter_um) * 1.0
  - Propagation delay = total_length / velocity

Nodes of Ranvier have 10x Na_v channel density for AP regeneration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AxonSegmentType(Enum):
    """Types of axon segments with distinct conduction properties."""

    UNMYELINATED = "unmyelinated"
    MYELINATED = "myelinated"
    NODE_OF_RANVIER = "node_of_ranvier"


# Typical Na_v channel density per um^2 of membrane
_BASE_NAV_DENSITY = 100  # channels / um^2
_NODE_NAV_DENSITY = 1000  # 10x at Nodes of Ranvier


@dataclass
class AxonSegment:
    """A single segment of axon with defined geometry and myelination.

    Attributes:
        segment_type: Whether this segment is myelinated, unmyelinated,
            or a Node of Ranvier.
        length_um: Length of the segment in micrometers (typical: 100-1000).
        diameter_um: Axon diameter in micrometers (typical: 0.2-20).
        myelin_thickness: Relative myelin thickness 0-1, set by
            oligodendrocyte wrapping. 0 = no myelin, 1 = maximum.
    """

    segment_type: AxonSegmentType
    length_um: float
    diameter_um: float
    myelin_thickness: float = 0.0

    def __post_init__(self) -> None:
        if self.length_um <= 0:
            raise ValueError(f"Segment length must be positive, got {self.length_um}")
        if self.diameter_um <= 0:
            raise ValueError(f"Segment diameter must be positive, got {self.diameter_um}")
        if not 0.0 <= self.myelin_thickness <= 1.0:
            raise ValueError(
                f"Myelin thickness must be in [0, 1], got {self.myelin_thickness}"
            )

        # Enforce consistency: myelinated segments need myelin
        if self.segment_type == AxonSegmentType.MYELINATED and self.myelin_thickness == 0.0:
            self.myelin_thickness = 0.6  # Default wrapping

        # Nodes and unmyelinated segments have no myelin
        if self.segment_type in (AxonSegmentType.UNMYELINATED, AxonSegmentType.NODE_OF_RANVIER):
            self.myelin_thickness = 0.0

    @property
    def nav_density(self) -> int:
        """Na_v channel density (channels / um^2) for this segment."""
        if self.segment_type == AxonSegmentType.NODE_OF_RANVIER:
            return _NODE_NAV_DENSITY
        elif self.segment_type == AxonSegmentType.MYELINATED:
            # Myelin insulates -- very few channels needed
            return int(_BASE_NAV_DENSITY * 0.01)
        else:
            return _BASE_NAV_DENSITY

    @property
    def conduction_velocity_ms(self) -> float:
        """Conduction velocity in m/s for this individual segment.

        Myelinated: v = 6.0 * diameter (Hursh's law, 1939).
        Unmyelinated: v = sqrt(diameter) * 1.0 (cable theory).
        Node of Ranvier: treated as unmyelinated for local velocity,
            but AP regeneration enables saltatory conduction globally.
        """
        if self.segment_type == AxonSegmentType.MYELINATED:
            return 6.0 * self.diameter_um
        else:
            # Unmyelinated and Nodes of Ranvier
            return math.sqrt(self.diameter_um) * 1.0

    @property
    def segment_delay_ms(self) -> float:
        """Time for signal to traverse this segment, in milliseconds.

        delay = length_um / (velocity_m_s * 1000)
        where factor of 1000 converts m/s to um/ms.
        """
        v = self.conduction_velocity_ms
        if v <= 0:
            return float("inf")
        # velocity in m/s = velocity in um/ms * 1e-3
        # length_um / (velocity_m_s * 1000 um_per_mm) but need um->ms
        # v m/s = v * 1e3 um/ms ... wait, 1 m = 1e6 um, 1 s = 1e3 ms
        # v m/s = v * 1e6 um / 1e3 ms = v * 1e3 um/ms
        return self.length_um / (v * 1e3)


@dataclass
class NodeOfRanvier:
    """A Node of Ranvier: gap in myelin sheath for AP regeneration.

    Nodes are ~1 um long with high-density Na_v channels (10x normal).
    Each node regenerates the action potential during saltatory conduction,
    preventing signal attenuation along long myelinated axons.
    """

    diameter_um: float = 1.0
    nav_channel_count: int = _NODE_NAV_DENSITY
    length_um: float = 1.0

    # AP regeneration adds a small delay (~0.01 ms per node)
    regeneration_delay_ms: float = 0.01

    def to_segment(self) -> AxonSegment:
        """Convert to an AxonSegment for inclusion in an Axon."""
        return AxonSegment(
            segment_type=AxonSegmentType.NODE_OF_RANVIER,
            length_um=self.length_um,
            diameter_um=self.diameter_um,
            myelin_thickness=0.0,
        )


@dataclass
class Axon:
    """Complete axon model with segments, myelination, and conduction delay.

    Computes biophysically grounded conduction velocity and propagation
    delay based on segment geometry and myelination state.

    Examples:
        >>> axon = Axon.unmyelinated(length_um=500.0, diameter_um=0.5)
        >>> axon.conduction_velocity()  # ~0.71 m/s
        >>> axon.propagation_delay()    # ~0.71 ms

        >>> axon = Axon.myelinated(length_um=1000.0, diameter_um=2.0, n_segments=5)
        >>> axon.conduction_velocity()  # ~12 m/s (Hursh's law)
        >>> axon.propagation_delay()    # ~0.08 ms
    """

    segments: List[AxonSegment] = field(default_factory=list)

    # Optional metadata
    source_neuron_id: Optional[int] = None

    @property
    def total_length_um(self) -> float:
        """Total axon length in micrometers (sum of all segments)."""
        return sum(seg.length_um for seg in self.segments)

    @property
    def n_nodes(self) -> int:
        """Number of Nodes of Ranvier."""
        return sum(
            1 for seg in self.segments
            if seg.segment_type == AxonSegmentType.NODE_OF_RANVIER
        )

    @property
    def is_myelinated(self) -> bool:
        """Whether this axon has any myelinated segments."""
        return any(
            seg.segment_type == AxonSegmentType.MYELINATED for seg in self.segments
        )

    @property
    def mean_diameter_um(self) -> float:
        """Length-weighted mean diameter across all segments."""
        if not self.segments:
            return 0.0
        total_length = self.total_length_um
        if total_length == 0:
            return 0.0
        return sum(
            seg.diameter_um * seg.length_um for seg in self.segments
        ) / total_length

    def conduction_velocity(self) -> float:
        """Effective conduction velocity in m/s.

        For uniform axons (all same type), returns the type-specific velocity.
        For mixed axons, computes the effective velocity as:
            v_eff = total_length / total_delay
        which naturally weights by segment length.

        Returns:
            Conduction velocity in meters per second.
        """
        if not self.segments:
            return 0.0

        total_delay = self.propagation_delay()
        if total_delay <= 0:
            return float("inf")

        # Convert: total_length in um, delay in ms
        # v = length_um / (delay_ms * 1e3) [um / (um/ms * factor)] => m/s
        return self.total_length_um / (total_delay * 1e3)

    def propagation_delay(self) -> float:
        """Total propagation delay in milliseconds.

        Sum of per-segment delays plus node regeneration delays.

        Returns:
            Propagation delay in ms.
        """
        if not self.segments:
            return 0.0

        delay = 0.0
        for seg in self.segments:
            delay += seg.segment_delay_ms
            # Nodes of Ranvier add regeneration delay
            if seg.segment_type == AxonSegmentType.NODE_OF_RANVIER:
                delay += 0.01  # AP regeneration overhead

        return delay

    def propagate(self, spike_time: float) -> float:
        """Compute arrival time of a spike at the axon terminal.

        Args:
            spike_time: Time of action potential initiation (ms).

        Returns:
            Arrival time at terminal (ms) = spike_time + propagation_delay.
        """
        return spike_time + self.propagation_delay()

    # ---- Factory class methods ----

    @classmethod
    def unmyelinated(cls, length_um: float, diameter_um: float) -> Axon:
        """Create an unmyelinated axon as a single continuous segment.

        Typical of C-fibers (pain), diameter 0.2-2.0 um, velocity <2 m/s.

        Args:
            length_um: Total axon length in micrometers.
            diameter_um: Axon diameter in micrometers.
        """
        segment = AxonSegment(
            segment_type=AxonSegmentType.UNMYELINATED,
            length_um=length_um,
            diameter_um=diameter_um,
            myelin_thickness=0.0,
        )
        return cls(segments=[segment])

    @classmethod
    def myelinated(
        cls,
        length_um: float,
        diameter_um: float,
        n_segments: int = 5,
        myelin_thickness: float = 0.6,
    ) -> Axon:
        """Create a myelinated axon with alternating internodes and Nodes of Ranvier.

        Structure: [myelin][node][myelin][node]...[myelin]
        The n_segments parameter sets the number of myelinated internodes.
        Nodes of Ranvier are inserted between them (n_segments - 1 nodes).

        Typical of A-fibers, diameter 1-20 um, velocity 6-120 m/s.

        Args:
            length_um: Total axon length in micrometers.
            diameter_um: Axon fiber diameter in micrometers.
            n_segments: Number of myelinated internode segments.
            myelin_thickness: Relative myelin thickness (0-1).
        """
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")

        # Nodes of Ranvier are each 1 um long
        n_nodes = max(0, n_segments - 1)
        total_node_length = n_nodes * 1.0  # 1 um per node

        # Remaining length distributed among myelinated internodes
        internode_length = max(
            1.0, (length_um - total_node_length) / n_segments
        )

        segments: List[AxonSegment] = []
        for i in range(n_segments):
            # Myelinated internode
            segments.append(
                AxonSegment(
                    segment_type=AxonSegmentType.MYELINATED,
                    length_um=internode_length,
                    diameter_um=diameter_um,
                    myelin_thickness=myelin_thickness,
                )
            )
            # Node of Ranvier between internodes (not after the last one)
            if i < n_segments - 1:
                node = NodeOfRanvier(diameter_um=diameter_um)
                segments.append(node.to_segment())

        return cls(segments=segments)

    @classmethod
    def from_distance(
        cls,
        distance_um: float,
        myelinated: bool = True,
        diameter_um: float = 1.0,
    ) -> Axon:
        """Auto-create an appropriate axon for a given distance.

        Automatically determines segment count for myelinated axons based
        on typical internode spacing (~100x diameter, capped at 1000 um).

        Args:
            distance_um: Distance between neurons in micrometers.
            myelinated: Whether to create a myelinated axon.
            diameter_um: Axon diameter in micrometers.

        Returns:
            An Axon configured for the given distance and type.
        """
        if not myelinated:
            return cls.unmyelinated(length_um=distance_um, diameter_um=diameter_um)

        # Internode length scales with diameter: ~100 * diameter, max 1000 um
        internode_length = min(1000.0, 100.0 * diameter_um)
        n_segments = max(1, int(math.ceil(distance_um / internode_length)))

        return cls.myelinated(
            length_um=distance_um,
            diameter_um=diameter_um,
            n_segments=n_segments,
        )

    def __repr__(self) -> str:
        kind = "myelinated" if self.is_myelinated else "unmyelinated"
        return (
            f"Axon({kind}, {len(self.segments)} segments, "
            f"{self.total_length_um:.0f} um, "
            f"v={self.conduction_velocity():.1f} m/s, "
            f"delay={self.propagation_delay():.3f} ms)"
        )
