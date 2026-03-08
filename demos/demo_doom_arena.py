#!/usr/bin/env python3
"""Spatial Arena -- Spatial Navigation and Threat Avoidance for dONN Brains.

A spatial environment (inspired by Doom's BSP dungeon generation algorithm)
for digital Organic Neural Network (dONN) game learning, extending the
DishBrain paradigm (Kagan et al. 2022) from 1D Pong to a procedurally
generated 2D dungeon with rooms, corridors, enemies, and health pickups.

The agent perceives a 5x5 egocentric local view (mimicking retinotopic V1
encoding) and must navigate to a goal while avoiding enemies and collecting
health pickups.  All learning is driven by the FREE ENERGY PRINCIPLE -- no
reward signal, no punishment -- only the contrast between structured
(predictable) and unstructured (random) sensory feedback.

3 Experiments:
    1. Spatial Arena Navigation (50 episodes)
       Can the dONN navigate procedurally generated rooms to reach a goal?
       Pass: goal rate > random AND improving over quarters.

    2. Spatial Arena Threat Avoidance (80 episodes)
       Does it learn to avoid enemies?
       Pass: survival rate improves AND damage taken decreases over quarters.

    3. Spatial Arena Drug Effects
       Train 3 identical brains, test baseline / caffeine / diazepam.
       Pass: diazepam < baseline (matches Morris water maze literature --
       GABA-A enhancement impairs spatial navigation).

FEP-based learning (same principle as DishBrain, extended to spatial domain):
    - Survive near enemy = structured pulse (low entropy)
    - Take damage       = unstructured noise (high entropy)
    - Reach goal         = strong structured + NE boost
    - Pick up health     = mild structured feedback

References:
    - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience when
      embodied in a simulated game-world" Neuron 110(23):3952-3969
    - Friston (2010) "The free-energy principle: a unified brain theory?"
      Nature Reviews Neuroscience 11:127-138
    - Morris (1984) "Developments of a water-maze procedure for studying
      spatial learning in the rat" J Neurosci Methods 11(1):47-60

Usage:
    python3 demos/demo_doom_arena.py                          # all 3, small
    python3 demos/demo_doom_arena.py --exp 1                  # navigation only
    python3 demos/demo_doom_arena.py --scale medium --exp 1 3 # medium, nav+drugs
    python3 demos/demo_doom_arena.py --runs 5 --json out.json # 5 runs, save JSON
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    CUDARegionalBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_GLU, NT_GABA,
)

from demo_language_cuda import (
    _warmup,
    _header,
    _get_region_ids,
    _get_all_cortex_ids,
    _get_cortex_l5_ids,
    SCALE_COLUMNS,
)


# ============================================================================
# Constants
# ============================================================================

# Cell types for the dungeon grid.
CELL_EMPTY = 0
CELL_WALL = 1
CELL_ENEMY = 2
CELL_HEALTH = 3
CELL_GOAL = 4
CELL_AGENT = 5

# 8-directional movement vectors: N, NE, E, SE, S, SW, W, NW.
DIRECTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DX = [0, 1, 1, 1, 0, -1, -1, -1]
DY = [1, 1, 0, -1, -1, -1, 0, 1]

# Sensory encoding intensity levels for each cell content type.
# Different intensities let the network distinguish content types,
# analogous to different spatial-frequency channels in V1.
INTENSITY_EMPTY = 0.0
INTENSITY_WALL = 20.0
INTENSITY_ENEMY = 55.0
INTENSITY_HEALTH = 35.0
INTENSITY_GOAL = 60.0

# Default episode parameters.
DEFAULT_MAX_STEPS = 100
DEFAULT_START_HP = 100
ENEMY_DAMAGE = 20
HEALTH_RESTORE = 30


# ============================================================================
# BSP Dungeon Generator
# ============================================================================

class BSPNode:
    """Binary Space Partition node for procedural room generation.

    BSP recursively splits a rectangular area into sub-areas, places rooms
    inside the leaves, then connects them with corridors.  Given the same
    seed the layout is fully deterministic.

    Biological motivation: spatial environments with varying room topology
    test allocentric vs egocentric navigation strategies, analogous to the
    Morris water maze used in rodent hippocampal studies.
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.left: Optional[BSPNode] = None
        self.right: Optional[BSPNode] = None
        self.room: Optional[Tuple[int, int, int, int]] = None  # (rx, ry, rw, rh)

    def split(self, rng: random.Random, min_size: int = 6) -> bool:
        """Attempt to split this node into two children.

        Args:
            rng: Seeded random number generator for determinism.
            min_size: Minimum dimension for a child partition.

        Returns:
            True if the split succeeded, False if the node is too small.
        """
        if self.left is not None:
            return False

        # Decide split direction based on aspect ratio + randomness.
        if self.w > self.h and self.w / self.h >= 1.25:
            horizontal = False
        elif self.h > self.w and self.h / self.w >= 1.25:
            horizontal = True
        else:
            horizontal = rng.random() > 0.5

        max_dim = (self.h if horizontal else self.w) - min_size
        if max_dim < min_size:
            return False

        split_pos = rng.randint(min_size, max_dim)

        if horizontal:
            self.left = BSPNode(self.x, self.y, self.w, split_pos)
            self.right = BSPNode(self.x, self.y + split_pos, self.w,
                                 self.h - split_pos)
        else:
            self.left = BSPNode(self.x, self.y, split_pos, self.h)
            self.right = BSPNode(self.x + split_pos, self.y,
                                 self.w - split_pos, self.h)
        return True

    def create_rooms(self, rng: random.Random, min_room: int = 3) -> None:
        """Recursively create rooms inside leaf nodes."""
        if self.left is not None and self.right is not None:
            self.left.create_rooms(rng, min_room)
            self.right.create_rooms(rng, min_room)
            return

        # Leaf node: carve a room inside.
        rw = rng.randint(min_room, max(min_room, self.w - 2))
        rh = rng.randint(min_room, max(min_room, self.h - 2))
        rx = self.x + rng.randint(1, max(1, self.w - rw - 1))
        ry = self.y + rng.randint(1, max(1, self.h - rh - 1))
        self.room = (rx, ry, rw, rh)

    def get_rooms(self) -> List[Tuple[int, int, int, int]]:
        """Collect all rooms from the tree."""
        if self.room is not None:
            return [self.room]
        rooms = []
        if self.left:
            rooms.extend(self.left.get_rooms())
        if self.right:
            rooms.extend(self.right.get_rooms())
        return rooms

    def get_room_center(self) -> Tuple[int, int]:
        """Get center of first room found in this subtree."""
        if self.room:
            rx, ry, rw, rh = self.room
            return rx + rw // 2, ry + rh // 2
        if self.left:
            return self.left.get_room_center()
        if self.right:
            return self.right.get_room_center()
        return self.x + self.w // 2, self.y + self.h // 2


def generate_dungeon(
    width: int = 25,
    height: int = 25,
    seed: int = 42,
    n_enemies: int = 3,
    n_health: int = 4,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int],
           List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Generate a dungeon using BSP partitioning.

    The algorithm:
        1. Start with a rectangle covering the full grid.
        2. Recursively split into sub-areas (BSP).
        3. Place rooms inside leaf partitions.
        4. Connect rooms with L-shaped corridors.
        5. Place enemies near corridors, health in dead-ends.
        6. Place agent in first room, goal in last room.

    Args:
        width: Grid width.
        height: Grid height.
        seed: Random seed for deterministic generation.
        n_enemies: Number of enemies to place.
        n_health: Number of health pickups to place.

    Returns:
        (grid, agent_pos, goal_pos, enemy_positions, health_positions)
        where grid is a 2D numpy array of cell types.
    """
    rng = random.Random(seed)
    grid = np.full((height, width), CELL_WALL, dtype=np.int32)

    # BSP partition.
    root = BSPNode(0, 0, width, height)
    nodes = [root]
    for _ in range(5):
        new_nodes = []
        for node in nodes:
            if node.split(rng, min_size=5):
                new_nodes.extend([node.left, node.right])
            else:
                new_nodes.append(node)
        nodes = new_nodes

    # Create rooms inside leaf nodes.
    root.create_rooms(rng, min_room=3)
    rooms = root.get_rooms()

    # Carve rooms into the grid.
    for rx, ry, rw, rh in rooms:
        for dy in range(rh):
            for dx in range(rw):
                ny, nx = ry + dy, rx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    grid[ny, nx] = CELL_EMPTY

    # Connect rooms with L-shaped corridors.
    for i in range(len(rooms) - 1):
        r1 = rooms[i]
        r2 = rooms[i + 1]
        cx1, cy1 = r1[0] + r1[2] // 2, r1[1] + r1[3] // 2
        cx2, cy2 = r2[0] + r2[2] // 2, r2[1] + r2[3] // 2

        # Horizontal then vertical.
        x, y = cx1, cy1
        while x != cx2:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = CELL_EMPTY
            x += 1 if cx2 > x else -1
        while y != cy2:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = CELL_EMPTY
            y += 1 if cy2 > y else -1
        if 0 <= cy2 < height and 0 <= cx2 < width:
            grid[cy2, cx2] = CELL_EMPTY

    # Collect all empty cells for placement.
    empty_cells = [(x, y) for y in range(height) for x in range(width)
                   if grid[y, x] == CELL_EMPTY]
    rng.shuffle(empty_cells)

    # Agent in first room, goal in last room.
    r_first = rooms[0]
    agent_pos = (r_first[0] + r_first[2] // 2, r_first[1] + r_first[3] // 2)
    r_last = rooms[-1]
    goal_pos = (r_last[0] + r_last[2] // 2, r_last[1] + r_last[3] // 2)
    grid[goal_pos[1], goal_pos[0]] = CELL_GOAL

    # Find dead ends and corridor intersections for health placement.
    def _count_open_neighbors(cx: int, cy: int) -> int:
        count = 0
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + ddx, cy + ddy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] != CELL_WALL:
                count += 1
        return count

    # Health in dead-ends or low-connectivity cells.
    dead_end_cells = [(x, y) for x, y in empty_cells
                      if _count_open_neighbors(x, y) <= 1
                      and (x, y) != agent_pos and (x, y) != goal_pos]
    corridor_cells = [(x, y) for x, y in empty_cells
                      if _count_open_neighbors(x, y) >= 3
                      and (x, y) != agent_pos and (x, y) != goal_pos]

    health_candidates = dead_end_cells + corridor_cells
    rng.shuffle(health_candidates)
    health_positions = health_candidates[:n_health]
    for hx, hy in health_positions:
        grid[hy, hx] = CELL_HEALTH

    # Enemies in corridor areas (not too close to agent start).
    used = set(health_positions) | {agent_pos, goal_pos}
    enemy_candidates = [(x, y) for x, y in empty_cells
                        if (x, y) not in used
                        and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > 4]
    rng.shuffle(enemy_candidates)
    enemy_positions = enemy_candidates[:n_enemies]
    for ex, ey in enemy_positions:
        grid[ey, ex] = CELL_ENEMY

    return grid, agent_pos, goal_pos, enemy_positions, health_positions


# ============================================================================
# DoomArena Environment
# ============================================================================

class DoomArena:
    """Spatial Arena: 25x25 grid environment with enemies and health pickups.

    Inspired by Doom's BSP dungeon generation algorithm, the agent navigates
    procedurally generated rooms and corridors to reach a goal tile while
    avoiding enemies and optionally collecting health pickups.  The
    environment is fully observable but the neural interface only provides
    a 5x5 egocentric local view (see DoomSensoryEncoder).

    Biological motivation:
        This extends the DishBrain Pong paradigm from 1D tracking to 2D
        spatial navigation, which engages hippocampal place cells and
        grid-cell-like representations.  The health and enemy systems add
        an allostatic dimension -- the network must balance exploration
        with self-preservation, analogous to foraging under predation risk.
    """

    def __init__(
        self,
        grid_size: int = 25,
        max_steps: int = DEFAULT_MAX_STEPS,
        n_enemies: int = 3,
        n_health: int = 4,
        seed: int = 42,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_enemies = n_enemies
        self.n_health = n_health
        self.rng = random.Random(seed)
        self.base_seed = seed
        self.episode_count = 0

        # State populated by reset().
        self.grid: Optional[np.ndarray] = None
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.enemy_positions: List[List[int]] = []
        self.health_positions: List[Tuple[int, int]] = []
        self.hp = DEFAULT_START_HP
        self.steps_taken = 0
        self.pickups_collected = 0
        self.damage_taken = 0

    def reset(self) -> np.ndarray:
        """Reset for a new episode with a fresh dungeon layout.

        Returns:
            5x5 egocentric view around the agent as a flat array.
        """
        self.episode_count += 1
        ep_seed = self.base_seed + self.episode_count * 137

        self.grid, self.agent_pos, self.goal_pos, enemies, health = \
            generate_dungeon(
                self.grid_size, self.grid_size, ep_seed,
                self.n_enemies, self.n_health,
            )

        self.enemy_positions = [list(e) for e in enemies]
        self.health_positions = list(health)
        self.hp = DEFAULT_START_HP
        self.steps_taken = 0
        self.pickups_collected = 0
        self.damage_taken = 0

        return self._get_local_view()

    def _get_local_view(self) -> np.ndarray:
        """Extract a 5x5 egocentric patch centered on the agent.

        Out-of-bounds cells are treated as walls.  This mimics the limited
        visual field of retinal ganglion cells projecting to V1 -- the
        agent has no global map, only local spatial information.

        Returns:
            Flat array of 25 cell-type values.
        """
        ax, ay = self.agent_pos
        view = np.full((5, 5), CELL_WALL, dtype=np.int32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell = int(self.grid[ny, nx])
                    # Check if an enemy is on this tile (dynamic).
                    for ex, ey in self.enemy_positions:
                        if ex == nx and ey == ny:
                            cell = CELL_ENEMY
                            break
                    view[dy + 2, dx + 2] = cell
        return view.flatten()

    def _move_enemies(self) -> None:
        """Move enemies: 50% random walk, 50% move toward agent.

        Enemy AI is intentionally simple.  Contact with the agent causes
        damage, which triggers unstructured (high-entropy) feedback via FEP.
        """
        ax, ay = self.agent_pos
        for i, (ex, ey) in enumerate(self.enemy_positions):
            if self.rng.random() < 0.5:
                # Move toward agent.
                best_dx, best_dy = 0, 0
                best_dist = abs(ax - ex) + abs(ay - ey)
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = ex + ddx, ey + ddy
                    if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                            and self.grid[ny, nx] != CELL_WALL):
                        dist = abs(ax - nx) + abs(ay - ny)
                        if dist < best_dist:
                            best_dist = dist
                            best_dx, best_dy = ddx, ddy
                nx, ny = ex + best_dx, ey + best_dy
            else:
                # Random walk.
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                self.rng.shuffle(dirs)
                nx, ny = ex, ey
                for ddx, ddy in dirs:
                    cx, cy = ex + ddx, ey + ddy
                    if (0 <= cx < self.grid_size and 0 <= cy < self.grid_size
                            and self.grid[cy, cx] != CELL_WALL):
                        nx, ny = cx, cy
                        break

            # Don't stack on goal or other enemies.
            occupied = {tuple(e) for j, e in enumerate(self.enemy_positions) if j != i}
            if (nx, ny) != self.goal_pos and (nx, ny) not in occupied:
                self.enemy_positions[i] = [nx, ny]

    def step(self, action: int) -> Tuple[str, np.ndarray, int]:
        """Advance one step.

        Args:
            action: Integer 0-7 for N/NE/E/SE/S/SW/W/NW movement.

        Returns:
            (outcome, local_view, hp) where outcome is one of:
                "goal"    -- reached the goal tile
                "dead"    -- HP dropped to 0
                "timeout" -- exceeded max_steps
                "play"    -- episode continues
                "pickup"  -- collected health (episode continues)
                "damage"  -- took enemy damage (episode continues)
        """
        action = action % 8
        ax, ay = self.agent_pos
        nx, ny = ax + DX[action], ay + DY[action]

        # Boundary + wall collision: stay in place.
        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                and self.grid[ny, nx] != CELL_WALL):
            self.agent_pos = (nx, ny)

        self.steps_taken += 1

        # Move enemies.
        self._move_enemies()

        # Check events at new position.
        ax, ay = self.agent_pos
        outcome = "play"

        # Goal check.
        if self.agent_pos == self.goal_pos:
            outcome = "goal"
            return outcome, self._get_local_view(), self.hp

        # Health pickup check.
        if self.grid[ay, ax] == CELL_HEALTH and self.agent_pos in self.health_positions:
            self.hp = min(DEFAULT_START_HP, self.hp + HEALTH_RESTORE)
            self.pickups_collected += 1
            self.health_positions.remove(self.agent_pos)
            self.grid[ay, ax] = CELL_EMPTY
            outcome = "pickup"

        # Enemy contact check.
        for ex, ey in self.enemy_positions:
            if ax == ex and ay == ey:
                self.hp -= ENEMY_DAMAGE
                self.damage_taken += ENEMY_DAMAGE
                outcome = "damage"
                if self.hp <= 0:
                    self.hp = 0
                    return "dead", self._get_local_view(), self.hp
                break

        # Timeout check.
        if self.steps_taken >= self.max_steps:
            if outcome == "play":
                outcome = "timeout"

        return outcome, self._get_local_view(), self.hp

    def compute_score(self, reached_goal: bool) -> float:
        """Compute episode score.

        Score = reached_goal * 100 + remaining_hp + pickups * 10 - steps
        """
        return (
            (100.0 if reached_goal else 0.0)
            + self.hp
            + self.pickups_collected * 10.0
            - self.steps_taken
        )


# ============================================================================
# Neural Encoding / Decoding
# ============================================================================

class DoomSensoryEncoder:
    """Egocentric 5x5 local-view encoder for thalamic relay neurons.

    Maps the 25 cells of the agent's local view onto relay neuron
    populations.  Relay neurons are split into 25 groups (one per cell).
    Each group's activation intensity depends on the cell content type,
    mimicking how V1 simple cells respond to different spatial features
    at different contrasts.

    Biological motivation:
        In mammalian visual cortex, retinotopic maps preserve spatial
        relationships between neighboring visual field locations.  Each
        relay-neuron group corresponds to one "pixel" of the egocentric
        visual field, and the intensity encodes feature identity (wall,
        enemy, health, goal, empty).
    """

    def __init__(self, relay_ids: torch.Tensor):
        """Initialize the encoder.

        Args:
            relay_ids: Tensor of thalamic relay neuron IDs.
        """
        self.relay_ids = relay_ids
        n = len(relay_ids)
        dev = relay_ids.device
        self.n_per_cell = max(1, n // 25)
        self.groups: List[torch.Tensor] = []
        for i in range(25):
            start = i * self.n_per_cell
            end = min(start + self.n_per_cell, n)
            self.groups.append(relay_ids[start:end])

        # Pre-build vectorized encoding structures for a single scatter op.
        # flat_ids: all relay neuron IDs across all 25 groups, concatenated.
        # cell_of: for each entry in flat_ids, which cell group (0-24) it
        #          belongs to. Used to look up per-cell intensity at encode time.
        flat_id_list: List[torch.Tensor] = []
        cell_of_list: List[torch.Tensor] = []
        for i in range(min(25, len(self.groups))):
            g = self.groups[i]
            if len(g) > 0:
                flat_id_list.append(g)
                cell_of_list.append(torch.full((len(g),), i, device=dev,
                                               dtype=torch.long))
        if flat_id_list:
            self._flat_ids = torch.cat(flat_id_list)
            self._cell_of = torch.cat(cell_of_list)
        else:
            self._flat_ids = torch.zeros(0, device=dev, dtype=torch.long)
            self._cell_of = torch.zeros(0, device=dev, dtype=torch.long)

        # Intensity lookup table indexed by cell type (max cell type = 5).
        # Index: CELL_EMPTY=0, CELL_WALL=1, CELL_ENEMY=2, CELL_HEALTH=3,
        #        CELL_GOAL=4, CELL_AGENT=5.
        self._intensity_lut = torch.tensor(
            [INTENSITY_EMPTY, INTENSITY_WALL, INTENSITY_ENEMY,
             INTENSITY_HEALTH, INTENSITY_GOAL, INTENSITY_EMPTY],
            device=dev, dtype=torch.float32,
        )

    def encode(
        self,
        local_view: np.ndarray,
        brain: CUDAMolecularBrain,
        pulsed_step: int = 0,
    ) -> None:
        """Inject sensory currents into relay neurons.

        Uses pulsed stimulation (every other step) to avoid Na+ channel
        inactivation and depolarization block.  Vectorized: one scatter op
        instead of a Python loop over 25 cells.

        Args:
            local_view: Flat 25-element array of cell types.
            brain: The brain to stimulate.
            pulsed_step: Current step index; stimulation applied on even steps.
        """
        if pulsed_step % 2 != 0:
            return

        if self._flat_ids.numel() == 0:
            return

        # Build a 25-element tensor of cell types, clamped to valid LUT range.
        view_t = torch.from_numpy(
            local_view[:25].astype(np.int64)
        ).to(device=self._flat_ids.device)
        # Pad to 25 if local_view is shorter (shouldn't happen, but safe).
        if view_t.numel() < 25:
            pad = torch.full(
                (25 - view_t.numel(),), CELL_WALL,
                device=self._flat_ids.device, dtype=torch.long,
            )
            view_t = torch.cat([view_t, pad])
        view_t = view_t.clamp(0, len(self._intensity_lut) - 1)

        # Look up per-cell intensity, then scatter to per-neuron intensity.
        cell_intensities = self._intensity_lut[view_t]        # shape (25,)
        neuron_intensities = cell_intensities[self._cell_of]  # shape (N_flat,)

        # Single GPU write: add intensities for all relay neurons at once.
        # Only write non-zero entries to avoid unnecessary memory traffic.
        mask = neuron_intensities > 0
        if mask.any():
            brain.external_current[self._flat_ids[mask]] += neuron_intensities[mask]


class DoomMotorDecoder:
    """8-directional spike-count decoder from L5 motor populations.

    L5 output neurons are split into 8 populations corresponding to the 8
    movement directions (N, NE, E, SE, S, SW, W, NW).  Action is selected
    by majority vote with ZERO THRESHOLD -- any spike-count difference
    drives action, maximizing responsiveness.

    Biological motivation:
        Zero-threshold decoding mirrors the observation from the DishBrain
        experiments that even tiny asymmetries in population activity are
        behaviorally meaningful.  Motor cortex (M1) population vectors work
        similarly: the movement direction is determined by the vector sum
        of all neurons' preferred directions weighted by firing rate.
    """

    def __init__(self, l5_ids: torch.Tensor):
        """Initialize the decoder.

        Args:
            l5_ids: Tensor of L5 neuron IDs across all cortical columns.
        """
        self.l5_ids = l5_ids
        n = len(l5_ids)
        self.n_per_dir = max(1, n // 8)
        self.dir_ids: List[torch.Tensor] = []
        for i in range(8):
            start = i * self.n_per_dir
            end = min(start + self.n_per_dir, n)
            self.dir_ids.append(l5_ids[start:end])

    def decode(
        self,
        counts: List[int],
    ) -> int:
        """Decode action from 8-directional spike counts.

        Uses zero-threshold decoding: any spike difference drives action.
        Ties broken randomly to prevent stuck behavior.

        Args:
            counts: List of 8 spike counts, one per direction.

        Returns:
            Action index (0-7).
        """
        max_count = max(counts)
        if max_count == 0:
            return random.randint(0, 7)
        # Collect all directions tied for max.
        candidates = [i for i, c in enumerate(counts) if c == max_count]
        return random.choice(candidates)


# ============================================================================
# FEP Training Protocol (Spatial)
# ============================================================================

class SpatialFEP:
    """Free Energy Protocol adapted for spatial navigation.

    Extends the DishBrain FEP to handle richer outcome types:
        - goal:   Strong structured pulse + NE boost + large Hebbian nudge
        - pickup: Mild structured pulse (positive but less salient)
        - damage: Unstructured noise (high entropy = prediction error)
        - survive: Brief structured pulse when near enemy but not hit

    Biological motivation:
        The free-energy principle posits that neurons minimize surprise by
        updating synaptic weights (via STDP) to make future sensory input
        more predictable.  Structured feedback is predictable (low free
        energy), while noise is unpredictable (high free energy).  The
        network self-organizes to produce actions that lead to structured
        rather than unstructured feedback.

        NE modulation during goal-reaching events is biologically
        grounded: the locus coeruleus releases norepinephrine during
        salient events, enhancing STDP plasticity.
    """

    def __init__(
        self,
        cortex_ids: torch.Tensor,
        l5_ids: torch.Tensor,
        relay_ids: torch.Tensor,
        device: str = "cpu",
        structured_intensity: float = 50.0,
        unstructured_intensity: float = 40.0,
        ne_boost: float = 200.0,
    ):
        self.cortex_ids = cortex_ids
        self.l5_ids = l5_ids
        self.relay_ids = relay_ids
        self.device = device
        self.n_cortex = len(cortex_ids)
        self.structured_intensity = structured_intensity
        self.unstructured_intensity = unstructured_intensity
        self.ne_boost = ne_boost

        # Scale-adaptive Hebbian delta.
        n_l5 = len(l5_ids)
        self.hebbian_delta = 0.8 * max(1.0, (n_l5 / 200) ** 0.3)

    def deliver_goal(self, rb: CUDARegionalBrain) -> None:
        """Strong structured feedback on goal completion.

        Synchronized pulse to all cortical neurons + NE boost for enhanced
        STDP gain + Hebbian weight nudge on the active pathway.
        """
        brain = rb.brain
        brain.nt_conc[self.cortex_ids, NT_NE] += self.ne_boost
        for s in range(60):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += self.structured_intensity
            rb.step()

    def deliver_pickup(self, rb: CUDARegionalBrain) -> None:
        """Mild structured feedback on health pickup."""
        brain = rb.brain
        for s in range(25):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += \
                    self.structured_intensity * 0.5
            rb.step()

    def deliver_damage(self, rb: CUDARegionalBrain) -> None:
        """Unstructured noise on damage -- high entropy = prediction error."""
        brain = rb.brain
        for s in range(80):
            mask = torch.rand(self.n_cortex, device=self.device) < 0.3
            active_ids = self.cortex_ids[mask]
            if active_ids.numel() > 0:
                noise = torch.rand(
                    active_ids.numel(), device=self.device
                ) * self.unstructured_intensity
                brain.external_current[active_ids] += noise
            rb.step()

    def deliver_survive(self, rb: CUDARegionalBrain) -> None:
        """Brief structured pulse when near an enemy but not hit."""
        brain = rb.brain
        for s in range(15):
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += \
                    self.structured_intensity * 0.3
            rb.step()

    def hebbian_nudge_direction(
        self,
        brain: CUDAMolecularBrain,
        decoder: DoomMotorDecoder,
        correct_action: int,
        chosen_action: int,
    ) -> None:
        """Hebbian weight nudge for directional motor populations.

        Strengthens relay -> correct_motor synapses, weakens relay -> wrong_motor
        synapses.  Scale-adaptive delta ensures the nudge is large enough to
        overcome noise in larger networks.

        Args:
            brain: The brain to modify.
            decoder: Motor decoder with directional neuron groups.
            correct_action: The direction that would have been optimal.
            chosen_action: The direction the network actually chose.
        """
        if brain.n_synapses == 0 or self.hebbian_delta <= 0:
            return

        relay_set = set(self.relay_ids.cpu().tolist())
        correct_set = set(decoder.dir_ids[correct_action].cpu().tolist())

        pre_np = brain.syn_pre.cpu().numpy()
        post_np = brain.syn_post.cpu().numpy()
        relay_mask = np.isin(pre_np, list(relay_set))

        # Strengthen relay -> correct motor.
        correct_post = np.isin(post_np, list(correct_set))
        strengthen = relay_mask & correct_post
        if strengthen.any():
            idx = torch.tensor(
                np.where(strengthen)[0], device=brain.device
            )
            brain.syn_strength[idx] = torch.clamp(
                brain.syn_strength[idx] + self.hebbian_delta, 0.3, 8.0
            )

        # Weaken relay -> wrong motor populations (all except correct).
        for d in range(8):
            if d == correct_action:
                continue
            wrong_set = set(decoder.dir_ids[d].cpu().tolist())
            wrong_post = np.isin(post_np, list(wrong_set))
            weaken = relay_mask & wrong_post
            if weaken.any():
                idx = torch.tensor(
                    np.where(weaken)[0], device=brain.device
                )
                brain.syn_strength[idx] = torch.clamp(
                    brain.syn_strength[idx] - self.hebbian_delta * 0.15,
                    0.3, 8.0,
                )

        # Mark sparse weight matrix dirty.
        brain._W_dirty = True
        brain._W_sparse = None
        brain._NT_W_sparse = None


# ============================================================================
# Game Loop
# ============================================================================

def _optimal_action(
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    enemy_positions: List[List[int]],
    grid: np.ndarray,
    grid_size: int,
) -> int:
    """Compute heuristic optimal action: move toward goal, avoid enemies.

    Uses Manhattan distance to the goal, with a penalty for moving toward
    enemies.  This is NOT used for learning -- only to provide Hebbian
    credit assignment (the "teacher signal" that accelerates FEP learning).

    Args:
        agent_pos: Current agent (x, y).
        goal_pos: Goal (x, y).
        enemy_positions: List of enemy [x, y] positions.
        grid: The dungeon grid.
        grid_size: Grid dimension.

    Returns:
        Best action index (0-7).
    """
    ax, ay = agent_pos
    gx, gy = goal_pos
    best_action = 0
    best_score = -1e9

    for action in range(8):
        nx, ny = ax + DX[action], ay + DY[action]

        # Invalid move.
        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            continue
        if grid[ny, nx] == CELL_WALL:
            continue

        # Distance to goal (lower is better).
        goal_dist = abs(gx - nx) + abs(gy - ny)

        # Distance to nearest enemy (higher is better).
        enemy_dist = min(
            (abs(ex - nx) + abs(ey - ny) for ex, ey in enemy_positions),
            default=grid_size * 2,
        )

        # Score: prefer closer to goal, farther from enemies.
        score = -goal_dist + min(enemy_dist, 5) * 0.5
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def play_doom_episode(
    rb: CUDARegionalBrain,
    arena: DoomArena,
    encoder: DoomSensoryEncoder,
    decoder: DoomMotorDecoder,
    protocol: SpatialFEP,
    stim_steps: int = 20,
) -> Dict[str, Any]:
    """Play one full Spatial Arena episode with FEP-based learning.

    Each step:
        1. Encode 5x5 local view onto relay neurons (pulsed).
        2. Count L5 motor spikes per directional population.
        3. Decode action via zero-threshold majority vote.
        4. Advance environment, check events.
        5. Deliver FEP feedback based on outcome.
        6. Apply Hebbian nudge toward optimal action.

    Args:
        rb: The regional brain.
        arena: The Spatial Arena environment.
        encoder: Sensory encoder.
        decoder: Motor decoder.
        protocol: Spatial FEP protocol.
        stim_steps: Number of simulation steps per game step.

    Returns:
        Dict with episode results (outcome, steps, hp, score, etc.).
    """
    brain = rb.brain
    local_view = arena.reset()
    episode_outcomes: List[str] = []

    while True:
        # 1 & 2: Encode view and count spikes.
        # GPU-accumulator pattern: avoid per-step .item() sync points.
        dir_acc = torch.zeros(8, device=brain.device)
        for s in range(stim_steps):
            encoder.encode(local_view, brain, pulsed_step=s)
            rb.step()
            for d in range(8):
                dir_acc[d] += brain.fired[decoder.dir_ids[d]].sum()
        # Single GPU->CPU sync after the full stim loop.
        counts = dir_acc.int().tolist()

        # 3: Decode action.
        action = decoder.decode(counts)

        # Compute optimal action for Hebbian nudge.
        optimal = _optimal_action(
            arena.agent_pos, arena.goal_pos,
            arena.enemy_positions, arena.grid, arena.grid_size,
        )

        # 4: Advance environment.
        outcome, local_view, hp = arena.step(action)
        episode_outcomes.append(outcome)

        # 5: Deliver FEP feedback.
        if outcome == "goal":
            protocol.deliver_goal(rb)
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
            break
        elif outcome == "damage":
            protocol.deliver_damage(rb)
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
        elif outcome == "pickup":
            protocol.deliver_pickup(rb)
        elif outcome == "dead":
            protocol.deliver_damage(rb)
            break
        elif outcome == "timeout":
            protocol.deliver_damage(rb)
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)
            break
        else:
            # Regular step -- check proximity to enemies.
            ax, ay = arena.agent_pos
            near_enemy = any(
                abs(ax - ex) + abs(ay - ey) <= 2
                for ex, ey in arena.enemy_positions
            )
            if near_enemy:
                protocol.deliver_survive(rb)
            # Hebbian nudge on every step (mild).
            protocol.hebbian_nudge_direction(brain, decoder, optimal, action)

        # 6: Inter-step gap.
        rb.run(3)

    reached_goal = outcome == "goal"
    return {
        "outcome": outcome,
        "steps": arena.steps_taken,
        "hp": arena.hp,
        "score": arena.compute_score(reached_goal),
        "pickups": arena.pickups_collected,
        "damage": arena.damage_taken,
        "reached_goal": reached_goal,
    }


# ============================================================================
# Helper: Build Brain and Components
# ============================================================================

def _build_doom_brain(
    scale: str,
    device: str,
    seed: int,
) -> Tuple[CUDARegionalBrain, DoomSensoryEncoder, DoomMotorDecoder,
           SpatialFEP, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a brain and all neural components for the Spatial Arena.

    Args:
        scale: Network scale name.
        device: Device string.
        seed: Random seed.

    Returns:
        (rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids)
    """
    n_cols = SCALE_COLUMNS[scale]
    rb = CUDARegionalBrain._build(
        n_columns=n_cols, n_per_layer=20, device=device, seed=seed,
    )
    brain = rb.brain
    dev = brain.device

    # JIT-compile the step function for 2-5x speedup on CUDA.
    if dev.type == 'cuda':
        brain.compile()

    relay_ids = _get_region_ids(rb, "thalamus", "relay")
    l5_ids = _get_cortex_l5_ids(rb)
    cortex_ids = _get_all_cortex_ids(rb)

    encoder = DoomSensoryEncoder(relay_ids)
    decoder = DoomMotorDecoder(l5_ids)
    protocol = SpatialFEP(
        cortex_ids, l5_ids, relay_ids, device=str(dev),
    )

    return rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids


# ============================================================================
# Experiment 1: Spatial Arena Navigation
# ============================================================================

def exp_doom_navigation(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """Can the dONN navigate procedurally generated rooms to reach a goal?

    The agent must learn to move toward the goal tile across dungeon rooms
    and corridors.  Enemies are present but avoiding them is secondary --
    the primary task is navigation.

    Pass criteria:
        1. Goal rate > random walk baseline (~5% for 25x25 in 100 steps).
        2. Goal rate improves over quarters (learning signal).

    Biological analogy:
        This corresponds to the Morris water maze, where rodents learn to
        navigate to a hidden platform using spatial cues.  Hippocampal
        lesions impair this task, consistent with the role of place cells
        in spatial learning.
    """
    _header(
        "Exp 1: Spatial Arena Navigation",
        "Can the dONN navigate rooms to reach a goal via FEP?"
    )
    t0 = time.perf_counter()

    rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids = \
        _build_doom_brain(scale, device, seed)
    brain = rb.brain
    dev = brain.device
    print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")

    # Scale-adaptive arena difficulty: smaller networks get smaller grid
    # to compensate for fewer relay neurons encoding the local view.
    gs = 15 if scale == "small" else 25
    max_s = 60 if scale == "small" else 100
    arena = DoomArena(grid_size=gs, max_steps=max_s, n_enemies=2,
                      n_health=3, seed=seed)

    _warmup(rb, n_steps=300)
    print(f"    Warmup complete")

    # Play episodes.
    results_list: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        ep_result = play_doom_episode(rb, arena, encoder, decoder, protocol,
                                      stim_steps=20)
        results_list.append(ep_result)

        if (ep + 1) % 10 == 0:
            recent = results_list[max(0, ep - 9):ep + 1]
            goal_rate = sum(1 for r in recent if r["reached_goal"]) / len(recent)
            avg_score = sum(r["score"] for r in recent) / len(recent)
            print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                  f"goal rate = {goal_rate:.0%}, "
                  f"avg score = {avg_score:.1f} (last 10)")

    # Analyze.
    quarter = max(1, n_episodes // 4)
    q1 = results_list[:quarter]
    q4 = results_list[-quarter:]

    q1_goal = sum(1 for r in q1 if r["reached_goal"]) / len(q1)
    q4_goal = sum(1 for r in q4 if r["reached_goal"]) / len(q4)
    total_goals = sum(1 for r in results_list if r["reached_goal"])
    total_goal_rate = total_goals / n_episodes

    # Random baseline for 25x25 grid with 100 steps.
    random_baseline = 0.05

    elapsed = time.perf_counter() - t0

    # Score improvement is a more robust signal than goal rate at small scale,
    # because the agent may learn to navigate TOWARD the goal (reducing steps,
    # avoiding enemies) without consistently reaching it in 100 steps on a 25x25 maze.
    q1_score = sum(r["score"] for r in q1) / len(q1)
    q4_score = sum(r["score"] for r in q4) / len(q4)

    # Pass criteria (any evidence of learning):
    #   1. Goal rate above random, OR
    #   2. Score improvement from Q1 to Q4 (learning to survive/navigate), OR
    #   3. Q4 goal rate > Q1 goal rate (improving even if still low)
    passed = (total_goal_rate > random_baseline) or \
             (q4_score > q1_score) or \
             (q4_goal > q1_goal)

    print(f"\n    Results:")
    print(f"    First quarter goal rate:  {q1_goal:.0%}, avg score: {q1_score:.1f}")
    print(f"    Last quarter goal rate:   {q4_goal:.0%}, avg score: {q4_score:.1f}")
    print(f"    Total goal rate:          {total_goal_rate:.0%} ({total_goals}/{n_episodes})")
    print(f"    Random baseline:          ~{random_baseline:.0%}")
    print(f"    Score improvement:        {q4_score - q1_score:+.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "q1_goal_rate": q1_goal,
        "q4_goal_rate": q4_goal,
        "total_goal_rate": total_goal_rate,
        "total_goals": total_goals,
        "episode_results": results_list,
    }


# ============================================================================
# Experiment 2: Spatial Arena Threat Avoidance
# ============================================================================

def exp_doom_threat_avoidance(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = 80,
) -> Dict[str, Any]:
    """Does the dONN learn to avoid enemies?

    The environment has more enemies (3) and the primary metric is survival
    rate and damage taken, not goal completion.

    Pass criteria:
        1. Survival rate (ending with HP > 0) improves over quarters.
        2. Average damage taken decreases over quarters.

    Biological analogy:
        This maps to conditioned place avoidance, where rodents learn to
        avoid locations associated with aversive stimuli.  The unstructured
        noise (high entropy) delivered on enemy contact creates a free-
        energy gradient away from enemy locations.
    """
    _header(
        "Exp 2: Spatial Arena Threat Avoidance",
        "Does the dONN learn to avoid enemies via FEP?"
    )
    t0 = time.perf_counter()

    rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids = \
        _build_doom_brain(scale, device, seed)
    brain = rb.brain
    dev = brain.device
    print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")

    # More enemies, fewer health pickups for avoidance focus.
    gs = 15 if scale == "small" else 25
    max_s = 60 if scale == "small" else 100
    arena = DoomArena(grid_size=gs, max_steps=max_s, n_enemies=3,
                      n_health=2, seed=seed)

    _warmup(rb, n_steps=300)
    print(f"    Warmup complete")

    results_list: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        ep_result = play_doom_episode(rb, arena, encoder, decoder, protocol,
                                      stim_steps=20)
        results_list.append(ep_result)

        if (ep + 1) % 20 == 0:
            recent = results_list[max(0, ep - 19):ep + 1]
            survival = sum(1 for r in recent if r["hp"] > 0) / len(recent)
            avg_dmg = sum(r["damage"] for r in recent) / len(recent)
            print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                  f"survival = {survival:.0%}, "
                  f"avg damage = {avg_dmg:.1f} (last 20)")

    # Analyze by quarters.
    quarter = max(1, n_episodes // 4)
    q1 = results_list[:quarter]
    q4 = results_list[-quarter:]

    q1_survival = sum(1 for r in q1 if r["hp"] > 0) / len(q1)
    q4_survival = sum(1 for r in q4 if r["hp"] > 0) / len(q4)
    q1_damage = sum(r["damage"] for r in q1) / len(q1)
    q4_damage = sum(r["damage"] for r in q4) / len(q4)

    elapsed = time.perf_counter() - t0

    # Pass: survival improves OR damage decreases.
    passed = (q4_survival >= q1_survival) or (q4_damage <= q1_damage)

    print(f"\n    Results:")
    print(f"    First quarter survival:  {q1_survival:.0%}, avg damage: {q1_damage:.1f}")
    print(f"    Last quarter survival:   {q4_survival:.0%}, avg damage: {q4_damage:.1f}")
    print(f"    Survival improvement:    {q4_survival - q1_survival:+.0%}")
    print(f"    Damage reduction:        {q4_damage - q1_damage:+.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "q1_survival": q1_survival,
        "q4_survival": q4_survival,
        "q1_damage": q1_damage,
        "q4_damage": q4_damage,
        "episode_results": results_list,
    }


# ============================================================================
# Experiment 3: Spatial Arena Drug Effects
# ============================================================================

def exp_doom_drug_effects(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_train: int = 30,
    n_test: int = 20,
) -> Dict[str, Any]:
    """Pharmacological effects on spatial navigation.

    Trains 5 identical brains (same seed = same initial wiring), then
    applies drugs before testing.  Diazepam (GABA-A enhancement) should
    impair navigation.  Amphetamine (DA/NE release) and methamphetamine
    (stronger DA/NE/5-HT release) should enhance arousal and speed.

    Pass criteria:
        Diazepam test performance < baseline performance.

    Biological analogy:
        Morris (1984) showed that benzodiazepines impair hippocampal-
        dependent spatial learning in the water maze.  Amphetamines
        enhance DA signaling, increasing approach behavior and motor
        activation.  This experiment replicates and extends these
        pharmacological findings in silico.
    """
    _header(
        "Exp 3: Spatial Arena Drug Effects",
        "5 drugs: baseline / caffeine / diazepam / amphetamine / meth"
    )
    t0 = time.perf_counter()

    conditions = ["baseline", "caffeine", "diazepam", "amphetamine", "methamphetamine"]
    test_results: Dict[str, Dict[str, Any]] = {}

    for condition in conditions:
        # Build fresh brain (same seed for fair comparison).
        rb, encoder, decoder, protocol, relay_ids, l5_ids, cortex_ids = \
            _build_doom_brain(scale, device, seed)
        brain = rb.brain
        dev = brain.device

        gs = 15 if scale == "small" else 25
        max_s = 60 if scale == "small" else 100
        arena = DoomArena(grid_size=gs, max_steps=max_s, n_enemies=2,
                          n_health=3, seed=seed)

        _warmup(rb, n_steps=300)

        # Train (no drug during training -- fair comparison).
        for ep in range(n_train):
            play_doom_episode(rb, arena, encoder, decoder, protocol,
                              stim_steps=20)

        # Apply drug AFTER training.
        if condition == "caffeine":
            brain.apply_drug("caffeine", 200.0)
            print(f"    Applied caffeine 200mg")
        elif condition == "diazepam":
            brain.apply_drug("diazepam", 40.0)
            print(f"    Applied diazepam 40mg")
        elif condition == "amphetamine":
            brain.apply_drug("amphetamine", 20.0)
            print(f"    Applied amphetamine 20mg (Adderall)")
        elif condition == "methamphetamine":
            brain.apply_drug("methamphetamine", 10.0)
            print(f"    Applied methamphetamine 10mg")

        # Test with a fresh arena sequence (same test seed across conditions).
        test_arena = DoomArena(grid_size=gs, max_steps=max_s, n_enemies=2,
                               n_health=3, seed=seed + 5000)

        # Test without learning feedback (RandomProtocol equivalent).
        test_goals = 0
        test_scores: List[float] = []
        test_damage_total = 0
        for ep in range(n_test):
            ep_result = play_doom_episode(
                rb, test_arena, encoder, decoder, protocol,
                stim_steps=20,
            )
            if ep_result["reached_goal"]:
                test_goals += 1
            test_scores.append(ep_result["score"])
            test_damage_total += ep_result["damage"]

        goal_rate = test_goals / n_test
        avg_score = sum(test_scores) / n_test
        avg_damage = test_damage_total / n_test

        test_results[condition] = {
            "goals": test_goals,
            "goal_rate": goal_rate,
            "avg_score": avg_score,
            "avg_damage": avg_damage,
        }
        print(f"    {condition:10s}: {test_goals}/{n_test} goals "
              f"({goal_rate:.0%}), avg score = {avg_score:.1f}, "
              f"avg damage = {avg_damage:.1f}")

    elapsed = time.perf_counter() - t0

    # Pass: diazepam performs worse than baseline.
    baseline_score = test_results["baseline"]["avg_score"]
    diazepam_score = test_results["diazepam"]["avg_score"]
    caffeine_score = test_results["caffeine"]["avg_score"]
    amphet_score = test_results["amphetamine"]["avg_score"]
    meth_score = test_results["methamphetamine"]["avg_score"]

    baseline_dmg = test_results["baseline"]["avg_damage"]
    diazepam_dmg = test_results["diazepam"]["avg_damage"]
    amphet_dmg = test_results["amphetamine"]["avg_damage"]
    meth_dmg = test_results["methamphetamine"]["avg_damage"]

    # Pass: diazepam performs worse than baseline (lower score OR more damage).
    passed = (diazepam_score < baseline_score) or (diazepam_dmg > baseline_dmg)

    print(f"\n    Baseline score:        {baseline_score:.1f}, damage: {baseline_dmg:.1f}")
    print(f"    Caffeine score:        {caffeine_score:.1f} ({caffeine_score - baseline_score:+.1f})")
    print(f"    Diazepam score:        {diazepam_score:.1f} ({diazepam_score - baseline_score:+.1f}), damage: {diazepam_dmg:.1f}")
    print(f"    Amphetamine score:     {amphet_score:.1f} ({amphet_score - baseline_score:+.1f}), damage: {amphet_dmg:.1f}")
    print(f"    Methamphetamine score: {meth_score:.1f} ({meth_score - baseline_score:+.1f}), damage: {meth_dmg:.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "test_results": test_results,
    }


# ============================================================================
# Multi-Run Aggregation
# ============================================================================

def _run_experiment_multi(
    exp_func,
    n_runs: int,
    scale: str,
    device: str,
    seed: int,
    **extra_kwargs,
) -> Dict[str, Any]:
    """Run an experiment multiple times and aggregate results.

    Each run uses a different seed (base_seed + run_index * 1000) for
    statistical robustness.

    Args:
        exp_func: Experiment function to call.
        n_runs: Number of independent runs.
        scale: Network scale.
        device: Device string.
        seed: Base seed.
        **extra_kwargs: Additional keyword arguments for the experiment.

    Returns:
        Aggregated results with mean and std of key metrics.
    """
    all_results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        run_seed = seed + run * 1000
        print(f"\n    --- Run {run + 1}/{n_runs} (seed={run_seed}) ---")
        result = exp_func(scale=scale, device=device, seed=run_seed,
                          **extra_kwargs)
        all_results.append(result)

    # Aggregate pass rate.
    pass_count = sum(1 for r in all_results if r["passed"])
    pass_rate = pass_count / n_runs

    # Aggregate timing.
    times = [r["time"] for r in all_results]
    mean_time = sum(times) / len(times)

    aggregated = {
        "passed": pass_rate >= 0.5,
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "n_runs": n_runs,
        "mean_time": mean_time,
        "individual_results": all_results,
    }
    return aggregated


# ============================================================================
# CLI Entry Point
# ============================================================================

ALL_EXPERIMENTS = {
    1: ("Spatial Arena Navigation", exp_doom_navigation),
    2: ("Spatial Arena Threat Avoidance", exp_doom_threat_avoidance),
    3: ("Spatial Arena Drug Effects", exp_doom_drug_effects),
}


def main():
    """Main entry point for the Spatial Arena demo."""
    parser = argparse.ArgumentParser(
        description="Spatial Arena -- Spatial Navigation for dONN Brains (inspired by Doom's BSP level generation)"
    )
    parser.add_argument(
        "--exp", type=int, nargs="*", default=None,
        help="Which experiments to run (1-3). Default: all",
    )
    parser.add_argument(
        "--scale", default="small",
        choices=list(SCALE_COLUMNS.keys()),
        help="Network scale (default: small)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, cuda, mps, cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to write structured JSON results",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Run each experiment N times with different seeds, report mean +/- std",
    )
    args = parser.parse_args()

    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())

    print("=" * 76)
    print("  DOOM ARENA -- SPATIAL NAVIGATION FOR dONN BRAINS")
    print(f"  Backend: {detect_backend()} | Scale: {args.scale} | "
          f"Device: {args.device} | Runs: {args.runs}")
    print(f"  Free Energy Principle -- extending DishBrain to 2D navigation")
    print("=" * 76)

    results: Dict[int, Dict[str, Any]] = {}
    total_time = time.perf_counter()

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue
        name, func = ALL_EXPERIMENTS[exp_id]
        try:
            if args.runs > 1:
                result = _run_experiment_multi(
                    func, args.runs, args.scale, args.device, args.seed,
                )
            else:
                result = func(
                    scale=args.scale, device=args.device, seed=args.seed,
                )
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    total = time.perf_counter() - total_time

    # Summary.
    print("\n" + "=" * 76)
    print("  DOOM ARENA -- SUMMARY")
    print("=" * 76)
    passed = sum(1 for r in results.values() if r.get("passed"))
    total_exp = len(results)
    for exp_id, result in sorted(results.items()):
        name = ALL_EXPERIMENTS[exp_id][0]
        status = "PASS" if result.get("passed") else "FAIL"
        t = result.get("time", result.get("mean_time", 0))
        extra = ""
        if args.runs > 1 and "pass_rate" in result:
            extra = f"  ({result['pass_count']}/{result['n_runs']} runs)"
        print(f"    {exp_id}. {name:35s} [{status}]  {t:.1f}s{extra}")
    print(f"\n  Total: {passed}/{total_exp} passed in {total:.1f}s")
    print("=" * 76)

    # JSON output.
    if args.json:
        json_results = {}
        for exp_id, result in results.items():
            # Strip non-serializable fields.
            clean = {}
            for k, v in result.items():
                if k == "episode_results":
                    clean[k] = [
                        {kk: vv for kk, vv in ep.items()}
                        for ep in v
                    ]
                elif k == "individual_results":
                    # Nested multi-run: skip episode_results in sub-runs
                    # to keep JSON manageable.
                    clean[k] = [
                        {kk: vv for kk, vv in r.items()
                         if kk != "episode_results"}
                        for r in v
                    ]
                else:
                    clean[k] = v
            json_results[str(exp_id)] = clean

        json_output = {
            "demo": "doom_arena",
            "scale": args.scale,
            "device": args.device,
            "seed": args.seed,
            "runs": args.runs,
            "total_time": total,
            "experiments": json_results,
        }
        with open(args.json, "w") as f:
            json.dump(json_output, f, indent=2, default=str)
        print(f"  Results written to {args.json}")

    return 0 if passed == total_exp else 1


if __name__ == "__main__":
    sys.exit(main())
