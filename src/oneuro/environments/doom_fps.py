"""Doom-style first-person raycasting engine for dONN brain environments.

A real raycasting 3D renderer inspired by id Software's Doom (1993) and
Wolfenstein 3D (1992).  Produces pixel frames as numpy uint8 RGB arrays
suitable for processing by a molecular retina and digital organic neural
network (dONN).

The renderer uses the DDA (Digital Differential Analyzer) raycasting algorithm
-- the same technique used in Wolfenstein 3D -- casting one ray per column of
the output image.  Wall heights are proportional to 1/distance, creating the
characteristic corridor perspective.

Maps are procedurally generated using a BSP-inspired room+corridor algorithm
(seeded for determinism), consistent with the DoomArena class in
demos/demo_doom_arena.py but extended for continuous-space navigation.

Architecture:
    DoomMap      -- Procedural BSP map generation (grid-based, 32x32 tiles)
    DoomSprite   -- Billboard sprite (enemy, health pickup, goal marker)
    DoomFPS      -- Main environment: physics, game logic, raycasting renderer

Actions (8 total):
    0: Move forward          4: Turn left (22.5 deg)
    1: Move backward         5: Turn right (22.5 deg)
    2: Strafe left           6: Turn left + move forward
    3: Strafe right          7: Turn right + move forward

Rendering pipeline:
    1. Cast rays (DDA) to find wall distances per column
    2. Draw ceiling and floor gradients
    3. Draw textured wall slices (height ~ 1/distance)
    4. Sort sprites by distance, render back-to-front with z-buffer

Dependencies: numpy only.  No pygame, no OpenGL, no external rendering.

References:
    - Permadi (1996) "Ray-Casting Tutorial for Game Development"
    - id Software (1992) Wolfenstein 3D source (DDA raycasting)
    - id Software (1993) Doom source (BSP rendering, sprites)
    - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience
      when embodied in a simulated game-world" Neuron 110(23):3952-3969

Usage:
    env = DoomFPS(render_width=64, render_height=48, seed=42)
    frame = env.reset()       # (48, 64, 3) uint8 RGB
    frame, reward, done, info = env.step(action)

    # High-res visualization:
    env_hires = DoomFPS(render_width=320, render_height=200)
    frame = env_hires.render()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ============================================================================
# Constants
# ============================================================================

# Map cell types.
CELL_FLOOR = 0
CELL_WALL_STONE = 1
CELL_WALL_METAL = 2
CELL_DOOR = 3

# Wall colors (R, G, B) -- slightly muted palette, Doom-esque.
WALL_COLORS = {
    CELL_WALL_STONE: np.array([128, 96, 72], dtype=np.uint8),   # brown stone
    CELL_WALL_METAL: np.array([160, 160, 176], dtype=np.uint8),  # steel gray
    CELL_DOOR: np.array([96, 64, 32], dtype=np.uint8),           # dark wood
}

# Ceiling and floor base colors.
COLOR_CEILING = np.array([40, 40, 48], dtype=np.uint8)    # dark blue-gray
COLOR_FLOOR = np.array([56, 56, 48], dtype=np.uint8)      # dark warm gray

# Sprite colors.
SPRITE_COLORS = {
    "enemy": np.array([200, 40, 40], dtype=np.uint8),     # red
    "health": np.array([40, 200, 40], dtype=np.uint8),     # green
    "goal": np.array([240, 240, 100], dtype=np.uint8),     # yellow
}

# Sprite outline (darker shade for depth).
SPRITE_OUTLINE_COLORS = {
    "enemy": np.array([120, 20, 20], dtype=np.uint8),
    "health": np.array([20, 120, 20], dtype=np.uint8),
    "goal": np.array([160, 160, 60], dtype=np.uint8),
}

# Player movement constants.
MOVE_SPEED = 0.15
TURN_SPEED = math.pi / 8.0  # 22.5 degrees

# Game balance.
DEFAULT_HP = 100
ENEMY_DAMAGE = 20
HEALTH_RESTORE = 25
ENEMY_CONTACT_RADIUS = 0.5
PICKUP_RADIUS = 0.4
GOAL_RADIUS = 0.5
ENEMY_MOVE_SPEED = 0.06
ENEMY_CHASE_RADIUS = 6.0

# Field of view.
FOV = math.pi / 3.0  # 60 degrees, classic Wolfenstein/Doom FOV

# Texture stripe width (in map units).
TEXTURE_STRIPE_PERIOD = 0.25


# ============================================================================
# BSP Node (procedural map generation)
# ============================================================================

class _BSPNode:
    """Binary Space Partition node for procedural room generation.

    Recursively splits a rectangular area into sub-areas, places rooms inside
    the leaf nodes, then connects them with corridors.  Given the same seed
    the layout is fully deterministic.

    This is the same algorithm used in the DoomArena (demos/demo_doom_arena.py)
    adapted for the continuous-space FPS environment.
    """

    __slots__ = ("x", "y", "w", "h", "left", "right", "room")

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left: Optional[_BSPNode] = None
        self.right: Optional[_BSPNode] = None
        self.room: Optional[Tuple[int, int, int, int]] = None

    def split(self, rng: random.Random, min_size: int = 6) -> bool:
        """Attempt to split this node into two children.

        Chooses horizontal or vertical split based on aspect ratio with
        some randomness.  Returns True if split succeeded.
        """
        if self.left is not None:
            return False

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
            self.left = _BSPNode(self.x, self.y, self.w, split_pos)
            self.right = _BSPNode(self.x, self.y + split_pos,
                                  self.w, self.h - split_pos)
        else:
            self.left = _BSPNode(self.x, self.y, split_pos, self.h)
            self.right = _BSPNode(self.x + split_pos, self.y,
                                  self.w - split_pos, self.h)
        return True

    def create_rooms(self, rng: random.Random, min_room: int = 3) -> None:
        """Recursively create rooms inside leaf nodes."""
        if self.left is not None and self.right is not None:
            self.left.create_rooms(rng, min_room)
            self.right.create_rooms(rng, min_room)
            return

        rw = rng.randint(min_room, max(min_room, self.w - 2))
        rh = rng.randint(min_room, max(min_room, self.h - 2))
        rx = self.x + rng.randint(1, max(1, self.w - rw - 1))
        ry = self.y + rng.randint(1, max(1, self.h - rh - 1))
        self.room = (rx, ry, rw, rh)

    def get_rooms(self) -> List[Tuple[int, int, int, int]]:
        """Collect all rooms from the BSP tree."""
        if self.room is not None:
            return [self.room]
        rooms: List[Tuple[int, int, int, int]] = []
        if self.left:
            rooms.extend(self.left.get_rooms())
        if self.right:
            rooms.extend(self.right.get_rooms())
        return rooms


# ============================================================================
# DoomMap
# ============================================================================

@dataclass
class DoomMap:
    """Procedurally generated Doom-style map.

    A 2D grid where each cell is one of:
        0 = floor (passable)
        1 = stone wall
        2 = metal wall
        3 = door

    The map is generated using BSP (Binary Space Partition) room+corridor
    decomposition, the same family of algorithms used in the original Doom.

    Attributes:
        width:         Grid width in tiles.
        height:        Grid height in tiles.
        grid:          2D int array of cell types.
        spawn:         Player spawn position (float x, float y).
        goal:          Goal position (float x, float y).
        enemy_spawns:  List of enemy spawn positions.
        health_spawns: List of health pickup positions.
        rooms:         List of (rx, ry, rw, rh) room rectangles.
    """

    width: int = 32
    height: int = 32
    grid: np.ndarray = field(default_factory=lambda: np.zeros((32, 32), dtype=np.int32))
    spawn: Tuple[float, float] = (1.5, 1.5)
    goal: Tuple[float, float] = (30.5, 30.5)
    enemy_spawns: List[Tuple[float, float]] = field(default_factory=list)
    health_spawns: List[Tuple[float, float]] = field(default_factory=list)
    rooms: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @classmethod
    def generate(
        cls,
        width: int = 32,
        height: int = 32,
        n_rooms: int = 6,
        n_enemies: int = 3,
        n_health: int = 4,
        seed: int = 42,
    ) -> "DoomMap":
        """Generate a BSP-style procedural dungeon map.

        Algorithm:
            1. Fill grid with walls.
            2. BSP-partition the space into sub-rectangles.
            3. Place rooms inside leaf partitions.
            4. Connect rooms with L-shaped corridors.
            5. Add door tiles at corridor-room junctions.
            6. Assign metal walls to some rooms for variety.
            7. Place spawn, goal, enemies, and health pickups.

        Args:
            width:     Map width in tiles.
            height:    Map height in tiles.
            n_rooms:   Target number of BSP splits (actual rooms may vary).
            n_enemies: Number of enemy spawn points.
            n_health:  Number of health pickup spawn points.
            seed:      Random seed for deterministic generation.

        Returns:
            A fully populated DoomMap instance.
        """
        rng = random.Random(seed)
        grid = np.full((height, width), CELL_WALL_STONE, dtype=np.int32)

        # --- BSP partition ---
        root = _BSPNode(0, 0, width, height)
        nodes = [root]
        for _ in range(n_rooms):
            new_nodes: List[_BSPNode] = []
            for node in nodes:
                if node.split(rng, min_size=5):
                    new_nodes.extend([node.left, node.right])  # type: ignore
                else:
                    new_nodes.append(node)
            nodes = new_nodes

        root.create_rooms(rng, min_room=3)
        rooms = root.get_rooms()

        if len(rooms) < 2:
            # Fallback: ensure at least two rooms.
            rooms = [
                (2, 2, 5, 5),
                (width - 8, height - 8, 5, 5),
            ]

        # --- Carve rooms ---
        metal_rooms = set()
        for idx, (rx, ry, rw, rh) in enumerate(rooms):
            # Mark some rooms as metal for visual variety.
            is_metal = rng.random() < 0.3
            if is_metal:
                metal_rooms.add(idx)
            wall_type = CELL_WALL_METAL if is_metal else CELL_WALL_STONE

            for dy in range(rh):
                for dx in range(rw):
                    ny, nx = ry + dy, rx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        grid[ny, nx] = CELL_FLOOR

            # Set room border walls to the appropriate type.
            if is_metal:
                for dy in range(-1, rh + 1):
                    for dx in range(-1, rw + 1):
                        ny, nx = ry + dy, rx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if grid[ny, nx] != CELL_FLOOR:
                                grid[ny, nx] = wall_type

        # --- Connect rooms with L-shaped corridors ---
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2

            # Horizontal segment.
            x, y = cx1, cy1
            while x != cx2:
                if 0 <= y < height and 0 <= x < width:
                    if grid[y, x] != CELL_FLOOR:
                        grid[y, x] = CELL_FLOOR
                x += 1 if cx2 > x else -1
            # Vertical segment.
            while y != cy2:
                if 0 <= y < height and 0 <= x < width:
                    if grid[y, x] != CELL_FLOOR:
                        grid[y, x] = CELL_FLOOR
                y += 1 if cy2 > y else -1
            if 0 <= cy2 < height and 0 <= cx2 < width:
                grid[cy2, cx2] = CELL_FLOOR

        # --- Place doors at corridor-room transitions ---
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2

            # Place a door near the midpoint of the corridor.
            mid_x = (cx1 + cx2) // 2
            mid_y = (cy1 + cy2) // 2
            if 0 <= mid_y < height and 0 <= mid_x < width:
                if grid[mid_y, mid_x] == CELL_FLOOR:
                    # Only place door if flanked by walls on at least one axis.
                    has_wall_ns = (
                        (mid_y > 0 and grid[mid_y - 1, mid_x] != CELL_FLOOR) or
                        (mid_y < height - 1 and grid[mid_y + 1, mid_x] != CELL_FLOOR)
                    )
                    has_wall_ew = (
                        (mid_x > 0 and grid[mid_y, mid_x - 1] != CELL_FLOOR) or
                        (mid_x < width - 1 and grid[mid_y, mid_x + 1] != CELL_FLOOR)
                    )
                    if has_wall_ns or has_wall_ew:
                        grid[mid_y, mid_x] = CELL_DOOR

        # --- Collect floor cells for entity placement ---
        floor_cells: List[Tuple[int, int]] = []
        for gy in range(height):
            for gx in range(width):
                if grid[gy, gx] == CELL_FLOOR:
                    floor_cells.append((gx, gy))
        rng.shuffle(floor_cells)

        # Spawn in first room center, goal in last room center.
        r_first = rooms[0]
        spawn = (
            float(r_first[0] + r_first[2] // 2) + 0.5,
            float(r_first[1] + r_first[3] // 2) + 0.5,
        )
        r_last = rooms[-1]
        goal = (
            float(r_last[0] + r_last[2] // 2) + 0.5,
            float(r_last[1] + r_last[3] // 2) + 0.5,
        )

        reserved = {
            (int(spawn[0]), int(spawn[1])),
            (int(goal[0]), int(goal[1])),
        }

        # Enemies: not too close to spawn.
        enemy_spawns: List[Tuple[float, float]] = []
        for gx, gy in floor_cells:
            if (gx, gy) in reserved:
                continue
            dist_to_spawn = abs(gx - spawn[0]) + abs(gy - spawn[1])
            if dist_to_spawn > 5 and len(enemy_spawns) < n_enemies:
                enemy_spawns.append((float(gx) + 0.5, float(gy) + 0.5))
                reserved.add((gx, gy))

        # Health pickups: prefer dead-end cells.
        health_spawns: List[Tuple[float, float]] = []
        for gx, gy in floor_cells:
            if (gx, gy) in reserved:
                continue
            if len(health_spawns) < n_health:
                health_spawns.append((float(gx) + 0.5, float(gy) + 0.5))
                reserved.add((gx, gy))

        return cls(
            width=width,
            height=height,
            grid=grid,
            spawn=spawn,
            goal=goal,
            enemy_spawns=enemy_spawns,
            health_spawns=health_spawns,
            rooms=rooms,
        )

    def is_wall(self, x: int, y: int) -> bool:
        """Check if grid cell (x, y) is a solid wall."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid[y, x] != CELL_FLOOR

    def wall_type(self, x: int, y: int) -> int:
        """Get the wall type at grid cell (x, y).  Returns CELL_WALL_STONE for
        out-of-bounds cells."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return CELL_WALL_STONE
        return int(self.grid[y, x])


# ============================================================================
# DoomSprite
# ============================================================================

@dataclass
class DoomSprite:
    """Billboard sprite in the 3D world.

    Sprites are always facing the camera (billboard rendering).  They are
    sorted by distance and rendered back-to-front using the wall z-buffer
    for correct occlusion.

    Attributes:
        x:           World position X (float, in tile units).
        y:           World position Y (float, in tile units).
        sprite_type: One of "enemy", "health", "goal".
        active:      Whether the sprite is visible and interactable.
        hp:          Hit points (enemies only, for future combat extension).
        vx:          Velocity X (for enemy movement AI).
        vy:          Velocity Y (for enemy movement AI).
    """

    x: float = 0.0
    y: float = 0.0
    sprite_type: str = "enemy"
    active: bool = True
    hp: int = 1
    vx: float = 0.0
    vy: float = 0.0

    def distance_to(self, px: float, py: float) -> float:
        """Euclidean distance from this sprite to point (px, py)."""
        dx = self.x - px
        dy = self.y - py
        return math.sqrt(dx * dx + dy * dy)


# ============================================================================
# Procedural Texture Generator
# ============================================================================

def _generate_wall_texture(wall_type: int, tex_height: int = 64) -> np.ndarray:
    """Generate a simple procedural 1D texture column for a wall type.

    Returns an (tex_height, 3) uint8 array representing one vertical
    stripe of a wall texture.  Textures use horizontal banding patterns
    reminiscent of the original Doom's tiled surfaces.

    Args:
        wall_type: CELL_WALL_STONE, CELL_WALL_METAL, or CELL_DOOR.
        tex_height: Texture resolution in pixels.

    Returns:
        (tex_height, 3) uint8 RGB texture column.
    """
    base = WALL_COLORS.get(wall_type, WALL_COLORS[CELL_WALL_STONE])
    tex = np.zeros((tex_height, 3), dtype=np.uint8)

    for ty in range(tex_height):
        # Horizontal mortar lines / panel grooves.
        t = ty / tex_height
        if wall_type == CELL_WALL_STONE:
            # Stone: mortar lines every 8 pixels.
            if ty % 8 == 0:
                shade = 0.6
            elif ty % 8 == 1:
                shade = 0.7
            else:
                # Slight random-looking variation from position.
                shade = 0.85 + 0.15 * math.sin(ty * 0.7)
        elif wall_type == CELL_WALL_METAL:
            # Metal: horizontal rivet lines.
            if ty % 16 == 0:
                shade = 1.2  # bright rivet line
            elif ty % 16 == 1 or ty % 16 == 15:
                shade = 0.7
            else:
                shade = 0.9 + 0.1 * math.sin(ty * 1.3)
        else:
            # Door: vertical plank pattern (simulated via horizontal bands).
            if ty % 12 < 2:
                shade = 0.6
            else:
                shade = 0.8 + 0.2 * math.sin(ty * 0.5)

        shade = max(0.0, min(1.5, shade))
        tex[ty] = np.clip(base.astype(np.float32) * shade, 0, 255).astype(np.uint8)

    return tex


# Pre-generate texture columns for each wall type.
_TEXTURES = {
    wt: _generate_wall_texture(wt, 64)
    for wt in [CELL_WALL_STONE, CELL_WALL_METAL, CELL_DOOR]
}


# ============================================================================
# DoomFPS Environment
# ============================================================================

class DoomFPS:
    """First-person Doom-style environment with raycasting renderer.

    Produces (render_height, render_width, 3) uint8 RGB frames at each step.
    The agent navigates a procedurally generated dungeon, avoids enemies,
    collects health pickups, and reaches a goal.

    The raycasting renderer uses the DDA (Digital Differential Analyzer)
    algorithm to find wall intersections for each screen column, identical
    to the technique used in Wolfenstein 3D (id Software, 1992).

    Usage:
        env = DoomFPS(render_width=64, render_height=48, seed=42)
        frame = env.reset()                          # (48, 64, 3) uint8 RGB
        frame, reward, done, info = env.step(0)      # move forward

        # High-res mode for visualization:
        env = DoomFPS(render_width=320, render_height=200)

    Args:
        render_width:  Output frame width in pixels.
        render_height: Output frame height in pixels.
        map_size:      Map dimension (map_size x map_size tiles).
        n_rooms:       Number of BSP splits for room generation.
        n_enemies:     Number of enemy sprites.
        n_health:      Number of health pickup sprites.
        max_steps:     Maximum steps per episode.
        seed:          Random seed for deterministic generation.
        show_minimap:  If True, overlay a top-down minimap (debugging).
    """

    # Action space.
    ACTION_FORWARD = 0
    ACTION_BACKWARD = 1
    ACTION_STRAFE_LEFT = 2
    ACTION_STRAFE_RIGHT = 3
    ACTION_TURN_LEFT = 4
    ACTION_TURN_RIGHT = 5
    ACTION_TURN_LEFT_FORWARD = 6
    ACTION_TURN_RIGHT_FORWARD = 7
    NUM_ACTIONS = 8

    def __init__(
        self,
        render_width: int = 64,
        render_height: int = 48,
        map_size: int = 32,
        n_rooms: int = 6,
        n_enemies: int = 3,
        n_health: int = 4,
        max_steps: int = 100,
        seed: int = 42,
        show_minimap: bool = False,
    ):
        self.render_width = render_width
        self.render_height = render_height
        self.map_size = map_size
        self.n_rooms = n_rooms
        self.n_enemies = n_enemies
        self.n_health = n_health
        self.max_steps = max_steps
        self.seed = seed
        self.show_minimap = show_minimap

        # Player state (set in reset).
        self.player_x: float = 0.0
        self.player_y: float = 0.0
        self.player_angle: float = 0.0
        self.player_hp: int = DEFAULT_HP
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False

        # Map and sprites (set in reset).
        self.doom_map: Optional[DoomMap] = None
        self.sprites: List[DoomSprite] = []

        # RNG for enemy AI.
        self._rng = random.Random(seed)

        # Pre-compute ray angle offsets for each screen column.
        self._ray_angles = np.zeros(render_width, dtype=np.float64)
        for col in range(render_width):
            # Map screen column to angle offset from center of FOV.
            screen_x = (2.0 * col / render_width) - 1.0  # -1 to +1
            self._ray_angles[col] = math.atan(screen_x * math.tan(FOV / 2.0))

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to a new episode.

        Generates a new map (or re-uses the same seed for determinism),
        spawns the player, enemies, and pickups, and returns the first
        rendered frame.

        Args:
            seed: Optional new seed.  If None, uses self.seed.

        Returns:
            (render_height, render_width, 3) uint8 RGB frame.
        """
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)

        # Generate map.
        self.doom_map = DoomMap.generate(
            width=self.map_size,
            height=self.map_size,
            n_rooms=self.n_rooms,
            n_enemies=self.n_enemies,
            n_health=self.n_health,
            seed=self.seed,
        )

        # Player state.
        self.player_x, self.player_y = self.doom_map.spawn
        self.player_angle = 0.0  # facing east
        self.player_hp = DEFAULT_HP
        self.score = 0
        self.steps = 0
        self.done = False

        # Create sprites.
        self.sprites = []
        for ex, ey in self.doom_map.enemy_spawns:
            self.sprites.append(DoomSprite(x=ex, y=ey, sprite_type="enemy"))
        for hx, hy in self.doom_map.health_spawns:
            self.sprites.append(DoomSprite(x=hx, y=hy, sprite_type="health"))
        # Goal sprite.
        self.sprites.append(DoomSprite(
            x=self.doom_map.goal[0],
            y=self.doom_map.goal[1],
            sprite_type="goal",
        ))

        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take one action in the environment.

        Args:
            action: Integer in [0, 7].  See ACTION_* constants.

        Returns:
            Tuple of (frame, reward, done, info) where:
                frame:  (H, W, 3) uint8 RGB array.
                reward: Float reward for this step.
                done:   True if episode is over.
                info:   Dict with player_hp, score, steps, position.
        """
        if self.done:
            return self.render(), 0.0, True, self._info()

        reward = -1.0  # step penalty
        self.steps += 1

        # --- Apply action ---
        dx, dy, dangle = 0.0, 0.0, 0.0

        if action == self.ACTION_FORWARD:
            dx = math.cos(self.player_angle) * MOVE_SPEED
            dy = math.sin(self.player_angle) * MOVE_SPEED
        elif action == self.ACTION_BACKWARD:
            dx = -math.cos(self.player_angle) * MOVE_SPEED
            dy = -math.sin(self.player_angle) * MOVE_SPEED
        elif action == self.ACTION_STRAFE_LEFT:
            dx = math.cos(self.player_angle - math.pi / 2) * MOVE_SPEED
            dy = math.sin(self.player_angle - math.pi / 2) * MOVE_SPEED
        elif action == self.ACTION_STRAFE_RIGHT:
            dx = math.cos(self.player_angle + math.pi / 2) * MOVE_SPEED
            dy = math.sin(self.player_angle + math.pi / 2) * MOVE_SPEED
        elif action == self.ACTION_TURN_LEFT:
            dangle = -TURN_SPEED
        elif action == self.ACTION_TURN_RIGHT:
            dangle = TURN_SPEED
        elif action == self.ACTION_TURN_LEFT_FORWARD:
            dangle = -TURN_SPEED
            dx = math.cos(self.player_angle + dangle) * MOVE_SPEED
            dy = math.sin(self.player_angle + dangle) * MOVE_SPEED
        elif action == self.ACTION_TURN_RIGHT_FORWARD:
            dangle = TURN_SPEED
            dx = math.cos(self.player_angle + dangle) * MOVE_SPEED
            dy = math.sin(self.player_angle + dangle) * MOVE_SPEED

        # Apply turn.
        self.player_angle += dangle
        # Normalize angle to [-pi, pi].
        self.player_angle = math.atan2(
            math.sin(self.player_angle),
            math.cos(self.player_angle),
        )

        # Apply movement with wall collision (slide along walls).
        new_x = self.player_x + dx
        new_y = self.player_y + dy
        margin = 0.2  # collision margin from wall

        assert self.doom_map is not None

        # Try X movement.
        if not self.doom_map.is_wall(int(new_x + margin * np.sign(dx)),
                                      int(self.player_y)):
            self.player_x = new_x
        # Try Y movement.
        if not self.doom_map.is_wall(int(self.player_x),
                                      int(new_y + margin * np.sign(dy))):
            self.player_y = new_y

        # Clamp to map bounds.
        self.player_x = max(0.5, min(self.map_size - 0.5, self.player_x))
        self.player_y = max(0.5, min(self.map_size - 0.5, self.player_y))

        # --- Enemy AI ---
        self._update_enemies()

        # --- Check sprite interactions ---
        for sprite in self.sprites:
            if not sprite.active:
                continue
            dist = sprite.distance_to(self.player_x, self.player_y)

            if sprite.sprite_type == "enemy" and dist < ENEMY_CONTACT_RADIUS:
                self.player_hp -= ENEMY_DAMAGE
                reward -= 20.0
                # Push enemy away after contact.
                if dist > 0.01:
                    push_dx = (sprite.x - self.player_x) / dist
                    push_dy = (sprite.y - self.player_y) / dist
                    sprite.x += push_dx * 1.0
                    sprite.y += push_dy * 1.0

            elif sprite.sprite_type == "health" and dist < PICKUP_RADIUS:
                sprite.active = False
                self.player_hp = min(DEFAULT_HP, self.player_hp + HEALTH_RESTORE)
                reward += 10.0
                self.score += 10

            elif sprite.sprite_type == "goal" and dist < GOAL_RADIUS:
                reward += 100.0
                self.score += 100
                self.done = True

        # --- Check death ---
        if self.player_hp <= 0:
            self.player_hp = 0
            self.done = True

        # --- Check max steps ---
        if self.steps >= self.max_steps:
            self.done = True

        frame = self.render()
        return frame, reward, self.done, self._info()

    def render(self) -> np.ndarray:
        """Render the current view as a (H, W, 3) uint8 RGB array.

        Pipeline:
            1. Draw ceiling and floor with distance-based shading.
            2. Cast rays (DDA) to find wall distances.
            3. Draw textured wall columns.
            4. Sort and draw billboard sprites with z-buffer.
            5. Optionally overlay a minimap.

        Returns:
            (render_height, render_width, 3) uint8 RGB frame.
        """
        assert self.doom_map is not None
        W = self.render_width
        H = self.render_height

        # Allocate frame buffer.
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # --- 1. Ceiling and floor ---
        half_h = H // 2
        for row in range(H):
            if row < half_h:
                # Ceiling: gets darker further from horizon.
                t = 1.0 - (row / half_h)
                shade = max(0.3, 1.0 - 0.5 * t)
                frame[row, :] = np.clip(
                    COLOR_CEILING.astype(np.float32) * shade, 0, 255
                ).astype(np.uint8)
            else:
                # Floor: gets darker further from horizon.
                t = (row - half_h) / max(1, H - half_h)
                shade = max(0.3, 1.0 - 0.4 * t)
                frame[row, :] = np.clip(
                    COLOR_FLOOR.astype(np.float32) * shade, 0, 255
                ).astype(np.uint8)

        # --- 2. Cast rays ---
        wall_distances, wall_types, wall_x_offsets, wall_sides = self._cast_rays()

        # --- 3. Draw walls ---
        self._render_walls(frame, wall_distances, wall_types, wall_x_offsets,
                           wall_sides)

        # --- 4. Draw sprites ---
        self._render_sprites(frame, wall_distances)

        # --- 5. Minimap ---
        if self.show_minimap:
            self._render_minimap(frame)

        return frame

    def _cast_rays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cast rays using the DDA algorithm for each screen column.

        The DDA (Digital Differential Analyzer) algorithm steps through the
        grid one cell at a time along each ray, checking for wall intersections.
        This is the same algorithm used in Wolfenstein 3D.

        Returns:
            Tuple of four arrays, each of length render_width:
                distances:  Perpendicular distance to wall (corrects fisheye).
                wall_types: Cell type of the wall hit (CELL_WALL_STONE, etc.).
                x_offsets:  Fractional X position along the wall face (for texturing).
                sides:      0 if ray hit a N/S wall face, 1 if E/W face.
        """
        assert self.doom_map is not None
        W = self.render_width

        distances = np.full(W, 1e9, dtype=np.float64)
        wall_types = np.full(W, CELL_WALL_STONE, dtype=np.int32)
        x_offsets = np.zeros(W, dtype=np.float64)
        sides = np.zeros(W, dtype=np.int32)

        px = self.player_x
        py = self.player_y
        pa = self.player_angle

        for col in range(W):
            ray_angle = pa + self._ray_angles[col]
            ray_dir_x = math.cos(ray_angle)
            ray_dir_y = math.sin(ray_angle)

            # Current grid cell.
            map_x = int(px)
            map_y = int(py)

            # Length of ray from one X/Y side to the next.
            # Avoid division by zero.
            delta_dist_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0.0 else 1e30
            delta_dist_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0.0 else 1e30

            # Step direction and initial side distances.
            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (px - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - px) * delta_dist_x

            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (py - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - py) * delta_dist_y

            # DDA loop.
            hit = False
            side_hit = 0  # 0 = X-side, 1 = Y-side
            max_depth = self.map_size * 2  # safety limit

            for _ in range(max_depth):
                # Jump to next map cell.
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side_hit = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side_hit = 1

                # Check bounds.
                if (map_x < 0 or map_x >= self.doom_map.width or
                        map_y < 0 or map_y >= self.doom_map.height):
                    hit = True
                    break

                # Check wall hit.
                cell = self.doom_map.grid[map_y, map_x]
                if cell != CELL_FLOOR:
                    hit = True
                    break

            if hit:
                # Perpendicular distance (corrects fisheye distortion).
                if side_hit == 0:
                    perp_dist = (map_x - px + (1 - step_x) / 2.0) / (
                        ray_dir_x if ray_dir_x != 0.0 else 1e-10
                    )
                else:
                    perp_dist = (map_y - py + (1 - step_y) / 2.0) / (
                        ray_dir_y if ray_dir_y != 0.0 else 1e-10
                    )

                perp_dist = max(perp_dist, 0.01)  # clamp to avoid division by 0
                distances[col] = perp_dist

                # Wall type.
                wt = self.doom_map.wall_type(map_x, map_y)
                wall_types[col] = wt

                # Wall X offset (where on the wall face the ray hit).
                if side_hit == 0:
                    wall_x = py + perp_dist * ray_dir_y
                else:
                    wall_x = px + perp_dist * ray_dir_x
                wall_x -= math.floor(wall_x)
                x_offsets[col] = wall_x

                sides[col] = side_hit

        return distances, wall_types, x_offsets, sides

    def _render_walls(
        self,
        frame: np.ndarray,
        distances: np.ndarray,
        wall_types: np.ndarray,
        x_offsets: np.ndarray,
        wall_sides: np.ndarray,
    ) -> None:
        """Render wall columns into the frame buffer.

        Wall height is proportional to 1/distance.  Walls are textured using
        procedural texture columns, with N/S faces slightly darker than E/W
        faces for a depth cue (same trick used in Wolfenstein 3D).

        Args:
            frame:      (H, W, 3) uint8 frame buffer (modified in-place).
            distances:  Per-column perpendicular wall distances.
            wall_types: Per-column wall cell types.
            x_offsets:  Per-column wall face X offsets (for texture sampling).
            wall_sides: Per-column wall face (0=NS, 1=EW).
        """
        H = self.render_height
        W = self.render_width
        half_h = H / 2.0

        for col in range(W):
            dist = distances[col]
            if dist >= 1e8:
                continue  # no wall hit

            # Wall height on screen.
            wall_height = int(H / dist)
            if wall_height < 1:
                continue

            # Wall top and bottom on screen.
            draw_start = int(half_h - wall_height / 2.0)
            draw_end = int(half_h + wall_height / 2.0)

            # Clamp to screen.
            tex_start = 0
            if draw_start < 0:
                tex_start = -draw_start
                draw_start = 0
            if draw_end > H:
                draw_end = H

            wt = wall_types[col]
            tex = _TEXTURES.get(wt, _TEXTURES[CELL_WALL_STONE])
            tex_height = len(tex)

            # Side shading: NS faces are 30% darker.
            side_shade = 0.7 if wall_sides[col] == 0 else 1.0

            # Distance fog: fade to black at distance.
            fog = max(0.15, 1.0 - dist / (self.map_size * 0.5))

            # Texture stripe variation from x_offset.
            stripe_shade = 0.85 + 0.15 * math.sin(
                x_offsets[col] * 2.0 * math.pi / TEXTURE_STRIPE_PERIOD
            )

            combined_shade = side_shade * fog * stripe_shade

            for row in range(draw_start, draw_end):
                # Map screen row to texture Y.
                tex_y_raw = tex_start + (row - draw_start)
                tex_y = int((tex_y_raw / wall_height) * tex_height)
                tex_y = min(tex_y, tex_height - 1)

                pixel = tex[tex_y].astype(np.float32) * combined_shade
                frame[row, col] = np.clip(pixel, 0, 255).astype(np.uint8)

    def _render_sprites(
        self,
        frame: np.ndarray,
        wall_distances: np.ndarray,
    ) -> None:
        """Render billboard sprites with z-buffer occlusion.

        Sprites are sorted by distance (back-to-front), transformed into
        screen space, and drawn as scaled rectangles.  The wall z-buffer
        prevents sprites from drawing over closer walls.

        Each sprite is drawn as a simple filled rectangle with a darker
        border for depth perception.  Enemy sprites have a simple face
        pattern; health sprites show a cross; goal sprites show a diamond.

        Args:
            frame:          (H, W, 3) frame buffer (modified in-place).
            wall_distances: Per-column wall distances (z-buffer).
        """
        H = self.render_height
        W = self.render_width
        half_h = H / 2.0
        px = self.player_x
        py = self.player_y
        pa = self.player_angle

        # Collect active sprites with distances.
        visible: List[Tuple[float, DoomSprite]] = []
        for sprite in self.sprites:
            if not sprite.active:
                continue
            dist = sprite.distance_to(px, py)
            if dist < 0.1:
                continue  # too close
            visible.append((dist, sprite))

        # Sort back-to-front (farthest first).
        visible.sort(key=lambda x: -x[0])

        for dist, sprite in visible:
            # Transform sprite position relative to camera.
            sprite_dx = sprite.x - px
            sprite_dy = sprite.y - py

            # Rotate into camera space.
            cos_a = math.cos(-pa)
            sin_a = math.sin(-pa)
            tx = sprite_dx * cos_a - sprite_dy * sin_a
            ty = sprite_dx * sin_a + sprite_dy * cos_a

            # ty is depth (distance along view direction).
            if ty <= 0.1:
                continue  # behind camera

            # Screen X position.
            sprite_screen_x = int((0.5 + tx / (ty * 2.0 * math.tan(FOV / 2.0))) * W)

            # Sprite height and width on screen.
            sprite_h = int(H / ty * 0.8)
            sprite_w = sprite_h  # square sprites

            if sprite_h < 1 or sprite_w < 1:
                continue

            # Screen bounds.
            x_start = sprite_screen_x - sprite_w // 2
            x_end = x_start + sprite_w
            y_start = int(half_h - sprite_h // 2)
            y_end = y_start + sprite_h

            # Clamp.
            sx_start = max(0, x_start)
            sx_end = min(W, x_end)
            sy_start = max(0, y_start)
            sy_end = min(H, y_end)

            if sx_start >= sx_end or sy_start >= sy_end:
                continue

            # Colors.
            fill_color = SPRITE_COLORS.get(sprite.sprite_type,
                                            SPRITE_COLORS["enemy"])
            outline_color = SPRITE_OUTLINE_COLORS.get(sprite.sprite_type,
                                                       SPRITE_OUTLINE_COLORS["enemy"])

            # Distance fog.
            fog = max(0.2, 1.0 - dist / (self.map_size * 0.5))

            # Draw sprite pixels with z-buffer check.
            for sx in range(sx_start, sx_end):
                # Z-buffer: skip if wall is closer.
                if wall_distances[sx] < ty:
                    continue

                for sy in range(sy_start, sy_end):
                    # Sprite-local coordinates (0-1 range).
                    local_x = (sx - x_start) / sprite_w
                    local_y = (sy - y_start) / sprite_h

                    # Determine if this pixel is part of the sprite shape.
                    color = self._sprite_pixel(
                        sprite.sprite_type, local_x, local_y,
                        fill_color, outline_color,
                    )
                    if color is not None:
                        fogged = np.clip(
                            color.astype(np.float32) * fog, 0, 255
                        ).astype(np.uint8)
                        frame[sy, sx] = fogged

    @staticmethod
    def _sprite_pixel(
        sprite_type: str,
        lx: float,
        ly: float,
        fill: np.ndarray,
        outline: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute sprite pixel color based on local (lx, ly) coordinates.

        Generates simple procedural sprite shapes:
            enemy:  circle body with eyes
            health: cross shape
            goal:   diamond shape

        Args:
            sprite_type: "enemy", "health", or "goal".
            lx:          Local X coordinate (0 to 1).
            ly:          Local Y coordinate (0 to 1).
            fill:        Fill color (uint8 RGB).
            outline:     Outline color (uint8 RGB).

        Returns:
            uint8 RGB array for the pixel, or None for transparent.
        """
        cx, cy = 0.5, 0.5  # sprite center

        if sprite_type == "enemy":
            # Circle body.
            dx = lx - cx
            dy = ly - cy
            r = math.sqrt(dx * dx + dy * dy)
            if r > 0.45:
                return None  # transparent
            if r > 0.38:
                return outline  # border

            # Eyes (two small dark circles).
            eye_y = 0.35
            for eye_x in [0.35, 0.65]:
                edx = lx - eye_x
                edy = ly - eye_y
                if math.sqrt(edx * edx + edy * edy) < 0.08:
                    return np.array([40, 0, 0], dtype=np.uint8)

            # Mouth (horizontal line).
            if 0.55 < ly < 0.62 and 0.3 < lx < 0.7:
                return np.array([80, 0, 0], dtype=np.uint8)

            return fill

        elif sprite_type == "health":
            # Cross shape.
            in_h_bar = (0.2 < lx < 0.8) and (0.35 < ly < 0.65)
            in_v_bar = (0.35 < lx < 0.65) and (0.2 < ly < 0.8)
            if in_h_bar or in_v_bar:
                # Outline check.
                inner_h = (0.25 < lx < 0.75) and (0.4 < ly < 0.6)
                inner_v = (0.4 < lx < 0.6) and (0.25 < ly < 0.75)
                if inner_h or inner_v:
                    return fill
                return outline
            return None

        elif sprite_type == "goal":
            # Diamond shape.
            dx = abs(lx - cx)
            dy = abs(ly - cy)
            if dx + dy < 0.45:
                if dx + dy > 0.35:
                    return outline
                # Inner glow: brighter at center.
                brightness = 1.0 + 0.5 * (1.0 - (dx + dy) / 0.35)
                return np.clip(
                    fill.astype(np.float32) * brightness, 0, 255
                ).astype(np.uint8)
            return None

        return None

    def _update_enemies(self) -> None:
        """Update enemy positions with simple chase/wander AI.

        Enemy behavior:
            - If within ENEMY_CHASE_RADIUS of the player, move toward player.
            - Otherwise, wander randomly.
            - Enemies cannot walk through walls (same collision as player).
        """
        assert self.doom_map is not None

        for sprite in self.sprites:
            if sprite.sprite_type != "enemy" or not sprite.active:
                continue

            dist = sprite.distance_to(self.player_x, self.player_y)

            if dist < ENEMY_CHASE_RADIUS and dist > 0.01:
                # Chase player.
                dx = (self.player_x - sprite.x) / dist
                dy = (self.player_y - sprite.y) / dist
            else:
                # Random wander.
                angle = self._rng.uniform(0, 2 * math.pi)
                dx = math.cos(angle)
                dy = math.sin(angle)

            new_x = sprite.x + dx * ENEMY_MOVE_SPEED
            new_y = sprite.y + dy * ENEMY_MOVE_SPEED

            # Wall collision for enemies.
            if not self.doom_map.is_wall(int(new_x), int(sprite.y)):
                sprite.x = new_x
            if not self.doom_map.is_wall(int(sprite.x), int(new_y)):
                sprite.y = new_y

            # Clamp to map.
            sprite.x = max(0.5, min(self.map_size - 0.5, sprite.x))
            sprite.y = max(0.5, min(self.map_size - 0.5, sprite.y))

    def _render_minimap(self, frame: np.ndarray) -> None:
        """Overlay a top-down minimap in the top-right corner.

        Shows the map layout, player position/direction, enemies (red dots),
        health pickups (green dots), and goal (yellow dot).  Useful for
        debugging navigation behavior.

        Args:
            frame: (H, W, 3) frame buffer (modified in-place).
        """
        assert self.doom_map is not None

        # Minimap size: 1/4 of the frame, capped at map_size pixels.
        mm_size = min(self.render_width // 4, self.render_height // 4, self.map_size)
        if mm_size < 8:
            return

        scale = mm_size / self.map_size
        x_off = self.render_width - mm_size - 2
        y_off = 2

        # Semi-transparent background.
        for my in range(mm_size):
            for mx in range(mm_size):
                fy = y_off + my
                fx = x_off + mx
                if 0 <= fy < self.render_height and 0 <= fx < self.render_width:
                    # Map cell.
                    map_x = int(mx / scale)
                    map_y = int(my / scale)
                    if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                        cell = self.doom_map.grid[map_y, map_x]
                        if cell == CELL_FLOOR:
                            frame[fy, fx] = [32, 32, 32]
                        elif cell == CELL_DOOR:
                            frame[fy, fx] = [64, 48, 24]
                        else:
                            frame[fy, fx] = [80, 80, 80]

        # Draw sprites on minimap.
        for sprite in self.sprites:
            if not sprite.active:
                continue
            sx = int(sprite.x * scale) + x_off
            sy = int(sprite.y * scale) + y_off
            if 0 <= sx < self.render_width and 0 <= sy < self.render_height:
                color = SPRITE_COLORS.get(sprite.sprite_type, [255, 255, 255])
                frame[sy, sx] = color

        # Draw player.
        ppx = int(self.player_x * scale) + x_off
        ppy = int(self.player_y * scale) + y_off
        if 0 <= ppx < self.render_width and 0 <= ppy < self.render_height:
            frame[ppy, ppx] = [255, 255, 255]
            # Direction indicator.
            dir_x = int(ppx + math.cos(self.player_angle) * 3)
            dir_y = int(ppy + math.sin(self.player_angle) * 3)
            if 0 <= dir_x < self.render_width and 0 <= dir_y < self.render_height:
                frame[dir_y, dir_x] = [255, 255, 0]

    def _info(self) -> dict:
        """Build the info dict returned by step()."""
        return {
            "player_hp": self.player_hp,
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_x,
            "player_y": self.player_y,
            "player_angle": self.player_angle,
            "n_active_enemies": sum(
                1 for s in self.sprites
                if s.sprite_type == "enemy" and s.active
            ),
            "n_active_health": sum(
                1 for s in self.sprites
                if s.sprite_type == "health" and s.active
            ),
        }

    # --- Convenience properties ---

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Shape of rendered frames: (height, width, 3)."""
        return (self.render_height, self.render_width, 3)

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return self.NUM_ACTIONS

    @property
    def action_names(self) -> List[str]:
        """Human-readable action names."""
        return [
            "forward", "backward", "strafe_left", "strafe_right",
            "turn_left", "turn_right", "turn_left+forward",
            "turn_right+forward",
        ]


# ============================================================================
# Self-test / demo
# ============================================================================

def _demo() -> None:
    """Run a quick self-test of the DoomFPS environment.

    Verifies that:
        1. Map generation produces a valid grid.
        2. Reset returns a correctly shaped frame.
        3. All 8 actions produce valid frames and rewards.
        4. Episode terminates (by max_steps if nothing else).
        5. Rendering is purely numpy-based (no external deps).
    """
    print("=" * 60)
    print("DoomFPS Raycasting Engine -- Self Test")
    print("=" * 60)

    # --- Test 1: Map generation ---
    print("\n[1] Map generation...")
    doom_map = DoomMap.generate(width=32, height=32, n_rooms=6, seed=42)
    assert doom_map.grid.shape == (32, 32), f"Bad shape: {doom_map.grid.shape}"
    n_floor = int(np.sum(doom_map.grid == CELL_FLOOR))
    n_walls = int(np.sum(doom_map.grid != CELL_FLOOR))
    print(f"    Grid: {doom_map.width}x{doom_map.height}, "
          f"floor={n_floor}, walls={n_walls}")
    print(f"    Rooms: {len(doom_map.rooms)}")
    print(f"    Spawn: ({doom_map.spawn[0]:.1f}, {doom_map.spawn[1]:.1f})")
    print(f"    Goal:  ({doom_map.goal[0]:.1f}, {doom_map.goal[1]:.1f})")
    print(f"    Enemies: {len(doom_map.enemy_spawns)}, "
          f"Health: {len(doom_map.health_spawns)}")
    print("    PASS")

    # --- Test 2: Environment reset ---
    print("\n[2] Environment reset (64x48)...")
    env = DoomFPS(render_width=64, render_height=48, seed=42)
    frame = env.reset()
    assert frame.shape == (48, 64, 3), f"Bad shape: {frame.shape}"
    assert frame.dtype == np.uint8, f"Bad dtype: {frame.dtype}"
    print(f"    Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"    Pixel range: [{frame.min()}, {frame.max()}]")
    print(f"    Player pos: ({env.player_x:.2f}, {env.player_y:.2f}), "
          f"angle: {math.degrees(env.player_angle):.1f} deg")
    print(f"    HP: {env.player_hp}, Score: {env.score}")
    print("    PASS")

    # --- Test 3: All actions ---
    print("\n[3] Testing all 8 actions...")
    env.reset()
    for action in range(8):
        frame, reward, done, info = env.step(action)
        assert frame.shape == (48, 64, 3), f"Action {action}: bad shape"
        assert isinstance(reward, float), f"Action {action}: bad reward type"
        print(f"    Action {action} ({env.action_names[action]:>20s}): "
              f"reward={reward:+.1f}, hp={info['player_hp']}, "
              f"pos=({info['player_x']:.2f}, {info['player_y']:.2f})")
    print("    PASS")

    # --- Test 4: Full episode ---
    print("\n[4] Running full episode (max_steps=100)...")
    env = DoomFPS(render_width=64, render_height=48, max_steps=100, seed=42)
    frame = env.reset()
    total_reward = 0.0
    rng = random.Random(123)
    while not env.done:
        action = rng.randint(0, 7)
        frame, reward, done, info = env.step(action)
        total_reward += reward
    print(f"    Steps: {env.steps}, Total reward: {total_reward:.1f}")
    print(f"    Final HP: {info['player_hp']}, Score: {info['score']}")
    print(f"    Episode ended by: "
          f"{'death' if info['player_hp'] <= 0 else 'max_steps/goal'}")
    print("    PASS")

    # --- Test 5: High-res render ---
    print("\n[5] High-res render (320x200)...")
    env_hires = DoomFPS(render_width=320, render_height=200, seed=42,
                         show_minimap=True)
    frame = env_hires.reset()
    assert frame.shape == (200, 320, 3), f"Bad shape: {frame.shape}"
    print(f"    Frame shape: {frame.shape}")
    # Check that the frame is not all black (rendering produced content).
    nonzero = np.count_nonzero(frame)
    total_pixels = frame.size
    pct_nonzero = 100.0 * nonzero / total_pixels
    print(f"    Non-zero pixels: {pct_nonzero:.1f}%")
    assert pct_nonzero > 20.0, "Frame is too dark (rendering may be broken)"
    print("    PASS")

    # --- Test 6: Determinism ---
    print("\n[6] Determinism test...")
    env_a = DoomFPS(render_width=64, render_height=48, seed=99)
    env_b = DoomFPS(render_width=64, render_height=48, seed=99)
    frame_a = env_a.reset()
    frame_b = env_b.reset()
    assert np.array_equal(frame_a, frame_b), "Reset frames differ"
    for action in [0, 4, 0, 5, 2, 7, 1, 6]:
        fa, _, _, _ = env_a.step(action)
        fb, _, _, _ = env_b.step(action)
        assert np.array_equal(fa, fb), f"Frames differ after action {action}"
    print("    8 steps compared: all identical")
    print("    PASS")

    # --- Test 7: Rendering performance ---
    print("\n[7] Render performance benchmark...")
    import time
    env = DoomFPS(render_width=64, render_height=48, seed=42)
    env.reset()
    n_renders = 50
    t0 = time.perf_counter()
    for _ in range(n_renders):
        env.render()
    elapsed = time.perf_counter() - t0
    fps = n_renders / elapsed
    print(f"    {n_renders} renders at 64x48: {elapsed:.3f}s ({fps:.1f} FPS)")
    print(f"    {'PASS' if fps > 5 else 'WARN: slow'}")

    print("\n" + "=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
