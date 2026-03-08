"""oNeuro environments -- simulated worlds for dONN brains.

Environments produce pixel frames (numpy uint8 RGB arrays) that can be
processed by a molecular retina and fed into a digital organic neural network.

Available environments:
    DoomFPS       -- First-person Doom-style raycasting renderer
    DoomMap       -- Procedurally generated BSP-style map
    DoomSprite    -- Billboard sprite in the 3D world
"""

from oneuro.environments.doom_fps import DoomFPS, DoomMap, DoomSprite

__all__ = ["DoomFPS", "DoomMap", "DoomSprite"]
