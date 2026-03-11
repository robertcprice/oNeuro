#!/usr/bin/env python3
"""ViZDoom via Molecular Retina — digital Organic Neural Networks (dONNs) Play Doom.

A dONN (digital Organic Neural Network) plays actual Doom through biologically
faithful "organic eyes" (molecular retina). ViZDoom renders game frames, the
MolecularRetina converts RGB pixels to spike trains through three biological
layers (photoreceptors, bipolar cells, retinal ganglion cells), and a
CUDARegionalBrain processes those spikes via Hodgkin-Huxley neurons with
molecular neurotransmitter dynamics. Motor neuron populations drive game actions.
Learning uses the FREE ENERGY PRINCIPLE — no reward signal, just structured
vs unstructured sensory feedback.

Terminology:
  - ONN:    Organic Neural Network — real biological neurons (DishBrain, FinalSpark)
  - dONN:   digital Organic Neural Network — oNeuro's biophysically faithful simulation
  - oNeuro: The platform for building and running dONNs

3 Experiments:
   1. Doom Navigation (30 episodes): health gathering via free energy principle
   2. Learning Speed Comparison (30 episodes x 3 protocols): FEP vs DA vs Random
   3. Doom Drug Effects: baseline / caffeine / diazepam pharmacological modulation

Key innovation: Learning via FREE ENERGY PRINCIPLE, not reward/punishment.
  - Positive event (health gain): STRUCTURED pulse to cortex (predictable = low entropy)
  - Negative event (damage taken): RANDOM noise to 30% cortex (unpredictable = high entropy)
  - Neurons self-organize via STDP to prefer states that produce predictable feedback.

References:
  - Kagan et al. (2022) "In vitro neurons learn and exhibit sentience when
    embodied in a simulated game-world" Neuron 110(23):3952-3969
  - Friston (2010) "The free-energy principle: a unified brain theory?"
    Nature Reviews Neuroscience 11:127-138

Usage:
    python3 demos/demo_doom_vizdoom.py                         # all 3 experiments
    python3 demos/demo_doom_vizdoom.py --exp 1                 # just navigation
    python3 demos/demo_doom_vizdoom.py --scenario take_cover   # different scenario
    python3 demos/demo_doom_vizdoom.py --scale medium          # more neurons
    python3 demos/demo_doom_vizdoom.py --exp 1 --pretrain-episodes 2 \
        --teacher-motor-intensity 60 --teacher-hebbian-delta 1.2  # tuned teacher-guided warmup
    python3 demos/demo_doom_vizdoom.py --json results.json     # structured output
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Force unbuffered stdout for real-time progress reporting
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from oneuro.molecular.cuda_backend import (
    CUDAMolecularBrain,
    CUDARegionalBrain,
    detect_backend,
    NT_DA, NT_5HT, NT_NE, NT_GLU, NT_GABA,
)
from oneuro.molecular.retina import MolecularRetina

from demo_language_cuda import (
    _warmup,
    _header,
    _get_region_ids,
    _get_all_cortex_ids,
    _get_cortex_l5_ids,
    SCALE_COLUMNS,
)

try:
    import vizdoom as vzd
    HAS_VIZDOOM = True
except ImportError:
    HAS_VIZDOOM = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# Constants
# ============================================================================

RETINA_WIDTH = 64
RETINA_HEIGHT = 48
VIZDOOM_WIDTH = 160
VIZDOOM_HEIGHT = 120

HOSTILE_KEYWORDS = (
    "zombieman", "shotgunguy", "chaingunguy", "chaingunner", "demon",
    "imp", "cacodemon", "hellknight", "baron", "revenant", "mancubus",
    "arachnotron", "lostsoul", "spectre",
)
NON_HOSTILE_KEYWORDS = (
    "doomplayer", "marine", "medikit", "stimpack", "health", "armor",
    "greenarmor", "bluearmor", "redarmor", "clip", "ammo", "weapon",
    "shotgun", "chaingun", "chainsaw",
)

# Motor populations: L5 neurons split into 6 groups (5 movement + 1 attack)
MOTOR_FORWARD = 0
MOTOR_TURN_LEFT = 1
MOTOR_TURN_RIGHT = 2
MOTOR_STRAFE_LEFT = 3
MOTOR_STRAFE_RIGHT = 4
MOTOR_ATTACK = 5
N_MOTOR_POPULATIONS = 6

MOTOR_NAMES = ["forward", "turn_left", "turn_right", "strafe_left", "strafe_right", "attack"]

# Scale-aware simulation parameters: smaller networks need fewer steps
# to keep total runtime manageable while preserving experiment validity.
SCALE_PARAMS = {
    "small": {
        "stim_steps": 5,           # brain steps per game frame
        "max_game_steps": 80,      # frames per episode
        "structured_steps": 10,    # FEP positive feedback steps
        "unstructured_steps": 15,  # FEP negative feedback steps
        "neutral_steps": 3,        # settling steps on neutral events
        "n_episodes": 10,          # default episodes for exp 1/2
        "n_train_episodes": 8,     # training episodes for exp 3
        "n_test_episodes": 5,      # test episodes for exp 3
        "warmup_steps": 200,
    },
    "medium": {
        "stim_steps": 10,
        "max_game_steps": 200,
        "structured_steps": 25,
        "unstructured_steps": 40,
        "neutral_steps": 5,
        "n_episodes": 20,
        "n_train_episodes": 15,
        "n_test_episodes": 8,
        "warmup_steps": 300,
    },
    "large": {
        "stim_steps": 20,
        "max_game_steps": 500,
        "structured_steps": 50,
        "unstructured_steps": 100,
        "neutral_steps": 5,
        "n_episodes": 30,
        "n_train_episodes": 20,
        "n_test_episodes": 10,
        "warmup_steps": 300,
    },
}
# Default fallback uses "large" params for any unrecognized scale
for _s in ("mega", "100k", "1m", "xlarge", "standard", "minimal"):
    SCALE_PARAMS[_s] = SCALE_PARAMS["large"]

# Scenario configurations
SCENARIOS = {
    "health_gathering": {
        "cfg": "health_gathering.cfg",
        "description": "Navigate to collect health vials",
        "positive_metric": "health_gained",
    },
    "my_way_home": {
        "cfg": "my_way_home.cfg",
        "description": "Maze navigation to a vest",
        "positive_metric": "survived",
    },
    "take_cover": {
        "cfg": "take_cover.cfg",
        "description": "Dodge fireballs (avoidance learning)",
        "positive_metric": "survived",
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
        "description": "Navigate corridor with enemies",
        "positive_metric": "survived",
    },
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
        "description": "Defend against approaching enemies",
        "positive_metric": "survived",
    },
}


def _scenario_positive_metric(scenario: str) -> str:
    """Return the primary success metric for a scenario."""
    return SCENARIOS.get(
        scenario, SCENARIOS["health_gathering"]
    ).get("positive_metric", "health_gained")


def _episode_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    """Read an episode metric as a numeric value."""
    value = metrics.get(metric_name, 0.0)
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _metric_label(metric_name: str) -> str:
    """Human-readable metric label for logs."""
    if metric_name == "survived":
        return "survival"
    return metric_name.replace("_", " ")


def _format_metric_value(metric_name: str, value: float) -> str:
    """Format a metric value for progress logs."""
    if metric_name == "survived":
        return f"{value * 100:.0f}%"
    return f"{value:.1f}"


def _active_relay_source_ids(
    relay_ids: torch.Tensor,
    activation: torch.Tensor,
    threshold_ratio: float = 0.35,
    min_count: int = 8,
) -> torch.Tensor:
    """Select the currently active relay subset for targeted credit assignment."""
    if activation.numel() == 0:
        return relay_ids

    max_val = float(activation.max().item())
    if max_val <= 0.0:
        return relay_ids

    active_idx = torch.nonzero(
        activation >= max_val * threshold_ratio, as_tuple=False
    ).flatten()

    if active_idx.numel() < min_count:
        k = min(len(relay_ids), max(min_count, len(relay_ids) // 12))
        _, active_idx = torch.topk(activation, k=k)

    active_idx = torch.unique(active_idx).to(device=relay_ids.device, dtype=torch.long)
    if active_idx.numel() == 0:
        return relay_ids
    return relay_ids[active_idx]


def _is_hostile_name(name: str) -> bool:
    """Return whether a ViZDoom label/object name looks like an enemy."""
    lower = name.lower()
    if any(token in lower for token in NON_HOSTILE_KEYWORDS):
        return False
    return any(token in lower for token in HOSTILE_KEYWORDS)


def _normalize_angle_deg(angle: float) -> float:
    """Wrap degrees to [-180, 180)."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return wrapped


def _default_video_path(stem: str) -> str:
    """Return a repo-local path for saved Doom recordings."""
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", "doom_videos")
    )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem)


# ============================================================================
# ViZDoom Game Wrapper
# ============================================================================

class DoomGame:
    """Wraps ViZDoom to provide a clean interface for the dONN game loop.

    Handles initialization, frame capture, downsampling, action execution,
    and health/damage tracking for FEP-based learning.

    Args:
        scenario: One of the supported scenario names.
        seed: Random seed for game reproducibility.
        visible: Whether to show the game window.
    """

    def __init__(self, scenario: str = "health_gathering", seed: int = 42,
                 visible: bool = False):
        if not HAS_VIZDOOM:
            raise ImportError(
                "ViZDoom is required: pip install vizdoom")
        if not HAS_PIL:
            raise ImportError(
                "Pillow is required for frame downsampling: pip install Pillow")

        self.scenario = scenario
        self.seed = seed
        self._game = vzd.DoomGame()
        self._setup(scenario, visible)
        self._prev_health = 100.0
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_kills = 0
        self._episode_survived = True
        self._episode_steps = 0
        self._total_episodes = 0
        self._last_health_delta = 0.0
        self._corridor_evasive_steps = 0
        self._corridor_strafe_dir = MOTOR_STRAFE_LEFT

    def _setup(self, scenario: str, visible: bool) -> None:
        """Configure ViZDoom with the selected scenario.

        Loads the scenario .cfg (which includes the .wad path, rewards, and
        episode timeout), then overrides screen resolution, buttons, and
        game variables to match our 5-action motor decoder.

        Args:
            scenario: Scenario name from SCENARIOS dict.
            visible: Whether to render the game window.
        """
        cfg = SCENARIOS[scenario]["cfg"]
        self._game.load_config(os.path.join(vzd.scenarios_path, cfg))

        # Override rendering for our retina pipeline
        self._game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        self._game.set_window_visible(visible)
        self._game.set_mode(vzd.Mode.PLAYER)
        self._game.set_objects_info_enabled(True)
        self._game.set_labels_buffer_enabled(True)

        # Override buttons: clear cfg defaults, add our 6-action set (5 movement + attack)
        self._game.clear_available_buttons()
        self._game.add_available_button(vzd.Button.MOVE_FORWARD)
        self._game.add_available_button(vzd.Button.TURN_LEFT)
        self._game.add_available_button(vzd.Button.TURN_RIGHT)
        self._game.add_available_button(vzd.Button.MOVE_LEFT)
        self._game.add_available_button(vzd.Button.MOVE_RIGHT)
        self._game.add_available_button(vzd.Button.ATTACK)

        # Override game variables: health and kills
        self._game.clear_available_game_variables()
        self._game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self._game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self._game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        self._game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        self._game.add_available_game_variable(vzd.GameVariable.ANGLE)

        self._game.set_seed(self.seed)
        self._game.init()

        # Track kills for event detection
        self._episode_kills = 0
        self._prev_kills = 0

        # Video recording: save screen buffer each frame
        self._frame_buffer = []

    def new_episode(self) -> np.ndarray:
        """Start a new episode.

        Returns:
            Downsampled first frame as (48, 64, 3) uint8 array.
        """
        self._game.new_episode()
        self._prev_health = self._get_health()
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_kills = 0
        self._prev_kills = self._get_kills()
        self._episode_survived = True
        self._episode_steps = 0
        self._last_health_delta = 0.0
        self._corridor_evasive_steps = 0
        self._corridor_strafe_dir = MOTOR_STRAFE_LEFT
        self._total_episodes += 1
        return self._get_frame()

    def step(
        self,
        action_idx: Optional[int] = None,
        *,
        action_vector: Optional[List[int]] = None,
    ) -> Tuple[str, float, bool, np.ndarray]:
        """Execute one action and return results.

        Args:
            action_idx: Optional motor population index (0-5, ATTACK is 5).
            action_vector: Optional full 6-button action vector. If provided,
                it is used directly and can enable simultaneous movement+attack.

        Returns:
            Tuple of (event, health_delta, done, frame), where event is one of
            "health_gained", "damage_taken", "kill", "survived", "neutral", or "episode_end".
        """
        if action_vector is not None:
            action = [1 if int(v) else 0 for v in action_vector[:N_MOTOR_POPULATIONS]]
            if len(action) < N_MOTOR_POPULATIONS:
                action.extend([0] * (N_MOTOR_POPULATIONS - len(action)))
        else:
            if action_idx is None:
                raise ValueError("Either action_idx or action_vector must be provided")
            # Build one-hot action vector
            action = [0] * N_MOTOR_POPULATIONS
            action[action_idx] = 1

        self._game.make_action(action)
        self._episode_steps += 1

        # Track kills (positive event - killed enemy)
        current_kills = self._get_kills()
        kills_delta = max(0, current_kills - self._prev_kills)
        if kills_delta > 0:
            self._episode_kills += kills_delta
        self._prev_kills = current_kills

        # Track health
        current_health = self._get_health()
        health_delta = current_health - self._prev_health
        self._last_health_delta = health_delta
        self._prev_health = current_health
        if health_delta > 0:
            self._episode_health_gained += health_delta
        elif health_delta < 0:
            self._episode_damage_taken += abs(health_delta)

        if self.scenario == "deadly_corridor":
            if health_delta < 0:
                self._corridor_evasive_steps = 6
                self._corridor_strafe_dir = (
                    MOTOR_STRAFE_RIGHT
                    if self._corridor_strafe_dir == MOTOR_STRAFE_LEFT
                    else MOTOR_STRAFE_LEFT
                )
            elif self._corridor_evasive_steps > 0:
                self._corridor_evasive_steps -= 1
            if kills_delta > 0:
                self._corridor_evasive_steps = 0

        episode_finished = self._game.is_episode_finished()
        if episode_finished and self._is_player_dead(current_health):
            self._episode_survived = False
            return "episode_end", health_delta, True, np.zeros(
                (RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8)

        if kills_delta > 0:
            event = "kill"
        elif health_delta > 0:
            event = "health_gained"
        elif health_delta < 0:
            event = "damage_taken"
        elif episode_finished:
            event = "survived"
        else:
            event = "neutral"

        frame = np.zeros(
            (RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8
        ) if episode_finished else self._get_frame()
        return event, health_delta, episode_finished, frame

    def _get_health(self) -> float:
        """Read current health from game variables."""
        try:
            return float(self._game.get_game_variable(vzd.GameVariable.HEALTH))
        except Exception:
            return 0.0

    def _get_kills(self) -> int:
        """Read current kill count from game variables."""
        try:
            return int(self._game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        except Exception:
            return 0

    def _is_player_dead(self, current_health: Optional[float] = None) -> bool:
        """Return whether the player died this episode."""
        try:
            return bool(self._game.is_player_dead())
        except Exception:
            health = self._get_health() if current_health is None else current_health
            return health <= 0.0

    def visible_labels(self) -> List[Any]:
        """Return currently visible labeled objects, if the scenario exposes them."""
        try:
            state = self._game.get_state()
            if state is None or getattr(state, "labels", None) is None:
                return []
            return list(state.labels)
        except Exception:
            return []

    def visible_enemy_labels(self) -> List[Any]:
        """Return currently visible hostile labels."""
        return [
            label for label in self.visible_labels()
            if _is_hostile_name(getattr(label, "object_name", ""))
        ]

    def objects_info(self) -> List[Any]:
        """Return world objects when ViZDoom object info is enabled."""
        try:
            state = self._game.get_state()
            if state is None or getattr(state, "objects", None) is None:
                return []
            return list(state.objects)
        except Exception:
            return []

    def enemy_objects(self) -> List[Any]:
        """Return hostile world objects."""
        return [
            obj for obj in self.objects_info()
            if _is_hostile_name(getattr(obj, "name", ""))
        ]

    def player_pose(self) -> Tuple[float, float, float]:
        """Return approximate player (x, y, angle) pose."""
        try:
            x = float(self._game.get_game_variable(vzd.GameVariable.POSITION_X))
            y = float(self._game.get_game_variable(vzd.GameVariable.POSITION_Y))
            angle = float(self._game.get_game_variable(vzd.GameVariable.ANGLE))
            return x, y, angle
        except Exception:
            return 0.0, 0.0, 0.0

    def teacher_guidance(self) -> Dict[str, Any]:
        """Heuristic action plus confidence metadata for teacher shaping."""
        if self.scenario == "deadly_corridor":
            enemy_labels = self.visible_enemy_labels()
            if enemy_labels:
                target = max(enemy_labels, key=lambda label: float(label.width * label.height))
                center_x = float(target.x + target.width * 0.5)
                area = float(target.width * target.height)
                if center_x < VIZDOOM_WIDTH * 0.40:
                    return {
                        "action": MOTOR_TURN_LEFT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.1,
                    }
                if center_x > VIZDOOM_WIDTH * 0.60:
                    return {
                        "action": MOTOR_TURN_RIGHT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.1,
                    }
                if self._corridor_evasive_steps > 0:
                    if area >= 110.0 and self._corridor_evasive_steps % 3 == 0:
                        return {
                            "action": MOTOR_ATTACK,
                            "enemy_visible": True,
                            "attack_window": True,
                            "confidence": 1.5,
                        }
                    return {
                        "action": self._corridor_strafe_dir,
                        "enemy_visible": True,
                        "attack_window": area >= 90.0,
                        "confidence": 1.35,
                    }
                if area < 70.0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.0,
                    }
                if area < 130.0 and self._episode_steps % 3 == 0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.9,
                    }
                if area >= 160.0 and self._episode_steps % 4 == 0:
                    return {
                        "action": self._corridor_strafe_dir,
                        "enemy_visible": True,
                        "attack_window": True,
                        "confidence": 1.1,
                    }
                return {
                    "action": MOTOR_ATTACK,
                    "enemy_visible": True,
                    "attack_window": True,
                    "confidence": 1.7,
                }

            px, py, angle = self.player_pose()
            enemies = self.enemy_objects()
            if enemies:
                def _enemy_key(obj: Any) -> float:
                    dx = float(getattr(obj, "position_x", 0.0)) - px
                    dy = float(getattr(obj, "position_y", 0.0)) - py
                    return dx * dx + dy * dy

                target = min(enemies, key=_enemy_key)
                dx = float(getattr(target, "position_x", 0.0)) - px
                dy = float(getattr(target, "position_y", 0.0)) - py
                distance = math.sqrt(dx * dx + dy * dy)
                target_angle = math.degrees(math.atan2(dy, dx))
                angle_error = _normalize_angle_deg(target_angle - angle)

                if angle_error < -6.0:
                    return {
                        "action": MOTOR_TURN_RIGHT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.0,
                    }
                if angle_error > 6.0:
                    return {
                        "action": MOTOR_TURN_LEFT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.0,
                    }
                if self._corridor_evasive_steps > 0:
                    return {
                        "action": self._corridor_strafe_dir,
                        "enemy_visible": True,
                        "attack_window": distance < 220.0,
                        "confidence": 1.2,
                    }
                if distance > 260.0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.95,
                    }
                if distance > 170.0 and self._episode_steps % 3 == 0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.9,
                    }
                return {
                    "action": MOTOR_ATTACK,
                    "enemy_visible": True,
                    "attack_window": True,
                    "confidence": 1.4,
                }

            if self._corridor_evasive_steps > 0:
                return {
                    "action": self._corridor_strafe_dir,
                    "enemy_visible": False,
                    "attack_window": False,
                    "confidence": 0.7,
                }

            phase = (self._episode_steps // 4) % 4
            if phase in (0, 2):
                return {
                    "action": MOTOR_FORWARD,
                    "enemy_visible": False,
                    "attack_window": False,
                    "confidence": 0.45,
                }
            if phase == 1:
                return {
                    "action": self._corridor_strafe_dir,
                    "enemy_visible": False,
                    "attack_window": False,
                    "confidence": 0.35,
                }
            return {
                "action": MOTOR_TURN_LEFT if (self._episode_steps // 8) % 2 == 0 else MOTOR_TURN_RIGHT,
                "enemy_visible": False,
                "attack_window": False,
                "confidence": 0.25,
            }

        if self.scenario == "defend_the_center":
            enemy_labels = self.visible_enemy_labels()
            if enemy_labels:
                target = max(enemy_labels, key=lambda label: float(label.width * label.height))
                center_x = float(target.x + target.width * 0.5)
                area = float(target.width * target.height)
                if center_x < VIZDOOM_WIDTH * 0.44:
                    return {
                        "action": MOTOR_TURN_LEFT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.0,
                    }
                if center_x > VIZDOOM_WIDTH * 0.56:
                    return {
                        "action": MOTOR_TURN_RIGHT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 1.0,
                    }
                if self.scenario == "deadly_corridor" and area < 140.0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.9,
                    }
                return {
                    "action": MOTOR_ATTACK,
                    "enemy_visible": True,
                    "attack_window": True,
                    "confidence": 1.7,
                }

            px, py, angle = self.player_pose()
            enemies = self.enemy_objects()
            if enemies:
                def _enemy_key(obj: Any) -> float:
                    dx = float(getattr(obj, "position_x", 0.0)) - px
                    dy = float(getattr(obj, "position_y", 0.0)) - py
                    return dx * dx + dy * dy

                target = min(enemies, key=_enemy_key)
                dx = float(getattr(target, "position_x", 0.0)) - px
                dy = float(getattr(target, "position_y", 0.0)) - py
                distance = math.sqrt(dx * dx + dy * dy)
                target_angle = math.degrees(math.atan2(dy, dx))
                angle_error = _normalize_angle_deg(target_angle - angle)

                if angle_error < -8.0:
                    return {
                        "action": MOTOR_TURN_RIGHT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.9,
                    }
                if angle_error > 8.0:
                    return {
                        "action": MOTOR_TURN_LEFT,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.9,
                    }
                if self.scenario == "deadly_corridor" and distance > 220.0:
                    return {
                        "action": MOTOR_FORWARD,
                        "enemy_visible": True,
                        "attack_window": False,
                        "confidence": 0.8,
                    }
                return {
                    "action": MOTOR_ATTACK,
                    "enemy_visible": True,
                    "attack_window": True,
                    "confidence": 1.4,
                }

            sweep = (self._episode_steps // 6) % 2
            return {
                "action": MOTOR_TURN_LEFT if sweep == 0 else MOTOR_TURN_RIGHT,
                "enemy_visible": False,
                "attack_window": False,
                "confidence": 0.0,
            }

        medikits = [
            label for label in self.visible_labels()
            if "medikit" in getattr(label, "object_name", "").lower()
            or "stimpack" in getattr(label, "object_name", "").lower()
            or "health" in getattr(label, "object_name", "").lower()
        ]
        if medikits:
            target = max(medikits, key=lambda label: float(label.width * label.height))
            center_x = float(target.x + target.width * 0.5)
            if center_x < VIZDOOM_WIDTH * 0.42:
                return {
                    "action": MOTOR_TURN_LEFT,
                    "enemy_visible": False,
                    "attack_window": False,
                    "confidence": 1.0,
                }
            if center_x > VIZDOOM_WIDTH * 0.58:
                return {
                    "action": MOTOR_TURN_RIGHT,
                    "enemy_visible": False,
                    "attack_window": False,
                    "confidence": 1.0,
                }
            return {
                "action": MOTOR_FORWARD,
                "enemy_visible": False,
                "attack_window": False,
                "confidence": 1.2,
            }

        _, _, angle = self.player_pose()
        return {
            "action": MOTOR_TURN_LEFT if int(angle // 45) % 2 == 0 else MOTOR_FORWARD,
            "enemy_visible": False,
            "attack_window": False,
            "confidence": 0.0,
        }

    def teacher_action(self) -> int:
        """Heuristic action used for teacher-forced visual pretraining."""
        return int(self.teacher_guidance()["action"])

    def _get_frame(self) -> np.ndarray:
        """Capture and downsample the current frame.

        Returns:
            (48, 64, 3) uint8 RGB array.
        """
        state = self._game.get_state()
        if state is None:
            return np.zeros((RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8)
        buf = state.screen_buffer  # (120, 160, 3) uint8
        # Save full-resolution frame for video
        if hasattr(self, '_frame_buffer'):
            self._frame_buffer.append(buf.copy())
        img = Image.fromarray(buf)
        img = img.resize((RETINA_WIDTH, RETINA_HEIGHT), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    @property
    def is_running(self) -> bool:
        """Whether the game instance is active."""
        return not self._game.is_episode_finished()

    @property
    def episode_health_gained(self) -> float:
        """Total health gained in the current episode."""
        return self._episode_health_gained

    @property
    def episode_kills(self) -> int:
        """Total kills in the current episode."""
        return self._episode_kills

    @property
    def episode_survived(self) -> bool:
        """Whether the player survived the episode horizon."""
        return self._episode_survived

    @property
    def episode_damage_taken(self) -> float:
        """Total damage taken in the current episode."""
        return self._episode_damage_taken

    @property
    def episode_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._episode_steps

    def save_video(self, path: str, fps: int = 30) -> None:
        """Save recorded frames as video.

        Args:
            path: Output video path (.mp4)
            fps: Frames per second
        """
        if not hasattr(self, '_frame_buffer') or not self._frame_buffer:
            print(f"    No frames recorded")
            return
        frames = self._frame_buffer
        if not frames:
            print(f"    No frames recorded")
            return
        if HAS_CV2:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (w, h))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"    Saved video: {path} ({len(frames)} frames)")
        else:
            print(f"    OpenCV not available, saving frames as PNG")
            import os
            os.makedirs(path.replace('.mp4', ''), exist_ok=True)
            for i, frame in enumerate(self._frame_buffer):
                Image.fromarray(frame).save(f"{path.replace('.mp4', '')}/frame_{i:04d}.png")
            print(f"    Saved {len(self._frame_buffer)} frames to {path.replace('.mp4', '')}/")
        self._frame_buffer = []

    def close(self) -> None:
        """Shut down the ViZDoom instance."""
        self._game.close()


# ============================================================================
# Retina-to-Brain Bridge
# ============================================================================

class RetinaBridge:
    """Connects MolecularRetina output (RGC spike IDs) to CUDARegionalBrain
    thalamic relay neurons.

    Maps each RGC neuron to one or more relay neurons using a sparse
    connectivity pattern that preserves retinotopic organization.

    Args:
        retina: The MolecularRetina instance.
        relay_ids: Tensor of thalamic relay neuron IDs.
        device: Torch device for tensor operations.
        seed: Random seed for reproducible mapping.
    """

    def __init__(self, retina: MolecularRetina, relay_ids: torch.Tensor,
                 device: str = "cpu", seed: int = 42):
        self.retina = retina
        self.relay_ids = relay_ids
        self.device = device
        self.n_rgc = retina.n_rgc
        self.n_relay = len(relay_ids)

        # Build RGC-to-relay mapping: each RGC maps to a subset of relay neurons
        rng = np.random.RandomState(seed)
        self._rgc_to_relay: Dict[int, List[int]] = {}
        relay_np = relay_ids.cpu().numpy()
        self._relay_index = {
            int(nid): idx for idx, nid in enumerate(relay_np.tolist())
        }

        for rgc_idx in range(self.n_rgc):
            rgc = retina.rgc_cells[rgc_idx]
            # Map RGC position to relay subpopulation (retinotopic)
            center_idx = int(rgc.x * self.n_relay)
            center_idx = min(center_idx, self.n_relay - 1)
            # Fan-out: each RGC activates 2-5 nearby relay neurons
            fan_out = min(max(2, self.n_relay // max(self.n_rgc, 1) * 2), 5)
            half = fan_out // 2
            start = max(0, center_idx - half)
            end = min(self.n_relay, start + fan_out)
            indices = list(range(start, end))
            self._rgc_to_relay[rgc.neuron_id] = [int(relay_np[i]) for i in indices]

        self._total_injections = 0

    def activation_from_spikes(self, fired_rgc_ids: List[int],
                               intensity: float = 45.0) -> torch.Tensor:
        """Convert fired RGC IDs into a dense relay activation vector."""
        activation = torch.zeros(
            self.n_relay, dtype=torch.float32, device=self.relay_ids.device
        )
        if not fired_rgc_ids:
            return activation

        for rgc_id in fired_rgc_ids:
            if rgc_id not in self._rgc_to_relay:
                continue
            for relay_nid in self._rgc_to_relay[rgc_id]:
                activation[self._relay_index[relay_nid]] += 1.0

        max_val = float(activation.max().item())
        if max_val > 0:
            activation = activation * (float(intensity) / max_val)
        return activation

    def inject_activation(self, brain: CUDAMolecularBrain,
                          activation: torch.Tensor) -> int:
        """Inject a precomputed relay activation vector into the brain."""
        if activation.numel() == 0:
            return 0
        if float(activation.max().item()) <= 0:
            return 0
        brain.external_current[self.relay_ids] += activation.to(
            device=brain.device, dtype=torch.float32
        )
        self._total_injections += 1
        return int((activation > 0).sum().item())

    def inject_spikes(self, brain: CUDAMolecularBrain, fired_rgc_ids: List[int],
                      intensity: float = 45.0) -> int:
        """Convert RGC spikes into thalamic relay currents.

        For each fired RGC, injects current into corresponding relay neurons.
        Uses pulsed injection pattern to avoid depolarization block.

        Args:
            brain: The CUDAMolecularBrain instance.
            fired_rgc_ids: List of RGC neuron IDs that fired.
            intensity: Current injection amplitude (uA/cm^2).

        Returns:
            Number of relay neurons activated.
        """
        activation = self.activation_from_spikes(fired_rgc_ids, intensity=intensity)
        return self.inject_activation(brain, activation)


# ============================================================================
# Motor Decoder
# ============================================================================

class DoomMotorDecoder:
    """5-way motor decoder for Doom actions from L5 cortical spike counts.

    L5 neurons are split into 5 equal populations corresponding to:
    forward, turn_left, turn_right, strafe_left, strafe_right.

    Uses zero-threshold decoding: ANY spike count difference drives action
    selection. This maximizes responsiveness and breaks initial symmetry.

    Args:
        l5_ids: Tensor of all L5 neuron IDs across cortical columns.
    """

    def __init__(self, l5_ids: torch.Tensor):
        n = len(l5_ids)
        pop_size = n // N_MOTOR_POPULATIONS
        self.populations: List[torch.Tensor] = []
        for i in range(N_MOTOR_POPULATIONS):
            start = i * pop_size
            end = start + pop_size if i < N_MOTOR_POPULATIONS - 1 else n
            self.populations.append(l5_ids[start:end])

    def decode(self, brain: CUDAMolecularBrain,
               counts: Optional[List[int]] = None,
               biases: Optional[List[float]] = None) -> Tuple[int, List[int]]:
        """Decode motor action from spike counts.

        Args:
            brain: The brain to read fired neurons from.
            counts: Pre-computed spike counts per population. If None,
                reads from brain.fired.
            biases: Optional additive per-action decoder bias.

        Returns:
            Tuple of (action_index, spike_counts_per_population).
        """
        if counts is None:
            counts = []
            for pop in self.populations:
                counts.append(int(brain.fired[pop].sum().item()))

        total = sum(counts)
        if total == 0:
            if biases is not None:
                best_score = max(biases)
                if best_score > 0:
                    return biases.index(best_score), counts
            # No spikes: random action to explore
            return random.randint(0, N_MOTOR_POPULATIONS - 1), counts

        # Zero-threshold: pick population with most biased spikes
        if biases is not None:
            scores = [float(c) + float(b) for c, b in zip(counts, biases)]
        else:
            scores = [float(c) for c in counts]
        max_count = max(scores)
        action = scores.index(max_count)
        return action, counts


# ============================================================================
# FEP Protocol for Doom
# ============================================================================

class DoomFEPProtocol:
    """Free Energy Principle learning protocol adapted for Doom.

    Positive events (health gain, survival): STRUCTURED pulsed stimulation
    to all cortical neurons. Predictable input creates correlated pre-post
    firing that STDP strengthens. NE boost enhances signal-to-noise.

    Negative events (damage taken): RANDOM noise to 30% of cortical neurons.
    Unpredictable input creates uncorrelated activity, no systematic STDP.

    Hebbian weight nudge directly strengthens relay-to-correct-motor pathways
    on positive events.

    Args:
        cortex_ids: Tensor of all cortical neuron IDs.
        relay_ids: Tensor of thalamic relay neuron IDs.
        l5_ids: Tensor of L5 neuron IDs.
        device: Torch device string.
        structured_steps: Number of steps for structured feedback delivery.
        unstructured_steps: Number of steps for unstructured feedback delivery.
        structured_intensity: Current amplitude for structured pulse (uA/cm^2).
        unstructured_intensity: Current amplitude for random noise (uA/cm^2).
        ne_boost: Norepinephrine boost during structured feedback (nM).
        hebbian_delta: Scale-adaptive Hebbian weight nudge magnitude.
    """

    def __init__(self, cortex_ids: torch.Tensor, relay_ids: torch.Tensor,
                 l5_ids: torch.Tensor, device: str = "cpu",
                 structured_steps: int = 50, unstructured_steps: int = 100,
                 structured_intensity: float = 5.0,
                 unstructured_intensity: float = 5.0,
                 ne_boost: float = 50.0,
                 hebbian_delta: float = 1.5,
                 structured_replay_scale: float = 0.0):
        self.cortex_ids = cortex_ids
        self.relay_ids = relay_ids
        self.l5_ids = l5_ids
        self.device = device
        self.n_cortex = len(cortex_ids)
        self.structured_steps = structured_steps
        self.unstructured_steps = unstructured_steps
        self.structured_intensity = structured_intensity
        self.unstructured_intensity = unstructured_intensity
        self.ne_boost = ne_boost
        self.hebbian_delta = hebbian_delta
        self.structured_replay_scale = structured_replay_scale
        # Track last active motor population for Hebbian nudge
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None
        self.last_activation: Optional[torch.Tensor] = None
        self.last_active_relay_ids: Optional[torch.Tensor] = None
        # ACTION HISTORY for proper credit assignment (RL)
        # Reward the action that led to health (2-3 steps delay)
        self.action_history: List[Tuple[int, int]] = []  # (action, steps_ago)
        self.max_history = 5

    def deliver_positive(self, rb: CUDARegionalBrain) -> None:
        """Structured feedback for positive events (health gain, survival).

        Low entropy: every cortical neuron gets same intensity, same timing.
        Creates correlated activity that STDP strengthens.
        NE boost enhances STDP gain (biologically: locus coeruleus activation).

        Now with MINIMAL cortex disruption: onlyHebbian nudge, no cortex stimulation.
        This allows natural relay->L5 pathway to work while Hebbian learning
        strengthens the correct motor output.
        """
        brain = rb.brain

        if self.ne_boost > 0:
            brain.nt_conc[self.cortex_ids, NT_NE] += self.ne_boost

        replay_activation = None
        if self.last_activation is not None and self.structured_replay_scale > 0:
            replay_activation = self.last_activation * self.structured_replay_scale

        n_steps = max(2, self.structured_steps)
        for s in range(n_steps):
            if replay_activation is not None and s % 2 == 0:
                brain.external_current[self.relay_ids] += replay_activation
            rb.step()

        # ONLY Hebbian nudge on relay->correct motor pathway (no cortex stimulation)
        if self.motor_populations is not None and self.hebbian_delta > 0:
            correct_pop = self.motor_populations[self.last_action]
            wrong_pops = [self.motor_populations[i]
                          for i in range(N_MOTOR_POPULATIONS)
                          if i != self.last_action]
            _doom_hebbian_nudge(brain, self.relay_ids, correct_pop,
                                wrong_pops, self.hebbian_delta,
                                source_ids=self.last_active_relay_ids)

    def deliver_negative(self, rb: CUDARegionalBrain) -> None:
        """Unstructured feedback for negative events (damage taken).

        MINIMAL version: Just let network settle briefly without disrupting
        the natural relay->L5 pathway. TheHebbian nudge on positive events
        will strengthen correct pathways, while negative events do nothing
        (no strengthening of wrong pathways).
        """
        # MINIMAL: Just brief settling, no random cortex stimulation
        for _ in range(3):
            rb.step()

    def deliver_survival_reward(self, rb: CUDARegionalBrain) -> None:
        """Treat survival as another structured positive event."""
        self.deliver_positive(rb)


class DoomDAProtocol:
    """Standard dopamine reward protocol for comparison with FEP.

    Positive events: DA release at L5 motor neurons.
    Negative events: no DA, network settles.
    """

    def __init__(self, cortex_ids: torch.Tensor, l5_ids: torch.Tensor,
                 device: str = "cpu",
                 reward_steps: int = 50, da_amount: float = 50.0,
                 settle_steps: int = 15):
        self.cortex_ids = cortex_ids
        self.l5_ids = l5_ids
        self.device = device
        self.reward_steps = reward_steps
        self.da_amount = da_amount
        self.settle_steps = settle_steps
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None
        self.last_active_relay_ids: Optional[torch.Tensor] = None

    def deliver_positive(self, rb: CUDARegionalBrain) -> None:
        """DA reward at L5 motor neurons."""
        brain = rb.brain
        for s in range(self.reward_steps):
            if s % 3 == 0:
                brain.nt_conc[self.l5_ids, NT_DA] += self.da_amount
            if s % 2 == 0:
                brain.external_current[self.cortex_ids] += 5.0
            rb.step()

    def deliver_negative(self, rb: CUDARegionalBrain) -> None:
        """No reward — let the network settle."""
        rb.run(self.settle_steps)


class DoomRandomProtocol:
    """Control: identical feedback regardless of outcome."""

    def __init__(self, cortex_ids: torch.Tensor, device: str = "cpu",
                 settle_steps: int = 15):
        self.cortex_ids = cortex_ids
        self.device = device
        self.settle_steps = settle_steps
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None
        self.last_active_relay_ids: Optional[torch.Tensor] = None

    def deliver_positive(self, rb: CUDARegionalBrain) -> None:
        """Same as negative — no differential feedback."""
        rb.run(self.settle_steps)

    def deliver_negative(self, rb: CUDARegionalBrain) -> None:
        """Same as positive — no differential feedback."""
        rb.run(self.settle_steps)


class DoomRLProtocol:
    """Reinforcement Learning with proper dopamine/cortisol signals.

    This implements CLASSIC RL, not FEP:
    - Health collected → Dopamine burst at motor neurons → strengthen that action
    - Damage taken → Cortisol (stress) → weaken that action
    - Credit assignment: reward the action that occurred 2-3 steps ago

    This is what actual brains do: reward prediction error drives learning.
    """

    def __init__(self, cortex_ids: torch.Tensor, relay_ids: torch.Tensor,
                 l5_ids: torch.Tensor, device: str = "cpu",
                 da_amount: float = 200.0,  # Strong dopamine
                 cortisol_amount: float = 150.0,  # Stress signal
                 reward_steps: int = 10,
                 settle_steps: int = 5):
        self.cortex_ids = cortex_ids
        self.relay_ids = relay_ids
        self.l5_ids = l5_ids
        self.device = device
        self.da_amount = da_amount
        self.cortisol_amount = cortisol_amount
        self.reward_steps = reward_steps
        self.settle_steps = settle_steps
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None
        self.last_active_relay_ids: Optional[torch.Tensor] = None

        # ACTION HISTORY for temporal credit assignment
        # Track (action, steps_since) - reward the action that led to outcome
        self.action_history: List[Tuple[int, int]] = []
        self.max_history = 4  # How far back to credit

    def record_action(self, action: int) -> None:
        """Record action and increment step counters."""
        # Add new action with 0 steps
        self.action_history.insert(0, (action, 0))
        # Increment step counter for all historical actions
        self.action_history = [(a, s+1) for a, s in self.action_history]
        # Keep only recent history
        self.action_history = [(a, s) for a, s in self.action_history if s <= self.max_history]

    def deliver_positive(self, rb: CUDARegionalBrain) -> None:
        """DOPAMINE burst for health collected!

        Credit assignment: reward the action from 2-3 steps ago that led to health.
        This is how real brains learn - reward prediction error.
        """
        brain = rb.brain

        # Credit assignment: find action from 2-3 steps ago
        rewarded_actions = []
        for action, steps_ago in self.action_history:
            if 1 <= steps_ago <= 3:  # Reward delayed actions
                rewarded_actions.append(action)

        if rewarded_actions and self.motor_populations:
            # Apply dopamine to the ACTIONS that led to reward
            # (not just the last action!)
            unique_actions = list(set(rewarded_actions))
            for action in unique_actions:
                count = rewarded_actions.count(action)
                pop = self.motor_populations[action]
                # Dopamine at motor neurons strengthens that action pathway
                brain.nt_conc[pop, NT_DA] += self.da_amount * (count / len(rewarded_actions))

        # Also briefly stimulate to mark the event
        for _ in range(self.reward_steps):
            rb.step()

        # Hebbian nudge to strengthen relay->correct motor
        if self.motor_populations and rewarded_actions:
            # Strengthen the credited action(s)
            for action in unique_actions:
                correct_pop = self.motor_populations[action]
                wrong_pops = [self.motor_populations[i]
                              for i in range(N_MOTOR_POPULATIONS)
                              if i != action]
                _doom_hebbian_nudge(brain, self.relay_ids, correct_pop,
                                    wrong_pops, 2.0,
                                    source_ids=self.last_active_relay_ids)  # Stronger nudge for RL

    def deliver_negative(self, rb: CUDARegionalBrain) -> None:
        """CORTISOL (stress) for damage taken!

        Weaken the action that led to damage.
        """
        brain = rb.brain

        # Credit assignment: find action from 2-3 steps ago
        punished_actions = []
        for action, steps_ago in self.action_history:
            if 1 <= steps_ago <= 3:
                punished_actions.append(action)

        if punished_actions and self.motor_populations:
            unique_actions = list(set(punished_actions))
            for action in unique_actions:
                count = punished_actions.count(action)
                pop = self.motor_populations[action]
                # Cortisol weakens synapses - reduce motor output for that action
                brain.nt_conc[pop, NT_5HT] += self.cortisol_amount * (count / len(punished_actions))

        # Brief settling
        for _ in range(self.settle_steps):
            rb.step()

    def deliver_kill_reward(self, rb: CUDARegionalBrain) -> None:
        """HUGE DOPAMINE for killing an enemy!

        This is the key for combat - brain learns to shoot enemies.
        """
        brain = rb.brain

        # Credit assignment: find attack action from 2-4 steps ago
        killer_actions = []
        for action, steps_ago in self.action_history:
            if 1 <= steps_ago <= 4 and action == 5:  # action 5 = attack
                killer_actions.append(action)

        if killer_actions and self.motor_populations:
            # BIG dopamine for the kill!
            pop = self.motor_populations[5]  # Attack
            brain.nt_conc[pop, NT_DA] += self.da_amount * 2.0  # Double dopamine for kills!

        # Brief stimulation
        for _ in range(3):
            rb.step()

    def deliver_miss_punishment(self, rb: CUDARegionalBrain) -> None:
        """CORTISOL for missing a shot!

        Attack but didn't kill - punish that action.
        """
        brain = rb.brain

        # Find recent attack actions
        for action, steps_ago in self.action_history:
            if 1 <= steps_ago <= 2 and action == 5 and self.motor_populations:
                pop = self.motor_populations[5]  # Attack
                brain.nt_conc[pop, NT_5HT] += self.cortisol_amount * 0.5
                break

        for _ in range(2):
            rb.step()


# ============================================================================
# Hebbian Weight Nudge
# ============================================================================

def _doom_hebbian_nudge(brain: CUDAMolecularBrain, relay_ids: torch.Tensor,
                        correct_pop: torch.Tensor,
                        wrong_pops: List[torch.Tensor],
                        delta: float = 0.5,
                        source_ids: Optional[torch.Tensor] = None) -> None:
    """Hebbian weight update for Doom motor populations.

    Strengthens relay->correct_motor synapses, weakens relay->wrong_motor.
    This provides targeted credit assignment to accelerate FEP learning.

    Args:
        brain: The molecular brain instance.
        relay_ids: Thalamic relay neuron IDs.
        correct_pop: Neuron IDs of the correct motor population.
        wrong_pops: List of neuron ID tensors for incorrect motor populations.
        delta: Weight update magnitude.
        source_ids: Optional relay subset to credit instead of all relays.
    """
    if brain.n_synapses == 0:
        return

    if source_ids is None:
        source_ids = relay_ids

    relay_set = set(source_ids.cpu().tolist())
    correct_set = set(correct_pop.cpu().tolist())

    pre_np = brain.syn_pre.cpu().numpy()
    post_np = brain.syn_post.cpu().numpy()
    relay_mask = np.isin(pre_np, list(relay_set))

    # Strengthen: relay -> correct motor
    correct_post_mask = np.isin(post_np, list(correct_set))
    strengthen_mask = relay_mask & correct_post_mask
    if strengthen_mask.any():
        idx = torch.tensor(np.where(strengthen_mask)[0], device=brain.device)
        brain.syn_strength[idx] = torch.clamp(
            brain.syn_strength[idx] + delta, 0.3, 8.0)

    # Weaken: relay -> wrong motors
    for wrong_pop in wrong_pops:
        wrong_set = set(wrong_pop.cpu().tolist())
        wrong_post_mask = np.isin(post_np, list(wrong_set))
        weaken_mask = relay_mask & wrong_post_mask
        if weaken_mask.any():
            idx = torch.tensor(np.where(weaken_mask)[0], device=brain.device)
            brain.syn_strength[idx] = torch.clamp(
                brain.syn_strength[idx] - delta * 0.15, 0.3, 8.0)

    brain._W_dirty = True
    brain._W_sparse = None
    brain._NT_W_sparse = None


def _present_doom_stimulus(
    rb: CUDARegionalBrain,
    bridge: RetinaBridge,
    decoder: DoomMotorDecoder,
    activation: torch.Tensor,
    stim_steps: int = 20,
    teacher_motor_ids: Optional[torch.Tensor] = None,
    teacher_motor_intensity: float = 0.0,
) -> List[int]:
    """Present one retinal activation pattern and return motor spike counts."""
    brain = rb.brain
    motor_acc = torch.zeros(N_MOTOR_POPULATIONS, device=brain.device)
    for s in range(stim_steps):
        if s % 2 == 0:
            bridge.inject_activation(brain, activation)
            if teacher_motor_ids is not None and teacher_motor_intensity > 0:
                brain.external_current[teacher_motor_ids] += teacher_motor_intensity
        rb.step()
        for pop_idx, pop_ids in enumerate(decoder.populations):
            motor_acc[pop_idx] += brain.fired[pop_ids].sum()
    return motor_acc.int().tolist()


def _apply_teacher_action_shaping(
    brain: CUDAMolecularBrain,
    decoder: DoomMotorDecoder,
    relay_ids: torch.Tensor,
    activation: torch.Tensor,
    target_action: int,
    delta: float,
) -> None:
    """Apply a targeted relay-to-motor nudge for the teacher-indicated action."""
    if delta <= 0:
        return
    source_ids = _active_relay_source_ids(relay_ids, activation)
    wrong_pops = [
        decoder.populations[i]
        for i in range(N_MOTOR_POPULATIONS)
        if i != target_action
    ]
    _doom_hebbian_nudge(
        brain,
        relay_ids,
        decoder.populations[target_action],
        wrong_pops,
        delta=delta,
        source_ids=source_ids,
    )


def _apply_motor_penalty(
    brain: CUDAMolecularBrain,
    motor_pop: torch.Tensor,
    source_ids: torch.Tensor,
    delta: float,
) -> None:
    """Weaken source->motor synapses for a specific motor population."""
    if delta <= 0 or brain.n_synapses == 0 or source_ids.numel() == 0:
        return

    source_set = set(source_ids.cpu().tolist())
    motor_set = set(motor_pop.cpu().tolist())
    pre_np = brain.syn_pre.cpu().numpy()
    post_np = brain.syn_post.cpu().numpy()
    weaken_mask = np.isin(pre_np, list(source_set)) & np.isin(post_np, list(motor_set))
    if weaken_mask.any():
        idx = torch.tensor(np.where(weaken_mask)[0], device=brain.device)
        brain.syn_strength[idx] = torch.clamp(
            brain.syn_strength[idx] - delta, 0.3, 8.0
        )
        brain._W_dirty = True
        brain._W_sparse = None
        brain._NT_W_sparse = None


def _combat_decoder_biases(
    guidance: Dict[str, Any],
    attack_bonus: float,
    attack_penalty: float,
) -> List[float]:
    """Return decoder biases for combat attack selection."""
    biases = [0.0] * N_MOTOR_POPULATIONS
    if guidance["attack_window"]:
        biases[MOTOR_ATTACK] += float(attack_bonus)
    else:
        biases[MOTOR_ATTACK] -= float(attack_penalty)
    return biases


def _apply_doom_event_feedback(
    rb: CUDARegionalBrain,
    protocol,
    event: str,
    action: int,
    neutral_steps: int,
) -> Tuple[int, int]:
    """Apply event feedback and return (positive_count, negative_count)."""
    positive = 0
    negative = 0

    if event == "health_gained":
        protocol.deliver_positive(rb)
        positive += 1
    elif event == "kill":
        if hasattr(protocol, 'deliver_kill_reward'):
            protocol.deliver_kill_reward(rb)
        protocol.deliver_positive(rb)
        positive += 5
    elif event == "survived":
        if hasattr(protocol, 'deliver_survival_reward'):
            protocol.deliver_survival_reward(rb)
        else:
            protocol.deliver_positive(rb)
        positive += 1
    elif event == "damage_taken":
        protocol.deliver_negative(rb)
        negative += 1
    elif event == "episode_end":
        protocol.deliver_negative(rb)
        negative += 10
    else:
        if action == MOTOR_ATTACK and hasattr(protocol, 'deliver_miss_punishment'):
            if np.random.random() < 0.3:
                protocol.deliver_miss_punishment(rb)
        rb.run(neutral_steps)

    return positive, negative


# ============================================================================
# Doom Game Loop
# ============================================================================

def play_doom_episode(
    rb: CUDARegionalBrain,
    game: DoomGame,
    retina: MolecularRetina,
    bridge: RetinaBridge,
    decoder: DoomMotorDecoder,
    protocol,
    relay_ids: torch.Tensor,
    stim_steps: int = 20,
    max_game_steps: int = 500,
    neutral_steps: int = 5,
    combat_teacher_shaping_delta: float = 0.0,
    combat_attack_window_delta: float = 0.0,
    combat_attack_miss_delta: float = 0.0,
    combat_decoder_attack_bonus: float = 0.0,
    combat_decoder_attack_penalty: float = 0.0,
    record_video: bool = False,
    video_path: str = None,
) -> Dict[str, Any]:
    """Play one Doom episode through the molecular retina pipeline.

    Full loop: ViZDoom frame -> retina spikes -> relay injection ->
    cortical processing -> motor decode -> game action -> FEP feedback.

    Args:
        rb: The regional brain.
        game: The DoomGame wrapper.
        retina: Molecular retina for frame processing.
        bridge: Retina-to-brain bridge.
        decoder: Motor decoder for L5 readout.
        protocol: Learning protocol (FEP, DA, or Random).
        relay_ids: Thalamic relay neuron IDs.
        stim_steps: Neural processing steps per game frame.
        max_game_steps: Maximum steps before forced episode end.

    Returns:
        Dict with episode metrics: health_gained, kills, survived,
        damage_taken, steps, action counts, and event counts.
    """
    brain = rb.brain
    frame = game.new_episode()
    retina.reset()

    total_positive = 0
    total_negative = 0
    action_counts = [0] * N_MOTOR_POPULATIONS
    step_count = 0
    teacher_match_steps = 0
    teacher_match_hits = 0
    attack_window_steps = 0
    attack_window_shots = 0
    missed_attack_windows = 0
    blind_attack_shots = 0

    while game.is_running and step_count < max_game_steps:
        # 1. Process frame through molecular retina (pixel -> RGC spikes)
        fired_rgc_ids = retina.process_frame(frame, n_steps=5)
        activation = bridge.activation_from_spikes(fired_rgc_ids, intensity=45.0)
        active_relay_ids = _active_relay_source_ids(relay_ids, activation)
        guidance = game.teacher_guidance()

        # 2. Present retinal activation and decode motor action
        counts = _present_doom_stimulus(
            rb, bridge, decoder, activation, stim_steps=stim_steps
        )
        decoder_biases = None
        if combat_decoder_attack_bonus > 0 or combat_decoder_attack_penalty > 0:
            decoder_biases = _combat_decoder_biases(
                guidance,
                attack_bonus=combat_decoder_attack_bonus,
                attack_penalty=combat_decoder_attack_penalty,
            )
        action, _ = decoder.decode(brain, counts=counts, biases=decoder_biases)

        if guidance["enemy_visible"]:
            teacher_match_steps += 1
            if action == guidance["action"]:
                teacher_match_hits += 1
            if combat_teacher_shaping_delta > 0:
                shaped_delta = combat_teacher_shaping_delta * max(
                    1.0, float(guidance["confidence"])
                )
                _apply_teacher_action_shaping(
                    brain,
                    decoder,
                    relay_ids,
                    activation,
                    int(guidance["action"]),
                    delta=shaped_delta,
                )

        if guidance["enemy_visible"] and guidance["attack_window"]:
            attack_window_steps += 1
            if action == MOTOR_ATTACK:
                attack_window_shots += 1
                if combat_attack_window_delta > 0:
                    _apply_teacher_action_shaping(
                        brain,
                        decoder,
                        relay_ids,
                        activation,
                        MOTOR_ATTACK,
                        delta=combat_attack_window_delta * max(
                            1.0, float(guidance["confidence"])
                        ),
                    )
            else:
                missed_attack_windows += 1
        elif action == MOTOR_ATTACK:
            blind_attack_shots += 1
            if combat_attack_miss_delta > 0:
                _apply_motor_penalty(
                    brain,
                    decoder.populations[MOTOR_ATTACK],
                    active_relay_ids,
                    delta=combat_attack_miss_delta,
                )

        # Track which action the protocol should credit
        if hasattr(protocol, 'last_action'):
            protocol.last_action = action
        if hasattr(protocol, 'last_activation'):
            protocol.last_activation = activation
        if hasattr(protocol, 'last_active_relay_ids'):
            protocol.last_active_relay_ids = active_relay_ids

        # FOR RL: Record action in history for temporal credit assignment
        if hasattr(protocol, 'record_action'):
            protocol.record_action(action)

        action_counts[action] += 1

        # 5. Execute action in ViZDoom
        event, health_delta, done, frame = game.step(action)
        step_count += 1

        # 6. Deliver feedback based on game event
        pos_delta, neg_delta = _apply_doom_event_feedback(
            rb, protocol, event, action, neutral_steps
        )
        total_positive += pos_delta
        total_negative += neg_delta

        if done:
            break

    if step_count >= max_game_steps and game.is_running and game.episode_survived:
        if hasattr(protocol, 'deliver_survival_reward'):
            protocol.deliver_survival_reward(rb)
        else:
            protocol.deliver_positive(rb)
        total_positive += 1

    # Save video if requested
    if record_video and video_path:
        game.save_video(video_path)

    return {
        "health_gained": game.episode_health_gained,
        "kills": game.episode_kills,
        "survived": float(game.episode_survived),
        "damage_taken": game.episode_damage_taken,
        "steps": game.episode_steps,
        "positive_events": total_positive,
        "negative_events": total_negative,
        "action_counts": action_counts,
        "teacher_match_rate": (
            teacher_match_hits / max(1, teacher_match_steps)
        ),
        "attack_window_fire_rate": (
            attack_window_shots / max(1, attack_window_steps)
        ),
        "missed_attack_windows": missed_attack_windows,
        "blind_attack_shots": blind_attack_shots,
    }


def teacher_forced_doom_episode(
    rb: CUDARegionalBrain,
    game: DoomGame,
    retina: MolecularRetina,
    bridge: RetinaBridge,
    decoder: DoomMotorDecoder,
    protocol,
    relay_ids: torch.Tensor,
    stim_steps: int = 20,
    max_game_steps: int = 500,
    neutral_steps: int = 5,
    teacher_motor_intensity: float = 30.0,
    teacher_hebbian_delta: float = 1.2,
) -> Dict[str, Any]:
    """Teacher-guided visual warmup before autonomous Doom play."""
    brain = rb.brain
    frame = game.new_episode()
    retina.reset()

    total_positive = 0
    total_negative = 0
    step_count = 0
    action_counts = [0] * N_MOTOR_POPULATIONS
    teacher_match_steps = 0
    teacher_match_hits = 0

    while game.is_running and step_count < max_game_steps:
        fired_rgc_ids = retina.process_frame(frame, n_steps=5)
        activation = bridge.activation_from_spikes(fired_rgc_ids, intensity=45.0)
        guidance = game.teacher_guidance()
        teacher_action = int(guidance["action"])
        teacher_motor_ids = decoder.populations[teacher_action]
        active_relay_ids = _active_relay_source_ids(relay_ids, activation)
        motor_intensity = teacher_motor_intensity * max(1.0, float(guidance["confidence"]))

        counts = _present_doom_stimulus(
            rb,
            bridge,
            decoder,
            activation,
            stim_steps=stim_steps,
            teacher_motor_ids=teacher_motor_ids,
            teacher_motor_intensity=motor_intensity,
        )
        decoded_action, _ = decoder.decode(brain, counts=counts)
        teacher_match_steps += 1
        if decoded_action == teacher_action:
            teacher_match_hits += 1

        if hasattr(protocol, 'last_action'):
            protocol.last_action = teacher_action
        if hasattr(protocol, 'last_activation'):
            protocol.last_activation = activation
        if hasattr(protocol, 'last_active_relay_ids'):
            protocol.last_active_relay_ids = active_relay_ids
        if hasattr(protocol, 'record_action'):
            protocol.record_action(teacher_action)

        if teacher_hebbian_delta > 0:
            shaped_delta = teacher_hebbian_delta * max(1.0, float(guidance["confidence"]))
            _apply_teacher_action_shaping(
                brain,
                decoder,
                relay_ids,
                activation,
                teacher_action,
                delta=shaped_delta,
            )

        action_counts[teacher_action] += 1
        event, _, done, frame = game.step(teacher_action)
        step_count += 1
        pos_delta, neg_delta = _apply_doom_event_feedback(
            rb, protocol, event, teacher_action, neutral_steps
        )
        total_positive += pos_delta
        total_negative += neg_delta

        if done:
            break

    if step_count >= max_game_steps and game.is_running and game.episode_survived:
        if hasattr(protocol, 'deliver_survival_reward'):
            protocol.deliver_survival_reward(rb)
        else:
            protocol.deliver_positive(rb)
        total_positive += 1

    return {
        "health_gained": game.episode_health_gained,
        "kills": game.episode_kills,
        "survived": float(game.episode_survived),
        "damage_taken": game.episode_damage_taken,
        "steps": game.episode_steps,
        "positive_events": total_positive,
        "negative_events": total_negative,
        "action_counts": action_counts,
        "teacher_match_rate": (
            teacher_match_hits / max(1, teacher_match_steps)
        ),
    }


# ============================================================================
# Brain + Retina Setup Helper
# ============================================================================

def _build_doom_brain(
    scale: str, device: str, seed: int,
) -> Tuple[CUDARegionalBrain, MolecularRetina, RetinaBridge,
           DoomMotorDecoder, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build complete neural architecture for Doom: brain + retina + bridge.

    Args:
        scale: Network scale ("small", "medium", "large").
        device: Torch device ("auto", "cuda", "mps", "cpu").
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (regional_brain, retina, bridge, motor_decoder,
                  relay_ids, l5_ids, cortex_ids).
    """
    n_cols = SCALE_COLUMNS.get(scale, 10)
    rb = CUDARegionalBrain._build(n_columns=n_cols, n_per_layer=20,
                                   device=device, seed=seed)
    brain = rb.brain
    dev = brain.device
    if dev.type == 'cuda':
        brain.compile()

    relay_ids = _get_region_ids(rb, "thalamus", "relay")
    l5_ids = _get_cortex_l5_ids(rb)
    cortex_ids = _get_all_cortex_ids(rb)

    # Build molecular retina
    retina = MolecularRetina(
        resolution=(RETINA_WIDTH, RETINA_HEIGHT),
        fovea_ratio=0.3,
        device="cpu",  # retina runs on CPU (numpy-based HH)
        seed=seed,
    )

    # Build retina-to-brain bridge
    bridge = RetinaBridge(retina, relay_ids, device=str(dev), seed=seed)

    # Motor decoder from L5
    decoder = DoomMotorDecoder(l5_ids)

    return rb, retina, bridge, decoder, relay_ids, l5_ids, cortex_ids


# ============================================================================
# Experiment 1: Doom Navigation (Health Gathering)
# ============================================================================

def exp_doom_navigation(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = None,
    scenario: str = "health_gathering",
    record_video: bool = False,
    pretrain_episodes: int = 0,
    structured_replay_scale: float = 0.0,
    teacher_motor_intensity: float = 30.0,
    teacher_hebbian_delta: float = 1.2,
    combat_teacher_shaping_delta: float = 0.0,
    combat_attack_window_delta: float = 0.0,
    combat_attack_miss_delta: float = 0.0,
    combat_decoder_attack_bonus: float = 0.0,
    combat_decoder_attack_penalty: float = 0.0,
    metric_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Can a dONN learn to gather health in Doom via the free energy principle?

    Tracks the scenario's configured positive metric per episode. Learning is
    evidenced by improvement in that metric over time as the FEP protocol
    strengthens sensorimotor pathways through structured feedback.

    Args:
        scale: Network scale.
        device: Torch device.
        seed: Random seed.
        n_episodes: Number of episodes to run (None = scale default).
        scenario: ViZDoom scenario to use.

    Returns:
        Dict with pass/fail status, per-episode metrics, and timing.
    """
    sp = SCALE_PARAMS.get(scale, SCALE_PARAMS["large"])
    if n_episodes is None:
        n_episodes = sp["n_episodes"]
    metric_name = metric_override or _scenario_positive_metric(scenario)
    metric_label = _metric_label(metric_name)

    _header(
        f"Exp 1: Doom Navigation ({scenario})",
        "Free energy principle — structured vs unstructured feedback, NO reward"
    )
    t0 = time.perf_counter()

    # Build brain + retina
    rb, retina, bridge, decoder, relay_ids, l5_ids, cortex_ids = \
        _build_doom_brain(scale, device, seed)
    brain = rb.brain
    dev = brain.device
    print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")
    print(f"    Retina: {retina.n_rgc} RGCs, {retina.total_neurons} total retinal neurons")

    # Scale-adaptive Hebbian delta
    n_l5 = len(l5_ids)
    delta = 0.8 * max(1.0, (n_l5 / 200) ** 0.3)

    # Create FEP protocol with scale-appropriate step counts
    protocol = DoomFEPProtocol(
        cortex_ids, relay_ids, l5_ids, device=dev,
        structured_steps=sp["structured_steps"],
        unstructured_steps=sp["unstructured_steps"],
        structured_intensity=40.0, unstructured_intensity=40.0,
        hebbian_delta=delta,
        structured_replay_scale=structured_replay_scale,
    )
    protocol.motor_populations = decoder.populations

    # Warmup brain
    _warmup(rb, n_steps=sp["warmup_steps"])
    print(f"    Warmup complete")

    # Game
    game = DoomGame(scenario=scenario, seed=seed, visible=False)

    pretrain_metrics = []
    if pretrain_episodes > 0:
        print(f"    Teacher pretraining: {pretrain_episodes} episodes")
        for _ in range(pretrain_episodes):
            metrics = teacher_forced_doom_episode(
                rb, game, retina, bridge, decoder, protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
                teacher_motor_intensity=teacher_motor_intensity,
                teacher_hebbian_delta=teacher_hebbian_delta,
            )
            pretrain_metrics.append(metrics)
        avg_teacher = sum(m["teacher_match_rate"] for m in pretrain_metrics) / len(pretrain_metrics)
        avg_pre_metric = sum(_episode_metric(m, metric_name) for m in pretrain_metrics) / len(pretrain_metrics)
        print(f"    Teacher warmup complete: match {avg_teacher:.0%}, "
              f"{metric_label} {_format_metric_value(metric_name, avg_pre_metric)}")

    # Play episodes
    report_interval = max(1, n_episodes // 5)
    episode_metrics = []
    for ep in range(n_episodes):
        # Record first episode video if requested
        record_this = (record_video and ep == 0)
        video_path = _default_video_path(
            f"doom_exp1_{scenario}_ep{ep+1}.mp4"
        ) if record_this else None
        metrics = play_doom_episode(
            rb, game, retina, bridge, decoder, protocol,
            relay_ids, stim_steps=sp["stim_steps"],
            max_game_steps=sp["max_game_steps"],
            neutral_steps=sp["neutral_steps"],
            combat_teacher_shaping_delta=combat_teacher_shaping_delta,
            combat_attack_window_delta=combat_attack_window_delta,
            combat_attack_miss_delta=combat_attack_miss_delta,
            combat_decoder_attack_bonus=combat_decoder_attack_bonus,
            combat_decoder_attack_penalty=combat_decoder_attack_penalty,
            record_video=record_this, video_path=video_path)
        episode_metrics.append(metrics)

        if (ep + 1) % report_interval == 0 or ep == n_episodes - 1:
            recent = episode_metrics[max(0, ep - report_interval + 1):ep + 1]
            avg_metric = sum(_episode_metric(m, metric_name) for m in recent) / len(recent)
            avg_damage = sum(m["damage_taken"] for m in recent) / len(recent)
            print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                  f"{metric_label} {_format_metric_value(metric_name, avg_metric)}, "
                  f"damage -{avg_damage:.0f} "
                  f"(last {len(recent)})")

    game.close()

    # Analyze results
    metric_per_ep = [_episode_metric(m, metric_name) for m in episode_metrics]
    quarter = max(1, n_episodes // 4)
    first_q = metric_per_ep[:quarter]
    last_q = metric_per_ep[-quarter:]
    first_avg = sum(first_q) / len(first_q) if first_q else 0
    last_avg = sum(last_q) / len(last_q) if last_q else 0
    total_metric = sum(metric_per_ep)

    elapsed = time.perf_counter() - t0

    # Pass: positive metric improves over episodes OR shows non-zero success.
    passed = (last_avg > first_avg) or (total_metric > 0)

    print(f"\n    Results:")
    print(f"    First quarter avg {metric_label}: {first_avg:.1f}")
    print(f"    Last quarter avg {metric_label}:  {last_avg:.1f}")
    print(f"    Total {metric_label}:             {total_metric:.1f}")
    print(f"    Improvement:              {last_avg - first_avg:+.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "metric_name": metric_name,
        "first_q_avg": first_avg,
        "last_q_avg": last_avg,
        "total_metric": total_metric,
        "pretrain_episodes": pretrain_episodes,
        "combat_teacher_shaping_delta": combat_teacher_shaping_delta,
        "combat_attack_window_delta": combat_attack_window_delta,
        "combat_attack_miss_delta": combat_attack_miss_delta,
        "combat_decoder_attack_bonus": combat_decoder_attack_bonus,
        "combat_decoder_attack_penalty": combat_decoder_attack_penalty,
        "pretrain_metrics": pretrain_metrics,
        "episode_metrics": episode_metrics,
    }


# ============================================================================
# Experiment 2: Learning Speed Comparison
# ============================================================================

def exp_learning_speed(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_episodes: int = None,
    scenario: str = "health_gathering",
    record_video: bool = False,
) -> Dict[str, Any]:
    """Compare free energy vs DA reward vs random protocols on Doom.

    Each protocol trains an identical brain (same seed) and plays the same
    scenario. Metrics use that scenario's configured positive metric.

    Args:
        scale: Network scale.
        device: Torch device.
        seed: Random seed.
        n_episodes: Episodes per protocol (None = scale default).
        scenario: ViZDoom scenario name.

    Returns:
        Dict with per-protocol results and pass/fail status.
    """
    sp = SCALE_PARAMS.get(scale, SCALE_PARAMS["large"])
    if n_episodes is None:
        n_episodes = sp["n_episodes"]
    metric_name = _scenario_positive_metric(scenario)
    metric_label = _metric_label(metric_name)

    _header(
        "Exp 2: Learning Speed Comparison",
        "Free energy vs DA reward vs random — which learns fastest in Doom?"
    )
    t0 = time.perf_counter()

    conditions = ["free_energy", "da_reward", "random", "rl"]
    all_results = {}

    for condition in conditions:
        print(f"\n    --- {condition} ---")

        # Build fresh brain (same seed for fair comparison)
        rb, retina, bridge, decoder, relay_ids, l5_ids, cortex_ids = \
            _build_doom_brain(scale, device, seed)
        brain = rb.brain
        dev = brain.device

        n_l5 = len(l5_ids)
        delta = 5.0 * max(1.0, (n_l5 / 200) ** 0.3)

        # Create protocol with scale-appropriate step counts
        if condition == "free_energy":
            protocol = DoomFEPProtocol(
                cortex_ids, relay_ids, l5_ids, device=dev,
                structured_steps=sp["structured_steps"],
                unstructured_steps=sp["unstructured_steps"],
                hebbian_delta=delta)
            protocol.motor_populations = decoder.populations
        elif condition == "da_reward":
            protocol = DoomDAProtocol(cortex_ids, l5_ids, device=dev,
                                       reward_steps=sp["structured_steps"],
                                       settle_steps=sp["neutral_steps"] * 3)
            protocol.motor_populations = decoder.populations
        elif condition == "rl":
            # RL: Dopamine for health, Cortisol for damage
            protocol = DoomRLProtocol(
                cortex_ids, relay_ids, l5_ids, device=dev,
                da_amount=200.0, cortisol_amount=150.0)
            protocol.motor_populations = decoder.populations
        else:
            protocol = DoomRandomProtocol(cortex_ids, device=dev,
                                           settle_steps=sp["neutral_steps"] * 3)
            protocol.motor_populations = decoder.populations

        _warmup(rb, n_steps=sp["warmup_steps"])

        game = DoomGame(scenario=scenario, seed=seed, visible=False)

        episode_metrics = []
        for ep in range(n_episodes):
            record_this = (record_video and ep == 0)
            video_path = _default_video_path(
                f"doom_exp2_{condition}_ep{ep+1}.mp4"
            ) if record_this else None
            metrics = play_doom_episode(
                rb, game, retina, bridge, decoder, protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
                record_video=record_this, video_path=video_path)
            episode_metrics.append(metrics)

        game.close()

        total_metric = sum(_episode_metric(m, metric_name) for m in episode_metrics)
        last_q = episode_metrics[-(n_episodes // 4):]
        last_q_metric = sum(_episode_metric(m, metric_name) for m in last_q) / len(last_q)

        all_results[condition] = {
            "total_metric": total_metric,
            "last_q_avg": last_q_metric,
            "episode_metrics": episode_metrics,
        }
        print(f"    {condition:15s}: total {metric_label} = {total_metric:.1f}, "
              f"last quarter avg = {last_q_metric:.1f}")

    elapsed = time.perf_counter() - t0

    # Pass: FEP or DA outperforms random
    fe_total = all_results["free_energy"]["total_metric"]
    da_total = all_results["da_reward"]["total_metric"]
    rand_total = all_results["random"]["total_metric"]
    passed = (fe_total > rand_total or da_total > rand_total)

    print(f"\n    Free Energy: {fe_total:.1f} total {metric_label}")
    print(f"    DA Reward:   {da_total:.1f} total {metric_label}")
    print(f"    Random:      {rand_total:.1f} total {metric_label}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "metric_name": metric_name,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "episode_metrics"}
                    for k, v in all_results.items()},
        "all_results": all_results,
    }


# ============================================================================
# Experiment 3: Pharmacological Effects on Doom
# ============================================================================

def exp_pharmacology(
    scale: str = "small",
    device: str = "auto",
    seed: int = 42,
    n_train_episodes: int = None,
    n_test_episodes: int = None,
    scenario: str = "health_gathering",
    record_video: bool = False,
) -> Dict[str, Any]:
    """Drug effects on Doom performance — IMPOSSIBLE on real DishBrain tissue.

    Train 3 identical brains, then apply drugs before testing:
    - Baseline: no drug
    - Caffeine: adenosine antagonist (expected: mild improvement)
    - Diazepam: GABA-A enhancer (expected: impairment)

    On real tissue, drugs are irreversible. In simulation, we train identical
    networks and compare post-drug performance.

    Args:
        scale: Network scale.
        device: Torch device.
        seed: Random seed.
        n_train_episodes: Episodes for pre-drug training (None = scale default).
        n_test_episodes: Episodes for post-drug testing (None = scale default).
        scenario: ViZDoom scenario name.

    Returns:
        Dict with per-condition test results and pass/fail status.
    """
    sp = SCALE_PARAMS.get(scale, SCALE_PARAMS["large"])
    if n_train_episodes is None:
        n_train_episodes = sp["n_train_episodes"]
    if n_test_episodes is None:
        n_test_episodes = sp["n_test_episodes"]
    metric_name = _scenario_positive_metric(scenario)
    metric_label = _metric_label(metric_name)

    _header(
        "Exp 3: Pharmacological Effects on Doom Performance",
        "3 conditions: baseline / caffeine / diazepam"
    )
    t0 = time.perf_counter()

    conditions = ["baseline", "caffeine", "diazepam"]
    test_results = {}

    for condition in conditions:
        print(f"\n    --- {condition} ---")

        # Build and train brain
        rb, retina, bridge, decoder, relay_ids, l5_ids, cortex_ids = \
            _build_doom_brain(scale, device, seed)
        brain = rb.brain
        dev = brain.device

        n_l5 = len(l5_ids)
        delta = 5.0 * max(1.0, (n_l5 / 200) ** 0.3)

        protocol = DoomFEPProtocol(
            cortex_ids, relay_ids, l5_ids, device=dev,
            structured_steps=sp["structured_steps"],
            unstructured_steps=sp["unstructured_steps"],
            hebbian_delta=delta)
        protocol.motor_populations = decoder.populations

        _warmup(rb, n_steps=sp["warmup_steps"])

        # Train phase
        game = DoomGame(scenario=scenario, seed=seed, visible=False)
        for ep in range(n_train_episodes):
            play_doom_episode(
                rb, game, retina, bridge, decoder, protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"])
        game.close()
        print(f"    Training complete ({n_train_episodes} episodes)")

        # Apply drug AFTER training
        if condition == "caffeine":
            brain.apply_drug("caffeine", 200.0)
            print(f"    Applied caffeine 200mg")
        elif condition == "diazepam":
            brain.apply_drug("diazepam", 40.0)
            print(f"    Applied diazepam 40mg")

        # Test phase: Use FEP protocol (NOT RandomProtocol) to measure
        # drug effects on LEARNING ability, not just performance
        test_protocol = DoomFEPProtocol(
            cortex_ids, relay_ids, l5_ids, device=dev,
            structured_steps=sp["structured_steps"],
            unstructured_steps=sp["unstructured_steps"],
            hebbian_delta=delta)
        test_protocol.motor_populations = decoder.populations

        test_game = DoomGame(scenario=scenario, seed=seed + 1000, visible=False)
        test_metrics = []
        for ep in range(n_test_episodes):
            record_this = (record_video and ep == 0)
            video_path = _default_video_path(
                f"doom_exp3_{condition}_ep{ep+1}.mp4"
            ) if record_this else None
            metrics = play_doom_episode(
                rb, test_game, retina, bridge, decoder, test_protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"],
                record_video=record_this, video_path=video_path)
            test_metrics.append(metrics)
        test_game.close()

        total_metric = sum(_episode_metric(m, metric_name) for m in test_metrics)
        avg_metric = total_metric / n_test_episodes
        test_results[condition] = {
            "total_metric": total_metric,
            "avg_metric": avg_metric,
            "test_metrics": test_metrics,
        }
        print(f"    {condition:10s}: avg {metric_label} = {avg_metric:.1f} "
              f"(total {total_metric:.1f} over {n_test_episodes} episodes)")

    elapsed = time.perf_counter() - t0

    # Pass: diazepam < baseline (GABA-A enhancement impairs performance)
    baseline_metric = test_results["baseline"]["total_metric"]
    diazepam_metric = test_results["diazepam"]["total_metric"]
    caffeine_metric = test_results["caffeine"]["total_metric"]

    passed = diazepam_metric < baseline_metric

    print(f"\n    Baseline:  {baseline_metric:.1f} total {metric_label}")
    print(f"    Caffeine:  {caffeine_metric:.1f} total {metric_label} "
          f"({caffeine_metric - baseline_metric:+.1f})")
    print(f"    Diazepam:  {diazepam_metric:.1f} total {metric_label} "
          f"({diazepam_metric - baseline_metric:+.1f})")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "metric_name": metric_name,
        "test_results": {k: {kk: vv for kk, vv in v.items() if kk != "test_metrics"}
                         for k, v in test_results.items()},
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

ALL_EXPERIMENTS = {
    1: ("Doom Navigation", exp_doom_navigation),
    2: ("Learning Speed Comparison", exp_learning_speed),
    3: ("Pharmacological Effects", exp_pharmacology),
}


def _system_info() -> Dict[str, Any]:
    """Collect system information for JSON output.

    Returns:
        Dict with platform, python version, torch version, backend, and GPU info.
    """
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": detect_backend(),
        "vizdoom": HAS_VIZDOOM,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["gpu"] = "CPU only"
    return info


def _make_json_safe(obj: Any) -> Any:
    """Convert results dict to JSON-serializable form.

    Handles numpy arrays, torch tensors, and numpy scalars.

    Args:
        obj: Any Python object to make JSON-safe.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


def _run_single(args, seed: int) -> Dict[str, Any]:
    """Run all requested experiments with a single seed.

    Args:
        args: Parsed CLI arguments.
        seed: Random seed for this run.

    Returns:
        Dict mapping experiment ID to results.
    """
    exps = args.exp if args.exp else list(ALL_EXPERIMENTS.keys())
    results = {}

    for exp_id in exps:
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n  Unknown experiment: {exp_id}")
            continue
        name, func = ALL_EXPERIMENTS[exp_id]

        try:
            kwargs = {
                "scale": args.scale,
                "device": args.device,
                "seed": seed,
                "scenario": args.scenario,
                "record_video": args.video,
            }
            if args.episodes and exp_id == 1:
                kwargs["n_episodes"] = args.episodes
                kwargs["pretrain_episodes"] = args.pretrain_episodes
                kwargs["structured_replay_scale"] = args.structured_replay_scale
                kwargs["teacher_motor_intensity"] = args.teacher_motor_intensity
                kwargs["teacher_hebbian_delta"] = args.teacher_hebbian_delta
                kwargs["combat_teacher_shaping_delta"] = args.combat_teacher_shaping_delta
                kwargs["combat_attack_window_delta"] = args.combat_attack_window_delta
                kwargs["combat_attack_miss_delta"] = args.combat_attack_miss_delta
                kwargs["combat_decoder_attack_bonus"] = args.combat_decoder_attack_bonus
                kwargs["combat_decoder_attack_penalty"] = args.combat_decoder_attack_penalty
                kwargs["metric_override"] = args.metric_override
            elif args.episodes and exp_id == 2:
                kwargs["n_episodes"] = args.episodes
            elif exp_id == 1:
                kwargs["pretrain_episodes"] = args.pretrain_episodes
                kwargs["structured_replay_scale"] = args.structured_replay_scale
                kwargs["teacher_motor_intensity"] = args.teacher_motor_intensity
                kwargs["teacher_hebbian_delta"] = args.teacher_hebbian_delta
                kwargs["combat_teacher_shaping_delta"] = args.combat_teacher_shaping_delta
                kwargs["combat_attack_window_delta"] = args.combat_attack_window_delta
                kwargs["combat_attack_miss_delta"] = args.combat_attack_miss_delta
                kwargs["combat_decoder_attack_bonus"] = args.combat_decoder_attack_bonus
                kwargs["combat_decoder_attack_penalty"] = args.combat_decoder_attack_penalty
                kwargs["metric_override"] = args.metric_override
            result = func(**kwargs)
            results[exp_id] = result
        except Exception as e:
            print(f"\n  EXPERIMENT {exp_id} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {"passed": False, "error": str(e)}

    return results


def main():
    """CLI entry point for Doom via Molecular Retina experiments."""
    parser = argparse.ArgumentParser(
        description="ViZDoom via Molecular Retina — dONN Plays Doom (Free Energy Principle)"
    )
    parser.add_argument("--exp", type=int, nargs="*", default=None,
                        help="Which experiments to run (1-3). Default: all")
    parser.add_argument("--scale", default="small",
                        choices=list(SCALE_COLUMNS.keys()),
                        help="Network scale (default: small)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default="health_gathering",
                        choices=list(SCENARIOS.keys()),
                        help="ViZDoom scenario (default: health_gathering)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes for experiments 1 & 2")
    parser.add_argument("--pretrain-episodes", type=int, default=0,
                        help="Teacher-guided visual warmup episodes for exp 1")
    parser.add_argument("--structured-replay-scale", type=float, default=0.0,
                        help="Relay replay strength on positive feedback for exp 1")
    parser.add_argument("--teacher-motor-intensity", type=float, default=30.0,
                        help="Teacher motor clamp strength during exp 1 pretraining")
    parser.add_argument("--teacher-hebbian-delta", type=float, default=1.2,
                        help="Teacher relay->motor Hebbian nudge during exp 1 pretraining")
    parser.add_argument("--combat-teacher-shaping-delta", type=float, default=0.0,
                        help="Combat-only online teacher Hebbian shaping during exp 1 play")
    parser.add_argument("--combat-attack-window-delta", type=float, default=0.0,
                        help="Combat-only Hebbian bonus for attack actions in valid attack windows")
    parser.add_argument("--combat-attack-miss-delta", type=float, default=0.0,
                        help="Combat-only Hebbian suppression for blind attack actions")
    parser.add_argument("--combat-decoder-attack-bonus", type=float, default=0.0,
                        help="Combat-only decoder bonus for attack in valid attack windows")
    parser.add_argument("--combat-decoder-attack-penalty", type=float, default=0.0,
                        help="Combat-only decoder penalty for attack outside valid attack windows")
    parser.add_argument("--metric-override", type=str, default=None,
                        help="Override exp 1 success metric, e.g. kills")
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="Write structured JSON results to file")
    parser.add_argument("--runs", type=int, default=1,
                        help="Run each experiment N times with different seeds")
    parser.add_argument("--video", action="store_true",
                        help="Record video of gameplay (requires OpenCV)")
    args = parser.parse_args()

    if not HAS_VIZDOOM:
        print("ERROR: ViZDoom is required. Install with: pip install vizdoom")
        print("       See: https://github.com/Farama-Foundation/ViZDoom")
        return 1
    if not HAS_PIL:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        return 1

    print("=" * 76)
    print("  VIZDOOM VIA MOLECULAR RETINA — dONN PLAYS DOOM")
    print(f"  Backend: {detect_backend()} | Scale: {args.scale} | "
          f"Device: {args.device} | Scenario: {args.scenario}")
    if args.runs > 1:
        print(f"  Multi-seed: {args.runs} runs "
              f"(seeds {args.seed}..{args.seed + args.runs - 1})")
    print(f"  Free energy principle: Kagan et al. (2022) Neuron")
    print("=" * 76)

    total_time = time.perf_counter()

    if args.runs == 1:
        results = _run_single(args, args.seed)
        all_run_results = [results]
    else:
        all_run_results = []
        for run_idx in range(args.runs):
            s = args.seed + run_idx
            print(f"\n{'~' * 76}")
            print(f"  RUN {run_idx + 1}/{args.runs} (seed={s})")
            print(f"{'~' * 76}")
            results = _run_single(args, s)
            all_run_results.append(results)

    total = time.perf_counter() - total_time

    # ── Summary ──
    print("\n" + "=" * 76)
    print("  VIZDOOM DOOM — SUMMARY")
    print("=" * 76)

    final_results = all_run_results[-1]
    if args.runs > 1:
        exp_ids = sorted(set().union(*[r.keys() for r in all_run_results]))
        for exp_id in exp_ids:
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            pass_rates = [r[exp_id].get("passed", False) for r in all_run_results
                          if exp_id in r]
            times = [r[exp_id].get("time", 0) for r in all_run_results
                     if exp_id in r]
            avg_t = sum(times) / len(times) if times else 0
            print(f"    {exp_id}. {name:35s} "
                  f"[{sum(pass_rates)}/{len(pass_rates)} PASS]  "
                  f"avg {avg_t:.1f}s")
    else:
        passed = sum(1 for r in final_results.values() if r.get("passed"))
        total_exp = len(final_results)
        for exp_id, result in sorted(final_results.items()):
            if exp_id not in ALL_EXPERIMENTS:
                continue
            name = ALL_EXPERIMENTS[exp_id][0]
            status = "PASS" if result.get("passed") else "FAIL"
            t = result.get("time", 0)
            print(f"    {exp_id}. {name:35s} [{status}]  {t:.1f}s")
        print(f"\n  Total: {passed}/{total_exp} passed in {total:.1f}s")

    print("=" * 76)

    # ── JSON output ──
    if args.json:
        json_data = {
            "experiment": "doom_vizdoom",
            "scale": args.scale,
            "device": args.device,
            "scenario": args.scenario,
            "n_runs": args.runs,
            "base_seed": args.seed,
            "total_time_s": round(total, 2),
            "system": _system_info(),
            "runs": [],
        }
        for run_idx, run_results in enumerate(all_run_results):
            run_data = {
                "seed": args.seed + run_idx,
                "experiments": {},
            }
            for exp_id, result in sorted(run_results.items()):
                if exp_id not in ALL_EXPERIMENTS:
                    continue
                name = ALL_EXPERIMENTS[exp_id][0]
                # Strip bulky per-episode metrics for JSON
                clean = {}
                for k, v in result.items():
                    if k in ("episode_metrics", "all_results", "test_metrics"):
                        continue
                    clean[k] = v
                clean = _make_json_safe(clean)
                clean["name"] = name
                run_data["experiments"][str(exp_id)] = clean
            json_data["runs"].append(run_data)

        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  JSON results written to: {args.json}")

    all_passed = all(
        r.get("passed", False)
        for run in all_run_results
        for r in run.values()
        if isinstance(r, dict) and "passed" in r
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
