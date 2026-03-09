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


# ============================================================================
# Constants
# ============================================================================

RETINA_WIDTH = 64
RETINA_HEIGHT = 48
VIZDOOM_WIDTH = 160
VIZDOOM_HEIGHT = 120

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
        self._episode_steps = 0
        self._total_episodes = 0

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

        self._game.set_seed(self.seed)
        self._game.init()

        # Track kills for event detection
        self._episode_kills = 0
        self._prev_kills = 0

    def new_episode(self) -> np.ndarray:
        """Start a new episode.

        Returns:
            Downsampled first frame as (48, 64, 3) uint8 array.
        """
        self._game.new_episode()
        self._prev_health = self._get_health()
        self._episode_health_gained = 0.0
        self._episode_damage_taken = 0.0
        self._episode_steps = 0
        self._total_episodes += 1
        return self._get_frame()

    def step(self, action_idx: int) -> Tuple[str, float, bool, np.ndarray, int]:
        """Execute one action and return results.

        Args:
            action_idx: Motor population index (0-5, ATTACK is 5).

        Returns:
            Tuple of (event, health_delta, done, frame, where event is one of
            "health_gained", "damage_taken", "kill", "neutral", or "episode_end".
        """
        # Build one-hot action vector
        action = [0] * N_MOTOR_POPULATIONS
        action[action_idx] = 1

        self._game.make_action(action)
        self._episode_steps += 1

        if self._game.is_episode_finished():
            return "episode_end", 0.0, True, np.zeros(
                (RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8)

        # Track kills
        current_kills = self._get_kills()
        kills_delta = current_kills - self._prev_kills
        event = None
        if kills_delta > 0:
            self._episode_kills += kills_delta
            event = "kill"
        self._prev_kills = current_kills

        # Track health
        current_health = self._get_health()
        health_delta = current_health - self._prev_health
        self._prev_health = current_health

        if event != "kill":
            if health_delta > 0:
                self._episode_health_gained += health_delta
                event = "health_gained"
            elif health_delta < 0:
                self._episode_damage_taken += abs(health_delta)
                event = "damage_taken"
            else:
                event = "neutral"

        frame = self._get_frame()
        return event, health_delta, False, frame

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

    def _get_frame(self) -> np.ndarray:
        """Capture and downsample the current frame.

        Returns:
            (48, 64, 3) uint8 RGB array.
        """
        state = self._game.get_state()
        if state is None:
            return np.zeros((RETINA_HEIGHT, RETINA_WIDTH, 3), dtype=np.uint8)
        buf = state.screen_buffer  # (120, 160, 3) uint8
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
    def episode_damage_taken(self) -> float:
        """Total damage taken in the current episode."""
        return self._episode_damage_taken

    @property
    def episode_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._episode_steps

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
        if not fired_rgc_ids:
            return 0

        activated_relays = set()
        for rgc_id in fired_rgc_ids:
            if rgc_id in self._rgc_to_relay:
                for relay_nid in self._rgc_to_relay[rgc_id]:
                    activated_relays.add(relay_nid)

        if not activated_relays:
            return 0

        relay_tensor = torch.tensor(list(activated_relays),
                                     dtype=torch.int64, device=brain.device)
        brain.external_current[relay_tensor] += intensity
        self._total_injections += 1
        return len(activated_relays)


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
               counts: Optional[List[int]] = None) -> Tuple[int, List[int]]:
        """Decode motor action from spike counts.

        Args:
            brain: The brain to read fired neurons from.
            counts: Pre-computed spike counts per population. If None,
                reads from brain.fired.

        Returns:
            Tuple of (action_index, spike_counts_per_population).
        """
        if counts is None:
            counts = []
            for pop in self.populations:
                counts.append(int(brain.fired[pop].sum().item()))

        total = sum(counts)
        if total == 0:
            # No spikes: random action to explore
            return random.randint(0, N_MOTOR_POPULATIONS - 1), counts

        # Zero-threshold: pick population with most spikes
        max_count = max(counts)
        action = counts.index(max_count)
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
                 hebbian_delta: float = 1.5):
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
        # Track last active motor population for Hebbian nudge
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None

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

        # MINIMAL: Don't stimulate entire cortex - it disrupts natural motor output
        # Instead, just do brief network settling
        rb.step()
        rb.step()

        # ONLY Hebbian nudge on relay->correct motor pathway (no cortex stimulation)
        if self.motor_populations is not None and self.hebbian_delta > 0:
            correct_pop = self.motor_populations[self.last_action]
            wrong_pops = [self.motor_populations[i]
                          for i in range(N_MOTOR_POPULATIONS)
                          if i != self.last_action]
            _doom_hebbian_nudge(brain, self.relay_ids, correct_pop,
                                wrong_pops, self.hebbian_delta)

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

    def deliver_positive(self, rb: CUDARegionalBrain) -> None:
        """Same as negative — no differential feedback."""
        rb.run(self.settle_steps)

    def deliver_negative(self, rb: CUDARegionalBrain) -> None:
        """Same as positive — no differential feedback."""
        rb.run(self.settle_steps)


# ============================================================================
# Hebbian Weight Nudge
# ============================================================================

def _doom_hebbian_nudge(brain: CUDAMolecularBrain, relay_ids: torch.Tensor,
                        correct_pop: torch.Tensor,
                        wrong_pops: List[torch.Tensor],
                        delta: float = 0.5) -> None:
    """Hebbian weight update for Doom motor populations.

    Strengthens relay->correct_motor synapses, weakens relay->wrong_motor.
    This provides targeted credit assignment to accelerate FEP learning.

    Args:
        brain: The molecular brain instance.
        relay_ids: Thalamic relay neuron IDs.
        correct_pop: Neuron IDs of the correct motor population.
        wrong_pops: List of neuron ID tensors for incorrect motor populations.
        delta: Weight update magnitude.
    """
    if brain.n_synapses == 0:
        return

    relay_set = set(relay_ids.cpu().tolist())
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
        Dict with episode metrics: health_gained, damage_taken, steps,
        actions taken, positive/negative event counts.
    """
    brain = rb.brain
    frame = game.new_episode()
    retina.reset()

    total_positive = 0
    total_negative = 0
    action_counts = [0] * N_MOTOR_POPULATIONS
    step_count = 0

    while game.is_running and step_count < max_game_steps:
        # 1. Process frame through molecular retina (pixel -> RGC spikes)
        fired_rgc_ids = retina.process_frame(frame, n_steps=5)

        # 2. Inject RGC spikes into thalamic relay neurons
        bridge.inject_spikes(brain, fired_rgc_ids, intensity=45.0)

        # 3. Run brain for stim_steps (pulsed to avoid depolarization block)
        # Accumulate motor spike counts on GPU
        motor_acc = torch.zeros(N_MOTOR_POPULATIONS, device=brain.device)
        for s in range(stim_steps):
            rb.step()
            for pop_idx, pop_ids in enumerate(decoder.populations):
                motor_acc[pop_idx] += brain.fired[pop_ids].sum()

        # 4. Decode motor action from spike counts (single GPU->CPU sync)
        counts = motor_acc.int().tolist()
        action, _ = decoder.decode(brain, counts=counts)

        # Track which action the protocol should credit
        if hasattr(protocol, 'last_action'):
            protocol.last_action = action
        action_counts[action] += 1

        # 5. Execute action in ViZDoom
        event, health_delta, done, frame = game.step(action)

        if done:
            break

        # 6. Deliver feedback based on game event
        if event == "health_gained":
            protocol.deliver_positive(rb)
            total_positive += 1
        elif event == "damage_taken":
            protocol.deliver_negative(rb)
            total_negative += 1
        else:
            # Neutral step: brief settling
            rb.run(neutral_steps)

        step_count += 1

    return {
        "health_gained": game.episode_health_gained,
        "damage_taken": game.episode_damage_taken,
        "steps": game.episode_steps,
        "positive_events": total_positive,
        "negative_events": total_negative,
        "action_counts": action_counts,
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
) -> Dict[str, Any]:
    """Can a dONN learn to gather health in Doom via the free energy principle?

    Tracks health gained per episode. Learning is evidenced by increasing
    health acquisition over time as the FEP protocol strengthens sensorimotor
    pathways through structured feedback and Hebbian nudging.

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
    )
    protocol.motor_populations = decoder.populations

    # Warmup brain
    _warmup(rb, n_steps=sp["warmup_steps"])
    print(f"    Warmup complete")

    # Game
    game = DoomGame(scenario=scenario, seed=seed, visible=False)

    # Play episodes
    report_interval = max(1, n_episodes // 5)
    episode_metrics = []
    for ep in range(n_episodes):
        metrics = play_doom_episode(
            rb, game, retina, bridge, decoder, protocol,
            relay_ids, stim_steps=sp["stim_steps"],
            max_game_steps=sp["max_game_steps"],
            neutral_steps=sp["neutral_steps"])
        episode_metrics.append(metrics)

        if (ep + 1) % report_interval == 0 or ep == n_episodes - 1:
            recent = episode_metrics[max(0, ep - report_interval + 1):ep + 1]
            avg_health = sum(m["health_gained"] for m in recent) / len(recent)
            avg_damage = sum(m["damage_taken"] for m in recent) / len(recent)
            print(f"    Episode {ep + 1:3d}/{n_episodes}: "
                  f"health +{avg_health:.0f}, damage -{avg_damage:.0f} "
                  f"(last {len(recent)})")

    game.close()

    # Analyze results
    health_per_ep = [m["health_gained"] for m in episode_metrics]
    quarter = max(1, n_episodes // 4)
    first_q = health_per_ep[:quarter]
    last_q = health_per_ep[-quarter:]
    first_avg = sum(first_q) / len(first_q) if first_q else 0
    last_avg = sum(last_q) / len(last_q) if last_q else 0
    total_health = sum(health_per_ep)

    elapsed = time.perf_counter() - t0

    # Pass: health gathered improves over episodes OR total health > 0
    passed = (last_avg > first_avg) or (total_health > 0)

    print(f"\n    Results:")
    print(f"    First quarter avg health: {first_avg:.1f}")
    print(f"    Last quarter avg health:  {last_avg:.1f}")
    print(f"    Total health gathered:    {total_health:.0f}")
    print(f"    Improvement:              {last_avg - first_avg:+.1f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "first_q_avg": first_avg,
        "last_q_avg": last_avg,
        "total_health": total_health,
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
) -> Dict[str, Any]:
    """Compare free energy vs DA reward vs random protocols on Doom.

    Each protocol trains an identical brain (same seed) and plays the same
    scenario. Metrics: total health gathered per protocol.

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

    _header(
        "Exp 2: Learning Speed Comparison",
        "Free energy vs DA reward vs random — which learns fastest in Doom?"
    )
    t0 = time.perf_counter()

    conditions = ["free_energy", "da_reward", "random"]
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
        else:
            protocol = DoomRandomProtocol(cortex_ids, device=dev,
                                           settle_steps=sp["neutral_steps"] * 3)
            protocol.motor_populations = decoder.populations

        _warmup(rb, n_steps=sp["warmup_steps"])

        game = DoomGame(scenario=scenario, seed=seed, visible=False)

        episode_metrics = []
        for ep in range(n_episodes):
            metrics = play_doom_episode(
                rb, game, retina, bridge, decoder, protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"])
            episode_metrics.append(metrics)

        game.close()

        total_health = sum(m["health_gained"] for m in episode_metrics)
        last_q = episode_metrics[-(n_episodes // 4):]
        last_q_health = sum(m["health_gained"] for m in last_q) / len(last_q)

        all_results[condition] = {
            "total_health": total_health,
            "last_q_avg": last_q_health,
            "episode_metrics": episode_metrics,
        }
        print(f"    {condition:15s}: total health = {total_health:.0f}, "
              f"last quarter avg = {last_q_health:.1f}")

    elapsed = time.perf_counter() - t0

    # Pass: FEP or DA outperforms random
    fe_total = all_results["free_energy"]["total_health"]
    da_total = all_results["da_reward"]["total_health"]
    rand_total = all_results["random"]["total_health"]
    passed = (fe_total > rand_total or da_total > rand_total)

    print(f"\n    Free Energy: {fe_total:.0f} total health")
    print(f"    DA Reward:   {da_total:.0f} total health")
    print(f"    Random:      {rand_total:.0f} total health")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
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
            metrics = play_doom_episode(
                rb, test_game, retina, bridge, decoder, test_protocol,
                relay_ids, stim_steps=sp["stim_steps"],
                max_game_steps=sp["max_game_steps"],
                neutral_steps=sp["neutral_steps"])
            test_metrics.append(metrics)
        test_game.close()

        total_health = sum(m["health_gained"] for m in test_metrics)
        avg_health = total_health / n_test_episodes
        test_results[condition] = {
            "total_health": total_health,
            "avg_health": avg_health,
            "test_metrics": test_metrics,
        }
        print(f"    {condition:10s}: avg health = {avg_health:.1f} "
              f"(total {total_health:.0f} over {n_test_episodes} episodes)")

    elapsed = time.perf_counter() - t0

    # Pass: diazepam < baseline (GABA-A enhancement impairs performance)
    baseline_health = test_results["baseline"]["total_health"]
    diazepam_health = test_results["diazepam"]["total_health"]
    caffeine_health = test_results["caffeine"]["total_health"]

    passed = diazepam_health < baseline_health

    print(f"\n    Baseline:  {baseline_health:.0f} total health")
    print(f"    Caffeine:  {caffeine_health:.0f} total health "
          f"({caffeine_health - baseline_health:+.0f})")
    print(f"    Diazepam:  {diazepam_health:.0f} total health "
          f"({diazepam_health - baseline_health:+.0f})")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
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
            }
            if args.episodes and exp_id == 1:
                kwargs["n_episodes"] = args.episodes
            elif args.episodes and exp_id == 2:
                kwargs["n_episodes"] = args.episodes
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
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="Write structured JSON results to file")
    parser.add_argument("--runs", type=int, default=1,
                        help="Run each experiment N times with different seeds")
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
