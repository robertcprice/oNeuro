#!/usr/bin/env python3
"""Hybrid PPO Doom with a digital brain substrate.

Phase 2 adaptation of the hybrid control idea used in hardware-backed systems
such as doom-neuron, with DishBrain-style closed-loop sensory feedback:

- A fixed sensory path drives the digital brain substrate.
- The brain's spike activity is treated as the recurrent latent state.
- A trainable PPO policy reads spike features and selects actions.
- Optional biological feedback can use either:
  - DishBrain-style predictable vs disruptive sensory stimulation
  - dopamine / serotonin-like motor credit assignment

This is intentionally narrower than the hardware setup:
- No CL1 / UDP / electrode transport layer.
- Uses compact observation features and a trainable stimulation encoder.
- Includes ablations for spike features, stimulation, and decoder freezing.
- Includes deterministic held-out evaluation after training.

Usage:
    python3 demos/demo_doom_hybrid_ppo.py
    python3 demos/demo_doom_hybrid_ppo.py --scenario defend_the_center
    python3 demos/demo_doom_hybrid_ppo.py --episodes 24 --scale small
    python3 demos/demo_doom_hybrid_ppo.py --spike-ablation zero
    python3 demos/demo_doom_hybrid_ppo.py --freeze-decoder
    python3 demos/demo_doom_hybrid_ppo.py --no-bio-feedback
    python3 demos/demo_doom_hybrid_ppo.py --feedback-style dishbrain --eval-episodes 8
    python3 demos/demo_doom_hybrid_ppo.py --scenario deadly_corridor --split-attack-head --dagger-episodes 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from demo_doom_vizdoom import (
    DoomGame,
    DoomRLProtocol,
    HAS_PIL,
    HAS_VIZDOOM,
    MOTOR_ATTACK,
    MOTOR_FORWARD,
    MOTOR_STRAFE_LEFT,
    MOTOR_STRAFE_RIGHT,
    N_MOTOR_POPULATIONS,
    NT_DA,
    NT_NE,
    NT_5HT,
    RETINA_HEIGHT,
    RETINA_WIDTH,
    SCALE_COLUMNS,
    SCALE_PARAMS,
    SCENARIOS,
    _build_doom_brain,
    _episode_metric,
    _format_metric_value,
    _header,
    _metric_label,
    _scenario_positive_metric,
    _warmup,
    detect_backend,
)


@dataclass
class HybridPPOConfig:
    """Training hyperparameters for the hybrid PPO readout."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 1.0
    epochs: int = 4
    batch_size: int = 128
    episodes_per_update: int = 4
    hidden_dim: int = 32
    encoder_hidden_dim: int = 64
    stim_group_count: int = 8
    stim_current_max: float = 35.0
    zero_bias: bool = True
    split_attack_head: bool = False
    factorized_attack_controls: bool = False
    use_biological_feedback: bool = True
    feedback_style: str = "dishbrain"
    spike_ablation: str = "none"
    stim_ablation: str = "none"
    freeze_decoder: bool = False
    freeze_encoder: bool = False
    dagger_episodes: int = 0
    dagger_epochs: int = 2
    dagger_batch_size: int = 128
    dagger_coef: float = 1.0
    dagger_replay_capacity: int = 8192
    dagger_replay_ratio: float = 1.0
    dagger_priority_disagreement: float = 2.0
    dagger_priority_negative: float = 0.75
    dagger_priority_death: float = 2.0
    dagger_priority_attack_miss: float = 1.0
    dagger_priority_attack_window: float = 1.0
    aux_combat_heads: bool = False
    aux_head_coef: float = 0.15
    aux_attack_gate_coef: float = 0.25
    predictable_feedback_steps: int = 4
    predictable_feedback_current: float = 18.0
    disruptive_feedback_steps: int = 6
    disruptive_feedback_current: float = 28.0
    disruptive_feedback_fraction: float = 0.5
    post_error_settle_steps: int = 4
    reward_kill: float = 5.0
    reward_survival: float = 2.0
    reward_health_gain_scale: float = 0.05
    penalty_damage_scale: float = 0.1
    penalty_death: float = 6.0
    penalty_neutral_step: float = 0.01


MOVE_CHOICE_NONE = 0
MOVE_BUTTON_IDS = tuple(
    action for action in range(N_MOTOR_POPULATIONS) if action != MOTOR_ATTACK
)
N_MOVE_CHOICES = 1 + len(MOVE_BUTTON_IDS)


def _move_choice_to_button(move_choice: int) -> Optional[int]:
    """Map a factorized movement choice to a Doom button index."""
    move_choice = int(move_choice)
    if move_choice <= MOVE_CHOICE_NONE:
        return None
    idx = move_choice - 1
    if 0 <= idx < len(MOVE_BUTTON_IDS):
        return int(MOVE_BUTTON_IDS[idx])
    return None


def _button_to_move_choice(button_idx: Optional[int]) -> int:
    """Map a Doom movement button to a factorized movement choice."""
    if button_idx is None:
        return MOVE_CHOICE_NONE
    try:
        return 1 + MOVE_BUTTON_IDS.index(int(button_idx))
    except ValueError:
        return MOVE_CHOICE_NONE


def _controls_to_button_vector(
    move_choice: int,
    attack_on: bool,
    device: torch.device,
) -> torch.Tensor:
    """Convert factorized controls into the 6-button Doom action vector."""
    buttons = torch.zeros(N_MOTOR_POPULATIONS, device=device, dtype=torch.float32)
    move_button = _move_choice_to_button(move_choice)
    if move_button is not None:
        buttons[move_button] = 1.0
    if attack_on:
        buttons[MOTOR_ATTACK] = 1.0
    return buttons


def _guidance_to_teacher_controls(guidance: Dict[str, Any]) -> Tuple[int, bool]:
    """Derive factorized movement/attack targets from scalar teacher guidance."""
    action = int(guidance["action"])
    enemy_visible = bool(guidance.get("enemy_visible", False))
    attack_window = bool(guidance.get("attack_window", False))

    if action == MOTOR_ATTACK:
        return MOVE_CHOICE_NONE, True

    move_choice = _button_to_move_choice(action)
    allow_combo_attack = enemy_visible and attack_window and action in (
        MOTOR_FORWARD,
        MOTOR_STRAFE_LEFT,
        MOTOR_STRAFE_RIGHT,
    )
    return move_choice, bool(allow_combo_attack)


class DishBrainFeedbackProtocol:
    """Closed-loop sensory feedback inspired by the DishBrain training protocol.

    Positive outcomes trigger a brief predictable pulse across the relay layer.
    Negative outcomes trigger disruptive random stimulation over relay subgroups,
    followed by a short settling window. The aim is to make "good" states more
    predictable and "bad" states more surprising.
    """

    def __init__(
        self,
        relay_ids: torch.Tensor,
        stim_groups: List[torch.Tensor],
        predictable_steps: int = 4,
        predictable_current: float = 18.0,
        disruptive_steps: int = 6,
        disruptive_current: float = 28.0,
        disruptive_fraction: float = 0.5,
        settle_steps: int = 4,
    ):
        self.relay_ids = relay_ids
        self.stim_groups = stim_groups if stim_groups else [relay_ids]
        self.predictable_steps = max(1, int(predictable_steps))
        self.predictable_current = float(predictable_current)
        self.disruptive_steps = max(1, int(disruptive_steps))
        self.disruptive_current = float(disruptive_current)
        self.disruptive_fraction = min(1.0, max(0.1, float(disruptive_fraction)))
        self.settle_steps = max(0, int(settle_steps))
        self.last_action: int = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None

    def record_action(self, action: int) -> None:
        self.last_action = int(action)

    def _pulse_predictable(self, rb, current_scale: float = 1.0, extra_steps: int = 0) -> None:
        brain = rb.brain
        current = self.predictable_current * current_scale
        n_steps = self.predictable_steps + max(0, int(extra_steps))
        for _ in range(n_steps):
            brain.external_current[self.relay_ids] += current
            rb.step()

    def _pulse_disruptive(self, rb, current_scale: float = 1.0, extra_steps: int = 0) -> None:
        brain = rb.brain
        n_groups = len(self.stim_groups)
        group_count = max(1, int(round(n_groups * self.disruptive_fraction)))
        n_steps = self.disruptive_steps + max(0, int(extra_steps))
        for _ in range(n_steps):
            chosen = random.sample(range(n_groups), k=min(group_count, n_groups))
            for idx in chosen:
                amplitude = self.disruptive_current * current_scale * random.uniform(0.4, 1.0)
                brain.external_current[self.stim_groups[idx]] += amplitude
            rb.step()
        if self.settle_steps:
            rb.run(self.settle_steps)

    def deliver_positive(self, rb) -> None:
        self._pulse_predictable(rb)

    def deliver_kill_reward(self, rb) -> None:
        self._pulse_predictable(rb, current_scale=1.35, extra_steps=2)

    def deliver_survival_reward(self, rb) -> None:
        self._pulse_predictable(rb, current_scale=1.15, extra_steps=1)

    def deliver_negative(self, rb) -> None:
        self._pulse_disruptive(rb)

    def deliver_miss_punishment(self, rb) -> None:
        self._pulse_disruptive(rb, current_scale=0.7, extra_steps=-2)


class EligibilityTraceFeedbackProtocol:
    """Reward/punishment protocol that relies on backend eligibility traces.

    Instead of manually nudging weights, this targets dopamine and 5-HT onto
    recently active motor populations. The CUDA backend then converts existing
    synaptic eligibility traces into permanent updates.
    """

    def __init__(
        self,
        cortex_ids: torch.Tensor,
        l5_ids: torch.Tensor,
        da_amount: float = 120.0,
        serotonin_amount: float = 110.0,
        ne_amount: float = 45.0,
        reward_steps: int = 4,
        settle_steps: int = 4,
        max_history: int = 4,
    ):
        self.cortex_ids = cortex_ids
        self.l5_ids = l5_ids
        self.da_amount = float(da_amount)
        self.serotonin_amount = float(serotonin_amount)
        self.ne_amount = float(ne_amount)
        self.reward_steps = max(1, int(reward_steps))
        self.settle_steps = max(1, int(settle_steps))
        self.max_history = max(1, int(max_history))
        self.last_action = 0
        self.motor_populations: Optional[List[torch.Tensor]] = None
        self.action_history: List[Tuple[int, int]] = []
        self.last_active_relay_ids: Optional[torch.Tensor] = None
        self.last_motor_counts: Optional[torch.Tensor] = None

    def record_action(self, action: int) -> None:
        self.last_action = int(action)
        self.action_history.insert(0, (self.last_action, 0))
        self.action_history = [(a, s + 1) for a, s in self.action_history]
        self.action_history = [(a, s) for a, s in self.action_history if s <= self.max_history]

    def record_step_activity(
        self,
        active_relay_ids: torch.Tensor,
        motor_counts: torch.Tensor,
    ) -> None:
        self.last_active_relay_ids = active_relay_ids.detach().clone()
        self.last_motor_counts = motor_counts.detach().clone()

    def _recent_actions(
        self,
        min_steps: int,
        max_steps: int,
        attack_only: bool = False,
    ) -> List[int]:
        actions: List[int] = []
        for action, steps_ago in self.action_history:
            if min_steps <= steps_ago <= max_steps:
                if attack_only and action != MOTOR_ATTACK:
                    continue
                actions.append(action)
        return actions

    def _apply_neuromodulators(
        self,
        rb,
        actions: List[int],
        dopamine_scale: float = 0.0,
        serotonin_scale: float = 0.0,
        ne_scale: float = 0.0,
        n_steps: Optional[int] = None,
    ) -> None:
        brain = rb.brain
        n_steps = self.reward_steps if n_steps is None else max(1, int(n_steps))

        if actions and self.motor_populations:
            unique_actions = list(set(actions))
            for action in unique_actions:
                pop = self.motor_populations[action]
                frac = actions.count(action) / float(len(actions))
                if dopamine_scale > 0.0:
                    brain.nt_conc[pop, NT_DA] += self.da_amount * dopamine_scale * frac
                if serotonin_scale > 0.0:
                    brain.nt_conc[pop, NT_5HT] += self.serotonin_amount * serotonin_scale * frac

        if ne_scale > 0.0:
            brain.nt_conc[self.cortex_ids, NT_NE] += self.ne_amount * ne_scale

        for _ in range(n_steps):
            rb.step()

    def deliver_positive(self, rb) -> None:
        rewarded = self._recent_actions(1, 3)
        self._apply_neuromodulators(
            rb,
            actions=rewarded,
            dopamine_scale=1.0,
            ne_scale=0.75,
            n_steps=self.reward_steps,
        )

    def deliver_kill_reward(self, rb) -> None:
        rewarded = self._recent_actions(1, 4, attack_only=True)
        if not rewarded:
            rewarded = [MOTOR_ATTACK]
        self._apply_neuromodulators(
            rb,
            actions=rewarded,
            dopamine_scale=2.0,
            ne_scale=1.0,
            n_steps=self.reward_steps + 2,
        )

    def deliver_survival_reward(self, rb) -> None:
        rewarded = self._recent_actions(1, 3)
        self._apply_neuromodulators(
            rb,
            actions=rewarded,
            dopamine_scale=0.8,
            ne_scale=0.5,
            n_steps=self.reward_steps,
        )

    def deliver_negative(self, rb) -> None:
        punished = self._recent_actions(1, 3)
        self._apply_neuromodulators(
            rb,
            actions=punished,
            serotonin_scale=1.0,
            n_steps=self.settle_steps,
        )

    def deliver_miss_punishment(self, rb) -> None:
        punished = self._recent_actions(1, 2, attack_only=True)
        self._apply_neuromodulators(
            rb,
            actions=punished,
            serotonin_scale=0.6,
            n_steps=max(1, self.settle_steps - 1),
        )


class CombinedFeedbackProtocol:
    """Compose multiple feedback controllers into one protocol."""

    def __init__(self, *protocols: Any):
        self.protocols = [p for p in protocols if p is not None]
        self._motor_populations: Optional[List[torch.Tensor]] = None
        self.last_action = 0

    @property
    def motor_populations(self) -> Optional[List[torch.Tensor]]:
        return self._motor_populations

    @motor_populations.setter
    def motor_populations(self, populations: Optional[List[torch.Tensor]]) -> None:
        self._motor_populations = populations
        for protocol in self.protocols:
            if hasattr(protocol, "motor_populations"):
                protocol.motor_populations = populations

    def record_action(self, action: int) -> None:
        self.last_action = int(action)
        for protocol in self.protocols:
            if hasattr(protocol, "last_action"):
                protocol.last_action = self.last_action
            if hasattr(protocol, "record_action"):
                protocol.record_action(self.last_action)

    def record_step_activity(self, active_relay_ids: torch.Tensor, motor_counts: torch.Tensor) -> None:
        for protocol in self.protocols:
            if hasattr(protocol, "record_step_activity"):
                protocol.record_step_activity(active_relay_ids, motor_counts)

    def deliver_positive(self, rb) -> None:
        for protocol in self.protocols:
            protocol.deliver_positive(rb)

    def deliver_kill_reward(self, rb) -> None:
        for protocol in self.protocols:
            if hasattr(protocol, "deliver_kill_reward"):
                protocol.deliver_kill_reward(rb)
            else:
                protocol.deliver_positive(rb)

    def deliver_survival_reward(self, rb) -> None:
        for protocol in self.protocols:
            if hasattr(protocol, "deliver_survival_reward"):
                protocol.deliver_survival_reward(rb)
            else:
                protocol.deliver_positive(rb)

    def deliver_negative(self, rb) -> None:
        for protocol in self.protocols:
            protocol.deliver_negative(rb)

    def deliver_miss_punishment(self, rb) -> None:
        for protocol in self.protocols:
            if hasattr(protocol, "deliver_miss_punishment"):
                protocol.deliver_miss_punishment(rb)
            else:
                protocol.deliver_negative(rb)


class StimEncoderNetwork(nn.Module):
    """Observation-to-current encoder for relay-group stimulation."""

    def __init__(
        self,
        obs_dim: int,
        n_groups: int,
        hidden_dim: int = 64,
        current_max: float = 35.0,
    ):
        super().__init__()
        hidden_dim = max(16, int(hidden_dim))
        self.current_max = float(current_max)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, n_groups)
        self.beta_head = nn.Linear(hidden_dim, n_groups)

    def _dist(self, obs_features: torch.Tensor):
        hidden = self.backbone(obs_features)
        alpha = F.softplus(self.alpha_head(hidden)) + 1.0
        beta = F.softplus(self.beta_head(hidden)) + 1.0
        return torch.distributions.Beta(alpha, beta)

    def sample(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs_features)
        unit = dist.mean if deterministic else dist.sample()
        currents = unit * self.current_max
        log_prob = dist.log_prob(unit).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return currents, log_prob, entropy

    def log_prob(self, obs_features: torch.Tensor, currents: torch.Tensor) -> torch.Tensor:
        dist = self._dist(obs_features)
        unit = torch.clamp(currents / max(self.current_max, 1e-6), 1e-6, 1.0 - 1e-6)
        return dist.log_prob(unit).sum(dim=-1)

    def entropy(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self._dist(obs_features).entropy().sum(dim=-1)


class HybridSpikePolicy(nn.Module):
    """Hybrid policy with trainable stim encoder and spike decoder."""

    def __init__(
        self,
        obs_dim: int,
        spike_feature_dim: int,
        n_actions: int,
        n_stim_groups: int,
        hidden_dim: int = 32,
        encoder_hidden_dim: int = 64,
        stim_current_max: float = 35.0,
        zero_bias: bool = True,
        split_attack_head: bool = False,
        factorized_attack_controls: bool = False,
        aux_combat_heads: bool = False,
        attack_action: int = MOTOR_ATTACK,
        freeze_decoder: bool = False,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        hidden_dim = max(0, int(hidden_dim))
        self.use_hidden = hidden_dim > 0
        self.n_actions = int(n_actions)
        self.split_attack_head = bool(split_attack_head)
        self.factorized_attack_controls = bool(factorized_attack_controls and split_attack_head)
        self.aux_combat_heads = bool(aux_combat_heads)
        self.attack_action = int(attack_action)
        self.movement_action_ids = [
            action for action in range(self.n_actions) if action != self.attack_action
        ]
        self.stim_encoder = StimEncoderNetwork(
            obs_dim=obs_dim,
            n_groups=n_stim_groups,
            hidden_dim=encoder_hidden_dim,
            current_max=stim_current_max,
        )

        if self.use_hidden:
            self.backbone = nn.Sequential(
                nn.Linear(spike_feature_dim, hidden_dim),
                nn.SiLU(),
            )
            action_in = hidden_dim
        else:
            self.backbone = nn.Identity()
            action_in = spike_feature_dim

        if self.split_attack_head:
            move_dim = N_MOVE_CHOICES if self.factorized_attack_controls else len(self.movement_action_ids)
            self.move_head = nn.Linear(action_in, move_dim)
            self.attack_head = nn.Linear(action_in, 1)
        else:
            self.action_head = nn.Linear(action_in, n_actions)
        if self.aux_combat_heads:
            self.enemy_head = nn.Linear(action_in, 1)
            self.attack_window_head = nn.Linear(action_in, 1)
        value_hidden = max(32, hidden_dim if hidden_dim > 0 else spike_feature_dim)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim + spike_feature_dim, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 1),
        )

        if zero_bias and self.split_attack_head:
            self.move_head.bias.data.zero_()
            self.move_head.bias.requires_grad = False
            # Keep attack rare by default without hard-coding it off.
            self.attack_head.bias.data.fill_(-1.25 if self.factorized_attack_controls else -1.5)
        elif zero_bias:
            self.action_head.bias.data.zero_()
            self.action_head.bias.requires_grad = False
        if isinstance(self.value_net[-1], nn.Linear) and self.value_net[-1].bias is not None:
            self.value_net[-1].bias.data.zero_()

        if freeze_decoder:
            decoder_modules: List[nn.Module] = [self.backbone]
            if self.split_attack_head:
                decoder_modules.extend([self.move_head, self.attack_head])
            else:
                decoder_modules.append(self.action_head)
            if self.aux_combat_heads:
                decoder_modules.extend([self.enemy_head, self.attack_window_head])
            for module in decoder_modules:
                for param in module.parameters():
                    param.requires_grad = False
        if freeze_encoder:
            for param in self.stim_encoder.parameters():
                param.requires_grad = False

    def backbone_features(self, spike_features: torch.Tensor) -> torch.Tensor:
        return self.backbone(spike_features)

    def _movement_distribution(self, hidden: torch.Tensor) -> Categorical:
        return Categorical(probs=F.softmax(self.move_head(hidden), dim=-1))

    def _attack_distribution(self, hidden: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self.attack_head(hidden).squeeze(-1))

    def _action_probs_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.split_attack_head:
            return F.softmax(self.action_head(hidden), dim=-1)
        if self.factorized_attack_controls:
            move_probs = self._movement_distribution(hidden).probs
            attack_prob = self._attack_distribution(hidden).probs
            probs = torch.zeros(
                hidden.size(0),
                self.n_actions,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            probs[:, self.movement_action_ids] = move_probs[:, 1:] * (1.0 - attack_prob.unsqueeze(-1))
            probs[:, self.attack_action] = attack_prob
            probs = probs.clamp_min(1e-6)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            return probs

        move_probs = F.softmax(self.move_head(hidden), dim=-1)
        attack_prob = torch.sigmoid(self.attack_head(hidden))
        probs = torch.zeros(
            hidden.size(0),
            self.n_actions,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        probs[:, self.movement_action_ids] = move_probs * (1.0 - attack_prob)
        probs[:, self.attack_action] = attack_prob.squeeze(-1)
        probs = probs.clamp_min(1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def _action_probs(self, spike_features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone_features(spike_features)
        return self._action_probs_from_hidden(hidden)

    def action_distribution(self, spike_features: torch.Tensor) -> Categorical:
        probs = self._action_probs(spike_features)
        return Categorical(probs=probs)

    def aux_logits(self, spike_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone_features(spike_features)
        if not self.aux_combat_heads:
            zeros = torch.zeros(hidden.size(0), device=hidden.device, dtype=hidden.dtype)
            return {
                "enemy_visible": zeros,
                "attack_window": zeros,
                "attack_gate": zeros,
            }
        attack_gate = (
            self.attack_head(hidden).squeeze(-1)
            if self.split_attack_head
            else self.action_head(hidden)[:, self.attack_action]
        )
        return {
            "enemy_visible": self.enemy_head(hidden).squeeze(-1),
            "attack_window": self.attack_window_head(hidden).squeeze(-1),
            "attack_gate": attack_gate,
        }

    def value(self, obs_features: torch.Tensor, spike_features: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([obs_features, spike_features], dim=-1)
        return self.value_net(joint).squeeze(-1)

    def act(
        self,
        obs_features: torch.Tensor,
        spike_features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone_features(spike_features)
        value = self.value(obs_features, spike_features)
        if self.factorized_attack_controls:
            move_dist = self._movement_distribution(hidden)
            attack_dist = self._attack_distribution(hidden)
            if deterministic:
                move_action = move_dist.probs.argmax(dim=-1)
                attack_action = (attack_dist.probs >= 0.5).to(dtype=torch.long)
            else:
                move_action = move_dist.sample()
                attack_action = attack_dist.sample().to(dtype=torch.long)
            action = torch.stack([move_action, attack_action], dim=-1)
            log_prob = move_dist.log_prob(move_action) + attack_dist.log_prob(
                attack_action.float()
            )
            entropy = move_dist.entropy() + attack_dist.entropy()
            return action, log_prob, entropy, value

        dist = self.action_distribution(spike_features)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs_features: torch.Tensor,
        spike_features: torch.Tensor,
        actions: torch.Tensor,
        stim_currents: Optional[torch.Tensor],
        include_encoder_log_prob: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone_features(spike_features)
        value = self.value(obs_features, spike_features)
        if self.factorized_attack_controls:
            move_dist = self._movement_distribution(hidden)
            attack_dist = self._attack_distribution(hidden)
            move_actions = actions[:, 0].long()
            attack_actions = actions[:, 1].float()
            action_log_prob = move_dist.log_prob(move_actions) + attack_dist.log_prob(
                attack_actions
            )
            action_entropy = move_dist.entropy() + attack_dist.entropy()
            encoder_entropy = torch.zeros_like(action_entropy)
        else:
            dist = self.action_distribution(spike_features)
            action_log_prob = dist.log_prob(actions)
            action_entropy = dist.entropy()
            encoder_entropy = torch.zeros_like(action_entropy)

        total_log_prob = action_log_prob
        if include_encoder_log_prob and stim_currents is not None:
            total_log_prob = total_log_prob + self.stim_encoder.log_prob(obs_features, stim_currents)
            encoder_entropy = self.stim_encoder.entropy(obs_features)

        return total_log_prob, value, action_entropy, encoder_entropy

    def sample_stim(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.stim_encoder.sample(obs_features, deterministic=deterministic)

    def imitation_loss(
        self,
        spike_features: torch.Tensor,
        teacher_actions: torch.Tensor,
        teacher_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted behavior-cloning loss over teacher-labeled states."""
        if self.factorized_attack_controls:
            hidden = self.backbone_features(spike_features)
            move_dist = self._movement_distribution(hidden)
            attack_dist = self._attack_distribution(hidden)
            move_targets = teacher_actions[:, 0].long()
            attack_targets = teacher_actions[:, 1].float()
            loss = (
                -move_dist.log_prob(move_targets)
                - attack_dist.log_prob(attack_targets)
            )
        else:
            dist = self.action_distribution(spike_features)
            loss = -dist.log_prob(teacher_actions)
        if teacher_weights is not None:
            weights = teacher_weights / teacher_weights.mean().clamp_min(1e-6)
            loss = loss * weights
        return loss.mean()

    def auxiliary_losses(
        self,
        spike_features: torch.Tensor,
        enemy_visible: torch.Tensor,
        attack_window: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.aux_logits(spike_features)
        enemy_target = enemy_visible.float()
        attack_target = attack_window.float()
        return {
            "enemy_loss": F.binary_cross_entropy_with_logits(
                logits["enemy_visible"], enemy_target
            ),
            "attack_window_loss": F.binary_cross_entropy_with_logits(
                logits["attack_window"], attack_target
            ),
            "attack_gate_loss": F.binary_cross_entropy_with_logits(
                logits["attack_gate"], attack_target
            ),
        }


def _make_stim_groups(relay_ids: torch.Tensor, n_groups: int) -> List[torch.Tensor]:
    """Split relay neurons into stimulation subgroups."""
    n_groups = max(1, min(int(n_groups), int(len(relay_ids))))
    return [chunk for chunk in torch.chunk(relay_ids, n_groups) if len(chunk) > 0]


def _ablate_tensor(
    tensor: torch.Tensor,
    mode: str,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a deterministic ablation mode."""
    if mode == "zero":
        return torch.zeros_like(tensor)
    if mode == "random":
        return torch.rand_like(tensor) * scale
    return tensor


def _build_obs_features(
    frame: np.ndarray,
    prev_buttons: Optional[torch.Tensor],
    step_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Compact observation vector for the stimulation encoder."""
    frame_t = torch.from_numpy(frame).to(device=device, dtype=torch.float32)
    gray = frame_t.mean(dim=-1, keepdim=False).unsqueeze(0).unsqueeze(0) / 255.0
    pooled = F.avg_pool2d(gray, kernel_size=8, stride=8).reshape(-1)
    prev_action_1h = torch.zeros(N_MOTOR_POPULATIONS, device=device)
    if prev_buttons is not None:
        prev_action_1h = prev_buttons.to(device=device, dtype=torch.float32)
    step_feat = torch.tensor([step_fraction], device=device, dtype=torch.float32)
    return torch.cat([pooled, prev_action_1h, step_feat], dim=0)


def _apply_stim_encoder(
    brain,
    stim_groups: List[torch.Tensor],
    stim_currents: torch.Tensor,
) -> None:
    """Inject encoder currents into relay subgroups for the next brain step."""
    for group, current in zip(stim_groups, stim_currents):
        brain.external_current[group] += current


def _build_spike_features(motor_counts: torch.Tensor) -> torch.Tensor:
    """Convert raw motor spike counts into policy features."""
    counts = motor_counts.float()
    total = counts.sum()
    norm = counts / (total + 1e-6)
    summary = torch.stack([
        total,
        counts.max(),
        counts.mean(),
    ])
    return torch.cat([counts, norm, summary], dim=0)


def _set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_feedback_protocol(
    config: HybridPPOConfig,
    enabled: bool,
    relay_ids: torch.Tensor,
    l5_ids: torch.Tensor,
    cortex_ids: torch.Tensor,
    stim_groups: List[torch.Tensor],
    decoder,
    scale_params: Dict[str, Any],
    device: torch.device,
):
    """Construct the configured substrate feedback controller."""
    if not enabled:
        return None

    dishbrain_protocol = DishBrainFeedbackProtocol(
        relay_ids=relay_ids,
        stim_groups=stim_groups,
        predictable_steps=config.predictable_feedback_steps,
        predictable_current=config.predictable_feedback_current,
        disruptive_steps=config.disruptive_feedback_steps,
        disruptive_current=config.disruptive_feedback_current,
        disruptive_fraction=config.disruptive_feedback_fraction,
        settle_steps=max(config.post_error_settle_steps, scale_params["neutral_steps"]),
    )
    eligibility_protocol = EligibilityTraceFeedbackProtocol(
        cortex_ids=cortex_ids,
        l5_ids=l5_ids,
        da_amount=140.0,
        serotonin_amount=120.0,
        ne_amount=50.0,
        reward_steps=max(3, scale_params["structured_steps"] // 2),
        settle_steps=max(2, scale_params["neutral_steps"]),
        max_history=4,
    )

    if config.feedback_style == "dishbrain":
        protocol = dishbrain_protocol
    elif config.feedback_style == "eligibility_da":
        protocol = eligibility_protocol
    elif config.feedback_style == "hybrid_bio":
        protocol = CombinedFeedbackProtocol(dishbrain_protocol, eligibility_protocol)
    else:
        protocol = DoomRLProtocol(
            cortex_ids,
            relay_ids=relay_ids,
            l5_ids=l5_ids,
            device=str(device),
            da_amount=200.0,
            cortisol_amount=150.0,
            reward_steps=max(4, scale_params["structured_steps"] // 2),
            settle_steps=max(2, scale_params["neutral_steps"]),
        )

    if hasattr(protocol, "motor_populations"):
        protocol.motor_populations = decoder.populations
    return protocol


def _summarize_episode_metrics(
    episode_metrics: List[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, float]:
    """Aggregate scalar metrics across a set of episodes."""
    if not episode_metrics:
        return {
            "avg_metric": 0.0,
            "avg_return": 0.0,
            "avg_damage": 0.0,
            "avg_kills": 0.0,
        }

    count = len(episode_metrics)
    return {
        "avg_metric": sum(_episode_metric(m, metric_name) for m in episode_metrics) / count,
        "avg_return": sum(m["episode_return"] for m in episode_metrics) / count,
        "avg_damage": sum(m["damage_taken"] for m in episode_metrics) / count,
        "avg_kills": sum(m["kills"] for m in episode_metrics) / count,
    }


def _event_reward(
    event: str,
    health_delta: float,
    config: HybridPPOConfig,
) -> float:
    """Scalar reward used by PPO."""
    if event == "health_gained":
        return 1.0 + max(0.0, health_delta) * config.reward_health_gain_scale
    if event == "kill":
        return config.reward_kill
    if event == "survived":
        return config.reward_survival
    if event == "damage_taken":
        return -max(1.0, abs(health_delta) * config.penalty_damage_scale)
    if event == "episode_end":
        return -config.penalty_death
    return -config.penalty_neutral_step


def _deliver_biological_feedback(
    protocol,
    rb,
    event: str,
    did_attack: bool,
    neutral_steps: int,
) -> Tuple[float, float]:
    """Apply substrate feedback for the observed event."""
    total_positive = 0.0
    total_negative = 0.0

    if event == "health_gained":
        protocol.deliver_positive(rb)
        total_positive += 1.0
    elif event == "kill":
        if hasattr(protocol, "deliver_kill_reward"):
            protocol.deliver_kill_reward(rb)
        else:
            protocol.deliver_positive(rb)
        total_positive += 5.0
    elif event == "survived":
        if hasattr(protocol, "deliver_survival_reward"):
            protocol.deliver_survival_reward(rb)
        else:
            protocol.deliver_positive(rb)
        total_positive += 1.0
    elif event == "damage_taken":
        protocol.deliver_negative(rb)
        total_negative += 1.0
    elif event == "episode_end":
        protocol.deliver_negative(rb)
        total_negative += 10.0
    else:
        if did_attack and hasattr(protocol, "deliver_miss_punishment"):
            protocol.deliver_miss_punishment(rb)
            total_negative += 0.5
        else:
            rb.run(neutral_steps)

    return total_positive, total_negative


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(rewards.numel())):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def _merge_rollouts(rollouts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate rollout tensors across episodes."""
    keys = rollouts[0].keys()
    merged = {}
    for key in keys:
        merged[key] = torch.cat([r[key] for r in rollouts], dim=0)
    return merged


class DAggerReplayBuffer:
    """Priority buffer for hard teacher-labeled states."""

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.spike_features: Optional[torch.Tensor] = None
        self.teacher_actions: Optional[torch.Tensor] = None
        self.teacher_weights: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        if self.spike_features is None:
            return 0
        return int(self.spike_features.size(0))

    def add_rollout(self, rollout: Dict[str, torch.Tensor]) -> int:
        if self.capacity <= 0:
            return 0
        teacher_mask = rollout["teacher_mask"] > 0.0
        if teacher_mask.sum().item() == 0:
            return 0

        new_spike = rollout["spike_features"][teacher_mask].detach().cpu()
        new_actions = rollout["teacher_actions"][teacher_mask].detach().cpu()
        new_weights = rollout["teacher_weights"][teacher_mask].detach().cpu().clamp_min(1e-3)

        if self.spike_features is None:
            self.spike_features = new_spike
            self.teacher_actions = new_actions
            self.teacher_weights = new_weights
        else:
            self.spike_features = torch.cat([self.spike_features, new_spike], dim=0)
            self.teacher_actions = torch.cat([self.teacher_actions, new_actions], dim=0)
            self.teacher_weights = torch.cat([self.teacher_weights, new_weights], dim=0)

        if len(self) > self.capacity:
            keep = torch.topk(self.teacher_weights, k=self.capacity).indices
            self.spike_features = self.spike_features[keep]
            self.teacher_actions = self.teacher_actions[keep]
            self.teacher_weights = self.teacher_weights[keep]

        return int(new_spike.size(0))

    def sample(
        self,
        n_samples: int,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if len(self) == 0 or n_samples <= 0:
            return None

        n_samples = min(int(n_samples), len(self))
        probs = self.teacher_weights / self.teacher_weights.sum().clamp_min(1e-6)
        replace = n_samples > len(self)
        indices = torch.multinomial(probs, num_samples=n_samples, replacement=replace)
        return {
            "spike_features": self.spike_features[indices].to(device=device),
            "teacher_actions": self.teacher_actions[indices].to(device=device),
            "teacher_weights": self.teacher_weights[indices].to(device=device),
        }


def update_policy(
    policy: HybridSpikePolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: HybridPPOConfig,
) -> Dict[str, float]:
    """Run PPO updates on the collected rollout."""
    obs_features = rollout["obs_features"]
    spike_features = rollout["spike_features"]
    stim_currents = rollout["stim_currents"]
    actions = rollout["actions"]
    old_log_probs = rollout["log_probs"]
    returns = rollout["returns"]
    advantages = rollout["advantages"]
    enemy_visible = rollout["enemy_visible"]
    attack_window = rollout["attack_window"]
    include_encoder_log_prob = (config.stim_ablation == "none")

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = min(config.batch_size, spike_features.size(0))

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_action_entropy = 0.0
    last_encoder_entropy = 0.0
    last_enemy_loss = 0.0
    last_attack_window_loss = 0.0
    last_attack_gate_loss = 0.0

    for _ in range(config.epochs):
        indices = torch.randperm(spike_features.size(0), device=spike_features.device)
        for start in range(0, spike_features.size(0), batch_size):
            batch_idx = indices[start:start + batch_size]
            log_probs, values, action_entropy, encoder_entropy = policy.evaluate_actions(
                obs_features[batch_idx],
                spike_features[batch_idx],
                actions[batch_idx],
                stim_currents[batch_idx],
                include_encoder_log_prob=include_encoder_log_prob,
            )
            entropy = action_entropy.mean()
            encoder_entropy_mean = encoder_entropy.mean()

            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_idx])

            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            enemy_loss = torch.zeros((), device=spike_features.device)
            attack_window_loss = torch.zeros((), device=spike_features.device)
            attack_gate_loss = torch.zeros((), device=spike_features.device)
            if config.aux_combat_heads:
                aux_losses = policy.auxiliary_losses(
                    spike_features[batch_idx],
                    enemy_visible[batch_idx],
                    attack_window[batch_idx],
                )
                enemy_loss = aux_losses["enemy_loss"]
                attack_window_loss = aux_losses["attack_window_loss"]
                attack_gate_loss = aux_losses["attack_gate_loss"]
                loss = (
                    loss
                    + config.aux_head_coef * (enemy_loss + attack_window_loss)
                    + config.aux_attack_gate_coef * attack_gate_loss
                )
            if include_encoder_log_prob:
                loss = loss - (0.25 * config.entropy_coef) * encoder_entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_action_entropy = float(entropy.item())
            last_encoder_entropy = float(encoder_entropy_mean.item())
            last_enemy_loss = float(enemy_loss.item())
            last_attack_window_loss = float(attack_window_loss.item())
            last_attack_gate_loss = float(attack_gate_loss.item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "action_entropy": last_action_entropy,
        "encoder_entropy": last_encoder_entropy,
        "enemy_loss": last_enemy_loss,
        "attack_window_loss": last_attack_window_loss,
        "attack_gate_loss": last_attack_gate_loss,
    }


def update_imitation_policy(
    policy: HybridSpikePolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: HybridPPOConfig,
    replay_buffer: Optional[DAggerReplayBuffer] = None,
) -> Dict[str, float]:
    """Run DAgger-style imitation updates on teacher-labeled learner states."""
    teacher_mask = rollout["teacher_mask"] > 0.0
    online_spike = rollout["spike_features"][teacher_mask]
    online_actions = rollout["teacher_actions"][teacher_mask]
    online_weights = rollout["teacher_weights"][teacher_mask]

    device = rollout["spike_features"].device
    replay_samples = None
    replay_count = 0
    if replay_buffer is not None and len(replay_buffer) > 0:
        target_count = max(
            config.dagger_batch_size,
            int(round(max(1, online_spike.size(0)) * max(0.0, config.dagger_replay_ratio))),
        )
        replay_samples = replay_buffer.sample(target_count, device=device)
        if replay_samples is not None:
            replay_count = int(replay_samples["spike_features"].size(0))

    if online_spike.size(0) == 0 and replay_samples is None:
        return {
            "imitation_loss": 0.0,
            "imitation_samples": 0.0,
            "replay_samples": 0.0,
        }

    datasets_spike = []
    datasets_actions = []
    datasets_weights = []
    if online_spike.size(0) > 0:
        datasets_spike.append(online_spike)
        datasets_actions.append(online_actions)
        datasets_weights.append(online_weights)
    if replay_samples is not None:
        datasets_spike.append(replay_samples["spike_features"])
        datasets_actions.append(replay_samples["teacher_actions"])
        datasets_weights.append(replay_samples["teacher_weights"])

    spike_features = torch.cat(datasets_spike, dim=0)
    teacher_actions = torch.cat(datasets_actions, dim=0)
    teacher_weights = torch.cat(datasets_weights, dim=0)
    batch_size = min(config.dagger_batch_size, spike_features.size(0))
    last_loss = 0.0

    for _ in range(max(1, config.dagger_epochs)):
        indices = torch.randperm(spike_features.size(0), device=spike_features.device)
        for start in range(0, spike_features.size(0), batch_size):
            batch_idx = indices[start:start + batch_size]
            loss = config.dagger_coef * policy.imitation_loss(
                spike_features[batch_idx],
                teacher_actions[batch_idx],
                teacher_weights[batch_idx],
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.item())

    return {
        "imitation_loss": last_loss,
        "imitation_samples": float(spike_features.size(0)),
        "replay_samples": float(replay_count),
    }


def play_hybrid_episode(
    rb,
    game: DoomGame,
    retina,
    bridge,
    decoder,
    stim_groups: List[torch.Tensor],
    policy: HybridSpikePolicy,
    protocol: Optional[Any],
    config: HybridPPOConfig,
    stim_steps: int,
    max_game_steps: int,
    neutral_steps: int,
    collect_teacher: bool = False,
    deterministic: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Collect one episode rollout with the hybrid brain-in-the-loop policy."""
    brain = rb.brain
    frame = game.new_episode()
    retina.reset()
    rb.run(max(1, neutral_steps))

    if protocol is not None and hasattr(protocol, "action_history"):
        protocol.action_history.clear()

    obs_features_list: List[torch.Tensor] = []
    spike_features_list: List[torch.Tensor] = []
    stim_currents_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    log_probs_list: List[torch.Tensor] = []
    values_list: List[torch.Tensor] = []
    rewards_list: List[torch.Tensor] = []
    dones_list: List[torch.Tensor] = []
    teacher_actions_list: List[torch.Tensor] = []
    teacher_weights_list: List[torch.Tensor] = []
    teacher_mask_list: List[torch.Tensor] = []
    enemy_visible_list: List[torch.Tensor] = []
    attack_window_list: List[torch.Tensor] = []

    action_counts = [0] * N_MOTOR_POPULATIONS
    total_positive = 0.0
    total_negative = 0.0
    total_reward = 0.0
    step_count = 0
    prev_buttons: Optional[torch.Tensor] = None
    teacher_matches = 0
    teacher_labels = 0

    while game.is_running and step_count < max_game_steps:
        obs_features = _build_obs_features(
            frame=frame,
            prev_buttons=prev_buttons,
            step_fraction=step_count / max(1, max_game_steps),
            device=brain.device,
        ).unsqueeze(0)
        guidance = game.teacher_guidance()
        teacher_action = int(guidance["action"])
        teacher_move_choice, teacher_attack_on = _guidance_to_teacher_controls(guidance)
        teacher_weight = float(guidance.get("confidence", 1.0))
        teacher_mask = 1.0 if collect_teacher else 0.0
        with torch.no_grad():
            stim_currents, encoder_log_prob, _ = policy.sample_stim(
                obs_features, deterministic=deterministic
            )
        stim_currents = _ablate_tensor(
            stim_currents, config.stim_ablation, scale=config.stim_current_max
        )

        fired_rgc_ids = retina.process_frame(frame, n_steps=5)
        relay_activation = bridge.activation_from_spikes(fired_rgc_ids, intensity=45.0)
        active_relay_ids = bridge.relay_ids[relay_activation > 0]
        bridge.inject_activation(brain, relay_activation)
        _apply_stim_encoder(brain, stim_groups, stim_currents.squeeze(0))

        motor_acc = torch.zeros(N_MOTOR_POPULATIONS, device=brain.device)
        for _ in range(stim_steps):
            rb.step()
            for pop_idx, pop_ids in enumerate(decoder.populations):
                motor_acc[pop_idx] += brain.fired[pop_ids].sum()

        spike_features = _build_spike_features(motor_acc).unsqueeze(0)
        spike_features = _ablate_tensor(
            spike_features, config.spike_ablation, scale=1.0
        )
        if protocol is not None and hasattr(protocol, "record_step_activity"):
            protocol.record_step_activity(active_relay_ids, motor_acc)
        with torch.no_grad():
            action, action_log_prob, _, value = policy.act(
                obs_features, spike_features, deterministic=deterministic
            )
        if policy.factorized_attack_controls:
            move_choice = int(action[0, 0].item())
            attack_on = bool(action[0, 1].item())
            move_button = _move_choice_to_button(move_choice)
            action_idx = MOTOR_ATTACK if attack_on else (
                move_button if move_button is not None else MOTOR_FORWARD
            )
            action_buttons = _controls_to_button_vector(
                move_choice,
                attack_on,
                device=brain.device,
            )
        else:
            action_idx = int(action.item())
            attack_on = action_idx == MOTOR_ATTACK
            action_buttons = torch.zeros(
                N_MOTOR_POPULATIONS, device=brain.device, dtype=torch.float32
            )
            action_buttons[action_idx] = 1.0
        if collect_teacher:
            teacher_labels += 1
            if policy.factorized_attack_controls:
                teacher_matches += int(
                    move_choice == teacher_move_choice and attack_on == teacher_attack_on
                )
            else:
                teacher_matches += int(action_idx == teacher_action)
        prev_buttons = action_buttons.detach()

        if protocol is not None:
            protocol.last_action = action_idx
            protocol.record_action(action_idx)

        for button_idx in torch.nonzero(action_buttons > 0.5, as_tuple=False).view(-1).tolist():
            action_counts[int(button_idx)] += 1
        event, health_delta, done, frame = game.step(
            action_vector=action_buttons.detach().cpu().to(dtype=torch.int64).tolist()
        )
        reward = _event_reward(event, health_delta, config)
        teacher_priority = teacher_weight
        if collect_teacher:
            if policy.factorized_attack_controls:
                disagrees = (move_choice != teacher_move_choice) or (attack_on != teacher_attack_on)
            else:
                disagrees = action_idx != teacher_action
            if disagrees:
                teacher_priority *= (1.0 + config.dagger_priority_disagreement)
            if reward < 0.0:
                teacher_priority *= (
                    1.0 + config.dagger_priority_negative * min(2.0, abs(reward))
                )
            if event == "episode_end":
                teacher_priority *= (1.0 + config.dagger_priority_death)
            if attack_on and event != "kill":
                teacher_priority *= (1.0 + config.dagger_priority_attack_miss)
            attack_window = bool(guidance.get("attack_window", False))
            if attack_window != attack_on:
                teacher_priority *= (1.0 + config.dagger_priority_attack_window)
        teacher_priority = float(min(teacher_priority, 25.0))

        if protocol is not None:
            pos, neg = _deliver_biological_feedback(
                protocol, rb, event, attack_on, neutral_steps
            )
            total_positive += pos
            total_negative += neg
        else:
            rb.run(neutral_steps)

        total_log_prob = action_log_prob
        if config.stim_ablation == "none":
            total_log_prob = total_log_prob + encoder_log_prob

        obs_features_list.append(obs_features.squeeze(0).detach())
        spike_features_list.append(spike_features.squeeze(0).detach())
        stim_currents_list.append(stim_currents.squeeze(0).detach())
        if policy.factorized_attack_controls:
            actions_list.append(action.squeeze(0).detach())
        else:
            actions_list.append(action.detach())
        log_probs_list.append(total_log_prob.detach())
        values_list.append(value.detach())
        rewards_list.append(torch.tensor(reward, device=brain.device))
        dones_list.append(torch.tensor(float(done), device=brain.device))
        if policy.factorized_attack_controls:
            teacher_actions_list.append(
                torch.tensor(
                    [teacher_move_choice, float(teacher_attack_on)],
                    device=brain.device,
                    dtype=torch.float32,
                )
            )
        else:
            teacher_actions_list.append(torch.tensor(teacher_action, device=brain.device))
        teacher_weights_list.append(torch.tensor(teacher_priority, device=brain.device))
        teacher_mask_list.append(torch.tensor(teacher_mask, device=brain.device))
        enemy_visible_list.append(
            torch.tensor(float(bool(guidance.get("enemy_visible", False))), device=brain.device)
        )
        attack_window_list.append(
            torch.tensor(float(bool(guidance.get("attack_window", False))), device=brain.device)
        )

        total_reward += reward
        step_count += 1
        if done:
            break

    if step_count >= max_game_steps and game.is_running and game.episode_survived:
        reward = config.reward_survival
        if protocol is not None:
            protocol.deliver_positive(rb)
            total_positive += 1.0
        total_reward += reward
        rewards_list.append(torch.tensor(reward, device=brain.device))
        dones_list.append(torch.tensor(1.0, device=brain.device))
        obs_features_list.append(obs_features_list[-1].clone())
        spike_features_list.append(spike_features_list[-1].clone())
        stim_currents_list.append(stim_currents_list[-1].clone())
        actions_list.append(actions_list[-1].clone())
        log_probs_list.append(log_probs_list[-1].clone())
        values_list.append(values_list[-1].clone())
        teacher_actions_list.append(teacher_actions_list[-1].clone())
        teacher_weights_list.append(teacher_weights_list[-1].clone())
        teacher_mask_list.append(teacher_mask_list[-1].clone())
        enemy_visible_list.append(enemy_visible_list[-1].clone())
        attack_window_list.append(attack_window_list[-1].clone())

    if not spike_features_list:
        raise RuntimeError("Episode produced no rollout steps.")

    rewards = torch.stack(rewards_list)
    values = torch.stack(values_list).view(-1)
    dones = torch.stack(dones_list).view(-1)
    advantages, returns = _compute_gae(
        rewards, values, dones, config.gamma, config.gae_lambda
    )

    rollout = {
        "obs_features": torch.stack(obs_features_list),
        "spike_features": torch.stack(spike_features_list),
        "stim_currents": torch.stack(stim_currents_list),
        "actions": (
            torch.stack(actions_list)
            if policy.factorized_attack_controls
            else torch.stack(actions_list).view(-1)
        ),
        "log_probs": torch.stack(log_probs_list).view(-1),
        "advantages": advantages.detach(),
        "returns": returns.detach(),
        "teacher_actions": (
            torch.stack(teacher_actions_list)
            if policy.factorized_attack_controls
            else torch.stack(teacher_actions_list).view(-1)
        ),
        "teacher_weights": torch.stack(teacher_weights_list).view(-1),
        "teacher_mask": torch.stack(teacher_mask_list).view(-1),
        "enemy_visible": torch.stack(enemy_visible_list).view(-1),
        "attack_window": torch.stack(attack_window_list).view(-1),
    }

    metrics = {
        "episode_return": total_reward,
        "health_gained": game.episode_health_gained,
        "kills": game.episode_kills,
        "survived": float(game.episode_survived),
        "damage_taken": game.episode_damage_taken,
        "steps": game.episode_steps,
        "positive_events": total_positive,
        "negative_events": total_negative,
        "action_counts": action_counts,
        "teacher_agreement": (
            float(teacher_matches) / float(max(1, teacher_labels))
            if collect_teacher
            else None
        ),
        "teacher_labels": teacher_labels,
    }
    return rollout, metrics


def evaluate_hybrid_policy(
    rb,
    retina,
    bridge,
    decoder,
    relay_ids: torch.Tensor,
    l5_ids: torch.Tensor,
    cortex_ids: torch.Tensor,
    stim_groups: List[torch.Tensor],
    policy: HybridSpikePolicy,
    config: HybridPPOConfig,
    scale_params: Dict[str, Any],
    scenario: str,
    eval_seeds: List[int],
    use_biological_feedback: bool = False,
) -> Dict[str, Any]:
    """Run deterministic held-out evaluation without PPO updates."""
    metric_name = _scenario_positive_metric(scenario)
    metric_label = _metric_label(metric_name)
    eval_metrics: List[Dict[str, Any]] = []

    protocol = _make_feedback_protocol(
        config=config,
        enabled=use_biological_feedback,
        relay_ids=relay_ids,
        l5_ids=l5_ids,
        cortex_ids=cortex_ids,
        stim_groups=stim_groups,
        decoder=decoder,
        scale_params=scale_params,
        device=rb.brain.device,
    )

    print("\n    Held-out deterministic evaluation:")
    for idx, eval_seed in enumerate(eval_seeds, start=1):
        _set_global_seeds(eval_seed)
        game = DoomGame(scenario=scenario, seed=eval_seed, visible=False)
        try:
            _, metrics = play_hybrid_episode(
                rb=rb,
                game=game,
                retina=retina,
                bridge=bridge,
                decoder=decoder,
                stim_groups=stim_groups,
                policy=policy,
                protocol=protocol,
                config=config,
                stim_steps=scale_params["stim_steps"],
                max_game_steps=scale_params["max_game_steps"],
                neutral_steps=scale_params["neutral_steps"],
                deterministic=True,
            )
        finally:
            game.close()

        eval_metrics.append(metrics)
        print(
            f"      Eval {idx:2d}/{len(eval_seeds)} seed {eval_seed}: "
            f"{metric_label} {_format_metric_value(metric_name, _episode_metric(metrics, metric_name))}, "
            f"return {metrics['episode_return']:+.2f}, "
            f"damage -{metrics['damage_taken']:.1f}, kills {metrics['kills']:.2f}"
        )

    summary = _summarize_episode_metrics(eval_metrics, metric_name)
    print(
        f"    Eval avg {metric_label}: {_format_metric_value(metric_name, summary['avg_metric'])}, "
        f"return {summary['avg_return']:+.2f}, "
        f"damage -{summary['avg_damage']:.1f}, kills {summary['avg_kills']:.2f}"
    )
    return {
        "metric_name": metric_name,
        "seeds": eval_seeds,
        "use_biological_feedback": use_biological_feedback,
        "avg_metric": summary["avg_metric"],
        "avg_return": summary["avg_return"],
        "avg_damage": summary["avg_damage"],
        "avg_kills": summary["avg_kills"],
        "episode_metrics": eval_metrics,
    }


def train_hybrid_ppo(
    scale: str,
    scenario: str,
    device: str,
    seed: int,
    n_episodes: int,
    config: HybridPPOConfig,
    eval_episodes: int = 0,
    eval_seed_offset: int = 1000,
    eval_use_biological_feedback: bool = False,
) -> Dict[str, Any]:
    """Train the hybrid PPO readout on top of the digital brain substrate."""
    sp = SCALE_PARAMS.get(scale, SCALE_PARAMS["large"])
    metric_name = _scenario_positive_metric(scenario)
    metric_label = _metric_label(metric_name)

    _header(
        f"Hybrid PPO Doom ({scenario})",
        "PPO readout over digital-brain spike features with optional bio feedback"
    )
    t0 = time.perf_counter()

    rb, retina, bridge, decoder, relay_ids, l5_ids, cortex_ids = _build_doom_brain(
        scale, device, seed
    )
    brain = rb.brain
    dev = brain.device
    obs_dim = (RETINA_HEIGHT // 8) * (RETINA_WIDTH // 8) + N_MOTOR_POPULATIONS + 1
    stim_groups = _make_stim_groups(relay_ids, config.stim_group_count)

    policy = HybridSpikePolicy(
        obs_dim=obs_dim,
        spike_feature_dim=N_MOTOR_POPULATIONS * 2 + 3,
        n_actions=N_MOTOR_POPULATIONS,
        n_stim_groups=len(stim_groups),
        hidden_dim=config.hidden_dim,
        encoder_hidden_dim=config.encoder_hidden_dim,
        stim_current_max=config.stim_current_max,
        zero_bias=config.zero_bias,
        split_attack_head=config.split_attack_head,
        factorized_attack_controls=config.factorized_attack_controls,
        aux_combat_heads=config.aux_combat_heads,
        attack_action=MOTOR_ATTACK,
        freeze_decoder=config.freeze_decoder,
        freeze_encoder=config.freeze_encoder,
    ).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    protocol = _make_feedback_protocol(
        config=config,
        enabled=config.use_biological_feedback,
        relay_ids=relay_ids,
        l5_ids=l5_ids,
        cortex_ids=cortex_ids,
        stim_groups=stim_groups,
        decoder=decoder,
        scale_params=sp,
        device=dev,
    )

    _warmup(rb, n_steps=sp["warmup_steps"])
    print(f"    Brain: {rb.n_neurons} neurons, {rb.n_synapses} synapses on {dev}")
    print(f"    Warmup complete")
    print(
        f"    Policy: obs_dim={obs_dim}, spike_dim={N_MOTOR_POPULATIONS * 2 + 3}, "
        f"stim_groups={len(stim_groups)}, decoder_hidden={config.hidden_dim}, "
        f"encoder_hidden={config.encoder_hidden_dim}, split_attack_head={config.split_attack_head}, "
        f"factorized_attack_controls={config.factorized_attack_controls}, "
        f"aux_heads={config.aux_combat_heads}"
    )
    print(
        f"    Ablations: spike={config.spike_ablation}, stim={config.stim_ablation}, "
        f"freeze_decoder={config.freeze_decoder}, freeze_encoder={config.freeze_encoder}, "
        f"bio_feedback={config.use_biological_feedback}, feedback_style={config.feedback_style}, "
        f"dagger_episodes={config.dagger_episodes}, replay_capacity={config.dagger_replay_capacity}, "
        f"replay_ratio={config.dagger_replay_ratio:.2f}"
    )

    game = DoomGame(scenario=scenario, seed=seed, visible=False)
    report_interval = max(1, n_episodes // 6)

    episode_metrics: List[Dict[str, Any]] = []
    pending_rollouts: List[Dict[str, torch.Tensor]] = []
    dagger_replay = (
        DAggerReplayBuffer(config.dagger_replay_capacity)
        if config.dagger_episodes > 0 and config.dagger_replay_capacity > 0
        else None
    )
    last_update: Dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "action_entropy": 0.0,
        "encoder_entropy": 0.0,
        "imitation_loss": 0.0,
        "imitation_samples": 0.0,
        "replay_samples": 0.0,
        "enemy_loss": 0.0,
        "attack_window_loss": 0.0,
        "attack_gate_loss": 0.0,
    }

    for ep in range(n_episodes):
        collect_teacher = ep < config.dagger_episodes
        rollout, metrics = play_hybrid_episode(
            rb=rb,
            game=game,
            retina=retina,
            bridge=bridge,
            decoder=decoder,
            stim_groups=stim_groups,
            policy=policy,
            protocol=protocol,
            config=config,
            stim_steps=sp["stim_steps"],
            max_game_steps=sp["max_game_steps"],
            neutral_steps=sp["neutral_steps"],
            collect_teacher=collect_teacher,
            deterministic=False,
        )
        episode_metrics.append(metrics)
        pending_rollouts.append(rollout)
        if collect_teacher and dagger_replay is not None:
            dagger_replay.add_rollout(rollout)

        if len(pending_rollouts) >= config.episodes_per_update:
            merged = _merge_rollouts(pending_rollouts)
            last_update = update_policy(policy, optimizer, merged, config)
            if config.dagger_episodes > 0:
                last_update.update(
                    update_imitation_policy(
                        policy,
                        optimizer,
                        merged,
                        config,
                        replay_buffer=dagger_replay,
                    )
                )
            else:
                last_update.setdefault("imitation_loss", 0.0)
                last_update.setdefault("imitation_samples", 0.0)
                last_update.setdefault("replay_samples", 0.0)
            pending_rollouts = []

        if (ep + 1) % report_interval == 0 or ep == n_episodes - 1:
            recent = episode_metrics[max(0, ep - report_interval + 1):ep + 1]
            avg_metric = sum(_episode_metric(m, metric_name) for m in recent) / len(recent)
            avg_return = sum(m["episode_return"] for m in recent) / len(recent)
            avg_damage = sum(m["damage_taken"] for m in recent) / len(recent)
            avg_kills = sum(m["kills"] for m in recent) / len(recent)
            teacher_recent = [m["teacher_agreement"] for m in recent if m["teacher_agreement"] is not None]
            teacher_text = ""
            if teacher_recent:
                avg_teacher = sum(teacher_recent) / len(teacher_recent)
                teacher_text = f", teacher {avg_teacher:.2f}"
            print(
                f"    Episode {ep + 1:3d}/{n_episodes}: "
                f"{metric_label} {_format_metric_value(metric_name, avg_metric)}, "
                f"return {avg_return:+.2f}, damage -{avg_damage:.1f}, kills {avg_kills:.2f}"
                f"{teacher_text}"
            )

    if pending_rollouts:
        merged = _merge_rollouts(pending_rollouts)
        last_update = update_policy(policy, optimizer, merged, config)
        if config.dagger_episodes > 0:
            last_update.update(
                update_imitation_policy(
                    policy,
                    optimizer,
                    merged,
                    config,
                    replay_buffer=dagger_replay,
                )
            )
        else:
            last_update.setdefault("imitation_loss", 0.0)
            last_update.setdefault("imitation_samples", 0.0)
            last_update.setdefault("replay_samples", 0.0)

    game.close()

    quarter = max(1, n_episodes // 4)
    first_q = episode_metrics[:quarter]
    last_q = episode_metrics[-quarter:]
    first_metric = sum(_episode_metric(m, metric_name) for m in first_q) / len(first_q)
    last_metric = sum(_episode_metric(m, metric_name) for m in last_q) / len(last_q)
    first_return = sum(m["episode_return"] for m in first_q) / len(first_q)
    last_return = sum(m["episode_return"] for m in last_q) / len(last_q)
    total_kills = sum(m["kills"] for m in episode_metrics)
    avg_damage = sum(m["damage_taken"] for m in episode_metrics) / len(episode_metrics)
    passed = (last_return > first_return) or (last_metric > first_metric)
    eval_results = None

    if eval_episodes > 0:
        eval_seeds = [seed + eval_seed_offset + i for i in range(eval_episodes)]
        eval_results = evaluate_hybrid_policy(
            rb=rb,
            retina=retina,
            bridge=bridge,
            decoder=decoder,
            relay_ids=relay_ids,
            l5_ids=l5_ids,
            cortex_ids=cortex_ids,
            stim_groups=stim_groups,
            policy=policy,
            config=config,
            scale_params=sp,
            scenario=scenario,
            eval_seeds=eval_seeds,
            use_biological_feedback=eval_use_biological_feedback,
        )

    elapsed = time.perf_counter() - t0

    print("\n    Results:")
    print(f"    First quarter avg {metric_label}: {first_metric:.2f}")
    print(f"    Last quarter avg {metric_label}:  {last_metric:.2f}")
    print(f"    First quarter avg return:         {first_return:+.2f}")
    print(f"    Last quarter avg return:          {last_return:+.2f}")
    print(f"    Total kills:                      {total_kills}")
    print(f"    Avg damage / episode:             {avg_damage:.2f}")
    print(
        f"    Last update losses:               "
        f"pi={last_update['policy_loss']:.4f}, "
        f"vf={last_update['value_loss']:.4f}, "
        f"act_ent={last_update['action_entropy']:.4f}, "
        f"enc_ent={last_update['encoder_entropy']:.4f}, "
        f"imitation={last_update['imitation_loss']:.4f}, "
        f"enemy={last_update['enemy_loss']:.4f}, "
        f"window={last_update['attack_window_loss']:.4f}, "
        f"gate={last_update['attack_gate_loss']:.4f}, "
        f"replay={last_update['replay_samples']:.0f}"
    )
    if eval_results is not None:
        print(
            f"    Held-out eval avg {metric_label}:      "
            f"{_format_metric_value(metric_name, eval_results['avg_metric'])}"
        )
        print(f"    Held-out eval avg return:        {eval_results['avg_return']:+.2f}")
        print(f"    Held-out eval avg damage:        {eval_results['avg_damage']:.2f}")
        print(f"    Held-out eval avg kills:         {eval_results['avg_kills']:.2f}")
    print(f"    {'PASS' if passed else 'FAIL'} in {elapsed:.1f}s")

    return {
        "passed": passed,
        "time": elapsed,
        "metric_name": metric_name,
        "bio_feedback": config.use_biological_feedback,
        "feedback_style": config.feedback_style,
        "spike_ablation": config.spike_ablation,
        "stim_ablation": config.stim_ablation,
        "freeze_decoder": config.freeze_decoder,
        "freeze_encoder": config.freeze_encoder,
        "first_q_metric": first_metric,
        "last_q_metric": last_metric,
        "first_q_return": first_return,
        "last_q_return": last_return,
        "avg_damage": avg_damage,
        "total_kills": total_kills,
        "dagger_replay_size": 0 if dagger_replay is None else len(dagger_replay),
        "last_update": last_update,
        "episode_metrics": episode_metrics,
        "eval_results": eval_results,
    }


def _make_json_safe(obj: Any) -> Any:
    """Convert nested results to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid PPO readout over a digital-brain Doom substrate"
    )
    parser.add_argument("--scale", default="small", choices=list(SCALE_COLUMNS.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario", default="defend_the_center", choices=list(SCENARIOS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--encoder-hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--stim-groups", type=int, default=8)
    parser.add_argument("--stim-current-max", type=float, default=35.0)
    parser.add_argument("--feedback-style", default="dishbrain",
                        choices=["dishbrain", "eligibility_da", "hybrid_bio", "rl"])
    parser.add_argument("--spike-ablation", default="none",
                        choices=["none", "zero", "random"])
    parser.add_argument("--stim-ablation", default="none",
                        choices=["none", "zero", "random"])
    parser.add_argument("--split-attack-head", action="store_true")
    parser.add_argument("--factorized-attack-controls", action="store_true")
    parser.add_argument("--aux-combat-heads", action="store_true")
    parser.add_argument("--aux-head-coef", type=float, default=0.15)
    parser.add_argument("--aux-attack-gate-coef", type=float, default=0.25)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-bio-feedback", action="store_true")
    parser.add_argument("--dagger-episodes", type=int, default=0)
    parser.add_argument("--dagger-epochs", type=int, default=2)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-coef", type=float, default=1.0)
    parser.add_argument("--dagger-replay-capacity", type=int, default=8192)
    parser.add_argument("--dagger-replay-ratio", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument("--eval-bio-feedback", action="store_true")
    parser.add_argument("--json", type=str, default=None, metavar="PATH")
    args = parser.parse_args()

    if not HAS_VIZDOOM:
        print("ERROR: ViZDoom is required. Install with: pip install vizdoom")
        return 1
    if not HAS_PIL:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        return 1

    _set_global_seeds(args.seed)

    print("=" * 76)
    print("  HYBRID PPO DOOM — DIGITAL BRAIN SUBSTRATE")
    print(
        f"  Backend: {detect_backend()} | Scale: {args.scale} | "
        f"Device: {args.device} | Scenario: {args.scenario}"
    )
    print("=" * 76)

    config = HybridPPOConfig(
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        episodes_per_update=args.episodes_per_update,
        stim_group_count=args.stim_groups,
        stim_current_max=args.stim_current_max,
        feedback_style=args.feedback_style,
        spike_ablation=args.spike_ablation,
        stim_ablation=args.stim_ablation,
        split_attack_head=args.split_attack_head,
        factorized_attack_controls=args.factorized_attack_controls,
        aux_combat_heads=args.aux_combat_heads,
        aux_head_coef=args.aux_head_coef,
        aux_attack_gate_coef=args.aux_attack_gate_coef,
        freeze_decoder=args.freeze_decoder,
        freeze_encoder=args.freeze_encoder,
        dagger_episodes=args.dagger_episodes,
        dagger_epochs=args.dagger_epochs,
        dagger_batch_size=args.dagger_batch_size,
        dagger_coef=args.dagger_coef,
        dagger_replay_capacity=args.dagger_replay_capacity,
        dagger_replay_ratio=args.dagger_replay_ratio,
        use_biological_feedback=not args.no_bio_feedback,
    )

    results = train_hybrid_ppo(
        scale=args.scale,
        scenario=args.scenario,
        device=args.device,
        seed=args.seed,
        n_episodes=args.episodes,
        config=config,
        eval_episodes=args.eval_episodes,
        eval_seed_offset=args.eval_seed_offset,
        eval_use_biological_feedback=args.eval_bio_feedback,
    )

    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    status = "PASS" if results.get("passed") else "FAIL"
    print(f"    Hybrid PPO training [{status}]  {results.get('time', 0.0):.1f}s")
    if results.get("eval_results") is not None:
        eval_results = results["eval_results"]
        metric_label = _metric_label(results.get("metric_name", "metric"))
        print(
            f"    Held-out eval {metric_label}: "
            f"{_format_metric_value(results['metric_name'], eval_results['avg_metric'])}, "
            f"return {eval_results['avg_return']:+.2f}"
        )
    print("=" * 76)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_make_json_safe(results), f, indent=2)
        print(f"\n  JSON results written to: {args.json}")

    return 0 if results.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
