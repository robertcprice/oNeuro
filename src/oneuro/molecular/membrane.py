"""Molecular membrane — THE CORE.

MolecularMembrane composes an IonChannelEnsemble with SynapticReceptors.
Membrane potential is NOT a settable float — it EMERGES from ion channel physics.

The update loop:
  1. Apply neurotransmitter concentrations to synaptic receptors
  2. Receptors open/close ion channels
  3. Feed metabotropic cascade signals to second messenger system
  4. Compute total ionic current from all channels
  5. dV/dt = (-I_ion + I_external) / C_m
  6. Detect action potential threshold crossing
  7. Update calcium system (multi-compartment) and second messengers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from oneuro.molecular.ion_channels import IonChannelEnsemble, IonChannelType
from oneuro.molecular.receptors import SynapticReceptor, ReceptorType


# Standard membrane capacitance (uF/cm²)
DEFAULT_C_M = 1.0

# Resting potential emerges from channel equilibria, but we initialize here
INITIAL_VOLTAGE = -65.0

# Action potential detection threshold (mV)
AP_THRESHOLD = -20.0


@dataclass
class MolecularMembrane:
    """A biophysical membrane whose voltage emerges from ion channel dynamics.

    The membrane potential is a read-only property that results from the
    balance of all ionic currents. It cannot be directly set.

    Optionally integrates:
    - CalciumSystem (multi-compartment Ca²⁺ with ER stores)
    - SecondMessengerSystem (cAMP/PKA/PKC/CaMKII/CREB cascades)
    """

    channels: IonChannelEnsemble = field(default_factory=IonChannelEnsemble.standard_hh)
    receptors: List[SynapticReceptor] = field(default_factory=list)
    C_m: float = DEFAULT_C_M  # Membrane capacitance (uF/cm²)

    # Internal state
    _voltage: float = field(init=False, default=INITIAL_VOLTAGE)
    _prev_voltage: float = field(init=False, default=INITIAL_VOLTAGE)
    _fired: bool = field(init=False, default=False)
    _refractory_timer: float = field(init=False, default=0.0)
    _refractory_period: float = 2.0  # ms
    _spike_count: int = field(init=False, default=0)
    _time: float = field(init=False, default=0.0)

    # Intracellular calcium — simple float for backward compat, or CalciumSystem
    _ca_internal_nM: float = field(init=False, default=50.0)  # Resting ~50-100 nM

    # Optional advanced subsystems (None = use simple model)
    _calcium_system: Optional[object] = field(init=False, default=None)
    _second_messenger_system: Optional[object] = field(init=False, default=None)

    # Phosphorylation state from second messengers → modifies channel conductances
    _phosphorylation: Dict[str, float] = field(init=False, default_factory=dict)

    # ATP availability from metabolism — set by neuron/network before each step
    _atp_ok: bool = field(init=False, default=True)

    @property
    def voltage(self) -> float:
        """Membrane potential in mV. Read-only — emerges from channel physics."""
        return self._voltage

    @property
    def fired(self) -> bool:
        """Whether an action potential was detected on the last step."""
        return self._fired

    @property
    def spike_count(self) -> int:
        return self._spike_count

    @property
    def ca_internal(self) -> float:
        """Intracellular calcium concentration in nM."""
        if self._calcium_system is not None:
            return self._calcium_system.cytoplasmic_nM
        return self._ca_internal_nM

    @property
    def ca_microdomain(self) -> float:
        """Near-channel microdomain Ca²⁺ in nM."""
        if self._calcium_system is not None:
            return self._calcium_system.microdomain_nM
        return self._ca_internal_nM

    @property
    def camkii_activation(self) -> float:
        """CaMKII activation level [0, 1] from calcium system."""
        if self._calcium_system is not None:
            return self._calcium_system.camkii_activation
        # Simple fallback: threshold-based
        return min(1.0, max(0.0, (self._ca_internal_nM - 300.0) / 2000.0))

    @property
    def is_refractory(self) -> bool:
        return self._refractory_timer > 0

    def set_calcium_system(self, calcium_system) -> None:
        """Attach a CalciumSystem for multi-compartment Ca²⁺ modeling."""
        self._calcium_system = calcium_system

    def set_second_messenger_system(self, sms) -> None:
        """Attach a SecondMessengerSystem for intracellular signaling cascades."""
        self._second_messenger_system = sms

    def step(
        self,
        dt: float,
        nt_concentrations: Optional[Dict[str, float]] = None,
        external_current: float = 0.0,
    ) -> bool:
        """Advance membrane state by dt milliseconds.

        Args:
            dt: Timestep in ms.
            nt_concentrations: Dict of NT name -> concentration in nM.
            external_current: Injected current in uA/cm².

        Returns:
            True if an action potential was fired during this step.
        """
        self._fired = False
        self._time += dt

        # 1. Apply NT concentrations to receptors → update channel conductances
        cascade_signals = {}
        if nt_concentrations:
            cascade_signals = self._apply_neurotransmitters(nt_concentrations, dt)

        # 2. Transfer ionotropic receptor activation to channel ensemble
        self._receptor_to_channel_coupling()

        # 3. Feed metabotropic cascade signals to second messenger system
        if self._second_messenger_system is not None and cascade_signals:
            ca_for_sms = self.ca_microdomain if self._calcium_system else self._ca_internal_nM
            self._second_messenger_system.step(dt, cascade_signals, ca_for_sms)
            # Apply phosphorylation effects to channels
            self._apply_phosphorylation()

        # 4. Update voltage-gated channel kinetics
        self.channels.update_all(self._voltage, dt)

        # 5. Compute total ionic current
        I_ion = self.channels.total_current(self._voltage)

        # 6. Integrate membrane equation: C_m * dV/dt = -I_ion + I_ext
        self._prev_voltage = self._voltage
        dV = (-I_ion + external_current) / self.C_m * dt
        self._voltage += dV

        # 7. Clamp voltage to physiological range
        self._voltage = max(-100.0, min(60.0, self._voltage))

        # 8. Action potential detection
        if self._refractory_timer > 0:
            self._refractory_timer -= dt
        elif self._prev_voltage < AP_THRESHOLD <= self._voltage:
            self._fired = True
            self._spike_count += 1
            self._refractory_timer = self._refractory_period

            # Spike-triggered Ca2+ influx
            if self._calcium_system is not None:
                self._calcium_system.spike_influx()
            else:
                self._ca_internal_nM += 500.0

        # 9. Calcium dynamics
        if self._calcium_system is not None:
            # Voltage-dependent Ca²⁺ influx through Ca_v channels
            ca_ch = self.channels.get_channel(IonChannelType.Ca_v)
            if ca_ch is not None:
                ca_current = ca_ch.current(self._voltage)
                # Convert current to Ca²⁺ influx (negative current = inward = Ca²⁺ in)
                if ca_current < 0:
                    ca_influx_nM = abs(ca_current) * 10.0 * dt  # Scaling factor
                    self._calcium_system.add_channel_influx(ca_influx_nM)

            # Get IP3 level from second messengers for ER release
            ip3 = 0.0
            if self._second_messenger_system is not None:
                ip3 = getattr(self._second_messenger_system, 'ip3_level', 0.0)

            self._calcium_system.step(dt, ip3_level=ip3, atp_available=self._atp_ok)
        else:
            # Simple calcium dynamics: exponential decay toward resting
            ca_rest = 50.0
            ca_tau = 50.0  # ms
            self._ca_internal_nM += dt * (ca_rest - self._ca_internal_nM) / ca_tau
            self._ca_internal_nM = max(0.0, self._ca_internal_nM)

        return self._fired

    def _apply_neurotransmitters(self, nt_concs: Dict[str, float],
                                  dt: float) -> Dict[str, float]:
        """Route NT concentrations to the appropriate receptors.

        Returns dict of cascade_effect → activation level for metabotropic receptors.
        """
        cascade_signals: Dict[str, float] = {}
        for receptor in self.receptors:
            nt_name = receptor.neurotransmitter
            conc = nt_concs.get(nt_name, 0.0)
            receptor.update(conc, dt)

            # Collect metabotropic cascade signals
            if not receptor.is_ionotropic and receptor.cascade_signal > 0.01:
                effect = receptor._props.get("cascade_effect", "")
                if effect:
                    cascade_signals[effect] = max(
                        cascade_signals.get(effect, 0.0),
                        receptor.cascade_signal,
                    )

        return cascade_signals

    def _receptor_to_channel_coupling(self) -> None:
        """Transfer ionotropic receptor activation to ion channels."""
        for receptor in self.receptors:
            if receptor.is_ionotropic and receptor.channel_type is not None:
                ch = self.channels.get_channel(receptor.channel_type)
                if ch is not None:
                    ch.ligand_open_fraction = receptor.activation

    def _apply_phosphorylation(self) -> None:
        """Apply second messenger phosphorylation effects to channels."""
        if self._second_messenger_system is None:
            return

        phos = getattr(self._second_messenger_system, 'phosphorylation_state', None)
        if phos is None:
            return

        # PKA phosphorylation of AMPA → increased conductance
        ampa_p = getattr(phos, 'AMPA_p', 0.0)
        if ampa_p > 0.01:
            ch = self.channels.get_channel(IonChannelType.AMPA)
            if ch is not None:
                ch.conductance_scale = 1.0 + ampa_p * 0.5  # Up to 50% enhancement

        # PKA phosphorylation of K_v → reduced conductance (increases excitability)
        kv_p = getattr(phos, 'Kv_p', 0.0)
        if kv_p > 0.01:
            ch = self.channels.get_channel(IonChannelType.K_v)
            if ch is not None:
                ch.conductance_scale = 1.0 - kv_p * 0.3  # Up to 30% reduction

        # PKC phosphorylation of Ca_v → increased conductance
        cav_p = getattr(phos, 'Cav_p', 0.0)
        if cav_p > 0.01:
            ch = self.channels.get_channel(IonChannelType.Ca_v)
            if ch is not None:
                ch.conductance_scale = 1.0 + cav_p * 0.3

    def add_receptor(self, receptor_type: ReceptorType, count: int = 1) -> None:
        """Add synaptic receptors to this membrane."""
        self.receptors.append(SynapticReceptor(receptor_type=receptor_type, count=count))
        # Ensure corresponding ion channel exists for ionotropic receptors
        props = self.receptors[-1]._props
        if props.get("ionotropic") and props.get("channel_type"):
            ch_type = props["channel_type"]
            if ch_type not in self.channels.channels:
                self.channels.add_channel(ch_type, count=1)

    def remove_receptors(self, receptor_type: ReceptorType, count: int = 1) -> None:
        """Remove receptors (for LTD / receptor trafficking)."""
        for receptor in self.receptors:
            if receptor.receptor_type == receptor_type:
                receptor.count = max(0, receptor.count - count)
                return

    def get_receptor(self, receptor_type: ReceptorType) -> Optional[SynapticReceptor]:
        """Find a receptor by type."""
        for r in self.receptors:
            if r.receptor_type == receptor_type:
                return r
        return None

    def receptor_count(self, receptor_type: ReceptorType) -> int:
        """Total receptor count for a type."""
        return sum(
            r.count for r in self.receptors if r.receptor_type == receptor_type
        )

    @classmethod
    def excitatory(cls) -> "MolecularMembrane":
        """Create an excitatory neuron membrane (glutamatergic postsynaptic)."""
        membrane = cls(channels=IonChannelEnsemble.excitatory_postsynaptic())
        membrane.add_receptor(ReceptorType.AMPA, count=50)
        membrane.add_receptor(ReceptorType.NMDA, count=20)
        return membrane

    @classmethod
    def inhibitory(cls) -> "MolecularMembrane":
        """Create an inhibitory neuron membrane (GABAergic postsynaptic)."""
        membrane = cls(channels=IonChannelEnsemble.inhibitory_postsynaptic())
        membrane.add_receptor(ReceptorType.GABA_A, count=40)
        return membrane

    @classmethod
    def cholinergic(cls) -> "MolecularMembrane":
        """Create a cholinergic postsynaptic membrane."""
        membrane = cls(channels=IonChannelEnsemble.standard_hh())
        membrane.channels.add_channel(IonChannelType.nAChR, count=1)
        membrane.add_receptor(ReceptorType.nAChR, count=30)
        membrane.add_receptor(ReceptorType.mAChR_M1, count=10)
        return membrane
