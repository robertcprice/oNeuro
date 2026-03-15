//! Membrane and division-state mechanics for the native whole-cell runtime.
//!
//! This module owns the explicit envelope state update path, derived geometry
//! accessors that come from that state, and membrane-specific support signals.

use super::*;

impl WholeCellSimulator {
    pub(super) fn membrane_cardiolipin_share(&self) -> f32 {
        self.organism_data
            .as_ref()
            .map(|organism| (0.08 + 0.40 * organism.composition.lipid_fraction).clamp(0.08, 0.32))
            .unwrap_or(0.16)
    }

    pub(super) fn seeded_membrane_division_state(&self) -> WholeCellMembraneDivisionState {
        let preferred_area_nm2 = self.surface_area_nm2.max(1.0);
        let cardiolipin_share = self.membrane_cardiolipin_share();
        let division_progress = self.current_division_progress();
        let septum_radius_fraction = (1.0 - division_progress).clamp(0.01, 1.0);
        let septum_localization = (0.08 + 0.22 * division_progress).clamp(0.05, 0.60);
        let divisome_order_progress = (0.06 + 0.28 * division_progress).clamp(0.0, 1.0);
        let ring_occupancy = (0.05 + 0.30 * division_progress).clamp(0.0, 1.0);
        let chromosome_occlusion = (1.0 - self.chromosome_state.segregation_progress)
            .clamp(0.0, 1.0)
            * (0.35 + 0.65 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0));
        let membrane_protein_insertion = Self::saturating_signal(
            self.complex_assembly.membrane_complexes
                + 0.35 * self.complex_assembly.atp_band_complexes,
            self.complex_assembly.membrane_target.max(8.0),
        );
        let polar_patch_fraction =
            (0.08 + 0.18 * cardiolipin_share + 0.06 * division_progress).clamp(0.05, 0.28);
        let band_patch_fraction =
            (0.26 + 0.18 * membrane_protein_insertion - 0.08 * division_progress).clamp(0.16, 0.50);
        WholeCellMembraneDivisionState {
            membrane_area_nm2: preferred_area_nm2,
            preferred_membrane_area_nm2: preferred_area_nm2,
            phospholipid_inventory_nm2: preferred_area_nm2 * (1.0 - cardiolipin_share),
            cardiolipin_inventory_nm2: preferred_area_nm2 * cardiolipin_share,
            septal_lipid_inventory_nm2: preferred_area_nm2 * septum_localization * 0.08,
            membrane_band_lipid_inventory_nm2: preferred_area_nm2 * band_patch_fraction,
            polar_lipid_inventory_nm2: preferred_area_nm2 * polar_patch_fraction,
            membrane_protein_insertion,
            insertion_debt: 0.0,
            curvature_stress: (0.10 + 0.25 * division_progress).clamp(0.0, 1.5),
            septum_localization,
            divisome_occupancy: divisome_order_progress * ring_occupancy,
            divisome_order_progress,
            ring_occupancy,
            ring_tension: (0.10 + 0.50 * division_progress).clamp(0.0, 1.5),
            constriction_force: 0.0,
            septum_radius_fraction,
            septum_thickness_nm: (0.018 * self.radius_nm).clamp(3.0, 12.0),
            envelope_integrity: 1.0,
            osmotic_balance: 1.0,
            chromosome_occlusion,
            failure_pressure: 0.0,
            band_turnover_pressure: 0.0,
            pole_turnover_pressure: 0.0,
            septum_turnover_pressure: 0.0,
            scission_events: 0,
        }
    }

    pub(super) fn normalize_membrane_division_state(
        &self,
        mut state: WholeCellMembraneDivisionState,
    ) -> WholeCellMembraneDivisionState {
        let seeded = self.seeded_membrane_division_state();
        state.preferred_membrane_area_nm2 = state
            .preferred_membrane_area_nm2
            .max(seeded.preferred_membrane_area_nm2.max(1.0));
        state.membrane_area_nm2 = state
            .membrane_area_nm2
            .clamp(1.0, state.preferred_membrane_area_nm2 * 1.5);
        state.phospholipid_inventory_nm2 = state.phospholipid_inventory_nm2.max(0.0);
        state.cardiolipin_inventory_nm2 = state.cardiolipin_inventory_nm2.max(0.0);
        state.septal_lipid_inventory_nm2 = state.septal_lipid_inventory_nm2.max(0.0);
        state.membrane_band_lipid_inventory_nm2 = state.membrane_band_lipid_inventory_nm2.max(0.0);
        state.polar_lipid_inventory_nm2 = state.polar_lipid_inventory_nm2.max(0.0);
        state.membrane_protein_insertion = state.membrane_protein_insertion.clamp(0.0, 1.5);
        state.insertion_debt = state.insertion_debt.clamp(0.0, 2.0);
        state.curvature_stress = state.curvature_stress.clamp(0.0, 2.0);
        state.septum_localization = state.septum_localization.clamp(0.0, 1.0);
        state.divisome_occupancy = state.divisome_occupancy.clamp(0.0, 1.0);
        state.divisome_order_progress = state.divisome_order_progress.clamp(0.0, 1.0);
        state.ring_occupancy = state.ring_occupancy.clamp(0.0, 1.0);
        state.ring_tension = state.ring_tension.clamp(0.0, 2.0);
        state.constriction_force = state.constriction_force.clamp(0.0, 2.0);
        state.septum_radius_fraction = state.septum_radius_fraction.clamp(0.01, 1.0);
        state.septum_thickness_nm = state.septum_thickness_nm.clamp(2.0, 20.0);
        state.envelope_integrity = state.envelope_integrity.clamp(0.10, 1.20);
        state.osmotic_balance = state.osmotic_balance.clamp(0.65, 1.35);
        state.chromosome_occlusion = state.chromosome_occlusion.clamp(0.0, 1.5);
        state.failure_pressure = state.failure_pressure.clamp(0.0, 2.0);
        state.band_turnover_pressure = state.band_turnover_pressure.clamp(0.0, 2.0);
        state.pole_turnover_pressure = state.pole_turnover_pressure.clamp(0.0, 2.0);
        state.septum_turnover_pressure = state.septum_turnover_pressure.clamp(0.0, 2.0);
        state
    }

    pub(super) fn synchronize_membrane_division_summary(&mut self) {
        self.membrane_division_state =
            self.normalize_membrane_division_state(self.membrane_division_state.clone());
        self.surface_area_nm2 = self.membrane_division_state.membrane_area_nm2.max(1.0);
        self.radius_nm = (self.surface_area_nm2 / (4.0 * PI)).sqrt();
        self.volume_nm3 =
            Self::volume_from_radius(self.radius_nm) * self.membrane_division_state.osmotic_balance;
        self.division_progress =
            (1.0 - self.membrane_division_state.septum_radius_fraction).clamp(0.0, 0.99);
    }

    pub(super) fn current_surface_area_nm2(&self) -> f32 {
        if self.membrane_division_state.membrane_area_nm2 > 1.0 {
            self.membrane_division_state.membrane_area_nm2.max(1.0)
        } else {
            self.surface_area_nm2.max(1.0)
        }
    }

    pub(super) fn current_radius_nm(&self) -> f32 {
        (self.current_surface_area_nm2() / (4.0 * PI)).sqrt()
    }

    pub(super) fn current_volume_nm3(&self) -> f32 {
        if self.membrane_division_state.preferred_membrane_area_nm2 > 1.0
            || self.membrane_division_state.membrane_area_nm2 > 1.0
        {
            Self::volume_from_radius(self.current_radius_nm())
                * self.membrane_division_state.osmotic_balance.max(0.1)
        } else {
            self.volume_nm3.max(1.0)
        }
    }

    pub(super) fn current_division_progress(&self) -> f32 {
        if self.membrane_division_state.preferred_membrane_area_nm2 > 1.0
            || self.membrane_division_state.membrane_area_nm2 > 1.0
        {
            (1.0 - self.membrane_division_state.septum_radius_fraction).clamp(0.0, 0.99)
        } else {
            self.division_progress.clamp(0.0, 0.99)
        }
    }

    pub(super) fn membrane_chromosome_occlusion(&self) -> f32 {
        ((1.0 - self.chromosome_state.segregation_progress).clamp(0.0, 1.0)
            * (0.45 + 0.55 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
            * (0.70 + 0.30 * self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)))
        .clamp(0.0, 1.5)
    }

    pub(super) fn update_membrane_division_state(
        &mut self,
        dt: f32,
        membrane_growth_nm2: f32,
        constriction_flux: f32,
        constriction_drive: f32,
    ) {
        self.refresh_spatial_fields();
        self.refresh_rdme_drive_fields();
        let inventory = self.assembly_inventory();
        let mut state =
            self.normalize_membrane_division_state(self.membrane_division_state.clone());
        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let membrane_support =
            Self::finite_scale(self.organism_expression.membrane_support, 1.0, 0.55, 1.55);
        let constriction_support = Self::finite_scale(
            self.organism_expression.process_scales.constriction,
            1.0,
            0.55,
            1.55,
        );
        let membrane_process_scale = Self::finite_scale(
            self.organism_expression.process_scales.membrane,
            1.0,
            0.55,
            1.55,
        );
        let constriction_process_scale = Self::finite_scale(
            self.organism_expression.process_scales.constriction,
            1.0,
            0.55,
            1.55,
        );
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let cardiolipin_share = self.membrane_cardiolipin_share();
        let local_membrane_precursors =
            Self::saturating_signal(self.localized_membrane_precursor_pool_mm(), 0.6);
        let local_membrane_atp =
            Self::saturating_signal(self.localized_membrane_atp_pool_mm(), 1.0);
        let local_band_precursors =
            Self::saturating_signal(self.localized_membrane_band_precursor_pool_mm(), 0.45);
        let local_polar_precursors =
            Self::saturating_signal(self.localized_polar_precursor_pool_mm(), 0.35);
        let local_band_atp = Self::saturating_signal(
            self.spatial_species_mean(
                IntracellularSpecies::ATP,
                IntracellularSpatialField::MembraneBandZone,
            ),
            0.8,
        );
        let local_polar_atp = Self::saturating_signal(
            self.spatial_species_mean(
                IntracellularSpecies::ATP,
                IntracellularSpatialField::PoleZone,
            ),
            0.8,
        );
        let band_membrane_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_membrane_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::PoleZone,
        );
        let septum_membrane_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::SeptumZone,
        );
        let band_membrane_demand = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneDemand,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_membrane_demand = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneDemand,
            IntracellularSpatialField::PoleZone,
        );
        let septum_membrane_demand = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneDemand,
            IntracellularSpatialField::SeptumZone,
        );
        let band_crowding = self.localized_drive_mean(
            WholeCellRdmeDriveField::Crowding,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_crowding = self.localized_drive_mean(
            WholeCellRdmeDriveField::Crowding,
            IntracellularSpatialField::PoleZone,
        );
        let septum_crowding = self.localized_drive_mean(
            WholeCellRdmeDriveField::Crowding,
            IntracellularSpatialField::SeptumZone,
        );
        let membrane_complex_signal = Self::saturating_signal(
            inventory.membrane_complexes + 0.35 * inventory.atp_band_complexes,
            inventory.membrane_target.max(8.0),
        );
        let ring_component_signal = Self::saturating_signal(
            inventory.ftsz_polymer + 0.40 * inventory.membrane_complexes,
            inventory.ftsz_target.max(12.0),
        );
        let occlusion = self.membrane_chromosome_occlusion();
        let occlusion_gate = (1.0 - 0.75 * occlusion).clamp(0.05, 1.0);
        let protein_insertion_support = Self::finite_scale(
            0.42 * membrane_support
                + 0.20 * membrane_process_scale
                + 0.28 * membrane_complex_signal
                + 0.10 * local_membrane_precursors
                + 0.08 * local_membrane_atp
                + 0.10 * local_band_precursors
                + 0.08 * band_membrane_source
                + 0.10 * self.md_membrane_scale.max(0.0)
                + 0.10 * self.quantum_profile.membrane_synthesis_efficiency,
            1.0,
            0.25,
            1.85,
        );
        let membrane_growth_efficiency = Self::finite_scale(
            0.46 * membrane_support
                + 0.24 * protein_insertion_support
                + 0.10 * local_membrane_precursors
                + 0.08 * local_band_precursors
                + 0.15 * crowding
                + 0.10 * self.quantum_profile.membrane_synthesis_efficiency,
            1.0,
            0.25,
            1.90,
        );
        state.band_turnover_pressure = (0.55 * state.band_turnover_pressure
            + 0.45
                * (0.52 * band_membrane_demand
                    + 0.18 * band_crowding
                    + 0.16 * state.insertion_debt
                    - 0.22 * local_band_precursors
                    - 0.12 * band_membrane_source))
            .clamp(0.0, 2.0);
        state.pole_turnover_pressure = (0.55 * state.pole_turnover_pressure
            + 0.45
                * (0.48 * pole_membrane_demand
                    + 0.16 * pole_crowding
                    + 0.14 * state.curvature_stress
                    - 0.20 * local_polar_precursors
                    - 0.10 * pole_membrane_source))
            .clamp(0.0, 2.0);
        state.septum_turnover_pressure = (0.52 * state.septum_turnover_pressure
            + 0.48
                * (0.58 * septum_membrane_demand
                    + 0.18 * septum_crowding
                    + 0.16 * state.constriction_force
                    - 0.18 * septum_membrane_source))
            .clamp(0.0, 2.0);
        let phospholipid_supply =
            membrane_growth_nm2.max(0.0) * membrane_growth_efficiency * (1.0 - cardiolipin_share);
        let cardiolipin_supply = membrane_growth_nm2.max(0.0)
            * membrane_growth_efficiency
            * cardiolipin_share
            * (0.75 + 0.25 * state.curvature_stress);
        state.phospholipid_inventory_nm2 = (state.phospholipid_inventory_nm2
            + dt_scale
                * (phospholipid_supply
                    - 0.020 * state.septal_lipid_inventory_nm2
                    - 0.015 * state.insertion_debt * state.preferred_membrane_area_nm2))
            .max(0.0);
        state.cardiolipin_inventory_nm2 = (state.cardiolipin_inventory_nm2
            + dt_scale
                * (cardiolipin_supply
                    - 0.008 * state.septal_lipid_inventory_nm2
                    - 0.006 * state.curvature_stress * state.cardiolipin_inventory_nm2))
            .max(0.0);
        state.membrane_band_lipid_inventory_nm2 = (state.membrane_band_lipid_inventory_nm2
            + dt_scale
                * ((0.032 * state.phospholipid_inventory_nm2
                    + 0.022 * state.cardiolipin_inventory_nm2)
                    * (0.55 + 0.45 * local_band_precursors)
                    * (0.55 + 0.45 * local_band_atp)
                    - 0.022 * state.membrane_band_lipid_inventory_nm2
                    - 0.018
                        * state.band_turnover_pressure
                        * state.membrane_band_lipid_inventory_nm2))
            .max(0.0);
        state.polar_lipid_inventory_nm2 = (state.polar_lipid_inventory_nm2
            + dt_scale
                * ((0.024 * state.phospholipid_inventory_nm2
                    + 0.034 * state.cardiolipin_inventory_nm2)
                    * (0.55 + 0.45 * local_polar_precursors)
                    * (0.55 + 0.45 * local_polar_atp)
                    - 0.018 * state.polar_lipid_inventory_nm2
                    - 0.016 * state.pole_turnover_pressure * state.polar_lipid_inventory_nm2))
            .max(0.0);
        state.membrane_protein_insertion = (state.membrane_protein_insertion
            + dt_scale
                * (0.060 * protein_insertion_support
                    + 0.020 * local_band_atp
                    + 0.018 * local_band_precursors
                    - state.membrane_protein_insertion * (0.030 + 0.020 * state.failure_pressure)))
            .clamp(0.0, 1.5);
        state.insertion_debt = (state.insertion_debt
            + dt_scale
                * (0.060 * (1.0 - state.membrane_protein_insertion)
                    + 0.040 * membrane_complex_signal
                    + 0.025 * state.band_turnover_pressure
                    - 0.070 * protein_insertion_support))
            .clamp(0.0, 2.0);
        state.septum_localization = (state.septum_localization
            + dt_scale
                * (0.090 * state.membrane_protein_insertion + 0.080 * ring_component_signal
                    - 0.045 * state.septum_localization * (1.0 + occlusion)))
            .clamp(0.0, 1.0);
        state.divisome_order_progress = (state.divisome_order_progress
            + dt_scale
                * (0.100 * state.septum_localization * membrane_process_scale
                    + 0.080 * protein_insertion_support
                    + 0.060 * membrane_complex_signal * membrane_process_scale
                    - 0.040 * state.divisome_order_progress * (1.0 + state.failure_pressure)))
            .clamp(0.0, 1.0);
        state.divisome_occupancy = (state.divisome_occupancy
            + dt_scale
                * (0.110
                    * state.divisome_order_progress
                    * ring_component_signal
                    * constriction_process_scale
                    * occlusion_gate
                    - 0.038 * state.divisome_occupancy * (1.0 + state.failure_pressure)))
            .clamp(0.0, 1.0);
        state.ring_occupancy = (state.ring_occupancy
            + dt_scale
                * (0.130
                    * ring_component_signal
                    * state.divisome_order_progress
                    * constriction_process_scale
                    * occlusion_gate
                    - 0.040 * state.ring_occupancy * (1.0 + state.failure_pressure)))
            .clamp(0.0, 1.0);
        state.curvature_stress = (0.62 * state.curvature_stress
            + 0.38
                * ((1.0 - state.septum_radius_fraction)
                    + 0.35 * (1.0 - crowding)
                    + 0.22 * state.insertion_debt
                    - 0.18
                        * Self::saturating_signal(
                            state.cardiolipin_inventory_nm2,
                            state.preferred_membrane_area_nm2.max(1.0) * cardiolipin_share,
                        )))
        .clamp(0.0, 2.0);
        state.ring_tension = Self::finite_scale(
            0.42 * constriction_support
                + 0.28 * state.ring_occupancy
                + 0.20 * state.divisome_occupancy
                + 0.10 * self.quantum_profile.translation_efficiency,
            1.0,
            0.15,
            2.0,
        );
        let constriction_signal = Self::saturating_signal(
            constriction_drive.max(0.0) + 6.0 * constriction_flux.max(0.0),
            0.35,
        );
        state.constriction_force = constriction_signal
            * state.divisome_occupancy
            * state.ring_tension
            * constriction_process_scale
            * state.envelope_integrity
            * occlusion_gate;
        let septal_supply = (0.020 * state.phospholipid_inventory_nm2
            + 0.018 * state.cardiolipin_inventory_nm2 * (0.70 + 0.30 * state.curvature_stress))
            * state.septum_localization
            / state.preferred_membrane_area_nm2.max(1.0);
        state.septal_lipid_inventory_nm2 = (state.septal_lipid_inventory_nm2
            + dt_scale
                * (septal_supply * state.preferred_membrane_area_nm2
                    + 0.030 * septum_membrane_source * state.preferred_membrane_area_nm2
                    - 0.020 * state.septal_lipid_inventory_nm2
                    - 0.016 * state.septum_turnover_pressure * state.septal_lipid_inventory_nm2
                    - 0.010 * state.constriction_force * state.septal_lipid_inventory_nm2))
            .max(0.0);
        let previous_septum_radius_fraction = state.septum_radius_fraction;
        state.septum_radius_fraction = (state.septum_radius_fraction
            - dt_scale
                * (0.12
                    * state.constriction_force
                    * membrane_process_scale
                    * constriction_process_scale
                    * (0.55 + 0.45 * state.septum_localization)
                    * (0.55 + 0.45 * state.divisome_order_progress))
            + dt_scale * 0.0025 * occlusion)
            .clamp(0.01, 1.0);
        state.septum_thickness_nm = (state.septum_thickness_nm
            + dt_scale
                * (0.040 * state.septum_localization
                    + 0.030
                        * Self::saturating_signal(
                            state.septal_lipid_inventory_nm2,
                            0.08 * state.preferred_membrane_area_nm2.max(1.0),
                        )
                    - 0.020 * state.constriction_force))
            .clamp(2.0, 20.0);
        state.failure_pressure = (0.34 * state.curvature_stress
            + 0.24 * state.insertion_debt
            + 0.18 * occlusion
            + 0.10 * state.band_turnover_pressure
            + 0.08 * state.pole_turnover_pressure
            + 0.12 * state.septum_turnover_pressure
            + 0.14 * (1.0 - crowding).max(0.0)
            + 0.10 * (1.0 - state.envelope_integrity).max(0.0))
        .clamp(0.0, 2.0);
        state.envelope_integrity = (state.envelope_integrity
            + dt_scale
                * (0.028 * state.membrane_protein_insertion
                    + 0.014
                        * Self::saturating_signal(
                            state.membrane_band_lipid_inventory_nm2,
                            0.18 * state.preferred_membrane_area_nm2.max(1.0),
                        )
                    + 0.012
                        * Self::saturating_signal(
                            state.polar_lipid_inventory_nm2,
                            0.10 * state.preferred_membrane_area_nm2.max(1.0),
                        )
                    + 0.020
                        * Self::saturating_signal(
                            state.septal_lipid_inventory_nm2,
                            0.06 * state.preferred_membrane_area_nm2.max(1.0),
                        )
                    - 0.030 * state.failure_pressure
                    - 0.012 * state.constriction_force * state.curvature_stress))
            .clamp(0.10, 1.20);
        state.osmotic_balance = (0.72
            + 0.16 * Self::saturating_signal(self.atp_mm + self.glucose_mm, 1.8)
            + 0.12 * state.envelope_integrity
            - 0.10 * state.failure_pressure)
            .clamp(0.65, 1.35);
        state.preferred_membrane_area_nm2 = (state.preferred_membrane_area_nm2
            + dt_scale * membrane_growth_nm2.max(0.0) * membrane_growth_efficiency)
            .max(1.0);
        state.membrane_area_nm2 = (state.membrane_area_nm2
            + dt_scale
                * (membrane_growth_nm2.max(0.0) * membrane_growth_efficiency
                    - 0.025 * state.constriction_force * state.curvature_stress))
            .clamp(1.0, state.preferred_membrane_area_nm2 * 1.20);
        state.chromosome_occlusion = occlusion;
        if previous_septum_radius_fraction > 0.06
            && state.septum_radius_fraction <= 0.06
            && state.divisome_occupancy > 0.60
            && state.envelope_integrity > 0.70
        {
            state.scission_events = state.scission_events.saturating_add(1);
        }
        self.membrane_division_state = self.normalize_membrane_division_state(state);
        self.synchronize_membrane_division_summary();
        self.synchronize_chromosome_summary();
        self.refresh_spatial_fields();
    }
}
