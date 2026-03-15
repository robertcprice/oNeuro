//! Chromosome and polymer-state mechanics for the native whole-cell runtime.
//!
//! This module owns chromosome-domain lookup, fork/locus state maintenance,
//! and explicit chromosome progression so the main runtime file does not have
//! to carry polymer-state logic inline.

use super::*;

impl WholeCellSimulator {
    pub(super) fn circular_distance_bp(a: f32, b: f32, genome_bp: f32) -> f32 {
        let delta = (a - b).abs();
        delta.min((genome_bp - delta).abs())
    }

    pub(super) fn circular_position_bp(position_bp: i64, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1) as i64;
        position_bp.rem_euclid(genome_bp) as u32
    }

    pub(super) fn clockwise_distance_bp(origin_bp: u32, target_bp: u32, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1);
        if target_bp >= origin_bp {
            target_bp - origin_bp
        } else {
            genome_bp - (origin_bp - target_bp)
        }
    }

    pub(super) fn counter_clockwise_distance_bp(
        origin_bp: u32,
        target_bp: u32,
        genome_bp: u32,
    ) -> u32 {
        Self::clockwise_distance_bp(target_bp, origin_bp, genome_bp)
    }

    pub(super) fn chromosome_position_from_origin(
        origin_bp: u32,
        traveled_bp: u32,
        direction: WholeCellChromosomeForkDirection,
        genome_bp: u32,
    ) -> u32 {
        let traveled_bp = traveled_bp.min(genome_bp.max(1));
        match direction {
            WholeCellChromosomeForkDirection::Clockwise => {
                Self::circular_position_bp(origin_bp as i64 + traveled_bp as i64, genome_bp)
            }
            WholeCellChromosomeForkDirection::CounterClockwise => {
                Self::circular_position_bp(origin_bp as i64 - traveled_bp as i64, genome_bp)
            }
        }
    }

    pub(super) fn chromosome_arm_length(
        origin_bp: u32,
        terminus_bp: u32,
        direction: WholeCellChromosomeForkDirection,
        genome_bp: u32,
    ) -> u32 {
        match direction {
            WholeCellChromosomeForkDirection::Clockwise => {
                Self::clockwise_distance_bp(origin_bp, terminus_bp, genome_bp).max(1)
            }
            WholeCellChromosomeForkDirection::CounterClockwise => {
                Self::counter_clockwise_distance_bp(origin_bp, terminus_bp, genome_bp).max(1)
            }
        }
    }

    pub(super) fn compiled_chromosome_domains(&self) -> Option<&[WholeCellChromosomeDomainSpec]> {
        self.organism_data
            .as_ref()
            .and_then(|organism| {
                (!organism.chromosome_domains.is_empty())
                    .then_some(organism.chromosome_domains.as_slice())
            })
            .or_else(|| {
                self.organism_assets.as_ref().and_then(|assets| {
                    (!assets.chromosome_domains.is_empty())
                        .then_some(assets.chromosome_domains.as_slice())
                })
            })
            .or_else(|| {
                self.organism_process_registry
                    .as_ref()
                    .and_then(|registry| {
                        (!registry.chromosome_domains.is_empty())
                            .then_some(registry.chromosome_domains.as_slice())
                    })
            })
    }

    pub(super) fn chromosome_domain_index(&self, midpoint_bp: u32, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1);
        if let Some(domains) = self.compiled_chromosome_domains() {
            if let Some((index, _)) = domains.iter().enumerate().find(|(_, domain)| {
                let start_bp = domain.start_bp.min(genome_bp.saturating_sub(1));
                let end_bp = domain.end_bp.min(genome_bp.saturating_sub(1));
                let (start_bp, end_bp) = if start_bp <= end_bp {
                    (start_bp, end_bp)
                } else {
                    (end_bp, start_bp)
                };
                midpoint_bp >= start_bp && midpoint_bp <= end_bp
            }) {
                return index as u32;
            }
            if let Some((index, _)) = domains.iter().enumerate().min_by(|(_, left), (_, right)| {
                let left_center_bp =
                    (left.axial_center_fraction.clamp(0.0, 1.0) * genome_bp as f32).round() as u32;
                let right_center_bp =
                    (right.axial_center_fraction.clamp(0.0, 1.0) * genome_bp as f32).round() as u32;
                left_center_bp
                    .abs_diff(midpoint_bp)
                    .cmp(&right_center_bp.abs_diff(midpoint_bp))
            }) {
                return index as u32;
            }
        }
        0
    }

    pub(super) fn chromosome_domain_count(&self) -> usize {
        self.compiled_chromosome_domains()
            .map(|domains| domains.len().max(1))
            .unwrap_or(1)
    }

    pub(super) fn chromosome_domain_index_by_id(&self, domain_id: &str) -> Option<u32> {
        self.compiled_chromosome_domains().and_then(|domains| {
            domains
                .iter()
                .position(|domain| domain.id == domain_id)
                .map(|index| index as u32)
        })
    }

    pub(super) fn chromosome_domain_value(values: &[f32], domain_index: usize) -> f32 {
        values
            .get(domain_index)
            .copied()
            .or_else(|| values.last().copied())
            .unwrap_or(1.0)
    }

    pub(super) fn chromosome_locus_records(
        &self,
        organism: &WholeCellOrganismSpec,
    ) -> Vec<(String, u32, i8)> {
        let mut loci: Vec<(String, u32, i8)> = organism
            .genes
            .iter()
            .map(|feature| {
                (
                    feature.gene.clone(),
                    ((feature.start_bp as u64 + feature.end_bp as u64) / 2) as u32,
                    feature.strand,
                )
            })
            .collect();
        if loci.is_empty() {
            loci = organism
                .transcription_units
                .iter()
                .enumerate()
                .map(|(index, unit)| {
                    let midpoint = ((index + 1) as f32
                        * organism.chromosome_length_bp.max(1) as f32
                        / (organism.transcription_units.len().max(1) + 1) as f32)
                        .round() as u32;
                    (unit.name.clone(), midpoint, 1)
                })
                .collect();
        }
        loci.sort_by_key(|(_, midpoint, _)| *midpoint);
        loci
    }

    pub(super) fn chromosome_forks_from_progress(
        genome_bp: u32,
        origin_bp: u32,
        terminus_bp: u32,
        replicated_bp: u32,
    ) -> Vec<WholeCellChromosomeForkState> {
        if replicated_bp == 0 {
            return Vec::new();
        }
        let genome_bp = genome_bp.max(1);
        let replicated_bp = replicated_bp.min(genome_bp);
        let arm_cw = Self::chromosome_arm_length(
            origin_bp,
            terminus_bp,
            WholeCellChromosomeForkDirection::Clockwise,
            genome_bp,
        );
        let arm_ccw = Self::chromosome_arm_length(
            origin_bp,
            terminus_bp,
            WholeCellChromosomeForkDirection::CounterClockwise,
            genome_bp,
        );
        let cw_traveled = (replicated_bp as f32 * 0.5).round() as u32;
        let ccw_traveled = replicated_bp.saturating_sub(cw_traveled);
        [
            (
                "fork_clockwise",
                WholeCellChromosomeForkDirection::Clockwise,
                cw_traveled.min(arm_cw),
                arm_cw,
            ),
            (
                "fork_counter_clockwise",
                WholeCellChromosomeForkDirection::CounterClockwise,
                ccw_traveled.min(arm_ccw),
                arm_ccw,
            ),
        ]
        .into_iter()
        .map(|(id, direction, traveled_bp, arm_length)| {
            let completed = traveled_bp >= arm_length;
            WholeCellChromosomeForkState {
                id: id.to_string(),
                direction,
                position_bp: Self::chromosome_position_from_origin(
                    origin_bp,
                    traveled_bp,
                    direction,
                    genome_bp,
                ),
                traveled_bp,
                active: !completed,
                paused: false,
                pause_pressure: 0.0,
                collision_pressure: 0.0,
                pause_events: 0,
                completion_fraction: traveled_bp as f32 / arm_length.max(1) as f32,
                completed,
            }
        })
        .collect()
    }

    pub(super) fn seeded_chromosome_state(&self) -> WholeCellChromosomeState {
        let (genome_bp, origin_bp, terminus_bp, loci) =
            if let Some(organism) = self.organism_data.as_ref() {
                (
                    organism.chromosome_length_bp.max(1),
                    organism.origin_bp.min(organism.chromosome_length_bp.max(1)),
                    organism
                        .terminus_bp
                        .min(organism.chromosome_length_bp.max(1)),
                    self.chromosome_locus_records(organism),
                )
            } else {
                (
                    self.genome_bp.max(1),
                    0,
                    (self.genome_bp.max(1) / 2).max(1),
                    Vec::new(),
                )
            };
        let replicated_bp = self.replicated_bp.min(genome_bp);
        let replicated_fraction = replicated_bp as f32 / genome_bp.max(1) as f32;
        let segregation_progress =
            (self.chromosome_separation_nm / (self.radius_nm * 1.8).max(1.0)).clamp(0.0, 1.0);
        let loci = loci
            .into_iter()
            .map(|(id, midpoint_bp, strand)| WholeCellChromosomeLocusState {
                id,
                midpoint_bp,
                strand,
                copy_number: 1.0,
                accessibility: (0.82 + 0.16 * (1.0 - replicated_fraction)).clamp(0.55, 1.25),
                torsional_stress: 0.0,
                replicated: false,
                segregating: segregation_progress > 0.45,
                domain_index: self.chromosome_domain_index(midpoint_bp, genome_bp),
            })
            .collect();
        WholeCellChromosomeState {
            chromosome_length_bp: genome_bp,
            origin_bp,
            terminus_bp,
            initiation_potential: 0.35,
            initiation_events: u32::from(replicated_bp > 0),
            completion_events: u32::from(replicated_bp >= genome_bp),
            replicated_bp,
            replicated_fraction,
            segregation_progress,
            compaction_fraction: (0.32 + 0.20 * replicated_fraction).clamp(0.20, 0.90),
            torsional_stress: 0.0,
            mean_locus_accessibility: (0.82 + 0.16 * (1.0 - replicated_fraction)).clamp(0.55, 1.25),
            forks: Self::chromosome_forks_from_progress(
                genome_bp,
                origin_bp,
                terminus_bp,
                replicated_bp,
            ),
            loci,
        }
    }

    pub(super) fn normalize_chromosome_state(
        &self,
        mut state: WholeCellChromosomeState,
    ) -> WholeCellChromosomeState {
        let seeded = self.seeded_chromosome_state();
        state.chromosome_length_bp = state.chromosome_length_bp.max(seeded.chromosome_length_bp);
        state.origin_bp = state.origin_bp.min(state.chromosome_length_bp.max(1));
        state.terminus_bp = state.terminus_bp.min(state.chromosome_length_bp.max(1));
        if state.origin_bp == state.terminus_bp {
            state.origin_bp = seeded.origin_bp;
            state.terminus_bp = seeded.terminus_bp;
        }
        state.replicated_bp = state.replicated_bp.min(state.chromosome_length_bp.max(1));
        state.replicated_fraction =
            state.replicated_bp as f32 / state.chromosome_length_bp.max(1) as f32;
        state.compaction_fraction = state.compaction_fraction.clamp(0.0, 1.0);
        state.segregation_progress = state.segregation_progress.clamp(0.0, 1.0);
        state.torsional_stress = state.torsional_stress.clamp(0.0, 2.0);
        state.mean_locus_accessibility = state.mean_locus_accessibility.clamp(0.25, 1.50);
        if state.loci.is_empty() {
            state.loci = seeded.loci;
        }
        if state.forks.is_empty()
            && state.replicated_bp > 0
            && state.replicated_bp < state.chromosome_length_bp
        {
            state.forks = Self::chromosome_forks_from_progress(
                state.chromosome_length_bp,
                state.origin_bp,
                state.terminus_bp,
                state.replicated_bp,
            );
        }
        for locus in &mut state.loci {
            locus.copy_number = locus.copy_number.clamp(1.0, 2.0);
            locus.accessibility = locus.accessibility.clamp(0.25, 1.50);
            locus.torsional_stress = locus.torsional_stress.clamp(0.0, 2.0);
            locus.domain_index =
                self.chromosome_domain_index(locus.midpoint_bp, state.chromosome_length_bp);
        }
        for fork in &mut state.forks {
            let arm_length = Self::chromosome_arm_length(
                state.origin_bp,
                state.terminus_bp,
                fork.direction,
                state.chromosome_length_bp,
            );
            fork.traveled_bp = fork.traveled_bp.min(arm_length);
            fork.position_bp = Self::chromosome_position_from_origin(
                state.origin_bp,
                fork.traveled_bp,
                fork.direction,
                state.chromosome_length_bp,
            );
            fork.pause_pressure = fork.pause_pressure.clamp(0.0, 1.5);
            fork.collision_pressure = fork.collision_pressure.clamp(0.0, 1.5);
            fork.completion_fraction = fork.traveled_bp as f32 / arm_length.max(1) as f32;
            fork.completed = fork.completed || fork.traveled_bp >= arm_length;
            fork.active = fork.active && !fork.completed;
        }
        state
    }

    pub(super) fn synchronize_chromosome_summary(&mut self) {
        let genome_bp = self.chromosome_state.chromosome_length_bp.max(1);
        self.genome_bp = genome_bp;
        self.replicated_bp = self.chromosome_state.replicated_bp.min(genome_bp);
        let radius_scale = self
            .organism_data
            .as_ref()
            .map(|organism| organism.geometry.chromosome_radius_fraction.max(0.1))
            .unwrap_or(0.55);
        let target_separation = self.radius_nm
            * radius_scale
            * (0.35 + 1.45 * self.chromosome_state.segregation_progress);
        self.chromosome_separation_nm = target_separation.max(10.0);
    }

    pub(super) fn current_genome_bp(&self) -> u32 {
        let chromosome_length_bp = self.chromosome_state.chromosome_length_bp.max(1);
        if chromosome_length_bp > 1 {
            chromosome_length_bp
        } else {
            self.genome_bp.max(1)
        }
    }

    pub(super) fn current_replicated_bp(&self) -> u32 {
        let genome_bp = self.current_genome_bp();
        if self.chromosome_state.chromosome_length_bp > 1
            || self.chromosome_state.replicated_bp > 0
            || !self.chromosome_state.forks.is_empty()
        {
            self.chromosome_state.replicated_bp.min(genome_bp)
        } else {
            self.replicated_bp.min(genome_bp)
        }
    }

    pub(super) fn current_replicated_fraction(&self) -> f32 {
        self.current_replicated_bp() as f32 / self.current_genome_bp().max(1) as f32
    }

    pub(super) fn current_chromosome_separation_nm(&self) -> f32 {
        if self.chromosome_state.chromosome_length_bp > 1
            || self.chromosome_state.replicated_bp > 0
            || !self.chromosome_state.forks.is_empty()
        {
            let radius_scale = self
                .organism_data
                .as_ref()
                .map(|organism| organism.geometry.chromosome_radius_fraction.max(0.1))
                .unwrap_or(0.55);
            (self.current_radius_nm()
                * radius_scale
                * (0.35 + 1.45 * self.chromosome_state.segregation_progress))
                .max(10.0)
        } else {
            self.chromosome_separation_nm.max(0.0)
        }
    }

    pub(super) fn chromosome_copy_number_for_state(
        state: &WholeCellChromosomeState,
        midpoint_bp: u32,
    ) -> f32 {
        let genome_bp = state.chromosome_length_bp.max(1);
        if state.replicated_bp >= genome_bp {
            return 2.0;
        }
        let mut copy_number: f32 = 1.0;
        for fork in &state.forks {
            let distance = match fork.direction {
                WholeCellChromosomeForkDirection::Clockwise => {
                    Self::clockwise_distance_bp(state.origin_bp, midpoint_bp, genome_bp)
                }
                WholeCellChromosomeForkDirection::CounterClockwise => {
                    Self::counter_clockwise_distance_bp(state.origin_bp, midpoint_bp, genome_bp)
                }
            };
            if fork.traveled_bp >= distance {
                copy_number += 1.0;
            }
        }
        copy_number.clamp(1.0, 2.0)
    }

    pub(super) fn chromosome_copy_number_at(&self, midpoint_bp: u32) -> f32 {
        Self::chromosome_copy_number_for_state(&self.chromosome_state, midpoint_bp)
    }

    pub(super) fn chromosome_locus_accessibility_at(&self, midpoint_bp: u32) -> f32 {
        self.chromosome_state
            .loci
            .iter()
            .min_by_key(|locus| {
                Self::circular_distance_bp(
                    locus.midpoint_bp as f32,
                    midpoint_bp as f32,
                    self.chromosome_state.chromosome_length_bp.max(1) as f32,
                ) as u32
            })
            .map(|locus| locus.accessibility)
            .unwrap_or(
                self.chromosome_state
                    .mean_locus_accessibility
                    .clamp(0.25, 1.50),
            )
    }

    pub(super) fn chromosome_locus_torsional_stress_at(&self, midpoint_bp: u32) -> f32 {
        self.chromosome_state
            .loci
            .iter()
            .min_by_key(|locus| {
                Self::circular_distance_bp(
                    locus.midpoint_bp as f32,
                    midpoint_bp as f32,
                    self.chromosome_state.chromosome_length_bp.max(1) as f32,
                ) as u32
            })
            .map(|locus| locus.torsional_stress)
            .unwrap_or(self.chromosome_state.torsional_stress.clamp(0.0, 2.0))
    }

    pub(super) fn chromosome_collision_pressure(
        &self,
        direction: WholeCellChromosomeForkDirection,
        fork_position_bp: u32,
        genome_bp: u32,
    ) -> f32 {
        let Some(organism) = self.organism_data.as_ref() else {
            return 0.0;
        };
        let window_bp = (0.12 * genome_bp as f32).round() as u32;
        let mut pressure = 0.0;
        for feature in &organism.genes {
            let midpoint_bp = ((feature.start_bp as u64 + feature.end_bp as u64) / 2) as u32;
            let ahead_distance = match direction {
                WholeCellChromosomeForkDirection::Clockwise => {
                    Self::clockwise_distance_bp(fork_position_bp, midpoint_bp, genome_bp)
                }
                WholeCellChromosomeForkDirection::CounterClockwise => {
                    Self::counter_clockwise_distance_bp(fork_position_bp, midpoint_bp, genome_bp)
                }
            };
            if ahead_distance > window_bp {
                continue;
            }
            let head_on = match direction {
                WholeCellChromosomeForkDirection::Clockwise => feature.strand < 0,
                WholeCellChromosomeForkDirection::CounterClockwise => feature.strand > 0,
            };
            if !head_on {
                continue;
            }
            let local_weight = 1.0 - ahead_distance as f32 / window_bp.max(1) as f32;
            pressure += feature.basal_expression.max(0.0) * (0.35 + 0.65 * local_weight);
        }
        Self::saturating_signal(pressure, 6.0).clamp(0.0, 1.5)
    }

    pub(super) fn advance_chromosome_state(
        &mut self,
        dt: f32,
        replication_drive: f32,
        segregation_drive: f32,
    ) {
        let mut state = self.normalize_chromosome_state(self.chromosome_state.clone());
        let genome_bp = state.chromosome_length_bp.max(1);
        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let chromosome_domain_count = self.chromosome_domain_count();
        let chromosome_domain_energy_support = (0..chromosome_domain_count)
            .map(|index| self.chromosome_domain_energy_support(index as u32))
            .collect::<Vec<_>>();
        let chromosome_domain_nucleotide_support = (0..chromosome_domain_count)
            .map(|index| self.chromosome_domain_nucleotide_support(index as u32))
            .collect::<Vec<_>>();
        let origin_domain_index = self.chromosome_domain_index(state.origin_bp, genome_bp) as usize;
        let chromosome_local_energy_support =
            Self::chromosome_domain_value(&chromosome_domain_energy_support, origin_domain_index);
        let chromosome_local_nucleotide_support = Self::chromosome_domain_value(
            &chromosome_domain_nucleotide_support,
            origin_domain_index,
        );
        let chromosome_local_nucleoid_atp =
            Self::saturating_signal(self.localized_nucleoid_atp_pool_mm(), 0.90);
        let chromosome_local_nucleoid_nucleotides =
            Self::saturating_signal(self.localized_nucleotide_pool_mm(), 0.55);
        let initiation_support = Self::finite_scale(
            0.32 * Self::saturating_signal(self.complex_assembly.dnaa_activity, 64.0)
                + 0.18 * chromosome_local_nucleotide_support
                + 0.14 * chromosome_local_energy_support
                + 0.10 * chromosome_local_nucleoid_nucleotides
                + 0.08 * chromosome_local_nucleoid_atp
                + 0.10 * crowding
                + 0.08 * Self::saturating_signal(self.complex_assembly.replisome_complexes, 16.0),
            1.0,
            0.40,
            1.60,
        );
        state.initiation_potential = (0.72 * state.initiation_potential
            + 0.20 * initiation_support
            + 0.08 * Self::saturating_signal(replication_drive.max(0.0), 12.0))
        .clamp(0.0, 1.5);

        if state.forks.is_empty()
            && state.replicated_bp < genome_bp
            && (state.initiation_potential > 0.42 || replication_drive > 4.0)
        {
            state.initiation_events = state.initiation_events.saturating_add(1);
            state.forks = Self::chromosome_forks_from_progress(
                genome_bp,
                state.origin_bp,
                state.terminus_bp,
                2,
            );
        }

        let mut total_replicated_bp = 0u32;
        let mut total_collision = 0.0;
        let mut active_fork_count = 0.0;
        let mut completion_increment = 0u32;
        for fork in &mut state.forks {
            let arm_length = Self::chromosome_arm_length(
                state.origin_bp,
                state.terminus_bp,
                fork.direction,
                genome_bp,
            );
            if fork.completed {
                total_replicated_bp = total_replicated_bp.saturating_add(arm_length);
                continue;
            }
            let collision_pressure =
                self.chromosome_collision_pressure(fork.direction, fork.position_bp, genome_bp);
            let domain_index = self.chromosome_domain_index(fork.position_bp, genome_bp) as usize;
            let chromosome_local_energy_support =
                Self::chromosome_domain_value(&chromosome_domain_energy_support, domain_index);
            let chromosome_local_nucleotide_support =
                Self::chromosome_domain_value(&chromosome_domain_nucleotide_support, domain_index);
            let chromosome_resource_support = Self::finite_scale(
                0.44 * chromosome_local_nucleotide_support
                    + 0.32 * chromosome_local_energy_support
                    + 0.14 * crowding
                    + 0.10 * self.organism_expression.process_scales.replication.max(0.0),
                1.0,
                0.45,
                1.65,
            );
            let pause_pressure = (0.46 * collision_pressure
                + 0.18 * (1.0 - crowding).max(0.0)
                + 0.20 * (1.0 - chromosome_local_nucleotide_support).max(0.0)
                + 0.16 * (1.0 - chromosome_local_energy_support).max(0.0))
            .clamp(0.0, 1.5);
            let paused = pause_pressure > 0.85;
            let progress_scale = if paused {
                0.18
            } else {
                (1.0 - 0.70 * pause_pressure).clamp(0.12, 1.0)
            } * chromosome_resource_support.clamp(0.35, 1.30);
            let raw_delta = (0.5
                * replication_drive.max(0.0)
                * self.organism_expression.process_scales.replication.max(0.1)
                * progress_scale
                * dt_scale)
                .round();
            let minimal_progress_allowed = replication_drive > 0.0
                && !paused
                && chromosome_resource_support > 0.58
                && chromosome_local_nucleotide_support > 0.55
                && chromosome_local_energy_support > 0.55;
            let delta_bp = if raw_delta <= 0.0 {
                u32::from(minimal_progress_allowed)
            } else {
                raw_delta as u32
            };
            let next_traveled_bp = (fork.traveled_bp.saturating_add(delta_bp)).min(arm_length);
            if paused && !fork.paused {
                fork.pause_events = fork.pause_events.saturating_add(1);
            }
            fork.traveled_bp = next_traveled_bp;
            fork.position_bp = Self::chromosome_position_from_origin(
                state.origin_bp,
                fork.traveled_bp,
                fork.direction,
                genome_bp,
            );
            fork.pause_pressure = pause_pressure;
            fork.collision_pressure = collision_pressure;
            fork.paused = paused;
            fork.completion_fraction = fork.traveled_bp as f32 / arm_length.max(1) as f32;
            fork.completed = fork.traveled_bp >= arm_length;
            fork.active = !fork.completed;
            if fork.completed {
                completion_increment = completion_increment.saturating_add(1);
            }
            total_replicated_bp =
                total_replicated_bp.saturating_add(fork.traveled_bp.min(arm_length));
            total_collision += collision_pressure;
            if fork.active {
                active_fork_count += 1.0;
            }
        }
        if completion_increment > 0 && state.forks.iter().all(|fork| fork.completed) {
            state.completion_events = state.completion_events.saturating_add(1);
        }
        state.replicated_bp = total_replicated_bp.min(genome_bp);
        state.replicated_fraction = state.replicated_bp as f32 / genome_bp.max(1) as f32;
        let mean_collision = if active_fork_count > 0.0 {
            total_collision / active_fork_count
        } else {
            0.0
        };
        state.torsional_stress = (0.58 * state.torsional_stress
            + 0.42 * (0.55 * mean_collision + 0.20 * state.replicated_fraction))
            .clamp(0.0, 2.0);
        state.compaction_fraction = (0.32
            + 0.18 * state.replicated_fraction
            + 0.12 * mean_collision
            + 0.10 * (1.0 - crowding).max(0.0))
        .clamp(0.20, 0.95);
        state.segregation_progress = (state.segregation_progress
            + 0.004 * segregation_drive.max(0.0) * dt_scale
            + 0.020 * state.replicated_fraction)
            .clamp(0.0, 1.0);

        let state_for_copy = state.clone();
        let mut accessibility_total = 0.0;
        for locus in &mut state.loci {
            let copy_number =
                Self::chromosome_copy_number_for_state(&state_for_copy, locus.midpoint_bp);
            let local_collision = state
                .forks
                .iter()
                .map(|fork| {
                    let distance = Self::circular_distance_bp(
                        locus.midpoint_bp as f32,
                        fork.position_bp as f32,
                        genome_bp as f32,
                    );
                    let proximity =
                        1.0 - (distance / (0.18 * genome_bp as f32).max(1.0)).clamp(0.0, 1.0);
                    fork.collision_pressure * proximity
                })
                .fold(0.0, f32::max);
            let accessibility = (0.74 + 0.18 * copy_number
                - 0.20 * state.compaction_fraction
                - 0.18 * state.torsional_stress
                - 0.14 * local_collision)
                .clamp(0.25, 1.50);
            locus.copy_number = copy_number;
            locus.accessibility = accessibility;
            locus.torsional_stress =
                (0.55 * state.torsional_stress + 0.45 * local_collision).clamp(0.0, 2.0);
            locus.replicated = copy_number > 1.05;
            locus.segregating = locus.replicated && state.segregation_progress > 0.45;
            locus.domain_index = self.chromosome_domain_index(locus.midpoint_bp, genome_bp);
            accessibility_total += accessibility;
        }
        if !state.loci.is_empty() {
            state.mean_locus_accessibility = accessibility_total / state.loci.len() as f32;
        }
        self.chromosome_state = self.normalize_chromosome_state(state);
        self.synchronize_chromosome_summary();
        self.refresh_spatial_fields();
    }
}
