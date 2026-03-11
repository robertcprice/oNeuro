//! Generic substrate-level reaction and assembly IR.
//!
//! This is the first substrate-first layer for whole-cell work: reactions are
//! represented as generic stoichiometric rules with site-local execution
//! against the terrarium lattice instead of being hard-coded as bespoke
//! subsystem behavior.

use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxChannel {
    Neutral,
    Substrate,
    Energy,
    Biosynthetic,
    Waste,
}

impl FluxChannel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Substrate => "substrate",
            Self::Energy => "energy",
            Self::Biosynthetic => "biosynthetic",
            Self::Waste => "waste",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "neutral" => Some(Self::Neutral),
            "substrate" => Some(Self::Substrate),
            "energy" => Some(Self::Energy),
            "biosynthetic" | "biosynthesis" => Some(Self::Biosynthetic),
            "waste" => Some(Self::Waste),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionTerm {
    pub species: TerrariumSpecies,
    pub stoich: f32,
    pub channel: FluxChannel,
}

impl ReactionTerm {
    pub const fn new(species: TerrariumSpecies, stoich: f32, channel: FluxChannel) -> Self {
        Self {
            species,
            stoich,
            channel,
        }
    }
}

pub const EMPTY_REACTION_TERM: ReactionTerm =
    ReactionTerm::new(TerrariumSpecies::Water, 0.0, FluxChannel::Neutral);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ReactionDriver {
    Activity = 0,
    Energy = 1,
    Biosynthesis = 2,
    Replication = 3,
    Division = 4,
    Oxygen = 5,
    Carbon = 6,
    Translation = 7,
}

impl ReactionDriver {
    pub const COUNT: usize = 8;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionContext {
    pub catalyst_scale: f32,
    pub drivers: [f32; ReactionDriver::COUNT],
}

impl ReactionContext {
    pub fn driver(self, driver: ReactionDriver) -> f32 {
        self.drivers[driver as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionLaw {
    pub base_rate: f32,
    pub driver_weights: [f32; ReactionDriver::COUNT],
}

impl ReactionLaw {
    pub const fn new(base_rate: f32, driver_weights: [f32; ReactionDriver::COUNT]) -> Self {
        Self {
            base_rate,
            driver_weights,
        }
    }

    pub fn target_amount(self, context: ReactionContext, dt_ms: f32) -> f32 {
        let mut drive = 1.0f32;
        for idx in 0..ReactionDriver::COUNT {
            drive += self.driver_weights[idx] * context.drivers[idx];
        }
        let drive = drive.max(0.0);
        (self.base_rate.max(0.0) * dt_ms.max(0.0) * drive * context.catalyst_scale.max(0.0))
            .max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReactionRule {
    pub name: &'static str,
    pub substrate_count: usize,
    pub substrates: [ReactionTerm; 4],
    pub product_count: usize,
    pub products: [ReactionTerm; 4],
    pub law: ReactionLaw,
}

impl ReactionRule {
    pub const fn new(
        name: &'static str,
        substrate_count: usize,
        substrates: [ReactionTerm; 4],
        product_count: usize,
        products: [ReactionTerm; 4],
        law: ReactionLaw,
    ) -> Self {
        Self {
            name,
            substrate_count,
            substrates,
            product_count,
            products,
            law,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReactionFlux {
    pub substrate_draw: f32,
    pub energy_draw: f32,
    pub biosynthetic_draw: f32,
    pub byproduct_load: f32,
    pub removed_total: f32,
    pub target_total: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssemblyComponent {
    pub species: TerrariumSpecies,
    pub weight: f32,
    pub half_saturation: f32,
}

impl AssemblyComponent {
    pub const fn new(species: TerrariumSpecies, weight: f32, half_saturation: f32) -> Self {
        Self {
            species,
            weight,
            half_saturation,
        }
    }
}

pub const EMPTY_ASSEMBLY_COMPONENT: AssemblyComponent =
    AssemblyComponent::new(TerrariumSpecies::Water, 0.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssemblyContext {
    pub catalyst_scale: f32,
    pub support_scale: f32,
    pub demand_satisfaction: f32,
    pub crowding_penalty: f32,
    pub byproduct_load: f32,
    pub substrate_pressure: f32,
    pub energy_pressure: f32,
    pub biosynthetic_pressure: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AssemblyState {
    pub component_availability: f32,
    pub occupancy: f32,
    pub stability: f32,
    pub turnover: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssemblyRule {
    pub name: &'static str,
    pub component_count: usize,
    pub components: [AssemblyComponent; 4],
    pub occupancy_gain_scale: f32,
    pub stability_bias: f32,
    pub turnover_bias: f32,
}

impl AssemblyRule {
    pub const fn new(
        name: &'static str,
        component_count: usize,
        components: [AssemblyComponent; 4],
        occupancy_gain_scale: f32,
        stability_bias: f32,
        turnover_bias: f32,
    ) -> Self {
        Self {
            name,
            component_count,
            components,
            occupancy_gain_scale,
            stability_bias,
            turnover_bias,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialChannel {
    Species(TerrariumSpecies),
    Hydration,
    MicrobialActivity,
    PlantDrive,
    CenterProximity,
    RadialCenterProximity,
    VerticalMidplaneProximity,
    BoundaryProximity,
}

impl SpatialChannel {
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized = name.trim().to_lowercase();
        if let Some(species_name) = normalized.strip_prefix("species:") {
            return TerrariumSpecies::from_name(species_name).map(Self::Species);
        }

        match normalized.as_str() {
            "hydration" => Some(Self::Hydration),
            "microbial_activity" | "microbes" => Some(Self::MicrobialActivity),
            "plant_drive" | "plants" => Some(Self::PlantDrive),
            "center_proximity" | "center" => Some(Self::CenterProximity),
            "radial_center_proximity" | "radial_center" => Some(Self::RadialCenterProximity),
            "vertical_midplane_proximity" | "midplane" => Some(Self::VerticalMidplaneProximity),
            "boundary_proximity" | "boundary" => Some(Self::BoundaryProximity),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizationCue {
    pub channel: SpatialChannel,
    pub weight: f32,
    pub half_saturation: f32,
}

impl LocalizationCue {
    pub const fn new(channel: SpatialChannel, weight: f32, half_saturation: f32) -> Self {
        Self {
            channel,
            weight,
            half_saturation,
        }
    }
}

pub const EMPTY_LOCALIZATION_CUE: LocalizationCue =
    LocalizationCue::new(SpatialChannel::Species(TerrariumSpecies::Water), 0.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizationRule {
    pub name: &'static str,
    pub patch_radius: usize,
    pub cue_count: usize,
    pub cues: [LocalizationCue; 8],
    pub persistence_weight: f32,
    pub exclusion_padding: f32,
    pub exclusion_strength: f32,
}

impl LocalizationRule {
    pub const fn new(
        name: &'static str,
        patch_radius: usize,
        cue_count: usize,
        cues: [LocalizationCue; 8],
        persistence_weight: f32,
        exclusion_padding: f32,
        exclusion_strength: f32,
    ) -> Self {
        Self {
            name,
            patch_radius,
            cue_count,
            cues,
            persistence_weight,
            exclusion_padding,
            exclusion_strength,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalizedPatch {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub radius: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarContext<const N: usize> {
    pub signals: [f32; N],
}

impl<const N: usize> ScalarContext<N> {
    pub fn signal(self, index: usize) -> f32 {
        self.signals.get(index).copied().unwrap_or(0.0).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarFactor {
    pub signal: usize,
    pub bias: f32,
    pub scale: f32,
    pub exponent: f32,
}

impl ScalarFactor {
    pub const fn new(signal: usize, bias: f32, scale: f32, exponent: f32) -> Self {
        Self {
            signal,
            bias,
            scale,
            exponent,
        }
    }

    pub fn evaluate<const N: usize>(self, context: ScalarContext<N>) -> f32 {
        let signal = context.signal(self.signal);
        let exponent = self.exponent.max(0.0);
        let transformed = if exponent <= 1.0e-6 {
            1.0
        } else if (exponent - 1.0).abs() <= 1.0e-6 {
            signal
        } else {
            signal.powf(exponent)
        };
        (self.bias + self.scale * transformed).max(0.0)
    }
}

pub const EMPTY_SCALAR_FACTOR: ScalarFactor = ScalarFactor::new(0, 1.0, 0.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarBranch {
    pub coefficient: f32,
    pub factor_count: usize,
    pub factors: [ScalarFactor; 8],
}

impl ScalarBranch {
    pub const fn new(coefficient: f32, factor_count: usize, factors: [ScalarFactor; 8]) -> Self {
        Self {
            coefficient,
            factor_count,
            factors,
        }
    }

    pub fn evaluate<const N: usize>(self, context: ScalarContext<N>) -> f32 {
        if self.coefficient.abs() <= 1.0e-9 {
            return 0.0;
        }
        let mut value = self.coefficient;
        for factor in self.factors.iter().take(self.factor_count) {
            value *= factor.evaluate(context);
        }
        value
    }
}

pub const EMPTY_SCALAR_BRANCH: ScalarBranch = ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarRule {
    pub bias: f32,
    pub branch_count: usize,
    pub branches: [ScalarBranch; 4],
    pub min_value: f32,
    pub max_value: f32,
}

impl ScalarRule {
    pub const fn new(
        bias: f32,
        branch_count: usize,
        branches: [ScalarBranch; 4],
        min_value: f32,
        max_value: f32,
    ) -> Self {
        Self {
            bias,
            branch_count,
            branches,
            min_value,
            max_value,
        }
    }

    pub fn evaluate<const N: usize>(self, context: ScalarContext<N>) -> f32 {
        let mut value = self.bias;
        for branch in self.branches.iter().take(self.branch_count) {
            value += branch.evaluate(context);
        }
        let min_value = self.min_value.min(self.max_value);
        let max_value = self.max_value.max(self.min_value);
        if value.is_finite() {
            value.clamp(min_value, max_value)
        } else {
            min_value
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineRule<const N: usize> {
    pub bias: f32,
    pub weights: [f32; N],
    pub min_value: f32,
    pub max_value: f32,
}

impl<const N: usize> AffineRule<N> {
    pub const fn new(bias: f32, weights: [f32; N], min_value: f32, max_value: f32) -> Self {
        Self {
            bias,
            weights,
            min_value,
            max_value,
        }
    }

    pub fn evaluate(self, context: ScalarContext<N>) -> f32 {
        let mut value = self.bias;
        for (idx, weight) in self.weights.iter().enumerate() {
            value += *weight * context.signal(idx);
        }
        let min_value = self.min_value.min(self.max_value);
        let max_value = self.max_value.max(self.min_value);
        if value.is_finite() {
            value.clamp(min_value, max_value)
        } else {
            min_value
        }
    }
}

fn channel_signal(
    substrate: &BatchedAtomTerrarium,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    cue: LocalizationCue,
) -> f32 {
    let gid = z * substrate.y_dim * substrate.x_dim + y * substrate.x_dim + x;
    let x_mid = (substrate.x_dim as f32 - 1.0) * 0.5;
    let y_mid = (substrate.y_dim as f32 - 1.0) * 0.5;
    let z_mid = (substrate.z_dim as f32 - 1.0) * 0.5;
    let x_den = x_mid.max(1.0);
    let y_den = y_mid.max(1.0);
    let z_den = z_mid.max(1.0);

    match cue.channel {
        SpatialChannel::Species(species) => {
            let mean = substrate
                .patch_mean_species(species, x, y, z, radius)
                .max(0.0);
            let half_saturation = cue.half_saturation.max(1.0e-6);
            (mean / (mean + half_saturation)).clamp(0.0, 1.0)
        }
        SpatialChannel::Hydration => substrate.hydration[gid].clamp(0.0, 1.0),
        SpatialChannel::MicrobialActivity => (substrate.microbial_activity[gid]
            / (substrate.microbial_activity[gid] + 1.0))
            .clamp(0.0, 1.0),
        SpatialChannel::PlantDrive => substrate.plant_drive[gid].clamp(0.0, 1.0),
        SpatialChannel::CenterProximity => {
            let dx = (x as f32 - x_mid) / x_den;
            let dy = (y as f32 - y_mid) / y_den;
            let dz = (z as f32 - z_mid) / z_den;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt() / 3.0f32.sqrt();
            (1.0 - distance).clamp(0.0, 1.0)
        }
        SpatialChannel::RadialCenterProximity => {
            let dx = (x as f32 - x_mid) / x_den;
            let dy = (y as f32 - y_mid) / y_den;
            let distance = (dx * dx + dy * dy).sqrt() / 2.0f32.sqrt();
            (1.0 - distance).clamp(0.0, 1.0)
        }
        SpatialChannel::VerticalMidplaneProximity => {
            (1.0 - ((z as f32 - z_mid).abs() / z_den)).clamp(0.0, 1.0)
        }
        SpatialChannel::BoundaryProximity => {
            let x_dist = x.min(substrate.x_dim.saturating_sub(1).saturating_sub(x)) as f32;
            let y_dist = y.min(substrate.y_dim.saturating_sub(1).saturating_sub(y)) as f32;
            let z_dist = z.min(substrate.z_dim.saturating_sub(1).saturating_sub(z)) as f32;
            let boundary_distance = x_dist.min(y_dist).min(z_dist);
            let max_distance =
                ((substrate.x_dim.min(substrate.y_dim).min(substrate.z_dim) as f32) * 0.5).max(1.0);
            (1.0 - boundary_distance / max_distance).clamp(0.0, 1.0)
        }
    }
}

pub fn localize_patch(
    substrate: &BatchedAtomTerrarium,
    rule: LocalizationRule,
    previous_patch: Option<LocalizedPatch>,
    occupied_sites: &[LocalizedPatch],
) -> LocalizedPatch {
    let mut best = LocalizedPatch {
        x: substrate.x_dim / 2,
        y: substrate.y_dim / 2,
        z: substrate.z_dim / 2,
        radius: rule.patch_radius,
        score: f32::NEG_INFINITY,
    };

    let x_den = (substrate.x_dim.max(2) - 1) as f32;
    let y_den = (substrate.y_dim.max(2) - 1) as f32;
    let z_den = (substrate.z_dim.max(2) - 1) as f32;
    let spatial_den = (x_den * x_den + y_den * y_den + z_den * z_den)
        .sqrt()
        .max(1.0);

    for z in 0..substrate.z_dim {
        for y in 0..substrate.y_dim {
            for x in 0..substrate.x_dim {
                let mut score = 0.0f32;
                for cue in rule.cues.iter().take(rule.cue_count) {
                    score +=
                        cue.weight * channel_signal(substrate, x, y, z, rule.patch_radius, *cue);
                }

                if let Some(previous) = previous_patch {
                    let dx = x as f32 - previous.x as f32;
                    let dy = y as f32 - previous.y as f32;
                    let dz = z as f32 - previous.z as f32;
                    let continuity = 1.0 - (dx * dx + dy * dy + dz * dz).sqrt() / spatial_den;
                    score += rule.persistence_weight * continuity.clamp(0.0, 1.0);
                }

                for occupied in occupied_sites {
                    let dx = x as f32 - occupied.x as f32;
                    let dy = y as f32 - occupied.y as f32;
                    let dz = z as f32 - occupied.z as f32;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let exclusion_radius = rule.patch_radius as f32
                        + occupied.radius as f32
                        + rule.exclusion_padding.max(0.0);
                    if exclusion_radius > 1.0e-6 && distance < exclusion_radius {
                        let overlap = 1.0 - distance / exclusion_radius;
                        score -= rule.exclusion_strength * overlap.clamp(0.0, 1.0);
                    }
                }

                if score > best.score {
                    best = LocalizedPatch {
                        x,
                        y,
                        z,
                        radius: rule.patch_radius,
                        score,
                    };
                }
            }
        }
    }

    best
}

pub fn evaluate_patch_assembly(
    substrate: &BatchedAtomTerrarium,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    rule: AssemblyRule,
    context: AssemblyContext,
) -> AssemblyState {
    let mut weighted_component_signal = 0.0f32;
    let mut total_weight = 0.0f32;
    for component in rule.components.iter().take(rule.component_count) {
        let weight = component.weight.max(0.0);
        if weight <= 1.0e-9 {
            continue;
        }
        let mean = substrate
            .patch_mean_species(component.species, x, y, z, radius)
            .max(0.0);
        let half_saturation = component.half_saturation.max(1.0e-6);
        let availability = (mean / (mean + half_saturation)).clamp(0.0, 1.0);
        weighted_component_signal += availability * weight;
        total_weight += weight;
    }

    let component_availability = if total_weight <= 1.0e-9 {
        1.0
    } else {
        (weighted_component_signal / total_weight).clamp(0.0, 1.0)
    };
    let support_gate = (context.support_scale.max(0.0)
        * context.catalyst_scale.max(0.0)
        * context.demand_satisfaction.clamp(0.0, 1.0))
    .sqrt()
    .clamp(0.0, 1.6);
    let pressure = (0.35 * context.substrate_pressure.max(0.0)
        + 0.35 * context.energy_pressure.max(0.0)
        + 0.30 * context.biosynthetic_pressure.max(0.0))
    .clamp(0.0, 4.0);

    let occupancy = (component_availability
        * (0.60 + 0.40 * support_gate)
        * context.crowding_penalty.clamp(0.0, 1.0)
        * rule.occupancy_gain_scale.max(0.0))
    .clamp(0.0, 1.5);
    let stability = (rule.stability_bias
        + component_availability * 0.32
        + occupancy * 0.36
        + context.crowding_penalty.clamp(0.0, 1.0) * 0.16
        + context.demand_satisfaction.clamp(0.0, 1.0) * 0.10
        - context.byproduct_load.max(0.0) * 0.10
        - pressure * 0.08)
        .clamp(0.0, 1.5);
    let turnover = (rule.turnover_bias
        + (1.0 - component_availability) * 0.30
        + (1.0 - context.demand_satisfaction.clamp(0.0, 1.0)) * 0.30
        + context.byproduct_load.max(0.0) * 0.12
        + pressure * 0.10
        + (1.0 - context.crowding_penalty.clamp(0.0, 1.0)) * 0.20)
        .clamp(0.0, 1.5);

    AssemblyState {
        component_availability,
        occupancy,
        stability,
        turnover,
    }
}

pub fn execute_patch_reaction(
    substrate: &mut BatchedAtomTerrarium,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    rule: ReactionRule,
    context: ReactionContext,
    dt_ms: f32,
) -> ReactionFlux {
    let reaction_extent = rule.law.target_amount(context, dt_ms);
    if reaction_extent <= 1.0e-9 {
        return ReactionFlux::default();
    }

    let mut flux = ReactionFlux::default();
    for term in rule.substrates.iter().take(rule.substrate_count) {
        let desired = (reaction_extent * term.stoich.max(0.0)).max(0.0);
        if desired <= 1.0e-9 {
            continue;
        }
        let removed = substrate.extract_patch_species(term.species, x, y, z, radius, desired);
        flux.target_total += desired;
        flux.removed_total += removed;
        match term.channel {
            FluxChannel::Substrate => flux.substrate_draw += removed,
            FluxChannel::Energy => flux.energy_draw += removed,
            FluxChannel::Biosynthetic => flux.biosynthetic_draw += removed,
            FluxChannel::Neutral | FluxChannel::Waste => {}
        }
    }

    let satisfaction = if flux.target_total <= 1.0e-9 {
        1.0
    } else {
        (flux.removed_total / flux.target_total).clamp(0.0, 1.0)
    };
    let product_extent = reaction_extent * satisfaction;

    for term in rule.products.iter().take(rule.product_count) {
        let produced = (product_extent * term.stoich.max(0.0)).max(0.0);
        if produced <= 1.0e-9 {
            continue;
        }
        substrate.add_hotspot(term.species, x, y, z, produced);
        if term.channel == FluxChannel::Waste {
            flux.byproduct_load += produced;
        }
    }

    flux
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_rule_evaluates_sum_of_affine_product_branches() {
        let context = ScalarContext::<4> {
            signals: [0.5, 0.8, 0.25, 0.9],
        };
        let rule = ScalarRule::new(
            0.2,
            2,
            [
                ScalarBranch::new(
                    1.5,
                    2,
                    [
                        ScalarFactor::new(0, 0.4, 0.6, 1.0),
                        ScalarFactor::new(1, 0.3, 0.4, 1.0),
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                    ],
                ),
                ScalarBranch::new(
                    0.8,
                    2,
                    [
                        ScalarFactor::new(2, 0.0, 1.0, 1.0),
                        ScalarFactor::new(3, 0.5, 0.5, 1.0),
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                        EMPTY_SCALAR_FACTOR,
                    ],
                ),
                EMPTY_SCALAR_BRANCH,
                EMPTY_SCALAR_BRANCH,
            ],
            0.0,
            10.0,
        );

        let expected =
            0.2 + 1.5 * (0.4 + 0.6 * 0.5) * (0.3 + 0.4 * 0.8) + 0.8 * 0.25 * (0.5 + 0.5 * 0.9);
        assert!((rule.evaluate(context) - expected).abs() < 1.0e-6);
    }

    #[test]
    fn affine_rule_evaluates_high_fan_in_linear_context() {
        let context = ScalarContext::<5> {
            signals: [0.50, 0.25, 0.80, 0.10, 0.40],
        };
        let rule = AffineRule::new(0.20, [0.30, -0.10, 0.25, 0.00, 0.15], 0.0, 1.0);

        assert!((rule.evaluate(context) - 0.585).abs() < 1.0e-6);
    }

    #[test]
    fn assembly_rule_reports_higher_order_for_supported_components() {
        let mut terrarium = BatchedAtomTerrarium::new(8, 8, 4, 0.5, false);
        terrarium.fill_species(TerrariumSpecies::AtpFlux, 0.6);
        terrarium.fill_species(TerrariumSpecies::Phosphorus, 0.4);
        terrarium.fill_species(TerrariumSpecies::OxygenGas, 0.5);

        let rich = evaluate_patch_assembly(
            &terrarium,
            4,
            4,
            2,
            2,
            AssemblyRule::new(
                "membrane_patch",
                3,
                [
                    AssemblyComponent::new(TerrariumSpecies::AtpFlux, 1.0, 0.05),
                    AssemblyComponent::new(TerrariumSpecies::Phosphorus, 0.8, 0.05),
                    AssemblyComponent::new(TerrariumSpecies::OxygenGas, 0.4, 0.08),
                    EMPTY_ASSEMBLY_COMPONENT,
                ],
                1.10,
                0.48,
                0.08,
            ),
            AssemblyContext {
                catalyst_scale: 1.1,
                support_scale: 1.0,
                demand_satisfaction: 0.95,
                crowding_penalty: 0.96,
                byproduct_load: 0.04,
                substrate_pressure: 0.08,
                energy_pressure: 0.06,
                biosynthetic_pressure: 0.05,
            },
        );

        let poor = evaluate_patch_assembly(
            &terrarium,
            4,
            4,
            2,
            2,
            AssemblyRule::new(
                "membrane_patch",
                3,
                [
                    AssemblyComponent::new(TerrariumSpecies::AtpFlux, 1.0, 0.05),
                    AssemblyComponent::new(TerrariumSpecies::Phosphorus, 0.8, 0.05),
                    AssemblyComponent::new(TerrariumSpecies::OxygenGas, 0.4, 0.08),
                    EMPTY_ASSEMBLY_COMPONENT,
                ],
                1.10,
                0.48,
                0.08,
            ),
            AssemblyContext {
                catalyst_scale: 0.7,
                support_scale: 0.75,
                demand_satisfaction: 0.45,
                crowding_penalty: 0.75,
                byproduct_load: 0.40,
                substrate_pressure: 0.55,
                energy_pressure: 0.52,
                biosynthetic_pressure: 0.36,
            },
        );

        assert!(rich.component_availability > 0.0);
        assert!(rich.occupancy > poor.occupancy);
        assert!(rich.stability > poor.stability);
        assert!(rich.turnover < poor.turnover);
    }

    #[test]
    fn localization_rule_tracks_hotspot_and_avoids_overlap() {
        let mut terrarium = BatchedAtomTerrarium::new(10, 10, 6, 0.5, false);
        terrarium.fill_species(TerrariumSpecies::OxygenGas, 0.02);
        terrarium.add_hotspot(TerrariumSpecies::OxygenGas, 5, 5, 0, 2.0);
        terrarium.add_hotspot(TerrariumSpecies::OxygenGas, 5, 5, 5, 1.0);

        let rule = LocalizationRule::new(
            "membrane_localizer",
            2,
            3,
            [
                LocalizationCue::new(
                    SpatialChannel::Species(TerrariumSpecies::OxygenGas),
                    0.8,
                    0.05,
                ),
                LocalizationCue::new(SpatialChannel::BoundaryProximity, 0.7, 1.0),
                LocalizationCue::new(SpatialChannel::RadialCenterProximity, 0.4, 1.0),
                EMPTY_LOCALIZATION_CUE,
                EMPTY_LOCALIZATION_CUE,
                EMPTY_LOCALIZATION_CUE,
                EMPTY_LOCALIZATION_CUE,
                EMPTY_LOCALIZATION_CUE,
            ],
            0.0,
            1.0,
            1.0,
        );

        let first = localize_patch(&terrarium, rule, None, &[]);
        let second = localize_patch(&terrarium, rule, None, &[first]);

        assert!(
            first.z <= 1,
            "expected top boundary patch, got z={}",
            first.z
        );
        assert!((first.x as isize - 5).abs() <= 1);
        assert!((first.y as isize - 5).abs() <= 1);

        let dx = second.x as f32 - first.x as f32;
        let dy = second.y as f32 - first.y as f32;
        let dz = second.z as f32 - first.z as f32;
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(distance >= 1.0, "expected exclusion to move second patch");
    }

    #[test]
    fn reaction_rule_consumes_reactants_and_emits_products() {
        let mut terrarium = BatchedAtomTerrarium::new(8, 8, 4, 0.5, false);
        terrarium.fill_species(TerrariumSpecies::Glucose, 0.4);
        let before_glucose = terrarium.patch_mean_species(TerrariumSpecies::Glucose, 4, 4, 2, 2);
        let before_co2 = terrarium.patch_mean_species(TerrariumSpecies::CarbonDioxide, 4, 4, 2, 2);

        let rule = ReactionRule::new(
            "test_respiration",
            1,
            [
                ReactionTerm::new(TerrariumSpecies::Glucose, 1.0, FluxChannel::Substrate),
                EMPTY_REACTION_TERM,
                EMPTY_REACTION_TERM,
                EMPTY_REACTION_TERM,
            ],
            1,
            [
                ReactionTerm::new(TerrariumSpecies::CarbonDioxide, 0.5, FluxChannel::Waste),
                EMPTY_REACTION_TERM,
                EMPTY_REACTION_TERM,
                EMPTY_REACTION_TERM,
            ],
            ReactionLaw::new(0.2, [0.0; ReactionDriver::COUNT]),
        );

        let flux = execute_patch_reaction(
            &mut terrarium,
            4,
            4,
            2,
            2,
            rule,
            ReactionContext {
                catalyst_scale: 1.0,
                drivers: [0.0; ReactionDriver::COUNT],
            },
            1.0,
        );
        let after_glucose = terrarium.patch_mean_species(TerrariumSpecies::Glucose, 4, 4, 2, 2);
        let after_co2 = terrarium.patch_mean_species(TerrariumSpecies::CarbonDioxide, 4, 4, 2, 2);

        assert!(flux.substrate_draw > 0.0);
        assert!(flux.byproduct_load > 0.0);
        assert!(after_glucose < before_glucose);
        assert!(after_co2 > before_co2);
    }

    #[test]
    fn declarative_channel_names_parse_cleanly() {
        assert_eq!(
            FluxChannel::from_name("biosynthetic"),
            Some(FluxChannel::Biosynthetic)
        );
        assert_eq!(
            SpatialChannel::from_name("species:oxygen_gas"),
            Some(SpatialChannel::Species(TerrariumSpecies::OxygenGas))
        );
        assert_eq!(
            SpatialChannel::from_name("vertical_midplane_proximity"),
            Some(SpatialChannel::VerticalMidplaneProximity)
        );
        assert!(SpatialChannel::from_name("unknown_channel").is_none());
    }
}
