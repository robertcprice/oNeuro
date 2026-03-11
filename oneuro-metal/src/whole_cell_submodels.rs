//! Local chemistry submodels that can feed the native whole-cell runtime.
//!
//! The main whole-cell simulator stays coarse and fast. This module hosts
//! optional submodels that can be sampled periodically:
//! - a batched chemistry lattice for microdomain substrate support
//! - localized molecular dynamics probes for short-range structural signals

use crate::atomistic_topology::{atomistic_template_for_site_name, AtomisticTemplateDescriptor};
use crate::molecular_dynamics::GPUMolecularDynamics;
use crate::substrate_ir::{
    evaluate_patch_assembly, execute_patch_reaction, localize_patch, AffineRule, AssemblyComponent,
    AssemblyContext, AssemblyRule, AssemblyState, FluxChannel, LocalizationCue, LocalizationRule,
    LocalizedPatch, ReactionContext, ReactionLaw, ReactionRule, ReactionTerm, ScalarBranch,
    ScalarContext, ScalarFactor, ScalarRule, SpatialChannel, EMPTY_LOCALIZATION_CUE,
    EMPTY_REACTION_TERM, EMPTY_SCALAR_FACTOR,
};
use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies};
use crate::whole_cell::{WholeCellBackend, WholeCellQuantumProfile, WholeCellSnapshot};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

fn finite_clamped(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min_value, max_value)
    } else {
        fallback.clamp(min_value, max_value)
    }
}

fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
    let value = value.max(0.0);
    let half_saturation = half_saturation.max(1.0e-6);
    (value / (value + half_saturation)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WholeCellChemistrySite {
    Cytosol,
    AtpSynthaseBand,
    RibosomeCluster,
    SeptumRing,
    ChromosomeTrack,
}

impl WholeCellChemistrySite {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cytosol => "cytosol",
            Self::AtpSynthaseBand => "atp_synthase_band",
            Self::RibosomeCluster => "ribosome_cluster",
            Self::SeptumRing => "septum_ring",
            Self::ChromosomeTrack => "chromosome_track",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "cytosol" | "cytoplasm" => Some(Self::Cytosol),
            "atp_synthase_band" | "atp_band" | "respiratory_band" => Some(Self::AtpSynthaseBand),
            "ribosome_cluster" | "ribosome" | "translation" => Some(Self::RibosomeCluster),
            "septum_ring" | "septum" | "division_ring" => Some(Self::SeptumRing),
            "chromosome_track" | "chromosome" | "dna_track" => Some(Self::ChromosomeTrack),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum Syn3ASubsystemPreset {
    AtpSynthaseMembraneBand,
    RibosomePolysomeCluster,
    ReplisomeTrack,
    FtsZSeptumRing,
}

impl Syn3ASubsystemPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AtpSynthaseMembraneBand => "atp_synthase_membrane_band",
            Self::RibosomePolysomeCluster => "ribosome_polysome_cluster",
            Self::ReplisomeTrack => "replisome_track",
            Self::FtsZSeptumRing => "ftsz_septum_ring",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "atp_synthase_membrane_band" | "atp_band" | "atp_synthase" => {
                Some(Self::AtpSynthaseMembraneBand)
            }
            "ribosome_polysome_cluster" | "ribosome_cluster" | "polysome" => {
                Some(Self::RibosomePolysomeCluster)
            }
            "replisome_track" | "replisome" | "dna_track" => Some(Self::ReplisomeTrack),
            "ftsz_septum_ring" | "septum_ring" | "ftsz" | "septum" => Some(Self::FtsZSeptumRing),
            _ => None,
        }
    }

    pub fn chemistry_site(self) -> WholeCellChemistrySite {
        subsystem_signature(self).chemistry_site
    }

    pub fn default_interval_steps(self) -> u64 {
        subsystem_signature(self).default_interval_steps
    }

    pub fn default_probe_request(self) -> LocalMDProbeRequest {
        subsystem_signature(self).default_probe_request
    }

    pub fn all() -> &'static [Self] {
        const PRESETS: [Syn3ASubsystemPreset; 4] = [
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            Syn3ASubsystemPreset::ReplisomeTrack,
            Syn3ASubsystemPreset::FtsZSeptumRing,
        ];
        &PRESETS
    }
}

const SUBSYSTEM_PRESET_COUNT: usize = Syn3ASubsystemPreset::FtsZSeptumRing as usize + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledSubsystemProbe {
    pub preset: Syn3ASubsystemPreset,
    pub interval_steps: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PatchSpeciesMetrics {
    mean_glucose: f32,
    mean_oxygen: f32,
    mean_atp_flux: f32,
    mean_carbon_dioxide: f32,
    mean_nitrate: f32,
    mean_ammonium: f32,
    mean_proton: f32,
    mean_phosphorus: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalPatchSignals {
    oxygen_signal: f32,
    carbon_signal: f32,
    energy_signal: f32,
    nitrogen_signal: f32,
    phosphorus_signal: f32,
    biosynthesis_signal: f32,
    structural_signal: f32,
    stress_signal: f32,
}

impl LocalPatchSignals {
    fn feature(self, idx: usize) -> f32 {
        match idx {
            0 => self.oxygen_signal,
            1 => self.carbon_signal,
            2 => self.energy_signal,
            3 => self.nitrogen_signal,
            4 => self.phosphorus_signal,
            5 => self.biosynthesis_signal,
            6 => self.structural_signal,
            7 => self.stress_signal,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum CrowdingSignal {
    CarbonDioxide = 0,
    Proton,
}

impl CrowdingSignal {
    const COUNT: usize = Self::Proton as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct CrowdingContext {
    signals: [f32; CrowdingSignal::COUNT],
}

impl Default for CrowdingContext {
    fn default() -> Self {
        Self {
            signals: [0.0; CrowdingSignal::COUNT],
        }
    }
}

impl CrowdingContext {
    fn set(&mut self, signal: CrowdingSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ CrowdingSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum StructuralReducerSignal {
    AssemblyAvailability = 0,
    AssemblyOccupancy,
    AssemblyStability,
    StructuralOrder,
    CrowdingPenalty,
    AssemblyTurnover,
}

impl StructuralReducerSignal {
    const COUNT: usize = Self::AssemblyTurnover as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct StructuralReducerContext {
    signals: [f32; StructuralReducerSignal::COUNT],
}

impl Default for StructuralReducerContext {
    fn default() -> Self {
        Self {
            signals: [0.0; StructuralReducerSignal::COUNT],
        }
    }
}

impl StructuralReducerContext {
    fn set(&mut self, signal: StructuralReducerSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ StructuralReducerSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum PatchReducerSignal {
    Carbon = 0,
    Nitrogen,
    Phosphorus,
    Energy,
    Structural,
    CarbonDioxide,
    Proton,
    AssemblyTurnover,
    CrowdingPenalty,
    DemandSatisfaction,
}

impl PatchReducerSignal {
    const COUNT: usize = Self::DemandSatisfaction as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct PatchReducerContext {
    signals: [f32; PatchReducerSignal::COUNT],
}

impl Default for PatchReducerContext {
    fn default() -> Self {
        Self {
            signals: [0.0; PatchReducerSignal::COUNT],
        }
    }
}

impl PatchReducerContext {
    fn set(&mut self, signal: PatchReducerSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ PatchReducerSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum AssemblyReducerSignal {
    Oxygen = 0,
    Carbon,
    Energy,
    Nitrogen,
    Phosphorus,
    Biosynthesis,
    Stress,
    StructuralOrder,
    AtpScale,
    TranslationScale,
    ReplicationScale,
    MembraneScale,
    ConstrictionScale,
    DemandSatisfaction,
    ByproductLoad,
}

impl AssemblyReducerSignal {
    const COUNT: usize = Self::ByproductLoad as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct AssemblyReducerContext {
    signals: [f32; AssemblyReducerSignal::COUNT],
}

impl Default for AssemblyReducerContext {
    fn default() -> Self {
        Self {
            signals: [0.0; AssemblyReducerSignal::COUNT],
        }
    }
}

impl AssemblyReducerContext {
    fn set(&mut self, signal: AssemblyReducerSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ AssemblyReducerSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum GlobalReportSignal {
    Glucose = 0,
    Oxygen,
    AtpFlux,
    Nitrate,
    Ammonium,
    CrowdingPenalty,
}

impl GlobalReportSignal {
    const COUNT: usize = Self::CrowdingPenalty as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct GlobalReportContext {
    signals: [f32; GlobalReportSignal::COUNT],
}

impl Default for GlobalReportContext {
    fn default() -> Self {
        Self {
            signals: [0.0; GlobalReportSignal::COUNT],
        }
    }
}

impl GlobalReportContext {
    fn set(&mut self, signal: GlobalReportSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ GlobalReportSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum AtomisticReducerSignal {
    PolarFraction = 0,
    PhosphateFraction,
    HydrogenFraction,
    BondDensity,
    AngleDensity,
    DihedralDensity,
    ChargeDensity,
    StructuralOrder,
    Compactness,
    ShellOrder,
    AxisAnisotropy,
    ThermalStability,
    ElectrostaticOrder,
    VdwCohesion,
    CrowdingPenalty,
}

impl AtomisticReducerSignal {
    const COUNT: usize = Self::CrowdingPenalty as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct AtomisticReducerContext {
    signals: [f32; AtomisticReducerSignal::COUNT],
}

impl Default for AtomisticReducerContext {
    fn default() -> Self {
        Self {
            signals: [0.0; AtomisticReducerSignal::COUNT],
        }
    }
}

impl AtomisticReducerContext {
    fn set(&mut self, signal: AtomisticReducerSignal, value: f32) {
        self.signals[signal as usize] = finite_clamped(value, 0.0, 0.0, 1.0);
    }

    fn scalar(self) -> ScalarContext<{ AtomisticReducerSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const PATCH_CROWDING_RULE: AffineRule<{ CrowdingSignal::COUNT }> =
    AffineRule::new(1.0, [-0.24, -0.18], 0.68, 1.0);

const PATCH_STRUCTURAL_SIGNAL_RULE: AffineRule<{ StructuralReducerSignal::COUNT }> =
    AffineRule::new(0.0, [0.20, 0.28, 0.30, 0.12, 0.10, -0.18], 0.0, 1.2);

const PATCH_BIOSYNTHESIS_SIGNAL_RULE: AffineRule<{ PatchReducerSignal::COUNT }> = AffineRule::new(
    0.0,
    [0.28, 0.28, 0.16, 0.16, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
    0.0,
    1.2,
);

const PATCH_STRESS_SIGNAL_RULE: AffineRule<{ PatchReducerSignal::COUNT }> = AffineRule::new(
    0.26,
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.32, 0.24, 0.18, -0.14, -0.12],
    0.0,
    1.2,
);

const ASSEMBLY_CATALYST_SCALE_RULE: AffineRule<{ AssemblyReducerSignal::COUNT }> = AffineRule::new(
    0.30,
    [
        0.0, 0.0, 0.12, 0.0, 0.0, 0.0, -0.10, 0.18, 0.14, 0.10, 0.10, 0.10, 0.08, 0.0, 0.0,
    ],
    0.25,
    1.8,
);

const ASSEMBLY_SUPPORT_SCALE_RULE: AffineRule<{ AssemblyReducerSignal::COUNT }> = AffineRule::new(
    0.34,
    [
        0.18, 0.18, 0.0, 0.16, 0.10, 0.18, -0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, -0.05,
    ],
    0.25,
    1.6,
);

const SITE_STRUCTURAL_TARGET_RULE: AffineRule<{ StructuralReducerSignal::COUNT }> =
    AffineRule::new(0.20, [0.20, 0.25, 0.30, 0.08, 0.10, -0.10], 0.2, 1.0);

const GLOBAL_ATP_SUPPORT_RULE: AffineRule<{ GlobalReportSignal::COUNT }> =
    AffineRule::new(0.92, [0.0, 0.18, 0.06, 0.0, 0.0, 0.10], 0.85, 1.35);

const GLOBAL_TRANSLATION_SUPPORT_RULE: AffineRule<{ GlobalReportSignal::COUNT }> =
    AffineRule::new(0.92, [0.16, 0.0, 0.0, 0.0, 0.06, 0.08], 0.85, 1.35);

const GLOBAL_NUCLEOTIDE_SUPPORT_RULE: AffineRule<{ GlobalReportSignal::COUNT }> =
    AffineRule::new(0.92, [0.04, 0.0, 0.0, 0.16, 0.0, 0.08], 0.85, 1.35);

const GLOBAL_MEMBRANE_SUPPORT_RULE: AffineRule<{ GlobalReportSignal::COUNT }> =
    AffineRule::new(0.92, [0.0, 0.05, 0.10, 0.0, 0.0, 0.08], 0.85, 1.30);

const ATOMISTIC_ATP_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> = AffineRule::new(
    0.88,
    [
        0.12, 0.24, 0.02, 0.08, 0.04, 0.06, 0.12, 0.12, 0.04, 0.14, 0.06, 0.10, 0.12, 0.12, 0.18,
    ],
    0.75,
    1.35,
);

const ATOMISTIC_TRANSLATION_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> =
    AffineRule::new(
        0.86,
        [
            0.18, 0.02, 0.02, 0.12, 0.10, 0.08, 0.12, 0.16, 0.18, 0.02, 0.04, 0.12, 0.08, 0.10,
            0.12,
        ],
        0.75,
        1.35,
    );

const ATOMISTIC_REPLICATION_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> =
    AffineRule::new(
        0.86,
        [
            0.08, 0.04, 0.00, 0.08, 0.12, 0.18, 0.18, 0.12, 0.02, 0.06, 0.22, 0.10, 0.12, 0.08,
            0.12,
        ],
        0.75,
        1.35,
    );

const ATOMISTIC_SEGREGATION_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> =
    AffineRule::new(
        0.86,
        [
            0.06, 0.02, 0.00, 0.08, 0.10, 0.18, 0.14, 0.10, 0.04, 0.08, 0.26, 0.10, 0.10, 0.10,
            0.12,
        ],
        0.75,
        1.35,
    );

const ATOMISTIC_MEMBRANE_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> =
    AffineRule::new(
        0.86,
        [
            0.14, 0.30, 0.04, 0.08, 0.04, 0.04, 0.10, 0.10, -0.06, 0.22, 0.08, 0.08, 0.12, 0.18,
            0.18,
        ],
        0.75,
        1.35,
    );

const ATOMISTIC_CONSTRICTION_SCALE_RULE: AffineRule<{ AtomisticReducerSignal::COUNT }> =
    AffineRule::new(
        0.86,
        [
            0.08, 0.08, 0.02, 0.14, 0.18, 0.12, 0.10, 0.12, -0.02, 0.24, 0.12, 0.10, 0.10, 0.14,
            0.16,
        ],
        0.75,
        1.35,
    );

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalActivityProfile {
    activity_bias: f32,
    activity_weights: [f32; 8],
    activity_state_weights: [f32; 6],
    activity_min: f32,
    activity_max: f32,
    catalyst_bias: f32,
    catalyst_weights: [f32; 8],
    catalyst_state_weights: [f32; 6],
    catalyst_min: f32,
    catalyst_max: f32,
}

impl LocalActivityProfile {
    fn evaluate_signal(
        bias: f32,
        weights: [f32; 8],
        state_weights: [f32; 6],
        min: f32,
        max: f32,
        signals: LocalPatchSignals,
        state: WholeCellSubsystemState,
    ) -> f32 {
        let state_features = [
            state.atp_scale,
            state.translation_scale,
            state.replication_scale,
            state.segregation_scale,
            state.membrane_scale,
            state.constriction_scale,
        ];
        let mut value = bias;
        for (idx, weight) in weights.iter().enumerate() {
            value += weight * signals.feature(idx);
        }
        for (idx, weight) in state_weights.iter().enumerate() {
            value += weight * state_features[idx];
        }
        value.clamp(min, max)
    }

    fn activity(self, signals: LocalPatchSignals, state: WholeCellSubsystemState) -> f32 {
        Self::evaluate_signal(
            self.activity_bias,
            self.activity_weights,
            self.activity_state_weights,
            self.activity_min,
            self.activity_max,
            signals,
            state,
        )
    }

    fn catalyst(self, signals: LocalPatchSignals, state: WholeCellSubsystemState) -> f32 {
        Self::evaluate_signal(
            self.catalyst_bias,
            self.catalyst_weights,
            self.catalyst_state_weights,
            self.catalyst_min,
            self.catalyst_max,
            signals,
            state,
        )
    }
}

/// Weight order: glucose, oxygen, atp_flux, nitrate, ammonium, phosphorus,
/// assembly_occupancy_delta, assembly_stability_delta.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SupportProfile {
    bias: f32,
    weights: [f32; 8],
    turnover_penalty: f32,
    min: f32,
    max: f32,
}

impl SupportProfile {
    fn evaluate(self, metrics: PatchSpeciesMetrics, assembly: AssemblyState) -> f32 {
        let weighted = self.bias
            + self.weights[0] * metrics.mean_glucose
            + self.weights[1] * metrics.mean_oxygen
            + self.weights[2] * metrics.mean_atp_flux
            + self.weights[3] * metrics.mean_nitrate
            + self.weights[4] * metrics.mean_ammonium
            + self.weights[5] * metrics.mean_phosphorus
            + self.weights[6] * (assembly.occupancy - 1.0)
            + self.weights[7] * (assembly.stability - 1.0)
            - self.turnover_penalty * assembly.turnover;
        weighted.clamp(self.min, self.max)
    }
}

/// Positive weight order: atp, translation, nucleotide, membrane, assembly
/// occupancy delta, assembly stability delta, crowding delta, demand delta.
/// Penalty order: substrate draw, energy draw, biosynthetic draw, byproduct
/// load, assembly turnover.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScaleProfile {
    baseline: f32,
    weights: [f32; 8],
    penalties: [f32; 5],
    min: f32,
    max: f32,
}

impl ScaleProfile {
    fn evaluate(self, report: LocalChemistrySiteReport) -> f32 {
        let drive = self.baseline
            + self.weights[0] * (report.atp_support - 1.0)
            + self.weights[1] * (report.translation_support - 1.0)
            + self.weights[2] * (report.nucleotide_support - 1.0)
            + self.weights[3] * (report.membrane_support - 1.0)
            + self.weights[4] * (report.assembly_occupancy - 1.0)
            + self.weights[5] * (report.assembly_stability - 1.0)
            + self.weights[6] * (report.crowding_penalty - 1.0)
            + self.weights[7] * (report.demand_satisfaction - 1.0)
            - self.penalties[0] * report.substrate_draw
            - self.penalties[1] * report.energy_draw
            - self.penalties[2] * report.biosynthetic_draw
            - self.penalties[3] * report.byproduct_load
            - self.penalties[4] * report.assembly_turnover;
        drive.clamp(self.min, self.max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SubsystemCouplingProfile {
    localization_rule: LocalizationRule,
    assembly_rule: AssemblyRule,
    activity_profile: LocalActivityProfile,
    atp_support: SupportProfile,
    translation_support: SupportProfile,
    nucleotide_support: SupportProfile,
    membrane_support: SupportProfile,
    atp_scale: ScaleProfile,
    translation_scale: ScaleProfile,
    replication_scale: ScaleProfile,
    segregation_scale: ScaleProfile,
    membrane_scale: ScaleProfile,
    constriction_scale: ScaleProfile,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalActivitySignature {
    activity_bias: f32,
    activity_weights: [f32; 8],
    activity_state_weights: [f32; 6],
    activity_min: f32,
    activity_max: f32,
    catalyst_bias: f32,
    catalyst_weights: [f32; 8],
    catalyst_state_weights: [f32; 6],
    catalyst_min: f32,
    catalyst_max: f32,
}

const fn compile_local_activity_profile(signature: LocalActivitySignature) -> LocalActivityProfile {
    LocalActivityProfile {
        activity_bias: signature.activity_bias,
        activity_weights: signature.activity_weights,
        activity_state_weights: signature.activity_state_weights,
        activity_min: signature.activity_min,
        activity_max: signature.activity_max,
        catalyst_bias: signature.catalyst_bias,
        catalyst_weights: signature.catalyst_weights,
        catalyst_state_weights: signature.catalyst_state_weights,
        catalyst_min: signature.catalyst_min,
        catalyst_max: signature.catalyst_max,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SupportSignature {
    bias: f32,
    weights: [f32; 8],
    turnover_penalty: f32,
    min: f32,
    max: f32,
}

const fn compile_support_profile(signature: SupportSignature) -> SupportProfile {
    SupportProfile {
        bias: signature.bias,
        weights: signature.weights,
        turnover_penalty: signature.turnover_penalty,
        min: signature.min,
        max: signature.max,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScaleSignature {
    baseline: f32,
    weights: [f32; 8],
    penalties: [f32; 5],
    min: f32,
    max: f32,
}

const fn compile_scale_profile(signature: ScaleSignature) -> ScaleProfile {
    ScaleProfile {
        baseline: signature.baseline,
        weights: signature.weights,
        penalties: signature.penalties,
        min: signature.min,
        max: signature.max,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalizationSignature {
    name: &'static str,
    patch_radius: usize,
    cue_count: usize,
    cues: [LocalizationCue; 8],
    persistence_weight: f32,
    exclusion_padding: f32,
    exclusion_strength: f32,
}

const fn compile_localization_rule(signature: LocalizationSignature) -> LocalizationRule {
    LocalizationRule::new(
        signature.name,
        signature.patch_radius,
        signature.cue_count,
        signature.cues,
        signature.persistence_weight,
        signature.exclusion_padding,
        signature.exclusion_strength,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AssemblySignature {
    name: &'static str,
    component_count: usize,
    components: [AssemblyComponent; 4],
    target_occupancy: f32,
    stability_scale: f32,
    baseline_turnover: f32,
}

const fn compile_assembly_rule(signature: AssemblySignature) -> AssemblyRule {
    AssemblyRule::new(
        signature.name,
        signature.component_count,
        signature.components,
        signature.target_occupancy,
        signature.stability_scale,
        signature.baseline_turnover,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReactionRuleSignature {
    name: &'static str,
    substrate_count: usize,
    substrates: [ReactionTerm; 4],
    product_count: usize,
    products: [ReactionTerm; 4],
    law: ReactionLaw,
}

const fn compile_reaction_rule(signature: ReactionRuleSignature) -> ReactionRule {
    ReactionRule::new(
        signature.name,
        signature.substrate_count,
        signature.substrates,
        signature.product_count,
        signature.products,
        signature.law,
    )
}

const EMPTY_REACTION_RULE_SIGNATURE: ReactionRuleSignature = ReactionRuleSignature {
    name: "empty_reaction_rule",
    substrate_count: 0,
    substrates: [EMPTY_REACTION_TERM; 4],
    product_count: 0,
    products: [EMPTY_REACTION_TERM; 4],
    law: ReactionLaw::new(0.0, [0.0; 8]),
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReactionProgramSignature {
    rule_count: usize,
    rules: [ReactionRuleSignature; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReactionProgram {
    rule_count: usize,
    rules: [ReactionRule; 4],
}

impl ReactionProgram {
    fn as_slice(&self) -> &[ReactionRule] {
        &self.rules[..self.rule_count]
    }
}

const fn compile_reaction_program(signature: ReactionProgramSignature) -> ReactionProgram {
    ReactionProgram {
        rule_count: signature.rule_count,
        rules: [
            compile_reaction_rule(signature.rules[0]),
            compile_reaction_rule(signature.rules[1]),
            compile_reaction_rule(signature.rules[2]),
            compile_reaction_rule(signature.rules[3]),
        ],
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SubsystemSignature {
    chemistry_site: WholeCellChemistrySite,
    default_interval_steps: u64,
    default_probe_request: LocalMDProbeRequest,
    localization: LocalizationSignature,
    assembly: AssemblySignature,
    activity: LocalActivitySignature,
    atp_support: SupportSignature,
    translation_support: SupportSignature,
    nucleotide_support: SupportSignature,
    membrane_support: SupportSignature,
    atp_scale: ScaleSignature,
    translation_scale: ScaleSignature,
    replication_scale: ScaleSignature,
    segregation_scale: ScaleSignature,
    membrane_scale: ScaleSignature,
    constriction_scale: ScaleSignature,
    reaction_program: ReactionProgramSignature,
}

const fn compile_subsystem_coupling_profile(
    signature: SubsystemSignature,
) -> SubsystemCouplingProfile {
    SubsystemCouplingProfile {
        localization_rule: compile_localization_rule(signature.localization),
        assembly_rule: compile_assembly_rule(signature.assembly),
        activity_profile: compile_local_activity_profile(signature.activity),
        atp_support: compile_support_profile(signature.atp_support),
        translation_support: compile_support_profile(signature.translation_support),
        nucleotide_support: compile_support_profile(signature.nucleotide_support),
        membrane_support: compile_support_profile(signature.membrane_support),
        atp_scale: compile_scale_profile(signature.atp_scale),
        translation_scale: compile_scale_profile(signature.translation_scale),
        replication_scale: compile_scale_profile(signature.replication_scale),
        segregation_scale: compile_scale_profile(signature.segregation_scale),
        membrane_scale: compile_scale_profile(signature.membrane_scale),
        constriction_scale: compile_scale_profile(signature.constriction_scale),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SubsystemRuleBundle {
    chemistry_site: WholeCellChemistrySite,
    default_interval_steps: u64,
    default_probe_request: LocalMDProbeRequest,
    coupling_profile: SubsystemCouplingProfile,
    reaction_program: ReactionProgram,
}

const fn compile_subsystem_rule_bundle(signature: SubsystemSignature) -> SubsystemRuleBundle {
    SubsystemRuleBundle {
        chemistry_site: signature.chemistry_site,
        default_interval_steps: signature.default_interval_steps,
        default_probe_request: signature.default_probe_request,
        coupling_profile: compile_subsystem_coupling_profile(signature),
        reaction_program: compile_reaction_program(signature.reaction_program),
    }
}

const SUBSYSTEM_SIGNATURE_SPEC_JSON: &str = include_str!("../specs/whole_cell_subsystems.json");
const DERIVATION_CALIBRATION_SPEC_JSON: &str =
    include_str!("../specs/whole_cell_derivation_calibration.json");

const EMPTY_LOCAL_ACTIVITY_SIGNATURE: LocalActivitySignature = LocalActivitySignature {
    activity_bias: 0.0,
    activity_weights: [0.0; 8],
    activity_state_weights: [0.0; 6],
    activity_min: 0.0,
    activity_max: 1.0,
    catalyst_bias: 0.0,
    catalyst_weights: [0.0; 8],
    catalyst_state_weights: [0.0; 6],
    catalyst_min: 0.0,
    catalyst_max: 1.0,
};

const EMPTY_SUPPORT_SIGNATURE: SupportSignature = SupportSignature {
    bias: 0.0,
    weights: [0.0; 8],
    turnover_penalty: 0.0,
    min: 0.0,
    max: 1.0,
};

const EMPTY_SCALE_SIGNATURE: ScaleSignature = ScaleSignature {
    baseline: 1.0,
    weights: [0.0; 8],
    penalties: [0.0; 5],
    min: 0.0,
    max: 1.0,
};

const EMPTY_LOCALIZATION_SIGNATURE: LocalizationSignature = LocalizationSignature {
    name: "empty_localization",
    patch_radius: 1,
    cue_count: 0,
    cues: [EMPTY_LOCALIZATION_CUE; 8],
    persistence_weight: 0.0,
    exclusion_padding: 0.0,
    exclusion_strength: 0.0,
};

const EMPTY_ASSEMBLY_SIGNATURE: AssemblySignature = AssemblySignature {
    name: "empty_assembly",
    component_count: 0,
    components: [crate::substrate_ir::EMPTY_ASSEMBLY_COMPONENT; 4],
    target_occupancy: 1.0,
    stability_scale: 0.0,
    baseline_turnover: 0.0,
};

const EMPTY_REACTION_PROGRAM_SIGNATURE: ReactionProgramSignature = ReactionProgramSignature {
    rule_count: 0,
    rules: [EMPTY_REACTION_RULE_SIGNATURE; 4],
};

const EMPTY_PROBE_REQUEST: LocalMDProbeRequest = LocalMDProbeRequest {
    site: WholeCellChemistrySite::Cytosol,
    n_atoms: 0,
    steps: 0,
    dt_ps: 0.0,
    box_size_angstrom: 0.0,
    temperature_k: 0.0,
};

const EMPTY_SUBSYSTEM_SIGNATURE: SubsystemSignature = SubsystemSignature {
    chemistry_site: WholeCellChemistrySite::Cytosol,
    default_interval_steps: 0,
    default_probe_request: EMPTY_PROBE_REQUEST,
    localization: EMPTY_LOCALIZATION_SIGNATURE,
    assembly: EMPTY_ASSEMBLY_SIGNATURE,
    activity: EMPTY_LOCAL_ACTIVITY_SIGNATURE,
    atp_support: EMPTY_SUPPORT_SIGNATURE,
    translation_support: EMPTY_SUPPORT_SIGNATURE,
    nucleotide_support: EMPTY_SUPPORT_SIGNATURE,
    membrane_support: EMPTY_SUPPORT_SIGNATURE,
    atp_scale: EMPTY_SCALE_SIGNATURE,
    translation_scale: EMPTY_SCALE_SIGNATURE,
    replication_scale: EMPTY_SCALE_SIGNATURE,
    segregation_scale: EMPTY_SCALE_SIGNATURE,
    membrane_scale: EMPTY_SCALE_SIGNATURE,
    constriction_scale: EMPTY_SCALE_SIGNATURE,
    reaction_program: EMPTY_REACTION_PROGRAM_SIGNATURE,
};

#[derive(Debug, Deserialize, Clone)]
struct SubsystemSignatureSpec {
    preset: String,
    chemistry_site: String,
    default_interval_steps: u64,
    default_probe_request: LocalMDProbeRequestSpec,
    localization: LocalizationSignatureSpec,
    assembly: AssemblySignatureSpec,
    #[serde(default)]
    process_focus: Option<ProcessFocusSpec>,
    #[serde(default)]
    activity: Option<LocalActivitySignatureSpec>,
    #[serde(default)]
    atp_support: Option<SupportSignatureSpec>,
    #[serde(default)]
    translation_support: Option<SupportSignatureSpec>,
    #[serde(default)]
    nucleotide_support: Option<SupportSignatureSpec>,
    #[serde(default)]
    membrane_support: Option<SupportSignatureSpec>,
    #[serde(default)]
    atp_scale: Option<ScaleSignatureSpec>,
    #[serde(default)]
    translation_scale: Option<ScaleSignatureSpec>,
    #[serde(default)]
    replication_scale: Option<ScaleSignatureSpec>,
    #[serde(default)]
    segregation_scale: Option<ScaleSignatureSpec>,
    #[serde(default)]
    membrane_scale: Option<ScaleSignatureSpec>,
    #[serde(default)]
    constriction_scale: Option<ScaleSignatureSpec>,
    reaction_program: Vec<ReactionRuleSignatureSpec>,
}

#[derive(Debug, Deserialize, Clone)]
struct LocalMDProbeRequestSpec {
    site: String,
    n_atoms: usize,
    steps: usize,
    dt_ps: f32,
    box_size_angstrom: f32,
    temperature_k: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct LocalizationSignatureSpec {
    name: String,
    patch_radius: usize,
    cues: Vec<LocalizationCueSpec>,
    persistence_weight: f32,
    exclusion_padding: f32,
    exclusion_strength: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct LocalizationCueSpec {
    channel: String,
    weight: f32,
    half_saturation: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct AssemblySignatureSpec {
    name: String,
    components: Vec<AssemblyComponentSpec>,
    target_occupancy: f32,
    stability_scale: f32,
    baseline_turnover: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct AssemblyComponentSpec {
    species: String,
    weight: f32,
    half_saturation: f32,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ProcessFocusSpec {
    energy: f32,
    translation: f32,
    replication: f32,
    segregation: f32,
    membrane: f32,
    constriction: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct LocalActivitySignatureSpec {
    activity_bias: f32,
    activity_weights: [f32; 8],
    activity_state_weights: [f32; 6],
    activity_min: f32,
    activity_max: f32,
    catalyst_bias: f32,
    catalyst_weights: [f32; 8],
    catalyst_state_weights: [f32; 6],
    catalyst_min: f32,
    catalyst_max: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct SupportSignatureSpec {
    bias: f32,
    weights: [f32; 8],
    turnover_penalty: f32,
    min: f32,
    max: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct ScaleSignatureSpec {
    baseline: f32,
    weights: [f32; 8],
    penalties: [f32; 5],
    min: f32,
    max: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct ReactionRuleSignatureSpec {
    name: String,
    substrates: Vec<ReactionTermSpec>,
    products: Vec<ReactionTermSpec>,
    law: ReactionLawSpec,
}

#[derive(Debug, Deserialize, Clone)]
struct ReactionTermSpec {
    species: String,
    stoich: f32,
    channel: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ReactionLawSpec {
    base_rate: f32,
    #[serde(default)]
    driver_weights: Option<[f32; 8]>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
struct DerivationCalibrationSearchSpec {
    step_scales: [f32; 4],
    gain_min: f32,
    gain_max: f32,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
struct DerivationCalibrationObjectiveSpec {
    support: f32,
    scale: f32,
    demand: f32,
    activity: f32,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
struct DerivationCalibrationConfig {
    gains: WholeCellDerivationCalibration,
    search: DerivationCalibrationSearchSpec,
    objective: DerivationCalibrationObjectiveSpec,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
pub struct WholeCellDerivationCalibration {
    pub context_occupancy_gain: f32,
    pub context_stability_gain: f32,
    pub context_persistence_gain: f32,
    pub context_focus_gain: f32,
    pub context_turnover_gain: f32,
    pub context_probe_complexity_gain: f32,
    pub activity_resource_gain: f32,
    pub activity_focus_gain: f32,
    pub activity_structural_gain: f32,
    pub activity_state_gain: f32,
    pub activity_stress_gain: f32,
    pub support_bias_gain: f32,
    pub support_weight_gain: f32,
    pub support_turnover_gain: f32,
    pub scale_weight_gain: f32,
    pub scale_penalty_gain: f32,
    pub scale_structural_gain: f32,
    pub reaction_focus_gain: f32,
    pub reaction_channel_gain: f32,
    pub reaction_resource_gain: f32,
    pub reaction_spatial_gain: f32,
    pub reaction_structural_gain: f32,
}

impl WholeCellDerivationCalibration {
    const PARAMETER_COUNT: usize = 22;

    fn clamped(self, search: DerivationCalibrationSearchSpec) -> Self {
        let clamp = |value: f32| value.clamp(search.gain_min, search.gain_max);
        Self {
            context_occupancy_gain: clamp(self.context_occupancy_gain),
            context_stability_gain: clamp(self.context_stability_gain),
            context_persistence_gain: clamp(self.context_persistence_gain),
            context_focus_gain: clamp(self.context_focus_gain),
            context_turnover_gain: clamp(self.context_turnover_gain),
            context_probe_complexity_gain: clamp(self.context_probe_complexity_gain),
            activity_resource_gain: clamp(self.activity_resource_gain),
            activity_focus_gain: clamp(self.activity_focus_gain),
            activity_structural_gain: clamp(self.activity_structural_gain),
            activity_state_gain: clamp(self.activity_state_gain),
            activity_stress_gain: clamp(self.activity_stress_gain),
            support_bias_gain: clamp(self.support_bias_gain),
            support_weight_gain: clamp(self.support_weight_gain),
            support_turnover_gain: clamp(self.support_turnover_gain),
            scale_weight_gain: clamp(self.scale_weight_gain),
            scale_penalty_gain: clamp(self.scale_penalty_gain),
            scale_structural_gain: clamp(self.scale_structural_gain),
            reaction_focus_gain: clamp(self.reaction_focus_gain),
            reaction_channel_gain: clamp(self.reaction_channel_gain),
            reaction_resource_gain: clamp(self.reaction_resource_gain),
            reaction_spatial_gain: clamp(self.reaction_spatial_gain),
            reaction_structural_gain: clamp(self.reaction_structural_gain),
        }
    }

    fn get(self, index: usize) -> f32 {
        match index {
            0 => self.context_occupancy_gain,
            1 => self.context_stability_gain,
            2 => self.context_persistence_gain,
            3 => self.context_focus_gain,
            4 => self.context_turnover_gain,
            5 => self.context_probe_complexity_gain,
            6 => self.activity_resource_gain,
            7 => self.activity_focus_gain,
            8 => self.activity_structural_gain,
            9 => self.activity_state_gain,
            10 => self.activity_stress_gain,
            11 => self.support_bias_gain,
            12 => self.support_weight_gain,
            13 => self.support_turnover_gain,
            14 => self.scale_weight_gain,
            15 => self.scale_penalty_gain,
            16 => self.scale_structural_gain,
            17 => self.reaction_focus_gain,
            18 => self.reaction_channel_gain,
            19 => self.reaction_resource_gain,
            20 => self.reaction_spatial_gain,
            21 => self.reaction_structural_gain,
            _ => 1.0,
        }
    }

    fn with(self, index: usize, value: f32) -> Self {
        let mut next = self;
        match index {
            0 => next.context_occupancy_gain = value,
            1 => next.context_stability_gain = value,
            2 => next.context_persistence_gain = value,
            3 => next.context_focus_gain = value,
            4 => next.context_turnover_gain = value,
            5 => next.context_probe_complexity_gain = value,
            6 => next.activity_resource_gain = value,
            7 => next.activity_focus_gain = value,
            8 => next.activity_structural_gain = value,
            9 => next.activity_state_gain = value,
            10 => next.activity_stress_gain = value,
            11 => next.support_bias_gain = value,
            12 => next.support_weight_gain = value,
            13 => next.support_turnover_gain = value,
            14 => next.scale_weight_gain = value,
            15 => next.scale_penalty_gain = value,
            16 => next.scale_structural_gain = value,
            17 => next.reaction_focus_gain = value,
            18 => next.reaction_channel_gain = value,
            19 => next.reaction_resource_gain = value,
            20 => next.reaction_spatial_gain = value,
            21 => next.reaction_structural_gain = value,
            _ => {}
        }
        next
    }
}

fn derivation_calibration_config() -> &'static DerivationCalibrationConfig {
    static DERIVATION_CALIBRATION_CONFIG: OnceLock<DerivationCalibrationConfig> = OnceLock::new();
    DERIVATION_CALIBRATION_CONFIG.get_or_init(|| {
        serde_json::from_str(DERIVATION_CALIBRATION_SPEC_JSON).unwrap_or_else(|error| {
            panic!("failed to parse whole-cell derivation calibration JSON: {error}")
        })
    })
}

fn derivation_calibration() -> WholeCellDerivationCalibration {
    derivation_calibration_config().gains
}

pub fn default_whole_cell_derivation_calibration() -> WholeCellDerivationCalibration {
    derivation_calibration()
}

#[derive(Debug, Clone, Copy)]
struct ProcessFocus {
    energy: f32,
    translation: f32,
    replication: f32,
    segregation: f32,
    membrane: f32,
    constriction: f32,
}

impl ProcessFocus {
    fn from_spec(spec: ProcessFocusSpec) -> Self {
        Self {
            energy: spec.energy.clamp(0.0, 1.25),
            translation: spec.translation.clamp(0.0, 1.25),
            replication: spec.replication.clamp(0.0, 1.25),
            segregation: spec.segregation.clamp(0.0, 1.25),
            membrane: spec.membrane.clamp(0.0, 1.25),
            constriction: spec.constriction.clamp(0.0, 1.25),
        }
    }

    fn gated(value: f32) -> f32 {
        if value < 0.08 {
            0.0
        } else {
            value.clamp(0.0, 1.25)
        }
    }

    fn max(self) -> f32 {
        self.energy
            .max(self.translation)
            .max(self.replication)
            .max(self.segregation)
            .max(self.membrane)
            .max(self.constriction)
    }

    fn translation_like(self) -> f32 {
        Self::gated(self.translation)
    }

    fn replication_like(self) -> f32 {
        (Self::gated(self.replication) + 0.55 * Self::gated(self.segregation)).clamp(0.0, 1.25)
    }

    fn membrane_like(self) -> f32 {
        (Self::gated(self.membrane) + 0.65 * Self::gated(self.constriction)).clamp(0.0, 1.25)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SpeciesAffinity {
    glucose: f32,
    oxygen: f32,
    atp_flux: f32,
    nitrate: f32,
    ammonium: f32,
    phosphorus: f32,
}

impl SpeciesAffinity {
    fn add_species(&mut self, species: TerrariumSpecies, weight: f32) {
        let weight = weight.max(0.0);
        match species {
            TerrariumSpecies::Glucose => self.glucose += weight,
            TerrariumSpecies::OxygenGas => self.oxygen += weight,
            TerrariumSpecies::AtpFlux => self.atp_flux += weight,
            TerrariumSpecies::Nitrate => self.nitrate += weight,
            TerrariumSpecies::Ammonium => self.ammonium += weight,
            TerrariumSpecies::Phosphorus => self.phosphorus += weight,
            _ => {}
        }
    }

    fn normalized(self) -> Self {
        let total = self.glucose
            + self.oxygen
            + self.atp_flux
            + self.nitrate
            + self.ammonium
            + self.phosphorus;
        if total <= 1.0e-6 {
            return self;
        }

        Self {
            glucose: self.glucose / total,
            oxygen: self.oxygen / total,
            atp_flux: self.atp_flux / total,
            nitrate: self.nitrate / total,
            ammonium: self.ammonium / total,
            phosphorus: self.phosphorus / total,
        }
    }

    fn as_array(self) -> [f32; 6] {
        [
            self.glucose,
            self.oxygen,
            self.atp_flux,
            self.nitrate,
            self.ammonium,
            self.phosphorus,
        ]
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ReactionChannelBalance {
    substrate: f32,
    energy: f32,
    biosynthetic: f32,
    waste: f32,
}

impl ReactionChannelBalance {
    fn accumulate(&mut self, channel: FluxChannel, value: f32) {
        let value = value.max(0.0);
        match channel {
            FluxChannel::Neutral => {}
            FluxChannel::Substrate => self.substrate += value,
            FluxChannel::Energy => self.energy += value,
            FluxChannel::Biosynthetic => self.biosynthetic += value,
            FluxChannel::Waste => self.waste += value,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SpatialPreference {
    boundary: f32,
    center: f32,
    radial_center: f32,
    midplane: f32,
}

#[derive(Debug, Clone, Copy)]
struct DerivationContext {
    calibration: WholeCellDerivationCalibration,
    focus: ProcessFocus,
    resources: SpeciesAffinity,
    channels: ReactionChannelBalance,
    structural_emphasis: f32,
    turnover_pressure: f32,
    waste_pressure: f32,
    probe_complexity: f32,
    spatial: SpatialPreference,
}

impl DerivationContext {
    fn neutral_scale_signature() -> ScaleSignature {
        ScaleSignature {
            baseline: 1.0,
            weights: [0.0; 8],
            penalties: [0.0; 5],
            min: 0.75,
            max: 1.35,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SupportKind {
    Atp,
    Translation,
    Nucleotide,
    Membrane,
}

#[derive(Debug, Clone, Copy)]
enum ScaleKind {
    Atp,
    Translation,
    Replication,
    Segregation,
    Membrane,
    Constriction,
}

fn channel_resource_weight(channel: FluxChannel) -> f32 {
    match channel {
        FluxChannel::Neutral => 0.35,
        FluxChannel::Substrate => 1.0,
        FluxChannel::Energy => 1.1,
        FluxChannel::Biosynthetic => 1.05,
        FluxChannel::Waste => 0.15,
    }
}

fn derive_subsystem_context(
    calibration: WholeCellDerivationCalibration,
    focus: ProcessFocus,
    localization: &LocalizationSignature,
    assembly: &AssemblySignature,
    reaction_specs: &[ReactionRuleSignatureSpec],
    probe_request: LocalMDProbeRequest,
) -> Result<DerivationContext, String> {
    let mut resources = SpeciesAffinity::default();
    let mut channels = ReactionChannelBalance::default();
    let mut spatial = SpatialPreference::default();
    let mut waste_pressure = 0.0;

    for cue in localization.cues.iter().take(localization.cue_count) {
        match cue.channel {
            SpatialChannel::Species(species) => {
                resources.add_species(species, cue.weight.max(0.0) * 0.75);
            }
            SpatialChannel::BoundaryProximity => {
                spatial.boundary += cue.weight.max(0.0);
            }
            SpatialChannel::CenterProximity => {
                spatial.center += cue.weight.max(0.0);
            }
            SpatialChannel::RadialCenterProximity => {
                spatial.radial_center += cue.weight.max(0.0);
            }
            SpatialChannel::VerticalMidplaneProximity => {
                spatial.midplane += cue.weight.max(0.0);
            }
            SpatialChannel::Hydration
            | SpatialChannel::MicrobialActivity
            | SpatialChannel::PlantDrive => {}
        }
    }

    for component in assembly.components.iter().take(assembly.component_count) {
        resources.add_species(component.species, component.weight);
    }

    for rule in reaction_specs {
        for term in &rule.substrates {
            let species = parse_species(&term.species)?;
            let channel = parse_flux_channel(&term.channel)?;
            channels.accumulate(channel, term.stoich);
            resources.add_species(species, term.stoich * channel_resource_weight(channel));
        }
        for term in &rule.products {
            let species = parse_species(&term.species)?;
            let channel = parse_flux_channel(&term.channel)?;
            channels.accumulate(channel, term.stoich * 0.35);
            resources.add_species(
                species,
                term.stoich * channel_resource_weight(channel) * 0.35,
            );
            if channel == FluxChannel::Waste {
                waste_pressure += term.stoich.max(0.0);
            }
        }
    }

    let structural_emphasis =
        ((assembly.target_occupancy - 1.0).max(0.0) * 2.1 * calibration.context_occupancy_gain
            + assembly.stability_scale * 0.7 * calibration.context_stability_gain
            + localization.persistence_weight * 0.2 * calibration.context_persistence_gain
            + focus.max() * 0.18 * calibration.context_focus_gain)
            .clamp(0.0, 1.2);
    let turnover_pressure =
        (assembly.baseline_turnover * 4.0 * calibration.context_turnover_gain).clamp(0.0, 1.0);
    let probe_complexity =
        ((((probe_request.n_atoms as f32 * probe_request.steps as f32) / (36.0 * 14.0)).sqrt())
            * calibration.context_probe_complexity_gain)
            .clamp(0.5, 1.5);

    Ok(DerivationContext {
        calibration,
        focus,
        resources: resources.normalized(),
        channels,
        structural_emphasis,
        turnover_pressure,
        waste_pressure: waste_pressure.clamp(0.0, 1.0),
        probe_complexity,
        spatial,
    })
}

fn derive_local_activity_signature(context: DerivationContext) -> LocalActivitySignature {
    let calibration = context.calibration;
    let resources = context.resources.as_array();
    let translation = context.focus.translation_like();
    let replication = context.focus.replication_like();
    let membrane = context.focus.membrane_like();
    let energy = ProcessFocus::gated(context.focus.energy);
    let segregation = ProcessFocus::gated(context.focus.segregation);
    let constriction = ProcessFocus::gated(context.focus.constriction);
    let activity_bias = (0.06
        + 0.05 * calibration.activity_focus_gain * context.focus.max()
        + 0.04 * calibration.activity_structural_gain * context.structural_emphasis)
        .clamp(0.0, 0.3);
    let catalyst_bias = (0.14
        + 0.06 * calibration.activity_focus_gain * context.focus.max()
        + 0.03 * calibration.activity_structural_gain * context.probe_complexity)
        .clamp(0.0, 0.35);
    let activity_stress = (0.12
        + calibration.activity_stress_gain
            * (0.10 * context.waste_pressure + 0.08 * context.turnover_pressure))
        .clamp(0.08, 0.30);
    let catalyst_stress = (0.10
        + calibration.activity_stress_gain
            * (0.08 * context.waste_pressure + 0.06 * context.turnover_pressure))
        .clamp(0.08, 0.24);

    let state_weight = |focus: f32, base: f32, scale: f32| {
        if focus <= 0.0 {
            0.0
        } else {
            (calibration.activity_state_gain * (base + scale * focus)).clamp(0.0, 0.28)
        }
    };

    LocalActivitySignature {
        activity_bias,
        activity_weights: [
            (0.03
                + calibration.activity_resource_gain * 0.18 * resources[0]
                + calibration.activity_focus_gain * (0.06 * translation + 0.04 * membrane))
                .clamp(-1.0, 1.0),
            (0.03
                + calibration.activity_resource_gain * 0.18 * resources[1]
                + calibration.activity_focus_gain * 0.16 * energy
                + calibration.activity_structural_gain * 0.04 * context.spatial.boundary)
                .clamp(-1.0, 1.0),
            (0.10
                + calibration.activity_resource_gain * 0.20 * resources[2]
                + calibration.activity_focus_gain
                    * (0.14 * energy + 0.08 * translation + 0.08 * replication + 0.08 * membrane))
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.18 * resources[3]
                + calibration.activity_focus_gain * 0.18 * replication)
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.18 * resources[4]
                + calibration.activity_focus_gain * 0.18 * translation)
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.18 * resources[5]
                + calibration.activity_focus_gain * (0.14 * membrane + 0.10 * constriction))
                .clamp(-1.0, 1.0),
            (0.14
                + calibration.activity_structural_gain
                    * (0.08 * context.structural_emphasis + 0.04 * context.probe_complexity))
                .clamp(-1.0, 1.0),
            -activity_stress,
        ],
        activity_state_weights: [
            state_weight(energy, 0.04, 0.16),
            state_weight(translation, 0.04, 0.16),
            state_weight(ProcessFocus::gated(context.focus.replication), 0.04, 0.18),
            state_weight(segregation, 0.04, 0.16),
            state_weight(ProcessFocus::gated(context.focus.membrane), 0.04, 0.16),
            state_weight(constriction, 0.04, 0.18),
        ],
        activity_min: 0.0,
        activity_max: 3.0,
        catalyst_bias,
        catalyst_weights: [
            (0.02
                + calibration.activity_resource_gain * 0.14 * resources[0]
                + calibration.activity_focus_gain * (0.04 * translation + 0.02 * membrane))
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.14 * resources[1]
                + calibration.activity_focus_gain * 0.12 * energy
                + calibration.activity_structural_gain * 0.03 * context.spatial.boundary)
                .clamp(-1.0, 1.0),
            (0.08
                + calibration.activity_resource_gain * 0.16 * resources[2]
                + calibration.activity_focus_gain
                    * (0.12 * energy + 0.06 * translation + 0.06 * replication + 0.06 * membrane))
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.14 * resources[3]
                + calibration.activity_focus_gain * 0.14 * replication)
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.14 * resources[4]
                + calibration.activity_focus_gain * 0.14 * translation)
                .clamp(-1.0, 1.0),
            (0.02
                + calibration.activity_resource_gain * 0.14 * resources[5]
                + calibration.activity_focus_gain * (0.10 * membrane + 0.08 * constriction))
                .clamp(-1.0, 1.0),
            (0.12
                + calibration.activity_structural_gain
                    * (0.06 * context.structural_emphasis + 0.03 * context.probe_complexity))
                .clamp(-1.0, 1.0),
            -catalyst_stress,
        ],
        catalyst_state_weights: [
            state_weight(energy, 0.04, 0.18),
            state_weight(translation, 0.04, 0.18),
            state_weight(ProcessFocus::gated(context.focus.replication), 0.04, 0.20),
            state_weight(segregation, 0.04, 0.18),
            state_weight(ProcessFocus::gated(context.focus.membrane), 0.04, 0.18),
            state_weight(constriction, 0.04, 0.20),
        ],
        catalyst_min: (0.36 + 0.05 * context.probe_complexity).clamp(0.35, 0.5),
        catalyst_max: (1.55 + 0.18 * context.focus.max()).clamp(1.55, 1.85),
    }
}

fn support_emphasis(context: DerivationContext, kind: SupportKind) -> f32 {
    match kind {
        SupportKind::Atp => (0.65 * ProcessFocus::gated(context.focus.energy)
            + 0.20 * context.focus.membrane_like()
            + 0.10 * context.focus.translation_like()
            + 0.05 * context.spatial.boundary)
            .clamp(0.0, 1.2),
        SupportKind::Translation => (0.72 * context.focus.translation_like()
            + 0.12 * ProcessFocus::gated(context.focus.energy)
            + 0.08 * ProcessFocus::gated(context.focus.membrane)
            + 0.08 * context.spatial.center)
            .clamp(0.0, 1.2),
        SupportKind::Nucleotide => (0.72 * context.focus.replication_like()
            + 0.14 * ProcessFocus::gated(context.focus.energy)
            + 0.08 * context.spatial.center)
            .clamp(0.0, 1.2),
        SupportKind::Membrane => (0.60 * context.focus.membrane_like()
            + 0.20 * ProcessFocus::gated(context.focus.energy)
            + 0.12 * ProcessFocus::gated(context.focus.constriction)
            + 0.08 * context.spatial.boundary)
            .clamp(0.0, 1.2),
    }
}

fn support_dependencies(kind: SupportKind) -> [f32; 6] {
    match kind {
        SupportKind::Atp => [0.02, 0.20, 0.10, 0.02, 0.02, 0.04],
        SupportKind::Translation => [0.14, 0.02, 0.08, 0.02, 0.12, 0.02],
        SupportKind::Nucleotide => [0.04, 0.02, 0.06, 0.18, 0.02, 0.04],
        SupportKind::Membrane => [0.04, 0.04, 0.10, 0.02, 0.02, 0.16],
    }
}

fn derive_support_signature(context: DerivationContext, kind: SupportKind) -> SupportSignature {
    let calibration = context.calibration;
    let emphasis = support_emphasis(context, kind);
    let resources = context.resources.as_array();
    let dependencies = support_dependencies(kind);
    let mut weights = [0.0; 8];
    let scale = 0.55 + calibration.support_weight_gain * 0.95 * emphasis;
    for idx in 0..6 {
        weights[idx] = dependencies[idx] * scale * (0.45 + resources[idx]);
    }
    weights[6] = calibration.support_weight_gain
        * (0.04 + 0.08 * context.structural_emphasis)
        * (0.4 + 0.6 * emphasis);
    weights[7] = calibration.support_weight_gain
        * (0.04 + 0.10 * context.structural_emphasis)
        * (0.4 + 0.6 * emphasis);

    SupportSignature {
        bias: (0.90
            + calibration.support_bias_gain
                * (0.06 * emphasis + 0.02 * context.structural_emphasis))
            .clamp(0.82, 0.98),
        weights,
        turnover_penalty: (0.04
            + calibration.support_turnover_gain
                * (0.02 * context.turnover_pressure
                    + 0.01 * context.waste_pressure
                    + 0.01 * emphasis))
            .clamp(0.04, 0.07),
        min: (0.82 + 0.03 * emphasis).clamp(0.82, 0.86),
        max: (1.18
            + 0.24 * emphasis
            + calibration.scale_structural_gain * 0.04 * context.structural_emphasis)
            .clamp(1.18, 1.45),
    }
}

fn scale_focus(context: DerivationContext, kind: ScaleKind) -> f32 {
    match kind {
        ScaleKind::Atp => (0.75 * ProcessFocus::gated(context.focus.energy)
            + 0.15 * context.focus.membrane_like()
            + 0.10 * context.focus.translation_like())
        .clamp(0.0, 1.2),
        ScaleKind::Translation => (0.85 * context.focus.translation_like()
            + 0.10 * ProcessFocus::gated(context.focus.energy))
        .clamp(0.0, 1.2),
        ScaleKind::Replication => (0.90 * ProcessFocus::gated(context.focus.replication)
            + 0.10 * ProcessFocus::gated(context.focus.energy))
        .clamp(0.0, 1.2),
        ScaleKind::Segregation => (0.75 * ProcessFocus::gated(context.focus.segregation)
            + 0.20 * ProcessFocus::gated(context.focus.replication)
            + 0.05 * ProcessFocus::gated(context.focus.energy))
        .clamp(0.0, 1.2),
        ScaleKind::Membrane => (0.85 * ProcessFocus::gated(context.focus.membrane)
            + 0.10 * ProcessFocus::gated(context.focus.energy)
            + 0.05 * ProcessFocus::gated(context.focus.constriction))
        .clamp(0.0, 1.2),
        ScaleKind::Constriction => (0.85 * ProcessFocus::gated(context.focus.constriction)
            + 0.10 * ProcessFocus::gated(context.focus.membrane)
            + 0.05 * ProcessFocus::gated(context.focus.energy))
        .clamp(0.0, 1.2),
    }
}

fn scale_dependencies(kind: ScaleKind) -> [f32; 4] {
    match kind {
        ScaleKind::Atp => [0.90, 0.08, 0.04, 0.10],
        ScaleKind::Translation => [0.10, 0.90, 0.06, 0.04],
        ScaleKind::Replication => [0.08, 0.04, 0.90, 0.04],
        ScaleKind::Segregation => [0.12, 0.02, 0.32, 0.02],
        ScaleKind::Membrane => [0.08, 0.04, 0.04, 0.90],
        ScaleKind::Constriction => [0.10, 0.04, 0.04, 0.92],
    }
}

fn derive_scale_signature(context: DerivationContext, kind: ScaleKind) -> ScaleSignature {
    let calibration = context.calibration;
    let focus = scale_focus(context, kind);
    if focus <= 0.0 {
        return DerivationContext::neutral_scale_signature();
    }

    let dependencies = scale_dependencies(kind);
    let mut weights = [0.0; 8];
    let multiplier = 0.55 + calibration.scale_weight_gain * 0.55 * focus;
    for idx in 0..4 {
        weights[idx] = dependencies[idx] * multiplier;
    }
    weights[4] = calibration.scale_structural_gain
        * (0.10 + 0.10 * context.structural_emphasis)
        * (0.55 + 0.45 * focus);
    weights[5] = calibration.scale_structural_gain
        * (0.12 + 0.08 * context.structural_emphasis)
        * (0.55 + 0.45 * focus);
    weights[6] = (calibration.scale_weight_gain
        * (0.10 + 0.08 * focus + 0.04 * context.probe_complexity.min(1.0)))
    .clamp(0.0, 1.2);
    weights[7] = (calibration.scale_weight_gain * (0.10 + 0.08 * focus)).clamp(0.0, 1.2);
    let demand_scale = 0.45 + 0.55 * focus;

    ScaleSignature {
        baseline: 1.0,
        weights,
        penalties: [
            calibration.scale_penalty_gain
                * (0.04 + 0.08 * context.channels.substrate)
                * demand_scale,
            calibration.scale_penalty_gain * (0.04 + 0.10 * context.channels.energy) * demand_scale,
            calibration.scale_penalty_gain
                * (0.03 + 0.10 * context.channels.biosynthetic)
                * demand_scale,
            calibration.scale_penalty_gain * (0.06 + 0.08 * context.waste_pressure) * demand_scale,
            calibration.scale_penalty_gain
                * (0.05 + 0.06 * context.turnover_pressure)
                * demand_scale,
        ],
        min: (0.75 + 0.03 * focus).clamp(0.75, 0.80),
        max: (1.20
            + 0.20 * focus
            + calibration.scale_structural_gain * 0.05 * context.structural_emphasis)
            .clamp(1.20, 1.45),
    }
}

fn derive_reaction_driver_weights(
    context: DerivationContext,
    rule: &ReactionRuleSignatureSpec,
) -> Result<[f32; 8], String> {
    let calibration = context.calibration;
    let mut resources = SpeciesAffinity::default();
    let mut channels = ReactionChannelBalance::default();

    for term in &rule.substrates {
        let species = parse_species(&term.species)?;
        let channel = parse_flux_channel(&term.channel)?;
        channels.accumulate(channel, term.stoich);
        resources.add_species(species, term.stoich * channel_resource_weight(channel));
    }
    for term in &rule.products {
        let species = parse_species(&term.species)?;
        let channel = parse_flux_channel(&term.channel)?;
        channels.accumulate(channel, term.stoich * 0.35);
        resources.add_species(
            species,
            term.stoich * channel_resource_weight(channel) * 0.35,
        );
    }

    let resources = resources.normalized();
    Ok([
        (0.14
            + calibration.reaction_focus_gain * 0.10 * context.focus.max()
            + calibration.reaction_structural_gain * 0.06 * context.structural_emphasis)
            .clamp(0.0, 0.50),
        (0.06
            + calibration.reaction_focus_gain * 0.14 * ProcessFocus::gated(context.focus.energy)
            + calibration.reaction_channel_gain * 0.08 * channels.energy
            + calibration.reaction_resource_gain * 0.06 * resources.atp_flux
            + calibration.reaction_resource_gain * 0.03 * context.resources.atp_flux)
            .clamp(0.0, 0.50),
        (0.04
            + calibration.reaction_channel_gain * 0.14 * channels.biosynthetic
            + calibration.reaction_focus_gain
                * 0.12
                * context
                    .focus
                    .translation_like()
                    .max(context.focus.replication_like())
                    .max(context.focus.membrane_like()))
        .clamp(0.0, 0.50),
        (calibration.reaction_focus_gain * 0.24 * ProcessFocus::gated(context.focus.replication)
            + calibration.reaction_focus_gain
                * 0.10
                * ProcessFocus::gated(context.focus.segregation)
            + calibration.reaction_resource_gain * 0.08 * resources.nitrate)
            .clamp(0.0, 0.50),
        (calibration.reaction_focus_gain * 0.18 * ProcessFocus::gated(context.focus.membrane)
            + calibration.reaction_focus_gain
                * 0.24
                * ProcessFocus::gated(context.focus.constriction)
            + calibration.reaction_spatial_gain * 0.04 * context.spatial.midplane)
            .clamp(0.0, 0.60),
        (calibration.reaction_resource_gain * 0.18 * resources.oxygen
            + calibration.reaction_spatial_gain * 0.06 * context.spatial.boundary)
            .clamp(0.0, 0.40),
        (calibration.reaction_resource_gain * 0.18 * resources.glucose).clamp(0.0, 0.40),
        (calibration.reaction_focus_gain * 0.20 * context.focus.translation_like()
            + calibration.reaction_resource_gain * 0.08 * resources.ammonium)
            .clamp(0.0, 0.50),
    ])
}

fn leak_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

fn copy_slice_into_array<T: Copy, const N: usize>(
    values: &[T],
    filler: T,
    label: &str,
) -> Result<[T; N], String> {
    if values.len() > N {
        return Err(format!("{label} has {} entries, max {N}", values.len()));
    }

    let mut array = [filler; N];
    for (idx, value) in values.iter().copied().enumerate() {
        array[idx] = value;
    }
    Ok(array)
}

fn parse_chemistry_site(name: &str) -> Result<WholeCellChemistrySite, String> {
    WholeCellChemistrySite::from_name(name)
        .ok_or_else(|| format!("unknown chemistry site in subsystem spec: {name}"))
}

fn parse_species(name: &str) -> Result<TerrariumSpecies, String> {
    TerrariumSpecies::from_name(name)
        .ok_or_else(|| format!("unknown terrarium species in subsystem spec: {name}"))
}

fn parse_spatial_channel(name: &str) -> Result<SpatialChannel, String> {
    SpatialChannel::from_name(name)
        .ok_or_else(|| format!("unknown spatial channel in subsystem spec: {name}"))
}

fn parse_flux_channel(name: &str) -> Result<FluxChannel, String> {
    FluxChannel::from_name(name)
        .ok_or_else(|| format!("unknown flux channel in subsystem spec: {name}"))
}

fn build_probe_request(
    spec: LocalMDProbeRequestSpec,
    chemistry_site: WholeCellChemistrySite,
) -> Result<LocalMDProbeRequest, String> {
    let site = parse_chemistry_site(&spec.site)?;
    if site != chemistry_site {
        return Err(format!(
            "probe site {} does not match subsystem chemistry site {}",
            spec.site,
            chemistry_site.as_str()
        ));
    }

    Ok(LocalMDProbeRequest {
        site,
        n_atoms: spec.n_atoms,
        steps: spec.steps,
        dt_ps: spec.dt_ps,
        box_size_angstrom: spec.box_size_angstrom,
        temperature_k: spec.temperature_k,
    })
}

fn build_localization_signature(
    spec: LocalizationSignatureSpec,
) -> Result<LocalizationSignature, String> {
    let cues = spec
        .cues
        .into_iter()
        .map(|cue| {
            Ok(LocalizationCue::new(
                parse_spatial_channel(&cue.channel)?,
                cue.weight,
                cue.half_saturation,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let cue_count = cues.len();

    Ok(LocalizationSignature {
        name: leak_name(spec.name),
        patch_radius: spec.patch_radius,
        cue_count,
        cues: copy_slice_into_array(&cues, EMPTY_LOCALIZATION_CUE, "localization cues")?,
        persistence_weight: spec.persistence_weight,
        exclusion_padding: spec.exclusion_padding,
        exclusion_strength: spec.exclusion_strength,
    })
}

fn build_assembly_signature(spec: AssemblySignatureSpec) -> Result<AssemblySignature, String> {
    let components = spec
        .components
        .into_iter()
        .map(|component| {
            Ok(AssemblyComponent::new(
                parse_species(&component.species)?,
                component.weight,
                component.half_saturation,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let component_count = components.len();

    Ok(AssemblySignature {
        name: leak_name(spec.name),
        component_count,
        components: copy_slice_into_array(
            &components,
            crate::substrate_ir::EMPTY_ASSEMBLY_COMPONENT,
            "assembly components",
        )?,
        target_occupancy: spec.target_occupancy,
        stability_scale: spec.stability_scale,
        baseline_turnover: spec.baseline_turnover,
    })
}

fn build_local_activity_signature(spec: LocalActivitySignatureSpec) -> LocalActivitySignature {
    LocalActivitySignature {
        activity_bias: spec.activity_bias,
        activity_weights: spec.activity_weights,
        activity_state_weights: spec.activity_state_weights,
        activity_min: spec.activity_min,
        activity_max: spec.activity_max,
        catalyst_bias: spec.catalyst_bias,
        catalyst_weights: spec.catalyst_weights,
        catalyst_state_weights: spec.catalyst_state_weights,
        catalyst_min: spec.catalyst_min,
        catalyst_max: spec.catalyst_max,
    }
}

fn build_support_signature(spec: SupportSignatureSpec) -> SupportSignature {
    SupportSignature {
        bias: spec.bias,
        weights: spec.weights,
        turnover_penalty: spec.turnover_penalty,
        min: spec.min,
        max: spec.max,
    }
}

fn build_scale_signature(spec: ScaleSignatureSpec) -> ScaleSignature {
    ScaleSignature {
        baseline: spec.baseline,
        weights: spec.weights,
        penalties: spec.penalties,
        min: spec.min,
        max: spec.max,
    }
}

fn build_reaction_term(spec: ReactionTermSpec) -> Result<ReactionTerm, String> {
    Ok(ReactionTerm::new(
        parse_species(&spec.species)?,
        spec.stoich,
        parse_flux_channel(&spec.channel)?,
    ))
}

fn build_reaction_rule_signature(
    spec: ReactionRuleSignatureSpec,
    derivation: Option<DerivationContext>,
) -> Result<ReactionRuleSignature, String> {
    let rule_name = spec.name.clone();
    let driver_weights = match spec.law.driver_weights {
        Some(driver_weights) => driver_weights,
        None => {
            let context = derivation.ok_or_else(|| {
                format!(
                    "reaction rule {} is missing explicit driver weights and no derivation context was supplied",
                    rule_name
                )
            })?;
            derive_reaction_driver_weights(context, &spec)?
        }
    };
    let substrates = spec
        .substrates
        .into_iter()
        .map(build_reaction_term)
        .collect::<Result<Vec<_>, String>>()?;
    let products = spec
        .products
        .into_iter()
        .map(build_reaction_term)
        .collect::<Result<Vec<_>, String>>()?;

    Ok(ReactionRuleSignature {
        name: leak_name(spec.name),
        substrate_count: substrates.len(),
        substrates: copy_slice_into_array(&substrates, EMPTY_REACTION_TERM, "reaction substrates")?,
        product_count: products.len(),
        products: copy_slice_into_array(&products, EMPTY_REACTION_TERM, "reaction products")?,
        law: ReactionLaw::new(spec.law.base_rate, driver_weights),
    })
}

fn build_reaction_program_signature(
    specs: Vec<ReactionRuleSignatureSpec>,
    derivation: Option<DerivationContext>,
) -> Result<ReactionProgramSignature, String> {
    let rules = specs
        .into_iter()
        .map(|spec| build_reaction_rule_signature(spec, derivation))
        .collect::<Result<Vec<_>, String>>()?;
    let rule_count = rules.len();

    Ok(ReactionProgramSignature {
        rule_count,
        rules: copy_slice_into_array(&rules, EMPTY_REACTION_RULE_SIGNATURE, "reaction rules")?,
    })
}

fn build_subsystem_signature(
    spec: SubsystemSignatureSpec,
    calibration: WholeCellDerivationCalibration,
) -> Result<SubsystemSignature, String> {
    let chemistry_site = parse_chemistry_site(&spec.chemistry_site)?;
    let focus = spec.process_focus.map(ProcessFocus::from_spec);
    let probe_request = build_probe_request(spec.default_probe_request, chemistry_site)?;
    let localization = build_localization_signature(spec.localization)?;
    let assembly = build_assembly_signature(spec.assembly)?;
    let derivation = match focus {
        Some(focus) => Some(derive_subsystem_context(
            calibration,
            focus,
            &localization,
            &assembly,
            &spec.reaction_program,
            probe_request,
        )?),
        None => None,
    };

    let activity = match (spec.activity, derivation) {
        (Some(explicit), _) => build_local_activity_signature(explicit),
        (None, Some(context)) => derive_local_activity_signature(context),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit activity profile",
                spec.preset
            ))
        }
    };

    let atp_support = match (spec.atp_support, derivation) {
        (Some(explicit), _) => build_support_signature(explicit),
        (None, Some(context)) => derive_support_signature(context, SupportKind::Atp),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit ATP support profile",
                spec.preset
            ))
        }
    };
    let translation_support = match (spec.translation_support, derivation) {
        (Some(explicit), _) => build_support_signature(explicit),
        (None, Some(context)) => derive_support_signature(context, SupportKind::Translation),
        (None, None) => {
            return Err(format!(
            "subsystem {} is missing both process_focus and explicit translation support profile",
            spec.preset
        ))
        }
    };
    let nucleotide_support = match (spec.nucleotide_support, derivation) {
        (Some(explicit), _) => build_support_signature(explicit),
        (None, Some(context)) => derive_support_signature(context, SupportKind::Nucleotide),
        (None, None) => {
            return Err(format!(
            "subsystem {} is missing both process_focus and explicit nucleotide support profile",
            spec.preset
        ))
        }
    };
    let membrane_support = match (spec.membrane_support, derivation) {
        (Some(explicit), _) => build_support_signature(explicit),
        (None, Some(context)) => derive_support_signature(context, SupportKind::Membrane),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit membrane support profile",
                spec.preset
            ))
        }
    };
    let atp_scale = match (spec.atp_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Atp),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit ATP scale profile",
                spec.preset
            ))
        }
    };
    let translation_scale = match (spec.translation_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Translation),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit translation scale profile",
                spec.preset
            ))
        }
    };
    let replication_scale = match (spec.replication_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Replication),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit replication scale profile",
                spec.preset
            ))
        }
    };
    let segregation_scale = match (spec.segregation_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Segregation),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit segregation scale profile",
                spec.preset
            ))
        }
    };
    let membrane_scale = match (spec.membrane_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Membrane),
        (None, None) => {
            return Err(format!(
                "subsystem {} is missing both process_focus and explicit membrane scale profile",
                spec.preset
            ))
        }
    };
    let constriction_scale = match (spec.constriction_scale, derivation) {
        (Some(explicit), _) => build_scale_signature(explicit),
        (None, Some(context)) => derive_scale_signature(context, ScaleKind::Constriction),
        (None, None) => {
            return Err(format!(
            "subsystem {} is missing both process_focus and explicit constriction scale profile",
            spec.preset
        ))
        }
    };

    Ok(SubsystemSignature {
        chemistry_site,
        default_interval_steps: spec.default_interval_steps,
        default_probe_request: probe_request,
        localization,
        assembly,
        activity,
        atp_support,
        translation_support,
        nucleotide_support,
        membrane_support,
        atp_scale,
        translation_scale,
        replication_scale,
        segregation_scale,
        membrane_scale,
        constriction_scale,
        reaction_program: build_reaction_program_signature(spec.reaction_program, derivation)?,
    })
}

fn subsystem_signature_specs() -> &'static [SubsystemSignatureSpec] {
    static SUBSYSTEM_SIGNATURE_SPECS: OnceLock<Vec<SubsystemSignatureSpec>> = OnceLock::new();
    SUBSYSTEM_SIGNATURE_SPECS.get_or_init(|| {
        serde_json::from_str(SUBSYSTEM_SIGNATURE_SPEC_JSON)
            .unwrap_or_else(|error| panic!("failed to parse subsystem signature JSON: {error}"))
    })
}

fn load_subsystem_signatures_with_calibration(
    calibration: WholeCellDerivationCalibration,
) -> Result<[SubsystemSignature; SUBSYSTEM_PRESET_COUNT], String> {
    let mut loaded = [EMPTY_SUBSYSTEM_SIGNATURE; SUBSYSTEM_PRESET_COUNT];
    let mut seen = [false; SUBSYSTEM_PRESET_COUNT];

    for spec in subsystem_signature_specs().iter().cloned() {
        let preset = Syn3ASubsystemPreset::from_name(&spec.preset)
            .ok_or_else(|| format!("unknown subsystem preset in spec: {}", spec.preset))?;
        if seen[preset as usize] {
            return Err(format!(
                "duplicate subsystem preset in spec: {}",
                preset.as_str()
            ));
        }
        loaded[preset as usize] = build_subsystem_signature(spec, calibration)?;
        seen[preset as usize] = true;
    }

    for preset in Syn3ASubsystemPreset::all().iter().copied() {
        if !seen[preset as usize] {
            return Err(format!(
                "missing subsystem preset in spec: {}",
                preset.as_str()
            ));
        }
    }

    Ok(loaded)
}

fn load_subsystem_signatures() -> Result<[SubsystemSignature; SUBSYSTEM_PRESET_COUNT], String> {
    load_subsystem_signatures_with_calibration(derivation_calibration())
}

fn subsystem_signatures() -> &'static [SubsystemSignature; SUBSYSTEM_PRESET_COUNT] {
    static SUBSYSTEM_SIGNATURE_REGISTRY: OnceLock<[SubsystemSignature; SUBSYSTEM_PRESET_COUNT]> =
        OnceLock::new();
    SUBSYSTEM_SIGNATURE_REGISTRY.get_or_init(|| {
        load_subsystem_signatures()
            .unwrap_or_else(|error| panic!("failed to load whole-cell subsystem specs: {error}"))
    })
}

fn subsystem_signature(preset: Syn3ASubsystemPreset) -> &'static SubsystemSignature {
    &subsystem_signatures()[preset as usize]
}

fn subsystem_rule_bundles() -> &'static [SubsystemRuleBundle; SUBSYSTEM_PRESET_COUNT] {
    static SUBSYSTEM_RULE_BUNDLE_REGISTRY: OnceLock<[SubsystemRuleBundle; SUBSYSTEM_PRESET_COUNT]> =
        OnceLock::new();
    SUBSYSTEM_RULE_BUNDLE_REGISTRY.get_or_init(|| {
        let signatures = subsystem_signatures();
        [
            compile_subsystem_rule_bundle(signatures[0]),
            compile_subsystem_rule_bundle(signatures[1]),
            compile_subsystem_rule_bundle(signatures[2]),
            compile_subsystem_rule_bundle(signatures[3]),
        ]
    })
}

fn subsystem_rule_bundle(preset: Syn3ASubsystemPreset) -> &'static SubsystemRuleBundle {
    &subsystem_rule_bundles()[preset as usize]
}

fn build_subsystem_rule_bundles_with_calibration(
    calibration: WholeCellDerivationCalibration,
) -> Result<[SubsystemRuleBundle; SUBSYSTEM_PRESET_COUNT], String> {
    let signatures = load_subsystem_signatures_with_calibration(calibration)?;
    Ok([
        compile_subsystem_rule_bundle(signatures[0]),
        compile_subsystem_rule_bundle(signatures[1]),
        compile_subsystem_rule_bundle(signatures[2]),
        compile_subsystem_rule_bundle(signatures[3]),
    ])
}

fn subsystem_coupling_profile(preset: Syn3ASubsystemPreset) -> &'static SubsystemCouplingProfile {
    &subsystem_rule_bundle(preset).coupling_profile
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalChemistryReport {
    pub atp_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub mean_glucose: f32,
    pub mean_oxygen: f32,
    pub mean_atp_flux: f32,
    pub mean_carbon_dioxide: f32,
}

impl Default for LocalChemistryReport {
    fn default() -> Self {
        Self {
            atp_support: 1.0,
            translation_support: 1.0,
            nucleotide_support: 1.0,
            membrane_support: 1.0,
            crowding_penalty: 1.0,
            mean_glucose: 0.0,
            mean_oxygen: 0.0,
            mean_atp_flux: 0.0,
            mean_carbon_dioxide: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalChemistrySiteReport {
    pub preset: Syn3ASubsystemPreset,
    pub site: WholeCellChemistrySite,
    pub patch_radius: usize,
    pub site_x: usize,
    pub site_y: usize,
    pub site_z: usize,
    pub localization_score: f32,
    pub atp_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub mean_glucose: f32,
    pub mean_oxygen: f32,
    pub mean_atp_flux: f32,
    pub mean_carbon_dioxide: f32,
    pub mean_nitrate: f32,
    pub mean_ammonium: f32,
    pub mean_proton: f32,
    pub mean_phosphorus: f32,
    pub assembly_component_availability: f32,
    pub assembly_occupancy: f32,
    pub assembly_stability: f32,
    pub assembly_turnover: f32,
    pub substrate_draw: f32,
    pub energy_draw: f32,
    pub biosynthetic_draw: f32,
    pub byproduct_load: f32,
    pub demand_satisfaction: f32,
}

impl LocalChemistrySiteReport {
    pub fn as_report(self) -> LocalChemistryReport {
        LocalChemistryReport {
            atp_support: self.atp_support,
            translation_support: self.translation_support,
            nucleotide_support: self.nucleotide_support,
            membrane_support: self.membrane_support,
            crowding_penalty: self.crowding_penalty,
            mean_glucose: self.mean_glucose,
            mean_oxygen: self.mean_oxygen,
            mean_atp_flux: self.mean_atp_flux,
            mean_carbon_dioxide: self.mean_carbon_dioxide,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalChemistryDemandSummary {
    substrate_draw: f32,
    energy_draw: f32,
    biosynthetic_draw: f32,
    byproduct_load: f32,
    demand_satisfaction: f32,
}

impl Default for LocalChemistryDemandSummary {
    fn default() -> Self {
        Self {
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum SnapshotExchangeSignal {
    Pool = 0,
    LocalMean,
    Support,
    Progress,
}

impl SnapshotExchangeSignal {
    const COUNT: usize = Self::Progress as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct SnapshotExchangeContext {
    signals: [f32; SnapshotExchangeSignal::COUNT],
}

impl Default for SnapshotExchangeContext {
    fn default() -> Self {
        Self {
            signals: [0.0; SnapshotExchangeSignal::COUNT],
        }
    }
}

impl SnapshotExchangeContext {
    fn set(&mut self, signal: SnapshotExchangeSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ SnapshotExchangeSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const fn exchange_factor(signal: SnapshotExchangeSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const GLUCOSE_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.02,
    3,
    [
        ScalarBranch::new(
            0.09,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.05,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::LocalMean, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.03,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    1.2,
);

const OXYGEN_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.02,
    3,
    [
        ScalarBranch::new(
            0.10,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.05,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::LocalMean, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.02,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    1.0,
);

const ATP_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    3,
    [
        ScalarBranch::new(
            0.14,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.08,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::LocalMean, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.04,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    1.5,
);

const AMMONIUM_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.01,
    2,
    [
        ScalarBranch::new(
            0.06,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.03,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    0.8,
);

const NITRATE_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.01,
    2,
    [
        ScalarBranch::new(
            0.05,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.03,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    0.8,
);

const PHOSPHORUS_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.004,
    2,
    [
        ScalarBranch::new(
            0.025,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.015,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Support, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    0.20,
);

const CARBON_DIOXIDE_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.01,
    3,
    [
        ScalarBranch::new(
            0.03,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.04,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::LocalMean, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.02,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Progress, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    0.8,
);

const PROTON_EXCHANGE_RULE: ScalarRule = ScalarRule::new(
    0.002,
    2,
    [
        ScalarBranch::new(
            0.006,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Pool, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(
            0.015,
            1,
            [
                exchange_factor(SnapshotExchangeSignal::Progress, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
        ScalarBranch::new(0.0, 0, [EMPTY_SCALAR_FACTOR; 8]),
    ],
    0.0,
    0.3,
);

#[derive(Debug, Clone, Copy, PartialEq)]
struct SnapshotExchangeTargets {
    reactive_species: [(TerrariumSpecies, f32); 8],
}

impl SnapshotExchangeTargets {
    fn from_snapshot(snapshot: &WholeCellSnapshot) -> Self {
        let chemistry = snapshot.local_chemistry.unwrap_or_default();
        let evaluate_target =
            |rule: ScalarRule, pool: f32, local_mean: f32, support: f32, progress: f32| {
                let mut ctx = SnapshotExchangeContext::default();
                ctx.set(SnapshotExchangeSignal::Pool, pool);
                ctx.set(SnapshotExchangeSignal::LocalMean, local_mean);
                ctx.set(SnapshotExchangeSignal::Support, support);
                ctx.set(SnapshotExchangeSignal::Progress, progress);
                rule.evaluate(ctx.scalar())
            };
        Self {
            reactive_species: [
                (
                    TerrariumSpecies::Glucose,
                    evaluate_target(
                        GLUCOSE_EXCHANGE_RULE,
                        snapshot.glucose_mm,
                        chemistry.mean_glucose,
                        chemistry.atp_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::OxygenGas,
                    evaluate_target(
                        OXYGEN_EXCHANGE_RULE,
                        snapshot.oxygen_mm,
                        chemistry.mean_oxygen,
                        chemistry.atp_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::AtpFlux,
                    evaluate_target(
                        ATP_EXCHANGE_RULE,
                        snapshot.atp_mm,
                        chemistry.mean_atp_flux,
                        chemistry.atp_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::Ammonium,
                    evaluate_target(
                        AMMONIUM_EXCHANGE_RULE,
                        snapshot.amino_acids_mm,
                        chemistry.mean_glucose,
                        chemistry.translation_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::Nitrate,
                    evaluate_target(
                        NITRATE_EXCHANGE_RULE,
                        snapshot.nucleotides_mm,
                        chemistry.mean_atp_flux,
                        chemistry.nucleotide_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::Phosphorus,
                    evaluate_target(
                        PHOSPHORUS_EXCHANGE_RULE,
                        snapshot.membrane_precursors_mm,
                        chemistry.mean_oxygen,
                        chemistry.membrane_support,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::CarbonDioxide,
                    evaluate_target(
                        CARBON_DIOXIDE_EXCHANGE_RULE,
                        snapshot.adp_mm,
                        chemistry.mean_carbon_dioxide,
                        chemistry.crowding_penalty,
                        snapshot.division_progress,
                    ),
                ),
                (
                    TerrariumSpecies::Proton,
                    evaluate_target(
                        PROTON_EXCHANGE_RULE,
                        snapshot.adp_mm,
                        chemistry.mean_carbon_dioxide,
                        chemistry.crowding_penalty,
                        snapshot.division_progress,
                    ),
                ),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalMDProbeRequest {
    pub site: WholeCellChemistrySite,
    pub n_atoms: usize,
    pub steps: usize,
    pub dt_ps: f32,
    pub box_size_angstrom: f32,
    pub temperature_k: f32,
}

impl LocalMDProbeRequest {
    pub fn new(site: WholeCellChemistrySite) -> Self {
        Self {
            site,
            ..Self::default()
        }
    }
}

impl Default for LocalMDProbeRequest {
    fn default() -> Self {
        Self {
            site: WholeCellChemistrySite::Cytosol,
            n_atoms: 32,
            steps: 32,
            dt_ps: 0.001,
            box_size_angstrom: 18.0,
            temperature_k: 310.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalMDProbeReport {
    pub site: WholeCellChemistrySite,
    pub mean_temperature: f32,
    pub mean_total_energy: f32,
    pub mean_vdw_energy: f32,
    pub mean_electrostatic_energy: f32,
    pub structural_order: f32,
    pub crowding_penalty: f32,
    pub compactness: f32,
    pub shell_order: f32,
    pub axis_anisotropy: f32,
    pub thermal_stability: f32,
    pub electrostatic_order: f32,
    pub vdw_cohesion: f32,
    pub polar_fraction: f32,
    pub phosphate_fraction: f32,
    pub hydrogen_fraction: f32,
    pub bond_density: f32,
    pub angle_density: f32,
    pub dihedral_density: f32,
    pub charge_density: f32,
    pub recommended_atp_scale: f32,
    pub recommended_translation_scale: f32,
    pub recommended_replication_scale: f32,
    pub recommended_segregation_scale: f32,
    pub recommended_membrane_scale: f32,
    pub recommended_constriction_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AtomisticDynamicDescriptor {
    structural_order: f32,
    compactness: f32,
    shell_order: f32,
    axis_anisotropy: f32,
    thermal_stability: f32,
    electrostatic_order: f32,
    vdw_cohesion: f32,
    crowding_penalty: f32,
}

fn generic_atomistic_probe_descriptor(
    n_atoms: usize,
    bond_count: usize,
    angle_count: usize,
    dihedral_count: usize,
    charges: &[f32],
) -> AtomisticTemplateDescriptor {
    let atom_count = n_atoms.max(1) as f32;
    let charge_density = charges.iter().map(|charge| charge.abs()).sum::<f32>() / atom_count;
    AtomisticTemplateDescriptor::new(
        0.50,
        0.16,
        0.16,
        bond_count as f32 / atom_count,
        angle_count as f32 / atom_count,
        dihedral_count as f32 / atom_count,
        charge_density,
    )
}

fn atomistic_dynamic_descriptor(
    positions: &[f32],
    center: f32,
    box_size: f32,
    target_temperature: f32,
    mean_temperature: f32,
    mean_vdw_energy: f32,
    mean_electrostatic_energy: f32,
    structural_order: f32,
    crowding_penalty: f32,
) -> AtomisticDynamicDescriptor {
    let n_atoms = (positions.len() / 3).max(1);
    let mut mean_radius = 0.0;
    let mut mean_radius_sq = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut var_z = 0.0;
    for i in 0..n_atoms {
        let i3 = i * 3;
        let dx = positions[i3] - center;
        let dy = positions[i3 + 1] - center;
        let dz = positions[i3 + 2] - center;
        let radius = (dx * dx + dy * dy + dz * dz).sqrt();
        mean_radius += radius;
        mean_radius_sq += radius * radius;
        var_x += dx * dx;
        var_y += dy * dy;
        var_z += dz * dz;
    }
    let n_atoms_f = n_atoms as f32;
    mean_radius /= n_atoms_f;
    mean_radius_sq /= n_atoms_f;
    var_x /= n_atoms_f;
    var_y /= n_atoms_f;
    var_z /= n_atoms_f;

    let mut axes = [var_x, var_y, var_z];
    axes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let axis_sum = axes[0] + axes[1] + axes[2];
    let axis_anisotropy = if axis_sum <= 1.0e-6 {
        0.0
    } else {
        ((axes[2] - axes[0]) / axis_sum).clamp(0.0, 1.0)
    };

    let half_box = (box_size * 0.5).max(1.0);
    let radial_fraction = (mean_radius / half_box).clamp(0.0, 1.0);
    let _variance = (mean_radius_sq - mean_radius * mean_radius).max(0.0);
    let compactness = (1.0 - radial_fraction).clamp(0.0, 1.0);
    let shell_order = (radial_fraction * structural_order).clamp(0.0, 1.0);
    let thermal_stability = (1.0
        / (1.0
            + 4.0 * ((mean_temperature - target_temperature).abs() / target_temperature.max(1.0))))
    .clamp(0.0, 1.0);
    let electrostatic_order =
        (1.0 / (1.0 + mean_electrostatic_energy.abs() / 120.0)).clamp(0.0, 1.0);
    let vdw_cohesion = (1.0 / (1.0 + mean_vdw_energy.abs() / 180.0)).clamp(0.0, 1.0);

    AtomisticDynamicDescriptor {
        structural_order,
        compactness,
        shell_order,
        axis_anisotropy,
        thermal_stability,
        electrostatic_order,
        vdw_cohesion,
        crowding_penalty,
    }
}

fn atomistic_scale_context(
    template: AtomisticTemplateDescriptor,
    dynamic: AtomisticDynamicDescriptor,
) -> ScalarContext<{ AtomisticReducerSignal::COUNT }> {
    let mut ctx = AtomisticReducerContext::default();
    ctx.set(
        AtomisticReducerSignal::PolarFraction,
        template.polar_fraction,
    );
    ctx.set(
        AtomisticReducerSignal::PhosphateFraction,
        template.phosphate_fraction,
    );
    ctx.set(
        AtomisticReducerSignal::HydrogenFraction,
        template.hydrogen_fraction,
    );
    ctx.set(
        AtomisticReducerSignal::BondDensity,
        saturating_signal(template.bond_density, 0.45),
    );
    ctx.set(
        AtomisticReducerSignal::AngleDensity,
        saturating_signal(template.angle_density, 0.45),
    );
    ctx.set(
        AtomisticReducerSignal::DihedralDensity,
        saturating_signal(template.dihedral_density, 0.30),
    );
    ctx.set(
        AtomisticReducerSignal::ChargeDensity,
        saturating_signal(template.charge_density, 0.16),
    );
    ctx.set(
        AtomisticReducerSignal::StructuralOrder,
        dynamic.structural_order,
    );
    ctx.set(AtomisticReducerSignal::Compactness, dynamic.compactness);
    ctx.set(AtomisticReducerSignal::ShellOrder, dynamic.shell_order);
    ctx.set(
        AtomisticReducerSignal::AxisAnisotropy,
        dynamic.axis_anisotropy,
    );
    ctx.set(
        AtomisticReducerSignal::ThermalStability,
        dynamic.thermal_stability,
    );
    ctx.set(
        AtomisticReducerSignal::ElectrostaticOrder,
        dynamic.electrostatic_order,
    );
    ctx.set(AtomisticReducerSignal::VdwCohesion, dynamic.vdw_cohesion);
    ctx.set(
        AtomisticReducerSignal::CrowdingPenalty,
        dynamic.crowding_penalty,
    );
    ctx.scalar()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WholeCellDerivationCalibrationSample {
    pub preset: Syn3ASubsystemPreset,
    pub dt_ms: f32,
    pub site_report: LocalChemistrySiteReport,
    pub md_report: LocalMDProbeReport,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WholeCellDerivationCalibrationFit {
    pub sample_count: usize,
    pub baseline_loss: f32,
    pub fitted_loss: f32,
    pub support_rmse: f32,
    pub scale_rmse: f32,
    pub demand_rmse: f32,
    pub activity_rmse: f32,
    pub calibration: WholeCellDerivationCalibration,
}

/// Persistent per-subsystem coupling state used by the native whole-cell runtime.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSubsystemState {
    pub preset: Syn3ASubsystemPreset,
    pub site: WholeCellChemistrySite,
    pub site_x: usize,
    pub site_y: usize,
    pub site_z: usize,
    pub localization_score: f32,
    pub structural_order: f32,
    pub crowding_penalty: f32,
    pub assembly_component_availability: f32,
    pub assembly_occupancy: f32,
    pub assembly_stability: f32,
    pub assembly_turnover: f32,
    pub substrate_draw: f32,
    pub energy_draw: f32,
    pub biosynthetic_draw: f32,
    pub byproduct_load: f32,
    pub demand_satisfaction: f32,
    pub atp_scale: f32,
    pub translation_scale: f32,
    pub replication_scale: f32,
    pub segregation_scale: f32,
    pub membrane_scale: f32,
    pub constriction_scale: f32,
    pub last_probe_step: Option<u64>,
}

impl WholeCellSubsystemState {
    pub fn new(preset: Syn3ASubsystemPreset) -> Self {
        Self {
            preset,
            site: preset.chemistry_site(),
            site_x: 0,
            site_y: 0,
            site_z: 0,
            localization_score: 0.0,
            structural_order: 1.0,
            crowding_penalty: 1.0,
            assembly_component_availability: 1.0,
            assembly_occupancy: 1.0,
            assembly_stability: 1.0,
            assembly_turnover: 0.0,
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
            atp_scale: 1.0,
            translation_scale: 1.0,
            replication_scale: 1.0,
            segregation_scale: 1.0,
            membrane_scale: 1.0,
            constriction_scale: 1.0,
            last_probe_step: None,
        }
    }

    pub fn apply_chemistry_report(&mut self, report: LocalChemistryReport) {
        self.apply_site_report(LocalChemistrySiteReport {
            preset: self.preset,
            site: self.site,
            patch_radius: 1,
            site_x: self.site_x,
            site_y: self.site_y,
            site_z: self.site_z,
            localization_score: self.localization_score,
            atp_support: report.atp_support,
            translation_support: report.translation_support,
            nucleotide_support: report.nucleotide_support,
            membrane_support: report.membrane_support,
            crowding_penalty: report.crowding_penalty,
            mean_glucose: report.mean_glucose,
            mean_oxygen: report.mean_oxygen,
            mean_atp_flux: report.mean_atp_flux,
            mean_carbon_dioxide: report.mean_carbon_dioxide,
            mean_nitrate: 0.0,
            mean_ammonium: 0.0,
            mean_proton: 0.0,
            mean_phosphorus: 0.0,
            assembly_component_availability: 1.0,
            assembly_occupancy: 1.0,
            assembly_stability: 1.0,
            assembly_turnover: 0.0,
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
        });
    }

    pub fn apply_site_report(&mut self, report: LocalChemistrySiteReport) {
        let profile = subsystem_coupling_profile(self.preset);
        self.site = report.site;
        self.site_x = report.site_x;
        self.site_y = report.site_y;
        self.site_z = report.site_z;
        self.localization_score = finite_clamped(
            report.localization_score,
            self.localization_score,
            -10.0,
            10.0,
        );
        self.assembly_component_availability = finite_clamped(
            report.assembly_component_availability,
            self.assembly_component_availability,
            0.0,
            1.0,
        );
        self.assembly_occupancy =
            finite_clamped(report.assembly_occupancy, self.assembly_occupancy, 0.0, 1.5);
        self.assembly_stability =
            finite_clamped(report.assembly_stability, self.assembly_stability, 0.0, 1.5);
        self.assembly_turnover =
            finite_clamped(report.assembly_turnover, self.assembly_turnover, 0.0, 1.5);
        self.substrate_draw = finite_clamped(report.substrate_draw, self.substrate_draw, 0.0, 4.0);
        self.energy_draw = finite_clamped(report.energy_draw, self.energy_draw, 0.0, 4.0);
        self.biosynthetic_draw =
            finite_clamped(report.biosynthetic_draw, self.biosynthetic_draw, 0.0, 4.0);
        self.byproduct_load = finite_clamped(report.byproduct_load, self.byproduct_load, 0.0, 4.0);
        self.demand_satisfaction = finite_clamped(
            report.demand_satisfaction,
            self.demand_satisfaction,
            0.35,
            1.0,
        );
        self.crowding_penalty =
            finite_clamped(report.crowding_penalty, self.crowding_penalty, 0.65, 1.0);
        let mut structural_ctx = StructuralReducerContext::default();
        structural_ctx.set(
            StructuralReducerSignal::AssemblyAvailability,
            report.assembly_component_availability,
        );
        structural_ctx.set(
            StructuralReducerSignal::AssemblyOccupancy,
            report.assembly_occupancy,
        );
        structural_ctx.set(
            StructuralReducerSignal::AssemblyStability,
            report.assembly_stability,
        );
        structural_ctx.set(
            StructuralReducerSignal::StructuralOrder,
            report.demand_satisfaction,
        );
        structural_ctx.set(
            StructuralReducerSignal::CrowdingPenalty,
            report.crowding_penalty,
        );
        structural_ctx.set(
            StructuralReducerSignal::AssemblyTurnover,
            report.assembly_turnover,
        );
        let structural_target = SITE_STRUCTURAL_TARGET_RULE.evaluate(structural_ctx.scalar());
        self.structural_order = finite_clamped(
            0.60 * self.structural_order + 0.40 * structural_target,
            self.structural_order,
            0.2,
            1.0,
        );

        self.atp_scale = finite_clamped(
            0.65 * self.atp_scale + 0.35 * profile.atp_scale.evaluate(report),
            self.atp_scale,
            profile.atp_scale.min,
            profile.atp_scale.max,
        );
        self.translation_scale = finite_clamped(
            0.65 * self.translation_scale + 0.35 * profile.translation_scale.evaluate(report),
            self.translation_scale,
            profile.translation_scale.min,
            profile.translation_scale.max,
        );
        self.replication_scale = finite_clamped(
            0.65 * self.replication_scale + 0.35 * profile.replication_scale.evaluate(report),
            self.replication_scale,
            profile.replication_scale.min,
            profile.replication_scale.max,
        );
        self.segregation_scale = finite_clamped(
            0.65 * self.segregation_scale + 0.35 * profile.segregation_scale.evaluate(report),
            self.segregation_scale,
            profile.segregation_scale.min,
            profile.segregation_scale.max,
        );
        self.membrane_scale = finite_clamped(
            0.65 * self.membrane_scale + 0.35 * profile.membrane_scale.evaluate(report),
            self.membrane_scale,
            profile.membrane_scale.min,
            profile.membrane_scale.max,
        );
        self.constriction_scale = finite_clamped(
            0.65 * self.constriction_scale + 0.35 * profile.constriction_scale.evaluate(report),
            self.constriction_scale,
            profile.constriction_scale.min,
            profile.constriction_scale.max,
        );
    }

    pub fn apply_probe_report(&mut self, report: LocalMDProbeReport, step_count: u64) {
        self.site = report.site;
        self.structural_order =
            finite_clamped(report.structural_order, self.structural_order, 0.2, 1.0);
        self.crowding_penalty =
            finite_clamped(report.crowding_penalty, self.crowding_penalty, 0.65, 1.0);
        self.atp_scale = finite_clamped(
            0.55 * self.atp_scale + 0.45 * report.recommended_atp_scale,
            self.atp_scale,
            0.70,
            1.45,
        );
        self.translation_scale = finite_clamped(
            0.55 * self.translation_scale + 0.45 * report.recommended_translation_scale,
            self.translation_scale,
            0.70,
            1.45,
        );
        self.replication_scale = finite_clamped(
            0.55 * self.replication_scale + 0.45 * report.recommended_replication_scale,
            self.replication_scale,
            0.70,
            1.45,
        );
        self.segregation_scale = finite_clamped(
            0.55 * self.segregation_scale + 0.45 * report.recommended_segregation_scale,
            self.segregation_scale,
            0.70,
            1.45,
        );
        self.membrane_scale = finite_clamped(
            0.55 * self.membrane_scale + 0.45 * report.recommended_membrane_scale,
            self.membrane_scale,
            0.70,
            1.45,
        );
        self.constriction_scale = finite_clamped(
            0.55 * self.constriction_scale + 0.45 * report.recommended_constriction_scale,
            self.constriction_scale,
            0.70,
            1.45,
        );
        self.last_probe_step = Some(step_count);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CalibrationLossAccumulator {
    support_sq: f32,
    support_count: usize,
    scale_sq: f32,
    scale_count: usize,
    demand_sq: f32,
    demand_count: usize,
    activity_sq: f32,
    activity_count: usize,
}

impl CalibrationLossAccumulator {
    fn add_support(&mut self, predicted: f32, target: f32) {
        self.support_sq += (predicted - target).powi(2);
        self.support_count += 1;
    }

    fn add_scale(&mut self, predicted: f32, target: f32) {
        self.scale_sq += (predicted - target).powi(2);
        self.scale_count += 1;
    }

    fn add_demand(&mut self, predicted: f32, target: f32) {
        self.demand_sq += (predicted - target).powi(2);
        self.demand_count += 1;
    }

    fn add_activity(&mut self, predicted: f32, target: f32) {
        self.activity_sq += (predicted - target).powi(2);
        self.activity_count += 1;
    }

    fn mse(sum_sq: f32, count: usize) -> f32 {
        if count == 0 {
            0.0
        } else {
            sum_sq / count as f32
        }
    }

    fn rmse(sum_sq: f32, count: usize) -> f32 {
        Self::mse(sum_sq, count).sqrt()
    }

    fn support_rmse(self) -> f32 {
        Self::rmse(self.support_sq, self.support_count)
    }

    fn scale_rmse(self) -> f32 {
        Self::rmse(self.scale_sq, self.scale_count)
    }

    fn demand_rmse(self) -> f32 {
        Self::rmse(self.demand_sq, self.demand_count)
    }

    fn activity_rmse(self) -> f32 {
        Self::rmse(self.activity_sq, self.activity_count)
    }

    fn total_loss(self, objective: DerivationCalibrationObjectiveSpec) -> f32 {
        objective.support * Self::mse(self.support_sq, self.support_count)
            + objective.scale * Self::mse(self.scale_sq, self.scale_count)
            + objective.demand * Self::mse(self.demand_sq, self.demand_count)
            + objective.activity * Self::mse(self.activity_sq, self.activity_count)
    }
}

fn representative_calibration_snapshot() -> WholeCellSnapshot {
    WholeCellSnapshot {
        backend: WholeCellBackend::Cpu,
        time_ms: 8.0,
        step_count: 16,
        atp_mm: 1.08,
        amino_acids_mm: 0.96,
        nucleotides_mm: 0.88,
        membrane_precursors_mm: 0.82,
        adp_mm: 0.24,
        glucose_mm: 0.92,
        oxygen_mm: 0.86,
        ftsz: 42.0,
        dnaa: 18.0,
        active_ribosomes: 72.0,
        active_rnap: 20.0,
        genome_bp: 543_000,
        replicated_bp: 126_000,
        chromosome_separation_nm: 72.0,
        radius_nm: 210.0,
        surface_area_nm2: 554_176.9,
        volume_nm3: 38_792_384.0,
        division_progress: 0.24,
        quantum_profile: WholeCellQuantumProfile::default(),
        local_chemistry: None,
        local_chemistry_sites: Vec::new(),
        local_md_probe: None,
        subsystem_states: Vec::new(),
    }
}

fn patch_metrics_from_site_report(report: LocalChemistrySiteReport) -> PatchSpeciesMetrics {
    PatchSpeciesMetrics {
        mean_glucose: report.mean_glucose,
        mean_oxygen: report.mean_oxygen,
        mean_atp_flux: report.mean_atp_flux,
        mean_carbon_dioxide: report.mean_carbon_dioxide,
        mean_nitrate: report.mean_nitrate,
        mean_ammonium: report.mean_ammonium,
        mean_proton: report.mean_proton,
        mean_phosphorus: report.mean_phosphorus,
    }
}

fn assembly_from_site_report(report: LocalChemistrySiteReport) -> AssemblyState {
    AssemblyState {
        component_availability: report.assembly_component_availability,
        occupancy: report.assembly_occupancy,
        stability: report.assembly_stability,
        turnover: report.assembly_turnover,
    }
}

fn state_from_calibration_sample(
    sample: WholeCellDerivationCalibrationSample,
) -> WholeCellSubsystemState {
    WholeCellSubsystemState {
        preset: sample.preset,
        site: sample.site_report.site,
        site_x: sample.site_report.site_x,
        site_y: sample.site_report.site_y,
        site_z: sample.site_report.site_z,
        localization_score: sample.site_report.localization_score,
        structural_order: sample.md_report.structural_order.clamp(0.2, 1.0),
        crowding_penalty: sample.site_report.crowding_penalty.clamp(0.65, 1.0),
        assembly_component_availability: sample.site_report.assembly_component_availability,
        assembly_occupancy: sample.site_report.assembly_occupancy,
        assembly_stability: sample.site_report.assembly_stability,
        assembly_turnover: sample.site_report.assembly_turnover,
        substrate_draw: sample.site_report.substrate_draw,
        energy_draw: sample.site_report.energy_draw,
        biosynthetic_draw: sample.site_report.biosynthetic_draw,
        byproduct_load: sample.site_report.byproduct_load,
        demand_satisfaction: sample.site_report.demand_satisfaction,
        atp_scale: sample.md_report.recommended_atp_scale,
        translation_scale: sample.md_report.recommended_translation_scale,
        replication_scale: sample.md_report.recommended_replication_scale,
        segregation_scale: sample.md_report.recommended_segregation_scale,
        membrane_scale: sample.md_report.recommended_membrane_scale,
        constriction_scale: sample.md_report.recommended_constriction_scale,
        last_probe_step: Some(0),
    }
}

fn activity_targets_from_sample(sample: WholeCellDerivationCalibrationSample) -> (f32, f32) {
    let total_draw = sample.site_report.substrate_draw
        + sample.site_report.energy_draw
        + sample.site_report.biosynthetic_draw;
    let activity_target = (0.18
        + 1.10 * sample.site_report.demand_satisfaction
        + 0.42 * total_draw
        + 0.22 * (sample.site_report.assembly_occupancy - 1.0).max(0.0)
        - 0.18 * sample.site_report.byproduct_load)
        .clamp(0.0, 3.0);
    let catalyst_target = (0.40
        + 0.55 * sample.md_report.structural_order
        + 0.20 * sample.site_report.assembly_stability
        + 0.10 * sample.site_report.demand_satisfaction
        - 0.16 * (1.0 - sample.site_report.crowding_penalty))
        .clamp(0.35, 1.85);
    (activity_target, catalyst_target)
}

fn predicted_demand_from_bundle(
    bundle: &SubsystemRuleBundle,
    sample: WholeCellDerivationCalibrationSample,
) -> LocalChemistryDemandSummary {
    let metrics = patch_metrics_from_site_report(sample.site_report);
    let assembly = assembly_from_site_report(sample.site_report);
    let state = state_from_calibration_sample(sample);
    let signals = WholeCellChemistryBridge::local_patch_signals(
        metrics,
        assembly,
        state,
        sample.site_report.crowding_penalty,
    );
    let context =
        WholeCellChemistryBridge::reaction_context_from_local_patch(sample.preset, signals, state);
    let demand_scale = sample.site_report.demand_satisfaction.clamp(0.35, 1.0);
    let mut summary = LocalChemistryDemandSummary {
        demand_satisfaction: sample.site_report.demand_satisfaction,
        ..LocalChemistryDemandSummary::default()
    };

    for rule in bundle.reaction_program.as_slice() {
        let target = rule.law.target_amount(context, sample.dt_ms) * demand_scale;
        for term in rule.substrates.iter().take(rule.substrate_count) {
            match term.channel {
                FluxChannel::Substrate => {
                    summary.substrate_draw += term.stoich.max(0.0) * target;
                }
                FluxChannel::Energy => {
                    summary.energy_draw += term.stoich.max(0.0) * target;
                }
                FluxChannel::Biosynthetic => {
                    summary.biosynthetic_draw += term.stoich.max(0.0) * target;
                }
                FluxChannel::Neutral | FluxChannel::Waste => {}
            }
        }
        for term in rule.products.iter().take(rule.product_count) {
            if term.channel == FluxChannel::Waste {
                summary.byproduct_load += term.stoich.max(0.0) * target;
            }
        }
    }

    summary
}

fn evaluate_calibration_samples(
    calibration: WholeCellDerivationCalibration,
    samples: &[WholeCellDerivationCalibrationSample],
) -> Result<CalibrationLossAccumulator, String> {
    let bundles = build_subsystem_rule_bundles_with_calibration(calibration)?;
    let mut loss = CalibrationLossAccumulator::default();

    for sample in samples {
        let bundle = &bundles[sample.preset as usize];
        let metrics = patch_metrics_from_site_report(sample.site_report);
        let assembly = assembly_from_site_report(sample.site_report);
        let state = state_from_calibration_sample(*sample);
        let signals = WholeCellChemistryBridge::local_patch_signals(
            metrics,
            assembly,
            state,
            sample.site_report.crowding_penalty,
        );
        let activity = bundle
            .coupling_profile
            .activity_profile
            .activity(signals, state);
        let catalyst = bundle
            .coupling_profile
            .activity_profile
            .catalyst(signals, state);
        let (target_activity, target_catalyst) = activity_targets_from_sample(*sample);

        loss.add_activity(activity, target_activity);
        loss.add_activity(catalyst, target_catalyst);

        loss.add_support(
            bundle
                .coupling_profile
                .atp_support
                .evaluate(metrics, assembly),
            sample.site_report.atp_support,
        );
        loss.add_support(
            bundle
                .coupling_profile
                .translation_support
                .evaluate(metrics, assembly),
            sample.site_report.translation_support,
        );
        loss.add_support(
            bundle
                .coupling_profile
                .nucleotide_support
                .evaluate(metrics, assembly),
            sample.site_report.nucleotide_support,
        );
        loss.add_support(
            bundle
                .coupling_profile
                .membrane_support
                .evaluate(metrics, assembly),
            sample.site_report.membrane_support,
        );

        loss.add_scale(
            bundle
                .coupling_profile
                .atp_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_atp_scale,
        );
        loss.add_scale(
            bundle
                .coupling_profile
                .translation_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_translation_scale,
        );
        loss.add_scale(
            bundle
                .coupling_profile
                .replication_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_replication_scale,
        );
        loss.add_scale(
            bundle
                .coupling_profile
                .segregation_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_segregation_scale,
        );
        loss.add_scale(
            bundle
                .coupling_profile
                .membrane_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_membrane_scale,
        );
        loss.add_scale(
            bundle
                .coupling_profile
                .constriction_scale
                .evaluate(sample.site_report),
            sample.md_report.recommended_constriction_scale,
        );

        let demand = predicted_demand_from_bundle(bundle, *sample);
        loss.add_demand(demand.substrate_draw, sample.site_report.substrate_draw);
        loss.add_demand(demand.energy_draw, sample.site_report.energy_draw);
        loss.add_demand(
            demand.biosynthetic_draw,
            sample.site_report.biosynthetic_draw,
        );
        loss.add_demand(demand.byproduct_load, sample.site_report.byproduct_load);
    }

    Ok(loss)
}

fn fit_derivation_calibration_from_samples(
    samples: &[WholeCellDerivationCalibrationSample],
    initial: WholeCellDerivationCalibration,
) -> Result<WholeCellDerivationCalibrationFit, String> {
    let config = *derivation_calibration_config();
    let search = config.search;
    let objective = config.objective;
    let mut best = initial.clamped(search);
    let mut best_loss = evaluate_calibration_samples(best, samples)?;
    let baseline_loss = best_loss.total_loss(objective);
    let mut best_total = baseline_loss;

    for step in search.step_scales {
        let mut improved = true;
        while improved {
            improved = false;
            for index in 0..WholeCellDerivationCalibration::PARAMETER_COUNT {
                let current = best.get(index);
                for direction in [-1.0_f32, 1.0_f32] {
                    let candidate = best.with(index, current + direction * step).clamped(search);
                    let candidate_loss = evaluate_calibration_samples(candidate, samples)?;
                    let candidate_total = candidate_loss.total_loss(objective);
                    if candidate_total + 1.0e-6 < best_total {
                        best = candidate;
                        best_loss = candidate_loss;
                        best_total = candidate_total;
                        improved = true;
                    }
                }
            }
        }
    }

    Ok(WholeCellDerivationCalibrationFit {
        sample_count: samples.len(),
        baseline_loss,
        fitted_loss: best_total,
        support_rmse: best_loss.support_rmse(),
        scale_rmse: best_loss.scale_rmse(),
        demand_rmse: best_loss.demand_rmse(),
        activity_rmse: best_loss.activity_rmse(),
        calibration: best,
    })
}

pub struct WholeCellChemistryBridge {
    substrate: BatchedAtomTerrarium,
    last_report: LocalChemistryReport,
    last_site_reports: Vec<LocalChemistrySiteReport>,
    last_md_report: Option<LocalMDProbeReport>,
    exchange_targets: SnapshotExchangeTargets,
    initialized_from_snapshot: bool,
}

impl WholeCellChemistryBridge {
    pub fn new(
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) -> Self {
        Self {
            substrate: BatchedAtomTerrarium::new(x_dim, y_dim, z_dim, voxel_size_au, use_gpu),
            last_report: LocalChemistryReport::default(),
            last_site_reports: Vec::new(),
            last_md_report: None,
            exchange_targets: SnapshotExchangeTargets {
                reactive_species: [
                    (TerrariumSpecies::Glucose, 0.0),
                    (TerrariumSpecies::OxygenGas, 0.0),
                    (TerrariumSpecies::AtpFlux, 0.0),
                    (TerrariumSpecies::Ammonium, 0.0),
                    (TerrariumSpecies::Nitrate, 0.0),
                    (TerrariumSpecies::Phosphorus, 0.0),
                    (TerrariumSpecies::CarbonDioxide, 0.0),
                    (TerrariumSpecies::Proton, 0.0),
                ],
            },
            initialized_from_snapshot: false,
        }
    }

    pub fn synchronize_from_snapshot(&mut self, snapshot: &WholeCellSnapshot) {
        self.exchange_targets = SnapshotExchangeTargets::from_snapshot(snapshot);
        if !self.initialized_from_snapshot {
            for (species, target) in self.exchange_targets.reactive_species {
                self.substrate.fill_species(species, target);
            }
            self.initialized_from_snapshot = true;
        }
    }

    fn apply_bulk_exchange(&mut self, dt_ms: f32) {
        if !self.initialized_from_snapshot {
            return;
        }

        let exchange_fraction = (1.0 - (-dt_ms.max(0.0) / 8.0).exp()).clamp(0.0, 0.28);
        for (species, target) in self.exchange_targets.reactive_species {
            self.substrate
                .relax_species_toward(species, target, exchange_fraction);
        }
    }

    fn subsystem_state_from_snapshot(
        snapshot: &WholeCellSnapshot,
        preset: Syn3ASubsystemPreset,
    ) -> WholeCellSubsystemState {
        snapshot
            .subsystem_states
            .iter()
            .copied()
            .find(|state| state.preset == preset)
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset))
    }

    fn previous_localized_patch(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
    ) -> Option<LocalizedPatch> {
        let radius = subsystem_coupling_profile(preset)
            .localization_rule
            .patch_radius;
        snapshot
            .and_then(|snap| {
                let state = Self::subsystem_state_from_snapshot(snap, preset);
                if state.localization_score.is_finite()
                    && (state.localization_score.abs() > 1.0e-6
                        || state.site_x != 0
                        || state.site_y != 0
                        || state.site_z != 0)
                {
                    Some(LocalizedPatch {
                        x: state.site_x,
                        y: state.site_y,
                        z: state.site_z,
                        radius,
                        score: state.localization_score,
                    })
                } else {
                    None
                }
            })
            .or_else(|| {
                self.last_site_reports
                    .iter()
                    .find(|report| report.preset == preset)
                    .map(|report| LocalizedPatch {
                        x: report.site_x,
                        y: report.site_y,
                        z: report.site_z,
                        radius: report.patch_radius,
                        score: report.localization_score,
                    })
            })
    }

    fn resolve_localized_patch(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
        occupied_sites: &[LocalizedPatch],
    ) -> LocalizedPatch {
        let profile = subsystem_coupling_profile(preset);
        localize_patch(
            &self.substrate,
            profile.localization_rule,
            self.previous_localized_patch(snapshot, preset),
            occupied_sites,
        )
    }

    fn patch_species_metrics(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> PatchSpeciesMetrics {
        PatchSpeciesMetrics {
            mean_glucose: self.substrate.patch_mean_species(
                TerrariumSpecies::Glucose,
                x,
                y,
                z,
                radius,
            ),
            mean_oxygen: self.substrate.patch_mean_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                z,
                radius,
            ),
            mean_atp_flux: self.substrate.patch_mean_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                z,
                radius,
            ),
            mean_carbon_dioxide: self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                z,
                radius,
            ),
            mean_nitrate: self.substrate.patch_mean_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
            ),
            mean_ammonium: self.substrate.patch_mean_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                z,
                radius,
            ),
            mean_proton: self.substrate.patch_mean_species(
                TerrariumSpecies::Proton,
                x,
                y,
                z,
                radius,
            ),
            mean_phosphorus: self.substrate.patch_mean_species(
                TerrariumSpecies::Phosphorus,
                x,
                y,
                z,
                radius,
            ),
        }
    }

    fn patch_crowding_penalty(metrics: PatchSpeciesMetrics) -> f32 {
        let mut ctx = CrowdingContext::default();
        ctx.set(
            CrowdingSignal::CarbonDioxide,
            saturating_signal(metrics.mean_carbon_dioxide, 0.04),
        );
        ctx.set(
            CrowdingSignal::Proton,
            saturating_signal(metrics.mean_proton, 0.01),
        );
        PATCH_CROWDING_RULE.evaluate(ctx.scalar())
    }

    fn base_assembly_context(
        metrics: PatchSpeciesMetrics,
        state: WholeCellSubsystemState,
        demand: LocalChemistryDemandSummary,
        crowding_penalty: f32,
    ) -> AssemblyContext {
        let oxygen_signal = saturating_signal(metrics.mean_oxygen, 0.05);
        let carbon_signal = saturating_signal(metrics.mean_glucose, 0.05);
        let energy_signal = (0.75 * saturating_signal(metrics.mean_atp_flux, 0.04)
            + 0.25 * oxygen_signal)
            .clamp(0.0, 1.0);
        let nitrogen_signal = (0.55 * saturating_signal(metrics.mean_ammonium, 0.04)
            + 0.45 * saturating_signal(metrics.mean_nitrate, 0.04))
        .clamp(0.0, 1.0);
        let phosphorus_signal = saturating_signal(metrics.mean_phosphorus, 0.03);
        let stress_signal = (0.45 * saturating_signal(metrics.mean_carbon_dioxide, 0.04)
            + 0.30 * saturating_signal(metrics.mean_proton, 0.01)
            + 0.25 * (1.0 - crowding_penalty))
            .clamp(0.0, 1.0);
        let biosynthesis_signal = (0.30 * carbon_signal
            + 0.32 * nitrogen_signal
            + 0.14 * phosphorus_signal
            + 0.24 * energy_signal)
            .clamp(0.0, 1.0);

        let mut ctx = AssemblyReducerContext::default();
        ctx.set(AssemblyReducerSignal::Oxygen, oxygen_signal);
        ctx.set(AssemblyReducerSignal::Carbon, carbon_signal);
        ctx.set(AssemblyReducerSignal::Energy, energy_signal);
        ctx.set(AssemblyReducerSignal::Nitrogen, nitrogen_signal);
        ctx.set(AssemblyReducerSignal::Phosphorus, phosphorus_signal);
        ctx.set(AssemblyReducerSignal::Biosynthesis, biosynthesis_signal);
        ctx.set(AssemblyReducerSignal::Stress, stress_signal);
        ctx.set(
            AssemblyReducerSignal::StructuralOrder,
            state.structural_order,
        );
        ctx.set(AssemblyReducerSignal::AtpScale, state.atp_scale);
        ctx.set(
            AssemblyReducerSignal::TranslationScale,
            state.translation_scale,
        );
        ctx.set(
            AssemblyReducerSignal::ReplicationScale,
            state.replication_scale,
        );
        ctx.set(AssemblyReducerSignal::MembraneScale, state.membrane_scale);
        ctx.set(
            AssemblyReducerSignal::ConstrictionScale,
            state.constriction_scale,
        );
        ctx.set(
            AssemblyReducerSignal::DemandSatisfaction,
            demand.demand_satisfaction,
        );
        ctx.set(AssemblyReducerSignal::ByproductLoad, demand.byproduct_load);

        AssemblyContext {
            catalyst_scale: ASSEMBLY_CATALYST_SCALE_RULE.evaluate(ctx.scalar()),
            support_scale: ASSEMBLY_SUPPORT_SCALE_RULE.evaluate(ctx.scalar()),
            demand_satisfaction: demand.demand_satisfaction,
            crowding_penalty,
            byproduct_load: demand.byproduct_load,
            substrate_pressure: demand.substrate_draw,
            energy_pressure: demand.energy_draw,
            biosynthetic_pressure: demand.biosynthetic_draw,
        }
    }

    fn local_patch_signals(
        metrics: PatchSpeciesMetrics,
        assembly: AssemblyState,
        state: WholeCellSubsystemState,
        crowding_penalty: f32,
    ) -> LocalPatchSignals {
        let oxygen_signal = saturating_signal(metrics.mean_oxygen, 0.05);
        let carbon_signal = saturating_signal(metrics.mean_glucose, 0.05);
        let energy_signal = (0.72 * saturating_signal(metrics.mean_atp_flux, 0.04)
            + 0.28 * oxygen_signal)
            .clamp(0.0, 1.0);
        let nitrogen_signal = (0.55 * saturating_signal(metrics.mean_ammonium, 0.04)
            + 0.45 * saturating_signal(metrics.mean_nitrate, 0.04))
        .clamp(0.0, 1.0);
        let phosphorus_signal = saturating_signal(metrics.mean_phosphorus, 0.03);
        let assembly_availability = saturating_signal(assembly.component_availability, 0.35);
        let assembly_occupancy = saturating_signal(assembly.occupancy, 0.55);
        let assembly_stability = saturating_signal(assembly.stability, 0.55);
        let assembly_turnover = saturating_signal(assembly.turnover, 0.30);
        let carbon_dioxide_signal = saturating_signal(metrics.mean_carbon_dioxide, 0.04);
        let proton_signal = saturating_signal(metrics.mean_proton, 0.01);
        let mut structural_ctx = StructuralReducerContext::default();
        structural_ctx.set(
            StructuralReducerSignal::AssemblyAvailability,
            assembly_availability,
        );
        structural_ctx.set(
            StructuralReducerSignal::AssemblyOccupancy,
            assembly_occupancy,
        );
        structural_ctx.set(
            StructuralReducerSignal::AssemblyStability,
            assembly_stability,
        );
        structural_ctx.set(
            StructuralReducerSignal::StructuralOrder,
            state.structural_order,
        );
        structural_ctx.set(StructuralReducerSignal::CrowdingPenalty, crowding_penalty);
        structural_ctx.set(StructuralReducerSignal::AssemblyTurnover, assembly_turnover);
        let structural_signal = PATCH_STRUCTURAL_SIGNAL_RULE.evaluate(structural_ctx.scalar());

        let mut patch_ctx = PatchReducerContext::default();
        patch_ctx.set(PatchReducerSignal::Carbon, carbon_signal);
        patch_ctx.set(PatchReducerSignal::Nitrogen, nitrogen_signal);
        patch_ctx.set(PatchReducerSignal::Phosphorus, phosphorus_signal);
        patch_ctx.set(PatchReducerSignal::Energy, energy_signal);
        patch_ctx.set(PatchReducerSignal::Structural, structural_signal);
        patch_ctx.set(PatchReducerSignal::CarbonDioxide, carbon_dioxide_signal);
        patch_ctx.set(PatchReducerSignal::Proton, proton_signal);
        patch_ctx.set(PatchReducerSignal::AssemblyTurnover, assembly_turnover);
        patch_ctx.set(PatchReducerSignal::CrowdingPenalty, crowding_penalty);
        patch_ctx.set(
            PatchReducerSignal::DemandSatisfaction,
            state.demand_satisfaction.clamp(0.0, 1.0),
        );
        let biosynthesis_signal = PATCH_BIOSYNTHESIS_SIGNAL_RULE.evaluate(patch_ctx.scalar());
        let stress_signal = PATCH_STRESS_SIGNAL_RULE.evaluate(patch_ctx.scalar());

        LocalPatchSignals {
            oxygen_signal,
            carbon_signal,
            energy_signal,
            nitrogen_signal,
            phosphorus_signal,
            biosynthesis_signal,
            structural_signal,
            stress_signal,
        }
    }

    fn reaction_rules_for_preset(preset: Syn3ASubsystemPreset) -> &'static [ReactionRule] {
        subsystem_rule_bundle(preset).reaction_program.as_slice()
    }

    fn reaction_context_from_local_patch(
        preset: Syn3ASubsystemPreset,
        signals: LocalPatchSignals,
        state: WholeCellSubsystemState,
    ) -> ReactionContext {
        let profile = subsystem_coupling_profile(preset);
        let activity = profile.activity_profile.activity(signals, state);
        let catalyst_scale = profile.activity_profile.catalyst(signals, state);
        let energy_driver =
            (signals.energy_signal * (0.82 + 0.18 * state.atp_scale)).clamp(0.0, 3.0);
        let biosynthesis_driver = (signals.biosynthesis_signal
            * (0.72
                + 0.08 * state.translation_scale
                + 0.10 * state.replication_scale
                + 0.10 * state.membrane_scale))
            .clamp(0.0, 4.0);
        let replication_driver = (0.42 * signals.biosynthesis_signal
            + 0.18 * signals.energy_signal
            + 0.26 * signals.structural_signal
            + 0.14 * state.replication_scale
            - 0.14 * signals.stress_signal)
            .clamp(0.0, 3.0);
        let division_driver = (0.18 * signals.energy_signal
            + 0.24 * signals.phosphorus_signal
            + 0.24 * signals.structural_signal
            + 0.14 * state.membrane_scale
            + 0.16 * state.constriction_scale
            - 0.14 * signals.stress_signal)
            .clamp(0.0, 3.0);
        let translation_driver = (0.30 * signals.carbon_signal
            + 0.28 * signals.nitrogen_signal
            + 0.18 * signals.energy_signal
            + 0.14 * signals.structural_signal
            + 0.10 * state.translation_scale
            - 0.12 * signals.stress_signal)
            .clamp(0.0, 3.0);

        ReactionContext {
            catalyst_scale,
            drivers: [
                activity,
                energy_driver,
                biosynthesis_driver,
                replication_driver,
                division_driver,
                signals.oxygen_signal.clamp(0.0, 3.0),
                signals.carbon_signal.clamp(0.0, 3.0),
                translation_driver,
            ],
        }
    }

    fn apply_subsystem_demand(
        &mut self,
        snapshot: &WholeCellSnapshot,
        preset: Syn3ASubsystemPreset,
        patch: LocalizedPatch,
        dt_ms: f32,
    ) -> LocalChemistryDemandSummary {
        let state = Self::subsystem_state_from_snapshot(snapshot, preset);
        let demand_window = dt_ms.max(0.05);
        let metrics = self.patch_species_metrics(patch.x, patch.y, patch.z, patch.radius);
        let crowding_penalty = Self::patch_crowding_penalty(metrics);
        let assembly = evaluate_patch_assembly(
            &self.substrate,
            patch.x,
            patch.y,
            patch.z,
            patch.radius,
            subsystem_coupling_profile(preset).assembly_rule,
            Self::base_assembly_context(
                metrics,
                state,
                LocalChemistryDemandSummary::default(),
                crowding_penalty,
            ),
        );
        let context = Self::reaction_context_from_local_patch(
            preset,
            Self::local_patch_signals(metrics, assembly, state, crowding_penalty),
            state,
        );

        let mut substrate_draw = 0.0;
        let mut energy_draw = 0.0;
        let mut biosynthetic_draw = 0.0;
        let mut byproduct_load = 0.0;
        let mut removed_total = 0.0;
        let mut target_total = 0.0;

        for rule in Self::reaction_rules_for_preset(preset) {
            let flux = execute_patch_reaction(
                &mut self.substrate,
                patch.x,
                patch.y,
                patch.z,
                patch.radius,
                *rule,
                context,
                demand_window,
            );
            substrate_draw += flux.substrate_draw;
            energy_draw += flux.energy_draw;
            biosynthetic_draw += flux.biosynthetic_draw;
            byproduct_load += flux.byproduct_load;
            removed_total += flux.removed_total;
            target_total += flux.target_total;
        }

        LocalChemistryDemandSummary {
            substrate_draw,
            energy_draw,
            biosynthetic_draw,
            byproduct_load,
            demand_satisfaction: if target_total <= 1.0e-6 {
                1.0
            } else {
                (removed_total / target_total).clamp(0.0, 1.0)
            },
        }
    }

    fn site_support_report(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
        patch: LocalizedPatch,
        demand: LocalChemistryDemandSummary,
    ) -> LocalChemistrySiteReport {
        let profile = subsystem_coupling_profile(preset);
        let metrics = self.patch_species_metrics(patch.x, patch.y, patch.z, patch.radius);
        let crowding_penalty = Self::patch_crowding_penalty(metrics);
        let state = snapshot
            .map(|snap| Self::subsystem_state_from_snapshot(snap, preset))
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset));
        let assembly_context =
            Self::base_assembly_context(metrics, state, demand, crowding_penalty);
        let assembly = evaluate_patch_assembly(
            &self.substrate,
            patch.x,
            patch.y,
            patch.z,
            patch.radius,
            profile.assembly_rule,
            assembly_context,
        );
        let atp_support = profile.atp_support.evaluate(metrics, assembly);
        let translation_support = profile.translation_support.evaluate(metrics, assembly);
        let nucleotide_support = profile.nucleotide_support.evaluate(metrics, assembly);
        let membrane_support = profile.membrane_support.evaluate(metrics, assembly);

        LocalChemistrySiteReport {
            preset,
            site: preset.chemistry_site(),
            patch_radius: patch.radius,
            site_x: patch.x,
            site_y: patch.y,
            site_z: patch.z,
            localization_score: patch.score,
            atp_support,
            translation_support,
            nucleotide_support,
            membrane_support,
            crowding_penalty,
            mean_glucose: metrics.mean_glucose,
            mean_oxygen: metrics.mean_oxygen,
            mean_atp_flux: metrics.mean_atp_flux,
            mean_carbon_dioxide: metrics.mean_carbon_dioxide,
            mean_nitrate: metrics.mean_nitrate,
            mean_ammonium: metrics.mean_ammonium,
            mean_proton: metrics.mean_proton,
            mean_phosphorus: metrics.mean_phosphorus,
            assembly_component_availability: assembly.component_availability,
            assembly_occupancy: assembly.occupancy,
            assembly_stability: assembly.stability,
            assembly_turnover: assembly.turnover,
            substrate_draw: demand.substrate_draw,
            energy_draw: demand.energy_draw,
            biosynthetic_draw: demand.biosynthetic_draw,
            byproduct_load: demand.byproduct_load,
            demand_satisfaction: demand.demand_satisfaction,
        }
    }

    pub fn step_with_snapshot(
        &mut self,
        dt_ms: f32,
        snapshot: Option<&WholeCellSnapshot>,
    ) -> LocalChemistryReport {
        if let Some(snapshot) = snapshot {
            self.synchronize_from_snapshot(snapshot);
        }

        let mut localized_sites = Vec::with_capacity(Syn3ASubsystemPreset::all().len());
        let mut occupied_sites = Vec::with_capacity(Syn3ASubsystemPreset::all().len());
        for preset in Syn3ASubsystemPreset::all().iter().copied() {
            let patch = self.resolve_localized_patch(snapshot, preset, &occupied_sites);
            occupied_sites.push(patch);
            localized_sites.push((preset, patch));
        }

        let mut site_demands = Vec::with_capacity(localized_sites.len());
        for (preset, patch) in &localized_sites {
            let demand = if let Some(snapshot) = snapshot {
                self.apply_subsystem_demand(snapshot, *preset, *patch, dt_ms)
            } else {
                LocalChemistryDemandSummary::default()
            };
            site_demands.push((*preset, *patch, demand));
        }

        self.substrate.step(dt_ms);
        self.apply_bulk_exchange(dt_ms);

        let mean_glucose = self.substrate.mean_species(TerrariumSpecies::Glucose);
        let mean_oxygen = self.substrate.mean_species(TerrariumSpecies::OxygenGas);
        let mean_atp_flux = self.substrate.mean_species(TerrariumSpecies::AtpFlux);
        let mean_co2 = self.substrate.mean_species(TerrariumSpecies::CarbonDioxide);
        let mean_nitrate = self.substrate.mean_species(TerrariumSpecies::Nitrate);
        let mean_ammonium = self.substrate.mean_species(TerrariumSpecies::Ammonium);
        let mean_proton = self.substrate.mean_species(TerrariumSpecies::Proton);

        let mut crowding_ctx = CrowdingContext::default();
        crowding_ctx.set(
            CrowdingSignal::CarbonDioxide,
            saturating_signal(mean_co2, 0.04),
        );
        crowding_ctx.set(CrowdingSignal::Proton, saturating_signal(mean_proton, 0.01));
        let crowding_penalty = PATCH_CROWDING_RULE.evaluate(crowding_ctx.scalar());

        let mut report_ctx = GlobalReportContext::default();
        report_ctx.set(GlobalReportSignal::Glucose, mean_glucose);
        report_ctx.set(GlobalReportSignal::Oxygen, mean_oxygen);
        report_ctx.set(GlobalReportSignal::AtpFlux, mean_atp_flux);
        report_ctx.set(GlobalReportSignal::Nitrate, mean_nitrate);
        report_ctx.set(GlobalReportSignal::Ammonium, mean_ammonium);
        report_ctx.set(GlobalReportSignal::CrowdingPenalty, crowding_penalty);

        let report = LocalChemistryReport {
            atp_support: GLOBAL_ATP_SUPPORT_RULE.evaluate(report_ctx.scalar()),
            translation_support: GLOBAL_TRANSLATION_SUPPORT_RULE.evaluate(report_ctx.scalar()),
            nucleotide_support: GLOBAL_NUCLEOTIDE_SUPPORT_RULE.evaluate(report_ctx.scalar()),
            membrane_support: GLOBAL_MEMBRANE_SUPPORT_RULE.evaluate(report_ctx.scalar()),
            crowding_penalty,
            mean_glucose,
            mean_oxygen,
            mean_atp_flux,
            mean_carbon_dioxide: mean_co2,
        };

        self.last_report = report;
        self.last_site_reports = site_demands
            .into_iter()
            .map(|(preset, patch, demand)| {
                self.site_support_report(snapshot, preset, patch, demand)
            })
            .collect();
        report
    }

    pub fn step(&mut self, dt_ms: f32) -> LocalChemistryReport {
        self.step_with_snapshot(dt_ms, None)
    }

    pub fn last_report(&self) -> LocalChemistryReport {
        self.last_report
    }

    pub fn last_md_report(&self) -> Option<LocalMDProbeReport> {
        self.last_md_report
    }

    pub fn lattice_shape(&self) -> (usize, usize, usize) {
        (
            self.substrate.x_dim,
            self.substrate.y_dim,
            self.substrate.z_dim,
        )
    }

    pub fn voxel_size_au(&self) -> f32 {
        self.substrate.voxel_size_mm
    }

    pub fn use_gpu_backend(&self) -> bool {
        matches!(
            self.substrate.backend(),
            crate::terrarium::TerrariumBackend::Metal
        )
    }

    pub fn site_reports(&self) -> Vec<LocalChemistrySiteReport> {
        self.last_site_reports.clone()
    }

    pub fn derivation_calibration_samples(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Vec<WholeCellDerivationCalibrationSample> {
        let snapshot = representative_calibration_snapshot();
        self.synchronize_from_snapshot(&snapshot);
        let steps = equilibration_steps.max(1);
        for _ in 0..steps {
            self.step_with_snapshot(dt_ms, Some(&snapshot));
        }

        let site_reports = self.site_reports();
        Syn3ASubsystemPreset::all()
            .iter()
            .copied()
            .filter_map(|preset| {
                let site_report = site_reports
                    .iter()
                    .copied()
                    .find(|report| report.preset == preset)?;
                let md_report = self.run_md_probe(preset.default_probe_request());
                Some(WholeCellDerivationCalibrationSample {
                    preset,
                    dt_ms,
                    site_report,
                    md_report,
                })
            })
            .collect()
    }

    pub fn fit_derivation_calibration(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Result<WholeCellDerivationCalibrationFit, String> {
        let samples = self.derivation_calibration_samples(dt_ms, equilibration_steps);
        fit_derivation_calibration_from_samples(&samples, derivation_calibration())
    }

    pub fn run_md_probe(&mut self, request: LocalMDProbeRequest) -> LocalMDProbeReport {
        let template = atomistic_template_for_site_name(request.site.as_str());
        let n_atoms = template
            .map(|template| template.atom_count())
            .unwrap_or_else(|| request.n_atoms.max(8));
        let box_size = template
            .map(|template| {
                request
                    .box_size_angstrom
                    .max(template.recommended_box_angstrom)
            })
            .unwrap_or_else(|| request.box_size_angstrom.max(8.0));
        let mut md = GPUMolecularDynamics::new(n_atoms, "auto");
        let center = box_size * 0.5;
        let template_descriptor = if let Some(template) = template {
            template.configure_md(&mut md, box_size, request.temperature_k);
            template.descriptor()
        } else {
            let mut positions = vec![0.0f32; n_atoms * 3];
            let mut masses = vec![12.0f32; n_atoms];
            let mut charges = vec![0.0f32; n_atoms];
            let mut sigma = vec![3.3f32; n_atoms];
            let mut epsilon = vec![0.10f32; n_atoms];

            for i in 0..n_atoms {
                let i3 = i * 3;
                let frac = i as f32 / n_atoms as f32;
                let angle = frac * std::f32::consts::TAU * 1.618;
                let radial = (0.20 + 0.45 * frac) * box_size * 0.5;
                positions[i3] = center + radial * angle.cos();
                positions[i3 + 1] = center + radial * angle.sin();
                positions[i3 + 2] = center + (((i % 5) as f32) - 2.0) * 0.8;

                match i % 6 {
                    0 => {
                        masses[i] = 12.0;
                        charges[i] = 0.00;
                        sigma[i] = 3.4;
                        epsilon[i] = 0.10;
                    }
                    1 => {
                        masses[i] = 16.0;
                        charges[i] = -0.20;
                        sigma[i] = 3.0;
                        epsilon[i] = 0.12;
                    }
                    2 => {
                        masses[i] = 14.0;
                        charges[i] = -0.10;
                        sigma[i] = 3.2;
                        epsilon[i] = 0.11;
                    }
                    3 => {
                        masses[i] = 1.0;
                        charges[i] = 0.08;
                        sigma[i] = 2.4;
                        epsilon[i] = 0.02;
                    }
                    4 => {
                        masses[i] = 31.0;
                        charges[i] = 0.24;
                        sigma[i] = 3.6;
                        epsilon[i] = 0.16;
                    }
                    _ => {
                        masses[i] = 32.0;
                        charges[i] = -0.04;
                        sigma[i] = 3.7;
                        epsilon[i] = 0.20;
                    }
                }
            }

            md.set_positions(&positions);
            md.set_masses(&masses);
            md.set_charges(&charges);
            md.set_lj_params(&sigma, &epsilon);
            md.set_box([box_size, box_size, box_size]);
            md.set_temperature(request.temperature_k);
            let mut bond_count = 0usize;
            let mut angle_count = 0usize;
            let mut dihedral_count = 0usize;
            for i in 0..n_atoms.saturating_sub(1) {
                md.add_bond(i, i + 1, 2.1, 20.0);
                bond_count += 1;
            }
            for i in 0..n_atoms.saturating_sub(2) {
                md.add_angle(i, i + 1, i + 2, 110.0f32.to_radians(), 5.5);
                angle_count += 1;
            }
            for i in 0..n_atoms.saturating_sub(3) {
                md.add_dihedral(i, i + 1, i + 2, i + 3, 3, 180.0f32.to_radians(), 0.8);
                dihedral_count += 1;
            }

            generic_atomistic_probe_descriptor(
                n_atoms,
                bond_count,
                angle_count,
                dihedral_count,
                &charges,
            )
        };

        md.initialize_velocities();

        let mut temp_acc = 0.0;
        let mut total_acc = 0.0;
        let mut vdw_acc = 0.0;
        let mut elec_acc = 0.0;
        for _ in 0..request.steps.max(1) {
            let stats = md.step(request.dt_ps.max(0.0001));
            temp_acc += stats.temperature;
            total_acc += stats.total_energy;
            vdw_acc += stats.vdw_energy;
            elec_acc += stats.electrostatic_energy;
        }

        let steps_f = request.steps.max(1) as f32;
        let positions = md.positions();
        let mut mean_radius = 0.0;
        let mut mean_radius_sq = 0.0;
        for i in 0..n_atoms {
            let i3 = i * 3;
            let dx = positions[i3] - center;
            let dy = positions[i3 + 1] - center;
            let dz = positions[i3 + 2] - center;
            let radius = (dx * dx + dy * dy + dz * dz).sqrt();
            mean_radius += radius;
            mean_radius_sq += radius * radius;
        }
        mean_radius /= n_atoms as f32;
        mean_radius_sq /= n_atoms as f32;
        let variance = (mean_radius_sq - mean_radius * mean_radius).max(0.0);
        let structural_order = (1.0 / (1.0 + variance / (box_size * 0.5).max(1.0))).clamp(0.2, 1.0);

        let mean_temperature = temp_acc / steps_f;
        let mean_total_energy = total_acc / steps_f;
        let mean_vdw_energy = vdw_acc / steps_f;
        let mean_electrostatic_energy = elec_acc / steps_f;
        let crowding_penalty = (1.0
            / (1.0 + mean_vdw_energy.abs() / 150.0 + mean_total_energy.abs() / 600.0))
            .clamp(0.65, 1.0);
        let dynamic_descriptor = atomistic_dynamic_descriptor(
            &positions,
            center,
            box_size,
            request.temperature_k,
            mean_temperature,
            mean_vdw_energy,
            mean_electrostatic_energy,
            structural_order,
            crowding_penalty,
        );
        let scale_ctx = atomistic_scale_context(template_descriptor, dynamic_descriptor);
        let recommended_atp_scale = ATOMISTIC_ATP_SCALE_RULE.evaluate(scale_ctx);
        let recommended_translation_scale = ATOMISTIC_TRANSLATION_SCALE_RULE.evaluate(scale_ctx);
        let recommended_replication_scale = ATOMISTIC_REPLICATION_SCALE_RULE.evaluate(scale_ctx);
        let recommended_segregation_scale = ATOMISTIC_SEGREGATION_SCALE_RULE.evaluate(scale_ctx);
        let recommended_membrane_scale = ATOMISTIC_MEMBRANE_SCALE_RULE.evaluate(scale_ctx);
        let recommended_constriction_scale = ATOMISTIC_CONSTRICTION_SCALE_RULE.evaluate(scale_ctx);

        let report = LocalMDProbeReport {
            site: request.site,
            mean_temperature,
            mean_total_energy,
            mean_vdw_energy,
            mean_electrostatic_energy,
            structural_order,
            crowding_penalty,
            compactness: dynamic_descriptor.compactness,
            shell_order: dynamic_descriptor.shell_order,
            axis_anisotropy: dynamic_descriptor.axis_anisotropy,
            thermal_stability: dynamic_descriptor.thermal_stability,
            electrostatic_order: dynamic_descriptor.electrostatic_order,
            vdw_cohesion: dynamic_descriptor.vdw_cohesion,
            polar_fraction: template_descriptor.polar_fraction,
            phosphate_fraction: template_descriptor.phosphate_fraction,
            hydrogen_fraction: template_descriptor.hydrogen_fraction,
            bond_density: template_descriptor.bond_density,
            angle_density: template_descriptor.angle_density,
            dihedral_density: template_descriptor.dihedral_density,
            charge_density: template_descriptor.charge_density,
            recommended_atp_scale: recommended_atp_scale.clamp(0.75, 1.35),
            recommended_translation_scale: recommended_translation_scale.clamp(0.75, 1.35),
            recommended_replication_scale: recommended_replication_scale.clamp(0.75, 1.35),
            recommended_segregation_scale: recommended_segregation_scale.clamp(0.75, 1.35),
            recommended_membrane_scale: recommended_membrane_scale.clamp(0.75, 1.35),
            recommended_constriction_scale: recommended_constriction_scale.clamp(0.75, 1.35),
        };
        self.last_md_report = Some(report);
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell::{WholeCellConfig, WholeCellSimulator};

    #[test]
    fn chemistry_bridge_generates_support_report() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        let report = bridge.step(0.25);

        assert!(report.atp_support > 0.0);
        assert!(report.translation_support > 0.0);
        assert!(report.nucleotide_support > 0.0);
        assert!(report.crowding_penalty > 0.0);
        let site_reports = bridge.site_reports();
        assert_eq!(site_reports.len(), Syn3ASubsystemPreset::all().len());
        assert!(site_reports.iter().all(|report| report.patch_radius > 0));
    }

    #[test]
    fn subsystem_rule_bundle_registry_is_complete() {
        for preset in Syn3ASubsystemPreset::all().iter().copied() {
            let signature = subsystem_signature(preset);
            let bundle = subsystem_rule_bundle(preset);
            assert_eq!(bundle.chemistry_site, signature.chemistry_site);
            assert_eq!(
                bundle.default_interval_steps,
                signature.default_interval_steps
            );
            assert_eq!(
                bundle.default_probe_request,
                signature.default_probe_request
            );
            assert_eq!(
                bundle.coupling_profile.localization_rule.patch_radius,
                signature.localization.patch_radius
            );
            assert_eq!(
                bundle.coupling_profile.assembly_rule.name,
                signature.assembly.name
            );
            assert_eq!(
                bundle.reaction_program.rule_count,
                signature.reaction_program.rule_count
            );
            assert!(!bundle.reaction_program.as_slice().is_empty());
        }
    }

    #[test]
    fn subsystem_spec_is_descriptor_driven() {
        assert!(SUBSYSTEM_SIGNATURE_SPEC_JSON.contains("\"process_focus\""));
        assert!(!SUBSYSTEM_SIGNATURE_SPEC_JSON.contains("\"activity_weights\""));
        assert!(!SUBSYSTEM_SIGNATURE_SPEC_JSON.contains("\"driver_weights\""));
    }

    #[test]
    fn derived_profiles_follow_process_focus() {
        let atp = subsystem_signature(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let ribosome = subsystem_signature(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        let replisome = subsystem_signature(Syn3ASubsystemPreset::ReplisomeTrack);
        let septum = subsystem_signature(Syn3ASubsystemPreset::FtsZSeptumRing);

        assert!(atp.atp_scale.weights[0] > ribosome.atp_scale.weights[0]);
        assert!(ribosome.translation_scale.weights[1] > atp.translation_scale.weights[1]);
        assert!(replisome.replication_scale.weights[2] > ribosome.replication_scale.weights[2]);
        assert!(septum.constriction_scale.weights[3] > replisome.constriction_scale.weights[3]);
        assert!(
            subsystem_rule_bundle(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
                .reaction_program
                .as_slice()[0]
                .law
                .driver_weights[1]
                > subsystem_rule_bundle(Syn3ASubsystemPreset::RibosomePolysomeCluster)
                    .reaction_program
                    .as_slice()[0]
                    .law
                    .driver_weights[1]
        );
    }

    #[test]
    fn chemistry_bridge_applies_localized_substrate_demand() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        bridge.step_with_snapshot(0.25, Some(&snap));

        let site_reports = bridge.site_reports();
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");
        let replisome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome site report");

        assert!(ribosome.substrate_draw > 0.0);
        assert!(ribosome.energy_draw > 0.0);
        assert!(ribosome.demand_satisfaction > 0.0);
        assert!(replisome.biosynthetic_draw > 0.0);
    }

    #[test]
    fn snapshot_exchange_targets_use_local_chemistry_context() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let mut snapshot = sim.snapshot();
        let baseline = SnapshotExchangeTargets::from_snapshot(&snapshot);

        snapshot.local_chemistry = Some(LocalChemistryReport {
            atp_support: 1.15,
            translation_support: 1.10,
            nucleotide_support: 1.08,
            membrane_support: 1.05,
            crowding_penalty: 0.94,
            mean_glucose: 0.90,
            mean_oxygen: 0.82,
            mean_atp_flux: 0.88,
            mean_carbon_dioxide: 0.26,
        });
        let enriched = SnapshotExchangeTargets::from_snapshot(&snapshot);

        let target_for = |targets: SnapshotExchangeTargets, species: TerrariumSpecies| {
            targets
                .reactive_species
                .iter()
                .find(|(candidate, _)| *candidate == species)
                .map(|(_, value)| *value)
                .expect("exchange target")
        };

        assert!(
            target_for(enriched, TerrariumSpecies::Glucose)
                > target_for(baseline, TerrariumSpecies::Glucose)
        );
        assert!(
            target_for(enriched, TerrariumSpecies::OxygenGas)
                > target_for(baseline, TerrariumSpecies::OxygenGas)
        );
        assert!(
            target_for(enriched, TerrariumSpecies::AtpFlux)
                > target_for(baseline, TerrariumSpecies::AtpFlux)
        );
        assert!(
            target_for(enriched, TerrariumSpecies::CarbonDioxide)
                > target_for(baseline, TerrariumSpecies::CarbonDioxide)
        );
    }

    #[test]
    fn snapshot_resync_preserves_local_depletion_memory() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();

        let mut persisted = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        persisted.synchronize_from_snapshot(&snap);
        persisted.step_with_snapshot(0.25, Some(&snap));
        persisted.synchronize_from_snapshot(&snap);
        persisted.step(0.05);

        let persisted_ribosome = persisted
            .site_reports()
            .into_iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("persisted ribosome report");

        let mut fresh = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        fresh.synchronize_from_snapshot(&snap);
        fresh.step(0.05);
        let fresh_ribosome = fresh
            .site_reports()
            .into_iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("fresh ribosome report");

        assert!(
            persisted_ribosome.mean_carbon_dioxide > fresh_ribosome.mean_carbon_dioxide,
            "persisted co2={} fresh co2={}",
            persisted_ribosome.mean_carbon_dioxide,
            fresh_ribosome.mean_carbon_dioxide
        );
        assert!(
            persisted_ribosome.crowding_penalty <= fresh_ribosome.crowding_penalty,
            "persisted crowding={} fresh crowding={}",
            persisted_ribosome.crowding_penalty,
            fresh_ribosome.crowding_penalty
        );
    }

    #[test]
    fn subsystem_state_responds_to_local_depletion_pressure() {
        let rich = LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
            site: WholeCellChemistrySite::RibosomeCluster,
            patch_radius: 2,
            site_x: 4,
            site_y: 4,
            site_z: 2,
            localization_score: 1.10,
            atp_support: 1.05,
            translation_support: 1.28,
            nucleotide_support: 1.02,
            membrane_support: 1.00,
            crowding_penalty: 0.98,
            mean_glucose: 0.35,
            mean_oxygen: 0.22,
            mean_atp_flux: 0.30,
            mean_carbon_dioxide: 0.02,
            mean_nitrate: 0.18,
            mean_ammonium: 0.28,
            mean_proton: 0.01,
            mean_phosphorus: 0.14,
            assembly_component_availability: 1.08,
            assembly_occupancy: 1.04,
            assembly_stability: 1.02,
            assembly_turnover: 0.03,
            substrate_draw: 0.04,
            energy_draw: 0.03,
            biosynthetic_draw: 0.02,
            byproduct_load: 0.02,
            demand_satisfaction: 1.0,
        };
        let depleted = LocalChemistrySiteReport {
            crowding_penalty: 0.82,
            assembly_component_availability: 0.78,
            assembly_occupancy: 0.74,
            assembly_stability: 0.70,
            assembly_turnover: 0.30,
            substrate_draw: 0.55,
            energy_draw: 0.48,
            biosynthetic_draw: 0.20,
            byproduct_load: 0.42,
            demand_satisfaction: 0.42,
            ..rich
        };

        let mut rich_state =
            WholeCellSubsystemState::new(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        rich_state.apply_site_report(rich);

        let mut depleted_state =
            WholeCellSubsystemState::new(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        depleted_state.apply_site_report(depleted);

        assert!(depleted_state.translation_scale < rich_state.translation_scale);
        assert!(depleted_state.structural_order < rich_state.structural_order);
        assert!(depleted_state.demand_satisfaction < rich_state.demand_satisfaction);
        assert!(depleted_state.byproduct_load > rich_state.byproduct_load);
    }

    #[test]
    fn acidic_byproduct_load_reduces_global_crowding_penalty() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();

        let mut baseline = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        baseline.synchronize_from_snapshot(&snap);
        let baseline_report = baseline.step(0.10);

        let mut stressed = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        stressed.synchronize_from_snapshot(&snap);
        stressed
            .substrate
            .add_hotspot(TerrariumSpecies::CarbonDioxide, 6, 6, 3, 3.5);
        stressed
            .substrate
            .add_hotspot(TerrariumSpecies::Proton, 6, 6, 3, 2.5);
        let stressed_report = stressed.step(0.10);

        assert!(stressed_report.crowding_penalty < baseline_report.crowding_penalty);
        assert!(stressed_report.atp_support <= baseline_report.atp_support);
    }

    #[test]
    fn local_md_probe_returns_finite_metrics() {
        let mut bridge = WholeCellChemistryBridge::new(8, 8, 4, 0.5, false);
        let report = bridge.run_md_probe(LocalMDProbeRequest {
            site: WholeCellChemistrySite::RibosomeCluster,
            n_atoms: 16,
            steps: 8,
            dt_ps: 0.001,
            box_size_angstrom: 14.0,
            temperature_k: 310.0,
        });

        assert!(report.mean_temperature.is_finite());
        assert!(report.mean_total_energy.is_finite());
        assert!(report.structural_order > 0.0);
        assert!(report.crowding_penalty > 0.0);
        assert!(report.compactness >= 0.0);
        assert!(report.axis_anisotropy >= 0.0);
        assert!(report.polar_fraction > 0.0);
        assert!(report.bond_density > 0.0);
    }

    #[test]
    fn demand_driven_reactions_shift_local_pools() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        let before = bridge.step(0.10);

        let after = bridge.step_with_snapshot(0.25, Some(&snap));

        assert!(after.mean_carbon_dioxide >= before.mean_carbon_dioxide);
        assert!(
            after.mean_glucose != before.mean_glucose
                || after.mean_atp_flux != before.mean_atp_flux
        );
    }

    #[test]
    fn localization_tracks_hotspots_instead_of_fixed_anchors() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);

        let bottom_z = bridge.substrate.z_dim.saturating_sub(1);
        bridge
            .substrate
            .add_hotspot(TerrariumSpecies::OxygenGas, 6, 6, bottom_z, 3.0);

        let patch = bridge.resolve_localized_patch(
            Some(&snap),
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            &[],
        );
        assert!(patch.z >= bridge.substrate.z_dim.saturating_sub(2));
        assert!(patch.score.is_finite());
    }

    #[test]
    fn demand_is_driven_by_local_patch_state_not_snapshot_counters() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snapshot = sim.snapshot();
        let mut altered_snapshot = snapshot.clone();
        altered_snapshot.active_ribosomes = 2.0;
        altered_snapshot.dnaa = 1.0;
        altered_snapshot.ftsz = 1.0;

        let mut baseline_bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        baseline_bridge.synchronize_from_snapshot(&snapshot);
        let baseline_patch = baseline_bridge.resolve_localized_patch(
            Some(&snapshot),
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            &[],
        );
        let baseline_demand = baseline_bridge.apply_subsystem_demand(
            &snapshot,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            baseline_patch,
            0.25,
        );

        let mut altered_bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        altered_bridge.synchronize_from_snapshot(&altered_snapshot);
        let altered_patch = altered_bridge.resolve_localized_patch(
            Some(&altered_snapshot),
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            &[],
        );
        let altered_demand = altered_bridge.apply_subsystem_demand(
            &altered_snapshot,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            altered_patch,
            0.25,
        );

        assert_eq!(
            (baseline_patch.x, baseline_patch.y, baseline_patch.z),
            (altered_patch.x, altered_patch.y, altered_patch.z)
        );
        assert!((baseline_demand.substrate_draw - altered_demand.substrate_draw).abs() < 1.0e-6);
        assert!((baseline_demand.energy_draw - altered_demand.energy_draw).abs() < 1.0e-6);
        assert!(
            (baseline_demand.biosynthetic_draw - altered_demand.biosynthetic_draw).abs() < 1.0e-6
        );
    }

    #[test]
    fn site_reports_capture_distinct_microdomains() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        bridge.step_with_snapshot(0.20, Some(&snap));

        let site_reports = bridge.site_reports();
        let atp_band = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("atp-band site report");
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");
        let unique_sites = site_reports
            .iter()
            .map(|report| (report.site_x, report.site_y, report.site_z))
            .collect::<std::collections::HashSet<_>>();

        assert!(
            atp_band.atp_support > ribosome.atp_support
                || atp_band.mean_oxygen > ribosome.mean_oxygen
        );
        assert!(ribosome.translation_support >= atp_band.translation_support);
        assert!(atp_band.demand_satisfaction > 0.0);
        assert!(ribosome.substrate_draw > 0.0);
        assert!(atp_band.localization_score.is_finite());
        assert!(unique_sites.len() > 1);
    }

    #[test]
    fn calibration_sweep_samples_cover_all_subsystems() {
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        let samples = bridge.derivation_calibration_samples(0.25, 2);

        assert_eq!(samples.len(), Syn3ASubsystemPreset::all().len());
        assert!(samples
            .iter()
            .all(|sample| sample.md_report.structural_order > 0.0));
        assert!(samples
            .iter()
            .all(|sample| sample.site_report.mean_nitrate >= 0.0));
        assert!(samples
            .iter()
            .all(|sample| sample.site_report.mean_ammonium >= 0.0));
        assert!(samples
            .iter()
            .all(|sample| sample.site_report.mean_phosphorus >= 0.0));
    }

    #[test]
    fn fitter_reduces_loss_from_perturbed_calibration() {
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        let samples = bridge.derivation_calibration_samples(0.25, 2);
        let base = default_whole_cell_derivation_calibration();
        let initial = WholeCellDerivationCalibration {
            context_occupancy_gain: 0.55,
            context_stability_gain: 1.35,
            context_persistence_gain: 0.70,
            context_focus_gain: 1.30,
            context_turnover_gain: 1.40,
            context_probe_complexity_gain: 0.80,
            activity_resource_gain: 1.35,
            activity_focus_gain: 0.70,
            activity_structural_gain: 1.25,
            activity_state_gain: 0.72,
            activity_stress_gain: 1.35,
            support_bias_gain: 0.80,
            support_weight_gain: 1.30,
            support_turnover_gain: 1.28,
            scale_weight_gain: 0.75,
            scale_penalty_gain: 1.30,
            scale_structural_gain: 1.22,
            reaction_focus_gain: 0.72,
            reaction_channel_gain: 1.32,
            reaction_resource_gain: 0.78,
            reaction_spatial_gain: 1.25,
            reaction_structural_gain: 1.18,
        };

        let fit = fit_derivation_calibration_from_samples(&samples, initial).expect("fit");

        assert_eq!(fit.sample_count, samples.len());
        assert!(fit.fitted_loss < fit.baseline_loss);
        assert!(fit.fitted_loss.is_finite());
        assert!(fit.support_rmse.is_finite());
        assert!(fit.scale_rmse.is_finite());
        assert!(fit.demand_rmse.is_finite());
        assert!(fit.activity_rmse.is_finite());
        assert_ne!(fit.calibration, base);
    }
}
