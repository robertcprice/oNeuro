//! Native whole-cell runtime focused on minimal bacterial cells.
//!
//! This is a coarse but performance-oriented native core: voxelized
//! intracellular RDME on a GPU-ready lattice, plus staged CME/ODE/BD/geometry
//! updates in Rust. It is meant to replace the Python skeleton as the
//! performance-critical simulation path.

use std::collections::HashMap;
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::gpu;
use crate::gpu::whole_cell_rdme::{
    cpu_whole_cell_rdme, dispatch_whole_cell_rdme, IntracellularLattice, IntracellularSpatialField,
    IntracellularSpatialState, IntracellularSpecies, WholeCellRdmeContext,
    WholeCellRdmeDriveField, WholeCellRdmeDriveState,
};
use crate::substrate_ir::{
    ScalarBranch, ScalarContext, ScalarFactor, ScalarRule, EMPTY_SCALAR_BRANCH, EMPTY_SCALAR_FACTOR,
};
use crate::whole_cell_data::{
    bundled_syn3a_genome_asset_package_json, bundled_syn3a_program_spec,
    bundled_syn3a_program_spec_json, compile_genome_asset_package, compile_genome_process_registry,
    compile_program_spec_from_bundle_manifest_path, derive_organism_profile,
    initialize_runtime_reaction_state, initialize_runtime_species_state, parse_program_spec_json,
    parse_saved_state_json, saved_state_to_json, WholeCellAssemblyFamily, WholeCellAssetClass,
    WholeCellBulkField,
    WholeCellChromosomeForkDirection, WholeCellChromosomeForkState, WholeCellChromosomeLocusState,
    WholeCellChromosomeState, WholeCellComplexAssemblyState, WholeCellComplexSpec,
    WholeCellContractSchema, WholeCellGenomeAssetPackage, WholeCellGenomeAssetSummary,
    WholeCellGenomeFeature, WholeCellGenomeProcessRegistry, WholeCellGenomeProcessRegistrySummary,
    WholeCellLatticeState, WholeCellLocalChemistrySpec, WholeCellMembraneDivisionState,
    WholeCellMoleculePoolSpec, WholeCellNamedComplexState, WholeCellOrganismExpressionState,
    WholeCellOrganismProfile, WholeCellOrganismSpec, WholeCellOrganismSummary,
    WholeCellPatchDomain, WholeCellProcessWeights, WholeCellProgramSpec, WholeCellProvenance, WholeCellReactionClass,
    WholeCellReactionRuntimeState, WholeCellSavedCoreState, WholeCellSavedState,
    WholeCellSchedulerState, WholeCellSolverStage, WholeCellSpatialFieldState,
    WholeCellSpatialScope, WholeCellSpeciesClass, WholeCellSpeciesRuntimeState,
    WholeCellStageClockState, WholeCellTranscriptionUnitState,
};
use crate::whole_cell_submodels::{
    LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeReport, LocalMDProbeRequest,
    ScheduledSubsystemProbe, Syn3ASubsystemPreset, WholeCellChemistryBridge,
    WholeCellDerivationCalibrationFit, WholeCellDerivationCalibrationSample,
    WholeCellSubsystemState,
};

#[cfg(target_os = "macos")]
use crate::gpu::GpuContext;

/// Execution backend chosen for the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WholeCellBackend {
    Cpu,
    Metal,
}

impl WholeCellBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            WholeCellBackend::Cpu => "cpu",
            WholeCellBackend::Metal => "metal",
        }
    }
}

/// Optional quantum-chemistry correction profile supplied by nQPU or another
/// external chemistry backend. The runtime uses these as multiplicative
/// modifiers rather than pushing Python or quantum logic into the hot loop.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellQuantumProfile {
    pub oxphos_efficiency: f32,
    pub translation_efficiency: f32,
    pub nucleotide_polymerization_efficiency: f32,
    pub membrane_synthesis_efficiency: f32,
    pub chromosome_segregation_efficiency: f32,
}

impl Default for WholeCellQuantumProfile {
    fn default() -> Self {
        Self {
            oxphos_efficiency: 1.0,
            translation_efficiency: 1.0,
            nucleotide_polymerization_efficiency: 1.0,
            membrane_synthesis_efficiency: 1.0,
            chromosome_segregation_efficiency: 1.0,
        }
    }
}

impl WholeCellQuantumProfile {
    fn normalized(self) -> Self {
        Self {
            oxphos_efficiency: self.oxphos_efficiency.clamp(0.5, 2.5),
            translation_efficiency: self.translation_efficiency.clamp(0.5, 2.5),
            nucleotide_polymerization_efficiency: self
                .nucleotide_polymerization_efficiency
                .clamp(0.5, 2.5),
            membrane_synthesis_efficiency: self.membrane_synthesis_efficiency.clamp(0.5, 2.5),
            chromosome_segregation_efficiency: self
                .chromosome_segregation_efficiency
                .clamp(0.5, 2.5),
        }
    }
}

/// Static runtime configuration for the whole-cell simulator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellConfig {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_nm: f32,
    pub dt_ms: f32,
    pub cme_interval: u64,
    pub ode_interval: u64,
    pub bd_interval: u64,
    pub geometry_interval: u64,
    pub use_gpu: bool,
}

impl Default for WholeCellConfig {
    fn default() -> Self {
        Self {
            x_dim: 24,
            y_dim: 24,
            z_dim: 12,
            voxel_size_nm: 20.0,
            dt_ms: 0.25,
            cme_interval: 4,
            ode_interval: 1,
            bd_interval: 2,
            geometry_interval: 4,
            use_gpu: true,
        }
    }
}

/// Flattened view of the current simulation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WholeCellSnapshot {
    pub backend: WholeCellBackend,
    pub time_ms: f32,
    pub step_count: u64,
    pub atp_mm: f32,
    pub amino_acids_mm: f32,
    pub nucleotides_mm: f32,
    pub membrane_precursors_mm: f32,
    pub adp_mm: f32,
    pub glucose_mm: f32,
    pub oxygen_mm: f32,
    pub ftsz: f32,
    pub dnaa: f32,
    pub active_ribosomes: f32,
    pub active_rnap: f32,
    pub genome_bp: u32,
    pub replicated_bp: u32,
    pub chromosome_separation_nm: f32,
    pub chromosome_state: WholeCellChromosomeState,
    pub membrane_division_state: WholeCellMembraneDivisionState,
    pub radius_nm: f32,
    pub surface_area_nm2: f32,
    pub volume_nm3: f32,
    pub division_progress: f32,
    pub quantum_profile: WholeCellQuantumProfile,
    pub local_chemistry: Option<LocalChemistryReport>,
    pub local_chemistry_sites: Vec<LocalChemistrySiteReport>,
    pub local_md_probe: Option<LocalMDProbeReport>,
    pub subsystem_states: Vec<WholeCellSubsystemState>,
    pub scheduler_state: WholeCellSchedulerState,
}

type WholeCellAssemblyInventory = WholeCellComplexAssemblyState;

#[derive(Debug, Clone, Copy)]
struct WholeCellProcessFluxes {
    energy_capacity: f32,
    transcription_capacity: f32,
    translation_capacity: f32,
    replication_capacity: f32,
    segregation_capacity: f32,
    membrane_capacity: f32,
    constriction_capacity: f32,
}

#[derive(Debug, Clone, Copy)]
struct WholeCellOrganismProcessScales {
    energy_scale: f32,
    transcription_scale: f32,
    translation_scale: f32,
    replication_scale: f32,
    segregation_scale: f32,
    membrane_scale: f32,
    constriction_scale: f32,
    amino_cost_scale: f32,
    nucleotide_cost_scale: f32,
}

impl Default for WholeCellOrganismProcessScales {
    fn default() -> Self {
        Self {
            energy_scale: 1.0,
            transcription_scale: 1.0,
            translation_scale: 1.0,
            replication_scale: 1.0,
            segregation_scale: 1.0,
            membrane_scale: 1.0,
            constriction_scale: 1.0,
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WholeCellSpatialCouplingCache {
    bulk_concentrations: [[f32; 7]; 4],
    overlap: [[f32; 4]; 4],
    patch_bulk_concentrations: [[f32; 7]; 5],
    patch_overlap: [[f32; 5]; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum SubsystemEstimatorSignal {
    HealthMix = 0,
    SupportScale,
    DemandCrowdingMix,
    PenaltyMix,
}

impl SubsystemEstimatorSignal {
    const COUNT: usize = Self::PenaltyMix as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct SubsystemEstimatorContext {
    signals: [f32; SubsystemEstimatorSignal::COUNT],
}

impl Default for SubsystemEstimatorContext {
    fn default() -> Self {
        Self {
            signals: [0.0; SubsystemEstimatorSignal::COUNT],
        }
    }
}

impl SubsystemEstimatorContext {
    fn set(&mut self, signal: SubsystemEstimatorSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ SubsystemEstimatorSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum ResourceEstimatorSignal {
    RawPool = 0,
    LocalMean,
    SupportMix,
    Pressure,
}

impl ResourceEstimatorSignal {
    const COUNT: usize = Self::Pressure as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct ResourceEstimatorContext {
    signals: [f32; ResourceEstimatorSignal::COUNT],
}

impl Default for ResourceEstimatorContext {
    fn default() -> Self {
        Self {
            signals: [0.0; ResourceEstimatorSignal::COUNT],
        }
    }
}

impl ResourceEstimatorContext {
    fn set(&mut self, signal: ResourceEstimatorSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn scalar(self) -> ScalarContext<{ ResourceEstimatorSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const fn subsystem_factor(signal: SubsystemEstimatorSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const fn resource_factor(signal: ResourceEstimatorSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const SUBSYSTEM_INVENTORY_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.108,
    4,
    [
        ScalarBranch::new(
            1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::HealthMix, 0.0, 1.0),
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
                subsystem_factor(SubsystemEstimatorSignal::SupportScale, 0.0, 1.0),
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
            1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::DemandCrowdingMix, 0.0, 1.0),
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
            -1.0,
            1,
            [
                subsystem_factor(SubsystemEstimatorSignal::PenaltyMix, 0.0, 1.0),
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
                EMPTY_SCALAR_FACTOR,
            ],
        ),
    ],
    0.15,
    1.60,
);

const GLUCOSE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.42,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.20,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const OXYGEN_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.50,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.16,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const AMINO_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.08,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.62,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.08,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const NUCLEOTIDE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.12,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.62,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.10,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.16,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

const MEMBRANE_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.10,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            1.00,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.08,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.12,
        ),
    ],
    0.0,
    1.0,
);

const ENERGY_SIGNAL_RULE: ScalarRule = ScalarRule::new(
    0.08,
    4,
    [
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::RawPool, 0.0, 1.0),
            0.40,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::LocalMean, 0.0, 1.0),
            0.22,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::SupportMix, 0.0, 1.0),
            0.18,
        ),
        scalar_branch_1(
            resource_factor(ResourceEstimatorSignal::Pressure, 0.0, 1.0),
            -0.10,
        ),
    ],
    0.0,
    1.0,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum WholeCellRuleSignal {
    Dt = 0,
    AtpBandSignal,
    RibosomeSignal,
    ReplisomeSignal,
    SeptumSignal,
    WeightedRibosomeReplisomeSignal,
    WeightedAtpSeptumSignal,
    GlucoseSignal,
    OxygenSignal,
    AminoSignal,
    NucleotideSignal,
    MembraneSignal,
    EnergySignal,
    ReplicatedFraction,
    InverseReplicatedFraction,
    DivisionReadiness,
    LocalizedSupplyScale,
    CrowdingPenalty,
    AtpSupport,
    TranslationSupport,
    NucleotideSupport,
    MembraneSupport,
    AtpBandScale,
    RibosomeTranslationScale,
    ReplisomeReplicationScale,
    ReplisomeSegregationScale,
    MembraneAssemblyScale,
    FtszConstrictionScale,
    MdTranslationScale,
    MdMembraneScale,
    QuantumOxphosEfficiency,
    QuantumTranslationEfficiency,
    QuantumNucleotideEfficiency,
    QuantumMembraneEfficiency,
    QuantumSegregationEfficiency,
    EffectiveMetabolicLoad,
    MembranePrecursorFloor,
    AtpBandComplexes,
    RibosomeComplexes,
    RnapComplexes,
    ReplisomeComplexes,
    MembraneComplexes,
    FtszPolymer,
    DnaaActivity,
    EnergyCapacity,
    EnergyCapacityCapped16,
    EnergyCapacityCapped18,
    TranscriptionCapacity,
    TranscriptionCapacityCapped16,
    TranslationCapacity,
    ReplicationCapacity,
    SegregationCapacity,
    MembraneCapacity,
    ConstrictionCapacity,
    DnaaSignal,
    ReplisomeAssemblySignal,
    ConstrictionSignal,
    TranscriptionDriveMix,
    TranslationDriveMix,
    BiosyntheticLoadMix,
    ConstrictionFlux,
}

impl WholeCellRuleSignal {
    const COUNT: usize = Self::ConstrictionFlux as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct WholeCellRuleContext {
    signals: [f32; WholeCellRuleSignal::COUNT],
}

impl Default for WholeCellRuleContext {
    fn default() -> Self {
        Self {
            signals: [0.0; WholeCellRuleSignal::COUNT],
        }
    }
}

impl WholeCellRuleContext {
    fn set(&mut self, signal: WholeCellRuleSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn get(self, signal: WholeCellRuleSignal) -> f32 {
        self.signals[signal as usize]
    }

    fn scalar(self) -> ScalarContext<{ WholeCellRuleSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const fn scalar_factor(signal: WholeCellRuleSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const fn scalar_branch_1(f1: ScalarFactor, coefficient: f32) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        1,
        [
            f1,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_2(f1: ScalarFactor, f2: ScalarFactor, coefficient: f32) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        2,
        [
            f1,
            f2,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_3(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        3,
        [
            f1,
            f2,
            f3,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_4(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        4,
        [
            f1,
            f2,
            f3,
            f4,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_5(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        5,
        [
            f1,
            f2,
            f3,
            f4,
            f5,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_6(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        6,
        [
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_7(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    f7: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        7,
        [f1, f2, f3, f4, f5, f6, f7, EMPTY_SCALAR_FACTOR],
    )
}

const fn scalar_branch_8(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    f7: ScalarFactor,
    f8: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(coefficient, 8, [f1, f2, f3, f4, f5, f6, f7, f8])
}

const ATP_BAND_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    18.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::AtpBandSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.70, 0.30),
            30.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    6.0,
    128.0,
);

const RIBOSOME_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    18.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::RibosomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.65, 0.35),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.60, 0.40),
            42.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    12.0,
    256.0,
);

const RNAP_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    10.0,
    1,
    [
        scalar_branch_3(
            scalar_factor(
                WholeCellRuleSignal::WeightedRibosomeReplisomeSignal,
                0.0,
                1.0,
            ),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.70, 0.30),
            20.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    8.0,
    128.0,
);

const REPLISOME_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    8.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.68, 0.32),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.85, 0.15),
            26.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    4.0,
    160.0,
);

const MEMBRANE_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    14.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::WeightedAtpSeptumSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.65, 0.35),
            26.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    6.0,
    160.0,
);

const FTSZ_POLYMER_RULE: ScalarRule = ScalarRule::new(
    28.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::SeptumSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.58, 0.42),
            scalar_factor(WholeCellRuleSignal::DivisionReadiness, 0.50, 0.50),
            78.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    12.0,
    320.0,
);

const DNAA_ACTIVITY_RULE: ScalarRule = ScalarRule::new(
    20.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.62, 0.38),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.80, 0.20),
            56.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    8.0,
    192.0,
);

const ENERGY_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_6(
            scalar_factor(WholeCellRuleSignal::AtpBandComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AtpSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AtpBandScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.40,
    2.40,
);

const TRANSCRIPTION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::RnapComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.40,
);

const TRANSLATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_6(
            scalar_factor(WholeCellRuleSignal::RibosomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranslationSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::RibosomeTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.55, 0.45),
            1.0 / 46.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.60,
);

const REPLICATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeReplicationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            1.0 / 28.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.40,
);

const SEGREGATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeSegregationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumSegregationEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.60, 0.40),
            1.0 / 28.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.40,
);

const MEMBRANE_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::MembraneComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneAssemblyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdMembraneScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.40,
);

const CONSTRICTION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::FtszPolymer, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::FtszConstrictionScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            1.0 / 90.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.60,
);

const TRANSCRIPTION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_7(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranscriptionCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::TranscriptionDriveMix, 0.50, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.65, 0.35),
            0.060,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const TRANSLATION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_7(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranslationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::TranslationDriveMix, 0.55, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumTranslationEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            scalar_factor(
                WholeCellRuleSignal::TranscriptionCapacityCapped16,
                0.65,
                0.35,
            ),
            0.085,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const ENERGY_GAIN_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergyCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumOxphosEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.55, 0.45),
            0.0155,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const ENERGY_COST_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_3(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EffectiveMetabolicLoad, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::BiosyntheticLoadMix, 0.34, 1.0),
            0.010,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const NUCLEOTIDE_RECHARGE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergyCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            0.0032,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const MEMBRANE_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumMembraneEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::EnergyCapacityCapped18, 0.55, 0.45),
            0.0028,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const REPLICATION_DRIVE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_8(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplicationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::DnaaSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplisomeAssemblySignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            18.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const SEGREGATION_STEP_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::SegregationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeAssemblySignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 3.0, 18.0),
            1.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const MEMBRANE_GROWTH_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembranePrecursorFloor, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumMembraneEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneCapacity, 0.0, 1.0),
            14.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const CONSTRICTION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 0.55, 0.45),
            1.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const CONSTRICTION_DRIVE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    3,
    [
        scalar_branch_1(scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0), 0.002),
        scalar_branch_2(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 0.0, 1.0),
            0.012,
        ),
        scalar_branch_3(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionFlux, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumTranslationEfficiency, 0.0, 1.0),
            0.005,
        ),
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

/// Native whole-cell simulator with a Rust-owned state and scheduler.
pub struct WholeCellSimulator {
    config: WholeCellConfig,
    backend: WholeCellBackend,
    program_name: Option<String>,
    contract: WholeCellContractSchema,
    provenance: WholeCellProvenance,
    organism_data_ref: Option<String>,
    organism_data: Option<WholeCellOrganismSpec>,
    organism_assets: Option<WholeCellGenomeAssetPackage>,
    organism_process_registry: Option<WholeCellGenomeProcessRegistry>,
    organism_expression: WholeCellOrganismExpressionState,
    organism_species: Vec<WholeCellSpeciesRuntimeState>,
    organism_reactions: Vec<WholeCellReactionRuntimeState>,
    complex_assembly: WholeCellComplexAssemblyState,
    named_complexes: Vec<WholeCellNamedComplexState>,
    #[cfg(target_os = "macos")]
    gpu: Option<GpuContext>,
    lattice: IntracellularLattice,
    spatial_fields: IntracellularSpatialState,
    rdme_drive_fields: WholeCellRdmeDriveState,
    time_ms: f32,
    step_count: u64,
    atp_mm: f32,
    amino_acids_mm: f32,
    nucleotides_mm: f32,
    membrane_precursors_mm: f32,
    adp_mm: f32,
    glucose_mm: f32,
    oxygen_mm: f32,
    ftsz: f32,
    dnaa: f32,
    active_ribosomes: f32,
    active_rnap: f32,
    genome_bp: u32,
    replicated_bp: u32,
    chromosome_separation_nm: f32,
    chromosome_state: WholeCellChromosomeState,
    membrane_division_state: WholeCellMembraneDivisionState,
    radius_nm: f32,
    surface_area_nm2: f32,
    volume_nm3: f32,
    division_progress: f32,
    metabolic_load: f32,
    quantum_profile: WholeCellQuantumProfile,
    chemistry_bridge: Option<WholeCellChemistryBridge>,
    chemistry_report: LocalChemistryReport,
    chemistry_site_reports: Vec<LocalChemistrySiteReport>,
    last_md_probe: Option<LocalMDProbeReport>,
    scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
    subsystem_states: Vec<WholeCellSubsystemState>,
    scheduler_state: WholeCellSchedulerState,
    md_translation_scale: f32,
    md_membrane_scale: f32,
}

impl WholeCellSimulator {
    const SOLVER_STAGE_ORDER: [WholeCellSolverStage; 6] = [
        WholeCellSolverStage::AtomisticRefinement,
        WholeCellSolverStage::Rdme,
        WholeCellSolverStage::Cme,
        WholeCellSolverStage::Ode,
        WholeCellSolverStage::ChromosomeBd,
        WholeCellSolverStage::Geometry,
    ];

    fn finite_scale(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
        if value.is_finite() {
            value.clamp(min_value, max_value)
        } else {
            fallback.clamp(min_value, max_value)
        }
    }

    fn process_scale(signal: f32, mean_signal: f32) -> f32 {
        if mean_signal <= 1.0e-6 {
            1.0
        } else {
            (0.82 + 0.28 * (signal / mean_signal)).clamp(0.68, 1.45)
        }
    }

    fn scaled_interval(base: u64, drive: f32, max_multiplier: u64) -> u64 {
        let base = base.max(1);
        let base_f = base as f32;
        let interval = if drive >= 1.0 {
            base_f / drive.max(1.0)
        } else {
            base_f * (1.0 + 0.80 * (1.0 - drive.max(0.0)))
        };
        interval
            .round()
            .clamp(1.0, (base * max_multiplier.max(1)) as f32) as u64
    }

    fn build_scheduler_state(config: &WholeCellConfig) -> WholeCellSchedulerState {
        WholeCellSchedulerState {
            stage_clocks: Self::SOLVER_STAGE_ORDER
                .iter()
                .copied()
                .map(|stage| WholeCellStageClockState {
                    stage,
                    base_interval_steps: Self::base_stage_interval_from_config(config, stage),
                    dynamic_interval_steps: Self::base_stage_interval_from_config(config, stage),
                    next_due_step: 0,
                    run_count: 0,
                    last_run_step: None,
                    last_run_time_ms: 0.0,
                })
                .collect(),
        }
    }

    fn normalized_scheduler_state(
        config: &WholeCellConfig,
        saved: WholeCellSchedulerState,
    ) -> WholeCellSchedulerState {
        let mut scheduler_state = Self::build_scheduler_state(config);
        for saved_clock in saved.stage_clocks {
            if let Some(clock) = scheduler_state
                .stage_clocks
                .iter_mut()
                .find(|clock| clock.stage == saved_clock.stage)
            {
                clock.base_interval_steps =
                    Self::base_stage_interval_from_config(config, saved_clock.stage).max(1);
                clock.dynamic_interval_steps = saved_clock.dynamic_interval_steps.max(1);
                clock.next_due_step = saved_clock.next_due_step;
                clock.run_count = saved_clock.run_count;
                clock.last_run_step = saved_clock.last_run_step;
                clock.last_run_time_ms = saved_clock.last_run_time_ms.max(0.0);
            }
        }
        scheduler_state
    }

    fn base_stage_interval_from_config(
        config: &WholeCellConfig,
        stage: WholeCellSolverStage,
    ) -> u64 {
        match stage {
            WholeCellSolverStage::AtomisticRefinement => 1,
            WholeCellSolverStage::Rdme => 1,
            WholeCellSolverStage::Cme => config.cme_interval.max(1),
            WholeCellSolverStage::Ode => config.ode_interval.max(1),
            WholeCellSolverStage::ChromosomeBd => config.bd_interval.max(1),
            WholeCellSolverStage::Geometry => config.geometry_interval.max(1),
        }
    }

    fn scheduler_clock(&self, stage: WholeCellSolverStage) -> Option<&WholeCellStageClockState> {
        self.scheduler_state
            .stage_clocks
            .iter()
            .find(|clock| clock.stage == stage)
    }

    fn scheduler_clock_mut(
        &mut self,
        stage: WholeCellSolverStage,
    ) -> Option<&mut WholeCellStageClockState> {
        self.scheduler_state
            .stage_clocks
            .iter_mut()
            .find(|clock| clock.stage == stage)
    }

    fn average_unit_stress_penalty(&self) -> f32 {
        if self.organism_expression.transcription_units.is_empty() {
            1.0
        } else {
            self.organism_expression
                .transcription_units
                .iter()
                .map(|unit| unit.stress_penalty.max(0.0))
                .sum::<f32>()
                / self.organism_expression.transcription_units.len() as f32
        }
    }

    fn adaptive_interval_for_stage(&self, stage: WholeCellSolverStage) -> u64 {
        let base = Self::base_stage_interval_from_config(&self.config, stage);
        let expression = &self.organism_expression;
        let stress = (self.average_unit_stress_penalty() - 1.0).max(0.0);
        let load = (self.effective_metabolic_load() - 1.0).max(0.0);
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        match stage {
            WholeCellSolverStage::AtomisticRefinement => {
                if self.chemistry_bridge.is_none() {
                    base
                } else {
                    let probe_floor = self
                        .scheduled_subsystem_probes
                        .iter()
                        .map(|probe| probe.interval_steps.max(1))
                        .min()
                        .unwrap_or(base.max(1));
                    let drive = (0.85
                        + 0.45 * load
                        + 0.25 * stress
                        + 0.18 * self.division_progress
                        + 0.12 * (1.0 - self.chemistry_report.crowding_penalty).max(0.0))
                    .clamp(0.6, 2.6);
                    Self::scaled_interval(base.max(1), drive, 3).min(probe_floor)
                }
            }
            WholeCellSolverStage::Rdme => 1,
            WholeCellSolverStage::Cme => {
                let drive = (0.70
                    + 0.28 * expression.global_activity
                    + 0.22 * expression.process_scales.transcription
                    + 0.18 * expression.translation_support
                    + 0.18 * stress)
                    .clamp(0.45, 2.8);
                Self::scaled_interval(base, drive, 3)
            }
            WholeCellSolverStage::Ode => {
                let energy_stress =
                    ((self.adp_mm + 0.25) / (self.atp_mm + 0.25)).clamp(0.35, 3.0) - 0.35;
                let drive = (0.82
                    + 0.34 * energy_stress.max(0.0)
                    + 0.28 * load
                    + 0.16 * (1.0 - expression.energy_support).max(0.0))
                .clamp(0.55, 3.0);
                Self::scaled_interval(base, drive, 3)
            }
            WholeCellSolverStage::ChromosomeBd => {
                let drive = (0.58
                    + 0.62 * replicated_fraction
                    + 0.24 * expression.process_scales.replication
                    + 0.18 * self.complex_assembly.replisome_complexes.sqrt() / 6.0)
                    .clamp(0.40, 2.8);
                Self::scaled_interval(base, drive, 4)
            }
            WholeCellSolverStage::Geometry => {
                let drive = (0.54
                    + 0.52 * self.division_progress
                    + 0.22 * expression.process_scales.membrane
                    + 0.18 * expression.process_scales.constriction
                    + 0.12 * self.complex_assembly.ftsz_polymer.sqrt() / 8.0)
                    .clamp(0.35, 2.6);
                Self::scaled_interval(base, drive, 4)
            }
        }
    }

    fn refresh_multirate_scheduler(&mut self) {
        let updates = Self::SOLVER_STAGE_ORDER
            .iter()
            .copied()
            .map(|stage| {
                (
                    stage,
                    Self::base_stage_interval_from_config(&self.config, stage),
                    self.adaptive_interval_for_stage(stage),
                )
            })
            .collect::<Vec<_>>();
        for (stage, base_interval, dynamic_interval) in updates {
            if let Some(clock) = self.scheduler_clock_mut(stage) {
                clock.base_interval_steps = base_interval.max(1);
                clock.dynamic_interval_steps = dynamic_interval.max(1);
            }
        }
    }

    fn stage_due(&self, stage: WholeCellSolverStage) -> bool {
        if stage == WholeCellSolverStage::AtomisticRefinement && self.chemistry_bridge.is_none() {
            return false;
        }
        self.scheduler_clock(stage)
            .map(|clock| self.step_count >= clock.next_due_step)
            .unwrap_or(false)
    }

    fn stage_dt_ms(&self, stage: WholeCellSolverStage, base_dt: f32) -> f32 {
        if let Some(clock) = self.scheduler_clock(stage) {
            let steps = if let Some(last_run_step) = clock.last_run_step {
                self.step_count
                    .saturating_sub(last_run_step)
                    .max(clock.dynamic_interval_steps.max(1))
            } else {
                clock.dynamic_interval_steps.max(1)
            };
            base_dt * steps.max(1) as f32
        } else {
            base_dt
        }
    }

    fn record_stage_run(&mut self, stage: WholeCellSolverStage, dt_ms: f32) {
        let next_interval = self.adaptive_interval_for_stage(stage).max(1);
        let step_count = self.step_count;
        let time_ms = self.time_ms;
        if let Some(clock) = self.scheduler_clock_mut(stage) {
            clock.dynamic_interval_steps = next_interval;
            clock.run_count += 1;
            clock.last_run_step = Some(step_count);
            clock.last_run_time_ms = time_ms;
            clock.next_due_step = step_count.saturating_add(next_interval);
        }
        if dt_ms.is_finite() && dt_ms > 0.0 {
            let _ = dt_ms;
        }
    }

    fn circular_distance_bp(a: f32, b: f32, genome_bp: f32) -> f32 {
        let delta = (a - b).abs();
        delta.min((genome_bp - delta).abs())
    }

    fn circular_position_bp(position_bp: i64, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1) as i64;
        position_bp.rem_euclid(genome_bp) as u32
    }

    fn clockwise_distance_bp(origin_bp: u32, target_bp: u32, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1);
        if target_bp >= origin_bp {
            target_bp - origin_bp
        } else {
            genome_bp - (origin_bp - target_bp)
        }
    }

    fn counter_clockwise_distance_bp(origin_bp: u32, target_bp: u32, genome_bp: u32) -> u32 {
        Self::clockwise_distance_bp(target_bp, origin_bp, genome_bp)
    }

    fn chromosome_position_from_origin(
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

    fn chromosome_arm_length(
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

    fn chromosome_domain_index(midpoint_bp: u32, genome_bp: u32) -> u32 {
        let genome_bp = genome_bp.max(1);
        ((4 * midpoint_bp.min(genome_bp.saturating_sub(1))) / genome_bp).min(3)
    }

    fn chromosome_locus_records(&self, organism: &WholeCellOrganismSpec) -> Vec<(String, u32, i8)> {
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

    fn chromosome_forks_from_progress(
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

    fn seeded_chromosome_state(&self) -> WholeCellChromosomeState {
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
                domain_index: Self::chromosome_domain_index(midpoint_bp, genome_bp),
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

    fn normalize_chromosome_state(
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
                Self::chromosome_domain_index(locus.midpoint_bp, state.chromosome_length_bp);
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

    fn synchronize_chromosome_summary(&mut self) {
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

    fn membrane_cardiolipin_share(&self) -> f32 {
        self.organism_data
            .as_ref()
            .map(|organism| (0.08 + 0.40 * organism.composition.lipid_fraction).clamp(0.08, 0.32))
            .unwrap_or(0.16)
    }

    fn seeded_membrane_division_state(&self) -> WholeCellMembraneDivisionState {
        let preferred_area_nm2 = self.surface_area_nm2.max(1.0);
        let cardiolipin_share = self.membrane_cardiolipin_share();
        let division_progress = self.division_progress.clamp(0.0, 0.99);
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

    fn normalize_membrane_division_state(
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

    fn synchronize_membrane_division_summary(&mut self) {
        self.membrane_division_state =
            self.normalize_membrane_division_state(self.membrane_division_state.clone());
        self.surface_area_nm2 = self.membrane_division_state.membrane_area_nm2.max(1.0);
        self.radius_nm = (self.surface_area_nm2 / (4.0 * PI)).sqrt();
        self.volume_nm3 =
            Self::volume_from_radius(self.radius_nm) * self.membrane_division_state.osmotic_balance;
        self.division_progress =
            (1.0 - self.membrane_division_state.septum_radius_fraction).clamp(0.0, 0.99);
    }

    fn membrane_chromosome_occlusion(&self) -> f32 {
        ((1.0 - self.chromosome_state.segregation_progress).clamp(0.0, 1.0)
            * (0.45 + 0.55 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
            * (0.70 + 0.30 * self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)))
        .clamp(0.0, 1.5)
    }

    fn update_membrane_division_state(
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
            self.spatial_species_mean(IntracellularSpecies::ATP, IntracellularSpatialField::MembraneBandZone),
            0.8,
        );
        let local_polar_atp = Self::saturating_signal(
            self.spatial_species_mean(IntracellularSpecies::ATP, IntracellularSpatialField::PoleZone),
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
                * ((0.032 * state.phospholipid_inventory_nm2 + 0.022 * state.cardiolipin_inventory_nm2)
                    * (0.55 + 0.45 * local_band_precursors)
                    * (0.55 + 0.45 * local_band_atp)
                    - 0.022 * state.membrane_band_lipid_inventory_nm2
                    - 0.018
                        * state.band_turnover_pressure
                        * state.membrane_band_lipid_inventory_nm2))
        .max(0.0);
        state.polar_lipid_inventory_nm2 = (state.polar_lipid_inventory_nm2
            + dt_scale
                * ((0.024 * state.phospholipid_inventory_nm2 + 0.034 * state.cardiolipin_inventory_nm2)
                    * (0.55 + 0.45 * local_polar_precursors)
                    * (0.55 + 0.45 * local_polar_atp)
                    - 0.018 * state.polar_lipid_inventory_nm2
                    - 0.016
                        * state.pole_turnover_pressure
                        * state.polar_lipid_inventory_nm2))
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

    fn chromosome_copy_number_for_state(state: &WholeCellChromosomeState, midpoint_bp: u32) -> f32 {
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

    fn chromosome_copy_number_at(&self, midpoint_bp: u32) -> f32 {
        Self::chromosome_copy_number_for_state(&self.chromosome_state, midpoint_bp)
    }

    fn chromosome_locus_accessibility_at(&self, midpoint_bp: u32) -> f32 {
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

    fn chromosome_locus_torsional_stress_at(&self, midpoint_bp: u32) -> f32 {
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

    fn chromosome_collision_pressure(
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

    fn advance_chromosome_state(
        &mut self,
        dt: f32,
        replication_drive: f32,
        segregation_drive: f32,
    ) {
        let mut state = self.normalize_chromosome_state(self.chromosome_state.clone());
        let genome_bp = state.chromosome_length_bp.max(1);
        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let initiation_support = Self::finite_scale(
            0.40 * Self::saturating_signal(self.dnaa, 64.0)
                + 0.24 * self.organism_expression.nucleotide_support
                + 0.18 * crowding
                + 0.18 * Self::saturating_signal(self.complex_assembly.replisome_complexes, 16.0),
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
            let pause_pressure = (0.52 * collision_pressure
                + 0.22 * (1.0 - crowding).max(0.0)
                + 0.26 * (1.0 - self.organism_expression.nucleotide_support).max(0.0))
            .clamp(0.0, 1.5);
            let paused = pause_pressure > 0.85;
            let progress_scale = if paused {
                0.18
            } else {
                (1.0 - 0.70 * pause_pressure).clamp(0.12, 1.0)
            };
            let raw_delta = (0.5
                * replication_drive.max(0.0)
                * self.organism_expression.process_scales.replication.max(0.1)
                * progress_scale
                * dt_scale)
                .round();
            let delta_bp = if raw_delta <= 0.0 {
                u32::from(replication_drive > 0.0)
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
            locus.domain_index = Self::chromosome_domain_index(locus.midpoint_bp, genome_bp);
            accessibility_total += accessibility;
        }
        if !state.loci.is_empty() {
            state.mean_locus_accessibility = accessibility_total / state.loci.len() as f32;
        }
        self.chromosome_state = self.normalize_chromosome_state(state);
        self.synchronize_chromosome_summary();
        self.refresh_spatial_fields();
    }

    fn molecule_pool_count(spec: &WholeCellMoleculePoolSpec) -> f32 {
        if spec.count.is_finite() {
            spec.count.max(0.0)
        } else {
            0.0
        }
    }

    fn apply_pool_seed(&mut self, pool: &WholeCellMoleculePoolSpec) {
        let name = pool.species.trim().to_lowercase();
        if pool.concentration_mm.is_finite() && pool.concentration_mm > 0.0 {
            let concentration = pool.concentration_mm.max(0.0);
            if name.contains("amino") {
                self.lattice
                    .fill_species(IntracellularSpecies::AminoAcids, concentration);
            } else if name.contains("nucleotide") {
                self.lattice
                    .fill_species(IntracellularSpecies::Nucleotides, concentration);
            } else if name.contains("membrane") {
                self.lattice
                    .fill_species(IntracellularSpecies::MembranePrecursors, concentration);
            } else if name == "atp" {
                self.lattice
                    .fill_species(IntracellularSpecies::ATP, concentration);
            } else if name == "adp" {
                self.adp_mm = concentration;
            } else if name.contains("glucose") {
                self.glucose_mm = concentration;
            } else if name.contains("oxygen") {
                self.oxygen_mm = concentration;
            }
        }

        let count = Self::molecule_pool_count(pool);
        if count <= 0.0 {
            return;
        }
        if name.contains("ribosome") {
            self.active_ribosomes = count;
        } else if name.contains("rnap") || name.contains("rna_polymerase") {
            self.active_rnap = count;
        } else if name.contains("dnaa") {
            self.dnaa = count;
        } else if name.contains("ftsz") {
            self.ftsz = count;
        }
    }

    fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
        let value = value.max(0.0);
        let half_saturation = half_saturation.max(1.0e-6);
        (value / (value + half_saturation)).clamp(0.0, 1.0)
    }

    fn evaluate_resource_signal(
        rule: ScalarRule,
        raw_pool: f32,
        local_mean: f32,
        support_mix: f32,
        pressure: f32,
    ) -> f32 {
        let mut ctx = ResourceEstimatorContext::default();
        ctx.set(ResourceEstimatorSignal::RawPool, raw_pool);
        ctx.set(ResourceEstimatorSignal::LocalMean, local_mean);
        ctx.set(ResourceEstimatorSignal::SupportMix, support_mix);
        ctx.set(ResourceEstimatorSignal::Pressure, pressure);
        rule.evaluate(ctx.scalar())
    }

    fn subsystem_inventory_signal(state: WholeCellSubsystemState, support: f32) -> f32 {
        let mut ctx = SubsystemEstimatorContext::default();
        ctx.set(
            SubsystemEstimatorSignal::HealthMix,
            0.18 * state.structural_order
                + 0.16 * state.assembly_component_availability
                + 0.20 * state.assembly_occupancy
                + 0.16 * state.assembly_stability,
        );
        ctx.set(SubsystemEstimatorSignal::SupportScale, support);
        ctx.set(
            SubsystemEstimatorSignal::DemandCrowdingMix,
            0.118 * state.demand_satisfaction + 0.074 * state.crowding_penalty,
        );
        ctx.set(
            SubsystemEstimatorSignal::PenaltyMix,
            0.14 * state.assembly_turnover + 0.02 * state.byproduct_load,
        );
        SUBSYSTEM_INVENTORY_SIGNAL_RULE.evaluate(ctx.scalar())
    }

    fn base_rule_context(&self, dt: f32) -> WholeCellRuleContext {
        let atp_band = self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let ribosome = self.subsystem_state(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        let replisome = self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack);
        let septum = self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing);

        let atp_support = Self::finite_scale(self.chemistry_report.atp_support, 1.0, 0.70, 1.50);
        let translation_support =
            Self::finite_scale(self.chemistry_report.translation_support, 1.0, 0.70, 1.50);
        let nucleotide_support =
            Self::finite_scale(self.chemistry_report.nucleotide_support, 1.0, 0.70, 1.50);
        let membrane_support =
            Self::finite_scale(self.chemistry_report.membrane_support, 1.0, 0.70, 1.50);
        let crowding_penalty =
            Self::finite_scale(self.chemistry_report.crowding_penalty, 1.0, 0.65, 1.0);

        let atp_band_signal = Self::subsystem_inventory_signal(atp_band, atp_support);
        let ribosome_signal = Self::subsystem_inventory_signal(ribosome, translation_support);
        let replisome_signal = Self::subsystem_inventory_signal(replisome, nucleotide_support);
        let septum_signal = Self::subsystem_inventory_signal(septum, membrane_support);
        let localized_nucleotide_pool = self.localized_nucleotide_pool_mm();
        let localized_membrane_pool = self.localized_membrane_precursor_pool_mm();
        let localized_membrane_atp = self.localized_membrane_atp_pool_mm();

        let localized_supply_scale = self.localized_supply_scale();
        let localized_resource_pressure =
            Self::finite_scale(self.localized_resource_pressure(), 0.0, 0.0, 1.5);
        let glucose_signal = Self::evaluate_resource_signal(
            GLUCOSE_SIGNAL_RULE,
            self.glucose_mm,
            self.chemistry_report.mean_glucose,
            localized_supply_scale,
            localized_resource_pressure,
        );
        let oxygen_signal = Self::evaluate_resource_signal(
            OXYGEN_SIGNAL_RULE,
            self.oxygen_mm,
            self.chemistry_report.mean_oxygen,
            localized_supply_scale,
            localized_resource_pressure,
        );
        let amino_signal = Self::evaluate_resource_signal(
            AMINO_SIGNAL_RULE,
            self.amino_acids_mm,
            0.5 * (self.chemistry_report.mean_glucose + self.chemistry_report.mean_atp_flux),
            0.60 * translation_support + 0.40 * localized_supply_scale,
            localized_resource_pressure,
        );
        let nucleotide_signal = Self::evaluate_resource_signal(
            NUCLEOTIDE_SIGNAL_RULE,
            0.55 * self.nucleotides_mm + 0.45 * localized_nucleotide_pool,
            0.5 * self.chemistry_report.mean_atp_flux,
            0.60 * nucleotide_support + 0.40 * localized_supply_scale,
            localized_resource_pressure,
        );
        let membrane_signal = Self::evaluate_resource_signal(
            MEMBRANE_SIGNAL_RULE,
            0.45 * self.membrane_precursors_mm + 0.55 * localized_membrane_pool,
            0.5 * self.chemistry_report.mean_oxygen,
            0.60 * membrane_support + 0.40 * localized_supply_scale,
            localized_resource_pressure,
        );
        let energy_signal = Self::evaluate_resource_signal(
            ENERGY_SIGNAL_RULE,
            0.60 * self.atp_mm + 0.40 * localized_membrane_atp,
            self.chemistry_report.mean_atp_flux,
            0.55 * atp_support + 0.45 * localized_supply_scale,
            localized_resource_pressure,
        );
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        let division_readiness = (0.35 + 0.65 * replicated_fraction).clamp(0.35, 1.0);

        let mut ctx = WholeCellRuleContext::default();
        ctx.set(WholeCellRuleSignal::Dt, dt);
        ctx.set(WholeCellRuleSignal::AtpBandSignal, atp_band_signal);
        ctx.set(WholeCellRuleSignal::RibosomeSignal, ribosome_signal);
        ctx.set(WholeCellRuleSignal::ReplisomeSignal, replisome_signal);
        ctx.set(WholeCellRuleSignal::SeptumSignal, septum_signal);
        ctx.set(
            WholeCellRuleSignal::WeightedRibosomeReplisomeSignal,
            0.55 * ribosome_signal + 0.45 * replisome_signal,
        );
        ctx.set(
            WholeCellRuleSignal::WeightedAtpSeptumSignal,
            0.55 * atp_band_signal + 0.45 * septum_signal,
        );
        ctx.set(WholeCellRuleSignal::GlucoseSignal, glucose_signal);
        ctx.set(WholeCellRuleSignal::OxygenSignal, oxygen_signal);
        ctx.set(WholeCellRuleSignal::AminoSignal, amino_signal);
        ctx.set(WholeCellRuleSignal::NucleotideSignal, nucleotide_signal);
        ctx.set(WholeCellRuleSignal::MembraneSignal, membrane_signal);
        ctx.set(WholeCellRuleSignal::EnergySignal, energy_signal);
        ctx.set(WholeCellRuleSignal::ReplicatedFraction, replicated_fraction);
        ctx.set(
            WholeCellRuleSignal::InverseReplicatedFraction,
            (1.0 - replicated_fraction).clamp(0.0, 1.0),
        );
        ctx.set(WholeCellRuleSignal::DivisionReadiness, division_readiness);
        ctx.set(
            WholeCellRuleSignal::LocalizedSupplyScale,
            self.localized_supply_scale(),
        );
        ctx.set(WholeCellRuleSignal::CrowdingPenalty, crowding_penalty);
        ctx.set(WholeCellRuleSignal::AtpSupport, atp_support);
        ctx.set(WholeCellRuleSignal::TranslationSupport, translation_support);
        ctx.set(WholeCellRuleSignal::NucleotideSupport, nucleotide_support);
        ctx.set(WholeCellRuleSignal::MembraneSupport, membrane_support);
        ctx.set(
            WholeCellRuleSignal::AtpBandScale,
            Self::finite_scale(self.atp_band_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::RibosomeTranslationScale,
            Self::finite_scale(self.ribosome_translation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::ReplisomeReplicationScale,
            Self::finite_scale(self.replisome_replication_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::ReplisomeSegregationScale,
            Self::finite_scale(self.replisome_segregation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MembraneAssemblyScale,
            Self::finite_scale(self.membrane_assembly_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::FtszConstrictionScale,
            Self::finite_scale(self.ftsz_constriction_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MdTranslationScale,
            Self::finite_scale(self.md_translation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MdMembraneScale,
            Self::finite_scale(self.md_membrane_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::QuantumOxphosEfficiency,
            self.quantum_profile.oxphos_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumTranslationEfficiency,
            self.quantum_profile.translation_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumNucleotideEfficiency,
            self.quantum_profile.nucleotide_polymerization_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumMembraneEfficiency,
            self.quantum_profile.membrane_synthesis_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumSegregationEfficiency,
            self.quantum_profile.chromosome_segregation_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::EffectiveMetabolicLoad,
            self.effective_metabolic_load(),
        );
        ctx.set(
            WholeCellRuleSignal::MembranePrecursorFloor,
            (0.45 * self.membrane_precursors_mm + 0.55 * localized_membrane_pool).max(0.1),
        );
        ctx
    }

    fn process_rule_context(
        &self,
        dt: f32,
        inventory: WholeCellAssemblyInventory,
    ) -> WholeCellRuleContext {
        let mut ctx = self.base_rule_context(dt);
        ctx.set(
            WholeCellRuleSignal::AtpBandComplexes,
            inventory.atp_band_complexes,
        );
        ctx.set(
            WholeCellRuleSignal::RibosomeComplexes,
            inventory.ribosome_complexes,
        );
        ctx.set(WholeCellRuleSignal::RnapComplexes, inventory.rnap_complexes);
        ctx.set(
            WholeCellRuleSignal::ReplisomeComplexes,
            inventory.replisome_complexes,
        );
        ctx.set(
            WholeCellRuleSignal::MembraneComplexes,
            inventory.membrane_complexes,
        );
        ctx.set(WholeCellRuleSignal::FtszPolymer, inventory.ftsz_polymer);
        ctx.set(WholeCellRuleSignal::DnaaActivity, inventory.dnaa_activity);
        ctx
    }

    fn stage_rule_context(
        &self,
        dt: f32,
        inventory: WholeCellAssemblyInventory,
        fluxes: WholeCellProcessFluxes,
    ) -> WholeCellRuleContext {
        let mut ctx = self.process_rule_context(dt, inventory);
        ctx.set(WholeCellRuleSignal::EnergyCapacity, fluxes.energy_capacity);
        ctx.set(
            WholeCellRuleSignal::EnergyCapacityCapped16,
            fluxes.energy_capacity.min(1.6),
        );
        ctx.set(
            WholeCellRuleSignal::EnergyCapacityCapped18,
            fluxes.energy_capacity.min(1.8),
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionCapacity,
            fluxes.transcription_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionCapacityCapped16,
            fluxes.transcription_capacity.min(1.6),
        );
        ctx.set(
            WholeCellRuleSignal::TranslationCapacity,
            fluxes.translation_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::ReplicationCapacity,
            fluxes.replication_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::SegregationCapacity,
            fluxes.segregation_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::MembraneCapacity,
            fluxes.membrane_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::ConstrictionCapacity,
            fluxes.constriction_capacity,
        );
        let replisome_replication_scale = self.replisome_replication_scale();
        ctx.set(
            WholeCellRuleSignal::DnaaSignal,
            Self::saturating_signal(inventory.dnaa_activity * replisome_replication_scale, 48.0),
        );
        let replisome_structure_scale =
            0.5 * (replisome_replication_scale + self.replisome_segregation_scale());
        ctx.set(
            WholeCellRuleSignal::ReplisomeAssemblySignal,
            Self::saturating_signal(
                inventory.replisome_complexes * replisome_structure_scale,
                28.0,
            ),
        );
        ctx.set(
            WholeCellRuleSignal::ConstrictionSignal,
            Self::saturating_signal(inventory.ftsz_polymer, 90.0),
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionDriveMix,
            0.35 * fluxes.energy_capacity.min(1.6)
                + 0.15 * ctx.get(WholeCellRuleSignal::GlucoseSignal),
        );
        ctx.set(
            WholeCellRuleSignal::TranslationDriveMix,
            0.30 * fluxes.energy_capacity.min(1.8) + 0.15 * fluxes.transcription_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::BiosyntheticLoadMix,
            0.28 * fluxes.translation_capacity
                + 0.16 * fluxes.transcription_capacity
                + 0.12 * fluxes.replication_capacity
                + 0.10 * fluxes.constriction_capacity,
        );
        ctx
    }

    fn assembly_inventory(&self) -> WholeCellAssemblyInventory {
        if self.complex_assembly.total_complexes() > 1.0e-6 {
            self.complex_assembly
        } else {
            self.derived_complex_assembly_target()
        }
    }

    fn process_fluxes(&self, inventory: WholeCellAssemblyInventory) -> WholeCellProcessFluxes {
        let ctx = self.process_rule_context(0.0, inventory);
        let scalar = ctx.scalar();
        WholeCellProcessFluxes {
            energy_capacity: ENERGY_CAPACITY_RULE.evaluate(scalar),
            transcription_capacity: TRANSCRIPTION_CAPACITY_RULE.evaluate(scalar),
            translation_capacity: TRANSLATION_CAPACITY_RULE.evaluate(scalar),
            replication_capacity: REPLICATION_CAPACITY_RULE.evaluate(scalar),
            segregation_capacity: SEGREGATION_CAPACITY_RULE.evaluate(scalar),
            membrane_capacity: MEMBRANE_CAPACITY_RULE.evaluate(scalar),
            constriction_capacity: CONSTRICTION_CAPACITY_RULE.evaluate(scalar),
        }
    }

    fn refresh_surrogate_pool_diagnostics(
        &mut self,
        inventory: WholeCellAssemblyInventory,
        transcription_flux: f32,
        translation_flux: f32,
        replication_flux: f32,
        membrane_flux: f32,
        constriction_flux: f32,
    ) {
        let replisome_scale =
            Self::finite_scale(self.replisome_replication_scale(), 1.0, 0.70, 1.45);
        let ftsz_translation_scale =
            Self::finite_scale(self.ftsz_translation_scale(), 1.0, 0.70, 1.45);
        self.active_rnap = (0.72 * self.active_rnap
            + 0.28 * (inventory.rnap_complexes + 8.0 * transcription_flux))
            .clamp(8.0, 256.0);
        self.active_ribosomes = (0.70 * self.active_ribosomes
            + 0.30 * (inventory.ribosome_complexes + 6.0 * translation_flux))
            .clamp(12.0, 320.0);
        self.dnaa = (0.72 * self.dnaa
            + 0.28 * (inventory.dnaa_activity + 4.0 * replication_flux * replisome_scale))
            .clamp(8.0, 256.0);
        self.ftsz = (0.70 * self.ftsz
            + 0.30
                * (inventory.ftsz_polymer
                    + 8.0 * membrane_flux
                    + 6.0 * constriction_flux
                    + 4.0 * translation_flux * ftsz_translation_scale))
            .clamp(12.0, 384.0);
    }

    fn initialize_surrogate_pool_diagnostics(&mut self) {
        let inventory = self.assembly_inventory();
        self.active_rnap = inventory.rnap_complexes.clamp(8.0, 256.0);
        self.active_ribosomes = inventory.ribosome_complexes.clamp(12.0, 320.0);
        self.dnaa = inventory.dnaa_activity.clamp(8.0, 256.0);
        self.ftsz = inventory.ftsz_polymer.clamp(12.0, 384.0);
    }

    fn apply_organism_data_initialization(&mut self) {
        let Some(organism) = self.organism_data.clone() else {
            return;
        };

        self.genome_bp = organism.chromosome_length_bp.max(1);
        self.replicated_bp = self.replicated_bp.min(self.genome_bp);
        if organism.geometry.radius_nm.is_finite() {
            self.radius_nm = organism.geometry.radius_nm.max(50.0);
            self.surface_area_nm2 = Self::surface_area_from_radius(self.radius_nm);
            self.volume_nm3 = Self::volume_from_radius(self.radius_nm);
        }

        let diagnostic_seeded = organism.pools.iter().any(|pool| {
            let name = pool.species.trim().to_lowercase();
            Self::molecule_pool_count(pool) > 0.0
                && (name.contains("ribosome")
                    || name.contains("rnap")
                    || name.contains("rna_polymerase")
                    || name.contains("dnaa")
                    || name.contains("ftsz"))
        });

        for pool in &organism.pools {
            self.apply_pool_seed(pool);
        }
        self.sync_from_lattice();
        if !diagnostic_seeded {
            self.initialize_surrogate_pool_diagnostics();
        }
    }

    fn unit_support_level(
        weights: WholeCellProcessWeights,
        energy_support: f32,
        translation_support: f32,
        nucleotide_support: f32,
        membrane_support: f32,
        localized_supply: f32,
    ) -> f32 {
        let total = weights.total();
        if total <= 1.0e-6 {
            return 1.0;
        }
        let translation_mix = weights.transcription + weights.translation;
        let nucleotide_mix = weights.replication + weights.segregation;
        let membrane_mix = weights.membrane + weights.constriction;
        let raw = (energy_support * weights.energy
            + translation_support * translation_mix
            + nucleotide_support * nucleotide_mix
            + membrane_support * membrane_mix)
            / total.max(1.0e-6);
        Self::finite_scale(raw * (0.85 + 0.15 * localized_supply), 1.0, 0.55, 1.55)
    }

    fn unit_inventory_scale(
        transcript_abundance: f32,
        protein_abundance: f32,
        gene_count: usize,
    ) -> f32 {
        let gene_scale = (gene_count.max(1) as f32).sqrt();
        let transcript_signal =
            Self::saturating_signal(transcript_abundance, 6.0 + 2.0 * gene_scale);
        let protein_signal = Self::saturating_signal(protein_abundance, 10.0 + 3.0 * gene_scale);
        (0.78 + 0.18 * transcript_signal + 0.24 * protein_signal).clamp(0.70, 1.45)
    }

    fn expression_lengths_for_operon(
        organism: &WholeCellOrganismSpec,
        genes: &[String],
    ) -> (f32, f32) {
        let mut total_nt = 0.0;
        let mut gene_count = 0usize;
        for gene_name in genes {
            if let Some(feature) = organism
                .genes
                .iter()
                .find(|feature| feature.gene == *gene_name)
            {
                let gene_nt = feature
                    .end_bp
                    .saturating_sub(feature.start_bp)
                    .saturating_add(1)
                    .max(90) as f32;
                total_nt += gene_nt;
                gene_count += 1;
            }
        }
        if gene_count == 0 {
            total_nt = 900.0 * genes.len().max(1) as f32;
        }
        let total_aa = (total_nt / 3.0).max(30.0);
        (total_nt.max(90.0), total_aa)
    }

    fn expression_lengths_for_gene(feature: &WholeCellGenomeFeature) -> (f32, f32) {
        let total_nt = feature
            .end_bp
            .saturating_sub(feature.start_bp)
            .saturating_add(1)
            .max(90) as f32;
        let total_aa = (total_nt / 3.0).max(30.0);
        (total_nt, total_aa)
    }

    fn normalize_expression_execution_state(unit: &mut WholeCellTranscriptionUnitState) {
        unit.transcription_length_nt = unit.transcription_length_nt.max(90.0);
        unit.translation_length_aa = unit.translation_length_aa.max(30.0);

        let transcript_pool_total = unit.nascent_transcript_abundance.max(0.0)
            + unit.mature_transcript_abundance.max(0.0)
            + unit.damaged_transcript_abundance.max(0.0);
        if transcript_pool_total <= 1.0e-6 && unit.transcript_abundance > 0.0 {
            unit.mature_transcript_abundance = unit.transcript_abundance.max(0.0);
        }

        let protein_pool_total = unit.nascent_protein_abundance.max(0.0)
            + unit.mature_protein_abundance.max(0.0)
            + unit.damaged_protein_abundance.max(0.0);
        if protein_pool_total <= 1.0e-6 && unit.protein_abundance > 0.0 {
            unit.mature_protein_abundance = unit.protein_abundance.max(0.0);
        }

        unit.transcription_progress_nt = unit
            .transcription_progress_nt
            .clamp(0.0, unit.transcription_length_nt.max(1.0));
        unit.translation_progress_aa = unit
            .translation_progress_aa
            .clamp(0.0, unit.translation_length_aa.max(1.0));
        unit.promoter_open_fraction = unit.promoter_open_fraction.clamp(0.0, 1.0);
        unit.active_rnap_occupancy = unit.active_rnap_occupancy.max(0.0);
        unit.active_ribosome_occupancy = unit.active_ribosome_occupancy.max(0.0);
        unit.nascent_transcript_abundance = unit.nascent_transcript_abundance.max(0.0);
        unit.mature_transcript_abundance = unit.mature_transcript_abundance.max(0.0);
        unit.damaged_transcript_abundance = unit.damaged_transcript_abundance.max(0.0);
        unit.nascent_protein_abundance = unit.nascent_protein_abundance.max(0.0);
        unit.mature_protein_abundance = unit.mature_protein_abundance.max(0.0);
        unit.damaged_protein_abundance = unit.damaged_protein_abundance.max(0.0);
        unit.transcript_abundance = (unit.nascent_transcript_abundance
            + unit.mature_transcript_abundance
            + unit.damaged_transcript_abundance)
            .clamp(0.0, 1024.0);
        unit.protein_abundance = (unit.nascent_protein_abundance
            + unit.mature_protein_abundance
            + unit.damaged_protein_abundance)
            .clamp(0.0, 2048.0);
    }

    fn subtract_expression_pool(pool: &mut f32, delta: f32) -> f32 {
        let available = pool.max(0.0);
        let removed = available.min(delta.max(0.0));
        *pool = (available - removed).max(0.0);
        removed
    }

    fn complex_assembly_signal_scale(signal: f32, mean_signal: f32, fallback: f32) -> f32 {
        if mean_signal <= 1.0e-6 {
            fallback
        } else {
            Self::finite_scale(0.84 + 0.32 * (signal / mean_signal), fallback, 0.60, 1.65)
        }
    }

    fn complex_channel_step(
        current: f32,
        target: f32,
        assembly_support: f32,
        degradation_pressure: f32,
        dt_scale: f32,
        max_value: f32,
    ) -> (f32, f32, f32) {
        let current = current.max(0.0);
        let target = target.max(0.0);
        let assembly_rate =
            (target - current).max(0.0) * (0.06 + 0.10 * assembly_support.clamp(0.55, 1.65));
        let degradation_rate = current * (0.005 + 0.018 * degradation_pressure.clamp(0.65, 1.80));
        let next = (current + dt_scale * (assembly_rate - degradation_rate)).clamp(0.0, max_value);
        (next, assembly_rate.max(0.0), degradation_rate.max(0.0))
    }

    fn prior_assembly_inventory(&self) -> WholeCellAssemblyInventory {
        let ctx = self.base_rule_context(0.0);
        let scalar = ctx.scalar();
        let atp_band_complexes = ATP_BAND_INVENTORY_RULE.evaluate(scalar);
        let ribosome_complexes = RIBOSOME_INVENTORY_RULE.evaluate(scalar);
        let rnap_complexes = RNAP_INVENTORY_RULE.evaluate(scalar);
        let replisome_complexes = REPLISOME_INVENTORY_RULE.evaluate(scalar);
        let membrane_complexes = MEMBRANE_INVENTORY_RULE.evaluate(scalar);
        let ftsz_polymer = FTSZ_POLYMER_RULE.evaluate(scalar);
        let dnaa_activity = DNAA_ACTIVITY_RULE.evaluate(scalar);
        WholeCellComplexAssemblyState {
            atp_band_complexes,
            ribosome_complexes,
            rnap_complexes,
            replisome_complexes,
            membrane_complexes,
            ftsz_polymer,
            dnaa_activity,
            atp_band_target: atp_band_complexes,
            ribosome_target: ribosome_complexes,
            rnap_target: rnap_complexes,
            replisome_target: replisome_complexes,
            membrane_target: membrane_complexes,
            ftsz_target: ftsz_polymer,
            dnaa_target: dnaa_activity,
            ..WholeCellComplexAssemblyState::default()
        }
    }

    fn asset_class_process_scale(
        &self,
        asset_class: crate::whole_cell_data::WholeCellAssetClass,
    ) -> f32 {
        match asset_class {
            crate::whole_cell_data::WholeCellAssetClass::Energy => {
                self.organism_expression.process_scales.energy
            }
            crate::whole_cell_data::WholeCellAssetClass::Translation => {
                self.organism_expression.process_scales.translation
            }
            crate::whole_cell_data::WholeCellAssetClass::Replication => {
                self.organism_expression.process_scales.replication
            }
            crate::whole_cell_data::WholeCellAssetClass::Segregation => {
                self.organism_expression.process_scales.segregation
            }
            crate::whole_cell_data::WholeCellAssetClass::Membrane => {
                self.organism_expression.process_scales.membrane
            }
            crate::whole_cell_data::WholeCellAssetClass::Constriction => {
                self.organism_expression.process_scales.constriction
            }
            crate::whole_cell_data::WholeCellAssetClass::QualityControl => Self::finite_scale(
                0.55 * self.organism_expression.process_scales.translation
                    + 0.45 * self.organism_expression.process_scales.energy,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Homeostasis => Self::finite_scale(
                0.55 * self.organism_expression.process_scales.transcription
                    + 0.45 * self.organism_expression.process_scales.membrane,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Generic => 1.0,
        }
    }

    fn asset_class_process_template(
        asset_class: crate::whole_cell_data::WholeCellAssetClass,
    ) -> WholeCellProcessWeights {
        match asset_class {
            crate::whole_cell_data::WholeCellAssetClass::Energy => WholeCellProcessWeights {
                energy: 1.0,
                membrane: 0.15,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Translation => WholeCellProcessWeights {
                translation: 1.0,
                transcription: 0.18,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Replication => WholeCellProcessWeights {
                replication: 1.0,
                segregation: 0.30,
                transcription: 0.12,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Segregation => WholeCellProcessWeights {
                segregation: 1.0,
                replication: 0.28,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Membrane => WholeCellProcessWeights {
                membrane: 1.0,
                energy: 0.18,
                constriction: 0.12,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Constriction => WholeCellProcessWeights {
                constriction: 1.0,
                membrane: 0.22,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::QualityControl => {
                WholeCellProcessWeights {
                    translation: 0.40,
                    energy: 0.35,
                    transcription: 0.18,
                    ..WholeCellProcessWeights::default()
                }
            }
            crate::whole_cell_data::WholeCellAssetClass::Homeostasis => WholeCellProcessWeights {
                transcription: 0.40,
                membrane: 0.28,
                energy: 0.18,
                ..WholeCellProcessWeights::default()
            },
            crate::whole_cell_data::WholeCellAssetClass::Generic => WholeCellProcessWeights {
                energy: 0.18,
                transcription: 0.16,
                translation: 0.16,
                replication: 0.16,
                segregation: 0.12,
                membrane: 0.12,
                constriction: 0.10,
            },
        }
    }

    fn registry_process_drive(&self) -> WholeCellProcessWeights {
        if !self.organism_species.is_empty() || !self.organism_reactions.is_empty() {
            let mut drive = WholeCellProcessWeights::default();

            for species in &self.organism_species {
                let class_weight = match species.species_class {
                    WholeCellSpeciesClass::Pool => 0.12,
                    WholeCellSpeciesClass::Rna => 0.32,
                    WholeCellSpeciesClass::Protein => 0.36,
                    WholeCellSpeciesClass::ComplexSubunitPool => 0.24,
                    WholeCellSpeciesClass::ComplexNucleationIntermediate => 0.28,
                    WholeCellSpeciesClass::ComplexElongationIntermediate => 0.30,
                    WholeCellSpeciesClass::ComplexMature => 0.34,
                };
                let abundance_weight = class_weight * species.count.max(0.0).sqrt();
                drive.add_weighted(
                    Self::asset_class_process_template(species.asset_class),
                    abundance_weight,
                );
                if matches!(species.species_class, WholeCellSpeciesClass::Rna) {
                    drive.transcription += 0.20 * abundance_weight;
                }
                if matches!(species.species_class, WholeCellSpeciesClass::Protein) {
                    drive.translation += 0.22 * abundance_weight;
                }
            }

            for reaction in &self.organism_reactions {
                let reaction_weight = reaction.current_flux.max(0.0)
                    * match reaction.reaction_class {
                        WholeCellReactionClass::PoolTransport => 0.55,
                        WholeCellReactionClass::LocalizedPoolTransfer => 0.60,
                        WholeCellReactionClass::LocalizedPoolTurnover => 0.38,
                        WholeCellReactionClass::MembranePatchTransfer => 0.72,
                        WholeCellReactionClass::MembranePatchTurnover => 0.44,
                        WholeCellReactionClass::Transcription => 1.00,
                        WholeCellReactionClass::Translation => 1.10,
                        WholeCellReactionClass::RnaDegradation => 0.46,
                        WholeCellReactionClass::ProteinDegradation => 0.50,
                        WholeCellReactionClass::StressResponse => 0.62,
                        WholeCellReactionClass::SubunitPoolFormation => 0.85,
                        WholeCellReactionClass::ComplexNucleation => 0.70,
                        WholeCellReactionClass::ComplexElongation => 0.75,
                        WholeCellReactionClass::ComplexMaturation => 0.78,
                        WholeCellReactionClass::ComplexRepair => 0.88,
                        WholeCellReactionClass::ComplexTurnover => 0.42,
                    };
                drive.add_weighted(
                    Self::asset_class_process_template(reaction.asset_class),
                    reaction_weight,
                );
                match reaction.reaction_class {
                    WholeCellReactionClass::PoolTransport => {
                        drive.energy += 0.08 * reaction_weight;
                    }
                    WholeCellReactionClass::LocalizedPoolTransfer => {
                        drive.energy += 0.06 * reaction_weight;
                    }
                    WholeCellReactionClass::LocalizedPoolTurnover => {
                        drive.energy += 0.03 * reaction_weight;
                    }
                    WholeCellReactionClass::MembranePatchTransfer => {
                        drive.membrane += 0.16 * reaction_weight;
                        drive.constriction += 0.04 * reaction_weight;
                    }
                    WholeCellReactionClass::MembranePatchTurnover => {
                        drive.membrane += 0.10 * reaction_weight;
                    }
                    WholeCellReactionClass::Transcription => {
                        drive.transcription += 0.28 * reaction_weight;
                    }
                    WholeCellReactionClass::Translation => {
                        drive.translation += 0.30 * reaction_weight;
                    }
                    WholeCellReactionClass::RnaDegradation => {
                        drive.transcription += 0.06 * reaction_weight;
                    }
                    WholeCellReactionClass::ProteinDegradation => {
                        drive.translation += 0.08 * reaction_weight;
                    }
                    WholeCellReactionClass::StressResponse => {
                        drive.energy += 0.10 * reaction_weight;
                        drive.transcription += 0.04 * reaction_weight;
                    }
                    WholeCellReactionClass::SubunitPoolFormation => {
                        drive.translation += 0.12 * reaction_weight;
                    }
                    WholeCellReactionClass::ComplexNucleation
                    | WholeCellReactionClass::ComplexElongation
                    | WholeCellReactionClass::ComplexMaturation
                    | WholeCellReactionClass::ComplexRepair => {
                        drive.replication += 0.02 * reaction_weight;
                        drive.constriction += 0.02 * reaction_weight;
                    }
                    WholeCellReactionClass::ComplexTurnover => {
                        drive.transcription += 0.04 * reaction_weight;
                    }
                }
            }

            return drive;
        }

        let Some(registry) = self.organism_process_registry() else {
            return WholeCellProcessWeights::default();
        };
        let mut drive = WholeCellProcessWeights::default();

        for species in registry.species {
            let class_weight = match species.species_class {
                crate::whole_cell_data::WholeCellSpeciesClass::Pool => 0.12,
                crate::whole_cell_data::WholeCellSpeciesClass::Rna => 0.32,
                crate::whole_cell_data::WholeCellSpeciesClass::Protein => 0.36,
                crate::whole_cell_data::WholeCellSpeciesClass::ComplexSubunitPool => 0.24,
                crate::whole_cell_data::WholeCellSpeciesClass::ComplexNucleationIntermediate => {
                    0.28
                }
                crate::whole_cell_data::WholeCellSpeciesClass::ComplexElongationIntermediate => {
                    0.30
                }
                crate::whole_cell_data::WholeCellSpeciesClass::ComplexMature => 0.34,
            };
            let abundance_weight = class_weight * species.basal_abundance.max(0.0).sqrt();
            drive.add_weighted(
                Self::asset_class_process_template(species.asset_class),
                abundance_weight,
            );
            if matches!(
                species.species_class,
                crate::whole_cell_data::WholeCellSpeciesClass::Rna
            ) {
                drive.transcription += 0.20 * abundance_weight;
            }
            if matches!(
                species.species_class,
                crate::whole_cell_data::WholeCellSpeciesClass::Protein
            ) {
                drive.translation += 0.22 * abundance_weight;
            }
        }

        for reaction in registry.reactions {
            let reaction_weight = reaction.nominal_rate.max(0.0)
                * match reaction.reaction_class {
                    crate::whole_cell_data::WholeCellReactionClass::PoolTransport => 0.55,
                    crate::whole_cell_data::WholeCellReactionClass::LocalizedPoolTransfer => 0.60,
                    crate::whole_cell_data::WholeCellReactionClass::LocalizedPoolTurnover => 0.38,
                    crate::whole_cell_data::WholeCellReactionClass::MembranePatchTransfer => 0.72,
                    crate::whole_cell_data::WholeCellReactionClass::MembranePatchTurnover => 0.44,
                    crate::whole_cell_data::WholeCellReactionClass::Transcription => 1.00,
                    crate::whole_cell_data::WholeCellReactionClass::Translation => 1.10,
                    crate::whole_cell_data::WholeCellReactionClass::RnaDegradation => 0.46,
                    crate::whole_cell_data::WholeCellReactionClass::ProteinDegradation => 0.50,
                    crate::whole_cell_data::WholeCellReactionClass::StressResponse => 0.62,
                    crate::whole_cell_data::WholeCellReactionClass::SubunitPoolFormation => 0.85,
                    crate::whole_cell_data::WholeCellReactionClass::ComplexNucleation => 0.70,
                    crate::whole_cell_data::WholeCellReactionClass::ComplexElongation => 0.75,
                    crate::whole_cell_data::WholeCellReactionClass::ComplexMaturation => 0.78,
                    crate::whole_cell_data::WholeCellReactionClass::ComplexRepair => 0.88,
                    crate::whole_cell_data::WholeCellReactionClass::ComplexTurnover => 0.42,
                };
            drive.add_weighted(
                Self::asset_class_process_template(reaction.asset_class),
                reaction_weight,
            );
            match reaction.reaction_class {
                crate::whole_cell_data::WholeCellReactionClass::PoolTransport => {
                    drive.energy += 0.08 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::LocalizedPoolTransfer => {
                    drive.energy += 0.06 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::LocalizedPoolTurnover => {
                    drive.energy += 0.03 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::MembranePatchTransfer => {
                    drive.membrane += 0.16 * reaction_weight;
                    drive.constriction += 0.04 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::MembranePatchTurnover => {
                    drive.membrane += 0.10 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::Transcription => {
                    drive.transcription += 0.28 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::Translation => {
                    drive.translation += 0.30 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::RnaDegradation => {
                    drive.transcription += 0.06 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::ProteinDegradation => {
                    drive.translation += 0.08 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::StressResponse => {
                    drive.energy += 0.10 * reaction_weight;
                    drive.transcription += 0.04 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::SubunitPoolFormation => {
                    drive.translation += 0.12 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::ComplexNucleation
                | crate::whole_cell_data::WholeCellReactionClass::ComplexElongation
                | crate::whole_cell_data::WholeCellReactionClass::ComplexMaturation
                | crate::whole_cell_data::WholeCellReactionClass::ComplexRepair => {
                    drive.replication += 0.02 * reaction_weight;
                    drive.constriction += 0.02 * reaction_weight;
                }
                crate::whole_cell_data::WholeCellReactionClass::ComplexTurnover => {
                    drive.transcription += 0.04 * reaction_weight;
                }
            }
        }

        drive
    }

    fn ensure_process_registry(&mut self) -> Option<WholeCellGenomeProcessRegistry> {
        if self.organism_process_registry.is_none() {
            self.organism_process_registry = self
                .organism_assets
                .as_ref()
                .map(compile_genome_process_registry);
        }
        self.organism_process_registry.clone()
    }

    fn rebuild_process_registry_from_assets(&mut self) -> bool {
        self.organism_process_registry = self
            .organism_assets
            .as_ref()
            .map(compile_genome_process_registry);
        if self.organism_process_registry.is_none() {
            self.organism_species.clear();
            self.organism_reactions.clear();
            return false;
        }
        self.organism_species.clear();
        self.organism_reactions.clear();
        self.initialize_runtime_process_state();
        true
    }

    fn initialize_runtime_process_state(&mut self) {
        let Some(registry) = self.ensure_process_registry() else {
            self.organism_species.clear();
            self.organism_reactions.clear();
            return;
        };
        self.organism_species = initialize_runtime_species_state(&registry);
        self.organism_reactions = initialize_runtime_reaction_state(&registry);
        self.sync_runtime_process_species(0.0);
        self.update_runtime_process_reactions(0.0, 0.0, 0.0);
    }

    fn species_runtime_count(&self, species_id: &str) -> Option<f32> {
        self.organism_species
            .iter()
            .find(|species| species.id == species_id)
            .map(|species| species.count)
    }

    fn asset_class_expression_support(
        expression: &WholeCellOrganismExpressionState,
        asset_class: crate::whole_cell_data::WholeCellAssetClass,
    ) -> f32 {
        match asset_class {
            crate::whole_cell_data::WholeCellAssetClass::Energy => expression.process_scales.energy,
            crate::whole_cell_data::WholeCellAssetClass::Translation => {
                expression.process_scales.translation
            }
            crate::whole_cell_data::WholeCellAssetClass::Replication => {
                expression.process_scales.replication
            }
            crate::whole_cell_data::WholeCellAssetClass::Segregation => {
                expression.process_scales.segregation
            }
            crate::whole_cell_data::WholeCellAssetClass::Membrane => {
                expression.process_scales.membrane
            }
            crate::whole_cell_data::WholeCellAssetClass::Constriction => {
                expression.process_scales.constriction
            }
            crate::whole_cell_data::WholeCellAssetClass::QualityControl => Self::finite_scale(
                0.55 * expression.process_scales.translation
                    + 0.45 * expression.process_scales.energy,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Homeostasis => Self::finite_scale(
                0.55 * expression.process_scales.transcription
                    + 0.45 * expression.process_scales.membrane,
                1.0,
                0.70,
                1.55,
            ),
            crate::whole_cell_data::WholeCellAssetClass::Generic => Self::finite_scale(
                0.18 * expression.process_scales.energy
                    + 0.16 * expression.process_scales.transcription
                    + 0.16 * expression.process_scales.translation
                    + 0.16 * expression.process_scales.replication
                    + 0.12 * expression.process_scales.segregation
                    + 0.12 * expression.process_scales.membrane
                    + 0.10 * expression.process_scales.constriction,
                1.0,
                0.70,
                1.45,
            ),
        }
    }

    fn runtime_species_upper_bound(species_class: WholeCellSpeciesClass) -> f32 {
        match species_class {
            WholeCellSpeciesClass::Pool => 4096.0,
            WholeCellSpeciesClass::Rna => 2048.0,
            WholeCellSpeciesClass::Protein => 4096.0,
            WholeCellSpeciesClass::ComplexSubunitPool => 2048.0,
            WholeCellSpeciesClass::ComplexNucleationIntermediate => 1024.0,
            WholeCellSpeciesClass::ComplexElongationIntermediate => 1024.0,
            WholeCellSpeciesClass::ComplexMature => 1024.0,
        }
    }

    fn spatial_scope_field(scope: WholeCellSpatialScope) -> Option<IntracellularSpatialField> {
        match scope {
            WholeCellSpatialScope::WellMixed => None,
            WholeCellSpatialScope::MembraneAdjacent => {
                Some(IntracellularSpatialField::MembraneAdjacency)
            }
            WholeCellSpatialScope::SeptumLocal => Some(IntracellularSpatialField::SeptumZone),
            WholeCellSpatialScope::NucleoidLocal => {
                Some(IntracellularSpatialField::NucleoidOccupancy)
            }
        }
    }

    fn patch_domain_field(domain: WholeCellPatchDomain) -> Option<IntracellularSpatialField> {
        match domain {
            WholeCellPatchDomain::Distributed => None,
            WholeCellPatchDomain::MembraneBand => Some(IntracellularSpatialField::MembraneBandZone),
            WholeCellPatchDomain::SeptumPatch => Some(IntracellularSpatialField::SeptumZone),
            WholeCellPatchDomain::PolarPatch => Some(IntracellularSpatialField::PoleZone),
            WholeCellPatchDomain::NucleoidTrack => {
                Some(IntracellularSpatialField::NucleoidOccupancy)
            }
        }
    }

    fn spatial_scope_index(scope: WholeCellSpatialScope) -> usize {
        match scope {
            WholeCellSpatialScope::WellMixed => 0,
            WholeCellSpatialScope::MembraneAdjacent => 1,
            WholeCellSpatialScope::SeptumLocal => 2,
            WholeCellSpatialScope::NucleoidLocal => 3,
        }
    }

    fn patch_domain_index(domain: WholeCellPatchDomain) -> usize {
        match domain {
            WholeCellPatchDomain::Distributed => 0,
            WholeCellPatchDomain::MembraneBand => 1,
            WholeCellPatchDomain::SeptumPatch => 2,
            WholeCellPatchDomain::PolarPatch => 3,
            WholeCellPatchDomain::NucleoidTrack => 4,
        }
    }

    fn lattice_species_for_bulk_field(field: WholeCellBulkField) -> Option<IntracellularSpecies> {
        match field {
            WholeCellBulkField::ATP => Some(IntracellularSpecies::ATP),
            WholeCellBulkField::AminoAcids => Some(IntracellularSpecies::AminoAcids),
            WholeCellBulkField::Nucleotides => Some(IntracellularSpecies::Nucleotides),
            WholeCellBulkField::MembranePrecursors => {
                Some(IntracellularSpecies::MembranePrecursors)
            }
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                None
            }
        }
    }

    fn bulk_field_species_scale(field: WholeCellBulkField) -> f32 {
        match field {
            WholeCellBulkField::ATP => 56.0,
            WholeCellBulkField::ADP => 48.0,
            WholeCellBulkField::Glucose => 40.0,
            WholeCellBulkField::Oxygen => 36.0,
            WholeCellBulkField::AminoAcids => 64.0,
            WholeCellBulkField::Nucleotides => 64.0,
            WholeCellBulkField::MembranePrecursors => 52.0,
        }
    }

    fn bulk_field_index(field: WholeCellBulkField) -> usize {
        match field {
            WholeCellBulkField::ATP => 0,
            WholeCellBulkField::ADP => 1,
            WholeCellBulkField::Glucose => 2,
            WholeCellBulkField::Oxygen => 3,
            WholeCellBulkField::AminoAcids => 4,
            WholeCellBulkField::Nucleotides => 5,
            WholeCellBulkField::MembranePrecursors => 6,
        }
    }

    fn bulk_field_concentration(
        field: WholeCellBulkField,
        atp_mm: f32,
        adp_mm: f32,
        glucose_mm: f32,
        oxygen_mm: f32,
        amino_acids_mm: f32,
        nucleotides_mm: f32,
        membrane_precursors_mm: f32,
    ) -> f32 {
        match field {
            WholeCellBulkField::ATP => atp_mm,
            WholeCellBulkField::ADP => adp_mm,
            WholeCellBulkField::Glucose => glucose_mm,
            WholeCellBulkField::Oxygen => oxygen_mm,
            WholeCellBulkField::AminoAcids => amino_acids_mm,
            WholeCellBulkField::Nucleotides => nucleotides_mm,
            WholeCellBulkField::MembranePrecursors => membrane_precursors_mm,
        }
    }

    fn compute_spatial_coupling_cache(&self) -> WholeCellSpatialCouplingCache {
        let mut bulk_concentrations = [[0.0; 7]; 4];
        let mut patch_bulk_concentrations = [[0.0; 7]; 5];
        let global_fields = [
            self.atp_mm,
            self.adp_mm,
            self.glucose_mm,
            self.oxygen_mm,
            self.amino_acids_mm,
            self.nucleotides_mm,
            self.membrane_precursors_mm,
        ];
        bulk_concentrations[Self::spatial_scope_index(WholeCellSpatialScope::WellMixed)] =
            global_fields;
        patch_bulk_concentrations[Self::patch_domain_index(WholeCellPatchDomain::Distributed)] =
            global_fields;
        for scope in [
            WholeCellSpatialScope::MembraneAdjacent,
            WholeCellSpatialScope::SeptumLocal,
            WholeCellSpatialScope::NucleoidLocal,
        ] {
            let scope_index = Self::spatial_scope_index(scope);
            bulk_concentrations[scope_index] = global_fields;
            let Some(field) = Self::spatial_scope_field(scope) else {
                continue;
            };
            for bulk_field in [
                WholeCellBulkField::ATP,
                WholeCellBulkField::AminoAcids,
                WholeCellBulkField::Nucleotides,
                WholeCellBulkField::MembranePrecursors,
            ] {
                if let Some(lattice_species) = Self::lattice_species_for_bulk_field(bulk_field) {
                    bulk_concentrations[scope_index][Self::bulk_field_index(bulk_field)] =
                        self.spatial_species_mean(lattice_species, field);
                }
            }
        }
        for domain in [
            WholeCellPatchDomain::MembraneBand,
            WholeCellPatchDomain::SeptumPatch,
            WholeCellPatchDomain::PolarPatch,
            WholeCellPatchDomain::NucleoidTrack,
        ] {
            let domain_index = Self::patch_domain_index(domain);
            patch_bulk_concentrations[domain_index] = global_fields;
            let Some(field) = Self::patch_domain_field(domain) else {
                continue;
            };
            for bulk_field in [
                WholeCellBulkField::ATP,
                WholeCellBulkField::AminoAcids,
                WholeCellBulkField::Nucleotides,
                WholeCellBulkField::MembranePrecursors,
            ] {
                if let Some(lattice_species) = Self::lattice_species_for_bulk_field(bulk_field) {
                    patch_bulk_concentrations[domain_index][Self::bulk_field_index(bulk_field)] =
                        self.spatial_species_mean(lattice_species, field);
                }
            }
        }

        let mut overlap = [[1.0; 4]; 4];
        for source_scope in [
            WholeCellSpatialScope::WellMixed,
            WholeCellSpatialScope::MembraneAdjacent,
            WholeCellSpatialScope::SeptumLocal,
            WholeCellSpatialScope::NucleoidLocal,
        ] {
            for target_scope in [
                WholeCellSpatialScope::WellMixed,
                WholeCellSpatialScope::MembraneAdjacent,
                WholeCellSpatialScope::SeptumLocal,
                WholeCellSpatialScope::NucleoidLocal,
            ] {
                let source_index = Self::spatial_scope_index(source_scope);
                let target_index = Self::spatial_scope_index(target_scope);
                if source_scope == WholeCellSpatialScope::WellMixed
                    || target_scope == WholeCellSpatialScope::WellMixed
                    || source_scope == target_scope
                {
                    overlap[source_index][target_index] = 1.0;
                    continue;
                }
                let Some(source_field) = Self::spatial_scope_field(source_scope) else {
                    continue;
                };
                let Some(target_field) = Self::spatial_scope_field(target_scope) else {
                    continue;
                };
                let source = self.spatial_fields.field_slice(source_field);
                let target = self.spatial_fields.field_slice(target_field);
                let source_total = source.iter().sum::<f32>().max(1.0e-6);
                let shared = source
                    .iter()
                    .zip(target.iter())
                    .map(|(lhs, rhs)| lhs.min(*rhs))
                    .sum::<f32>();
                overlap[source_index][target_index] = (shared / source_total).clamp(0.02, 1.0);
            }
        }
        let mut patch_overlap = [[1.0; 5]; 5];
        for source_domain in [
            WholeCellPatchDomain::Distributed,
            WholeCellPatchDomain::MembraneBand,
            WholeCellPatchDomain::SeptumPatch,
            WholeCellPatchDomain::PolarPatch,
            WholeCellPatchDomain::NucleoidTrack,
        ] {
            for target_domain in [
                WholeCellPatchDomain::Distributed,
                WholeCellPatchDomain::MembraneBand,
                WholeCellPatchDomain::SeptumPatch,
                WholeCellPatchDomain::PolarPatch,
                WholeCellPatchDomain::NucleoidTrack,
            ] {
                let source_index = Self::patch_domain_index(source_domain);
                let target_index = Self::patch_domain_index(target_domain);
                if source_domain == WholeCellPatchDomain::Distributed
                    || target_domain == WholeCellPatchDomain::Distributed
                    || source_domain == target_domain
                {
                    patch_overlap[source_index][target_index] = 1.0;
                    continue;
                }
                let Some(source_field) = Self::patch_domain_field(source_domain) else {
                    continue;
                };
                let Some(target_field) = Self::patch_domain_field(target_domain) else {
                    continue;
                };
                let source = self.spatial_fields.field_slice(source_field);
                let target = self.spatial_fields.field_slice(target_field);
                let source_total = source.iter().sum::<f32>().max(1.0e-6);
                let shared = source
                    .iter()
                    .zip(target.iter())
                    .map(|(lhs, rhs)| lhs.min(*rhs))
                    .sum::<f32>();
                patch_overlap[source_index][target_index] = (shared / source_total).clamp(0.02, 1.0);
            }
        }

        WholeCellSpatialCouplingCache {
            bulk_concentrations,
            overlap,
            patch_bulk_concentrations,
            patch_overlap,
        }
    }

    fn bulk_field_concentration_for_scope(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
    ) -> f32 {
        cache.bulk_concentrations[Self::spatial_scope_index(spatial_scope)]
            [Self::bulk_field_index(field)]
    }

    fn bulk_field_concentration_for_patch_domain(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        cache.patch_bulk_concentrations[Self::patch_domain_index(patch_domain)]
            [Self::bulk_field_index(field)]
    }

    fn bulk_field_concentration_for_locality(
        cache: &WholeCellSpatialCouplingCache,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        let scope_value = Self::bulk_field_concentration_for_scope(cache, field, spatial_scope);
        if patch_domain == WholeCellPatchDomain::Distributed {
            scope_value
        } else {
            let patch_value =
                Self::bulk_field_concentration_for_patch_domain(cache, field, patch_domain);
            if spatial_scope == WholeCellSpatialScope::WellMixed {
                patch_value
            } else {
                (scope_value.max(0.0) * patch_value.max(0.0)).sqrt()
            }
        }
    }

    fn localized_pool_signal_target(field: WholeCellBulkField) -> f32 {
        match field {
            WholeCellBulkField::ATP => 0.52,
            WholeCellBulkField::AminoAcids => 0.44,
            WholeCellBulkField::Nucleotides => 0.40,
            WholeCellBulkField::MembranePrecursors => 0.36,
            WholeCellBulkField::ADP | WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => {
                0.40
            }
        }
    }

    fn drive_field_mean(&self, field: WholeCellRdmeDriveField) -> f32 {
        let values = self.rdme_drive_fields.field_slice(field);
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

    fn drive_field_mean_for_locality(
        &self,
        field: WholeCellRdmeDriveField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        let scope_value = Self::spatial_scope_field(spatial_scope)
            .map(|scope_field| self.localized_drive_mean(field, scope_field))
            .unwrap_or_else(|| self.drive_field_mean(field));
        if patch_domain == WholeCellPatchDomain::Distributed {
            scope_value
        } else {
            let patch_value = Self::patch_domain_field(patch_domain)
                .map(|patch_field| self.localized_drive_mean(field, patch_field))
                .unwrap_or(scope_value);
            if spatial_scope == WholeCellSpatialScope::WellMixed {
                patch_value
            } else {
                (scope_value.max(0.0) * patch_value.max(0.0)).sqrt()
            }
        }
    }

    fn localized_pool_reaction_field(
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        use_products: bool,
    ) -> Option<(
        WholeCellBulkField,
        WholeCellSpatialScope,
        WholeCellPatchDomain,
    )> {
        let participants = if use_products {
            &reaction.products
        } else {
            &reaction.reactants
        };
        participants.iter().find_map(|participant| {
            let species = species_state.get(&participant.species_id)?;
            let field = species.bulk_field?;
            if species.spatial_scope == WholeCellSpatialScope::WellMixed
                && species.patch_domain == WholeCellPatchDomain::Distributed
            {
                return None;
            }
            Some((field, species.spatial_scope, species.patch_domain))
        })
    }

    fn localized_pool_transfer_hint(
        &self,
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let Some((field, spatial_scope, patch_domain)) =
            Self::localized_pool_reaction_field(reaction, species_state, true)
        else {
            return 1.0;
        };
        let local_level = Self::saturating_signal(
            Self::bulk_field_concentration_for_locality(cache, field, spatial_scope, patch_domain),
            Self::localized_pool_signal_target(field),
        );
        let crowding = self.drive_field_mean_for_locality(
            WholeCellRdmeDriveField::Crowding,
            spatial_scope,
            patch_domain,
        );
        let (demand, support, base, max_scale) = match field {
            WholeCellBulkField::ATP => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::AtpDemand,
                    spatial_scope,
                    patch_domain,
                ),
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::EnergySource,
                    spatial_scope,
                    patch_domain,
                ),
                0.16,
                2.6,
            ),
            WholeCellBulkField::AminoAcids => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::AminoDemand,
                    spatial_scope,
                    patch_domain,
                ),
                0.42
                    * self.drive_field_mean_for_locality(
                        WholeCellRdmeDriveField::EnergySource,
                        spatial_scope,
                        patch_domain,
                    ),
                0.14,
                2.4,
            ),
            WholeCellBulkField::Nucleotides => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::NucleotideDemand,
                    spatial_scope,
                    patch_domain,
                ),
                0.36
                    * self.drive_field_mean_for_locality(
                        WholeCellRdmeDriveField::EnergySource,
                        spatial_scope,
                        patch_domain,
                    ),
                0.15,
                2.5,
            ),
            WholeCellBulkField::MembranePrecursors => (
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::MembraneDemand,
                    spatial_scope,
                    patch_domain,
                ),
                self.drive_field_mean_for_locality(
                    WholeCellRdmeDriveField::MembraneSource,
                    spatial_scope,
                    patch_domain,
                ),
                0.16,
                2.6,
            ),
            WholeCellBulkField::ADP
            | WholeCellBulkField::Glucose
            | WholeCellBulkField::Oxygen => (0.0, 0.0, 0.10, 1.8),
        };
        Self::finite_scale(
            base + 0.56 * demand + 0.20 * support - 0.18 * crowding - 0.28 * local_level,
            1.0,
            0.04,
            max_scale,
        )
    }

    fn localized_pool_turnover_hint(
        &self,
        reaction: &WholeCellReactionRuntimeState,
        species_state: &HashMap<String, WholeCellSpeciesRuntimeState>,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let Some((field, spatial_scope, patch_domain)) =
            Self::localized_pool_reaction_field(reaction, species_state, false)
        else {
            return 1.0;
        };
        let local_level = Self::saturating_signal(
            Self::bulk_field_concentration_for_locality(cache, field, spatial_scope, patch_domain),
            Self::localized_pool_signal_target(field),
        );
        let crowding = self.drive_field_mean_for_locality(
            WholeCellRdmeDriveField::Crowding,
            spatial_scope,
            patch_domain,
        );
        let demand = match field {
            WholeCellBulkField::ATP => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::AtpDemand,
                spatial_scope,
                patch_domain,
            ),
            WholeCellBulkField::AminoAcids => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::AminoDemand,
                spatial_scope,
                patch_domain,
            ),
            WholeCellBulkField::Nucleotides => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::NucleotideDemand,
                spatial_scope,
                patch_domain,
            ),
            WholeCellBulkField::MembranePrecursors => self.drive_field_mean_for_locality(
                WholeCellRdmeDriveField::MembraneDemand,
                spatial_scope,
                patch_domain,
            ),
            WholeCellBulkField::ADP
            | WholeCellBulkField::Glucose
            | WholeCellBulkField::Oxygen => 0.0,
        };
        Self::finite_scale(
            0.12 + 0.34 * local_level + 0.18 * crowding - 0.30 * demand,
            1.0,
            0.03,
            2.2,
        )
    }

    fn pool_species_anchor(
        bulk_field: Option<WholeCellBulkField>,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        species_id: &str,
        basal_abundance: f32,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let anchor = if let Some(field) = bulk_field {
            Self::bulk_field_species_scale(field)
                * Self::bulk_field_concentration_for_locality(cache, field, spatial_scope, patch_domain)
                    .max(0.0)
        } else {
            let lowered = species_id.to_ascii_lowercase();
            if lowered.contains("atp") {
                56.0 * cache.bulk_concentrations[0][Self::bulk_field_index(WholeCellBulkField::ATP)]
                    .max(0.0)
            } else if lowered.contains("adp") {
                48.0 * cache.bulk_concentrations[0][Self::bulk_field_index(WholeCellBulkField::ADP)]
                    .max(0.0)
            } else if lowered.contains("glucose") {
                40.0 * cache.bulk_concentrations[0]
                    [Self::bulk_field_index(WholeCellBulkField::Glucose)]
                .max(0.0)
            } else if lowered.contains("oxygen") {
                36.0 * cache.bulk_concentrations[0]
                    [Self::bulk_field_index(WholeCellBulkField::Oxygen)]
                .max(0.0)
            } else if lowered.contains("amino") {
                64.0 * cache.bulk_concentrations[0]
                    [Self::bulk_field_index(WholeCellBulkField::AminoAcids)]
                .max(0.0)
            } else if lowered.contains("nucleotide") {
                64.0 * cache.bulk_concentrations[0]
                    [Self::bulk_field_index(WholeCellBulkField::Nucleotides)]
                .max(0.0)
            } else if lowered.contains("membrane") || lowered.contains("lipid") {
                52.0 * cache.bulk_concentrations[0]
                    [Self::bulk_field_index(WholeCellBulkField::MembranePrecursors)]
                .max(0.0)
            } else {
                basal_abundance.max(0.0)
            }
        };
        anchor.clamp(0.0, 4096.0)
    }

    fn spatial_scope_overlap(
        cache: &WholeCellSpatialCouplingCache,
        source_scope: WholeCellSpatialScope,
        target_scope: WholeCellSpatialScope,
    ) -> f32 {
        cache.overlap[Self::spatial_scope_index(source_scope)]
            [Self::spatial_scope_index(target_scope)]
    }

    fn patch_domain_overlap(
        cache: &WholeCellSpatialCouplingCache,
        source_patch_domain: WholeCellPatchDomain,
        target_patch_domain: WholeCellPatchDomain,
    ) -> f32 {
        cache.patch_overlap[Self::patch_domain_index(source_patch_domain)]
            [Self::patch_domain_index(target_patch_domain)]
    }

    fn effective_runtime_species_count_for_locality(
        species: &WholeCellSpeciesRuntimeState,
        target_scope: WholeCellSpatialScope,
        target_patch_domain: WholeCellPatchDomain,
        cache: &WholeCellSpatialCouplingCache,
    ) -> f32 {
        let preferred_scope = if target_scope == WholeCellSpatialScope::WellMixed {
            species.spatial_scope
        } else {
            target_scope
        };
        let preferred_patch_domain = if target_patch_domain == WholeCellPatchDomain::Distributed {
            species.patch_domain
        } else {
            target_patch_domain
        };
        if let Some(field) = species.bulk_field {
            return Self::bulk_field_species_scale(field)
                * Self::bulk_field_concentration_for_locality(
                    cache,
                    field,
                    preferred_scope,
                    preferred_patch_domain,
                )
                .max(0.0);
        }
        species.count.max(0.0)
            * Self::spatial_scope_overlap(cache, species.spatial_scope, preferred_scope)
            * Self::patch_domain_overlap(
                cache,
                species.patch_domain,
                preferred_patch_domain,
            )
    }

    fn locality_weights(
        &self,
        scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        uniform_weights: &[f32],
    ) -> Vec<f32> {
        let scope_field = Self::spatial_scope_field(scope);
        let patch_field = Self::patch_domain_field(patch_domain);
        match (scope_field, patch_field) {
            (None, None) => uniform_weights.to_vec(),
            (Some(field), None) | (None, Some(field)) => self.spatial_fields.field_slice(field).to_vec(),
            (Some(lhs), Some(rhs)) if lhs == rhs => self.spatial_fields.field_slice(lhs).to_vec(),
            (Some(lhs), Some(rhs)) => self
                .spatial_fields
                .field_slice(lhs)
                .iter()
                .zip(self.spatial_fields.field_slice(rhs).iter())
                .map(|(left, right)| (left.max(0.0) * right.max(0.0)).sqrt())
                .collect(),
        }
    }

    fn localized_drive_mean(
        &self,
        drive_field: WholeCellRdmeDriveField,
        spatial_field: IntracellularSpatialField,
    ) -> f32 {
        self.rdme_drive_fields
            .weighted_mean(drive_field, self.spatial_fields.field_slice(spatial_field))
    }

    fn accumulate_normalized_signal(target: &mut [f32], weights: &[f32], amplitude: f32) {
        if !amplitude.is_finite() || amplitude.abs() <= 1.0e-6 || target.len() != weights.len() {
            return;
        }
        let weight_sum = weights
            .iter()
            .map(|weight| weight.max(0.0))
            .sum::<f32>()
            .max(1.0e-6);
        let mean_scale = target.len() as f32 / weight_sum;
        for (value, weight) in target.iter_mut().zip(weights.iter()) {
            *value += amplitude * weight.max(0.0) * mean_scale;
        }
    }

    fn accumulate_patch_signal(
        target: &mut [f32],
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        center_x: usize,
        center_y: usize,
        center_z: usize,
        radius: usize,
        amplitude: f32,
    ) {
        if !amplitude.is_finite() || amplitude.abs() <= 1.0e-6 || target.is_empty() {
            return;
        }
        let radius = radius.max(1) as isize;
        let sigma = (radius as f32 * 0.65).max(0.75);
        let x_start = center_x.saturating_sub(radius as usize);
        let y_start = center_y.saturating_sub(radius as usize);
        let z_start = center_z.saturating_sub(radius as usize);
        let x_end = (center_x + radius as usize + 1).min(x_dim);
        let y_end = (center_y + radius as usize + 1).min(y_dim);
        let z_end = (center_z + radius as usize + 1).min(z_dim);
        let mut weighted_indices = Vec::new();
        let mut weight_total = 0.0f32;
        for z in z_start..z_end {
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let dx = x as isize - center_x as isize;
                    let dy = y as isize - center_y as isize;
                    let dz = z as isize - center_z as isize;
                    let dist2 = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist2 > (radius * radius) as f32 {
                        continue;
                    }
                    let weight = (-0.5 * dist2 / (sigma * sigma)).exp().max(1.0e-4);
                    let gid = z * y_dim * x_dim + y * x_dim + x;
                    weighted_indices.push((gid, weight));
                    weight_total += weight;
                }
            }
        }
        if weight_total <= 1.0e-6 {
            return;
        }
        let mean_scale = weighted_indices.len() as f32 / weight_total;
        for (gid, weight) in weighted_indices {
            target[gid] += amplitude * weight * mean_scale;
        }
    }

    fn species_crowding_weight(species_class: WholeCellSpeciesClass) -> f32 {
        match species_class {
            WholeCellSpeciesClass::Pool => 0.0,
            WholeCellSpeciesClass::Rna => 0.10,
            WholeCellSpeciesClass::Protein => 0.14,
            WholeCellSpeciesClass::ComplexSubunitPool => 0.18,
            WholeCellSpeciesClass::ComplexNucleationIntermediate => 0.22,
            WholeCellSpeciesClass::ComplexElongationIntermediate => 0.24,
            WholeCellSpeciesClass::ComplexMature => 0.28,
        }
    }

    fn reaction_flux_signal(reaction: &WholeCellReactionRuntimeState) -> f32 {
        let fallback = reaction.nominal_rate.max(0.0)
            * reaction.reactant_satisfaction.max(0.0)
            * reaction.catalyst_support.max(0.0)
            * 0.08;
        reaction.current_flux.max(fallback).clamp(0.0, 12.0).sqrt().clamp(0.0, 2.5)
    }

    fn clamp_signal_field(values: &mut [f32], max_value: f32) {
        for value in values {
            *value = value.clamp(0.0, max_value);
        }
    }

    fn refresh_rdme_drive_fields(&mut self) {
        let total_voxels = self.lattice.total_voxels();
        if total_voxels == 0 {
            return;
        }

        let uniform_weights = vec![1.0; total_voxels];
        let mut energy_source = vec![0.0; total_voxels];
        let mut atp_demand = vec![0.0; total_voxels];
        let mut amino_demand = vec![0.0; total_voxels];
        let mut nucleotide_demand = vec![0.0; total_voxels];
        let mut membrane_source = vec![0.0; total_voxels];
        let mut membrane_demand = vec![0.0; total_voxels];
        let mut crowding = vec![0.0; total_voxels];

        let species_by_id = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species))
            .collect::<HashMap<_, _>>();

        for species in &self.organism_species {
            let weights = self.locality_weights(
                species.spatial_scope,
                species.patch_domain,
                &uniform_weights,
            );
            let anchor = species
                .anchor_count
                .max(species.basal_abundance)
                .max(1.0);
            let abundance_signal = (species.count.max(0.0) / anchor).sqrt().clamp(0.0, 2.5);
            let crowding_signal =
                abundance_signal * Self::species_crowding_weight(species.species_class);
            if crowding_signal > 0.0 {
                Self::accumulate_normalized_signal(&mut crowding, &weights, crowding_signal);
            }
            if species.spatial_scope != WholeCellSpatialScope::WellMixed
                && species.species_class != WholeCellSpeciesClass::Pool
            {
                match species.asset_class {
                    WholeCellAssetClass::Energy => {
                        Self::accumulate_normalized_signal(
                            &mut energy_source,
                            &weights,
                            0.26 * abundance_signal,
                        );
                    }
                    WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction => {
                        Self::accumulate_normalized_signal(
                            &mut membrane_source,
                            &weights,
                            0.22 * abundance_signal,
                        );
                    }
                    WholeCellAssetClass::Replication => {
                        Self::accumulate_normalized_signal(
                            &mut nucleotide_demand,
                            &weights,
                            0.16 * abundance_signal,
                        );
                    }
                    _ => {}
                }
            }
        }

        for reaction in &self.organism_reactions {
            let weights = self.locality_weights(
                reaction.spatial_scope,
                reaction.patch_domain,
                &uniform_weights,
            );
            let flux_signal = Self::reaction_flux_signal(reaction);
            if flux_signal <= 0.0 {
                continue;
            }
            if reaction.spatial_scope != WholeCellSpatialScope::WellMixed {
                Self::accumulate_normalized_signal(&mut crowding, &weights, 0.04 * flux_signal);
            }
            for participant in &reaction.reactants {
                let Some(species) = species_by_id.get(&participant.species_id) else {
                    continue;
                };
                let amplitude = flux_signal * participant.stoichiometry.max(0.0);
                match species.bulk_field {
                    Some(WholeCellBulkField::ATP) => {
                        Self::accumulate_normalized_signal(&mut atp_demand, &weights, amplitude);
                    }
                    Some(WholeCellBulkField::AminoAcids) => {
                        Self::accumulate_normalized_signal(&mut amino_demand, &weights, amplitude);
                    }
                    Some(WholeCellBulkField::Nucleotides) => Self::accumulate_normalized_signal(
                        &mut nucleotide_demand,
                        &weights,
                        amplitude,
                    ),
                    Some(WholeCellBulkField::MembranePrecursors) => Self::accumulate_normalized_signal(
                        &mut membrane_demand,
                        &weights,
                        amplitude,
                    ),
                    _ => {}
                }
            }
            for participant in &reaction.products {
                let Some(species) = species_by_id.get(&participant.species_id) else {
                    continue;
                };
                let amplitude = flux_signal * participant.stoichiometry.max(0.0);
                match species.bulk_field {
                    Some(WholeCellBulkField::ATP) => Self::accumulate_normalized_signal(
                        &mut energy_source,
                        &weights,
                        0.85 * amplitude,
                    ),
                    Some(WholeCellBulkField::MembranePrecursors) => Self::accumulate_normalized_signal(
                        &mut membrane_source,
                        &weights,
                        amplitude,
                    ),
                    _ => {}
                }
            }
        }

        for report in &self.chemistry_site_reports {
            let patch_scale = report.localization_score.clamp(0.0, 1.5);
            if patch_scale <= 0.0 {
                continue;
            }
            let radius = report.patch_radius.max(1);
            let energy_source_signal =
                patch_scale * (0.22 * report.atp_support.max(0.0) + 0.28 * report.mean_atp_flux.max(0.0));
            let atp_demand_signal = patch_scale * report.energy_draw.max(0.0);
            let amino_demand_signal =
                patch_scale * (0.45 * report.substrate_draw.max(0.0) + 0.55 * report.biosynthetic_draw.max(0.0));
            let nucleotide_signal =
                patch_scale * (0.60 * report.biosynthetic_draw.max(0.0) + 0.35 * (1.0 - report.nucleotide_support).max(0.0));
            let membrane_source_signal =
                patch_scale * (0.40 * report.membrane_support.max(0.0) + 0.24 * report.assembly_stability.max(0.0));
            let membrane_demand_signal =
                patch_scale * (0.38 * report.assembly_turnover.max(0.0) + 0.22 * report.byproduct_load.max(0.0));
            let crowding_signal = patch_scale
                * (0.25 * report.assembly_occupancy.max(0.0)
                    + 0.20 * report.byproduct_load.max(0.0)
                    + 0.30 * (1.0 - report.demand_satisfaction).max(0.0));

            Self::accumulate_patch_signal(
                &mut energy_source,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                energy_source_signal,
            );
            Self::accumulate_patch_signal(
                &mut atp_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                atp_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut amino_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                amino_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut nucleotide_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                nucleotide_signal,
            );
            Self::accumulate_patch_signal(
                &mut membrane_source,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                membrane_source_signal,
            );
            Self::accumulate_patch_signal(
                &mut membrane_demand,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                membrane_demand_signal,
            );
            Self::accumulate_patch_signal(
                &mut crowding,
                self.lattice.x_dim,
                self.lattice.y_dim,
                self.lattice.z_dim,
                report.site_x,
                report.site_y,
                report.site_z,
                radius,
                crowding_signal,
            );
        }

        Self::clamp_signal_field(&mut energy_source, 2.5);
        Self::clamp_signal_field(&mut atp_demand, 2.5);
        Self::clamp_signal_field(&mut amino_demand, 2.5);
        Self::clamp_signal_field(&mut nucleotide_demand, 2.5);
        Self::clamp_signal_field(&mut membrane_source, 2.5);
        Self::clamp_signal_field(&mut membrane_demand, 2.5);
        Self::clamp_signal_field(&mut crowding, 1.6);

        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::EnergySource, &energy_source);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::AtpDemand, &atp_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::AminoDemand, &amino_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::NucleotideDemand, &nucleotide_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::MembraneSource, &membrane_source);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::MembraneDemand, &membrane_demand);
        let _ = self
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::Crowding, &crowding);
    }

    fn apply_bulk_field_delta(
        &mut self,
        field: WholeCellBulkField,
        spatial_scope: WholeCellSpatialScope,
        patch_domain: WholeCellPatchDomain,
        delta_count: f32,
    ) -> bool {
        if !delta_count.is_finite() || delta_count.abs() <= 1.0e-6 {
            return false;
        }
        let delta_mm = delta_count / Self::bulk_field_species_scale(field).max(1.0);
        if !delta_mm.is_finite() || delta_mm.abs() <= 1.0e-8 {
            return false;
        }
        let uniform_weights = vec![1.0; self.lattice.total_voxels()];
        let weights = self.locality_weights(spatial_scope, patch_domain, &uniform_weights);
        match field {
            WholeCellBulkField::ATP => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::ATP, delta_mm);
                } else {
                    self.lattice
                        .apply_weighted_delta(IntracellularSpecies::ATP, delta_mm, &weights);
                }
                self.atp_mm = (self.atp_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::AminoAcids => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::AminoAcids, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::AminoAcids,
                        delta_mm,
                        &weights,
                    );
                }
                self.amino_acids_mm = (self.amino_acids_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::Nucleotides => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::Nucleotides, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::Nucleotides,
                        delta_mm,
                        &weights,
                    );
                }
                self.nucleotides_mm = (self.nucleotides_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::MembranePrecursors => {
                if spatial_scope == WholeCellSpatialScope::WellMixed
                    && patch_domain == WholeCellPatchDomain::Distributed
                {
                    self.lattice
                        .apply_uniform_delta(IntracellularSpecies::MembranePrecursors, delta_mm);
                } else {
                    self.lattice.apply_weighted_delta(
                        IntracellularSpecies::MembranePrecursors,
                        delta_mm,
                        &weights,
                    );
                }
                self.membrane_precursors_mm = (self.membrane_precursors_mm + delta_mm).max(0.0);
                true
            }
            WholeCellBulkField::ADP => {
                self.adp_mm = (self.adp_mm + delta_mm).max(0.0);
                false
            }
            WholeCellBulkField::Glucose => {
                self.glucose_mm = (self.glucose_mm + delta_mm).max(0.0);
                false
            }
            WholeCellBulkField::Oxygen => {
                self.oxygen_mm = (self.oxygen_mm + delta_mm).max(0.0);
                false
            }
        }
    }

    fn apply_operon_inventory_delta(
        &mut self,
        operon: &str,
        transcript_delta: f32,
        protein_delta: f32,
        dt_scale: f32,
    ) -> bool {
        let Some(unit) = self
            .organism_expression
            .transcription_units
            .iter_mut()
            .find(|unit| unit.name == operon)
        else {
            return false;
        };
        Self::normalize_expression_execution_state(unit);
        if transcript_delta.is_finite() && transcript_delta.abs() > 1.0e-6 {
            if transcript_delta >= 0.0 {
                unit.mature_transcript_abundance =
                    (unit.mature_transcript_abundance + transcript_delta).clamp(0.0, 1024.0);
            } else {
                let mut remaining = -transcript_delta;
                remaining -= Self::subtract_expression_pool(
                    &mut unit.damaged_transcript_abundance,
                    remaining,
                );
                remaining -= Self::subtract_expression_pool(
                    &mut unit.nascent_transcript_abundance,
                    remaining,
                );
                let _ = Self::subtract_expression_pool(
                    &mut unit.mature_transcript_abundance,
                    remaining,
                );
            }
            if dt_scale > 0.0 {
                if transcript_delta >= 0.0 {
                    unit.transcript_synthesis_rate += transcript_delta / dt_scale;
                } else {
                    unit.transcript_turnover_rate += (-transcript_delta) / dt_scale;
                }
            }
        }
        if protein_delta.is_finite() && protein_delta.abs() > 1.0e-6 {
            if protein_delta >= 0.0 {
                unit.mature_protein_abundance =
                    (unit.mature_protein_abundance + protein_delta).clamp(0.0, 2048.0);
            } else {
                let mut remaining = -protein_delta;
                remaining -=
                    Self::subtract_expression_pool(&mut unit.damaged_protein_abundance, remaining);
                remaining -=
                    Self::subtract_expression_pool(&mut unit.nascent_protein_abundance, remaining);
                let _ =
                    Self::subtract_expression_pool(&mut unit.mature_protein_abundance, remaining);
            }
            if dt_scale > 0.0 {
                if protein_delta >= 0.0 {
                    unit.protein_synthesis_rate += protein_delta / dt_scale;
                } else {
                    unit.protein_turnover_rate += (-protein_delta) / dt_scale;
                }
            }
        }
        Self::normalize_expression_execution_state(unit);
        true
    }

    fn refresh_expression_inventory_totals(&mut self) {
        for unit in &mut self.organism_expression.transcription_units {
            Self::normalize_expression_execution_state(unit);
        }
        self.organism_expression.total_transcript_abundance = self
            .organism_expression
            .transcription_units
            .iter()
            .map(|unit| unit.transcript_abundance.max(0.0))
            .sum::<f32>();
        self.organism_expression.total_protein_abundance = self
            .organism_expression
            .transcription_units
            .iter()
            .map(|unit| unit.protein_abundance.max(0.0))
            .sum::<f32>();
    }

    fn apply_named_complex_species_delta(
        &mut self,
        complex_id: &str,
        species_class: WholeCellSpeciesClass,
        delta: f32,
    ) -> bool {
        let Some(state) = self
            .named_complexes
            .iter_mut()
            .find(|state| state.id == complex_id)
        else {
            return false;
        };
        match species_class {
            WholeCellSpeciesClass::ComplexSubunitPool => {
                state.subunit_pool = (state.subunit_pool + delta).clamp(0.0, 2048.0);
            }
            WholeCellSpeciesClass::ComplexNucleationIntermediate => {
                state.nucleation_intermediate =
                    (state.nucleation_intermediate + delta).clamp(0.0, 512.0);
            }
            WholeCellSpeciesClass::ComplexElongationIntermediate => {
                state.elongation_intermediate =
                    (state.elongation_intermediate + delta).clamp(0.0, 512.0);
            }
            WholeCellSpeciesClass::ComplexMature => {
                state.abundance = (state.abundance + delta).clamp(0.0, 512.0);
            }
            _ => return false,
        }
        true
    }

    fn sync_runtime_process_species(&mut self, dt: f32) {
        if self.organism_species.is_empty() {
            if self.organism_assets.is_some() {
                self.initialize_runtime_process_state();
            }
            return;
        }
        let Some(assets) = self.organism_assets.clone() else {
            self.organism_species.clear();
            return;
        };

        let dt_scale = if dt > 0.0 {
            (dt / self.config.dt_ms.max(0.05)).clamp(0.25, 6.0)
        } else {
            0.0
        };
        let expression = self.organism_expression.clone();
        let rna_totals = assets.rnas.iter().fold(HashMap::new(), |mut acc, rna| {
            *acc.entry(rna.operon.clone()).or_insert(0.0) += rna.basal_abundance.max(0.0);
            acc
        });
        let protein_totals = assets
            .proteins
            .iter()
            .fold(HashMap::new(), |mut acc, protein| {
                *acc.entry(protein.operon.clone()).or_insert(0.0) +=
                    protein.basal_abundance.max(0.0);
                acc
            });
        let unit_transcripts = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), unit.transcript_abundance))
            .collect::<HashMap<_, _>>();
        let unit_proteins = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), unit.protein_abundance))
            .collect::<HashMap<_, _>>();
        let named_complexes = self
            .named_complexes
            .iter()
            .map(|state| (state.id.clone(), state.clone()))
            .collect::<HashMap<_, _>>();
        let spatial_cache = self.compute_spatial_coupling_cache();

        for species in &mut self.organism_species {
            let anchor = match species.species_class {
                WholeCellSpeciesClass::Pool => Self::pool_species_anchor(
                    species.bulk_field,
                    species.spatial_scope,
                    species.patch_domain,
                    &species.id,
                    species.basal_abundance,
                    &spatial_cache,
                ),
                WholeCellSpeciesClass::Rna => {
                    if let Some(operon) = species.operon.as_ref() {
                        let total = rna_totals
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.01))
                            .max(0.01);
                        let unit_total = unit_transcripts
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.0));
                        (unit_total * species.basal_abundance.max(0.01) / total).clamp(0.0, 2048.0)
                    } else {
                        species.basal_abundance.max(0.0)
                    }
                }
                WholeCellSpeciesClass::Protein => {
                    if let Some(operon) = species.operon.as_ref() {
                        let total = protein_totals
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.01))
                            .max(0.01);
                        let unit_total = unit_proteins
                            .get(operon)
                            .copied()
                            .unwrap_or(species.basal_abundance.max(0.0));
                        (unit_total * species.basal_abundance.max(0.01) / total).clamp(0.0, 4096.0)
                    } else {
                        species.basal_abundance.max(0.0)
                    }
                }
                WholeCellSpeciesClass::ComplexSubunitPool => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.subunit_pool)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexNucleationIntermediate => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.nucleation_intermediate)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexElongationIntermediate => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.elongation_intermediate)
                    .unwrap_or(species.basal_abundance.max(0.0)),
                WholeCellSpeciesClass::ComplexMature => species
                    .parent_complex
                    .as_ref()
                    .and_then(|id| named_complexes.get(id))
                    .map(|state| state.abundance)
                    .unwrap_or(species.basal_abundance.max(0.0)),
            };
            let previous = species.count.max(0.0);
            let support = Self::asset_class_expression_support(&expression, species.asset_class);
            let next = if dt_scale <= 0.0 {
                anchor
            } else {
                let coupling = ((0.16 + 0.08 * support) * dt_scale).clamp(0.08, 0.95);
                previous + coupling * (anchor - previous)
            }
            .clamp(
                0.0,
                Self::runtime_species_upper_bound(species.species_class),
            );
            species.anchor_count = anchor;
            if dt_scale > 0.0 {
                let delta = next - previous;
                species.synthesis_rate = (delta.max(0.0) / dt_scale).max(0.0);
                species.turnover_rate = ((-delta).max(0.0) / dt_scale).max(0.0);
            } else {
                species.synthesis_rate = 0.0;
                species.turnover_rate = 0.0;
            }
            species.count = next;
        }
    }

    fn update_runtime_process_reactions(
        &mut self,
        dt: f32,
        transcription_flux: f32,
        translation_flux: f32,
    ) {
        if self.organism_reactions.is_empty() {
            if self.organism_assets.is_some() {
                self.initialize_runtime_process_state();
            }
            return;
        }
        let dt_scale = if dt > 0.0 {
            (dt / self.config.dt_ms.max(0.05)).clamp(0.25, 6.0)
        } else {
            0.0
        };
        let expression = self.organism_expression.clone();
        let effective_load = self.effective_metabolic_load();
        let species_counts = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.count))
            .collect::<HashMap<_, _>>();
        let species_state = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.clone()))
            .collect::<HashMap<_, _>>();
        let species_anchors = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.anchor_count))
            .collect::<HashMap<_, _>>();
        let species_basal = self
            .organism_species
            .iter()
            .map(|species| (species.id.clone(), species.basal_abundance))
            .collect::<HashMap<_, _>>();
        let rna_operon_basal_totals = self.organism_species.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, f32>, species| {
                if species.species_class == WholeCellSpeciesClass::Rna {
                    if let Some(operon) = species.operon.as_ref() {
                        *acc.entry(operon.clone()).or_insert(0.0) +=
                            species.basal_abundance.max(0.0);
                    }
                }
                acc
            },
        );
        let protein_operon_basal_totals = self.organism_species.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, f32>, species| {
                if species.species_class == WholeCellSpeciesClass::Protein {
                    if let Some(operon) = species.operon.as_ref() {
                        *acc.entry(operon.clone()).or_insert(0.0) +=
                            species.basal_abundance.max(0.0);
                    }
                }
                acc
            },
        );
        let unit_state_by_operon = expression
            .transcription_units
            .iter()
            .map(|unit| (unit.name.clone(), (unit.support_level, unit.stress_penalty)))
            .collect::<HashMap<_, _>>();
        let spatial_cache = self.compute_spatial_coupling_cache();
        let mut species_deltas: HashMap<String, f32> = HashMap::new();
        let mut bulk_field_deltas: HashMap<
            (WholeCellBulkField, WholeCellSpatialScope, WholeCellPatchDomain),
            f32,
        > = HashMap::new();
        let mut stress_relief: HashMap<String, f32> = HashMap::new();
        let mut metabolic_load_relief = 0.0f32;
        let band_precursors = Self::saturating_signal(
            self.localized_membrane_band_precursor_pool_mm(),
            0.45,
        );
        let pole_precursors =
            Self::saturating_signal(self.localized_polar_precursor_pool_mm(), 0.35);
        let septum_precursors = Self::saturating_signal(
            self.spatial_species_mean(
                IntracellularSpecies::MembranePrecursors,
                IntracellularSpatialField::SeptumZone,
            ),
            0.40,
        );
        let band_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::PoleZone,
        );
        let septum_source = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneSource,
            IntracellularSpatialField::SeptumZone,
        );
        let band_demand = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneDemand,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_demand = self.localized_drive_mean(
            WholeCellRdmeDriveField::MembraneDemand,
            IntracellularSpatialField::PoleZone,
        );
        let septum_demand = self.localized_drive_mean(
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
        let patch_transfer_hints = [
            1.0,
            Self::finite_scale(
                0.22
                    + 0.55 * self.membrane_division_state.band_turnover_pressure
                    + 0.28 * band_demand
                    + 0.20 * band_source
                    + 0.18 * expression.membrane_support
                    - 0.16 * band_crowding
                    - 0.22 * band_precursors,
                1.0,
                0.05,
                2.6,
            ),
            Self::finite_scale(
                0.26
                    + 0.58 * self.membrane_division_state.septum_turnover_pressure
                    + 0.30 * septum_demand
                    + 0.20 * septum_source
                    + 0.20 * expression.process_scales.constriction
                    - 0.15 * septum_crowding
                    - 0.18 * septum_precursors,
                1.0,
                0.05,
                2.8,
            ),
            Self::finite_scale(
                0.18
                    + 0.42 * self.membrane_division_state.pole_turnover_pressure
                    + 0.24 * pole_demand
                    + 0.16 * pole_source
                    + 0.16 * expression.membrane_support
                    - 0.12 * pole_crowding
                    - 0.20 * pole_precursors,
                1.0,
                0.05,
                2.4,
            ),
            0.25,
        ];
        let patch_turnover_hints = [
            1.0,
            Self::finite_scale(
                0.18
                    + 0.48 * self.membrane_division_state.band_turnover_pressure
                    + 0.18 * band_crowding
                    - 0.20 * band_source
                    - 0.16 * band_precursors,
                1.0,
                0.05,
                2.2,
            ),
            Self::finite_scale(
                0.22
                    + 0.54 * self.membrane_division_state.septum_turnover_pressure
                    + 0.18 * septum_crowding
                    - 0.22 * septum_source
                    - 0.14 * septum_precursors,
                1.0,
                0.05,
                2.4,
            ),
            Self::finite_scale(
                0.16
                    + 0.40 * self.membrane_division_state.pole_turnover_pressure
                    + 0.14 * pole_crowding
                    - 0.18 * pole_source
                    - 0.16 * pole_precursors,
                1.0,
                0.05,
                2.0,
            ),
            0.18,
        ];

        for reaction_index in 0..self.organism_reactions.len() {
            let reaction = &self.organism_reactions[reaction_index];
            let reaction_scope = reaction.spatial_scope;
            let reaction_patch_domain = reaction.patch_domain;
            let reactant_satisfaction = if reaction.reactants.is_empty() {
                1.0
            } else {
                reaction
                    .reactants
                    .iter()
                    .map(|participant| {
                        let count = species_state
                            .get(&participant.species_id)
                            .map(|species| {
                                Self::effective_runtime_species_count_for_locality(
                                    species,
                                    reaction_scope,
                                    reaction_patch_domain,
                                    &spatial_cache,
                                )
                            })
                            .or_else(|| species_counts.get(&participant.species_id).copied())
                            .unwrap_or(0.0);
                        Self::saturating_signal(
                            count,
                            1.0 + 2.0 * participant.stoichiometry.max(0.25),
                        )
                    })
                    .fold(1.0, f32::min)
            };
            let catalyst_support = reaction
                .catalyst
                .as_ref()
                .map(|species_id| {
                    let count = species_state
                        .get(species_id)
                        .map(|species| {
                            Self::effective_runtime_species_count_for_locality(
                                species,
                                reaction_scope,
                                reaction_patch_domain,
                                &spatial_cache,
                            )
                        })
                        .or_else(|| species_counts.get(species_id).copied())
                        .unwrap_or(0.0);
                    Self::saturating_signal(count, 4.0)
                })
                .unwrap_or(1.0);
            let asset_support =
                Self::asset_class_expression_support(&expression, reaction.asset_class);
            let external_hint = match reaction.reaction_class {
                WholeCellReactionClass::PoolTransport => {
                    let pool_deficit = reaction
                        .products
                        .iter()
                        .map(|participant| {
                            let count = species_counts
                                .get(&participant.species_id)
                                .copied()
                                .unwrap_or(0.0);
                            let target = species_anchors
                                .get(&participant.species_id)
                                .copied()
                                .or_else(|| species_basal.get(&participant.species_id).copied())
                                .unwrap_or(1.0)
                                .max(1.0);
                            ((target - count).max(0.0) / target).clamp(0.0, 1.0)
                        })
                        .fold(0.0, f32::max);
                    Self::finite_scale(
                        0.12 + 1.65 * pool_deficit + 0.14 * expression.energy_support,
                        1.0,
                        0.05,
                        2.5,
                    )
                }
                WholeCellReactionClass::LocalizedPoolTransfer => {
                    self.localized_pool_transfer_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::LocalizedPoolTurnover => {
                    self.localized_pool_turnover_hint(reaction, &species_state, &spatial_cache)
                }
                WholeCellReactionClass::MembranePatchTransfer => patch_transfer_hints
                    [Self::patch_domain_index(reaction_patch_domain)],
                WholeCellReactionClass::MembranePatchTurnover => patch_turnover_hints
                    [Self::patch_domain_index(reaction_patch_domain)],
                WholeCellReactionClass::Transcription => {
                    Self::finite_scale(1.0 + 0.12 * transcription_flux.max(0.0), 1.0, 0.75, 2.5)
                }
                WholeCellReactionClass::Translation => {
                    Self::finite_scale(1.0 + 0.12 * translation_flux.max(0.0), 1.0, 0.75, 2.5)
                }
                WholeCellReactionClass::RnaDegradation => Self::finite_scale(
                    0.72 + 0.24 * expression.crowding_penalty.max(0.0)
                        + 0.12 * (1.0 - expression.process_scales.transcription).max(0.0),
                    1.0,
                    0.40,
                    2.0,
                ),
                WholeCellReactionClass::ProteinDegradation => Self::finite_scale(
                    0.68 + 0.28 * expression.crowding_penalty.max(0.0)
                        + 0.14 * (1.0 - expression.translation_support).max(0.0),
                    1.0,
                    0.35,
                    2.2,
                ),
                WholeCellReactionClass::StressResponse => {
                    let (support_level, stress_penalty) = reaction
                        .operon
                        .as_ref()
                        .and_then(|operon| unit_state_by_operon.get(operon).copied())
                        .unwrap_or((1.0, 1.0));
                    let stress_signal = (stress_penalty - 1.0).max(0.0);
                    Self::finite_scale(
                        0.08 + 1.05 * stress_signal
                            + 0.18 * (1.0 - support_level).max(0.0)
                            + 0.12 * (effective_load - 1.0).max(0.0),
                        1.0,
                        0.0,
                        2.4,
                    )
                }
                WholeCellReactionClass::SubunitPoolFormation => {
                    Self::finite_scale(0.9 + 0.08 * translation_flux.max(0.0), 1.0, 0.70, 2.0)
                }
                WholeCellReactionClass::ComplexNucleation
                | WholeCellReactionClass::ComplexElongation
                | WholeCellReactionClass::ComplexMaturation => Self::finite_scale(
                    0.78 + 0.22 * expression.translation_support
                        + 0.12 * expression.membrane_support,
                    1.0,
                    0.70,
                    2.0,
                ),
                WholeCellReactionClass::ComplexRepair => {
                    let repair_deficit = reaction
                        .products
                        .iter()
                        .map(|participant| {
                            let count = species_counts
                                .get(&participant.species_id)
                                .copied()
                                .unwrap_or(0.0);
                            let target = species_anchors
                                .get(&participant.species_id)
                                .copied()
                                .or_else(|| species_basal.get(&participant.species_id).copied())
                                .unwrap_or(1.0)
                                .max(1.0);
                            ((target - count).max(0.0) / target).clamp(0.0, 1.0)
                        })
                        .fold(0.0, f32::max);
                    Self::finite_scale(
                        0.14 + 1.10 * repair_deficit
                            + 0.16 * expression.translation_support
                            + 0.08 * expression.membrane_support,
                        1.0,
                        0.05,
                        2.6,
                    )
                }
                WholeCellReactionClass::ComplexTurnover => Self::finite_scale(
                    0.72 + 0.30 * expression.crowding_penalty.max(0.0),
                    1.0,
                    0.60,
                    1.8,
                ),
            };
            let current_flux = (reaction.nominal_rate.max(0.0)
                * reactant_satisfaction
                * catalyst_support
                * asset_support
                * external_hint)
                .max(0.0);
            let reaction_class = reaction.reaction_class;
            let reaction_operon = reaction.operon.clone();
            let reaction_reactants = if dt_scale > 0.0 {
                reaction.reactants.clone()
            } else {
                Vec::new()
            };
            let reaction_products = if dt_scale > 0.0 {
                reaction.products.clone()
            } else {
                Vec::new()
            };
            let reaction = &mut self.organism_reactions[reaction_index];
            reaction.current_flux = current_flux;
            reaction.reactant_satisfaction = reactant_satisfaction;
            reaction.catalyst_support = catalyst_support;
            if dt_scale > 0.0 {
                reaction.cumulative_extent += current_flux * dt_scale;
                let extent = 0.05 * current_flux * dt_scale;
                if reaction_class == WholeCellReactionClass::StressResponse {
                    metabolic_load_relief += 0.08 * extent;
                    if let Some(operon) = reaction_operon.as_ref() {
                        *stress_relief.entry(operon.clone()).or_insert(0.0) += 0.18 * extent;
                    }
                }
                for participant in &reaction_reactants {
                    *species_deltas
                        .entry(participant.species_id.clone())
                        .or_insert(0.0) -= extent * participant.stoichiometry.max(0.0);
                    if let Some(species) = species_state.get(&participant.species_id) {
                        if let Some(field) = species.bulk_field {
                            *bulk_field_deltas
                                .entry((field, species.spatial_scope, species.patch_domain))
                                .or_insert(0.0) -=
                                extent * participant.stoichiometry.max(0.0);
                        }
                    }
                }
                for participant in &reaction_products {
                    *species_deltas
                        .entry(participant.species_id.clone())
                        .or_insert(0.0) += extent * participant.stoichiometry.max(0.0);
                    if let Some(species) = species_state.get(&participant.species_id) {
                        if let Some(field) = species.bulk_field {
                            *bulk_field_deltas
                                .entry((field, species.spatial_scope, species.patch_domain))
                                .or_insert(0.0) +=
                                extent * participant.stoichiometry.max(0.0);
                        }
                    }
                }
            }
        }

        if dt_scale > 0.0 {
            let mut transcript_delta_sum: HashMap<String, f32> = HashMap::new();
            let mut transcript_delta_count: HashMap<String, f32> = HashMap::new();
            let mut protein_delta_sum: HashMap<String, f32> = HashMap::new();
            let mut protein_delta_count: HashMap<String, f32> = HashMap::new();
            let mut complex_subunit_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_nucleation_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_elongation_deltas: HashMap<String, f32> = HashMap::new();
            let mut complex_mature_deltas: HashMap<String, f32> = HashMap::new();
            for species in &mut self.organism_species {
                if let Some(delta) = species_deltas.get(&species.id).copied() {
                    species.count = (species.count + delta).clamp(
                        0.0,
                        Self::runtime_species_upper_bound(species.species_class),
                    );
                    if delta >= 0.0 {
                        species.synthesis_rate += delta / dt_scale;
                    } else {
                        species.turnover_rate += (-delta) / dt_scale;
                    }
                    if species.bulk_field.is_none() {
                        match species.species_class {
                            WholeCellSpeciesClass::Rna => {
                                if let Some(operon) = species.operon.as_ref() {
                                    let total = rna_operon_basal_totals
                                        .get(operon)
                                        .copied()
                                        .unwrap_or(species.basal_abundance.max(0.01))
                                        .max(0.01);
                                    let candidate =
                                        delta * total / species.basal_abundance.max(0.01);
                                    *transcript_delta_sum.entry(operon.clone()).or_insert(0.0) +=
                                        candidate;
                                    *transcript_delta_count.entry(operon.clone()).or_insert(0.0) +=
                                        1.0;
                                }
                            }
                            WholeCellSpeciesClass::Protein => {
                                if let Some(operon) = species.operon.as_ref() {
                                    let total = protein_operon_basal_totals
                                        .get(operon)
                                        .copied()
                                        .unwrap_or(species.basal_abundance.max(0.01))
                                        .max(0.01);
                                    let candidate =
                                        delta * total / species.basal_abundance.max(0.01);
                                    *protein_delta_sum.entry(operon.clone()).or_insert(0.0) +=
                                        candidate;
                                    *protein_delta_count.entry(operon.clone()).or_insert(0.0) +=
                                        1.0;
                                }
                            }
                            WholeCellSpeciesClass::ComplexSubunitPool => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_subunit_deltas.entry(parent.clone()).or_insert(0.0) +=
                                        delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexNucleationIntermediate => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_nucleation_deltas
                                        .entry(parent.clone())
                                        .or_insert(0.0) += delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexElongationIntermediate => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_elongation_deltas
                                        .entry(parent.clone())
                                        .or_insert(0.0) += delta;
                                }
                            }
                            WholeCellSpeciesClass::ComplexMature => {
                                if let Some(parent) = species.parent_complex.as_ref() {
                                    *complex_mature_deltas.entry(parent.clone()).or_insert(0.0) +=
                                        delta;
                                }
                            }
                            WholeCellSpeciesClass::Pool => {}
                        }
                    }
                }
            }
            let mut lattice_bulk_changed = false;
            for ((field, scope, patch_domain), delta) in bulk_field_deltas {
                lattice_bulk_changed |=
                    self.apply_bulk_field_delta(field, scope, patch_domain, delta);
            }
            let mut expression_changed = false;
            for (operon, sum) in transcript_delta_sum {
                let average = sum
                    / transcript_delta_count
                        .get(&operon)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1.0);
                expression_changed |=
                    self.apply_operon_inventory_delta(&operon, average, 0.0, dt_scale);
            }
            for (operon, sum) in protein_delta_sum {
                let average = sum
                    / protein_delta_count
                        .get(&operon)
                        .copied()
                        .unwrap_or(1.0)
                        .max(1.0);
                expression_changed |=
                    self.apply_operon_inventory_delta(&operon, 0.0, average, dt_scale);
            }
            for (operon, relief) in stress_relief {
                if let Some(unit) = self
                    .organism_expression
                    .transcription_units
                    .iter_mut()
                    .find(|unit| unit.name == operon)
                {
                    unit.stress_penalty = (unit.stress_penalty - relief).clamp(0.80, 1.60);
                    unit.support_level = (unit.support_level + 0.10 * relief).clamp(0.55, 1.55);
                    expression_changed = true;
                }
            }
            if expression_changed {
                self.refresh_expression_inventory_totals();
            }
            let mut complex_changed = false;
            for (complex_id, delta) in complex_subunit_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexSubunitPool,
                    delta,
                );
            }
            for (complex_id, delta) in complex_nucleation_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexNucleationIntermediate,
                    delta,
                );
            }
            for (complex_id, delta) in complex_elongation_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexElongationIntermediate,
                    delta,
                );
            }
            for (complex_id, delta) in complex_mature_deltas {
                complex_changed |= self.apply_named_complex_species_delta(
                    &complex_id,
                    WholeCellSpeciesClass::ComplexMature,
                    delta,
                );
            }
            if complex_changed {
                if let Some(assets) = self.organism_assets.clone() {
                    self.complex_assembly = self.aggregate_named_complex_assembly_state(&assets);
                }
            }
            if metabolic_load_relief > 0.0 {
                self.metabolic_load =
                    (self.metabolic_load - metabolic_load_relief).clamp(0.25, 8.0);
            }
            if lattice_bulk_changed {
                self.sync_from_lattice();
            }
        }
    }

    fn expression_state_for_operon(
        &self,
        operon: &str,
    ) -> Option<&WholeCellTranscriptionUnitState> {
        self.organism_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == operon)
    }

    fn operon_gene_count(assets: &WholeCellGenomeAssetPackage, operon: &str) -> usize {
        assets
            .operons
            .iter()
            .find(|candidate| candidate.name == operon)
            .map(|candidate| candidate.genes.len().max(1))
            .unwrap_or(1)
    }

    fn named_complex_total_stoichiometry(complex: &WholeCellComplexSpec) -> f32 {
        complex
            .components
            .iter()
            .map(|component| component.stoichiometry.max(1) as f32)
            .sum::<f32>()
            .max(1.0)
    }

    fn named_complex_component_supply_signal(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut mean_signal = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let per_subunit = unit_abundance / component.stoichiometry.max(1) as f32;
            let half_saturation = 2.0
                + 0.75 * component.stoichiometry.max(1) as f32
                + 0.01 * protein.aa_length as f32;
            mean_signal += Self::saturating_signal(per_subunit, half_saturation);
            counted += 1;
        }

        if counted == 0 {
            1.0
        } else {
            (mean_signal / counted as f32).clamp(0.0, 1.0)
        }
    }

    fn named_complex_component_satisfaction(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut min_signal: f32 = 1.0;
        let mut mean_signal = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let half_saturation = 2.0
                + 1.25 * component.stoichiometry.max(1) as f32
                + 0.015 * protein.aa_length as f32;
            let signal = Self::saturating_signal(unit_abundance, half_saturation);
            min_signal = min_signal.min(signal);
            mean_signal += signal;
            counted += 1;
        }

        if counted == 0 {
            1.0
        } else {
            (0.65 * min_signal + 0.35 * (mean_signal / counted as f32)).clamp(0.0, 1.0)
        }
    }

    fn named_complex_limiting_component_signal(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        if complex.components.is_empty() {
            return 1.0;
        }
        let mut limiting_signal: f32 = 1.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let unit_abundance = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| {
                            state.mature_protein_abundance
                                / Self::operon_gene_count(assets, &protein.operon) as f32
                        })
                })
                .unwrap_or(protein.basal_abundance.max(0.0));
            let per_subunit = unit_abundance / component.stoichiometry.max(1) as f32;
            let half_saturation = 2.0 + 1.5 * component.stoichiometry.max(1) as f32;
            limiting_signal =
                limiting_signal.min(Self::saturating_signal(per_subunit, half_saturation));
            counted += 1;
        }
        if counted == 0 {
            1.0
        } else {
            limiting_signal.clamp(0.0, 1.0)
        }
    }

    fn named_complex_component_demand_map(
        &self,
        assets: &WholeCellGenomeAssetPackage,
    ) -> HashMap<String, f32> {
        let mut demand = HashMap::new();
        for (state, complex) in self.named_complexes.iter().zip(assets.complexes.iter()) {
            let assembly_drive =
                (state.target_abundance.max(state.abundance) + state.stalled_intermediate).max(0.0);
            for component in &complex.components {
                *demand.entry(component.protein_id.clone()).or_insert(0.0) +=
                    assembly_drive * component.stoichiometry.max(1) as f32;
            }
        }
        demand
    }

    fn named_complex_shared_component_pressure(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
        component_demand: &HashMap<String, f32>,
    ) -> f32 {
        if complex.components.is_empty() {
            return 0.0;
        }
        let mut mean_pressure = 0.0;
        let mut counted = 0usize;
        for component in &complex.components {
            let Some(protein) = assets
                .proteins
                .iter()
                .find(|protein| protein.id == component.protein_id)
            else {
                continue;
            };
            let available = self
                .species_runtime_count(&protein.id)
                .or_else(|| {
                    self.expression_state_for_operon(&protein.operon)
                        .map(|state| state.mature_protein_abundance.max(0.0))
                })
                .unwrap_or(protein.basal_abundance.max(0.0))
                .max(1.0);
            let demand = component_demand
                .get(&component.protein_id)
                .copied()
                .unwrap_or(0.0)
                .max(0.0);
            let pressure = ((demand / available) - 1.0).max(0.0);
            mean_pressure += pressure;
            counted += 1;
        }
        if counted == 0 {
            0.0
        } else {
            (mean_pressure / counted as f32).clamp(0.0, 4.0)
        }
    }

    fn named_complex_family_gate(&self, complex: &WholeCellComplexSpec) -> f32 {
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        match complex.family {
            WholeCellAssemblyFamily::Ribosome => 1.0,
            WholeCellAssemblyFamily::RnaPolymerase => {
                (0.82 + 0.18 * replicated_fraction).clamp(0.75, 1.15)
            }
            WholeCellAssemblyFamily::Replisome => {
                (0.70 + 0.50 * (1.0 - replicated_fraction)).clamp(0.55, 1.25)
            }
            WholeCellAssemblyFamily::AtpSynthase => Self::finite_scale(
                0.55 * self.chemistry_report.atp_support
                    + 0.25 * self.organism_expression.membrane_support
                    + 0.20 * self.localized_supply_scale(),
                1.0,
                0.55,
                1.25,
            ),
            WholeCellAssemblyFamily::Transporter => Self::finite_scale(
                0.60 * self.localized_supply_scale()
                    + 0.25 * self.organism_expression.membrane_support
                    + 0.15 * self.organism_expression.energy_support,
                1.0,
                0.55,
                1.25,
            ),
            WholeCellAssemblyFamily::MembraneEnzyme => Self::finite_scale(
                0.55 * self.organism_expression.membrane_support
                    + 0.20 * self.localized_supply_scale()
                    + 0.25 * (1.0 - self.division_progress).clamp(0.0, 1.0),
                1.0,
                0.55,
                1.20,
            ),
            WholeCellAssemblyFamily::ChaperoneClient => Self::finite_scale(
                0.80 + 0.25 * (self.effective_metabolic_load() - 1.0).max(0.0),
                1.0,
                0.75,
                1.40,
            ),
            WholeCellAssemblyFamily::Divisome => {
                (0.55 + 0.80 * replicated_fraction + 0.25 * self.division_progress)
                    .clamp(0.45, 1.45)
            }
            WholeCellAssemblyFamily::Generic => 1.0,
        }
    }

    fn named_complex_structural_support(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let localized_supply = self.localized_supply_scale();
        let process_scale = self.asset_class_process_scale(complex.asset_class);
        let operon_support = self
            .expression_state_for_operon(&complex.operon)
            .map(|state| {
                Self::finite_scale(
                    0.42 * state.support_level
                        + 0.28 * state.effective_activity
                        + 0.18 * (1.0 / state.stress_penalty.clamp(0.80, 1.60))
                        + 0.12 * localized_supply,
                    1.0,
                    0.55,
                    1.65,
                )
            })
            .unwrap_or(1.0);
        let component_operon_mean = if complex.components.is_empty() {
            1.0
        } else {
            let sum = complex
                .components
                .iter()
                .filter_map(|component| {
                    assets
                        .proteins
                        .iter()
                        .find(|protein| protein.id == component.protein_id)
                        .and_then(|protein| self.expression_state_for_operon(&protein.operon))
                        .map(|state| state.effective_activity)
                })
                .sum::<f32>();
            if sum <= 1.0e-6 {
                1.0
            } else {
                sum / complex.components.len() as f32
            }
        };
        Self::finite_scale(
            0.38 * process_scale
                + 0.32 * operon_support
                + 0.18 * component_operon_mean
                + 0.12 * crowding,
            1.0,
            0.55,
            1.80,
        )
    }

    fn named_complex_subunit_pool_target(
        &self,
        assets: &WholeCellGenomeAssetPackage,
        complex: &WholeCellComplexSpec,
    ) -> f32 {
        let supply_signal = self.named_complex_component_supply_signal(assets, complex);
        let structural_support = self.named_complex_structural_support(assets, complex);
        let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        (complex.basal_abundance.max(0.2)
            * (0.70 + 1.25 * supply_signal)
            * (0.85 + 0.15 * structural_support)
            * total_stoichiometry.sqrt()
            * crowding)
            .clamp(0.0, 1024.0)
    }

    fn initialize_named_complexes_state(&mut self) -> bool {
        let Some(assets) = self.organism_assets.clone() else {
            self.named_complexes.clear();
            return false;
        };
        if assets.complexes.is_empty() {
            self.named_complexes.clear();
            return false;
        }
        self.named_complexes = assets
            .complexes
            .iter()
            .map(|complex| {
                let component_satisfaction =
                    self.named_complex_component_satisfaction(&assets, complex);
                let structural_support = self.named_complex_structural_support(&assets, complex);
                let subunit_pool = self.named_complex_subunit_pool_target(&assets, complex);
                let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
                let target_abundance = (complex.basal_abundance.max(0.1)
                    * component_satisfaction
                    * structural_support
                    * self.organism_expression.crowding_penalty.clamp(0.65, 1.10))
                .clamp(0.0, 512.0);
                let nucleation_intermediate =
                    (0.10 * target_abundance * component_satisfaction * total_stoichiometry.sqrt())
                        .clamp(0.0, 256.0);
                let elongation_intermediate =
                    (0.08 * target_abundance * structural_support * total_stoichiometry.sqrt())
                        .clamp(0.0, 256.0);
                let assembly_progress = (0.42 * component_satisfaction
                    + 0.30 * structural_support
                    + 0.18
                        * Self::saturating_signal(
                            subunit_pool,
                            6.0 + 0.5 * total_stoichiometry.max(1.0),
                        )
                    + 0.10
                        * Self::saturating_signal(
                            nucleation_intermediate + elongation_intermediate,
                            2.0 + 0.4 * total_stoichiometry.max(1.0),
                        ))
                .clamp(0.0, 1.0);
                let limiting_component_signal =
                    self.named_complex_limiting_component_signal(&assets, complex);
                let insertion_progress = if complex.membrane_inserted {
                    (0.58 * structural_support + 0.42 * component_satisfaction).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                WholeCellNamedComplexState {
                    id: complex.id.clone(),
                    operon: complex.operon.clone(),
                    asset_class: complex.asset_class,
                    family: complex.family,
                    subsystem_targets: complex.subsystem_targets.clone(),
                    subunit_pool,
                    nucleation_intermediate,
                    elongation_intermediate,
                    abundance: target_abundance,
                    target_abundance,
                    assembly_rate: 0.0,
                    degradation_rate: 0.0,
                    nucleation_rate: 0.0,
                    elongation_rate: 0.0,
                    maturation_rate: 0.0,
                    component_satisfaction,
                    structural_support,
                    assembly_progress,
                    stalled_intermediate: 0.0,
                    damaged_abundance: 0.0,
                    limiting_component_signal,
                    shared_component_pressure: 0.0,
                    insertion_progress,
                    failure_count: 0.0,
                }
            })
            .collect();
        true
    }

    fn aggregate_named_complex_assembly_state(
        &self,
        assets: &WholeCellGenomeAssetPackage,
    ) -> WholeCellComplexAssemblyState {
        let mut aggregate = WholeCellComplexAssemblyState::default();
        for (state, complex) in self.named_complexes.iter().zip(assets.complexes.iter()) {
            let insertion_gate = if complex.membrane_inserted {
                (0.65 + 0.35 * state.insertion_progress).clamp(0.40, 1.0)
            } else {
                1.0
            };
            let damage_penalty = (1.0
                - 0.45
                    * Self::saturating_signal(
                        state.damaged_abundance,
                        1.0 + 0.2 * state.target_abundance.max(1.0),
                    ))
            .clamp(0.35, 1.0);
            let effective_abundance = (state.abundance * insertion_gate * damage_penalty)
                + 0.22 * state.elongation_intermediate
                + 0.08 * state.nucleation_intermediate;
            let effective_target = (state.target_abundance * insertion_gate)
                + 0.18 * state.elongation_intermediate
                + 0.06 * state.nucleation_intermediate
                + 0.04 * state.stalled_intermediate;
            let effective_assembly_rate = state.assembly_rate
                + 0.55 * state.maturation_rate
                + 0.30 * state.elongation_rate
                + 0.15 * state.nucleation_rate;
            let mut weights = complex.process_weights.clamped();
            for target in &complex.subsystem_targets {
                match target {
                    Syn3ASubsystemPreset::AtpSynthaseMembraneBand => {
                        weights.energy += 1.2;
                        weights.membrane += 0.35;
                    }
                    Syn3ASubsystemPreset::RibosomePolysomeCluster => {
                        weights.translation += 1.25;
                        weights.transcription += 0.20;
                    }
                    Syn3ASubsystemPreset::ReplisomeTrack => {
                        weights.replication += 1.15;
                        weights.segregation += 0.55;
                    }
                    Syn3ASubsystemPreset::FtsZSeptumRing => {
                        weights.constriction += 1.25;
                        weights.membrane += 0.30;
                    }
                }
            }
            if weights.transcription <= 1.0e-6 {
                weights.transcription += 0.18
                    * (weights.translation
                        + weights.replication
                        + weights.membrane
                        + weights.energy)
                        .max(0.1);
            }
            let total = weights.total().max(1.0e-6);
            let atp_share = (weights.energy + 0.20 * weights.membrane) / total;
            let ribosome_share = (weights.translation + 0.10 * weights.transcription) / total;
            let rnap_share = (weights.transcription + 0.10 * weights.translation) / total;
            let replisome_share = (weights.replication + 0.70 * weights.segregation) / total;
            let membrane_share = (weights.membrane + 0.15 * weights.energy) / total;
            let constriction_share = (weights.constriction + 0.20 * weights.membrane) / total;
            let dnaa_share = (0.65 * weights.replication + 0.35 * weights.transcription) / total;

            aggregate.atp_band_complexes += effective_abundance * atp_share;
            aggregate.ribosome_complexes += effective_abundance * ribosome_share;
            aggregate.rnap_complexes += effective_abundance * rnap_share;
            aggregate.replisome_complexes += effective_abundance * replisome_share;
            aggregate.membrane_complexes += effective_abundance * membrane_share;
            aggregate.ftsz_polymer += effective_abundance * constriction_share;
            aggregate.dnaa_activity += effective_abundance * dnaa_share;

            aggregate.atp_band_target += effective_target * atp_share;
            aggregate.ribosome_target += effective_target * ribosome_share;
            aggregate.rnap_target += effective_target * rnap_share;
            aggregate.replisome_target += effective_target * replisome_share;
            aggregate.membrane_target += effective_target * membrane_share;
            aggregate.ftsz_target += effective_target * constriction_share;
            aggregate.dnaa_target += effective_target * dnaa_share;

            aggregate.atp_band_assembly_rate += effective_assembly_rate * atp_share;
            aggregate.ribosome_assembly_rate += effective_assembly_rate * ribosome_share;
            aggregate.rnap_assembly_rate += effective_assembly_rate * rnap_share;
            aggregate.replisome_assembly_rate += effective_assembly_rate * replisome_share;
            aggregate.membrane_assembly_rate += effective_assembly_rate * membrane_share;
            aggregate.ftsz_assembly_rate += effective_assembly_rate * constriction_share;
            aggregate.dnaa_assembly_rate += effective_assembly_rate * dnaa_share;

            aggregate.atp_band_degradation_rate += state.degradation_rate * atp_share;
            aggregate.ribosome_degradation_rate += state.degradation_rate * ribosome_share;
            aggregate.rnap_degradation_rate += state.degradation_rate * rnap_share;
            aggregate.replisome_degradation_rate += state.degradation_rate * replisome_share;
            aggregate.membrane_degradation_rate += state.degradation_rate * membrane_share;
            aggregate.ftsz_degradation_rate += state.degradation_rate * constriction_share;
            aggregate.dnaa_degradation_rate += state.degradation_rate * dnaa_share;
        }
        aggregate
    }

    fn update_named_complexes_state(&mut self, dt: f32) -> bool {
        let Some(assets) = self.organism_assets.clone() else {
            self.named_complexes.clear();
            return false;
        };
        if assets.complexes.is_empty() {
            self.named_complexes.clear();
            return false;
        }
        if self.named_complexes.len() != assets.complexes.len() {
            self.initialize_named_complexes_state();
        }

        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let effective_load = self.effective_metabolic_load();
        let degradation_pressure =
            (0.68 + 0.22 * (effective_load - 1.0).max(0.0) + 0.16 * (1.0 - crowding).max(0.0))
                .clamp(0.60, 1.80);
        let component_demand = self.named_complex_component_demand_map(&assets);

        let updated_states = self
            .named_complexes
            .iter()
            .zip(assets.complexes.iter())
            .map(|(state, complex)| {
                let component_satisfaction =
                    self.named_complex_component_satisfaction(&assets, complex);
                let limiting_component_signal =
                    self.named_complex_limiting_component_signal(&assets, complex);
                let component_supply_signal =
                    self.named_complex_component_supply_signal(&assets, complex);
                let structural_support = self.named_complex_structural_support(&assets, complex);
                let shared_component_pressure = self.named_complex_shared_component_pressure(
                    &assets,
                    complex,
                    &component_demand,
                );
                let subunit_pool_target = self.named_complex_subunit_pool_target(&assets, complex);
                let total_stoichiometry = Self::named_complex_total_stoichiometry(complex);
                let complexity_penalty = 1.0 / total_stoichiometry.sqrt().max(1.0);
                let family_gate = self.named_complex_family_gate(complex);
                let assembly_support = Self::finite_scale(
                    0.36 * component_satisfaction
                        + 0.20 * limiting_component_signal
                        + 0.24 * structural_support
                        + 0.12 * crowding
                        + 0.08 * family_gate
                        - 0.12 * shared_component_pressure,
                    1.0,
                    0.45,
                    1.75,
                );
                let target_abundance = (complex.basal_abundance.max(0.1)
                    * component_satisfaction
                    * structural_support
                    * crowding
                    * family_gate)
                    .clamp(0.0, 512.0);
                let subunit_supply_rate = (subunit_pool_target - state.subunit_pool).max(0.0)
                    * (0.10 + 0.16 * component_supply_signal)
                    * (1.0 - 0.18 * shared_component_pressure.clamp(0.0, 1.5));
                let subunit_turnover_rate = state.subunit_pool
                    * (0.010
                        + 0.012 * (1.0 - component_satisfaction).max(0.0)
                        + 0.008 * (1.0 - crowding).max(0.0));
                let stall_pressure = (0.30 * (1.0 - limiting_component_signal).max(0.0)
                    + 0.24 * (1.0 - structural_support).max(0.0)
                    + 0.22 * shared_component_pressure
                    + 0.18 * (1.0 - family_gate).max(0.0)
                    + 0.12 * (degradation_pressure - 1.0).max(0.0))
                .clamp(0.0, 2.0);
                let nucleation_rate = state.subunit_pool
                    * (0.012 + 0.020 * component_satisfaction)
                    * complexity_penalty
                    * assembly_support
                    * (1.0 - 0.35 * stall_pressure.min(1.0));
                let nucleation_turnover_rate = state.nucleation_intermediate
                    * (0.012
                        + 0.012 * (1.0 - structural_support).max(0.0)
                        + 0.008 * (1.0 - component_satisfaction).max(0.0)
                        + 0.010 * stall_pressure);
                let elongation_rate = state.nucleation_intermediate
                    * (0.016 + 0.024 * structural_support)
                    * (0.82 + 0.18 * component_supply_signal)
                    * (1.0 - 0.28 * stall_pressure.min(1.0));
                let elongation_turnover_rate = state.elongation_intermediate
                    * (0.010
                        + 0.010 * (1.0 - structural_support).max(0.0)
                        + 0.006 * (1.0 - crowding).max(0.0)
                        + 0.012 * stall_pressure);
                let insertion_target = if complex.membrane_inserted {
                    (0.48 * structural_support
                        + 0.26 * component_satisfaction
                        + 0.16 * family_gate
                        + 0.10 * crowding)
                        .clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let insertion_progress = if complex.membrane_inserted {
                    (state.insertion_progress
                        + dt_scale
                            * ((0.020 + 0.028 * structural_support) * insertion_target
                                - state.insertion_progress
                                    * (0.010
                                        + 0.012 * stall_pressure
                                        + 0.008 * (1.0 - crowding).max(0.0))))
                    .clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let maturation_rate = state.elongation_intermediate
                    * (0.020 + 0.032 * assembly_support)
                    * (0.80 + 0.20 * component_satisfaction)
                    * (0.75 + 0.25 * insertion_progress)
                    * (1.0 - 0.30 * stall_pressure.min(1.0));
                let stalled_resolution_rate =
                    state.stalled_intermediate * (0.010 + 0.014 * structural_support);
                let stalled_intermediate = (state.stalled_intermediate
                    + dt_scale
                        * (0.42 * nucleation_turnover_rate
                            + 0.38 * elongation_turnover_rate
                            + 0.30
                                * stall_pressure
                                * (state.nucleation_intermediate + state.elongation_intermediate)
                            - stalled_resolution_rate))
                    .clamp(0.0, 512.0);
                let channel_degradation_pressure = (degradation_pressure
                    + 0.45 * (1.0 - component_satisfaction).max(0.0)
                    + 0.20 * (1.0 - structural_support).max(0.0)
                    + 0.24 * stall_pressure)
                    .clamp(0.60, 2.10);
                let (abundance, assembly_rate, degradation_rate) = Self::complex_channel_step(
                    state.abundance,
                    target_abundance,
                    assembly_support,
                    channel_degradation_pressure,
                    dt_scale,
                    512.0,
                );
                let subunit_pool = (state.subunit_pool
                    + dt_scale
                        * (subunit_supply_rate - 0.70 * nucleation_rate - subunit_turnover_rate))
                    .clamp(0.0, 2048.0);
                let nucleation_intermediate = (state.nucleation_intermediate
                    + dt_scale
                        * (nucleation_rate - 0.65 * elongation_rate - nucleation_turnover_rate))
                    .clamp(0.0, 512.0);
                let elongation_intermediate = (state.elongation_intermediate
                    + dt_scale
                        * (elongation_rate - 0.75 * maturation_rate - elongation_turnover_rate))
                    .clamp(0.0, 512.0);
                let damaged_abundance = (state.damaged_abundance
                    + dt_scale
                        * (abundance
                            * (0.004
                                + 0.006 * stall_pressure
                                + 0.004 * shared_component_pressure)
                            - state.damaged_abundance
                                * (0.008 + 0.010 * structural_support + 0.006 * family_gate)))
                    .clamp(0.0, 512.0);
                let assembly_progress = (0.36 * component_satisfaction
                    + 0.26 * structural_support
                    + 0.18
                        * Self::saturating_signal(
                            subunit_pool,
                            6.0 + 0.5 * total_stoichiometry.max(1.0),
                        )
                    + 0.20
                        * Self::saturating_signal(
                            nucleation_intermediate + elongation_intermediate,
                            2.0 + 0.35 * total_stoichiometry.max(1.0),
                        ))
                .clamp(0.0, 1.0);
                let failure_count = (state.failure_count
                    + dt_scale * (0.08 * stall_pressure + 0.04 * shared_component_pressure))
                    .clamp(0.0, 1.0e6);
                WholeCellNamedComplexState {
                    id: state.id.clone(),
                    operon: state.operon.clone(),
                    asset_class: state.asset_class,
                    family: state.family,
                    subsystem_targets: state.subsystem_targets.clone(),
                    subunit_pool,
                    nucleation_intermediate,
                    elongation_intermediate,
                    abundance,
                    target_abundance,
                    assembly_rate,
                    degradation_rate,
                    nucleation_rate,
                    elongation_rate,
                    maturation_rate,
                    component_satisfaction,
                    structural_support,
                    assembly_progress,
                    stalled_intermediate,
                    damaged_abundance,
                    limiting_component_signal,
                    shared_component_pressure,
                    insertion_progress,
                    failure_count,
                }
            })
            .collect();

        self.named_complexes = updated_states;
        self.complex_assembly = self.aggregate_named_complex_assembly_state(&assets);
        true
    }

    fn derived_complex_assembly_target(&self) -> WholeCellComplexAssemblyState {
        let prior = self.prior_assembly_inventory();
        let expression = &self.organism_expression;
        if self.organism_data.is_none() || expression.transcription_units.is_empty() {
            let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
            let energy_signal = Self::saturating_signal(
                0.70 * self.atp_mm.max(0.0) + 0.30 * self.chemistry_report.atp_support.max(0.0),
                1.2,
            );
            let transcription_signal = Self::saturating_signal(
                0.65 * self.nucleotides_mm.max(0.0) + 0.35 * self.glucose_mm.max(0.0),
                1.0,
            );
            let translation_signal = Self::saturating_signal(
                0.70 * self.amino_acids_mm.max(0.0) + 0.30 * self.md_translation_scale.max(0.0),
                1.1,
            );
            let membrane_signal = Self::saturating_signal(
                0.75 * self.membrane_precursors_mm.max(0.0)
                    + 0.25 * self.md_membrane_scale.max(0.0),
                0.8,
            );
            return WholeCellComplexAssemblyState {
                atp_band_target: (prior.atp_band_complexes * (0.76 + 0.44 * energy_signal))
                    .clamp(4.0, 512.0),
                ribosome_target: (prior.ribosome_complexes * (0.74 + 0.42 * translation_signal))
                    .clamp(8.0, 640.0),
                rnap_target: (prior.rnap_complexes * (0.76 + 0.38 * transcription_signal))
                    .clamp(6.0, 384.0),
                replisome_target: (prior.replisome_complexes
                    * (0.74 + 0.42 * transcription_signal)
                    * (0.82 + 0.24 * (1.0 - replicated_fraction)))
                    .clamp(2.0, 192.0),
                membrane_target: (prior.membrane_complexes * (0.74 + 0.40 * membrane_signal))
                    .clamp(4.0, 384.0),
                ftsz_target: (prior.ftsz_polymer
                    * (0.72 + 0.36 * membrane_signal + 0.36 * replicated_fraction))
                    .clamp(8.0, 768.0),
                dnaa_target: (prior.dnaa_activity
                    * (0.74 + 0.34 * transcription_signal)
                    * (0.84 + 0.20 * (1.0 - replicated_fraction)))
                    .clamp(4.0, 256.0),
                ..prior
            };
        }

        let mut protein_drive = WholeCellProcessWeights::default();
        let mut total_signal = 0.0;
        for unit in &expression.transcription_units {
            let assembly_mass = (0.76 * unit.protein_abundance.max(0.0)
                + 0.24 * unit.transcript_abundance.max(0.0))
                * unit.support_level.clamp(0.55, 1.55)
                / unit.stress_penalty.clamp(0.80, 1.60);
            protein_drive.add_weighted(unit.process_drive, assembly_mass);
            total_signal += unit.process_drive.total() * assembly_mass;
        }

        let mean_signal = if total_signal > 1.0e-6 {
            total_signal / 7.0
        } else {
            1.0
        };
        let protein_signal = Self::saturating_signal(
            expression.total_protein_abundance,
            36.0 + 3.0 * expression.transcription_units.len() as f32,
        );
        let transcript_signal = Self::saturating_signal(
            expression.total_transcript_abundance,
            24.0 + 2.0 * expression.transcription_units.len() as f32,
        );
        let localized_supply = self.localized_supply_scale();
        let crowding = expression.crowding_penalty.clamp(0.65, 1.10);
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;

        let energy_scale = Self::finite_scale(
            0.45 * expression.process_scales.energy
                + 0.30
                    * Self::complex_assembly_signal_scale(protein_drive.energy, mean_signal, 1.0)
                + 0.15 * expression.energy_support
                + 0.10 * localized_supply,
            1.0,
            0.55,
            1.85,
        );
        let transcription_scale = Self::finite_scale(
            0.42 * expression.process_scales.transcription
                + 0.30
                    * Self::complex_assembly_signal_scale(
                        protein_drive.transcription,
                        mean_signal,
                        1.0,
                    )
                + 0.18 * expression.translation_support
                + 0.10 * transcript_signal,
            1.0,
            0.55,
            1.85,
        );
        let translation_scale = Self::finite_scale(
            0.40 * expression.process_scales.translation
                + 0.34
                    * Self::complex_assembly_signal_scale(
                        protein_drive.translation,
                        mean_signal,
                        1.0,
                    )
                + 0.16 * expression.translation_support
                + 0.10 * protein_signal,
            1.0,
            0.55,
            1.95,
        );
        let replication_signal = 0.5 * (protein_drive.replication + protein_drive.segregation);
        let replication_scale = Self::finite_scale(
            0.38 * expression.process_scales.replication
                + 0.26 * Self::complex_assembly_signal_scale(replication_signal, mean_signal, 1.0)
                + 0.22 * expression.nucleotide_support
                + 0.14 * (1.0 - 0.35 * replicated_fraction).clamp(0.65, 1.15),
            1.0,
            0.55,
            1.95,
        );
        let membrane_scale = Self::finite_scale(
            0.44 * expression.process_scales.membrane
                + 0.30
                    * Self::complex_assembly_signal_scale(protein_drive.membrane, mean_signal, 1.0)
                + 0.16 * expression.membrane_support
                + 0.10 * protein_signal,
            1.0,
            0.55,
            1.95,
        );
        let constriction_scale = Self::finite_scale(
            0.36 * expression.process_scales.constriction
                + 0.26
                    * Self::complex_assembly_signal_scale(
                        protein_drive.constriction,
                        mean_signal,
                        1.0,
                    )
                + 0.18 * expression.membrane_support
                + 0.20 * (0.70 + 0.60 * replicated_fraction).clamp(0.70, 1.30),
            1.0,
            0.55,
            2.10,
        );
        let replication_window = (0.78 + 0.30 * (1.0 - replicated_fraction)).clamp(0.60, 1.20);
        let constriction_window = (0.65 + 0.60 * replicated_fraction).clamp(0.65, 1.35);

        WholeCellComplexAssemblyState {
            atp_band_complexes: prior.atp_band_complexes,
            ribosome_complexes: prior.ribosome_complexes,
            rnap_complexes: prior.rnap_complexes,
            replisome_complexes: prior.replisome_complexes,
            membrane_complexes: prior.membrane_complexes,
            ftsz_polymer: prior.ftsz_polymer,
            dnaa_activity: prior.dnaa_activity,
            atp_band_target: (prior.atp_band_complexes
                * energy_scale
                * crowding
                * (0.72 + 0.28 * protein_signal))
                .clamp(4.0, 512.0),
            ribosome_target: (prior.ribosome_complexes
                * translation_scale
                * crowding
                * (0.68 + 0.32 * protein_signal))
                .clamp(8.0, 640.0),
            rnap_target: (prior.rnap_complexes
                * transcription_scale
                * crowding
                * (0.72 + 0.28 * transcript_signal))
                .clamp(6.0, 384.0),
            replisome_target: (prior.replisome_complexes
                * replication_scale
                * replication_window
                * (0.72 + 0.28 * transcript_signal))
                .clamp(2.0, 192.0),
            membrane_target: (prior.membrane_complexes
                * membrane_scale
                * crowding
                * (0.72 + 0.28 * protein_signal))
                .clamp(4.0, 384.0),
            ftsz_target: (prior.ftsz_polymer
                * constriction_scale
                * constriction_window
                * (0.66 + 0.34 * protein_signal))
                .clamp(8.0, 768.0),
            dnaa_target: (prior.dnaa_activity
                * replication_scale
                * replication_window
                * (0.70 + 0.30 * transcript_signal))
                .clamp(4.0, 256.0),
            ..WholeCellComplexAssemblyState::default()
        }
    }

    fn initialize_complex_assembly_state(&mut self) {
        if self.initialize_named_complexes_state() {
            if let Some(assets) = self.organism_assets.as_ref() {
                self.complex_assembly = self.aggregate_named_complex_assembly_state(assets);
                return;
            }
        }
        let target = self.derived_complex_assembly_target();
        self.complex_assembly = WholeCellComplexAssemblyState {
            atp_band_complexes: target.atp_band_target,
            ribosome_complexes: target.ribosome_target,
            rnap_complexes: target.rnap_target,
            replisome_complexes: target.replisome_target,
            membrane_complexes: target.membrane_target,
            ftsz_polymer: target.ftsz_target,
            dnaa_activity: target.dnaa_target,
            ..target
        };
    }

    fn update_complex_assembly_state(&mut self, dt: f32) {
        if self.update_named_complexes_state(dt) {
            self.sync_runtime_process_species(dt);
            return;
        }
        let target = self.derived_complex_assembly_target();
        let current = if self.complex_assembly.total_complexes() > 1.0e-6 {
            self.complex_assembly
        } else {
            WholeCellComplexAssemblyState {
                atp_band_complexes: target.atp_band_target,
                ribosome_complexes: target.ribosome_target,
                rnap_complexes: target.rnap_target,
                replisome_complexes: target.replisome_target,
                membrane_complexes: target.membrane_target,
                ftsz_polymer: target.ftsz_target,
                dnaa_activity: target.dnaa_target,
                ..target
            }
        };

        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 6.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let effective_load = self.effective_metabolic_load();
        let degradation_pressure =
            (0.70 + 0.24 * (effective_load - 1.0).max(0.0) + 0.18 * (1.0 - crowding).max(0.0))
                .clamp(0.65, 1.80);

        let energy_support = Self::finite_scale(
            0.55 * self.organism_expression.energy_support
                + 0.45 * self.chemistry_report.atp_support,
            1.0,
            0.55,
            1.55,
        );
        let transcription_support = Self::finite_scale(
            0.55 * self.organism_expression.translation_support
                + 0.45 * self.organism_expression.process_scales.transcription,
            1.0,
            0.55,
            1.60,
        );
        let translation_support = Self::finite_scale(
            0.60 * self.organism_expression.translation_support
                + 0.40 * self.organism_expression.process_scales.translation,
            1.0,
            0.55,
            1.60,
        );
        let replication_support = Self::finite_scale(
            0.55 * self.organism_expression.nucleotide_support
                + 0.45
                    * (0.5
                        * (self.organism_expression.process_scales.replication
                            + self.organism_expression.process_scales.segregation)),
            1.0,
            0.55,
            1.60,
        );
        let membrane_support = Self::finite_scale(
            0.60 * self.organism_expression.membrane_support
                + 0.40 * self.organism_expression.process_scales.membrane,
            1.0,
            0.55,
            1.60,
        );
        let constriction_support = Self::finite_scale(
            0.50 * self.organism_expression.membrane_support
                + 0.30 * self.organism_expression.process_scales.constriction
                + 0.20 * (0.70 + 0.60 * (self.replicated_bp as f32 / self.genome_bp.max(1) as f32)),
            1.0,
            0.55,
            1.70,
        );

        let (atp_band_complexes, atp_band_assembly_rate, atp_band_degradation_rate) =
            Self::complex_channel_step(
                current.atp_band_complexes,
                target.atp_band_target,
                energy_support,
                degradation_pressure,
                dt_scale,
                512.0,
            );
        let (ribosome_complexes, ribosome_assembly_rate, ribosome_degradation_rate) =
            Self::complex_channel_step(
                current.ribosome_complexes,
                target.ribosome_target,
                translation_support,
                degradation_pressure,
                dt_scale,
                640.0,
            );
        let (rnap_complexes, rnap_assembly_rate, rnap_degradation_rate) =
            Self::complex_channel_step(
                current.rnap_complexes,
                target.rnap_target,
                transcription_support,
                degradation_pressure,
                dt_scale,
                384.0,
            );
        let (replisome_complexes, replisome_assembly_rate, replisome_degradation_rate) =
            Self::complex_channel_step(
                current.replisome_complexes,
                target.replisome_target,
                replication_support,
                degradation_pressure,
                dt_scale,
                192.0,
            );
        let (membrane_complexes, membrane_assembly_rate, membrane_degradation_rate) =
            Self::complex_channel_step(
                current.membrane_complexes,
                target.membrane_target,
                membrane_support,
                degradation_pressure,
                dt_scale,
                384.0,
            );
        let (ftsz_polymer, ftsz_assembly_rate, ftsz_degradation_rate) = Self::complex_channel_step(
            current.ftsz_polymer,
            target.ftsz_target,
            constriction_support,
            degradation_pressure,
            dt_scale,
            768.0,
        );
        let (dnaa_activity, dnaa_assembly_rate, dnaa_degradation_rate) = Self::complex_channel_step(
            current.dnaa_activity,
            target.dnaa_target,
            replication_support,
            degradation_pressure,
            dt_scale,
            256.0,
        );

        self.complex_assembly = WholeCellComplexAssemblyState {
            atp_band_complexes,
            ribosome_complexes,
            rnap_complexes,
            replisome_complexes,
            membrane_complexes,
            ftsz_polymer,
            dnaa_activity,
            atp_band_target: target.atp_band_target,
            ribosome_target: target.ribosome_target,
            rnap_target: target.rnap_target,
            replisome_target: target.replisome_target,
            membrane_target: target.membrane_target,
            ftsz_target: target.ftsz_target,
            dnaa_target: target.dnaa_target,
            atp_band_assembly_rate,
            ribosome_assembly_rate,
            rnap_assembly_rate,
            replisome_assembly_rate,
            membrane_assembly_rate,
            ftsz_assembly_rate,
            dnaa_assembly_rate,
            atp_band_degradation_rate,
            ribosome_degradation_rate,
            rnap_degradation_rate,
            replisome_degradation_rate,
            membrane_degradation_rate,
            ftsz_degradation_rate,
            dnaa_degradation_rate,
        };
        self.sync_runtime_process_species(dt);
    }

    fn refresh_organism_expression_state(&mut self) {
        let Some(organism) = self.organism_data.clone() else {
            self.organism_expression = WholeCellOrganismExpressionState::default();
            return;
        };
        let previous_units = self.organism_expression.transcription_units.clone();
        let profile = derive_organism_profile(&organism);
        let localized_supply = self.localized_supply_scale();
        let crowding_penalty =
            Self::finite_scale(self.chemistry_report.crowding_penalty, 1.0, 0.65, 1.10);
        let adenylate_ratio = self.atp_mm.max(0.0) / (self.adp_mm + 0.25).max(0.25);
        let energy_support = Self::finite_scale(
            0.44 * self.chemistry_report.atp_support
                + 0.24 * Self::saturating_signal(adenylate_ratio, 1.0)
                + 0.16 * Self::saturating_signal(self.glucose_mm, 0.8)
                + 0.16 * Self::saturating_signal(self.oxygen_mm, 0.7),
            1.0,
            0.55,
            1.55,
        );
        let translation_support = Self::finite_scale(
            0.72 * self.chemistry_report.translation_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let nucleotide_support = Self::finite_scale(
            0.72 * self.chemistry_report.nucleotide_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let membrane_support = Self::finite_scale(
            0.72 * self.chemistry_report.membrane_support + 0.28 * localized_supply,
            1.0,
            0.55,
            1.55,
        );
        let load_penalty = (1.0 + 0.22 * (self.metabolic_load - 1.0).max(0.0)).clamp(1.0, 1.45);

        let mut process_signal = WholeCellProcessWeights::default();
        let mut transcription_units = Vec::new();
        let mut activity_total = 0.0;
        let mut amino_cost_signal = 0.0;
        let mut nucleotide_cost_signal = 0.0;
        let mut total_transcript_abundance = 0.0;
        let mut total_protein_abundance = 0.0;

        for unit in &organism.transcription_units {
            let mut unit_weights = unit.process_weights.clamped();
            let mut midpoint_sum = 0.0;
            let mut midpoint_count = 0usize;
            let mut mean_translation_cost = 1.0;
            let mut mean_nucleotide_cost = 1.0;
            let mut cost_count = 0.0;

            for gene_name in &unit.genes {
                if let Some(feature) = organism
                    .genes
                    .iter()
                    .find(|feature| feature.gene == *gene_name)
                {
                    let midpoint_bp = 0.5 * (feature.start_bp as f32 + feature.end_bp as f32);
                    midpoint_sum += midpoint_bp;
                    midpoint_count += 1;
                    unit_weights.add_weighted(
                        feature.process_weights,
                        0.35 * feature.basal_expression.max(0.1),
                    );
                    mean_translation_cost += feature.translation_cost.max(0.0);
                    mean_nucleotide_cost += feature.nucleotide_cost.max(0.0);
                    cost_count += 1.0;
                }
            }

            if cost_count > 0.0 {
                mean_translation_cost /= 1.0 + cost_count;
                mean_nucleotide_cost /= 1.0 + cost_count;
            }

            let midpoint_bp = if midpoint_count > 0 {
                (midpoint_sum / midpoint_count as f32).round() as u32
            } else {
                organism.origin_bp.min(organism.chromosome_length_bp.max(1))
            };
            let copy_gain = self.chromosome_copy_number_at(midpoint_bp);
            let chromosome_accessibility = self.chromosome_locus_accessibility_at(midpoint_bp);
            let chromosome_torsion = self.chromosome_locus_torsional_stress_at(midpoint_bp);
            let (transcription_length_nt, translation_length_aa) =
                Self::expression_lengths_for_operon(&organism, &unit.genes);
            let previous_state = previous_units
                .iter()
                .find(|previous| previous.name == unit.name);
            let transcript_abundance = previous_state
                .map(|previous| previous.transcript_abundance)
                .unwrap_or_else(|| {
                    (8.0 + 5.0 * unit.basal_activity.max(0.05) * copy_gain)
                        * (unit.genes.len().max(1) as f32).sqrt()
                });
            let protein_abundance = previous_state
                .map(|previous| previous.protein_abundance)
                .unwrap_or_else(|| {
                    (14.0 + 9.0 * unit.basal_activity.max(0.05) * copy_gain)
                        * (unit.genes.len().max(1) as f32).sqrt()
                });
            let support_level = Self::unit_support_level(
                unit_weights,
                energy_support,
                translation_support,
                nucleotide_support,
                membrane_support,
                localized_supply,
            );
            let stress_penalty = (load_penalty
                + 0.22 * (1.0 - crowding_penalty).max(0.0)
                + 0.18 * (1.0 - support_level).max(0.0))
            .clamp(0.80, 1.60);
            let effective_activity = (unit.basal_activity.max(0.05)
                * copy_gain
                * support_level
                * chromosome_accessibility
                / (stress_penalty * (1.0 + 0.18 * chromosome_torsion)))
                .clamp(0.05, 2.50);
            let inventory_scale = Self::unit_inventory_scale(
                transcript_abundance,
                protein_abundance,
                unit.genes.len(),
            );

            process_signal.add_weighted(unit_weights, effective_activity * inventory_scale);
            activity_total += effective_activity;
            amino_cost_signal += mean_translation_cost * effective_activity * inventory_scale;
            nucleotide_cost_signal += mean_nucleotide_cost * effective_activity * inventory_scale;
            total_transcript_abundance += transcript_abundance;
            total_protein_abundance += protein_abundance;
            transcription_units.push(WholeCellTranscriptionUnitState {
                name: unit.name.clone(),
                gene_count: unit.genes.len(),
                copy_gain,
                basal_activity: unit.basal_activity.max(0.0),
                effective_activity,
                support_level,
                stress_penalty,
                transcript_abundance,
                protein_abundance,
                transcript_synthesis_rate: previous_state
                    .map(|previous| previous.transcript_synthesis_rate)
                    .unwrap_or(0.0),
                protein_synthesis_rate: previous_state
                    .map(|previous| previous.protein_synthesis_rate)
                    .unwrap_or(0.0),
                transcript_turnover_rate: previous_state
                    .map(|previous| previous.transcript_turnover_rate)
                    .unwrap_or(0.0),
                protein_turnover_rate: previous_state
                    .map(|previous| previous.protein_turnover_rate)
                    .unwrap_or(0.0),
                promoter_open_fraction: previous_state
                    .map(|previous| previous.promoter_open_fraction)
                    .unwrap_or(0.0),
                active_rnap_occupancy: previous_state
                    .map(|previous| previous.active_rnap_occupancy)
                    .unwrap_or(0.0),
                transcription_length_nt: previous_state
                    .map(|previous| previous.transcription_length_nt.max(90.0))
                    .unwrap_or(transcription_length_nt),
                transcription_progress_nt: previous_state
                    .map(|previous| previous.transcription_progress_nt)
                    .unwrap_or(0.0),
                nascent_transcript_abundance: previous_state
                    .map(|previous| previous.nascent_transcript_abundance)
                    .unwrap_or(0.0),
                mature_transcript_abundance: previous_state
                    .map(|previous| previous.mature_transcript_abundance)
                    .unwrap_or(transcript_abundance),
                damaged_transcript_abundance: previous_state
                    .map(|previous| previous.damaged_transcript_abundance)
                    .unwrap_or(0.0),
                active_ribosome_occupancy: previous_state
                    .map(|previous| previous.active_ribosome_occupancy)
                    .unwrap_or(0.0),
                translation_length_aa: previous_state
                    .map(|previous| previous.translation_length_aa.max(30.0))
                    .unwrap_or(translation_length_aa),
                translation_progress_aa: previous_state
                    .map(|previous| previous.translation_progress_aa)
                    .unwrap_or(0.0),
                nascent_protein_abundance: previous_state
                    .map(|previous| previous.nascent_protein_abundance)
                    .unwrap_or(0.0),
                mature_protein_abundance: previous_state
                    .map(|previous| previous.mature_protein_abundance)
                    .unwrap_or(protein_abundance),
                damaged_protein_abundance: previous_state
                    .map(|previous| previous.damaged_protein_abundance)
                    .unwrap_or(0.0),
                process_drive: unit_weights.clamped(),
            });
        }

        if transcription_units.is_empty() && !organism.genes.is_empty() {
            for feature in &organism.genes {
                let midpoint_bp = 0.5 * (feature.start_bp as f32 + feature.end_bp as f32);
                let midpoint_bp = midpoint_bp.round() as u32;
                let copy_gain = self.chromosome_copy_number_at(midpoint_bp);
                let chromosome_accessibility = self.chromosome_locus_accessibility_at(midpoint_bp);
                let chromosome_torsion = self.chromosome_locus_torsional_stress_at(midpoint_bp);
                let (transcription_length_nt, translation_length_aa) =
                    Self::expression_lengths_for_gene(feature);
                let previous_state = previous_units
                    .iter()
                    .find(|previous| previous.name == feature.gene);
                let transcript_abundance = previous_state
                    .map(|previous| previous.transcript_abundance)
                    .unwrap_or_else(|| 6.0 + 4.0 * feature.basal_expression.max(0.05) * copy_gain);
                let protein_abundance = previous_state
                    .map(|previous| previous.protein_abundance)
                    .unwrap_or_else(|| 10.0 + 8.0 * feature.basal_expression.max(0.05) * copy_gain);
                let support_level = Self::unit_support_level(
                    feature.process_weights,
                    energy_support,
                    translation_support,
                    nucleotide_support,
                    membrane_support,
                    localized_supply,
                );
                let stress_penalty = (load_penalty
                    + 0.22 * (1.0 - crowding_penalty).max(0.0)
                    + 0.18 * (1.0 - support_level).max(0.0))
                .clamp(0.80, 1.60);
                let effective_activity =
                    (feature.basal_expression.max(0.05) * copy_gain * chromosome_accessibility
                        / (stress_penalty * (1.0 + 0.18 * chromosome_torsion)))
                        .clamp(0.05, 2.50);
                let inventory_scale =
                    Self::unit_inventory_scale(transcript_abundance, protein_abundance, 1);
                process_signal.add_weighted(
                    feature.process_weights,
                    effective_activity * inventory_scale,
                );
                activity_total += effective_activity;
                amino_cost_signal +=
                    feature.translation_cost.max(0.0) * effective_activity * inventory_scale;
                nucleotide_cost_signal +=
                    feature.nucleotide_cost.max(0.0) * effective_activity * inventory_scale;
                total_transcript_abundance += transcript_abundance;
                total_protein_abundance += protein_abundance;
                transcription_units.push(WholeCellTranscriptionUnitState {
                    name: feature.gene.clone(),
                    gene_count: 1,
                    copy_gain,
                    basal_activity: feature.basal_expression.max(0.0),
                    effective_activity,
                    support_level,
                    stress_penalty,
                    transcript_abundance,
                    protein_abundance,
                    transcript_synthesis_rate: previous_state
                        .map(|previous| previous.transcript_synthesis_rate)
                        .unwrap_or(0.0),
                    protein_synthesis_rate: previous_state
                        .map(|previous| previous.protein_synthesis_rate)
                        .unwrap_or(0.0),
                    transcript_turnover_rate: previous_state
                        .map(|previous| previous.transcript_turnover_rate)
                        .unwrap_or(0.0),
                    protein_turnover_rate: previous_state
                        .map(|previous| previous.protein_turnover_rate)
                        .unwrap_or(0.0),
                    promoter_open_fraction: previous_state
                        .map(|previous| previous.promoter_open_fraction)
                        .unwrap_or(0.0),
                    active_rnap_occupancy: previous_state
                        .map(|previous| previous.active_rnap_occupancy)
                        .unwrap_or(0.0),
                    transcription_length_nt: previous_state
                        .map(|previous| previous.transcription_length_nt.max(90.0))
                        .unwrap_or(transcription_length_nt),
                    transcription_progress_nt: previous_state
                        .map(|previous| previous.transcription_progress_nt)
                        .unwrap_or(0.0),
                    nascent_transcript_abundance: previous_state
                        .map(|previous| previous.nascent_transcript_abundance)
                        .unwrap_or(0.0),
                    mature_transcript_abundance: previous_state
                        .map(|previous| previous.mature_transcript_abundance)
                        .unwrap_or(transcript_abundance),
                    damaged_transcript_abundance: previous_state
                        .map(|previous| previous.damaged_transcript_abundance)
                        .unwrap_or(0.0),
                    active_ribosome_occupancy: previous_state
                        .map(|previous| previous.active_ribosome_occupancy)
                        .unwrap_or(0.0),
                    translation_length_aa: previous_state
                        .map(|previous| previous.translation_length_aa.max(30.0))
                        .unwrap_or(translation_length_aa),
                    translation_progress_aa: previous_state
                        .map(|previous| previous.translation_progress_aa)
                        .unwrap_or(0.0),
                    nascent_protein_abundance: previous_state
                        .map(|previous| previous.nascent_protein_abundance)
                        .unwrap_or(0.0),
                    mature_protein_abundance: previous_state
                        .map(|previous| previous.mature_protein_abundance)
                        .unwrap_or(protein_abundance),
                    damaged_protein_abundance: previous_state
                        .map(|previous| previous.damaged_protein_abundance)
                        .unwrap_or(0.0),
                    process_drive: feature.process_weights.clamped(),
                });
            }
        }

        let registry_drive = self.registry_process_drive();
        process_signal.add_weighted(registry_drive, 0.14);
        activity_total += 0.02 * registry_drive.total().max(0.0);
        amino_cost_signal +=
            0.06 * (registry_drive.translation + 0.45 * registry_drive.membrane).max(0.0);
        nucleotide_cost_signal +=
            0.05 * (registry_drive.transcription + 0.70 * registry_drive.replication).max(0.0);

        let process_mean = (process_signal.energy
            + process_signal.transcription
            + process_signal.translation
            + process_signal.replication
            + process_signal.segregation
            + process_signal.membrane
            + process_signal.constriction)
            / 7.0;
        let cost_mean = if activity_total > 1.0e-6 {
            (amino_cost_signal + nucleotide_cost_signal) / (2.0 * activity_total.max(1.0e-6))
        } else {
            1.0
        };
        let global_activity = if transcription_units.is_empty() {
            1.0
        } else {
            activity_total / transcription_units.len() as f32
        };
        let process_scales = WholeCellProcessWeights {
            energy: Self::finite_scale(
                profile.process_scales.energy
                    * Self::process_scale(process_signal.energy, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            transcription: Self::finite_scale(
                profile.process_scales.transcription
                    * Self::process_scale(process_signal.transcription, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            translation: Self::finite_scale(
                profile.process_scales.translation
                    * Self::process_scale(process_signal.translation, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            replication: Self::finite_scale(
                profile.process_scales.replication
                    * Self::process_scale(process_signal.replication, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            segregation: Self::finite_scale(
                profile.process_scales.segregation
                    * Self::process_scale(process_signal.segregation, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            membrane: Self::finite_scale(
                profile.process_scales.membrane
                    * Self::process_scale(process_signal.membrane, process_mean),
                1.0,
                0.70,
                1.45,
            ),
            constriction: Self::finite_scale(
                profile.process_scales.constriction
                    * Self::process_scale(process_signal.constriction, process_mean),
                1.0,
                0.70,
                1.45,
            ),
        };

        self.organism_expression = WholeCellOrganismExpressionState {
            global_activity: Self::finite_scale(global_activity, 1.0, 0.50, 1.80),
            energy_support,
            translation_support,
            nucleotide_support,
            membrane_support,
            crowding_penalty,
            metabolic_burden_scale: Self::finite_scale(
                profile.metabolic_burden_scale * (0.92 + 0.08 * global_activity),
                profile.metabolic_burden_scale,
                0.85,
                1.65,
            ),
            process_scales,
            amino_cost_scale: Self::process_scale(amino_cost_signal, cost_mean),
            nucleotide_cost_scale: Self::process_scale(nucleotide_cost_signal, cost_mean),
            total_transcript_abundance,
            total_protein_abundance,
            transcription_units,
        };
    }

    fn update_organism_inventory_dynamics(
        &mut self,
        dt: f32,
        transcription_flux: f32,
        translation_flux: f32,
    ) {
        if self.organism_data.is_none() || self.organism_expression.transcription_units.is_empty() {
            return;
        }
        let dt_scale = (dt / self.config.dt_ms.max(0.05)).clamp(0.5, 4.0);
        let crowding = self.organism_expression.crowding_penalty.clamp(0.65, 1.10);
        let transcription_flux = transcription_flux.max(0.0);
        let translation_flux = translation_flux.max(0.0);
        let engaged_rnap_capacity =
            self.active_rnap.clamp(8.0, 256.0) * (0.08 + 0.04 * transcription_flux.clamp(0.0, 2.5));
        let engaged_ribosome_capacity = self.active_ribosomes.clamp(12.0, 320.0)
            * (0.10 + 0.05 * translation_flux.clamp(0.0, 2.5));

        let mut transcription_demands =
            Vec::with_capacity(self.organism_expression.transcription_units.len());
        let mut translation_demands =
            Vec::with_capacity(self.organism_expression.transcription_units.len());

        for unit in &mut self.organism_expression.transcription_units {
            Self::normalize_expression_execution_state(unit);
            let gene_scale = (unit.gene_count.max(1) as f32).sqrt();
            let support = unit.support_level.clamp(0.55, 1.55);
            let stress = unit.stress_penalty.clamp(0.80, 1.60);
            unit.promoter_open_fraction = Self::finite_scale(
                0.34 + 0.30 * support + 0.16 * unit.copy_gain.clamp(0.8, 1.5) + 0.12 * crowding
                    - 0.18 * (stress - 1.0).max(0.0),
                0.5,
                0.05,
                1.0,
            );
            let transcription_demand = unit.promoter_open_fraction
                * unit.effective_activity
                * (0.55 + 0.25 * gene_scale)
                * (0.75 + 0.25 * crowding)
                * (0.85 + 0.25 * transcription_flux);
            let transcript_signal = Self::saturating_signal(
                unit.mature_transcript_abundance + 0.5 * unit.nascent_transcript_abundance,
                8.0 + 2.0 * gene_scale,
            );
            let translation_demand = transcript_signal
                * unit.effective_activity
                * support
                * (0.45 + 0.20 * gene_scale)
                * (0.80 + 0.20 * translation_flux);
            transcription_demands.push(transcription_demand.max(0.0));
            translation_demands.push(translation_demand.max(0.0));
        }

        let total_transcription_demand = transcription_demands
            .iter()
            .copied()
            .sum::<f32>()
            .max(1.0e-6);
        let total_translation_demand = translation_demands.iter().copied().sum::<f32>().max(1.0e-6);

        for ((unit, transcription_demand), translation_demand) in self
            .organism_expression
            .transcription_units
            .iter_mut()
            .zip(transcription_demands.into_iter())
            .zip(translation_demands.into_iter())
        {
            let gene_scale = (unit.gene_count.max(1) as f32).sqrt();
            let support = unit.support_level.clamp(0.55, 1.55);
            let stress = unit.stress_penalty.clamp(0.80, 1.60);

            let rnap_target = engaged_rnap_capacity
                * (transcription_demand / total_transcription_demand)
                * (0.60 + 0.40 * unit.promoter_open_fraction);
            let ribosome_target = engaged_ribosome_capacity
                * (translation_demand / total_translation_demand)
                * (0.55 + 0.45 * Self::saturating_signal(unit.mature_transcript_abundance, 6.0));
            let rnap_ceiling = (3.0 + 2.5 * gene_scale * unit.copy_gain.clamp(0.8, 1.8)).max(1.0);
            let ribosome_ceiling =
                (4.0 + 3.5 * gene_scale * unit.copy_gain.clamp(0.8, 1.8)).max(1.0);
            unit.active_rnap_occupancy =
                (0.58 * unit.active_rnap_occupancy + 0.42 * rnap_target).clamp(0.0, rnap_ceiling);
            unit.active_ribosome_occupancy = (0.56 * unit.active_ribosome_occupancy
                + 0.44 * ribosome_target)
                .clamp(0.0, ribosome_ceiling);

            let transcription_elongation_rate = (18.0 + 14.0 * support + 5.0 * transcription_flux
                - 3.0 * (stress - 1.0).max(0.0))
            .max(6.0);
            unit.transcription_progress_nt +=
                unit.active_rnap_occupancy * transcription_elongation_rate * dt_scale * crowding;
            let completed_transcripts =
                (unit.transcription_progress_nt / unit.transcription_length_nt.max(1.0)).max(0.0);
            unit.transcription_progress_nt %= unit.transcription_length_nt.max(1.0);

            let nascent_transcript_target =
                unit.active_rnap_occupancy * (0.35 + 0.10 * support + 0.08 * gene_scale);
            unit.nascent_transcript_abundance = (0.62 * unit.nascent_transcript_abundance
                + 0.38 * nascent_transcript_target)
                .clamp(0.0, 512.0);
            let transcript_damage = unit.mature_transcript_abundance
                * (0.0015 + 0.008 * (stress - 1.0).max(0.0) + 0.003 * (1.0 - support).max(0.0))
                * dt_scale;
            let transcript_turnover = unit.mature_transcript_abundance
                * (0.008 + 0.010 * (stress - 1.0).max(0.0) + 0.004 * (1.0 - support).max(0.0))
                * dt_scale;
            let transcript_damage_clearance =
                unit.damaged_transcript_abundance * (0.012 + 0.010 * support) * dt_scale;
            let matured_transcripts = completed_transcripts
                * (0.68 + 0.22 * support + 0.10 * crowding)
                * transcription_flux.max(0.25);
            unit.mature_transcript_abundance = (unit.mature_transcript_abundance
                + matured_transcripts
                - transcript_turnover
                - transcript_damage)
                .clamp(0.0, 1024.0);
            unit.damaged_transcript_abundance = (unit.damaged_transcript_abundance
                + transcript_damage
                - transcript_damage_clearance)
                .clamp(0.0, 512.0);

            let translation_elongation_rate = (9.0 + 8.0 * support + 4.0 * translation_flux
                - 2.0 * (stress - 1.0).max(0.0))
            .max(3.0);
            unit.translation_progress_aa +=
                unit.active_ribosome_occupancy * translation_elongation_rate * dt_scale * crowding;
            let completed_proteins =
                (unit.translation_progress_aa / unit.translation_length_aa.max(1.0)).max(0.0);
            unit.translation_progress_aa %= unit.translation_length_aa.max(1.0);

            let nascent_protein_target =
                unit.active_ribosome_occupancy * (0.28 + 0.08 * support + 0.06 * gene_scale);
            unit.nascent_protein_abundance = (0.64 * unit.nascent_protein_abundance
                + 0.36 * nascent_protein_target)
                .clamp(0.0, 768.0);
            let protein_damage = unit.mature_protein_abundance
                * (0.0008 + 0.005 * (stress - 1.0).max(0.0) + 0.002 * (1.0 - support).max(0.0))
                * dt_scale;
            let protein_turnover = unit.mature_protein_abundance
                * (0.003 + 0.005 * (stress - 1.0).max(0.0) + 0.002 * (1.0 - support).max(0.0))
                * dt_scale;
            let protein_damage_clearance =
                unit.damaged_protein_abundance * (0.007 + 0.008 * support) * dt_scale;
            let matured_proteins = completed_proteins
                * (0.70 + 0.20 * support + 0.10 * crowding)
                * translation_flux.max(0.25);
            unit.mature_protein_abundance = (unit.mature_protein_abundance + matured_proteins
                - protein_turnover
                - protein_damage)
                .clamp(0.0, 2048.0);
            unit.damaged_protein_abundance = (unit.damaged_protein_abundance + protein_damage
                - protein_damage_clearance)
                .clamp(0.0, 1024.0);

            unit.transcript_synthesis_rate = (matured_transcripts / dt_scale.max(1.0e-6)).max(0.0);
            unit.protein_synthesis_rate = (matured_proteins / dt_scale.max(1.0e-6)).max(0.0);
            unit.transcript_turnover_rate = ((transcript_turnover + transcript_damage_clearance)
                / dt_scale.max(1.0e-6))
            .max(0.0);
            unit.protein_turnover_rate =
                ((protein_turnover + protein_damage_clearance) / dt_scale.max(1.0e-6)).max(0.0);
            Self::normalize_expression_execution_state(unit);
        }

        self.refresh_expression_inventory_totals();
    }

    fn organism_process_scales(&self) -> WholeCellOrganismProcessScales {
        let expression = &self.organism_expression;
        if self.organism_data.is_none() || expression.transcription_units.is_empty() {
            return WholeCellOrganismProcessScales::default();
        }
        WholeCellOrganismProcessScales {
            energy_scale: expression.process_scales.energy,
            transcription_scale: expression.process_scales.transcription,
            translation_scale: expression.process_scales.translation,
            replication_scale: expression.process_scales.replication,
            segregation_scale: expression.process_scales.segregation,
            membrane_scale: expression.process_scales.membrane,
            constriction_scale: expression.process_scales.constriction,
            amino_cost_scale: expression.amino_cost_scale,
            nucleotide_cost_scale: expression.nucleotide_cost_scale,
        }
    }

    pub fn organism_summary(&self) -> Option<WholeCellOrganismSummary> {
        self.organism_data
            .as_ref()
            .map(WholeCellOrganismSummary::from)
    }

    pub fn organism_profile(&self) -> Option<WholeCellOrganismProfile> {
        self.organism_data.as_ref().map(derive_organism_profile)
    }

    pub fn organism_asset_summary(&self) -> Option<WholeCellGenomeAssetSummary> {
        self.organism_assets
            .as_ref()
            .map(WholeCellGenomeAssetSummary::from)
    }

    pub fn organism_asset_package(&self) -> Option<WholeCellGenomeAssetPackage> {
        self.organism_assets.clone()
    }

    pub fn organism_process_registry_summary(
        &self,
    ) -> Option<WholeCellGenomeProcessRegistrySummary> {
        self.organism_process_registry()
            .map(|registry| WholeCellGenomeProcessRegistrySummary::from(&registry))
    }

    pub fn organism_process_registry(&self) -> Option<WholeCellGenomeProcessRegistry> {
        self.organism_process_registry.clone().or_else(|| {
            self.organism_assets
                .as_ref()
                .map(compile_genome_process_registry)
        })
    }

    pub fn organism_expression_state(&self) -> Option<WholeCellOrganismExpressionState> {
        if self.organism_data.is_none() {
            None
        } else {
            Some(self.organism_expression.clone())
        }
    }

    pub fn chromosome_state(&self) -> WholeCellChromosomeState {
        self.chromosome_state.clone()
    }

    pub fn membrane_division_state(&self) -> WholeCellMembraneDivisionState {
        self.membrane_division_state.clone()
    }

    pub fn named_complexes_state(&self) -> Vec<WholeCellNamedComplexState> {
        self.named_complexes.clone()
    }

    pub fn complex_assembly_state(&self) -> WholeCellComplexAssemblyState {
        self.assembly_inventory()
    }

    fn surface_area_from_radius(radius_nm: f32) -> f32 {
        4.0 * PI * radius_nm * radius_nm
    }

    fn volume_from_radius(radius_nm: f32) -> f32 {
        4.0 / 3.0 * PI * radius_nm.powi(3)
    }

    fn refresh_spatial_fields(&mut self) {
        let total_voxels = self.lattice.total_voxels();
        if total_voxels == 0 {
            return;
        }
        let x_dim = self.lattice.x_dim.max(1);
        let y_dim = self.lattice.y_dim.max(1);
        let z_dim = self.lattice.z_dim.max(1);
        let center_x = (x_dim as f32 - 1.0) * 0.5;
        let center_y = (y_dim as f32 - 1.0) * 0.5;
        let center_z = (z_dim as f32 - 1.0) * 0.5;
        let axial_scale = center_x.max(1.0);
        let radial_y_scale = center_y.max(1.0);
        let radial_z_scale = center_z.max(1.0);
        let membrane_depth = (0.18
            + self
                .organism_data
                .as_ref()
                .map(|organism| organism.geometry.membrane_fraction.clamp(0.08, 0.32))
                .unwrap_or(0.16))
        .clamp(0.10, 0.35);
        let septum_width =
            (0.10 + 0.18 * self.membrane_division_state.septum_radius_fraction).clamp(0.06, 0.32);
        let septum_gain =
            (0.35 + 0.65 * self.membrane_division_state.septum_localization).clamp(0.20, 1.25);
        let division_progress = self.division_progress.clamp(0.0, 0.99);
        let cardiolipin_share = self.membrane_cardiolipin_share();
        let pole_width = (0.10 + 0.18 * cardiolipin_share + 0.08 * division_progress)
            .clamp(0.08, 0.26);
        let chromosome_radius_fraction = self
            .organism_data
            .as_ref()
            .map(|organism| {
                organism
                    .geometry
                    .chromosome_radius_fraction
                    .clamp(0.18, 0.70)
            })
            .unwrap_or(0.40);
        let axial_spread = (0.12
            + 0.22 * (1.0 - self.chromosome_state.compaction_fraction.clamp(0.0, 1.0))
            + 0.08 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
        .clamp(0.08, 0.45);
        let radial_spread = (0.10
            + 0.30 * chromosome_radius_fraction
            + 0.12 * (1.0 - self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)))
        .clamp(0.10, 0.42);
        let separation_fraction = (self.chromosome_separation_nm
            / (self.radius_nm * 2.2).max(self.lattice.voxel_size_nm))
        .clamp(0.0, 0.45);
        let left_center = 0.5 - 0.5 * separation_fraction;
        let right_center = 0.5 + 0.5 * separation_fraction;
        let occupancy_gain = (0.45
            + 0.55 * self.chromosome_state.compaction_fraction.clamp(0.0, 1.0)
            + 0.20 * self.chromosome_state.replicated_fraction.clamp(0.0, 1.0))
        .clamp(0.20, 1.35);

        let mut membrane = vec![0.0; total_voxels];
        let mut septum = vec![0.0; total_voxels];
        let mut nucleoid = vec![0.0; total_voxels];
        let mut membrane_band = vec![0.0; total_voxels];
        let mut poles = vec![0.0; total_voxels];

        for gid in 0..total_voxels {
            let z = gid / (y_dim * x_dim);
            let rem = gid - z * y_dim * x_dim;
            let y = rem / x_dim;
            let x = rem - y * x_dim;

            let x_norm = (x as f32 - center_x) / axial_scale;
            let y_norm = (y as f32 - center_y) / radial_y_scale;
            let z_norm = (z as f32 - center_z) / radial_z_scale;

            let boundary_distance = x_norm
                .abs()
                .max(y_norm.abs())
                .max(z_norm.abs())
                .clamp(0.0, 1.0);
            let membrane_adjacency =
                ((boundary_distance - (1.0 - membrane_depth)) / membrane_depth).clamp(0.0, 1.0);
            let septum_axis = (-0.5 * (x_norm / septum_width).powi(2)).exp();
            let radial_centering = (1.0 - 0.45 * (y_norm.abs().max(z_norm.abs()))).clamp(0.15, 1.0);
            let septum_zone =
                (septum_gain * septum_axis * membrane_adjacency * radial_centering).clamp(0.0, 1.0);
            let pole_axis =
                (-0.5 * ((1.0 - x_norm.abs()).max(0.0) / pole_width.max(1.0e-3)).powi(2)).exp();
            let pole_zone = (membrane_adjacency
                * pole_axis
                * (0.65 + 0.35 * radial_centering)
                * (0.65 + 0.35 * cardiolipin_share))
                .clamp(0.0, 1.0);
            let membrane_band_zone = (membrane_adjacency
                * (1.0 - pole_axis).clamp(0.0, 1.0)
                * (1.0 - 0.70 * septum_axis).clamp(0.0, 1.0)
                * radial_centering)
                .clamp(0.0, 1.0);

            let y_frac = (y as f32 + 0.5) / y_dim as f32 - 0.5;
            let z_frac = (z as f32 + 0.5) / z_dim as f32 - 0.5;
            let radial_distance = (y_frac.powi(2) + z_frac.powi(2)).sqrt();
            let radial_term = (radial_distance / radial_spread.max(1.0e-3)).powi(2);
            let x_frac = (x as f32 + 0.5) / x_dim as f32;
            let left_term = ((x_frac - left_center) / axial_spread.max(1.0e-3)).powi(2);
            let right_term = ((x_frac - right_center) / axial_spread.max(1.0e-3)).powi(2);
            let nucleoid_occupancy = (occupancy_gain
                * ((-0.5 * (left_term + radial_term)).exp()
                    + (-0.5 * (right_term + radial_term)).exp()))
            .clamp(0.0, 1.0);

            membrane[gid] = membrane_adjacency;
            septum[gid] = septum_zone;
            nucleoid[gid] = nucleoid_occupancy;
            membrane_band[gid] = membrane_band_zone;
            poles[gid] = pole_zone;
        }

        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::MembraneAdjacency, &membrane);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::SeptumZone, &septum);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::NucleoidOccupancy, &nucleoid);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::MembraneBandZone, &membrane_band);
        let _ = self
            .spatial_fields
            .set_field(IntracellularSpatialField::PoleZone, &poles);
    }

    fn spatial_species_mean(
        &self,
        species: IntracellularSpecies,
        field: IntracellularSpatialField,
    ) -> f32 {
        self.spatial_fields
            .weighted_mean_species(&self.lattice, species, field)
    }

    fn localized_nucleotide_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::Nucleotides,
            IntracellularSpatialField::NucleoidOccupancy,
        )
    }

    fn localized_membrane_precursor_pool_mm(&self) -> f32 {
        0.55 * self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::MembraneAdjacency,
        ) + 0.45
            * self.spatial_species_mean(
                IntracellularSpecies::MembranePrecursors,
                IntracellularSpatialField::SeptumZone,
            )
    }

    fn localized_membrane_atp_pool_mm(&self) -> f32 {
        0.65 * self.spatial_species_mean(
            IntracellularSpecies::ATP,
            IntracellularSpatialField::MembraneAdjacency,
        ) + 0.35
            * self.spatial_species_mean(
                IntracellularSpecies::ATP,
                IntracellularSpatialField::SeptumZone,
            )
    }

    fn localized_membrane_band_precursor_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::MembraneBandZone,
        )
    }

    fn localized_polar_precursor_pool_mm(&self) -> f32 {
        self.spatial_species_mean(
            IntracellularSpecies::MembranePrecursors,
            IntracellularSpatialField::PoleZone,
        )
    }

    fn restore_saved_state(&mut self, saved: WholeCellSavedState) -> Result<(), String> {
        let saved_scheduler_state = saved.scheduler_state.clone();
        self.program_name = saved.program_name.clone();
        self.contract = saved.contract.clone();
        self.provenance = saved.provenance.clone();
        self.organism_data_ref = saved.organism_data_ref.clone();
        self.organism_data = saved.organism_data.clone();
        self.organism_assets = saved.organism_assets.clone();
        self.organism_process_registry = saved.organism_process_registry.clone();
        self.organism_expression = saved.organism_expression.clone();
        self.chromosome_state = saved.chromosome_state.clone();
        self.membrane_division_state = saved.membrane_division_state.clone();
        self.organism_species = saved.organism_species.clone();
        self.organism_reactions = saved.organism_reactions.clone();
        self.complex_assembly = saved.complex_assembly;
        self.named_complexes = saved.named_complexes.clone();
        self.lattice
            .set_species(IntracellularSpecies::ATP, &saved.lattice.atp)?;
        self.lattice
            .set_species(IntracellularSpecies::AminoAcids, &saved.lattice.amino_acids)?;
        self.lattice.set_species(
            IntracellularSpecies::Nucleotides,
            &saved.lattice.nucleotides,
        )?;
        self.lattice.set_species(
            IntracellularSpecies::MembranePrecursors,
            &saved.lattice.membrane_precursors,
        )?;
        if let Some(spatial) = saved.spatial_fields {
            let _ = self.spatial_fields.set_field(
                IntracellularSpatialField::MembraneAdjacency,
                &spatial.membrane_adjacency,
            );
            let _ = self
                .spatial_fields
                .set_field(IntracellularSpatialField::SeptumZone, &spatial.septum_zone);
            let _ = self.spatial_fields.set_field(
                IntracellularSpatialField::NucleoidOccupancy,
                &spatial.nucleoid_occupancy,
            );
            let _ = self.spatial_fields.set_field(
                IntracellularSpatialField::MembraneBandZone,
                &spatial.membrane_band_zone,
            );
            let _ = self
                .spatial_fields
                .set_field(IntracellularSpatialField::PoleZone, &spatial.pole_zone);
        }
        self.sync_from_lattice();

        self.time_ms = saved.core.time_ms.max(0.0);
        self.step_count = saved.core.step_count;
        self.adp_mm = saved.core.adp_mm.max(0.0);
        self.glucose_mm = saved.core.glucose_mm.max(0.0);
        self.oxygen_mm = saved.core.oxygen_mm.max(0.0);
        self.ftsz = saved.core.ftsz.max(0.0);
        self.dnaa = saved.core.dnaa.max(0.0);
        self.active_ribosomes = saved.core.active_ribosomes.max(0.0);
        self.active_rnap = saved.core.active_rnap.max(0.0);
        self.genome_bp = saved.core.genome_bp.max(1);
        self.replicated_bp = saved.core.replicated_bp.min(self.genome_bp);
        self.chromosome_separation_nm = saved.core.chromosome_separation_nm.max(0.0);
        self.radius_nm = saved.core.radius_nm.max(50.0);
        self.surface_area_nm2 = saved.core.surface_area_nm2.max(1.0);
        self.volume_nm3 = saved.core.volume_nm3.max(1.0);
        self.division_progress = saved.core.division_progress.clamp(0.0, 1.0);
        self.metabolic_load = saved.core.metabolic_load.max(0.1);
        self.quantum_profile = saved.core.quantum_profile.normalized();
        if self.organism_data.is_some() {
            self.genome_bp = self
                .organism_data
                .as_ref()
                .map(|organism| organism.chromosome_length_bp.max(1))
                .unwrap_or(self.genome_bp);
            self.replicated_bp = self.replicated_bp.min(self.genome_bp);
            if self.organism_assets.is_none() {
                self.organism_assets = self
                    .organism_data
                    .as_ref()
                    .map(compile_genome_asset_package);
            }
            if self.organism_process_registry.is_none() {
                self.organism_process_registry = self
                    .organism_assets
                    .as_ref()
                    .map(compile_genome_process_registry);
            }
            if self.organism_expression.transcription_units.is_empty() {
                self.refresh_organism_expression_state();
            }
            if self.organism_species.is_empty() || self.organism_reactions.is_empty() {
                self.initialize_runtime_process_state();
            }
            if self.named_complexes.is_empty() {
                self.initialize_named_complexes_state();
            }
            self.chromosome_state = if self.chromosome_state.chromosome_length_bp > 1
                || !self.chromosome_state.loci.is_empty()
            {
                self.normalize_chromosome_state(self.chromosome_state.clone())
            } else {
                self.seeded_chromosome_state()
            };
            self.synchronize_chromosome_summary();
            self.membrane_division_state =
                if self.membrane_division_state.preferred_membrane_area_nm2 > 1.0 {
                    self.normalize_membrane_division_state(self.membrane_division_state.clone())
                } else {
                    self.seeded_membrane_division_state()
                };
            self.synchronize_membrane_division_summary();
            if !self.named_complexes.is_empty() {
                if let Some(assets) = self.organism_assets.as_ref() {
                    self.complex_assembly = self.aggregate_named_complex_assembly_state(assets);
                }
            } else if self.complex_assembly.total_complexes() <= 1.0e-6 {
                self.initialize_complex_assembly_state();
            }
        } else {
            self.organism_expression = WholeCellOrganismExpressionState::default();
            self.organism_process_registry = None;
            self.organism_species.clear();
            self.organism_reactions.clear();
            self.named_complexes.clear();
            self.chromosome_state = self.seeded_chromosome_state();
            self.synchronize_chromosome_summary();
            self.membrane_division_state = self.seeded_membrane_division_state();
            self.synchronize_membrane_division_summary();
            if self.complex_assembly.total_complexes() <= 1.0e-6 {
                self.initialize_complex_assembly_state();
            }
        }
        self.refresh_spatial_fields();
        self.refresh_rdme_drive_fields();

        self.disable_local_chemistry();
        if let Some(local) = saved.local_chemistry {
            self.enable_local_chemistry(
                local.x_dim,
                local.y_dim,
                local.z_dim,
                local.voxel_size_au,
                local.use_gpu,
            );
            if local.enable_default_syn3a_subsystems {
                self.enable_default_syn3a_subsystems();
            }
            if !local.scheduled_subsystem_probes.is_empty() {
                self.clear_syn3a_subsystem_probes();
                for probe in local.scheduled_subsystem_probes {
                    self.schedule_syn3a_subsystem_probe(probe.preset, probe.interval_steps);
                }
            }
        }

        self.chemistry_report = saved.chemistry_report;
        self.chemistry_site_reports = saved.chemistry_site_reports;
        self.last_md_probe = saved.last_md_probe;
        self.scheduled_subsystem_probes = saved.scheduled_subsystem_probes;
        self.subsystem_states = if saved.subsystem_states.is_empty() {
            Syn3ASubsystemPreset::all()
                .iter()
                .copied()
                .map(WholeCellSubsystemState::new)
                .collect()
        } else {
            saved.subsystem_states
        };
        self.md_translation_scale = Self::finite_scale(saved.md_translation_scale, 1.0, 0.70, 1.45);
        self.md_membrane_scale = Self::finite_scale(saved.md_membrane_scale, 1.0, 0.70, 1.45);
        self.scheduler_state =
            Self::normalized_scheduler_state(&self.config, saved_scheduler_state);
        if self
            .scheduler_state
            .stage_clocks
            .iter()
            .all(|clock| clock.run_count == 0)
        {
            self.refresh_multirate_scheduler();
        }
        self.initialize_surrogate_pool_diagnostics();
        Ok(())
    }

    /// Create a simulator with JCVI-syn3A-like defaults.
    pub fn new(config: WholeCellConfig) -> Self {
        let scheduler_state = Self::build_scheduler_state(&config);
        let backend = if config.use_gpu && gpu::has_gpu() {
            WholeCellBackend::Metal
        } else {
            WholeCellBackend::Cpu
        };
        let spatial_dims = (config.x_dim, config.y_dim, config.z_dim);

        #[cfg(target_os = "macos")]
        let gpu = if backend == WholeCellBackend::Metal {
            GpuContext::new().ok()
        } else {
            None
        };

        let backend = {
            #[cfg(target_os = "macos")]
            {
                if backend == WholeCellBackend::Metal && gpu.is_some() {
                    WholeCellBackend::Metal
                } else {
                    WholeCellBackend::Cpu
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                WholeCellBackend::Cpu
            }
        };

        let mut lattice = IntracellularLattice::new(
            config.x_dim,
            config.y_dim,
            config.z_dim,
            config.voxel_size_nm,
        );
        lattice.fill_species(IntracellularSpecies::ATP, 1.20);
        lattice.fill_species(IntracellularSpecies::AminoAcids, 0.95);
        lattice.fill_species(IntracellularSpecies::Nucleotides, 0.80);
        lattice.fill_species(IntracellularSpecies::MembranePrecursors, 0.35);

        let radius_nm = 200.0;
        let surface_area_nm2 = Self::surface_area_from_radius(radius_nm);
        let volume_nm3 = Self::volume_from_radius(radius_nm);

        let mut simulator = Self {
            config,
            backend,
            program_name: None,
            contract: WholeCellContractSchema::default(),
            provenance: WholeCellProvenance::default(),
            organism_data_ref: None,
            organism_data: None,
            organism_assets: None,
            organism_process_registry: None,
            organism_expression: WholeCellOrganismExpressionState::default(),
            organism_species: Vec::new(),
            organism_reactions: Vec::new(),
            complex_assembly: WholeCellComplexAssemblyState::default(),
            named_complexes: Vec::new(),
            #[cfg(target_os = "macos")]
            gpu,
            lattice,
            spatial_fields: IntracellularSpatialState::new(
                spatial_dims.0,
                spatial_dims.1,
                spatial_dims.2,
            ),
            rdme_drive_fields: WholeCellRdmeDriveState::new(
                spatial_dims.0,
                spatial_dims.1,
                spatial_dims.2,
            ),
            time_ms: 0.0,
            step_count: 0,
            atp_mm: 1.20,
            amino_acids_mm: 0.95,
            nucleotides_mm: 0.80,
            membrane_precursors_mm: 0.35,
            adp_mm: 0.30,
            glucose_mm: 1.0,
            oxygen_mm: 0.85,
            ftsz: 0.0,
            dnaa: 0.0,
            active_ribosomes: 0.0,
            active_rnap: 0.0,
            genome_bp: 543_000,
            replicated_bp: 0,
            chromosome_separation_nm: 40.0,
            chromosome_state: WholeCellChromosomeState::default(),
            membrane_division_state: WholeCellMembraneDivisionState::default(),
            radius_nm,
            surface_area_nm2,
            volume_nm3,
            division_progress: 0.0,
            metabolic_load: 1.0,
            quantum_profile: WholeCellQuantumProfile::default(),
            chemistry_bridge: None,
            chemistry_report: LocalChemistryReport::default(),
            chemistry_site_reports: Vec::new(),
            last_md_probe: None,
            scheduled_subsystem_probes: Vec::new(),
            subsystem_states: Syn3ASubsystemPreset::all()
                .iter()
                .copied()
                .map(WholeCellSubsystemState::new)
                .collect(),
            scheduler_state,
            md_translation_scale: 1.0,
            md_membrane_scale: 1.0,
        };
        simulator.sync_from_lattice();
        simulator.chromosome_state = simulator.seeded_chromosome_state();
        simulator.synchronize_chromosome_summary();
        simulator.membrane_division_state = simulator.seeded_membrane_division_state();
        simulator.synchronize_membrane_division_summary();
        simulator.refresh_spatial_fields();
        simulator.refresh_rdme_drive_fields();
        simulator.refresh_organism_expression_state();
        simulator.initialize_complex_assembly_state();
        simulator.initialize_runtime_process_state();
        simulator.refresh_multirate_scheduler();
        simulator.initialize_surrogate_pool_diagnostics();
        simulator
    }

    /// Create a simulator from a data-driven whole-cell program spec.
    pub fn from_program_spec(spec: WholeCellProgramSpec) -> Self {
        let config = spec.config.clone();
        let local_chemistry = spec.local_chemistry.clone();
        let mut simulator = Self::new(config);
        simulator.program_name = spec.program_name.clone();
        simulator.contract = spec.contract.clone();
        simulator.provenance = spec.provenance.clone();
        simulator.organism_data_ref = spec.organism_data_ref.clone();
        simulator.organism_data = spec.organism_data.clone();
        simulator.organism_assets = spec.organism_assets.clone().or_else(|| {
            simulator
                .organism_data
                .as_ref()
                .map(compile_genome_asset_package)
        });
        simulator.organism_process_registry =
            spec.organism_process_registry.clone().or_else(|| {
                simulator
                    .organism_assets
                    .as_ref()
                    .map(compile_genome_process_registry)
            });

        simulator
            .lattice
            .fill_species(IntracellularSpecies::ATP, spec.initial_lattice.atp.max(0.0));
        simulator.lattice.fill_species(
            IntracellularSpecies::AminoAcids,
            spec.initial_lattice.amino_acids.max(0.0),
        );
        simulator.lattice.fill_species(
            IntracellularSpecies::Nucleotides,
            spec.initial_lattice.nucleotides.max(0.0),
        );
        simulator.lattice.fill_species(
            IntracellularSpecies::MembranePrecursors,
            spec.initial_lattice.membrane_precursors.max(0.0),
        );
        simulator.sync_from_lattice();

        simulator.adp_mm = spec.initial_state.adp_mm.max(0.0);
        simulator.glucose_mm = spec.initial_state.glucose_mm.max(0.0);
        simulator.oxygen_mm = spec.initial_state.oxygen_mm.max(0.0);
        simulator.genome_bp = spec.initial_state.genome_bp.max(1);
        simulator.replicated_bp = spec.initial_state.replicated_bp.min(simulator.genome_bp);
        simulator.chromosome_separation_nm = spec.initial_state.chromosome_separation_nm.max(0.0);
        simulator.radius_nm = spec.initial_state.radius_nm.max(50.0);
        simulator.surface_area_nm2 = Self::surface_area_from_radius(simulator.radius_nm);
        simulator.volume_nm3 = Self::volume_from_radius(simulator.radius_nm);
        simulator.division_progress = spec.initial_state.division_progress.clamp(0.0, 1.0);
        simulator.metabolic_load = spec.initial_state.metabolic_load.max(0.1);
        simulator.quantum_profile = spec.quantum_profile.normalized();
        simulator.apply_organism_data_initialization();
        simulator.chromosome_state = spec
            .chromosome_state
            .clone()
            .map(|state| simulator.normalize_chromosome_state(state))
            .unwrap_or_else(|| simulator.seeded_chromosome_state());
        simulator.synchronize_chromosome_summary();
        simulator.membrane_division_state = spec
            .membrane_division_state
            .clone()
            .map(|state| simulator.normalize_membrane_division_state(state))
            .unwrap_or_else(|| simulator.seeded_membrane_division_state());
        simulator.synchronize_membrane_division_summary();
        simulator.refresh_spatial_fields();
        simulator.refresh_rdme_drive_fields();

        simulator.disable_local_chemistry();
        if let Some(local) = local_chemistry {
            simulator.enable_local_chemistry(
                local.x_dim,
                local.y_dim,
                local.z_dim,
                local.voxel_size_au,
                local.use_gpu,
            );
            if local.enable_default_syn3a_subsystems {
                simulator.enable_default_syn3a_subsystems();
            }
            if !local.scheduled_subsystem_probes.is_empty() {
                simulator.clear_syn3a_subsystem_probes();
                for probe in local.scheduled_subsystem_probes {
                    simulator.schedule_syn3a_subsystem_probe(probe.preset, probe.interval_steps);
                }
            }
        }

        simulator.refresh_organism_expression_state();
        simulator.initialize_complex_assembly_state();
        simulator.initialize_runtime_process_state();
        simulator.refresh_multirate_scheduler();
        simulator.initialize_surrogate_pool_diagnostics();
        simulator
    }

    /// Create a simulator from bundled Syn3A reference data.
    pub fn bundled_syn3a_reference() -> Result<Self, String> {
        bundled_syn3a_program_spec().map(Self::from_program_spec)
    }

    /// Create a simulator from a native organism bundle manifest path.
    pub fn from_bundle_manifest_path(manifest_path: &str) -> Result<Self, String> {
        compile_program_spec_from_bundle_manifest_path(manifest_path).map(Self::from_program_spec)
    }

    /// Return the bundled Syn3A reference spec JSON.
    pub fn bundled_syn3a_reference_spec_json() -> &'static str {
        bundled_syn3a_program_spec_json()
    }

    /// Return the bundled Syn3A organism descriptor JSON.
    pub fn bundled_syn3a_organism_spec_json() -> &'static str {
        crate::whole_cell_data::bundled_syn3a_organism_spec_json()
    }

    /// Return the bundled Syn3A compiled genome asset package JSON.
    pub fn bundled_syn3a_genome_asset_package_json() -> Result<&'static str, String> {
        bundled_syn3a_genome_asset_package_json()
    }

    /// Return the bundled Syn3A compiled process registry JSON.
    pub fn bundled_syn3a_process_registry_json() -> Result<&'static str, String> {
        crate::whole_cell_data::bundled_syn3a_process_registry_json()
    }

    /// Create a simulator from a JSON-encoded whole-cell program spec.
    pub fn from_program_spec_json(spec_json: &str) -> Result<Self, String> {
        parse_program_spec_json(spec_json).map(Self::from_program_spec)
    }

    /// Serialize the current simulator state into a restartable JSON payload.
    pub fn save_state_json(&self) -> Result<String, String> {
        let saved = WholeCellSavedState {
            program_name: self
                .program_name
                .clone()
                .or_else(|| Some("native_runtime".to_string())),
            contract: self.contract.clone(),
            provenance: {
                let mut provenance = self.provenance.clone();
                provenance.backend = Some(self.backend.as_str().to_string());
                provenance
            },
            organism_data_ref: self.organism_data_ref.clone(),
            organism_data: self.organism_data.clone(),
            organism_assets: self.organism_assets.clone(),
            organism_process_registry: self.organism_process_registry.clone(),
            organism_expression: self.organism_expression.clone(),
            chromosome_state: self.chromosome_state.clone(),
            membrane_division_state: self.membrane_division_state.clone(),
            organism_species: self.organism_species.clone(),
            organism_reactions: self.organism_reactions.clone(),
            complex_assembly: self.complex_assembly,
            named_complexes: self.named_complexes.clone(),
            scheduler_state: self.scheduler_state.clone(),
            config: self.config.clone(),
            core: WholeCellSavedCoreState {
                time_ms: self.time_ms,
                step_count: self.step_count,
                adp_mm: self.adp_mm,
                glucose_mm: self.glucose_mm,
                oxygen_mm: self.oxygen_mm,
                ftsz: self.ftsz,
                dnaa: self.dnaa,
                active_ribosomes: self.active_ribosomes,
                active_rnap: self.active_rnap,
                genome_bp: self.genome_bp,
                replicated_bp: self.replicated_bp,
                chromosome_separation_nm: self.chromosome_separation_nm,
                radius_nm: self.radius_nm,
                surface_area_nm2: self.surface_area_nm2,
                volume_nm3: self.volume_nm3,
                division_progress: self.division_progress,
                metabolic_load: self.metabolic_load,
                quantum_profile: self.quantum_profile,
            },
            lattice: WholeCellLatticeState {
                atp: self.lattice.clone_species(IntracellularSpecies::ATP),
                amino_acids: self.lattice.clone_species(IntracellularSpecies::AminoAcids),
                nucleotides: self
                    .lattice
                    .clone_species(IntracellularSpecies::Nucleotides),
                membrane_precursors: self
                    .lattice
                    .clone_species(IntracellularSpecies::MembranePrecursors),
            },
            spatial_fields: Some(WholeCellSpatialFieldState {
                membrane_adjacency: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::MembraneAdjacency),
                septum_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::SeptumZone),
                nucleoid_occupancy: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::NucleoidOccupancy),
                membrane_band_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::MembraneBandZone),
                pole_zone: self
                    .spatial_fields
                    .clone_field(IntracellularSpatialField::PoleZone),
            }),
            local_chemistry: self.chemistry_bridge.as_ref().map(|bridge| {
                let (x_dim, y_dim, z_dim) = bridge.lattice_shape();
                WholeCellLocalChemistrySpec {
                    x_dim,
                    y_dim,
                    z_dim,
                    voxel_size_au: bridge.voxel_size_au(),
                    use_gpu: bridge.use_gpu_backend(),
                    enable_default_syn3a_subsystems: false,
                    scheduled_subsystem_probes: self.scheduled_subsystem_probes.clone(),
                }
            }),
            chemistry_report: self.chemistry_report,
            chemistry_site_reports: self.chemistry_site_reports.clone(),
            last_md_probe: self.last_md_probe,
            scheduled_subsystem_probes: self.scheduled_subsystem_probes.clone(),
            subsystem_states: self.subsystem_states.clone(),
            md_translation_scale: self.md_translation_scale,
            md_membrane_scale: self.md_membrane_scale,
        };
        saved_state_to_json(&saved)
    }

    /// Restore a simulator from a JSON-encoded saved state.
    pub fn from_saved_state_json(state_json: &str) -> Result<Self, String> {
        let saved = parse_saved_state_json(state_json)?;
        let mut simulator = Self::new(saved.config.clone());
        simulator.restore_saved_state(saved)?;
        Ok(simulator)
    }

    /// Step the simulator by one configured time quantum.
    pub fn step(&mut self) {
        let base_dt = self.config.dt_ms;
        self.refresh_organism_expression_state();
        self.refresh_multirate_scheduler();
        self.update_complex_assembly_state(base_dt);

        if self.stage_due(WholeCellSolverStage::AtomisticRefinement) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::AtomisticRefinement, base_dt);
            self.update_local_chemistry(stage_dt);
            self.record_stage_run(WholeCellSolverStage::AtomisticRefinement, stage_dt);
        }

        if self.stage_due(WholeCellSolverStage::Rdme) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::Rdme, base_dt);
            self.rdme_stage(stage_dt);
            self.record_stage_run(WholeCellSolverStage::Rdme, stage_dt);
        }

        if self.stage_due(WholeCellSolverStage::Cme) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::Cme, base_dt);
            self.cme_stage(stage_dt);
            self.record_stage_run(WholeCellSolverStage::Cme, stage_dt);
        }

        if self.stage_due(WholeCellSolverStage::Ode) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::Ode, base_dt);
            self.ode_stage(stage_dt);
            self.record_stage_run(WholeCellSolverStage::Ode, stage_dt);
        }

        if self.stage_due(WholeCellSolverStage::ChromosomeBd) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::ChromosomeBd, base_dt);
            self.bd_stage(stage_dt);
            self.record_stage_run(WholeCellSolverStage::ChromosomeBd, stage_dt);
        }

        if self.stage_due(WholeCellSolverStage::Geometry) {
            let stage_dt = self.stage_dt_ms(WholeCellSolverStage::Geometry, base_dt);
            self.geometry_stage(stage_dt);
            self.record_stage_run(WholeCellSolverStage::Geometry, stage_dt);
        }

        self.time_ms += base_dt;
        self.step_count += 1;
    }

    /// Run the simulator for a fixed number of steps.
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Current backend name.
    pub fn backend(&self) -> WholeCellBackend {
        self.backend
    }

    /// Current simulation time in milliseconds.
    pub fn time_ms(&self) -> f32 {
        self.time_ms
    }

    /// Number of integration steps completed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Expose the current voxel dimensions.
    pub fn lattice_shape(&self) -> (usize, usize, usize) {
        (self.lattice.x_dim, self.lattice.y_dim, self.lattice.z_dim)
    }

    /// Update metabolic load. Values >1.0 increase sink terms.
    pub fn set_metabolic_load(&mut self, load: f32) {
        self.metabolic_load = load.max(0.1);
    }

    /// Update the quantum correction profile used by the coarse stage models.
    pub fn set_quantum_profile(&mut self, profile: WholeCellQuantumProfile) {
        self.quantum_profile = profile.normalized();
    }

    /// Current quantum correction profile.
    pub fn quantum_profile(&self) -> WholeCellQuantumProfile {
        self.quantum_profile
    }

    /// Enable the local chemistry lattice submodel.
    pub fn enable_local_chemistry(
        &mut self,
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) {
        self.chemistry_bridge = Some(WholeCellChemistryBridge::new(
            x_dim,
            y_dim,
            z_dim,
            voxel_size_au,
            use_gpu,
        ));
    }

    /// Disable the local chemistry submodel.
    pub fn disable_local_chemistry(&mut self) {
        self.chemistry_bridge = None;
        self.chemistry_report = LocalChemistryReport::default();
        self.chemistry_site_reports.clear();
        self.last_md_probe = None;
        self.scheduled_subsystem_probes.clear();
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Latest local chemistry report, if enabled.
    pub fn local_chemistry_report(&self) -> Option<LocalChemistryReport> {
        self.chemistry_bridge
            .as_ref()
            .map(|_| self.chemistry_report)
    }

    /// Latest per-subsystem local chemistry reports, if enabled.
    pub fn local_chemistry_sites(&self) -> Vec<LocalChemistrySiteReport> {
        if self.chemistry_bridge.is_some() {
            self.chemistry_site_reports.clone()
        } else {
            Vec::new()
        }
    }

    /// Latest localized MD probe report.
    pub fn last_md_probe(&self) -> Option<LocalMDProbeReport> {
        self.last_md_probe
    }

    /// Current persistent coupling state for each Syn3A subsystem preset.
    pub fn subsystem_states(&self) -> Vec<WholeCellSubsystemState> {
        self.subsystem_states.clone()
    }

    /// Run a localized MD probe through the optional chemistry bridge.
    pub fn run_local_md_probe(
        &mut self,
        request: LocalMDProbeRequest,
    ) -> Option<LocalMDProbeReport> {
        let report = {
            let bridge = self.chemistry_bridge.as_mut()?;
            bridge.run_md_probe(request)
        };
        if let Some(preset) = Self::preset_for_site(report.site) {
            self.apply_probe_to_subsystem(preset, report);
        }
        self.last_md_probe = Some(report);
        self.md_translation_scale = report.recommended_translation_scale;
        self.md_membrane_scale = report.recommended_membrane_scale;
        Some(report)
    }

    /// Run a named Syn3A subsystem probe using the default request.
    pub fn run_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<LocalMDProbeReport> {
        self.run_local_md_probe(preset.default_probe_request())
    }

    /// Generate lower-scale calibration sweep samples from the active chemistry bridge.
    pub fn derivation_calibration_samples(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Vec<WholeCellDerivationCalibrationSample> {
        match self.chemistry_bridge.as_mut() {
            Some(bridge) => bridge.derivation_calibration_samples(dt_ms, equilibration_steps),
            None => Vec::new(),
        }
    }

    /// Fit descriptor-to-signature derivation gains against bridge sweep outputs.
    pub fn fit_derivation_calibration(
        &mut self,
        dt_ms: f32,
        equilibration_steps: usize,
    ) -> Result<Option<WholeCellDerivationCalibrationFit>, String> {
        match self.chemistry_bridge.as_mut() {
            Some(bridge) => bridge
                .fit_derivation_calibration(dt_ms, equilibration_steps)
                .map(Some),
            None => Ok(None),
        }
    }

    /// Schedule a Syn3A subsystem probe to run periodically.
    pub fn schedule_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
        interval_steps: u64,
    ) {
        let interval_steps = interval_steps.max(1);
        if let Some(existing) = self
            .scheduled_subsystem_probes
            .iter_mut()
            .find(|probe| probe.preset == preset)
        {
            existing.interval_steps = interval_steps;
            return;
        }
        self.scheduled_subsystem_probes
            .push(ScheduledSubsystemProbe {
                preset,
                interval_steps,
            });
    }

    /// Clear all scheduled Syn3A subsystem probes.
    pub fn clear_syn3a_subsystem_probes(&mut self) {
        self.scheduled_subsystem_probes.clear();
        for state in &mut self.subsystem_states {
            *state = WholeCellSubsystemState::new(state.preset);
            state.apply_chemistry_report(self.chemistry_report);
        }
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Return a copy of the scheduled subsystem probes.
    pub fn scheduled_syn3a_subsystem_probes(&self) -> Vec<ScheduledSubsystemProbe> {
        self.scheduled_subsystem_probes.clone()
    }

    /// Enable the default set of Syn3A subsystem probes.
    pub fn enable_default_syn3a_subsystems(&mut self) {
        if self.chemistry_bridge.is_none() {
            self.enable_local_chemistry(12, 12, 6, 0.5, true);
        }
        self.clear_syn3a_subsystem_probes();
        for preset in Syn3ASubsystemPreset::all() {
            self.schedule_syn3a_subsystem_probe(*preset, preset.default_interval_steps());
        }
    }

    /// Mean ATP concentration across the cell.
    pub fn atp_mm(&self) -> f32 {
        self.atp_mm
    }

    /// FtsZ pool used for division ring assembly.
    pub fn ftsz(&self) -> f32 {
        self.ftsz
    }

    /// Current chromosome replication progress in base pairs.
    pub fn replicated_bp(&self) -> u32 {
        self.replicated_bp
    }

    /// Current division progress (0-1).
    pub fn division_progress(&self) -> f32 {
        self.division_progress
    }

    /// Return a copied ATP lattice channel.
    pub fn atp_lattice(&self) -> Vec<f32> {
        self.lattice.clone_species(IntracellularSpecies::ATP)
    }

    /// Seed a hotspot into a species channel.
    pub fn add_hotspot(
        &mut self,
        species: IntracellularSpecies,
        x: usize,
        y: usize,
        z: usize,
        delta: f32,
    ) {
        self.lattice.add_hotspot(species, x, y, z, delta);
        self.sync_from_lattice();
    }

    /// Snapshot the coarse state for diagnostics or bindings.
    pub fn snapshot(&self) -> WholeCellSnapshot {
        WholeCellSnapshot {
            backend: self.backend,
            time_ms: self.time_ms,
            step_count: self.step_count,
            atp_mm: self.atp_mm,
            amino_acids_mm: self.amino_acids_mm,
            nucleotides_mm: self.nucleotides_mm,
            membrane_precursors_mm: self.membrane_precursors_mm,
            adp_mm: self.adp_mm,
            glucose_mm: self.glucose_mm,
            oxygen_mm: self.oxygen_mm,
            ftsz: self.ftsz,
            dnaa: self.dnaa,
            active_ribosomes: self.active_ribosomes,
            active_rnap: self.active_rnap,
            genome_bp: self.genome_bp,
            replicated_bp: self.replicated_bp,
            chromosome_separation_nm: self.chromosome_separation_nm,
            chromosome_state: self.chromosome_state.clone(),
            membrane_division_state: self.membrane_division_state.clone(),
            radius_nm: self.radius_nm,
            surface_area_nm2: self.surface_area_nm2,
            volume_nm3: self.volume_nm3,
            division_progress: self.division_progress,
            quantum_profile: self.quantum_profile,
            local_chemistry: self.local_chemistry_report(),
            local_chemistry_sites: self.local_chemistry_sites(),
            local_md_probe: self.last_md_probe,
            subsystem_states: self.subsystem_states(),
            scheduler_state: self.scheduler_state.clone(),
        }
    }

    fn update_local_chemistry(&mut self, dt: f32) {
        let snapshot = self.snapshot();
        let scheduled_probes = self.scheduled_subsystem_probes.clone();
        let Some((chemistry_report, chemistry_site_reports, last_md_report, due_reports)) = ({
            let Some(ref mut bridge) = self.chemistry_bridge else {
                return;
            };
            let chemistry_report = bridge.step_with_snapshot((dt * 2.0).max(0.1), Some(&snapshot));
            let chemistry_site_reports = bridge.site_reports();
            let last_md_report = bridge.last_md_report();
            let mut due_reports = Vec::new();
            for scheduled in &scheduled_probes {
                if self.step_count % scheduled.interval_steps == 0 {
                    let report = bridge.run_md_probe(scheduled.preset.default_probe_request());
                    due_reports.push((scheduled.preset, report));
                }
            }
            Some((
                chemistry_report,
                chemistry_site_reports,
                last_md_report,
                due_reports,
            ))
        }) else {
            return;
        };

        self.chemistry_report = chemistry_report;
        self.chemistry_site_reports = chemistry_site_reports;
        self.refresh_subsystem_chemistry_state();
        if scheduled_probes.is_empty() {
            self.last_md_probe = last_md_report;
            return;
        }

        if due_reports.is_empty() {
            if let Some(report) = last_md_report {
                self.last_md_probe = Some(report);
            }
        } else {
            for (preset, report) in &due_reports {
                self.apply_probe_to_subsystem(*preset, *report);
            }
            let count = due_reports.len() as f32;
            self.md_translation_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_translation_scale)
                .sum::<f32>()
                / count;
            self.md_membrane_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_membrane_scale)
                .sum::<f32>()
                / count;
            self.last_md_probe = due_reports.last().map(|(_, report)| *report);
        }
    }

    fn md_translation_scale(&self) -> f32 {
        self.md_translation_scale
    }

    fn md_membrane_scale(&self) -> f32 {
        self.md_membrane_scale
    }

    fn preset_for_site(
        site: crate::whole_cell_submodels::WholeCellChemistrySite,
    ) -> Option<Syn3ASubsystemPreset> {
        match site {
            crate::whole_cell_submodels::WholeCellChemistrySite::AtpSynthaseBand => {
                Some(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::RibosomeCluster => {
                Some(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::ChromosomeTrack => {
                Some(Syn3ASubsystemPreset::ReplisomeTrack)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::SeptumRing => {
                Some(Syn3ASubsystemPreset::FtsZSeptumRing)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::Cytosol => None,
        }
    }

    fn subsystem_state(&self, preset: Syn3ASubsystemPreset) -> WholeCellSubsystemState {
        self.subsystem_states
            .iter()
            .copied()
            .find(|state| state.preset == preset)
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset))
    }

    fn subsystem_state_mut(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<&mut WholeCellSubsystemState> {
        self.subsystem_states
            .iter_mut()
            .find(|state| state.preset == preset)
    }

    fn refresh_subsystem_chemistry_state(&mut self) {
        for state in &mut self.subsystem_states {
            if let Some(report) = self
                .chemistry_site_reports
                .iter()
                .find(|report| report.preset == state.preset)
                .copied()
            {
                state.apply_site_report(report);
            } else {
                state.apply_chemistry_report(self.chemistry_report);
            }
        }
    }

    fn apply_probe_to_subsystem(
        &mut self,
        preset: Syn3ASubsystemPreset,
        report: LocalMDProbeReport,
    ) {
        let step_count = self.step_count;
        if let Some(state) = self.subsystem_state_mut(preset) {
            state.apply_probe_report(report, step_count);
        }
    }

    fn atp_band_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .atp_scale
    }

    fn ribosome_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .translation_scale
    }

    fn replisome_replication_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .replication_scale
    }

    fn replisome_segregation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .segregation_scale
    }

    fn ftsz_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .translation_scale
    }

    fn ftsz_constriction_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .constriction_scale
    }

    fn membrane_assembly_scale(&self) -> f32 {
        let atp_band = self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let septum = self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing);
        (0.55 * atp_band.membrane_scale + 0.45 * septum.membrane_scale).clamp(0.70, 1.45)
    }

    fn localized_supply_scale(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 1.0;
        }
        let mean_satisfaction = self
            .chemistry_site_reports
            .iter()
            .map(|report| report.demand_satisfaction)
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32;
        Self::finite_scale(mean_satisfaction, 1.0, 0.55, 1.0)
    }

    fn localized_resource_pressure(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 0.0;
        }
        self.chemistry_site_reports
            .iter()
            .map(|report| {
                0.45 * report.substrate_draw
                    + 0.55 * report.energy_draw
                    + 0.50 * report.biosynthetic_draw
                    + 0.60 * report.byproduct_load
                    + (1.0 - report.demand_satisfaction).max(0.0) * 1.2
            })
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32
    }

    fn effective_metabolic_load(&self) -> f32 {
        let supply_scale = self.localized_supply_scale();
        let pressure = self.localized_resource_pressure();
        let local_multiplier =
            (1.0 + pressure * 0.16 + (1.0 - supply_scale).max(0.0) * 0.35).clamp(1.0, 2.2);
        let organism_multiplier = self
            .organism_expression
            .metabolic_burden_scale
            .clamp(0.85, 1.65);
        self.metabolic_load.max(0.1) * local_multiplier * organism_multiplier
    }

    fn rdme_stage(&mut self, dt: f32) {
        self.refresh_spatial_fields();
        self.refresh_rdme_drive_fields();
        let effective_metabolic_load = self.effective_metabolic_load();
        let rdme_context = WholeCellRdmeContext {
            metabolic_load: effective_metabolic_load,
        };
        match self.backend {
            WholeCellBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(ref gpu) = self.gpu {
                        dispatch_whole_cell_rdme(
                            gpu,
                            &mut self.lattice,
                            &self.spatial_fields,
                            &self.rdme_drive_fields,
                            dt,
                            rdme_context,
                        );
                    } else {
                        cpu_whole_cell_rdme(
                            &mut self.lattice,
                            &self.spatial_fields,
                            &self.rdme_drive_fields,
                            dt,
                            rdme_context,
                        );
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    cpu_whole_cell_rdme(
                        &mut self.lattice,
                        &self.spatial_fields,
                        &self.rdme_drive_fields,
                        dt,
                        rdme_context,
                    );
                }
            }
            WholeCellBackend::Cpu => cpu_whole_cell_rdme(
                &mut self.lattice,
                &self.spatial_fields,
                &self.rdme_drive_fields,
                dt,
                rdme_context,
            ),
        }
        self.sync_from_lattice();
    }

    fn cme_stage(&mut self, dt: f32) {
        self.refresh_organism_expression_state();
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let organism_scales = self.organism_process_scales();
        let transcription_flux =
            TRANSCRIPTION_FLUX_RULE.evaluate(scalar) * organism_scales.transcription_scale;
        let translation_flux =
            TRANSLATION_FLUX_RULE.evaluate(scalar) * organism_scales.translation_scale;

        self.lattice.apply_uniform_delta(
            IntracellularSpecies::ATP,
            -0.00030 * translation_flux * organism_scales.amino_cost_scale,
        );
        self.lattice.apply_uniform_delta(
            IntracellularSpecies::AminoAcids,
            -0.00022 * translation_flux * organism_scales.amino_cost_scale,
        );
        self.lattice.apply_uniform_delta(
            IntracellularSpecies::Nucleotides,
            -0.00016 * transcription_flux * organism_scales.nucleotide_cost_scale,
        );
        self.sync_from_lattice();
        self.update_organism_inventory_dynamics(dt, transcription_flux, translation_flux);
        self.update_complex_assembly_state(dt);
        self.update_runtime_process_reactions(dt, transcription_flux, translation_flux);
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(
            inventory,
            transcription_flux,
            translation_flux,
            0.0,
            0.0,
            0.0,
        );
    }

    fn ode_stage(&mut self, dt: f32) {
        self.refresh_organism_expression_state();
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let organism_scales = self.organism_process_scales();
        let effective_metabolic_load = ctx.get(WholeCellRuleSignal::EffectiveMetabolicLoad);
        let energy_gain = ENERGY_GAIN_RULE.evaluate(scalar) * organism_scales.energy_scale;
        let energy_cost = ENERGY_COST_RULE.evaluate(scalar)
            * 0.5
            * (organism_scales.transcription_scale + organism_scales.translation_scale);
        let nucleotide_recharge =
            NUCLEOTIDE_RECHARGE_RULE.evaluate(scalar) * organism_scales.replication_scale;
        let membrane_flux = MEMBRANE_FLUX_RULE.evaluate(scalar) * organism_scales.membrane_scale;

        self.adp_mm = (self.adp_mm + energy_cost - 0.65 * energy_gain).clamp(0.05, 4.0);
        self.glucose_mm = (self.glucose_mm + 0.0020 * dt - 0.0012 * effective_metabolic_load * dt)
            .clamp(0.2, 3.0);
        self.oxygen_mm =
            (self.oxygen_mm + 0.0018 * dt - 0.0010 * effective_metabolic_load * dt).clamp(0.2, 2.5);

        self.lattice
            .apply_uniform_delta(IntracellularSpecies::ATP, energy_gain - energy_cost);
        self.lattice
            .apply_uniform_delta(IntracellularSpecies::Nucleotides, nucleotide_recharge);
        self.lattice
            .apply_uniform_delta(IntracellularSpecies::MembranePrecursors, membrane_flux);
        self.sync_from_lattice();
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(inventory, 0.0, 0.0, 0.0, membrane_flux, 0.0);
    }

    fn bd_stage(&mut self, dt: f32) {
        self.refresh_organism_expression_state();
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let organism_scales = self.organism_process_scales();
        let replication_drive =
            REPLICATION_DRIVE_RULE.evaluate(scalar) * organism_scales.replication_scale;
        let replication_flux = replication_drive / 18.0;

        let mut segregation_ctx = ctx;
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        segregation_ctx.set(WholeCellRuleSignal::ReplicatedFraction, replicated_fraction);
        segregation_ctx.set(
            WholeCellRuleSignal::InverseReplicatedFraction,
            (1.0 - replicated_fraction).clamp(0.0, 1.0),
        );
        let segregation_drive = SEGREGATION_STEP_RULE.evaluate(segregation_ctx.scalar())
            * organism_scales.segregation_scale;
        self.advance_chromosome_state(dt, replication_drive, segregation_drive);
        self.refresh_surrogate_pool_diagnostics(inventory, 0.0, 0.0, replication_flux, 0.0, 0.0);
    }

    fn geometry_stage(&mut self, dt: f32) {
        self.refresh_organism_expression_state();
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let mut ctx = self.stage_rule_context(dt, inventory, fluxes);
        let organism_scales = self.organism_process_scales();
        let membrane_growth_nm2 =
            MEMBRANE_GROWTH_RULE.evaluate(ctx.scalar()) * organism_scales.membrane_scale;

        let constriction_flux =
            CONSTRICTION_FLUX_RULE.evaluate(ctx.scalar()) * organism_scales.constriction_scale;
        ctx.set(WholeCellRuleSignal::ConstrictionFlux, constriction_flux);
        let constriction_drive =
            CONSTRICTION_DRIVE_RULE.evaluate(ctx.scalar()) * organism_scales.constriction_scale;
        self.update_membrane_division_state(
            dt,
            membrane_growth_nm2,
            constriction_flux,
            constriction_drive,
        );

        self.lattice.apply_uniform_delta(
            IntracellularSpecies::MembranePrecursors,
            -0.00020 * membrane_growth_nm2,
        );
        self.sync_from_lattice();
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(
            inventory,
            0.0,
            0.0,
            0.0,
            membrane_growth_nm2 * 0.001,
            constriction_flux,
        );
    }

    fn sync_from_lattice(&mut self) {
        self.atp_mm = self.lattice.mean_species(IntracellularSpecies::ATP);
        self.amino_acids_mm = self.lattice.mean_species(IntracellularSpecies::AminoAcids);
        self.nucleotides_mm = self.lattice.mean_species(IntracellularSpecies::Nucleotides);
        self.membrane_precursors_mm = self
            .lattice
            .mean_species(IntracellularSpecies::MembranePrecursors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell_data::{WholeCellAssetClass, WholeCellBulkField};
    use crate::whole_cell_submodels::{
        LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeRequest, Syn3ASubsystemPreset,
        WholeCellChemistrySite,
    };

    fn distribute_total_by_weights(weights: &[f32], mean_value: f32) -> Vec<f32> {
        let total = mean_value.max(0.0) * weights.len() as f32;
        let weight_sum = weights
            .iter()
            .map(|weight| weight.max(0.0))
            .sum::<f32>()
            .max(1.0e-6);
        weights
            .iter()
            .map(|weight| total * weight.max(0.0) / weight_sum)
            .collect()
    }

    #[test]
    fn test_cpu_whole_cell_progresses_state() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.5,
            ..WholeCellConfig::default()
        });
        let start = sim.snapshot();

        sim.run(40);

        let end = sim.snapshot();
        let complex = sim.complex_assembly_state();
        assert_eq!(sim.backend(), WholeCellBackend::Cpu);
        assert!(end.time_ms > start.time_ms);
        assert!(end.replicated_bp > start.replicated_bp);
        assert!(end.surface_area_nm2 > start.surface_area_nm2);
        assert!(end.division_progress >= start.division_progress);
        assert!(end.atp_mm > 0.0);
        assert!(complex.total_complexes() > 0.0);
        assert!(complex.ftsz_target > 0.0);
    }

    #[test]
    fn test_registry_transport_reaction_updates_authoritative_glucose_pool() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.glucose_mm = 0.0;
        sim.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "pool_glucose".to_string(),
            name: "glucose".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 24.0,
            bulk_field: Some(WholeCellBulkField::Glucose),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 0.0,
            anchor_count: 24.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "pool_glucose_transport".to_string(),
            name: "glucose transport".to_string(),
            reaction_class: WholeCellReactionClass::PoolTransport,
            asset_class: WholeCellAssetClass::Energy,
            nominal_rate: 8.0,
            catalyst: None,
            operon: None,
            reactants: Vec::new(),
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_glucose".to_string(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        assert!(sim.glucose_mm > 0.0);
        assert!(sim.organism_species[0].count > 0.0);
        assert!(sim.organism_reactions[0].current_flux > 0.0);
    }

    #[test]
    fn test_registry_degradation_reaction_returns_mass_to_nucleotide_lattice() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.lattice
            .fill_species(IntracellularSpecies::Nucleotides, 0.05);
        sim.sync_from_lattice();
        sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
            name: "test_operon".to_string(),
            gene_count: 1,
            copy_gain: 1.0,
            basal_activity: 1.0,
            effective_activity: 1.0,
            support_level: 1.0,
            stress_penalty: 1.20,
            transcript_abundance: 16.0,
            protein_abundance: 10.0,
            transcript_synthesis_rate: 0.0,
            protein_synthesis_rate: 0.0,
            transcript_turnover_rate: 0.0,
            protein_turnover_rate: 0.0,
            promoter_open_fraction: 0.0,
            active_rnap_occupancy: 0.0,
            transcription_length_nt: 900.0,
            transcription_progress_nt: 0.0,
            nascent_transcript_abundance: 0.0,
            mature_transcript_abundance: 16.0,
            damaged_transcript_abundance: 0.0,
            active_ribosome_occupancy: 0.0,
            translation_length_aa: 300.0,
            translation_progress_aa: 0.0,
            nascent_protein_abundance: 0.0,
            mature_protein_abundance: 10.0,
            damaged_protein_abundance: 0.0,
            process_drive: WholeCellProcessWeights::default(),
        }];
        sim.refresh_expression_inventory_totals();
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleotides".to_string(),
                name: "nucleotides".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 32.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::NucleoidLocal,
                patch_domain: WholeCellPatchDomain::NucleoidTrack,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 3.2,
                anchor_count: 3.2,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "test_rna".to_string(),
                name: "test_rna".to_string(),
                species_class: WholeCellSpeciesClass::Rna,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Homeostasis,
                basal_abundance: 4.0,
                bulk_field: None,
                spatial_scope: WholeCellSpatialScope::WellMixed,
                patch_domain: WholeCellPatchDomain::Distributed,
                operon: Some("test_operon".to_string()),
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 16.0,
                anchor_count: 16.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "test_rna_degradation".to_string(),
            name: "test_rna degradation".to_string(),
            reaction_class: WholeCellReactionClass::RnaDegradation,
            asset_class: WholeCellAssetClass::Homeostasis,
            nominal_rate: 6.0,
            catalyst: None,
            operon: Some("test_operon".to_string()),
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "test_rna".to_string(),
                stoichiometry: 1.0,
            }],
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleotides".to_string(),
                stoichiometry: 2.5,
            }],
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];
        let before_nucleotides = sim.lattice.mean_species(IntracellularSpecies::Nucleotides);

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        let after_nucleotides = sim.lattice.mean_species(IntracellularSpecies::Nucleotides);
        let rna_state = sim
            .organism_species
            .iter()
            .find(|species| species.id == "test_rna")
            .expect("RNA species");
        let unit_state = sim
            .organism_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "test_operon")
            .expect("unit state");
        assert!(after_nucleotides > before_nucleotides);
        assert!(rna_state.count < 16.0);
        assert!(sim.nucleotides_mm > before_nucleotides);
        assert!(unit_state.transcript_abundance < 16.0);
        assert!(sim.organism_expression.total_transcript_abundance < 16.0);
    }

    #[test]
    fn test_registry_stress_response_reduces_metabolic_load_and_operon_stress() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.metabolic_load = 2.2;
        sim.lattice.fill_species(IntracellularSpecies::ATP, 0.50);
        sim.sync_from_lattice();
        sim.organism_expression.transcription_units = vec![WholeCellTranscriptionUnitState {
            name: "stress_operon".to_string(),
            gene_count: 2,
            copy_gain: 1.0,
            basal_activity: 1.0,
            effective_activity: 1.0,
            support_level: 0.72,
            stress_penalty: 1.45,
            transcript_abundance: 12.0,
            protein_abundance: 16.0,
            transcript_synthesis_rate: 0.0,
            protein_synthesis_rate: 0.0,
            transcript_turnover_rate: 0.0,
            protein_turnover_rate: 0.0,
            promoter_open_fraction: 0.0,
            active_rnap_occupancy: 0.0,
            transcription_length_nt: 1800.0,
            transcription_progress_nt: 0.0,
            nascent_transcript_abundance: 0.0,
            mature_transcript_abundance: 12.0,
            damaged_transcript_abundance: 0.0,
            active_ribosome_occupancy: 0.0,
            translation_length_aa: 600.0,
            translation_progress_aa: 0.0,
            nascent_protein_abundance: 0.0,
            mature_protein_abundance: 16.0,
            damaged_protein_abundance: 0.0,
            process_drive: WholeCellProcessWeights::default(),
        }];
        sim.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "pool_atp".to_string(),
            name: "atp".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 28.0,
            bulk_field: Some(WholeCellBulkField::ATP),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            count: 28.0,
            anchor_count: 28.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "stress_operon_stress_response".to_string(),
            name: "stress response".to_string(),
            reaction_class: WholeCellReactionClass::StressResponse,
            asset_class: WholeCellAssetClass::Homeostasis,
            nominal_rate: 8.0,
            catalyst: None,
            operon: Some("stress_operon".to_string()),
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_atp".to_string(),
                stoichiometry: 1.0,
            }],
            products: Vec::new(),
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];
        let before_load = sim.metabolic_load;
        let before_atp = sim.atp_mm;

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        let unit_state = sim
            .organism_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "stress_operon")
            .expect("stress unit");
        assert!(sim.metabolic_load < before_load);
        assert!(sim.atp_mm < before_atp);
        assert!(unit_state.stress_penalty < 1.45);
        assert!(unit_state.support_level > 0.72);
    }

    #[test]
    fn test_registry_complex_repair_updates_named_complex_state() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.lattice
            .fill_species(IntracellularSpecies::AminoAcids, 0.50);
        sim.sync_from_lattice();
        sim.named_complexes = vec![WholeCellNamedComplexState {
            id: "test_complex".to_string(),
            operon: "repair_operon".to_string(),
            asset_class: WholeCellAssetClass::Membrane,
            family: WholeCellAssemblyFamily::MembraneEnzyme,
            subsystem_targets: Vec::new(),
            subunit_pool: 10.0,
            nucleation_intermediate: 1.0,
            elongation_intermediate: 1.0,
            abundance: 2.0,
            target_abundance: 8.0,
            assembly_rate: 0.0,
            degradation_rate: 0.0,
            nucleation_rate: 0.0,
            elongation_rate: 0.0,
            maturation_rate: 0.0,
            component_satisfaction: 0.8,
            structural_support: 0.8,
            assembly_progress: 0.4,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.8,
            shared_component_pressure: 0.0,
            insertion_progress: 0.6,
            failure_count: 0.0,
        }];
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "test_complex_subunit_pool".to_string(),
                name: "test complex subunit pool".to_string(),
                species_class: WholeCellSpeciesClass::ComplexSubunitPool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Membrane,
                basal_abundance: 10.0,
                bulk_field: None,
                spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
                patch_domain: WholeCellPatchDomain::MembraneBand,
                operon: Some("repair_operon".to_string()),
                parent_complex: Some("test_complex".to_string()),
                subsystem_targets: Vec::new(),
                count: 10.0,
                anchor_count: 10.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "test_complex_mature".to_string(),
                name: "test complex".to_string(),
                species_class: WholeCellSpeciesClass::ComplexMature,
                compartment: "membrane".to_string(),
                asset_class: WholeCellAssetClass::Membrane,
                basal_abundance: 8.0,
                bulk_field: None,
                spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
                patch_domain: WholeCellPatchDomain::MembraneBand,
                operon: Some("repair_operon".to_string()),
                parent_complex: Some("test_complex".to_string()),
                subsystem_targets: Vec::new(),
                count: 2.0,
                anchor_count: 8.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "pool_amino_acids".to_string(),
                name: "amino acids".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Translation,
                basal_abundance: 32.0,
                bulk_field: Some(WholeCellBulkField::AminoAcids),
                spatial_scope: WholeCellSpatialScope::WellMixed,
                patch_domain: WholeCellPatchDomain::Distributed,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 32.0,
                anchor_count: 32.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "test_complex_repair".to_string(),
            name: "test complex repair".to_string(),
            reaction_class: WholeCellReactionClass::ComplexRepair,
            asset_class: WholeCellAssetClass::Membrane,
            nominal_rate: 6.0,
            catalyst: None,
            operon: Some("repair_operon".to_string()),
            reactants: vec![
                crate::whole_cell_data::WholeCellReactionParticipantSpec {
                    species_id: "test_complex_subunit_pool".to_string(),
                    stoichiometry: 1.0,
                },
                crate::whole_cell_data::WholeCellReactionParticipantSpec {
                    species_id: "pool_amino_acids".to_string(),
                    stoichiometry: 0.5,
                },
            ],
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "test_complex_mature".to_string(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        let complex_state = sim
            .named_complexes
            .iter()
            .find(|state| state.id == "test_complex")
            .expect("complex state");
        assert!(complex_state.abundance > 2.0);
        assert!(complex_state.subunit_pool < 10.0);
    }

    #[test]
    fn test_atp_hotspot_diffuses_on_cpu() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            x_dim: 8,
            y_dim: 8,
            z_dim: 4,
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let center = (4, 4, 2);
        let neighbor_idx = center.2 * 8 * 8 + center.1 * 8 + (center.0 + 1);
        let center_idx = center.2 * 8 * 8 + center.1 * 8 + center.0;

        sim.add_hotspot(IntracellularSpecies::ATP, center.0, center.1, center.2, 4.0);
        let before = sim.atp_lattice();

        sim.step();

        let after = sim.atp_lattice();
        assert!(after[center_idx] < before[center_idx]);
        assert!(after[neighbor_idx] > before[neighbor_idx]);
    }

    #[test]
    fn test_initial_lattice_has_no_seeded_hotspots() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let atp = sim.atp_lattice();
        let first = atp.first().copied().expect("atp lattice");
        assert!(atp.iter().all(|value| (*value - first).abs() < 1.0e-6));
    }

    #[test]
    fn test_spatial_fields_round_trip_through_saved_state() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.run(4);
        let saved = sim.save_state_json().expect("saved state json");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");

        let original_membrane = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneAdjacency);
        let restored_membrane = restored
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneAdjacency);
        let original_nucleoid = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::NucleoidOccupancy);
        let restored_nucleoid = restored
            .spatial_fields
            .clone_field(IntracellularSpatialField::NucleoidOccupancy);
        let original_band = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneBandZone);
        let restored_band = restored
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneBandZone);
        let original_poles = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::PoleZone);
        let restored_poles = restored
            .spatial_fields
            .clone_field(IntracellularSpatialField::PoleZone);

        assert_eq!(original_membrane.len(), restored_membrane.len());
        assert_eq!(original_nucleoid.len(), restored_nucleoid.len());
        assert_eq!(original_band.len(), restored_band.len());
        assert_eq!(original_poles.len(), restored_poles.len());
        assert!(original_membrane
            .iter()
            .zip(restored_membrane.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
        assert!(original_nucleoid
            .iter()
            .zip(restored_nucleoid.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
        assert!(original_band
            .iter()
            .zip(restored_band.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
        assert!(original_poles
            .iter()
            .zip(restored_poles.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1.0e-6));
    }

    #[test]
    fn test_nucleoid_localization_biases_nucleotide_signal() {
        let mut nucleoid_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let mut pole_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let nucleoid_weights = nucleoid_loaded
            .spatial_fields
            .clone_field(IntracellularSpatialField::NucleoidOccupancy);
        let pole_weights: Vec<f32> = nucleoid_weights
            .iter()
            .map(|weight| (1.0 - *weight).max(0.02))
            .collect();
        let nucleoid_values = distribute_total_by_weights(&nucleoid_weights, 0.80);
        let pole_values = distribute_total_by_weights(&pole_weights, 0.80);

        nucleoid_loaded
            .lattice
            .set_species(IntracellularSpecies::Nucleotides, &nucleoid_values)
            .expect("nucleoid-targeted nucleotides");
        pole_loaded
            .lattice
            .set_species(IntracellularSpecies::Nucleotides, &pole_values)
            .expect("pole-targeted nucleotides");
        nucleoid_loaded.sync_from_lattice();
        pole_loaded.sync_from_lattice();

        let nucleoid_ctx = nucleoid_loaded.base_rule_context(0.0);
        let pole_ctx = pole_loaded.base_rule_context(0.0);

        assert!((nucleoid_loaded.nucleotides_mm - pole_loaded.nucleotides_mm).abs() < 1.0e-6);
        assert!(
            nucleoid_loaded.localized_nucleotide_pool_mm()
                > pole_loaded.localized_nucleotide_pool_mm()
        );
        assert!(
            nucleoid_ctx.get(WholeCellRuleSignal::NucleotideSignal)
                > pole_ctx.get(WholeCellRuleSignal::NucleotideSignal)
        );
    }

    #[test]
    fn test_septum_localization_biases_membrane_constriction_support() {
        let mut septum_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let mut pole_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let septum_weights = septum_loaded
            .spatial_fields
            .clone_field(IntracellularSpatialField::SeptumZone);
        let pole_weights: Vec<f32> = septum_loaded
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneAdjacency)
            .into_iter()
            .zip(septum_weights.iter().copied())
            .map(|(membrane, septum)| (membrane - septum).max(0.02))
            .collect();
        let septum_values = distribute_total_by_weights(&septum_weights, 0.35);
        let pole_values = distribute_total_by_weights(&pole_weights, 0.35);

        septum_loaded
            .lattice
            .set_species(IntracellularSpecies::MembranePrecursors, &septum_values)
            .expect("septum-targeted membrane precursors");
        pole_loaded
            .lattice
            .set_species(IntracellularSpecies::MembranePrecursors, &pole_values)
            .expect("pole-targeted membrane precursors");
        septum_loaded.sync_from_lattice();
        pole_loaded.sync_from_lattice();
        assert!(
            (septum_loaded.membrane_precursors_mm - pole_loaded.membrane_precursors_mm).abs()
                < 1.0e-4
        );
        septum_loaded.refresh_organism_expression_state();
        pole_loaded.refresh_organism_expression_state();

        for _ in 0..8 {
            septum_loaded.geometry_stage(septum_loaded.config.dt_ms);
            pole_loaded.geometry_stage(pole_loaded.config.dt_ms);
        }

        assert!(
            septum_loaded.localized_membrane_precursor_pool_mm()
                > pole_loaded.localized_membrane_precursor_pool_mm()
        );
        assert!(
            septum_loaded.membrane_division_state.constriction_force
                > pole_loaded.membrane_division_state.constriction_force
        );
        assert!(
            septum_loaded
                .membrane_division_state
                .septal_lipid_inventory_nm2
                > pole_loaded
                    .membrane_division_state
                    .septal_lipid_inventory_nm2
        );
    }

    #[test]
    fn test_membrane_band_zone_exceeds_poles_for_band_localized_drive() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "atp_band_complex_mature".to_string(),
            name: "atp band complex".to_string(),
            species_class: WholeCellSpeciesClass::ComplexMature,
            compartment: "membrane".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 8.0,
            bulk_field: None,
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            operon: Some("energy_operon".to_string()),
            parent_complex: Some("atp_band_complex".to_string()),
            subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
            count: 8.0,
            anchor_count: 8.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];
        sim.organism_reactions.clear();
        sim.refresh_spatial_fields();
        sim.refresh_rdme_drive_fields();

        let band_energy = sim.localized_drive_mean(
            WholeCellRdmeDriveField::EnergySource,
            IntracellularSpatialField::MembraneBandZone,
        );
        let pole_energy = sim.localized_drive_mean(
            WholeCellRdmeDriveField::EnergySource,
            IntracellularSpatialField::PoleZone,
        );

        assert!(band_energy > pole_energy);
    }

    #[test]
    fn test_localized_pool_transfer_moves_membrane_precursors_into_band_zone() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.lattice
            .fill_species(IntracellularSpecies::MembranePrecursors, 0.22);
        sim.sync_from_lattice();
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "pool_membrane_precursors".to_string(),
                name: "membrane precursors".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Membrane,
                basal_abundance: 48.0,
                bulk_field: Some(WholeCellBulkField::MembranePrecursors),
                spatial_scope: WholeCellSpatialScope::WellMixed,
                patch_domain: WholeCellPatchDomain::Distributed,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 48.0,
                anchor_count: 48.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "pool_membrane_band_membrane_precursors".to_string(),
                name: "membrane band precursors".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "membrane".to_string(),
                asset_class: WholeCellAssetClass::Membrane,
                basal_abundance: 8.0,
                bulk_field: Some(WholeCellBulkField::MembranePrecursors),
                spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
                patch_domain: WholeCellPatchDomain::MembraneBand,
                operon: None,
                parent_complex: None,
                subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
                count: 2.0,
                anchor_count: 2.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "pool_membrane_band_membrane_precursors_localized_transfer".to_string(),
            name: "membrane band precursors localized transfer".to_string(),
            reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
            asset_class: WholeCellAssetClass::Membrane,
            nominal_rate: 9.0,
            catalyst: None,
            operon: None,
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_membrane_precursors".to_string(),
                stoichiometry: 1.0,
            }],
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_membrane_band_membrane_precursors".to_string(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
            spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
            patch_domain: WholeCellPatchDomain::MembraneBand,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        let before_band = sim.localized_membrane_band_precursor_pool_mm();

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        let after_band = sim.localized_membrane_band_precursor_pool_mm();
        let after_pole = sim.localized_polar_precursor_pool_mm();
        assert!(after_band > before_band);
        assert!(after_band > after_pole);
        assert!(sim.organism_reactions[0].current_flux > 0.0);
    }

    #[test]
    fn test_localized_pool_transfer_moves_nucleotides_into_nucleoid_track() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.lattice
            .fill_species(IntracellularSpecies::Nucleotides, 0.18);
        sim.sync_from_lattice();
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleotides".to_string(),
                name: "nucleotides".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 42.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::WellMixed,
                patch_domain: WholeCellPatchDomain::Distributed,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 42.0,
                anchor_count: 42.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleoid_track_nucleotides".to_string(),
                name: "nucleoid track nucleotides pool".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "chromosome".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 10.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::NucleoidLocal,
                patch_domain: WholeCellPatchDomain::NucleoidTrack,
                operon: None,
                parent_complex: None,
                subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
                count: 3.0,
                anchor_count: 3.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "pool_nucleoid_track_nucleotides_localized_transfer".to_string(),
            name: "nucleoid track nucleotides localized transfer".to_string(),
            reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
            asset_class: WholeCellAssetClass::Replication,
            nominal_rate: 8.5,
            catalyst: None,
            operon: None,
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleotides".to_string(),
                stoichiometry: 1.0,
            }],
            products: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleoid_track_nucleotides".to_string(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];
        let mut nucleotide_demand = vec![0.0; sim.lattice.total_voxels()];
        let nucleoid_weights = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::NucleoidOccupancy);
        for (value, weight) in nucleotide_demand.iter_mut().zip(nucleoid_weights.iter()) {
            *value = 1.6 * weight.max(0.0);
        }
        let _ = sim
            .rdme_drive_fields
            .set_field(WholeCellRdmeDriveField::NucleotideDemand, &nucleotide_demand);

        let before_nucleoid = sim.localized_nucleotide_pool_mm();

        sim.update_runtime_process_reactions(1.0, 0.0, 0.0);

        let after_nucleoid = sim.localized_nucleotide_pool_mm();
        let membrane_nucleotides = sim.spatial_species_mean(
            IntracellularSpecies::Nucleotides,
            IntracellularSpatialField::MembraneAdjacency,
        );
        assert!(after_nucleoid > before_nucleoid);
        assert!(after_nucleoid > membrane_nucleotides);
        assert!(sim.organism_reactions[0].current_flux > 0.0);
    }

    #[test]
    fn test_rdme_drive_fields_follow_compiled_scopes() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        sim.organism_species = vec![
            WholeCellSpeciesRuntimeState {
                id: "pool_nucleotides".to_string(),
                name: "nucleotides".to_string(),
                species_class: WholeCellSpeciesClass::Pool,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Replication,
                basal_abundance: 32.0,
                bulk_field: Some(WholeCellBulkField::Nucleotides),
                spatial_scope: WholeCellSpatialScope::NucleoidLocal,
                patch_domain: WholeCellPatchDomain::NucleoidTrack,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                count: 24.0,
                anchor_count: 24.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "atp_band_complex_mature".to_string(),
                name: "atp band complex".to_string(),
                species_class: WholeCellSpeciesClass::ComplexMature,
                compartment: "membrane".to_string(),
                asset_class: WholeCellAssetClass::Energy,
                basal_abundance: 6.0,
                bulk_field: None,
                spatial_scope: WholeCellSpatialScope::MembraneAdjacent,
                patch_domain: WholeCellPatchDomain::MembraneBand,
                operon: Some("energy_operon".to_string()),
                parent_complex: Some("atp_band_complex".to_string()),
                subsystem_targets: vec![Syn3ASubsystemPreset::AtpSynthaseMembraneBand],
                count: 6.0,
                anchor_count: 6.0,
                synthesis_rate: 0.0,
                turnover_rate: 0.0,
            },
        ];
        sim.organism_reactions = vec![WholeCellReactionRuntimeState {
            id: "replication_drive".to_string(),
            name: "replication drive".to_string(),
            reaction_class: WholeCellReactionClass::Transcription,
            asset_class: WholeCellAssetClass::Replication,
            nominal_rate: 9.0,
            catalyst: None,
            operon: Some("replication_operon".to_string()),
            reactants: vec![crate::whole_cell_data::WholeCellReactionParticipantSpec {
                species_id: "pool_nucleotides".to_string(),
                stoichiometry: 1.6,
            }],
            products: Vec::new(),
            subsystem_targets: vec![Syn3ASubsystemPreset::ReplisomeTrack],
            spatial_scope: WholeCellSpatialScope::NucleoidLocal,
            patch_domain: WholeCellPatchDomain::NucleoidTrack,
            current_flux: 6.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        sim.refresh_spatial_fields();
        sim.refresh_rdme_drive_fields();

        let membrane_weights = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneAdjacency);
        let nucleoid_weights = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::NucleoidOccupancy);

        let membrane_energy = sim.rdme_drive_fields.weighted_mean(
            WholeCellRdmeDriveField::EnergySource,
            &membrane_weights,
        );
        let nucleoid_energy = sim.rdme_drive_fields.weighted_mean(
            WholeCellRdmeDriveField::EnergySource,
            &nucleoid_weights,
        );
        let nucleoid_demand = sim.rdme_drive_fields.weighted_mean(
            WholeCellRdmeDriveField::NucleotideDemand,
            &nucleoid_weights,
        );
        let membrane_demand = sim.rdme_drive_fields.weighted_mean(
            WholeCellRdmeDriveField::NucleotideDemand,
            &membrane_weights,
        );
        let band_weights = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneBandZone);
        let pole_weights = sim
            .spatial_fields
            .clone_field(IntracellularSpatialField::PoleZone);
        let band_energy = sim
            .rdme_drive_fields
            .weighted_mean(WholeCellRdmeDriveField::EnergySource, &band_weights);
        let pole_energy = sim
            .rdme_drive_fields
            .weighted_mean(WholeCellRdmeDriveField::EnergySource, &pole_weights);

        assert!(membrane_energy > nucleoid_energy);
        assert!(nucleoid_demand > membrane_demand);
        assert!(band_energy > pole_energy);
    }

    #[test]
    fn test_membrane_patch_turnover_prefers_band_loaded_precursors() {
        let mut band_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let mut pole_loaded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let band_weights = band_loaded
            .spatial_fields
            .clone_field(IntracellularSpatialField::MembraneBandZone);
        let pole_weights = pole_loaded
            .spatial_fields
            .clone_field(IntracellularSpatialField::PoleZone);
        let band_values = distribute_total_by_weights(&band_weights, 0.35);
        let pole_values = distribute_total_by_weights(&pole_weights, 0.35);

        band_loaded
            .lattice
            .set_species(IntracellularSpecies::MembranePrecursors, &band_values)
            .expect("band-targeted membrane precursors");
        pole_loaded
            .lattice
            .set_species(IntracellularSpecies::MembranePrecursors, &pole_values)
            .expect("pole-targeted membrane precursors");
        band_loaded.sync_from_lattice();
        pole_loaded.sync_from_lattice();

        for _ in 0..8 {
            band_loaded.geometry_stage(band_loaded.config.dt_ms);
            pole_loaded.geometry_stage(pole_loaded.config.dt_ms);
        }

        assert!(
            band_loaded
                .membrane_division_state
                .membrane_band_lipid_inventory_nm2
                > pole_loaded
                    .membrane_division_state
                    .membrane_band_lipid_inventory_nm2
        );
        assert!(
            band_loaded.membrane_division_state.band_turnover_pressure
                < pole_loaded.membrane_division_state.band_turnover_pressure
        );
    }

    #[test]
    fn test_resource_estimators_ingest_local_chemistry_context() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let baseline = sim.base_rule_context(0.0);

        sim.chemistry_report = LocalChemistryReport {
            atp_support: 1.12,
            translation_support: 1.08,
            nucleotide_support: 1.10,
            membrane_support: 1.06,
            crowding_penalty: 0.95,
            mean_glucose: 0.85,
            mean_oxygen: 0.78,
            mean_atp_flux: 0.92,
            mean_carbon_dioxide: 0.18,
        };

        let enriched = sim.base_rule_context(0.0);
        assert!(
            enriched.get(WholeCellRuleSignal::GlucoseSignal)
                > baseline.get(WholeCellRuleSignal::GlucoseSignal)
        );
        assert!(
            enriched.get(WholeCellRuleSignal::OxygenSignal)
                > baseline.get(WholeCellRuleSignal::OxygenSignal)
        );
        assert!(
            enriched.get(WholeCellRuleSignal::EnergySignal)
                > baseline.get(WholeCellRuleSignal::EnergySignal)
        );
    }

    #[test]
    fn test_quantum_profile_accelerates_growth() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.5,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        let mut accelerated = WholeCellSimulator::new(config);
        accelerated.set_quantum_profile(WholeCellQuantumProfile {
            oxphos_efficiency: 1.60,
            translation_efficiency: 1.45,
            nucleotide_polymerization_efficiency: 1.50,
            membrane_synthesis_efficiency: 1.35,
            chromosome_segregation_efficiency: 1.30,
        });

        baseline.run(120);
        accelerated.run(120);

        let baseline_snapshot = baseline.snapshot();
        let accelerated_snapshot = accelerated.snapshot();
        let baseline_membrane = baseline.membrane_division_state();
        let accelerated_membrane = accelerated.membrane_division_state();

        assert!(accelerated_snapshot.atp_mm >= baseline_snapshot.atp_mm);
        assert!(accelerated_snapshot.ftsz > baseline_snapshot.ftsz);
        assert!(accelerated_snapshot.surface_area_nm2 > baseline_snapshot.surface_area_nm2);
        assert!(accelerated_membrane.constriction_force > baseline_membrane.constriction_force);
    }

    #[test]
    fn test_surrogate_pools_are_diagnostics_not_stage_drivers() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        let mut perturbed = WholeCellSimulator::new(config);

        perturbed.active_rnap = 256.0;
        perturbed.active_ribosomes = 320.0;
        perturbed.dnaa = 256.0;
        perturbed.ftsz = 384.0;

        baseline.run(16);
        perturbed.run(16);

        let baseline_snapshot = baseline.snapshot();
        let perturbed_snapshot = perturbed.snapshot();

        assert_eq!(
            perturbed_snapshot.replicated_bp,
            baseline_snapshot.replicated_bp
        );
        assert!(
            (perturbed_snapshot.division_progress - baseline_snapshot.division_progress).abs()
                < 1.0e-6
        );
        assert!(
            (perturbed_snapshot.surface_area_nm2 - baseline_snapshot.surface_area_nm2).abs()
                < 1.0e-4
        );
        assert!(perturbed_snapshot.active_rnap < 256.0);
        assert!(perturbed_snapshot.active_ribosomes < 320.0);
        assert!(perturbed_snapshot.dnaa < 256.0);
        assert!(perturbed_snapshot.ftsz < 384.0);
    }

    #[test]
    fn test_local_chemistry_bridge_updates_report_and_md_probe() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_local_chemistry(10, 10, 6, 0.5, false);

        sim.run(8);
        let chemistry = sim
            .local_chemistry_report()
            .expect("local chemistry report");
        assert!(chemistry.atp_support > 0.0);
        assert!(chemistry.translation_support > 0.0);
        assert!(chemistry.crowding_penalty > 0.0);

        let probe = sim
            .run_local_md_probe(LocalMDProbeRequest {
                site: WholeCellChemistrySite::RibosomeCluster,
                n_atoms: 16,
                steps: 8,
                dt_ps: 0.001,
                box_size_angstrom: 14.0,
                temperature_k: 310.0,
            })
            .expect("md probe");
        assert!(probe.structural_order > 0.0);
        assert!(probe.crowding_penalty > 0.0);
        assert!(sim.last_md_probe().is_some());
    }

    #[test]
    fn test_default_syn3a_subsystems_schedule_and_run() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();

        let scheduled = sim.scheduled_syn3a_subsystem_probes();
        assert_eq!(scheduled.len(), Syn3ASubsystemPreset::all().len());

        sim.run(12);

        assert!(sim.local_chemistry_report().is_some());
        assert!(sim.last_md_probe().is_some());
        assert!(sim.md_translation_scale() > 0.0);
        assert!(sim.md_membrane_scale() > 0.0);
    }

    #[test]
    fn test_derivation_calibration_is_exposed_on_simulator() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_local_chemistry(12, 12, 6, 0.5, false);

        let samples = sim.derivation_calibration_samples(0.25, 2);
        assert_eq!(samples.len(), Syn3ASubsystemPreset::all().len());

        let fit = sim
            .fit_derivation_calibration(0.25, 2)
            .expect("fit result")
            .expect("bridge-enabled fit");
        assert_eq!(fit.sample_count, Syn3ASubsystemPreset::all().len());
        assert!(fit.fitted_loss < fit.baseline_loss);
    }

    #[test]
    fn test_subsystem_states_capture_probe_couplings() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();
        sim.run(16);

        let states = sim.subsystem_states();
        assert_eq!(states.len(), Syn3ASubsystemPreset::all().len());

        let atp_band = states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("ATP synthase state");
        assert!(atp_band.atp_scale > 0.0);
        assert!(atp_band.membrane_scale > 0.0);
        assert!(atp_band.last_probe_step.is_some());

        let replisome = states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome state");
        assert!(replisome.replication_scale > 0.0);
        assert!(replisome.segregation_scale > 0.0);
        assert!(replisome.last_probe_step.is_some());
    }

    #[test]
    fn test_local_chemistry_sites_are_exposed_and_site_resolved() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();
        sim.run(8);

        let site_reports = sim.local_chemistry_sites();
        assert_eq!(site_reports.len(), Syn3ASubsystemPreset::all().len());

        let atp_band = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("ATP site report");
        let replisome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome site report");
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");

        assert!(atp_band.patch_radius > 0);
        assert!(atp_band.localization_score != 0.0);
        assert!(atp_band.site_z < sim.config.z_dim);
        assert!(replisome.nucleotide_support > 0.0);
        assert!(ribosome.substrate_draw > 0.0);
        assert!(replisome.biosynthetic_draw > 0.0);
        assert!(atp_band.demand_satisfaction > 0.0);
        assert!(atp_band.assembly_occupancy > 0.0);
        assert!(ribosome.assembly_stability > 0.0);
        let unique_sites = site_reports
            .iter()
            .map(|report| (report.site_x, report.site_y, report.site_z))
            .collect::<std::collections::HashSet<_>>();
        assert!(unique_sites.len() > 1);
        assert!(
            atp_band.mean_oxygen != replisome.mean_oxygen
                || atp_band.mean_atp_flux != replisome.mean_atp_flux
        );
    }

    #[test]
    fn test_localized_resource_pressure_increases_effective_metabolic_load() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.chemistry_site_reports = vec![LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
            site: WholeCellChemistrySite::RibosomeCluster,
            patch_radius: 2,
            site_x: 4,
            site_y: 4,
            site_z: 2,
            localization_score: 0.92,
            atp_support: 0.95,
            translation_support: 0.90,
            nucleotide_support: 0.92,
            membrane_support: 0.94,
            crowding_penalty: 0.88,
            mean_glucose: 0.10,
            mean_oxygen: 0.08,
            mean_atp_flux: 0.06,
            mean_carbon_dioxide: 0.14,
            mean_nitrate: 0.05,
            mean_ammonium: 0.07,
            mean_proton: 0.02,
            mean_phosphorus: 0.04,
            assembly_component_availability: 0.76,
            assembly_occupancy: 0.72,
            assembly_stability: 0.70,
            assembly_turnover: 0.38,
            substrate_draw: 0.60,
            energy_draw: 0.55,
            biosynthetic_draw: 0.24,
            byproduct_load: 0.42,
            demand_satisfaction: 0.46,
        }];

        assert!(sim.effective_metabolic_load() > sim.metabolic_load);
        assert!(sim.localized_supply_scale() < 1.0);
        assert!(sim.localized_resource_pressure() > 0.0);
    }

    #[test]
    fn test_replisome_probe_accelerates_replication() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

        let mut targeted = WholeCellSimulator::new(config);
        targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::ReplisomeTrack, 1);

        baseline.run(24);
        targeted.run(24);

        let baseline_snapshot = baseline.snapshot();
        let targeted_snapshot = targeted.snapshot();
        let baseline_replisome = baseline_snapshot
            .subsystem_states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("baseline replisome state");
        let targeted_replisome = targeted_snapshot
            .subsystem_states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("targeted replisome state");

        assert!(
            targeted_snapshot.replicated_bp >= baseline_snapshot.replicated_bp,
            "replication baseline={} targeted={}",
            baseline_snapshot.replicated_bp,
            targeted_snapshot.replicated_bp
        );
        assert!(
            targeted_replisome.replication_scale > baseline_replisome.replication_scale,
            "replication scale baseline={} targeted={}",
            baseline_replisome.replication_scale,
            targeted_replisome.replication_scale
        );
        assert!(
            targeted_snapshot.chromosome_separation_nm > baseline_snapshot.chromosome_separation_nm,
            "segregation baseline={} targeted={}",
            baseline_snapshot.chromosome_separation_nm,
            targeted_snapshot.chromosome_separation_nm
        );
    }

    #[test]
    fn test_membrane_and_septum_probes_accelerate_division() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

        let mut targeted = WholeCellSimulator::new(config);
        targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::AtpSynthaseMembraneBand, 1);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::RibosomePolysomeCluster, 1);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::FtsZSeptumRing, 1);

        baseline.run(12);
        targeted.run(12);

        let baseline_snapshot = baseline.snapshot();
        let targeted_snapshot = targeted.snapshot();

        assert!(
            targeted_snapshot.division_progress > baseline_snapshot.division_progress,
            "division baseline={} targeted={}",
            baseline_snapshot.division_progress,
            targeted_snapshot.division_progress
        );
        assert!(
            targeted_snapshot.surface_area_nm2 > baseline_snapshot.surface_area_nm2,
            "surface baseline={} targeted={}",
            baseline_snapshot.surface_area_nm2,
            targeted_snapshot.surface_area_nm2
        );
    }

    #[test]
    fn test_bundled_syn3a_reference_spec_builds_native_simulator() {
        let sim = WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

        assert_eq!(sim.genome_bp, 543_000);
        assert!(sim.chemistry_bridge.is_some());
        assert_eq!(sim.scheduled_subsystem_probes.len(), 4);
        assert!(sim.lattice.mean_species(IntracellularSpecies::ATP) > 0.0);
        let summary = sim.organism_summary().expect("organism summary");
        let asset_summary = sim
            .organism_asset_summary()
            .expect("organism asset summary");
        assert_eq!(summary.organism, "JCVI-syn3A");
        assert!(summary.gene_count >= 10);
        assert!(summary.transcription_unit_count >= 4);
        assert!(asset_summary.operon_count >= summary.transcription_unit_count);
        assert_eq!(asset_summary.protein_count, summary.gene_count);
        assert!(asset_summary.targeted_complex_count >= 4);
        let registry_summary = sim
            .organism_process_registry_summary()
            .expect("organism process registry summary");
        let registry = sim
            .organism_process_registry()
            .expect("organism process registry");
        assert!(registry_summary.species_count > asset_summary.protein_count);
        assert_eq!(registry_summary.rna_species_count, asset_summary.rna_count);
        assert_eq!(
            registry_summary.protein_species_count,
            asset_summary.protein_count
        );
        assert_eq!(
            registry_summary.complex_species_count,
            asset_summary.complex_count
        );
        assert!(
            registry_summary.assembly_intermediate_species_count >= asset_summary.complex_count * 3
        );
        assert!(registry_summary.transcription_reaction_count >= summary.transcription_unit_count);
        assert!(registry_summary.translation_reaction_count >= asset_summary.protein_count);
        assert!(registry_summary.assembly_reaction_count >= asset_summary.complex_count * 4);
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "ribosome_biogenesis_operon_complex_mature"));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.id == "ribosome_biogenesis_operon_complex_maturation"));
        assert!(sim.provenance.compiled_ir_hash.is_some());
        let profile = sim.organism_profile().expect("organism profile");
        assert!(profile.process_scales.translation > 0.9);
        assert!(profile.metabolic_burden_scale > 0.9);
        let expression = sim
            .organism_expression_state()
            .expect("organism expression state");
        assert!(expression.global_activity > 0.0);
        assert!(expression.transcription_units.len() >= 4);
        assert!(expression.total_transcript_abundance > 0.0);
        assert!(expression.total_protein_abundance > 0.0);
    }

    #[test]
    fn test_saved_state_round_trip_preserves_core_progress() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.run(8);

        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");

        let original = sim.snapshot();
        let reloaded = restored.snapshot();

        assert_eq!(reloaded.step_count, original.step_count);
        assert_eq!(reloaded.replicated_bp, original.replicated_bp);
        assert_eq!(reloaded.genome_bp, original.genome_bp);
        assert!((reloaded.time_ms - original.time_ms).abs() < 1.0e-6);
        assert!((reloaded.atp_mm - original.atp_mm).abs() < 1.0e-6);
        assert_eq!(
            restored.scheduled_subsystem_probes.len(),
            sim.scheduled_subsystem_probes.len()
        );
        assert_eq!(restored.organism_summary(), sim.organism_summary());
        assert_eq!(
            restored.organism_asset_summary(),
            sim.organism_asset_summary()
        );
        assert_eq!(
            restored.organism_process_registry_summary(),
            sim.organism_process_registry_summary()
        );
        assert_eq!(
            restored
                .organism_expression_state()
                .expect("restored organism expression")
                .transcription_units
                .len(),
            sim.organism_expression_state()
                .expect("original organism expression")
                .transcription_units
                .len()
        );
        assert!(
            restored
                .organism_expression_state()
                .expect("restored expression")
                .total_transcript_abundance
                > 0.0
        );
        let restored_complex = restored.complex_assembly_state();
        let original_complex = sim.complex_assembly_state();
        assert!(restored_complex.total_complexes() > 0.0);
        assert!(
            (restored_complex.ribosome_complexes - original_complex.ribosome_complexes).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn test_saved_state_round_trip_preserves_scheduler_state() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.enable_default_syn3a_subsystems();
        sim.run(10);

        let original = sim.snapshot();
        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
        let reloaded = restored.snapshot();

        assert_eq!(reloaded.scheduler_state, original.scheduler_state);
        assert!(reloaded
            .scheduler_state
            .stage_clocks
            .iter()
            .any(|clock| clock.run_count > 0));
        assert!(
            reloaded
                .scheduler_state
                .stage_clocks
                .iter()
                .find(|clock| clock.stage == WholeCellSolverStage::AtomisticRefinement)
                .expect("atomistic clock")
                .run_count
                > 0
        );
    }

    #[test]
    fn test_multirate_scheduler_allows_stress_driven_early_cme_reexecution() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.config.cme_interval = 8;
        sim.config.ode_interval = 8;
        sim.atp_mm = 0.08;
        sim.adp_mm = 2.10;
        sim.glucose_mm = 0.12;
        sim.oxygen_mm = 0.14;
        sim.metabolic_load = 2.8;
        sim.chemistry_report.translation_support = 0.58;
        sim.chemistry_report.nucleotide_support = 0.60;
        sim.chemistry_report.crowding_penalty = 0.74;
        sim.refresh_organism_expression_state();
        sim.refresh_multirate_scheduler();

        let cme_clock = sim
            .scheduler_clock(WholeCellSolverStage::Cme)
            .expect("CME clock")
            .clone();
        let ode_clock = sim
            .scheduler_clock(WholeCellSolverStage::Ode)
            .expect("ODE clock")
            .clone();
        assert!(cme_clock.dynamic_interval_steps < sim.config.cme_interval);
        assert!(ode_clock.dynamic_interval_steps < sim.config.ode_interval);

        sim.step();
        let initial_cme_runs = sim
            .scheduler_clock(WholeCellSolverStage::Cme)
            .expect("CME clock after first step")
            .run_count;
        let initial_ode_runs = sim
            .scheduler_clock(WholeCellSolverStage::Ode)
            .expect("ODE clock after first step")
            .run_count;
        assert_eq!(initial_cme_runs, 1);
        assert_eq!(initial_ode_runs, 1);

        let followup_steps = cme_clock
            .dynamic_interval_steps
            .max(ode_clock.dynamic_interval_steps);
        sim.run(followup_steps);

        let cme_after = sim
            .scheduler_clock(WholeCellSolverStage::Cme)
            .expect("CME clock after followup");
        let ode_after = sim
            .scheduler_clock(WholeCellSolverStage::Ode)
            .expect("ODE clock after followup");
        assert!(cme_after.run_count >= 2);
        assert!(ode_after.run_count >= 2);
        assert!(sim.step_count < sim.config.cme_interval + 2);
    }

    #[test]
    fn test_organism_expression_state_responds_to_energy_and_load_stress() {
        let mut baseline =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        baseline.refresh_organism_expression_state();
        let baseline_expression = baseline
            .organism_expression_state()
            .expect("baseline expression");

        let mut stressed =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        stressed.atp_mm = 0.10;
        stressed.adp_mm = 1.80;
        stressed.glucose_mm = 0.15;
        stressed.oxygen_mm = 0.12;
        stressed.metabolic_load = 2.4;
        stressed.chemistry_report.translation_support = 0.62;
        stressed.chemistry_report.nucleotide_support = 0.58;
        stressed.chemistry_report.membrane_support = 0.64;
        stressed.chemistry_report.crowding_penalty = 0.72;
        stressed.refresh_organism_expression_state();
        let stressed_expression = stressed
            .organism_expression_state()
            .expect("stressed expression");
        let baseline_ribosome = baseline_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("baseline ribosome operon");
        let stressed_ribosome = stressed_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("stressed ribosome operon");

        assert!(stressed_expression.global_activity < baseline_expression.global_activity);
        assert!(stressed_ribosome.effective_activity < baseline_ribosome.effective_activity);
        assert!(stressed_ribosome.stress_penalty > baseline_ribosome.stress_penalty);
    }

    #[test]
    fn test_runtime_process_scales_consume_compiled_registry_signals() {
        let baseline =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let baseline_ribosome_rna = baseline
            .organism_species
            .iter()
            .cloned()
            .into_iter()
            .find(|species| species.id == "ribosome_small_subunit_core_rna")
            .expect("baseline ribosome RNA species");
        let baseline_ftsz_rna = baseline
            .organism_species
            .iter()
            .cloned()
            .into_iter()
            .find(|species| species.id == "ftsz_ring_polymerization_core_rna")
            .expect("baseline FtsZ RNA species");

        let mut registry_boosted =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        {
            let assets = registry_boosted
                .organism_assets
                .as_mut()
                .expect("compiled genome asset package");
            for rna in &mut assets.rnas {
                if rna.operon == "ribosome_biogenesis_operon" {
                    rna.basal_abundance *= 18.0;
                }
                if rna.operon == "division_ring_operon" {
                    rna.basal_abundance *= 16.0;
                }
            }
        }
        assert!(registry_boosted.rebuild_process_registry_from_assets());
        registry_boosted.refresh_organism_expression_state();
        let boosted_ribosome_rna = registry_boosted
            .organism_species
            .iter()
            .cloned()
            .into_iter()
            .find(|species| species.id == "ribosome_small_subunit_core_rna")
            .expect("boosted ribosome RNA species");
        let boosted_ftsz_rna = registry_boosted
            .organism_species
            .iter()
            .cloned()
            .into_iter()
            .find(|species| species.id == "ftsz_ring_polymerization_core_rna")
            .expect("boosted FtsZ RNA species");

        assert!(boosted_ribosome_rna.basal_abundance > baseline_ribosome_rna.basal_abundance);
        assert!(boosted_ftsz_rna.basal_abundance > baseline_ftsz_rna.basal_abundance);
    }

    #[test]
    fn test_organism_inventory_state_accumulates_transcripts_and_proteins() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let start_expression = sim
            .organism_expression_state()
            .expect("initial organism expression");
        let start_ribosome = start_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("initial ribosome unit");

        sim.run(12);

        let end_expression = sim
            .organism_expression_state()
            .expect("updated organism expression");
        let end_ribosome = end_expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("updated ribosome unit");

        assert!(end_expression.total_transcript_abundance > 0.0);
        assert!(end_expression.total_protein_abundance > 0.0);
        assert!(end_ribosome.transcript_synthesis_rate > 0.0);
        assert!(end_ribosome.protein_synthesis_rate > 0.0);
        assert!(end_ribosome.transcript_abundance != start_ribosome.transcript_abundance);
        assert!(end_ribosome.protein_abundance != start_ribosome.protein_abundance);
    }

    #[test]
    fn test_expression_execution_state_tracks_occupancy_and_product_pools() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

        sim.run(12);

        let expression = sim
            .organism_expression_state()
            .expect("updated organism expression");
        let ribosome_unit = expression
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("ribosome operon");

        assert!(ribosome_unit.transcription_length_nt >= 90.0);
        assert!(ribosome_unit.translation_length_aa >= 30.0);
        assert!(ribosome_unit.promoter_open_fraction > 0.0);
        assert!(ribosome_unit.active_rnap_occupancy > 0.0);
        assert!(ribosome_unit.active_ribosome_occupancy > 0.0);
        assert!(ribosome_unit.mature_transcript_abundance > 0.0);
        assert!(ribosome_unit.mature_protein_abundance > 0.0);
        assert!(ribosome_unit.transcript_abundance >= ribosome_unit.mature_transcript_abundance);
        assert!(ribosome_unit.protein_abundance >= ribosome_unit.mature_protein_abundance);
    }

    #[test]
    fn test_saved_state_round_trip_preserves_expression_execution_state() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

        sim.run(10);

        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
        let original = sim
            .organism_expression_state()
            .expect("original expression state");
        let reloaded = restored
            .organism_expression_state()
            .expect("restored expression state");
        let original_unit = original
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("original ribosome unit");
        let reloaded_unit = reloaded
            .transcription_units
            .iter()
            .find(|unit| unit.name == "ribosome_biogenesis_operon")
            .expect("reloaded ribosome unit");

        assert!(
            (reloaded_unit.promoter_open_fraction - original_unit.promoter_open_fraction).abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.active_rnap_occupancy - original_unit.active_rnap_occupancy).abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.transcription_progress_nt - original_unit.transcription_progress_nt)
                .abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.mature_transcript_abundance - original_unit.mature_transcript_abundance)
                .abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.active_ribosome_occupancy - original_unit.active_ribosome_occupancy)
                .abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.translation_progress_aa - original_unit.translation_progress_aa).abs()
                < 1.0e-6
        );
        assert!(
            (reloaded_unit.mature_protein_abundance - original_unit.mature_protein_abundance).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn test_chromosome_state_tracks_forks_loci_and_restart() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.dnaa = 128.0;
        sim.complex_assembly.replisome_complexes = 24.0;
        sim.chromosome_state.replicated_bp = 2;
        sim.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
            sim.chromosome_state.chromosome_length_bp,
            sim.chromosome_state.origin_bp,
            sim.chromosome_state.terminus_bp,
            2,
        );
        sim.synchronize_chromosome_summary();
        let start = sim.chromosome_state();

        assert_eq!(start.chromosome_length_bp, sim.genome_bp);
        assert!(!start.loci.is_empty());

        sim.run(24);

        let end = sim.chromosome_state();
        assert!(end.initiation_events >= start.initiation_events);
        assert!(!end.forks.is_empty());
        assert!(end.replicated_bp > 0);
        assert!(end.mean_locus_accessibility > 0.0);
        assert!(end
            .loci
            .iter()
            .zip(start.loci.iter())
            .any(|(end_locus, start_locus)| {
                (end_locus.accessibility - start_locus.accessibility).abs() > 1.0e-6
                    || (end_locus.torsional_stress - start_locus.torsional_stress).abs() > 1.0e-6
            }));

        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
        let restored_state = restored.chromosome_state();

        assert_eq!(restored_state.loci.len(), end.loci.len());
        assert_eq!(restored_state.forks.len(), end.forks.len());
        assert_eq!(restored_state.initiation_events, end.initiation_events);
        assert_eq!(restored_state.completion_events, end.completion_events);
        assert_eq!(restored_state.replicated_bp, end.replicated_bp);
        assert!(
            (restored_state.mean_locus_accessibility - end.mean_locus_accessibility).abs() < 1.0e-6
        );
        if let (Some(restored_fork), Some(end_fork)) =
            (restored_state.forks.first(), end.forks.first())
        {
            assert_eq!(restored_fork.pause_events, end_fork.pause_events);
            assert_eq!(restored_fork.traveled_bp, end_fork.traveled_bp);
            assert!(
                (restored_fork.collision_pressure - end_fork.collision_pressure).abs() < 1.0e-6
            );
        }
    }

    #[test]
    fn test_head_on_transcription_increases_chromosome_collision_pressure() {
        let mut baseline =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        baseline.dnaa = 128.0;
        baseline.complex_assembly.replisome_complexes = 24.0;
        baseline.chromosome_state.replicated_bp = 2;
        baseline.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
            baseline.chromosome_state.chromosome_length_bp,
            baseline.chromosome_state.origin_bp,
            baseline.chromosome_state.terminus_bp,
            2,
        );
        baseline.synchronize_chromosome_summary();
        baseline.run(24);
        let baseline_state = baseline.chromosome_state();
        let baseline_collision = baseline_state
            .forks
            .iter()
            .map(|fork| fork.collision_pressure)
            .sum::<f32>();
        let baseline_pauses = baseline_state
            .forks
            .iter()
            .map(|fork| fork.pause_events)
            .sum::<u32>();

        let mut stressed =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        stressed.dnaa = 128.0;
        stressed.complex_assembly.replisome_complexes = 24.0;
        stressed.chromosome_state.replicated_bp = 2;
        stressed.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
            stressed.chromosome_state.chromosome_length_bp,
            stressed.chromosome_state.origin_bp,
            stressed.chromosome_state.terminus_bp,
            2,
        );
        stressed.synchronize_chromosome_summary();
        if let Some(organism) = stressed.organism_data.as_mut() {
            for feature in &mut organism.genes {
                feature.strand = -1;
                feature.basal_expression *= 18.0;
            }
        }
        stressed.run(24);
        let stressed_state = stressed.chromosome_state();
        let stressed_collision = stressed_state
            .forks
            .iter()
            .map(|fork| fork.collision_pressure)
            .sum::<f32>();
        let stressed_pauses = stressed_state
            .forks
            .iter()
            .map(|fork| fork.pause_events)
            .sum::<u32>();

        assert!(stressed_collision >= baseline_collision);
        assert!(stressed_pauses >= baseline_pauses);
        assert!(stressed_state.torsional_stress >= baseline_state.torsional_stress);
    }

    #[test]
    fn test_membrane_division_state_persists_across_restart() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

        sim.run(16);

        let original = sim.membrane_division_state();
        assert!(original.membrane_area_nm2 > 0.0);
        assert!(original.preferred_membrane_area_nm2 > 0.0);
        assert!(original.envelope_integrity > 0.0);

        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
        let reloaded = restored.membrane_division_state();

        assert!((reloaded.membrane_area_nm2 - original.membrane_area_nm2).abs() < 1.0e-6);
        assert!(
            (reloaded.preferred_membrane_area_nm2 - original.preferred_membrane_area_nm2).abs()
                < 1.0e-6
        );
        assert!((reloaded.divisome_occupancy - original.divisome_occupancy).abs() < 1.0e-6);
        assert!((reloaded.ring_tension - original.ring_tension).abs() < 1.0e-6);
        assert!((reloaded.chromosome_occlusion - original.chromosome_occlusion).abs() < 1.0e-6);
        assert!((reloaded.failure_pressure - original.failure_pressure).abs() < 1.0e-6);
    }

    #[test]
    fn test_chromosome_occlusion_penalizes_division_mechanics() {
        let mut occluded =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let mut cleared =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");

        for sim in [&mut occluded, &mut cleared] {
            let genome_bp = sim.chromosome_state.chromosome_length_bp.max(1);
            sim.chromosome_state.replicated_bp = genome_bp / 2;
            sim.chromosome_state.forks = WholeCellSimulator::chromosome_forks_from_progress(
                genome_bp,
                sim.chromosome_state.origin_bp,
                sim.chromosome_state.terminus_bp,
                genome_bp / 2,
            );
            sim.chromosome_state = sim.normalize_chromosome_state(sim.chromosome_state.clone());
            sim.synchronize_chromosome_summary();
            sim.membrane_division_state = sim.seeded_membrane_division_state();
            sim.synchronize_membrane_division_summary();
        }

        occluded.chromosome_state.segregation_progress = 0.0;
        occluded.chromosome_state.compaction_fraction = 0.85;
        occluded.chromosome_state =
            occluded.normalize_chromosome_state(occluded.chromosome_state.clone());
        occluded.synchronize_chromosome_summary();

        cleared.chromosome_state.segregation_progress = 0.90;
        cleared.chromosome_state.compaction_fraction = 0.35;
        cleared.chromosome_state =
            cleared.normalize_chromosome_state(cleared.chromosome_state.clone());
        cleared.synchronize_chromosome_summary();

        occluded.run(24);
        cleared.run(24);

        let occluded_state = occluded.membrane_division_state();
        let cleared_state = cleared.membrane_division_state();

        assert!(occluded_state.chromosome_occlusion > cleared_state.chromosome_occlusion);
        assert!(cleared_state.divisome_occupancy > occluded_state.divisome_occupancy);
        assert!(cleared_state.ring_occupancy > occluded_state.ring_occupancy);
        assert!(cleared.snapshot().division_progress > occluded.snapshot().division_progress);
    }

    #[test]
    fn test_complex_assembly_state_accumulates_from_inventory_and_preserves_targets() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let start = sim.complex_assembly_state();

        sim.run(12);

        let end = sim.complex_assembly_state();
        assert!(end.total_complexes() > 0.0);
        assert!(end.ribosome_target > 0.0);
        assert!(end.ribosome_assembly_rate > 0.0);
        assert!(end.rnap_assembly_rate > 0.0);
        assert!(end.ribosome_complexes != start.ribosome_complexes);
        assert!(end.rnap_complexes != start.rnap_complexes);
    }

    #[test]
    fn test_named_complex_state_is_compiled_from_assets_and_persists_across_restart() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let start_named = sim.named_complexes_state();
        let asset_package = sim
            .organism_asset_package()
            .expect("compiled genome asset package");

        assert_eq!(start_named.len(), asset_package.complexes.len());
        let start_ribosome = start_named
            .iter()
            .find(|state| state.id == "ribosome_biogenesis_operon_complex")
            .expect("ribosome complex state");
        assert_eq!(start_ribosome.family, WholeCellAssemblyFamily::Ribosome);
        assert!(start_ribosome.subunit_pool > 0.0);
        assert!(start_ribosome.nucleation_intermediate > 0.0);
        assert!(start_ribosome.elongation_intermediate > 0.0);
        assert!(start_ribosome.abundance > 0.0);
        assert!(start_ribosome.component_satisfaction > 0.0);
        assert!(start_ribosome.structural_support > 0.0);
        assert!(start_ribosome.assembly_progress > 0.0);
        assert!(start_ribosome.limiting_component_signal > 0.0);
        assert!(start_ribosome.insertion_progress >= 0.0);

        sim.run(12);

        let end_named = sim.named_complexes_state();
        let end_ribosome = end_named
            .iter()
            .find(|state| state.id == "ribosome_biogenesis_operon_complex")
            .expect("updated ribosome complex state");
        assert!(end_ribosome.target_abundance > 0.0);
        assert!(end_ribosome.assembly_rate > 0.0);
        assert!(end_ribosome.nucleation_rate > 0.0);
        assert!(end_ribosome.elongation_rate > 0.0);
        assert!(end_ribosome.maturation_rate > 0.0);
        assert!(end_ribosome.abundance != start_ribosome.abundance);
        assert!(end_ribosome.assembly_progress != start_ribosome.assembly_progress);
        assert!(end_ribosome.failure_count >= start_ribosome.failure_count);

        let saved = sim.save_state_json().expect("serialize saved state");
        let restored =
            WholeCellSimulator::from_saved_state_json(&saved).expect("restore saved state");
        let restored_named = restored.named_complexes_state();
        let restored_ribosome = restored_named
            .iter()
            .find(|state| state.id == "ribosome_biogenesis_operon_complex")
            .expect("restored ribosome complex state");

        assert_eq!(restored_named.len(), end_named.len());
        assert!((restored_ribosome.subunit_pool - end_ribosome.subunit_pool).abs() < 1.0e-6);
        assert!(
            (restored_ribosome.nucleation_intermediate - end_ribosome.nucleation_intermediate)
                .abs()
                < 1.0e-6
        );
        assert!(
            (restored_ribosome.elongation_intermediate - end_ribosome.elongation_intermediate)
                .abs()
                < 1.0e-6
        );
        assert!((restored_ribosome.abundance - end_ribosome.abundance).abs() < 1.0e-6);
        assert!(
            (restored_ribosome.target_abundance - end_ribosome.target_abundance).abs() < 1.0e-6
        );
        assert!(
            (restored_ribosome.shared_component_pressure - end_ribosome.shared_component_pressure)
                .abs()
                < 1.0e-6
        );
        assert!(
            (restored_ribosome.limiting_component_signal - end_ribosome.limiting_component_signal)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn test_named_complex_state_tracks_stall_and_damage_under_stress() {
        let mut baseline =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        let mut stressed =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        stressed.metabolic_load = 2.5;
        stressed.chemistry_report.atp_support = 0.62;
        stressed.chemistry_report.translation_support = 0.60;
        stressed.chemistry_report.membrane_support = 0.58;
        stressed.chemistry_report.crowding_penalty = 0.72;

        baseline.run(16);
        stressed.run(16);

        let stressed_complex = stressed
            .named_complexes_state()
            .into_iter()
            .find(|state| {
                matches!(
                    state.family,
                    WholeCellAssemblyFamily::AtpSynthase
                        | WholeCellAssemblyFamily::Transporter
                        | WholeCellAssemblyFamily::MembraneEnzyme
                        | WholeCellAssemblyFamily::Divisome
                )
            })
            .expect("stressed membrane-coupled complex");
        let baseline_complex = baseline
            .named_complexes_state()
            .into_iter()
            .find(|state| state.id == stressed_complex.id)
            .expect("baseline matching complex");

        assert!(stressed_complex.stalled_intermediate >= baseline_complex.stalled_intermediate);
        assert!(
            stressed_complex.shared_component_pressure
                >= baseline_complex.shared_component_pressure
        );
        assert!(stressed_complex.failure_count >= baseline_complex.failure_count);
        assert!(
            stressed_complex.insertion_progress <= baseline_complex.insertion_progress + 1.0e-6
        );
    }

    #[test]
    fn test_process_capacity_tracks_persistent_complex_assembly_state() {
        let mut sim =
            WholeCellSimulator::bundled_syn3a_reference().expect("bundled Syn3A simulator");
        sim.refresh_organism_expression_state();
        sim.initialize_complex_assembly_state();

        let baseline_inventory = sim.assembly_inventory();
        let baseline_fluxes = sim.process_fluxes(baseline_inventory);

        for state in &mut sim.named_complexes {
            state.subunit_pool *= 0.10;
            state.nucleation_intermediate *= 0.10;
            state.elongation_intermediate *= 0.10;
            state.abundance *= 0.10;
            state.target_abundance *= 0.10;
        }
        let assets = sim
            .organism_assets
            .clone()
            .expect("compiled genome asset package");
        sim.complex_assembly = sim.aggregate_named_complex_assembly_state(&assets);
        let reduced_inventory = sim.assembly_inventory();
        let reduced_fluxes = sim.process_fluxes(reduced_inventory);

        assert!(reduced_inventory.ribosome_complexes < baseline_inventory.ribosome_complexes);
        assert!(reduced_inventory.rnap_complexes < baseline_inventory.rnap_complexes);
        assert!(reduced_inventory.replisome_complexes < baseline_inventory.replisome_complexes);
        assert!(reduced_fluxes.translation_capacity <= baseline_fluxes.translation_capacity);
        assert!(reduced_fluxes.transcription_capacity <= baseline_fluxes.transcription_capacity);
        assert!(reduced_fluxes.replication_capacity <= baseline_fluxes.replication_capacity);
    }

    #[test]
    fn test_organism_descriptor_drives_division_and_replication_scales() {
        let baseline_spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
        let mut constrained_spec = baseline_spec.clone();
        let organism = constrained_spec
            .organism_data
            .as_mut()
            .expect("bundled organism data");
        for gene in &mut organism.genes {
            gene.process_weights.replication *= 0.35;
            gene.process_weights.segregation *= 0.35;
            gene.process_weights.membrane *= 0.40;
            gene.process_weights.constriction *= 0.30;
        }
        for unit in &mut organism.transcription_units {
            unit.process_weights.replication *= 0.35;
            unit.process_weights.segregation *= 0.35;
            unit.process_weights.membrane *= 0.40;
            unit.process_weights.constriction *= 0.30;
        }

        let mut baseline = WholeCellSimulator::from_program_spec(baseline_spec);
        let mut constrained = WholeCellSimulator::from_program_spec(constrained_spec);
        let baseline_expression = baseline
            .organism_expression_state()
            .expect("baseline organism expression");
        let constrained_expression = constrained
            .organism_expression_state()
            .expect("constrained organism expression");
        let baseline_complex = baseline.complex_assembly_state();
        let constrained_complex = constrained.complex_assembly_state();

        assert!(
            baseline_expression.process_scales.replication
                > constrained_expression.process_scales.replication
        );
        assert!(
            baseline_expression.process_scales.membrane
                > constrained_expression.process_scales.membrane
        );
        assert!(
            baseline_expression.process_scales.constriction
                > constrained_expression.process_scales.constriction
        );
        assert!(baseline_complex.replisome_target > constrained_complex.replisome_target);
        assert!(baseline_complex.ftsz_target > constrained_complex.ftsz_target);

        baseline.run(96);
        constrained.run(96);

        let baseline_snapshot = baseline.snapshot();
        let constrained_snapshot = constrained.snapshot();
        let baseline_membrane = baseline.membrane_division_state();
        let constrained_membrane = constrained.membrane_division_state();

        assert!(baseline_snapshot.replicated_bp >= constrained_snapshot.replicated_bp);
        assert!(baseline_membrane.divisome_occupancy > constrained_membrane.divisome_occupancy);
        assert!(baseline_membrane.ring_tension > constrained_membrane.ring_tension);
        assert!(baseline_snapshot.division_progress >= constrained_snapshot.division_progress);
        assert!(baseline_snapshot.surface_area_nm2 > constrained_snapshot.surface_area_nm2);
    }
}
