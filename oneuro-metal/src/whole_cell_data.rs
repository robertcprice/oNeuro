//! Data-driven whole-cell program and saved-state payloads.
//!
//! The native runtime already supports program-spec initialization and JSON
//! save/restore. This module is the serialized contract behind those flows.

use crate::whole_cell::{WholeCellConfig, WholeCellQuantumProfile};
use crate::whole_cell_submodels::{
    LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeReport, ScheduledSubsystemProbe,
    Syn3ASubsystemPreset, WholeCellSubsystemState,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const BUNDLED_SYN3A_PROGRAM_JSON: &str = include_str!("../specs/whole_cell_syn3a_reference.json");
const BUNDLED_SYN3A_BUNDLE_MANIFEST_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/manifest.json");
const BUNDLED_SYN3A_BUNDLE_METADATA_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/metadata.json");
const BUNDLED_SYN3A_BUNDLE_GENE_FEATURES_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_features.json");
const BUNDLED_SYN3A_BUNDLE_GENE_PRODUCTS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_products.json");
const BUNDLED_SYN3A_BUNDLE_GENE_SEMANTICS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/gene_semantics.json");
const BUNDLED_SYN3A_BUNDLE_TRANSCRIPTION_UNITS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/transcription_units.json");
const BUNDLED_SYN3A_BUNDLE_TRANSCRIPTION_UNIT_SEMANTICS_JSON: &str = include_str!(
    "../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/transcription_unit_semantics.json"
);
const BUNDLED_SYN3A_BUNDLE_CHROMOSOME_DOMAINS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/chromosome_domains.json");
const BUNDLED_SYN3A_BUNDLE_POOLS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/pools.json");
const BUNDLED_SYN3A_BUNDLE_OPERONS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/operons.json");
const BUNDLED_SYN3A_BUNDLE_RNAS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/rnas.json");
const BUNDLED_SYN3A_BUNDLE_PROTEINS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/proteins.json");
const BUNDLED_SYN3A_BUNDLE_COMPLEXES_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/complexes.json");
const BUNDLED_SYN3A_BUNDLE_OPERON_SEMANTICS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/operon_semantics.json");
const BUNDLED_SYN3A_BUNDLE_PROTEIN_SEMANTICS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/protein_semantics.json");
const BUNDLED_SYN3A_BUNDLE_COMPLEX_SEMANTICS_JSON: &str =
    include_str!("../../src/oneuro/whole_cell/assets/bundles/jcvi_syn3a/complex_semantics.json");
pub const WHOLE_CELL_CONTRACT_VERSION: &str = "whole_cell_phase0";
pub const WHOLE_CELL_PROGRAM_SCHEMA_VERSION: u32 = 1;
pub const WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION: u32 = 1;
pub const WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellContractSchema {
    #[serde(default = "default_contract_version")]
    pub contract_version: String,
    #[serde(default = "default_program_schema_version")]
    pub program_schema_version: u32,
    #[serde(default = "default_saved_state_schema_version")]
    pub saved_state_schema_version: u32,
    #[serde(default = "default_runtime_manifest_schema_version")]
    pub runtime_manifest_schema_version: u32,
}

impl Default for WholeCellContractSchema {
    fn default() -> Self {
        Self {
            contract_version: default_contract_version(),
            program_schema_version: default_program_schema_version(),
            saved_state_schema_version: default_saved_state_schema_version(),
            runtime_manifest_schema_version: default_runtime_manifest_schema_version(),
        }
    }
}

impl WholeCellContractSchema {
    pub fn normalized_for_program(mut self) -> Self {
        if self.contract_version.trim().is_empty() {
            self.contract_version = default_contract_version();
        }
        self.program_schema_version = default_program_schema_version();
        if self.saved_state_schema_version == 0 {
            self.saved_state_schema_version = default_saved_state_schema_version();
        }
        if self.runtime_manifest_schema_version == 0 {
            self.runtime_manifest_schema_version = default_runtime_manifest_schema_version();
        }
        self
    }

    pub fn normalized_for_saved_state(mut self) -> Self {
        self = self.normalized_for_program();
        self.saved_state_schema_version = default_saved_state_schema_version();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WholeCellProvenance {
    #[serde(default)]
    pub source_dataset: Option<String>,
    #[serde(default)]
    pub organism_asset_hash: Option<String>,
    #[serde(default)]
    pub compiled_ir_hash: Option<String>,
    #[serde(default)]
    pub calibration_bundle_hash: Option<String>,
    #[serde(default)]
    pub run_manifest_hash: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellInitialLatticeSpec {
    pub atp: f32,
    pub amino_acids: f32,
    pub nucleotides: f32,
    pub membrane_precursors: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellInitialStateSpec {
    pub adp_mm: f32,
    pub glucose_mm: f32,
    pub oxygen_mm: f32,
    pub genome_bp: u32,
    pub replicated_bp: u32,
    pub chromosome_separation_nm: f32,
    pub radius_nm: f32,
    pub division_progress: f32,
    pub metabolic_load: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct WholeCellProcessWeights {
    #[serde(default)]
    pub energy: f32,
    #[serde(default)]
    pub transcription: f32,
    #[serde(default)]
    pub translation: f32,
    #[serde(default)]
    pub replication: f32,
    #[serde(default)]
    pub segregation: f32,
    #[serde(default)]
    pub membrane: f32,
    #[serde(default)]
    pub constriction: f32,
}

impl WholeCellProcessWeights {
    pub fn clamped(self) -> Self {
        Self {
            energy: self.energy.max(0.0),
            transcription: self.transcription.max(0.0),
            translation: self.translation.max(0.0),
            replication: self.replication.max(0.0),
            segregation: self.segregation.max(0.0),
            membrane: self.membrane.max(0.0),
            constriction: self.constriction.max(0.0),
        }
    }

    pub fn total(self) -> f32 {
        self.energy
            + self.transcription
            + self.translation
            + self.replication
            + self.segregation
            + self.membrane
            + self.constriction
    }

    pub fn add_weighted(&mut self, other: Self, scale: f32) {
        let other = other.clamped();
        let scale = scale.max(0.0);
        self.energy += other.energy * scale;
        self.transcription += other.transcription * scale;
        self.translation += other.translation * scale;
        self.replication += other.replication * scale;
        self.segregation += other.segregation * scale;
        self.membrane += other.membrane * scale;
        self.constriction += other.constriction * scale;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGeometryPrior {
    pub radius_nm: f32,
    #[serde(default = "default_chromosome_radius_fraction")]
    pub chromosome_radius_fraction: f32,
    #[serde(default = "default_membrane_fraction")]
    pub membrane_fraction: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellCompositionPrior {
    #[serde(default = "default_dry_mass_fg")]
    pub dry_mass_fg: f32,
    #[serde(default = "default_gc_fraction")]
    pub gc_fraction: f32,
    #[serde(default = "default_protein_fraction")]
    pub protein_fraction: f32,
    #[serde(default = "default_rna_fraction")]
    pub rna_fraction: f32,
    #[serde(default = "default_lipid_fraction")]
    pub lipid_fraction: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellMoleculePoolSpec {
    pub species: String,
    #[serde(default)]
    pub bulk_field: Option<WholeCellBulkField>,
    #[serde(default)]
    pub role: Option<WholeCellPoolRole>,
    #[serde(default)]
    pub concentration_mm: f32,
    #[serde(default)]
    pub count: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeFeature {
    pub gene: String,
    pub start_bp: u32,
    pub end_bp: u32,
    #[serde(default)]
    pub strand: i8,
    #[serde(default)]
    pub essential: bool,
    #[serde(default = "default_expression_level")]
    pub basal_expression: f32,
    #[serde(default = "default_translation_cost")]
    pub translation_cost: f32,
    #[serde(default = "default_nucleotide_cost")]
    pub nucleotide_cost: f32,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellTranscriptionUnitSpec {
    pub name: String,
    #[serde(default)]
    pub genes: Vec<String>,
    #[serde(default = "default_expression_level")]
    pub basal_activity: f32,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismSpec {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub origin_bp: u32,
    pub terminus_bp: u32,
    pub geometry: WholeCellGeometryPrior,
    pub composition: WholeCellCompositionPrior,
    #[serde(default)]
    pub chromosome_domains: Vec<WholeCellChromosomeDomainSpec>,
    #[serde(default)]
    pub pools: Vec<WholeCellMoleculePoolSpec>,
    #[serde(default)]
    pub genes: Vec<WholeCellGenomeFeature>,
    #[serde(default)]
    pub transcription_units: Vec<WholeCellTranscriptionUnitSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellChromosomeDomainSpec {
    pub id: String,
    pub start_bp: u32,
    pub end_bp: u32,
    #[serde(default)]
    pub axial_center_fraction: f32,
    #[serde(default = "default_chromosome_domain_spread_fraction")]
    pub axial_spread_fraction: f32,
    #[serde(default)]
    pub genes: Vec<String>,
    #[serde(default)]
    pub transcription_units: Vec<String>,
    #[serde(default)]
    pub operons: Vec<String>,
}

fn default_chromosome_domain_spread_fraction() -> f32 {
    0.16
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellAssetClass {
    Energy,
    Translation,
    Replication,
    Segregation,
    Membrane,
    Constriction,
    QualityControl,
    Homeostasis,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellBulkField {
    ATP,
    ADP,
    Glucose,
    Oxygen,
    AminoAcids,
    Nucleotides,
    MembranePrecursors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellPoolRole {
    ActiveRibosomes,
    ActiveRnap,
    Dnaa,
    Ftsz,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOperonSpec {
    pub name: String,
    #[serde(default)]
    pub genes: Vec<String>,
    pub promoter_bp: u32,
    pub terminator_bp: u32,
    pub basal_activity: f32,
    pub polycistronic: bool,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOperonSemanticSpec {
    pub name: String,
    pub asset_class: WholeCellAssetClass,
    #[serde(default = "default_assembly_family")]
    pub complex_family: WholeCellAssemblyFamily,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellRnaProductSpec {
    pub id: String,
    pub gene: String,
    pub operon: String,
    pub length_nt: u32,
    pub basal_abundance: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellProteinProductSpec {
    pub id: String,
    pub gene: String,
    pub operon: String,
    pub rna_id: String,
    pub aa_length: u32,
    pub basal_abundance: f32,
    pub translation_cost: f32,
    pub nucleotide_cost: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellProteinSemanticSpec {
    pub id: String,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellComplexComponentSpec {
    pub protein_id: String,
    pub stoichiometry: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellComplexSpec {
    pub id: String,
    pub name: String,
    pub operon: String,
    #[serde(default)]
    pub components: Vec<WholeCellComplexComponentSpec>,
    pub basal_abundance: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default = "default_assembly_family")]
    pub family: WholeCellAssemblyFamily,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub membrane_inserted: bool,
    #[serde(default)]
    pub chromosome_coupled: bool,
    #[serde(default)]
    pub division_coupled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellComplexSemanticSpec {
    pub id: String,
    pub asset_class: WholeCellAssetClass,
    #[serde(default = "default_assembly_family")]
    pub family: WholeCellAssemblyFamily,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub membrane_inserted: bool,
    #[serde(default)]
    pub chromosome_coupled: bool,
    #[serde(default)]
    pub division_coupled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellAssemblyFamily {
    Ribosome,
    RnaPolymerase,
    Replisome,
    AtpSynthase,
    Transporter,
    MembraneEnzyme,
    ChaperoneClient,
    Divisome,
    Generic,
}

fn default_assembly_family() -> WholeCellAssemblyFamily {
    WholeCellAssemblyFamily::Generic
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeAssetPackage {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub origin_bp: u32,
    pub terminus_bp: u32,
    #[serde(default)]
    pub chromosome_domains: Vec<WholeCellChromosomeDomainSpec>,
    #[serde(default)]
    pub operons: Vec<WholeCellOperonSpec>,
    #[serde(default)]
    pub operon_semantics: Vec<WholeCellOperonSemanticSpec>,
    #[serde(default)]
    pub rnas: Vec<WholeCellRnaProductSpec>,
    #[serde(default)]
    pub proteins: Vec<WholeCellProteinProductSpec>,
    #[serde(default)]
    pub protein_semantics: Vec<WholeCellProteinSemanticSpec>,
    #[serde(default)]
    pub complex_semantics: Vec<WholeCellComplexSemanticSpec>,
    #[serde(default)]
    pub complexes: Vec<WholeCellComplexSpec>,
    #[serde(default)]
    pub pools: Vec<WholeCellMoleculePoolSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellSpeciesClass {
    Pool,
    Rna,
    Protein,
    ComplexSubunitPool,
    ComplexNucleationIntermediate,
    ComplexElongationIntermediate,
    ComplexMature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellReactionClass {
    PoolTransport,
    LocalizedPoolTransfer,
    LocalizedPoolTurnover,
    MembranePatchTransfer,
    MembranePatchTurnover,
    Transcription,
    Translation,
    RnaDegradation,
    ProteinDegradation,
    StressResponse,
    SubunitPoolFormation,
    ComplexNucleation,
    ComplexElongation,
    ComplexMaturation,
    ComplexRepair,
    ComplexTurnover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellSpatialScope {
    #[default]
    WellMixed,
    MembraneAdjacent,
    SeptumLocal,
    NucleoidLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellPatchDomain {
    #[default]
    Distributed,
    MembraneBand,
    SeptumPatch,
    PolarPatch,
    NucleoidTrack,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSpeciesSpec {
    pub id: String,
    pub name: String,
    pub species_class: WholeCellSpeciesClass,
    pub compartment: String,
    pub asset_class: WholeCellAssetClass,
    pub basal_abundance: f32,
    #[serde(default)]
    pub bulk_field: Option<WholeCellBulkField>,
    #[serde(default)]
    pub operon: Option<String>,
    #[serde(default)]
    pub parent_complex: Option<String>,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub spatial_scope: WholeCellSpatialScope,
    #[serde(default)]
    pub patch_domain: WholeCellPatchDomain,
    #[serde(default)]
    pub chromosome_domain: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellReactionParticipantSpec {
    pub species_id: String,
    pub stoichiometry: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellReactionSpec {
    pub id: String,
    pub name: String,
    pub reaction_class: WholeCellReactionClass,
    pub asset_class: WholeCellAssetClass,
    pub nominal_rate: f32,
    #[serde(default)]
    pub catalyst: Option<String>,
    #[serde(default)]
    pub operon: Option<String>,
    #[serde(default)]
    pub reactants: Vec<WholeCellReactionParticipantSpec>,
    #[serde(default)]
    pub products: Vec<WholeCellReactionParticipantSpec>,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub spatial_scope: WholeCellSpatialScope,
    #[serde(default)]
    pub patch_domain: WholeCellPatchDomain,
    #[serde(default)]
    pub chromosome_domain: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeProcessRegistry {
    pub organism: String,
    #[serde(default)]
    pub chromosome_domains: Vec<WholeCellChromosomeDomainSpec>,
    #[serde(default)]
    pub species: Vec<WholeCellSpeciesSpec>,
    #[serde(default)]
    pub reactions: Vec<WholeCellReactionSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSpeciesRuntimeState {
    pub id: String,
    pub name: String,
    pub species_class: WholeCellSpeciesClass,
    pub compartment: String,
    pub asset_class: WholeCellAssetClass,
    pub basal_abundance: f32,
    #[serde(default)]
    pub bulk_field: Option<WholeCellBulkField>,
    #[serde(default)]
    pub operon: Option<String>,
    #[serde(default)]
    pub parent_complex: Option<String>,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub spatial_scope: WholeCellSpatialScope,
    #[serde(default)]
    pub patch_domain: WholeCellPatchDomain,
    #[serde(default)]
    pub chromosome_domain: Option<String>,
    pub count: f32,
    #[serde(default)]
    pub anchor_count: f32,
    #[serde(default)]
    pub synthesis_rate: f32,
    #[serde(default)]
    pub turnover_rate: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellReactionRuntimeState {
    pub id: String,
    pub name: String,
    pub reaction_class: WholeCellReactionClass,
    pub asset_class: WholeCellAssetClass,
    pub nominal_rate: f32,
    #[serde(default)]
    pub catalyst: Option<String>,
    #[serde(default)]
    pub operon: Option<String>,
    #[serde(default)]
    pub reactants: Vec<WholeCellReactionParticipantSpec>,
    #[serde(default)]
    pub products: Vec<WholeCellReactionParticipantSpec>,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub spatial_scope: WholeCellSpatialScope,
    #[serde(default)]
    pub patch_domain: WholeCellPatchDomain,
    #[serde(default)]
    pub chromosome_domain: Option<String>,
    #[serde(default)]
    pub current_flux: f32,
    #[serde(default)]
    pub cumulative_extent: f32,
    #[serde(default)]
    pub reactant_satisfaction: f32,
    #[serde(default = "default_scale")]
    pub catalyst_support: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellGenomeProcessRegistrySummary {
    pub organism: String,
    pub species_count: usize,
    pub pool_species_count: usize,
    pub rna_species_count: usize,
    pub protein_species_count: usize,
    pub complex_species_count: usize,
    pub assembly_intermediate_species_count: usize,
    pub reaction_count: usize,
    pub transcription_reaction_count: usize,
    pub translation_reaction_count: usize,
    pub transport_reaction_count: usize,
    pub degradation_reaction_count: usize,
    pub stress_reaction_count: usize,
    pub assembly_reaction_count: usize,
    pub repair_reaction_count: usize,
    pub turnover_reaction_count: usize,
}

impl From<&WholeCellGenomeProcessRegistry> for WholeCellGenomeProcessRegistrySummary {
    fn from(registry: &WholeCellGenomeProcessRegistry) -> Self {
        let pool_species_count = registry
            .species
            .iter()
            .filter(|species| species.species_class == WholeCellSpeciesClass::Pool)
            .count();
        let rna_species_count = registry
            .species
            .iter()
            .filter(|species| species.species_class == WholeCellSpeciesClass::Rna)
            .count();
        let protein_species_count = registry
            .species
            .iter()
            .filter(|species| species.species_class == WholeCellSpeciesClass::Protein)
            .count();
        let complex_species_count = registry
            .species
            .iter()
            .filter(|species| species.species_class == WholeCellSpeciesClass::ComplexMature)
            .count();
        let assembly_intermediate_species_count = registry
            .species
            .iter()
            .filter(|species| {
                matches!(
                    species.species_class,
                    WholeCellSpeciesClass::ComplexSubunitPool
                        | WholeCellSpeciesClass::ComplexNucleationIntermediate
                        | WholeCellSpeciesClass::ComplexElongationIntermediate
                )
            })
            .count();
        let transcription_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::Transcription)
            .count();
        let translation_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::Translation)
            .count();
        let transport_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| {
                matches!(
                    reaction.reaction_class,
                    WholeCellReactionClass::PoolTransport
                        | WholeCellReactionClass::LocalizedPoolTransfer
                        | WholeCellReactionClass::MembranePatchTransfer
                )
            })
            .count();
        let degradation_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| {
                matches!(
                    reaction.reaction_class,
                    WholeCellReactionClass::RnaDegradation
                        | WholeCellReactionClass::ProteinDegradation
                )
            })
            .count();
        let stress_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::StressResponse)
            .count();
        let assembly_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| {
                matches!(
                    reaction.reaction_class,
                    WholeCellReactionClass::SubunitPoolFormation
                        | WholeCellReactionClass::ComplexNucleation
                        | WholeCellReactionClass::ComplexElongation
                        | WholeCellReactionClass::ComplexMaturation
                )
            })
            .count();
        let repair_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::ComplexRepair)
            .count();
        let turnover_reaction_count = registry
            .reactions
            .iter()
            .filter(|reaction| {
                matches!(
                    reaction.reaction_class,
                    WholeCellReactionClass::ComplexTurnover
                        | WholeCellReactionClass::LocalizedPoolTurnover
                        | WholeCellReactionClass::MembranePatchTurnover
                )
            })
            .count();
        Self {
            organism: registry.organism.clone(),
            species_count: registry.species.len(),
            pool_species_count,
            rna_species_count,
            protein_species_count,
            complex_species_count,
            assembly_intermediate_species_count,
            reaction_count: registry.reactions.len(),
            transcription_reaction_count,
            translation_reaction_count,
            transport_reaction_count,
            degradation_reaction_count,
            stress_reaction_count,
            assembly_reaction_count,
            repair_reaction_count,
            turnover_reaction_count,
        }
    }
}

pub fn initialize_runtime_species_state(
    registry: &WholeCellGenomeProcessRegistry,
) -> Vec<WholeCellSpeciesRuntimeState> {
    registry
        .species
        .iter()
        .map(|species| WholeCellSpeciesRuntimeState {
            id: species.id.clone(),
            name: species.name.clone(),
            species_class: species.species_class,
            compartment: species.compartment.clone(),
            asset_class: species.asset_class,
            basal_abundance: species.basal_abundance.max(0.0),
            bulk_field: species.bulk_field,
            operon: species.operon.clone(),
            parent_complex: species.parent_complex.clone(),
            subsystem_targets: species.subsystem_targets.clone(),
            spatial_scope: species.spatial_scope,
            patch_domain: species.patch_domain,
            chromosome_domain: species.chromosome_domain.clone(),
            count: species.basal_abundance.max(0.0),
            anchor_count: species.basal_abundance.max(0.0),
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        })
        .collect()
}

pub fn initialize_runtime_reaction_state(
    registry: &WholeCellGenomeProcessRegistry,
) -> Vec<WholeCellReactionRuntimeState> {
    registry
        .reactions
        .iter()
        .map(|reaction| WholeCellReactionRuntimeState {
            id: reaction.id.clone(),
            name: reaction.name.clone(),
            reaction_class: reaction.reaction_class,
            asset_class: reaction.asset_class,
            nominal_rate: reaction.nominal_rate.max(0.0),
            catalyst: reaction.catalyst.clone(),
            operon: reaction.operon.clone(),
            reactants: reaction.reactants.clone(),
            products: reaction.products.clone(),
            subsystem_targets: reaction.subsystem_targets.clone(),
            spatial_scope: reaction.spatial_scope,
            patch_domain: reaction.patch_domain,
            chromosome_domain: reaction.chromosome_domain.clone(),
            current_flux: 0.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        })
        .collect()
}

pub fn parse_genome_process_registry_json(
    registry_json: &str,
) -> Result<WholeCellGenomeProcessRegistry, String> {
    serde_json::from_str(registry_json)
        .map_err(|error| format!("failed to parse genome process registry: {error}"))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellGenomeAssetSummary {
    pub organism: String,
    pub chromosome_domain_count: usize,
    pub operon_count: usize,
    pub polycistronic_operon_count: usize,
    pub rna_count: usize,
    pub protein_count: usize,
    pub complex_count: usize,
    pub targeted_complex_count: usize,
}

impl From<&WholeCellGenomeAssetPackage> for WholeCellGenomeAssetSummary {
    fn from(package: &WholeCellGenomeAssetPackage) -> Self {
        Self {
            organism: package.organism.clone(),
            chromosome_domain_count: package.chromosome_domains.len(),
            operon_count: package.operons.len(),
            polycistronic_operon_count: package
                .operons
                .iter()
                .filter(|operon| operon.polycistronic)
                .count(),
            rna_count: package.rnas.len(),
            protein_count: package.proteins.len(),
            complex_count: package.complexes.len(),
            targeted_complex_count: package
                .complexes
                .iter()
                .filter(|complex| !complex.subsystem_targets.is_empty())
                .count(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellOrganismSummary {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub chromosome_domain_count: usize,
    pub gene_count: usize,
    pub transcription_unit_count: usize,
    pub pool_count: usize,
}

impl From<&WholeCellOrganismSpec> for WholeCellOrganismSummary {
    fn from(spec: &WholeCellOrganismSpec) -> Self {
        Self {
            organism: spec.organism.clone(),
            chromosome_length_bp: spec.chromosome_length_bp,
            chromosome_domain_count: spec.chromosome_domains.len(),
            gene_count: spec.genes.len(),
            transcription_unit_count: spec.transcription_units.len(),
            pool_count: spec.pools.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismProfile {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub chromosome_domain_count: usize,
    pub gene_count: usize,
    pub transcription_unit_count: usize,
    pub pool_count: usize,
    pub essential_gene_fraction: f32,
    pub polycistronic_fraction: f32,
    pub coding_density: f32,
    pub mean_gene_length_bp: f32,
    pub process_scales: WholeCellProcessWeights,
    pub metabolic_burden_scale: f32,
    pub crowding_scale: f32,
    pub preferred_radius_nm: f32,
    pub chromosome_radius_fraction: f32,
    pub membrane_fraction: f32,
}

impl Default for WholeCellOrganismProfile {
    fn default() -> Self {
        Self {
            organism: "generic".to_string(),
            chromosome_length_bp: 1,
            chromosome_domain_count: 0,
            gene_count: 0,
            transcription_unit_count: 0,
            pool_count: 0,
            essential_gene_fraction: 0.0,
            polycistronic_fraction: 0.0,
            coding_density: 0.0,
            mean_gene_length_bp: 0.0,
            process_scales: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.0,
                translation: 1.0,
                replication: 1.0,
                segregation: 1.0,
                membrane: 1.0,
                constriction: 1.0,
            },
            metabolic_burden_scale: 1.0,
            crowding_scale: 1.0,
            preferred_radius_nm: 200.0,
            chromosome_radius_fraction: default_chromosome_radius_fraction(),
            membrane_fraction: default_membrane_fraction(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellChromosomeForkDirection {
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellChromosomeForkState {
    pub id: String,
    pub direction: WholeCellChromosomeForkDirection,
    pub position_bp: u32,
    #[serde(default)]
    pub traveled_bp: u32,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub pause_pressure: f32,
    #[serde(default)]
    pub collision_pressure: f32,
    #[serde(default)]
    pub pause_events: u32,
    #[serde(default)]
    pub completion_fraction: f32,
    #[serde(default)]
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellChromosomeLocusState {
    pub id: String,
    pub midpoint_bp: u32,
    #[serde(default)]
    pub strand: i8,
    #[serde(default = "default_copy_number")]
    pub copy_number: f32,
    #[serde(default = "default_accessibility")]
    pub accessibility: f32,
    #[serde(default)]
    pub torsional_stress: f32,
    #[serde(default)]
    pub replicated: bool,
    #[serde(default)]
    pub segregating: bool,
    #[serde(default)]
    pub domain_index: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellChromosomeState {
    pub chromosome_length_bp: u32,
    pub origin_bp: u32,
    pub terminus_bp: u32,
    #[serde(default)]
    pub initiation_potential: f32,
    #[serde(default)]
    pub initiation_events: u32,
    #[serde(default)]
    pub completion_events: u32,
    #[serde(default)]
    pub replicated_bp: u32,
    #[serde(default)]
    pub replicated_fraction: f32,
    #[serde(default)]
    pub segregation_progress: f32,
    #[serde(default = "default_compaction_fraction")]
    pub compaction_fraction: f32,
    #[serde(default)]
    pub torsional_stress: f32,
    #[serde(default = "default_accessibility")]
    pub mean_locus_accessibility: f32,
    #[serde(default)]
    pub forks: Vec<WholeCellChromosomeForkState>,
    #[serde(default)]
    pub loci: Vec<WholeCellChromosomeLocusState>,
}

impl Default for WholeCellChromosomeState {
    fn default() -> Self {
        Self {
            chromosome_length_bp: 1,
            origin_bp: 0,
            terminus_bp: 0,
            initiation_potential: 0.0,
            initiation_events: 0,
            completion_events: 0,
            replicated_bp: 0,
            replicated_fraction: 0.0,
            segregation_progress: 0.0,
            compaction_fraction: default_compaction_fraction(),
            torsional_stress: 0.0,
            mean_locus_accessibility: default_accessibility(),
            forks: Vec::new(),
            loci: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellMembraneDivisionState {
    pub membrane_area_nm2: f32,
    pub preferred_membrane_area_nm2: f32,
    pub phospholipid_inventory_nm2: f32,
    pub cardiolipin_inventory_nm2: f32,
    pub septal_lipid_inventory_nm2: f32,
    #[serde(default)]
    pub membrane_band_lipid_inventory_nm2: f32,
    #[serde(default)]
    pub polar_lipid_inventory_nm2: f32,
    #[serde(default)]
    pub membrane_protein_insertion: f32,
    #[serde(default)]
    pub insertion_debt: f32,
    #[serde(default)]
    pub curvature_stress: f32,
    #[serde(default)]
    pub septum_localization: f32,
    #[serde(default)]
    pub divisome_occupancy: f32,
    #[serde(default)]
    pub divisome_order_progress: f32,
    #[serde(default)]
    pub ring_occupancy: f32,
    #[serde(default)]
    pub ring_tension: f32,
    #[serde(default)]
    pub constriction_force: f32,
    #[serde(default = "default_septum_radius_fraction")]
    pub septum_radius_fraction: f32,
    #[serde(default = "default_septum_thickness_nm")]
    pub septum_thickness_nm: f32,
    #[serde(default = "default_envelope_integrity")]
    pub envelope_integrity: f32,
    #[serde(default = "default_osmotic_balance")]
    pub osmotic_balance: f32,
    #[serde(default)]
    pub chromosome_occlusion: f32,
    #[serde(default)]
    pub failure_pressure: f32,
    #[serde(default)]
    pub band_turnover_pressure: f32,
    #[serde(default)]
    pub pole_turnover_pressure: f32,
    #[serde(default)]
    pub septum_turnover_pressure: f32,
    #[serde(default)]
    pub scission_events: u32,
}

impl Default for WholeCellMembraneDivisionState {
    fn default() -> Self {
        Self {
            membrane_area_nm2: 1.0,
            preferred_membrane_area_nm2: 1.0,
            phospholipid_inventory_nm2: 1.0,
            cardiolipin_inventory_nm2: 0.1,
            septal_lipid_inventory_nm2: 0.0,
            membrane_band_lipid_inventory_nm2: 0.0,
            polar_lipid_inventory_nm2: 0.0,
            membrane_protein_insertion: 0.0,
            insertion_debt: 0.0,
            curvature_stress: 0.0,
            septum_localization: 0.0,
            divisome_occupancy: 0.0,
            divisome_order_progress: 0.0,
            ring_occupancy: 0.0,
            ring_tension: 0.0,
            constriction_force: 0.0,
            septum_radius_fraction: default_septum_radius_fraction(),
            septum_thickness_nm: default_septum_thickness_nm(),
            envelope_integrity: default_envelope_integrity(),
            osmotic_balance: default_osmotic_balance(),
            chromosome_occlusion: 0.0,
            failure_pressure: 0.0,
            band_turnover_pressure: 0.0,
            pole_turnover_pressure: 0.0,
            septum_turnover_pressure: 0.0,
            scission_events: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellTranscriptionUnitState {
    pub name: String,
    pub gene_count: usize,
    pub copy_gain: f32,
    pub basal_activity: f32,
    pub effective_activity: f32,
    pub support_level: f32,
    pub stress_penalty: f32,
    pub transcript_abundance: f32,
    pub protein_abundance: f32,
    pub transcript_synthesis_rate: f32,
    pub protein_synthesis_rate: f32,
    pub transcript_turnover_rate: f32,
    pub protein_turnover_rate: f32,
    #[serde(default)]
    pub promoter_open_fraction: f32,
    #[serde(default)]
    pub active_rnap_occupancy: f32,
    #[serde(default = "default_expression_length_nt")]
    pub transcription_length_nt: f32,
    #[serde(default)]
    pub transcription_progress_nt: f32,
    #[serde(default)]
    pub nascent_transcript_abundance: f32,
    #[serde(default)]
    pub mature_transcript_abundance: f32,
    #[serde(default)]
    pub damaged_transcript_abundance: f32,
    #[serde(default)]
    pub active_ribosome_occupancy: f32,
    #[serde(default = "default_expression_length_aa")]
    pub translation_length_aa: f32,
    #[serde(default)]
    pub translation_progress_aa: f32,
    #[serde(default)]
    pub nascent_protein_abundance: f32,
    #[serde(default)]
    pub mature_protein_abundance: f32,
    #[serde(default)]
    pub damaged_protein_abundance: f32,
    pub process_drive: WholeCellProcessWeights,
}

fn default_expression_length_nt() -> f32 {
    90.0
}

fn default_expression_length_aa() -> f32 {
    30.0
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellComplexAssemblyState {
    pub atp_band_complexes: f32,
    pub ribosome_complexes: f32,
    pub rnap_complexes: f32,
    pub replisome_complexes: f32,
    pub membrane_complexes: f32,
    pub ftsz_polymer: f32,
    pub dnaa_activity: f32,
    pub atp_band_target: f32,
    pub ribosome_target: f32,
    pub rnap_target: f32,
    pub replisome_target: f32,
    pub membrane_target: f32,
    pub ftsz_target: f32,
    pub dnaa_target: f32,
    pub atp_band_assembly_rate: f32,
    pub ribosome_assembly_rate: f32,
    pub rnap_assembly_rate: f32,
    pub replisome_assembly_rate: f32,
    pub membrane_assembly_rate: f32,
    pub ftsz_assembly_rate: f32,
    pub dnaa_assembly_rate: f32,
    pub atp_band_degradation_rate: f32,
    pub ribosome_degradation_rate: f32,
    pub rnap_degradation_rate: f32,
    pub replisome_degradation_rate: f32,
    pub membrane_degradation_rate: f32,
    pub ftsz_degradation_rate: f32,
    pub dnaa_degradation_rate: f32,
}

impl WholeCellComplexAssemblyState {
    pub fn total_complexes(self) -> f32 {
        self.atp_band_complexes
            + self.ribosome_complexes
            + self.rnap_complexes
            + self.replisome_complexes
            + self.membrane_complexes
            + self.ftsz_polymer
            + self.dnaa_activity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellNamedComplexState {
    pub id: String,
    pub operon: String,
    pub asset_class: WholeCellAssetClass,
    #[serde(default = "default_assembly_family")]
    pub family: WholeCellAssemblyFamily,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub subunit_pool: f32,
    #[serde(default)]
    pub nucleation_intermediate: f32,
    #[serde(default)]
    pub elongation_intermediate: f32,
    pub abundance: f32,
    pub target_abundance: f32,
    pub assembly_rate: f32,
    pub degradation_rate: f32,
    #[serde(default)]
    pub nucleation_rate: f32,
    #[serde(default)]
    pub elongation_rate: f32,
    #[serde(default)]
    pub maturation_rate: f32,
    pub component_satisfaction: f32,
    pub structural_support: f32,
    #[serde(default)]
    pub assembly_progress: f32,
    #[serde(default)]
    pub stalled_intermediate: f32,
    #[serde(default)]
    pub damaged_abundance: f32,
    #[serde(default)]
    pub limiting_component_signal: f32,
    #[serde(default)]
    pub shared_component_pressure: f32,
    #[serde(default)]
    pub insertion_progress: f32,
    #[serde(default)]
    pub failure_count: f32,
}

impl Default for WholeCellComplexAssemblyState {
    fn default() -> Self {
        Self {
            atp_band_complexes: 0.0,
            ribosome_complexes: 0.0,
            rnap_complexes: 0.0,
            replisome_complexes: 0.0,
            membrane_complexes: 0.0,
            ftsz_polymer: 0.0,
            dnaa_activity: 0.0,
            atp_band_target: 0.0,
            ribosome_target: 0.0,
            rnap_target: 0.0,
            replisome_target: 0.0,
            membrane_target: 0.0,
            ftsz_target: 0.0,
            dnaa_target: 0.0,
            atp_band_assembly_rate: 0.0,
            ribosome_assembly_rate: 0.0,
            rnap_assembly_rate: 0.0,
            replisome_assembly_rate: 0.0,
            membrane_assembly_rate: 0.0,
            ftsz_assembly_rate: 0.0,
            dnaa_assembly_rate: 0.0,
            atp_band_degradation_rate: 0.0,
            ribosome_degradation_rate: 0.0,
            rnap_degradation_rate: 0.0,
            replisome_degradation_rate: 0.0,
            membrane_degradation_rate: 0.0,
            ftsz_degradation_rate: 0.0,
            dnaa_degradation_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismExpressionState {
    pub global_activity: f32,
    pub energy_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub metabolic_burden_scale: f32,
    pub process_scales: WholeCellProcessWeights,
    pub amino_cost_scale: f32,
    pub nucleotide_cost_scale: f32,
    pub total_transcript_abundance: f32,
    pub total_protein_abundance: f32,
    #[serde(default)]
    pub transcription_units: Vec<WholeCellTranscriptionUnitState>,
}

impl Default for WholeCellOrganismExpressionState {
    fn default() -> Self {
        Self {
            global_activity: 1.0,
            energy_support: 1.0,
            translation_support: 1.0,
            nucleotide_support: 1.0,
            membrane_support: 1.0,
            crowding_penalty: 1.0,
            metabolic_burden_scale: 1.0,
            process_scales: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.0,
                translation: 1.0,
                replication: 1.0,
                segregation: 1.0,
                membrane: 1.0,
                constriction: 1.0,
            },
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
            total_transcript_abundance: 0.0,
            total_protein_abundance: 0.0,
            transcription_units: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellLocalChemistrySpec {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_au: f32,
    pub use_gpu: bool,
    #[serde(default)]
    pub enable_default_syn3a_subsystems: bool,
    #[serde(default)]
    pub scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellProgramSpec {
    #[serde(default, alias = "name")]
    pub program_name: Option<String>,
    #[serde(default)]
    pub contract: WholeCellContractSchema,
    #[serde(default)]
    pub provenance: WholeCellProvenance,
    #[serde(default)]
    pub organism_data_ref: Option<String>,
    #[serde(default)]
    pub organism_data: Option<WholeCellOrganismSpec>,
    #[serde(default)]
    pub organism_assets: Option<WholeCellGenomeAssetPackage>,
    #[serde(default)]
    pub organism_process_registry: Option<WholeCellGenomeProcessRegistry>,
    #[serde(default)]
    pub chromosome_state: Option<WholeCellChromosomeState>,
    #[serde(default)]
    pub membrane_division_state: Option<WholeCellMembraneDivisionState>,
    pub config: WholeCellConfig,
    pub initial_lattice: WholeCellInitialLatticeSpec,
    pub initial_state: WholeCellInitialStateSpec,
    #[serde(default)]
    pub quantum_profile: WholeCellQuantumProfile,
    #[serde(default)]
    pub local_chemistry: Option<WholeCellLocalChemistrySpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct WholeCellProgramDefaultsSpec {
    #[serde(default)]
    pub program_name: Option<String>,
    #[serde(default)]
    pub config: Option<WholeCellConfig>,
    #[serde(default)]
    pub initial_lattice: Option<WholeCellInitialLatticeSpec>,
    #[serde(default)]
    pub initial_state: Option<WholeCellInitialStateSpec>,
    #[serde(default)]
    pub quantum_profile: Option<WholeCellQuantumProfile>,
    #[serde(default)]
    pub local_chemistry: Option<WholeCellLocalChemistrySpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellOrganismBundleManifest {
    #[serde(default)]
    pub organism: Option<String>,
    #[serde(default)]
    pub source_dataset: Option<String>,
    #[serde(default)]
    pub organism_spec_json: Option<String>,
    #[serde(default)]
    pub require_structured_bundle: bool,
    #[serde(default)]
    pub require_explicit_organism_sources: bool,
    #[serde(default)]
    pub require_explicit_gene_semantics: bool,
    #[serde(default)]
    pub require_explicit_transcription_unit_semantics: bool,
    #[serde(default)]
    pub require_explicit_asset_entities: bool,
    #[serde(default)]
    pub require_explicit_asset_semantics: bool,
    #[serde(default)]
    pub require_explicit_program_defaults: bool,
    #[serde(default)]
    pub metadata_json: Option<String>,
    #[serde(default)]
    pub genome_fasta: Option<String>,
    #[serde(default)]
    pub gene_features_json: Option<String>,
    #[serde(default)]
    pub gene_features_gff: Option<String>,
    #[serde(default)]
    pub gene_products_json: Option<String>,
    #[serde(default)]
    pub gene_semantics_json: Option<String>,
    #[serde(default)]
    pub transcription_units_json: Option<String>,
    #[serde(default)]
    pub transcription_unit_semantics_json: Option<String>,
    #[serde(default)]
    pub chromosome_domains_json: Option<String>,
    #[serde(default)]
    pub pools_json: Option<String>,
    #[serde(default)]
    pub operons_json: Option<String>,
    #[serde(default)]
    pub rnas_json: Option<String>,
    #[serde(default)]
    pub proteins_json: Option<String>,
    #[serde(default)]
    pub complexes_json: Option<String>,
    #[serde(default)]
    pub operon_semantics_json: Option<String>,
    #[serde(default)]
    pub protein_semantics_json: Option<String>,
    #[serde(default)]
    pub complex_semantics_json: Option<String>,
    #[serde(default)]
    pub program_defaults_json: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismBundleMetadata {
    pub organism: String,
    #[serde(default)]
    pub chromosome_length_bp: Option<u32>,
    #[serde(default)]
    pub origin_bp: u32,
    #[serde(default)]
    pub terminus_bp: u32,
    pub geometry: WholeCellGeometryPrior,
    pub composition: WholeCellCompositionPrior,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGeneProductAnnotation {
    pub gene: String,
    #[serde(default)]
    pub essential: bool,
    #[serde(default = "default_expression_level")]
    pub basal_expression: f32,
    #[serde(default = "default_translation_cost")]
    pub translation_cost: f32,
    #[serde(default = "default_nucleotide_cost")]
    pub nucleotide_cost: f32,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGeneSemanticAnnotation {
    pub gene: String,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellTranscriptionUnitSemanticAnnotation {
    pub name: String,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    #[serde(default)]
    pub asset_class: Option<WholeCellAssetClass>,
    #[serde(default)]
    pub complex_family: Option<WholeCellAssemblyFamily>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellSolverStage {
    AtomisticRefinement,
    Rdme,
    Cme,
    Ode,
    ChromosomeBd,
    Geometry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellStageClockState {
    pub stage: WholeCellSolverStage,
    #[serde(default = "default_stage_interval_steps")]
    pub base_interval_steps: u64,
    #[serde(default = "default_stage_interval_steps")]
    pub dynamic_interval_steps: u64,
    #[serde(default)]
    pub next_due_step: u64,
    #[serde(default)]
    pub run_count: u64,
    #[serde(default)]
    pub last_run_step: Option<u64>,
    #[serde(default)]
    pub last_run_time_ms: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSchedulerState {
    #[serde(default)]
    pub stage_clocks: Vec<WholeCellStageClockState>,
}

impl Default for WholeCellSchedulerState {
    fn default() -> Self {
        Self {
            stage_clocks: Vec::new(),
        }
    }
}

fn default_stage_interval_steps() -> u64 {
    1
}

fn default_copy_number() -> f32 {
    1.0
}

fn default_accessibility() -> f32 {
    1.0
}

fn default_compaction_fraction() -> f32 {
    0.35
}

fn default_septum_radius_fraction() -> f32 {
    1.0
}

fn default_septum_thickness_nm() -> f32 {
    4.0
}

fn default_envelope_integrity() -> f32 {
    1.0
}

fn default_osmotic_balance() -> f32 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSavedCoreState {
    pub time_ms: f32,
    pub step_count: u64,
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
    pub radius_nm: f32,
    pub surface_area_nm2: f32,
    pub volume_nm3: f32,
    pub division_progress: f32,
    pub metabolic_load: f32,
    pub quantum_profile: WholeCellQuantumProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellLatticeState {
    pub atp: Vec<f32>,
    pub amino_acids: Vec<f32>,
    pub nucleotides: Vec<f32>,
    pub membrane_precursors: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct WholeCellSpatialFieldState {
    #[serde(default)]
    pub membrane_adjacency: Vec<f32>,
    #[serde(default)]
    pub septum_zone: Vec<f32>,
    #[serde(default)]
    pub nucleoid_occupancy: Vec<f32>,
    #[serde(default)]
    pub membrane_band_zone: Vec<f32>,
    #[serde(default)]
    pub pole_zone: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSavedState {
    pub program_name: Option<String>,
    #[serde(default)]
    pub contract: WholeCellContractSchema,
    #[serde(default)]
    pub provenance: WholeCellProvenance,
    #[serde(default)]
    pub organism_data_ref: Option<String>,
    #[serde(default)]
    pub organism_data: Option<WholeCellOrganismSpec>,
    #[serde(default)]
    pub organism_assets: Option<WholeCellGenomeAssetPackage>,
    #[serde(default)]
    pub organism_expression: WholeCellOrganismExpressionState,
    #[serde(default)]
    pub organism_process_registry: Option<WholeCellGenomeProcessRegistry>,
    #[serde(default)]
    pub chromosome_state: WholeCellChromosomeState,
    #[serde(default)]
    pub membrane_division_state: WholeCellMembraneDivisionState,
    #[serde(default)]
    pub organism_species: Vec<WholeCellSpeciesRuntimeState>,
    #[serde(default)]
    pub organism_reactions: Vec<WholeCellReactionRuntimeState>,
    #[serde(default)]
    pub complex_assembly: WholeCellComplexAssemblyState,
    #[serde(default)]
    pub named_complexes: Vec<WholeCellNamedComplexState>,
    #[serde(default)]
    pub scheduler_state: WholeCellSchedulerState,
    pub config: WholeCellConfig,
    pub core: WholeCellSavedCoreState,
    pub lattice: WholeCellLatticeState,
    #[serde(default)]
    pub spatial_fields: Option<WholeCellSpatialFieldState>,
    #[serde(default)]
    pub local_chemistry: Option<WholeCellLocalChemistrySpec>,
    #[serde(default)]
    pub chemistry_report: LocalChemistryReport,
    #[serde(default)]
    pub chemistry_site_reports: Vec<LocalChemistrySiteReport>,
    #[serde(default)]
    pub last_md_probe: Option<LocalMDProbeReport>,
    #[serde(default)]
    pub scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
    #[serde(default)]
    pub subsystem_states: Vec<WholeCellSubsystemState>,
    #[serde(default = "default_scale")]
    pub md_translation_scale: f32,
    #[serde(default = "default_scale")]
    pub md_membrane_scale: f32,
}

fn default_scale() -> f32 {
    1.0
}

fn default_contract_version() -> String {
    WHOLE_CELL_CONTRACT_VERSION.to_string()
}

fn default_program_schema_version() -> u32 {
    WHOLE_CELL_PROGRAM_SCHEMA_VERSION
}

fn default_saved_state_schema_version() -> u32 {
    WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION
}

fn default_runtime_manifest_schema_version() -> u32 {
    WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION
}

fn default_expression_level() -> f32 {
    1.0
}

fn default_translation_cost() -> f32 {
    1.0
}

fn default_nucleotide_cost() -> f32 {
    1.0
}

fn default_chromosome_radius_fraction() -> f32 {
    0.55
}

fn default_membrane_fraction() -> f32 {
    0.24
}

fn default_dry_mass_fg() -> f32 {
    130.0
}

fn default_gc_fraction() -> f32 {
    0.31
}

fn default_protein_fraction() -> f32 {
    0.56
}

fn default_rna_fraction() -> f32 {
    0.22
}

fn default_lipid_fraction() -> f32 {
    0.12
}

fn stable_checksum_hex(bytes: &[u8]) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn stable_json_checksum<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| stable_checksum_hex(&bytes))
        .map_err(|error| format!("failed to serialize whole-cell contract payload: {error}"))
}

fn populate_program_contract_metadata(spec: &mut WholeCellProgramSpec) -> Result<(), String> {
    spec.contract = spec.contract.clone().normalized_for_program();
    if spec.provenance.organism_asset_hash.is_none() {
        if let Some(assets) = spec.organism_assets.as_ref() {
            spec.provenance.organism_asset_hash = Some(stable_json_checksum(assets)?);
        }
    }
    if spec.provenance.compiled_ir_hash.is_none() {
        if let Some(registry) = spec.organism_process_registry.as_ref() {
            spec.provenance.compiled_ir_hash = Some(stable_json_checksum(registry)?);
        }
    }
    if spec.provenance.run_manifest_hash.is_none() {
        let manifest_view = (
            &spec.program_name,
            &spec.organism_data_ref,
            &spec.config,
            &spec.local_chemistry,
            &spec.contract,
        );
        spec.provenance.run_manifest_hash = Some(stable_json_checksum(&manifest_view)?);
    }
    Ok(())
}

fn populate_saved_state_contract_metadata(state: &mut WholeCellSavedState) -> Result<(), String> {
    state.contract = state.contract.clone().normalized_for_saved_state();
    if state.provenance.organism_asset_hash.is_none() {
        if let Some(assets) = state.organism_assets.as_ref() {
            state.provenance.organism_asset_hash = Some(stable_json_checksum(assets)?);
        }
    }
    if state.provenance.compiled_ir_hash.is_none() {
        if let Some(registry) = state.organism_process_registry.as_ref() {
            state.provenance.compiled_ir_hash = Some(stable_json_checksum(registry)?);
        }
    }
    if state.provenance.run_manifest_hash.is_none() {
        let manifest_view = (
            &state.program_name,
            &state.organism_data_ref,
            &state.config,
            &state.local_chemistry,
            &state.contract,
        );
        state.provenance.run_manifest_hash = Some(stable_json_checksum(&manifest_view)?);
    }
    Ok(())
}

fn normalized_share(raw: f32, total: f32) -> f32 {
    let total = total.max(1.0e-6);
    (raw.max(0.0) / total).clamp(0.0, 1.0)
}

fn derived_process_scale(
    share: f32,
    essential_fraction: f32,
    polycistronic_fraction: f32,
    bias: f32,
) -> f32 {
    (0.78 + 1.12 * share + 0.08 * essential_fraction + 0.06 * polycistronic_fraction + bias)
        .clamp(0.70, 1.45)
}

pub fn derive_organism_profile(spec: &WholeCellOrganismSpec) -> WholeCellOrganismProfile {
    let chromosome_length_bp = spec.chromosome_length_bp.max(1);
    let gene_count = spec.genes.len();
    let transcription_unit_count = spec.transcription_units.len();
    let pool_count = spec.pools.len();
    let essential_gene_fraction = if gene_count > 0 {
        spec.genes.iter().filter(|gene| gene.essential).count() as f32 / gene_count as f32
    } else {
        0.0
    };
    let polycistronic_fraction = if transcription_unit_count > 0 {
        spec.transcription_units
            .iter()
            .filter(|unit| unit.genes.len() > 1)
            .count() as f32
            / transcription_unit_count as f32
    } else {
        0.0
    };
    let total_gene_bp = spec
        .genes
        .iter()
        .map(|gene| {
            if gene.end_bp >= gene.start_bp {
                (gene.end_bp - gene.start_bp + 1) as f32
            } else {
                0.0
            }
        })
        .sum::<f32>();
    let mean_gene_length_bp = if gene_count > 0 {
        total_gene_bp / gene_count as f32
    } else {
        0.0
    };
    let coding_density = (total_gene_bp / chromosome_length_bp as f32).clamp(0.0, 1.0);

    let mut process_totals = WholeCellProcessWeights::default();
    for gene in &spec.genes {
        let scale = gene.basal_expression.max(0.0);
        process_totals.add_weighted(gene.process_weights, scale);
        process_totals.translation += gene.translation_cost.max(0.0) * 0.06 * scale;
        process_totals.replication += gene.nucleotide_cost.max(0.0) * 0.04 * scale;
    }
    for unit in &spec.transcription_units {
        process_totals.add_weighted(unit.process_weights, unit.basal_activity.max(0.0));
    }
    for pool in &spec.pools {
        let species = pool.species.to_ascii_lowercase();
        let scale = pool.concentration_mm.max(0.0) + 0.002 * pool.count.max(0.0);
        if species.contains("atp") || species.contains("oxygen") || species.contains("glucose") {
            process_totals.energy += 0.04 * scale;
        }
        if species.contains("rib") || species.contains("amino") {
            process_totals.translation += 0.03 * scale;
        }
        if species.contains("nucleotide") || species.contains("dna") {
            process_totals.replication += 0.03 * scale;
        }
        if species.contains("membrane") || species.contains("lipid") || species.contains("phosph") {
            process_totals.membrane += 0.03 * scale;
        }
    }

    let process_total = process_totals.total();
    let composition = spec.composition;
    let geometry = spec.geometry;
    let energy_bias = 0.08 * geometry.membrane_fraction + 0.05 * composition.lipid_fraction;
    let transcription_bias = 0.10 * composition.rna_fraction + 0.04 * polycistronic_fraction;
    let translation_bias = 0.08 * composition.protein_fraction + 0.04 * polycistronic_fraction;
    let replication_bias = 0.05 * (1.0 - composition.gc_fraction.clamp(0.0, 1.0));
    let segregation_bias = 0.05 * geometry.chromosome_radius_fraction.clamp(0.0, 1.0);
    let membrane_bias = 0.12 * geometry.membrane_fraction + 0.06 * composition.lipid_fraction;
    let constriction_bias = 0.10 * geometry.membrane_fraction + 0.05 * essential_gene_fraction;

    let process_scales = WholeCellProcessWeights {
        energy: derived_process_scale(
            normalized_share(process_totals.energy, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            energy_bias,
        ),
        transcription: derived_process_scale(
            normalized_share(process_totals.transcription, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            transcription_bias,
        ),
        translation: derived_process_scale(
            normalized_share(process_totals.translation, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            translation_bias,
        ),
        replication: derived_process_scale(
            normalized_share(process_totals.replication, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            replication_bias,
        ),
        segregation: derived_process_scale(
            normalized_share(process_totals.segregation, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            segregation_bias,
        ),
        membrane: derived_process_scale(
            normalized_share(process_totals.membrane, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            membrane_bias,
        ),
        constriction: derived_process_scale(
            normalized_share(process_totals.constriction, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            constriction_bias,
        ),
    };

    let metabolic_burden_scale = (0.84
        + 0.16 * coding_density
        + 0.10 * essential_gene_fraction
        + 0.08 * composition.protein_fraction
        + 0.04 * polycistronic_fraction)
        .clamp(0.85, 1.55);
    let crowding_scale = (0.88
        + 0.20 * coding_density
        + 0.12 * composition.protein_fraction
        + 0.05 * geometry.chromosome_radius_fraction)
        .clamp(0.85, 1.25);

    WholeCellOrganismProfile {
        organism: spec.organism.clone(),
        chromosome_length_bp,
        chromosome_domain_count: spec.chromosome_domains.len(),
        gene_count,
        transcription_unit_count,
        pool_count,
        essential_gene_fraction,
        polycistronic_fraction,
        coding_density,
        mean_gene_length_bp,
        process_scales,
        metabolic_burden_scale,
        crowding_scale,
        preferred_radius_nm: geometry.radius_nm.max(50.0),
        chromosome_radius_fraction: geometry.chromosome_radius_fraction.clamp(0.1, 0.95),
        membrane_fraction: geometry.membrane_fraction.clamp(0.05, 0.95),
    }
}

fn gene_length_bp(feature: &WholeCellGenomeFeature) -> u32 {
    if feature.end_bp >= feature.start_bp {
        feature.end_bp - feature.start_bp + 1
    } else {
        0
    }
}

fn gene_midpoint_bp(feature: &WholeCellGenomeFeature) -> u32 {
    ((feature.start_bp as u64 + feature.end_bp as u64) / 2) as u32
}

fn midpoint_bp(start_bp: u32, end_bp: u32) -> u32 {
    ((start_bp as u64 + end_bp as u64) / 2) as u32
}

fn normalized_chromosome_interval(start_bp: u32, end_bp: u32, genome_bp: u32) -> (u32, u32) {
    let genome_bp = genome_bp.max(1);
    let start_bp = start_bp.min(genome_bp.saturating_sub(1));
    let end_bp = end_bp.min(genome_bp.saturating_sub(1));
    if start_bp <= end_bp {
        (start_bp, end_bp)
    } else {
        (end_bp, start_bp)
    }
}

fn interval_contains_bp(start_bp: u32, end_bp: u32, position_bp: u32) -> bool {
    let (start_bp, end_bp) = if start_bp <= end_bp {
        (start_bp, end_bp)
    } else {
        (end_bp, start_bp)
    };
    position_bp >= start_bp && position_bp <= end_bp
}

fn chromosome_domain_id_for_position(
    domains: &[WholeCellChromosomeDomainSpec],
    position_bp: u32,
) -> Option<String> {
    domains
        .iter()
        .find(|domain| interval_contains_bp(domain.start_bp, domain.end_bp, position_bp))
        .map(|domain| domain.id.clone())
        .or_else(|| {
            domains
                .iter()
                .min_by_key(|domain| {
                    let center_bp = midpoint_bp(domain.start_bp, domain.end_bp);
                    center_bp.abs_diff(position_bp)
                })
                .map(|domain| domain.id.clone())
        })
}

fn inferred_asset_class(
    weights: WholeCellProcessWeights,
    subsystem_targets: &[Syn3ASubsystemPreset],
    name: &str,
) -> WholeCellAssetClass {
    if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand) {
        return WholeCellAssetClass::Energy;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::RibosomePolysomeCluster) {
        return WholeCellAssetClass::Translation;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack) {
        return WholeCellAssetClass::Replication;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing) {
        return WholeCellAssetClass::Constriction;
    }

    let name = name.to_ascii_lowercase();
    if name.contains("chaperone") || name.contains("quality_control") {
        return WholeCellAssetClass::QualityControl;
    }
    if name.contains("transport") || name.contains("homeostasis") {
        return WholeCellAssetClass::Homeostasis;
    }

    let weights = weights.clamped();
    let ranked = [
        (weights.energy, WholeCellAssetClass::Energy),
        (weights.translation, WholeCellAssetClass::Translation),
        (weights.replication, WholeCellAssetClass::Replication),
        (weights.segregation, WholeCellAssetClass::Segregation),
        (weights.membrane, WholeCellAssetClass::Membrane),
        (weights.constriction, WholeCellAssetClass::Constriction),
        (weights.transcription, WholeCellAssetClass::Homeostasis),
    ];
    ranked
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, asset_class)| asset_class)
        .unwrap_or(WholeCellAssetClass::Generic)
}

fn inferred_complex_family(
    asset_class: WholeCellAssetClass,
    subsystem_targets: &[Syn3ASubsystemPreset],
    operon_name: &str,
) -> WholeCellAssemblyFamily {
    let lowered = operon_name.to_ascii_lowercase();
    if subsystem_targets.contains(&Syn3ASubsystemPreset::RibosomePolysomeCluster)
        || lowered.contains("ribosome")
    {
        return WholeCellAssemblyFamily::Ribosome;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack)
        || lowered.contains("replisome")
        || lowered.contains("replication")
        || lowered.contains("dna")
    {
        return WholeCellAssemblyFamily::Replisome;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        || lowered.contains("atp_synthase")
        || lowered.contains("respir")
    {
        return WholeCellAssemblyFamily::AtpSynthase;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
        || lowered.contains("ftsz")
        || lowered.contains("division")
        || lowered.contains("sept")
        || lowered.contains("divisome")
    {
        return WholeCellAssemblyFamily::Divisome;
    }
    if lowered.contains("rnap") || lowered.contains("rna_polymerase") || lowered.contains("sigma") {
        return WholeCellAssemblyFamily::RnaPolymerase;
    }
    if lowered.contains("chaperone") || lowered.contains("fold") || lowered.contains("client") {
        return WholeCellAssemblyFamily::ChaperoneClient;
    }
    if lowered.contains("transport") || lowered.contains("porin") || lowered.contains("pump") {
        return WholeCellAssemblyFamily::Transporter;
    }
    if matches!(
        asset_class,
        WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction
    ) {
        return WholeCellAssemblyFamily::MembraneEnzyme;
    }
    WholeCellAssemblyFamily::Generic
}

fn canonical_species_fragment(name: &str) -> String {
    let mut fragment = String::with_capacity(name.len());
    let mut last_was_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            fragment.push(ch.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore {
            fragment.push('_');
            last_was_underscore = true;
        }
    }
    fragment.trim_matches('_').to_string()
}

fn infer_pool_bulk_field(species_name: &str) -> Option<WholeCellBulkField> {
    let lowered = species_name.trim().to_ascii_lowercase();
    if lowered == "atp" {
        Some(WholeCellBulkField::ATP)
    } else if lowered == "adp" {
        Some(WholeCellBulkField::ADP)
    } else if lowered.contains("glucose") {
        Some(WholeCellBulkField::Glucose)
    } else if lowered.contains("oxygen") {
        Some(WholeCellBulkField::Oxygen)
    } else if lowered.contains("amino") {
        Some(WholeCellBulkField::AminoAcids)
    } else if lowered.contains("nucleotide") {
        Some(WholeCellBulkField::Nucleotides)
    } else if lowered.contains("membrane") || lowered.contains("lipid") {
        Some(WholeCellBulkField::MembranePrecursors)
    } else {
        None
    }
}

fn infer_pool_role(species_name: &str) -> Option<WholeCellPoolRole> {
    let lowered = species_name.trim().to_ascii_lowercase();
    if lowered.contains("ribosome") {
        Some(WholeCellPoolRole::ActiveRibosomes)
    } else if lowered.contains("rnap") || lowered.contains("rna_polymerase") {
        Some(WholeCellPoolRole::ActiveRnap)
    } else if lowered.contains("dnaa") {
        Some(WholeCellPoolRole::Dnaa)
    } else if lowered.contains("ftsz") {
        Some(WholeCellPoolRole::Ftsz)
    } else {
        None
    }
}

fn normalize_pool_metadata(pools: &mut [WholeCellMoleculePoolSpec]) {
    for pool in pools {
        if pool.bulk_field.is_none() {
            pool.bulk_field = infer_pool_bulk_field(&pool.species);
        }
        if pool.role.is_none() {
            pool.role = infer_pool_role(&pool.species);
        }
    }
}

fn with_normalized_pool_metadata(mut spec: WholeCellOrganismSpec) -> WholeCellOrganismSpec {
    normalize_pool_metadata(&mut spec.pools);
    spec
}

fn normalize_gene_semantic_metadata(genes: &mut [WholeCellGenomeFeature]) {
    for gene in genes {
        let asset_class = gene.asset_class.unwrap_or_else(|| {
            inferred_asset_class(gene.process_weights, &gene.subsystem_targets, &gene.gene)
        });
        gene.asset_class = Some(asset_class);
        if gene.complex_family.is_none() {
            gene.complex_family = Some(inferred_complex_family(
                asset_class,
                &gene.subsystem_targets,
                &gene.gene,
            ));
        }
    }
}

fn normalize_transcription_unit_semantic_metadata(spec: &mut WholeCellOrganismSpec) {
    for unit in &mut spec.transcription_units {
        let mut subsystem_targets = unit.subsystem_targets.clone();
        for gene_name in &unit.genes {
            if let Some(gene) = spec.genes.iter().find(|gene| gene.gene == *gene_name) {
                push_unique_subsystem_targets(&mut subsystem_targets, &gene.subsystem_targets);
            }
        }
        unit.subsystem_targets = subsystem_targets;
        let asset_class = unit.asset_class.unwrap_or_else(|| {
            inferred_asset_class(unit.process_weights, &unit.subsystem_targets, &unit.name)
        });
        unit.asset_class = Some(asset_class);
        if unit.complex_family.is_none() {
            unit.complex_family = Some(inferred_complex_family(
                asset_class,
                &unit.subsystem_targets,
                &unit.name,
            ));
        }
    }
}

fn with_normalized_semantic_metadata(mut spec: WholeCellOrganismSpec) -> WholeCellOrganismSpec {
    normalize_gene_semantic_metadata(&mut spec.genes);
    normalize_transcription_unit_semantic_metadata(&mut spec);
    spec
}

fn with_normalized_asset_pool_metadata(
    mut package: WholeCellGenomeAssetPackage,
) -> WholeCellGenomeAssetPackage {
    normalize_pool_metadata(&mut package.pools);
    package
}

fn normalize_runtime_species_bulk_fields_from_registry(
    species: &mut [WholeCellSpeciesRuntimeState],
    registry: Option<&WholeCellGenomeProcessRegistry>,
) {
    let registry_bulk_fields = registry
        .map(|registry| {
            registry
                .species
                .iter()
                .map(|species| (species.id.clone(), species.bulk_field))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    for runtime_species in species {
        if runtime_species.bulk_field.is_some()
            || runtime_species.species_class != WholeCellSpeciesClass::Pool
        {
            continue;
        }
        if let Some(bulk_field) = registry_bulk_fields
            .get(&runtime_species.id)
            .copied()
            .flatten()
        {
            runtime_species.bulk_field = Some(bulk_field);
        }
    }
}

fn backfill_legacy_runtime_species_bulk_fields(species: &mut [WholeCellSpeciesRuntimeState]) {
    for runtime_species in species {
        if runtime_species.bulk_field.is_some()
            || runtime_species.species_class != WholeCellSpeciesClass::Pool
        {
            continue;
        }
        if let Some(bulk_field) = infer_pool_bulk_field(&runtime_species.id)
            .or_else(|| infer_pool_bulk_field(&runtime_species.name))
        {
            runtime_species.bulk_field = Some(bulk_field);
        }
    }
}

fn compile_operon_semantic_specs(
    operons: &[WholeCellOperonSpec],
) -> Vec<WholeCellOperonSemanticSpec> {
    let mut semantics: Vec<WholeCellOperonSemanticSpec> = operons
        .iter()
        .filter_map(|operon| {
            Some(WholeCellOperonSemanticSpec {
                name: operon.name.clone(),
                asset_class: operon.asset_class?,
                complex_family: operon.complex_family?,
                subsystem_targets: operon.subsystem_targets.clone(),
            })
        })
        .collect();
    semantics.sort_by(|left, right| left.name.cmp(&right.name));
    semantics
}

fn compile_complex_semantic_specs(
    complexes: &[WholeCellComplexSpec],
) -> Vec<WholeCellComplexSemanticSpec> {
    let mut semantics: Vec<WholeCellComplexSemanticSpec> = complexes
        .iter()
        .map(|complex| WholeCellComplexSemanticSpec {
            id: complex.id.clone(),
            asset_class: complex.asset_class,
            family: complex.family,
            subsystem_targets: complex.subsystem_targets.clone(),
            membrane_inserted: complex.membrane_inserted,
            chromosome_coupled: complex.chromosome_coupled,
            division_coupled: complex.division_coupled,
        })
        .collect();
    semantics.sort_by(|left, right| left.id.cmp(&right.id));
    semantics
}

fn compile_protein_semantic_specs(
    proteins: &[WholeCellProteinProductSpec],
) -> Vec<WholeCellProteinSemanticSpec> {
    let mut semantics: Vec<WholeCellProteinSemanticSpec> = proteins
        .iter()
        .map(|protein| WholeCellProteinSemanticSpec {
            id: protein.id.clone(),
            asset_class: protein.asset_class,
            subsystem_targets: protein.subsystem_targets.clone(),
        })
        .collect();
    semantics.sort_by(|left, right| left.id.cmp(&right.id));
    semantics
}

fn normalize_asset_package_semantic_metadata(package: &mut WholeCellGenomeAssetPackage) {
    let mut operon_targets = HashMap::<String, Vec<Syn3ASubsystemPreset>>::new();
    for protein in &package.proteins {
        let entry = operon_targets.entry(protein.operon.clone()).or_default();
        push_unique_subsystem_targets(entry, &protein.subsystem_targets);
    }

    for operon in &mut package.operons {
        let mut subsystem_targets = operon.subsystem_targets.clone();
        if let Some(extra_targets) = operon_targets.get(&operon.name) {
            push_unique_subsystem_targets(&mut subsystem_targets, extra_targets);
        }
        operon.subsystem_targets = subsystem_targets;
        let asset_class = operon.asset_class.unwrap_or_else(|| {
            inferred_asset_class(
                operon.process_weights,
                &operon.subsystem_targets,
                &operon.name,
            )
        });
        operon.asset_class = Some(asset_class);
        if operon.complex_family.is_none() {
            operon.complex_family = Some(inferred_complex_family(
                asset_class,
                &operon.subsystem_targets,
                &operon.name,
            ));
        }
    }

    let mut operon_semantics_by_name: HashMap<String, WholeCellOperonSemanticSpec> = package
        .operon_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.name.clone(), semantic))
        .collect();
    for operon in &package.operons {
        let Some(asset_class) = operon.asset_class else {
            continue;
        };
        let Some(complex_family) = operon.complex_family else {
            continue;
        };
        let semantic = operon_semantics_by_name
            .entry(operon.name.clone())
            .or_insert_with(|| WholeCellOperonSemanticSpec {
                name: operon.name.clone(),
                asset_class,
                complex_family,
                subsystem_targets: Vec::new(),
            });
        if matches!(semantic.asset_class, WholeCellAssetClass::Generic) {
            semantic.asset_class = asset_class;
        }
        if matches!(semantic.complex_family, WholeCellAssemblyFamily::Generic) {
            semantic.complex_family = complex_family;
        }
        push_unique_subsystem_targets(&mut semantic.subsystem_targets, &operon.subsystem_targets);
    }
    let mut operon_semantics = operon_semantics_by_name
        .into_values()
        .collect::<Vec<WholeCellOperonSemanticSpec>>();
    operon_semantics.sort_by(|left, right| left.name.cmp(&right.name));
    package.operon_semantics = operon_semantics;

    let operon_semantic_map: HashMap<
        String,
        (
            Vec<Syn3ASubsystemPreset>,
            WholeCellAssetClass,
            WholeCellAssemblyFamily,
        ),
    > = package
        .operon_semantics
        .iter()
        .map(|semantic| {
            (
                semantic.name.clone(),
                (
                    semantic.subsystem_targets.clone(),
                    semantic.asset_class,
                    semantic.complex_family,
                ),
            )
        })
        .collect();

    for operon in &mut package.operons {
        if let Some((subsystem_targets, asset_class, family)) =
            operon_semantic_map.get(&operon.name)
        {
            if operon.subsystem_targets.is_empty() {
                operon.subsystem_targets = subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(&mut operon.subsystem_targets, subsystem_targets);
            }
            operon.asset_class = Some(*asset_class);
            operon.complex_family = Some(*family);
        }
    }

    let operon_semantics: HashMap<
        String,
        (
            Vec<Syn3ASubsystemPreset>,
            WholeCellAssetClass,
            WholeCellAssemblyFamily,
        ),
    > = package
        .operon_semantics
        .iter()
        .map(|semantic| {
            (
                semantic.name.clone(),
                (
                    semantic.subsystem_targets.clone(),
                    semantic.asset_class,
                    semantic.complex_family,
                ),
            )
        })
        .collect();

    for protein in &mut package.proteins {
        if let Some((subsystem_targets, asset_class, _)) = operon_semantics.get(&protein.operon) {
            if protein.subsystem_targets.is_empty() {
                protein.subsystem_targets = subsystem_targets.clone();
            }
            if matches!(protein.asset_class, WholeCellAssetClass::Generic) {
                protein.asset_class = *asset_class;
            }
        }
    }

    for complex in &mut package.complexes {
        if let Some((subsystem_targets, asset_class, family)) =
            operon_semantics.get(&complex.operon)
        {
            if complex.subsystem_targets.is_empty() {
                complex.subsystem_targets = subsystem_targets.clone();
            }
            if matches!(complex.asset_class, WholeCellAssetClass::Generic) {
                complex.asset_class = *asset_class;
            }
            if matches!(complex.family, WholeCellAssemblyFamily::Generic) {
                complex.family = *family;
            }
            if !complex.membrane_inserted {
                complex.membrane_inserted = matches!(
                    complex.family,
                    WholeCellAssemblyFamily::AtpSynthase
                        | WholeCellAssemblyFamily::Transporter
                        | WholeCellAssemblyFamily::MembraneEnzyme
                        | WholeCellAssemblyFamily::Divisome
                );
            }
            if !complex.chromosome_coupled {
                complex.chromosome_coupled = matches!(
                    complex.family,
                    WholeCellAssemblyFamily::Replisome | WholeCellAssemblyFamily::RnaPolymerase
                );
            }
            if !complex.division_coupled {
                complex.division_coupled =
                    matches!(complex.family, WholeCellAssemblyFamily::Divisome);
            }
        }
    }

    let mut protein_semantics_by_id: HashMap<String, WholeCellProteinSemanticSpec> = package
        .protein_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.id.clone(), semantic))
        .collect();
    for protein in &package.proteins {
        let semantic = protein_semantics_by_id
            .entry(protein.id.clone())
            .or_insert_with(|| WholeCellProteinSemanticSpec {
                id: protein.id.clone(),
                asset_class: protein.asset_class,
                subsystem_targets: Vec::new(),
            });
        if matches!(semantic.asset_class, WholeCellAssetClass::Generic) {
            semantic.asset_class = protein.asset_class;
        }
        push_unique_subsystem_targets(&mut semantic.subsystem_targets, &protein.subsystem_targets);
    }
    let mut protein_semantics = protein_semantics_by_id
        .into_values()
        .collect::<Vec<WholeCellProteinSemanticSpec>>();
    protein_semantics.sort_by(|left, right| left.id.cmp(&right.id));
    package.protein_semantics = protein_semantics;

    let protein_semantic_map: HashMap<String, (Vec<Syn3ASubsystemPreset>, WholeCellAssetClass)> =
        package
            .protein_semantics
            .iter()
            .map(|semantic| {
                (
                    semantic.id.clone(),
                    (semantic.subsystem_targets.clone(), semantic.asset_class),
                )
            })
            .collect();

    for protein in &mut package.proteins {
        if let Some((subsystem_targets, asset_class)) = protein_semantic_map.get(&protein.id) {
            if protein.subsystem_targets.is_empty() {
                protein.subsystem_targets = subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(&mut protein.subsystem_targets, subsystem_targets);
            }
            protein.asset_class = *asset_class;
        }
    }

    let mut complex_semantics_by_id: HashMap<String, WholeCellComplexSemanticSpec> = package
        .complex_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.id.clone(), semantic))
        .collect();
    for complex in &package.complexes {
        let semantic = complex_semantics_by_id
            .entry(complex.id.clone())
            .or_insert_with(|| WholeCellComplexSemanticSpec {
                id: complex.id.clone(),
                asset_class: complex.asset_class,
                family: complex.family,
                subsystem_targets: Vec::new(),
                membrane_inserted: complex.membrane_inserted,
                chromosome_coupled: complex.chromosome_coupled,
                division_coupled: complex.division_coupled,
            });
        if matches!(semantic.asset_class, WholeCellAssetClass::Generic) {
            semantic.asset_class = complex.asset_class;
        }
        if matches!(semantic.family, WholeCellAssemblyFamily::Generic) {
            semantic.family = complex.family;
        }
        push_unique_subsystem_targets(&mut semantic.subsystem_targets, &complex.subsystem_targets);
    }
    let mut complex_semantics = complex_semantics_by_id
        .into_values()
        .collect::<Vec<WholeCellComplexSemanticSpec>>();
    complex_semantics.sort_by(|left, right| left.id.cmp(&right.id));
    package.complex_semantics = complex_semantics;

    let complex_semantic_map: HashMap<
        String,
        (
            Vec<Syn3ASubsystemPreset>,
            WholeCellAssetClass,
            WholeCellAssemblyFamily,
            bool,
            bool,
            bool,
        ),
    > = package
        .complex_semantics
        .iter()
        .map(|semantic| {
            (
                semantic.id.clone(),
                (
                    semantic.subsystem_targets.clone(),
                    semantic.asset_class,
                    semantic.family,
                    semantic.membrane_inserted,
                    semantic.chromosome_coupled,
                    semantic.division_coupled,
                ),
            )
        })
        .collect();

    for complex in &mut package.complexes {
        if let Some((
            subsystem_targets,
            asset_class,
            family,
            membrane_inserted,
            chromosome_coupled,
            division_coupled,
        )) = complex_semantic_map.get(&complex.id)
        {
            if complex.subsystem_targets.is_empty() {
                complex.subsystem_targets = subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(&mut complex.subsystem_targets, subsystem_targets);
            }
            complex.asset_class = *asset_class;
            complex.family = *family;
            complex.membrane_inserted = *membrane_inserted;
            complex.chromosome_coupled = *chromosome_coupled;
            complex.division_coupled = *division_coupled;
        }
    }
}

fn with_normalized_asset_semantic_metadata(
    mut package: WholeCellGenomeAssetPackage,
) -> WholeCellGenomeAssetPackage {
    normalize_asset_package_semantic_metadata(&mut package);
    package
}

fn pool_bulk_field(pool: &WholeCellMoleculePoolSpec) -> Option<WholeCellBulkField> {
    pool.bulk_field
}

fn transport_asset_class_for_bulk_field(field: WholeCellBulkField) -> WholeCellAssetClass {
    match field {
        WholeCellBulkField::ATP
        | WholeCellBulkField::ADP
        | WholeCellBulkField::Glucose
        | WholeCellBulkField::Oxygen => WholeCellAssetClass::Energy,
        WholeCellBulkField::AminoAcids => WholeCellAssetClass::Translation,
        WholeCellBulkField::Nucleotides => WholeCellAssetClass::Replication,
        WholeCellBulkField::MembranePrecursors => WholeCellAssetClass::Membrane,
    }
}

fn bulk_field_supports_transport(field: WholeCellBulkField) -> bool {
    matches!(
        field,
        WholeCellBulkField::Glucose
            | WholeCellBulkField::Oxygen
            | WholeCellBulkField::AminoAcids
            | WholeCellBulkField::Nucleotides
            | WholeCellBulkField::MembranePrecursors
    )
}

fn bulk_field_transport_rate(field: WholeCellBulkField) -> f32 {
    match field {
        WholeCellBulkField::Glucose => 0.12,
        WholeCellBulkField::Oxygen => 0.16,
        WholeCellBulkField::AminoAcids => 0.08,
        WholeCellBulkField::Nucleotides => 0.07,
        WholeCellBulkField::MembranePrecursors => 0.05,
        WholeCellBulkField::ATP | WholeCellBulkField::ADP => 0.0,
    }
}

fn pool_species_id(species: &str) -> String {
    format!("pool_{}", canonical_species_fragment(species))
}

fn localized_pool_supports_bulk_field(field: WholeCellBulkField) -> bool {
    matches!(
        field,
        WholeCellBulkField::ATP
            | WholeCellBulkField::ADP
            | WholeCellBulkField::AminoAcids
            | WholeCellBulkField::Nucleotides
            | WholeCellBulkField::MembranePrecursors
    )
}

fn localized_pool_fields_for_asset_class(
    asset_class: WholeCellAssetClass,
) -> &'static [(WholeCellBulkField, f32)] {
    match asset_class {
        WholeCellAssetClass::Energy => &[(WholeCellBulkField::ATP, 1.0)],
        WholeCellAssetClass::Translation => &[
            (WholeCellBulkField::ATP, 0.45),
            (WholeCellBulkField::AminoAcids, 1.0),
        ],
        WholeCellAssetClass::Replication => &[
            (WholeCellBulkField::ATP, 0.50),
            (WholeCellBulkField::Nucleotides, 1.0),
        ],
        WholeCellAssetClass::Segregation => &[(WholeCellBulkField::ATP, 0.70)],
        WholeCellAssetClass::Membrane => &[
            (WholeCellBulkField::ATP, 0.78),
            (WholeCellBulkField::MembranePrecursors, 1.0),
        ],
        WholeCellAssetClass::Constriction => &[
            (WholeCellBulkField::ATP, 0.82),
            (WholeCellBulkField::MembranePrecursors, 1.0),
        ],
        WholeCellAssetClass::QualityControl => &[
            (WholeCellBulkField::ATP, 0.72),
            (WholeCellBulkField::AminoAcids, 0.28),
        ],
        WholeCellAssetClass::Homeostasis => &[(WholeCellBulkField::ATP, 0.42)],
        WholeCellAssetClass::Generic => &[(WholeCellBulkField::ATP, 0.36)],
    }
}

fn localized_pool_request_weight(
    field: WholeCellBulkField,
    asset_class: WholeCellAssetClass,
) -> Option<f32> {
    localized_pool_fields_for_asset_class(asset_class)
        .iter()
        .copied()
        .find(|(candidate_field, _)| *candidate_field == field)
        .map(|(_, weight)| weight)
        .or_else(|| match field {
            WholeCellBulkField::ADP => localized_pool_fields_for_asset_class(asset_class)
                .iter()
                .copied()
                .find(|(candidate_field, _)| *candidate_field == WholeCellBulkField::ATP)
                .map(|(_, atp_weight)| (0.82 * atp_weight).clamp(0.18, 1.0))
                .or(Some(0.30)),
            _ => None,
        })
}

fn localized_pool_transfer_rate(field: WholeCellBulkField) -> f32 {
    match field {
        WholeCellBulkField::ATP => 0.085,
        WholeCellBulkField::ADP => 0.080,
        WholeCellBulkField::AminoAcids => 0.072,
        WholeCellBulkField::Nucleotides => 0.066,
        WholeCellBulkField::MembranePrecursors => 0.058,
        WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => 0.0,
    }
}

fn localized_pool_turnover_rate(field: WholeCellBulkField) -> f32 {
    match field {
        WholeCellBulkField::ATP => 0.030,
        WholeCellBulkField::ADP => 0.028,
        WholeCellBulkField::AminoAcids => 0.025,
        WholeCellBulkField::Nucleotides => 0.022,
        WholeCellBulkField::MembranePrecursors => 0.020,
        WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => 0.0,
    }
}

fn localized_pool_basal_scale(field: WholeCellBulkField) -> f32 {
    match field {
        WholeCellBulkField::ATP => 0.11,
        WholeCellBulkField::ADP => 0.10,
        WholeCellBulkField::AminoAcids => 0.09,
        WholeCellBulkField::Nucleotides => 0.08,
        WholeCellBulkField::MembranePrecursors => 0.07,
        WholeCellBulkField::Glucose | WholeCellBulkField::Oxygen => 0.0,
    }
}

fn bulk_field_fragment(field: WholeCellBulkField) -> &'static str {
    match field {
        WholeCellBulkField::ATP => "atp",
        WholeCellBulkField::ADP => "adp",
        WholeCellBulkField::Glucose => "glucose",
        WholeCellBulkField::Oxygen => "oxygen",
        WholeCellBulkField::AminoAcids => "amino_acids",
        WholeCellBulkField::Nucleotides => "nucleotides",
        WholeCellBulkField::MembranePrecursors => "membrane_precursors",
    }
}

fn bulk_field_display_name(field: WholeCellBulkField) -> &'static str {
    match field {
        WholeCellBulkField::ATP => "ATP",
        WholeCellBulkField::ADP => "ADP",
        WholeCellBulkField::Glucose => "glucose",
        WholeCellBulkField::Oxygen => "oxygen",
        WholeCellBulkField::AminoAcids => "amino acids",
        WholeCellBulkField::Nucleotides => "nucleotides",
        WholeCellBulkField::MembranePrecursors => "membrane precursors",
    }
}

fn spatial_scope_fragment(scope: WholeCellSpatialScope) -> &'static str {
    match scope {
        WholeCellSpatialScope::WellMixed => "well_mixed",
        WholeCellSpatialScope::MembraneAdjacent => "membrane_adjacent",
        WholeCellSpatialScope::SeptumLocal => "septum_local",
        WholeCellSpatialScope::NucleoidLocal => "nucleoid_local",
    }
}

fn spatial_scope_display_name(scope: WholeCellSpatialScope) -> &'static str {
    match scope {
        WholeCellSpatialScope::WellMixed => "well-mixed",
        WholeCellSpatialScope::MembraneAdjacent => "membrane-adjacent",
        WholeCellSpatialScope::SeptumLocal => "septum-local",
        WholeCellSpatialScope::NucleoidLocal => "nucleoid-local",
    }
}

fn patch_domain_fragment(domain: WholeCellPatchDomain) -> &'static str {
    match domain {
        WholeCellPatchDomain::Distributed => "distributed",
        WholeCellPatchDomain::MembraneBand => "membrane_band",
        WholeCellPatchDomain::SeptumPatch => "septum_patch",
        WholeCellPatchDomain::PolarPatch => "polar_patch",
        WholeCellPatchDomain::NucleoidTrack => "nucleoid_track",
    }
}

fn patch_domain_display_name(domain: WholeCellPatchDomain) -> &'static str {
    match domain {
        WholeCellPatchDomain::Distributed => "distributed",
        WholeCellPatchDomain::MembraneBand => "membrane band",
        WholeCellPatchDomain::SeptumPatch => "septum patch",
        WholeCellPatchDomain::PolarPatch => "polar patch",
        WholeCellPatchDomain::NucleoidTrack => "nucleoid track",
    }
}

fn localized_pool_locality_fragment(
    spatial_scope: WholeCellSpatialScope,
    patch_domain: WholeCellPatchDomain,
) -> &'static str {
    if patch_domain != WholeCellPatchDomain::Distributed {
        patch_domain_fragment(patch_domain)
    } else {
        spatial_scope_fragment(spatial_scope)
    }
}

fn localized_pool_locality_display_name(
    spatial_scope: WholeCellSpatialScope,
    patch_domain: WholeCellPatchDomain,
) -> &'static str {
    if patch_domain != WholeCellPatchDomain::Distributed {
        patch_domain_display_name(patch_domain)
    } else {
        spatial_scope_display_name(spatial_scope)
    }
}

fn normalized_chromosome_domain_id(chromosome_domain: Option<&str>) -> Option<String> {
    chromosome_domain
        .map(str::trim)
        .filter(|domain_id| !domain_id.is_empty())
        .map(str::to_string)
}

fn localized_pool_domain_fragment(chromosome_domain: Option<&str>) -> Option<String> {
    normalized_chromosome_domain_id(chromosome_domain).map(|domain_id| {
        let fragment = canonical_species_fragment(&domain_id);
        if fragment.is_empty() {
            "chromosome_domain".to_string()
        } else {
            fragment
        }
    })
}

fn localized_pool_species_id(
    field: WholeCellBulkField,
    spatial_scope: WholeCellSpatialScope,
    patch_domain: WholeCellPatchDomain,
    chromosome_domain: Option<&str>,
) -> String {
    let base = format!(
        "pool_{}_{}",
        localized_pool_locality_fragment(spatial_scope, patch_domain),
        bulk_field_fragment(field)
    );
    if let Some(domain_fragment) = localized_pool_domain_fragment(chromosome_domain) {
        format!("{base}_{domain_fragment}")
    } else {
        base
    }
}

#[derive(Debug, Clone, Default)]
struct LocalizedPoolRequest {
    signal: f32,
    subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

fn register_localized_pool_request(
    requests: &mut HashMap<
        (
            WholeCellBulkField,
            WholeCellSpatialScope,
            WholeCellPatchDomain,
            Option<String>,
        ),
        LocalizedPoolRequest,
    >,
    field: WholeCellBulkField,
    spatial_scope: WholeCellSpatialScope,
    patch_domain: WholeCellPatchDomain,
    chromosome_domain: Option<&str>,
    signal_seed: f32,
    weight: f32,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> String {
    let chromosome_domain = if patch_domain == WholeCellPatchDomain::NucleoidTrack
        || spatial_scope == WholeCellSpatialScope::NucleoidLocal
    {
        normalized_chromosome_domain_id(chromosome_domain)
    } else {
        None
    };
    let key = (
        field,
        spatial_scope,
        patch_domain,
        chromosome_domain.clone(),
    );
    let entry = requests.entry(key).or_default();
    entry.signal += (0.30 + 0.24 * signal_seed.max(0.0).sqrt()) * weight.max(0.05);
    for target in subsystem_targets {
        if !entry.subsystem_targets.contains(target) {
            entry.subsystem_targets.push(*target);
        }
    }
    localized_pool_species_id(
        field,
        spatial_scope,
        patch_domain,
        chromosome_domain.as_deref(),
    )
}

fn complex_stage_species_id(complex_id: &str, stage: &str) -> String {
    format!("{}_{}", canonical_species_fragment(complex_id), stage)
}

fn registry_compartment_for_asset_class(
    asset_class: WholeCellAssetClass,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> &'static str {
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack) {
        "chromosome"
    } else if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        || subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
        || matches!(
            asset_class,
            WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction
        )
    {
        "membrane"
    } else {
        "cytosol"
    }
}

fn registry_spatial_scope(
    asset_class: WholeCellAssetClass,
    compartment: &str,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellSpatialScope {
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack)
        || compartment.eq_ignore_ascii_case("chromosome")
        || matches!(asset_class, WholeCellAssetClass::Replication)
    {
        WholeCellSpatialScope::NucleoidLocal
    } else if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
        || matches!(asset_class, WholeCellAssetClass::Constriction)
    {
        WholeCellSpatialScope::SeptumLocal
    } else if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        || compartment.eq_ignore_ascii_case("membrane")
        || matches!(asset_class, WholeCellAssetClass::Membrane)
    {
        WholeCellSpatialScope::MembraneAdjacent
    } else {
        WholeCellSpatialScope::WellMixed
    }
}

fn registry_patch_domain(
    asset_class: WholeCellAssetClass,
    compartment: &str,
    subsystem_targets: &[Syn3ASubsystemPreset],
    name: &str,
) -> WholeCellPatchDomain {
    let lowered = name.to_ascii_lowercase();
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack)
        || compartment.eq_ignore_ascii_case("chromosome")
        || matches!(asset_class, WholeCellAssetClass::Replication)
    {
        WholeCellPatchDomain::NucleoidTrack
    } else if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
        || matches!(asset_class, WholeCellAssetClass::Constriction)
        || lowered.contains("sept")
        || lowered.contains("divisome")
    {
        WholeCellPatchDomain::SeptumPatch
    } else if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
        || lowered.contains("atp_synthase")
        || lowered.contains("respir")
        || lowered.contains("membrane_band")
    {
        WholeCellPatchDomain::MembraneBand
    } else if lowered.contains("pole") || lowered.contains("polar") {
        WholeCellPatchDomain::PolarPatch
    } else {
        WholeCellPatchDomain::Distributed
    }
}

fn find_pool_species_id_by_field(
    package: &WholeCellGenomeAssetPackage,
    field: WholeCellBulkField,
) -> Option<String> {
    package
        .pools
        .iter()
        .find(|pool| pool_bulk_field(pool) == Some(field))
        .map(|pool| pool_species_id(&pool.species))
}

fn asset_class_for_pool(pool: &WholeCellMoleculePoolSpec) -> WholeCellAssetClass {
    if let Some(field) = pool_bulk_field(pool) {
        transport_asset_class_for_bulk_field(field)
    } else {
        let lowered = pool.species.to_ascii_lowercase();
        if lowered.contains("membrane") || lowered.contains("lipid") {
            WholeCellAssetClass::Membrane
        } else if lowered.contains("atp")
            || lowered.contains("oxygen")
            || lowered.contains("glucose")
        {
            WholeCellAssetClass::Energy
        } else if lowered.contains("amino") {
            WholeCellAssetClass::Translation
        } else if lowered.contains("nucleotide") {
            WholeCellAssetClass::Replication
        } else {
            WholeCellAssetClass::Generic
        }
    }
}

fn compartment_for_pool(pool: &WholeCellMoleculePoolSpec) -> &'static str {
    if let Some(field) = pool_bulk_field(pool) {
        match field {
            WholeCellBulkField::MembranePrecursors => "membrane",
            _ => "cytosol",
        }
    } else if pool.species.to_ascii_lowercase().contains("membrane")
        || pool.species.to_ascii_lowercase().contains("lipid")
    {
        "membrane"
    } else {
        "cytosol"
    }
}

fn total_complex_stoichiometry(complex: &WholeCellComplexSpec) -> f32 {
    complex
        .components
        .iter()
        .map(|component| component.stoichiometry.max(1) as f32)
        .sum::<f32>()
        .max(1.0)
}

fn operon_bounds(spec: &WholeCellOrganismSpec, genes: &[String]) -> (u32, u32) {
    let mut promoter_bp = u32::MAX;
    let mut terminator_bp = 0u32;
    for gene_name in genes {
        if let Some(feature) = spec.genes.iter().find(|feature| feature.gene == *gene_name) {
            promoter_bp = promoter_bp.min(feature.start_bp.min(feature.end_bp));
            terminator_bp = terminator_bp.max(feature.start_bp.max(feature.end_bp));
        }
    }
    if promoter_bp == u32::MAX {
        (0, 0)
    } else {
        (promoter_bp, terminator_bp)
    }
}

fn transcription_unit_midpoint_bp(
    spec: &WholeCellOrganismSpec,
    unit: &WholeCellTranscriptionUnitSpec,
) -> u32 {
    let (start_bp, end_bp) = operon_bounds(spec, &unit.genes);
    if start_bp == 0 && end_bp == 0 {
        0
    } else {
        midpoint_bp(start_bp, end_bp)
    }
}

fn compile_chromosome_domains(
    spec: &WholeCellOrganismSpec,
    operons: &[WholeCellOperonSpec],
) -> Vec<WholeCellChromosomeDomainSpec> {
    const DEFAULT_DOMAIN_COUNT: usize = 4;

    let genome_bp = spec.chromosome_length_bp.max(1);
    let mut domains = if spec.chromosome_domains.is_empty() {
        let domain_count = DEFAULT_DOMAIN_COUNT.max(1);
        (0..domain_count)
            .map(|index| {
                let start_bp = ((index as u64 * genome_bp as u64) / domain_count as u64) as u32;
                let mut end_bp =
                    (((index as u64 + 1) * genome_bp as u64) / domain_count as u64) as u32;
                end_bp = end_bp.saturating_sub(1);
                let (start_bp, end_bp) =
                    normalized_chromosome_interval(start_bp, end_bp.max(start_bp), genome_bp);
                let center_fraction = ((midpoint_bp(start_bp, end_bp) as f32 + 0.5)
                    / genome_bp as f32)
                    .clamp(0.02, 0.98);
                let spread_fraction =
                    (((end_bp.saturating_sub(start_bp) + 1) as f32 / genome_bp as f32) * 0.75)
                        .clamp(0.08, 0.24);
                WholeCellChromosomeDomainSpec {
                    id: format!("chromosome_domain_{}", index),
                    start_bp,
                    end_bp,
                    axial_center_fraction: center_fraction,
                    axial_spread_fraction: spread_fraction,
                    genes: Vec::new(),
                    transcription_units: Vec::new(),
                    operons: Vec::new(),
                }
            })
            .collect::<Vec<_>>()
    } else {
        spec.chromosome_domains
            .iter()
            .enumerate()
            .map(|(index, domain)| {
                let (start_bp, end_bp) =
                    normalized_chromosome_interval(domain.start_bp, domain.end_bp, genome_bp);
                let center_fraction = if domain.axial_center_fraction > 0.0 {
                    domain.axial_center_fraction.clamp(0.02, 0.98)
                } else {
                    ((midpoint_bp(start_bp, end_bp) as f32 + 0.5) / genome_bp as f32)
                        .clamp(0.02, 0.98)
                };
                WholeCellChromosomeDomainSpec {
                    id: if domain.id.trim().is_empty() {
                        format!("chromosome_domain_{}", index)
                    } else {
                        domain.id.clone()
                    },
                    start_bp,
                    end_bp,
                    axial_center_fraction: center_fraction,
                    axial_spread_fraction: domain.axial_spread_fraction.clamp(0.05, 0.28),
                    genes: domain.genes.clone(),
                    transcription_units: domain.transcription_units.clone(),
                    operons: domain.operons.clone(),
                }
            })
            .collect::<Vec<_>>()
    };
    domains.sort_by_key(|domain| (domain.start_bp, domain.end_bp, domain.id.clone()));

    for domain in &mut domains {
        for gene in &spec.genes {
            let gene_midpoint = gene_midpoint_bp(gene);
            if interval_contains_bp(domain.start_bp, domain.end_bp, gene_midpoint)
                && !domain.genes.contains(&gene.gene)
            {
                domain.genes.push(gene.gene.clone());
            }
        }
        for unit in &spec.transcription_units {
            let midpoint = transcription_unit_midpoint_bp(spec, unit);
            if interval_contains_bp(domain.start_bp, domain.end_bp, midpoint)
                && !domain.transcription_units.contains(&unit.name)
            {
                domain.transcription_units.push(unit.name.clone());
            }
        }
        for operon in operons {
            let midpoint = midpoint_bp(operon.promoter_bp, operon.terminator_bp);
            if interval_contains_bp(domain.start_bp, domain.end_bp, midpoint)
                && !domain.operons.contains(&operon.name)
            {
                domain.operons.push(operon.name.clone());
            }
        }
        domain.genes.sort();
        domain.genes.dedup();
        domain.transcription_units.sort();
        domain.transcription_units.dedup();
        domain.operons.sort();
        domain.operons.dedup();
    }

    domains
}

fn with_compiled_chromosome_domains(mut spec: WholeCellOrganismSpec) -> WholeCellOrganismSpec {
    let mut operons = Vec::new();
    for transcription_unit in &spec.transcription_units {
        let (promoter_bp, terminator_bp) = operon_bounds(&spec, &transcription_unit.genes);
        let subsystem_targets = transcription_unit_subsystem_targets(&spec, transcription_unit);
        let asset_class = transcription_unit_asset_class(transcription_unit, &subsystem_targets);
        let complex_family =
            transcription_unit_complex_family(transcription_unit, asset_class, &subsystem_targets);
        operons.push(WholeCellOperonSpec {
            name: transcription_unit.name.clone(),
            genes: transcription_unit.genes.clone(),
            promoter_bp,
            terminator_bp,
            basal_activity: transcription_unit.basal_activity.max(0.0),
            polycistronic: transcription_unit.genes.len() > 1,
            process_weights: transcription_unit.process_weights.clamped(),
            subsystem_targets,
            asset_class: Some(asset_class),
            complex_family: Some(complex_family),
        });
    }
    for gene in &spec.genes {
        if operons.iter().any(|operon| operon.name == gene.gene) {
            continue;
        }
        if spec
            .transcription_units
            .iter()
            .any(|unit| unit.genes.contains(&gene.gene))
        {
            continue;
        }
        let asset_class = gene_asset_class(gene);
        operons.push(WholeCellOperonSpec {
            name: gene.gene.clone(),
            genes: vec![gene.gene.clone()],
            promoter_bp: gene.start_bp.min(gene.end_bp),
            terminator_bp: gene.start_bp.max(gene.end_bp),
            basal_activity: gene.basal_expression.max(0.0),
            polycistronic: false,
            process_weights: gene.process_weights.clamped(),
            subsystem_targets: gene.subsystem_targets.clone(),
            asset_class: Some(asset_class),
            complex_family: Some(gene_complex_family(gene, asset_class)),
        });
    }
    spec.chromosome_domains = compile_chromosome_domains(&spec, &operons);
    spec
}

fn push_unique_subsystem_targets(
    targets: &mut Vec<Syn3ASubsystemPreset>,
    candidates: &[Syn3ASubsystemPreset],
) {
    for candidate in candidates {
        if !targets.contains(candidate) {
            targets.push(*candidate);
        }
    }
}

fn transcription_unit_subsystem_targets(
    spec: &WholeCellOrganismSpec,
    transcription_unit: &WholeCellTranscriptionUnitSpec,
) -> Vec<Syn3ASubsystemPreset> {
    let mut subsystem_targets = transcription_unit.subsystem_targets.clone();
    for gene_name in &transcription_unit.genes {
        if let Some(gene) = spec.genes.iter().find(|gene| gene.gene == *gene_name) {
            push_unique_subsystem_targets(&mut subsystem_targets, &gene.subsystem_targets);
        }
    }
    subsystem_targets
}

fn transcription_unit_asset_class(
    transcription_unit: &WholeCellTranscriptionUnitSpec,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellAssetClass {
    transcription_unit.asset_class.unwrap_or_else(|| {
        inferred_asset_class(
            transcription_unit.process_weights,
            subsystem_targets,
            &transcription_unit.name,
        )
    })
}

fn transcription_unit_complex_family(
    transcription_unit: &WholeCellTranscriptionUnitSpec,
    asset_class: WholeCellAssetClass,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellAssemblyFamily {
    transcription_unit.complex_family.unwrap_or_else(|| {
        inferred_complex_family(asset_class, subsystem_targets, &transcription_unit.name)
    })
}

fn gene_asset_class(gene: &WholeCellGenomeFeature) -> WholeCellAssetClass {
    gene.asset_class.unwrap_or_else(|| {
        inferred_asset_class(gene.process_weights, &gene.subsystem_targets, &gene.gene)
    })
}

fn gene_complex_family(
    gene: &WholeCellGenomeFeature,
    asset_class: WholeCellAssetClass,
) -> WholeCellAssemblyFamily {
    gene.complex_family.unwrap_or_else(|| {
        inferred_complex_family(asset_class, &gene.subsystem_targets, &gene.gene)
    })
}

pub fn compile_genome_asset_package(spec: &WholeCellOrganismSpec) -> WholeCellGenomeAssetPackage {
    let spec = with_compiled_chromosome_domains(with_normalized_semantic_metadata(
        with_normalized_pool_metadata(spec.clone()),
    ));
    let mut gene_to_operon = HashMap::<String, String>::new();
    let mut operons = Vec::new();

    for transcription_unit in &spec.transcription_units {
        let (promoter_bp, terminator_bp) = operon_bounds(&spec, &transcription_unit.genes);
        for gene_name in &transcription_unit.genes {
            gene_to_operon.insert(gene_name.clone(), transcription_unit.name.clone());
        }
        let subsystem_targets = transcription_unit_subsystem_targets(&spec, transcription_unit);
        let asset_class = transcription_unit_asset_class(transcription_unit, &subsystem_targets);
        let complex_family =
            transcription_unit_complex_family(transcription_unit, asset_class, &subsystem_targets);
        operons.push(WholeCellOperonSpec {
            name: transcription_unit.name.clone(),
            genes: transcription_unit.genes.clone(),
            promoter_bp,
            terminator_bp,
            basal_activity: transcription_unit.basal_activity.max(0.0),
            polycistronic: transcription_unit.genes.len() > 1,
            process_weights: transcription_unit.process_weights.clamped(),
            subsystem_targets,
            asset_class: Some(asset_class),
            complex_family: Some(complex_family),
        });
    }

    for gene in &spec.genes {
        if gene_to_operon.contains_key(&gene.gene) {
            continue;
        }
        gene_to_operon.insert(gene.gene.clone(), gene.gene.clone());
        let asset_class = gene_asset_class(gene);
        operons.push(WholeCellOperonSpec {
            name: gene.gene.clone(),
            genes: vec![gene.gene.clone()],
            promoter_bp: gene.start_bp.min(gene.end_bp),
            terminator_bp: gene.start_bp.max(gene.end_bp),
            basal_activity: gene.basal_expression.max(0.0),
            polycistronic: false,
            process_weights: gene.process_weights.clamped(),
            subsystem_targets: gene.subsystem_targets.clone(),
            asset_class: Some(asset_class),
            complex_family: Some(gene_complex_family(gene, asset_class)),
        });
    }

    let mut rnas = Vec::new();
    let mut proteins = Vec::new();
    for gene in &spec.genes {
        let length_nt = gene_length_bp(gene).max(1);
        let operon_name = gene_to_operon
            .get(&gene.gene)
            .cloned()
            .unwrap_or_else(|| gene.gene.clone());
        let asset_class = gene_asset_class(gene);
        let rna_id = format!("{}_rna", gene.gene);
        let protein_id = format!("{}_protein", gene.gene);
        rnas.push(WholeCellRnaProductSpec {
            id: rna_id.clone(),
            gene: gene.gene.clone(),
            operon: operon_name.clone(),
            length_nt,
            basal_abundance: (4.0 + 6.0 * gene.basal_expression.max(0.05)).clamp(0.5, 256.0),
            asset_class,
            process_weights: gene.process_weights.clamped(),
        });
        proteins.push(WholeCellProteinProductSpec {
            id: protein_id,
            gene: gene.gene.clone(),
            operon: operon_name,
            rna_id,
            aa_length: (length_nt / 3).max(1),
            basal_abundance: (8.0 + 10.0 * gene.basal_expression.max(0.05)).clamp(0.5, 512.0),
            translation_cost: gene.translation_cost.max(0.0),
            nucleotide_cost: gene.nucleotide_cost.max(0.0),
            asset_class,
            process_weights: gene.process_weights.clamped(),
            subsystem_targets: gene.subsystem_targets.clone(),
        });
    }

    let mut complexes = Vec::new();
    for operon in &operons {
        let mut components = Vec::new();
        let mut process_weights = operon.process_weights;
        let mut subsystem_targets = operon.subsystem_targets.clone();
        for gene_name in &operon.genes {
            components.push(WholeCellComplexComponentSpec {
                protein_id: format!("{}_protein", gene_name),
                stoichiometry: 1,
            });
            if let Some(gene) = spec.genes.iter().find(|gene| gene.gene == *gene_name) {
                process_weights.add_weighted(gene.process_weights, 0.35);
                push_unique_subsystem_targets(&mut subsystem_targets, &gene.subsystem_targets);
            }
        }
        let asset_class = operon.asset_class.unwrap_or_else(|| {
            inferred_asset_class(process_weights, &subsystem_targets, &operon.name)
        });
        let family = operon.complex_family.unwrap_or_else(|| {
            inferred_complex_family(asset_class, &subsystem_targets, &operon.name)
        });
        complexes.push(WholeCellComplexSpec {
            id: format!("{}_complex", operon.name),
            name: format!("{} complex", operon.name),
            operon: operon.name.clone(),
            components,
            basal_abundance: (3.0
                + 7.0
                    * operon.basal_activity.max(0.05)
                    * (operon.genes.len().max(1) as f32).sqrt())
            .clamp(0.5, 256.0),
            asset_class,
            family,
            process_weights: process_weights.clamped(),
            subsystem_targets,
            membrane_inserted: matches!(
                family,
                WholeCellAssemblyFamily::AtpSynthase
                    | WholeCellAssemblyFamily::Transporter
                    | WholeCellAssemblyFamily::MembraneEnzyme
                    | WholeCellAssemblyFamily::Divisome
            ),
            chromosome_coupled: matches!(
                family,
                WholeCellAssemblyFamily::Replisome | WholeCellAssemblyFamily::RnaPolymerase
            ),
            division_coupled: matches!(family, WholeCellAssemblyFamily::Divisome),
        });
    }

    let operon_semantics = compile_operon_semantic_specs(&operons);
    let protein_semantics = compile_protein_semantic_specs(&proteins);
    let complex_semantics = compile_complex_semantic_specs(&complexes);

    WholeCellGenomeAssetPackage {
        organism: spec.organism.clone(),
        chromosome_length_bp: spec.chromosome_length_bp.max(1),
        origin_bp: spec.origin_bp.min(spec.chromosome_length_bp.max(1)),
        terminus_bp: spec.terminus_bp.min(spec.chromosome_length_bp.max(1)),
        chromosome_domains: spec.chromosome_domains.clone(),
        operon_semantics,
        operons,
        rnas,
        proteins,
        protein_semantics,
        complex_semantics,
        complexes,
        pools: spec.pools.clone(),
    }
}

fn empty_genome_asset_package(spec: &WholeCellOrganismSpec) -> WholeCellGenomeAssetPackage {
    WholeCellGenomeAssetPackage {
        organism: spec.organism.clone(),
        chromosome_length_bp: spec.chromosome_length_bp,
        origin_bp: spec.origin_bp,
        terminus_bp: spec.terminus_bp,
        chromosome_domains: spec.chromosome_domains.clone(),
        operons: Vec::new(),
        operon_semantics: Vec::new(),
        rnas: Vec::new(),
        proteins: Vec::new(),
        protein_semantics: Vec::new(),
        complex_semantics: Vec::new(),
        complexes: Vec::new(),
        pools: spec.pools.clone(),
    }
}

pub fn compile_genome_process_registry(
    package: &WholeCellGenomeAssetPackage,
) -> WholeCellGenomeProcessRegistry {
    let mut species = Vec::new();
    let mut reactions = Vec::new();
    let operon_domain_map: HashMap<String, String> = package
        .chromosome_domains
        .iter()
        .flat_map(|domain| {
            domain
                .operons
                .iter()
                .map(|operon| (operon.clone(), domain.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    let gene_domain_map: HashMap<String, String> = package
        .chromosome_domains
        .iter()
        .flat_map(|domain| {
            domain
                .genes
                .iter()
                .map(|gene| (gene.clone(), domain.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut localized_pool_requests: HashMap<
        (
            WholeCellBulkField,
            WholeCellSpatialScope,
            WholeCellPatchDomain,
            Option<String>,
        ),
        LocalizedPoolRequest,
    > = HashMap::new();
    let atp_pool = find_pool_species_id_by_field(package, WholeCellBulkField::ATP);
    let adp_pool = find_pool_species_id_by_field(package, WholeCellBulkField::ADP);
    let nucleotide_pool = find_pool_species_id_by_field(package, WholeCellBulkField::Nucleotides);
    let amino_pool = find_pool_species_id_by_field(package, WholeCellBulkField::AminoAcids);
    let global_pools_by_field: HashMap<WholeCellBulkField, (String, &WholeCellMoleculePoolSpec)> =
        package
            .pools
            .iter()
            .filter_map(|pool| {
                pool_bulk_field(pool).map(|field| (field, (pool_species_id(&pool.species), pool)))
            })
            .collect();
    let mut localized_pool_participant =
        |global_species_id: &String,
         field: WholeCellBulkField,
         spatial_scope: WholeCellSpatialScope,
         patch_domain: WholeCellPatchDomain,
         chromosome_domain: Option<&str>,
         signal_seed: f32,
         asset_class: WholeCellAssetClass,
         subsystem_targets: &[Syn3ASubsystemPreset]| {
            if (spatial_scope == WholeCellSpatialScope::WellMixed
                && patch_domain == WholeCellPatchDomain::Distributed)
                || !localized_pool_supports_bulk_field(field)
            {
                return global_species_id.clone();
            }
            let Some(weight) = localized_pool_request_weight(field, asset_class) else {
                return global_species_id.clone();
            };
            register_localized_pool_request(
                &mut localized_pool_requests,
                field,
                spatial_scope,
                patch_domain,
                chromosome_domain,
                signal_seed,
                weight,
                subsystem_targets,
            )
        };
    let operon_domain = |operon: &str| operon_domain_map.get(operon).cloned();
    let gene_domain = |gene: &str| gene_domain_map.get(gene).cloned();

    for pool in &package.pools {
        let species_name = pool.species.clone();
        let bulk_field = pool_bulk_field(pool);
        let asset_class = asset_class_for_pool(pool);
        let compartment = compartment_for_pool(pool);
        let spatial_scope = registry_spatial_scope(asset_class, compartment, &[]);
        let patch_domain = registry_patch_domain(asset_class, compartment, &[], &species_name);
        species.push(WholeCellSpeciesSpec {
            id: pool_species_id(&pool.species),
            name: species_name,
            species_class: WholeCellSpeciesClass::Pool,
            compartment: compartment.to_string(),
            asset_class,
            basal_abundance: (pool.count.max(0.0) + 24.0 * pool.concentration_mm.max(0.0))
                .clamp(0.0, 4096.0),
            bulk_field,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            spatial_scope,
            patch_domain,
            chromosome_domain: None,
        });
    }

    for rna in &package.rnas {
        let compartment = registry_compartment_for_asset_class(
            rna.asset_class,
            &Vec::<Syn3ASubsystemPreset>::new(),
        );
        let spatial_scope = registry_spatial_scope(rna.asset_class, compartment, &[]);
        let patch_domain = registry_patch_domain(rna.asset_class, compartment, &[], &rna.id);
        species.push(WholeCellSpeciesSpec {
            id: rna.id.clone(),
            name: rna.id.clone(),
            species_class: WholeCellSpeciesClass::Rna,
            compartment: compartment.to_string(),
            asset_class: rna.asset_class,
            basal_abundance: rna.basal_abundance.max(0.0),
            bulk_field: None,
            operon: Some(rna.operon.clone()),
            parent_complex: None,
            subsystem_targets: Vec::new(),
            spatial_scope,
            patch_domain,
            chromosome_domain: operon_domain(&rna.operon),
        });
    }

    for protein in &package.proteins {
        let compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let spatial_scope =
            registry_spatial_scope(protein.asset_class, compartment, &protein.subsystem_targets);
        let patch_domain = registry_patch_domain(
            protein.asset_class,
            compartment,
            &protein.subsystem_targets,
            &protein.id,
        );
        species.push(WholeCellSpeciesSpec {
            id: protein.id.clone(),
            name: protein.id.clone(),
            species_class: WholeCellSpeciesClass::Protein,
            compartment: compartment.to_string(),
            asset_class: protein.asset_class,
            basal_abundance: protein.basal_abundance.max(0.0),
            bulk_field: None,
            operon: Some(protein.operon.clone()),
            parent_complex: None,
            subsystem_targets: protein.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: gene_domain(&protein.gene)
                .or_else(|| operon_domain(&protein.operon)),
        });
    }

    for complex in &package.complexes {
        let compartment =
            registry_compartment_for_asset_class(complex.asset_class, &complex.subsystem_targets);
        let spatial_scope =
            registry_spatial_scope(complex.asset_class, compartment, &complex.subsystem_targets);
        let patch_domain = registry_patch_domain(
            complex.asset_class,
            compartment,
            &complex.subsystem_targets,
            &complex.name,
        );
        let total_stoichiometry = total_complex_stoichiometry(complex);
        let subunit_pool_id = complex_stage_species_id(&complex.id, "subunit_pool");
        let nucleation_id = complex_stage_species_id(&complex.id, "nucleation");
        let elongation_id = complex_stage_species_id(&complex.id, "elongation");
        let mature_id = complex_stage_species_id(&complex.id, "mature");
        species.push(WholeCellSpeciesSpec {
            id: subunit_pool_id.clone(),
            name: format!("{} subunit pool", complex.name),
            species_class: WholeCellSpeciesClass::ComplexSubunitPool,
            compartment: compartment.to_string(),
            asset_class: complex.asset_class,
            basal_abundance: (complex.basal_abundance.max(0.0) * total_stoichiometry.sqrt())
                .clamp(0.0, 2048.0),
            bulk_field: None,
            operon: Some(complex.operon.clone()),
            parent_complex: Some(complex.id.clone()),
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: operon_domain(&complex.operon),
        });
        species.push(WholeCellSpeciesSpec {
            id: nucleation_id.clone(),
            name: format!("{} nucleation intermediate", complex.name),
            species_class: WholeCellSpeciesClass::ComplexNucleationIntermediate,
            compartment: compartment.to_string(),
            asset_class: complex.asset_class,
            basal_abundance: (0.10 * complex.basal_abundance.max(0.0) * total_stoichiometry.sqrt())
                .clamp(0.0, 512.0),
            bulk_field: None,
            operon: Some(complex.operon.clone()),
            parent_complex: Some(complex.id.clone()),
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: operon_domain(&complex.operon),
        });
        species.push(WholeCellSpeciesSpec {
            id: elongation_id.clone(),
            name: format!("{} elongation intermediate", complex.name),
            species_class: WholeCellSpeciesClass::ComplexElongationIntermediate,
            compartment: compartment.to_string(),
            asset_class: complex.asset_class,
            basal_abundance: (0.08 * complex.basal_abundance.max(0.0) * total_stoichiometry.sqrt())
                .clamp(0.0, 512.0),
            bulk_field: None,
            operon: Some(complex.operon.clone()),
            parent_complex: Some(complex.id.clone()),
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: operon_domain(&complex.operon),
        });
        species.push(WholeCellSpeciesSpec {
            id: mature_id.clone(),
            name: complex.name.clone(),
            species_class: WholeCellSpeciesClass::ComplexMature,
            compartment: compartment.to_string(),
            asset_class: complex.asset_class,
            basal_abundance: complex.basal_abundance.max(0.0),
            bulk_field: None,
            operon: Some(complex.operon.clone()),
            parent_complex: Some(complex.id.clone()),
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: operon_domain(&complex.operon),
        });
    }

    for pool in &package.pools {
        let pool_id = pool_species_id(&pool.species);
        let Some(field) = pool_bulk_field(pool) else {
            continue;
        };
        if !bulk_field_supports_transport(field) {
            continue;
        }
        let asset_class = transport_asset_class_for_bulk_field(field);
        let compartment = registry_compartment_for_asset_class(asset_class, &[]);
        let patch_domain = registry_patch_domain(asset_class, compartment, &[], &pool.species);
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_transport", canonical_species_fragment(&pool_id)),
            name: format!("{} transport", pool.species),
            reaction_class: WholeCellReactionClass::PoolTransport,
            asset_class,
            nominal_rate: bulk_field_transport_rate(field),
            catalyst: None,
            operon: None,
            reactants: Vec::new(),
            products: vec![WholeCellReactionParticipantSpec {
                species_id: pool_id,
                stoichiometry: 1.0,
            }],
            subsystem_targets: Vec::new(),
            spatial_scope: registry_spatial_scope(asset_class, compartment, &[]),
            patch_domain,
            chromosome_domain: None,
        });
    }

    for operon in &package.operons {
        let operon_chromosome_domain = operon_domain(&operon.name);
        let operon_asset_class = operon.asset_class.unwrap_or_else(|| {
            inferred_asset_class(
                operon.process_weights,
                &operon.subsystem_targets,
                &operon.name,
            )
        });
        let operon_compartment =
            registry_compartment_for_asset_class(operon_asset_class, &operon.subsystem_targets);
        let operon_spatial_scope = registry_spatial_scope(
            operon_asset_class,
            operon_compartment,
            &operon.subsystem_targets,
        );
        let operon_patch_domain = registry_patch_domain(
            operon_asset_class,
            operon_compartment,
            &operon.subsystem_targets,
            &operon.name,
        );
        let operon_signal_seed =
            operon.basal_activity.max(0.01) * (operon.genes.len().max(1) as f32).sqrt().max(1.0);
        let mut reactants = Vec::new();
        if let Some(species_id) = nucleotide_pool.as_ref() {
            reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::Nucleotides,
                    operon_spatial_scope,
                    operon_patch_domain,
                    operon_chromosome_domain.as_deref(),
                    operon_signal_seed,
                    operon_asset_class,
                    &[],
                ),
                stoichiometry: (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        let products = package
            .rnas
            .iter()
            .filter(|rna| rna.operon == operon.name)
            .map(|rna| WholeCellReactionParticipantSpec {
                species_id: rna.id.clone(),
                stoichiometry: 1.0,
            })
            .collect::<Vec<_>>();
        if !products.is_empty() {
            reactions.push(WholeCellReactionSpec {
                id: format!("{}_transcription", canonical_species_fragment(&operon.name)),
                name: format!("{} transcription", operon.name),
                reaction_class: WholeCellReactionClass::Transcription,
                asset_class: operon_asset_class,
                nominal_rate: operon.basal_activity.max(0.01),
                catalyst: None,
                operon: Some(operon.name.clone()),
                reactants,
                products,
                subsystem_targets: Vec::new(),
                spatial_scope: operon_spatial_scope,
                patch_domain: operon_patch_domain,
                chromosome_domain: operon_chromosome_domain.clone(),
            });
        }

        let mut stress_reactants = Vec::new();
        if let Some(species_id) = atp_pool.as_ref() {
            stress_reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::ATP,
                    operon_spatial_scope,
                    operon_patch_domain,
                    operon_chromosome_domain.as_deref(),
                    operon_signal_seed,
                    operon_asset_class,
                    &[],
                ),
                stoichiometry: (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        if let Some(species_id) = amino_pool.as_ref() {
            stress_reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::AminoAcids,
                    operon_spatial_scope,
                    operon_patch_domain,
                    operon_chromosome_domain.as_deref(),
                    operon_signal_seed,
                    operon_asset_class,
                    &[],
                ),
                stoichiometry: 0.40 * (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        let mut stress_products = Vec::new();
        if let Some(species_id) = adp_pool.as_ref() {
            stress_products.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::ADP,
                    operon_spatial_scope,
                    operon_patch_domain,
                    operon_chromosome_domain.as_deref(),
                    operon_signal_seed,
                    operon_asset_class,
                    &[],
                ),
                stoichiometry: (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!(
                "{}_stress_response",
                canonical_species_fragment(&operon.name)
            ),
            name: format!("{} stress response", operon.name),
            reaction_class: WholeCellReactionClass::StressResponse,
            asset_class: operon_asset_class,
            nominal_rate: (0.012 + 0.005 * (operon.genes.len().max(1) as f32).sqrt())
                .clamp(0.004, 4.0),
            catalyst: None,
            operon: Some(operon.name.clone()),
            reactants: stress_reactants,
            products: stress_products,
            subsystem_targets: Vec::new(),
            spatial_scope: operon_spatial_scope,
            patch_domain: operon_patch_domain,
            chromosome_domain: operon_chromosome_domain,
        });
    }

    for protein in &package.proteins {
        let protein_chromosome_domain =
            gene_domain(&protein.gene).or_else(|| operon_domain(&protein.operon));
        let protein_compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let protein_spatial_scope = registry_spatial_scope(
            protein.asset_class,
            protein_compartment,
            &protein.subsystem_targets,
        );
        let protein_patch_domain = registry_patch_domain(
            protein.asset_class,
            protein_compartment,
            &protein.subsystem_targets,
            &protein.id,
        );
        let protein_signal_seed = protein.translation_cost.max(1.0);
        let mut reactants = vec![WholeCellReactionParticipantSpec {
            species_id: protein.rna_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = amino_pool.as_ref() {
            reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::AminoAcids,
                    protein_spatial_scope,
                    protein_patch_domain,
                    protein_chromosome_domain.as_deref(),
                    protein_signal_seed,
                    protein.asset_class,
                    &protein.subsystem_targets,
                ),
                stoichiometry: (protein.aa_length.max(1) as f32 / 24.0).max(1.0),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_translation", canonical_species_fragment(&protein.id)),
            name: format!("{} translation", protein.id),
            reaction_class: WholeCellReactionClass::Translation,
            asset_class: protein.asset_class,
            nominal_rate: (0.04 + 0.002 * protein.translation_cost.max(0.0)).clamp(0.01, 8.0),
            catalyst: Some(protein.rna_id.clone()),
            operon: Some(protein.operon.clone()),
            reactants,
            products: vec![WholeCellReactionParticipantSpec {
                species_id: protein.id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: protein.subsystem_targets.clone(),
            spatial_scope: protein_spatial_scope,
            patch_domain: protein_patch_domain,
            chromosome_domain: protein_chromosome_domain,
        });
    }

    for rna in &package.rnas {
        let rna_chromosome_domain = operon_domain(&rna.operon);
        let rna_compartment = registry_compartment_for_asset_class(
            rna.asset_class,
            &Vec::<Syn3ASubsystemPreset>::new(),
        );
        let rna_spatial_scope = registry_spatial_scope(rna.asset_class, rna_compartment, &[]);
        let rna_patch_domain =
            registry_patch_domain(rna.asset_class, rna_compartment, &[], &rna.id);
        let mut products = Vec::new();
        if let Some(species_id) = nucleotide_pool.as_ref() {
            products.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::Nucleotides,
                    rna_spatial_scope,
                    rna_patch_domain,
                    rna_chromosome_domain.as_deref(),
                    (rna.length_nt.max(1) as f32 / 36.0).max(1.0),
                    rna.asset_class,
                    &[],
                ),
                stoichiometry: (rna.length_nt.max(1) as f32 / 36.0).max(1.0),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_degradation", canonical_species_fragment(&rna.id)),
            name: format!("{} degradation", rna.id),
            reaction_class: WholeCellReactionClass::RnaDegradation,
            asset_class: WholeCellAssetClass::Homeostasis,
            nominal_rate: (0.010 + 0.0025 * (rna.length_nt.max(1) as f32 / 120.0).sqrt())
                .clamp(0.005, 4.0),
            catalyst: None,
            operon: Some(rna.operon.clone()),
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: rna.id.clone(),
                stoichiometry: 1.0,
            }],
            products,
            subsystem_targets: Vec::new(),
            spatial_scope: rna_spatial_scope,
            patch_domain: rna_patch_domain,
            chromosome_domain: rna_chromosome_domain,
        });
    }

    for protein in &package.proteins {
        let protein_chromosome_domain =
            gene_domain(&protein.gene).or_else(|| operon_domain(&protein.operon));
        let protein_compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let protein_spatial_scope = registry_spatial_scope(
            protein.asset_class,
            protein_compartment,
            &protein.subsystem_targets,
        );
        let protein_patch_domain = registry_patch_domain(
            protein.asset_class,
            protein_compartment,
            &protein.subsystem_targets,
            &protein.id,
        );
        let mut products = Vec::new();
        if let Some(species_id) = amino_pool.as_ref() {
            products.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::AminoAcids,
                    protein_spatial_scope,
                    protein_patch_domain,
                    protein_chromosome_domain.as_deref(),
                    (protein.aa_length.max(1) as f32 / 18.0).max(1.0),
                    protein.asset_class,
                    &protein.subsystem_targets,
                ),
                stoichiometry: (protein.aa_length.max(1) as f32 / 18.0).max(1.0),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_degradation", canonical_species_fragment(&protein.id)),
            name: format!("{} degradation", protein.id),
            reaction_class: WholeCellReactionClass::ProteinDegradation,
            asset_class: WholeCellAssetClass::QualityControl,
            nominal_rate: (0.008 + 0.0020 * (protein.aa_length.max(1) as f32 / 90.0).sqrt())
                .clamp(0.004, 4.0),
            catalyst: None,
            operon: Some(protein.operon.clone()),
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: protein.id.clone(),
                stoichiometry: 1.0,
            }],
            products,
            subsystem_targets: protein.subsystem_targets.clone(),
            spatial_scope: protein_spatial_scope,
            patch_domain: protein_patch_domain,
            chromosome_domain: protein_chromosome_domain,
        });
    }

    for complex in &package.complexes {
        let complex_chromosome_domain = operon_domain(&complex.operon);
        let complex_compartment =
            registry_compartment_for_asset_class(complex.asset_class, &complex.subsystem_targets);
        let complex_spatial_scope = registry_spatial_scope(
            complex.asset_class,
            complex_compartment,
            &complex.subsystem_targets,
        );
        let complex_patch_domain = registry_patch_domain(
            complex.asset_class,
            complex_compartment,
            &complex.subsystem_targets,
            &complex.name,
        );
        let total_stoichiometry = total_complex_stoichiometry(complex);
        let subunit_pool_id = complex_stage_species_id(&complex.id, "subunit_pool");
        let nucleation_id = complex_stage_species_id(&complex.id, "nucleation");
        let elongation_id = complex_stage_species_id(&complex.id, "elongation");
        let mature_id = complex_stage_species_id(&complex.id, "mature");
        let assembly_energy_signal = (total_stoichiometry.max(1.0)
            * (1.0 + 0.35 * complex.basal_abundance.max(0.1).sqrt()))
        .max(1.0);
        let localized_assembly_atp_pool = atp_pool.as_ref().map(|species_id| {
            localized_pool_participant(
                species_id,
                WholeCellBulkField::ATP,
                complex_spatial_scope,
                complex_patch_domain,
                complex_chromosome_domain.as_deref(),
                assembly_energy_signal,
                complex.asset_class,
                &complex.subsystem_targets,
            )
        });
        let localized_assembly_adp_pool = adp_pool.as_ref().map(|species_id| {
            localized_pool_participant(
                species_id,
                WholeCellBulkField::ADP,
                complex_spatial_scope,
                complex_patch_domain,
                complex_chromosome_domain.as_deref(),
                assembly_energy_signal,
                complex.asset_class,
                &complex.subsystem_targets,
            )
        });
        let mut subunit_pool_reactants = complex
            .components
            .iter()
            .map(|component| WholeCellReactionParticipantSpec {
                species_id: component.protein_id.clone(),
                stoichiometry: component.stoichiometry.max(1) as f32,
            })
            .collect::<Vec<_>>();
        if let Some(species_id) = localized_assembly_atp_pool.as_ref() {
            subunit_pool_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.16 * total_stoichiometry.sqrt()).clamp(0.12, 0.90),
            });
        }
        let mut subunit_pool_products = vec![WholeCellReactionParticipantSpec {
            species_id: subunit_pool_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_adp_pool.as_ref() {
            subunit_pool_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.12 * total_stoichiometry.sqrt()).clamp(0.08, 0.75),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!(
                "{}_subunit_pool_formation",
                canonical_species_fragment(&complex.id)
            ),
            name: format!("{} subunit pool formation", complex.name),
            reaction_class: WholeCellReactionClass::SubunitPoolFormation,
            asset_class: complex.asset_class,
            nominal_rate: (0.05 + 0.015 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: subunit_pool_reactants,
            products: subunit_pool_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain.clone(),
        });
        let mut nucleation_reactants = vec![WholeCellReactionParticipantSpec {
            species_id: subunit_pool_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_atp_pool.as_ref() {
            nucleation_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.18 * total_stoichiometry.sqrt()).clamp(0.14, 1.0),
            });
        }
        let mut nucleation_products = vec![WholeCellReactionParticipantSpec {
            species_id: nucleation_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_adp_pool.as_ref() {
            nucleation_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.14 * total_stoichiometry.sqrt()).clamp(0.10, 0.85),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_nucleation", canonical_species_fragment(&complex.id)),
            name: format!("{} nucleation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexNucleation,
            asset_class: complex.asset_class,
            nominal_rate: (0.03 + 0.010 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: nucleation_reactants,
            products: nucleation_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain.clone(),
        });
        let mut elongation_reactants = vec![
            WholeCellReactionParticipantSpec {
                species_id: nucleation_id.clone(),
                stoichiometry: 1.0,
            },
            WholeCellReactionParticipantSpec {
                species_id: subunit_pool_id.clone(),
                stoichiometry: 0.5 * total_stoichiometry.max(1.0),
            },
        ];
        if let Some(species_id) = localized_assembly_atp_pool.as_ref() {
            elongation_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.20 * total_stoichiometry.sqrt()).clamp(0.16, 1.2),
            });
        }
        let mut elongation_products = vec![WholeCellReactionParticipantSpec {
            species_id: elongation_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_adp_pool.as_ref() {
            elongation_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.16 * total_stoichiometry.sqrt()).clamp(0.12, 0.95),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_elongation", canonical_species_fragment(&complex.id)),
            name: format!("{} elongation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexElongation,
            asset_class: complex.asset_class,
            nominal_rate: (0.04 + 0.012 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: elongation_reactants,
            products: elongation_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain.clone(),
        });
        let mut maturation_reactants = vec![WholeCellReactionParticipantSpec {
            species_id: elongation_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_atp_pool.as_ref() {
            maturation_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.14 + 0.06 * complex.basal_abundance.max(0.1).sqrt())
                    .clamp(0.12, 1.0),
            });
        }
        let mut maturation_products = vec![WholeCellReactionParticipantSpec {
            species_id: mature_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_adp_pool.as_ref() {
            maturation_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.10 + 0.04 * complex.basal_abundance.max(0.1).sqrt())
                    .clamp(0.08, 0.80),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_maturation", canonical_species_fragment(&complex.id)),
            name: format!("{} maturation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexMaturation,
            asset_class: complex.asset_class,
            nominal_rate: (0.05 + 0.015 * complex.basal_abundance.max(0.1).sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: maturation_reactants,
            products: maturation_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain.clone(),
        });
        let mut repair_reactants = vec![WholeCellReactionParticipantSpec {
            species_id: subunit_pool_id.clone(),
            stoichiometry: 0.5 * total_stoichiometry.max(1.0),
        }];
        if let Some(species_id) = atp_pool.as_ref() {
            repair_reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::ATP,
                    complex_spatial_scope,
                    complex_patch_domain,
                    complex_chromosome_domain.as_deref(),
                    total_stoichiometry.max(1.0),
                    complex.asset_class,
                    &complex.subsystem_targets,
                ),
                stoichiometry: 0.40 * total_stoichiometry.sqrt().max(1.0),
            });
        }
        if let Some(species_id) = amino_pool.as_ref() {
            repair_reactants.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::AminoAcids,
                    complex_spatial_scope,
                    complex_patch_domain,
                    complex_chromosome_domain.as_deref(),
                    total_stoichiometry.max(1.0),
                    complex.asset_class,
                    &complex.subsystem_targets,
                ),
                stoichiometry: 0.25 * total_stoichiometry.sqrt().max(1.0),
            });
        }
        let mut repair_products = vec![WholeCellReactionParticipantSpec {
            species_id: mature_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = adp_pool.as_ref() {
            repair_products.push(WholeCellReactionParticipantSpec {
                species_id: localized_pool_participant(
                    species_id,
                    WholeCellBulkField::ADP,
                    complex_spatial_scope,
                    complex_patch_domain,
                    complex_chromosome_domain.as_deref(),
                    total_stoichiometry.max(1.0),
                    complex.asset_class,
                    &complex.subsystem_targets,
                ),
                stoichiometry: 0.30 * total_stoichiometry.sqrt().max(1.0),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_repair", canonical_species_fragment(&complex.id)),
            name: format!("{} repair", complex.name),
            reaction_class: WholeCellReactionClass::ComplexRepair,
            asset_class: complex.asset_class,
            nominal_rate: (0.018 + 0.008 * total_stoichiometry.sqrt()).clamp(0.004, 4.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: repair_reactants,
            products: repair_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain.clone(),
        });
        let mut turnover_reactants = vec![WholeCellReactionParticipantSpec {
            species_id: mature_id,
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_atp_pool.as_ref() {
            turnover_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.10 * total_stoichiometry.sqrt()).clamp(0.06, 0.60),
            });
        }
        let mut turnover_products = vec![WholeCellReactionParticipantSpec {
            species_id: subunit_pool_id,
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = localized_assembly_adp_pool.as_ref() {
            turnover_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (0.08 * total_stoichiometry.sqrt()).clamp(0.05, 0.45),
            });
        }
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_turnover", canonical_species_fragment(&complex.id)),
            name: format!("{} turnover", complex.name),
            reaction_class: WholeCellReactionClass::ComplexTurnover,
            asset_class: complex.asset_class,
            nominal_rate: (0.02 + 0.010 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: turnover_reactants,
            products: turnover_products,
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: complex_spatial_scope,
            patch_domain: complex_patch_domain,
            chromosome_domain: complex_chromosome_domain,
        });
    }

    for species_spec in &species {
        if species_spec.spatial_scope == WholeCellSpatialScope::WellMixed
            && species_spec.patch_domain == WholeCellPatchDomain::Distributed
        {
            continue;
        }
        let signal_seed = match species_spec.species_class {
            WholeCellSpeciesClass::Pool => 0.35 * species_spec.basal_abundance.max(0.0),
            WholeCellSpeciesClass::Rna => 0.55 * species_spec.basal_abundance.max(0.0),
            WholeCellSpeciesClass::Protein => 0.70 * species_spec.basal_abundance.max(0.0),
            WholeCellSpeciesClass::ComplexSubunitPool
            | WholeCellSpeciesClass::ComplexNucleationIntermediate
            | WholeCellSpeciesClass::ComplexElongationIntermediate => {
                0.85 * species_spec.basal_abundance.max(0.0)
            }
            WholeCellSpeciesClass::ComplexMature => species_spec.basal_abundance.max(0.0),
        };
        for (field, weight) in localized_pool_fields_for_asset_class(species_spec.asset_class) {
            if !localized_pool_supports_bulk_field(*field) {
                continue;
            }
            let _ = register_localized_pool_request(
                &mut localized_pool_requests,
                *field,
                species_spec.spatial_scope,
                species_spec.patch_domain,
                species_spec.chromosome_domain.as_deref(),
                signal_seed,
                *weight,
                &species_spec.subsystem_targets,
            );
        }
        let membrane_weight = if species_spec.compartment.eq_ignore_ascii_case("membrane")
            || matches!(
                species_spec.spatial_scope,
                WholeCellSpatialScope::MembraneAdjacent | WholeCellSpatialScope::SeptumLocal
            )
            || matches!(
                species_spec.patch_domain,
                WholeCellPatchDomain::MembraneBand
                    | WholeCellPatchDomain::SeptumPatch
                    | WholeCellPatchDomain::PolarPatch
            ) {
            match species_spec.asset_class {
                WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction => 1.0,
                _ => 0.45,
            }
        } else {
            0.0
        };
        if membrane_weight > 0.0 {
            let _ = register_localized_pool_request(
                &mut localized_pool_requests,
                WholeCellBulkField::MembranePrecursors,
                species_spec.spatial_scope,
                species_spec.patch_domain,
                species_spec.chromosome_domain.as_deref(),
                signal_seed,
                membrane_weight,
                &species_spec.subsystem_targets,
            );
        }
    }

    for reaction_spec in &reactions {
        if reaction_spec.spatial_scope == WholeCellSpatialScope::WellMixed
            && reaction_spec.patch_domain == WholeCellPatchDomain::Distributed
        {
            continue;
        }
        let signal_seed = reaction_spec.nominal_rate.max(0.01)
            * (reaction_spec
                .reactants
                .len()
                .max(reaction_spec.products.len())
                .max(1) as f32)
                .sqrt();
        for (field, weight) in localized_pool_fields_for_asset_class(reaction_spec.asset_class) {
            if !localized_pool_supports_bulk_field(*field) {
                continue;
            }
            let _ = register_localized_pool_request(
                &mut localized_pool_requests,
                *field,
                reaction_spec.spatial_scope,
                reaction_spec.patch_domain,
                reaction_spec.chromosome_domain.as_deref(),
                signal_seed,
                *weight,
                &reaction_spec.subsystem_targets,
            );
        }
        let membrane_weight = if matches!(
            reaction_spec.spatial_scope,
            WholeCellSpatialScope::MembraneAdjacent | WholeCellSpatialScope::SeptumLocal
        ) || matches!(
            reaction_spec.patch_domain,
            WholeCellPatchDomain::MembraneBand
                | WholeCellPatchDomain::SeptumPatch
                | WholeCellPatchDomain::PolarPatch
        ) {
            match reaction_spec.asset_class {
                WholeCellAssetClass::Membrane | WholeCellAssetClass::Constriction => 1.0,
                _ => 0.45,
            }
        } else {
            0.0
        };
        if membrane_weight > 0.0 {
            let _ = register_localized_pool_request(
                &mut localized_pool_requests,
                WholeCellBulkField::MembranePrecursors,
                reaction_spec.spatial_scope,
                reaction_spec.patch_domain,
                reaction_spec.chromosome_domain.as_deref(),
                signal_seed,
                membrane_weight,
                &reaction_spec.subsystem_targets,
            );
        }
    }

    let mut localized_pool_keys = localized_pool_requests.keys().cloned().collect::<Vec<_>>();
    localized_pool_keys.sort_by_key(|(field, spatial_scope, patch_domain, chromosome_domain)| {
        (
            match field {
                WholeCellBulkField::ATP => 0usize,
                WholeCellBulkField::ADP => 1usize,
                WholeCellBulkField::Glucose => 2usize,
                WholeCellBulkField::Oxygen => 3usize,
                WholeCellBulkField::AminoAcids => 4usize,
                WholeCellBulkField::Nucleotides => 5usize,
                WholeCellBulkField::MembranePrecursors => 6usize,
            },
            match spatial_scope {
                WholeCellSpatialScope::WellMixed => 0usize,
                WholeCellSpatialScope::MembraneAdjacent => 1usize,
                WholeCellSpatialScope::SeptumLocal => 2usize,
                WholeCellSpatialScope::NucleoidLocal => 3usize,
            },
            match patch_domain {
                WholeCellPatchDomain::Distributed => 0usize,
                WholeCellPatchDomain::MembraneBand => 1usize,
                WholeCellPatchDomain::SeptumPatch => 2usize,
                WholeCellPatchDomain::PolarPatch => 3usize,
                WholeCellPatchDomain::NucleoidTrack => 4usize,
            },
            chromosome_domain.clone().unwrap_or_default(),
        )
    });
    for (field, spatial_scope, patch_domain, chromosome_domain) in localized_pool_keys {
        let Some((global_species_id, global_pool)) = global_pools_by_field.get(&field) else {
            continue;
        };
        let Some(request) = localized_pool_requests.get(&(
            field,
            spatial_scope,
            patch_domain,
            chromosome_domain.clone(),
        )) else {
            continue;
        };
        let species_id = localized_pool_species_id(
            field,
            spatial_scope,
            patch_domain,
            chromosome_domain.as_deref(),
        );
        let species_name = if let Some(domain_id) = chromosome_domain.as_deref() {
            format!(
                "{} {} {} pool",
                domain_id,
                localized_pool_locality_display_name(spatial_scope, patch_domain),
                bulk_field_display_name(field)
            )
        } else {
            format!(
                "{} {} pool",
                localized_pool_locality_display_name(spatial_scope, patch_domain),
                bulk_field_display_name(field)
            )
        };
        let target_signal = request.signal.max(0.5);
        let basal_abundance = (global_pool.count.max(0.0)
            + 24.0 * global_pool.concentration_mm.max(0.0))
            * (localized_pool_basal_scale(field) + 0.016 * target_signal.sqrt());
        species.push(WholeCellSpeciesSpec {
            id: species_id.clone(),
            name: species_name.clone(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: match patch_domain {
                WholeCellPatchDomain::MembraneBand
                | WholeCellPatchDomain::SeptumPatch
                | WholeCellPatchDomain::PolarPatch => "membrane".to_string(),
                WholeCellPatchDomain::NucleoidTrack => "chromosome".to_string(),
                WholeCellPatchDomain::Distributed => match spatial_scope {
                    WholeCellSpatialScope::MembraneAdjacent
                    | WholeCellSpatialScope::SeptumLocal => "membrane".to_string(),
                    WholeCellSpatialScope::NucleoidLocal => "chromosome".to_string(),
                    WholeCellSpatialScope::WellMixed => "cytosol".to_string(),
                },
            },
            asset_class: transport_asset_class_for_bulk_field(field),
            basal_abundance: basal_abundance.clamp(0.0, 1024.0),
            bulk_field: Some(field),
            operon: None,
            parent_complex: None,
            subsystem_targets: request.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: chromosome_domain.clone(),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!(
                "{}_localized_transfer",
                canonical_species_fragment(&species_id)
            ),
            name: format!("{} localized transfer", species_name),
            reaction_class: WholeCellReactionClass::LocalizedPoolTransfer,
            asset_class: transport_asset_class_for_bulk_field(field),
            nominal_rate: (localized_pool_transfer_rate(field) + 0.010 * target_signal.sqrt())
                .clamp(0.01, 4.0),
            catalyst: None,
            operon: None,
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: global_species_id.clone(),
                stoichiometry: 1.0,
            }],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: request.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: chromosome_domain.clone(),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!(
                "{}_localized_turnover",
                canonical_species_fragment(&species_id)
            ),
            name: format!("{} localized turnover", species_name),
            reaction_class: WholeCellReactionClass::LocalizedPoolTurnover,
            asset_class: transport_asset_class_for_bulk_field(field),
            nominal_rate: (localized_pool_turnover_rate(field) + 0.006 * target_signal.sqrt())
                .clamp(0.004, 2.0),
            catalyst: None,
            operon: None,
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: species_id,
                stoichiometry: 1.0,
            }],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: global_species_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: request.subsystem_targets.clone(),
            spatial_scope,
            patch_domain,
            chromosome_domain: chromosome_domain.clone(),
        });
    }

    WholeCellGenomeProcessRegistry {
        organism: package.organism.clone(),
        chromosome_domains: package.chromosome_domains.clone(),
        species,
        reactions,
    }
}

fn resolve_manifest_relative_path(
    manifest_path: &Path,
    relative_path: &str,
) -> Result<PathBuf, String> {
    let trimmed = relative_path.trim();
    if trimmed.is_empty() {
        return Err("bundle manifest contained an empty relative path".to_string());
    }
    Ok(manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(trimmed)
        .canonicalize()
        .unwrap_or_else(|_| {
            manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(trimmed)
        }))
}

fn read_text_file(path: &Path, label: &str) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("failed to read {label} {}: {error}", path.display()))
}

fn parse_fasta_sequence_length(path: &Path) -> Result<u32, String> {
    let contents = read_text_file(path, "FASTA file")?;
    let mut length = 0usize;
    let mut saw_header = false;
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('>') {
            saw_header = true;
            continue;
        }
        length += line.len();
    }
    if !saw_header || length == 0 {
        return Err(format!("invalid FASTA file: {}", path.display()));
    }
    Ok(length.min(u32::MAX as usize) as u32)
}

fn parse_gff_attributes(attributes: &str) -> HashMap<String, String> {
    let mut parsed = HashMap::new();
    for entry in attributes.split(';') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = if let Some((key, value)) = trimmed.split_once('=') {
            (key.trim(), value.trim())
        } else if let Some((key, value)) = trimmed.split_once(' ') {
            (key.trim(), value.trim())
        } else {
            (trimmed, "")
        };
        parsed.insert(key.to_string(), value.to_string());
    }
    parsed
}

fn parse_gff_gene_features(path: &Path) -> Result<Vec<WholeCellGenomeFeature>, String> {
    let contents = read_text_file(path, "GFF3 file")?;
    let mut genes = Vec::new();
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 9 {
            return Err(format!("invalid GFF3 row in {}: {}", path.display(), line));
        }
        let feature_type = parts[2].trim().to_ascii_lowercase();
        if feature_type != "gene" && feature_type != "cds" {
            continue;
        }
        let attributes = parse_gff_attributes(parts[8]);
        let gene = attributes
            .get("gene")
            .or_else(|| attributes.get("Name"))
            .or_else(|| attributes.get("ID"))
            .cloned()
            .ok_or_else(|| format!("missing gene identifier in {}", path.display()))?;
        genes.push(WholeCellGenomeFeature {
            gene,
            start_bp: parts[3].parse::<u32>().map_err(|error| {
                format!(
                    "invalid GFF start coordinate in {}: {error}",
                    path.display()
                )
            })?,
            end_bp: parts[4].parse::<u32>().map_err(|error| {
                format!("invalid GFF end coordinate in {}: {error}", path.display())
            })?,
            strand: if parts[6].trim() == "-" { -1 } else { 1 },
            essential: false,
            basal_expression: default_expression_level(),
            translation_cost: default_translation_cost(),
            nucleotide_cost: default_nucleotide_cost(),
            process_weights: WholeCellProcessWeights::default(),
            subsystem_targets: Vec::new(),
            asset_class: None,
            complex_family: None,
        });
    }
    Ok(genes)
}

fn merge_gene_product_annotations(
    genes: &mut [WholeCellGenomeFeature],
    annotations: &[WholeCellGeneProductAnnotation],
) {
    let annotation_map: HashMap<&str, &WholeCellGeneProductAnnotation> = annotations
        .iter()
        .map(|annotation| (annotation.gene.as_str(), annotation))
        .collect();
    for gene in genes {
        if let Some(annotation) = annotation_map.get(gene.gene.as_str()) {
            gene.essential = annotation.essential;
            gene.basal_expression = annotation.basal_expression;
            gene.translation_cost = annotation.translation_cost;
            gene.nucleotide_cost = annotation.nucleotide_cost;
            gene.process_weights = annotation.process_weights.clamped();
            gene.subsystem_targets = annotation.subsystem_targets.clone();
            gene.asset_class = annotation.asset_class;
            gene.complex_family = annotation.complex_family;
        }
    }
}

fn merge_gene_semantic_annotations(
    genes: &mut [WholeCellGenomeFeature],
    annotations: &[WholeCellGeneSemanticAnnotation],
) {
    let annotation_map: HashMap<&str, &WholeCellGeneSemanticAnnotation> = annotations
        .iter()
        .map(|annotation| (annotation.gene.as_str(), annotation))
        .collect();
    for gene in genes {
        if let Some(annotation) = annotation_map.get(gene.gene.as_str()) {
            gene.subsystem_targets = annotation.subsystem_targets.clone();
            if let Some(asset_class) = annotation.asset_class {
                gene.asset_class = Some(asset_class);
            }
            if let Some(complex_family) = annotation.complex_family {
                gene.complex_family = Some(complex_family);
            }
        }
    }
}

fn merge_transcription_unit_semantic_annotations(
    units: &mut [WholeCellTranscriptionUnitSpec],
    annotations: &[WholeCellTranscriptionUnitSemanticAnnotation],
) {
    let annotation_map: HashMap<&str, &WholeCellTranscriptionUnitSemanticAnnotation> = annotations
        .iter()
        .map(|annotation| (annotation.name.as_str(), annotation))
        .collect();
    for unit in units {
        if let Some(annotation) = annotation_map.get(unit.name.as_str()) {
            unit.subsystem_targets = annotation.subsystem_targets.clone();
            if let Some(asset_class) = annotation.asset_class {
                unit.asset_class = Some(asset_class);
            }
            if let Some(complex_family) = annotation.complex_family {
                unit.complex_family = Some(complex_family);
            }
        }
    }
}

fn validate_explicit_gene_semantics(genes: &[WholeCellGenomeFeature]) -> Result<(), String> {
    let missing: Vec<String> = genes
        .iter()
        .filter(|gene| {
            gene.asset_class.is_none()
                || gene.complex_family.is_none()
                || gene.subsystem_targets.is_empty()
        })
        .map(|gene| gene.gene.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit gene semantics but {} gene(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_transcription_unit_semantics(
    units: &[WholeCellTranscriptionUnitSpec],
) -> Result<(), String> {
    let missing: Vec<String> = units
        .iter()
        .filter(|unit| {
            unit.asset_class.is_none()
                || unit.complex_family.is_none()
                || unit.subsystem_targets.is_empty()
        })
        .map(|unit| unit.name.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit transcription unit semantics but {} unit(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_pool_metadata(pools: &[WholeCellMoleculePoolSpec]) -> Result<(), String> {
    let missing: Vec<String> = pools
        .iter()
        .filter(|pool| pool.bulk_field.is_none())
        .map(|pool| pool.species.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit pool metadata but {} pool(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_operon_assets(operons: &[WholeCellOperonSpec]) -> Result<(), String> {
    let missing: Vec<String> = operons
        .iter()
        .filter(|operon| {
            operon.asset_class.is_none()
                || operon.complex_family.is_none()
                || operon.subsystem_targets.is_empty()
        })
        .map(|operon| operon.name.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset entities but {} operon(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_protein_assets(
    proteins: &[WholeCellProteinProductSpec],
) -> Result<(), String> {
    let missing: Vec<String> = proteins
        .iter()
        .filter(|protein| protein.subsystem_targets.is_empty())
        .map(|protein| protein.id.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset entities but {} protein(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_complex_assets(complexes: &[WholeCellComplexSpec]) -> Result<(), String> {
    let missing: Vec<String> = complexes
        .iter()
        .filter(|complex| complex.subsystem_targets.is_empty())
        .map(|complex| complex.id.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset entities but {} complex(es) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_asset_entities(assets: &WholeCellGenomeAssetPackage) -> Result<(), String> {
    validate_explicit_operon_assets(&assets.operons)?;
    validate_explicit_protein_assets(&assets.proteins)?;
    validate_explicit_complex_assets(&assets.complexes)
}

fn validate_explicit_asset_entity_coverage(
    organism: &WholeCellOrganismSpec,
    assets: &WholeCellGenomeAssetPackage,
) -> Result<(), String> {
    let genes_in_units: HashMap<String, ()> = organism
        .transcription_units
        .iter()
        .flat_map(|unit| unit.genes.iter().cloned())
        .map(|gene| (gene, ()))
        .collect();
    let expected_operons: HashMap<String, ()> = organism
        .transcription_units
        .iter()
        .map(|unit| (unit.name.clone(), ()))
        .chain(
            organism
                .genes
                .iter()
                .filter(|gene| !genes_in_units.contains_key(&gene.gene))
                .map(|gene| (gene.gene.clone(), ())),
        )
        .collect();
    let operon_names: HashMap<String, ()> = assets
        .operons
        .iter()
        .map(|operon| (operon.name.clone(), ()))
        .collect();
    let missing_operons: Vec<String> = expected_operons
        .keys()
        .filter(|name| !operon_names.contains_key(*name))
        .cloned()
        .collect();
    if !missing_operons.is_empty() {
        return Err(format!(
            "bundle requires explicit asset entity coverage but {} operon(s) are missing: {}",
            missing_operons.len(),
            missing_operons.join(", ")
        ));
    }

    let rna_genes: HashMap<String, ()> = assets
        .rnas
        .iter()
        .map(|rna| (rna.gene.clone(), ()))
        .collect();
    let missing_rnas: Vec<String> = organism
        .genes
        .iter()
        .filter(|gene| !rna_genes.contains_key(&gene.gene))
        .map(|gene| gene.gene.clone())
        .collect();
    if !missing_rnas.is_empty() {
        return Err(format!(
            "bundle requires explicit asset entity coverage but {} RNA gene(s) are missing: {}",
            missing_rnas.len(),
            missing_rnas.join(", ")
        ));
    }

    let protein_genes: HashMap<String, ()> = assets
        .proteins
        .iter()
        .map(|protein| (protein.gene.clone(), ()))
        .collect();
    let missing_proteins: Vec<String> = organism
        .genes
        .iter()
        .filter(|gene| !protein_genes.contains_key(&gene.gene))
        .map(|gene| gene.gene.clone())
        .collect();
    if !missing_proteins.is_empty() {
        return Err(format!(
            "bundle requires explicit asset entity coverage but {} protein gene(s) are missing: {}",
            missing_proteins.len(),
            missing_proteins.join(", ")
        ));
    }

    let complex_operons: HashMap<String, ()> = assets
        .complexes
        .iter()
        .map(|complex| (complex.operon.clone(), ()))
        .collect();
    let missing_complexes: Vec<String> = expected_operons
        .keys()
        .filter(|name| !complex_operons.contains_key(*name))
        .cloned()
        .collect();
    if !missing_complexes.is_empty() {
        return Err(format!(
            "bundle requires explicit asset entity coverage but {} complex operon(s) are missing: {}",
            missing_complexes.len(),
            missing_complexes.join(", ")
        ));
    }

    Ok(())
}

fn validate_explicit_operon_semantics(assets: &WholeCellGenomeAssetPackage) -> Result<(), String> {
    let semantics: HashMap<&str, &WholeCellOperonSemanticSpec> = assets
        .operon_semantics
        .iter()
        .map(|semantic| (semantic.name.as_str(), semantic))
        .collect();
    let missing: Vec<String> = assets
        .operons
        .iter()
        .filter(|operon| {
            semantics
                .get(operon.name.as_str())
                .map_or(true, |semantic| semantic.subsystem_targets.is_empty())
        })
        .map(|operon| operon.name.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset semantics but {} operon semantic entry(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_protein_semantics(assets: &WholeCellGenomeAssetPackage) -> Result<(), String> {
    let semantics: HashMap<&str, &WholeCellProteinSemanticSpec> = assets
        .protein_semantics
        .iter()
        .map(|semantic| (semantic.id.as_str(), semantic))
        .collect();
    let missing: Vec<String> = assets
        .proteins
        .iter()
        .filter(|protein| {
            semantics
                .get(protein.id.as_str())
                .map_or(true, |semantic| semantic.subsystem_targets.is_empty())
        })
        .map(|protein| protein.id.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset semantics but {} protein semantic entry(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_complex_semantics(assets: &WholeCellGenomeAssetPackage) -> Result<(), String> {
    let semantics: HashMap<&str, &WholeCellComplexSemanticSpec> = assets
        .complex_semantics
        .iter()
        .map(|semantic| (semantic.id.as_str(), semantic))
        .collect();
    let missing: Vec<String> = assets
        .complexes
        .iter()
        .filter(|complex| {
            semantics
                .get(complex.id.as_str())
                .map_or(true, |semantic| semantic.subsystem_targets.is_empty())
        })
        .map(|complex| complex.id.clone())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle requires explicit asset semantics but {} complex semantic entry(s) are incomplete: {}",
            missing.len(),
            missing.join(", ")
        ))
    }
}

fn validate_explicit_asset_semantics(assets: &WholeCellGenomeAssetPackage) -> Result<(), String> {
    validate_explicit_operon_semantics(assets)?;
    validate_explicit_protein_semantics(assets)?;
    validate_explicit_complex_semantics(assets)
}

fn merge_explicit_asset_semantics_into_entities(assets: &mut WholeCellGenomeAssetPackage) {
    let operon_semantics: HashMap<String, WholeCellOperonSemanticSpec> = assets
        .operon_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.name.clone(), semantic))
        .collect();
    for operon in &mut assets.operons {
        if let Some(semantic) = operon_semantics.get(&operon.name) {
            operon.asset_class = Some(semantic.asset_class);
            operon.complex_family = Some(semantic.complex_family);
            if operon.subsystem_targets.is_empty() {
                operon.subsystem_targets = semantic.subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(
                    &mut operon.subsystem_targets,
                    &semantic.subsystem_targets,
                );
            }
        }
    }

    let protein_semantics: HashMap<String, WholeCellProteinSemanticSpec> = assets
        .protein_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.id.clone(), semantic))
        .collect();
    for protein in &mut assets.proteins {
        if let Some(semantic) = protein_semantics.get(&protein.id) {
            protein.asset_class = semantic.asset_class;
            if protein.subsystem_targets.is_empty() {
                protein.subsystem_targets = semantic.subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(
                    &mut protein.subsystem_targets,
                    &semantic.subsystem_targets,
                );
            }
        }
    }

    let complex_semantics: HashMap<String, WholeCellComplexSemanticSpec> = assets
        .complex_semantics
        .iter()
        .cloned()
        .map(|semantic| (semantic.id.clone(), semantic))
        .collect();
    for complex in &mut assets.complexes {
        if let Some(semantic) = complex_semantics.get(&complex.id) {
            complex.asset_class = semantic.asset_class;
            complex.family = semantic.family;
            complex.membrane_inserted = semantic.membrane_inserted;
            complex.chromosome_coupled = semantic.chromosome_coupled;
            complex.division_coupled = semantic.division_coupled;
            if complex.subsystem_targets.is_empty() {
                complex.subsystem_targets = semantic.subsystem_targets.clone();
            } else {
                push_unique_subsystem_targets(
                    &mut complex.subsystem_targets,
                    &semantic.subsystem_targets,
                );
            }
        }
    }
}

fn finalize_bundle_asset_package(
    organism: &WholeCellOrganismSpec,
    manifest: &WholeCellOrganismBundleManifest,
    mut assets: WholeCellGenomeAssetPackage,
) -> Result<WholeCellGenomeAssetPackage, String> {
    if manifest.require_explicit_asset_entities {
        validate_explicit_asset_entities(&assets)?;
        validate_explicit_asset_entity_coverage(organism, &assets)?;
    }
    if manifest.require_explicit_asset_semantics {
        merge_explicit_asset_semantics_into_entities(&mut assets);
        validate_explicit_asset_semantics(&assets)?;
        return Ok(assets);
    }
    if manifest.require_explicit_asset_entities {
        assets.operon_semantics = compile_operon_semantic_specs(&assets.operons);
        assets.protein_semantics = compile_protein_semantic_specs(&assets.proteins);
        assets.complex_semantics = compile_complex_semantic_specs(&assets.complexes);
        return Ok(assets);
    }
    Ok(with_normalized_asset_semantic_metadata(
        with_normalized_asset_pool_metadata(assets),
    ))
}

fn pool_concentration_for_field(
    pools: &[WholeCellMoleculePoolSpec],
    field: WholeCellBulkField,
    fallback: f32,
) -> f32 {
    pools
        .iter()
        .find(|pool| pool_bulk_field(pool) == Some(field))
        .map(|pool| pool.concentration_mm.max(0.0))
        .unwrap_or(fallback)
}

fn build_program_spec_from_organism(
    organism: WholeCellOrganismSpec,
    source_dataset: Option<String>,
) -> Result<WholeCellProgramSpec, String> {
    let assets = compile_genome_asset_package(&organism);
    let process_registry = compile_genome_process_registry(&assets);
    let mut spec = WholeCellProgramSpec {
        program_name: Some(format!(
            "{}_bundle_native",
            organism
                .organism
                .to_ascii_lowercase()
                .replace([' ', '-'], "_")
        )),
        contract: WholeCellContractSchema::default(),
        provenance: WholeCellProvenance {
            source_dataset,
            backend: Some("rust_bundle_compiler".to_string()),
            notes: vec!["compiled from whole-cell organism bundle manifest".to_string()],
            ..WholeCellProvenance::default()
        },
        organism_data_ref: None,
        organism_data: Some(organism.clone()),
        organism_assets: Some(assets),
        organism_process_registry: Some(process_registry),
        chromosome_state: None,
        membrane_division_state: None,
        config: WholeCellConfig::default(),
        initial_lattice: WholeCellInitialLatticeSpec {
            atp: pool_concentration_for_field(&organism.pools, WholeCellBulkField::ATP, 1.2),
            amino_acids: pool_concentration_for_field(
                &organism.pools,
                WholeCellBulkField::AminoAcids,
                0.95,
            ),
            nucleotides: pool_concentration_for_field(
                &organism.pools,
                WholeCellBulkField::Nucleotides,
                0.80,
            ),
            membrane_precursors: pool_concentration_for_field(
                &organism.pools,
                WholeCellBulkField::MembranePrecursors,
                0.35,
            ),
        },
        initial_state: WholeCellInitialStateSpec {
            adp_mm: pool_concentration_for_field(&organism.pools, WholeCellBulkField::ADP, 0.2),
            glucose_mm: pool_concentration_for_field(
                &organism.pools,
                WholeCellBulkField::Glucose,
                1.0,
            ),
            oxygen_mm: pool_concentration_for_field(
                &organism.pools,
                WholeCellBulkField::Oxygen,
                0.85,
            ),
            genome_bp: organism.chromosome_length_bp.max(1),
            replicated_bp: 0,
            chromosome_separation_nm: (organism.geometry.radius_nm
                * organism.geometry.chromosome_radius_fraction.max(0.1))
            .max(10.0),
            radius_nm: organism.geometry.radius_nm.max(50.0),
            division_progress: 0.0,
            metabolic_load: 1.0,
        },
        quantum_profile: WholeCellQuantumProfile::default(),
        local_chemistry: None,
    };
    finalize_parsed_program_spec(&mut spec)?;
    Ok(spec)
}

fn parse_bundle_manifest_json(
    manifest_json: &str,
) -> Result<WholeCellOrganismBundleManifest, String> {
    serde_json::from_str(manifest_json)
        .map_err(|error| format!("failed to parse organism bundle manifest: {error}"))
}

fn validate_bundle_manifest_mode(manifest: &WholeCellOrganismBundleManifest) -> Result<(), String> {
    if manifest.organism_spec_json.is_some() {
        return Err(
            "bundle manifests may not define organism_spec_json; use explicit structured bundle sources"
                .to_string(),
        );
    }
    if manifest.require_explicit_organism_sources {
        let mut missing = Vec::new();
        if manifest.metadata_json.is_none() {
            missing.push("metadata_json");
        }
        if manifest.gene_features_json.is_none() && manifest.gene_features_gff.is_none() {
            missing.push("gene_features_json|gene_features_gff");
        }
        if manifest.gene_products_json.is_none() {
            missing.push("gene_products_json");
        }
        if manifest.transcription_units_json.is_none() {
            missing.push("transcription_units_json");
        }
        if manifest.chromosome_domains_json.is_none() {
            missing.push("chromosome_domains_json");
        }
        if manifest.pools_json.is_none() {
            missing.push("pools_json");
        }
        if !missing.is_empty() {
            return Err(format!(
                "bundle requires explicit organism sources but is missing {}",
                missing.join(", ")
            ));
        }
    }
    Ok(())
}

fn validate_bundle_asset_contracts(
    manifest: &WholeCellOrganismBundleManifest,
) -> Result<(), String> {
    if manifest.require_explicit_asset_entities {
        let mut missing = Vec::new();
        if manifest.operons_json.is_none() {
            missing.push("operons_json");
        }
        if manifest.rnas_json.is_none() {
            missing.push("rnas_json");
        }
        if manifest.proteins_json.is_none() {
            missing.push("proteins_json");
        }
        if manifest.complexes_json.is_none() {
            missing.push("complexes_json");
        }
        if !missing.is_empty() {
            return Err(format!(
                "bundle requires explicit asset entities but is missing {}",
                missing.join(", ")
            ));
        }
    }
    if manifest.require_explicit_asset_semantics {
        let mut missing = Vec::new();
        if manifest.operon_semantics_json.is_none() {
            missing.push("operon_semantics_json");
        }
        if manifest.protein_semantics_json.is_none() {
            missing.push("protein_semantics_json");
        }
        if manifest.complex_semantics_json.is_none() {
            missing.push("complex_semantics_json");
        }
        if !missing.is_empty() {
            return Err(format!(
                "bundle requires explicit asset semantics but is missing {}",
                missing.join(", ")
            ));
        }
    }
    if manifest.require_explicit_program_defaults && manifest.program_defaults_json.is_none() {
        return Err(
            "bundle requires explicit program defaults but is missing program_defaults_json"
                .to_string(),
        );
    }
    Ok(())
}

fn compile_embedded_syn3a_organism_spec() -> Result<WholeCellOrganismSpec, String> {
    let manifest = parse_bundle_manifest_json(BUNDLED_SYN3A_BUNDLE_MANIFEST_JSON)?;
    validate_bundle_manifest_mode(&manifest)?;
    let metadata: WholeCellOrganismBundleMetadata =
        serde_json::from_str(BUNDLED_SYN3A_BUNDLE_METADATA_JSON)
            .map_err(|error| format!("failed to parse embedded Syn3A bundle metadata: {error}"))?;
    let chromosome_length_bp = metadata
        .chromosome_length_bp
        .ok_or_else(|| "embedded Syn3A bundle metadata missing chromosome_length_bp".to_string())?;

    let mut genes = serde_json::from_str::<Vec<WholeCellGenomeFeature>>(
        BUNDLED_SYN3A_BUNDLE_GENE_FEATURES_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A gene features: {error}"))?;
    let gene_products = serde_json::from_str::<Vec<WholeCellGeneProductAnnotation>>(
        BUNDLED_SYN3A_BUNDLE_GENE_PRODUCTS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A gene products: {error}"))?;
    merge_gene_product_annotations(&mut genes, &gene_products);
    let gene_semantics = serde_json::from_str::<Vec<WholeCellGeneSemanticAnnotation>>(
        BUNDLED_SYN3A_BUNDLE_GENE_SEMANTICS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A gene semantics: {error}"))?;
    merge_gene_semantic_annotations(&mut genes, &gene_semantics);
    if manifest.require_explicit_gene_semantics {
        validate_explicit_gene_semantics(&genes)?;
    }

    let mut transcription_units = serde_json::from_str::<Vec<WholeCellTranscriptionUnitSpec>>(
        BUNDLED_SYN3A_BUNDLE_TRANSCRIPTION_UNITS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A transcription units: {error}"))?;
    let unit_semantics = serde_json::from_str::<Vec<WholeCellTranscriptionUnitSemanticAnnotation>>(
        BUNDLED_SYN3A_BUNDLE_TRANSCRIPTION_UNIT_SEMANTICS_JSON,
    )
    .map_err(|error| {
        format!("failed to parse embedded Syn3A transcription unit semantics: {error}")
    })?;
    merge_transcription_unit_semantic_annotations(&mut transcription_units, &unit_semantics);
    if manifest.require_explicit_transcription_unit_semantics {
        validate_explicit_transcription_unit_semantics(&transcription_units)?;
    }

    let mut pools =
        serde_json::from_str::<Vec<WholeCellMoleculePoolSpec>>(BUNDLED_SYN3A_BUNDLE_POOLS_JSON)
            .map_err(|error| format!("failed to parse embedded Syn3A pools: {error}"))?;
    if manifest.require_explicit_organism_sources {
        validate_explicit_pool_metadata(&pools)?;
    } else {
        normalize_pool_metadata(&mut pools);
    }

    let chromosome_domains = serde_json::from_str::<Vec<WholeCellChromosomeDomainSpec>>(
        BUNDLED_SYN3A_BUNDLE_CHROMOSOME_DOMAINS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A chromosome domains: {error}"))?;

    let spec = WholeCellOrganismSpec {
        organism: manifest
            .organism
            .clone()
            .unwrap_or_else(|| metadata.organism.clone()),
        chromosome_length_bp: chromosome_length_bp.max(1),
        origin_bp: metadata.origin_bp.min(chromosome_length_bp.max(1)),
        terminus_bp: metadata.terminus_bp.min(chromosome_length_bp.max(1)),
        geometry: metadata.geometry,
        composition: metadata.composition,
        chromosome_domains,
        pools,
        genes,
        transcription_units,
    };

    if manifest.require_explicit_organism_sources
        || manifest.require_explicit_gene_semantics
        || manifest.require_explicit_transcription_unit_semantics
    {
        Ok(with_compiled_chromosome_domains(spec))
    } else {
        Ok(with_compiled_chromosome_domains(
            with_normalized_semantic_metadata(with_normalized_pool_metadata(spec)),
        ))
    }
}

pub fn compile_organism_spec_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<WholeCellOrganismSpec, String> {
    let manifest_path = Path::new(manifest_path).canonicalize().map_err(|error| {
        format!("failed to resolve bundle manifest path {manifest_path}: {error}")
    })?;
    let manifest_json = read_text_file(&manifest_path, "bundle manifest")?;
    let manifest = parse_bundle_manifest_json(&manifest_json)?;
    validate_bundle_manifest_mode(&manifest)?;

    let metadata_relative = manifest
        .metadata_json
        .as_deref()
        .ok_or_else(|| "bundle manifest missing metadata_json".to_string())?;
    let metadata_path = resolve_manifest_relative_path(&manifest_path, metadata_relative)?;
    let metadata: WholeCellOrganismBundleMetadata =
        serde_json::from_str(&read_text_file(&metadata_path, "bundle metadata JSON")?).map_err(
            |error| {
                format!(
                    "failed to parse bundle metadata {}: {error}",
                    metadata_path.display()
                )
            },
        )?;

    let chromosome_length_bp = if let Some(genome_fasta) = manifest.genome_fasta.as_deref() {
        let fasta_path = resolve_manifest_relative_path(&manifest_path, genome_fasta)?;
        parse_fasta_sequence_length(&fasta_path)?
    } else {
        metadata.chromosome_length_bp.ok_or_else(|| {
            "bundle metadata missing chromosome_length_bp and genome_fasta".to_string()
        })?
    };

    let mut genes = if let Some(features_json) = manifest.gene_features_json.as_deref() {
        let features_path = resolve_manifest_relative_path(&manifest_path, features_json)?;
        serde_json::from_str::<Vec<WholeCellGenomeFeature>>(&read_text_file(
            &features_path,
            "gene feature JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse gene features {}: {error}",
                features_path.display()
            )
        })?
    } else if let Some(features_gff) = manifest.gene_features_gff.as_deref() {
        let gff_path = resolve_manifest_relative_path(&manifest_path, features_gff)?;
        parse_gff_gene_features(&gff_path)?
    } else {
        return Err("bundle manifest missing gene feature source".to_string());
    };

    if let Some(gene_products_json) = manifest.gene_products_json.as_deref() {
        let products_path = resolve_manifest_relative_path(&manifest_path, gene_products_json)?;
        let annotations: Vec<WholeCellGeneProductAnnotation> = serde_json::from_str(
            &read_text_file(&products_path, "gene product JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse gene products {}: {error}",
                products_path.display()
            )
        })?;
        merge_gene_product_annotations(&mut genes, &annotations);
    }
    if let Some(gene_semantics_json) = manifest.gene_semantics_json.as_deref() {
        let semantics_path = resolve_manifest_relative_path(&manifest_path, gene_semantics_json)?;
        let annotations: Vec<WholeCellGeneSemanticAnnotation> = serde_json::from_str(
            &read_text_file(&semantics_path, "gene semantic JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse gene semantics {}: {error}",
                semantics_path.display()
            )
        })?;
        merge_gene_semantic_annotations(&mut genes, &annotations);
    }
    if manifest.require_explicit_gene_semantics {
        validate_explicit_gene_semantics(&genes)?;
    }

    let mut transcription_units = if let Some(transcription_units_json) =
        manifest.transcription_units_json.as_deref()
    {
        let units_path = resolve_manifest_relative_path(&manifest_path, transcription_units_json)?;
        serde_json::from_str::<Vec<WholeCellTranscriptionUnitSpec>>(&read_text_file(
            &units_path,
            "transcription unit JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse transcription units {}: {error}",
                units_path.display()
            )
        })?
    } else {
        Vec::new()
    };
    if let Some(unit_semantics_json) = manifest.transcription_unit_semantics_json.as_deref() {
        let semantics_path = resolve_manifest_relative_path(&manifest_path, unit_semantics_json)?;
        let annotations: Vec<WholeCellTranscriptionUnitSemanticAnnotation> = serde_json::from_str(
            &read_text_file(&semantics_path, "transcription unit semantic JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse transcription unit semantics {}: {error}",
                semantics_path.display()
            )
        })?;
        merge_transcription_unit_semantic_annotations(&mut transcription_units, &annotations);
    }
    if manifest.require_explicit_transcription_unit_semantics {
        validate_explicit_transcription_unit_semantics(&transcription_units)?;
    }

    let mut pools = if let Some(pools_json) = manifest.pools_json.as_deref() {
        let pools_path = resolve_manifest_relative_path(&manifest_path, pools_json)?;
        serde_json::from_str::<Vec<WholeCellMoleculePoolSpec>>(&read_text_file(
            &pools_path,
            "pool JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse molecule pools {}: {error}",
                pools_path.display()
            )
        })?
    } else {
        Vec::new()
    };
    if manifest.require_explicit_organism_sources {
        validate_explicit_pool_metadata(&pools)?;
    } else {
        normalize_pool_metadata(&mut pools);
    }

    let chromosome_domains = if let Some(chromosome_domains_json) =
        manifest.chromosome_domains_json.as_deref()
    {
        let domains_path = resolve_manifest_relative_path(&manifest_path, chromosome_domains_json)?;
        serde_json::from_str::<Vec<WholeCellChromosomeDomainSpec>>(&read_text_file(
            &domains_path,
            "chromosome domain JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse chromosome domains {}: {error}",
                domains_path.display()
            )
        })?
    } else {
        Vec::new()
    };

    let spec = WholeCellOrganismSpec {
        organism: manifest
            .organism
            .clone()
            .unwrap_or_else(|| metadata.organism.clone()),
        chromosome_length_bp: chromosome_length_bp.max(1),
        origin_bp: metadata.origin_bp.min(chromosome_length_bp.max(1)),
        terminus_bp: metadata.terminus_bp.min(chromosome_length_bp.max(1)),
        geometry: metadata.geometry,
        composition: metadata.composition,
        chromosome_domains,
        pools,
        genes,
        transcription_units,
    };

    if manifest.require_explicit_organism_sources
        || manifest.require_explicit_gene_semantics
        || manifest.require_explicit_transcription_unit_semantics
    {
        Ok(with_compiled_chromosome_domains(spec))
    } else {
        Ok(with_compiled_chromosome_domains(
            with_normalized_semantic_metadata(with_normalized_pool_metadata(spec)),
        ))
    }
}

fn apply_bundle_asset_semantic_overlays(
    manifest_path: &Path,
    manifest: &WholeCellOrganismBundleManifest,
    mut assets: WholeCellGenomeAssetPackage,
) -> Result<WholeCellGenomeAssetPackage, String> {
    if let Some(operon_semantics_json) = manifest.operon_semantics_json.as_deref() {
        let semantics_path = resolve_manifest_relative_path(manifest_path, operon_semantics_json)?;
        assets.operon_semantics = serde_json::from_str::<Vec<WholeCellOperonSemanticSpec>>(
            &read_text_file(&semantics_path, "operon semantic JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse operon semantics {}: {error}",
                semantics_path.display()
            )
        })?;
    }
    if let Some(protein_semantics_json) = manifest.protein_semantics_json.as_deref() {
        let semantics_path = resolve_manifest_relative_path(manifest_path, protein_semantics_json)?;
        assets.protein_semantics = serde_json::from_str::<Vec<WholeCellProteinSemanticSpec>>(
            &read_text_file(&semantics_path, "protein semantic JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse protein semantics {}: {error}",
                semantics_path.display()
            )
        })?;
    }
    if let Some(complex_semantics_json) = manifest.complex_semantics_json.as_deref() {
        let semantics_path = resolve_manifest_relative_path(manifest_path, complex_semantics_json)?;
        assets.complex_semantics = serde_json::from_str::<Vec<WholeCellComplexSemanticSpec>>(
            &read_text_file(&semantics_path, "complex semantic JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse complex semantics {}: {error}",
                semantics_path.display()
            )
        })?;
    }
    Ok(assets)
}

fn apply_bundle_asset_entity_overlays(
    manifest_path: &Path,
    manifest: &WholeCellOrganismBundleManifest,
    mut assets: WholeCellGenomeAssetPackage,
) -> Result<WholeCellGenomeAssetPackage, String> {
    let mut entity_overrides = false;
    if let Some(operons_json) = manifest.operons_json.as_deref() {
        let operons_path = resolve_manifest_relative_path(manifest_path, operons_json)?;
        assets.operons = serde_json::from_str::<Vec<WholeCellOperonSpec>>(&read_text_file(
            &operons_path,
            "operon JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse operons {}: {error}",
                operons_path.display()
            )
        })?;
        entity_overrides = true;
    }
    if let Some(rnas_json) = manifest.rnas_json.as_deref() {
        let rnas_path = resolve_manifest_relative_path(manifest_path, rnas_json)?;
        assets.rnas = serde_json::from_str::<Vec<WholeCellRnaProductSpec>>(&read_text_file(
            &rnas_path, "RNA JSON",
        )?)
        .map_err(|error| format!("failed to parse RNAs {}: {error}", rnas_path.display()))?;
        entity_overrides = true;
    }
    if let Some(proteins_json) = manifest.proteins_json.as_deref() {
        let proteins_path = resolve_manifest_relative_path(manifest_path, proteins_json)?;
        assets.proteins = serde_json::from_str::<Vec<WholeCellProteinProductSpec>>(
            &read_text_file(&proteins_path, "protein JSON")?,
        )
        .map_err(|error| {
            format!(
                "failed to parse proteins {}: {error}",
                proteins_path.display()
            )
        })?;
        entity_overrides = true;
    }
    if let Some(complexes_json) = manifest.complexes_json.as_deref() {
        let complexes_path = resolve_manifest_relative_path(manifest_path, complexes_json)?;
        assets.complexes = serde_json::from_str::<Vec<WholeCellComplexSpec>>(&read_text_file(
            &complexes_path,
            "complex JSON",
        )?)
        .map_err(|error| {
            format!(
                "failed to parse complexes {}: {error}",
                complexes_path.display()
            )
        })?;
        entity_overrides = true;
    }
    if entity_overrides {
        assets.operon_semantics.clear();
        assets.protein_semantics.clear();
        assets.complex_semantics.clear();
    }
    Ok(assets)
}

fn compile_genome_asset_package_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    let manifest_path_obj = Path::new(manifest_path).canonicalize().map_err(|error| {
        format!("failed to resolve bundle manifest path {manifest_path}: {error}")
    })?;
    let manifest_json = read_text_file(&manifest_path_obj, "bundle manifest")?;
    let manifest = parse_bundle_manifest_json(&manifest_json)?;
    validate_bundle_asset_contracts(&manifest)?;
    let organism = compile_organism_spec_from_bundle_manifest_path(manifest_path)?;
    let assets = if manifest.require_explicit_asset_entities {
        empty_genome_asset_package(&organism)
    } else {
        compile_genome_asset_package(&organism)
    };
    let assets = apply_bundle_asset_entity_overlays(&manifest_path_obj, &manifest, assets)?;
    let assets = apply_bundle_asset_semantic_overlays(&manifest_path_obj, &manifest, assets)?;
    finalize_bundle_asset_package(&organism, &manifest, assets)
}

fn apply_bundle_program_defaults(
    manifest_path: &Path,
    manifest: &WholeCellOrganismBundleManifest,
    spec: &mut WholeCellProgramSpec,
) -> Result<(), String> {
    let Some(program_defaults_json) = manifest.program_defaults_json.as_deref() else {
        return Ok(());
    };
    let defaults_path = resolve_manifest_relative_path(manifest_path, program_defaults_json)?;
    let defaults = serde_json::from_str::<WholeCellProgramDefaultsSpec>(&read_text_file(
        &defaults_path,
        "program defaults JSON",
    )?)
    .map_err(|error| {
        format!(
            "failed to parse program defaults {}: {error}",
            defaults_path.display()
        )
    })?;
    if defaults.program_name.is_some() {
        spec.program_name = defaults.program_name;
    }
    if let Some(config) = defaults.config {
        spec.config = config;
    }
    if let Some(initial_lattice) = defaults.initial_lattice {
        spec.initial_lattice = initial_lattice;
    }
    if let Some(initial_state) = defaults.initial_state {
        spec.initial_state = initial_state;
    }
    if let Some(quantum_profile) = defaults.quantum_profile {
        spec.quantum_profile = quantum_profile;
    }
    if defaults.local_chemistry.is_some() {
        spec.local_chemistry = defaults.local_chemistry;
    }
    Ok(())
}

pub fn compile_program_spec_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<WholeCellProgramSpec, String> {
    let manifest_path_obj = Path::new(manifest_path).canonicalize().map_err(|error| {
        format!("failed to resolve bundle manifest path {manifest_path}: {error}")
    })?;
    let manifest_json = read_text_file(&manifest_path_obj, "bundle manifest")?;
    let manifest = parse_bundle_manifest_json(&manifest_json)?;
    let organism = compile_organism_spec_from_bundle_manifest_path(manifest_path)?;
    let assets = compile_genome_asset_package_from_bundle_manifest_path(manifest_path)?;
    let registry = compile_genome_process_registry(&assets);
    let mut spec = build_program_spec_from_organism(organism, manifest.source_dataset.clone())?;
    spec.organism_assets = Some(assets);
    spec.organism_process_registry = Some(registry);
    apply_bundle_program_defaults(&manifest_path_obj, &manifest, &mut spec)?;
    finalize_parsed_program_spec(&mut spec)?;
    Ok(spec)
}

pub fn compile_program_spec_json_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<String, String> {
    let spec = compile_program_spec_from_bundle_manifest_path(manifest_path)?;
    serde_json::to_string_pretty(&spec)
        .map_err(|error| format!("failed to serialize compiled program spec: {error}"))
}

pub fn compile_organism_spec_json_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<String, String> {
    let organism = compile_organism_spec_from_bundle_manifest_path(manifest_path)?;
    serde_json::to_string_pretty(&organism)
        .map_err(|error| format!("failed to serialize compiled organism spec: {error}"))
}

pub fn compile_genome_asset_package_json_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<String, String> {
    let assets = compile_genome_asset_package_from_bundle_manifest_path(manifest_path)?;
    serde_json::to_string_pretty(&assets)
        .map_err(|error| format!("failed to serialize compiled genome asset package: {error}"))
}

pub fn compile_genome_process_registry_json_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<String, String> {
    let assets = compile_genome_asset_package_from_bundle_manifest_path(manifest_path)?;
    let registry = compile_genome_process_registry(&assets);
    serde_json::to_string_pretty(&registry)
        .map_err(|error| format!("failed to serialize compiled genome process registry: {error}"))
}

fn refresh_program_spec_registry_from_assets_if_needed(spec: &mut WholeCellProgramSpec) {
    let refresh_registry = match (
        spec.organism_assets.as_ref(),
        spec.organism_process_registry.as_ref(),
    ) {
        (Some(assets), Some(registry)) if !assets.chromosome_domains.is_empty() => {
            registry.chromosome_domains.is_empty()
                || (registry
                    .species
                    .iter()
                    .filter(|species| species.operon.is_some())
                    .all(|species| species.chromosome_domain.is_none())
                    && registry
                        .reactions
                        .iter()
                        .filter(|reaction| reaction.operon.is_some())
                        .all(|reaction| reaction.chromosome_domain.is_none()))
        }
        (_, None) => true,
        _ => false,
    };
    if refresh_registry {
        if let Some(assets) = spec.organism_assets.as_ref() {
            spec.organism_process_registry = Some(compile_genome_process_registry(assets));
        }
    }
}

fn finalize_parsed_program_spec(spec: &mut WholeCellProgramSpec) -> Result<(), String> {
    if spec.organism_data.is_none() {
        if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_data = Some(resolve_bundled_organism_spec(reference)?);
        }
    }
    if spec.organism_assets.is_none() {
        if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_assets = Some(resolve_bundled_genome_asset_package(reference)?);
        }
    }
    if spec.organism_process_registry.is_none() {
        if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_process_registry =
                Some(resolve_bundled_genome_process_registry(reference)?);
        }
    }
    populate_program_contract_metadata(spec)?;
    Ok(())
}

pub fn parse_program_spec_json(spec_json: &str) -> Result<WholeCellProgramSpec, String> {
    let mut spec: WholeCellProgramSpec = serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse program spec: {error}"))?;
    finalize_parsed_program_spec(&mut spec)?;
    Ok(spec)
}

pub fn parse_legacy_program_spec_json(spec_json: &str) -> Result<WholeCellProgramSpec, String> {
    let mut spec: WholeCellProgramSpec = serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse program spec: {error}"))?;
    if let Some(organism) = spec.organism_data.take() {
        spec.organism_data = Some(with_compiled_chromosome_domains(organism));
    }
    if spec.organism_assets.is_none() {
        if let Some(organism) = spec.organism_data.as_ref() {
            spec.organism_assets = Some(compile_genome_asset_package(organism));
        }
    } else if let (Some(organism), Some(assets)) =
        (spec.organism_data.as_ref(), spec.organism_assets.as_mut())
    {
        if assets.chromosome_domains.is_empty() {
            *assets = compile_genome_asset_package(organism);
        }
    }
    refresh_program_spec_registry_from_assets_if_needed(&mut spec);
    finalize_parsed_program_spec(&mut spec)?;
    Ok(spec)
}

pub fn bundled_syn3a_program_spec_json() -> &'static str {
    static BUNDLED_SPEC_JSON: OnceLock<String> = OnceLock::new();
    BUNDLED_SPEC_JSON
        .get_or_init(|| {
            let spec = bundled_syn3a_program_spec()
                .expect("bundled Syn3A program spec should hydrate successfully");
            serde_json::to_string_pretty(&spec)
                .expect("bundled Syn3A program spec should serialize successfully")
        })
        .as_str()
}

pub fn parse_organism_spec_json(spec_json: &str) -> Result<WholeCellOrganismSpec, String> {
    serde_json::from_str::<WholeCellOrganismSpec>(spec_json)
        .map_err(|error| format!("failed to parse organism spec: {error}"))
}

pub fn parse_legacy_organism_spec_json(spec_json: &str) -> Result<WholeCellOrganismSpec, String> {
    serde_json::from_str::<WholeCellOrganismSpec>(spec_json)
        .map(with_normalized_pool_metadata)
        .map(with_normalized_semantic_metadata)
        .map(with_compiled_chromosome_domains)
        .map_err(|error| format!("failed to parse organism spec: {error}"))
}

pub fn parse_genome_asset_package_json(
    spec_json: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse genome asset package: {error}"))
}

pub fn parse_legacy_genome_asset_package_json(
    spec_json: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    serde_json::from_str(spec_json)
        .map(with_normalized_asset_pool_metadata)
        .map(with_normalized_asset_semantic_metadata)
        .map_err(|error| format!("failed to parse genome asset package: {error}"))
}

pub fn bundled_syn3a_organism_spec_json() -> &'static str {
    static BUNDLED_ORGANISM_JSON: OnceLock<String> = OnceLock::new();
    BUNDLED_ORGANISM_JSON.get_or_init(|| {
        let organism =
            bundled_syn3a_organism_spec().expect("compile embedded structured Syn3A organism");
        serde_json::to_string_pretty(&organism)
            .expect("serialize embedded structured Syn3A organism")
    })
}

pub fn bundled_syn3a_organism_spec() -> Result<WholeCellOrganismSpec, String> {
    static BUNDLED_ORGANISM: OnceLock<Result<WholeCellOrganismSpec, String>> = OnceLock::new();
    BUNDLED_ORGANISM
        .get_or_init(compile_embedded_syn3a_organism_spec)
        .clone()
}

fn compile_embedded_syn3a_genome_asset_package() -> Result<WholeCellGenomeAssetPackage, String> {
    let manifest = parse_bundle_manifest_json(BUNDLED_SYN3A_BUNDLE_MANIFEST_JSON)?;
    validate_bundle_asset_contracts(&manifest)?;
    let organism = bundled_syn3a_organism_spec()?;
    let mut assets = if manifest.require_explicit_asset_entities {
        empty_genome_asset_package(&organism)
    } else {
        compile_genome_asset_package(&organism)
    };
    assets.operons =
        serde_json::from_str::<Vec<WholeCellOperonSpec>>(BUNDLED_SYN3A_BUNDLE_OPERONS_JSON)
            .map_err(|error| format!("failed to parse embedded Syn3A operons: {error}"))?;
    assets.rnas =
        serde_json::from_str::<Vec<WholeCellRnaProductSpec>>(BUNDLED_SYN3A_BUNDLE_RNAS_JSON)
            .map_err(|error| format!("failed to parse embedded Syn3A RNAs: {error}"))?;
    assets.proteins = serde_json::from_str::<Vec<WholeCellProteinProductSpec>>(
        BUNDLED_SYN3A_BUNDLE_PROTEINS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A proteins: {error}"))?;
    assets.complexes =
        serde_json::from_str::<Vec<WholeCellComplexSpec>>(BUNDLED_SYN3A_BUNDLE_COMPLEXES_JSON)
            .map_err(|error| format!("failed to parse embedded Syn3A complexes: {error}"))?;
    assets.operon_semantics.clear();
    assets.protein_semantics.clear();
    assets.complex_semantics.clear();
    assets.operon_semantics = serde_json::from_str::<Vec<WholeCellOperonSemanticSpec>>(
        BUNDLED_SYN3A_BUNDLE_OPERON_SEMANTICS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A operon semantics: {error}"))?;
    assets.protein_semantics = serde_json::from_str::<Vec<WholeCellProteinSemanticSpec>>(
        BUNDLED_SYN3A_BUNDLE_PROTEIN_SEMANTICS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A protein semantics: {error}"))?;
    assets.complex_semantics = serde_json::from_str::<Vec<WholeCellComplexSemanticSpec>>(
        BUNDLED_SYN3A_BUNDLE_COMPLEX_SEMANTICS_JSON,
    )
    .map_err(|error| format!("failed to parse embedded Syn3A complex semantics: {error}"))?;
    finalize_bundle_asset_package(&organism, &manifest, assets)
}

pub fn bundled_syn3a_genome_asset_package_json() -> Result<&'static str, String> {
    static BUNDLED_ASSET_JSON: OnceLock<Result<String, String>> = OnceLock::new();
    match BUNDLED_ASSET_JSON.get_or_init(|| {
        bundled_syn3a_genome_asset_package().and_then(|package| {
            serde_json::to_string_pretty(&package).map_err(|error| {
                format!("failed to serialize bundled genome asset package: {error}")
            })
        })
    }) {
        Ok(json) => Ok(json.as_str()),
        Err(error) => Err(error.clone()),
    }
}

pub fn bundled_syn3a_genome_asset_package() -> Result<WholeCellGenomeAssetPackage, String> {
    static BUNDLED_ASSET_PACKAGE: OnceLock<Result<WholeCellGenomeAssetPackage, String>> =
        OnceLock::new();
    BUNDLED_ASSET_PACKAGE
        .get_or_init(compile_embedded_syn3a_genome_asset_package)
        .clone()
}

pub fn bundled_syn3a_process_registry() -> Result<WholeCellGenomeProcessRegistry, String> {
    static BUNDLED_REGISTRY: OnceLock<Result<WholeCellGenomeProcessRegistry, String>> =
        OnceLock::new();
    BUNDLED_REGISTRY
        .get_or_init(|| {
            bundled_syn3a_genome_asset_package()
                .map(|package| compile_genome_process_registry(&package))
        })
        .clone()
}

pub fn bundled_syn3a_process_registry_json() -> Result<&'static str, String> {
    static BUNDLED_REGISTRY_JSON: OnceLock<Result<String, String>> = OnceLock::new();
    match BUNDLED_REGISTRY_JSON.get_or_init(|| {
        bundled_syn3a_process_registry().and_then(|registry| {
            serde_json::to_string_pretty(&registry).map_err(|error| {
                format!("failed to serialize bundled genome process registry: {error}")
            })
        })
    }) {
        Ok(json) => Ok(json.as_str()),
        Err(error) => Err(error.clone()),
    }
}

pub fn resolve_bundled_organism_spec(reference: &str) -> Result<WholeCellOrganismSpec, String> {
    match reference.trim().to_lowercase().as_str() {
        "jcvi_syn3a_reference" | "jcvi-syn3a" | "syn3a" | "syn3a_reference" => {
            bundled_syn3a_organism_spec()
        }
        _ => Err(format!("unknown bundled organism reference: {reference}")),
    }
}

pub fn resolve_bundled_genome_asset_package(
    reference: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    match reference.trim().to_lowercase().as_str() {
        "jcvi_syn3a_reference" | "jcvi-syn3a" | "syn3a" | "syn3a_reference" => {
            bundled_syn3a_genome_asset_package()
        }
        _ => Err(format!(
            "unknown bundled genome asset package reference: {reference}"
        )),
    }
}

pub fn resolve_bundled_genome_process_registry(
    reference: &str,
) -> Result<WholeCellGenomeProcessRegistry, String> {
    match reference.trim().to_lowercase().as_str() {
        "jcvi_syn3a_reference" | "jcvi-syn3a" | "syn3a" | "syn3a_reference" => {
            bundled_syn3a_process_registry()
        }
        _ => Err(format!(
            "unknown bundled genome process registry reference: {reference}"
        )),
    }
}

pub fn bundled_syn3a_program_spec() -> Result<WholeCellProgramSpec, String> {
    static BUNDLED_SPEC: OnceLock<Result<WholeCellProgramSpec, String>> = OnceLock::new();
    BUNDLED_SPEC
        .get_or_init(|| parse_program_spec_json(BUNDLED_SYN3A_PROGRAM_JSON))
        .clone()
}

fn refresh_saved_state_registry_from_assets_if_needed(state: &mut WholeCellSavedState) {
    let refresh_registry = match (
        state.organism_assets.as_ref(),
        state.organism_process_registry.as_ref(),
    ) {
        (Some(assets), Some(registry)) if !assets.chromosome_domains.is_empty() => {
            registry.chromosome_domains.is_empty()
                || (registry
                    .species
                    .iter()
                    .filter(|species| species.operon.is_some())
                    .all(|species| species.chromosome_domain.is_none())
                    && registry
                        .reactions
                        .iter()
                        .filter(|reaction| reaction.operon.is_some())
                        .all(|reaction| reaction.chromosome_domain.is_none()))
        }
        (_, None) => true,
        _ => false,
    };
    if refresh_registry {
        if let Some(assets) = state.organism_assets.as_ref() {
            state.organism_process_registry = Some(compile_genome_process_registry(assets));
        }
    }
}

fn normalize_saved_state_runtime_species_from_registry(state: &mut WholeCellSavedState) {
    normalize_runtime_species_bulk_fields_from_registry(
        &mut state.organism_species,
        state.organism_process_registry.as_ref(),
    );
}

fn finalize_parsed_saved_state(
    mut state: WholeCellSavedState,
) -> Result<WholeCellSavedState, String> {
    if state.organism_process_registry.is_none() {
        if let Some(reference) = state.organism_data_ref.as_deref() {
            state.organism_process_registry =
                Some(resolve_bundled_genome_process_registry(reference)?);
        }
    }
    populate_saved_state_contract_metadata(&mut state)?;
    Ok(state)
}

pub fn parse_saved_state_json(state_json: &str) -> Result<WholeCellSavedState, String> {
    let state: WholeCellSavedState = serde_json::from_str(state_json)
        .map_err(|error| format!("failed to parse saved state: {error}"))?;
    finalize_parsed_saved_state(state)
}

pub fn parse_legacy_saved_state_json(state_json: &str) -> Result<WholeCellSavedState, String> {
    let mut state: WholeCellSavedState = serde_json::from_str(state_json)
        .map_err(|error| format!("failed to parse saved state: {error}"))?;
    if let Some(organism) = state.organism_data.take() {
        state.organism_data = Some(with_compiled_chromosome_domains(
            with_normalized_pool_metadata(organism),
        ));
    }
    if state.organism_assets.is_none() {
        if let Some(organism) = state.organism_data.as_ref() {
            state.organism_assets = Some(compile_genome_asset_package(organism));
        }
    } else if let (Some(organism), Some(assets)) =
        (state.organism_data.as_ref(), state.organism_assets.as_mut())
    {
        if assets.chromosome_domains.is_empty() {
            *assets = compile_genome_asset_package(organism);
        }
    }
    refresh_saved_state_registry_from_assets_if_needed(&mut state);
    backfill_legacy_runtime_species_bulk_fields(&mut state.organism_species);
    normalize_saved_state_runtime_species_from_registry(&mut state);
    finalize_parsed_saved_state(state)
}

pub fn saved_state_to_json(state: &WholeCellSavedState) -> Result<String, String> {
    let mut hydrated = state.clone();
    populate_saved_state_contract_metadata(&mut hydrated)?;
    serde_json::to_string_pretty(&hydrated)
        .map_err(|error| format!("failed to serialize saved state: {error}"))
}

pub type WholeCellSeedSpec = WholeCellProgramSpec;
pub type WholeCellLocalChemistryConfig = WholeCellLocalChemistrySpec;
pub type WholeCellCheckpoint = WholeCellSavedState;

pub fn default_syn3a_seed_spec() -> Result<WholeCellProgramSpec, String> {
    bundled_syn3a_program_spec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn bundle_manifest_path(name: &str) -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../src/oneuro/whole_cell/assets/bundles")
            .join(name)
            .join("manifest.json")
            .display()
            .to_string()
    }

    fn minimal_saved_state_from_spec(spec: &WholeCellProgramSpec) -> WholeCellSavedState {
        WholeCellSavedState {
            program_name: spec.program_name.clone(),
            contract: WholeCellContractSchema::default(),
            provenance: WholeCellProvenance::default(),
            organism_data_ref: spec.organism_data_ref.clone(),
            organism_data: spec.organism_data.clone(),
            organism_assets: spec.organism_assets.clone(),
            organism_expression: WholeCellOrganismExpressionState::default(),
            organism_process_registry: spec.organism_process_registry.clone(),
            chromosome_state: WholeCellChromosomeState::default(),
            membrane_division_state: WholeCellMembraneDivisionState::default(),
            organism_species: Vec::new(),
            organism_reactions: Vec::new(),
            complex_assembly: WholeCellComplexAssemblyState::default(),
            named_complexes: Vec::new(),
            scheduler_state: WholeCellSchedulerState::default(),
            config: spec.config.clone(),
            core: WholeCellSavedCoreState {
                time_ms: 0.0,
                step_count: 0,
                adp_mm: 0.2,
                glucose_mm: 0.3,
                oxygen_mm: 0.4,
                ftsz: 1.0,
                dnaa: 1.0,
                active_ribosomes: 1.0,
                active_rnap: 1.0,
                genome_bp: 10,
                replicated_bp: 0,
                chromosome_separation_nm: 1.0,
                radius_nm: 100.0,
                surface_area_nm2: 10.0,
                volume_nm3: 10.0,
                division_progress: 0.0,
                metabolic_load: 1.0,
                quantum_profile: WholeCellQuantumProfile::default(),
            },
            lattice: WholeCellLatticeState {
                atp: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                amino_acids: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                nucleotides: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                membrane_precursors: vec![
                    1.0;
                    spec.config.x_dim * spec.config.y_dim * spec.config.z_dim
                ],
            },
            spatial_fields: None,
            local_chemistry: None,
            chemistry_report: LocalChemistryReport::default(),
            chemistry_site_reports: Vec::new(),
            last_md_probe: None,
            scheduled_subsystem_probes: Vec::new(),
            subsystem_states: Vec::new(),
            md_translation_scale: 1.0,
            md_membrane_scale: 1.0,
        }
    }

    #[test]
    fn bundled_syn3a_program_spec_resolves_organism_data() {
        let spec = bundled_syn3a_program_spec().expect("bundled program spec");
        let organism = spec.organism_data.as_ref().expect("bundled organism data");
        let assets = spec
            .organism_assets
            .as_ref()
            .expect("bundled organism asset package");
        let registry = spec
            .organism_process_registry
            .as_ref()
            .expect("bundled organism process registry");
        let profile = derive_organism_profile(organism);

        assert_eq!(
            spec.organism_data_ref.as_deref(),
            Some("jcvi_syn3a_reference")
        );
        assert_eq!(organism.organism, "JCVI-syn3A");
        assert!(profile.gene_count >= 8);
        assert!(profile.transcription_unit_count >= 4);
        assert!(profile.process_scales.translation > 0.9);
        assert!(profile.metabolic_burden_scale > 0.9);
        assert!(organism.chromosome_domains.len() >= 4);
        assert!(assets.operons.len() >= 4);
        assert_eq!(
            assets.chromosome_domains.len(),
            organism.chromosome_domains.len()
        );
        assert_eq!(assets.operon_semantics.len(), assets.operons.len());
        assert_eq!(assets.protein_semantics.len(), assets.proteins.len());
        assert_eq!(assets.complex_semantics.len(), assets.complexes.len());
        assert_eq!(assets.rnas.len(), organism.genes.len());
        assert_eq!(assets.proteins.len(), organism.genes.len());
        assert!(assets.complexes.len() >= 4);
        assert_eq!(
            registry.chromosome_domains.len(),
            organism.chromosome_domains.len()
        );
        assert!(registry.species.len() > assets.proteins.len());
        assert!(registry.reactions.len() >= assets.proteins.len());
        assert!(spec.provenance.compiled_ir_hash.is_some());
    }

    #[test]
    fn parse_program_spec_json_hydrates_bundled_organism_data() {
        let spec = parse_program_spec_json(
            r#"{
                "program_name": "test_program",
                "organism_data_ref": "syn3a",
                "config": {
                    "x_dim": 8,
                    "y_dim": 8,
                    "z_dim": 4,
                    "voxel_size_nm": 20.0,
                    "dt_ms": 0.25,
                    "cme_interval": 4,
                    "ode_interval": 1,
                    "bd_interval": 2,
                    "geometry_interval": 4,
                    "use_gpu": false
                },
                "initial_lattice": {
                    "atp": 1.0,
                    "amino_acids": 1.0,
                    "nucleotides": 1.0,
                    "membrane_precursors": 1.0
                },
                "initial_state": {
                    "adp_mm": 0.2,
                    "glucose_mm": 0.3,
                    "oxygen_mm": 0.4,
                    "genome_bp": 1,
                    "replicated_bp": 0,
                    "chromosome_separation_nm": 10.0,
                    "radius_nm": 100.0,
                    "division_progress": 0.0,
                    "metabolic_load": 1.0
                }
            }"#,
        )
        .expect("spec with bundled organism");

        assert!(spec.organism_data.is_some());
        assert!(spec.organism_assets.is_some());
        assert_eq!(spec.organism_data_ref.as_deref(), Some("syn3a"));
        assert_eq!(spec.contract.contract_version, WHOLE_CELL_CONTRACT_VERSION);
        assert_eq!(
            spec.contract.program_schema_version,
            WHOLE_CELL_PROGRAM_SCHEMA_VERSION
        );
        assert!(spec.provenance.organism_asset_hash.is_some());
        assert!(spec.provenance.run_manifest_hash.is_some());
    }

    #[test]
    fn parse_program_spec_json_keeps_inline_organism_specs_explicit() {
        let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
        spec.organism_data_ref = None;
        spec.organism_assets = None;
        spec.organism_process_registry = None;

        let parsed = parse_program_spec_json(
            &serde_json::to_string(&spec).expect("serialize explicit program spec"),
        )
        .expect("parse explicit program spec");

        assert!(parsed.organism_data.is_some());
        assert!(parsed.organism_assets.is_none());
        assert!(parsed.organism_process_registry.is_none());
        assert_eq!(
            parsed.contract.contract_version,
            WHOLE_CELL_CONTRACT_VERSION
        );
        assert_eq!(
            parsed.contract.program_schema_version,
            WHOLE_CELL_PROGRAM_SCHEMA_VERSION
        );
    }

    #[test]
    fn parse_legacy_program_spec_json_derives_inline_organism_assets() {
        let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
        spec.organism_data_ref = None;
        spec.organism_assets = None;
        spec.organism_process_registry = None;

        let parsed = parse_legacy_program_spec_json(
            &serde_json::to_string(&spec).expect("serialize legacy program spec"),
        )
        .expect("parse legacy program spec");

        assert!(parsed.organism_data.is_some());
        assert!(parsed.organism_assets.is_some());
        assert!(parsed.organism_process_registry.is_some());
        assert_eq!(
            parsed.contract.contract_version,
            WHOLE_CELL_CONTRACT_VERSION
        );
        assert_eq!(
            parsed.contract.program_schema_version,
            WHOLE_CELL_PROGRAM_SCHEMA_VERSION
        );
    }

    #[test]
    fn parse_program_spec_json_keeps_inline_assets_without_registry_explicit() {
        let mut spec = bundled_syn3a_program_spec().expect("bundled Syn3A program spec");
        spec.organism_data_ref = None;
        spec.organism_process_registry = None;

        let parsed = parse_program_spec_json(
            &serde_json::to_string(&spec).expect("serialize explicit asset-backed program spec"),
        )
        .expect("parse explicit asset-backed program spec");

        assert!(parsed.organism_assets.is_some());
        assert!(parsed.organism_process_registry.is_none());
    }

    #[test]
    fn bundled_syn3a_genome_asset_package_compiles_from_descriptor() {
        let organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let summary = WholeCellGenomeAssetSummary::from(&package);

        assert_eq!(package.organism, organism.organism);
        assert_eq!(package.rnas.len(), organism.genes.len());
        assert_eq!(package.proteins.len(), organism.genes.len());
        assert!(package.complexes.len() >= organism.transcription_units.len());
        assert_eq!(
            summary.chromosome_domain_count,
            package.chromosome_domains.len()
        );
        assert!(package.chromosome_domains.len() >= 4);
        assert!(package
            .chromosome_domains
            .iter()
            .all(|domain| !domain.operons.is_empty()));
        assert!(summary.operon_count >= organism.transcription_units.len());
        assert!(summary.protein_count >= 8);
        assert!(summary.targeted_complex_count >= 4);
    }

    #[test]
    fn bundled_syn3a_genome_asset_package_json_round_trips() {
        let package_json =
            bundled_syn3a_genome_asset_package_json().expect("bundled asset package json");
        let package = parse_genome_asset_package_json(package_json).expect("parsed asset package");

        assert_eq!(package.organism, "JCVI-syn3A");
        assert!(package.chromosome_domains.len() >= 4);
        assert!(package.operons.iter().any(|operon| operon.polycistronic));
        assert!(package
            .complexes
            .iter()
            .any(|complex| !complex.subsystem_targets.is_empty()));
    }

    #[test]
    fn explicit_asset_entity_validation_rejects_incomplete_operons() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let operon = package.operons.first_mut().expect("operon");
        let operon_name = operon.name.clone();
        operon.asset_class = None;
        operon.complex_family = None;
        operon.subsystem_targets.clear();

        let error = validate_explicit_asset_entities(&package)
            .expect_err("incomplete strict operon assets should fail");

        assert!(error.contains("requires explicit asset entities"));
        assert!(error.contains(&operon_name));
    }

    #[test]
    fn explicit_asset_semantic_validation_requires_full_operon_coverage() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let removed = package
            .operon_semantics
            .pop()
            .expect("bundled operon semantic");

        let error = validate_explicit_asset_semantics(&package)
            .expect_err("missing strict operon semantic coverage should fail");

        assert!(error.contains("requires explicit asset semantics"));
        assert!(error.contains(&removed.name));
    }

    #[test]
    fn explicit_asset_entity_coverage_requires_operon_presence() {
        let organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let removed = package.operons.pop().expect("bundled operon");

        let error = validate_explicit_asset_entity_coverage(&organism, &package)
            .expect_err("missing strict operon coverage should fail");

        assert!(error.contains("requires explicit asset entity coverage"));
        assert!(error.contains(&removed.name));
    }

    #[test]
    fn explicit_asset_semantics_merge_restores_entity_fields() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let semantic = package
            .operon_semantics
            .first()
            .expect("bundled operon semantic")
            .clone();
        let operon = package
            .operons
            .iter_mut()
            .find(|operon| operon.name == semantic.name)
            .expect("matching operon");
        operon.asset_class = None;
        operon.complex_family = None;
        operon.subsystem_targets.clear();

        merge_explicit_asset_semantics_into_entities(&mut package);

        let restored = package
            .operons
            .iter()
            .find(|operon| operon.name == semantic.name)
            .expect("restored operon");
        assert_eq!(restored.asset_class, Some(semantic.asset_class));
        assert_eq!(restored.complex_family, Some(semantic.complex_family));
        assert_eq!(restored.subsystem_targets, semantic.subsystem_targets);
    }

    #[test]
    fn explicit_transcription_unit_metadata_overrides_name_heuristics() {
        let mut organism = bundled_syn3a_organism_spec().expect("bundled organism");
        organism.genes.push(WholeCellGenomeFeature {
            gene: "opaque_control_gene".to_string(),
            start_bp: 200_100,
            end_bp: 200_360,
            strand: 1,
            essential: false,
            basal_expression: 0.62,
            translation_cost: 1.0,
            nucleotide_cost: 1.0,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            subsystem_targets: Vec::new(),
            asset_class: None,
            complex_family: None,
        });
        organism
            .transcription_units
            .push(WholeCellTranscriptionUnitSpec {
                name: "opaque_control_unit".to_string(),
                genes: vec!["opaque_control_gene".to_string()],
                basal_activity: 0.68,
                process_weights: WholeCellProcessWeights {
                    translation: 1.0,
                    ..WholeCellProcessWeights::default()
                },
                subsystem_targets: vec![Syn3ASubsystemPreset::FtsZSeptumRing],
                asset_class: Some(WholeCellAssetClass::QualityControl),
                complex_family: Some(WholeCellAssemblyFamily::ChaperoneClient),
            });

        let package = compile_genome_asset_package(&organism);
        let operon = package
            .operons
            .iter()
            .find(|operon| operon.name == "opaque_control_unit")
            .expect("compiled opaque operon");
        let complex = package
            .complexes
            .iter()
            .find(|complex| complex.operon == "opaque_control_unit")
            .expect("compiled opaque complex");

        assert_eq!(
            operon.asset_class,
            Some(WholeCellAssetClass::QualityControl)
        );
        assert_eq!(
            operon.complex_family,
            Some(WholeCellAssemblyFamily::ChaperoneClient)
        );
        assert_eq!(
            operon.subsystem_targets,
            vec![Syn3ASubsystemPreset::FtsZSeptumRing]
        );
        assert_eq!(complex.asset_class, WholeCellAssetClass::QualityControl);
        assert_eq!(complex.family, WholeCellAssemblyFamily::ChaperoneClient);
        assert_eq!(
            complex.subsystem_targets,
            vec![Syn3ASubsystemPreset::FtsZSeptumRing]
        );
    }

    #[test]
    fn explicit_gene_metadata_overrides_name_heuristics_for_singletons() {
        let mut organism = bundled_syn3a_organism_spec().expect("bundled organism");
        organism.genes.push(WholeCellGenomeFeature {
            gene: "opaque_singleton_gene".to_string(),
            start_bp: 201_100,
            end_bp: 201_360,
            strand: 1,
            essential: false,
            basal_expression: 0.55,
            translation_cost: 1.0,
            nucleotide_cost: 1.0,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            subsystem_targets: Vec::new(),
            asset_class: Some(WholeCellAssetClass::QualityControl),
            complex_family: Some(WholeCellAssemblyFamily::ChaperoneClient),
        });

        let package = compile_genome_asset_package(&organism);
        let operon = package
            .operons
            .iter()
            .find(|operon| operon.name == "opaque_singleton_gene")
            .expect("singleton operon");
        let protein = package
            .proteins
            .iter()
            .find(|protein| protein.gene == "opaque_singleton_gene")
            .expect("singleton protein");
        let complex = package
            .complexes
            .iter()
            .find(|complex| complex.operon == "opaque_singleton_gene")
            .expect("singleton complex");

        assert_eq!(
            operon.asset_class,
            Some(WholeCellAssetClass::QualityControl)
        );
        assert_eq!(
            operon.complex_family,
            Some(WholeCellAssemblyFamily::ChaperoneClient)
        );
        assert_eq!(protein.asset_class, WholeCellAssetClass::QualityControl);
        assert_eq!(complex.asset_class, WholeCellAssetClass::QualityControl);
        assert_eq!(complex.family, WholeCellAssemblyFamily::ChaperoneClient);
    }

    #[test]
    fn parse_legacy_organism_spec_json_backfills_legacy_semantic_metadata_at_boundary() {
        let mut organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let division_gene = organism
            .genes
            .iter_mut()
            .find(|gene| gene.gene == "ftsz_ring_polymerization_core")
            .expect("division gene");
        division_gene.asset_class = None;
        division_gene.complex_family = None;

        let division_unit = organism
            .transcription_units
            .iter_mut()
            .find(|unit| unit.name == "division_ring_operon")
            .expect("division unit");
        division_unit.subsystem_targets.clear();
        division_unit.asset_class = None;
        division_unit.complex_family = None;

        let json = serde_json::to_string(&organism).expect("serialize organism");
        let reparsed = parse_legacy_organism_spec_json(&json).expect("parse organism");
        let reparsed_gene = reparsed
            .genes
            .iter()
            .find(|gene| gene.gene == "ftsz_ring_polymerization_core")
            .expect("reparsed gene");
        let reparsed_unit = reparsed
            .transcription_units
            .iter()
            .find(|unit| unit.name == "division_ring_operon")
            .expect("reparsed unit");

        assert_eq!(
            reparsed_gene.asset_class,
            Some(WholeCellAssetClass::Constriction)
        );
        assert_eq!(
            reparsed_gene.complex_family,
            Some(WholeCellAssemblyFamily::Divisome)
        );
        assert_eq!(
            reparsed_unit.asset_class,
            Some(WholeCellAssetClass::Constriction)
        );
        assert_eq!(
            reparsed_unit.complex_family,
            Some(WholeCellAssemblyFamily::Divisome)
        );
        assert!(reparsed_unit
            .subsystem_targets
            .contains(&Syn3ASubsystemPreset::FtsZSeptumRing));
    }

    #[test]
    fn parse_organism_spec_json_round_trips_explicit_metadata() {
        let organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let json = serde_json::to_string(&organism).expect("serialize organism");
        let reparsed = parse_organism_spec_json(&json).expect("parse explicit organism");

        assert_eq!(reparsed, organism);
    }

    #[test]
    fn bundled_syn3a_process_registry_compiles_from_assets() {
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let registry = compile_genome_process_registry(&package);
        let summary = WholeCellGenomeProcessRegistrySummary::from(&registry);

        assert_eq!(registry.organism, "JCVI-syn3A");
        assert_eq!(
            registry.chromosome_domains.len(),
            package.chromosome_domains.len()
        );
        assert!(summary.species_count > package.proteins.len());
        assert_eq!(summary.rna_species_count, package.rnas.len());
        assert_eq!(summary.protein_species_count, package.proteins.len());
        assert_eq!(summary.complex_species_count, package.complexes.len());
        assert!(summary.assembly_intermediate_species_count >= package.complexes.len() * 3);
        assert!(summary.transcription_reaction_count >= package.operons.len());
        assert!(summary.translation_reaction_count >= package.proteins.len());
        assert!(summary.transport_reaction_count >= 3);
        assert_eq!(
            summary.degradation_reaction_count,
            package.rnas.len() + package.proteins.len()
        );
        assert_eq!(summary.stress_reaction_count, package.operons.len());
        assert!(summary.assembly_reaction_count >= package.complexes.len() * 4);
        assert_eq!(summary.repair_reaction_count, package.complexes.len());
        assert!(summary.turnover_reaction_count >= package.complexes.len());
        assert!(registry.species.iter().any(|species| {
            species.id == "pool_glucose" && species.bulk_field == Some(WholeCellBulkField::Glucose)
        }));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "ribosome_biogenesis_operon_complex_mature"));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "ribosome_biogenesis_operon_complex_subunit_pool"));
        assert!(registry
            .species
            .iter()
            .any(|species| species.operon.is_some() && species.chromosome_domain.is_some()));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::PoolTransport));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.operon.is_some() && reaction.chromosome_domain.is_some()));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::RnaDegradation));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::ProteinDegradation));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::StressResponse));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::ComplexRepair));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.id == "ribosome_biogenesis_operon_complex_maturation"));
        assert!(registry
            .reactions
            .iter()
            .any(|reaction| reaction.id == "ftsz_ring_polymerization_core_protein_translation"));
        assert!(registry.species.iter().any(|species| {
            species.asset_class == WholeCellAssetClass::Replication
                && species.spatial_scope == WholeCellSpatialScope::NucleoidLocal
        }));
        assert!(registry.species.iter().any(|species| {
            species.compartment == "membrane"
                && species.spatial_scope == WholeCellSpatialScope::MembraneAdjacent
        }));
        assert!(registry.species.iter().any(|species| {
            species
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
                && species.spatial_scope == WholeCellSpatialScope::SeptumLocal
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::Transcription
                && reaction.spatial_scope == WholeCellSpatialScope::NucleoidLocal
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
                && reaction.spatial_scope == WholeCellSpatialScope::MembraneAdjacent
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
                && reaction.spatial_scope == WholeCellSpatialScope::SeptumLocal
        }));
        assert!(registry.species.iter().any(|species| {
            species
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
                && species.patch_domain == WholeCellPatchDomain::MembraneBand
        }));
        assert!(registry.species.iter().any(|species| {
            species
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::FtsZSeptumRing)
                && species.patch_domain == WholeCellPatchDomain::SeptumPatch
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction
                .subsystem_targets
                .contains(&Syn3ASubsystemPreset::ReplisomeTrack)
                && reaction.patch_domain == WholeCellPatchDomain::NucleoidTrack
        }));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "pool_membrane_band_membrane_precursors"));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "pool_septum_patch_membrane_precursors"));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "pool_membrane_band_atp"));
        assert!(registry
            .species
            .iter()
            .any(|species| species.id == "pool_septum_patch_atp"));
        assert!(registry.species.iter().any(|species| {
            species.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && species.bulk_field == Some(WholeCellBulkField::Nucleotides)
                && species.chromosome_domain.is_some()
                && species.id.starts_with("pool_nucleoid_track_nucleotides_")
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::LocalizedPoolTransfer
                && reaction.patch_domain == WholeCellPatchDomain::MembraneBand
                && reaction.products.iter().any(|participant| {
                    participant.species_id == "pool_membrane_band_membrane_precursors"
                })
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::LocalizedPoolTurnover
                && reaction.patch_domain == WholeCellPatchDomain::SeptumPatch
                && reaction.reactants.iter().any(|participant| {
                    participant.species_id == "pool_septum_patch_membrane_precursors"
                })
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::LocalizedPoolTransfer
                && reaction.patch_domain == WholeCellPatchDomain::MembraneBand
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::LocalizedPoolTurnover
                && reaction.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && reaction.chromosome_domain.is_some()
                && reaction
                    .reactants
                    .iter()
                    .any(|participant| participant.species_id.starts_with("pool_nucleoid_track_"))
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            reaction.reaction_class == WholeCellReactionClass::LocalizedPoolTransfer
                && reaction.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && reaction.chromosome_domain.is_some()
                && reaction.products.iter().any(|participant| {
                    participant
                        .species_id
                        .starts_with("pool_nucleoid_track_nucleotides_")
                })
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            matches!(
                reaction.reaction_class,
                WholeCellReactionClass::SubunitPoolFormation
                    | WholeCellReactionClass::ComplexNucleation
                    | WholeCellReactionClass::ComplexElongation
                    | WholeCellReactionClass::ComplexMaturation
                    | WholeCellReactionClass::ComplexTurnover
            ) && reaction.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && reaction.reactants.iter().any(|participant| {
                    participant
                        .species_id
                        .starts_with("pool_nucleoid_track_atp")
                })
        }));
    }

    #[test]
    fn bundled_syn3a_embedded_structured_sources_match_manifest_compilation() {
        let embedded = bundled_syn3a_organism_spec().expect("embedded organism");
        let manifest_path = bundle_manifest_path("jcvi_syn3a");
        let from_manifest = compile_organism_spec_from_bundle_manifest_path(&manifest_path)
            .expect("manifest organism");

        assert_eq!(embedded.organism, from_manifest.organism);
        assert_eq!(embedded.genes, from_manifest.genes);
        assert_eq!(
            embedded.transcription_units,
            from_manifest.transcription_units
        );
        assert_eq!(embedded.pools, from_manifest.pools);
        assert_eq!(
            embedded.chromosome_domains,
            from_manifest.chromosome_domains
        );
    }

    #[test]
    fn bundled_syn3a_embedded_asset_package_matches_manifest_compilation() {
        let embedded = bundled_syn3a_genome_asset_package().expect("embedded assets");
        let manifest_path = bundle_manifest_path("jcvi_syn3a");
        let from_manifest = compile_genome_asset_package_from_bundle_manifest_path(&manifest_path)
            .expect("manifest assets");

        assert_eq!(embedded.organism, from_manifest.organism);
        assert_eq!(embedded.operons, from_manifest.operons);
        assert_eq!(embedded.rnas, from_manifest.rnas);
        assert_eq!(embedded.proteins, from_manifest.proteins);
        assert_eq!(embedded.complexes, from_manifest.complexes);
        assert_eq!(embedded.operon_semantics, from_manifest.operon_semantics);
        assert_eq!(embedded.protein_semantics, from_manifest.protein_semantics);
        assert_eq!(embedded.complex_semantics, from_manifest.complex_semantics);
    }

    #[test]
    fn localized_adp_pool_requests_compile_when_bundle_exposes_adp_pool() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        package.pools.push(WholeCellMoleculePoolSpec {
            species: "adp".to_string(),
            bulk_field: Some(WholeCellBulkField::ADP),
            role: None,
            concentration_mm: 0.35,
            count: 3200.0,
        });
        let registry = compile_genome_process_registry(&package);

        assert!(registry.species.iter().any(|species| {
            species.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && species.bulk_field == Some(WholeCellBulkField::ADP)
                && species.chromosome_domain.is_some()
                && species.id.starts_with("pool_nucleoid_track_adp_")
        }));
        assert!(registry.reactions.iter().any(|reaction| {
            matches!(
                reaction.reaction_class,
                WholeCellReactionClass::StressResponse
                    | WholeCellReactionClass::SubunitPoolFormation
                    | WholeCellReactionClass::ComplexNucleation
                    | WholeCellReactionClass::ComplexElongation
                    | WholeCellReactionClass::ComplexMaturation
                    | WholeCellReactionClass::ComplexRepair
                    | WholeCellReactionClass::ComplexTurnover
            ) && reaction.patch_domain == WholeCellPatchDomain::NucleoidTrack
                && reaction.chromosome_domain.is_some()
                && reaction.products.iter().any(|participant| {
                    participant
                        .species_id
                        .starts_with("pool_nucleoid_track_adp_")
                })
        }));
    }

    #[test]
    fn explicit_pool_bulk_field_overrides_name_heuristics_in_registry_compilation() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        package.pools.push(WholeCellMoleculePoolSpec {
            species: "opaque_pool_alpha".to_string(),
            bulk_field: Some(WholeCellBulkField::MembranePrecursors),
            role: None,
            concentration_mm: 0.22,
            count: 900.0,
        });

        let registry = compile_genome_process_registry(&package);
        let species = registry
            .species
            .iter()
            .find(|species| species.id == "pool_opaque_pool_alpha")
            .expect("compiled opaque pool");

        assert_eq!(
            species.bulk_field,
            Some(WholeCellBulkField::MembranePrecursors)
        );
        assert_eq!(species.asset_class, WholeCellAssetClass::Membrane);
        assert_eq!(species.compartment, "membrane");
        assert_eq!(
            species.spatial_scope,
            WholeCellSpatialScope::MembraneAdjacent
        );
    }

    #[test]
    fn parse_legacy_genome_asset_package_json_backfills_legacy_pool_bulk_field_metadata() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let atp_pool = package
            .pools
            .iter_mut()
            .find(|pool| pool.bulk_field == Some(WholeCellBulkField::ATP))
            .expect("ATP pool");
        atp_pool.bulk_field = None;

        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_legacy_genome_asset_package_json(&json).expect("parse package");
        let reparsed_atp_pool = reparsed
            .pools
            .iter()
            .find(|pool| pool.species.eq_ignore_ascii_case("atp"))
            .expect("reparsed ATP pool");
        assert_eq!(reparsed_atp_pool.bulk_field, Some(WholeCellBulkField::ATP));
    }

    #[test]
    fn parse_legacy_genome_asset_package_json_backfills_legacy_pool_role_metadata() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        package.pools.push(WholeCellMoleculePoolSpec {
            species: "ribosome_shadow_buffer".to_string(),
            bulk_field: None,
            role: None,
            concentration_mm: 0.0,
            count: 96.0,
        });

        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_legacy_genome_asset_package_json(&json).expect("parse package");
        let reparsed_pool = reparsed
            .pools
            .iter()
            .find(|pool| pool.species == "ribosome_shadow_buffer")
            .expect("reparsed role pool");
        assert_eq!(reparsed_pool.role, Some(WholeCellPoolRole::ActiveRibosomes));
    }

    #[test]
    fn parse_legacy_genome_asset_package_json_backfills_legacy_operon_semantics() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let opaque_operon = "opaque_operon_alpha";
        let semantic = package
            .operon_semantics
            .iter_mut()
            .find(|semantic| semantic.name == "division_ring_operon")
            .expect("division operon semantic");
        let expected_asset_class = semantic.asset_class;
        let expected_complex_family = semantic.complex_family;
        let expected_targets = semantic.subsystem_targets.clone();
        semantic.name = opaque_operon.to_string();

        let operon = package
            .operons
            .iter_mut()
            .find(|operon| operon.name == "division_ring_operon")
            .expect("division operon");
        operon.name = opaque_operon.to_string();
        operon.subsystem_targets.clear();
        operon.asset_class = None;
        operon.complex_family = None;

        for protein in package
            .proteins
            .iter_mut()
            .filter(|protein| protein.operon == "division_ring_operon")
        {
            protein.operon = opaque_operon.to_string();
            protein.subsystem_targets.clear();
        }

        let complex = package
            .complexes
            .iter_mut()
            .find(|complex| complex.operon == "division_ring_operon")
            .expect("division complex");
        let expected_membrane_inserted = complex.membrane_inserted;
        let expected_division_coupled = complex.division_coupled;
        complex.operon = opaque_operon.to_string();
        complex.subsystem_targets.clear();
        complex.asset_class = WholeCellAssetClass::Generic;
        complex.family = WholeCellAssemblyFamily::Generic;
        complex.membrane_inserted = false;
        complex.division_coupled = false;

        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_legacy_genome_asset_package_json(&json).expect("parse package");
        let reparsed_operon = reparsed
            .operons
            .iter()
            .find(|operon| operon.name == opaque_operon)
            .expect("reparsed operon");
        let reparsed_complex = reparsed
            .complexes
            .iter()
            .find(|complex| complex.operon == opaque_operon)
            .expect("reparsed complex");

        assert_eq!(reparsed_operon.asset_class, Some(expected_asset_class));
        assert_eq!(
            reparsed_operon.complex_family,
            Some(expected_complex_family)
        );
        for target in &expected_targets {
            assert!(reparsed_operon.subsystem_targets.contains(target));
        }
        assert_eq!(reparsed_complex.asset_class, expected_asset_class);
        assert_eq!(reparsed_complex.family, expected_complex_family);
        for target in &expected_targets {
            assert!(reparsed_complex.subsystem_targets.contains(target));
        }
        assert_eq!(reparsed_complex.division_coupled, expected_division_coupled);
        assert_eq!(
            reparsed_complex.membrane_inserted,
            expected_membrane_inserted
        );
    }

    #[test]
    fn parse_legacy_genome_asset_package_json_backfills_legacy_complex_semantics() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let complex = package
            .complexes
            .iter_mut()
            .find(|complex| complex.operon == "division_ring_operon")
            .expect("division complex");
        let expected_id = complex.id.clone();
        complex.operon = "opaque_complex_operon_alpha".to_string();
        complex.subsystem_targets.clear();
        complex.asset_class = WholeCellAssetClass::Generic;
        complex.family = WholeCellAssemblyFamily::Generic;
        complex.membrane_inserted = false;
        complex.chromosome_coupled = false;
        complex.division_coupled = false;

        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_legacy_genome_asset_package_json(&json).expect("parse package");
        let reparsed_complex = reparsed
            .complexes
            .iter()
            .find(|complex| complex.id == expected_id)
            .expect("reparsed complex");
        let reparsed_semantic = reparsed
            .complex_semantics
            .iter()
            .find(|semantic| semantic.id == expected_id)
            .expect("reparsed complex semantic");

        assert_eq!(reparsed_complex.asset_class, reparsed_semantic.asset_class);
        assert_eq!(reparsed_complex.family, reparsed_semantic.family);
        for target in &reparsed_semantic.subsystem_targets {
            assert!(reparsed_complex.subsystem_targets.contains(target));
        }
        assert_eq!(
            reparsed_complex.membrane_inserted,
            reparsed_semantic.membrane_inserted
        );
        assert_eq!(
            reparsed_complex.chromosome_coupled,
            reparsed_semantic.chromosome_coupled
        );
        assert_eq!(
            reparsed_complex.division_coupled,
            reparsed_semantic.division_coupled
        );
    }

    #[test]
    fn parse_legacy_genome_asset_package_json_backfills_legacy_protein_semantics() {
        let mut package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let protein = package
            .proteins
            .iter_mut()
            .find(|protein| protein.operon == "division_ring_operon")
            .expect("division protein");
        let expected_id = protein.id.clone();
        protein.operon = "opaque_protein_operon_alpha".to_string();
        protein.subsystem_targets.clear();
        protein.asset_class = WholeCellAssetClass::Generic;

        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_legacy_genome_asset_package_json(&json).expect("parse package");
        let reparsed_protein = reparsed
            .proteins
            .iter()
            .find(|protein| protein.id == expected_id)
            .expect("reparsed protein");
        let reparsed_semantic = reparsed
            .protein_semantics
            .iter()
            .find(|semantic| semantic.id == expected_id)
            .expect("reparsed protein semantic");

        assert_eq!(reparsed_protein.asset_class, reparsed_semantic.asset_class);
        for target in &reparsed_semantic.subsystem_targets {
            assert!(reparsed_protein.subsystem_targets.contains(target));
        }
    }

    #[test]
    fn parse_genome_asset_package_json_round_trips_explicit_metadata() {
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let json = serde_json::to_string(&package).expect("serialize package");
        let reparsed = parse_genome_asset_package_json(&json).expect("parse explicit package");

        assert_eq!(reparsed, package);
    }

    #[test]
    fn saved_state_json_hydrates_contract_defaults() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);

        let json = saved_state_to_json(&saved).expect("saved json");
        saved.contract.contract_version.clear();
        let reparsed = parse_saved_state_json(&json).expect("reparsed saved state");

        assert_eq!(
            reparsed.contract.saved_state_schema_version,
            WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION
        );
        assert_eq!(
            reparsed.contract.contract_version,
            WHOLE_CELL_CONTRACT_VERSION
        );
        assert!(reparsed.provenance.organism_asset_hash.is_some());
        assert!(reparsed.provenance.compiled_ir_hash.is_some());
        assert!(reparsed.provenance.run_manifest_hash.is_some());
        assert!(reparsed.organism_process_registry.is_some());
    }

    #[test]
    fn parse_legacy_saved_state_json_backfills_legacy_pool_bulk_fields_at_boundary() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);
        let organism = saved.organism_data.as_mut().expect("organism");
        let atp_pool = organism
            .pools
            .iter_mut()
            .find(|pool| pool.bulk_field == Some(WholeCellBulkField::ATP))
            .expect("ATP pool");
        atp_pool.bulk_field = None;

        let reparsed =
            parse_legacy_saved_state_json(&saved_state_to_json(&saved).expect("saved json"))
                .expect("reparsed saved state");

        let stored_pool = reparsed
            .organism_data
            .as_ref()
            .expect("stored organism")
            .pools
            .iter()
            .find(|pool| pool.species.eq_ignore_ascii_case("atp"))
            .expect("stored ATP pool");
        assert_eq!(stored_pool.bulk_field, Some(WholeCellBulkField::ATP));
    }

    #[test]
    fn parse_legacy_saved_state_json_backfills_runtime_pool_bulk_fields_at_boundary() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);
        saved.organism_data = None;
        saved.organism_assets = None;
        saved.organism_process_registry = None;
        saved.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "opaque_runtime_pool".to_string(),
            name: "ATP".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 8.0,
            bulk_field: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            count: 8.0,
            anchor_count: 8.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];

        let reparsed =
            parse_legacy_saved_state_json(&saved_state_to_json(&saved).expect("saved json"))
                .expect("reparsed saved state");

        assert_eq!(
            reparsed.organism_species[0].bulk_field,
            Some(WholeCellBulkField::ATP)
        );
    }

    #[test]
    fn parse_saved_state_json_round_trips_explicit_state() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let saved = minimal_saved_state_from_spec(&spec);
        let json = saved_state_to_json(&saved).expect("saved json");
        let reparsed = parse_saved_state_json(&json).expect("reparsed saved state");

        assert_eq!(reparsed.organism_data, saved.organism_data);
        assert_eq!(reparsed.organism_assets, saved.organism_assets);
        assert_eq!(
            reparsed.organism_process_registry,
            saved.organism_process_registry
        );
    }

    #[test]
    fn parse_saved_state_json_keeps_runtime_species_bulk_fields_explicit() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);
        saved.organism_species = vec![WholeCellSpeciesRuntimeState {
            id: "atp_runtime_pool".to_string(),
            name: "ATP".to_string(),
            species_class: WholeCellSpeciesClass::Pool,
            compartment: "cytosol".to_string(),
            asset_class: WholeCellAssetClass::Energy,
            basal_abundance: 8.0,
            bulk_field: None,
            operon: None,
            parent_complex: None,
            subsystem_targets: Vec::new(),
            spatial_scope: WholeCellSpatialScope::WellMixed,
            patch_domain: WholeCellPatchDomain::Distributed,
            chromosome_domain: None,
            count: 8.0,
            anchor_count: 8.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];

        let reparsed = parse_saved_state_json(&saved_state_to_json(&saved).expect("saved json"))
            .expect("reparsed explicit saved state");

        assert_eq!(reparsed.organism_species[0].bulk_field, None);
    }

    #[test]
    fn parse_legacy_saved_state_json_derives_inline_organism_assets() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);
        saved.organism_data_ref = None;
        saved.organism_assets = None;
        saved.organism_process_registry = None;

        let reparsed =
            parse_legacy_saved_state_json(&saved_state_to_json(&saved).expect("saved json"))
                .expect("reparsed legacy saved state");

        assert!(reparsed.organism_data.is_some());
        assert!(reparsed.organism_assets.is_some());
        assert!(reparsed.organism_process_registry.is_some());
    }

    #[test]
    fn parse_saved_state_json_keeps_inline_assets_without_registry_explicit() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = minimal_saved_state_from_spec(&spec);
        saved.organism_data_ref = None;
        saved.organism_process_registry = None;

        let reparsed = parse_saved_state_json(&saved_state_to_json(&saved).expect("saved json"))
            .expect("reparsed explicit saved state");

        assert!(reparsed.organism_assets.is_some());
        assert!(reparsed.organism_process_registry.is_none());
    }

    #[test]
    fn compile_syn3a_bundle_manifest_path_hydrates_program_spec() {
        let manifest_path = bundle_manifest_path("jcvi_syn3a");
        let spec =
            compile_program_spec_from_bundle_manifest_path(&manifest_path).expect("compiled spec");
        let organism = spec.organism_data.as_ref().expect("compiled organism");
        let assets = spec.organism_assets.as_ref().expect("compiled assets");

        assert_eq!(organism.organism, "JCVI-syn3A");
        assert_eq!(
            spec.program_name.as_deref(),
            Some("jcvi_syn3a_structured_bundle_native")
        );
        assert_eq!(assets.proteins.len(), organism.genes.len());
        assert!(assets.complexes.len() >= organism.transcription_units.len());
        assert_eq!(
            spec.provenance.source_dataset.as_deref(),
            Some("bundled_syn3a_structured_bundle")
        );
        assert!(spec.provenance.organism_asset_hash.is_some());
    }

    #[test]
    fn compile_demo_bundle_manifest_path_from_gff_and_fasta() {
        let manifest_path = bundle_manifest_path("mgen_minimal_demo");
        let organism = compile_organism_spec_from_bundle_manifest_path(&manifest_path)
            .expect("compiled demo organism");
        let assets_json =
            compile_genome_asset_package_json_from_bundle_manifest_path(&manifest_path)
                .expect("compiled demo assets json");
        let assets =
            parse_genome_asset_package_json(&assets_json).expect("parsed demo assets package");
        let registry_json =
            compile_genome_process_registry_json_from_bundle_manifest_path(&manifest_path)
                .expect("compiled demo process registry json");
        let registry = parse_genome_process_registry_json(&registry_json)
            .expect("parsed demo process registry");

        assert_eq!(organism.organism, "Mgen-minimal-demo");
        assert_eq!(organism.genes.len(), 4);
        assert_eq!(organism.transcription_units.len(), 3);
        assert!(organism.chromosome_length_bp > 1000);
        assert!(organism.genes.iter().all(|gene| gene.asset_class.is_some()));
        assert!(organism
            .genes
            .iter()
            .all(|gene| gene.complex_family.is_some()));
        assert!(organism
            .transcription_units
            .iter()
            .all(|unit| unit.asset_class.is_some()));
        assert!(organism
            .transcription_units
            .iter()
            .all(|unit| unit.complex_family.is_some()));
        assert_eq!(assets.proteins.len(), 4);
        assert!(assets.operons.iter().any(|operon| operon.polycistronic));
        assert!(registry.species.len() > assets.proteins.len());
        assert!(registry.reactions.len() >= assets.proteins.len());
    }
}
