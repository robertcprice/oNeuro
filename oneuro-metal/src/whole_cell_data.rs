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
const BUNDLED_SYN3A_ORGANISM_JSON: &str = include_str!("../specs/whole_cell_syn3a_organism.json");
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
    pub pools: Vec<WholeCellMoleculePoolSpec>,
    #[serde(default)]
    pub genes: Vec<WholeCellGenomeFeature>,
    #[serde(default)]
    pub transcription_units: Vec<WholeCellTranscriptionUnitSpec>,
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
    pub operons: Vec<WholeCellOperonSpec>,
    #[serde(default)]
    pub rnas: Vec<WholeCellRnaProductSpec>,
    #[serde(default)]
    pub proteins: Vec<WholeCellProteinProductSpec>,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeProcessRegistry {
    pub organism: String,
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
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::PoolTransport)
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
            .filter(|reaction| reaction.reaction_class == WholeCellReactionClass::ComplexTurnover)
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
    pub gene_count: usize,
    pub transcription_unit_count: usize,
    pub pool_count: usize,
}

impl From<&WholeCellOrganismSpec> for WholeCellOrganismSummary {
    fn from(spec: &WholeCellOrganismSpec) -> Self {
        Self {
            organism: spec.organism.clone(),
            chromosome_length_bp: spec.chromosome_length_bp,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellOrganismBundleManifest {
    #[serde(default)]
    pub organism: Option<String>,
    #[serde(default)]
    pub source_dataset: Option<String>,
    #[serde(default)]
    pub organism_spec_json: Option<String>,
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
    pub transcription_units_json: Option<String>,
    #[serde(default)]
    pub pools_json: Option<String>,
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

fn find_pool_species_id_by_hint(
    package: &WholeCellGenomeAssetPackage,
    hint: &str,
) -> Option<String> {
    package
        .pools
        .iter()
        .find(|pool| pool.species.to_ascii_lowercase().contains(hint))
        .map(|pool| pool_species_id(&pool.species))
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

pub fn compile_genome_asset_package(spec: &WholeCellOrganismSpec) -> WholeCellGenomeAssetPackage {
    let mut gene_to_operon = HashMap::<String, String>::new();
    let mut operons = Vec::new();

    for transcription_unit in &spec.transcription_units {
        let (promoter_bp, terminator_bp) = operon_bounds(spec, &transcription_unit.genes);
        for gene_name in &transcription_unit.genes {
            gene_to_operon.insert(gene_name.clone(), transcription_unit.name.clone());
        }
        operons.push(WholeCellOperonSpec {
            name: transcription_unit.name.clone(),
            genes: transcription_unit.genes.clone(),
            promoter_bp,
            terminator_bp,
            basal_activity: transcription_unit.basal_activity.max(0.0),
            polycistronic: transcription_unit.genes.len() > 1,
            process_weights: transcription_unit.process_weights.clamped(),
        });
    }

    for gene in &spec.genes {
        if gene_to_operon.contains_key(&gene.gene) {
            continue;
        }
        gene_to_operon.insert(gene.gene.clone(), gene.gene.clone());
        operons.push(WholeCellOperonSpec {
            name: gene.gene.clone(),
            genes: vec![gene.gene.clone()],
            promoter_bp: gene.start_bp.min(gene.end_bp),
            terminator_bp: gene.start_bp.max(gene.end_bp),
            basal_activity: gene.basal_expression.max(0.0),
            polycistronic: false,
            process_weights: gene.process_weights.clamped(),
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
        let asset_class =
            inferred_asset_class(gene.process_weights, &gene.subsystem_targets, &gene.gene);
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
        let mut subsystem_targets = Vec::new();
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
        let asset_class = inferred_asset_class(process_weights, &subsystem_targets, &operon.name);
        let family = inferred_complex_family(asset_class, &subsystem_targets, &operon.name);
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

    WholeCellGenomeAssetPackage {
        organism: spec.organism.clone(),
        chromosome_length_bp: spec.chromosome_length_bp.max(1),
        origin_bp: spec.origin_bp.min(spec.chromosome_length_bp.max(1)),
        terminus_bp: spec.terminus_bp.min(spec.chromosome_length_bp.max(1)),
        operons,
        rnas,
        proteins,
        complexes,
        pools: spec.pools.clone(),
    }
}

pub fn compile_genome_process_registry(
    package: &WholeCellGenomeAssetPackage,
) -> WholeCellGenomeProcessRegistry {
    let mut species = Vec::new();
    let mut reactions = Vec::new();
    let atp_pool = find_pool_species_id_by_hint(package, "atp");
    let adp_pool = find_pool_species_id_by_hint(package, "adp");
    let nucleotide_pool = find_pool_species_id_by_hint(package, "nucleotide");
    let amino_pool = find_pool_species_id_by_hint(package, "amino");

    for pool in &package.pools {
        let species_name = pool.species.clone();
        let bulk_field = infer_pool_bulk_field(&species_name);
        let asset_class = if species_name.to_ascii_lowercase().contains("membrane")
            || species_name.to_ascii_lowercase().contains("lipid")
        {
            WholeCellAssetClass::Membrane
        } else if species_name.to_ascii_lowercase().contains("atp")
            || species_name.to_ascii_lowercase().contains("oxygen")
            || species_name.to_ascii_lowercase().contains("glucose")
        {
            WholeCellAssetClass::Energy
        } else if species_name.to_ascii_lowercase().contains("amino") {
            WholeCellAssetClass::Translation
        } else if species_name.to_ascii_lowercase().contains("nucleotide") {
            WholeCellAssetClass::Replication
        } else {
            WholeCellAssetClass::Generic
        };
        let compartment = if species_name.to_ascii_lowercase().contains("membrane")
            || species_name.to_ascii_lowercase().contains("lipid")
        {
            "membrane"
        } else {
            "cytosol"
        };
        let spatial_scope = registry_spatial_scope(asset_class, compartment, &[]);
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
        });
    }

    for rna in &package.rnas {
        let compartment = registry_compartment_for_asset_class(
            rna.asset_class,
            &Vec::<Syn3ASubsystemPreset>::new(),
        );
        let spatial_scope = registry_spatial_scope(rna.asset_class, compartment, &[]);
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
        });
    }

    for protein in &package.proteins {
        let compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let spatial_scope =
            registry_spatial_scope(protein.asset_class, compartment, &protein.subsystem_targets);
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
        });
    }

    for complex in &package.complexes {
        let compartment =
            registry_compartment_for_asset_class(complex.asset_class, &complex.subsystem_targets);
        let spatial_scope =
            registry_spatial_scope(complex.asset_class, compartment, &complex.subsystem_targets);
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
        });
    }

    for pool in &package.pools {
        let pool_id = pool_species_id(&pool.species);
        let Some(field) = infer_pool_bulk_field(&pool.species) else {
            continue;
        };
        if !bulk_field_supports_transport(field) {
            continue;
        }
        let asset_class = transport_asset_class_for_bulk_field(field);
        let compartment = registry_compartment_for_asset_class(asset_class, &[]);
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
        });
    }

    for operon in &package.operons {
        let operon_asset_class = inferred_asset_class(
            operon.process_weights,
            &Vec::<Syn3ASubsystemPreset>::new(),
            &operon.name,
        );
        let operon_compartment = registry_compartment_for_asset_class(operon_asset_class, &[]);
        let mut reactants = Vec::new();
        if let Some(species_id) = nucleotide_pool.as_ref() {
            reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
                spatial_scope: registry_spatial_scope(operon_asset_class, operon_compartment, &[]),
            });
        }

        let mut stress_reactants = Vec::new();
        if let Some(species_id) = atp_pool.as_ref() {
            stress_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        if let Some(species_id) = amino_pool.as_ref() {
            stress_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: 0.40 * (operon.genes.len().max(1) as f32).sqrt().max(1.0),
            });
        }
        let mut stress_products = Vec::new();
        if let Some(species_id) = adp_pool.as_ref() {
            stress_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
            spatial_scope: registry_spatial_scope(operon_asset_class, operon_compartment, &[]),
        });
    }

    for protein in &package.proteins {
        let protein_compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let mut reactants = vec![WholeCellReactionParticipantSpec {
            species_id: protein.rna_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = amino_pool.as_ref() {
            reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
            spatial_scope: registry_spatial_scope(
                protein.asset_class,
                protein_compartment,
                &protein.subsystem_targets,
            ),
        });
    }

    for rna in &package.rnas {
        let rna_compartment = registry_compartment_for_asset_class(
            rna.asset_class,
            &Vec::<Syn3ASubsystemPreset>::new(),
        );
        let mut products = Vec::new();
        if let Some(species_id) = nucleotide_pool.as_ref() {
            products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
            spatial_scope: registry_spatial_scope(rna.asset_class, rna_compartment, &[]),
        });
    }

    for protein in &package.proteins {
        let protein_compartment =
            registry_compartment_for_asset_class(protein.asset_class, &protein.subsystem_targets);
        let mut products = Vec::new();
        if let Some(species_id) = amino_pool.as_ref() {
            products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
            spatial_scope: registry_spatial_scope(
                protein.asset_class,
                protein_compartment,
                &protein.subsystem_targets,
            ),
        });
    }

    for complex in &package.complexes {
        let complex_compartment =
            registry_compartment_for_asset_class(complex.asset_class, &complex.subsystem_targets);
        let total_stoichiometry = total_complex_stoichiometry(complex);
        let subunit_pool_id = complex_stage_species_id(&complex.id, "subunit_pool");
        let nucleation_id = complex_stage_species_id(&complex.id, "nucleation");
        let elongation_id = complex_stage_species_id(&complex.id, "elongation");
        let mature_id = complex_stage_species_id(&complex.id, "mature");
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
            reactants: complex
                .components
                .iter()
                .map(|component| WholeCellReactionParticipantSpec {
                    species_id: component.protein_id.clone(),
                    stoichiometry: component.stoichiometry.max(1) as f32,
                })
                .collect(),
            products: vec![WholeCellReactionParticipantSpec {
                species_id: subunit_pool_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_nucleation", canonical_species_fragment(&complex.id)),
            name: format!("{} nucleation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexNucleation,
            asset_class: complex.asset_class,
            nominal_rate: (0.03 + 0.010 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: subunit_pool_id.clone(),
                stoichiometry: 1.0,
            }],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: nucleation_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_elongation", canonical_species_fragment(&complex.id)),
            name: format!("{} elongation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexElongation,
            asset_class: complex.asset_class,
            nominal_rate: (0.04 + 0.012 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: vec![
                WholeCellReactionParticipantSpec {
                    species_id: nucleation_id.clone(),
                    stoichiometry: 1.0,
                },
                WholeCellReactionParticipantSpec {
                    species_id: subunit_pool_id.clone(),
                    stoichiometry: 0.5 * total_stoichiometry.max(1.0),
                },
            ],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: elongation_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_maturation", canonical_species_fragment(&complex.id)),
            name: format!("{} maturation", complex.name),
            reaction_class: WholeCellReactionClass::ComplexMaturation,
            asset_class: complex.asset_class,
            nominal_rate: (0.05 + 0.015 * complex.basal_abundance.max(0.1).sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: elongation_id.clone(),
                stoichiometry: 1.0,
            }],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: mature_id.clone(),
                stoichiometry: 1.0,
            }],
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
        let mut repair_reactants = vec![WholeCellReactionParticipantSpec {
            species_id: subunit_pool_id.clone(),
            stoichiometry: 0.5 * total_stoichiometry.max(1.0),
        }];
        if let Some(species_id) = atp_pool.as_ref() {
            repair_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: 0.40 * total_stoichiometry.sqrt().max(1.0),
            });
        }
        if let Some(species_id) = amino_pool.as_ref() {
            repair_reactants.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
                stoichiometry: 0.25 * total_stoichiometry.sqrt().max(1.0),
            });
        }
        let mut repair_products = vec![WholeCellReactionParticipantSpec {
            species_id: mature_id.clone(),
            stoichiometry: 1.0,
        }];
        if let Some(species_id) = adp_pool.as_ref() {
            repair_products.push(WholeCellReactionParticipantSpec {
                species_id: species_id.clone(),
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
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
        reactions.push(WholeCellReactionSpec {
            id: format!("{}_turnover", canonical_species_fragment(&complex.id)),
            name: format!("{} turnover", complex.name),
            reaction_class: WholeCellReactionClass::ComplexTurnover,
            asset_class: complex.asset_class,
            nominal_rate: (0.02 + 0.010 * total_stoichiometry.sqrt()).clamp(0.01, 8.0),
            catalyst: None,
            operon: Some(complex.operon.clone()),
            reactants: vec![WholeCellReactionParticipantSpec {
                species_id: mature_id,
                stoichiometry: 1.0,
            }],
            products: vec![WholeCellReactionParticipantSpec {
                species_id: subunit_pool_id,
                stoichiometry: 1.0,
            }],
            subsystem_targets: complex.subsystem_targets.clone(),
            spatial_scope: registry_spatial_scope(
                complex.asset_class,
                complex_compartment,
                &complex.subsystem_targets,
            ),
        });
    }

    WholeCellGenomeProcessRegistry {
        organism: package.organism.clone(),
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
        }
    }
}

fn pool_concentration(pools: &[WholeCellMoleculePoolSpec], name: &str, fallback: f32) -> f32 {
    pools
        .iter()
        .find(|pool| pool.species.eq_ignore_ascii_case(name))
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
            atp: pool_concentration(&organism.pools, "ATP", 1.2),
            amino_acids: pool_concentration(&organism.pools, "amino_acids", 0.95),
            nucleotides: pool_concentration(&organism.pools, "nucleotides", 0.80),
            membrane_precursors: pool_concentration(&organism.pools, "membrane_precursors", 0.35),
        },
        initial_state: WholeCellInitialStateSpec {
            adp_mm: pool_concentration(&organism.pools, "ADP", 0.2),
            glucose_mm: pool_concentration(&organism.pools, "glucose", 1.0),
            oxygen_mm: pool_concentration(&organism.pools, "oxygen", 0.85),
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
    hydrate_program_spec(&mut spec)?;
    Ok(spec)
}

fn parse_bundle_manifest_json(
    manifest_json: &str,
) -> Result<WholeCellOrganismBundleManifest, String> {
    serde_json::from_str(manifest_json)
        .map_err(|error| format!("failed to parse organism bundle manifest: {error}"))
}

pub fn compile_organism_spec_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<WholeCellOrganismSpec, String> {
    let manifest_path = Path::new(manifest_path).canonicalize().map_err(|error| {
        format!("failed to resolve bundle manifest path {manifest_path}: {error}")
    })?;
    let manifest_json = read_text_file(&manifest_path, "bundle manifest")?;
    let manifest = parse_bundle_manifest_json(&manifest_json)?;

    if let Some(spec_relative) = manifest.organism_spec_json.as_deref() {
        let spec_path = resolve_manifest_relative_path(&manifest_path, spec_relative)?;
        return parse_organism_spec_json(&read_text_file(&spec_path, "organism spec JSON")?);
    }

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

    let transcription_units = if let Some(transcription_units_json) =
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

    let pools = if let Some(pools_json) = manifest.pools_json.as_deref() {
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

    Ok(WholeCellOrganismSpec {
        organism: manifest
            .organism
            .clone()
            .unwrap_or_else(|| metadata.organism.clone()),
        chromosome_length_bp: chromosome_length_bp.max(1),
        origin_bp: metadata.origin_bp.min(chromosome_length_bp.max(1)),
        terminus_bp: metadata.terminus_bp.min(chromosome_length_bp.max(1)),
        geometry: metadata.geometry,
        composition: metadata.composition,
        pools,
        genes,
        transcription_units,
    })
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
    build_program_spec_from_organism(organism, manifest.source_dataset.clone())
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
    let organism = compile_organism_spec_from_bundle_manifest_path(manifest_path)?;
    let assets = compile_genome_asset_package(&organism);
    serde_json::to_string_pretty(&assets)
        .map_err(|error| format!("failed to serialize compiled genome asset package: {error}"))
}

pub fn compile_genome_process_registry_json_from_bundle_manifest_path(
    manifest_path: &str,
) -> Result<String, String> {
    let organism = compile_organism_spec_from_bundle_manifest_path(manifest_path)?;
    let assets = compile_genome_asset_package(&organism);
    let registry = compile_genome_process_registry(&assets);
    serde_json::to_string_pretty(&registry)
        .map_err(|error| format!("failed to serialize compiled genome process registry: {error}"))
}

fn hydrate_program_spec(spec: &mut WholeCellProgramSpec) -> Result<(), String> {
    if spec.organism_data.is_none() {
        if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_data = Some(resolve_bundled_organism_spec(reference)?);
        }
    }
    if spec.organism_assets.is_none() {
        if let Some(organism) = spec.organism_data.as_ref() {
            spec.organism_assets = Some(compile_genome_asset_package(organism));
        } else if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_assets = Some(resolve_bundled_genome_asset_package(reference)?);
        }
    }
    if spec.organism_process_registry.is_none() {
        if let Some(assets) = spec.organism_assets.as_ref() {
            spec.organism_process_registry = Some(compile_genome_process_registry(assets));
        }
    }
    populate_program_contract_metadata(spec)?;
    Ok(())
}

pub fn parse_program_spec_json(spec_json: &str) -> Result<WholeCellProgramSpec, String> {
    let mut spec: WholeCellProgramSpec = serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse program spec: {error}"))?;
    hydrate_program_spec(&mut spec)?;
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
    serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse organism spec: {error}"))
}

pub fn parse_genome_asset_package_json(
    spec_json: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse genome asset package: {error}"))
}

pub fn bundled_syn3a_organism_spec_json() -> &'static str {
    BUNDLED_SYN3A_ORGANISM_JSON
}

pub fn bundled_syn3a_organism_spec() -> Result<WholeCellOrganismSpec, String> {
    static BUNDLED_ORGANISM: OnceLock<Result<WholeCellOrganismSpec, String>> = OnceLock::new();
    BUNDLED_ORGANISM
        .get_or_init(|| parse_organism_spec_json(BUNDLED_SYN3A_ORGANISM_JSON))
        .clone()
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
        .get_or_init(|| {
            bundled_syn3a_organism_spec().map(|organism| compile_genome_asset_package(&organism))
        })
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

pub fn parse_saved_state_json(state_json: &str) -> Result<WholeCellSavedState, String> {
    let mut state: WholeCellSavedState = serde_json::from_str(state_json)
        .map_err(|error| format!("failed to parse saved state: {error}"))?;
    if state.organism_process_registry.is_none() {
        if let Some(assets) = state.organism_assets.as_ref() {
            state.organism_process_registry = Some(compile_genome_process_registry(assets));
        }
    }
    populate_saved_state_contract_metadata(&mut state)?;
    Ok(state)
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
        assert!(assets.operons.len() >= 4);
        assert_eq!(assets.rnas.len(), organism.genes.len());
        assert_eq!(assets.proteins.len(), organism.genes.len());
        assert!(assets.complexes.len() >= 4);
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
    fn bundled_syn3a_genome_asset_package_compiles_from_descriptor() {
        let organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let summary = WholeCellGenomeAssetSummary::from(&package);

        assert_eq!(package.organism, organism.organism);
        assert_eq!(package.rnas.len(), organism.genes.len());
        assert_eq!(package.proteins.len(), organism.genes.len());
        assert!(package.complexes.len() >= organism.transcription_units.len());
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
        assert!(package.operons.iter().any(|operon| operon.polycistronic));
        assert!(package
            .complexes
            .iter()
            .any(|complex| !complex.subsystem_targets.is_empty()));
    }

    #[test]
    fn bundled_syn3a_process_registry_compiles_from_assets() {
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let registry = compile_genome_process_registry(&package);
        let summary = WholeCellGenomeProcessRegistrySummary::from(&registry);

        assert_eq!(registry.organism, "JCVI-syn3A");
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
        assert_eq!(summary.turnover_reaction_count, package.complexes.len());
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
            .reactions
            .iter()
            .any(|reaction| reaction.reaction_class == WholeCellReactionClass::PoolTransport));
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
    }

    #[test]
    fn saved_state_json_hydrates_contract_defaults() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = WholeCellSavedState {
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
        };

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
    fn compile_syn3a_bundle_manifest_path_hydrates_program_spec() {
        let manifest_path = bundle_manifest_path("jcvi_syn3a");
        let spec =
            compile_program_spec_from_bundle_manifest_path(&manifest_path).expect("compiled spec");
        let organism = spec.organism_data.as_ref().expect("compiled organism");
        let assets = spec.organism_assets.as_ref().expect("compiled assets");

        assert_eq!(organism.organism, "JCVI-syn3A");
        assert_eq!(assets.proteins.len(), organism.genes.len());
        assert!(assets.complexes.len() >= organism.transcription_units.len());
        assert_eq!(
            spec.provenance.source_dataset.as_deref(),
            Some("bundled_syn3a_organism_spec")
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
        assert_eq!(assets.proteins.len(), 4);
        assert!(assets.operons.iter().any(|operon| operon.polycistronic));
        assert!(registry.species.len() > assets.proteins.len());
        assert!(registry.reactions.len() >= assets.proteins.len());
    }
}
