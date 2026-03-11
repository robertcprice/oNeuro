//! oNeuro-Metal -- GPU-accelerated molecular brain simulator.
//!
//! The world's first Metal-powered biophysical neural engine where every behavior
//! -- learning, memory, drug response, consciousness, sleep -- EMERGES from
//! molecular biochemistry simulated on Apple Silicon GPU.
//!
//! # Architecture
//!
//! All neuron state is stored in Structure-of-Arrays (`NeuronArrays`) for GPU-coalesced
//! memory access. Synaptic connectivity uses Compressed Sparse Row (`SynapseArrays`)
//! for efficient spike propagation. The simulation step pipeline is:
//!
//! 1. **HH gating** (GPU): advance Na_v, K_v, Ca_v gating variables via forward-Euler
//! 2. **Receptor binding** (GPU): compute ligand-gated channel open fractions via Hill eq
//! 3. **Membrane integration** (GPU): sum ionic currents, Euler-integrate dV/dt, spike detect
//! 4. **Calcium dynamics** (GPU): 4-compartment calcium ODE (microdomain/cyto/ER/mito)
//! 5. **Second messengers** (GPU): cAMP/PKA/PKC/CaMKII/CREB/MAPK cascades
//! 6. **Spike propagation** (CPU): vesicle release + PSC injection for fired neurons only
//! 7. **STDP** (CPU): receptor trafficking for synapses touching fired neurons
//! 8. **Cleft dynamics** (CPU): NT clearance via enzymatic degradation and reuptake
//! 9. **Interval-gated subsystems** (CPU): gene expression, metabolism, microtubules, glia
//!
//! Every step on CPU has a GPU dispatch counterpart (macOS/Metal) and a scalar CPU fallback
//! (all platforms). The decision is automatic: GPU for >= 64 neurons, CPU otherwise.
//!
//! # Crate Features
//!
//! - `python` (default): Builds PyO3 extension module for Python bindings.
//! - `cuda`: Enables NVIDIA CUDA backend for cross-platform GPU acceleration.

// ===== Core data structures =====
pub mod constants;
pub mod neuron_arrays;
pub mod synapse_arrays;
pub mod types;

// ===== GPU compute pipelines =====
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod gpu;

// ===== Per-step compute phases (CPU implementations) =====
pub mod spike_propagation;
pub mod stdp;

// ===== Interval-gated subsystems =====
pub mod cellular_metabolism;
pub mod circadian;
pub mod gene_expression;
pub mod glia;
pub mod metabolism;
pub mod microtubules;
pub mod molecular_atmosphere;
pub mod pharmacology;
pub mod plant_cellular;
pub mod plant_organism;
pub mod substrate_ir;
pub mod terrarium;
pub mod terrarium_field;
pub mod terrarium_world;
pub mod whole_cell;
pub mod whole_cell_data;
pub mod whole_cell_submodels;

// ===== High-level orchestration =====
pub mod atomistic_topology;
pub mod brain_regions;
pub mod consciousness;
pub mod dishbrain_pong;
pub mod network;

// ===== Retinal processing =====
pub mod retina;

// ===== Organism simulators =====
pub mod doom_brain;
pub mod drosophila;
pub mod ecology_events;
pub mod ecology_fields;
pub mod molecular_dynamics;
pub mod neural_molecular_simulator;
pub mod soil_broad;
pub mod soil_uptake;

// ===== CUDA simulation modules =====
#[cfg(feature = "cuda")]
pub mod body;
#[cfg(feature = "cuda")]
pub mod fep;
#[cfg(feature = "cuda")]
pub mod motor;
#[cfg(feature = "cuda")]
pub mod sensory;
#[cfg(feature = "cuda")]
pub mod world;

// ===== Python bindings =====
#[cfg(feature = "python")]
pub mod python;

// ===== Re-exports =====
pub use atomistic_topology::{
    atomistic_assembly_templates, atomistic_template_for_site_name, AtomisticAssemblyTemplate,
    AtomisticTemplateDescriptor,
};
pub use brain_regions::{BrainRegion, RegionalBrain};
pub use cellular_metabolism::CellularMetabolismSim;
pub use consciousness::{ConsciousnessMetrics, ConsciousnessMonitor};
pub use constants::{clamp, hill, michaelis_menten};
pub use dishbrain_pong::{DishBrainPongResult, DishBrainPongSim, PongScale};
pub use doom_brain::{DDAController, DoomBrainSim, DoomEngine, DoomExperimentResult, DoomMode};
pub use drosophila::{DrosophilaScale, DrosophilaSim, ExperimentResult};
pub use ecology_events::{
    step_food_patches, step_seed_bank, FoodPatchStepResult, SeedBankStepResult,
};
pub use ecology_fields::{build_dual_radial_fields, build_radial_field, RadialSource};
pub use gpu::whole_cell_rdme::{IntracellularLattice, IntracellularSpecies};
pub use molecular_atmosphere::{
    odorant_channel_params, step_molecular_atmosphere_fields, step_molecular_world_fields,
    FruitSourceState, OdorantChannelParams, PlantSourceState, WaterSourceState,
};
pub use molecular_dynamics::{GPUMolecularDynamics, MDStats};
pub use network::MolecularBrain;
pub use neural_molecular_simulator::NeuralMolecularSimulator;
pub use plant_cellular::{
    PlantCellularFeedback, PlantCellularStateSim, PlantClusterSnapshot, PlantTissue,
};
pub use plant_organism::{PlantOrganismSim, PlantStepReport};
pub use retina::MolecularRetina;
pub use soil_broad::{step_soil_broad_pools, SoilBroadStepResult};
pub use soil_uptake::{extract_root_resources_with_layers, SoilResourceExtraction};
pub use substrate_ir::{
    evaluate_patch_assembly, execute_patch_reaction, localize_patch, AssemblyComponent,
    AssemblyContext, AssemblyRule, AssemblyState, FluxChannel, LocalizationCue, LocalizationRule,
    LocalizedPatch, ReactionContext, ReactionDriver, ReactionFlux, ReactionLaw, ReactionRule,
    ReactionTerm, SpatialChannel, EMPTY_ASSEMBLY_COMPONENT, EMPTY_LOCALIZATION_CUE,
    EMPTY_REACTION_TERM,
};
pub use terrarium::{BatchedAtomTerrarium, TerrariumBackend, TerrariumSnapshot, TerrariumSpecies};
pub use terrarium_field::{FlySensorySample, TerrariumSensoryField};
pub use terrarium_world::{
    TerrariumFruitPatch, TerrariumPlant, TerrariumPlantGenome, TerrariumSeed, TerrariumTopdownView,
    TerrariumWorld, TerrariumWorldConfig, TerrariumWorldSnapshot,
};
pub use types::*;
pub use whole_cell::{
    WholeCellBackend, WholeCellConfig, WholeCellQuantumProfile, WholeCellSimulator,
    WholeCellSnapshot,
};
pub use whole_cell_data::{
    bundled_syn3a_genome_asset_package, bundled_syn3a_genome_asset_package_json,
    bundled_syn3a_organism_spec, bundled_syn3a_organism_spec_json, bundled_syn3a_process_registry,
    bundled_syn3a_process_registry_json, bundled_syn3a_program_spec,
    bundled_syn3a_program_spec_json, compile_genome_asset_package,
    compile_genome_process_registry, compile_genome_process_registry_json_from_bundle_manifest_path,
    default_syn3a_seed_spec, derive_organism_profile, parse_genome_asset_package_json,
    parse_genome_process_registry_json, parse_organism_spec_json, parse_program_spec_json,
    parse_saved_state_json, resolve_bundled_genome_asset_package,
    resolve_bundled_genome_process_registry, resolve_bundled_organism_spec, saved_state_to_json,
    WholeCellAssetClass, WholeCellCheckpoint, WholeCellComplexAssemblyState,
    WholeCellComplexComponentSpec, WholeCellComplexSpec, WholeCellCompositionPrior,
    WholeCellGenomeAssetPackage, WholeCellGenomeAssetSummary, WholeCellGenomeFeature,
    WholeCellGenomeProcessRegistry, WholeCellGenomeProcessRegistrySummary,
    WholeCellGeometryPrior, WholeCellInitialLatticeSpec, WholeCellInitialStateSpec,
    WholeCellLatticeState, WholeCellLocalChemistryConfig, WholeCellLocalChemistrySpec,
    WholeCellMoleculePoolSpec, WholeCellOperonSpec, WholeCellOrganismExpressionState,
    WholeCellOrganismProfile, WholeCellOrganismSpec, WholeCellOrganismSummary,
    WholeCellProcessWeights, WholeCellProgramSpec, WholeCellProteinProductSpec,
    WholeCellReactionClass, WholeCellReactionParticipantSpec, WholeCellReactionRuntimeState,
    WholeCellReactionSpec, WholeCellRnaProductSpec, WholeCellSavedCoreState,
    WholeCellSavedState, WholeCellSeedSpec, WholeCellSpeciesClass, WholeCellSpeciesRuntimeState,
    WholeCellSpeciesSpec, WholeCellTranscriptionUnitSpec, WholeCellTranscriptionUnitState,
};
pub use whole_cell_submodels::{
    default_whole_cell_derivation_calibration, LocalChemistryReport, LocalChemistrySiteReport,
    LocalMDProbeReport, LocalMDProbeRequest, ScheduledSubsystemProbe, Syn3ASubsystemPreset,
    WholeCellChemistryBridge, WholeCellChemistrySite, WholeCellDerivationCalibration,
    WholeCellDerivationCalibrationFit, WholeCellDerivationCalibrationSample,
    WholeCellSubsystemState,
};
