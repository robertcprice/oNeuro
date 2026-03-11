//! PyO3 Python bindings for oNeuro-Metal.
//!
//! Exposes `MolecularBrain` and `RegionalBrain` to Python with zero-copy
//! numpy array access via Apple Silicon unified memory.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

use crate::brain_regions::RegionalBrain;
use crate::cellular_metabolism::CellularMetabolismSim;
use crate::consciousness::{ConsciousnessMetrics, ConsciousnessMonitor};
use crate::ecology_events::{
    step_food_patches as rust_step_food_patches, step_seed_bank as rust_step_seed_bank,
};
use crate::ecology_fields::{
    build_dual_radial_fields as rust_build_dual_radial_fields,
    build_radial_field as rust_build_radial_field,
};
use crate::gpu::whole_cell_rdme::IntracellularSpecies;
use crate::molecular_atmosphere::{
    odorant_channel_params, step_molecular_world_fields, FruitSourceState, OdorantChannelParams,
    PlantSourceState, WaterSourceState,
};
use crate::molecular_dynamics;
use crate::network::MolecularBrain;
use crate::neural_molecular_simulator;
use crate::plant_cellular::{PlantCellularStateSim, PlantTissue};
use crate::plant_organism::PlantOrganismSim;
use crate::soil_broad::step_soil_broad_pools as rust_step_soil_broad_pools;
use crate::soil_uptake::extract_root_resources_with_layers as rust_extract_root_resources_with_layers;
use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies};
use crate::terrarium_field::TerrariumSensoryField;
use crate::types::*;
use crate::whole_cell::{WholeCellConfig, WholeCellQuantumProfile, WholeCellSimulator};
use crate::whole_cell_data::{
    compile_genome_asset_package_json_from_bundle_manifest_path,
    compile_organism_spec_json_from_bundle_manifest_path,
    compile_program_spec_json_from_bundle_manifest_path,
};
use crate::whole_cell_submodels::{
    LocalMDProbeRequest, Syn3ASubsystemPreset, WholeCellChemistrySite,
};

fn parse_drug_name(drug_name: &str) -> PyResult<DrugType> {
    match drug_name.to_lowercase().as_str() {
        "fluoxetine" | "prozac" => Ok(DrugType::Fluoxetine),
        "diazepam" | "valium" => Ok(DrugType::Diazepam),
        "caffeine" => Ok(DrugType::Caffeine),
        "amphetamine" => Ok(DrugType::Amphetamine),
        "ldopa" | "l-dopa" | "levodopa" => Ok(DrugType::LDOPA),
        "donepezil" | "aricept" => Ok(DrugType::Donepezil),
        "ketamine" => Ok(DrugType::Ketamine),
        _ => Err(PyValueError::new_err(format!(
            "Unknown drug: {}",
            drug_name
        ))),
    }
}

fn validate_neuron_indices(count: usize, neuron_indices: &[usize]) -> PyResult<()> {
    for &idx in neuron_indices {
        if idx >= count {
            return Err(PyValueError::new_err(format!(
                "Neuron index out of range: {} >= {}",
                idx, count
            )));
        }
    }
    Ok(())
}

fn validate_synapse_indices(count: usize, synapse_indices: &[usize]) -> PyResult<()> {
    for &idx in synapse_indices {
        if idx >= count {
            return Err(PyValueError::new_err(format!(
                "Synapse index out of range: {} >= {}",
                idx, count
            )));
        }
    }
    Ok(())
}

fn validate_nt_index(nt: usize) -> PyResult<usize> {
    if nt >= NTType::COUNT {
        return Err(PyValueError::new_err("NT index out of range (0-5)"));
    }
    Ok(nt)
}

fn build_synapse_pre(row_offsets: &[u32], n_synapses: usize) -> Vec<u32> {
    let mut pre = Vec::with_capacity(n_synapses);
    for neuron_idx in 0..row_offsets.len().saturating_sub(1) {
        let start = row_offsets[neuron_idx] as usize;
        let end = row_offsets[neuron_idx + 1] as usize;
        for _ in start..end {
            pre.push(neuron_idx as u32);
        }
    }
    pre
}

fn get_required_dict_item<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing key '{key}'")))
}

fn buffer_to_vec_f32(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    expected_len: usize,
    name: &str,
) -> PyResult<Vec<f32>> {
    if let Ok(buffer) = PyBuffer::<f32>::get(obj) {
        if buffer.item_count() != expected_len {
            return Err(PyValueError::new_err(format!(
                "{name} length mismatch: expected {expected_len}, got {}",
                buffer.item_count()
            )));
        }
        let mut values = vec![0.0f32; expected_len];
        buffer
            .copy_to_slice(py, &mut values)
            .map_err(PyValueError::new_err)?;
        return Ok(values);
    }
    if let Ok(buffer) = PyBuffer::<f64>::get(obj) {
        if buffer.item_count() != expected_len {
            return Err(PyValueError::new_err(format!(
                "{name} length mismatch: expected {expected_len}, got {}",
                buffer.item_count()
            )));
        }
        let mut values64 = vec![0.0f64; expected_len];
        buffer
            .copy_to_slice(py, &mut values64)
            .map_err(PyValueError::new_err)?;
        return Ok(values64.into_iter().map(|v| v as f32).collect());
    }
    Err(PyValueError::new_err(format!(
        "{name} must expose a contiguous float32/float64 buffer"
    )))
}

fn add_buffer_into_sum(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    accum: &mut [f32],
    name: &str,
) -> PyResult<()> {
    let values = buffer_to_vec_f32(py, obj, accum.len(), name)?;
    for (dst, value) in accum.iter_mut().zip(values.into_iter()) {
        *dst += value;
    }
    Ok(())
}

fn world_food_patches_flat(world: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let patches = world.getattr("food_patches")?;
    let mut flat = Vec::new();
    for patch in patches.try_iter()? {
        let patch = patch?;
        let patch = patch.cast::<PyDict>()?;
        flat.push(get_required_dict_item(&patch, "x")?.extract::<f32>()?);
        flat.push(get_required_dict_item(&patch, "y")?.extract::<f32>()?);
        flat.push(get_required_dict_item(&patch, "radius")?.extract::<f32>()?);
        flat.push(get_required_dict_item(&patch, "remaining")?.extract::<f32>()?);
    }
    Ok(flat)
}

fn extract_odorant_profile(
    profile: &Bound<'_, PyAny>,
    channel_map: &HashMap<String, usize>,
) -> PyResult<Vec<(usize, f32)>> {
    let dict = profile.cast::<PyDict>()?;
    let mut entries = Vec::new();
    for (key, value) in dict.iter() {
        let name = key.extract::<String>()?;
        if let Some(&channel_idx) = channel_map.get(&name) {
            entries.push((channel_idx, value.extract::<f32>()?));
        }
    }
    Ok(entries)
}

fn sync_odorant_profile(
    py: Python<'_>,
    profile: &Bound<'_, PyAny>,
    entries: &[(usize, f32)],
    channel_keys: &[Py<PyAny>],
) -> PyResult<()> {
    let dict = profile.cast::<PyDict>()?;
    for &(channel_idx, value) in entries {
        if let Some(key) = channel_keys.get(channel_idx) {
            dict.set_item(key.bind(py), value)?;
        }
    }
    Ok(())
}

fn write_vec_to_buffer_f32(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    values: &[f32],
    name: &str,
) -> PyResult<()> {
    if let Ok(buffer) = PyBuffer::<f32>::get(obj) {
        if buffer.item_count() != values.len() {
            return Err(PyValueError::new_err(format!(
                "{name} length mismatch: expected {}, got {}",
                values.len(),
                buffer.item_count()
            )));
        }
        let cells = buffer.as_mut_slice(py).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name} must expose a writable contiguous float32 buffer"
            ))
        })?;
        for (cell, value) in cells.iter().zip(values.iter().copied()) {
            cell.set(value);
        }
        return Ok(());
    }
    if let Ok(buffer) = PyBuffer::<f64>::get(obj) {
        if buffer.item_count() != values.len() {
            return Err(PyValueError::new_err(format!(
                "{name} length mismatch: expected {}, got {}",
                values.len(),
                buffer.item_count()
            )));
        }
        let cells = buffer.as_mut_slice(py).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name} must expose a writable contiguous float64 buffer"
            ))
        })?;
        for (cell, value) in cells.iter().zip(values.iter().copied()) {
            cell.set(value as f64);
        }
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} must expose a writable contiguous float32/float64 buffer"
    )))
}

#[pyfunction]
fn build_dual_radial_fields(
    width: usize,
    height: usize,
    canopy_sources: Vec<f32>,
    root_sources: Vec<f32>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    rust_build_dual_radial_fields(width, height, &canopy_sources, &root_sources)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn step_molecular_atmosphere(py: Python<'_>, world: &Bound<'_, PyAny>, dt: f32) -> PyResult<()> {
    let width = world.getattr("W")?.extract::<usize>()?;
    let height = world.getattr("H")?.extract::<usize>()?;
    let depth = world.getattr("D")?.extract::<usize>()?.max(1);
    let is_3d = world.getattr("is_3d")?.extract::<bool>()?;
    let cell_size_mm = world.getattr("cell_size_mm")?.extract::<f32>()?;
    let solar_factor = world
        .getattr("_solar_angle_factor")?
        .call0()?
        .extract::<f32>()?;
    let total = width * height * depth;

    let water_sources_obj = world.getattr("water_sources")?;
    let mut water_sources = Vec::new();
    for source in water_sources_obj.try_iter()? {
        let source = source?;
        if !source.getattr("alive")?.extract::<bool>()? {
            continue;
        }
        let x = source.getattr("x")?.extract::<usize>()?;
        let y = source.getattr("y")?.extract::<usize>()?;
        let z = source.getattr("z")?.extract::<usize>()?;
        water_sources.push((x, y, z));
    }

    let odorant_grids_obj = world.getattr("odorant_grids")?;
    let odorant_grids = odorant_grids_obj.cast::<PyDict>()?;
    let mut odorant_grid_objs: Vec<Py<PyAny>> = Vec::new();
    let mut channel_keys: Vec<Py<PyAny>> = Vec::new();
    let mut channel_map: HashMap<String, usize> = HashMap::new();
    let mut odorants = Vec::new();
    let mut odorant_params: Vec<OdorantChannelParams> = Vec::new();
    for (key, grid) in odorant_grids.iter() {
        let name = key.extract::<String>()?;
        let params = odorant_channel_params(&name).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported odorant channel '{name}'"))
        })?;
        let channel_idx = odorant_params.len();
        odorant_params.push(params);
        channel_map.insert(name, channel_idx);
        channel_keys.push(key.unbind());
        odorants.push(buffer_to_vec_f32(py, &grid, total, "odorant_grid")?);
        odorant_grid_objs.push(grid.unbind());
    }

    let ammonia_channel = channel_map.iter().find_map(|(name, idx)| {
        name.to_ascii_lowercase()
            .contains("ammonia")
            .then_some(*idx)
    });

    let fruit_sources_obj = world.getattr("fruit_sources")?;
    let mut fruit_objs: Vec<Py<PyAny>> = Vec::new();
    let mut fruits = Vec::new();
    for fruit in fruit_sources_obj.try_iter()? {
        let fruit = fruit?;
        let state = FruitSourceState {
            x: fruit.getattr("x")?.extract::<usize>()?,
            y: fruit.getattr("y")?.extract::<usize>()?,
            z: fruit.getattr("z")?.extract::<usize>()?,
            ripeness: fruit.getattr("ripeness")?.extract::<f32>()?,
            sugar_content: fruit.getattr("sugar_content")?.extract::<f32>()?,
            odorant_emission_rate: fruit.getattr("odorant_emission_rate")?.extract::<f32>()?,
            decay_rate: fruit.getattr("decay_rate")?.extract::<f32>()?,
            alive: fruit.getattr("alive")?.extract::<bool>()?,
            odorant_profile: extract_odorant_profile(
                &fruit.getattr("odorant_profile")?,
                &channel_map,
            )?,
        };
        fruits.push(state);
        fruit_objs.push(fruit.unbind());
    }

    let plant_sources_obj = world.getattr("plant_sources")?;
    let mut plants = Vec::new();
    for plant in plant_sources_obj.try_iter()? {
        let plant = plant?;
        plants.push(PlantSourceState {
            x: plant.getattr("x")?.extract::<usize>()?,
            y: plant.getattr("y")?.extract::<usize>()?,
            emission_z: plant.getattr("emission_z")?.extract::<usize>()?,
            odorant_emission_rate: plant.getattr("odorant_emission_rate")?.extract::<f32>()?,
            alive: plant.getattr("alive")?.extract::<bool>()?,
            odorant_profile: extract_odorant_profile(
                &plant.getattr("odorant_profile")?,
                &channel_map,
            )?,
        });
    }

    let water_sources_obj = world.getattr("water_sources")?;
    let mut water_objs: Vec<Py<PyAny>> = Vec::new();
    let mut waters = Vec::new();
    for water in water_sources_obj.try_iter()? {
        let water = water?;
        waters.push(WaterSourceState {
            x: water.getattr("x")?.extract::<usize>()?,
            y: water.getattr("y")?.extract::<usize>()?,
            z: water.getattr("z")?.extract::<usize>()?,
            volume: water.getattr("volume")?.extract::<f32>()?,
            evaporation_rate: water.getattr("evaporation_rate")?.extract::<f32>()?,
            alive: water.getattr("alive")?.extract::<bool>()?,
        });
        water_objs.push(water.unbind());
    }

    let temperature_obj = world.getattr("temperature")?;
    let humidity_obj = world.getattr("humidity")?;
    let wind_x_obj = world.getattr("wind_vx")?;
    let wind_y_obj = world.getattr("wind_vy")?;
    let wind_z_obj = world.getattr("wind_vz")?;
    let rng_state_obj = world.getattr("_rust_world_rng_state")?;
    let mut rng_state = rng_state_obj.extract::<u64>()?;
    let mut temperature = buffer_to_vec_f32(py, &temperature_obj, total, "temperature")?;
    let mut humidity = buffer_to_vec_f32(py, &humidity_obj, total, "humidity")?;
    let mut wind_x = buffer_to_vec_f32(py, &wind_x_obj, total, "wind_vx")?;
    let mut wind_y = buffer_to_vec_f32(py, &wind_y_obj, total, "wind_vy")?;
    let mut wind_z = buffer_to_vec_f32(py, &wind_z_obj, total, "wind_vz")?;

    step_molecular_world_fields(
        width,
        height,
        depth,
        dt,
        cell_size_mm,
        is_3d,
        solar_factor,
        ammonia_channel,
        &mut fruits,
        &plants,
        &mut waters,
        &mut odorants,
        &odorant_params,
        &mut temperature,
        &mut humidity,
        &mut wind_x,
        &mut wind_y,
        &mut wind_z,
        &mut rng_state,
    )
    .map_err(PyValueError::new_err)?;

    write_vec_to_buffer_f32(py, &temperature_obj, &temperature, "temperature")?;
    write_vec_to_buffer_f32(py, &humidity_obj, &humidity, "humidity")?;
    write_vec_to_buffer_f32(py, &wind_x_obj, &wind_x, "wind_vx")?;
    write_vec_to_buffer_f32(py, &wind_y_obj, &wind_y, "wind_vy")?;
    write_vec_to_buffer_f32(py, &wind_z_obj, &wind_z, "wind_vz")?;
    for (grid_obj, values) in odorant_grid_objs.into_iter().zip(odorants.iter()) {
        let bound = grid_obj.bind(py);
        write_vec_to_buffer_f32(py, bound.as_any(), values, "odorant_grid")?;
    }
    for (fruit_obj, fruit) in fruit_objs.into_iter().zip(fruits.iter()) {
        let bound = fruit_obj.bind(py);
        bound.setattr("ripeness", fruit.ripeness)?;
        bound.setattr("sugar_content", fruit.sugar_content)?;
        bound.setattr("alive", fruit.alive)?;
        sync_odorant_profile(
            py,
            &bound.getattr("odorant_profile")?,
            &fruit.odorant_profile,
            &channel_keys,
        )?;
    }
    for (water_obj, water) in water_objs.into_iter().zip(waters.iter()) {
        let bound = water_obj.bind(py);
        bound.setattr("volume", water.volume)?;
        bound.setattr("alive", water.alive)?;
    }
    world.setattr("_rust_world_rng_state", rng_state)?;
    Ok(())
}

#[pyfunction]
fn build_radial_field(width: usize, height: usize, sources: Vec<f32>) -> PyResult<Vec<f32>> {
    rust_build_radial_field(width, height, &sources).map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_soil_broad_pools<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    dt: f32,
    light: f32,
    temp_factor: f32,
    water_mask: Vec<f32>,
    canopy_cover: Vec<f32>,
    root_density: Vec<f32>,
    moisture: Vec<f32>,
    deep_moisture: Vec<f32>,
    dissolved_nutrients: Vec<f32>,
    mineral_nitrogen: Vec<f32>,
    shallow_nutrients: Vec<f32>,
    deep_minerals: Vec<f32>,
    organic_matter: Vec<f32>,
    litter_carbon: Vec<f32>,
    microbial_biomass: Vec<f32>,
    symbiont_biomass: Vec<f32>,
    root_exudates: Vec<f32>,
    soil_structure: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = rust_step_soil_broad_pools(
        width,
        height,
        dt,
        light,
        temp_factor,
        &water_mask,
        &canopy_cover,
        &root_density,
        &moisture,
        &deep_moisture,
        &dissolved_nutrients,
        &mineral_nitrogen,
        &shallow_nutrients,
        &deep_minerals,
        &organic_matter,
        &litter_carbon,
        &microbial_biomass,
        &symbiont_biomass,
        &root_exudates,
        &soil_structure,
    )
    .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("moisture", result.moisture)?;
    dict.set_item("deep_moisture", result.deep_moisture)?;
    dict.set_item("dissolved_nutrients", result.dissolved_nutrients)?;
    dict.set_item("mineral_nitrogen", result.mineral_nitrogen)?;
    dict.set_item("shallow_nutrients", result.shallow_nutrients)?;
    dict.set_item("deep_minerals", result.deep_minerals)?;
    dict.set_item("organic_matter", result.organic_matter)?;
    dict.set_item("litter_carbon", result.litter_carbon)?;
    dict.set_item("microbial_biomass", result.microbial_biomass)?;
    dict.set_item("symbiont_biomass", result.symbiont_biomass)?;
    dict.set_item("root_exudates", result.root_exudates)?;
    dict.set_item("decomposition", result.decomposition)?;
    dict.set_item("mineralized", result.mineralized)?;
    dict.set_item("litter_used", result.litter_used)?;
    dict.set_item("exudate_used", result.exudate_used)?;
    dict.set_item("organic_used", result.organic_used)?;
    dict.set_item("microbial_turnover", result.microbial_turnover)?;
    dict.set_item("sym_turnover", result.sym_turnover)?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn extract_root_resources_with_layers<'py>(
    py: Python<'py>,
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    radius: i32,
    water_demand: f32,
    nutrient_demand: f32,
    deep_fraction: f32,
    symbiosis_factor: f32,
    root_respiration: f32,
    moisture: Vec<f32>,
    deep_moisture: Vec<f32>,
    dissolved_nutrients: Vec<f32>,
    mineral_nitrogen: Vec<f32>,
    shallow_nutrients: Vec<f32>,
    deep_minerals: Vec<f32>,
    symbiont_biomass: Vec<f32>,
    ammonium_rhizo: Vec<f32>,
    nitrate_rhizo: Vec<f32>,
    nitrate_deep: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = rust_extract_root_resources_with_layers(
        width,
        height,
        x,
        y,
        radius,
        water_demand,
        nutrient_demand,
        deep_fraction,
        symbiosis_factor,
        root_respiration,
        &moisture,
        &deep_moisture,
        &dissolved_nutrients,
        &mineral_nitrogen,
        &shallow_nutrients,
        &deep_minerals,
        &symbiont_biomass,
        &ammonium_rhizo,
        &nitrate_rhizo,
        &nitrate_deep,
    )
    .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("moisture", result.moisture)?;
    dict.set_item("deep_moisture", result.deep_moisture)?;
    dict.set_item("dissolved_nutrients", result.dissolved_nutrients)?;
    dict.set_item("mineral_nitrogen", result.mineral_nitrogen)?;
    dict.set_item("shallow_nutrients", result.shallow_nutrients)?;
    dict.set_item("deep_minerals", result.deep_minerals)?;
    dict.set_item("water_take", result.water_take)?;
    dict.set_item("nutrient_take", result.nutrient_take)?;
    dict.set_item("surface_water_take", result.surface_water_take)?;
    dict.set_item("deep_water_take", result.deep_water_take)?;
    dict.set_item("ammonium_take", result.ammonium_take)?;
    dict.set_item("rhizo_nitrate_take", result.rhizo_nitrate_take)?;
    dict.set_item("deep_nitrate_take", result.deep_nitrate_take)?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_food_patches<'py>(
    py: Python<'py>,
    dt: f32,
    patch_remaining: Vec<f32>,
    previous_remaining: Vec<f32>,
    deposited_all: Vec<bool>,
    has_fruit: Vec<bool>,
    fruit_ripeness: Vec<f32>,
    fruit_sugar_content: Vec<f32>,
    microbial_biomass: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = rust_step_food_patches(
        dt,
        &patch_remaining,
        &previous_remaining,
        &deposited_all,
        &has_fruit,
        &fruit_ripeness,
        &fruit_sugar_content,
        &microbial_biomass,
    )
    .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("remaining", result.remaining)?;
    dict.set_item("sugar_content", result.sugar_content)?;
    dict.set_item("fruit_alive", result.fruit_alive)?;
    dict.set_item("deposited_all", result.deposited_all)?;
    dict.set_item("decay_detritus", result.decay_detritus)?;
    dict.set_item("lost_detritus", result.lost_detritus)?;
    dict.set_item("final_detritus", result.final_detritus)?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn step_seed_bank<'py>(
    py: Python<'py>,
    dt: f32,
    light: f32,
    current_plant_count: usize,
    max_plants: usize,
    dormancy_s: Vec<f32>,
    age_s: Vec<f32>,
    reserve_carbon: Vec<f32>,
    symbiosis_affinity: Vec<f32>,
    shade_tolerance: Vec<f32>,
    moisture: Vec<f32>,
    deep_moisture: Vec<f32>,
    nutrients: Vec<f32>,
    symbionts: Vec<f32>,
    canopy: Vec<f32>,
    litter: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = rust_step_seed_bank(
        dt,
        light,
        current_plant_count,
        max_plants,
        &dormancy_s,
        &age_s,
        &reserve_carbon,
        &symbiosis_affinity,
        &shade_tolerance,
        &moisture,
        &deep_moisture,
        &nutrients,
        &symbionts,
        &canopy,
        &litter,
    )
    .map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("dormancy_s", result.dormancy_s)?;
    dict.set_item("age_s", result.age_s)?;
    dict.set_item("keep", result.keep)?;
    dict.set_item("germinate", result.germinate)?;
    dict.set_item("seedling_scale", result.seedling_scale)?;
    Ok(dict)
}

fn parse_intracellular_species(species: &str) -> PyResult<IntracellularSpecies> {
    match species.to_lowercase().as_str() {
        "atp" => Ok(IntracellularSpecies::ATP),
        "amino_acids" | "aa" => Ok(IntracellularSpecies::AminoAcids),
        "nucleotides" | "nt" => Ok(IntracellularSpecies::Nucleotides),
        "membrane_precursors" | "lipids" | "membrane" => {
            Ok(IntracellularSpecies::MembranePrecursors)
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown intracellular species: {}",
            species
        ))),
    }
}

fn parse_chemistry_site(site: &str) -> PyResult<WholeCellChemistrySite> {
    WholeCellChemistrySite::from_name(site).ok_or_else(|| {
        PyValueError::new_err(format!("Unknown whole-cell chemistry site: {}", site))
    })
}

fn parse_syn3a_subsystem_preset(name: &str) -> PyResult<Syn3ASubsystemPreset> {
    Syn3ASubsystemPreset::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown Syn3A subsystem preset: {}", name)))
}

fn parse_terrarium_species(species: &str) -> PyResult<TerrariumSpecies> {
    TerrariumSpecies::from_name(species)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown terrarium species: {}", species)))
}

fn parse_plant_tissue(tissue: &str) -> PyResult<PlantTissue> {
    PlantTissue::from_name(tissue)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown plant tissue: {}", tissue)))
}

// =============================================================================
// PyMolecularBrain
// =============================================================================

/// GPU-accelerated molecular brain simulator.
///
/// Every neuron is a complete Hodgkin-Huxley biophysical model with:
/// - 8 ion channel types (Na_v, K_v, K_leak, Ca_v, NMDA, AMPA, GABA-A, nAChR)
/// - 4-compartment calcium dynamics (cytoplasmic, ER, mitochondrial, microdomain)
/// - Full second messenger cascades (cAMP/PKA, PLC/IP3/DAG/PKC, CaMKII, ERK, CREB)
/// - Gene expression, metabolism, circadian biology
/// - STDP via receptor trafficking, BCM metaplasticity
/// - 7 psychoactive drugs + general anesthesia
/// - 7 consciousness metrics
#[pyclass(name = "MolecularBrain")]
pub struct PyMolecularBrain {
    inner: MolecularBrain,
    consciousness: Option<ConsciousnessMonitor>,
}

#[pymethods]
impl PyMolecularBrain {
    /// Create a new molecular brain with n_neurons.
    #[new]
    #[pyo3(signature = (n_neurons, psc_scale=30.0, dt=0.1))]
    fn new(n_neurons: usize, psc_scale: f32, dt: f32) -> Self {
        let mut brain = MolecularBrain::new(n_neurons);
        brain.psc_scale = psc_scale;
        brain.dt = dt;
        Self {
            inner: brain,
            consciousness: None,
        }
    }

    /// Create from edge list: [(pre, post, nt_type), ...].
    /// nt_type: 0=DA, 1=5HT, 2=NE, 3=ACh, 4=GABA, 5=Glutamate.
    #[staticmethod]
    #[pyo3(signature = (n_neurons, edges, psc_scale=30.0, dt=0.1))]
    fn from_edges(
        n_neurons: usize,
        edges: Vec<(u32, u32, u8)>,
        psc_scale: f32,
        dt: f32,
    ) -> PyResult<Self> {
        let typed_edges: Vec<(u32, u32, NTType)> = edges
            .into_iter()
            .map(|(pre, post, nt)| {
                let nt_type = match nt {
                    0 => NTType::Dopamine,
                    1 => NTType::Serotonin,
                    2 => NTType::Norepinephrine,
                    3 => NTType::Acetylcholine,
                    4 => NTType::GABA,
                    5 => NTType::Glutamate,
                    _ => return Err(PyValueError::new_err(format!("Invalid NT type: {}", nt))),
                };
                Ok((pre, post, nt_type))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut brain = MolecularBrain::from_edges(n_neurons, &typed_edges);
        brain.psc_scale = psc_scale;
        brain.dt = dt;
        Ok(Self {
            inner: brain,
            consciousness: None,
        })
    }

    /// Run a single simulation step (dt milliseconds of bio time).
    fn step(&mut self) {
        self.inner.step();
        if let Some(ref mut cm) = self.consciousness {
            self.inner.sync_shadow_from_gpu();
            cm.record(&self.inner.neurons);
        }
    }

    /// Run multiple steps.
    fn run(&mut self, steps: u64) {
        if self.consciousness.is_none() {
            self.inner.run(steps);
        } else {
            for _ in 0..steps {
                self.step();
            }
        }
    }

    /// Run multiple steps without forcing a host shadow sync at the end.
    fn run_without_sync(&mut self, steps: u64) {
        self.inner.run_without_sync(steps);
    }

    /// Current simulation time in milliseconds.
    #[getter]
    fn time(&self) -> f32 {
        self.inner.time
    }

    /// Step size in milliseconds of biological time.
    #[getter]
    fn dt(&self) -> f32 {
        self.inner.dt
    }

    /// Total steps completed.
    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count
    }

    /// Number of neurons.
    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.neurons.count
    }

    /// Number of synapses.
    #[getter]
    fn n_synapses(&self) -> usize {
        self.inner.synapses.n_synapses
    }

    /// PSC scale factor.
    #[getter]
    fn psc_scale(&self) -> f32 {
        self.inner.psc_scale
    }

    #[setter]
    fn set_psc_scale(&mut self, value: f32) {
        self.inner.psc_scale = value;
    }

    // ========================================================================
    // State access — return copies as Python lists (numpy views would need
    // unsafe lifetime management; for safety, we copy)
    // ========================================================================

    /// Membrane voltages (mV) for all neurons.
    fn voltages(&mut self) -> Vec<f32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.neurons.voltage.clone()
    }

    /// Previous-step membrane voltages (mV) for all neurons.
    fn prev_voltages(&mut self) -> Vec<f32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.neurons.prev_voltage.clone()
    }

    /// Which neurons fired this step (bool as u8).
    fn fired(&mut self) -> Vec<u8> {
        self.inner.sync_shadow_from_gpu();
        self.inner.neurons.fired.clone()
    }

    /// Indices of neurons that fired this step.
    fn fired_indices(&mut self) -> Vec<usize> {
        self.inner.sync_shadow_from_gpu();
        self.inner.neurons.fired_indices()
    }

    /// Calcium concentrations: returns (cytoplasmic, er, mito, microdomain) as 4 lists.
    fn calcium(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner.sync_shadow_from_gpu();
        (
            self.inner.neurons.ca_cytoplasmic.clone(),
            self.inner.neurons.ca_er.clone(),
            self.inner.neurons.ca_mitochondrial.clone(),
            self.inner.neurons.ca_microdomain.clone(),
        )
    }

    /// Second messenger activities: returns dict-like tuple.
    fn second_messengers(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner.sync_shadow_from_gpu();
        (
            self.inner.neurons.camp.clone(),
            self.inner.neurons.pka_activity.clone(),
            self.inner.neurons.pkc_activity.clone(),
            self.inner.neurons.camkii_activity.clone(),
            self.inner.neurons.erk_activity.clone(),
        )
    }

    /// Phosphorylation states: (ampa_p, kv_p, cav_p, creb_p).
    fn phosphorylation(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner.sync_shadow_from_gpu();
        (
            self.inner.neurons.ampa_p.clone(),
            self.inner.neurons.kv_p.clone(),
            self.inner.neurons.cav_p.clone(),
            self.inner.neurons.creb_p.clone(),
        )
    }

    /// Spike counts for all neurons.
    fn spike_counts(&mut self) -> Vec<u32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.neurons.spike_count.clone()
    }

    /// Inject external current into a neuron (µA/cm²).
    fn stimulate(&mut self, neuron_idx: usize, current_ua: f32) -> PyResult<()> {
        if neuron_idx >= self.inner.neurons.count {
            return Err(PyValueError::new_err("Neuron index out of range"));
        }
        self.inner.stimulate(neuron_idx, current_ua);
        Ok(())
    }

    /// Stimulate multiple neurons with the same current.
    fn stimulate_range(&mut self, start: usize, end: usize, current_ua: f32) -> PyResult<()> {
        if end > self.inner.neurons.count {
            return Err(PyValueError::new_err("Range exceeds neuron count"));
        }
        for i in start..end {
            self.inner.stimulate(i, current_ua);
        }
        Ok(())
    }

    /// Stimulate an explicit set of neurons with the same current.
    fn stimulate_many(&mut self, neuron_indices: Vec<usize>, current_ua: f32) -> PyResult<()> {
        validate_neuron_indices(self.inner.neurons.count, &neuron_indices)?;
        for idx in neuron_indices {
            self.inner.stimulate(idx, current_ua);
        }
        Ok(())
    }

    /// Stimulate neurons with per-neuron currents.
    fn stimulate_weighted(
        &mut self,
        neuron_indices: Vec<usize>,
        currents_ua: Vec<f32>,
    ) -> PyResult<()> {
        if neuron_indices.len() != currents_ua.len() {
            return Err(PyValueError::new_err(
                "neuron_indices and currents_ua must have the same length",
            ));
        }
        validate_neuron_indices(self.inner.neurons.count, &neuron_indices)?;
        for (idx, current) in neuron_indices.into_iter().zip(currents_ua.into_iter()) {
            self.inner.stimulate(idx, current);
        }
        Ok(())
    }

    // ========================================================================
    // Pharmacology
    // ========================================================================

    /// Apply a drug. drug_name: "fluoxetine", "diazepam", "caffeine",
    /// "amphetamine", "ldopa", "donepezil", "ketamine".
    fn apply_drug(&mut self, drug_name: &str, dose_mg: f32) -> PyResult<()> {
        let drug = parse_drug_name(drug_name)?;
        self.inner.apply_drug(drug, dose_mg);
        Ok(())
    }

    /// Apply general anesthesia (GABA-A↑, NMDA↓, AMPA↓, Na_v↓, K_leak↑).
    fn apply_anesthesia(&mut self) {
        self.inner.apply_anesthesia();
    }

    // ========================================================================
    // Consciousness
    // ========================================================================

    /// Enable consciousness monitoring.
    fn enable_consciousness(&mut self) {
        self.consciousness = Some(ConsciousnessMonitor::new(self.inner.neurons.count));
    }

    /// Compute consciousness metrics (requires enable_consciousness() first).
    fn consciousness_metrics(&mut self) -> PyResult<PyConsciousnessMetrics> {
        self.inner.sync_shadow_from_gpu();
        match &self.consciousness {
            Some(cm) => {
                let m = cm.compute(&self.inner.neurons, &self.inner.synapses);
                Ok(PyConsciousnessMetrics { inner: m })
            }
            None => Err(PyValueError::new_err("Call enable_consciousness() first")),
        }
    }

    // ========================================================================
    // Feature flags
    // ========================================================================

    /// Enable/disable GPU acceleration.
    fn set_gpu_enabled(&mut self, enabled: bool) {
        self.inner.enable_gpu = enabled;
    }

    /// Enable/disable glia simulation.
    fn set_glia_enabled(&mut self, enabled: bool) {
        self.inner.enable_glia = enabled;
    }

    /// Enable/disable circadian clock.
    fn set_circadian_enabled(&mut self, enabled: bool) {
        self.inner.enable_circadian = enabled;
    }

    /// Enable/disable pharmacology updates.
    fn set_pharmacology_enabled(&mut self, enabled: bool) {
        self.inner.enable_pharmacology = enabled;
    }

    /// Enable/disable interval-gated gene expression updates.
    fn set_gene_expression_enabled(&mut self, enabled: bool) {
        self.inner.enable_gene_expression = enabled;
    }

    /// Enable/disable interval-gated metabolism updates.
    fn set_metabolism_enabled(&mut self, enabled: bool) {
        self.inner.enable_metabolism = enabled;
    }

    /// Enable/disable interval-gated microtubule updates.
    fn set_microtubules_enabled(&mut self, enabled: bool) {
        self.inner.enable_microtubules = enabled;
    }

    /// Disable nonessential interval biology for latency-sensitive full-GPU stepping.
    fn enable_latency_benchmark_mode(&mut self) {
        self.inner.enable_latency_benchmark_mode();
    }

    /// Set the effective membrane capacitance used by voltage integration (µF/cm²).
    fn set_membrane_capacitance(&mut self, capacitance_uf: f32) -> PyResult<()> {
        if !capacitance_uf.is_finite() || capacitance_uf <= 0.0 {
            return Err(PyValueError::new_err(
                "membrane capacitance must be a finite positive value",
            ));
        }
        self.inner.membrane_capacitance_uf = capacitance_uf;
        Ok(())
    }

    /// Set the spike threshold used for threshold-crossing detection (mV).
    fn set_spike_threshold(&mut self, threshold_mv: f32) -> PyResult<()> {
        if !threshold_mv.is_finite() {
            return Err(PyValueError::new_err("spike threshold must be finite"));
        }
        self.inner.spike_threshold_mv = threshold_mv;
        Ok(())
    }

    /// Set the absolute refractory period enforced after each spike (ms).
    fn set_refractory_period(&mut self, refractory_period_ms: f32) -> PyResult<()> {
        if !refractory_period_ms.is_finite() || refractory_period_ms <= 0.0 {
            return Err(PyValueError::new_err(
                "refractory period must be a finite positive value",
            ));
        }
        self.inner.refractory_period_ms = refractory_period_ms;
        Ok(())
    }

    /// Override the base maximal conductance for a channel family (mS/cm²).
    fn set_channel_g_max(&mut self, channel: usize, g_max: f32) -> PyResult<()> {
        if channel >= IonChannelType::COUNT {
            return Err(PyValueError::new_err("Channel index out of range (0-7)"));
        }
        if !g_max.is_finite() || g_max < 0.0 {
            return Err(PyValueError::new_err(
                "channel g_max must be a finite nonnegative value",
            ));
        }
        self.inner.channel_g_max[channel] = g_max;
        Ok(())
    }

    /// Set the reversal potential for the passive leak channel (mV).
    fn set_kleak_reversal(&mut self, reversal_mv: f32) -> PyResult<()> {
        if !reversal_mv.is_finite() {
            return Err(PyValueError::new_err("leak reversal must be finite"));
        }
        self.inner.kleak_reversal_mv = reversal_mv;
        Ok(())
    }

    /// Force an explicit host shadow sync from the resident GPU buffers.
    fn sync_shadow_from_gpu(&mut self) {
        self.inner.sync_shadow_from_gpu();
    }

    /// Whether Metal GPU is available on this system.
    #[staticmethod]
    fn has_gpu() -> bool {
        crate::gpu::has_gpu()
    }

    /// Set conductance scale for a specific channel type on a neuron.
    /// channel: 0=Nav, 1=Kv, 2=Kleak, 3=Cav, 4=NMDA, 5=AMPA, 6=GabaA, 7=nAChR.
    fn set_conductance_scale(
        &mut self,
        neuron_idx: usize,
        channel: usize,
        scale: f32,
    ) -> PyResult<()> {
        if neuron_idx >= self.inner.neurons.count {
            return Err(PyValueError::new_err("Neuron index out of range"));
        }
        if channel >= IonChannelType::COUNT {
            return Err(PyValueError::new_err("Channel index out of range (0-7)"));
        }
        self.inner.set_conductance_scale(neuron_idx, channel, scale);
        Ok(())
    }

    /// Set neurotransmitter concentration for a neuron.
    /// nt: 0=DA, 1=5HT, 2=NE, 3=ACh, 4=GABA, 5=Glu.
    fn set_nt_concentration(
        &mut self,
        neuron_idx: usize,
        nt: usize,
        concentration_nm: f32,
    ) -> PyResult<()> {
        if neuron_idx >= self.inner.neurons.count {
            return Err(PyValueError::new_err("Neuron index out of range"));
        }
        if nt >= NTType::COUNT {
            return Err(PyValueError::new_err("NT index out of range (0-5)"));
        }
        self.inner
            .set_nt_concentration(neuron_idx, nt, concentration_nm);
        Ok(())
    }

    /// Add a neurotransmitter concentration delta to many neurons.
    fn add_nt_concentration_many(
        &mut self,
        neuron_indices: Vec<usize>,
        nt: usize,
        delta_nm: f32,
    ) -> PyResult<()> {
        let nt_idx = validate_nt_index(nt)?;
        validate_neuron_indices(self.inner.neurons.count, &neuron_indices)?;
        self.inner
            .add_nt_concentration_many(&neuron_indices, nt_idx, delta_nm);
        Ok(())
    }

    /// Hebbian weight nudge: strengthen relay→correct_motor, weaken relay→wrong_motor.
    ///
    /// Used by the FEP/DishBrain learning protocol. Delta should be scale-adaptive:
    /// typically 0.8 * max(1.0, (n_l5 / 200)^0.3).
    fn hebbian_nudge(
        &mut self,
        relay_ids: Vec<u32>,
        correct_ids: Vec<u32>,
        wrong_ids: Vec<u32>,
        delta: f32,
    ) {
        self.inner
            .hebbian_nudge(&relay_ids, &correct_ids, &wrong_ids, delta);
    }

    /// Get synapse strength for a specific synapse index.
    fn synapse_strength(&mut self, idx: usize) -> PyResult<f32> {
        if idx >= self.inner.synapses.n_synapses {
            return Err(PyValueError::new_err("Synapse index out of range"));
        }
        self.inner.sync_shadow_from_gpu();
        Ok(self.inner.synapses.strength[idx])
    }

    /// Effective synaptic weight for a specific synapse index.
    fn synapse_weight(&mut self, idx: usize) -> PyResult<f32> {
        if idx >= self.inner.synapses.n_synapses {
            return Err(PyValueError::new_err("Synapse index out of range"));
        }
        self.inner.sync_shadow_from_gpu();
        Ok(self.inner.synapses.weight[idx])
    }

    /// Presynaptic neuron index for every synapse.
    fn synapse_pre(&mut self) -> Vec<u32> {
        self.inner.sync_shadow_from_gpu();
        build_synapse_pre(
            &self.inner.synapses.row_offsets,
            self.inner.synapses.n_synapses,
        )
    }

    /// Postsynaptic neuron index for every synapse.
    fn synapse_post(&mut self) -> Vec<u32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.synapses.col_indices.clone()
    }

    /// Synaptic strengths for every synapse.
    fn synapse_strengths(&mut self) -> Vec<f32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.synapses.strength.clone()
    }

    /// Effective synaptic weights for every synapse.
    fn synapse_weights(&mut self) -> Vec<f32> {
        self.inner.sync_shadow_from_gpu();
        self.inner.synapses.weight.clone()
    }

    /// Adjust selected synapse strengths by a fixed delta.
    #[pyo3(signature = (synapse_indices, delta, min_strength=0.3, max_strength=8.0))]
    fn adjust_synapse_strengths(
        &mut self,
        synapse_indices: Vec<usize>,
        delta: f32,
        min_strength: f32,
        max_strength: f32,
    ) -> PyResult<()> {
        validate_synapse_indices(self.inner.synapses.n_synapses, &synapse_indices)?;
        self.inner
            .adjust_synapse_strengths(&synapse_indices, delta, min_strength, max_strength);
        Ok(())
    }

    /// Set selected synapse strengths explicitly.
    #[pyo3(signature = (synapse_indices, strengths, min_strength=0.3, max_strength=8.0))]
    fn set_synapse_strengths(
        &mut self,
        synapse_indices: Vec<usize>,
        strengths: Vec<f32>,
        min_strength: f32,
        max_strength: f32,
    ) -> PyResult<()> {
        if synapse_indices.len() != strengths.len() {
            return Err(PyValueError::new_err(
                "synapse_indices and strengths must have the same length",
            ));
        }
        validate_synapse_indices(self.inner.synapses.n_synapses, &synapse_indices)?;
        self.inner
            .set_synapse_strengths(&synapse_indices, &strengths, min_strength, max_strength);
        Ok(())
    }

    /// Whether GPU acceleration is currently available/active for this brain.
    fn gpu_active(&self) -> bool {
        self.inner.gpu_available()
    }

    /// Whether the step loop will actually dispatch to Metal shaders.
    fn gpu_dispatch_active(&self) -> bool {
        self.inner.gpu_dispatch_active()
    }

    /// GPU initialization error, if the Metal context failed to initialize.
    fn gpu_init_error(&self) -> Option<String> {
        self.inner.gpu_init_error().map(|err| err.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "MolecularBrain(n_neurons={}, n_synapses={}, time={:.1}ms, steps={}, gpu={})",
            self.inner.neurons.count,
            self.inner.synapses.n_synapses,
            self.inner.time,
            self.inner.step_count,
            Self::has_gpu(),
        )
    }
}

// =============================================================================
// PyRegionalBrain
// =============================================================================

/// Pre-wired brain with cortical columns, thalamus, hippocampus, basal ganglia.
#[pyclass(name = "RegionalBrain")]
pub struct PyRegionalBrain {
    inner: RegionalBrain,
    consciousness: Option<ConsciousnessMonitor>,
}

#[pymethods]
impl PyRegionalBrain {
    /// Create a minimal brain (75 neurons) for testing.
    #[staticmethod]
    #[pyo3(signature = (seed=42))]
    fn minimal(seed: u64) -> Self {
        Self {
            inner: RegionalBrain::minimal(seed),
            consciousness: None,
        }
    }

    /// Create an xlarge brain (1018 neurons, 6 cortical columns + subcortical).
    #[staticmethod]
    #[pyo3(signature = (seed=42))]
    fn xlarge(seed: u64) -> Self {
        Self {
            inner: RegionalBrain::xlarge(seed),
            consciousness: None,
        }
    }

    /// Create with N cortical columns.
    #[staticmethod]
    #[pyo3(signature = (n_columns, seed=42))]
    fn with_columns(n_columns: usize, seed: u64) -> Self {
        Self {
            inner: RegionalBrain::with_columns(n_columns, seed),
            consciousness: None,
        }
    }

    fn step(&mut self) {
        self.inner.step();
        if let Some(ref mut cm) = self.consciousness {
            self.inner.brain.sync_shadow_from_gpu();
            cm.record(&self.inner.brain.neurons);
        }
    }

    fn run(&mut self, steps: u64) {
        if self.consciousness.is_none() {
            self.inner.run(steps);
        } else {
            for _ in 0..steps {
                self.step();
            }
        }
    }

    /// Stimulate thalamus with external current.
    fn stimulate_thalamus(&mut self, current: f32) {
        self.inner.stimulate_thalamus(current);
    }

    /// Get region info: list of (name, region_type, neuron_count).
    fn regions(&self) -> Vec<(String, u8, usize)> {
        self.inner
            .regions
            .iter()
            .map(|r| (r.name.clone(), r.region_type as u8, r.neuron_indices.len()))
            .collect()
    }

    /// Access the underlying MolecularBrain.
    fn voltages(&mut self) -> Vec<f32> {
        self.inner.brain.sync_shadow_from_gpu();
        self.inner.brain.neurons.voltage.clone()
    }

    fn fired_indices(&mut self) -> Vec<usize> {
        self.inner.brain.sync_shadow_from_gpu();
        self.inner.brain.neurons.fired_indices()
    }

    fn spike_counts(&mut self) -> Vec<u32> {
        self.inner.brain.sync_shadow_from_gpu();
        self.inner.brain.neurons.spike_count.clone()
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.brain.neurons.count
    }

    #[getter]
    fn n_synapses(&self) -> usize {
        self.inner.brain.synapses.n_synapses
    }

    #[getter]
    fn time(&self) -> f32 {
        self.inner.brain.time
    }

    #[getter]
    fn dt(&self) -> f32 {
        self.inner.brain.dt
    }

    fn enable_consciousness(&mut self) {
        self.consciousness = Some(ConsciousnessMonitor::new(self.inner.brain.neurons.count));
    }

    fn consciousness_metrics(&mut self) -> PyResult<PyConsciousnessMetrics> {
        self.inner.brain.sync_shadow_from_gpu();
        match &self.consciousness {
            Some(cm) => {
                let m = cm.compute(&self.inner.brain.neurons, &self.inner.brain.synapses);
                Ok(PyConsciousnessMetrics { inner: m })
            }
            None => Err(PyValueError::new_err("Call enable_consciousness() first")),
        }
    }

    fn apply_drug(&mut self, drug_name: &str, dose_mg: f32) -> PyResult<()> {
        let drug = parse_drug_name(drug_name)?;
        self.inner.brain.apply_drug(drug, dose_mg);
        Ok(())
    }

    fn apply_anesthesia(&mut self) {
        self.inner.brain.apply_anesthesia();
    }

    fn stimulate(&mut self, neuron_idx: usize, current_ua: f32) -> PyResult<()> {
        if neuron_idx >= self.inner.brain.neurons.count {
            return Err(PyValueError::new_err("Neuron index out of range"));
        }
        self.inner.brain.stimulate(neuron_idx, current_ua);
        Ok(())
    }

    fn stimulate_many(&mut self, neuron_indices: Vec<usize>, current_ua: f32) -> PyResult<()> {
        validate_neuron_indices(self.inner.brain.neurons.count, &neuron_indices)?;
        for idx in neuron_indices {
            self.inner.brain.stimulate(idx, current_ua);
        }
        Ok(())
    }

    fn stimulate_weighted(
        &mut self,
        neuron_indices: Vec<usize>,
        currents_ua: Vec<f32>,
    ) -> PyResult<()> {
        if neuron_indices.len() != currents_ua.len() {
            return Err(PyValueError::new_err(
                "neuron_indices and currents_ua must have the same length",
            ));
        }
        validate_neuron_indices(self.inner.brain.neurons.count, &neuron_indices)?;
        for (idx, current) in neuron_indices.into_iter().zip(currents_ua.into_iter()) {
            self.inner.brain.stimulate(idx, current);
        }
        Ok(())
    }

    fn add_nt_concentration_many(
        &mut self,
        neuron_indices: Vec<usize>,
        nt: usize,
        delta_nm: f32,
    ) -> PyResult<()> {
        let nt_idx = validate_nt_index(nt)?;
        validate_neuron_indices(self.inner.brain.neurons.count, &neuron_indices)?;
        self.inner
            .brain
            .add_nt_concentration_many(&neuron_indices, nt_idx, delta_nm);
        Ok(())
    }

    /// Hebbian weight nudge: strengthen relay→correct_motor, weaken relay→wrong_motor.
    fn hebbian_nudge(
        &mut self,
        relay_ids: Vec<u32>,
        correct_ids: Vec<u32>,
        wrong_ids: Vec<u32>,
        delta: f32,
    ) {
        self.inner
            .brain
            .hebbian_nudge(&relay_ids, &correct_ids, &wrong_ids, delta);
    }

    fn synapse_pre(&mut self) -> Vec<u32> {
        self.inner.brain.sync_shadow_from_gpu();
        build_synapse_pre(
            &self.inner.brain.synapses.row_offsets,
            self.inner.brain.synapses.n_synapses,
        )
    }

    fn synapse_post(&mut self) -> Vec<u32> {
        self.inner.brain.sync_shadow_from_gpu();
        self.inner.brain.synapses.col_indices.clone()
    }

    fn synapse_strengths(&mut self) -> Vec<f32> {
        self.inner.brain.sync_shadow_from_gpu();
        self.inner.brain.synapses.strength.clone()
    }

    #[pyo3(signature = (synapse_indices, delta, min_strength=0.3, max_strength=8.0))]
    fn adjust_synapse_strengths(
        &mut self,
        synapse_indices: Vec<usize>,
        delta: f32,
        min_strength: f32,
        max_strength: f32,
    ) -> PyResult<()> {
        validate_synapse_indices(self.inner.brain.synapses.n_synapses, &synapse_indices)?;
        self.inner.brain.adjust_synapse_strengths(
            &synapse_indices,
            delta,
            min_strength,
            max_strength,
        );
        Ok(())
    }

    #[pyo3(signature = (synapse_indices, strengths, min_strength=0.3, max_strength=8.0))]
    fn set_synapse_strengths(
        &mut self,
        synapse_indices: Vec<usize>,
        strengths: Vec<f32>,
        min_strength: f32,
        max_strength: f32,
    ) -> PyResult<()> {
        if synapse_indices.len() != strengths.len() {
            return Err(PyValueError::new_err(
                "synapse_indices and strengths must have the same length",
            ));
        }
        validate_synapse_indices(self.inner.brain.synapses.n_synapses, &synapse_indices)?;
        self.inner.brain.set_synapse_strengths(
            &synapse_indices,
            &strengths,
            min_strength,
            max_strength,
        );
        Ok(())
    }

    /// Get neuron indices for a named region (for example `cortex_0`).
    fn region_ids(&self, name: &str) -> PyResult<Vec<usize>> {
        self.inner
            .regions
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.neuron_indices.clone())
            .ok_or_else(|| PyValueError::new_err(format!("Unknown region: {}", name)))
    }

    fn region_names(&self) -> Vec<String> {
        self.inner.regions.iter().map(|r| r.name.clone()).collect()
    }

    fn gpu_active(&self) -> bool {
        self.inner.brain.gpu_available()
    }

    fn gpu_dispatch_active(&self) -> bool {
        self.inner.brain.gpu_dispatch_active()
    }

    fn gpu_init_error(&self) -> Option<String> {
        self.inner.brain.gpu_init_error().map(|err| err.to_string())
    }

    fn __repr__(&self) -> String {
        let regions: Vec<String> = self
            .inner
            .regions
            .iter()
            .map(|r| format!("{}({})", r.name, r.neuron_indices.len()))
            .collect();
        format!(
            "RegionalBrain(n_neurons={}, regions=[{}])",
            self.inner.brain.neurons.count,
            regions.join(", "),
        )
    }
}

// =============================================================================
// PyConsciousnessMetrics
// =============================================================================

#[pyclass(name = "ConsciousnessMetrics")]
pub struct PyConsciousnessMetrics {
    inner: ConsciousnessMetrics,
}

#[pymethods]
impl PyConsciousnessMetrics {
    #[getter]
    fn phi(&self) -> f32 {
        self.inner.phi
    }

    #[getter]
    fn pci(&self) -> f32 {
        self.inner.pci
    }

    #[getter]
    fn causal_density(&self) -> f32 {
        self.inner.causal_density
    }

    #[getter]
    fn criticality(&self) -> f32 {
        self.inner.criticality
    }

    #[getter]
    fn global_workspace(&self) -> f32 {
        self.inner.global_workspace
    }

    #[getter]
    fn orch_or(&self) -> f32 {
        self.inner.orch_or
    }

    #[getter]
    fn composite(&self) -> f32 {
        self.inner.composite
    }

    fn __repr__(&self) -> String {
        format!(
            "ConsciousnessMetrics(phi={:.3}, pci={:.3}, criticality={:.3}, gw={:.3}, orch_or={:.3}, composite={:.3})",
            self.inner.phi,
            self.inner.pci,
            self.inner.criticality,
            self.inner.global_workspace,
            self.inner.orch_or,
            self.inner.composite,
        )
    }
}

// =============================================================================
// PyTerrariumSensoryField
// =============================================================================

#[pyclass(name = "TerrariumSensoryField")]
pub struct PyTerrariumSensoryField {
    inner: TerrariumSensoryField,
}

#[pymethods]
impl PyTerrariumSensoryField {
    #[new]
    #[pyo3(signature = (width, height, depth=1))]
    fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            inner: TerrariumSensoryField::new(width, height, depth),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        odorant,
        temperature,
        wind_x,
        wind_y,
        wind_z,
        ambient_light=0.5,
        food_patches=None
    ))]
    fn load_state(
        &mut self,
        odorant: Vec<f32>,
        temperature: Vec<f32>,
        wind_x: Vec<f32>,
        wind_y: Vec<f32>,
        wind_z: Vec<f32>,
        ambient_light: f32,
        food_patches: Option<Vec<f32>>,
    ) -> PyResult<()> {
        self.inner
            .load_state(
                &odorant,
                &temperature,
                &wind_x,
                &wind_y,
                &wind_z,
                ambient_light,
                food_patches.as_deref().unwrap_or(&[]),
            )
            .map_err(PyValueError::new_err)
    }

    fn load_world_state(&mut self, py: Python<'_>, world: &Bound<'_, PyAny>) -> PyResult<()> {
        let width = world.getattr("W")?.extract::<usize>()?;
        let height = world.getattr("H")?.extract::<usize>()?;
        let depth = world.getattr("D")?.extract::<usize>()?;
        let expected_shape = self.inner.shape();
        if (width, height, depth) != expected_shape {
            return Err(PyValueError::new_err(format!(
                "world shape mismatch: field={expected_shape:?}, world=({}, {}, {})",
                width, height, depth
            )));
        }
        let expected_len = width * height * depth.max(1);
        let mut odorant = vec![0.0f32; expected_len];
        let odorant_grids_obj = world.getattr("odorant_grids")?;
        let odorant_grids = odorant_grids_obj.cast::<PyDict>()?;
        for (_, grid) in odorant_grids.iter() {
            add_buffer_into_sum(py, &grid, &mut odorant, "odorant_grid")?;
        }
        let temperature = buffer_to_vec_f32(
            py,
            &world.getattr("temperature")?,
            expected_len,
            "temperature",
        )?;
        let wind_x = buffer_to_vec_f32(py, &world.getattr("wind_vx")?, expected_len, "wind_vx")?;
        let wind_y = buffer_to_vec_f32(py, &world.getattr("wind_vy")?, expected_len, "wind_vy")?;
        let wind_z = buffer_to_vec_f32(py, &world.getattr("wind_vz")?, expected_len, "wind_vz")?;
        let ambient_light = match world.getattr("_light_intensity") {
            Ok(light_fn) => light_fn.call0()?.extract::<f32>()?,
            Err(_) => 0.5,
        };
        let food_patches = world_food_patches_flat(world)?;
        self.inner
            .load_state(
                &odorant,
                &temperature,
                &wind_x,
                &wind_y,
                &wind_z,
                ambient_light,
                &food_patches,
            )
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (x, y, z=0.0, heading=0.0, is_flying=false))]
    fn sample_fly<'py>(
        &self,
        py: Python<'py>,
        x: f32,
        y: f32,
        z: f32,
        heading: f32,
        is_flying: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sample = self.inner.sample_fly(x, y, z, heading, is_flying);
        let dict = PyDict::new(py);
        dict.set_item("odorant", sample.odorant)?;
        dict.set_item("left_light", sample.left_light)?;
        dict.set_item("right_light", sample.right_light)?;
        dict.set_item("temperature", sample.temperature)?;
        dict.set_item("wind_x", sample.wind_x)?;
        dict.set_item("wind_y", sample.wind_y)?;
        dict.set_item("wind_z", sample.wind_z)?;
        dict.set_item("sugar_taste", sample.sugar_taste)?;
        dict.set_item("bitter_taste", sample.bitter_taste)?;
        dict.set_item("amino_taste", sample.amino_taste)?;
        dict.set_item("food_available", sample.food_available)?;
        Ok(dict)
    }

    #[pyo3(signature = (x, y, eat_radius=2.0, amount=0.03))]
    fn consume_food_near(&mut self, x: f32, y: f32, eat_radius: f32, amount: f32) -> bool {
        self.inner.consume_food_near(x, y, eat_radius, amount)
    }

    fn food_patches(&self) -> Vec<f32> {
        self.inner.food_patches_flat()
    }

    fn sync_food_to_world(&self, world: &Bound<'_, PyAny>) -> PyResult<bool> {
        let flat = self.inner.food_patches_flat();
        let food_patches = world.getattr("food_patches")?;
        let fruit_sources = world.getattr("fruit_sources")?;
        let fruit_count = fruit_sources.len()?;
        let mut changed = false;

        for (idx, patch) in food_patches.try_iter()?.enumerate() {
            let patch = patch?;
            let patch = patch.cast::<PyDict>()?;
            let base = idx * 4;
            if base + 3 >= flat.len() {
                break;
            }
            let remaining = flat[base + 3];
            let current = get_required_dict_item(&patch, "remaining")?.extract::<f32>()?;
            if (current - remaining).abs() > 1e-9 {
                patch.set_item("remaining", remaining)?;
                changed = true;
            }
            if idx < fruit_count {
                let fruit = fruit_sources.get_item(idx)?;
                let sugar = fruit.getattr("sugar_content")?.extract::<f32>()?;
                fruit.setattr("sugar_content", sugar.min(remaining))?;
                if remaining <= 0.01 {
                    fruit.setattr("alive", false)?;
                }
            }
        }
        Ok(changed)
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }
}

// =============================================================================
// PyDrosophilaSim
// =============================================================================

/// Drosophila brain simulator with 6 experiment runners.
///
/// Uses Metal GPU on macOS Apple Silicon (>= 64 neurons), CPU fallback elsewhere.
///
/// Experiments:
/// 1. Olfactory learning — odor source approach
/// 2. Phototaxis — light gradient navigation
/// 3. Thermotaxis — temperature preference
/// 4. Foraging — multi-food search
/// 5. Drug response — TTX/picrotoxin effects
/// 6. Circadian — time-of-day activity modulation
#[pyclass(name = "DrosophilaSim")]
pub struct PyDrosophilaSim {
    inner: crate::drosophila::DrosophilaSim,
}

#[pymethods]
impl PyDrosophilaSim {
    #[new]
    #[pyo3(signature = (n_neurons=5000, seed=42))]
    fn new(n_neurons: usize, seed: u64) -> Self {
        let sim = crate::drosophila::DrosophilaSim::from_count(n_neurons, seed);
        Self { inner: sim }
    }

    /// Run a single body step (includes multiple neural steps).
    fn step(&mut self) {
        self.inner.body_step();
    }

    /// Override the native fly body state from an external world.
    #[pyo3(signature = (
        x,
        y,
        heading,
        z=None,
        pitch=None,
        is_flying=None,
        speed=None,
        energy=None,
        temperature=None,
        time_of_day=None,
        world_width=None,
        world_height=None
    ))]
    fn set_body_state(
        &mut self,
        x: f32,
        y: f32,
        heading: f32,
        z: Option<f32>,
        pitch: Option<f32>,
        is_flying: Option<bool>,
        speed: Option<f32>,
        energy: Option<f32>,
        temperature: Option<f32>,
        time_of_day: Option<f32>,
        world_width: Option<f32>,
        world_height: Option<f32>,
    ) {
        if let (Some(width), Some(height)) = (world_width, world_height) {
            self.inner.set_world_bounds(width, height);
        }
        self.inner.set_body_state(
            x,
            y,
            heading,
            z,
            pitch,
            is_flying,
            speed,
            energy,
            temperature,
            time_of_day,
        );
    }

    /// Run a single native fly step from externally sampled local sensory values.
    fn step_manual(&mut self, odorant: f32, left_light: f32, right_light: f32, temperature: f32) {
        self.inner
            .body_step_manual(odorant, left_light, right_light, temperature);
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        odorant,
        left_light,
        right_light,
        temperature,
        sugar_taste=0.0,
        bitter_taste=0.0,
        amino_taste=0.0,
        wind_x=0.0,
        wind_y=0.0,
        wind_z=0.0,
        food_available=0.0,
        reward_valence=0.0
    ))]
    fn step_terrarium<'py>(
        &mut self,
        py: Python<'py>,
        odorant: f32,
        left_light: f32,
        right_light: f32,
        temperature: f32,
        sugar_taste: f32,
        bitter_taste: f32,
        amino_taste: f32,
        wind_x: f32,
        wind_y: f32,
        wind_z: f32,
        food_available: f32,
        reward_valence: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let report = self.inner.body_step_terrarium(
            odorant,
            left_light,
            right_light,
            temperature,
            sugar_taste,
            bitter_taste,
            amino_taste,
            wind_x,
            wind_y,
            wind_z,
            food_available,
            reward_valence,
        );
        let dict = PyDict::new(py);
        dict.set_item("speed", report.speed)?;
        dict.set_item("turn", report.turn)?;
        dict.set_item("fly", report.fly_signal)?;
        dict.set_item("feed", report.feed_signal)?;
        dict.set_item("climb", report.climb_signal)?;
        dict.set_item("consumed_food", report.consumed_food)?;
        dict.set_item("x", report.x)?;
        dict.set_item("y", report.y)?;
        dict.set_item("z", report.z)?;
        dict.set_item("heading", report.heading)?;
        dict.set_item("pitch", report.pitch)?;
        dict.set_item("energy", report.energy)?;
        dict.set_item("is_flying", report.is_flying)?;
        Ok(dict)
    }

    /// Run N body steps.
    fn run_steps(&mut self, n: u32) {
        self.inner.run_body_steps(n);
    }

    /// Run olfactory learning experiment.
    fn run_olfactory(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_olfactory(n_episodes),
        }
    }

    /// Run phototaxis experiment.
    fn run_phototaxis(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_phototaxis(n_episodes),
        }
    }

    /// Run thermotaxis experiment.
    fn run_thermotaxis(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_thermotaxis(n_episodes),
        }
    }

    /// Run foraging experiment.
    fn run_foraging(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_foraging(n_episodes),
        }
    }

    /// Run drug response experiment.
    fn run_drug_response(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_drug_response(n_episodes),
        }
    }

    /// Run circadian experiment.
    fn run_circadian(&mut self, n_episodes: u32) -> PyExperimentResult {
        PyExperimentResult {
            inner: self.inner.run_circadian(n_episodes),
        }
    }

    /// Run all 6 experiments and return results.
    fn run_all(&mut self) -> Vec<PyExperimentResult> {
        self.inner
            .run_all()
            .into_iter()
            .map(|r| PyExperimentResult { inner: r })
            .collect()
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.brain.step_count
    }

    #[getter]
    fn body_x(&self) -> f32 {
        self.inner.body_state().x
    }

    #[getter]
    fn body_y(&self) -> f32 {
        self.inner.body_state().y
    }

    #[getter]
    fn body_z(&self) -> f32 {
        self.inner.body_state().z
    }

    #[getter]
    fn body_heading(&self) -> f32 {
        self.inner.body_state().heading
    }

    #[getter]
    fn body_pitch(&self) -> f32 {
        self.inner.body_state().pitch
    }

    #[getter]
    fn body_speed(&self) -> f32 {
        self.inner.body_state().speed
    }

    #[getter]
    fn body_energy(&self) -> f32 {
        self.inner.body_state().energy
    }

    #[getter]
    fn body_temperature(&self) -> f32 {
        self.inner.body_state().temperature
    }

    #[getter]
    fn body_time_of_day(&self) -> f32 {
        self.inner.body_state().time_of_day
    }

    #[getter]
    fn body_is_flying(&self) -> bool {
        self.inner.body_state().is_flying
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.brain.neurons.count
    }

    fn __repr__(&self) -> String {
        format!(
            "DrosophilaSim(n={}, scale={:?}, steps={})",
            self.inner.brain.neurons.count, self.inner.scale, self.inner.brain.step_count
        )
    }
}

/// Experiment result returned from Drosophila experiments.
#[pyclass(name = "ExperimentResult")]
pub struct PyExperimentResult {
    inner: crate::drosophila::ExperimentResult,
}

#[pymethods]
impl PyExperimentResult {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn passed(&self) -> bool {
        self.inner.passed
    }
    #[getter]
    fn metric_name(&self) -> &str {
        &self.inner.metric_name
    }
    #[getter]
    fn metric_value(&self) -> f64 {
        self.inner.metric_value
    }
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }
    #[getter]
    fn details(&self) -> &str {
        &self.inner.details
    }
    #[getter]
    fn trajectories(&self) -> Vec<(f32, f32)> {
        self.inner.trajectories.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "{}: {} ({}={:.3}, threshold={:.3}) — {}",
            if self.inner.passed { "PASS" } else { "FAIL" },
            self.inner.name,
            self.inner.metric_name,
            self.inner.metric_value,
            self.inner.threshold,
            self.inner.details
        )
    }
}

// =============================================================================
// PyDoomBrainSim
// =============================================================================

/// Doom FPS brain simulator with biophysical neurons.
///
/// Two modes:
/// - Embodied: 64 rays, retina pipeline
/// - Disembodied: 8 rays direct to cortex
#[pyclass(name = "DoomBrainSim")]
pub struct PyDoomBrainSim {
    inner: crate::doom_brain::DoomBrainSim,
}

#[pymethods]
impl PyDoomBrainSim {
    #[new]
    #[pyo3(signature = (n_neurons=5000, mode="disembodied", seed=42))]
    fn new(n_neurons: usize, mode: &str, seed: u64) -> PyResult<Self> {
        let doom_mode = match mode {
            "embodied" => crate::doom_brain::DoomMode::Embodied,
            "disembodied" => crate::doom_brain::DoomMode::Disembodied,
            _ => {
                return Err(PyValueError::new_err(
                    "mode must be 'embodied' or 'disembodied'",
                ))
            }
        };
        let sim = crate::doom_brain::DoomBrainSim::new(n_neurons, doom_mode, seed);
        Ok(Self { inner: sim })
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn run_steps(&mut self, n: u64) {
        self.inner.run_steps(n);
    }

    fn run_threat_avoidance(&mut self, n_episodes: u32) -> PyDoomExperimentResult {
        PyDoomExperimentResult {
            inner: self.inner.run_threat_avoidance(n_episodes),
        }
    }

    fn run_navigation(&mut self, n_episodes: u32) -> PyDoomExperimentResult {
        PyDoomExperimentResult {
            inner: self.inner.run_navigation(n_episodes),
        }
    }

    fn run_combat(&mut self, n_episodes: u32) -> PyDoomExperimentResult {
        PyDoomExperimentResult {
            inner: self.inner.run_combat(n_episodes),
        }
    }

    #[pyo3(signature = (episodes_per_experiment=10))]
    fn run_all(&mut self, episodes_per_experiment: u32) -> Vec<PyDoomExperimentResult> {
        self.inner
            .run_all(episodes_per_experiment)
            .into_iter()
            .map(|r| PyDoomExperimentResult { inner: r })
            .collect()
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count
    }

    #[getter]
    fn n_neurons(&self) -> usize {
        self.inner.n_neurons
    }

    #[getter]
    fn player_x(&self) -> f32 {
        self.inner.doom.player_x
    }

    #[getter]
    fn player_y(&self) -> f32 {
        self.inner.doom.player_y
    }

    #[getter]
    fn player_hp(&self) -> f32 {
        self.inner.doom.player_hp
    }

    fn __repr__(&self) -> String {
        format!(
            "DoomBrainSim(n={}, mode={:?}, steps={}, hp={:.0})",
            self.inner.n_neurons, self.inner.mode, self.inner.step_count, self.inner.doom.player_hp
        )
    }
}

/// Doom experiment result.
#[pyclass(name = "DoomExperimentResult")]
pub struct PyDoomExperimentResult {
    inner: crate::doom_brain::DoomExperimentResult,
}

#[pymethods]
impl PyDoomExperimentResult {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn passed(&self) -> bool {
        self.inner.passed
    }
    #[getter]
    fn metric_name(&self) -> &str {
        &self.inner.metric_name
    }
    #[getter]
    fn metric_value(&self) -> f64 {
        self.inner.metric_value
    }
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }
    #[getter]
    fn details(&self) -> &str {
        &self.inner.details
    }

    fn __repr__(&self) -> String {
        format!(
            "{}: {} ({}={:.3}, threshold={:.3}) — {}",
            if self.inner.passed { "PASS" } else { "FAIL" },
            self.inner.name,
            self.inner.metric_name,
            self.inner.metric_value,
            self.inner.threshold,
            self.inner.details
        )
    }
}

// =============================================================================
// PyMolecularRetina
// =============================================================================

/// Biophysical retina converting RGB frames to spike trains.
///
/// Three-layer circuit:
/// 1. Photoreceptors — Govardovskii spectral sensitivity, Weber adaptation,
///    graded hyperpolarization (dark=-40mV, bright=-70mV).
/// 2. Bipolar cells — ON/OFF pathways via mGluR6/iGluR, center-surround
///    antagonism with ribbon synapse threshold.
/// 3. RGCs — Full HH spiking (g_Na=120, g_K=36, g_leak=0.3), only spiking
///    layer — axons form optic nerve output.
#[pyclass(name = "MolecularRetina")]
pub struct PyMolecularRetina {
    inner: crate::retina::MolecularRetina,
}

#[pymethods]
impl PyMolecularRetina {
    /// Create a new retina for frames of width x height pixels.
    #[new]
    #[pyo3(signature = (width=64, height=48, seed=42))]
    fn new(width: u32, height: u32, seed: u64) -> Self {
        Self {
            inner: crate::retina::MolecularRetina::new(width, height, seed),
        }
    }

    /// Process one RGB frame (bytes, length = width*height*3) -> list of fired RGC IDs.
    #[pyo3(signature = (rgb, n_steps=10))]
    fn process_frame(&mut self, rgb: &[u8], n_steps: u32) -> PyResult<Vec<u32>> {
        let expected = (self.inner.width * self.inner.height * 3) as usize;
        if rgb.len() != expected {
            return Err(PyValueError::new_err(format!(
                "RGB buffer length {} != expected {} ({}x{}x3)",
                rgb.len(),
                expected,
                self.inner.width,
                self.inner.height
            )));
        }
        Ok(self.inner.process_frame(rgb, n_steps))
    }

    /// Reset all state to initial conditions.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Number of photoreceptors.
    #[getter]
    fn n_photo(&self) -> usize {
        self.inner.n_photo
    }

    /// Number of bipolar cells.
    #[getter]
    fn n_bipolar(&self) -> usize {
        self.inner.n_bipolar
    }

    /// Number of retinal ganglion cells (output neurons).
    #[getter]
    fn n_rgc(&self) -> usize {
        self.inner.n_rgc
    }

    /// Total neuron count across all 3 layers.
    #[getter]
    fn total_neurons(&self) -> usize {
        self.inner.total_neurons()
    }

    /// Width of the visual field.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Height of the visual field.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Total frames processed.
    #[getter]
    fn total_frames(&self) -> u64 {
        self.inner.total_frames()
    }

    /// Total spikes produced.
    #[getter]
    fn total_spikes(&self) -> u64 {
        self.inner.total_spikes()
    }

    /// RGC spike counts as a list.
    fn spike_counts(&self) -> Vec<u32> {
        self.inner.rgc_spike_counts().to_vec()
    }

    /// Photoreceptor voltages.
    fn photo_voltages(&self) -> Vec<f32> {
        self.inner.photo_voltages().to_vec()
    }

    /// Bipolar cell voltages.
    fn bipolar_voltages(&self) -> Vec<f32> {
        self.inner.bipolar_voltages().to_vec()
    }

    /// RGC voltages.
    fn rgc_voltages(&self) -> Vec<f32> {
        self.inner.rgc_voltages().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

// =============================================================================
// PyCellularMetabolism
// =============================================================================

#[pyclass(name = "CellularMetabolism")]
pub struct PyCellularMetabolism {
    inner: CellularMetabolismSim,
}

#[pymethods]
impl PyCellularMetabolism {
    #[new]
    #[pyo3(signature = (
        glucose=5.0,
        pyruvate=0.1,
        lactate=1.0,
        oxygen=0.05,
        atp=3.0,
        adp=0.3,
        amp=0.05,
        nad_plus=0.5,
        nadh=0.05
    ))]
    fn new(
        glucose: f32,
        pyruvate: f32,
        lactate: f32,
        oxygen: f32,
        atp: f32,
        adp: f32,
        amp: f32,
        nad_plus: f32,
        nadh: f32,
    ) -> Self {
        Self {
            inner: CellularMetabolismSim::new(
                glucose, pyruvate, lactate, oxygen, atp, adp, amp, nad_plus, nadh,
            ),
        }
    }

    fn supply_glucose(&mut self, amount: f32) {
        self.inner.supply_glucose(amount);
    }

    fn supply_oxygen(&mut self, amount: f32) {
        self.inner.supply_oxygen(amount);
    }

    fn supply_lactate(&mut self, amount: f32) {
        self.inner.supply_lactate(amount);
    }

    #[pyo3(signature = (amount_mm, purpose=None))]
    fn consume_atp(&mut self, amount_mm: f32, purpose: Option<&str>) -> bool {
        let _ = purpose;
        self.inner.consume_atp(amount_mm)
    }

    fn protein_synthesis_cost(&mut self, dt: f32, gene_expression_rate: f32) -> bool {
        self.inner.protein_synthesis_cost(dt, gene_expression_rate)
    }

    fn step(&mut self, dt: f32) {
        self.inner.step(dt);
    }

    #[getter]
    fn glucose(&self) -> f32 {
        self.inner.glucose()
    }

    #[getter]
    fn pyruvate(&self) -> f32 {
        self.inner.pyruvate()
    }

    #[getter]
    fn lactate(&self) -> f32 {
        self.inner.lactate()
    }

    #[getter]
    fn oxygen(&self) -> f32 {
        self.inner.oxygen()
    }

    #[getter]
    fn atp(&self) -> f32 {
        self.inner.atp()
    }

    #[getter]
    fn adp(&self) -> f32 {
        self.inner.adp()
    }

    #[getter]
    fn amp(&self) -> f32 {
        self.inner.amp()
    }

    #[getter]
    fn nad_plus(&self) -> f32 {
        self.inner.nad_plus()
    }

    #[getter]
    fn nadh(&self) -> f32 {
        self.inner.nadh()
    }

    #[getter]
    fn energy_ratio(&self) -> f32 {
        self.inner.energy_ratio()
    }

    #[getter]
    fn atp_available(&self) -> bool {
        self.inner.atp_available()
    }

    #[getter]
    fn is_hypoxic(&self) -> bool {
        self.inner.is_hypoxic()
    }

    fn __repr__(&self) -> String {
        format!(
            "CellularMetabolism(glucose={:.3}, oxygen={:.3}, atp={:.3}, ratio={:.3})",
            self.inner.glucose(),
            self.inner.oxygen(),
            self.inner.atp(),
            self.inner.energy_ratio(),
        )
    }
}

// =============================================================================
// PyPlantCellularState
// =============================================================================

#[pyclass(name = "PlantCellularState")]
pub struct PyPlantCellularState {
    inner: PlantCellularStateSim,
}

#[pymethods]
impl PyPlantCellularState {
    #[new]
    fn new(leaf_cells: f32, stem_cells: f32, root_cells: f32, meristem_cells: f32) -> Self {
        Self {
            inner: PlantCellularStateSim::new(leaf_cells, stem_cells, root_cells, meristem_cells),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        dt: f32,
        local_light: f32,
        temp_factor: f32,
        water_in: f32,
        nutrient_in: f32,
        water_status: f32,
        nutrient_status: f32,
        symbiosis: f32,
        stress_signal: f32,
        storage_signal: f32,
    ) -> Vec<f32> {
        let feedback = self.inner.step(
            dt,
            local_light,
            temp_factor,
            water_in,
            nutrient_in,
            water_status,
            nutrient_status,
            symbiosis,
            stress_signal,
            storage_signal,
        );
        vec![
            feedback.photosynthetic_capacity,
            feedback.maintenance_cost,
            feedback.storage_exchange,
            feedback.division_growth,
            feedback.senescence_mass,
            feedback.energy_charge,
            feedback.vitality,
            feedback.sugar_pool,
            feedback.water_pool,
            feedback.nitrogen_pool,
            feedback.division_signal,
            feedback.new_cells,
        ]
    }

    fn cluster_snapshot(&self, tissue: &str) -> PyResult<Vec<f32>> {
        let snapshot = self.inner.cluster_snapshot(parse_plant_tissue(tissue)?);
        Ok(vec![
            snapshot.cell_count,
            snapshot.vitality,
            snapshot.division_buffer,
            snapshot.state_atp,
            snapshot.state_adp,
            snapshot.state_glucose,
            snapshot.state_starch,
            snapshot.state_water,
            snapshot.state_nitrate,
            snapshot.state_amino_acid,
            snapshot.state_auxin,
            snapshot.cytoplasm_water,
            snapshot.cytoplasm_sucrose,
            snapshot.cytoplasm_nitrate,
            snapshot.apoplast_water,
            snapshot.apoplast_nitrate,
            snapshot.transcript_stress_response,
            snapshot.transcript_cell_cycle,
            snapshot.transcript_transport_program,
            snapshot.chem_glucose,
            snapshot.chem_pyruvate,
            snapshot.chem_lactate,
            snapshot.chem_oxygen,
            snapshot.chem_atp,
            snapshot.chem_adp,
            snapshot.chem_amp,
            snapshot.chem_nad_plus,
            snapshot.chem_nadh,
        ])
    }

    #[getter]
    fn total_cells(&self) -> f32 {
        self.inner.total_cells()
    }

    #[getter]
    fn vitality(&self) -> f32 {
        self.inner.vitality()
    }

    #[getter]
    fn energy_charge(&self) -> f32 {
        self.inner.energy_charge()
    }

    #[getter]
    fn sugar_pool(&self) -> f32 {
        self.inner.sugar_pool()
    }

    #[getter]
    fn water_pool(&self) -> f32 {
        self.inner.water_pool()
    }

    #[getter]
    fn nitrogen_pool(&self) -> f32 {
        self.inner.nitrogen_pool()
    }

    #[getter]
    fn division_signal(&self) -> f32 {
        self.inner.division_signal()
    }

    #[getter]
    fn division_events(&self) -> f32 {
        self.inner.division_events()
    }

    #[getter]
    fn last_new_cells(&self) -> f32 {
        self.inner.last_new_cells()
    }

    #[getter]
    fn last_senescence(&self) -> f32 {
        self.inner.last_senescence()
    }

    fn __repr__(&self) -> String {
        format!(
            "PlantCellularState(total_cells={:.2}, vitality={:.3}, energy_charge={:.3})",
            self.inner.total_cells(),
            self.inner.vitality(),
            self.inner.energy_charge(),
        )
    }
}

// =============================================================================
// PyWholeCellSimulator
// =============================================================================

#[pyclass(name = "PlantOrganism")]
pub struct PyPlantOrganism {
    inner: PlantOrganismSim,
}

#[pymethods]
impl PyPlantOrganism {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_height_mm: f32,
        canopy_radius_mm: f32,
        root_radius_mm: f32,
        leaf_efficiency: f32,
        root_uptake_efficiency: f32,
        water_use_efficiency: f32,
        volatile_scale: f32,
        fruiting_threshold: f32,
        litter_turnover: f32,
        shade_tolerance: f32,
        root_depth_bias: f32,
        symbiosis_affinity: f32,
        seed_mass: f32,
        leaf_biomass: f32,
        stem_biomass: f32,
        root_biomass: f32,
        storage_carbon: f32,
        water_buffer: f32,
        nitrogen_buffer: f32,
        fruit_timer_s: f32,
        seed_timer_s: f32,
    ) -> Self {
        Self {
            inner: PlantOrganismSim::new(
                max_height_mm,
                canopy_radius_mm,
                root_radius_mm,
                leaf_efficiency,
                root_uptake_efficiency,
                water_use_efficiency,
                volatile_scale,
                fruiting_threshold,
                litter_turnover,
                shade_tolerance,
                root_depth_bias,
                symbiosis_affinity,
                seed_mass,
                leaf_biomass,
                stem_biomass,
                root_biomass,
                storage_carbon,
                water_buffer,
                nitrogen_buffer,
                fruit_timer_s,
                seed_timer_s,
            ),
        }
    }

    fn resource_demands(
        &self,
        dt: f32,
        root_energy_gate: f32,
        root_water_deficit: f32,
        root_nitrogen_deficit: f32,
    ) -> (f32, f32) {
        self.inner.resource_demands(
            dt,
            root_energy_gate,
            root_water_deficit,
            root_nitrogen_deficit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        dt: f32,
        water_uptake: f32,
        nutrient_uptake: f32,
        local_light: f32,
        temp_factor: f32,
        root_pressure: f32,
        symbiosis_bonus: f32,
        water_factor: f32,
        nutrient_factor: f32,
        canopy_competition: f32,
        root_competition: f32,
        soil_glucose: f32,
        cell_photosynthetic_capacity: f32,
        cell_maintenance_cost: f32,
        cell_storage_exchange: f32,
        cell_division_growth: f32,
        cell_senescence_mass: f32,
        cell_energy_charge: f32,
        cell_vitality: f32,
        cell_sugar_pool: f32,
        cell_division_signal: f32,
        total_cells: f32,
        fruit_reset_s: f32,
        seed_reset_s: f32,
    ) -> Vec<f32> {
        let report = self.inner.step(
            dt,
            water_uptake,
            nutrient_uptake,
            local_light,
            temp_factor,
            root_pressure,
            symbiosis_bonus,
            water_factor,
            nutrient_factor,
            canopy_competition,
            root_competition,
            soil_glucose,
            1.0,
            1.0,
            cell_photosynthetic_capacity,
            cell_maintenance_cost,
            cell_storage_exchange,
            cell_division_growth,
            cell_senescence_mass,
            cell_energy_charge,
            cell_vitality,
            cell_sugar_pool,
            cell_division_signal,
            total_cells,
            fruit_reset_s,
            seed_reset_s,
        );
        vec![
            report.exudates,
            report.litter,
            if report.spawned_fruit { 1.0 } else { 0.0 },
            report.fruit_size,
            if report.spawned_seed { 1.0 } else { 0.0 },
        ]
    }

    #[getter]
    fn leaf_biomass(&self) -> f32 {
        self.inner.leaf_biomass()
    }
    #[getter]
    fn stem_biomass(&self) -> f32 {
        self.inner.stem_biomass()
    }
    #[getter]
    fn root_biomass(&self) -> f32 {
        self.inner.root_biomass()
    }
    #[getter]
    fn storage_carbon(&self) -> f32 {
        self.inner.storage_carbon()
    }
    #[getter]
    fn water_buffer(&self) -> f32 {
        self.inner.water_buffer()
    }
    #[getter]
    fn nitrogen_buffer(&self) -> f32 {
        self.inner.nitrogen_buffer()
    }
    #[getter]
    fn fruit_timer_s(&self) -> f32 {
        self.inner.fruit_timer_s()
    }
    #[getter]
    fn seed_timer_s(&self) -> f32 {
        self.inner.seed_timer_s()
    }
    #[getter]
    fn age_s(&self) -> f32 {
        self.inner.age_s()
    }
    #[getter]
    fn health(&self) -> f32 {
        self.inner.health()
    }
    #[getter]
    fn fruit_count(&self) -> u32 {
        self.inner.fruit_count()
    }
    #[getter]
    fn height_mm(&self) -> f32 {
        self.inner.height_mm()
    }
    #[getter]
    fn nectar_production_rate(&self) -> f32 {
        self.inner.nectar_production_rate()
    }
    #[getter]
    fn odorant_geraniol(&self) -> f32 {
        self.inner.odorant_geraniol()
    }
    #[getter]
    fn odorant_ethyl_acetate(&self) -> f32 {
        self.inner.odorant_ethyl_acetate()
    }
    #[getter]
    fn odorant_emission_rate(&self) -> f32 {
        self.inner.odorant_emission_rate()
    }
    #[getter]
    fn total_biomass(&self) -> f32 {
        self.inner.total_biomass()
    }

    fn is_dead(&self) -> bool {
        self.inner.is_dead()
    }
}

// =============================================================================
// PyWholeCellSimulator
// =============================================================================

/// Rust-native whole-cell simulator with optional Metal acceleration.
#[pyclass(name = "WholeCellSimulator")]
pub struct PyWholeCellSimulator {
    inner: WholeCellSimulator,
}

#[pymethods]
impl PyWholeCellSimulator {
    #[new]
    #[pyo3(signature = (x_dim=24, y_dim=24, z_dim=12, voxel_size_nm=20.0, dt_ms=0.25, use_gpu=true))]
    fn new(
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_nm: f32,
        dt_ms: f32,
        use_gpu: bool,
    ) -> Self {
        Self {
            inner: WholeCellSimulator::new(WholeCellConfig {
                x_dim,
                y_dim,
                z_dim,
                voxel_size_nm,
                dt_ms,
                use_gpu,
                ..WholeCellConfig::default()
            }),
        }
    }

    #[staticmethod]
    fn from_bundle_manifest_path(manifest_path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WholeCellSimulator::from_bundle_manifest_path(manifest_path)
                .map_err(PyValueError::new_err)?,
        })
    }

    #[staticmethod]
    fn compile_bundle_manifest_program_spec_json(manifest_path: &str) -> PyResult<String> {
        compile_program_spec_json_from_bundle_manifest_path(manifest_path)
            .map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn compile_bundle_manifest_organism_spec_json(manifest_path: &str) -> PyResult<String> {
        compile_organism_spec_json_from_bundle_manifest_path(manifest_path)
            .map_err(PyValueError::new_err)
    }

    #[staticmethod]
    fn compile_bundle_manifest_genome_asset_package_json(
        manifest_path: &str,
    ) -> PyResult<String> {
        compile_genome_asset_package_json_from_bundle_manifest_path(manifest_path)
            .map_err(PyValueError::new_err)
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn run(&mut self, steps: u64) {
        self.inner.run(steps);
    }

    fn set_metabolic_load(&mut self, load: f32) {
        self.inner.set_metabolic_load(load);
    }

    #[pyo3(signature = (
        oxphos_efficiency=1.0,
        translation_efficiency=1.0,
        nucleotide_polymerization_efficiency=1.0,
        membrane_synthesis_efficiency=1.0,
        chromosome_segregation_efficiency=1.0
    ))]
    fn set_quantum_profile(
        &mut self,
        oxphos_efficiency: f32,
        translation_efficiency: f32,
        nucleotide_polymerization_efficiency: f32,
        membrane_synthesis_efficiency: f32,
        chromosome_segregation_efficiency: f32,
    ) {
        self.inner.set_quantum_profile(WholeCellQuantumProfile {
            oxphos_efficiency,
            translation_efficiency,
            nucleotide_polymerization_efficiency,
            membrane_synthesis_efficiency,
            chromosome_segregation_efficiency,
        });
    }

    #[pyo3(signature = (x_dim=12, y_dim=12, z_dim=6, voxel_size_au=0.5, use_gpu=true))]
    fn enable_local_chemistry(
        &mut self,
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) {
        self.inner
            .enable_local_chemistry(x_dim, y_dim, z_dim, voxel_size_au, use_gpu);
    }

    fn disable_local_chemistry(&mut self) {
        self.inner.disable_local_chemistry();
    }

    fn enable_default_syn3a_subsystems(&mut self) {
        self.inner.enable_default_syn3a_subsystems();
    }

    #[pyo3(signature = (preset, interval_steps=None))]
    fn schedule_syn3a_subsystem_probe(
        &mut self,
        preset: &str,
        interval_steps: Option<u64>,
    ) -> PyResult<()> {
        let preset = parse_syn3a_subsystem_preset(preset)?;
        self.inner.schedule_syn3a_subsystem_probe(
            preset,
            interval_steps.unwrap_or_else(|| preset.default_interval_steps()),
        );
        Ok(())
    }

    fn clear_syn3a_subsystem_probes(&mut self) {
        self.inner.clear_syn3a_subsystem_probes();
    }

    fn scheduled_syn3a_subsystem_probes(&self) -> Vec<(String, u64)> {
        self.inner
            .scheduled_syn3a_subsystem_probes()
            .into_iter()
            .map(|probe| (probe.preset.as_str().to_string(), probe.interval_steps))
            .collect()
    }

    fn subsystem_states(
        &self,
    ) -> Vec<(
        String,
        String,
        (f32, f32, f32, f32, f32, f32),
        (f32, f32),
        (f32, f32, f32, f32),
        Option<u64>,
        (usize, usize, usize, f32),
    )> {
        self.inner
            .subsystem_states()
            .into_iter()
            .map(|state| {
                (
                    state.preset.as_str().to_string(),
                    state.site.as_str().to_string(),
                    (
                        state.structural_order,
                        state.atp_scale,
                        state.translation_scale,
                        state.replication_scale,
                        state.segregation_scale,
                        state.membrane_scale,
                    ),
                    (state.crowding_penalty, state.constriction_scale),
                    (
                        state.assembly_component_availability,
                        state.assembly_occupancy,
                        state.assembly_stability,
                        state.assembly_turnover,
                    ),
                    state.last_probe_step,
                    (
                        state.site_x,
                        state.site_y,
                        state.site_z,
                        state.localization_score,
                    ),
                )
            })
            .collect()
    }

    fn local_chemistry_report(&self) -> Option<(f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
        self.inner.local_chemistry_report().map(|report| {
            (
                report.atp_support,
                report.translation_support,
                report.nucleotide_support,
                report.membrane_support,
                report.crowding_penalty,
                report.mean_glucose,
                report.mean_oxygen,
                report.mean_atp_flux,
                report.mean_carbon_dioxide,
            )
        })
    }

    fn local_chemistry_sites(
        &self,
    ) -> Vec<(
        String,
        String,
        usize,
        (f32, f32, f32, f32, f32, f32, f32, f32, f32),
        (f32, f32, f32, f32, f32),
        (f32, f32, f32, f32),
        (usize, usize, usize, f32),
    )> {
        self.inner
            .local_chemistry_sites()
            .into_iter()
            .map(|report| {
                (
                    report.preset.as_str().to_string(),
                    report.site.as_str().to_string(),
                    report.patch_radius,
                    (
                        report.atp_support,
                        report.translation_support,
                        report.nucleotide_support,
                        report.membrane_support,
                        report.crowding_penalty,
                        report.mean_glucose,
                        report.mean_oxygen,
                        report.mean_atp_flux,
                        report.mean_carbon_dioxide,
                    ),
                    (
                        report.substrate_draw,
                        report.energy_draw,
                        report.biosynthetic_draw,
                        report.byproduct_load,
                        report.demand_satisfaction,
                    ),
                    (
                        report.assembly_component_availability,
                        report.assembly_occupancy,
                        report.assembly_stability,
                        report.assembly_turnover,
                    ),
                    (
                        report.site_x,
                        report.site_y,
                        report.site_z,
                        report.localization_score,
                    ),
                )
            })
            .collect()
    }

    #[pyo3(signature = (
        site="cytosol",
        n_atoms=32,
        steps=32,
        dt_ps=0.001,
        box_size_angstrom=18.0,
        temperature_k=310.0
    ))]
    fn run_local_md_probe(
        &mut self,
        site: &str,
        n_atoms: usize,
        steps: usize,
        dt_ps: f32,
        box_size_angstrom: f32,
        temperature_k: f32,
    ) -> PyResult<Option<(String, f32, f32, f32, f32, f32, f32, f32, f32)>> {
        let site = parse_chemistry_site(site)?;
        Ok(self
            .inner
            .run_local_md_probe(LocalMDProbeRequest {
                site,
                n_atoms,
                steps,
                dt_ps,
                box_size_angstrom,
                temperature_k,
            })
            .map(|report| {
                (
                    report.site.as_str().to_string(),
                    report.mean_temperature,
                    report.mean_total_energy,
                    report.mean_vdw_energy,
                    report.mean_electrostatic_energy,
                    report.structural_order,
                    report.crowding_penalty,
                    report.recommended_translation_scale,
                    report.recommended_membrane_scale,
                )
            }))
    }

    fn run_syn3a_subsystem_probe(
        &mut self,
        preset: &str,
    ) -> PyResult<Option<(String, f32, f32, f32, f32, f32, f32, f32, f32)>> {
        let preset = parse_syn3a_subsystem_preset(preset)?;
        Ok(self.inner.run_syn3a_subsystem_probe(preset).map(|report| {
            (
                report.site.as_str().to_string(),
                report.mean_temperature,
                report.mean_total_energy,
                report.mean_vdw_energy,
                report.mean_electrostatic_energy,
                report.structural_order,
                report.crowding_penalty,
                report.recommended_translation_scale,
                report.recommended_membrane_scale,
            )
        }))
    }

    fn add_hotspot(
        &mut self,
        species: &str,
        x: usize,
        y: usize,
        z: usize,
        delta: f32,
    ) -> PyResult<()> {
        let species = parse_intracellular_species(species)?;
        self.inner.add_hotspot(species, x, y, z, delta);
        Ok(())
    }

    fn lattice_shape(&self) -> (usize, usize, usize) {
        self.inner.lattice_shape()
    }

    fn atp_lattice(&self) -> Vec<f32> {
        self.inner.atp_lattice()
    }

    fn snapshot(&self) -> (String, f32, u64, f32, f32, f32, f32, u32, f32, f32) {
        let snapshot = self.inner.snapshot();
        (
            snapshot.backend.as_str().to_string(),
            snapshot.time_ms,
            snapshot.step_count,
            snapshot.atp_mm,
            snapshot.ftsz,
            snapshot.dnaa,
            snapshot.division_progress,
            snapshot.replicated_bp,
            snapshot.surface_area_nm2,
            snapshot.volume_nm3,
        )
    }

    fn quantum_profile(&self) -> (f32, f32, f32, f32, f32) {
        let profile = self.inner.quantum_profile();
        (
            profile.oxphos_efficiency,
            profile.translation_efficiency,
            profile.nucleotide_polymerization_efficiency,
            profile.membrane_synthesis_efficiency,
            profile.chromosome_segregation_efficiency,
        )
    }

    #[getter]
    fn backend(&self) -> String {
        self.inner.backend().as_str().to_string()
    }

    #[getter]
    fn time_ms(&self) -> f32 {
        self.inner.time_ms()
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    #[getter]
    fn atp_mm(&self) -> f32 {
        self.inner.atp_mm()
    }

    #[getter]
    fn ftsz(&self) -> f32 {
        self.inner.ftsz()
    }

    #[getter]
    fn replicated_bp(&self) -> u32 {
        self.inner.replicated_bp()
    }

    #[getter]
    fn division_progress(&self) -> f32 {
        self.inner.division_progress()
    }

    fn __repr__(&self) -> String {
        let snapshot = self.inner.snapshot();
        format!(
            "WholeCellSimulator(backend={}, time_ms={:.3}, ATP={:.3}, replicated_bp={}, division_progress={:.3})",
            snapshot.backend.as_str(),
            snapshot.time_ms,
            snapshot.atp_mm,
            snapshot.replicated_bp,
            snapshot.division_progress,
        )
    }
}

// =============================================================================
// PyBatchedAtomTerrarium
// =============================================================================

#[pyclass(name = "BatchedAtomTerrarium")]
pub struct PyBatchedAtomTerrarium {
    inner: BatchedAtomTerrarium,
}

#[pymethods]
impl PyBatchedAtomTerrarium {
    #[new]
    #[pyo3(signature = (x_dim, y_dim, z_dim, voxel_size_mm=0.5, use_gpu=true))]
    fn new(x_dim: usize, y_dim: usize, z_dim: usize, voxel_size_mm: f32, use_gpu: bool) -> Self {
        Self {
            inner: BatchedAtomTerrarium::new(x_dim, y_dim, z_dim, voxel_size_mm, use_gpu),
        }
    }

    fn seed_default_profile(&mut self) {
        self.inner.seed_default_profile();
    }

    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }

    fn step(&mut self, dt_ms: f32) {
        self.inner.step(dt_ms);
    }

    #[pyo3(signature = (steps, dt_ms=0.25))]
    fn run(&mut self, steps: u64, dt_ms: f32) {
        self.inner.run(steps, dt_ms);
    }

    fn mean_species(&self, species: &str) -> PyResult<f32> {
        let species = parse_terrarium_species(species)?;
        Ok(self.inner.mean_species(species))
    }

    fn species_field(&self, species: &str) -> PyResult<Vec<f32>> {
        let species = parse_terrarium_species(species)?;
        Ok(self.inner.species_field(species).to_vec())
    }

    fn patch_mean_species(
        &self,
        species: &str,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> PyResult<f32> {
        let species = parse_terrarium_species(species)?;
        Ok(self.inner.patch_mean_species(species, x, y, z, radius))
    }

    fn extract_patch_species(
        &mut self,
        species: &str,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        amount: f32,
    ) -> PyResult<f32> {
        let species = parse_terrarium_species(species)?;
        Ok(self
            .inner
            .extract_patch_species(species, x, y, z, radius, amount))
    }

    fn add_hotspot(
        &mut self,
        species: &str,
        x: usize,
        y: usize,
        z: usize,
        amplitude: f32,
    ) -> PyResult<()> {
        let species = parse_terrarium_species(species)?;
        self.inner.add_hotspot(species, x, y, z, amplitude);
        Ok(())
    }

    fn set_hydration_field(&mut self, values: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_hydration_field(&values)
            .map_err(PyValueError::new_err)
    }

    fn set_microbial_activity_field(&mut self, values: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_microbial_activity_field(&values)
            .map_err(PyValueError::new_err)
    }

    fn set_plant_drive_field(&mut self, values: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_plant_drive_field(&values)
            .map_err(PyValueError::new_err)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let snapshot = self.inner.snapshot();
        PyTuple::new(
            py,
            [
                snapshot
                    .backend
                    .as_str()
                    .to_string()
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot.time_ms.into_pyobject(py)?.unbind().into_any(),
                snapshot.step_count.into_pyobject(py)?.unbind().into_any(),
                snapshot
                    .mean_hydration
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot
                    .mean_microbes
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot
                    .mean_plant_drive
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot.mean_glucose.into_pyobject(py)?.unbind().into_any(),
                snapshot
                    .mean_oxygen_gas
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot
                    .mean_ammonium
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot.mean_nitrate.into_pyobject(py)?.unbind().into_any(),
                snapshot
                    .mean_carbon_dioxide
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
                snapshot
                    .mean_atp_flux
                    .into_pyobject(py)?
                    .unbind()
                    .into_any(),
            ],
        )
        .map_err(Into::into)
    }

    #[getter]
    fn backend(&self) -> String {
        self.inner.backend().as_str().to_string()
    }

    #[getter]
    fn time_ms(&self) -> f32 {
        self.inner.time_ms()
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn __repr__(&self) -> String {
        let snapshot = self.inner.snapshot();
        format!(
            "BatchedAtomTerrarium(backend={}, time_ms={:.3}, glucose={:.4}, oxygen_gas={:.4}, atp_flux={:.4})",
            snapshot.backend.as_str(),
            snapshot.time_ms,
            snapshot.mean_glucose,
            snapshot.mean_oxygen_gas,
            snapshot.mean_atp_flux,
        )
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// oNeuro-Metal: GPU-accelerated molecular brain simulator.
#[pymodule]
fn oneuro_metal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMolecularBrain>()?;
    m.add_class::<PyRegionalBrain>()?;
    m.add_class::<PyConsciousnessMetrics>()?;
    m.add_class::<PyTerrariumSensoryField>()?;
    m.add_class::<PyDrosophilaSim>()?;
    m.add_class::<PyDoomBrainSim>()?;
    m.add_class::<PyExperimentResult>()?;
    m.add_class::<PyDoomExperimentResult>()?;
    m.add_class::<PyMolecularRetina>()?;
    m.add_class::<PyCellularMetabolism>()?;
    m.add_class::<PyPlantCellularState>()?;
    m.add_class::<PyPlantOrganism>()?;
    m.add_class::<PyWholeCellSimulator>()?;
    m.add_class::<PyBatchedAtomTerrarium>()?;
    m.add_function(wrap_pyfunction!(build_radial_field, m)?)?;
    m.add_function(wrap_pyfunction!(build_dual_radial_fields, m)?)?;
    m.add_function(wrap_pyfunction!(step_molecular_atmosphere, m)?)?;
    m.add_function(wrap_pyfunction!(step_soil_broad_pools, m)?)?;
    m.add_function(wrap_pyfunction!(extract_root_resources_with_layers, m)?)?;
    m.add_function(wrap_pyfunction!(step_food_patches, m)?)?;
    m.add_function(wrap_pyfunction!(step_seed_bank, m)?)?;

    // Molecular Dynamics
    m.add_class::<molecular_dynamics::PyMD>()?;
    m.add_class::<molecular_dynamics::PyMDStats>()?;
    m.add_class::<neural_molecular_simulator::PyNeuralMDSim>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(has_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}

/// Whether Metal GPU acceleration is available.
#[pyfunction]
fn has_gpu() -> bool {
    crate::gpu::has_gpu()
}

/// Whether CUDA GPU acceleration is available.
#[pyfunction]
fn has_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::has_cuda()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Package version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
