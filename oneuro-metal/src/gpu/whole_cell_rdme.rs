//! GPU and CPU execution for whole-cell intracellular reaction-diffusion.
//!
//! The lattice stores a small set of core intracellular species in
//! structure-of-arrays form so each species can be diffused independently
//! while keeping the memory layout GPU-friendly.

use rayon::prelude::*;
use std::ops::Range;

#[cfg(target_os = "macos")]
use super::GpuContext;

/// Core intracellular species tracked on the lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IntracellularSpecies {
    ATP = 0,
    AminoAcids = 1,
    Nucleotides = 2,
    MembranePrecursors = 3,
}

impl IntracellularSpecies {
    pub const COUNT: usize = 4;

    pub fn index(self) -> usize {
        self as usize
    }

    fn diffusion_coeff_nm2_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 60_000.0,
            IntracellularSpecies::AminoAcids => 40_000.0,
            IntracellularSpecies::Nucleotides => 28_000.0,
            IntracellularSpecies::MembranePrecursors => 10_000.0,
        }
    }

    fn basal_source_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 0.012,
            IntracellularSpecies::AminoAcids => 0.006,
            IntracellularSpecies::Nucleotides => 0.004,
            IntracellularSpecies::MembranePrecursors => 0.002,
        }
    }

    fn basal_sink_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 0.010,
            IntracellularSpecies::AminoAcids => 0.005,
            IntracellularSpecies::Nucleotides => 0.003,
            IntracellularSpecies::MembranePrecursors => 0.0015,
        }
    }
}

const SPECIES_ORDER: [IntracellularSpecies; IntracellularSpecies::COUNT] = [
    IntracellularSpecies::ATP,
    IntracellularSpecies::AminoAcids,
    IntracellularSpecies::Nucleotides,
    IntracellularSpecies::MembranePrecursors,
];

fn stable_rdme_substeps(voxel_size_nm: f32, dt: f32) -> (u32, f32) {
    let dx2 = voxel_size_nm * voxel_size_nm;
    let max_diffusion = SPECIES_ORDER
        .iter()
        .map(|species| species.diffusion_coeff_nm2_per_ms())
        .fold(0.0, f32::max)
        .max(1.0);
    let max_stable_dt = (0.16 * dx2 / max_diffusion).max(1.0e-4);
    let substeps = (dt / max_stable_dt).ceil().max(1.0) as u32;
    let sub_dt = dt / substeps as f32;
    (substeps.max(1), sub_dt.max(1.0e-4))
}

/// Voxelized intracellular state used by the native whole-cell runtime.
pub struct IntracellularLattice {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_nm: f32,
    pub current: Vec<f32>,
    pub next: Vec<f32>,
}

impl IntracellularLattice {
    /// Create a new zero-filled lattice.
    pub fn new(x_dim: usize, y_dim: usize, z_dim: usize, voxel_size_nm: f32) -> Self {
        let total = IntracellularSpecies::COUNT * x_dim * y_dim * z_dim;
        Self {
            x_dim,
            y_dim,
            z_dim,
            voxel_size_nm,
            current: vec![0.0; total],
            next: vec![0.0; total],
        }
    }

    /// Total voxels in a single species channel.
    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    fn channel_range(&self, species: IntracellularSpecies) -> Range<usize> {
        let total_voxels = self.total_voxels();
        let start = species.index() * total_voxels;
        start..start + total_voxels
    }

    /// Set a channel to a uniform value.
    pub fn fill_species(&mut self, species: IntracellularSpecies, value: f32) {
        let range = self.channel_range(species);
        self.current[range.clone()].fill(value);
        self.next[range].fill(value);
    }

    /// Add a localized hotspot used for testing or seeding gradients.
    pub fn add_hotspot(
        &mut self,
        species: IntracellularSpecies,
        x: usize,
        y: usize,
        z: usize,
        delta: f32,
    ) {
        if x >= self.x_dim || y >= self.y_dim || z >= self.z_dim {
            return;
        }
        let idx = z * self.y_dim * self.x_dim + y * self.x_dim + x;
        let range = self.channel_range(species);
        self.current[range.start + idx] = (self.current[range.start + idx] + delta).max(0.0);
        self.next[range.start + idx] = self.current[range.start + idx];
    }

    /// Mean concentration for a species.
    pub fn mean_species(&self, species: IntracellularSpecies) -> f32 {
        let range = self.channel_range(species);
        let total = self.total_voxels() as f32;
        if total <= 0.0 {
            return 0.0;
        }
        self.current[range].iter().sum::<f32>() / total
    }

    /// Uniformly perturb a species channel.
    pub fn apply_uniform_delta(&mut self, species: IntracellularSpecies, delta: f32) {
        let range = self.channel_range(species);
        self.current[range.clone()]
            .par_iter_mut()
            .for_each(|value| *value = (*value + delta).max(0.0));
        let source = self.current[range.clone()].to_vec();
        self.next[range].copy_from_slice(&source);
    }

    /// Return a copy of a channel for callers that need direct inspection.
    pub fn clone_species(&self, species: IntracellularSpecies) -> Vec<f32> {
        self.current[self.channel_range(species)].to_vec()
    }

    /// Replace a species channel with explicit values.
    pub fn set_species(
        &mut self,
        species: IntracellularSpecies,
        values: &[f32],
    ) -> Result<(), String> {
        let expected = self.total_voxels();
        if values.len() != expected {
            return Err(format!(
                "species channel length mismatch: expected {}, got {}",
                expected,
                values.len()
            ));
        }
        let range = self.channel_range(species);
        self.current[range.clone()].copy_from_slice(values);
        self.next[range].copy_from_slice(values);
        Ok(())
    }

    /// Swap read/write buffers after an RDME update.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Params struct matching the Metal shader constant buffer.
#[repr(C)]
struct WholeCellRdmeParams {
    x_dim: u32,
    y_dim: u32,
    z_dim: u32,
    voxel_size_nm: f32,
    dt: f32,
    metabolic_load: f32,
}

/// Dispatch whole-cell RDME on Metal.
#[cfg(target_os = "macos")]
pub fn dispatch_whole_cell_rdme(
    gpu: &GpuContext,
    lattice: &mut IntracellularLattice,
    dt: f32,
    metabolic_load: f32,
) {
    let total_voxels = lattice.total_voxels() as u64;
    if total_voxels == 0 {
        return;
    }
    let (substeps, sub_dt) = stable_rdme_substeps(lattice.voxel_size_nm, dt.max(1.0e-4));
    for _ in 0..substeps {
        let buf_grid_in = gpu.buffer_from_slice(&lattice.current);
        let buf_grid_out = gpu.buffer_from_slice(&lattice.next);

        let params = WholeCellRdmeParams {
            x_dim: lattice.x_dim as u32,
            y_dim: lattice.y_dim as u32,
            z_dim: lattice.z_dim as u32,
            voxel_size_nm: lattice.voxel_size_nm,
            dt: sub_dt,
            metabolic_load,
        };
        let param_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const WholeCellRdmeParams as *const u8,
                std::mem::size_of::<WholeCellRdmeParams>(),
            )
        };

        gpu.dispatch_1d(
            &gpu.pipelines.whole_cell_rdme,
            &[(&buf_grid_in, 0), (&buf_grid_out, 0)],
            Some((param_bytes, 2)),
            total_voxels,
        );

        unsafe {
            let ptr = buf_grid_out.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, lattice.next.as_mut_ptr(), lattice.next.len());
        }

        lattice.swap_buffers();
    }
}

/// CPU fallback for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub fn dispatch_whole_cell_rdme(
    _gpu: &super::GpuContext,
    lattice: &mut IntracellularLattice,
    dt: f32,
    metabolic_load: f32,
) {
    cpu_whole_cell_rdme(lattice, dt, metabolic_load);
}

/// CPU reference implementation of the intracellular RDME step.
pub fn cpu_whole_cell_rdme(lattice: &mut IntracellularLattice, dt: f32, metabolic_load: f32) {
    let x_dim = lattice.x_dim;
    let y_dim = lattice.y_dim;
    let z_dim = lattice.z_dim;
    let total_voxels = lattice.total_voxels();
    if total_voxels == 0 {
        return;
    }
    let dx2 = lattice.voxel_size_nm * lattice.voxel_size_nm;
    let (substeps, sub_dt) = stable_rdme_substeps(lattice.voxel_size_nm, dt.max(1.0e-4));

    for _ in 0..substeps {
        for species in SPECIES_ORDER {
            let range = lattice.channel_range(species);
            let current = &lattice.current[range.clone()];
            let next = &mut lattice.next[range];
            let coeff = species.diffusion_coeff_nm2_per_ms() / dx2 * sub_dt;
            let source = species.basal_source_per_ms() * sub_dt;
            let sink = species.basal_sink_per_ms() * sub_dt * metabolic_load.max(0.1);

            next.par_iter_mut().enumerate().for_each(|(gid, out)| {
                let z = gid / (y_dim * x_dim);
                let rem = gid - z * y_dim * x_dim;
                let y = rem / x_dim;
                let x = rem - y * x_dim;

                let c = current[gid];

                let right = if x + 1 < x_dim {
                    current[z * y_dim * x_dim + y * x_dim + (x + 1)]
                } else {
                    c
                };
                let left = if x > 0 {
                    current[z * y_dim * x_dim + y * x_dim + (x - 1)]
                } else {
                    c
                };
                let up = if y + 1 < y_dim {
                    current[z * y_dim * x_dim + (y + 1) * x_dim + x]
                } else {
                    c
                };
                let down = if y > 0 {
                    current[z * y_dim * x_dim + (y - 1) * x_dim + x]
                } else {
                    c
                };
                let front = if z + 1 < z_dim {
                    current[(z + 1) * y_dim * x_dim + y * x_dim + x]
                } else {
                    c
                };
                let back = if z > 0 {
                    current[(z - 1) * y_dim * x_dim + y * x_dim + x]
                } else {
                    c
                };

                let laplacian = right + left + up + down + front + back - 6.0 * c;
                let updated = c + coeff * laplacian + source - sink * c;
                *out = updated.max(0.0);
            });
        }
        lattice.swap_buffers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_rdme_substeps_keep_default_dt_finite() {
        let mut lattice = IntracellularLattice::new(24, 24, 12, 20.0);
        lattice.fill_species(IntracellularSpecies::ATP, 1.2);
        lattice.fill_species(IntracellularSpecies::AminoAcids, 0.95);
        lattice.fill_species(IntracellularSpecies::Nucleotides, 0.80);
        lattice.fill_species(IntracellularSpecies::MembranePrecursors, 0.35);
        lattice.add_hotspot(IntracellularSpecies::ATP, 12, 12, 6, 2.5);

        for _ in 0..32 {
            cpu_whole_cell_rdme(&mut lattice, 0.25, 1.0);
        }

        for species in SPECIES_ORDER {
            let values = lattice.clone_species(species);
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(lattice.mean_species(species).is_finite());
        }
    }
}
