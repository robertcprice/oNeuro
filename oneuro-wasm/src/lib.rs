//! oNeuro WASM - WebGPU-based neural-molecular simulator
//!
//! This module provides GPU-accelerated molecular dynamics in the browser
//! using WebGPU for compute shaders.

use wasm_bindgen::prelude::*;

/// Molecular dynamics simulation state.
#[wasm_bindgen]
pub struct MDSimulation {
    n_atoms: usize,
    positions: Vec<f32>,
    velocities: Vec<f32>,
    forces: Vec<f32>,
    masses: Vec<f32>,
    sigma: Vec<f32>,
    epsilon: Vec<f32>,
    charges: Vec<f32>,
    temperature: f32,
    box_size: [f32; 3],
    cutoff: f32,
}

#[wasm_bindgen]
impl MDSimulation {
    /// Create a new MD simulation.
    #[wasm_bindgen(constructor)]
    pub fn new(n_atoms: usize) -> MDSimulation {
        let mut positions = vec![0.0f32; n_atoms * 3];
        let mut velocities = vec![0.0f32; n_atoms * 3];
        let mut forces = vec![0.0f32; n_atoms * 3];
        let mut masses = vec![1.0f32; n_atoms];
        let mut sigma = vec![3.4f32; n_atoms];
        let mut epsilon = vec![0.1f32; n_atoms];
        let mut charges = vec![0.0f32; n_atoms];

        // Random initial positions
        for i in 0..n_atoms {
            let i3 = i * 3;
            positions[i3] = rand::random::<f32>() * 50.0;
            positions[i3 + 1] = rand::random::<f32>() * 50.0;
            positions[i3 + 2] = rand::random::<f32>() * 50.0;

            // Random initial velocities (small)
            velocities[i3] = (rand::random::<f32>() - 0.5) * 0.1;
            velocities[i3 + 1] = (rand::random::<f32>() - 0.5) * 0.1;
            velocities[i3 + 2] = (rand::random::<f32>() - 0.5) * 0.1;
        }

        MDSimulation {
            n_atoms,
            positions,
            velocities,
            forces,
            masses,
            sigma,
            epsilon,
            charges,
            temperature: 300.0,
            box_size: [50.0, 50.0, 50.0],
            cutoff: 12.0,
        }
    }

    /// Run one simulation step.
    pub fn step(&mut self, dt: f32) {
        let kb = 0.001987;  // kcal/mol/K
        let gamma = 1.0;    // Friction

        // Clear forces
        self.forces.fill(0.0);

        // Compute LJ forces (simple O(N²) for now)
        let cutoff_sq = self.cutoff * self.cutoff;
        let box_size = self.box_size;

        for i in 0..self.n_atoms {
            let i3 = i * 3;

            for j in (i + 1)..self.n_atoms {
                let j3 = j * 3;

                let mut dx = self.positions[j3] - self.positions[i3];
                let mut dy = self.positions[j3 + 1] - self.positions[i3 + 1];
                let mut dz = self.positions[j3 + 2] - self.positions[i3 + 2];

                // PBC
                if box_size[0] > 0.0 {
                    dx -= box_size[0] * (dx / box_size[0]).round();
                    dy -= box_size[1] * (dy / box_size[1]).round();
                    dz -= box_size[2] * (dz / box_size[2]).round();
                }

                let r2 = dx * dx + dy * dy + dz * dz;

                if r2 > cutoff_sq || r2 < 1e-6 { continue; }

                let r = r2.sqrt();
                let sig_ij = 0.5 * (self.sigma[i] + self.sigma[j]);
                let eps_ij = (self.epsilon[i] * self.epsilon[j]).sqrt();

                // LJ
                let sig_r = sig_ij / r;
                let sig_r6 = sig_r.powi(6);
                let sig_r12 = sig_r6 * sig_r6;

                let force_mag = 12.0 * eps_ij * (sig_r12 - sig_r6) / r;

                let fx = force_mag * dx / r;
                let fy = force_mag * dy / r;
                let fz = force_mag * dz / r;

                self.forces[i3] += fx;
                self.forces[i3 + 1] += fy;
                self.forces[i3 + 2] += fz;
                self.forces[j3] -= fx;
                self.forces[j3 + 1] -= fy;
                self.forces[j3 + 2] -= fz;
            }
        }

        // Integrate with Langevin
        let sqrt_kbt = (kb * self.temperature).sqrt();
        let gamma_dt = gamma * dt;

        for i in 0..self.n_atoms {
            let i3 = i * 3;
            let m = self.masses[i];
            let inv_m = 1.0 / m;

            // Random force
            let rx: f32 = rand::random();
            let ry: f32 = rand::random();
            let rz: f32 = rand::random();
            let rand_f = sqrt_kbt * ((rx + ry + rz) - 1.5) * (2.0 * gamma_dt).sqrt() * inv_m;

            // Velocity Verlet with Langevin
            let vx = self.velocities[i3] + (self.forces[i3] * inv_m + rand_f) * dt;
            let vy = self.velocities[i3 + 1] + (self.forces[i3 + 1] * inv_m + rand_f) * dt;
            let vz = self.velocities[i3 + 2] + (self.forces[i3 + 2] * inv_m + rand_f) * dt;

            // Damping
            let damp = (-gamma_dt * 0.5).exp();
            self.velocities[i3] = vx * damp;
            self.velocities[i3 + 1] = vy * damp;
            self.velocities[i3 + 2] = vz * damp;

            // Position update
            self.positions[i3] += self.velocities[i3] * dt;
            self.positions[i3 + 1] += self.velocities[i3 + 1] * dt;
            self.positions[i3 + 2] += self.velocities[i3 + 2] * dt;

            // PBC
            for dim in 0..3 {
                let pos = self.positions[i3 + dim];
                let box_size = self.box_size[dim];
                if box_size > 0.0 {
                    self.positions[i3 + dim] = pos - box_size * (pos / box_size).floor();
                }
            }
        }
    }

    /// Get positions as flat array.
    pub fn get_positions(&self) -> Vec<f32> {
        self.positions.clone()
    }

    /// Get velocities as flat array.
    pub fn get_velocities(&self) -> Vec<f32> {
        self.velocities.clone()
    }

    /// Set temperature.
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp;
    }

    /// Set box size.
    pub fn set_box(&mut self, x: f32, y: f32, z: f32) {
        self.box_size = [x, y, z];
    }

    /// Get number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Compute kinetic energy.
    pub fn kinetic_energy(&self) -> f32 {
        let mut ke = 0.0f32;
        for i in 0..self.n_atoms {
            let i3 = i * 3;
            ke += 0.5 * self.masses[i] * (
                self.velocities[i3].powi(2) +
                self.velocities[i3 + 1].powi(2) +
                self.velocities[i3 + 2].powi(2)
            );
        }
        ke
    }

    /// Compute temperature from kinetic energy.
    pub fn compute_temperature(&self) -> f32 {
        let ke = self.kinetic_energy();
        2.0 * ke / (3.0 * self.n_atoms as f32 * 0.001987)
    }
}

/// Initialize logging for WASM.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"oNeuro WASM initialized".into());
}

mod console_error_panic_hook {
    use std::panic;
    pub fn set_once() {
        static SET: std::sync::Once = std::sync::Once::new();
        SET.call_once(|| {
            panic::set_hook(Box::new(|info| {
                web_sys::console::error_1(&format!("panic: {:?}", info).into());
            }));
        });
    }
}
