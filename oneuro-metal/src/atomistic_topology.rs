//! Atomistic local-topology templates for whole-cell microdomain refinement.
//!
//! These templates are not a full all-atom cell. They provide reusable,
//! descriptor-driven local assemblies that seed the Rust MD engine with
//! structured atoms, bonded topology, and coarse force-field parameters so the
//! whole-cell bridge can refine selected subsystems from a more physical base
//! than synthetic particle clouds.

use crate::molecular_dynamics::{Angle, Bond, Dihedral, Element, GPUMolecularDynamics};
use serde::Deserialize;
use std::sync::OnceLock;

const ATOMISTIC_TEMPLATE_SPEC_JSON: &str =
    include_str!("../specs/whole_cell_atomistic_templates.json");

#[derive(Debug, Deserialize)]
struct AtomisticTemplateSpec {
    name: String,
    site: String,
    recommended_box_angstrom: f32,
    atoms: Vec<AtomSpec>,
    sequence_topology: SequenceTopologySpec,
    #[serde(default)]
    extra_bonds: Vec<BondSpec>,
}

#[derive(Debug, Deserialize)]
struct AtomSpec {
    element: String,
    position_angstrom: [f32; 3],
    #[serde(default)]
    charge: Option<f32>,
    #[serde(default)]
    sigma: Option<f32>,
    #[serde(default)]
    epsilon: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SequenceTopologySpec {
    bond_r0: f32,
    bond_k: f32,
    angle_theta0_deg: f32,
    angle_ktheta: f32,
    dihedral_period: i32,
    dihedral_phase_deg: f32,
    dihedral_barrier: f32,
    #[serde(default)]
    breaks_after: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct BondSpec {
    i: usize,
    j: usize,
    r0: f32,
    k: f32,
}

#[derive(Debug, Clone)]
pub struct AtomisticAssemblyTemplate {
    pub name: String,
    pub site: String,
    pub recommended_box_angstrom: f32,
    elements: Vec<Element>,
    positions_angstrom: Vec<f32>,
    masses: Vec<f32>,
    charges: Vec<f32>,
    sigma: Vec<f32>,
    epsilon: Vec<f32>,
    bonds: Vec<Bond>,
    angles: Vec<Angle>,
    dihedrals: Vec<Dihedral>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomisticTemplateDescriptor {
    pub polar_fraction: f32,
    pub phosphate_fraction: f32,
    pub hydrogen_fraction: f32,
    pub bond_density: f32,
    pub angle_density: f32,
    pub dihedral_density: f32,
    pub charge_density: f32,
}

impl AtomisticTemplateDescriptor {
    pub const fn new(
        polar_fraction: f32,
        phosphate_fraction: f32,
        hydrogen_fraction: f32,
        bond_density: f32,
        angle_density: f32,
        dihedral_density: f32,
        charge_density: f32,
    ) -> Self {
        Self {
            polar_fraction,
            phosphate_fraction,
            hydrogen_fraction,
            bond_density,
            angle_density,
            dihedral_density,
            charge_density,
        }
    }
}

impl AtomisticAssemblyTemplate {
    pub fn atom_count(&self) -> usize {
        self.masses.len()
    }

    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    pub fn dihedral_count(&self) -> usize {
        self.dihedrals.len()
    }

    pub fn descriptor(&self) -> AtomisticTemplateDescriptor {
        let atom_count = self.atom_count().max(1) as f32;
        let polar_atoms = self
            .elements
            .iter()
            .filter(|element| matches!(element, Element::N | Element::O | Element::P | Element::S))
            .count() as f32;
        let phosphate_atoms = self
            .elements
            .iter()
            .filter(|element| matches!(element, Element::P))
            .count() as f32;
        let hydrogen_atoms = self
            .elements
            .iter()
            .filter(|element| matches!(element, Element::H))
            .count() as f32;
        let charge_density =
            self.charges.iter().map(|charge| charge.abs()).sum::<f32>() / atom_count;

        AtomisticTemplateDescriptor::new(
            polar_atoms / atom_count,
            phosphate_atoms / atom_count,
            hydrogen_atoms / atom_count,
            self.bond_count() as f32 / atom_count,
            self.angle_count() as f32 / atom_count,
            self.dihedral_count() as f32 / atom_count,
            charge_density,
        )
    }

    pub fn centered_positions(&self, box_size_angstrom: f32) -> Vec<f32> {
        let atom_count = self.atom_count().max(1);
        let mut centroid = [0.0f32; 3];
        for i in 0..atom_count {
            let i3 = i * 3;
            centroid[0] += self.positions_angstrom[i3];
            centroid[1] += self.positions_angstrom[i3 + 1];
            centroid[2] += self.positions_angstrom[i3 + 2];
        }
        centroid[0] /= atom_count as f32;
        centroid[1] /= atom_count as f32;
        centroid[2] /= atom_count as f32;

        let center = box_size_angstrom * 0.5;
        let mut shifted = self.positions_angstrom.clone();
        for i in 0..atom_count {
            let i3 = i * 3;
            shifted[i3] = shifted[i3] - centroid[0] + center;
            shifted[i3 + 1] = shifted[i3 + 1] - centroid[1] + center;
            shifted[i3 + 2] = shifted[i3 + 2] - centroid[2] + center;
        }
        shifted
    }

    pub fn configure_md(
        &self,
        md: &mut GPUMolecularDynamics,
        box_size_angstrom: f32,
        temperature_k: f32,
    ) {
        md.set_positions(&self.centered_positions(box_size_angstrom));
        md.set_masses(&self.masses);
        md.set_charges(&self.charges);
        md.set_lj_params(&self.sigma, &self.epsilon);
        md.set_box([box_size_angstrom, box_size_angstrom, box_size_angstrom]);
        md.set_temperature(temperature_k);

        for bond in &self.bonds {
            md.add_bond(bond.i, bond.j, bond.r0, bond.k);
        }
        for angle in &self.angles {
            md.add_angle(angle.i, angle.j, angle.k, angle.theta0, angle.ktheta);
        }
        for dihedral in &self.dihedrals {
            md.add_dihedral(
                dihedral.i,
                dihedral.j,
                dihedral.k,
                dihedral.l,
                dihedral.period,
                dihedral.phase,
                dihedral.barrier,
            );
        }
    }
}

fn build_atomistic_template(
    spec: AtomisticTemplateSpec,
) -> Result<AtomisticAssemblyTemplate, String> {
    if spec.atoms.is_empty() {
        return Err(format!("atomistic template {} has no atoms", spec.name));
    }

    let mut positions_angstrom = Vec::with_capacity(spec.atoms.len() * 3);
    let mut elements = Vec::with_capacity(spec.atoms.len());
    let mut masses = Vec::with_capacity(spec.atoms.len());
    let mut charges = Vec::with_capacity(spec.atoms.len());
    let mut sigma = Vec::with_capacity(spec.atoms.len());
    let mut epsilon = Vec::with_capacity(spec.atoms.len());

    for atom in spec.atoms {
        let element = Element::from_name(&atom.element).ok_or_else(|| {
            format!(
                "unknown atom element in template {}: {}",
                spec.name, atom.element
            )
        })?;
        let (default_sigma, default_epsilon) = element.lj_params();
        elements.push(element);
        positions_angstrom.extend_from_slice(&atom.position_angstrom);
        masses.push(element.mass());
        charges.push(atom.charge.unwrap_or(0.0));
        sigma.push(atom.sigma.unwrap_or(default_sigma));
        epsilon.push(atom.epsilon.unwrap_or(default_epsilon));
    }

    let atom_count = masses.len();
    let breaks = spec.sequence_topology.breaks_after;
    let has_break = |index: usize| breaks.contains(&index);
    let theta0 = spec.sequence_topology.angle_theta0_deg.to_radians();
    let phase = spec.sequence_topology.dihedral_phase_deg.to_radians();

    let mut bonds = Vec::new();
    for i in 0..atom_count.saturating_sub(1) {
        if !has_break(i) {
            bonds.push(Bond {
                i,
                j: i + 1,
                r0: spec.sequence_topology.bond_r0,
                k: spec.sequence_topology.bond_k,
            });
        }
    }
    for bond in spec.extra_bonds {
        bonds.push(Bond {
            i: bond.i,
            j: bond.j,
            r0: bond.r0,
            k: bond.k,
        });
    }

    let mut angles = Vec::new();
    for i in 0..atom_count.saturating_sub(2) {
        if !has_break(i) && !has_break(i + 1) {
            angles.push(Angle {
                i,
                j: i + 1,
                k: i + 2,
                theta0,
                ktheta: spec.sequence_topology.angle_ktheta,
            });
        }
    }

    let mut dihedrals = Vec::new();
    for i in 0..atom_count.saturating_sub(3) {
        if !has_break(i) && !has_break(i + 1) && !has_break(i + 2) {
            dihedrals.push(Dihedral {
                i,
                j: i + 1,
                k: i + 2,
                l: i + 3,
                period: spec.sequence_topology.dihedral_period,
                phase,
                barrier: spec.sequence_topology.dihedral_barrier,
            });
        }
    }

    Ok(AtomisticAssemblyTemplate {
        name: spec.name,
        site: spec.site,
        recommended_box_angstrom: spec.recommended_box_angstrom,
        elements,
        positions_angstrom,
        masses,
        charges,
        sigma,
        epsilon,
        bonds,
        angles,
        dihedrals,
    })
}

fn load_atomistic_templates() -> Result<Vec<AtomisticAssemblyTemplate>, String> {
    let specs: Vec<AtomisticTemplateSpec> = serde_json::from_str(ATOMISTIC_TEMPLATE_SPEC_JSON)
        .map_err(|error| format!("failed to parse atomistic template JSON: {error}"))?;
    specs
        .into_iter()
        .map(build_atomistic_template)
        .collect::<Result<Vec<_>, _>>()
}

pub fn atomistic_assembly_templates() -> &'static [AtomisticAssemblyTemplate] {
    static TEMPLATE_REGISTRY: OnceLock<Vec<AtomisticAssemblyTemplate>> = OnceLock::new();
    TEMPLATE_REGISTRY.get_or_init(|| {
        load_atomistic_templates().unwrap_or_else(|error| {
            panic!("failed to load atomistic whole-cell templates: {error}")
        })
    })
}

pub fn atomistic_template_for_site_name(site: &str) -> Option<&'static AtomisticAssemblyTemplate> {
    atomistic_assembly_templates()
        .iter()
        .find(|template| template.site.eq_ignore_ascii_case(site))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomistic_template_registry_is_complete_for_syn3a_sites() {
        let templates = atomistic_assembly_templates();
        assert_eq!(templates.len(), 4);
        assert!(templates.iter().all(|template| template.atom_count() >= 16));
        assert!(templates.iter().all(|template| template.bond_count() > 0));
        assert!(templates.iter().all(|template| template.angle_count() > 0));
        assert!(templates
            .iter()
            .all(|template| template.dihedral_count() > 0));
    }

    #[test]
    fn atomistic_template_can_configure_md() {
        let template = atomistic_template_for_site_name("ribosome_cluster")
            .expect("ribosome atomistic template");
        let mut md = GPUMolecularDynamics::new(template.atom_count(), "cpu");
        template.configure_md(&mut md, template.recommended_box_angstrom, 310.0);
        md.initialize_velocities();
        let stats = md.step(0.001);

        assert!(stats.total_energy.is_finite());
        assert_eq!(md.positions().len(), template.atom_count() * 3);
    }

    #[test]
    fn atomistic_template_descriptor_tracks_composition_and_topology() {
        let template = atomistic_template_for_site_name("atp_synthase_band")
            .expect("ATP synthase atomistic template");
        let descriptor = template.descriptor();

        assert!(descriptor.phosphate_fraction > 0.0);
        assert!(descriptor.polar_fraction >= descriptor.phosphate_fraction);
        assert!(descriptor.bond_density > 0.0);
        assert!(descriptor.angle_density > 0.0);
        assert!(descriptor.dihedral_density > 0.0);
        assert!(descriptor.charge_density > 0.0);
    }
}
