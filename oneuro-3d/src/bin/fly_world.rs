//! oNeuro 3D World - Realistic Fly Simulation

use bevy::prelude::*;
use std::f32::consts::PI;

const FLY_SCALE: f32 = 0.1;
const HEAD_RADIUS: f32 = 0.3 * FLY_SCALE;
const THORAX_LENGTH: f32 = 1.0 * FLY_SCALE;
const THORAX_WIDTH: f32 = 0.6 * FLY_SCALE;
const ABDOMEN_LENGTH: f32 = 1.5 * FLY_SCALE;
const ABDOMEN_WIDTH: f32 = 0.5 * FLY_SCALE;
const WING_LENGTH: f32 = 2.0 * FLY_SCALE;
const WING_WIDTH: f32 = 0.7 * FLY_SCALE;
const WORLD_SIZE: f32 = 50.0;

#[derive(Component)]
pub struct MolecularBrain { pub neurons: usize, pub synapses: usize, pub step_count: u64 }

impl MolecularBrain {
    pub fn new(neurons: usize, synapses: usize) -> Self { Self { neurons, synapses, step_count: 0 } }
    pub fn step(&mut self) { self.step_count += 1; }
}

#[derive(Component)]
pub struct Fly { pub energy: f32, pub age: f32, pub alive: bool, pub speed: f32, pub turn_rate: f32 }
impl Fly { pub fn new() -> Self { Self { energy: 100.0, age: 0.0, alive: true, speed: 0.5, turn_rate: 0.0 } } }
impl Default for Fly { fn default() -> Self { Self::new() } }

#[derive(Component)]
pub struct Food { pub nutrition: f32 }
impl Food { pub fn new(nutrition: f32) -> Self { Self { nutrition } } }

fn noise_height(x: f32, z: f32) -> f32 { (x.sin() + z.sin()) * 0.5 + 2.0 }

fn setup_scene(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });

    // Ambient light
    commands.insert_resource(AmbientLight { color: Color::rgb(0.4, 0.4, 0.5), brightness: 0.5 });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::rgb(1.0, 0.95, 0.8),
            illuminance: 100000.0,
            ..Default::default()
        },
        transform: Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn setup_terrain(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let terrain_size = 12usize;
    let cell_size = 1.0;
    let grass_mat = materials.add(StandardMaterial { base_color: Color::rgb(0.2, 0.6, 0.1), ..Default::default() });
    let dirt_mat = materials.add(StandardMaterial { base_color: Color::rgb(0.4, 0.25, 0.1), ..Default::default() });

    for x in 0..terrain_size {
        for z in 0..terrain_size {
            let height = noise_height(x as f32, z as f32);
            let gx = (x as f32 - terrain_size as f32 / 2.0) * cell_size;
            let gz = (z as f32 - terrain_size as f32 / 2.0) * cell_size;
            commands.spawn(PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: cell_size * 0.95 })),
                material: grass_mat.clone(),
                transform: Transform::from_xyz(gx, height, gz),
                ..Default::default()
            });
            commands.spawn(PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Cube { size: cell_size * 0.95 })),
                material: dirt_mat.clone(),
                transform: Transform::from_xyz(gx, height - 0.3, gz),
                ..Default::default()
            });
        }
    }
}

fn setup_fly(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let fly_mat = materials.add(StandardMaterial { base_color: Color::rgb(0.15, 0.12, 0.1), ..Default::default() });
    let eye_mat = materials.add(StandardMaterial { base_color: Color::rgb(0.8, 0.1, 0.1), ..Default::default() });
    let wing_mat = materials.add(StandardMaterial { base_color: Color::rgba(0.7, 0.7, 0.8, 0.4), alpha_mode: AlphaMode::Blend, ..Default::default() });

    // Fly parent entity
    commands.spawn((Transform::from_xyz(0.0, 3.0, 0.0), Fly::new(), MolecularBrain::new(1000, 10000)));

    // Head
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere { radius: HEAD_RADIUS, ..Default::default() })),
        material: fly_mat.clone(),
        transform: Transform::from_xyz(0.0, THORAX_LENGTH / 2.0 + HEAD_RADIUS + 3.0, 0.0),
        ..Default::default()
    });

    // Eyes
    for x_off in [-1.0_f32, 1.0] {
        commands.spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere { radius: HEAD_RADIUS * 0.6, ..Default::default() })),
            material: eye_mat.clone(),
            transform: Transform::from_xyz(x_off * HEAD_RADIUS * 0.8, THORAX_LENGTH / 2.0 + HEAD_RADIUS * 0.5 + 3.0, HEAD_RADIUS * 0.5),
            ..Default::default()
        });
    }

    // Thorax
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cylinder { radius: THORAX_WIDTH / 2.0, height: THORAX_LENGTH, ..Default::default() })),
        material: fly_mat.clone(),
        transform: Transform::from_xyz(0.0, 3.0, 0.0).with_rotation(Quat::from_rotation_x(PI / 2.0)),
        ..Default::default()
    });

    // Abdomen
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cylinder { radius: ABDOMEN_WIDTH / 2.0, height: ABDOMEN_LENGTH, ..Default::default() })),
        material: fly_mat.clone(),
        transform: Transform::from_xyz(0.0, 3.0 - THORAX_LENGTH / 2.0 - ABDOMEN_LENGTH / 2.0, 0.0).with_rotation(Quat::from_rotation_x(PI / 2.0)),
        ..Default::default()
    });

    // Wings
    for x_off in [-1.0_f32, 1.0] {
        commands.spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad { size: Vec2::new(WING_LENGTH, WING_WIDTH), ..Default::default() })),
            material: wing_mat.clone(),
            transform: Transform::from_xyz(x_off * (THORAX_WIDTH / 2.0 + 0.01), 3.05 * FLY_SCALE, 0.1).with_rotation(Quat::from_rotation_y(x_off * 0.3)),
            ..Default::default()
        });
    }
}

fn spawn_food(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let fruit_mat = materials.add(StandardMaterial { base_color: Color::rgb(1.0, 0.3, 0.1), ..Default::default() });
    for _ in 0..5 {
        let x = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 0.8;
        let z = (rand::random::<f32>() - 0.5) * WORLD_SIZE * 0.8;
        commands.spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere { radius: 0.3, ..Default::default() })),
            material: fruit_mat.clone(),
            transform: Transform::from_xyz(x, 0.5, z),
            ..Default::default()
        });
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn(TextBundle {
        text: Text::from_section("oNeuro 3D Fly World", TextStyle { font_size: 30.0, color: Color::WHITE, ..Default::default() }),
        style: Style { position_type: PositionType::Absolute, position: UiRect { top: Val::Px(10.0), left: Val::Px(10.0), ..Default::default() }, ..Default::default() },
        ..Default::default()
    });
}

fn brain_tick(mut query: Query<(&mut MolecularBrain, &mut Fly)>, time: Res<Time>) {
    for (mut brain, mut fly) in query.iter_mut() {
        if !fly.alive { continue; }
        brain.step();
        fly.age += time.delta_seconds();
        fly.energy -= 0.1 * time.delta_seconds();
        if fly.energy <= 0.0 { fly.alive = false; }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_startup_system(setup_scene)
        .add_startup_system(setup_terrain)
        .add_startup_system(spawn_food)
        .add_startup_system(setup_fly)
        .add_startup_system(setup_ui)
        .add_system(brain_tick)
        .run();
}
