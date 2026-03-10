#![allow(dead_code)]

use bevy::{
    asset::{load_internal_asset, HandleUntyped},
    app::AppExit,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, Shader, ShaderRef, ShaderType, TextureDimension,
            TextureFormat, TextureUsages,
        },
        texture::ImagePlugin,
    },
    sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle},
    window::{PresentMode, WindowPlugin},
};
use oneuro_metal::{TerrariumTopdownView, TerrariumWorld, TerrariumWorldSnapshot};
use std::env;

const TERRARIUM_GPU_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 0x5A17_9D33_4C82_11E7);

#[derive(Debug, Clone)]
struct Cli {
    seed: u64,
    fps: f32,
    frames: Option<usize>,
    cpu_substrate: bool,
    cell_px: f32,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: 7,
            fps: 30.0,
            frames: None,
            cpu_substrate: false,
            cell_px: 18.0,
        }
    }
}

#[derive(Resource, Clone)]
struct RenderTargets {
    field_image: Handle<Image>,
    material: Handle<TerrariumMaterial>,
}

struct SimState {
    world: TerrariumWorld,
    snapshot: TerrariumWorldSnapshot,
    view: TerrariumTopdownView,
    paused: bool,
    single_step: bool,
    fps: f32,
    accumulator: f32,
    frame_limit: Option<usize>,
    frame_idx: usize,
    seed: u64,
    use_gpu_substrate: bool,
    cell_px: f32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, ShaderType)]
struct TerrariumMaterialUniform {
    tone: Vec4,
    bio: Vec4,
    field_stats: Vec4,
    overlay_counts: Vec4,
    water_points: [Vec4; MAX_WATER_POINTS],
    plant_points: [Vec4; MAX_PLANT_POINTS],
    fruit_points: [Vec4; MAX_FRUIT_POINTS],
    fly_points: [Vec4; MAX_FLY_POINTS],
}

#[derive(AsBindGroup, TypeUuid, Clone)]
#[uuid = "9f3a5662-6784-4386-b9d1-d1ef2c1015af"]
struct TerrariumMaterial {
    #[texture(0)]
    #[sampler(1)]
    field_image: Handle<Image>,
    #[uniform(2)]
    uniform: TerrariumMaterialUniform,
}

impl Material2d for TerrariumMaterial {
    fn fragment_shader() -> ShaderRef {
        TERRARIUM_GPU_SHADER_HANDLE.typed::<Shader>().into()
    }
}

struct TerrariumMaterialPlugin;

impl Plugin for TerrariumMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            TERRARIUM_GPU_SHADER_HANDLE,
            "../shaders/terrarium_gpu_material.wgsl",
            Shader::from_wgsl
        );
        app.add_plugin(Material2dPlugin::<TerrariumMaterial>::default());
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --bin terrarium_gpu -- [--seed <n>] [--fps <n>] [--frames <n>] [--cell <px>] [--cpu-substrate]"
    );
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                cli.seed = args
                    .next()
                    .ok_or_else(|| "missing value for --seed".to_string())?
                    .parse()
                    .map_err(|_| "invalid integer for --seed".to_string())?;
            }
            "--fps" => {
                cli.fps = args
                    .next()
                    .ok_or_else(|| "missing value for --fps".to_string())?
                    .parse()
                    .map_err(|_| "invalid number for --fps".to_string())?;
            }
            "--frames" => {
                cli.frames = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --frames".to_string())?
                        .parse()
                        .map_err(|_| "invalid integer for --frames".to_string())?,
                );
            }
            "--cell" => {
                cli.cell_px = args
                    .next()
                    .ok_or_else(|| "missing value for --cell".to_string())?
                    .parse()
                    .map_err(|_| "invalid number for --cell".to_string())?;
                cli.cell_px = cli.cell_px.clamp(6.0, 40.0);
            }
            "--cpu-substrate" => cli.cpu_substrate = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    cli.fps = cli.fps.max(1.0);
    Ok(cli)
}

fn normalize_field(values: &[f32]) -> Vec<f32> {
    let peak = values.iter().copied().fold(0.0f32, f32::max);
    if peak <= 1.0e-9 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| *value / peak).collect()
    }
}

const MAX_WATER_POINTS: usize = 128;
const MAX_PLANT_POINTS: usize = 256;
const MAX_FRUIT_POINTS: usize = 256;
const MAX_FLY_POINTS: usize = 128;

fn encode_field(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

fn field_stats(values: &[f32]) -> Vec4 {
    if values.is_empty() {
        return Vec4::new(0.0, 0.0, 1.0, 0.0);
    }
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for value in values {
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
    }
    let span = (max_value - min_value).max(1.0e-6);
    Vec4::new(min_value, max_value, 1.0 / span, normalize_field(values).iter().copied().fold(0.0, f32::max))
}

fn store_point(points: &mut [Vec4], count: &mut usize, point: Vec4) {
    if *count < points.len() {
        points[*count] = point;
        *count += 1;
    }
}

fn encode_overlay_uniform(world: &TerrariumWorld) -> TerrariumMaterialUniform {
    let mut water_points = [Vec4::ZERO; MAX_WATER_POINTS];
    let mut plant_points = [Vec4::ZERO; MAX_PLANT_POINTS];
    let mut fruit_points = [Vec4::ZERO; MAX_FRUIT_POINTS];
    let mut fly_points = [Vec4::ZERO; MAX_FLY_POINTS];
    let mut water_count = 0usize;
    let mut plant_count = 0usize;
    let mut fruit_count = 0usize;
    let mut fly_count = 0usize;

    for water in &world.waters {
        if water.alive {
            store_point(
                &mut water_points,
                &mut water_count,
                Vec4::new(water.x as f32 + 0.5, water.y as f32 + 0.5, 1.0, 1.35),
            );
        }
    }
    for plant in &world.plants {
        store_point(
            &mut plant_points,
            &mut plant_count,
            Vec4::new(plant.x as f32 + 0.5, plant.y as f32 + 0.5, 1.0, 1.30),
        );
    }
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            store_point(
                &mut fruit_points,
                &mut fruit_count,
                Vec4::new(
                    fruit.source.x as f32 + 0.5,
                    fruit.source.y as f32 + 0.5,
                    fruit.source.sugar_content.clamp(0.15, 1.0),
                    1.15,
                ),
            );
        }
    }
    for fly in &world.flies {
        let body = fly.body_state();
        store_point(
            &mut fly_points,
            &mut fly_count,
            Vec4::new(
                body.x,
                body.y,
                if body.is_flying { 1.0 } else { 0.72 },
                if body.is_flying { 1.40 } else { 1.05 },
            ),
        );
    }

    TerrariumMaterialUniform {
        tone: Vec4::ZERO,
        bio: Vec4::ZERO,
        field_stats: Vec4::ZERO,
        overlay_counts: Vec4::new(
            water_count as f32,
            plant_count as f32,
            fruit_count as f32,
            fly_count as f32,
        ),
        water_points,
        plant_points,
        fruit_points,
        fly_points,
    }
}

fn terrarium_uniform(state: &SimState) -> TerrariumMaterialUniform {
    let field = state.world.topdown_field(state.view);
    let mut uniform = encode_overlay_uniform(&state.world);
    let view_id = match state.view {
        TerrariumTopdownView::Terrain => 0.0,
        TerrariumTopdownView::SoilMoisture => 1.0,
        TerrariumTopdownView::Canopy => 2.0,
        TerrariumTopdownView::Chemistry => 3.0,
        TerrariumTopdownView::Odor => 4.0,
    };
    uniform.tone = Vec4::new(
        state.snapshot.light.clamp(0.0, 1.0),
        if state.paused { 1.0 } else { 0.0 },
        view_id,
        (state.snapshot.time_s / 86_400.0).fract(),
    );
    uniform.bio = Vec4::new(
        state.snapshot.food_remaining.clamp(0.0, 1.0),
        state.snapshot.mean_cell_vitality.clamp(0.0, 1.0),
        state.snapshot.humidity.clamp(0.0, 1.0),
        state.snapshot.mean_cell_energy.clamp(0.0, 1.0),
    );
    uniform.field_stats = field_stats(&field);
    uniform
}

fn terrarium_field_bytes(world: &TerrariumWorld, view: TerrariumTopdownView) -> Vec<u8> {
    encode_field(&world.topdown_field(view))
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrariumMaterial>>,
    state: NonSend<SimState>,
) {
    let width = state.world.width();
    let height = state.world.height();
    let uniform = terrarium_uniform(&state);
    let mut field_image = Image::new_fill(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
    );
    field_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    field_image.data = terrarium_field_bytes(&state.world, state.view);
    let field_image_handle = images.add(field_image);
    let material_handle = materials.add(TerrariumMaterial {
        field_image: field_image_handle.clone(),
        uniform,
    });
    let quad = meshes.add(Mesh::from(shape::Quad::new(Vec2::new(
        width as f32 * state.cell_px,
        height as f32 * state.cell_px,
    ))));

    commands.insert_resource(RenderTargets {
        field_image: field_image_handle.clone(),
        material: material_handle.clone(),
    });

    commands.spawn(Camera2dBundle::default());
    commands.spawn(MaterialMesh2dBundle {
        mesh: quad.into(),
        material: material_handle,
        ..default()
    });
}

fn handle_input(
    keyboard: Res<Input<KeyCode>>,
    mut state: NonSendMut<SimState>,
    mut exit: EventWriter<AppExit>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
    }
    if keyboard.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }
    if keyboard.just_pressed(KeyCode::Right) {
        state.single_step = true;
    }
    if keyboard.just_pressed(KeyCode::Key1) {
        state.view = TerrariumTopdownView::Terrain;
    }
    if keyboard.just_pressed(KeyCode::Key2) {
        state.view = TerrariumTopdownView::SoilMoisture;
    }
    if keyboard.just_pressed(KeyCode::Key3) {
        state.view = TerrariumTopdownView::Canopy;
    }
    if keyboard.just_pressed(KeyCode::Key4) {
        state.view = TerrariumTopdownView::Chemistry;
    }
    if keyboard.just_pressed(KeyCode::Key5) {
        state.view = TerrariumTopdownView::Odor;
    }
    if keyboard.just_pressed(KeyCode::Up) {
        state.fps = (state.fps + 5.0).min(120.0);
    }
    if keyboard.just_pressed(KeyCode::Down) {
        state.fps = (state.fps - 5.0).max(1.0);
    }
    if keyboard.just_pressed(KeyCode::R) {
        state.world = TerrariumWorld::demo(state.seed, state.use_gpu_substrate)
            .unwrap_or_else(|err| panic!("failed to reset terrarium: {err}"));
        state.snapshot = state.world.snapshot();
        state.frame_idx = 0;
        state.accumulator = 0.0;
        state.single_step = false;
    }
}

fn step_world(time: Res<Time>, mut state: NonSendMut<SimState>, mut exit: EventWriter<AppExit>) {
    if state.paused && !state.single_step {
        return;
    }

    let dt = 1.0 / state.fps.max(1.0);
    state.accumulator += time.delta_seconds();
    if state.single_step {
        state.accumulator = dt;
    }

    while state.accumulator >= dt {
        state
            .world
            .step_frame()
            .unwrap_or_else(|err| panic!("terrarium step failed: {err}"));
        state.snapshot = state.world.snapshot();
        state.frame_idx += 1;
        state.accumulator -= dt;

        if let Some(limit) = state.frame_limit {
            if state.frame_idx >= limit {
                exit.send(AppExit);
                break;
            }
        }

        if state.single_step {
            state.single_step = false;
            state.accumulator = 0.0;
            break;
        }
    }
}

fn update_gpu_inputs(
    state: NonSend<SimState>,
    targets: Res<RenderTargets>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<TerrariumMaterial>>,
) {
    if let Some(image) = images.get_mut(&targets.field_image) {
        image.data = terrarium_field_bytes(&state.world, state.view);
    }
    if let Some(material) = materials.get_mut(&targets.material) {
        material.uniform = terrarium_uniform(&state);
    }
}

fn main() {
    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            std::process::exit(1);
        }
    };

    let world = TerrariumWorld::demo(cli.seed, !cli.cpu_substrate)
        .unwrap_or_else(|err| panic!("failed to create terrarium world: {err}"));
    let snapshot = world.snapshot();
    let resolution = Vec2::new(world.width() as f32 * cli.cell_px, world.height() as f32 * cli.cell_px);

    App::new()
        .insert_resource(ClearColor(Color::rgb(0.04, 0.05, 0.06)))
        .insert_non_send_resource(SimState {
            world,
            snapshot,
            view: TerrariumTopdownView::Terrain,
            paused: false,
            single_step: false,
            fps: cli.fps,
            accumulator: 0.0,
            frame_limit: cli.frames,
            frame_idx: 0,
            seed: cli.seed,
            use_gpu_substrate: !cli.cpu_substrate,
            cell_px: cli.cell_px,
        })
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()).set(WindowPlugin {
            primary_window: Some(Window {
                title: "oNeuro Terrarium GPU".into(),
                resolution: resolution.into(),
                present_mode: PresentMode::AutoVsync,
                resizable: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugin(TerrariumMaterialPlugin)
        .add_startup_system(setup)
        .add_system(handle_input)
        .add_system(step_world)
        .add_system(update_gpu_inputs.after(step_world))
        .run();
}
