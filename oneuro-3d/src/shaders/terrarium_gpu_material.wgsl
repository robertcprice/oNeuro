#import bevy_sprite::mesh2d_types
#import bevy_sprite::mesh2d_view_bindings

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping
#endif

struct TerrariumMaterial {
    tone: vec4<f32>,
    bio: vec4<f32>,
};

@group(1) @binding(0)
var field_image: texture_2d<f32>;
@group(1) @binding(1)
var field_sampler: sampler;
@group(1) @binding(2)
var overlay_image: texture_2d<f32>;
@group(1) @binding(3)
var overlay_sampler: sampler;
@group(1) @binding(4)
var<uniform> material: TerrariumMaterial;

@group(2) @binding(0)
var<uniform> mesh: Mesh2d;

struct FragmentInput {
    #import bevy_sprite::mesh2d_vertex_output
};

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn terrain_palette(value: f32, daylight: f32) -> vec3<f32> {
    let soil = lerp3(vec3<f32>(0.11, 0.08, 0.05), vec3<f32>(0.52, 0.42, 0.24), value);
    let dry = lerp3(soil, vec3<f32>(0.78, 0.70, 0.50), value * 0.45);
    return lerp3(dry * 0.70, dry * 1.05, daylight);
}

fn moisture_palette(value: f32, daylight: f32) -> vec3<f32> {
    let damp = lerp3(vec3<f32>(0.10, 0.07, 0.05), vec3<f32>(0.11, 0.26, 0.30), value);
    let rich = lerp3(damp, vec3<f32>(0.32, 0.64, 0.68), value * value);
    return lerp3(rich * 0.72, rich * 1.03, daylight);
}

fn canopy_palette(value: f32, daylight: f32) -> vec3<f32> {
    let canopy = lerp3(vec3<f32>(0.03, 0.10, 0.05), vec3<f32>(0.28, 0.74, 0.24), value);
    let bloom = lerp3(canopy, vec3<f32>(0.62, 0.88, 0.35), value * 0.35);
    return lerp3(bloom * 0.68, bloom * 1.04, daylight);
}

fn chemistry_palette(value: f32, daylight: f32) -> vec3<f32> {
    let base = lerp3(vec3<f32>(0.04, 0.06, 0.09), vec3<f32>(0.34, 0.22, 0.08), value);
    let reactive = lerp3(base, vec3<f32>(0.92, 0.62, 0.18), value * value);
    return lerp3(reactive * 0.72, reactive * 1.02, daylight);
}

fn odor_palette(value: f32, daylight: f32) -> vec3<f32> {
    let base = lerp3(vec3<f32>(0.02, 0.02, 0.03), vec3<f32>(0.38, 0.10, 0.06), value);
    let plume = lerp3(base, vec3<f32>(0.95, 0.30, 0.12), sqrt(value));
    return lerp3(plume * 0.74, plume * 1.02, daylight);
}

fn palette(view_id: f32, value: f32, daylight: f32) -> vec3<f32> {
    if (view_id < 0.5) {
        return terrain_palette(value, daylight);
    }
    if (view_id < 1.5) {
        return moisture_palette(value, daylight);
    }
    if (view_id < 2.5) {
        return canopy_palette(value, daylight);
    }
    if (view_id < 3.5) {
        return chemistry_palette(value, daylight);
    }
    return odor_palette(value, daylight);
}

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let dims = max(vec2<f32>(textureDimensions(field_image)), vec2<f32>(1.0, 1.0));
    let texel = 1.0 / dims;

    let field = textureSample(field_image, field_sampler, uv).r;
    let north = textureSample(field_image, field_sampler, uv + vec2<f32>(0.0, -texel.y)).r;
    let south = textureSample(field_image, field_sampler, uv + vec2<f32>(0.0, texel.y)).r;
    let east = textureSample(field_image, field_sampler, uv + vec2<f32>(texel.x, 0.0)).r;
    let west = textureSample(field_image, field_sampler, uv + vec2<f32>(-texel.x, 0.0)).r;

    let edge = clamp((abs(east - west) + abs(north - south)) * 1.45, 0.0, 1.0);

    let overlay = textureSample(overlay_image, overlay_sampler, uv);

    let daylight = clamp(material.tone.x, 0.0, 1.0);
    let paused = material.tone.y;
    let view_id = clamp(material.tone.z, 0.0, 4.0);
    let time_phase = material.tone.w;

    let food = clamp(material.bio.x, 0.0, 1.0);
    let vitality = clamp(material.bio.y, 0.0, 1.0);
    let humidity = clamp(material.bio.z, 0.0, 1.0);
    let energy = clamp(material.bio.w, 0.0, 1.0);

    var color = palette(view_id, clamp(field, 0.0, 1.0), daylight);

    let daylight_tint = lerp3(
        vec3<f32>(0.18, 0.22, 0.34),
        vec3<f32>(1.02, 0.98, 0.92),
        daylight,
    );
    color *= daylight_tint;

    let contour_tint = lerp3(
        vec3<f32>(0.02, 0.03, 0.05),
        vec3<f32>(0.18, 0.12, 0.05),
        clamp(view_id / 4.0, 0.0, 1.0),
    );
    color += edge * (0.05 + 0.08 * vitality) * contour_tint;

    let water_mask = overlay.r;
    let plant_mask = overlay.g;
    let fruit_mask = overlay.b;
    let fly_mask = overlay.a;

    color = lerp3(color, vec3<f32>(0.20, 0.46, 0.88), water_mask * 0.72);
    color = lerp3(color, vec3<f32>(0.10, 0.78, 0.22), plant_mask * 0.78);
    color = lerp3(color, vec3<f32>(0.96, 0.54, 0.12), fruit_mask * 0.82);

    let grounded_fly = smoothstep(0.60, 0.80, fly_mask) - smoothstep(0.82, 0.95, fly_mask);
    let flying_fly = smoothstep(0.88, 0.99, fly_mask);
    color = lerp3(color, vec3<f32>(0.02, 0.02, 0.02), grounded_fly * 0.92);
    color = lerp3(color, vec3<f32>(0.96, 0.96, 0.88), flying_fly * 0.90);

    let vignette = 1.0 - smoothstep(0.35, 0.95, distance(uv, vec2<f32>(0.5, 0.5)) * 1.4);
    color *= 0.75 + 0.25 * vignette;

    let sweep = 0.5 + 0.5 * sin((uv.y + time_phase) * 180.0);
    let bio_glow = vec3<f32>(
        0.50 + 0.50 * food,
        0.45 + 0.55 * energy,
        0.55 + 0.45 * humidity,
    );
    color += edge * sweep * 0.035 * bio_glow;

    if (paused > 0.5) {
        let gray = luma(color);
        color = lerp3(color, vec3<f32>(gray, gray, gray), 0.40);
    }

    var output_color = vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
#ifdef TONEMAP_IN_SHADER
    output_color = tone_mapping(output_color);
#endif
    return output_color;
}
