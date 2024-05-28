


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

struct Matricies
{
    data: array<mat4x4<f32>, 1024>
}

struct PushConstants
{
    id: u32,
    edge_dim: u32,
    time_alive: f32,
}

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;

@group(1) @binding(0) var t_diffuse: texture_2d<f32>;
@group(1) @binding(1) var s_diffuse: sampler;

var<push_constant> push_constants: PushConstants;

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;

    let angle = push_constants.time_alive;
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);

    let rotation_matrix = mat3x3<f32>(
        vec3<f32>(cos_angle, 0.0, -sin_angle),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(sin_angle, 0.0, cos_angle)
    );

    let model_space_pos = rotation_matrix * vec3<f32>(model.position) + vec3<f32>(f32(instance_index / push_constants.edge_dim), 0.0, f32(instance_index % push_constants.edge_dim));

    out.clip_position = global_model_view_projection.data[push_constants.id] * vec4<f32>(model_space_pos, 1.0);
    
    return out;
}
 

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
 