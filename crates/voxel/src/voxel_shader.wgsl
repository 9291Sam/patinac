

struct VertexInput
{
    @location(0) position: vec3<f32>
}

struct VertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       position:      vec3<f32>
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

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;

var<push_constant> push_constant_id: u32;

@vertex fn vs_main(in: VertexInput) -> VertexOutput
{
    var out: VertexOutput;

    out.clip_position =
        global_model_view_projection.data[push_constant_id]
        * vec4<f32>(in.position, 1.0);

    out.position = in.position;

    return out;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>
{
    return vec4<f32>(0.4, 0.21, 0.63, 1.0);
}