struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) voxel_offset: vec4<i32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>
}

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

alias Matricies = array<mat4x4<f32>, 1024>;

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;

var<push_constant> id: u32;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput
{
    var out: VertexOutput;

    let model = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(vec3<f32>(input.voxel_offset.xyz), 1.0)
    );

    out.clip_position = global_info.view_projection * model * vec4<f32>(input.position, 1.0);
    out.color = input.position;
  
    return out;
}

struct FragmentOutput
{
   @location(0) color: vec4<f32>
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput
{
    return FragmentOutput(vec4<f32>(in.color, 1.0));
}