struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) voxel_offset: vec4<i32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) voxel: u32,
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
    out.voxel = u32(input.voxel_offset.w);
  
    return out;
}

struct FragmentOutput
{
   @location(0) color: vec4<f32>
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput
{
    return FragmentOutput(get_voxel_color(in.voxel));
}

const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

fn get_voxel_color(voxel: u32) -> vec4<f32>
{
    switch voxel
    {
        case 0u: {return ERROR_COLOR;}                     // Error color for Air
        case 1u: {return vec4<f32>(0.5, 0.5, 0.5, 1.0);}   // Grey for Rock0
        case 2u: {return vec4<f32>(0.6, 0.6, 0.6, 1.0);}   // Light Grey for Rock1
        case 3u: {return vec4<f32>(0.7, 0.7, 0.7, 1.0);}   // Lighter Grey for Rock2
        case 4u: {return vec4<f32>(0.8, 0.8, 0.8, 1.0);}   // Even Lighter Grey for Rock3
        case 5u: {return vec4<f32>(0.9, 0.9, 0.9, 1.0);}   // Almost White for Rock4
        case 6u: {return vec4<f32>(1.0, 1.0, 1.0, 1.0);}   // White for Rock5
        case 7u: {return vec4<f32>(0.0, 0.5, 0.0, 1.0);}   // Dark Green for Grass0
        case 8u: {return vec4<f32>(0.0, 0.6, 0.0, 1.0);}   // Slightly Lighter Green for Grass1
        case 9u: {return vec4<f32>(0.0, 0.7, 0.0, 1.0);}   // Light Green for Grass2
        case 10u: {return vec4<f32>(0.0, 0.8, 0.0, 1.0);}  // Even Lighter Green for Grass3
        case 11u: {return vec4<f32>(0.0, 0.9, 0.0, 1.0);}  // Very Light Green for Grass4
        case 12u: {return vec4<f32>(0.2, 0.5, 0.2, 1.0);}  // Grey for Grass5
        default: {return ERROR_COLOR;}                      // Error color for unknown voxels
    }
}