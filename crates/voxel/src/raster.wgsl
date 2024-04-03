struct VertexInput {
    @location(0) voxel_data: vec2<u32>,
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

    let voxel_data: vec2<u32> = input.voxel_data;

    let nine_bit_mask: u32 = u32(511);
    let three_bit_mask: u32 = u32(7);
    let two_bit_mask: u32 = u32(3);
    let five_bit_mask: u32 = u32(31);
    let four_bit_mask: u32 = u32(15);

    let x_pos: u32 = voxel_data[0] & nine_bit_mask;
    let y_pos: u32 = (voxel_data[0] >> 9) & nine_bit_mask;
    let z_pos: u32 = (voxel_data[0] >> 18) & nine_bit_mask;

    let l_width_lo: u32 = (voxel_data[0] >> 27) & five_bit_mask;
    let l_width_hi: u32 = (voxel_data[1] & four_bit_mask) << 5;

    let l_width: u32 = l_width_lo | l_width_hi;
    let w_width: u32 = (voxel_data[1] >> 4) & nine_bit_mask;
    let face_id: u32 = (voxel_data[1] >> 13) & three_bit_mask;
    let voxel_id: u32 = (voxel_data[1] >> 16);

    out.clip_position = global_model_view_projection[id] * 
        vec4<f32>(f32(x_pos), f32(y_pos), f32(z_pos), 1.0);
    out.voxel = u32(voxel_id);
  
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