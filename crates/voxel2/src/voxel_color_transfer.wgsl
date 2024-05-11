
// Texture<FromFragmentShader>
@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
// [number_of_image_workgroups, 1, 1];
@group(0) @binding(1) var<storage, read_write> indirect_color_calc_buffer: array<atomic<u32>, 3>;
// [FaceIdInfo, ...]
@group(0) @binding(2) var<storage, read_write> face_id_buffer: array<FaceInfo>;
// number_of_voxels_in_buffer
@group(0) @binding(3) var<storage, read_write> number_of_unique_voxels: atomic<u32>;
// [FaceId, ...]
@group(0) @binding(4) var<storage, read_write> unique_voxel_buffer: array<atomic<u32>>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32>
{
    switch (index)
    {
        case 0u:      {return vec4<f32>(-1.0, 3.0, 0.5, 1.0); }
        case 1u:      {return vec4<f32>(3.0, -1.0, 0.5, 1.0); }
        case 2u:      {return vec4<f32>(-1.0, -1.0, 0.5, 1.0); }
        case default: {return vec4<f32>(0.0); }
    }
}

@fragment
fn fs_main(@builtin(position) in: vec4<f32>) -> @location(0) vec4<f32>
{
    let voxel_data: vec2<u32> = textureLoad(voxel_discovery_image, vec2<u32>(u32(in.x), u32(in.y)), 0).xy;

    if (all(voxel_data == vec2<u32>(0)))
    {
        discard;
    }
    
    let nine_bit_mask: u32 = u32(511);

    let x_pos: u32 = voxel_data[0] & nine_bit_mask;
    let y_pos: u32 = (voxel_data[0] >> 9) & nine_bit_mask;
    let z_pos: u32 = (voxel_data[0] >> 18) & nine_bit_mask;

    let face_id = voxel_data.y;

    face_id_buffer[face_id].low = 0u;


    
    return vec4<f32>(
        map(f32(x_pos), 0.0, 511.0, 0.0, 1.0),
        map(f32(y_pos), 0.0, 511.0, 0.0, 1.0),
        map(f32(z_pos), 0.0, 511.0, 0.0, 1.0),
        1.0
    ); 

}

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

const ERROR_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

fn get_voxel_color(voxel: u32) -> vec4<f32>
{
    switch voxel
    {
        case 0u:  {return ERROR_COLOR;}
        case 1u:  {return vec4<f32>(0.5, 0.5, 0.5, 1.0);}
        case 2u:  {return vec4<f32>(0.6, 0.6, 0.6, 1.0);}
        case 3u:  {return vec4<f32>(0.7, 0.7, 0.7, 1.0);}
        case 4u:  {return vec4<f32>(0.8, 0.8, 0.8, 1.0);}
        case 5u:  {return vec4<f32>(0.9, 0.9, 0.9, 1.0);}
        case 6u:  {return vec4<f32>(1.0, 1.0, 1.0, 1.0);}
        case 7u:  {return vec4<f32>(0.0, 0.5, 0.0, 1.0);}
        case 8u:  {return vec4<f32>(0.0, 0.6, 0.0, 1.0);}
        case 9u:  {return vec4<f32>(0.0, 0.7, 0.0, 1.0);}
        case 10u: {return vec4<f32>(0.0, 0.8, 0.0, 1.0);}
        case 11u: {return vec4<f32>(0.0, 0.9, 0.0, 1.0);}
        case 12u: {return vec4<f32>(0.2, 0.5, 0.2, 1.0);}
        default:  {return ERROR_COLOR;}
    }
}

fn rand(i: u32) -> u32
{
    let index = (i << 13) ^ i;

    return (index * (index * index * 15731 + 789221) + 1376312589);
}


struct FaceInfo
{
    ///   x [0]    y [1]
    /// [0,  15] [      ] | chunk_id
    /// [16, 31] [      ] | material
    /// [      ] [0,   0] | is_visible
    /// [      ] [1,  31] | unused
    low: atomic<u32>,
    high: atomic<u32>,
}
