@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;

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
    let dims = textureDimensions(voxel_discovery_image);

    let u: u32 = u32(round(map(in.x, -1.0, 1.0, 0.0, f32(dims.x))));
    let v: u32 = u32(round(map(in.y, 1.0, -1.0, 0.0, f32(dims.y))));

    let voxel_data: vec2<u32> = textureLoad(voxel_discovery_image, vec2<u32>(u32(in.x), u32(in.y)), 0).xy;

    if (all(voxel_data == vec2<u32>(0)))
    {
        discard;
    }
    
    let nine_bit_mask: u32 = u32(511);

    let x_pos: u32 = voxel_data[0] & nine_bit_mask;
    let y_pos: u32 = (voxel_data[0] >> 9) & nine_bit_mask;
    let z_pos: u32 = (voxel_data[0] >> 18) & nine_bit_mask;


    return vec4<f32>(
        map(f32(x_pos), 0.0, 511.0, 0.0, 1.0),
        map(f32(y_pos), 0.0, 511.0, 0.0, 1.0),
        map(f32(z_pos), 0.0, 511.0, 0.0, 1.0),
        1.0);

}

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}