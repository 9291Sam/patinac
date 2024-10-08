struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

struct MaterialData
{
    // special: u32,
    diffuse_color:    vec4<f32>,
    subsurface_color: vec4<f32>,
    specular_color:   vec4<f32>,

    diffuse_subsurface_weight: f32,
    specular:                  f32,
    roughness:                 f32,
    metallic:                  f32,

    emissive_color_and_power: vec4<f32>,
    coat_color_and_power:     vec4<f32>,
}

struct VoxelFaceData
{
    data: u32
}

// (face_number) -> chunk_local_position
fn voxel_face_data_load(face_number: u32) -> vec3<u32>
{
    let eight_bit_mask: u32 = u32(255);

    let voxel_data = face_data_buffer[face_number].data;

    let x_pos: u32 = voxel_data & eight_bit_mask;
    let y_pos: u32 = (voxel_data >> 8) & eight_bit_mask;
    let z_pos: u32 = (voxel_data >> 16) & eight_bit_mask;

    return vec3<u32>(x_pos, y_pos, z_pos);

}

struct BrickMap
{
    map: array<array<array<u32, 32>, 32>, 32>
}

// (chunk_id, brick_coordinate) -> maybe_brick_ptr
fn brick_map_load(chunk_id: u32, pos: vec3<u32>) -> u32
{
    return brick_map[chunk_id].map[pos.x][pos.y][pos.z];
}

struct MateralBrick
{
    voxels: array<array<array<u32, 4>, 8>, 8>,
}

// (brick_ptr, brick_local_pos) -> material_id
fn material_bricks_load(brick_ptr: u32, pos: vec3<u32>) -> u32
{
    let val = material_bricks[brick_ptr].voxels[pos.x][pos.y][pos.z / 2];

    if (pos.z % 2 == 0) // low u16
    {
        return val & u32(65535);
    }
    else // high u16
    {
        return val >> 16;
    }
}

struct VisibilityBrick
{
    data: array<u32, 16>
}

// (brick_ptr, brick_local_position) -> is_voxel_occupied
fn visiblity_brick_load(brick_ptr: u32, pos: vec3<u32>) -> bool
{
    //// !NOTE: the order is like this so that the cache lines are aligned
    //// ! vertically
    // i.e the bottom half is one cache line and the top is another
    let linear_index = pos.x + pos.z * 8 + pos.y * 8 * 8;

    // 32 is the number of bits in a u32
    let index = linear_index / 32;
    let bit = linear_index % 32;

    return (visiblity_bricks[brick_ptr].data[index] & (1u << bit)) != 0;
}

struct RenderedFaceInfo
{
    chunk_id: u32,
    combined_dir_and_pos: u32,
    color: vec4<f32>
}

struct PointLight
{
    position: vec4<f32>,
    color_and_power: vec4<f32>,
    falloffs: vec4<f32>
}

struct CalculatedLightPower
{
    diffuse_strength: vec3<f32>,
    specular_strength: vec3<f32>,
}

fn calculate_light_power(
    camera_position: vec3<f32>,
    voxel_position: vec3<f32>,
    voxel_normal: vec3<f32>,
    light: PointLight,
    specular_power: f32,
) -> CalculatedLightPower
{
    var to_light_vector = light.position.xyz - voxel_position;
    let to_light_distance = length(to_light_vector);
    to_light_vector /= to_light_distance;

    let view_vector = normalize(camera_position - voxel_position);
    let reflect_vector = 2.0 * dot(to_light_vector, voxel_normal) * voxel_normal - to_light_vector;

    let attenuation = 1.0 / (
        light.falloffs.x +
        light.falloffs.y * to_light_distance +
        light.falloffs.z * to_light_distance * to_light_distance +
        light.falloffs.w * to_light_distance * to_light_distance * to_light_distance);

    let diffuse_strength: vec3<f32> = dot(voxel_normal, to_light_vector) * light.color_and_power.xyz * light.color_and_power.w * attenuation;
    let specular_strength: vec3<f32> = pow(saturate(dot(reflect_vector, view_vector)), specular_power) * light.color_and_power.xyz * light.color_and_power.w * attenuation;

    return CalculatedLightPower(diffuse_strength, specular_strength);
}

fn calculate_single_face_color(
    camera_pos: vec3<f32>,
    pos: vec3<f32>,
    normal: vec3<f32>,
    material_id: u32,
    light: PointLight
) -> vec3<f32>
{    
    let color_and_power = light.color_and_power;
    let light_pos = light.position.xyz; 

    var light_dir = light_pos - pos;
    let distance = length(light_dir);
    light_dir /= distance;

    // intensity of diffuse
    let view_vector = normalize(camera_pos - pos);
    let r = 2 * dot(light_dir, normal) * normal - light_dir;

    let attenuation = 1.0 / (
        light.falloffs.x +
        light.falloffs.y * distance +
        light.falloffs.z * distance * distance +
        light.falloffs.w * distance * distance * distance);
 
    let diffuse = saturate(dot(normal, light_dir)) * color_and_power.xyz * material_buffer[material_id].diffuse_color.xyz * color_and_power.w * attenuation;
    let specular = pow(saturate(dot(r, view_vector)), material_buffer[material_id].specular) * color_and_power.xyz * material_buffer[material_id].specular_color.xyz * color_and_power.w * attenuation;

    let result: vec3<f32> = saturate(diffuse + specular);

    return result;
}

fn map(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

fn rand(i: u32) -> u32
{
    let index = (i << 13) ^ i;

    return (index * (index * index * 15731 + 789221) + 1376312589);
}
