@group(1) @binding(0) var<storage, read> face_data_buffer: array<VoxelFaceData>; 
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>; 
@group(1) @binding(3) var<storage, read> visiblity_bricks: array<VisibilityBrick>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> is_face_number_visible_bits: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> face_numbers_to_face_ids: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> next_face_id: u32;
@group(1) @binding(9) var<storage, read_write> renderered_face_info: array<RenderedFaceInfo>;

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    view_projection: mat4x4<f32>
}

@compute @workgroup_size(1024)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
){ 
    let global_invocation_index = global_invocation_id.x;
    
    if (global_invocation_index < next_face_id)
    {
        let chunk_id = renderered_face_info[global_invocation_index].chunk_id;
        let combined_dir_and_pos = renderered_face_info[global_invocation_index].combined_dir_and_pos;

        let eight_bit_mask: u32 = u32(255);

        let x_pos: u32 = combined_dir_and_pos & eight_bit_mask;
        let y_pos: u32 = (combined_dir_and_pos >> 8) & eight_bit_mask;
        let z_pos: u32 = (combined_dir_and_pos >> 16) & eight_bit_mask;
        let dir:   u32 = (combined_dir_and_pos >> 24) & u32(7);

        let face_voxel_pos = vec3<u32>(x_pos, y_pos, z_pos);

        var normal: vec3<f32>;

        switch (dir)
        {
            case 0u: {normal = vec3<f32>(0.0, 1.0, 0.0); }
            case 1u: {normal = vec3<f32>(0.0, -1.0, 0.0); }     
            case 2u: {normal = vec3<f32>(-1.0, 0.0, 0.0); }       
            case 3u: {normal = vec3<f32>(1.0, 0.0, 0.0); }       
            case 4u: {normal = vec3<f32>(0.0, 0.0, -1.0); }      
            case 5u: {normal = vec3<f32>(0.0, 0.0, 1.0); }
            case default: {normal = vec3<f32>(0.0); }
        }

        let global_face_voxel_position = gpu_chunk_data[chunk_id].xyz + vec3<f32>(face_voxel_pos);
        
        let brick_coordinate = face_voxel_pos / 8u;
        let brick_local_coordinate = face_voxel_pos % 8u;

        let brick_ptr = brick_map_load(chunk_id, brick_coordinate);
        let voxel = material_bricks_load(brick_ptr, brick_local_coordinate);

        var positions = array(
            vec4<f32>(120.0, 15.0, -40.0, 0.0),
            vec4<f32>(80.0, 25.0, -140.0, 0.0),
            vec4<f32>(190.0, 45.0, -10.0, 0.0),
            vec4<f32>(-20.0, 18.0, 30.0, 0.0),
        );

        var res = vec4<f32>(0.0);

        for (var i: i32 = 0; i < 4; i++)
        {
            res += vec4<f32>(calculate_single_face_color(
                global_face_voxel_position,
                normal,
                voxel,
                PointLight(
                    positions[i],
                    vec4<f32>(1.0, 1.0, 1.0, 32.0),
                )
            ), 1.0);
        }

        
        let ambient_strength = 0.005;
        let ambient_color = vec3<f32>(1.0);

        let ambient = ambient_strength * ambient_color;

        res += vec4<f32>(ambient, 1.0);

        // TODO: do a 10bit alpha ignoring packing?
        renderered_face_info[global_invocation_index].color = saturate(res);
    }
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
}

fn calculate_single_face_color(
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
    let view_vector = normalize(global_info.camera_pos.xyz - pos);
    let r = 2 * dot(light_dir, normal) * normal - light_dir;

    let constant_factor = 0.0;
    let linear_factor = 0.75;
    let quadratic_factor = 0.03;
    let attenuation = 1.0 / (constant_factor + linear_factor * distance + quadratic_factor * distance * distance);
 
    let diffuse = saturate(dot(normal, light_dir)) * color_and_power.xyz * material_buffer[material_id].diffuse_color.xyz * color_and_power.w * attenuation;
    let specular = pow(saturate(dot(r, view_vector)), material_buffer[material_id].specular) * color_and_power.xyz * material_buffer[material_id].specular_color.xyz * color_and_power.w * attenuation;

    let result: vec3<f32> = saturate(diffuse + specular);

    return result;
}