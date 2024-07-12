@group(0) @binding(0) var<storage, read> face_data_buffer: array<VoxelFaceData>; 
@group(0) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(0) @binding(2) var<storage, read> material_bricks: array<MateralBrick>; 
@group(0) @binding(3) var<storage, read> visiblity_bricks: array<VisibilityBrick>;
@group(0) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(0) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> is_face_number_visible_bits: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> face_numbers_to_face_ids: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> next_face_id: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> renderered_face_info: array<RenderedFaceInfo>;

@group(1) @binding(0) var<storage, read_write> color_raytracer_dispatches: array<atomic<u32>, 3>;


@compute @workgroup_size(1024)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
){ 
    let global_invocation_index = global_invocation_id.x;

    let idx = global_invocation_index / 32;
    let bit = global_invocation_index % 32;

    if (bit == 0)
    {
        is_face_number_visible_bits[idx] = 0u;
    }

    if (global_invocation_index == 0)
    {
        color_raytracer_dispatches[0] = 0u;
    }
}


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
    pos_and_dir: u32,
    packed_color: u32, // pack4x8unorm
}
