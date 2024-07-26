
@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;

@group(1) @binding(0) var<storage, read> face_data_buffer: array<VoxelFaceData>; 
@group(1) @binding(1) var<storage, read> brick_map: array<BrickMap>;
@group(1) @binding(2) var<storage, read> material_bricks: array<MateralBrick>; 
@group(1) @binding(3) var<storage, read> visiblity_bricks: array<VisibilityBrick>;
@group(1) @binding(4) var<storage, read> material_buffer: array<MaterialData>;
@group(1) @binding(5) var<storage, read> gpu_chunk_data: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> is_face_number_visible_bool: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> face_numbers_to_face_ids: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> next_face_id: atomic<u32>;
@group(1) @binding(9) var<storage, read_write> renderered_face_info: array<RenderedFaceInfo>;

@group(2) @binding(0) var<storage, read_write> color_raytracer_dispatches: array<atomic<u32>, 3>;

struct WorkgroupFaceData
{
    face_number: u32,
    other_packed_data: u32,
}

// var<workgroup> workgroup_subgroup_min_data: array<WorkgroupFaceData, 32>;

@compute @workgroup_size(32, 32)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
){
    let output_image_dimensions = textureDimensions(voxel_discovery_image).xy;

    if all(global_invocation_id.xy < output_image_dimensions)
    {
        let thread_face_data_raw: vec2<u32> = textureLoad(voxel_discovery_image, global_invocation_id.xy, 0).xy;
        
        try_global_dedup_of_face_number_unchecked(WorkgroupFaceData(thread_face_data_raw.y, thread_face_data_raw.x));
    }

}

fn try_global_dedup_of_face_number_unchecked(face_data: WorkgroupFaceData)
{
    let face_number = face_data.face_number;
    let pxx_data = face_data.other_packed_data;

    let chunk_id = pxx_data & u32(65535);
    let normal_id = (pxx_data >> 27) & u32(7);

    let face_voxel_pos = voxel_face_data_load(face_number);
    let face_number_index = face_number / 32;
    let face_number_bit = face_number % 32;

    let mask = (1u << face_number_bit);
    let prev = atomicOr(&is_face_number_visible_bool[face_number_index], mask);
    let is_first_write = (prev & mask) == 0u;

    if (is_first_write)
    {
        let this_face_id = atomicAdd(&next_face_id, 1u);

        face_numbers_to_face_ids[face_number] = this_face_id;
        let combined_dir_and_pos = face_voxel_pos.x | (face_voxel_pos.y << 8) | (face_voxel_pos.z << 16) | (normal_id << 24);
        renderered_face_info[this_face_id] = RenderedFaceInfo(chunk_id, combined_dir_and_pos, vec4<f32>(0.0));
        
        if (this_face_id % 256 == 0)
        {
            atomicAdd(&color_raytracer_dispatches[0], 1u);
        }
    }   
}