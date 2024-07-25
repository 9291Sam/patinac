
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

var<workgroup> workgroup_face_numbers: array<WorkgroupFaceData, 64>;
var<workgroup> workgroup_number_of_face_numbers: atomic<u32>;

@compute @workgroup_size(64)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
){
    let output_image_dimensions = textureDimensions(voxel_discovery_image).xy;

    let global_invocation_index = global_invocation_id.x;

    let global_workgroup_index = global_invocation_index / 64;
    let local_workgroup_index = global_invocation_index % 64;

    let global_workgroup_id = vec2u(
        global_workgroup_index % (output_image_dimensions.x / 8),
        global_workgroup_index / (output_image_dimensions.x / 8));

    let sample_idx: vec2<u32> = global_workgroup_id * 8 + vec2u(local_workgroup_index % 8, local_workgroup_index / 8);

    let this_px: vec2<u32> = textureLoad(voxel_discovery_image, sample_idx, 0).xy;

    let null_face_number = u32(4294967295);

    var maybe_face_number = null_face_number;

    if all(sample_idx < output_image_dimensions)
    {
        maybe_face_number = this_px.y;       
    }   

    loop
    {
        let min_face_number = subgroupMin(maybe_face_number);

        if (min_face_number == null_face_number)
        {
            break;
        }

        if (min_face_number == maybe_face_number)
        {
            if (tempSubgroupElect(subgroup_invocation_id))
            {
                let idx = atomicAdd(&workgroup_number_of_face_numbers, 1u);

                workgroup_face_numbers[idx] = WorkgroupFaceData(maybe_face_number, this_px.x);
            }

            maybe_face_number = null_face_number;
        }
    }

    workgroupBarrier();

    let number_of_mostly_unique_face_numbers = atomicLoad(&workgroup_number_of_face_numbers);

    for (var i = u32(0); i < number_of_mostly_unique_face_numbers; i++)
    {
        let face_number = workgroup_face_numbers[i].face_number;
        let pxx_data = workgroup_face_numbers[i].other_packed_data;

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
            
            if (this_face_id % 64 == 0)
            {
                atomicAdd(&color_raytracer_dispatches[0], 1u);
            }
        }   
    }
}

fn tempSubgroupElect(subgroup_invocation_id: u32) -> bool
{
    return subgroupBroadcastFirst(subgroup_invocation_id) == subgroup_invocation_id;
}