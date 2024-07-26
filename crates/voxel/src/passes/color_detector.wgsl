
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

var<workgroup> workgroup_subgroup_min_data: array<WorkgroupFaceData, 32>;
var<workgroup> workgroup_working_min: WorkgroupFaceData;

@compute @workgroup_size(1024)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
){
    let output_image_dimensions = textureDimensions(voxel_discovery_image).xy;

    let global_invocation_index = global_invocation_id.x;

    let global_workgroup_index = global_invocation_index / 1024; // should be 64!
    let local_workgroup_index = global_invocation_index % 1024;

    let subgroup_index_within_workgroup = local_workgroup_index / 32;

    let global_workgroup_id = vec2u(
        global_workgroup_index % (output_image_dimensions.x / 32),
        global_workgroup_index / (output_image_dimensions.x / 32));

    let sample_idx: vec2<u32> = global_workgroup_id * 32 + vec2u(local_workgroup_index % 32, local_workgroup_index / 32);
    
    let null_face_number = u32(4294967295);
    var thread_face_data = WorkgroupFaceData(null_face_number, null_face_number);

    if all(sample_idx < output_image_dimensions)
    {
        let thread_face_data_raw: vec2<u32> = textureLoad(voxel_discovery_image, sample_idx, 0).xy;
        thread_face_data = WorkgroupFaceData(thread_face_data_raw.y, thread_face_data_raw.x);
    }
    workgroupBarrier();

    // Each thread in our workgroup has a `thread_face_data` that's either null
    // or contains data that we want

    // The pseudocode looks as follows:
    // We have a 32x32 workgroup grid that's somehow divided into subgroup
    // arrays by the driver
    // Take the min of each subgroup, and then take the min of each of those mins
    // this gives us the minimum face_number in the entire workgroup in a few 
    // subgroup ops.
    // Write this global min to the global storage buffer, and then null all
    // threads that have this value

    for (var i = 0; i < 1024; i++)
    {
        let subgroup_min = subgroupMin(thread_face_data.face_number);

        if (thread_face_data.face_number == subgroup_min) // at least one thread
        {
            if (tempSubgroupElect(subgroup_invocation_id)) // pick the lowest
            {
                workgroup_subgroup_min_data[subgroup_index_within_workgroup] = thread_face_data;
            }
        }

        workgroupBarrier();

        // we now have a workgrpup buffer with each subgroup's min value.
        // use another subgroup op to find the minimum

        // we only want the first subgroup here
        let each_thread_subgroup_min = workgroup_subgroup_min_data[subgroup_invocation_id];
        
        let workgroup_min = subgroupMin(each_thread_subgroup_min.face_number);

        if (each_thread_subgroup_min.face_number == workgroup_min) // at least one thread
        {
            if (subgroup_index_within_workgroup == 0 && tempSubgroupElect(subgroup_invocation_id)) // pick the lowest
            {
                workgroup_working_min = each_thread_subgroup_min;
                
                try_global_dedup_of_face_number_unchecked(each_thread_subgroup_min);
            }
        }            
        
        workgroupBarrier();
        
        if (workgroup_working_min.face_number == null_face_number)
        {
            // the min of all threads was null there are no more threads
            break;
        }

        // now that we've done a global flush of the minimum value we may need to cleanup this
        // thread's data if it was flushed
        
        // Ok, great! this thread's data was flushed away so we can make
        // ourselves null now
        if (workgroup_working_min.face_number == thread_face_data.face_number)
        {
            thread_face_data.face_number = null_face_number;
        }

    }
}

fn tempSubgroupElect(subgroup_invocation_id: u32) -> bool
{
    return subgroupBroadcastFirst(subgroup_invocation_id) == subgroup_invocation_id;
}

fn try_global_dedup_of_face_number_unchecked(face_data: WorkgroupFaceData)
{
    let face_number = face_data.face_number;
    let pxx_data = face_data.other_packed_data;

    let chunk_id = pxx_data & u32(65535);
    let normal_id = (pxx_data >> 27) & u32(7);

    let face_voxel_pos = voxel_face_data_load(face_number);
    // let face_number_index = face_number / 32;
    // let face_number_bit = face_number % 32;

    // let mask = (1u << face_number_bit);
    // let prev = atomicOr(&is_face_number_visible_bool[face_number_index], mask);
    // let is_first_write = (prev & mask) == 0u; // this load is the slow bit

    let is_first_write = atomicExchange(&is_face_number_visible_bool[face_number], 1u) == 0u;

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