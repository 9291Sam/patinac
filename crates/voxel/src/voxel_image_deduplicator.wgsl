

const WORKGROUP_SET_SIZE: u32 = 32 * 32;
var<workgroup> workgroup_set: array<atomic<u32>, WORKGROUP_SET_SIZE>; // tune value. min required is 1024 max is 8192
const SetEmptySentinel: u32 = ~0u;

@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> indirect_rt_workgroups_buffer: array<atomic<u32>, 3>; // should be write only
// TODO: merge these lengths into their corresponding buffers
@group(0) @binding(2) var<storage, read> storage_set_len: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> storage_set: array<atomic<u32>>; 
@group(0) @binding(4) var<storage, read_write> unique_len: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> unique_voxel_buffer: array<atomic<u32>>; // should be write only


const WORKGROUP_X_SIZE: u32 = 32u;
const WORKGROUP_Y_SIZE: u32 = 32u;
@compute @workgroup_size(WORKGROUP_X_SIZE, WORKGROUP_Y_SIZE)
fn cs_main(
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
)
{
    let out_dims = textureDimensions(voxel_discovery_image).xy;

    if (all(global_invocation_id.xy < out_dims))
    {
        let recacl_global_invocation = global_invocation_id.x + out_dims.x * global_invocation_id.y;

        storage_set[recacl_global_invocation] = recacl_global_invocation;
        
    }
    else
    {
        // storage_set[global_invocation_index] = global_invocation_id.x * 100000 + global_invocation_id.y;
    }



}

fn u32pcgHash(in: u32) -> u32
{
    var state = in * 747796405u + 2891336453u;

    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    
    return (word >> 22u) ^ word;
}
