

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

@compute @workgroup_size(32, 32)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>)
{
    let out_dims = textureDimensions(voxel_discovery_image).xy;

    // Discard edges outside of the image's bound
    if (all(global_invocation_id.xy >= out_dims))
    {
        return;
    }
    

    let global_index = global_invocation_id.x + out_dims.x * global_invocation_id.y;

    storage_set[global_index] = SetEmptySentinel;

    storageBarrier();

    let image_px: vec2<u32> = textureLoad(voxel_discovery_image, global_invocation_id.xy, 0).xy;
    let image_location = image_px.x & 268435455u;

    var data_to_insert = image_location;
    var slot = u32pcgHash(image_location) % storage_set_len;

    loop
    {   
        if (data_to_insert == 0)
        {
            break;
        }

        // optimistic check
        var slot_data = atomicLoad(&storage_set[slot]);

        if (slot_data == data_to_insert)
        {
            // Neat, it's already there!
            break;
        }

        if (slot_data == SetEmptySentinel)
        {
            // awesome we probablly have unique access to this slot
            slot_data = atomicExchange(&storage_set[slot], data_to_insert);
        
            if (slot_data == SetEmptySentinel || slot_data == data_to_insert)
            {
                // Awesome we have the data we want in the slot
                break;
            }
            else
            {
                // shit, we stone another thread's data, we need to try again
                data_to_insert = slot_data;
            }
        }
        else
        {
            // welp, we know the slot is filled with some other thread's data
            // try again
        }
        
        slot = (slot + 1) % storage_set_len;
    }

        
    
}

fn u32pcgHash(in: u32) -> u32
{
    var state = in * 747796405u + 2891336453u;

    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    
    return (word >> 22u) ^ word;
}
