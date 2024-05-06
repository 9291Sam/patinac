

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


/// This shader is dispatched in 32x32 workgroups over a screen sized image
/// storage_set and unique_voxel_buffer are both preallocated to be of the same 
/// number of elements as the screen sized image.
/// storage_set_len is this length.
@compute @workgroup_size(32, 32)
fn cs_main(
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
)
{
    let workgroup_index =  
        workgroup_id.x +
        workgroup_id.y * num_workgroups.x +
        workgroup_id.z * num_workgroups.x * num_workgroups.y;
 
  
    let global_invocation_index =
        workgroup_index * 32 * 32 +
        local_invocation_index;


    // Fill set with the null sentienl
    storage_set[global_invocation_index] = SetEmptySentinel;

    storageBarrier();

    // Inserting our workgroup's pixel data into a workgroup local set
    {
        // Load our invocation's voxel data @ pixel
        var maybe_local_data: vec2<u32> = textureLoad(voxel_discovery_image, global_invocation_id.xy, 0).xy;

        // Ok, we may have just loaded garbage data if we did, let's pretend
        // it doesn't exist
        if all(global_invocation_id.xy < textureDimensions(voxel_discovery_image))
        {
            // load bottom 27 bits (location);
            let search_store_data = maybe_local_data.x & 268435455u; 

            StorageSet_insert(search_store_data);
        }
    }   

    // Ensure all workgroup local writes are visible
    storageBarrier(); 

    let maybe_storage_set_voxel = storage_set[global_invocation_index];

    if (storage_set[global_invocation_index] != SetEmptySentinel)
    {
        atomicAdd(&unique_len, 1u);
    }



    // // Now we have a global unique voxel set and we need to smash it down into
    // // an array
    // {
    //     // every invocation of this shader loads
    //     let maybe_storage_set_voxel = storage_set[global_invocation_index];

    //     if (maybe_storage_set_voxel != SetEmptySentinel)
    //     {
    //         // get the index of a guaranteed free value that's adjacent to 
    //         // all other elements
    //         let free_idx = atomicAdd(&unique_len, 1u);

    //         // write our data;
    //         unique_voxel_buffer[free_idx] = maybe_storage_set_voxel;

    //         // Ok, this is a little weird.
    //         // The shader stage that follows this one needs to be significantly smaller
    //         // and dispatched dynamically, we can use an indirect buffer for this
    //         // we need to dispatch workgroups of size 1024 x 1
    //         // unique_len.div_ceil(1024) times, this is a way of calculating this
    //         // efficiently
    //         if (free_idx % 1024 == 0)
    //         {
    //             atomicAdd(&indirect_rt_workgroups_buffer[0], 1u);
    //         }
    //     }
    // }
}

// insert a key, returns the index the element resides at
fn WorkgroupSet_insert(in_key: u32) -> u32
{
    if (in_key == 0u)
    {
        return 0u;
    }

    var key = in_key;
    var slot = u32hash(key) % WORKGROUP_SET_SIZE;

    for (var i = 0; i < 32; i += 1)
    {
        let current_value = atomicLoad(&workgroup_set[slot]);

        if (current_value == key)
        {
            return slot;
        }

        if (current_value == SetEmptySentinel)
        {
            // do exchange

            let hopefully_empty = atomicExchange(&workgroup_set[slot], key);

            workgroupBarrier();

            if (hopefully_empty == SetEmptySentinel)
            {
                // Excellent. nothing else touched this in the meantime and we 
                // have now put our elemnt in.
                return slot;
            }
            else
            {
                // Rats... Someone else wrote to the slot in the time it took 
                // for us to see if it was free
                // go forward another slot and try again
                // and worse, we stole their value which we now need to re-insert!
                key = hopefully_empty;

                continue;
            }
        }

        slot = u32hash(slot) % WORKGROUP_SET_SIZE;

        workgroupBarrier();
        
    }

    // impossible to happen, but we need it anyway
    return 0u;
}

fn StorageSet_insert(in_key: u32) -> u32
{
    if (in_key == 0u)
    {
        return 0u;
    }
    
    var key = in_key;
    var slot = u32hash(key) % storage_set_len;


    for (var i = 0; i < 32; i += 1)   
    {
        
        storageBarrier();
        let current_value = atomicLoad(&storage_set[slot]);
        storageBarrier();

        if (current_value == key)
        {
            return slot;
        }

        if (current_value == SetEmptySentinel)
        {
            // do exchange

            storageBarrier();
            let hopefully_empty = atomicExchange(&storage_set[slot], key);

            storageBarrier();

            if (hopefully_empty == SetEmptySentinel)
            {
                // Excellent. nothing else touched this in the meantime and we 
                // have now put our elemnt in.
                return slot;
            }
            else
            {
                // Rats... Someone else wrote to the slot in the time it took 
                // for us to see if it was free
                // go forward another slot and try again
                // and worse, we stole their value which we now need to re-insert!
                key = hopefully_empty;

                continue;
            }
        }

        slot = u32hash(slot) % WORKGROUP_SET_SIZE;

        storageBarrier();
        
    }

    // impossible to happen, but we need it anyway
    return 0u;
}

fn u32hash(in_x: u32) -> u32
{
    var x = in_x;

    x ^= x >> 17u;
    x *= 0xed5ad4bbu;
    x ^= x >> 11u;
    x *= 0xac4c1b51u;
    x ^= x >> 15u;
    x *= 0x31848babu;
    x ^= x >> 14u;

    return x;
}