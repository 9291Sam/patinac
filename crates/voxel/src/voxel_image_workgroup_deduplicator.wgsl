

const SET_SIZE: u32 = 32 * 32;
var<workgroup> workgroup_set: array<atomic<u32>, SET_SIZE>; // tune value. min required is 1024 max is 8192
const SetEmptySentinel: u32 = ~0u;

@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
@group(0) @binding(1) var indirect_rt_workgroups_buffer: array<atomic<u32>, 3>;
@group(0) @binding(2) var<storage> mostly_unique_len: atomic<u32>;
@group(0) @binding(3) var<storage> mostly_unique_voxel_buffer: array<atomic<u32>>; 
@group(0) @binding(4) var<storage> unique_len: atomic<u32>;
@group(0) @binding(5) var<storage> unique_voxel_buffer: array<atomic<u32>>;


@compute @workgroup_size(32, 32)
fn cs_main()
{
    // Fill set with ~0
    workgroup_set[local_invacation_index] = SetEmptySentinel;

    // Load our invocation's voxel data @ pixel
    // TODO: deal with out of bounds data
    var maybe_local_data: vec2<u32> = textureLoad(voxel_discovery_image, global_invocation_id.xy).xy;

    // Ok, we may have just loaded garbage data
    if any(global_invocation_id > textureDimensions(voxel_discovery_image))
    {
        maybe_local_data = vec2<u32>(0);
    }

    // If a voxel actually exists at the given pixel, store it's data into the set
    if all(maybe_local_data != vec2<u32>(0u))
    {
        let search_store_data = local_data.x & 268435455u; // load bottom 27 bits (location);

        Set_insert(&map, search_store_data);
    }

    // wait for all invocations of this workgroup to get ot here
    workgroupBarrier(); 

    // Use the fact that since our workgroup size is 32x32, this means that
    // we can do the work of another shader here.
    // Now we have a workgroup buffer of some very sparse data
    // for example
    // [null, null, 4, 84, 89806, null, 845, null, null 944, 45]
    // and we want it to look like (size [arr])
    // 6 [4, 84, 89806, 845, 944, 45]
    // We can shift our focus from a square around the image into a line in the 
    // buffer
    // so now we think of the 32x32 shader as a 1024x1 one
    // this means that for each element of this array, we test if its null,
    // and if it isnt we store its data into an output array, using a relaxed
    // atomic to always generate unique but adjacent indicies

    // take the data
    let maybe_data_to_put_in_large = workgroup_set[local_invacation_index];

    if (maybe_data_to_put_in_large != SetEmptySentinel)
    {
        // Ok, so we got one of the rare instances where we need to put the data
        // into the output buffer

        // get a new index
        let free_idx = atomicAdd(&mostly_unique_len, 1);

        // write our data into the buffer
        mostly_unique_voxel_buffer[free_idx] = maybe_data_to_put_in_large;

        // Ok, this is a little weird.
        // The shader stage that follows this one needs to be significantly smaller
        // and dispatched dynamically, we can use an indirect buffer for this
        if (free_idx % 1024 == 0)
        {
            atomicAdd(&indirect_buffer[0], 1)
        }
    }

}

// insert a key, returns the index the element resides at
fn Set_insert(key: u32) -> u32
{
    var slot = u32hash(key) % SET_SIZE;

    loop
    {
        let res: __atomic_compare_exchange_result<u32> =
            atomicCompareExchangeWeak(&workgroup_map[slot], SetEmptySentinel, key);

        if (res.exchanged)
        {
            // unique access, we now are the first one to get here
            return slot;
        }

        if (res.old_value == key)
        {
            // already there
            return slot;
        }

        if (res.old_value == 0)
        {
            // spurious failure
            continue;
        }
    
        if (res.old_value != key)
        {
            // there's another element there, incremenet the slot and try again
            slot = (slot + 1) % SET_SIZE;
        }
    }
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