@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
@group(0) @binding(1) var indirect_rt_workgroups_buffer: array<atomic<u32>, 3>;
@group(0) @binding(2) var<storage> mostly_unique_len: atomic<u32>;
@group(0) @binding(3) var<storage> mostly_unique_voxel_buffer: array<atomic<u32>>; 

@group(0) @binding(4) var<storage> unique_len: atomic<u32>;
@group(0) @binding(5) var<storage> unique_voxel_buffer: array<atomic<u32>>;

// hardcoded buffer of screen size 
@group(0) @binding(6) var<uniform> unique_voxel_set_nondense_xor_len: u32;
@group(0) @binding(7) var<storage> unique_voxel_set_nondense_xor: array<atomic<u32>>;

const SetEmptySentinel: u32 = ~0u;


@compute @workgroup_size(1024)
fn cs_main()
{
    if (global_invocation_index < mostly_unique_len)
    {
        SetNondense_insert(mostly_unique_voxel_buffer[global_invocation_index]);
    }

    storageBarrier();

    let number_of_elements_each_invocation_must_traverse = 
        div_ceil(unique_voxel_set_nondense_xor_len, 1024);

    let base_traversal = global_invocation_index * number_of_elements_each_invocation_must_traverse;
    let ceil_traversal = base_traversal + number_of_elements_each_invocation_must_traverse;

    for (var i = base_traversal; i < ceil_traversal; i += 1)
    {
        let maybe_nondense_data = SetNondense_get(i);
        
        if (maybe_nondense_data != ~SetEmptySentinel)
        {
            // Ok, so we got one of the rare instances where we need to put the data
            // into the output buffer

            // get a new index
            let free_idx = atomicAdd(&unique_len, 1);

            // write our data into the buffer
            unique_voxel_buffer[free_idx] = maybe_data_to_put_in_large;
        }
    }

    
}

fn SetNondense_get(idx: u32) -> u32
{
    return ~unique_voxel_set_nondense[idx]
}


// insert a key, returns the index the element resides at
fn SetNondense_insert(notkey: u32) -> u32
{
    var slot = u32hash(key) % unique_voxel_set_len;

    let key = ~notkey;

    loop
    {
        let res: __atomic_compare_exchange_result<u32> =
            atomicCompareExchangeWeak(&unique_voxel_set_nondense_xor[slot], ~SetEmptySentinel, key);

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
