// @group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
// @group(0) @binding(1) var indirect_rt_workgroups_buffer: array<atomic<u32>, 3>;
// @group(0) @binding(2) var<storage> mostly_unique_len: atomic<u32>;
// @group(0) @binding(3) var<storage> mostly_unique_voxel_buffer: array<atomic<u32>>; 
// @group(0) @binding(4) var<storage> unique_len: atomic<u32>;
// @group(0) @binding(5) var<storage> unique_voxel_buffer: array<atomic<u32>>;

// const SET_SIZE: u32 = 32 * 32;
// var<workgroup> workgroup_set: array<atomic<u32>, SET_SIZE>; // tune value. min required is 1024 max is 8192
// const SetEmptySentinel: u32 = ~0u;


// @compute @workgroup_size(32, 32)
// fn cs_main()
// {

// }