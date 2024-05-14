

// Texture<FromFragmentShader>
@group(0) @binding(0) var voxel_discovery_image: texture_2d<u32>;
// [number_of_image_workgroups, 1, 1];
@group(0) @binding(1) var<storage, read_write> indirect_color_calc_buffer: array<atomic<u32>, 3>;
// [FaceIdInfo, ...]
@group(0) @binding(2) var<storage, read_write> face_id_buffer: array<FaceInfo>;
// number_of_voxels_inx_buffer
@group(0) @binding(3) var<storage, read_write> number_of_unique_voxels: atomic<u32>;
// [FaceId, ...]
@group(0) @binding(4) var<storage, read_write> unique_voxel_buffer: array<atomic<u32>>;

@compute @workgroup_size(32, 32)
fn cs_main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
){}

struct FaceInfo
{
    ///   x [0]    y [1]
    /// [0,  15] [      ] | chunk_id
    /// [16, 31] [      ] | material
    /// [      ] [0,   0] | is_visible
    /// [      ] [1,  31] | unused
    low: atomic<u32>,
    high: atomic<u32>,
}

fn div_ceil(lhs: u32, rhs: u32) -> u32 {
    let d = lhs / rhs;
    let r = lhs % rhs;

    if r > 0 && rhs > 0 {
       return d + 1;
    } else {
       return d;
    }
}