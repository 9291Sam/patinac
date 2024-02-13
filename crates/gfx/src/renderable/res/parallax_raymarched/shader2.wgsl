var<private> Error: false = false;

// TODO: pipeline overrideable constant
var<private> EnableValidation: bool = true;

struct Voxel = u32;
type Brick = array<u32, BrickSideVoxels * BrickSideVoxels * BrickSideVoxels / 2>;
type BrickStorageBuffer = array<Brick, 131072>; // 128 MiB
type BrickPointer = u32;
type MaybeBrickPointer = u32;
type Chunk = array<array<array<MaybeBrickPointer, ChunkSideBricks>, ChunkSideBricks>, ChunkSideBricks>;
type ChunkStorageBuffer = array<Chunk, 128>; // 128MibB
type ChunkPointer = u32;

const BrickSideVoxels = 8;
const ChunkSideBricks = 64;

struct GlobalInfo
{
    camera_pos: vec4<f32>,
    
}

struct Matricies
{
    data: array<mat4x4<f32>, 1024>
}

struct PushConstants
{
    id: u32,
}

@group(0) @binding(0) var<uniform> global_info: GlobalInfo;
@group(0) @binding(1) var<uniform> global_model_view_projection: Matricies;
@group(0) @binding(2) var<uniform> global_model: Matricies;
@group(0) @binding(3) var<uniform> global_view_projection: Matricies;

@group(1) @binding(0) var<storage> ChunkBuffer: ChunkStorageBuffer;
@group(1) @binding(1) var<storage> BrickBuffer: BrickStorageBuffer;

var<push_constant> push_constants: PushConstants;

struct VertexInput
{
    position: vec3<f32>,
    chunk_ptr: ChunkPointer,
    local_offset: vec3<f32>,
}

struct VertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    chunk_ptr: ChunkPointer,
    local_offset: vec3<f32>,
    camera_pos: vec3<f32>,
}

struct FragmentOutput
{
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>
}

@vertex fn vs_main(in: VertexInput) -> VertexOutput
{
    var out: VertexOutput;
    out.clip_position = global_model_view_projection[push_constants.id] * vec4<f32>(input.position, 1.0);
    out.chunk_ptr = in.chunk_ptr;
    out.local_offset = in.local_offset;
    
    let world_pos_vertex_intercalc: vec4<f32> = global_model[push_constants.id] * vec4<f32>(input.position, 1.0);
    let world_pos_vertex: vec3<f32> = world_pos_vertex_intercalc.xyz / world_pos_vertex_intercalc.w;

    out.camera_pos = (world_pos_vertex - global_info.camera_pos);

    return out;
}

@fragment fn fs_main(in: VertexOutput) -> FragmentOutput
{
    let camera_in_chunk: bool =
        all(in.camera_pos > vec3<f32>(0.0)) &&
        all(in.camera_pos < vec3<f32>(f32(ChunkSideBricks * BrickSideVoxels)));

    var ray: Ray;
    ray.direction = normalize(in.local_offset - in.camera_pos);

    if (camera_in_chunk)
    {
        ray.origin = in.camera_pos;
    }
    else // camera is outside of chunk
    {
        ray.origin = in.local_offset
    }

    // two for loops with two seperate traces
    let result: VoxelTraceResult = Chunk_trace(in.chunk_ptr, ray);

    if (!result.hit)
    {
        discard;
    }

    var out: FragmentOutput;
    
    let depth_intercalc: vec4<f32> = global_view_projection[push_constants.id] * vec4<f32>(result.strike_pos_world, 1.0);
    out.depth = depth_intercalc.z / depth_intercalc.w;
    out.color = Voxel_get_material(result.voxel);

    return out;
}



/// TODO: proper materials...
/// @return Linear color of the given Voxel
fn Voxel_get_material(v: Voxel) -> vec4<f32>
{
    switch (v)
    {
        case 0: return vec4<f32>(0.0);
        case 1: return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        case 2: return vec4<f32>(0.0, 1.0, 0.0, 1.0);
        case 3: return vec4<f32>(0.0, 0.0, 1.0, 1.0);
        default: return vec4<f32>(0.0);
    }
}

fn Voxel_isVisible(v)
{
    switch (v)
    {
        case 0: return false;
        default: return true;
    }
}

fn Brick_access(self: BrickPointer, pos: vec3<u32>) -> Voxel;
{
    if (any(pos >= BrickSideVoxels))
    {
        Error = true;
    }

    let inital: u32 = (BrickSideVoxels * BrickSideVoxels * pos.x + BrickSideVoxels * pos.y + pos.z) / 2;
    let final: u32 = pos.z % 2;

    let val = BrickBuffer[self][inital];

    switch (final)
    {
        case 0: return extractBits(val, 0, 16);
        case 1: return extractBits(val, 16, 16);
        default: {Error = true; return 0;}
    }
}

struct VoxelTraceResult
{

}

fn Chunk_trace(self: ChunkPointer, ray: Ray) -> VoxelTraceResult
{
    var active_brick_ptr;

    for (bricks)
    {
        for (voxels in brick)
        {

        }
    }


}