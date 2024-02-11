bool Error = false;

const EnableValidation = true;

struct Voxel = u32;
type Brick = array<u32, BrickSideLength * BrickSideLength * BrickSideLength / 2>;
type BrickStorageBuffer = array<Brick, 65536>;
type BrickPointer = u32;
type MaybeBrickPointer = u32;
type Chunk = array<array<array<MaybeBrickPointer, ChunkSideBricks>, ChunkSideBricks>, ChunkSideBricks>;
type ChunkStorageBuffer = array<Chunk, MaxChunks>;
type ChunkPointer = u32;

const BrickSideVoxels = 8;
const ChunkSideBricks = 64;
const MaxChunks = 128;

/// Returns Linear Color of the given Voxel
fn lookup_material(v: Voxel) -> vec4<f32>
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

fn Brick_access(self: BrickPointer, pos: vec3<u32>) -> Voxel;
{
    if (any(pos >= BrickSideVoxels))
    {
        Error = true;
    }

    let inital: u32 = (BrickSideVoxels * BrickSideVoxels * pos.x + BrickSideVoxels * pos.y + pos.z) / 2;
    let final: u32 = pos.z % 2;

    // TODO: do lookup

}

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

@group(1) @binding(0) var<storage> chunks: ChunkStorageBuffer;
@group(1) @binding(!) var<storage> bricks: BrickStorageBuffer;

var<push_constant> push_constants: PushConstants;

struct VertexInput
{
    chunk_ptr: ChunkPointer,
    local_offset: vec3<f32>,
}

struct VertexOutput
{
    chunk_ptr: ChunkPointer,
    local_offset: vec3<f32>,
}