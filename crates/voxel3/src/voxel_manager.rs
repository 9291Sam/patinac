use bytemuck::{AnyBitPattern, NoUninit};
use gfx::glm;

struct VoxelManager
{
    face_allocator: util::FreelistAllocator,
    face_buffer:    gfx::CpuTrackedBuffer<FaceData> // chunk_data_buffer: gfx::CpuTrackedBuffer<>
}

impl VoxelManager {}

// make an algorithm that finds all of the ranges of things that need to be
// drawn upload that list of ranges into a
// TODO: use draw_indrect

// no vertex buffer each 6 looks at a different range

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit)]
struct FaceData
// is allocated at a specific index
{
    material:              u16,
    chunk_id:              u16,
    // 9 bits x
    // 9 bits y
    // 9 bits z
    // 3 bits normal
    // 1 bit visibility
    // 1 bit unused
    location_within_chunk: u32
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit)]
struct ChunkData
{
    position:              glm::Vec4,
    offset_into_brick_map: u32,
    adjacent_chunk_ids:    [u16; 6]
}
