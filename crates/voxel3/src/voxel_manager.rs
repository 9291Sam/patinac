use bytemuck::{AnyBitPattern, NoUninit, Zeroable};

struct VoxelManager
{
    face_allocator: util::FreelistAllocator,
    face_buffer:    gfx::CpuTrackedBuffer<FaceData>
    // chunk_data_buffer: gfx::CpuTrackedBuffer<>
}

impl VoxelManager {}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit)]
struct FaceData
// is allocated at a specific index
{
    material:              u16,
    chunk_id:              u16,
    location_within_chunk: u32 /* 9 bits each direction | 3 bits normal | 1 bit visibility | 1
                                * bit unused */
}
