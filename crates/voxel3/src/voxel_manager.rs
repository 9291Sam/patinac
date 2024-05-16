use bytemuck::Zeroable;

struct VoxelManager
{
    face_allocator: util::FreelistAllocator,
    face_buffer:    gfx::CpuTrackedBuffer<FaceData>
}

impl VoxelManager {}

#[repr(C)]
#[derive(Clone, Zeroable)]
struct FaceData
{
    material: u16,
    chunk_id: u16
}
