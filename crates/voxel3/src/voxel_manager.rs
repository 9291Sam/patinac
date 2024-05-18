use std::fmt::Debug;
use std::sync::Arc;

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::glm;

struct VoxelManager
{
    face_id_allocator: util::FreelistAllocator,
    face_id_buffer:    super::CpuTrackedDenseSet<u32>,
    face_buffer:       gfx::CpuTrackedBuffer<FaceData>
}

impl Debug for VoxelManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "Voxel Manager")
    }
}

impl VoxelManager
{
    pub fn new() -> Arc<Self>
    {
        Arc::new(VoxelManager {
            face_id_allocator: todo!(),
            face_id_buffer:    todo!(),
            face_buffer:       todo!()
        })
    }

    // no chunks for now, just one global chunk

    pub fn insert_face() {}
}

impl gfx::Recordable for VoxelManager
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        todo!()
    }

    fn get_uuid(&self) -> util::Uuid
    {
        todo!()
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        todo!()
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), None) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.draw(0..self.face_id_buffer.get_number_of_elements() as u32, 0..1)
    }
}

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

// one massive draw
// % 6
// [face_id_buffer] // use the data here to lookup everything else in the
// FaceData buffer
