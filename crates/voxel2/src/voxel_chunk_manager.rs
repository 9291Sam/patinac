use std::borrow::Cow;
use std::fmt::Debug;
use std::num::NonZero;
use std::sync::{Arc, Mutex, Weak};

use bytemuck::Contiguous;
use gfx::wgpu;
use itertools::Itertools;

use crate::{FaceId, VisibilityMarker};

/// The Pipeline
/// chunks render into Image<u64>
/// compute pass to mark visible faces and list their ids into a Buffer<FaceId>
/// compute pass over Buffer<FaceId> calculate colors && reset visibility
/// Raster pass over Image<u64> -> Screen Texture

struct VoxelChunkManager
{
    uuid: util::Uuid,
    game: Arc<game::Game>,
    this: Weak<VoxelChunkManager>,

    buffer_critical_section:      Mutex<Option<BufferCriticalSection>>,
    voxel_data_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    bind_group_windows: (
        util::Window<Arc<wgpu::BindGroup>>,
        util::WindowUpdater<Arc<wgpu::BindGroup>>
    ),

    face_id_allocator: Mutex<util::FreelistAllocator>,

    visibility_marker: Arc<VisibilityMarker>
}

impl Debug for VoxelChunkManager
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "VoxelChunkManager")
    }
}

impl VoxelChunkManager
{
    pub fn new(game: Arc<game::Game>) -> Arc<VoxelChunkManager>
    {
        // let (voxel_data_bind_group, buffers) =
        // Self::generate_voxel_buffers()

        // Arc::new_cyclic(|weak_this| {
        //     VoxelChunkManager {
        //         uuid:                           util::Uuid::new(),
        //         game:                           game.clone(),
        //         this:                           weak_this,
        //         indirect_color_calc_buffer:     todo!(),
        //         face_id_buffer:                 todo!(),
        //         number_of_unique_voxels_buffer: todo!(),
        //         unique_voxel_buffer:            todo!(),
        //         voxel_data_bind_group_layout:   todo!(),
        //         bind_group_windows:             todo!(),
        //         face_id_allocator:              todo!(),
        //         visibility_marker:              todo!()
        //     }
        // })

        todo!()
    }

    pub(crate) unsafe fn alloc_face_id(&self) -> FaceId
    {
        FaceId(
            self.face_id_allocator
                .lock()
                .unwrap()
                .allocate()
                .expect("Tried to allocate too many FaceId")
                .into_integer() as u32
        )
    }

    pub(crate) unsafe fn alloc_many_face_id(&self, amount: usize) -> Vec<FaceId>
    {
        let mut allocator = self.face_id_allocator.lock().unwrap();

        (0..amount)
            .map(|_| {
                FaceId(
                    allocator
                        .allocate()
                        .expect("Tried to allocate too many FaceId")
                        .into_integer() as u32
                )
            })
            .collect_vec()
    }

    pub(crate) unsafe fn dealloc_face_id(&self, id: FaceId)
    {
        self.face_id_allocator
            .lock()
            .unwrap()
            .free(NonZero::new(id.0 as usize).unwrap())
    }

    pub(crate) unsafe fn dealloc_many_face_id(&self, id: impl IntoIterator<Item = FaceId>)
    {
        let mut allocator = self.face_id_allocator.lock().unwrap();

        id.into_iter()
            .for_each(|i| allocator.free(NonZero::new(i.0 as usize).unwrap()))
    }

    fn generate_voxel_buffers(
        game: &game::Game,
        bind_group_layout: &wgpu::BindGroupLayout,
        maybe_old_buffers: Option<BufferCriticalSection>
    ) -> (wgpu::BindGroup, BufferCriticalSection)
    {
        let buffers = if let Some(old_buffers) = maybe_old_buffers
        {
        }
        else
        {
        };
    }
}

impl gfx::Recordable for VoxelChunkManager
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Voxel Chunk Manager")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::NoRecord {}
    }

    fn record<'s>(&'s self, _: &mut gfx::GenericPass<'s>, _: Option<gfx::DrawId>)
    {
        unreachable!()
    }
}

struct BufferCriticalSection
{
    indirect_color_calc_buffer:     wgpu::Buffer,
    face_id_buffer:                 wgpu::Buffer,
    number_of_unique_voxels_buffer: wgpu::Buffer,
    unique_voxel_buffer:            wgpu::Buffer
}
