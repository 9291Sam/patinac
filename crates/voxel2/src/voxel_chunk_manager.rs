use std::num::NonZero;
use std::sync::{Arc, Mutex, Weak};

use bytemuck::Contiguous;
use gfx::wgpu;
use itertools::Itertools;

use crate::FaceId;

struct VoxelChunkManager
{
    this:              Weak<VoxelChunkManager>,
    voxel_data_buffer: wgpu::Buffer,
    face_id_allocator: Mutex<util::FreelistAllocator>
}

impl VoxelChunkManager
{
    pub fn new() -> Arc<VoxelChunkManager>
    {
        Arc::new_cyclic(|weak_this| {
            VoxelChunkManager {
                this:              weak_this.clone(),
                voxel_data_buffer: todo!(),
                face_id_allocator: todo!()
            }
        })
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
}
