use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;

use gfx::{glm, wgpu};

use crate::cpu::{self, VoxelFaceDirection};
use crate::{gpu, BufferAllocation, ChunkLocalPosition, SubAllocatedCpuTrackedBuffer};

struct ChunkManager
{
    global_face_storage: Pin<Box<SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>>>,
    chunk:               Chunk
}

impl ChunkManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> ChunkManager
    {
        let mut allocator = Box::pin(SubAllocatedCpuTrackedBuffer::new(
            renderer,
            780974,
            "ChunkFacesSubBuffer",
            wgpu::BufferUsages::STORAGE
        ));

        ChunkManager {
            chunk:               Chunk::new(&mut allocator),
            global_face_storage: allocator
        }
    }

    pub fn insert_voxel(&mut self, pos: ChunkLocalPosition)
    {
        self.chunk.insert_voxel(pos);
    }
}

struct DirectionalFaceData
{
    owning_allocator: *mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>,
    dir:              cpu::VoxelFaceDirection,
    faces_allocation: BufferAllocation,
    faces_stored:     u32
}

impl DirectionalFaceData
{
    pub fn new(
        allocator: &mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>,
        dir: cpu::VoxelFaceDirection
    ) -> DirectionalFaceData
    {
        let alloc = allocator.allocate(96000);

        Self {
            owning_allocator: allocator as *mut _,
            dir,
            faces_allocation: alloc,
            faces_stored: 0
        }
    }

    pub fn insert_face(&mut self, face: gpu::VoxelFace)
    {
        if self.faces_allocation.get_length() > self.faces_stored
        {
            self.faces_stored += 1;

            unsafe {
                self.owning_allocator.as_mut_unchecked().write(
                    &self.faces_allocation,
                    self.faces_stored..(self.faces_stored + 1),
                    &[face]
                )
            }
        }
        else
        {
            panic!()
        }
    }
}

struct Chunk
{
    drawable_faces: [Option<DirectionalFaceData>; 6]
}

impl Chunk
{
    pub fn new(allocator: &mut SubAllocatedCpuTrackedBuffer<gpu::VoxelFace>) -> Chunk
    {
        Chunk {
            drawable_faces: std::array::from_fn(|i| {
                Some(DirectionalFaceData::new(
                    allocator,
                    VoxelFaceDirection::try_from(i as u8).unwrap()
                ))
            })
        }
    }

    pub fn insert_voxel(&mut self, local_pos: ChunkLocalPosition)
    {
        for d in VoxelFaceDirection::iterate()
        {
            self.drawable_faces[d as usize]
                .as_mut()
                .unwrap()
                .insert_face(gpu::VoxelFace::new(local_pos, glm::U8Vec2::new(1, 1)));
        }
    }

    pub fn get_draw_ranges(&self) -> [Option<(Range<u32>, VoxelFaceDirection)>; 6]
    {
        std::array::from_fn(|i| {
            unsafe {
                self.drawable_faces.get_unchecked(i).as_ref().map(|d| {
                    let start = d.faces_allocation.to_global_valid_range().start;

                    (start..(start + d.faces_stored), d.dir)
                })
            }
        })
    }
}
