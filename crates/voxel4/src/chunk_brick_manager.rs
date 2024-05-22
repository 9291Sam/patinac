use std::collections::HashSet;
use std::sync::Mutex;

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::glm;

use crate::gpu_data::{self, BrickMap, ChunkMetaData, MaterialBrick, VisibilityBrick};
use crate::material::Voxel;
use crate::{chunk_local_position_to_brick_position, BrickCoordinate, ChunkLocalPosition};

struct ChunkBrickManager
{
    chunk_id_allocator:  Mutex<util::FreelistAllocator>,
    brick_ptr_allocator: Mutex<util::FreelistAllocator>,

    #[cfg(debug_assertions)]
    active_chunks: Mutex<HashSet<ChunkId>>,

    // chunk_id indexed
    chunk_meta_data_buffer: gfx::CpuTrackedBuffer<gpu_data::ChunkMetaData>,
    chunk_brick_map_buffer: gfx::CpuTrackedBuffer<gpu_data::BrickMap>,

    // brick ptr indexed
    visibility_brick_buffer: gfx::CpuTrackedBuffer<gpu_data::VisibilityBrick>,
    material_brick_buffer:   gfx::CpuTrackedBuffer<gpu_data::MaterialBrick>
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AnyBitPattern, NoUninit, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkId(u16);

impl ChunkBrickManager
{
    pub fn new() {}

    // TODO: a function that scans the entire list and collapses homogenous Bricks
    // TODO: modify brick ptr to be able to store that :skull:

    // the previously contained voxel
    pub fn insert_voxel(&self, chunk: ChunkId, pos: ChunkLocalPosition) -> Voxel
    {
        // allocate brick
    }

    pub fn read_voxel(&self, chunk: ChunkId, pos: ChunkLocalPosition) -> Option<Voxel>
    {
        let (BrickCoordinate(brick_coordinate), brick_local_pos) =
            chunk_local_position_to_brick_position(pos);

        let brick_ptr = self
            .chunk_brick_map_buffer
            .access_ref(chunk.0 as usize, |brick_map: &BrickMap| {
                unsafe {
                    *brick_map
                        .brick_map
                        .get_unchecked(brick_coordinate.x as usize)
                        .get_unchecked(brick_coordinate.y as usize)
                        .get_unchecked(brick_coordinate.z as usize)
                }
            })
            .to_option()?;

        Some(
            self.material_brick_buffer
                .access_ref(brick_ptr.0 as usize, |brick: &MaterialBrick| {
                    brick.get_voxel(brick_local_pos)
                })
        )
    }

    pub fn allocate_chunk(&self) -> ChunkId
    {
        let new_chunk_id = ChunkId(
            self.chunk_id_allocator
                .lock()
                .unwrap()
                .allocate()
                .expect("Too many chunks") as u16
        );

        #[cfg(debug_assertions)]
        self.active_chunks.lock().unwrap().insert(new_chunk_id);

        new_chunk_id
    }

    pub fn dealloc_chunk(&self, id: ChunkId)
    {
        #[cfg(debug_assertions)]
        assert!(
            self.active_chunks.lock().unwrap().remove(&id),
            "ChunkId {id:?} was not in active_chunks"
        );

        // keep the mutex locked while we're freeing stuff
        let mut chunk_id_allocator = self.chunk_id_allocator.lock().unwrap();
        let mut brick_ptr_allocator = self.brick_ptr_allocator.lock().unwrap();

        #[cfg(debug_assertions)]
        self.chunk_meta_data_buffer.write(
            id.0 as usize,
            ChunkMetaData {
                pos: glm::Vec4::zeros()
            }
        );

        self.chunk_brick_map_buffer
            .access_mut(id.0 as usize, |brick_map: &mut BrickMap| {
                // free dependencies
                brick_map
                    .brick_map
                    .iter()
                    .flatten()
                    .flatten()
                    .for_each(|p| {
                        #[cfg(debug_assertions)]
                        {
                            self.visibility_brick_buffer
                                .write(p.0 as usize, VisibilityBrick::new_empty());

                            self.material_brick_buffer
                                .write(p.0 as usize, MaterialBrick::new_filled(Voxel::Air));
                        }

                        unsafe { brick_ptr_allocator.free(p.0 as usize) }
                    });

                brick_map.null_all_ptrs();
            });

        unsafe { chunk_id_allocator.free(id.0 as usize) };
    }
}
