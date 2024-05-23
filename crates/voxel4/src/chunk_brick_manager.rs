use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::{glm, wgpu};

use crate::gpu_data::{
    self,
    BrickMap,
    BrickPtr,
    ChunkMetaData,
    MaterialBrick,
    MaybeBrickPtr,
    VisibilityBrick
};
use crate::material::Voxel;
use crate::{chunk_local_position_to_brick_position, BrickCoordinate, ChunkLocalPosition};

struct ChunkBrickManager
{
    #[cfg(debug_assertions)]
    active_chunks: Mutex<HashSet<ChunkId>>,

    chunk_id_allocator:     Mutex<util::FreelistAllocator>,
    // chunk_id indexed
    chunk_meta_data_buffer: gfx::CpuTrackedBuffer<gpu_data::ChunkMetaData>,
    chunk_brick_map_buffer: gfx::CpuTrackedBuffer<gpu_data::BrickMap>,

    brick_ptr_allocator:     Mutex<util::FreelistAllocator>,
    // brick ptr indexed
    visibility_brick_buffer: gfx::CpuTrackedBuffer<gpu_data::VisibilityBrick>,
    material_brick_buffer:   gfx::CpuTrackedBuffer<gpu_data::MaterialBrick>
}

impl ChunkBrickManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> ChunkBrickManager
    {
        // values, fresh from my ass!
        ChunkBrickManager {
            active_chunks:           Mutex::new(HashSet::new()),
            chunk_id_allocator:      Mutex::new(util::FreelistAllocator::new(
                u16::MAX as usize - 1
            )),
            chunk_meta_data_buffer:  gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                16,
                String::from("ChunkBrickManager MetaData Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            chunk_brick_map_buffer:  gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                16,
                String::from("ChunkBrickManager BrickMap Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            brick_ptr_allocator:     Mutex::new(util::FreelistAllocator::new(2usize.pow(22))),
            visibility_brick_buffer: gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                32768,
                String::from("ChunkBrickManager VisibilityBrick Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            material_brick_buffer:   gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                32768,
                String::from("ChunkBrickManager MaterialBrick Buffer"),
                wgpu::BufferUsages::STORAGE
            )
        }
    }

    // TODO: a function that scans the entire list and collapses homogenous Bricks
    // TODO: modify brick ptr to be able to store that :skull:

    // the previously contained voxel
    pub fn insert_voxel(&self, chunk: ChunkId, pos: ChunkLocalPosition, voxel: Voxel) -> Voxel
    {
        let (BrickCoordinate(brick_coordinate), brick_local_pos) =
            chunk_local_position_to_brick_position(pos);

        let maybe_brick_ptr = self
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
            .to_option();

        let mut needs_zeroing = false;

        let brick_ptr = match maybe_brick_ptr
        {
            Some(ptr) => ptr,
            None =>
            {
                let new_ptr =
                    BrickPtr(self.brick_ptr_allocator.lock().unwrap().allocate().unwrap() as u32);

                self.chunk_brick_map_buffer.access_mut(
                    chunk.0 as usize,
                    |brick_map: &mut BrickMap| {
                        unsafe {
                            *brick_map
                                .brick_map
                                .get_unchecked_mut(brick_coordinate.x as usize)
                                .get_unchecked_mut(brick_coordinate.y as usize)
                                .get_unchecked_mut(brick_coordinate.z as usize) =
                                MaybeBrickPtr(new_ptr.0);
                        }
                    }
                );

                needs_zeroing = true;

                new_ptr
            }
        };

        self.visibility_brick_buffer.access_mut(
            brick_ptr.0 as usize,
            |visibility_brick: &mut VisibilityBrick| {
                if needs_zeroing
                {
                    *visibility_brick = VisibilityBrick::new_empty();
                }

                if let Voxel::Air = voxel
                {
                    visibility_brick.set_visibility(brick_local_pos, false);
                }
                else
                {
                    visibility_brick.set_visibility(brick_local_pos, true);
                }
            }
        );

        self.material_brick_buffer.access_mut(
            brick_ptr.0 as usize,
            |material_brick: &mut MaterialBrick| {
                if needs_zeroing
                {
                    *material_brick = MaterialBrick::new_filled(Voxel::Air);
                }

                let prev_voxel = material_brick.get_voxel(brick_local_pos);

                material_brick.set_voxel(brick_local_pos, voxel);

                prev_voxel
            }
        )
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
        self.ensure_chunk_id_length();

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

    // returns true if any bind groups need to be recreated
    pub fn replicate(&self) -> bool
    {
        let mut needs_replication = false;

        // black box is to prevent short-circuiting

        needs_replication |= std::hint::black_box(self.chunk_meta_data_buffer.replicate_to_gpu());
        std::hint::black_box(needs_replication);

        needs_replication |= std::hint::black_box(self.chunk_brick_map_buffer.replicate_to_gpu());
        std::hint::black_box(needs_replication);

        needs_replication |= std::hint::black_box(self.visibility_brick_buffer.replicate_to_gpu());
        std::hint::black_box(needs_replication);

        needs_replication |= std::hint::black_box(self.material_brick_buffer.replicate_to_gpu());
        std::hint::black_box(needs_replication);

        needs_replication
    }

    pub fn access_buffers<K>(
        &self,
        access_func: impl FnOnce(ChunkBrickManagerBufferViews<'_>) -> K
    ) -> K
    {
        self.chunk_meta_data_buffer.get_buffer(|meta_data_buffer| {
            self.chunk_brick_map_buffer.get_buffer(|brick_map_buffer| {
                self.visibility_brick_buffer
                    .get_buffer(|visibility_buffer| {
                        self.material_brick_buffer.get_buffer(|material_buffer| {
                            access_func(ChunkBrickManagerBufferViews {
                                chunk_meta_data_buffer:  meta_data_buffer,
                                chunk_brick_map_buffer:  brick_map_buffer,
                                visibility_brick_buffer: visibility_buffer,
                                material_brick_buffer:   material_buffer
                            })
                        })
                    })
            })
        })
    }

    fn ensure_chunk_id_length(&self)
    {
        let mut chunk_id_allocator = self.chunk_id_allocator.lock().unwrap();

        let temp_id = chunk_id_allocator.allocate().unwrap();

        let meta_len = self.chunk_meta_data_buffer.get_cpu_len();
        let brick_map_len = self.chunk_brick_map_buffer.get_cpu_len();

        if temp_id + 1 >= meta_len
        {
            self.chunk_meta_data_buffer.realloc(meta_len * 2);
        }

        if temp_id + 1 >= brick_map_len
        {
            self.chunk_brick_map_buffer.realloc(brick_map_len * 3 / 2)
        }

        unsafe { chunk_id_allocator.free(temp_id) }
    }

    fn ensure_brick_ptr_length(&self)
    {
        let mut brick_ptr_allocator = self.brick_ptr_allocator.lock().unwrap();

        let temp_id = brick_ptr_allocator.allocate().unwrap();

        let visibility_len = self.visibility_brick_buffer.get_cpu_len();
        let material_len = self.material_brick_buffer.get_cpu_len();

        if temp_id + 1 >= visibility_len
        {
            self.visibility_brick_buffer.realloc(visibility_len * 2);
        }

        if temp_id + 1 >= material_len
        {
            self.material_brick_buffer.realloc(material_len * 3 / 2);
        }

        unsafe { brick_ptr_allocator.free(temp_id) }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AnyBitPattern, NoUninit, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkId(u16);

pub struct ChunkBrickManagerBufferViews<'b>
{
    chunk_meta_data_buffer:  &'b wgpu::Buffer,
    chunk_brick_map_buffer:  &'b wgpu::Buffer,
    visibility_brick_buffer: &'b wgpu::Buffer,
    material_brick_buffer:   &'b wgpu::Buffer
}
