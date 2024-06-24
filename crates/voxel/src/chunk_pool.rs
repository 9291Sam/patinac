use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::usize::MAX;

use gfx::{glm, wgpu};

use crate::data::{self, ChunkMetaData, MaterialBrick, MaybeBrickPtr};
use crate::{ChunkCoordinate, ChunkLocalPosition, Voxel, CHUNK_EDGE_LEN_BRICKS};

const MAX_CHUNKS: usize = 256;
const BRICKS_TO_PREALLOCATE: usize =
    CHUNK_EDGE_LEN_BRICKS * CHUNK_EDGE_LEN_BRICKS * CHUNK_EDGE_LEN_BRICKS * 32;

pub struct Chunk
{
    id: u32
}

pub struct ChunkPool
{
    uuid: util::Uuid,

    critical_section: Mutex<ChunkPoolCriticalSection>
}

struct ChunkPoolCriticalSection
{
    chunk_id_allocator:      util::FreelistAllocator,
    brick_pointer_allocator: util::FreelistAllocator,

    // chunk_id -> brick map
    brick_maps:     gfx::CpuTrackedBuffer<data::BrickMap>,
    // chunk_id -> ChunkMetaData
    chunk_metadata: gfx::CpuTrackedBuffer<data::ChunkMetaData>,

    // chunk_id -> {ChunkLocalPosition}

    // brick_pointer -> Brick
    material_bricks:   gfx::CpuTrackedBuffer<data::MaterialBrick>,
    visibility_bricks: gfx::CpuTrackedBuffer<data::VisibilityBrick>
}

impl ChunkPool
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> ChunkPool
    {
        ChunkPool {
            uuid:             util::Uuid::new(),
            critical_section: Mutex::new(ChunkPoolCriticalSection {
                chunk_id_allocator:      util::FreelistAllocator::new(MAX_CHUNKS),
                brick_pointer_allocator: util::FreelistAllocator::new(BRICKS_TO_PREALLOCATE),
                brick_maps:              gfx::CpuTrackedBuffer::new(
                    renderer.clone(),
                    MAX_CHUNKS,
                    String::from("ChunkPool BrickMap Buffer"),
                    wgpu::BufferUsages::STORAGE
                ),
                chunk_metadata:          gfx::CpuTrackedBuffer::new(
                    renderer.clone(),
                    MAX_CHUNKS,
                    String::from("ChunkPool ChunkMetaData Buffer"),
                    wgpu::BufferUsages::STORAGE
                ),
                material_bricks:         gfx::CpuTrackedBuffer::new(
                    renderer.clone(),
                    BRICKS_TO_PREALLOCATE,
                    String::from("ChunkPool MaterialBrick Buffer"),
                    wgpu::BufferUsages::STORAGE
                ),
                visibility_bricks:       gfx::CpuTrackedBuffer::new(
                    renderer.clone(),
                    BRICKS_TO_PREALLOCATE,
                    String::from("ChunkPool VisibilityBrick Buffer"),
                    wgpu::BufferUsages::STORAGE
                )
            })
        }
    }

    pub fn allocate_chunk(&self, coordinate: ChunkCoordinate) -> Chunk
    {
        let ChunkPoolCriticalSection {
            chunk_id_allocator,
            chunk_metadata,
            brick_maps,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        let new_chunk_id = chunk_id_allocator
            .allocate()
            .expect("Failed to allocate a new chunk");

        chunk_metadata.write(
            new_chunk_id,
            ChunkMetaData {
                coordinate,
                bool_is_visible: true as u32
            }
        );

        brick_maps.access_mut(new_chunk_id, |brick_map: &mut data::BrickMap| {
            brick_map.null_all_ptrs()
        });

        Chunk {
            id: new_chunk_id as u32
        }
    }

    pub fn deallocate_chunk(&self, chunk: Chunk)
    {
        let ChunkPoolCriticalSection {
            chunk_id_allocator,
            brick_maps,
            brick_pointer_allocator,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        unsafe { chunk_id_allocator.free(chunk.id as usize) };

        brick_maps.access_mut(chunk.id as usize, |brick_map: &mut data::BrickMap| {
            brick_map.access_mut_all(|ptr: &mut MaybeBrickPtr| {
                if let Some(old_ptr) = std::mem::replace(ptr, MaybeBrickPtr::NULL).to_option()
                {
                    unsafe { brick_pointer_allocator.free(old_ptr.0 as usize) }
                }
            })
        });
    }

    pub fn write_many_voxel(
        &self,
        chunk: &Chunk,
        voxels_to_write: impl IntoIterator<Item = (ChunkLocalPosition, Voxel)>
    )
    {
        todo!()

        // update bricks, from there determine if faces need to be updated
    }

    pub fn read_many_voxel(
        &self,
        chunk: &Chunk,
        voxels_to_read: impl IntoIterator<Item = ChunkLocalPosition>,
        iterator_size_estimate: Option<usize>
    ) -> Vec<Voxel>
    {
        let mut output: Vec<Voxel> = Vec::new();
        if let Some(len) = iterator_size_estimate
        {
            output.reserve(len);
        }

        todo!()
        // self.brick_maps
        //     .access_ref(chunk.id as usize, |brick_map: &data::BrickMap| {
        //         for voxel in voxels_to_read
        //         {
        //             todo!()
        //         }
        //     });

        // output
    }
}

impl Debug for ChunkPool
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "ChunkPool")
    }
}

impl gfx::Recordable for ChunkPool
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        std::borrow::Cow::Borrowed("ChunkPool")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
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
        todo!()
    }
}
