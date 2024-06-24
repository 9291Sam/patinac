use std::fmt::Debug;
use std::pin::Pin;
use std::sync::Mutex;

use dashmap::DashMap;
use gfx::glm;

use crate::{ChunkCoordinate, ChunkLocalPosition, Voxel};

const MAX_CHUNKS: usize = 2usize.pow(16);

pub struct Chunk
{
    pub is_visible: bool,
    id:             u32
}

pub struct ChunkPool
{
    uuid: util::Uuid,

    chunk_data_critical_section: Mutex<ChunkDataCriticalSection>
}

impl ChunkPool
{
    pub fn new() -> ChunkPool
    {
        ChunkPool {
            chunk_data_critical_section: Mutex::new(ChunkDataCriticalSection {
                chunk_id_allocator: util::FreelistAllocator::new(MAX_CHUNKS),
                chunk_data:         vec![
                    ChunkData {
                        coordinate: ChunkCoordinate(glm::I32Vec3::zeros())
                    };
                    MAX_CHUNKS
                ]
            })
        }
    }

    pub fn allocate_chunk(&self, coordinate: ChunkCoordinate) -> Chunk
    {
        let ChunkDataCriticalSection {
            chunk_id_allocator,
            chunk_data
        } = &mut *self.chunk_data_critical_section.lock().unwrap();

        let id = chunk_id_allocator.allocate().unwrap_or_else(|_| {
            panic!(
                "Failed to allocate a ChunkId, {} chunks are currently allocated",
                chunk_data.len()
            )
        });

        chunk_data[id] = ChunkData {
            coordinate
        };

        Chunk {
            is_visible: true,
            id:         id as u32
        }
    }

    pub fn deallocate_chunk(&self, chunk: Chunk)
    {
        let ChunkDataCriticalSection {
            chunk_id_allocator, ..
        } = &mut *self.chunk_data_critical_section.lock().unwrap();

        let Chunk {
            id, ..
        } = chunk;

        unsafe { chunk_id_allocator.free(id as usize) };
    }

    pub fn write_many_voxel(
        &self,
        chunk: &Chunk,
        voxels_to_write: impl IntoIterator<Item = (ChunkLocalPosition, Voxel)>
    )
    {
        todo!()
    }

    pub fn read_many_voxel(
        &self,
        chunk: &Chunk,
        voxels_to_read: impl IntoIterator<Item = ChunkLocalPosition>
    )
    {
        todo!()
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

struct ChunkDataCriticalSection
{
    chunk_id_allocator: util::FreelistAllocator,
    chunk_data:         Vec<ChunkData>
}

#[derive(Clone)]
struct ChunkData
{
    is_visible: 
    coordinate: ChunkCoordinate
}
