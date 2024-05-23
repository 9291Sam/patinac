use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{AnyBitPattern, NoUninit};
use gfx::{glm, wgpu};

use crate::{ChunkCoordinate, ChunkLocalPosition, WorldPosition};

pub(crate) struct ChunkManager
{
    chunk_data:         gfx::CpuTrackedBuffer<GpuChunkData>,
    chunk_pos_id_map:   HashMap<ChunkCoordinate, u16>,
    chunk_id_allocator: util::FreelistAllocator
}

impl ChunkManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        let max_valid_id = u16::MAX - 1;

        ChunkManager {
            chunk_data:         gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                max_valid_id as usize,
                String::from("Chunk Data Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            chunk_pos_id_map:   HashMap::new(),
            chunk_id_allocator: util::FreelistAllocator::new(max_valid_id as usize)
        }
    }

    pub fn get_or_insert_chunk(&mut self, chunk_coord: ChunkCoordinate) -> u16
    {
        self.get_chunk_id(chunk_coord)
            .unwrap_or_else(|| self.insert_chunk_at(chunk_coord))
    }

    pub fn get_chunk_id(&self, chunk_coord: ChunkCoordinate) -> Option<u16>
    {
        self.chunk_pos_id_map.get(&chunk_coord).cloned()
    }

    pub fn insert_chunk_at(&mut self, chunk_coord: ChunkCoordinate) -> u16
    {
        assert!(self.get_chunk_id(chunk_coord).is_none());

        let new_id = self
            .chunk_id_allocator
            .allocate()
            .expect("Tried to allocate too many chunks");

        self.chunk_data.write(
            new_id,
            GpuChunkData {
                position: glm::vec3_to_vec4(&get_world_position_from_chunk(chunk_coord).0.cast()),
                scale:    glm::vec3_to_vec4(&glm::Vec3::new(1.0, 1.0, 1.0))
            }
        );

        self.chunk_pos_id_map
            .insert(chunk_coord, new_id as u16)
            .ok_or::<()>(())
            .unwrap_err();

        new_id as u16
    }

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        self.chunk_data.get_buffer(buf_access_func)
    }

    pub(crate) fn replicate_to_gpu(&self) -> bool
    {
        self.chunk_data.replicate_to_gpu()
    }
}

pub fn get_chunk_position_from_world(
    WorldPosition(world_pos): WorldPosition
) -> (ChunkCoordinate, ChunkLocalPosition)
{
    (
        ChunkCoordinate(world_pos.map(|w| w.div_euclid(512))),
        ChunkLocalPosition(world_pos.map(|w| w.rem_euclid(512)).try_cast().unwrap())
    )
}

pub fn get_world_position_from_chunk(chunk_coord: ChunkCoordinate) -> WorldPosition
{
    WorldPosition(chunk_coord.0.map(|p| p * 512).cast())
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug)]
struct GpuChunkData
{
    position: glm::Vec4,
    scale:    glm::Vec4
}