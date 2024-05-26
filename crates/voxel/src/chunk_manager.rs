use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use bytemuck::{AnyBitPattern, NoUninit};
use dashmap::DashMap;
use gfx::{glm, wgpu};

use crate::{get_world_position_from_chunk, ChunkCoordinate};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChunkId(pub(crate) u16);

pub(crate) struct ChunkManager
{
    chunk_id_allocator: util::FreelistAllocator,
    chunk_pos_id_map:   ChunkCoordinateToIdMap, // HashMap<ChunkCoordinate, ChunkId>,

    chunk_data: gfx::CpuTrackedBuffer<GpuChunkData> // TODO: brick map stuff
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
            chunk_pos_id_map:   ChunkCoordinateToIdMap::new(),
            chunk_id_allocator: util::FreelistAllocator::new(max_valid_id as usize)
        }
    }

    pub fn get_or_insert_chunk(&mut self, chunk_coord: ChunkCoordinate) -> ChunkId
    {
        self.get_chunk_id(chunk_coord)
            .unwrap_or_else(|| self.insert_chunk_at(chunk_coord))
    }

    pub fn get_chunk_id(&self, chunk_coord: ChunkCoordinate) -> Option<ChunkId>
    {
        self.chunk_pos_id_map.try_get(chunk_coord)
    }

    pub fn insert_chunk_at(&mut self, chunk_coord: ChunkCoordinate) -> ChunkId
    {
        debug_assert!(self.get_chunk_id(chunk_coord).is_none());

        let new_id = ChunkId(
            self.chunk_id_allocator
                .allocate()
                .expect("Tried to allocate too many chunks") as u16
        );

        self.chunk_data.write(
            new_id.0 as usize,
            GpuChunkData {
                position: glm::vec3_to_vec4(&get_world_position_from_chunk(chunk_coord).0.cast()),
                scale:    glm::vec3_to_vec4(&glm::Vec3::new(1.0, 1.0, 1.0))
            }
        );

        self.chunk_pos_id_map
            .insert(chunk_coord, new_id)
            .ok_or::<()>(())
            .unwrap_err();

        new_id
    }

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        self.chunk_data.get_buffer(buf_access_func)
    }

    pub(crate) fn replicate_to_gpu(&mut self) -> bool
    {
        self.chunk_data.replicate_to_gpu()
    }
}

#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern, NoUninit, Debug)]
struct GpuChunkData
{
    position: glm::Vec4,
    scale:    glm::Vec4
}

struct ChunkCoordinateToIdMap
{
    data: VecDeque<(ChunkCoordinate, ChunkId)>
}

impl ChunkCoordinateToIdMap
{
    pub fn new() -> Self
    {
        Self {
            data: VecDeque::new()
        }
    }

    /// Returns the value that may have been there before.
    /// None means first insertion
    pub fn insert(&mut self, coordinate: ChunkCoordinate, id: ChunkId) -> Option<ChunkId>
    {
        let seek = coordinate;

        match self.data.binary_search_by(|e| e.0.cmp(&seek))
        {
            Ok(idx) =>
            {
                Some(std::mem::replace(
                    unsafe { &mut self.data.get_mut(idx).unwrap_unchecked().1 },
                    id
                ))
            }
            Err(idx) =>
            {
                self.data.insert(idx, (coordinate, id));

                None
            }
        }
    }

    // returns the value if it was contained
    pub fn remove(&mut self, coordinate: ChunkCoordinate) -> Option<ChunkId>
    {
        let seek = coordinate;
        match self.data.binary_search_by(|e| e.0.cmp(&seek))
        {
            Ok(idx) =>
            unsafe { Some(self.data.remove(idx).unwrap_unchecked().1) },
            Err(_) => None
        }
    }

    pub fn try_get(&self, coordinate: ChunkCoordinate) -> Option<ChunkId>
    {
        let seek = coordinate;

        match self.data.binary_search_by(|e| e.0.cmp(&seek))
        {
            Ok(idx) =>
            unsafe { Some(self.data.get(idx).unwrap_unchecked().1) },
            Err(_) => None
        }
    }
}
