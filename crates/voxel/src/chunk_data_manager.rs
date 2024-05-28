use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use gfx::{glm, wgpu};
use smallvec::SmallVec;

use crate::face_manager::{FaceId, MaybeFaceId, VoxelFaceDirection};
use crate::material::Voxel;
use crate::{
    chunk_local_position_to_brick_position,
    get_world_offset_of_chunk,
    gpu_data,
    BrickCoordinate,
    BrickLocalPosition,
    ChunkCoordinate,
    ChunkLocalPosition,
    BRICK_EDGE_LEN_VOXELS
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ChunkId(pub(crate) u16);

pub(crate) struct ChunkDataManager
{
    chunk_pos_id_map: ChunkCoordinateToIdMap,

    chunk_id_allocator: util::FreelistAllocator,
    chunk_meta_data:    gfx::CpuTrackedBuffer<gpu_data::ChunkMetaData>,
    chunk_brick_map:    gfx::CpuTrackedBuffer<gpu_data::BrickMap>,

    brick_ptr_allocator:     util::FreelistAllocator,
    visibility_brick_buffer: gfx::CpuTrackedBuffer<gpu_data::VisibilityBrick>,
    material_brick_buffer:   gfx::CpuTrackedBuffer<gpu_data::MaterialBrick>,

    registered_faces: Vec<FaceIdBrick>
}

#[cfg(debug_assertions)]
impl Drop for ChunkDataManager
{
    fn drop(&mut self)
    {
        if self.chunk_id_allocator.peek().0 != 0
        {
            log::warn!("Retained {} ChunkIds", self.chunk_id_allocator.peek().0);
        }
    }
}

#[no_mangle]
static NUMBER_OF_VISIBLE_BRICKS: AtomicUsize = AtomicUsize::new(0);
#[no_mangle]
static NUMBER_OF_CHUNKS: AtomicUsize = AtomicUsize::new(0);

impl ChunkDataManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        let max_valid_id = u16::MAX - 1;

        // TODO: reisizing the buffers rather than preallocation
        const MAX_BRICKS: usize = 2usize.pow(18);

        ChunkDataManager {
            chunk_pos_id_map:        ChunkCoordinateToIdMap::new(),
            chunk_id_allocator:      util::FreelistAllocator::new(max_valid_id as usize),
            chunk_meta_data:         gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                max_valid_id as usize,
                String::from("Chunk Data Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            chunk_brick_map:         gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                48,
                String::from("ChunkBrickManager BrickMap Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            brick_ptr_allocator:     util::FreelistAllocator::new(2usize.pow(22)),
            visibility_brick_buffer: gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                MAX_BRICKS,
                String::from("ChunkBrickManager VisibilityBrick Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            material_brick_buffer:   gfx::CpuTrackedBuffer::new(
                renderer.clone(),
                MAX_BRICKS,
                String::from("ChunkBrickManager MaterialBrick Buffer"),
                wgpu::BufferUsages::STORAGE
            ),
            registered_faces:        vec![FaceIdBrick::new_empty(); MAX_BRICKS]
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

    // returns the voxel that was previously there
    pub fn insert_voxel(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition,
        voxel: Voxel
    ) -> Voxel
    {
        let (brick_coordinate, brick_local_pos) = chunk_local_position_to_brick_position(pos);

        let maybe_brick_ptr = self
            .chunk_brick_map
            .access_ref(chunk_id.0 as usize, |brick_map: &gpu_data::BrickMap| {
                brick_map.get(brick_coordinate)
            })
            .to_option();

        let mut is_brick_uninitialized = false;

        let brick_ptr = match maybe_brick_ptr
        {
            Some(ptr) => ptr,
            None =>
            {
                let new_ptr =
                    gpu_data::BrickPtr(self.brick_ptr_allocator.allocate().unwrap() as u32);

                NUMBER_OF_VISIBLE_BRICKS
                    .store(self.brick_ptr_allocator.peek().0, Ordering::Relaxed);

                self.chunk_brick_map.access_mut(
                    chunk_id.0 as usize,
                    |brick_map: &mut gpu_data::BrickMap| {
                        *brick_map.get_mut(brick_coordinate) = gpu_data::MaybeBrickPtr(new_ptr.0);
                    }
                );

                is_brick_uninitialized = true;

                new_ptr
            }
        };

        self.visibility_brick_buffer.access_mut(
            brick_ptr.0 as usize,
            |visibility_brick: &mut gpu_data::VisibilityBrick| {
                if is_brick_uninitialized
                {
                    *visibility_brick = gpu_data::VisibilityBrick::new_empty();
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
            |material_brick: &mut gpu_data::MaterialBrick| {
                if is_brick_uninitialized
                {
                    *material_brick = gpu_data::MaterialBrick::new_filled(Voxel::Air);
                }

                let prev_voxel = material_brick.get_voxel(brick_local_pos);

                material_brick.set_voxel(brick_local_pos, voxel);

                prev_voxel
            }
        )
    }

    pub fn remove_voxel(&mut self, chunk_id: ChunkId, pos: ChunkLocalPosition) -> Voxel
    {
        let (brick_coordinate, brick_local_pos) = chunk_local_position_to_brick_position(pos);

        let brick_ptr = self
            .chunk_brick_map
            .access_ref(chunk_id.0 as usize, |brick_map: &gpu_data::BrickMap| {
                brick_map.get(brick_coordinate)
            })
            .to_option()
            .expect("Tried to remove a voxel in an unallocated brick!");

        self.visibility_brick_buffer.access_mut(
            brick_ptr.0 as usize,
            |visibility_brick: &mut gpu_data::VisibilityBrick| {
                visibility_brick.set_visibility(brick_local_pos, false);
            }
        );

        self.material_brick_buffer.access_mut(
            brick_ptr.0 as usize,
            |material_brick: &mut gpu_data::MaterialBrick| {
                let prev_voxel = material_brick.get_voxel(brick_local_pos);

                material_brick.set_voxel(brick_local_pos, Voxel::Air);

                prev_voxel
            }
        )
    }

    pub fn is_voxel_visible(&self, chunk_id: ChunkId, pos: ChunkLocalPosition) -> bool
    {
        let (brick_coordinate, brick_local_pos) = chunk_local_position_to_brick_position(pos);

        self.chunk_brick_map
            .access_ref(chunk_id.0 as usize, |brick_map: &gpu_data::BrickMap| {
                match brick_map.get(brick_coordinate).to_option()
                {
                    Some(ptr) =>
                    {
                        self.visibility_brick_buffer.access_ref(
                            ptr.0 as usize,
                            |visibility_brick: &gpu_data::VisibilityBrick| {
                                visibility_brick.is_visible(brick_local_pos)
                            }
                        )
                    }
                    None => false
                }
            })
    }

    pub fn deregister_all_voxel_faces(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition
    ) -> SmallVec<[FaceId; 6]>
    {
        let mut out = SmallVec::new();

        out
    }

    // returns None if no brick is inserted
    fn get_voxel_faces(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition
    ) -> Option<&mut [MaybeFaceId; 6]>
    {
        let (brick_coordinate, brick_local_pos) = chunk_local_position_to_brick_position(pos);

        let brick_ptr = self
            .chunk_brick_map
            .access_ref(chunk_id.0 as usize, |brick_map: &gpu_data::BrickMap| {
                brick_map.get(brick_coordinate)
            })
            .to_option()?;

        Some(
            self.registered_faces
                .get_mut(brick_ptr.0 as usize)
                .unwrap()
                .get_mut(brick_local_pos)
        )
    }

    // returns None if no brick is inserted
    pub fn does_face_exist(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition,
        direction: VoxelFaceDirection
    ) -> bool
    {
        match self.get_voxel_faces(chunk_id, pos)
        {
            Some(faces) => faces[direction as usize].to_option().is_some(),
            None => false
        }
    }

    pub fn register_voxel_face(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition,
        direction: VoxelFaceDirection,
        face_id: FaceId
    )
    {
        let maybe_old_face_id = std::mem::replace(
            &mut self.get_voxel_faces(chunk_id, pos).as_mut().unwrap()[direction as usize],
            MaybeFaceId::from_face_id(face_id)
        );

        assert!(maybe_old_face_id.to_option().is_none())
    }

    pub fn deregister_voxel_face(
        &mut self,
        chunk_id: ChunkId,
        pos: ChunkLocalPosition,
        direction: VoxelFaceDirection
    ) -> FaceId
    {
        let old_face_id = std::mem::replace(
            &mut self.get_voxel_faces(chunk_id, pos).as_mut().unwrap()[direction as usize],
            MaybeFaceId::NULL
        );

        old_face_id.to_option().unwrap()
    }

    pub fn insert_chunk_at(&mut self, chunk_coord: ChunkCoordinate) -> ChunkId
    {
        assert!(self.get_chunk_id(chunk_coord).is_none());

        let new_id = ChunkId(
            self.chunk_id_allocator
                .allocate()
                .expect("Tried to allocate too many chunks") as u16
        );

        NUMBER_OF_CHUNKS.store(self.chunk_id_allocator.peek().0, Ordering::Relaxed);

        self.chunk_meta_data.write(
            new_id.0 as usize,
            gpu_data::ChunkMetaData {
                position: glm::vec3_to_vec4(&get_world_offset_of_chunk(chunk_coord).0.cast()),
                scale:    glm::vec3_to_vec4(&glm::Vec3::new(1.0, 1.0, 1.0))
            }
        );

        self.chunk_brick_map
            .access_mut(new_id.0 as usize, |brick_map: &mut gpu_data::BrickMap| {
                brick_map.null_all_ptrs();
            });

        self.chunk_pos_id_map
            .insert(chunk_coord, new_id)
            .ok_or::<()>(())
            .unwrap_err();

        new_id
    }

    pub fn delete_chunk_at(&mut self, chunk_coord: ChunkCoordinate, id: ChunkId) -> Vec<FaceId>
    {
        assert_eq!(self.get_chunk_id(chunk_coord), Some(id));

        unsafe { self.chunk_id_allocator.free(id.0 as usize) };

        let mut face_ids_to_dealloc: Vec<FaceId> = Vec::new();

        self.chunk_brick_map
            .access_mut(id.0 as usize, |brick_map: &mut gpu_data::BrickMap| {
                brick_map
                    .brick_map
                    .iter()
                    .flatten()
                    .flatten()
                    .for_each(|maybe_ptr| {
                        if let Some(ptr) = maybe_ptr.to_option()
                        {
                            unsafe { self.brick_ptr_allocator.free(ptr.0 as usize) }

                            self.registered_faces
                                .get_mut(ptr.0 as usize)
                                .unwrap()
                                .get_flattened_face_ids()
                                .iter()
                                .for_each(|f| {
                                    if let Some(face_id_to_remove) = f.to_option()
                                    {
                                        face_ids_to_dealloc.push(face_id_to_remove);
                                    }
                                })
                        }
                    })
            });

        face_ids_to_dealloc
    }

    pub fn get_buffer<R>(&self, buf_access_func: impl FnOnce(&wgpu::Buffer) -> R) -> R
    {
        self.chunk_meta_data.get_buffer(buf_access_func)
    }

    pub(crate) fn replicate_to_gpu(&mut self) -> bool
    {
        self.chunk_meta_data.replicate_to_gpu()
    }
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

#[derive(Clone)]
struct FaceIdBrick
{
    brick:
        [[[[MaybeFaceId; 6]; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS]
}

impl FaceIdBrick
{
    pub fn new_empty() -> Self
    {
        Self {
            brick: [[[[MaybeFaceId::NULL; 6]; BRICK_EDGE_LEN_VOXELS]; BRICK_EDGE_LEN_VOXELS];
                BRICK_EDGE_LEN_VOXELS]
        }
    }

    pub fn get_mut(&mut self, brick_local_pos: BrickLocalPosition) -> &mut [MaybeFaceId; 6]
    {
        unsafe {
            self.brick
                .get_unchecked_mut(brick_local_pos.0.x as usize)
                .get_unchecked_mut(brick_local_pos.0.y as usize)
                .get_unchecked_mut(brick_local_pos.0.z as usize)
        }
    }

    pub fn get_flattened_face_ids(&self) -> &[MaybeFaceId]
    {
        self.brick.as_flattened().as_flattened().as_flattened()
    }
}
