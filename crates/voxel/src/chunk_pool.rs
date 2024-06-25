use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::usize::MAX;

use gfx::{glm, wgpu};

use crate::data::{self, BrickMap, ChunkMetaData, MaterialBrick, MaybeBrickPtr};
use crate::suballocated_buffer::{SubAllocatedCpuTrackedBuffer, SubAllocatedCpuTrackedDenseSet};
use crate::{
    chunk_local_position_from_brick_positions,
    ChunkCoordinate,
    ChunkLocalPosition,
    Voxel,
    CHUNK_EDGE_LEN_BRICKS,
    CHUNK_EDGE_LEN_VOXELS
};

const MAX_CHUNKS: usize = 256;
const BRICKS_TO_PREALLOCATE: usize =
    CHUNK_EDGE_LEN_BRICKS * CHUNK_EDGE_LEN_BRICKS * CHUNK_EDGE_LEN_BRICKS * 32;
const FACES_TO_PREALLOCATE: usize = 1024 * 1024 * 128;

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
    chunk_metadata: Box<[Option<ChunkMetaData>]>,

    // brick_pointer -> Brick
    material_bricks:   gfx::CpuTrackedBuffer<data::MaterialBrick>,
    visibility_bricks: gfx::CpuTrackedBuffer<data::VisibilityBrick>,

    visible_face_set: SubAllocatedCpuTrackedBuffer<data::VoxelFace>
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
                chunk_metadata:          vec![None; MAX_CHUNKS].into_boxed_slice(),
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
                ),
                visible_face_set:        SubAllocatedCpuTrackedBuffer::new(
                    renderer.clone(),
                    FACES_TO_PREALLOCATE as u32,
                    "ChunkPool VisibleFaceSet Buffer",
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
            visible_face_set,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        let new_chunk_id = chunk_id_allocator
            .allocate()
            .expect("Failed to allocate a new chunk");

        chunk_metadata[new_chunk_id] = Some(ChunkMetaData {
            coordinate,
            is_visible: true,
            faces: std::array::from_fn(|_| {
                unsafe {
                    SubAllocatedCpuTrackedDenseSet::new(
                        CHUNK_EDGE_LEN_VOXELS * 16,
                        visible_face_set
                    )
                }
            })
        });

        brick_maps.write(new_chunk_id, BrickMap::new());

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
            visible_face_set,
            chunk_metadata,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        unsafe { chunk_id_allocator.free(chunk.id as usize) };

        brick_maps.access_mut(chunk.id as usize, |brick_map: &mut data::BrickMap| {
            brick_map.access_mut_all(|_, ptr: &mut MaybeBrickPtr| {
                if let Some(old_ptr) = std::mem::replace(ptr, MaybeBrickPtr::NULL).to_option()
                {
                    unsafe { brick_pointer_allocator.free(old_ptr.0 as usize) }
                }
            })
        });

        chunk_metadata[chunk.id as usize]
            .take()
            .unwrap()
            .faces
            .into_iter()
            .for_each(|data| visible_face_set.deallocate(unsafe { data.into_inner() }))

        // TODO: free metadata faces
    }

    pub fn write_many_voxel(
        &self,
        chunk: &Chunk,
        voxels_to_write: impl IntoIterator<Item = (ChunkLocalPosition, Voxel)>
    )
    {
        let iterator = voxels_to_write.into_iter();

        let mut chunks = iterator.array_chunks::<3072>();

        for i in chunks.by_ref()
        {
            self.write_many_voxel_deadlocking(chunk, i);
        }

        if let Some(remainder) = chunks.into_remainder()
        {
            self.write_many_voxel_deadlocking(chunk, remainder);
        }

        // update bricks, from there determine if faces need to be updated
    }

    fn write_many_voxel_deadlocking(
        &self,
        chunk: &Chunk,
        pos: impl IntoIterator<Item = (ChunkLocalPosition, Voxel)>
    )
    {
        let ChunkPoolCriticalSection {
            chunk_id_allocator,
            brick_pointer_allocator,
            brick_maps,
            chunk_metadata,
            material_bricks,
            visibility_bricks,
            visible_face_set
        } = &mut *self.critical_section.lock().unwrap();

        for p in pos
        {
            todo!()
            // insert voxel into material map
            // insert voxel into occupancy map
            // insert faces (check for opposing ones)
        }
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
