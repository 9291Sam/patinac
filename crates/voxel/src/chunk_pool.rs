use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use gfx::wgpu;

use crate::data::{self, BrickMap, ChunkMetaData, MaybeBrickPtr};
use crate::suballocated_buffer::{SubAllocatedCpuTrackedBuffer, SubAllocatedCpuTrackedDenseSet};
use crate::{
    chunk_local_position_to_brick_position,
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
    game:     Arc<game::Game>,
    uuid:     util::Uuid,
    pipeline: Arc<gfx::GenericPipeline>,

    chunk_data:     wgpu::Buffer, // chunk data smuggled via the instance buffer
    indirect_calls: wgpu::Buffer, // indirect offsets and lengths

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
        let critical_section = ChunkPoolCriticalSection {
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
        };

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("ChunkPool BindGroupLayout"),
                    entries: &const {
                        [
                            wgpu::BindGroupLayoutEntry {
                                binding:    0,
                                visibility: wgpu::ShaderStages::VERTEX,
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: true
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    1,
                                visibility: wgpu::ShaderStages::VERTEX
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: true
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    2,
                                visibility: wgpu::ShaderStages::VERTEX
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: true
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    3,
                                visibility: wgpu::ShaderStages::VERTEX
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: true
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            }
                        ]
                    }
                });

        let visible_face_set_buffer_len_bytes =
            critical_section.visible_face_set.get_buffer_size_bytes();

        let bind_group =
            critical_section
                .brick_maps
                .get_buffer(|brick_map_buffer: &wgpu::Buffer| {
                    critical_section.material_bricks.get_buffer(
                        |material_bricks_buffer: &wgpu::Buffer| {
                            critical_section.visibility_bricks.get_buffer(
                                |visibility_bricks_buffer: &wgpu::Buffer| {
                                    Arc::new(
                                        renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                                            label:   Some("ChunkPool BindGroup"),
                                            layout:  &bind_group_layout,
                                            entries: &[
                                                wgpu::BindGroupEntry {
                                                    binding:  0,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: critical_section
                                                                .visible_face_set
                                                                .access_buffer(),
                                                            offset: 0,
                                                            size:   Some(
                                                                visible_face_set_buffer_len_bytes
                                                            )
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  1,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: brick_map_buffer,
                                                            offset: 0,
                                                            size:   Some(
                                                                visible_face_set_buffer_len_bytes
                                                            )
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  2,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: material_bricks_buffer,
                                                            offset: 0,
                                                            size:   Some(
                                                                visible_face_set_buffer_len_bytes
                                                            )
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  3,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: visibility_bricks_buffer,
                                                            offset: 0,
                                                            size:   Some(
                                                                visible_face_set_buffer_len_bytes
                                                            )
                                                        }
                                                    )
                                                }
                                            ]
                                        })
                                    )
                                }
                            )
                        }
                    )
                });

        let foo = 3;

        let bar = 4;

        todo!()
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
    }

    fn write_many_voxel_deadlocking(
        &self,
        chunk: &Chunk,
        things_to_insert: impl IntoIterator<Item = (ChunkLocalPosition, Voxel)>
    )
    {
        let ChunkPoolCriticalSection {
            brick_pointer_allocator,
            brick_maps,
            chunk_metadata,
            material_bricks,
            visibility_bricks,
            visible_face_set,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        for (position, voxel) in things_to_insert
        {
            let (brick_coordinate, brick_local_coordinate) =
                chunk_local_position_to_brick_position(position);

            brick_maps.access_mut(chunk.id as usize, |brick_map: &mut data::BrickMap| {
                let maybe_brick_ptr = brick_map.get_mut(brick_coordinate);

                let mut needs_clear = false;

                if maybe_brick_ptr.0 == MaybeBrickPtr::NULL.0
                {
                    *maybe_brick_ptr =
                        MaybeBrickPtr(brick_pointer_allocator.allocate().unwrap() as u32);

                    needs_clear = true;
                }

                material_bricks.access_mut(
                    maybe_brick_ptr.0 as usize,
                    |material_brick: &mut data::MaterialBrick| {
                        if needs_clear
                        {
                            *material_brick = data::MaterialBrick::new_filled(Voxel::Air);
                        }

                        material_brick.set_voxel(brick_local_coordinate, voxel);
                    }
                );

                visibility_bricks.access_mut(
                    maybe_brick_ptr.0 as usize,
                    |visibility_brick: &mut data::VisibilityBrick| {
                        if needs_clear
                        {
                            *visibility_brick = data::VisibilityBrick::new_empty();
                        }

                        if voxel.is_air()
                        {
                            visibility_brick.set_visibility(brick_local_coordinate, false);
                        }
                        else
                        {
                            visibility_brick.set_visibility(brick_local_coordinate, true);
                        }
                    }
                )
            });

            let does_voxel_exist = |chunk_position: ChunkLocalPosition| -> bool {
                let (brick_coordinate, brick_local_coordinate) =
                    chunk_local_position_to_brick_position(chunk_position);

                brick_maps.access_ref(chunk.id as usize, |brick_map: &data::BrickMap| {
                    visibility_bricks.access_ref(
                        brick_map.get(brick_coordinate).to_option().unwrap().0 as usize,
                        |visibility_brick: &data::VisibilityBrick| {
                            visibility_brick.is_visible(brick_local_coordinate)
                        }
                    )
                })
            };

            let metadata: &mut ChunkMetaData = chunk_metadata[chunk.id as usize].as_mut().unwrap();

            for d in data::VoxelFaceDirection::iterate()
            {
                if let Some(adj_pos) = (position.0.cast() + d.get_axis())
                    .try_cast()
                    .map(ChunkLocalPosition)
                {
                    if !does_voxel_exist(adj_pos)
                    {
                        metadata.faces[d as usize]
                            .insert(data::VoxelFace::new(position), visible_face_set);
                    }
                    else
                    {
                        let _ = metadata.faces[d as usize]
                            .remove(data::VoxelFace::new(adj_pos), visible_face_set);
                    }
                }
            }
        }
    }

    pub fn read_many_voxel_material(
        &self,
        chunk: &Chunk,
        voxels_to_read: impl IntoIterator<Item = ChunkLocalPosition>
    ) -> Vec<Voxel>
    {
        let ChunkPoolCriticalSection {
            brick_maps,
            material_bricks,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        voxels_to_read
            .into_iter()
            .map(|chunk_position| {
                let (brick_coordinate, brick_local_coordinate) =
                    chunk_local_position_to_brick_position(chunk_position);

                brick_maps.access_ref(chunk.id as usize, |brick_map: &data::BrickMap| {
                    material_bricks.access_ref(
                        brick_map.get(brick_coordinate).to_option().unwrap().0 as usize,
                        |material_brick: &data::MaterialBrick| {
                            material_brick.get_voxel(brick_local_coordinate)
                        }
                    )
                })
            })
            .collect()
    }

    pub fn read_many_voxel_occupied(
        &self,
        chunk: &Chunk,
        voxels_to_read: impl IntoIterator<Item = ChunkLocalPosition>
    ) -> Vec<bool>
    {
        let ChunkPoolCriticalSection {
            brick_maps,
            visibility_bricks,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        voxels_to_read
            .into_iter()
            .map(|chunk_position| {
                let (brick_coordinate, brick_local_coordinate) =
                    chunk_local_position_to_brick_position(chunk_position);

                brick_maps.access_ref(chunk.id as usize, |brick_map: &data::BrickMap| {
                    visibility_bricks.access_ref(
                        brick_map.get(brick_coordinate).to_option().unwrap().0 as usize,
                        |visibility_brick: &data::VisibilityBrick| {
                            visibility_brick.is_visible(brick_local_coordinate)
                        }
                    )
                })
            })
            .collect()
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
