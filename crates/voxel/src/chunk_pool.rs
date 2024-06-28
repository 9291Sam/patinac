use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use fnv::FnvHashSet;
use gfx::{glm, wgpu};

use crate::data::{self, BrickMap, ChunkMetaData, MaterialManager, MaybeBrickPtr};
use crate::passes::VoxelColorTransferRecordable;
use crate::suballocated_buffer::{SubAllocatedCpuTrackedBuffer, SubAllocatedCpuTrackedDenseSet};
use crate::{
    chunk_local_position_to_brick_position,
    get_world_offset_of_chunk,
    passes,
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

// use the index in the visible_Face_set as a unique id
// make another buffer that stores this face data, marking

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chunk
{
    id: u32
}

pub struct ChunkPool
{
    game:       Arc<game::Game>,
    uuid:       util::Uuid,
    pipeline:   Arc<gfx::GenericPipeline>,
    bind_group: Arc<wgpu::BindGroup>,

    chunk_data:              wgpu::Buffer, // chunk data smuggled via the instance buffer
    indirect_calls:          wgpu::Buffer, // indirect offsets and lengths
    number_of_indirect_args: AtomicU32,
    material_manager:        MaterialManager,

    critical_section: Mutex<ChunkPoolCriticalSection>,

    color_transfer_pass: Arc<passes::VoxelColorTransferRecordable>
}

struct ChunkPoolCriticalSection
{
    active_chunk_ids: FnvHashSet<u32>,

    chunk_id_allocator:      util::FreelistAllocator,
    brick_pointer_allocator: util::FreelistAllocator,

    // chunk_id -> brick map
    brick_maps:     gfx::CpuTrackedBuffer<data::BrickMap>,
    // chunk_id -> ChunkMetaData
    chunk_metadata: Box<[Option<ChunkMetaData>]>, // CPU chunk metadata and gpu chunk metadata

    // brick_pointer -> Brick
    material_bricks:   gfx::CpuTrackedBuffer<data::MaterialBrick>,
    visibility_bricks: gfx::CpuTrackedBuffer<data::VisibilityBrick>,

    // face_number -> FaceData
    visible_face_set: SubAllocatedCpuTrackedBuffer<data::VoxelFace>
}

impl ChunkPool
{
    pub fn new(game: Arc<game::Game>) -> Arc<ChunkPool>
    {
        let renderer = game.get_renderer();

        let critical_section = ChunkPoolCriticalSection {
            active_chunk_ids:        FnvHashSet::default(),
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
                                visibility: wgpu::ShaderStages::all(),
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
                                visibility: wgpu::ShaderStages::all(),
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
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    4,
                                visibility: wgpu::ShaderStages::all(),
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

        let material_manager = MaterialManager::new(&renderer);

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
                                                            size:   None
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  1,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: brick_map_buffer,
                                                            offset: 0,
                                                            size:   None
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  2,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: material_bricks_buffer,
                                                            offset: 0,
                                                            size:   None
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  3,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: visibility_bricks_buffer,
                                                            offset: 0,
                                                            size:   None
                                                        }
                                                    )
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding:  4,
                                                    resource: wgpu::BindingResource::Buffer(
                                                        wgpu::BufferBinding {
                                                            buffer: &material_manager
                                                                .get_material_buffer(),
                                                            offset: 0,
                                                            size:   None
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

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("ChunkPool PipelineLayout"),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout.clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..4u32
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("chunk_pool_renderer.wgsl"));

        let this = Arc::new(ChunkPool {
            game: game.clone(),
            uuid: util::Uuid::new(),
            pipeline: renderer.render_cache.cache_render_pipeline(
                gfx::CacheableRenderPipelineDescriptor {
                    label: "ChunkPool Pipeline".into(),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![ChunkIndirectData::desc()],
                    fragment_state: Some(gfx::CacheableFragmentState {
                        module:                           shader,
                        entry_point:                      "fs_main".into(),
                        targets:                          vec![Some(wgpu::ColorTargetState {
                            format:     wgpu::TextureFormat::Rg32Uint,
                            blend:      None,
                            write_mask: wgpu::ColorWrites::ALL
                        })],
                        constants:                        None,
                        zero_initialize_workgroup_memory: false
                    }),
                    primitive_state: wgpu::PrimitiveState {
                        topology:           wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Ccw,
                        cull_mode:          None,
                        polygon_mode:       wgpu::PolygonMode::Fill,
                        unclipped_depth:    false,
                        conservative:       false
                    },
                    depth_stencil_state: Some(gfx::Renderer::get_default_depth_state()),
                    multisample_state: wgpu::MultisampleState {
                        count:                     1,
                        mask:                      !0,
                        alpha_to_coverage_enabled: false
                    },
                    multiview: None,
                    vertex_specialization: None,
                    zero_initialize_vertex_workgroup_memory: false,
                    fragment_specialization: None,
                    zero_initialize_fragment_workgroup_memory: false
                }
            ),
            bind_group: bind_group.clone(),
            chunk_data: renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("ChunkPool ChunkIndirectData Buffer"),
                size:               std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64
                    * MAX_CHUNKS as u64
                    * 6,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }),
            indirect_calls: renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("ChunkPool DrawIndirectArgs Buffer"),
                size:               std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64
                    * MAX_CHUNKS as u64
                    * 6,
                usage:              wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false
            }),
            material_manager,
            number_of_indirect_args: AtomicU32::new(0),
            critical_section: Mutex::new(critical_section),
            color_transfer_pass: VoxelColorTransferRecordable::new(
                game.clone(),
                bind_group_layout.clone(),
                bind_group.clone()
            )
        });

        renderer.register(this.clone());

        this
    }

    // TODO: eventually modify to take a Transform and then construct the rt
    // structures based on that.
    pub fn allocate_chunk(&self, coordinate: ChunkCoordinate) -> Chunk
    {
        let ChunkPoolCriticalSection {
            active_chunk_ids,
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

        assert!(active_chunk_ids.insert(new_chunk_id as u32));

        Chunk {
            id: new_chunk_id as u32
        }
    }

    pub fn deallocate_chunk(&self, chunk: Chunk)
    {
        let ChunkPoolCriticalSection {
            active_chunk_ids,
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

        assert!(active_chunk_ids.remove(&chunk.id));

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
                    if let Some(brick_ptr) = brick_map.get(brick_coordinate).to_option()
                    {
                        visibility_bricks.access_ref(
                            brick_ptr.0 as usize,
                            |visibility_brick: &data::VisibilityBrick| {
                                visibility_brick.is_visible(brick_local_coordinate)
                            }
                        )
                    }
                    else
                    {
                        false
                    }
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
                        let _ = metadata.faces[d.opposite() as usize]
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
                    if let Some(ptr) = brick_map.get(brick_coordinate).to_option()
                    {
                        visibility_bricks.access_ref(
                            ptr.0 as usize,
                            |visibility_brick: &data::VisibilityBrick| {
                                visibility_brick.is_visible(brick_local_coordinate)
                            }
                        )
                    }
                    else
                    {
                        false
                    }
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
        let ChunkPoolCriticalSection {
            active_chunk_ids,
            brick_maps,
            chunk_metadata,
            material_bricks,
            visibility_bricks,
            visible_face_set,
            brick_pointer_allocator,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        assert!(!brick_maps.replicate_to_gpu());
        assert!(!material_bricks.replicate_to_gpu());
        assert!(!visibility_bricks.replicate_to_gpu());
        visible_face_set.replicate_to_gpu();

        let mut indirect_args: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();
        let mut chunk_indirect_data: Vec<ChunkIndirectData> = Vec::new();

        let mut idx = 0;
        let mut total_number_of_faces = 0;
        let mut rendered_faces = 0;

        let chunk_offsets: [glm::I32Vec3; 8] = std::array::from_fn(|idx| {
            glm::I32Vec3::new(
                (CHUNK_EDGE_LEN_VOXELS * (idx / 4)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * ((idx / 2) % 2)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * (idx % 2)) as i32
            )
        });

        for chunk_id in active_chunk_ids.iter()
        {
            let metadata: &ChunkMetaData = chunk_metadata[*chunk_id as usize].as_ref().unwrap();

            if !metadata.is_visible
            {
                continue;
            }

            for dir in data::VoxelFaceDirection::iterate()
            {
                let draw_range = metadata.faces[dir as usize].get_global_range(&visible_face_set);

                if draw_range.is_empty()
                {
                    continue;
                }

                total_number_of_faces += draw_range.end + 1 - draw_range.start;

                let camera_world_position = camera.get_position();

                'outer: for offset in chunk_offsets.iter().map(|o| {
                    (o.cast() + get_world_offset_of_chunk(metadata.coordinate)).cast()
                        - camera_world_position
                })
                {
                    let is_camera_in_chunk = offset.magnitude() < CHUNK_EDGE_LEN_VOXELS as f32;

                    let is_chunk_visible = {
                        let is_chunk_in_camera_view =
                            offset.normalize().dot(&camera.get_forward_vector())
                                > (renderer.get_fov().max() / 2.0).cos();

                        is_chunk_in_camera_view || is_camera_in_chunk
                    };

                    let is_chunk_direction_visible =
                        offset.normalize().dot(&dir.get_axis().cast()) < 0.0;

                    if is_camera_in_chunk || (is_chunk_visible && is_chunk_direction_visible)
                    {
                        indirect_args.push(wgpu::util::DrawIndirectArgs {
                            vertex_count:   (draw_range.end + 1 - draw_range.start) * 6,
                            instance_count: 1,
                            first_vertex:   draw_range.start * 6,
                            first_instance: idx as u32
                        });

                        chunk_indirect_data.push(ChunkIndirectData {
                            position: get_world_offset_of_chunk(metadata.coordinate).cast(),
                            dir:      dir as u32,
                            id:       *chunk_id
                        });

                        idx += 1;

                        rendered_faces += draw_range.end + 1 - draw_range.start;
                        break 'outer;
                    }
                }
            }
        }

        fn draw_args_as_bytes(args: &[wgpu::util::DrawIndirectArgs]) -> &[u8]
        {
            unsafe {
                std::slice::from_raw_parts(
                    args.as_ptr() as *const u8,
                    args.len() * std::mem::size_of::<wgpu::util::DrawIndirectArgs>()
                )
            }
        }

        #[no_mangle]
        static NUMBER_OF_CHUNKS: AtomicUsize = AtomicUsize::new(0);

        #[no_mangle]
        static NUMBER_OF_VISIBLE_FACES: AtomicUsize = AtomicUsize::new(0);

        #[no_mangle]
        static NUMBER_OF_TOTAL_FACES: AtomicUsize = AtomicUsize::new(0);

        #[no_mangle]
        static NUMBER_OF_BRICKS_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

        #[no_mangle]
        static VRAM_USED_BYTES: AtomicUsize = AtomicUsize::new(0);

        NUMBER_OF_CHUNKS.store(active_chunk_ids.len(), Ordering::Relaxed);
        NUMBER_OF_VISIBLE_FACES.store(rendered_faces as usize, Ordering::Relaxed);
        NUMBER_OF_TOTAL_FACES.store(total_number_of_faces as usize, Ordering::Relaxed);
        NUMBER_OF_BRICKS_ALLOCATED.store(brick_pointer_allocator.peek().0, Ordering::Relaxed);
        VRAM_USED_BYTES.store(
            visible_face_set.get_memory_used_bytes() as usize,
            Ordering::Relaxed
        );

        self.number_of_indirect_args
            .store(indirect_args.len() as u32, Ordering::SeqCst);

        renderer.queue.write_buffer(
            &self.indirect_calls,
            0,
            draw_args_as_bytes(&indirect_args[..])
        );

        renderer
            .queue
            .write_buffer(&self.chunk_data, 0, cast_slice(&chunk_indirect_data[..]));

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::VoxelDiscovery),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.bind_group.clone()),
                None,
                None
            ],
            transform:   Some(gfx::Transform::new())
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.set_vertex_buffer(0, self.chunk_data.slice(..));
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.multi_draw_indirect(
            &self.indirect_calls,
            0,
            self.number_of_indirect_args.load(Ordering::SeqCst)
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkIndirectData
{
    pub position: glm::Vec3,
    pub dir:      u32,
    pub id:       u32
}

impl ChunkIndirectData
{
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Uint32];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Instance,
            attributes:   &Self::ATTRIBS
        }
    }
}
