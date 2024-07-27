use core::f32;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use fnv::FnvHashSet;
use gfx::wgpu::util::DeviceExt;
use gfx::{glm, wgpu};
use rand::Rng;

use crate::data::{self, BrickMap, MaterialManager, MaybeBrickPtr};
use crate::suballocated_buffer::{SubAllocatedCpuTrackedBuffer, SubAllocatedCpuTrackedDenseSet};
use crate::{
    chunk_local_position_to_brick_position,
    get_world_offset_of_chunk,
    passes,
    ChunkCoordinate,
    ChunkLocalPosition,
    PointLight,
    Voxel,
    CHUNK_EDGE_LEN_BRICKS,
    CHUNK_EDGE_LEN_VOXELS
};

////! some of these are hardcoded into shaders
const MAX_CHUNKS: usize = 256;
const BRICKS_TO_PREALLOCATE: usize = CHUNK_EDGE_LEN_BRICKS * CHUNK_EDGE_LEN_BRICKS * MAX_CHUNKS * 4;
const FACES_TO_PREALLOCATE: usize = 1024 * 1024 * 16;

// chunks. bricks. faces

// use the index in the visible_Face_set as a unique id
// make another buffer that stores this face data, marking

pub struct Chunk
{
    id:   u32,
    pool: Weak<ChunkPool>
}

impl Drop for Chunk
{
    fn drop(&mut self)
    {
        match self.pool.upgrade()
        {
            Some(pool) =>
            {
                pool.deallocate_chunk(Chunk {
                    id:   self.id,
                    pool: Arc::downgrade(&pool)
                })
            }
            None =>
            {
                log::warn!("Drop of Chunk after ChunkPool")
            }
        }
    }
}

impl Eq for Chunk {}

impl PartialEq for Chunk
{
    fn eq(&self, other: &Self) -> bool
    {
        self.id == other.id
    }
}

impl Ord for Chunk
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering
    {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Chunk
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>
    {
        Some(self.id.cmp(&other.id))
    }
}

impl Hash for Chunk
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H)
    {
        self.id.hash(state);
    }
}

pub struct ChunkPool
{
    game: Arc<game::Game>,
    uuid: util::Uuid,
    pipeline: Arc<gfx::GenericPipeline>,
    face_and_brick_info_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    face_and_brick_info_bind_group: Arc<wgpu::BindGroup>,

    chunk_data:              wgpu::Buffer, // chunk data smuggled via the instance buffer
    indirect_calls:          wgpu::Buffer, // indirect offsets and lengths
    number_of_indirect_args: AtomicU32,
    #[allow(dead_code)]
    material_manager:        MaterialManager,

    critical_section: Mutex<ChunkPoolCriticalSection>
}

struct ChunkPoolCriticalSection
{
    active_chunk_ids: FnvHashSet<u32>,

    chunk_id_allocator:      util::FreelistAllocator,
    brick_pointer_allocator: util::FreelistAllocator,

    // chunk_id -> brick map
    brick_maps:     gfx::CpuTrackedBuffer<data::BrickMap>,
    // chunk_id -> data::CpuChunkData
    cpu_chunk_data: Box<[Option<data::CpuChunkData>]>, // CPU chunk metadata and gpu chunk metadata
    gpu_chunk_data: gfx::CpuTrackedBuffer<data::GpuChunkData>,

    // brick_pointer -> Brick
    material_bricks:   gfx::CpuTrackedBuffer<data::MaterialBrick>,
    visibility_bricks: gfx::CpuTrackedBuffer<data::VisibilityBrick>,

    // face_number -> FaceData
    visible_face_set:         SubAllocatedCpuTrackedBuffer<data::VoxelFace>,
    // face_number -> face_id
    #[allow(dead_code)]
    face_numbers_to_face_ids: wgpu::Buffer, // <data::FaceId>,

    // face_id
    face_id_counter:    wgpu::Buffer, // <u32>
    #[allow(dead_code)]
    rendered_face_info: wgpu::Buffer, // <data::RenderedFaceInfo>

    is_face_visible_buffer: wgpu::Buffer,      // <bool bits>
    indirect_rt_dispatch:   Arc<wgpu::Buffer>, // <[X, 1, 1]>

    point_lights_buffer:        wgpu::Buffer,
    point_lights_number_buffer: wgpu::Buffer,

    // renderpasses
    voxel_discovery_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    voxel_discovery_bind_group:        Arc<wgpu::BindGroup>,

    raytrace_indirect_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    raytrace_indirect_bind_group:        Arc<wgpu::BindGroup>,

    resize_pinger: util::PingReceiver,

    color_detector_pass: Arc<passes::ColorDetectorRecordable>,
    color_raytrace_pass: Arc<passes::ColorRaytracerRecordable>,
    color_transfer_pass: Arc<passes::VoxelColorTransferRecordable>
}

impl ChunkPool
{
    pub fn new(game: Arc<game::Game>) -> Arc<ChunkPool>
    {
        let renderer = game.get_renderer();

        let voxel_discovery_image_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some(
                        "VoxelColorTransferRecordable VoxelDiscoveryImage BindGroupLayout"
                    ),
                    entries: const {
                        &[wgpu::BindGroupLayoutEntry {
                            binding:    0,
                            visibility: wgpu::ShaderStages::FRAGMENT
                                .union(wgpu::ShaderStages::COMPUTE),
                            ty:         wgpu::BindingType::Texture {
                                sample_type:    wgpu::TextureSampleType::Uint,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled:   false
                            },
                            count:      None
                        }]
                    }
                });

        let voxel_discovery_bind_group = create_voxel_discovery_bind_group_from_layout(
            &game,
            voxel_discovery_image_layout.clone()
        );

        let face_and_brick_info_bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("ChunkPool FaceAndBrickInfo BindGroupLayout"),
                    entries: &const {
                        [
                            wgpu::BindGroupLayoutEntry {
                                binding:    0,
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
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    5,
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
                                binding:    6,
                                visibility: wgpu::ShaderStages::FRAGMENT
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: false
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    7,
                                visibility: wgpu::ShaderStages::FRAGMENT
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: false
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    8,
                                visibility: wgpu::ShaderStages::FRAGMENT
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: false
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    9,
                                visibility: wgpu::ShaderStages::FRAGMENT
                                    .union(wgpu::ShaderStages::COMPUTE),
                                ty:         wgpu::BindingType::Buffer {
                                    ty:                 wgpu::BufferBindingType::Storage {
                                        read_only: false
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size:   None
                                },
                                count:      None
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding:    10,
                                visibility: wgpu::ShaderStages::COMPUTE,
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
                                binding:    11,
                                visibility: wgpu::ShaderStages::COMPUTE,
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

        let raytrace_indirect_bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("ChunkPool RaytraceIndirect BindGroupLayout"),
                    entries: &const {
                        [wgpu::BindGroupLayoutEntry {
                            binding:    0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::Buffer {
                                ty:                 wgpu::BufferBindingType::Storage {
                                    read_only: false
                                },
                                has_dynamic_offset: false,
                                min_binding_size:   None
                            },
                            count:      None
                        }]
                    }
                });

        let visible_face_set_buffer = SubAllocatedCpuTrackedBuffer::new(
            renderer.clone(),
            FACES_TO_PREALLOCATE as u32,
            "ChunkPool VisibleFaceSet Buffer",
            wgpu::BufferUsages::STORAGE
        );

        let brick_map_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            MAX_CHUNKS,
            String::from("ChunkPool BrickMap Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let material_bricks_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            BRICKS_TO_PREALLOCATE,
            String::from("ChunkPool MaterialBrick Buffer"),
            wgpu::BufferUsages::STORAGE
        );
        let visibility_bricks_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            BRICKS_TO_PREALLOCATE,
            String::from("ChunkPool VisibilityBrick Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let material_manager = MaterialManager::new(renderer);

        let gpu_chunk_data_buffer = gfx::CpuTrackedBuffer::new(
            renderer.clone(),
            MAX_CHUNKS,
            String::from("ChunkPool GpuChunkData Buffer"),
            wgpu::BufferUsages::STORAGE
        );

        let is_face_number_visible_bits_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool IsFaceVisible BitBuffer"),
            size:               (FACES_TO_PREALLOCATE as u64 * u8::BITS as u64)
                .div_ceil(u32::BITS as u64),
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let face_numbers_to_face_ids_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool VisibleFaceIds Buffer"),
            size:               FACES_TO_PREALLOCATE as u64
                * std::mem::size_of::<data::FaceId>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let face_id_counter_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool FaceIdCounter Buffer"),
            size:               std::mem::size_of::<u32>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let rendered_face_info_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool RenderedFaceInfo Buffer"),
            size:               FACES_TO_PREALLOCATE as u64
                * std::mem::size_of::<data::RenderedFaceInfo>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let point_lights_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool PointLights Buffer"),
            size:               4096 * std::mem::size_of::<data::PointLight>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let point_lights_number_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("ChunkPool PointLightsNumber Buffer"),
            size:               std::mem::size_of::<u32>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let face_and_brick_info_bind_group = {
            Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("ChunkPool BindGroup"),
                layout:  &face_and_brick_info_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: visible_face_set_buffer.access_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: brick_map_buffer.get_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: material_bricks_buffer.get_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: visibility_bricks_buffer.get_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &material_manager.get_material_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  5,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: gpu_chunk_data_buffer.get_buffer(),
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &is_face_number_visible_bits_buffer,
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  7,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &face_numbers_to_face_ids_buffer,
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  8,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &face_id_counter_buffer,
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  9,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &rendered_face_info_buffer,
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  10,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &point_lights_buffer,
                            offset: 0,
                            size:   None
                        })
                    },
                    wgpu::BindGroupEntry {
                        binding:  11,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &point_lights_number_buffer,
                            offset: 0,
                            size:   None
                        })
                    }
                ]
            }))
        };

        let color_raytrace_dispatches_indirect_buffer = Arc::new(renderer.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label:    Some("ChunkPool ColorRayTracerDispatchesIndirect Buffer"),
                contents: bytes_of(&[0, 1, 1]),
                usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
            }
        ));

        let raytrace_indirect_bind_group = {
            Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("ChunkPool BindGroup"),
                layout:  &raytrace_indirect_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &color_raytrace_dispatches_indirect_buffer,
                        offset: 0,
                        size:   None
                    })
                }]
            }))
        };

        let critical_section = ChunkPoolCriticalSection {
            active_chunk_ids: FnvHashSet::default(),
            chunk_id_allocator: util::FreelistAllocator::new(MAX_CHUNKS),
            brick_pointer_allocator: util::FreelistAllocator::new(BRICKS_TO_PREALLOCATE),
            brick_maps: brick_map_buffer,
            cpu_chunk_data: vec![None; MAX_CHUNKS].into_boxed_slice(),
            gpu_chunk_data: gpu_chunk_data_buffer,
            material_bricks: material_bricks_buffer,
            visibility_bricks: visibility_bricks_buffer,
            visible_face_set: visible_face_set_buffer,
            face_numbers_to_face_ids: face_numbers_to_face_ids_buffer,
            face_id_counter: face_id_counter_buffer,
            rendered_face_info: rendered_face_info_buffer,
            is_face_visible_buffer: is_face_number_visible_bits_buffer,
            indirect_rt_dispatch: color_raytrace_dispatches_indirect_buffer.clone(),
            point_lights_buffer,
            point_lights_number_buffer,
            voxel_discovery_bind_group_layout: voxel_discovery_image_layout.clone(),
            voxel_discovery_bind_group: voxel_discovery_bind_group.clone(),
            raytrace_indirect_bind_group_layout: raytrace_indirect_bind_group_layout.clone(),
            raytrace_indirect_bind_group: raytrace_indirect_bind_group.clone(),
            resize_pinger: renderer.get_resize_pinger(),
            color_detector_pass: passes::ColorDetectorRecordable::new(
                game.clone(),
                voxel_discovery_image_layout.clone(),
                voxel_discovery_bind_group.clone(),
                face_and_brick_info_bind_group_layout.clone(),
                face_and_brick_info_bind_group.clone(),
                raytrace_indirect_bind_group_layout.clone(),
                raytrace_indirect_bind_group.clone()
            ),
            color_raytrace_pass: passes::ColorRaytracerRecordable::new(
                game.clone(),
                face_and_brick_info_bind_group_layout.clone(),
                face_and_brick_info_bind_group.clone(),
                color_raytrace_dispatches_indirect_buffer
            ),
            color_transfer_pass: passes::VoxelColorTransferRecordable::new(
                game.clone(),
                voxel_discovery_image_layout.clone(),
                voxel_discovery_bind_group.clone(),
                face_and_brick_info_bind_group_layout.clone(),
                face_and_brick_info_bind_group.clone()
            ) // )
        };

        // voxel_discovery_bind_group_layout: voxel_discovery_image_layout.clone(),
        // voxel_discovery_bind_group:        voxel_discovery_bind_group.clone(),
        // let cpu_chunk_data = v;
        // let critical_section = ChunkPoolCriticalSection {
        //     active_chunk_ids:        FnvHashSet::default(),
        //     resize_pinger:           renderer.get_resize_pinger(),
        //
        // };

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("ChunkPool PipelineLayout"),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        face_and_brick_info_bind_group_layout.clone(),
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
            face_and_brick_info_bind_group_layout: face_and_brick_info_bind_group_layout.clone(),
            face_and_brick_info_bind_group: face_and_brick_info_bind_group.clone(),
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
            critical_section: Mutex::new(critical_section)
        });

        renderer.register(this.clone());

        this
    }

    // TODO: eventually modify to take a Transform and then construct the rt
    // structures based on that.
    pub fn allocate_chunk(self: &Arc<Self>, coordinate: ChunkCoordinate) -> Chunk
    {
        let ChunkPoolCriticalSection {
            active_chunk_ids,
            chunk_id_allocator,
            cpu_chunk_data,
            gpu_chunk_data,
            brick_maps,
            visible_face_set,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        let new_chunk_id = chunk_id_allocator
            .allocate()
            .expect("Failed to allocate a new chunk");

        cpu_chunk_data[new_chunk_id] = Some(data::CpuChunkData {
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

        let o = get_world_offset_of_chunk(coordinate);

        gpu_chunk_data.write(
            new_chunk_id,
            data::GpuChunkData {
                position: glm::Vec4::new(o.x, o.y, o.z, 0.0)
            }
        );

        brick_maps.write(new_chunk_id, BrickMap::new());

        assert!(active_chunk_ids.insert(new_chunk_id as u32));

        Chunk {
            id:   new_chunk_id as u32,
            pool: Arc::downgrade(self)
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
            cpu_chunk_data,
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

        cpu_chunk_data[chunk.id as usize]
            .take()
            .unwrap()
            .faces
            .into_iter()
            .for_each(|data| visible_face_set.deallocate(unsafe { data.into_inner() }));

        std::mem::forget(chunk)
    }

    pub fn write_lights(&self, lights: &[data::PointLight])
    {
        let ChunkPoolCriticalSection {
            point_lights_buffer,
            point_lights_number_buffer,
            ..
        } = &*self.critical_section.lock().unwrap();

        self.game
            .get_renderer()
            .queue
            .write_buffer(point_lights_buffer, 0, cast_slice(lights));

        self.game.get_renderer().queue.write_buffer(
            point_lights_number_buffer,
            0,
            cast_slice(&[lights.len() as u32])
        );
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
            cpu_chunk_data,
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

            let cpu_chunk_data: &mut data::CpuChunkData =
                cpu_chunk_data[chunk.id as usize].as_mut().unwrap();

            for d in data::VoxelFaceDirection::iterate()
            {
                if let Some(adj_pos) = (position.0.cast() + d.get_axis())
                    .try_cast()
                    .map(ChunkLocalPosition)
                {
                    if !does_voxel_exist(adj_pos)
                    {
                        cpu_chunk_data.faces[d as usize]
                            .insert(data::VoxelFace::new(position), visible_face_set);
                    }
                    else
                    {
                        let _ = cpu_chunk_data.faces[d.opposite() as usize]
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
        encoder: &mut wgpu::CommandEncoder,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        static TIME_ALIVE: util::AtomicF32 = util::AtomicF32::new(0.0);
        const MAX_LIGHTS: usize = 256;
        const RADIUS: usize = 256;

        static LIGHTS: Mutex<[PointLight; MAX_LIGHTS]> = Mutex::new(
            [PointLight {
                position:        glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                color_and_power: glm::Vec4::new(0.0, 0.0, 0.0, 0.0),
                falloffs:        glm::Vec4::new(0.0, 0.0, 0.02, 0.01)
            }; MAX_LIGHTS]
        );

        let lights: &mut [PointLight; MAX_LIGHTS] = &mut LIGHTS.lock().unwrap();

        if TIME_ALIVE.load(Ordering::Acquire) < 0.0001
        {
            for (idx, l) in lights.iter_mut().enumerate()
            {
                let percent_around = (idx as f32) / (MAX_LIGHTS as f32);
                let angle = percent_around * f32::consts::PI * 2.0;

                l.color_and_power = glm::Vec4::new(1.0, 1.0, 1.0, idx as f32);

                l.falloffs.y = 10.0; // rand::thread_rng().gen_range(.0..5.0);

                l.position = glm::Vec4::new(
                    angle.sin() * ((8.0 * angle).cos().mul(120.0) + RADIUS as f32),
                    13.0,
                    angle.cos() * ((8.0 * angle).cos().mul(120.0) + RADIUS as f32),
                    0.0
                );
            }
        }

        TIME_ALIVE.aba_add(renderer.get_delta_time(), Ordering::SeqCst);

        self.write_lights(&*lights);

        let ChunkPoolCriticalSection {
            active_chunk_ids,
            brick_maps,
            cpu_chunk_data,
            material_bricks,
            visibility_bricks,
            visible_face_set,
            brick_pointer_allocator,
            gpu_chunk_data,
            resize_pinger,
            voxel_discovery_bind_group,
            voxel_discovery_bind_group_layout,
            raytrace_indirect_bind_group_layout,
            raytrace_indirect_bind_group,
            color_detector_pass,
            color_transfer_pass,
            indirect_rt_dispatch,
            face_id_counter,
            is_face_visible_buffer,
            color_raytrace_pass,
            ..
        } = &mut *self.critical_section.lock().unwrap();

        // encoder.clear_buffer(&face_numbers_to_face_ids, 0, None);
        encoder.clear_buffer(face_id_counter, 0, None);
        // encoder.clear_buffer(&rendered_face_info, 0, None);
        encoder.clear_buffer(is_face_visible_buffer, 0, None);
        encoder.clear_buffer(indirect_rt_dispatch, 0, Some(4));

        // We might need to update our passes.
        if resize_pinger.recv_all()
        {
            let new_discovery_bind_group = create_voxel_discovery_bind_group_from_layout(
                &self.game,
                voxel_discovery_bind_group_layout.clone()
            );

            *voxel_discovery_bind_group = new_discovery_bind_group;

            *color_detector_pass = passes::ColorDetectorRecordable::new(
                self.game.clone(),
                voxel_discovery_bind_group_layout.clone(),
                voxel_discovery_bind_group.clone(),
                self.face_and_brick_info_bind_group_layout.clone(),
                self.face_and_brick_info_bind_group.clone(),
                raytrace_indirect_bind_group_layout.clone(),
                raytrace_indirect_bind_group.clone()
            );

            *color_raytrace_pass = passes::ColorRaytracerRecordable::new(
                self.game.clone(),
                self.face_and_brick_info_bind_group_layout.clone(),
                self.face_and_brick_info_bind_group.clone(),
                indirect_rt_dispatch.clone()
            );

            *color_transfer_pass = passes::VoxelColorTransferRecordable::new(
                self.game.clone(),
                voxel_discovery_bind_group_layout.clone(),
                voxel_discovery_bind_group.clone(),
                self.face_and_brick_info_bind_group_layout.clone(),
                self.face_and_brick_info_bind_group.clone()
            );
        }

        assert!(!brick_maps.replicate_to_gpu());
        assert!(!material_bricks.replicate_to_gpu());
        assert!(!visibility_bricks.replicate_to_gpu());
        assert!(!gpu_chunk_data.replicate_to_gpu());
        visible_face_set.replicate_to_gpu();

        let mut indirect_args: Vec<wgpu::util::DrawIndirectArgs> = Vec::new();
        let mut chunk_indirect_data: Vec<ChunkIndirectData> = Vec::new();

        let mut idx = 0;
        let mut total_number_of_faces = 0;
        let mut rendered_faces = 0;
        let mut rendered_chunks = 0;

        let chunk_offsets: [glm::I32Vec3; 8] = std::array::from_fn(|idx| {
            glm::I32Vec3::new(
                (CHUNK_EDGE_LEN_VOXELS * (idx / 4)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * ((idx / 2) % 2)) as i32,
                (CHUNK_EDGE_LEN_VOXELS * (idx % 2)) as i32
            )
        });

        for chunk_id in active_chunk_ids.iter()
        {
            let this_cpu_chunk_data: &data::CpuChunkData =
                cpu_chunk_data[*chunk_id as usize].as_ref().unwrap();

            if !this_cpu_chunk_data.is_visible
            {
                continue;
            }

            let faces_before_chunk = rendered_faces;

            for dir in data::VoxelFaceDirection::iterate()
            {
                let draw_range =
                    this_cpu_chunk_data.faces[dir as usize].get_global_range(visible_face_set);

                if draw_range.is_empty()
                {
                    continue;
                }

                total_number_of_faces += draw_range.end + 1 - draw_range.start;

                let camera_world_position = camera.get_position();

                'outer: for offset in chunk_offsets.iter().map(|o| {
                    (o.cast() + get_world_offset_of_chunk(this_cpu_chunk_data.coordinate)).cast()
                        - camera_world_position
                })
                {
                    let is_camera_in_chunk =
                        offset.magnitude() < CHUNK_EDGE_LEN_VOXELS as f32 / 1.5;

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
                            position: get_world_offset_of_chunk(this_cpu_chunk_data.coordinate)
                                .cast(),
                            dir:      dir as u32,
                            id:       *chunk_id
                        });

                        idx += 1;

                        rendered_faces += draw_range.end + 1 - draw_range.start;
                        break 'outer;
                    }
                }
            }

            if rendered_faces > faces_before_chunk
            {
                rendered_chunks += 1;
            }
        }

        fn draw_args_as_bytes(args: &[wgpu::util::DrawIndirectArgs]) -> &[u8]
        {
            unsafe {
                std::slice::from_raw_parts(args.as_ptr() as *const u8, std::mem::size_of_val(args))
            }
        }

        #[no_mangle]
        static VRAM_USED_BYTES: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static FACES_VISIBLE: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static FACES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static BRICKS_VISIBLE: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static BRICKS_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static CHUNKS_VISIBLE: AtomicUsize = AtomicUsize::new(0);
        #[no_mangle]
        static CHUNKS_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

        FACES_VISIBLE.store(rendered_faces as usize, Ordering::Relaxed);
        FACES_ALLOCATED.store(total_number_of_faces as usize, Ordering::Relaxed);

        BRICKS_ALLOCATED.store(brick_pointer_allocator.peek().0, Ordering::Relaxed);

        CHUNKS_VISIBLE.store(rendered_chunks, Ordering::Relaxed);
        CHUNKS_ALLOCATED.store(active_chunk_ids.len(), Ordering::Relaxed);

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
                Some(self.face_and_brick_info_bind_group.clone()),
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

fn create_voxel_discovery_bind_group_from_layout(
    game: &game::Game,
    discovery_image_layout: Arc<wgpu::BindGroupLayout>
) -> Arc<wgpu::BindGroup>
{
    Arc::new(
        game.get_renderer()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("VoxelColorTransferRecordable VoxelDiscoveryImage BindGroup"),
                layout:  &discovery_image_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(
                        &game
                            .get_renderpass_manager()
                            .get_voxel_discovery_texture()
                            .get_view()
                    )
                }]
            })
    )
}
