use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use bytemuck::bytes_of;
use gfx::wgpu::include_wgsl;
use gfx::{glm, wgpu, CacheablePipelineLayoutDescriptor, CacheableRenderPipelineDescriptor};
use smallvec::SmallVec;

use crate::chunk_brick_manager::{ChunkBrickManager, ChunkBrickManagerBufferViews, ChunkId};
use crate::face_manager::{FaceId, FaceManager, FaceManagerBuffers, GpuFaceData};
use crate::material::{MaterialManager, Voxel};
use crate::{world_position_to_chunk_position, ChunkCoordinate, WorldPosition};

pub struct VoxelWorld
{
    // TODO: replace with dedicated components
    game:              Arc<game::Game>,
    uuid:              util::Uuid,
    pipeline:          Arc<gfx::GenericPipeline>,
    bind_group:        Mutex<Arc<wgpu::BindGroup>>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // rendering_dispatcher: (),
    face_manager:        FaceManager,
    chunk_brick_manager: ChunkBrickManager,
    material_manager:    MaterialManager,

    // TODO: there's a better way to do this...
    voxel_face_cache:   BTreeMap<WorldPosition, SmallVec<[FaceId; 6]>>,
    chunk_position_ids: HashMap<ChunkCoordinate, ChunkId>
}

impl Debug for VoxelWorld
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "VoxelWorld")
    }
}

impl VoxelWorld
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let renderer = game.get_renderer();

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Voxel Manager Bind Group"),
                    entries: &[
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
                            binding:    2,
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
                            binding:    3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
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
                });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("Voxel Manager Pipeline Layout"),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout.clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..4
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(include_wgsl!("simple_color_voxel_world.wgsl"));

        let voxel_face_manager = FaceManager::new(renderer.clone());
        let chunk_manager = ChunkBrickManager::new(renderer.clone());
        let voxel_material_manager = MaterialManager::new(&renderer);

        let combined_bind_group = Self::generate_bind_group(
            &renderer,
            &bind_group_layout,
            &voxel_face_manager,
            &chunk_manager,
            &voxel_material_manager
        );

        let this = Arc::new(VoxelWorld {
            game:                game.clone(),
            uuid:                util::Uuid::new(),
            pipeline:            renderer.render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label: Cow::Borrowed("Voxel Manager Pipeline"),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    vertex_specialization: None,
                    zero_initalize_vertex_workgroup_memory: false,
                    fragment_state: Some(gfx::CacheableFragmentState {
                        module:                           shader,
                        entry_point:                      "fs_main".into(),
                        targets:                          vec![Some(wgpu::ColorTargetState {
                            format:     gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                            blend:      Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL
                        })],
                        constants:                        None,
                        zero_initialize_workgroup_memory: false
                    }),
                    primitive_state: wgpu::PrimitiveState {
                        topology:           wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Cw,
                        // TODO: test disabling backface culling because you're doing it on the CPU
                        // side!
                        cull_mode:          Some(wgpu::Face::Back),
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
                    multiview: None
                }
            ),
            bind_group:          Mutex::new(combined_bind_group),
            bind_group_layout:   bind_group_layout.clone(),
            face_manager:        voxel_face_manager,
            chunk_brick_manager: chunk_manager,
            voxel_face_cache:    BTreeMap::new(),
            chunk_position_ids:  HashMap::new(),
            material_manager:    voxel_material_manager
        });

        renderer.register(this.clone());

        this
    }

    pub fn write_voxel(&self, pos: WorldPosition, voxel: Voxel) -> Voxel
    {
        // insert voxel into chunk, check adjacent points for other things, fix
        // up the faces as required

        todo!()
    }

    pub fn read_voxel(&self, pos: WorldPosition) -> Option<Voxel>
    {
        let (chunk_coord, chunk_local_pos) = world_position_to_chunk_position(pos);

        if let Some(chunk_id) = self.chunk_position_ids.get(&chunk_coord)
        {
            self.chunk_brick_manager
                .read_voxel(*chunk_id, chunk_local_pos)
        }
        else
        {
            Some(Voxel::Air)
        }
    }

    // TODO: get this to work, otherwise start the refactor of the thing that
    // already works

    fn generate_bind_group(
        renderer: &gfx::Renderer,
        bind_group_layout: &wgpu::BindGroupLayout,
        face_manager: &FaceManager,
        chunk_manager: &ChunkBrickManager,
        material_manager: &MaterialManager
    ) -> Arc<wgpu::BindGroup>
    {
        face_manager.access_buffers(
            |FaceManagerBuffers {
                 face_id_buffer,
                 face_data_buffer
             }| {
                chunk_manager.access_buffers(
                    |ChunkBrickManagerBufferViews {
                         chunk_meta_data_buffer,
                         chunk_brick_map_buffer,
                         visibility_brick_buffer,
                         material_brick_buffer
                     }| {
                        Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                            label:   Some("Voxel Manager Bind Group"),
                            layout:  bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding:  0,
                                    resource: wgpu::BindingResource::Buffer(
                                        face_id_buffer.as_entire_buffer_binding()
                                    )
                                },
                                wgpu::BindGroupEntry {
                                    binding:  1,
                                    resource: wgpu::BindingResource::Buffer(
                                        face_data_buffer.as_entire_buffer_binding()
                                    )
                                },
                                wgpu::BindGroupEntry {
                                    binding:  2,
                                    resource: wgpu::BindingResource::Buffer(
                                        chunk_meta_data_buffer.as_entire_buffer_binding()
                                    )
                                },
                                wgpu::BindGroupEntry {
                                    binding:  3,
                                    resource: wgpu::BindingResource::Buffer(
                                        material_brick_buffer.as_entire_buffer_binding()
                                    )
                                }
                            ]
                        }))
                    }
                )
            }
        )
    }
}

impl gfx::Recordable for VoxelWorld
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Voxel Manager")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let mut needs_resize = false;

        needs_resize |= self.face_manager.replicate_to_gpu();
        needs_resize |= self.chunk_brick_manager.replicate();

        let mut bind_group = self.bind_group.lock().unwrap();

        if needs_resize
        {
            *bind_group = Self::generate_bind_group(
                renderer,
                &self.bind_group_layout,
                &self.face_manager,
                &self.chunk_brick_manager,
                &self.material_manager
            );
        }

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(bind_group.clone()),
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
            unreachable!()
        };

        let elements = self.face_manager.get_number_of_faces_to_draw() * 6;

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw(0..elements as u32, 0..1);
    }
}

// read write fns
// write face, lookup what faces to remove, remove them write in new ones, send
// writes to chunks
