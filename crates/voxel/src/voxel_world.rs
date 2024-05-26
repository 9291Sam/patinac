use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::bytes_of;
use gfx::wgpu::include_wgsl;
use gfx::{wgpu, CacheablePipelineLayoutDescriptor, CacheableRenderPipelineDescriptor};

use crate::chunk_manager::ChunkManager;
use crate::face_manager::{
    FaceId,
    FaceManager,
    FaceManagerBuffers,
    GpuFaceData,
    VoxelFaceDirection
};
use crate::material::{MaterialManager, Voxel};
use crate::{get_chunk_position_from_world, WorldPosition};

#[no_mangle]
static NUMBER_OF_VISIBLE_FACES: AtomicUsize = AtomicUsize::new(0);

pub struct VoxelWorld
{
    game:              Arc<game::Game>,
    pipeline:          Arc<gfx::GenericPipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    bind_group:        Mutex<Arc<wgpu::BindGroup>>,

    uuid: util::Uuid,

    estimate_number_of_visible_faces: AtomicU32,

    face_manager:     Mutex<FaceManager>,
    chunk_manager:    Mutex<ChunkManager>,
    material_manager: MaterialManager,
    world_voxel_list: Mutex<BTreeMap<WorldPosition, RefCell<[Option<FaceId>; 6]>>>
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

        let face_manager = FaceManager::new(game.clone());

        let voxel_chunk_manager = ChunkManager::new(renderer.clone());

        let mat_manager = MaterialManager::new(&renderer);

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
            .cache_shader_module(include_wgsl!("voxel_world.wgsl"));

        let this = Arc::new(Self {
            game:                             game.clone(),
            pipeline:                         renderer.render_cache.cache_render_pipeline(
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
            bind_group_layout:                bind_group_layout.clone(),
            bind_group:                       Mutex::new(Self::generate_bind_group(
                &renderer,
                &bind_group_layout.clone(),
                &face_manager,
                &voxel_chunk_manager,
                &mat_manager
            )),
            chunk_manager:                    Mutex::new(voxel_chunk_manager),
            material_manager:                 mat_manager,
            uuid:                             util::Uuid::new(),
            face_manager:                     Mutex::new(face_manager),
            world_voxel_list:                 Mutex::new(BTreeMap::new()),
            estimate_number_of_visible_faces: AtomicU32::new(0)
        });

        renderer.register(this.clone());

        this
    }

    pub fn insert_many_voxel(&self, it: impl IntoIterator<Item = (WorldPosition, Voxel)>)
    {
        let mut world_voxels = self.world_voxel_list.lock().unwrap();
        let mut chunk_manager = self.chunk_manager.lock().unwrap();
        let mut face_manager = self.face_manager.lock().unwrap();

        for (world_pos, voxel) in it.into_iter()
        {
            self.estimate_number_of_visible_faces
                .store(face_manager.get_number_of_faces(), Ordering::Relaxed);

            let (chunk_coordinate, chunk_position) = get_chunk_position_from_world(world_pos);

            let chunk_id = chunk_manager.get_or_insert_chunk(chunk_coordinate);

            for d in VoxelFaceDirection::iterate()
            {
                let adjacent_voxel_world_position =
                    WorldPosition(world_pos.0 + d.get_axis().cast());
                let adjacent_voxel_face_direction = d.opposite();

                let was_cell_already_occupied = world_voxels
                    .try_insert(world_pos, RefCell::new([const { None }; 6]))
                    .is_err();

                let this_voxel_faces = world_voxels.get(&world_pos);
                let adjacent_voxel_faces = world_voxels.get(&adjacent_voxel_world_position);

                if let Some(this_faces_refcell) = this_voxel_faces
                {
                    match adjacent_voxel_faces
                    {
                        Some(other_faces) =>
                        {
                            if let Some(conflicting_face) = other_faces.borrow_mut()
                                [adjacent_voxel_face_direction as usize]
                                .take()
                            {
                                face_manager.remove_face(conflicting_face);
                                // and don't add this face
                            }
                            else
                            {
                                #[allow(clippy::collapsible_else_if)]
                                if !was_cell_already_occupied
                                {
                                    log::warn!(
                                        "there was a naked voxel! @ \
                                         {adjacent_voxel_world_position:?}"
                                    )
                                }
                            }
                        }
                        None =>
                        {
                            this_faces_refcell.borrow_mut()[d as usize] =
                                Some(face_manager.insert_face(GpuFaceData::new(
                                    voxel as u16,
                                    chunk_id.0 as u16,
                                    chunk_position.0,
                                    d
                                )));
                        }
                    }
                }
                else
                {
                    unreachable!()
                }
            }
        }
    }

    // TODO: remove voxel

    fn get_bind_group(&self) -> Arc<wgpu::BindGroup>
    {
        let mut needs_resize = false;

        let mut bind_group = self.bind_group.lock().unwrap();

        // TODO: fix this lol
        let mut chunk_manager = match self.chunk_manager.try_lock()
        {
            Ok(g) => g,
            Err(e) =>
            {
                match e
                {
                    std::sync::TryLockError::Poisoned(_) => panic!(),
                    std::sync::TryLockError::WouldBlock => return bind_group.clone()
                }
            }
        };

        let mut face_manager = match self.face_manager.try_lock()
        {
            Ok(g) => g,
            Err(e) =>
            {
                match e
                {
                    std::sync::TryLockError::Poisoned(_) => panic!(),
                    std::sync::TryLockError::WouldBlock => return bind_group.clone()
                }
            }
        };

        needs_resize |= face_manager.replicate_to_gpu();
        needs_resize |= chunk_manager.replicate_to_gpu();

        if needs_resize
        {
            *bind_group = Self::generate_bind_group(
                self.game.get_renderer(),
                &self.bind_group_layout,
                &face_manager,
                &chunk_manager,
                &self.material_manager
            );
        }

        bind_group.clone()
    }

    fn generate_bind_group(
        renderer: &gfx::Renderer,
        bind_group_layout: &wgpu::BindGroupLayout,
        face_manager: &FaceManager,
        chunk_manager: &ChunkManager,
        material_manager: &MaterialManager
    ) -> Arc<wgpu::BindGroup>
    {
        face_manager.access_buffers(
            |FaceManagerBuffers {
                 face_id_buffer,
                 face_data_buffer
             }| {
                chunk_manager.get_buffer(|raw_chunk_buf| {
                    let material_buffer = material_manager.get_material_buffer();

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
                                    raw_chunk_buf.as_entire_buffer_binding()
                                )
                            },
                            wgpu::BindGroupEntry {
                                binding:  3,
                                resource: wgpu::BindingResource::Buffer(
                                    material_buffer.as_entire_buffer_binding()
                                )
                            }
                        ]
                    }))
                })
            }
        )
    }
}

impl gfx::Recordable for VoxelWorld
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Voxel World")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.get_bind_group()),
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

        let face_manager_faces = match self.face_manager.try_lock()
        {
            Ok(g) => g.get_number_of_faces(),
            Err(e) =>
            {
                match e
                {
                    std::sync::TryLockError::Poisoned(_) => panic!(),
                    std::sync::TryLockError::WouldBlock =>
                    {
                        self.estimate_number_of_visible_faces
                            .load(Ordering::Relaxed)
                    }
                }
            }
        };

        NUMBER_OF_VISIBLE_FACES.store(face_manager_faces as usize, Ordering::Relaxed);

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw(0..(face_manager_faces * 6), 0..1);
    }
}
