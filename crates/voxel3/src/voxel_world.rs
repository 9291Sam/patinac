use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use bytemuck::bytes_of;
use gfx::wgpu::include_wgsl;
use gfx::{wgpu, CacheablePipelineLayoutDescriptor, CacheableRenderPipelineDescriptor};

use crate::face_manager::{FaceManager, VoxelFace, VoxelFaceDirection};
use crate::material::Voxel;
use crate::WorldPosition;
pub struct VoxelWorld
{
    game:     Arc<game::Game>,
    pipeline: Arc<gfx::GenericPipeline>,

    uuid: util::Uuid,

    face_manager: FaceManager
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
    // insert remove voxels

    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let renderer = game.get_renderer();

        let (face_manager, bind_group_layout) = FaceManager::new(game.clone());

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
            game: game.clone(),
            pipeline: renderer.render_cache.cache_render_pipeline(
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
            uuid: util::Uuid::new(),
            face_manager
        });

        renderer.register(this.clone());

        this
    }

    pub fn insert_voxel(&self, world_pos: WorldPosition, voxel: Voxel)
    {
        for d in VoxelFaceDirection::iterate()
        {
            self.face_manager.insert_face(VoxelFace {
                direction: d,
                position:  world_pos,
                material:  voxel as u16
            })
        }
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
        renderer: &gfx::Renderer,
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
                Some(self.face_manager.get_bind_group(renderer)),
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

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw(0..(self.face_manager.get_number_of_faces() * 6), 0..1);
    }
}
