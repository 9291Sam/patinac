use std::borrow::Cow;
use std::sync::Arc;

use gfx::wgpu::{self, BindGroupLayoutDescriptor};
use gfx::{CacheablePipelineLayoutDescriptor, CacheableRenderPipelineDescriptor};

#[derive(Debug)]
struct InstancedIndirect
{
    game:     Arc<game::Game>,
    pipeline: Arc<gfx::GenericPipeline>,

    // add whatever you want
    uuid: util::Uuid
}

impl InstancedIndirect
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let renderer = game.get_renderer().clone();

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("InstancedIndirect PipelineLayout"),
                    bind_group_layouts:   vec![renderer.global_bind_group_layout.clone()],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..4
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(include_wgsl!("instanced_indirect.wgsl"));

        let pipeline =
            renderer
                .render_cache
                .cache_render_pipeline(CacheableRenderPipelineDescriptor {
                    label: Cow::Borrowed("InstancedIndirect Pipeline"),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![
                        wgpu::VertexBufferLayout { array_stride: todo!(), step_mode: todo!(), attributes: todo!() }
                    ],
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
                });

        let this = Arc::new(Self {
            game: game.clone(),
            pipeline,
            uuid: util::Uuid::new()
        });

        game.get_renderer().register(this.clone());

        this
    }
}

impl gfx::Recordable for InstancedIndirect
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Instanced Indirect")
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
            render_pass: self.game.get_renderpass_manager().get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [Some(global_bind_group.clone()), None, None, None],
            transform:   None
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>)
    {
        let gfx::GenericPass::Render(ref mut pass) = render_pass
        else
        {
            unreachable!()
        };

        pass.multi_draw_indirect(...);
    }
}
