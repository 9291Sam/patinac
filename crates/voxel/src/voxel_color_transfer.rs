use std::borrow::Cow;
use std::sync::Arc;

use gfx::{
    wgpu,
    CacheableFragmentState,
    CacheableRenderPipelineDescriptor,
    GenericPipeline,
    Renderer
};

#[derive(Debug)]
pub(crate) struct VoxelColorTransferRecordable
{
    game:              Arc<game::Game>,
    uuid:              util::Uuid,
    pipeline:          Arc<GenericPipeline>,
    bind_group_window: util::Window<Arc<wgpu::BindGroup>>
}

impl VoxelColorTransferRecordable
{
    pub fn new(
        game: Arc<game::Game>,
        color_transfer_layout: Arc<wgpu::BindGroupLayout>,
        bind_group_window: util::Window<Arc<wgpu::BindGroup>>
    ) -> Arc<Self>
    {
        let arc_renderer = game.get_renderer().clone();
        let renderer: &gfx::Renderer = &arc_renderer;

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("Voxel Color Transfer Pipeline Layout"),
                    bind_group_layouts:   vec![color_transfer_layout],
                    push_constant_ranges: vec![]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("voxel_color_transfer.wgsl"));

        let this = Arc::new(VoxelColorTransferRecordable {
            game: game.clone(),
            uuid: util::Uuid::new(),
            pipeline: renderer.render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label: Cow::Borrowed(
                        "Voxel Color Transfer
                                    Pipeline"
                    ),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    vertex_specialization: None,
                    zero_initalize_vertex_workgroup_memory: false,
                    fragment_state: Some(CacheableFragmentState {
                        module:                           shader,
                        entry_point:                      "fs_main".into(),
                        targets:                          vec![Some(wgpu::ColorTargetState {
                            format:     Renderer::SURFACE_TEXTURE_FORMAT,
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
                    depth_stencil_state: None,
                    multisample_state: wgpu::MultisampleState {
                        count:                     1,
                        mask:                      !0,
                        alpha_to_coverage_enabled: false
                    },
                    multiview: None
                }
            ),
            bind_group_window
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for VoxelColorTransferRecordable
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        std::borrow::Cow::Borrowed("VoxelColorTransferRecordable")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    // fn get_pass_stage(&self) -> crate::PassStage
    // {
    //     crate::PassStage::VoxelColorTransfer
    // }

    // fn get_pipeline(&self) -> Option<&crate::GenericPipeline>
    // {
    //     Some(&self.pipeline)
    // }

    fn pre_record_update(
        &self,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        _: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::VoxelColorTransfer),
            pipeline:    self.pipeline.clone(),
            bind_groups: [Some(self.bind_group_window.get()), None, None, None],
            transform:   None
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), None) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.draw(0..3, 0..1);
    }
}
