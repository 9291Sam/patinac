use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use gfx::{wgpu, CacheableComputePipelineDescriptor, CacheablePipelineLayoutDescriptor};

pub struct ColorCalculator
{
    game: Arc<game::Game>,
    uuid: util::Uuid,

    pipeline:          Arc<gfx::GenericPipeline>,
    bind_group_window: util::Window<Arc<wgpu::BindGroup>>,
    indirect_buffer:   Arc<wgpu::Buffer>
}

impl ColorCalculator
{
    pub fn new(
        game: Arc<game::Game>,
        bind_group_layout: Arc<wgpu::BindGroupLayout>,
        bind_group: util::Window<Arc<wgpu::BindGroup>>,
        indirect_buffer: Arc<wgpu::Buffer>
    ) -> Arc<Self>
    {
        let this = Arc::new(ColorCalculator {
            game: game.clone(),
            uuid: util::Uuid::new(),
            pipeline: game.get_renderer().render_cache.cache_compute_pipeline(
                CacheableComputePipelineDescriptor {
                    label:                            Cow::Borrowed("ColorCalculator Pipeline"),
                    layout:                           Some(
                        game.get_renderer().render_cache.cache_pipeline_layout(
                            CacheablePipelineLayoutDescriptor {
                                label:                Cow::Borrowed(
                                    "ColorCalculator Pipeline Layout"
                                ),
                                bind_group_layouts:   vec![bind_group_layout],
                                push_constant_ranges: vec![]
                            }
                        )
                    ),
                    module:                           game
                        .get_renderer()
                        .render_cache
                        .cache_shader_module(wgpu::include_wgsl!("color_calculator.wgsl")),
                    entry_point:                      Cow::Borrowed("cs_main"),
                    specialization_constants:         None,
                    zero_initialize_workgroup_memory: false
                }
            ),
            bind_group_window: bind_group,
            indirect_buffer
        });

        game.get_renderer().register(this.clone());

        this
    }
}

impl Debug for ColorCalculator
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "ColorCalculator")
    }
}

impl gfx::Recordable for ColorCalculator
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("ColorCalculator")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        _: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::VoxelColorCalculation),
            pipeline:    self.pipeline.clone(),
            bind_groups: [Some(self.bind_group_window.get()), None, None, None],
            transform:   None
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Compute(ref mut pass), None) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        // pass.dispatch_workgroups_indirect(&self.indirect_buffer, 0);
    }
}
