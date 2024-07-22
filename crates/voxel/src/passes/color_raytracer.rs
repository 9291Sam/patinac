use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use gfx::{wgpu, CacheableComputePipelineDescriptor, CacheablePipelineLayoutDescriptor};

pub struct ColorRaytracerRecordable
{
    game:            Arc<game::Game>,
    uuid:            util::Uuid,
    pipeline:        Arc<gfx::GenericPipeline>,
    indirect_buffer: Arc<wgpu::Buffer>,

    face_and_brick_info_bind_group: Arc<wgpu::BindGroup>
}

impl ColorRaytracerRecordable
{
    pub fn new(
        game: Arc<game::Game>,
        face_and_brick_info_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        face_and_brick_info_bind_group: Arc<wgpu::BindGroup>,
        indirect_buffer: Arc<wgpu::Buffer>
    ) -> Arc<Self>
    {
        let this = Arc::new(ColorRaytracerRecordable {
            game: game.clone(),
            uuid: util::Uuid::new(),
            pipeline: game.get_renderer().render_cache.cache_compute_pipeline(
                CacheableComputePipelineDescriptor {
                    label:                            Cow::Borrowed(
                        "ColorRaytracerRecordable Pipeline"
                    ),
                    layout:                           Some(
                        game.get_renderer().render_cache.cache_pipeline_layout(
                            CacheablePipelineLayoutDescriptor {
                                label:                Cow::Borrowed(
                                    "ColorRaytracerRecordable Pipeline Layout"
                                ),
                                bind_group_layouts:   vec![
                                    game.get_renderer().global_bind_group_layout.clone(),
                                    face_and_brick_info_bind_group_layout,
                                ],
                                push_constant_ranges: vec![]
                            }
                        )
                    ),
                    module:                           game
                        .get_renderer()
                        .render_cache
                        .cache_shader_module(wgpu::include_wgsl!("color_raytracer.wgsl")),
                    entry_point:                      Cow::Borrowed("cs_main"),
                    specialization_constants:         None,
                    zero_initialize_workgroup_memory: false
                }
            ),
            face_and_brick_info_bind_group: face_and_brick_info_bind_group,
            indirect_buffer
        });

        game.get_renderer().register(this.clone());

        this
    }
}

impl Debug for ColorRaytracerRecordable
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "ColorRaytracerRecordable")
    }
}

impl gfx::Recordable for ColorRaytracerRecordable
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("ColorRaytracerRecordable")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        _: &mut wgpu::CommandEncoder,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::VoxelColorCalculation),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.face_and_brick_info_bind_group.clone()),
                None,
                None
            ],
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

        pass.dispatch_workgroups_indirect(&self.indirect_buffer, 0);
    }
}
