use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use gfx::{wgpu, CacheableComputePipelineDescriptor, CacheablePipelineLayoutDescriptor};

pub struct ComputeResetRecordable
{
    game:     Arc<game::Game>,
    uuid:     util::Uuid,
    pipeline: Arc<gfx::GenericPipeline>,

    face_and_brick_info_bind_group: Arc<wgpu::BindGroup>,
    raytrace_indirect_bind_group:   Arc<wgpu::BindGroup>
}

impl ComputeResetRecordable
{
    pub fn new(
        game: Arc<game::Game>,
        face_and_brick_info_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        face_and_brick_info_bind_group: Arc<wgpu::BindGroup>,
        raytrace_indirect_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        raytrace_indirect_bind_group: Arc<wgpu::BindGroup>
    ) -> Arc<Self>
    {
        let this = Arc::new(ComputeResetRecordable {
            game:                           game.clone(),
            uuid:                           util::Uuid::new(),
            pipeline:                       game
                .get_renderer()
                .render_cache
                .cache_compute_pipeline(CacheableComputePipelineDescriptor {
                    label:                            Cow::Borrowed(
                        "ComputeResetRecordable Pipeline"
                    ),
                    layout:                           Some(
                        game.get_renderer().render_cache.cache_pipeline_layout(
                            CacheablePipelineLayoutDescriptor {
                                label:                Cow::Borrowed(
                                    "ComputeResetRecordable Pipeline Layout"
                                ),
                                bind_group_layouts:   vec![
                                    face_and_brick_info_bind_group_layout,
                                    raytrace_indirect_bind_group_layout,
                                ],
                                push_constant_ranges: vec![]
                            }
                        )
                    ),
                    module:                           game
                        .get_renderer()
                        .render_cache
                        .cache_shader_module(wgpu::include_wgsl!("compute_reset.wgsl")),
                    entry_point:                      Cow::Borrowed("cs_main"),
                    specialization_constants:         None,
                    zero_initialize_workgroup_memory: false
                }),
            face_and_brick_info_bind_group: face_and_brick_info_bind_group,
            raytrace_indirect_bind_group:   raytrace_indirect_bind_group
        });

        game.get_renderer().register(this.clone());

        this
    }
}

impl Debug for ComputeResetRecordable
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "ComputeResetRecordable")
    }
}

impl gfx::Recordable for ComputeResetRecordable
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("ComputeResetRecordable")
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
                .get_renderpass_id(game::PassStage::VoxelVisibilityDetection),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(self.face_and_brick_info_bind_group.clone()),
                Some(self.raytrace_indirect_bind_group.clone()),
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

        let total_face_ids: u32 = 1048576;

        pass.dispatch_workgroups(total_face_ids.div_ceil(32), 1, 1);
    }
}
