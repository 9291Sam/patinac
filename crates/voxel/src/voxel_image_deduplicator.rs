use std::borrow::Cow;
use std::sync::Arc;

use gfx::{wgpu, CacheableComputePipelineDescriptor, CacheablePipelineLayoutDescriptor};

#[derive(Debug)]
pub struct VoxelImageDeduplicator
{
    game: Arc<game::Game>,
    uuid: util::Uuid,

    pipeline:                  Arc<gfx::GenericPipeline>,
    voxel_lighting_bind_group: util::Window<Arc<wgpu::BindGroup>>
}

impl VoxelImageDeduplicator
{
    pub fn new(
        game: Arc<game::Game>,
        voxel_lighting_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        voxel_lighting_bind_group_window: util::Window<Arc<wgpu::BindGroup>>
    ) -> Arc<Self>
    {
        let this = Arc::new(VoxelImageDeduplicator {
            game:                      game.clone(),
            uuid:                      util::Uuid::new(),
            pipeline:                  game.get_renderer().render_cache.cache_compute_pipeline(
                CacheableComputePipelineDescriptor {
                    label:                            Cow::Borrowed(
                        "VoxelImageDeduplicator Compute Pipeline"
                    ),
                    layout:                           Some(
                        game.get_renderer().render_cache.cache_pipeline_layout(
                            CacheablePipelineLayoutDescriptor {
                                label:                Cow::Borrowed("Voxel Image"),
                                bind_group_layouts:   vec![voxel_lighting_bind_group_layout],
                                push_constant_ranges: vec![]
                            }
                        )
                    ),
                    module:                           game
                        .get_renderer()
                        .render_cache
                        .cache_shader_module(wgpu::include_wgsl!("voxel_image_deduplicator.wgsl")),
                    entry_point:                      Cow::Borrowed("cs_main"),
                    specialization_constants:         None,
                    zero_initialize_workgroup_memory: false
                }
            ),
            voxel_lighting_bind_group: voxel_lighting_bind_group_window
        });

        game.get_renderer().register(this.clone());

        this
    }
}

impl gfx::Recordable for VoxelImageDeduplicator
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Voxel Image Deduplicator")
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
                .get_renderpass_id(game::PassStage::PostVoxelDiscoveryCompute),
            pipeline:    self.pipeline.clone(),
            bind_groups: [Some(self.voxel_lighting_bind_group.get()), None, None, None],
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

        let size = self.game.get_renderer().get_framebuffer_size();

        pass.dispatch_workgroups(size.x.div_ceil(32), size.y.div_ceil(32), 1);
    }
}
