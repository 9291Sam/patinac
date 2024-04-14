use std::borrow::Cow;
use std::sync::Arc;

use crate::{CacheableFragmentState, CacheableRenderPipelineDescriptor, GenericPipeline, Renderer};

#[derive(Debug)]
pub(crate) struct VoxelColorTransferRecordable
{
    uuid:     util::Uuid,
    pipeline: Arc<GenericPipeline>
}

impl VoxelColorTransferRecordable
{
    pub fn new(renderer: &super::Renderer) -> Self
    {
        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(crate::CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("Voxel Color Transfer Pipeline Layout"),
                    bind_group_layouts:   vec![renderer.global_discovery_layout.clone()],
                    push_constant_ranges: vec![]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("voxel_color_transfer.wgsl"));

        VoxelColorTransferRecordable {
            uuid:     util::Uuid::new(),
            pipeline: renderer.render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label:                 Cow::Borrowed("Voxel Color Transfer Pipeline"),
                    layout:                Some(pipeline_layout),
                    vertex_module:         shader.clone(),
                    vertex_entry_point:    "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    fragment_state:        Some(CacheableFragmentState {
                        module:      shader,
                        entry_point: "fs_main".into(),
                        targets:     vec![Some(wgpu::ColorTargetState {
                            format:     Renderer::SURFACE_TEXTURE_FORMAT,
                            blend:      Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL
                        })]
                    }),
                    primitive_state:       wgpu::PrimitiveState {
                        topology:           wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Cw,
                        cull_mode:          Some(wgpu::Face::Back),
                        polygon_mode:       wgpu::PolygonMode::Fill,
                        unclipped_depth:    false,
                        conservative:       false
                    },
                    depth_stencil_state:   None,
                    multisample_state:     wgpu::MultisampleState {
                        count:                     1,
                        mask:                      !0,
                        alpha_to_coverage_enabled: false
                    },
                    multiview:             None
                }
            )
        }
    }
}

impl super::Recordable for VoxelColorTransferRecordable
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        std::borrow::Cow::Borrowed("VoxelColorTransferRecordable")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> crate::PassStage
    {
        crate::PassStage::VoxelColorTransfer
    }

    fn get_pipeline(&self) -> Option<&crate::GenericPipeline>
    {
        Some(&self.pipeline)
    }

    fn pre_record_update(
        &self,
        _: &crate::Renderer,
        _: &crate::Camera,
        _: &Arc<wgpu::BindGroup>,
        global_voxel_discovery_group: &Arc<wgpu::BindGroup>
    ) -> crate::RecordInfo
    {
        crate::RecordInfo {
            should_draw: true,
            transform:   None,
            bind_groups: [Some(global_voxel_discovery_group.clone()), None, None, None]
        }
    }

    fn record<'s>(
        &'s self,
        render_pass: &mut crate::GenericPass<'s>,
        maybe_id: Option<crate::DrawId>
    )
    {
        let (crate::GenericPass::Render(ref mut pass), None) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.draw(0..3, 0..1);
    }
}
