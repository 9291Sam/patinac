use std::collections::HashMap;
use std::sync::Arc;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use gfx::{wgpu, GenericPass};
use strum::{EnumIter, IntoEnumIterator};

#[derive(Copy, Clone, EnumIter, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PassStage
{
    VoxelDiscovery,
    PostVoxelDiscoveryCompute,
    VoxelColorTransfer,
    SimpleColor,
    MenuRender
}

pub struct RenderPassManager
{
    stage_to_id_map: DashMap<PassStage, gfx::RenderPassId>,
    depth_buffer:    Arc<gfx::ScreenSizedTexture>,
    voxel_discovery: Arc<gfx::ScreenSizedTexture>
}

impl RenderPassManager
{
    fn generate_renderpass_vec(self: Arc<Self>) -> gfx::RenderPassSendFunction
    {
        PassStage::iter()
            .map(|s| {
                let this = self.clone();

                let a: Arc<dyn Fn() -> (gfx::RenderPassId, gfx::EncoderToPassFn) + Sync + Send> =
                    Arc::new(move || -> (gfx::RenderPassId, gfx::EncoderToPassFn) {
                        (
                            match this.stage_to_id_map.entry(s)
                            {
                                Entry::Occupied(i) => *i.get(),
                                Entry::Vacant(i) => *i.insert(gfx::RenderPassId(util::Uuid::new()))
                            },
                            this.into_pass_func(s)
                        )
                    });

                a
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn into_pass_func(&self, pass_type: PassStage) -> gfx::EncoderToPassFn2
    {
        let voxel_discovery_view = self.voxel_discovery.get_view();
        let depth_buffer_view = self.depth_buffer.get_view();

        Box::new(
            move |encoder, screen_view, with_pass_func: &dyn FnOnce(&mut GenericPass)| {
                let mut pass = match pass_type
                {
                    PassStage::VoxelDiscovery =>
                    {
                        GenericPass::Render(encoder.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label:                    Some("Voxel Discovery Pass"),
                                color_attachments:        &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view:           &voxel_discovery_view,
                                        resolve_target: None,
                                        ops:            wgpu::Operations {
                                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.0,
                                                g: 0.0,
                                                b: 0.0,
                                                a: 0.0
                                            }),
                                            store: wgpu::StoreOp::Store
                                        }
                                    }
                                )],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view:        &depth_buffer_view,
                                        depth_ops:   Some(wgpu::Operations {
                                            load:  wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store
                                        }),
                                        stencil_ops: None
                                    }
                                ),
                                occlusion_query_set:      None,
                                timestamp_writes:         None
                            }
                        ))
                    }
                    PassStage::PostVoxelDiscoveryCompute =>
                    {
                        GenericPass::Compute(encoder.begin_compute_pass(
                            &wgpu::ComputePassDescriptor {
                                label:            Some("Post Voxel Discovery Compute"),
                                timestamp_writes: None
                            }
                        ))
                    }
                    PassStage::VoxelColorTransfer =>
                    {
                        GenericPass::Render(encoder.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label:                    Some("Voxel Color Transfer Pass"),
                                color_attachments:        &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view:           screen_view,
                                        resolve_target: None,
                                        ops:            wgpu::Operations {
                                            load:  wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.1,
                                                g: 0.2,
                                                b: 0.3,
                                                a: 1.0
                                            }),
                                            store: wgpu::StoreOp::Store
                                        }
                                    }
                                )],
                                depth_stencil_attachment: None,
                                occlusion_query_set:      None,
                                timestamp_writes:         None
                            }
                        ))
                    }
                    PassStage::SimpleColor =>
                    {
                        GenericPass::Render(encoder.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label:                    Some("Simple Color Pass"),
                                color_attachments:        &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view:           screen_view,
                                        resolve_target: None,
                                        ops:            wgpu::Operations {
                                            load:  wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store
                                        }
                                    }
                                )],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view:        &depth_buffer_view,
                                        depth_ops:   Some(wgpu::Operations {
                                            load:  wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store
                                        }),
                                        stencil_ops: None
                                    }
                                ),
                                occlusion_query_set:      None,
                                timestamp_writes:         None
                            }
                        ))
                    }
                    PassStage::MenuRender =>
                    {
                        GenericPass::Render(encoder.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label:                    None,
                                color_attachments:        &[Some(
                                    wgpu::RenderPassColorAttachment {
                                        view:           screen_view,
                                        resolve_target: None,
                                        ops:            wgpu::Operations {
                                            load:  wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store
                                        }
                                    }
                                )],
                                depth_stencil_attachment: None,
                                timestamp_writes:         None,
                                occlusion_query_set:      None
                            }
                        ))
                    }
                };

                with_pass_func(&mut pass);
            }
        )
    }
}
