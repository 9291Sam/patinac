use std::borrow::Cow;
use std::sync::Arc;

use dashmap::DashMap;
use gfx::{wgpu, GenericPass};
use strum::{EnumIter, IntoEnumIterator};

#[derive(Copy, Clone, EnumIter, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PassStage
{
    VoxelDiscovery,
    VoxelVisibilityDetection,
    VoxelColorCalculation,
    VoxelColorTransfer,
    SimpleColor,
    MenuRender,
    CleanupCompute
}

pub struct RenderPassManager
{
    stage_to_id_map: DashMap<PassStage, gfx::RenderPassId>,
    depth_buffer:    Arc<gfx::ScreenSizedTexture>,
    voxel_discovery: Arc<gfx::ScreenSizedTexture>
}

impl RenderPassManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> RenderPassManager
    {
        RenderPassManager {
            stage_to_id_map: PassStage::iter()
                .map(|s| (s, gfx::RenderPassId(util::Uuid::new())))
                .collect::<DashMap<_, _>>(),
            depth_buffer:    gfx::ScreenSizedTexture::new(
                renderer.clone(),
                gfx::ScreenSizedTextureDescriptor {
                    label:           Cow::Borrowed("Depth Buffer"),
                    mip_level_count: 1,
                    sample_count:    1,
                    format:          gfx::Renderer::DEPTH_FORMAT,
                    usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                }
            ),
            voxel_discovery: gfx::ScreenSizedTexture::new(
                renderer.clone(),
                gfx::ScreenSizedTextureDescriptor {
                    label:           Cow::Borrowed("Voxel Discovery Image"),
                    mip_level_count: 1,
                    sample_count:    1,
                    format:          wgpu::TextureFormat::Rg32Uint,
                    usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                }
            )
        }
    }

    pub fn get_voxel_discovery_texture(&self) -> Arc<gfx::ScreenSizedTexture>
    {
        self.voxel_discovery.clone()
    }

    pub fn get_renderpass_id(&self, stage: PassStage) -> gfx::RenderPassId
    {
        *self.stage_to_id_map.get(&stage).unwrap()
    }

    pub(crate) fn generate_renderpass_vec(self: Arc<Self>) -> gfx::RenderPassSendFunction
    {
        PassStage::iter()
            .map(|s| {
                let this = self.clone();

                let a: Arc<dyn Fn() -> (gfx::RenderPassId, gfx::EncoderToPassFn) + Sync + Send> =
                    Arc::new(move || -> (gfx::RenderPassId, gfx::EncoderToPassFn) {
                        (
                            *this.stage_to_id_map.get(&s).unwrap(),
                            this.get_pass_func(s)
                        )
                    });

                a
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn get_pass_func(&self, pass_type: PassStage) -> gfx::EncoderToPassFn
    {
        let voxel_discovery_view = self.voxel_discovery.get_view();
        let depth_buffer_view = self.depth_buffer.get_view();

        Box::new(move |encoder, screen_view, with_pass_func| {
            let mut pass = match pass_type
            {
                PassStage::VoxelDiscovery =>
                {
                    GenericPass::Render(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label:                    Some("Voxel Discovery Pass"),
                        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
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
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view:        &depth_buffer_view,
                            depth_ops:   Some(wgpu::Operations {
                                load:  wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store
                            }),
                            stencil_ops: None
                        }),
                        occlusion_query_set:      None,
                        timestamp_writes:         None
                    }))
                }
                PassStage::VoxelVisibilityDetection =>
                {
                    GenericPass::Compute(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:            Some("Voxel Visibility Detection Compute"),
                        timestamp_writes: None
                    }))
                }
                PassStage::VoxelColorCalculation =>
                {
                    GenericPass::Compute(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:            Some("Voxel Color Calculation Compute"),
                        timestamp_writes: None
                    }))
                }
                PassStage::VoxelColorTransfer =>
                {
                    GenericPass::Render(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label:                    Some("Voxel Color Transfer Pass"),
                        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                            view:           screen_view,
                            resolve_target: None,
                            ops:            wgpu::Operations {
                                load:  wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0
                                }),
                                store: wgpu::StoreOp::Store
                            }
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set:      None,
                        timestamp_writes:         None
                    }))
                }
                PassStage::SimpleColor =>
                {
                    GenericPass::Render(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label:                    Some("Simple Color Pass"),
                        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                            view:           screen_view,
                            resolve_target: None,
                            ops:            wgpu::Operations {
                                load:  wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store
                            }
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view:        &depth_buffer_view,
                            depth_ops:   Some(wgpu::Operations {
                                load:  wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store
                            }),
                            stencil_ops: None
                        }),
                        occlusion_query_set:      None,
                        timestamp_writes:         None
                    }))
                }
                PassStage::MenuRender =>
                {
                    GenericPass::Render(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label:                    None,
                        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                            view:           screen_view,
                            resolve_target: None,
                            ops:            wgpu::Operations {
                                load:  wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store
                            }
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes:         None,
                        occlusion_query_set:      None
                    }))
                }
                PassStage::CleanupCompute =>
                {
                    GenericPass::Compute(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:            Some("CleanUp Compute"),
                        timestamp_writes: None
                    }))
                }
            };

            with_pass_func(&mut pass);
        })
    }
}
