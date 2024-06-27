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
pub struct VoxelColorTransferRecordable
{
    game:                      Arc<game::Game>,
    uuid:                      util::Uuid,
    pipeline:                  Arc<GenericPipeline>,
    voxel_lighting_bind_group: Arc<wgpu::BindGroup>,

    discovery_image_layout: Arc<wgpu::BindGroupLayout>,
    discovery_bind_group:   util::JointWindow<Arc<wgpu::BindGroup>>,
    resize_pinger:          util::PingReceiver
}

impl VoxelColorTransferRecordable
{
    pub fn new(
        game: Arc<game::Game>,
        voxel_lighting_bind_group_layout: Arc<wgpu::BindGroupLayout>,
        voxel_lighting_bind_group: Arc<wgpu::BindGroup>
    ) -> Arc<Self>
    {
        let arc_renderer = game.get_renderer().clone();
        let renderer: &gfx::Renderer = &arc_renderer;

        let discovery_image_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some(
                        "VoxelColorTransferRecordable VoxelDiscoveryImage BindGroupLayout"
                    ),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:         wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false
                        },
                        count:      None
                    }]
                });

        let discovery_bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("VoxelColorTransferRecordable VoxelDiscoveryImage BindGroup"),
            layout:  &discovery_image_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: wgpu::BindingResource::TextureView(
                    &game
                        .get_renderpass_manager()
                        .get_voxel_discovery_texture()
                        .get_view()
                )
            }]
        });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                Cow::Borrowed("Voxel Color Transfer Pipeline Layout"),
                    bind_group_layouts:   vec![
                        discovery_image_layout.clone(),
                        voxel_lighting_bind_group_layout,
                    ],
                    push_constant_ranges: vec![]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("color_transferer.wgsl"));

        let this = Arc::new(VoxelColorTransferRecordable {
            game: game.clone(),
            uuid: util::Uuid::new(),
            pipeline: renderer.render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label: Cow::Borrowed("Voxel Color Transform Pipeline"),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![],
                    vertex_specialization: None,
                    fragment_specialization: None,
                    zero_initialize_fragment_workgroup_memory: false,
                    zero_initialize_vertex_workgroup_memory: false,
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
            voxel_lighting_bind_group,
            discovery_bind_group: util::JointWindow::new(Arc::new(discovery_bind_group)),
            discovery_image_layout,
            resize_pinger: game.get_renderer().get_resize_pinger()
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

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        _: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        if self.resize_pinger.recv_all()
        {
            self.discovery_bind_group.update(Arc::new(
                renderer.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:   Some("VoxelColorTransferRecordable VoxelDiscoveryImage BindGroup"),
                    layout:  &self.discovery_image_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::TextureView(
                            &self
                                .game
                                .get_renderpass_manager()
                                .get_voxel_discovery_texture()
                                .get_view()
                        )
                    }]
                })
            ));
        }

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::VoxelColorTransfer),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(self.discovery_bind_group.get()),
                Some(self.voxel_lighting_bind_group.clone()),
                None,
                None
            ],
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
