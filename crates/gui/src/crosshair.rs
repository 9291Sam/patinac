use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use gfx::{wgpu, CacheableComputePipelineDescriptor, CacheablePipelineLayoutDescriptor};

#[derive(Debug)]
pub struct Crosshair
{
    game: Arc<game::Game>,
    uuid: util::Uuid,

    screen_texture_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pipeline:                  Arc<gfx::GenericPipeline>,
    screen_texture_bind_group: Mutex<Option<(Arc<wgpu::BindGroup>, wgpu::Id<wgpu::Texture>)>>
}

impl Crosshair
{
    pub fn new(game: Arc<game::Game>) -> Arc<Self>
    {
        let renderer = game.get_renderer();

        let screen_texture_bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Crosshair BindGroupLayout"),
                    entries: &const {
                        [wgpu::BindGroupLayoutEntry {
                            binding:    0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty:         wgpu::BindingType::StorageTexture {
                                access:         wgpu::StorageTextureAccess::ReadWrite,
                                format:         gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                                view_dimension: wgpu::TextureViewDimension::D2
                            },
                            count:      None
                        }]
                    }
                });

        let this = Arc::new(Self {
            game:                             game.clone(),
            uuid:                             util::Uuid::new(),
            screen_texture_bind_group_layout: screen_texture_bind_group_layout.clone(),
            pipeline:                         renderer.render_cache.cache_compute_pipeline(
                CacheableComputePipelineDescriptor {
                    label:                            Cow::Borrowed("Crosshair Pipeline"),
                    layout:                           Some(
                        renderer.render_cache.cache_pipeline_layout(
                            CacheablePipelineLayoutDescriptor {
                                label:                Cow::Borrowed("Crosshair PipelineLayout"),
                                bind_group_layouts:   vec![
                                    screen_texture_bind_group_layout.clone(),
                                ],
                                push_constant_ranges: vec![]
                            }
                        )
                    ),
                    module:                           renderer
                        .render_cache
                        .cache_shader_module(wgpu::include_wgsl!("crosshair.wgsl")),
                    entry_point:                      Cow::Borrowed("cs_main"),
                    specialization_constants:         None,
                    zero_initialize_workgroup_memory: false
                }
            ),
            screen_texture_bind_group:        Mutex::new(None)
        });

        renderer.register(this.clone());

        this
    }

    fn create_screen_bind_group(
        layout: &wgpu::BindGroupLayout,
        renderer: &gfx::Renderer,
        texture: &wgpu::Texture
    ) -> Arc<wgpu::BindGroup>
    {
        Arc::new(renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Crosshair Screen Descriptor"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: wgpu::BindingResource::TextureView(&texture.create_view(
                    &wgpu::TextureViewDescriptor {
                        format: Some(wgpu::TextureFormat::R32Uint),
                        ..Default::default()
                    }
                ))
            }]
        }))
    }
}

impl gfx::Recordable for Crosshair
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        std::borrow::Cow::Borrowed("Crosshair")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn pre_record_update(
        &self,
        _: &mut gfx::wgpu::CommandEncoder,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        _: &std::sync::Arc<gfx::wgpu::BindGroup>,
        screen_texture: &wgpu::SurfaceTexture
    ) -> gfx::RecordInfo
    {
        let mut guard = self.screen_texture_bind_group.lock().unwrap();

        if let Some((bind_group, current_texture)) = &mut *guard
        {
            *bind_group = Self::create_screen_bind_group(
                &self.screen_texture_bind_group_layout,
                renderer,
                &screen_texture.texture
            );
            *current_texture = screen_texture.texture.global_id();
        }
        else
        {
            *guard = Some((
                Self::create_screen_bind_group(
                    &self.screen_texture_bind_group_layout,
                    renderer,
                    &screen_texture.texture
                ),
                screen_texture.texture.global_id()
            ));
        }

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::CleanupCompute),
            pipeline:    self.pipeline.clone(),
            bind_groups: [Some(guard.as_ref().unwrap().0.clone()), None, None, None],
            transform:   None
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, _: Option<gfx::DrawId>)
    {
        let gfx::GenericPass::Compute(ref mut pass) = render_pass
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.dispatch_workgroups(1, 1, 1);
    }
}
