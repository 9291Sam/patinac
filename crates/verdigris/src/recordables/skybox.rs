use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use gfx::{glm, wgpu};
use image::{GenericImageView, ImageBuffer, Rgba};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex
{
    pub position: glm::Vec4,
    pub color:    glm::Vec4,
    pub uv:       glm::Vec2
}

impl Vertex
{
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBS
        }
    }
}

#[derive(Debug)]
pub struct Skybox
{
    game:          Arc<game::Game>,
    id:            util::Uuid,
    vertex_buffer: wgpu::Buffer,

    skybox_bind_group: Arc<wgpu::BindGroup>,
    pipeline:          Arc<gfx::GenericPipeline>,

    time_alive:         Mutex<f32>,
    number_of_vertices: u32,
    transform:          gfx::Transform
}

impl Skybox
{
    pub const MASSIVE_CUBE_VERTICES: &'static [Vertex] = &[
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 1.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 0.0)
        },
        Vertex {
            position: glm::Vec4::new(1.0, -1.0, -1.0, 1.0),
            color:    glm::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv:       glm::Vec2::new(0.0, 1.0)
        },
        Vertex {
            position: glm::Vec4::new(-1.0, 1.0, -1.0, 1.0),
            color:    glm::Vec4::new(0.0, 1.0, 0.0, 1.0),
            uv:       glm::Vec2::new(1.0, 0.0)
        }
    ];

    pub fn new_skybox(game: Arc<game::Game>, transform: gfx::Transform) -> Arc<Self>
    {
        Self::new(
            game,
            transform,
            Self::MASSIVE_CUBE_VERTICES,
            [
                include_bytes!("res/skybox/posx.png"),
                include_bytes!("res/skybox/negx.png"),
                include_bytes!("res/skybox/posy.png"),
                include_bytes!("res/skybox/negy.png"),
                include_bytes!("res/skybox/posz.png"),
                include_bytes!("res/skybox/negz.png")
            ]
        )
    }

    pub fn new(
        game: Arc<game::Game>,
        transform: gfx::Transform,
        vertices: &[Vertex],
        textures: [&[u8]; 6]
    ) -> Arc<Self>
    {
        let renderer = game.get_renderer().clone();

        let vertex_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Skybox Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX
        });

        let images: [image::ImageBuffer<Rgba<u8>, Vec<u8>>; 6] =
            std::array::from_fn(|idx| image::load_from_memory(textures[idx]).unwrap().to_rgba8());

        let skybox_texture_size = wgpu::Extent3d {
            width:                 images[0].dimensions().0,
            height:                images[0].dimensions().1,
            depth_or_array_layers: 6
        };

        let skybox_texture = renderer.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Skybox Texture"),
            size:            skybox_texture_size,
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[]
        });

        for (idx, img) in images.iter().enumerate()
        {
            renderer.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture:   &skybox_texture,
                    mip_level: 0,
                    origin:    wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: idx as u32
                    },
                    aspect:    wgpu::TextureAspect::All
                },
                &img,
                wgpu::ImageDataLayout {
                    offset:         0,
                    bytes_per_row:  Some(
                        skybox_texture_size.width * std::mem::size_of::<Rgba<u8>>() as u32
                    ),
                    rows_per_image: Some(skybox_texture_size.height)
                },
                wgpu::Extent3d {
                    width:                 skybox_texture_size.width,
                    height:                skybox_texture_size.height,
                    depth_or_array_layers: 1
                }
            );
        }

        let skybox_view = skybox_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding:    0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty:         wgpu::BindingType::Texture {
                                multisampled:   false,
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                sample_type:    wgpu::TextureSampleType::Float {
                                    filterable: true
                                }
                            },
                            count:      None
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding:    1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty:         wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering
                            ),
                            count:      None
                        }
                    ],
                    label:   Some("texture_bind_group_layout")
                });

        let skybox_bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            layout:  &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&skybox_view)
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(
                        &renderer.create_sampler(&wgpu::SamplerDescriptor::default())
                    )
                }
            ],
            label:   Some("skybox_bind_group")
        });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                "Flat Textured Pipeline Layout".into(),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout,
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<glm::Mat4>() as u32)
                    }]
                });

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("res/skybox/skybox.wgsl"));

        let pipeline =
            renderer
                .render_cache
                .cache_render_pipeline(gfx::CacheableRenderPipelineDescriptor {
                    label: "Flat Textured Pipeline".into(),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![Vertex::desc()],
                    fragment_state: Some(gfx::CacheableFragmentState {
                        module:                           shader,
                        entry_point:                      "fs_main".into(),
                        targets:                          vec![Some(wgpu::ColorTargetState {
                            format:     gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                            blend:      Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL
                        })],
                        constants:                        None,
                        zero_initialize_workgroup_memory: false
                    }),
                    primitive_state: wgpu::PrimitiveState {
                        topology:           wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Ccw,
                        cull_mode:          None,
                        polygon_mode:       wgpu::PolygonMode::Fill,
                        unclipped_depth:    false,
                        conservative:       false
                    },
                    depth_stencil_state: Some(gfx::Renderer::get_default_depth_state()),
                    multisample_state: wgpu::MultisampleState {
                        count:                     1,
                        mask:                      !0,
                        alpha_to_coverage_enabled: false
                    },
                    multiview: None,
                    vertex_specialization: None,
                    zero_initialize_vertex_workgroup_memory: false,
                    fragment_specialization: None,
                    zero_initialize_fragment_workgroup_memory: false
                });

        let this = Arc::new(Self {
            id: util::Uuid::new(),
            vertex_buffer,
            skybox_bind_group: Arc::new(skybox_bind_group),
            time_alive: Mutex::new(0.0),
            transform,
            number_of_vertices: vertices.len() as u32,
            pipeline,
            game
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for Skybox
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Flat Textured Recordable")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        let time_alive = {
            let mut guard = self.time_alive.lock().unwrap();
            *guard += renderer.get_delta_time();
            *guard
        };

        let mut transform = self.transform.clone();

        transform.rotation *= *glm::UnitQuaternion::from_axis_angle(
            &gfx::Transform::global_right_vector(),
            0.01 * time_alive
        );
        transform.translation = camera.get_position();
        // transform.scale = glm::Vec3::repeat(1000000.0);

        transform.rotation.normalize_mut();

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.skybox_bind_group.clone()),
                None,
                None
            ],
            transform:   Some(transform)
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw(0..self.number_of_vertices, 0..1);
    }
}
