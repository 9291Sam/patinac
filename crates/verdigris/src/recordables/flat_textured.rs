use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use gfx::{glm, wgpu};
use image::GenericImageView;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex
{
    pub position:   glm::Vec3,
    pub tex_coords: glm::Vec2
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
pub struct FlatTextured
{
    game:          Arc<game::Game>,
    id:            util::Uuid,
    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,

    tree_bind_group: Arc<wgpu::BindGroup>,
    pipeline:        Arc<gfx::GenericPipeline>,

    time_alive:        Mutex<f32>,
    number_of_indices: u32,
    transform:         gfx::Transform
}

impl FlatTextured
{
    pub const PENTAGON_INDICES: &'static [u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4, /* padding */ 0];
    pub const PENTAGON_VERTICES: &'static [Vertex] = &[
        Vertex {
            position:   glm::Vec3::new(-0.0868241, 0.49240386, 0.0),
            tex_coords: glm::Vec2::new(0.4131759, 0.99240386)
        },
        Vertex {
            position:   glm::Vec3::new(-0.49513406, 0.06958647, 0.0),
            tex_coords: glm::Vec2::new(0.0048659444, 0.56958647)
        },
        Vertex {
            position:   glm::Vec3::new(-0.21918549, -0.44939706, 0.0),
            tex_coords: glm::Vec2::new(0.28081453, 0.05060294)
        },
        Vertex {
            position:   glm::Vec3::new(0.35966998, -0.3473291, 0.0),
            tex_coords: glm::Vec2::new(0.85967, 0.1526709)
        },
        Vertex {
            position:   glm::Vec3::new(0.44147372, 0.2347359, 0.0),
            tex_coords: glm::Vec2::new(0.9414737, 0.7347359)
        }
    ];

    pub fn new_pentagon(game: Arc<game::Game>, transform: gfx::Transform) -> Arc<Self>
    {
        Self::new(
            game,
            transform,
            Self::PENTAGON_VERTICES,
            Self::PENTAGON_INDICES,
            include_bytes!("res/flat_textured/happy-tree.png")
        )
    }

    pub fn new(
        game: Arc<game::Game>,
        transform: gfx::Transform,
        vertices: &[Vertex],
        indices: &[u16],
        texture: &[u8]
    ) -> Arc<Self>
    {
        let renderer = game.get_renderer().clone();

        let vertex_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX
        });

        let index_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX
        });

        let diffuse_image = image::load_from_memory(texture).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();
        let dimensions = diffuse_image.dimensions();

        let tree_texture_size = wgpu::Extent3d {
            width:                 dimensions.0,
            height:                dimensions.1,
            depth_or_array_layers: 1
        };

        let tree_texture = renderer.create_texture_with_data(
            &renderer.queue,
            &wgpu::TextureDescriptor {
                label:           Some("tree texture"),
                size:            tree_texture_size,
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8UnormSrgb,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats:    &[]
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &diffuse_rgba
        );

        let tree_texture_view = tree_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                                view_dimension: wgpu::TextureViewDimension::D2,
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

        let tree_bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            layout:  &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&tree_texture_view)
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&renderer.create_sampler(
                        &wgpu::SamplerDescriptor {
                            ..Default::default()
                        }
                    ))
                }
            ],
            label:   Some("tree_bind_group")
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
            .cache_shader_module(wgpu::include_wgsl!("res/flat_textured/flat_textured.wgsl"));

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
            index_buffer,
            tree_bind_group: Arc::new(tree_bind_group),
            time_alive: Mutex::new(0.0),
            number_of_indices: indices.len() as u32,
            transform,
            pipeline,
            game
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for FlatTextured
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
        _: &mut wgpu::CommandEncoder,
        renderer: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>,
        _: &gfx::wgpu::SurfaceTexture
    ) -> gfx::RecordInfo
    {
        let time_alive = {
            let mut guard = self.time_alive.lock().unwrap();
            *guard += renderer.get_delta_time();
            *guard
        };

        let mut transform = self.transform.clone();

        transform.rotation *= *glm::UnitQuaternion::from_axis_angle(
            &gfx::Transform::global_up_vector(),
            5.0 * time_alive
        );

        transform.rotation.normalize_mut();

        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.tree_bind_group.clone()),
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
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}
