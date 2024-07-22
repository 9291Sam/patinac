use std::borrow::Cow;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use gfx::{glm, wgpu, CacheableFragmentState};
use image::GenericImageView;
use wgpu::util::DeviceExt;
#[derive(Debug)]
pub struct LitTextured
{
    game:          Arc<game::Game>,
    pub transform: Mutex<gfx::Transform>,
    id:            util::Uuid,

    vertex_buffer:             wgpu::Buffer,
    index_buffer:              wgpu::Buffer,
    texture_normal_bind_group: Arc<wgpu::BindGroup>,
    pipeline:                  Arc<gfx::GenericPipeline>,
    number_of_indices:         u32
}

impl LitTextured
{
    pub fn new_cube(game: Arc<game::Game>, transform: gfx::Transform) -> Arc<Self>
    {
        let renderer = game.get_renderer().clone();

        let obj_data = include_bytes!("res/lit_textured/cube.obj");
        let mut obj_cursor = Cursor::new(obj_data);

        let (model, material) = tobj::load_obj_buf(
            &mut obj_cursor,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |p| {
                match p.file_name().unwrap().to_str().unwrap()
                {
                    "cube.mtl" =>
                    {
                        let mtl_data = include_bytes!("res/lit_textured/cube.mtl");
                        let mut mtl_cursor = Cursor::new(mtl_data);

                        tobj::load_mtl_buf(&mut mtl_cursor)
                    }
                    _ => unreachable!()
                }
            }
        )
        .map(|err| (err.0, err.1.unwrap()))
        .unwrap();

        assert_eq!((1, 1), (model.len(), material.len()));

        let tobj::Material {
            diffuse_texture,
            normal_texture,
            ..
        } = &material[0];

        assert_eq!(diffuse_texture.as_ref().unwrap(), "cube-diffuse.jpg");
        assert_eq!(normal_texture.as_ref().unwrap(), "cube-normal.png");

        let raw_diffuse =
            image::load_from_memory(include_bytes!("res/lit_textured/cube-diffuse.jpg")).unwrap();

        let diffuse_texture = renderer.create_texture_with_data(
            &renderer.queue,
            &wgpu::TextureDescriptor {
                label:           Some("cube-diffuse"),
                size:            wgpu::Extent3d {
                    width:                 raw_diffuse.dimensions().0,
                    height:                raw_diffuse.dimensions().1,
                    depth_or_array_layers: 1
                },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8UnormSrgb,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats:    &[]
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &raw_diffuse.to_rgba8()
        );

        let raw_normal =
            image::load_from_memory(include_bytes!("res/lit_textured/cube-normal.png")).unwrap();

        let normal_texture = renderer.create_texture_with_data(
            &renderer.queue,
            &wgpu::TextureDescriptor {
                label:           Some("cube-normal"),
                size:            wgpu::Extent3d {
                    width:                 raw_normal.dimensions().0,
                    height:                raw_normal.dimensions().1,
                    depth_or_array_layers: 1
                },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format:          wgpu::TextureFormat::Rgba8Snorm,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats:    &[]
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &raw_normal.to_rgba8()
        );

        let tobj::Mesh {
            positions,
            normals,
            texcoords,
            indices,
            ..
        } = &model[0].mesh;

        let vertices = (0..positions.len() / 3)
            .map(|i| {
                Vertex {
                    position:   glm::Vec3::new(
                        positions[i * 3],
                        positions[i * 3 + 1],
                        positions[i * 3 + 2]
                    ),
                    tex_coords: glm::Vec2::new(texcoords[i * 2], 1.0 - texcoords[i * 2 + 1]),
                    normal:     glm::Vec3::new(
                        normals[i * 3],
                        normals[i * 3 + 1],
                        normals[i * 3 + 2]
                    )
                }
            })
            .collect::<Vec<_>>();

        Self::new(
            game,
            diffuse_texture,
            normal_texture,
            &vertices,
            indices,
            transform
        )
    }

    pub fn new(
        game: Arc<game::Game>,
        diffuse_texture: wgpu::Texture,
        normal_texture: wgpu::Texture,
        vertices: &[Vertex],
        indices: &[u32],
        transform: gfx::Transform
    ) -> Arc<Self>
    {
        let renderer = game.get_renderer().clone();

        let vertex_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vertex buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX
        });
        let index_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("index buffer"),
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX
        });

        let bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Lit Textured BindGroup Layout"),
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
                            binding:    2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty:         wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering
                            ),
                            count:      None
                        }
                    ]
                });

        let bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Lit Textured Bind Group"),
            layout:  &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(
                        &diffuse_texture.create_view(&Default::default())
                    )
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(
                        &normal_texture.create_view(&Default::default())
                    )
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(
                        &renderer.create_sampler(&Default::default())
                    )
                }
            ]
        });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(gfx::CacheablePipelineLayoutDescriptor {
                    label:                "Lit Textured Pipeline Layout".into(),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        bind_group_layout,
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        // TODO: bad!!!!
        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("res/lit_textured/lit_textured.wgsl"));

        let pipeline =
            renderer
                .render_cache
                .cache_render_pipeline(gfx::CacheableRenderPipelineDescriptor {
                    label: "Lit Textured Pipeline".into(),
                    layout: Some(pipeline_layout),
                    vertex_module: shader.clone(),
                    vertex_entry_point: "vs_main".into(),
                    vertex_buffer_layouts: vec![Vertex::desc()],
                    vertex_specialization: None,
                    fragment_specialization: None,
                    zero_initialize_vertex_workgroup_memory: false,
                    zero_initialize_fragment_workgroup_memory: false,
                    fragment_state: Some(CacheableFragmentState {
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
                        topology:           wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face:         wgpu::FrontFace::Cw,
                        cull_mode:          Some(wgpu::Face::Back),
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
                    multiview: None
                });

        let this = Arc::new(Self {
            id: util::Uuid::new(),
            vertex_buffer,
            index_buffer,
            texture_normal_bind_group: Arc::new(bind_group),
            number_of_indices: indices.len() as u32,
            transform: Mutex::new(transform),
            pipeline,
            game
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for LitTextured
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("LitTextured")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn pre_record_update(
        &self,
        _: &mut wgpu::CommandEncoder,
        _: &gfx::Renderer,
        _: &gfx::Camera,
        global_bind_group: &Arc<wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo::Record {
            render_pass: self
                .game
                .get_renderpass_manager()
                .get_renderpass_id(game::PassStage::SimpleColor),
            pipeline:    self.pipeline.clone(),
            bind_groups: [
                Some(global_bind_group.clone()),
                Some(self.texture_normal_bind_group.clone()),
                None,
                None
            ],
            transform:   Some(self.transform.lock().unwrap().clone())
        }
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex
{
    pub position:   glm::Vec3,
    pub tex_coords: glm::Vec2,
    pub normal:     glm::Vec3
}

impl Vertex
{
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBUTES
        }
    }
}
