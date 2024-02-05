use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, Pod, Zeroable};
use image::GenericImageView;
use wgpu::util::DeviceExt;
use {crate as gfx, nalgebra_glm as glm};

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
    id:                util::Uuid,
    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    tree_bind_group:   wgpu::BindGroup,
    time_alive:        Mutex<f32>,
    number_of_indices: u32,
    translation:       glm::Vec3
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

    pub fn new(
        renderer: &gfx::Renderer,
        translation: glm::Vec3,
        vertices: &[Vertex],
        indices: &[u16]
    ) -> Arc<Self>
    {
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

        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
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

        let tree_bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            layout:  renderer
                .render_cache
                .lookup_bind_group_layout(gfx::BindGroupType::FlatSimpleTexture),
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

        let this = Arc::new(Self {
            id: util::Uuid::new(),
            vertex_buffer,
            index_buffer,
            tree_bind_group,
            time_alive: Mutex::new(0.0),
            number_of_indices: indices.len() as u32,
            translation
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for FlatTextured
{
    fn get_uuid(&self) -> util::Uuid
    {
        self.id
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline_type(&self) -> gfx::PipelineType
    {
        gfx::PipelineType::FlatTextured
    }

    fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4]
    {
        [Some(&self.tree_bind_group), None, None, None]
    }

    fn should_render(&self) -> bool
    {
        true
    }

    fn record<'s>(
        &'s self,
        active_render_pass: &mut gfx::GenericPass<'s>,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera
    )
    {
        let time_alive = {
            let mut guard = self.time_alive.lock().unwrap();
            *guard += renderer.get_delta_time();
            *guard
        };

        let gfx::GenericPass::Render(ref mut pass) = active_render_pass
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        let mut transform = gfx::Transform {
            translation: self.translation, // 2.0 + 2.0 * time_alive.sin()
            rotation:    *nalgebra::UnitQuaternion::new_normalize(glm::quat(1.0, 0.0, 0.0, 0.0)),
            scale:       glm::Vec3::new(1.0, 1.0, 1.0)
        };

        transform.rotation *= *nalgebra::UnitQuaternion::from_axis_angle(
            &gfx::Transform::global_up_vector(),
            5.0 * time_alive
        );

        transform.rotation.normalize_mut();

        let matrix = camera.get_perspective(renderer, &transform);

        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&matrix));

        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}
