use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use image::GenericImageView;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gfx;
pub struct Game<'r>
{
    renderer: &'r gfx::Renderer
}

impl<'r> Game<'r>
{
    pub fn new(renderer: &'r gfx::Renderer) -> Self
    {
        Game {
            renderer
        }
    }

    pub fn enter_tick_loop(&self, should_stop: &AtomicBool)
    {
        let all_u8s: [u8; u8::MAX as usize + 1] =
            std::array::from_fn(|i| u8::wrapping_add(255, i as u8));

        self.renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("wgpu test buffer"),
            contents: &all_u8s,
            usage:    wgpu::BufferUsages::STORAGE
        });

        // self.renderer.create_texture_with_data(
        //     &self.renderer.queue,
        //     &wgpu::TextureDescriptor {
        //         label:           todo!(),
        //         size:            todo!(),
        //         mip_level_count: todo!(),
        //         sample_count:    todo!(),
        //         dimension:       todo!(),
        //         format:          todo!(),
        //         usage:           todo!(),
        //         view_formats:    todo!()
        //     },
        //     wgpu::util::TextureDataOrder::MipMajor,
        //     &[]
        // );

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}

const VERTICES: &[gfx::Vertex] = &[
    gfx::Vertex {
        position:   cgmath::Vector3::new(-0.0868241, 0.49240386, 0.0),
        tex_coords: cgmath::Vector2::new(0.4131759, 0.99240386)
    }, // A
    gfx::Vertex {
        position:   cgmath::Vector3::new(-0.49513406, 0.06958647, 0.0),
        tex_coords: cgmath::Vector2::new(0.0048659444, 0.56958647)
    }, // B
    gfx::Vertex {
        position:   cgmath::Vector3::new(-0.21918549, -0.44939706, 0.0),
        tex_coords: cgmath::Vector2::new(0.28081453, 0.05060294)
    }, // C
    gfx::Vertex {
        position:   cgmath::Vector3::new(0.35966998, -0.3473291, 0.0),
        tex_coords: cgmath::Vector2::new(0.85967, 0.1526709)
    }, // D
    gfx::Vertex {
        position:   cgmath::Vector3::new(0.44147372, 0.2347359, 0.0),
        tex_coords: cgmath::Vector2::new(0.9414737, 0.7347359)
    } // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4, /* padding */ 0];

#[derive(Debug)]
struct PentagonalTreeRenderer
{
    vertex_buffer:   wgpu::Buffer,
    index_buffer:    wgpu::Buffer,
    tree_bind_group: wgpu::BindGroup
}

impl PentagonalTreeRenderer
{
    pub fn new(renderer: &gfx::Renderer) -> Arc<Self>
    {
        let vertex_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage:    wgpu::BufferUsages::VERTEX
        });

        let index_buffer = renderer.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
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
                .lookup_bind_group_layout(gfx::BindGroupType::TestSimpleTexture),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&tree_texture_view)
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&renderer.create_sampler(
                        &wgpu::SamplerDescriptor {
                            // address_mode_u: wgpu::AddressMode::ClampToEdge,
                            // address_mode_v: wgpu::AddressMode::ClampToEdge,
                            // address_mode_w: wgpu::AddressMode::ClampToEdge,
                            // mag_filter: wgpu::FilterMode::Linear,
                            // min_filter: wgpu::FilterMode::Nearest,
                            // mipmap_filter: wgpu::FilterMode::Nearest,
                            ..Default::default()
                        }
                    ))
                }
            ],
            label:   Some("tree_bind_group")
        });

        Arc::new(Self {
            vertex_buffer,
            index_buffer,
            tree_bind_group
        })
    }
}

impl gfx::Renderable for Arc<PentagonalTreeRenderer>
{
    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline_type(&self) -> gfx::PipelineType
    {
        gfx::PipelineType::TestSample
    }

    fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4]
    {
        [Some(&self.tree_bind_group), None, None, None]
    }

    fn should_render(&self) -> bool
    {
        true
    }

    fn bind_and_draw<'s>(&'s self, active_render_pass: &mut wgpu::RenderPass<'s>)
    {
        active_render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        active_render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        active_render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }
}
