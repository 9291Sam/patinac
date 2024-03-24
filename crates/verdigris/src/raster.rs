use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use bytemuck::{cast_slice, Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};

#[derive(Debug)]
pub struct RasterizedVoxelChunk
{
    uuid:                util::Uuid,
    vertex_buffer:       wgpu::Buffer,
    index_buffer:        wgpu::Buffer,
    instance_buffer:     wgpu::Buffer,
    number_of_instances: u32,
    number_of_indices:   u32,
    pipeline:            Arc<gfx::GenericPipeline>,

    transform: Mutex<gfx::Transform>
}

impl RasterizedVoxelChunk
{
    pub fn new(game: &game::Game, transform: gfx::Transform) -> Arc<RasterizedVoxelChunk>
    {
        let uuid = util::Uuid::new();

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("raster.wgsl"));

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "RasterizedVoxelChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![renderer.global_bind_group_layout.clone()],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let pipeline = game.get_renderer().render_cache.cache_render_pipeline(
            CacheableRenderPipelineDescriptor {
                label:                 "RasterizedVoxelChunk Pipeline".into(),
                layout:                Some(pipeline_layout),
                vertex_module:         shader.clone(),
                vertex_entry_point:    "vs_main".into(),
                vertex_buffer_layouts: vec![
                    VoxelVertex::describe(),
                    RasterizedVoxelVertexOffsetPosition::describe(),
                ],
                fragment_state:        Some(CacheableFragmentState {
                    module:      shader,
                    entry_point: "fs_main".into(),
                    targets:     vec![Some(wgpu::ColorTargetState {
                        format:     gfx::Renderer::SURFACE_TEXTURE_FORMAT,
                        blend:      Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL
                    })]
                }),
                primitive_state:       wgpu::PrimitiveState {
                    topology:           wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face:         wgpu::FrontFace::Cw,
                    cull_mode:          None,
                    polygon_mode:       wgpu::PolygonMode::Fill,
                    unclipped_depth:    false,
                    conservative:       false
                },
                depth_stencil_state:   Some(gfx::Renderer::get_default_depth_state()),
                multisample_state:     wgpu::MultisampleState {
                    count:                     1,
                    mask:                      !0,
                    alpha_to_coverage_enabled: false
                },
                multiview:             None
            }
        );

        let instances: &[RasterizedVoxelVertexOffsetPosition] = &[
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(1, 3, 0, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(2, 3, 0, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(3, 3, 1, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(1, 2, 0, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(3, 3, 0, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(4, 3, 4, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(6, 3, 0, 0)
            },
            RasterizedVoxelVertexOffsetPosition {
                offset: glm::I16Vec4::new(0, 3, 1, 0)
            }
        ];

        let this = Arc::new(RasterizedVoxelChunk {
            uuid,
            vertex_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterizedVoxelChunk Vertex Buffer"),
                contents: cast_slice(&VOXEL_VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            index_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterizedVoxelChunk Index Buffer"),
                contents: cast_slice(&VOXEL_INDICES),
                usage:    wgpu::BufferUsages::INDEX
            }),
            instance_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterizedVoxelChunk Instance Buffer"),
                contents: cast_slice(&instances),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            number_of_instances: instances.len() as u32,
            number_of_indices: VOXEL_INDICES.len() as u32,
            pipeline,
            transform: Mutex::new(transform)
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for RasterizedVoxelChunk
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline(&self) -> Option<&gfx::GenericPipeline>
    {
        Some(&self.pipeline)
    }

    fn pre_record_update(
        &self,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera,
        global_bind_group: &std::sync::Arc<gfx::wgpu::BindGroup>
    ) -> gfx::RecordInfo
    {
        gfx::RecordInfo {
            should_draw: true,
            transform:   Some(self.transform.lock().unwrap().clone()),
            bind_groups: [Some(global_bind_group.clone()), None, None, None]
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
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..self.number_of_instances);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct RasterizedVoxelVertexOffsetPosition
{
    offset: glm::I16Vec4
}

impl RasterizedVoxelVertexOffsetPosition
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Sint16x4];

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRS
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct VoxelVertex
{
    position: glm::Vec3
}

impl VoxelVertex
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRS
        }
    }
}

const VOXEL_VERTICES: [VoxelVertex; 8] = [
    VoxelVertex {
        position: glm::Vec3::new(0.0, 0.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 0.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 1.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 1.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 0.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 0.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 1.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 1.0, 1.0)
    }
];

#[rustfmt::skip]
const VOXEL_INDICES: [u16; 36] = [
    6, 2, 7,
    2, 3, 7,
    0, 4, 5,
    1, 0, 5,
    0, 2, 6,
    4, 0, 6,
    3, 1, 7,
    1, 5, 7,
    2, 0, 3,
    0, 1, 3,
    4, 6, 7,
    5, 4, 7
];
