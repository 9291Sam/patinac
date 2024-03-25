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
pub struct RasterChunk
{
    renderer: Arc<gfx::Renderer>,
    uuid:     util::Uuid,

    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,

    voxel_positions: Vec<RasterChunkVoxelInstance>,
    instance_buffer: wgpu::Buffer,
    pipeline:        Arc<gfx::GenericPipeline>,

    transform: Mutex<gfx::Transform>
}

impl RasterChunk
{
    pub fn new(game: &game::Game, transform: gfx::Transform) -> Arc<RasterChunk>
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
                    label:                "RasterChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![renderer.global_bind_group_layout.clone()],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let pipeline = game.get_renderer().render_cache.cache_render_pipeline(
            CacheableRenderPipelineDescriptor {
                label:                 "RasterChunk Pipeline".into(),
                layout:                Some(pipeline_layout),
                vertex_module:         shader.clone(),
                vertex_entry_point:    "vs_main".into(),
                vertex_buffer_layouts: vec![
                    VoxelVertex::describe(),
                    RasterChunkVoxelInstance::describe(),
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
                    topology:           wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face:         wgpu::FrontFace::Cw,
                    cull_mode:          Some(wgpu::Face::Back),
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

        let instances: &[RasterChunkVoxelInstance] = &[];

        let this = Arc::new(RasterChunk {
            uuid,
            vertex_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterChunk Vertex Buffer"),
                contents: cast_slice(&VOXEL_STRIP_VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            index_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterChunk Index Buffer"),
                contents: cast_slice(&VOXEL_STRIP_INDICES),
                usage:    wgpu::BufferUsages::INDEX
            }),
            instance_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some("RasterChunk Instance Buffer"),
                contents: cast_slice(instances),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            voxel_positions: Vec::from_iter(instances.iter().cloned()),
            pipeline,
            transform: Mutex::new(transform),
            renderer: game.get_renderer().clone()
        });

        renderer.register(this.clone());

        this
    }

    pub fn update_voxels(&mut self, positions: impl IntoIterator<Item = RasterChunkVoxelInstance>)
    {
        let instances: Vec<RasterChunkVoxelInstance> = positions.into_iter().collect();

        self.instance_buffer = self.renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Raster Instance Buffer"),
            contents: bytemuck::cast_slice(&instances[..]),
            usage:    wgpu::BufferUsages::VERTEX
        });

        self.voxel_positions = instances;
    }
}

impl gfx::Recordable for RasterChunk
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
        _: &gfx::Renderer,
        _: &gfx::Camera,
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
        pass.draw_indexed(
            0..VOXEL_STRIP_INDICES.len() as u32,
            0,
            0..self.voxel_positions.len() as u32
        );
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct RasterChunkVoxelInstance
{
    data: u32
}

impl RasterChunkVoxelInstance
{
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Uint32];

    pub fn new(x: u32, y: u32, z: u32, voxel: u32) -> Self
    {
        let five_bit_mask: u32 = 0b1_1111;

        assert!(x <= five_bit_mask);
        assert!(y <= five_bit_mask);
        assert!(z <= five_bit_mask);

        let x_data = five_bit_mask & x;
        let y_data = (five_bit_mask & y) << 5;
        let z_data = (five_bit_mask & z) << 10;
        let v_data = (five_bit_mask & voxel) << 15;

        Self {
            data: x_data | y_data | z_data | v_data
        }
    }

    pub fn destructure(self) -> (u32, u32, u32, u32)
    {
        let five_bit_mask: u32 = 0b1_1111;

        let x = five_bit_mask & self.data;
        let y = five_bit_mask & (self.data >> 5);
        let z = five_bit_mask & (self.data >> 10);
        let v = five_bit_mask & (self.data >> 15);

        (x, y, z, v)
    }

    pub fn describe() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Instance,
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

const VOXEL_STRIP_VERTICES: [VoxelVertex; 8] = [
    VoxelVertex {
        position: glm::Vec3::new(1.0, 1.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 1.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 0.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 0.0, 0.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 1.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 1.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(0.0, 0.0, 1.0)
    },
    VoxelVertex {
        position: glm::Vec3::new(1.0, 0.0, 1.0)
    }
];

// const VOXEL_VERTICES: [VoxelVertex; 8] = [
//     VoxelVertex {
//         position: glm::Vec3::new(0.0, 0.0, 0.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(0.0, 0.0, 1.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(0.0, 1.0, 0.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(0.0, 1.0, 1.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(1.0, 0.0, 0.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(1.0, 0.0, 1.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(1.0, 1.0, 0.0)
//     },
//     VoxelVertex {
//         position: glm::Vec3::new(1.0, 1.0, 1.0)
//     }
// ];

// #[rustfmt::skip]
// const VOXEL_INDICES: [u16; 36] = [
//     6, 2, 7,
//     2, 3, 7,
//     0, 4, 5,
//     1, 0, 5,
//     0, 2, 6,
//     4, 0, 6,
//     3, 1, 7,
//     1, 5, 7,
//     2, 0, 3,
//     0, 1, 3,
//     4, 6, 7,
//     5, 4, 7
// ];

const VOXEL_STRIP_INDICES: [u16; 14] = [3, 2, 6, 7, 4, 2, 0, 3, 1, 6, 5, 4, 1, 0];
