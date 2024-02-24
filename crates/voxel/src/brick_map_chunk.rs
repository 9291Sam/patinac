use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use game::{Entity, Positionable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{
    glm,
    CacheableFragmentState,
    CacheablePipelineLayoutDescriptor,
    CacheableRenderPipelineDescriptor
};

use crate::gpu_data::{BrickMap, VoxelChunkDataManager};

#[derive(Debug)]
pub struct BrickMapChunk
{
    uuid:     util::Uuid,
    name:     String,
    position: Mutex<glm::Vec3>,

    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    number_of_indices: u32,

    voxel_chunk_data: VoxelChunkDataManager,
    voxel_bind_group: wgpu::BindGroup,
    pipeline:         Arc<gfx::GenericPipeline>
}

impl BrickMapChunk
{
    pub fn new(game: &game::Game, center_position: glm::Vec3) -> Arc<Self>
    {
        let uuid = util::Uuid::new();

        let vertex_buffer_label = format!("BrickMapChunk {} Vertex Buffer", uuid);
        let index_buffer_label = format!("BrickMapChunk {} Index Buffer", uuid);

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("brick_map_chunk.wgsl"));

        static MIN_0_BINDING_SIZE: Option<NonZeroU64> =
            unsafe { Some(NonZeroU64::new_unchecked(2 * 1024 * 1024)) };

        static BINDINGS: &[wgpu::BindGroupLayoutEntry] = &[
            wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage {
                        read_only: true
                    },
                    has_dynamic_offset: false,
                    min_binding_size:   MIN_0_BINDING_SIZE
                },
                count:      None
            },
            wgpu::BindGroupLayoutEntry {
                binding:    1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage {
                        read_only: true
                    },
                    has_dynamic_offset: false,
                    min_binding_size:   MIN_0_BINDING_SIZE
                },
                count:      None
            }
        ];

        let voxel_bind_group_layout =
            renderer
                .render_cache
                .cache_bind_group_layout(wgpu::BindGroupLayoutDescriptor {
                    label:   Some("Brick Map Chunk BinGroup Layouts"),
                    entries: BINDINGS
                });

        let pipeline_layout =
            renderer
                .render_cache
                .cache_pipeline_layout(CacheablePipelineLayoutDescriptor {
                    label:                "Voxel BrickMapChunk Pipeline Layout".into(),
                    bind_group_layouts:   vec![
                        renderer.global_bind_group_layout.clone(),
                        voxel_bind_group_layout.clone(),
                    ],
                    push_constant_ranges: vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range:  0..(std::mem::size_of::<u32>() as u32)
                    }]
                });

        let voxel_data_manager = VoxelChunkDataManager::new(game.get_renderer().clone());
        let voxel_bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Voxel Bind Group"),
            layout:  &voxel_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: voxel_data_manager.gpu_brick_map.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: voxel_data_manager.gpu_brick_buffer.as_entire_binding()
                }
            ]
        });

        let this = Arc::new(Self {
            voxel_chunk_data: voxel_data_manager,
            voxel_bind_group,
            uuid,
            name: "Voxel BrickMapChunk".into(),
            position: Mutex::new(center_position),
            vertex_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some(&vertex_buffer_label),
                contents: cast_slice(&CUBE_VERTICES),
                usage:    wgpu::BufferUsages::VERTEX
            }),
            index_buffer: renderer.create_buffer_init(&BufferInitDescriptor {
                label:    Some(&index_buffer_label),
                contents: cast_slice(&CUBE_INDICES),
                usage:    wgpu::BufferUsages::INDEX
            }),
            number_of_indices: CUBE_INDICES.len() as u32,
            pipeline: game.get_renderer().render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label:                 "Voxel BrickMapChunk Pipeline".into(),
                    layout:                Some(pipeline_layout),
                    vertex_module:         shader.clone(),
                    vertex_entry_point:    "vs_main".into(),
                    vertex_buffer_layouts: vec![Vertex::describe_layout()],
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
            )
        });

        game.register(this.clone());
        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for BrickMapChunk
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed(&self.name)
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline(&self) -> &gfx::GenericPipeline
    {
        &self.pipeline
    }

    fn pre_record_update(&self, _: &gfx::Renderer, _: &gfx::Camera) -> gfx::RecordInfo
    {
        gfx::RecordInfo {
            should_draw: true,
            transform:   Some(gfx::Transform {
                translation: *self.position.lock().unwrap(),
                ..Default::default()
            })
        }
    }

    fn get_bind_groups<'s>(
        &'s self,
        global_bind_group: &'s gfx::wgpu::BindGroup
    ) -> [Option<&'s gfx::wgpu::BindGroup>; 4]
    {
        [
            Some(global_bind_group),
            Some(&self.voxel_bind_group),
            None,
            None
        ]
    }

    fn record<'s>(&'s self, render_pass: &mut gfx::GenericPass<'s>, maybe_id: Option<gfx::DrawId>)
    {
        let (gfx::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            unreachable!()
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytemuck::bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}

impl game::EntityCastDepot for BrickMapChunk
{
    fn as_entity(&self) -> Option<&dyn Entity>
    {
        Some(self)
    }

    fn as_positionable(&self) -> Option<&dyn Positionable>
    {
        Some(self)
    }

    fn as_transformable(&self) -> Option<&dyn game::Transformable>
    {
        None
    }
}

impl game::Entity for BrickMapChunk
{
    fn get_name(&self) -> Cow<'_, str>
    {
        gfx::Recordable::get_name(self)
    }

    fn get_uuid(&self) -> util::Uuid
    {
        gfx::Recordable::get_uuid(self)
    }

    fn tick(&self, _: &game::Game, _: game::TickTag) {}
}

impl game::Positionable for BrickMapChunk
{
    fn get_position(&self) -> glm::Vec3
    {
        *self.position.lock().unwrap()
    }

    fn get_position_mut(&self, func: &dyn Fn(&mut glm::Vec3))
    {
        func(&mut self.position.lock().unwrap())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Vertex
{
    position: glm::Vec3
}

impl Vertex
{
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    pub fn describe_layout() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBUTES
        }
    }
}

const CUBE_VERTICES: [Vertex; 8] = [
    Vertex {
        position: glm::Vec3::new(-512.0, -512.0, -512.0)
    },
    Vertex {
        position: glm::Vec3::new(-512.0, -512.0, 512.0)
    },
    Vertex {
        position: glm::Vec3::new(-512.0, 512.0, -512.0)
    },
    Vertex {
        position: glm::Vec3::new(-512.0, 512.0, 512.0)
    },
    Vertex {
        position: glm::Vec3::new(512.0, -512.0, -512.0)
    },
    Vertex {
        position: glm::Vec3::new(512.0, -512.0, 512.0)
    },
    Vertex {
        position: glm::Vec3::new(512.0, 512.0, -512.0)
    },
    Vertex {
        position: glm::Vec3::new(512.0, 512.0, 512.0)
    }
];

#[rustfmt::skip]
const CUBE_INDICES: [u16; 36] = [
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
