use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};
use gfx::{glm, CacheableRenderPipelineDescriptor};

#[derive(Debug)]
struct Chunk
{
    uuid:     util::Uuid,
    position: glm::Vec3,
    name:     String,

    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    number_of_indices: u32,

    pipeline: Arc<gfx::GenericPipeline>,

    brick_buffer:     wgpu::Buffer,
    voxel_bind_group: wgpu::BindGroup
}

impl Chunk
{
    pub fn new(game: &game::Game) -> Self
    {
        let uuid = util::Uuid::new();

        let vertex_buffer_label = format!("Chunk {} Vertex Buffer", uuid);
        let index_buffer_label = format!("Chunk {} Index Buffer", uuid);

        let renderer = &**game.get_renderer();

        let shader = renderer
            .render_cache
            .cache_shader_module(wgpu::include_wgsl!("voxel_shader.wgsl"));

        Self {
            uuid:              uuid,
            position:          glm::Vec3::repeat(0.0),
            name:              "Voxel Chunk".into(),
            vertex_buffer:     game
                .get_renderer()
                .create_buffer_init(&BufferInitDescriptor {
                    label:    Some(&vertex_buffer_label),
                    contents: cast_slice(&CUBE_VERTICES),
                    usage:    wgpu::BufferUsages::VERTEX
                }),
            index_buffer:      game
                .get_renderer()
                .create_buffer_init(&BufferInitDescriptor {
                    label:    Some(&index_buffer_label),
                    contents: cast_slice(&CUBE_INDICES),
                    usage:    wgpu::BufferUsages::INDEX
                }),
            number_of_indices: CUBE_INDICES.len() as u32,
            pipeline:          game.get_renderer().render_cache.cache_render_pipeline(
                CacheableRenderPipelineDescriptor {
                    label:                 todo!(),
                    layout:                todo!(),
                    vertex_module:         todo!(),
                    vertex_entry_point:    todo!(),
                    vertex_buffer_layouts: todo!(),
                    fragment_state:        todo!(),
                    primitive_state:       todo!(),
                    depth_stencil_state:   todo!(),
                    multisample_state:     todo!(),
                    multiview:             todo!()
                }
            ),
            brick_buffer:      todo!(),
            voxel_bind_group:  todo!()
        }
    }
}

impl gfx::Recordable for Chunk
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
                translation: self.position,
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
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}

impl game::Entity for Chunk
{
    fn get_name(&self) -> Cow<'_, str>
    {
        gfx::Recordable::get_name(self)
    }

    fn get_uuid(&self) -> util::Uuid
    {
        gfx::Recordable::get_uuid(self)
    }

    fn get_position(&self) -> Option<glm::Vec3>
    {
        Some(self.position)
    }

    fn tick(&self, game: &game::Game, _: game::TickTag)
    {
        todo!()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Vertex
{
    position: glm::Vec3
}

const CUBE_VERTICES: [Vertex; 8] = [
    Vertex {
        position: glm::Vec3::new(0.0, 0.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(0.0, 0.0, 8.0)
    },
    Vertex {
        position: glm::Vec3::new(0.0, 8.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(0.0, 8.0, 8.0)
    },
    Vertex {
        position: glm::Vec3::new(8.0, 0.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(8.0, 0.0, 8.0)
    },
    Vertex {
        position: glm::Vec3::new(8.0, 8.0, 0.0)
    },
    Vertex {
        position: glm::Vec3::new(8.0, 8.0, 8.0)
    }
];

#[rustfmt::skip]
const CUBE_INDICES: [u32; 36] = [
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
