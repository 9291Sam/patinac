use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::bytes_of;
use gfx::{glm, wgpu, Recordable};

#[derive(Debug)]
struct Chunk
{
    uuid:     util::Uuid,
    position: glm::Vec3,
    name:     String,

    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    number_of_indices: u32,
    pipeline:          Arc<gfx::GenericPipeline>,
    voxel_bind_group:  wgpu::BindGroup
}

impl Chunk
{
    pub fn new(game: &game::Game) -> Self
    {
        Self {
            uuid:              todo!(),
            position:          todo!(),
            name:              todo!(),
            vertex_buffer:     todo!(),
            index_buffer:      todo!(),
            number_of_indices: todo!(),
            pipeline:          todo!(),
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
