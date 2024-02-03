use std::io::Cursor;
use std::sync::Mutex;

use bytemuck::{bytes_of, Pod, Zeroable};
use nalgebra_glm as glm;

use crate::gfx;

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
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];

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
pub struct LitTextured
{
    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    texture_normal:    wgpu::BindGroup,
    number_of_indices: u32,
    transform:         Mutex<gfx::Transform>
}

impl LitTextured
{
    pub fn new_cube()
    {
        let obj_data = include_bytes!("cube.obj");
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
                        let mtl_data = include_bytes!("cube.mtl");
                        let mut mtl_cursor = Cursor::new(mtl_data);

                        tobj::load_mtl_buf(&mut mtl_cursor)
                    }
                    _ => unreachable!()
                }
            }
        )
        .map(|err| (err.0, err.1.unwrap()))
        .unwrap();

        log::info!("Size {} {}", model.len(), material.len());

        todo!()
    }

    pub fn new() {}
}
impl gfx::Recordable for LitTextured
{
    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline_type(&self) -> gfx::PipelineType
    {
        gfx::PipelineType::LitTextured
    }

    fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4]
    {
        [Some(&self.texture_normal), None, None, None]
    }

    fn should_render(&self) -> bool
    {
        true
    }

    fn record<'s>(
        &'s self,
        render_pass: &mut gfx::GenericPass<'s>,
        renderer: &gfx::Renderer,
        camera: &gfx::Camera
    )
    {
        let gfx::GenericPass::Render(ref mut pass) = render_pass
        else
        {
            panic!("Unexpected GenericPass")
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_push_constants(
            wgpu::ShaderStages::VERTEX,
            0,
            bytes_of(&camera.get_perspective(renderer, &*self.transform.lock().unwrap()))
        );
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}
