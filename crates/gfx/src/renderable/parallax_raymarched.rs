use std::borrow::{BorrowMut, Cow};
use std::sync::{Arc, Mutex};

use bytemuck::{bytes_of, cast_slice, Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use {crate as gfx, nalgebra_glm as glm};

use crate::{GenericPass, Transform};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex
{
    color:    glm::Vec3,
    position: glm::Vec3
}

impl Vertex
{
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBS
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PushConstants
{
    mvp:        glm::Mat4,
    model:      glm::Mat4,
    vp:         glm::Mat4,
    camera_pos: glm::Vec3
}

#[derive(Debug)]
pub struct ParallaxRaymarched
{
    uuid:              util::Uuid,
    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    brick_bind_group:  Arc<wgpu::BindGroup>,
    number_of_indices: u16,
    pub transform:     Mutex<gfx::Transform>,
    camera_track:      bool
}

impl ParallaxRaymarched
{
    // pub fn new_camera_tracked(renderer: &gfx::Renderer) -> Arc<Self>
    // {
    //     // TODO: bad!
    //     let vertex_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
    //         label:    Some("Parallax Raymarched Vertex Buffer"),
    //         contents: cast_slice(&CUBE_VERTICES),
    //         usage:    wgpu::BufferUsages::VERTEX
    //     });

    //     let index_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
    //         label:    Some("Parallax Raymarched Index Buffer"),
    //         contents: cast_slice(&CUBE_INDICES),
    //         usage:    wgpu::BufferUsages::INDEX
    //     });

    //     let this = Arc::new(Self {
    //         uuid: util::Uuid::new(),
    //         vertex_buffer,
    //         index_buffer,
    //         number_of_indices: CUBE_INDICES.len().try_into().unwrap(),
    //         transform: Mutex::new(Transform {
    //             scale: glm::Vec3::repeat(0.19),
    //             ..Default::default()
    //         }),
    //         camera_track: true,
    //         brick_bind_group: todo!()
    //     });

    //     renderer.register(this.clone());

    //     this
    // }

    pub fn new_cube(
        renderer: &gfx::Renderer,
        transform: gfx::Transform,
        brick_buffer_bind_group: Arc<wgpu::BindGroup>
    ) -> Arc<Self>
    {
        Self::new(
            renderer,
            transform,
            &CUBE_VERTICES,
            &CUBE_INDICES,
            brick_buffer_bind_group
        )
    }

    pub fn new(
        renderer: &gfx::Renderer,
        transform: gfx::Transform,
        vertices: &[Vertex],
        indices: &[u16],
        brick_buffer_bind_group: Arc<wgpu::BindGroup>
    ) -> Arc<Self>
    {
        let vertex_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Parallax Raymarched Vertex Buffer"),
            contents: cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX
        });

        let index_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Parallax Raymarched Index Buffer"),
            contents: cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX
        });

        let this = Arc::new(Self {
            uuid: util::Uuid::new(),
            vertex_buffer,
            index_buffer,
            number_of_indices: indices.len().try_into().unwrap(),
            transform: Mutex::new(transform),
            camera_track: false,
            brick_bind_group: brick_buffer_bind_group
        });

        renderer.register(this.clone());

        this
    }
}

impl gfx::Recordable for ParallaxRaymarched
{
    fn get_name(&self) -> std::borrow::Cow<'_, str>
    {
        Cow::Borrowed("Parallax Raymarched")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> gfx::PassStage
    {
        gfx::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline_type(&self) -> gfx::PipelineType
    {
        gfx::PipelineType::ParallaxRaymarched
    }

    fn get_bind_groups(&self) -> [Option<&'_ wgpu::BindGroup>; 4]
    {
        [Some(&*self.brick_bind_group), None, None, None]
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
        let GenericPass::Render(ref mut pass) = render_pass
        else
        {
            panic!("unexpected generic pass")
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        {
            let mut guard = self.transform.lock().unwrap();

            if self.camera_track
            {
                guard.translation = camera.get_position();
            }

            pass.set_push_constants(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                bytes_of(&PushConstants {
                    mvp:        camera.get_perspective(renderer, &guard),
                    model:      guard.as_model_matrix(),
                    vp:         camera.get_perspective(
                        renderer,
                        &Transform {
                            ..Default::default()
                        }
                    ),
                    camera_pos: camera.get_position()
                })
            );
        }

        pass.draw_indexed(0..self.number_of_indices as u32, 0, 0..1);
    }
}

const CUBE_VERTICES: [Vertex; 8] = [
    Vertex {
        color:    glm::Vec3::new(0.0, 0.0, 0.0),
        position: glm::Vec3::new(-1.0, -1.0, -1.0)
    },
    Vertex {
        color:    glm::Vec3::new(0.0, 0.0, 1.0),
        position: glm::Vec3::new(-1.0, -1.0, 1.0)
    },
    Vertex {
        color:    glm::Vec3::new(0.0, 1.0, 0.0),
        position: glm::Vec3::new(-1.0, 1.0, -1.0)
    },
    Vertex {
        color:    glm::Vec3::new(0.0, 1.0, 1.0),
        position: glm::Vec3::new(-1.0, 1.0, 1.0)
    },
    Vertex {
        color:    glm::Vec3::new(1.0, 0.0, 0.0),
        position: glm::Vec3::new(1.0, -1.0, -1.0)
    },
    Vertex {
        color:    glm::Vec3::new(1.0, 0.0, 1.0),
        position: glm::Vec3::new(1.0, -1.0, 1.0)
    },
    Vertex {
        color:    glm::Vec3::new(1.0, 1.0, 0.0),
        position: glm::Vec3::new(1.0, 1.0, -1.0)
    },
    Vertex {
        color:    glm::Vec3::new(1.0, 1.0, 1.0),
        position: glm::Vec3::new(1.0, 1.0, 1.0)
    }
];

const CUBE_INDICES: [u16; 36] = [
    6, 2, 7, 2, 3, 7, 0, 4, 5, 1, 0, 5, 0, 2, 6, 4, 0, 6, 3, 1, 7, 1, 5, 7, 2, 0, 3, 0, 1, 3, 4, 6,
    7, 5, 4, 7
];
