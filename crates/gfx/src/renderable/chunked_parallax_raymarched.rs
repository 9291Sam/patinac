use std::borrow::Cow;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use bytemuck::{bytes_of, Pod, Zeroable};
use nalgebra_glm as glm;
use strum::{EnumIter, IntoEnumIterator};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::Transform;

// TODO: instancing

#[derive(Debug)]
pub struct ChunkedParallaxRaymarched
{
    world:             Arc<VoxelWorld>,
    vertex_buffer:     wgpu::Buffer,
    index_buffer:      wgpu::Buffer,
    number_of_indices: u32,
    uuid:              util::Uuid // TODO: proper transforms please
}
impl ChunkedParallaxRaymarched
{
    pub fn new(renderer: &crate::Renderer, world: Arc<VoxelWorld>)
    -> Arc<ChunkedParallaxRaymarched>
    {
        let vertices = CUBE_VERTICES
            .iter()
            .map(|p| {
                Vertex {
                    position:     p.component_mul(&glm::Vec3::repeat(256.0)),
                    chunk_ptr:    0,
                    local_offset: p.component_mul(&glm::Vec3::repeat(256.0))
                        + glm::Vec3::repeat(256.0)
                }
            })
            .collect::<Vec<_>>();

        let vertex_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Chunked Parallax Raymarched Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices[..]),
            usage:    wgpu::BufferUsages::VERTEX
        });

        let index_buffer = renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("Chunked Parallax Raymarched Index Buffer"),
            contents: bytemuck::cast_slice(&CUBE_INDICES),
            usage:    wgpu::BufferUsages::INDEX
        });

        let this = Arc::new(ChunkedParallaxRaymarched {
            world,
            vertex_buffer,
            index_buffer,
            number_of_indices: CUBE_INDICES.len() as u32,
            uuid: util::Uuid::new()
        });

        renderer.register(this.clone());

        this
    }
}

impl super::Recordable for ChunkedParallaxRaymarched
{
    fn get_name(&self) -> Cow<'_, str>
    {
        Cow::Borrowed("Chunked Parallax Raymarched")
    }

    fn get_uuid(&self) -> util::Uuid
    {
        self.uuid
    }

    fn get_pass_stage(&self) -> crate::PassStage
    {
        crate::PassStage::GraphicsSimpleColor
    }

    fn get_pipeline_type(&self) -> crate::PipelineType
    {
        crate::PipelineType::ChunkedParallaxRaymarched
    }

    fn pre_record_update(&self, _: &crate::Renderer, _: &crate::Camera) -> crate::RecordInfo
    {
        // TODO: just sort the chunks!

        crate::RecordInfo {
            should_draw: true,
            transform:   Some(Transform {
                translation: glm::Vec3::new(0.0, 0.0, 270.0),
                ..Default::default()
            })
        }
    }

    fn get_bind_groups<'s>(
        &'s self,
        global_bind_group: &'s wgpu::BindGroup
    ) -> [Option<&'s wgpu::BindGroup>; 4]
    {
        [
            Some(global_bind_group),
            Some(&self.world.bind_group),
            None,
            None
        ]
    }

    fn record<'s>(
        &'s self,
        render_pass: &mut crate::GenericPass<'s>,
        maybe_id: Option<crate::DrawId>
    )
    {
        let (crate::GenericPass::Render(ref mut pass), Some(id)) = (render_pass, maybe_id)
        else
        {
            panic!("Generic RenderPass bound with incorrect type!")
        };

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, bytes_of(&id));
        pass.draw_indexed(0..self.number_of_indices, 0, 0..1);
    }
}

#[derive(Debug)]
pub struct VoxelWorld
{
    brick_buffer:          wgpu::Buffer,
    brick_allocator:       util::FreelistAllocator,
    chunk_tracking_buffer: Box<ChunkStorageBuffer>,
    chunk_buffer:          wgpu::Buffer,
    bind_group:            wgpu::BindGroup
}

impl VoxelWorld
{
    pub fn new(renderer: &crate::Renderer) -> Arc<VoxelWorld>
    {
        let brick_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Voxel World Brick Storage Buffer"),
            size:               std::mem::size_of::<BrickStorageBuffer>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let chunk_buffer = renderer.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Voxel World Chunk Storage Buffer"),
            size:               std::mem::size_of::<ChunkStorageBuffer>() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = renderer.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Voxel World Bind Group"),
            layout:  renderer
                .render_cache
                .lookup_bind_group_layout(crate::BindGroupType::BrickMap),
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: brick_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: chunk_buffer.as_entire_binding()
                }
            ]
        });

        Arc::new(VoxelWorld {
            brick_buffer,
            chunk_buffer,
            bind_group,
            brick_allocator: util::FreelistAllocator::new(
                NonZeroUsize::new(BRICK_STORAGE_BUFFER_LENGTH).unwrap()
            ),
            chunk_tracking_buffer: unsafe { Box::new_zeroed().assume_init() }
        })
    }

    // pub fn write_voxel()
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex
{
    position:     glm::Vec3,
    chunk_ptr:    u32,
    local_offset: glm::Vec3
}

impl Vertex
{
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32 , 2 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBUTES
        }
    }
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, EnumIter)]
pub enum Voxel
{
    #[default]
    Air   = 0,
    Red   = 1,
    Green = 2,
    Blue  = 3
}

impl Voxel
{
    pub fn get_material_lookup(&self) -> Box<[VoxelMaterial]>
    {
        Voxel::iter()
            .map(|v| v.get_material())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[inline(never)]
    pub fn get_material(&self) -> VoxelMaterial
    {
        match *self
        {
            Voxel::Air =>
            {
                VoxelMaterial {
                    is_visible: false,
                    srgb_r:     0,
                    srgb_g:     0,
                    srgb_b:     0
                }
            }
            Voxel::Red =>
            {
                VoxelMaterial {
                    is_visible: true,
                    srgb_r:     255,
                    srgb_g:     0,
                    srgb_b:     0
                }
            }
            Voxel::Green =>
            {
                VoxelMaterial {
                    is_visible: true,
                    srgb_r:     0,
                    srgb_g:     255,
                    srgb_b:     0
                }
            }
            Voxel::Blue =>
            {
                VoxelMaterial {
                    is_visible: true,
                    srgb_r:     0,
                    srgb_g:     0,
                    srgb_b:     255
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VoxelMaterial
{
    is_visible: bool,
    srgb_r:     u8,
    srgb_g:     u8,
    srgb_b:     u8
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Brick
{
    data: [[[Voxel; Self::SIDE_VOXELS]; Self::SIDE_VOXELS]; Self::SIDE_VOXELS]
}

impl Brick
{
    pub const SIDE_VOXELS: usize = 8;
}
pub const BRICK_STORAGE_BUFFER_LENGTH: usize = 131072;
pub type BrickStorageBuffer = [Brick; BRICK_STORAGE_BUFFER_LENGTH];
pub type BrickPointer = NonZeroU32;
pub type MaybeBrickPointer = Option<NonZeroU32>;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chunk
{
    data: [[[MaybeBrickPointer; Self::SIDE_BRICKS]; Self::SIDE_BRICKS]; Self::SIDE_BRICKS]
}

unsafe impl Zeroable for Chunk {}

impl Chunk
{
    pub const SIDE_BRICKS: usize = 64;
}

pub type ChunkStorageBuffer = [Chunk; 128];
pub type ChunkPointer = u32;

const CUBE_VERTICES: [glm::Vec3; 8] = [
    glm::Vec3::new(-1.0, -1.0, -1.0),
    glm::Vec3::new(-1.0, -1.0, 1.0),
    glm::Vec3::new(-1.0, 1.0, -1.0),
    glm::Vec3::new(-1.0, 1.0, 1.0),
    glm::Vec3::new(1.0, -1.0, -1.0),
    glm::Vec3::new(1.0, -1.0, 1.0),
    glm::Vec3::new(1.0, 1.0, -1.0),
    glm::Vec3::new(1.0, 1.0, 1.0)
];

const CUBE_INDICES: [u16; 36] = [
    6, 2, 7, 2, 3, 7, 0, 4, 5, 1, 0, 5, 0, 2, 6, 4, 0, 6, 3, 1, 7, 1, 5, 7, 2, 0, 3, 0, 1, 3, 4, 6,
    7, 5, 4, 7
];

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    pub fn assert_sizes()
    {
        assert_eq!(std::mem::size_of::<Brick>(), 1024);
        assert_eq!(std::mem::size_of::<BrickStorageBuffer>(), 128 * 1024 * 1024);
        assert_eq!(std::mem::size_of::<ChunkStorageBuffer>(), 128 * 1024 * 1024);
    }
}
