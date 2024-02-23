use std::num::NonZeroU32;

use gfx::glm;
use gfx::wgpu::util::{BufferInitDescriptor, DeviceExt};
use gfx::wgpu::{self};

#[repr(u16)]
pub enum Voxel
{
    Air   = 0,
    Red   = 1,
    Green = 2,
    Blue  = 3
}

struct VoxelBrick
{
    data: [[[Voxel; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
}

type BrickPointer = NonZeroU32;

type BrickMap = [[[Option<BrickPointer>; BRICK_EDGE_SIDE]; BRICK_EDGE_SIDE]; BRICK_EDGE_SIDE];

const VOXEL_BRICK_SIZE: usize = 8;
const BRICK_EDGE_SIDE: usize = 128;
const CHUNK_VOXEL_SIZE: usize = VOXEL_BRICK_SIZE * BRICK_EDGE_SIDE;

struct VoxelChunkDataManager
{
    cpu_brick_map: Box<BrickMap>,
    gpu_brick_map: wgpu::Buffer,

    number_of_bricks: usize,
    gpu_brick_buffer: wgpu::Buffer,
    brick_allocator:  util::FreelistAllocator
}

type ChunkPosition = glm::U16Vec3;

impl VoxelChunkDataManager
{
    pub fn new(renderer: &gfx::Renderer) -> Self
    {
        let number_of_starting_bricks = BRICK_EDGE_SIDE * BRICK_EDGE_SIDE;

        Self {
            cpu_brick_map:    vec![[[None; BRICK_EDGE_SIDE]; BRICK_EDGE_SIDE]; BRICK_EDGE_SIDE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            gpu_brick_map:    renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Voxel Chunk Data Manager Brickmap Buffer"),
                usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size:               std::mem::size_of::<BrickMap>() as u64,
                mapped_at_creation: false
            }),
            number_of_bricks: number_of_starting_bricks,
            gpu_brick_buffer: renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Voxel Chunk Data Manager Brick Buffer"),
                usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size:               std::mem::size_of::<VoxelBrick>() as u64
                    * number_of_starting_bricks as u64,
                mapped_at_creation: false
            }),
            brick_allocator:  util::FreelistAllocator::new(
                number_of_starting_bricks.try_into().unwrap()
            )
        }
    }

    pub fn write_voxel(v: Voxel, pos: ChunkPosition) {}
}

#[cfg(test)]
mod test
{
    pub use super::*;

    #[test]
    fn assert_sizes()
    {
        assert_eq!(1024, std::mem::size_of::<VoxelBrick>());
        assert_eq!(8 * 1024 * 1024, std::mem::size_of::<BrickMap>());
    }
}
