use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::{bytes_of, Contiguous};
use gfx::glm;
use gfx::wgpu::{self};

#[repr(u16)]
#[derive(Clone, Copy)]
pub enum Voxel
{
    Air   = 0,
    Red   = 1,
    Green = 2,
    Blue  = 3
}

impl Voxel
{
    pub fn as_bytes(&self) -> [u8; 2]
    {
        (*self as u16).to_ne_bytes()
    }
}

pub struct VoxelBrick
{
    data: [[[Voxel; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
}

impl VoxelBrick
{
    pub fn write(
        &mut self
    ) -> &mut [[[Voxel; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]; VOXEL_BRICK_SIZE]
    {
        &mut self.data
    }
}

type BrickPointer = NonZeroU32;

pub(crate) type BrickMap =
    [[[Option<BrickPointer>; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE];

const VOXEL_BRICK_SIZE: usize = 8;
const BRICK_MAP_EDGE_SIZE: usize = 128;
const CHUNK_VOXEL_SIZE: usize = VOXEL_BRICK_SIZE * BRICK_MAP_EDGE_SIZE;

#[derive(Debug)]
pub struct VoxelChunkDataManager
{
    renderer: Arc<gfx::Renderer>,

    cpu_brick_map:            Box<BrickMap>,
    pub(crate) gpu_brick_map: wgpu::Buffer,

    number_of_bricks:            usize,
    pub(crate) gpu_brick_buffer: wgpu::Buffer,
    brick_allocator:             util::FreelistAllocator
}

pub type ChunkPosition = glm::U16Vec3;

impl VoxelChunkDataManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        let number_of_starting_bricks = BRICK_MAP_EDGE_SIZE * BRICK_MAP_EDGE_SIZE * 8;

        let r = renderer.clone();

        Self {
            renderer,
            cpu_brick_map: vec![
                [[None; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE];
                BRICK_MAP_EDGE_SIZE
            ]
            .into_boxed_slice()
            .try_into()
            .unwrap(),
            gpu_brick_map: r.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Voxel Chunk Data Manager Brick Map Buffer"),
                usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size:               std::mem::size_of::<BrickMap>() as u64,
                mapped_at_creation: true
            }),
            number_of_bricks: number_of_starting_bricks,
            gpu_brick_buffer: r.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Voxel Chunk Data Manager Brick Buffer"),
                usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size:               std::mem::size_of::<VoxelBrick>() as u64
                    * number_of_starting_bricks as u64,
                mapped_at_creation: true
            }),
            brick_allocator: util::FreelistAllocator::new(
                number_of_starting_bricks.try_into().unwrap()
            )
        }
    }

    pub fn write_voxel(&mut self, v: Voxel, pos: ChunkPosition)
    {
        let voxel_bound = BRICK_MAP_EDGE_SIZE * VOXEL_BRICK_SIZE * 2;

        assert!((pos.x as usize) < voxel_bound);
        assert!((pos.y as usize) < voxel_bound);
        assert!((pos.z as usize) < voxel_bound);

        let voxel_pos = glm::U16Vec3::new(
            pos.x % VOXEL_BRICK_SIZE as u16,
            pos.y % VOXEL_BRICK_SIZE as u16,
            pos.z % VOXEL_BRICK_SIZE as u16
        );

        let brick_pos = glm::U16Vec3::new(
            pos.x / VOXEL_BRICK_SIZE as u16,
            pos.y / VOXEL_BRICK_SIZE as u16,
            pos.z / VOXEL_BRICK_SIZE as u16
        );

        let this_brick_head = (&self.cpu_brick_map[0][0][0]) as *const _;
        let this_brick = &mut self.cpu_brick_map[brick_pos.x as usize][brick_pos.y as usize]
            [brick_pos.z as usize];
        let this_brick_byte_offset =
            unsafe { (this_brick as *mut _ as *const u8).byte_offset_from(this_brick_head) };

        match this_brick
        {
            Some(brick_ptr) =>
            {
                let mapped_ptr = unsafe {
                    (self
                        .gpu_brick_buffer
                        .slice(..)
                        .get_mapped_range_mut()
                        .as_mut_ptr() as *mut VoxelBrick)
                        .add(brick_ptr.into_integer() as usize)
                };

                VoxelBrick::write(unsafe { &mut *mapped_ptr })[voxel_pos.x as usize]
                    [voxel_pos.y as usize][voxel_pos.z as usize] = Voxel::Blue;

                // let voxel_bytes = &v.as_bytes();

                // unsafe { (*mapped_ptr) }.data

                // self.renderer.queue.write_buffer(
                //     &self.gpu_brick_buffer,
                //     std::mem::size_of::<VoxelBrick>() as u64 *
                // brick_ptr.into_integer() as u64
                //         + voxel_offset_bytes,
                //     &v.as_bytes()
                // );
            }
            None =>
            {
                // Allocate new pointer, then recurse
                let new_brick_ptr: NonZeroU32 =
                    self.brick_allocator.allocate().unwrap().try_into().unwrap();

                // update cpu side
                *this_brick = Some(new_brick_ptr);

                // update gpu side
                let mapped_ptr = self
                    .gpu_brick_map
                    .slice(..)
                    .get_mapped_range_mut()
                    .as_mut_ptr();

                let brick_ptr_bytes = bytes_of(&new_brick_ptr);

                unsafe {
                    mapped_ptr
                        .add(this_brick_byte_offset.try_into().unwrap())
                        .copy_from_nonoverlapping(brick_ptr_bytes.as_ptr(), brick_ptr_bytes.len())
                }

                // self.renderer.queue.write_buffer(
                //     &self.gpu_brick_map,
                //     this_brick_byte_offset as u64,
                //     bytes_of(&new_brick_ptr)
                // );

                //
                self.write_voxel(v, pos);
            }
        }

        // update gpu side
    }

    pub fn stop_writes(&mut self)
    {
        self.gpu_brick_buffer.unmap();
        self.gpu_brick_map.unmap();
    }
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
