use std::array::from_fn;
use std::num::NonZeroUsize;

use wgpu::util::DeviceExt;

use crate::*;

#[repr(C)]
struct Brick
{
    bricks: [[[super::Voxel; Self::SIDE_LENGTH_VOXELS as usize]; Self::SIDE_LENGTH_VOXELS as usize];
        Self::SIDE_LENGTH_VOXELS as usize]
}

impl Brick
{
    pub const SIDE_LENGTH_VOXELS: i64 = 8;

    pub fn new_solid(voxel: super::Voxel) -> Brick
    {
        Brick {
            bricks: [[[voxel; Self::SIDE_LENGTH_VOXELS as usize];
                Self::SIDE_LENGTH_VOXELS as usize];
                Self::SIDE_LENGTH_VOXELS as usize]
        }
    }

    // function is called with args L:SIDE_LENGTH_VOXELS | (0..L, 0..L, 0..L)
    pub fn new_fill(mut fill_func: impl FnMut(usize, usize, usize) -> super::Voxel) -> Brick
    {
        Brick {
            bricks: from_fn(|x| from_fn(|y| from_fn(|z| fill_func(x, y, z))))
        }
    }
}

struct BrickMap
{
    tracking_array: Box<
        [[[Option<NonZeroUsize>; Self::SIDE_LENGTH_BRICKS as usize];
            Self::SIDE_LENGTH_BRICKS as usize]; Self::SIDE_LENGTH_BRICKS as usize]
    >,
    brick_allocator: util::FreelistAllocator,
    brick_buffer:    wgpu::Buffer
}
impl BrickMap
{
    pub const SIDE_LENGTH_BRICKS: i64 = 256;

    pub fn set_voxel(&mut self, voxel: Voxel, position: gfx::I64Vec3)
    {
        let div = position.component_div(&gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));
        let modulo = gfx::modf_vec(&position, &gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));

        match self.tracking_array[div.x as usize][div.y as usize][div.z as usize]
        {
            Some(brick_ptr) =>
            {
                let brick_addr: u64 = std::mem::size_of::<Brick>() as u64 * Into::into(brick_ptr); 
                let voxel_in_brick_addr = 
                let brick_range = brick_addr..(brick_addr + std::mem::size_of::<Brick>() as u64);

                self.brick_buffer
                    .slice(brick_range)
                    .map_async(wgpu::MapMode::Write, |map_result| {
                        let _ = map_result.unwrap();
                        self.brick_buffer.slice(brick_range).get_mapped_range_mut().fill_with(f)
                    })
            }
            None => todo!()
        }
    }

    pub fn new(renderer: &gfx::Renderer) {}
}
