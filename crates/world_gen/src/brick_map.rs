use std::array::from_fn;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

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
        [[[Option<NonZeroU64>; Self::SIDE_LENGTH_BRICKS as usize];
            Self::SIDE_LENGTH_BRICKS as usize]; Self::SIDE_LENGTH_BRICKS as usize]
    >,
    brick_allocator: util::FreelistAllocator,
    tracking_buffer: Arc<wgpu::Buffer>,
    brick_buffer:    Arc<wgpu::Buffer>
}
impl BrickMap
{
    pub const SIDE_LENGTH_BRICKS: i64 = 256;

    pub fn set_voxel(&mut self, voxel: Voxel, position: gfx::I64Vec3)
    {
        let div = position.component_div(&gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));
        let modulo = gfx::modf_vec(&position, &gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));

        let brick_of_voxel: &mut Option<NonZeroU64> = &mut self.tracking_array
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()];

        match brick_of_voxel
        {
            Some(brick_ptr) =>
            {
                let brick_size: u64 = std::mem::size_of::<Brick>() as u64;
                let brick_addr = brick_size * brick_ptr.get();
                let brick_range = brick_addr..(brick_addr + brick_size);

                let voxel_size: u64 = std::mem::size_of::<Voxel>() as u64;
                let voxel_addr =
                    ((voxel_size * voxel_size * TryInto::<u64>::try_into(modulo.x).unwrap())
                        + (voxel_size * TryInto::<u64>::try_into(modulo.y).unwrap())
                        + (TryInto::<u64>::try_into(modulo.z).unwrap()))
                        * voxel_size;
                let voxel_range = voxel_addr..(voxel_addr + voxel_size);

                let closure_buffer = self.brick_buffer.clone();

                self.brick_buffer.slice(brick_range.clone()).map_async(
                    wgpu::MapMode::Write,
                    move |map_result| {
                        map_result.unwrap();

                        let ptr: *mut Voxel = closure_buffer
                            .slice(voxel_range)
                            .get_mapped_range_mut()
                            .as_mut_ptr()
                            as *mut Voxel;

                        unsafe { ptr.write(voxel) };

                        log::info!("Wrote Voxel {voxel:?} @ {position}")
                    }
                )
            }
            None =>
            {
                let new_brick =
                    TryInto::<u64>::try_into(self.brick_allocator.allocate().unwrap().get())
                        .unwrap();

                *brick_of_voxel = NonZeroU64::new(new_brick);

                todo!("update tracking buffer");
                // self.tracking_buffer.update_async();

                // recurse and call the actual set method
                self.set_voxel(voxel, position);
            }
        }
    }

    pub fn new(renderer: &gfx::Renderer) -> (Self, Arc<wgpu::Buffer>)
    {
        let this = Self {
            tracking_array: vec![
                [[None; Self::SIDE_LENGTH_BRICKS as usize];
                    Self::SIDE_LENGTH_BRICKS as usize];
                Self::SIDE_LENGTH_BRICKS as usize
            ]
            .into_boxed_slice()
            .try_into()
            .unwrap(),

            brick_allocator: util::FreelistAllocator::new(NonZeroUsize::new(4096).unwrap()),
            brick_buffer:    Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Brickmap storage buffer"),
                size:               std::mem::size_of::<Brick>() as u64 * 4096,
                usage:              wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true
            }))
        };

        let buffer_ptr = this.brick_buffer.clone();

        (this, buffer_ptr)
    }
}
