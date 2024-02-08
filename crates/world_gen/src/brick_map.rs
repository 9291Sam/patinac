use std::array::from_fn;
use std::num::{NonZeroU32, NonZeroUsize};
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

#[derive(Debug)]
pub struct BrickMap
{
    tracking_array: Box<
        [[[Option<NonZeroU32>; Self::SIDE_LENGTH_BRICKS as usize];
            Self::SIDE_LENGTH_BRICKS as usize]; Self::SIDE_LENGTH_BRICKS as usize]
    >,
    brick_allocator: util::FreelistAllocator,
    tracking_buffer: Arc<wgpu::Buffer>,
    brick_buffer:    Arc<wgpu::Buffer>
}

pub struct BrickMapBuffers
{
    pub tracking_buffer: Arc<wgpu::Buffer>,
    pub brick_buffer:    Arc<wgpu::Buffer>
}

impl BrickMap
{
    pub const SIDE_LENGTH_BRICKS: i64 = 16;

    pub fn set_voxel(&mut self, voxel: Voxel, position: gfx::I64Vec3)
    {
        let div = position.component_div(&gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));
        let modulo = gfx::modf_vec(&position, &gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));

        // This feels hacky...
        let start_of_array: isize = self.tracking_array.as_ptr() as isize;
        let brick_of_voxel: &mut Option<NonZeroU32> = &mut self.tracking_array
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()];

        match brick_of_voxel
        {
            Some(brick_ptr) =>
            {
                let brick_size: u64 = std::mem::size_of::<Brick>() as u64;
                let brick_addr = brick_size * brick_ptr.get() as u64;
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
                let new_brick: u32 =
                    TryInto::<u32>::try_into(self.brick_allocator.allocate().unwrap().get())
                        .unwrap();

                *brick_of_voxel = NonZeroU32::new(new_brick);

                let brick_ptr: isize = brick_of_voxel as *mut _ as isize;

                let offset_into_buffer: isize = brick_ptr - start_of_array;
                let unsigned_offset: u64 = offset_into_buffer.try_into().unwrap();

                let closure_buffer = self.tracking_buffer.clone();

                let brick_ptr_range = unsigned_offset
                    ..(unsigned_offset + std::mem::size_of::<Option<NonZeroU32>>() as u64);

                self.tracking_buffer
                    .slice(brick_ptr_range.clone())
                    .map_async(wgpu::MapMode::Write, move |map_result| {
                        map_result.unwrap();

                        let ptr = closure_buffer
                            .slice(brick_ptr_range)
                            .get_mapped_range_mut()
                            .as_mut_ptr()
                            as *mut Option<NonZeroU32>;

                        unsafe { ptr.write(Some(NonZeroU32::new(new_brick).unwrap())) };

                        log::info!("Wrote Voxel {voxel:?} @ {position}")
                    });

                // recurse and call the actual set method
                self.set_voxel(voxel, position);
            }
        }
    }

    pub fn new(renderer: &gfx::Renderer) -> (Self, BrickMapBuffers)
    {
        let temp_fixed_size: u64 = 4096;

        let this = Self {
            tracking_array: vec![
                [[None; Self::SIDE_LENGTH_BRICKS as usize];
                    Self::SIDE_LENGTH_BRICKS as usize];
                Self::SIDE_LENGTH_BRICKS as usize
            ]
            .into_boxed_slice()
            .try_into()
            .unwrap(),

            brick_allocator: util::FreelistAllocator::new(
                NonZeroUsize::new(temp_fixed_size.try_into().unwrap()).unwrap()
            ),
            tracking_buffer: Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Brick Tracking storage buffer"),
                size:               std::mem::size_of::<Option<NonZeroU32>>() as u64
                    * temp_fixed_size,
                usage:              wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true
            })),
            brick_buffer:    Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Brick Map storage buffer"),
                size:               std::mem::size_of::<Brick>() as u64 * temp_fixed_size,
                usage:              wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true
            }))
        };

        let brick_map_buffers = BrickMapBuffers {
            tracking_buffer: this.tracking_buffer.clone(),
            brick_buffer:    this.brick_buffer.clone()
        };

        (this, brick_map_buffers)
    }
}
