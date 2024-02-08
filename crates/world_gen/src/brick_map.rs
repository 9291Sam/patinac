use std::array::from_fn;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use gfx::any;
use wgpu::util::DeviceExt;

use crate::*;

#[repr(C)]
struct Brick
{
    bricks: [[[super::Voxel; Self::SIDE_LENGTH_VOXELS as usize]; Self::SIDE_LENGTH_VOXELS as usize];
        Self::SIDE_LENGTH_VOXELS as usize]
}

// const _: () = assert_eq!(std::mem::size_of::<Brick>(), 1024);

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

    pub fn set(&mut self, local_pos: gfx::U64Vec3, voxel: Voxel)
    {
        self.bricks[local_pos.x as usize][local_pos.y as usize][local_pos.z as usize] = voxel;
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
        // let modulo = gfx::modf_vec(&position,
        // &gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS));

        let modulo = position.zip_map(&gfx::I64Vec3::repeat(Brick::SIDE_LENGTH_VOXELS), |l, r| {
            l.rem_euclid(r)
        });

        if modulo.x < 0 || modulo.y < 0 || modulo.z < 0
        {
            log::warn!("modulo returned negative! {position} {modulo}");
        }

        let head_tracking_array: *const Option<NonZeroU32> =
            self.tracking_array.as_ptr() as *const Option<NonZeroU32>;

        let brick_of_voxel: &mut Option<NonZeroU32> = &mut self.tracking_array
            [TryInto::<usize>::try_into(div.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.y + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
            [TryInto::<usize>::try_into(div.z + Self::SIDE_LENGTH_BRICKS / 2).unwrap()];

        match brick_of_voxel
        {
            // The brick has already been mapped, we just need to write to it
            Some(brick_ptr) =>
            {
                let brick_offset_in_buffer =
                    brick_ptr.get() as u64 * std::mem::size_of::<Brick>() as u64;
                let brick_range = brick_offset_in_buffer
                    ..(brick_offset_in_buffer + std::mem::size_of::<Brick>() as u64);

                let ptr = self
                    .brick_buffer
                    .slice(brick_range)
                    .get_mapped_range_mut()
                    .as_mut_ptr() as *mut Brick;

                Brick::set(
                    unsafe { &mut *ptr },
                    gfx::U64Vec3::new(modulo.x as u64, modulo.y as u64, modulo.z as u64),
                    voxel
                );
            }
            None =>
            {
                let new_brick_ptr = NonZeroU32::new(
                    TryInto::<u32>::try_into(self.brick_allocator.allocate().unwrap().get())
                        .unwrap()
                )
                .unwrap();

                // log::info!("Allocated new brick | {} @ {:?}", new_brick_ptr.get(), div);

                // Update CPU side
                *brick_of_voxel = Some(new_brick_ptr);

                let tail_tracking_array = brick_of_voxel as *const Option<NonZeroU32>;

                let offset_unaligned: u64 = TryInto::<u64>::try_into(unsafe {
                    tail_tracking_array.byte_offset_from(head_tracking_array)
                })
                .unwrap();

                let offset_aligned = (offset_unaligned / 8) * 8;
                let forward_offset = offset_unaligned.rem_euclid(8) as usize;

                let ptr = self
                    .tracking_buffer
                    .slice(
                        offset_aligned
                            ..(offset_aligned + std::mem::size_of::<Option<NonZeroU32>>() as u64)
                    )
                    .get_mapped_range_mut()
                    .as_mut_ptr() as *mut Option<NonZeroU32>;

                unsafe { ptr.byte_add(forward_offset).write(Some(new_brick_ptr)) };

                // recurse and call the actual set method
                self.set_voxel(voxel, position);
            }
        }
    }

    pub fn debug_print(&self)
    {
        for xyz in self.tracking_array.iter()
        {
            for yz in xyz.iter()
            {
                for z in yz.iter()
                {
                    log::info!("Id: {:?}", z);
                }
            }
        }
    }

    pub fn new(renderer: &gfx::Renderer) -> (Self, BrickMapBuffers)
    {
        let temp_fixed_size: u64 = 8192;

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
                usage:              wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true
            })),
            brick_buffer:    Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Brick Map storage buffer"),
                size:               std::mem::size_of::<Brick>() as u64 * temp_fixed_size,
                usage:              wgpu::BufferUsages::STORAGE,
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
