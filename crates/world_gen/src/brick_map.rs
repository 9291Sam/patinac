use std::array::from_fn;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::RangeInclusive;
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
    pub const SIDE_LENGTH_VOXELS: u64 = 8;

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
        debug_assert!(local_pos.x < Self::SIDE_LENGTH_VOXELS);

        debug_assert!(local_pos.y < Self::SIDE_LENGTH_VOXELS);

        debug_assert!(local_pos.z < Self::SIDE_LENGTH_VOXELS);

        self.bricks[local_pos.x as usize][local_pos.y as usize][local_pos.z as usize] = voxel;
    }
}

#[derive(Debug)]
struct BrickTrackingArray
{
    array: Box<
        [[[Option<NonZeroU32>; Self::SIDE_LENGTH_BRICKS as usize];
            Self::SIDE_LENGTH_BRICKS as usize]; Self::SIDE_LENGTH_BRICKS as usize]
    >
}

impl BrickTrackingArray
{
    const SIDE_LENGTH_BRICKS: u64 = 16;

    pub fn new() -> Self
    {
        Self {
            array: vec![
                [[None; Self::SIDE_LENGTH_BRICKS as usize];
                    Self::SIDE_LENGTH_BRICKS as usize];
                Self::SIDE_LENGTH_BRICKS as usize
            ]
            .into_boxed_slice()
            .try_into()
            .unwrap()
        }
    }

    pub fn access(&self, pos: &gfx::I64Vec3) -> &Option<NonZeroU32>
    {
        let indices = Self::get_pos_indices(pos);

        &self.array[indices.x][indices.y][indices.z]
    }

    pub fn access_mut(&mut self, pos: &gfx::I64Vec3) -> &mut Option<NonZeroU32>
    {
        let indices = Self::get_pos_indices(pos);

        &mut self.array[indices.x][indices.y][indices.z]
    }

    pub fn access_mut_with_head_offset(
        &mut self,
        pos: &gfx::I64Vec3
    ) -> (&mut Option<NonZeroU32>, usize)
    {
        let head = unsafe { self.access_head() };
        let this_brick = self.access_mut(pos);

        let brick_offset_elements =
            unsafe { (this_brick as *mut Option<NonZeroU32>).offset_from(head) };

        (this_brick, brick_offset_elements.try_into().unwrap())
    }

    pub unsafe fn access_head(&self) -> *const Option<NonZeroU32>
    {
        &self.array[0][0][0]
    }

    fn get_pos_indices(pos: &gfx::I64Vec3) -> gfx::TVec3<usize>
    {
        const UPPER_BOUND: i64 = (BrickTrackingArray::SIDE_LENGTH_BRICKS as i64 / 2) - 1;
        const LOWER_BOUND: i64 = -(BrickTrackingArray::SIDE_LENGTH_BRICKS as i64 / 2);
        const BOUND: RangeInclusive<i64> = LOWER_BOUND..=UPPER_BOUND;

        assert!(BOUND.contains(&pos.x));
        assert!(BOUND.contains(&pos.y));
        assert!(BOUND.contains(&pos.z));

        pos.map(|c| -> usize {
            TryInto::<usize>::try_into(c + Self::SIDE_LENGTH_BRICKS as i64 / 2).unwrap()
        })
    }
}

#[derive(Debug)]
pub struct BrickMap
{
    tracking_array:  BrickTrackingArray,
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
    pub fn set_voxel(&mut self, voxel: Voxel, position: gfx::I64Vec3)
    {
        let VoxelLocation {
            brick_pos,
            voxel_pos
        } = get_brick_pos(position, Brick::SIDE_LENGTH_VOXELS);

        let (cpu_maybe_brick_ptr, brick_ptr_offset_elements) =
            self.tracking_array.access_mut_with_head_offset(&brick_pos);

        match *cpu_maybe_brick_ptr
        {
            Some(brick_ptr) =>
            {
                let gpu_offset_bytes = brick_ptr.get() as u64 * std::mem::size_of::<Brick>() as u64;

                let gpu_brick: &mut Brick = unsafe {
                    &mut *(self
                        .brick_buffer
                        .slice(
                            gpu_offset_bytes
                                ..(gpu_offset_bytes + std::mem::size_of::<Brick>() as u64)
                        )
                        .get_mapped_range_mut()
                        .as_mut_ptr() as *mut Brick)
                };

                gpu_brick.set(voxel_pos, voxel);
            }
            None =>
            {
                let new_brick_ptr: u32 = self
                    .brick_allocator
                    .allocate()
                    .unwrap()
                    .get()
                    .try_into()
                    .unwrap();

                *cpu_maybe_brick_ptr = Some(NonZeroU32::new(new_brick_ptr).unwrap());

                let gpu_brick_ptr_ptr = self
                    .tracking_buffer
                    .slice(..) // TODO:
                    .get_mapped_range_mut()
                    .as_mut_ptr()
                    as *mut Option<NonZeroU32>;

                unsafe {
                    *gpu_brick_ptr_ptr.wrapping_add(brick_ptr_offset_elements) =
                        NonZeroU32::new(new_brick_ptr);
                }

                self.set_voxel(voxel, position);
            }
        }
    }

    pub fn new(renderer: &gfx::Renderer) -> (Self, BrickMapBuffers)
    {
        let temp_fixed_size: u64 = 8192;

        let this = Self {
            tracking_array:  BrickTrackingArray::new(),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Default)]
struct VoxelLocation
{
    pub brick_pos: gfx::I64Vec3,
    pub voxel_pos: gfx::U64Vec3
}

fn get_brick_pos(pos: gfx::I64Vec3, brick_side_length: u64) -> VoxelLocation
{
    let brick_side_vec = gfx::I64Vec3::repeat(brick_side_length.try_into().unwrap());

    VoxelLocation {
        brick_pos: pos.zip_map(&brick_side_vec, |l, r| l.div_floor(r)),
        voxel_pos: pos.zip_map(&brick_side_vec, |l, r| l.rem_euclid(r).try_into().unwrap())
    }
}

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_brick_pos()
    {
        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(0, 0, 0), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(0, 0, 0),
                voxel_pos: gfx::U64Vec3::new(0, 0, 0)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(0, 8, 0), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(0, 1, 0),
                voxel_pos: gfx::U64Vec3::new(0, 0, 0)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(7, 7, 7), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(0, 0, 0),
                voxel_pos: gfx::U64Vec3::new(7, 7, 7)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(3, 58, 21), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(0, 7, 2),
                voxel_pos: gfx::U64Vec3::new(3, 2, 5)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(-1, -1, -1), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(-1, -1, -1),
                voxel_pos: gfx::U64Vec3::new(7, 7, 7)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(-8, -8, -8), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(-1, -1, -1),
                voxel_pos: gfx::U64Vec3::new(0, 0, 0)
            }
        );

        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(-7, -7, -7), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(-1, -1, -1),
                voxel_pos: gfx::U64Vec3::new(1, 1, 1)
            }
        );

        // 0   ->  7  |  0
        // -8  ->  0  | -1
        // -16 -> -9  | -2
        // -24 -> -17 | -3
        // -32 -> -25 | -4
        // -40 -> -33 | -5
        // -48 -> -41 | -6
        // -56 -> -49 | -7
        // -64 -> -57 | -8
        assert_eq!(
            get_brick_pos(gfx::I64Vec3::new(-3, -58, -21), 8),
            VoxelLocation {
                brick_pos: gfx::I64Vec3::new(-1, -8, -3),
                voxel_pos: gfx::U64Vec3::new(5, 6, 3)
            }
        );
    }

    #[test]
    fn test_get_pos_indices()
    {
        assert_eq!(
            gfx::TVec3::<usize>::new(0, 0, 0),
            BrickTrackingArray::get_pos_indices(&gfx::I64Vec3::new(-8, -8, -8))
        );

        assert_eq!(
            gfx::TVec3::<usize>::new(1, 1, 1),
            BrickTrackingArray::get_pos_indices(&gfx::I64Vec3::new(-7, -7, -7))
        );

        assert_eq!(
            gfx::TVec3::<usize>::new(15, 14, 2),
            BrickTrackingArray::get_pos_indices(&gfx::I64Vec3::new(7, 6, -6))
        );
    }
}
