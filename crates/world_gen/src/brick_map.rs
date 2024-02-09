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
    array: Box<[[[Option<NonZeroU32>; Self::SIDE_LENGTH_BRICKS]; Self::SIDE_LENGTH_BRICKS];
        Self::SIDE_LENGTH_BRICKS]>
}

impl BrickTrackingArray
{
    type BrickPointer = Option<NonZeroU32>;

    const SIDE_LENGTH_BRICKS: usize = 16;

    pub fn new() -> Self
    {
        Self {
            array: vec![
                [[None; Self::SIDE_LENGTH_BRICKS];
                    Self::SIDE_LENGTH_BRICKS];
                Self::SIDE_LENGTH_BRICKS
            ]
            .into_boxed_slice()
            .try_into()
            .unwrap()
        }
    }

    pub fn get(&self, pos: &gfx::I64Vec3) -> Self::BrickPointer
    {
        let indices = Self::get_pos_indices(pos);

        self.array[indices.x][indices.y][indices.z]
    }

    // pub fn set(&mut self, new_ptr: BrickPointer)
    // // get set write

    fn get_pos_indices(pos: &gfx::I64Vec3) -> gfx::TVec3<usize>
    {
        pos.map(|c| -> usize {  TryInto::<usize>::try_into(c).unwrap() - TryInto::<usize>::try_into(Self::SIDE_LENGTH_BRICKS / 2).unwrap()} )
    }

}

#[derive(Debug)]
pub struct BrickMap
{
    tracking_array: BrickTrackingArray,
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
        todo!();

        // let VoxelLocation {
        //     brick_pos,
        //     voxel_pos
        // } = get_brick_pos(position, Brick::SIDE_LENGTH_VOXELS);

        // let start_addr_tracking_array: usize = self.tracking_array.as_ptr().addr();

        // let brick_of_voxel: &mut Option<NonZeroU32> = &mut self.tracking_array
        //     [TryInto::<usize>::try_into(brick_pos.x + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
        //     [TryInto::<usize>::try_into(brick_pos.y + Self::SIDE_LENGTH_BRICKS / 2).unwrap()]
        //     [TryInto::<usize>::try_into(brick_pos.z + Self::SIDE_LENGTH_BRICKS / 2).unwrap()];

        // match brick_of_voxel
        // {
        //     // The brick has already been mapped, we just need to write to it
        //     Some(brick_ptr) =>
        //     {
        //         let brick_offset_in_buffer =
        //             brick_ptr.get() as u64 * std::mem::size_of::<Brick>() as u64;
        //         let brick_range = brick_offset_in_buffer
        //             ..(brick_offset_in_buffer + std::mem::size_of::<Brick>() as u64);

        //         let ptr = self
        //             .brick_buffer
        //             .slice(brick_range)
        //             .get_mapped_range_mut()
        //             .as_mut_ptr() as *mut Brick;

        //         Brick::set(
        //             unsafe { &mut *ptr },
        //             gfx::U64Vec3::new(voxel_pos.x, voxel_pos.y, voxel_pos.z),
        //             voxel
        //         );
        //     }
        //     None =>
        //     {
        //         let new_brick_ptr = NonZeroU32::new(
        //             TryInto::<u32>::try_into(self.brick_allocator.allocate().unwrap().get())
        //                 .unwrap()
        //         )
        //         .unwrap();

        //         // log::info!("Allocated new brick | {} @ {:?}", new_brick_ptr.get(), div);

        //         // Update CPU side
        //         *brick_of_voxel = Some(new_brick_ptr);

        //         let tail_tracking_array = brick_of_voxel as *const Option<NonZeroU32>;

        //         let offset_unaligned: u64 = TryInto::<u64>::try_into(unsafe {
        //             tail_tracking_array.byte_offset_from(head_tracking_array)
        //         })
        //         .unwrap();

        //         let offset_aligned = (offset_unaligned / 8) * 8;
        //         let forward_offset = offset_unaligned.rem_euclid(8) as usize;

        //         let ptr = self
        //             .tracking_buffer
        //             .slice(
        //                 offset_aligned
        //                     ..(offset_aligned
        //                         + std::mem::size_of::<Option<NonZeroU32>>() as u64
        //                         + forward_offset as u64)
        //             )
        //             .get_mapped_range_mut()
        //             .as_mut_ptr() as *mut Option<NonZeroU32>;

        //         unsafe { ptr.byte_add(forward_offset).write(Some(new_brick_ptr)) };

        //         // recurse and call the actual set method
        //         self.set_voxel(voxel, position);
        //     }
        // }
    }

    pub fn debug_print(&self)
    {
        todo!()
        // for xyz in self.tracking_array.iter()
        // {
        //     for yz in xyz.iter()
        //     {
        //         for z in yz.iter()
        //         {
        //             log::info!("Id: {:?}", z);
        //         }
        //     }
        // }
    }

    pub fn new(renderer: &gfx::Renderer) -> (Self, BrickMapBuffers)
    {
        todo!()
        // let temp_fixed_size: u64 = 8192;

        // let this = Self {
        //     tracking_array: ,

        //     brick_allocator: util::FreelistAllocator::new(
        //         NonZeroUsize::new(temp_fixed_size.try_into().unwrap()).unwrap()
        //     ),
        //     tracking_buffer: Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
        //         label:              Some("Brick Tracking storage buffer"),
        //         size:               std::mem::size_of::<Option<NonZeroU32>>() as u64
        //             * temp_fixed_size,
        //         usage:              wgpu::BufferUsages::STORAGE,
        //         mapped_at_creation: true
        //     })),
        //     brick_buffer:    Arc::new(renderer.create_buffer(&wgpu::BufferDescriptor {
        //         label:              Some("Brick Map storage buffer"),
        //         size:               std::mem::size_of::<Brick>() as u64 * temp_fixed_size,
        //         usage:              wgpu::BufferUsages::STORAGE,
        //         mapped_at_creation: true
        //     }))
        // };

        // let brick_map_buffers = BrickMapBuffers {
        //     tracking_buffer: this.tracking_buffer.clone(),
        //     brick_buffer:    this.brick_buffer.clone()
        // };

        // (this, brick_map_buffers)
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
        assert_eq!(gfx::TVec3::<usize>::new(0, 0, 0), BrickTrackingArray::get_pos_indices(&gfx::I64Vec3::new(-16, -16, -16)));

    }
}
