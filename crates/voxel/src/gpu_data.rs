use std::assert_matches::assert_matches;
use std::num::NonZeroU32;
use std::sync::Arc;

use bytemuck::{bytes_of, Contiguous};
use gfx::glm;
use gfx::wgpu::{self};

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Voxel
{
    Air   = 0,
    Red   = 1,
    Green = 2,
    Blue  = 3
}

impl TryFrom<u16> for Voxel
{
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error>
    {
        match value
        {
            0 => Ok(Voxel::Air),
            1 => Ok(Voxel::Red),
            2 => Ok(Voxel::Green),
            3 => Ok(Voxel::Blue),
            _ => Err(())
        }
    }
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

    pub fn fill(&mut self, voxel: Voxel)
    {
        for slice in self.data.iter_mut()
        {
            for layer in slice.iter_mut()
            {
                for v in layer.iter_mut()
                {
                    *v = voxel;
                }
            }
        }
    }
}

/// Value Table
/// 0 - null brick pointer
/// [1, 2^32 - 2^16) - valid brick pointer
/// [2^32 - 2^16 - 2^32] - voxel

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct VoxelBrickPointer
{
    data: u32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum VoxelBrickPointerType
{
    Null,
    ValidBrickPointer(u32),
    Voxel(Voxel)
}

impl VoxelBrickPointer
{
    /// The first voxel
    const POINTER_TO_VOXEL_BOUND: u32 = u32::MAX - 2u32.pow(16);

    pub fn new_null() -> VoxelBrickPointer
    {
        VoxelBrickPointer {
            data: 0
        }
    }

    pub fn new_ptr(ptr: u32) -> VoxelBrickPointer
    {
        let maybe_new_ptr = VoxelBrickPointer {
            data: ptr
        };

        assert_matches!(
            maybe_new_ptr.classify(),
            VoxelBrickPointerType::ValidBrickPointer(ptr),
            "Tried to create a pointer with an invalid value. Valid range is [1, 2^32 - 2^16)"
        );

        maybe_new_ptr
    }

    pub fn new_voxel(voxel: Voxel) -> VoxelBrickPointer
    {
        let maybe_new_ptr = VoxelBrickPointer {
            data: voxel as u16 as u32 + Self::POINTER_TO_VOXEL_BOUND
        };

        assert!(voxel != Voxel::Air);
        assert_matches!(
            maybe_new_ptr.classify(),
            VoxelBrickPointerType::Voxel(voxel)
        );

        maybe_new_ptr
    }

    pub fn get_ptr(&self) -> u32
    {
        if let VoxelBrickPointerType::ValidBrickPointer(ptr) = self.classify()
        {
            ptr
        }
        else
        {
            panic!()
        }
    }

    pub fn classify(&self) -> VoxelBrickPointerType
    {
        match self.data
        {
            0 => VoxelBrickPointerType::Null,
            1..Self::POINTER_TO_VOXEL_BOUND => VoxelBrickPointerType::ValidBrickPointer(self.data),
            Self::POINTER_TO_VOXEL_BOUND..=u32::MAX =>
            {
                VoxelBrickPointerType::Voxel(
                    Voxel::try_from((self.data - Self::POINTER_TO_VOXEL_BOUND) as u16).unwrap()
                )
            }
        }
    }
}

// type BrickPointer = NonZeroU32;

pub(crate) type BrickMap =
    [[[VoxelBrickPointer; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE]; BRICK_MAP_EDGE_SIZE];

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

pub type ChunkPosition = glm::I16Vec3;

impl VoxelChunkDataManager
{
    pub fn new(renderer: Arc<gfx::Renderer>) -> Self
    {
        let number_of_starting_bricks = BRICK_MAP_EDGE_SIZE * BRICK_MAP_EDGE_SIZE * 64;

        let r = renderer.clone();

        Self {
            renderer,
            cpu_brick_map: vec![
                [[VoxelBrickPointer::new_null(); BRICK_MAP_EDGE_SIZE];
                    BRICK_MAP_EDGE_SIZE];
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
                (number_of_starting_bricks - 2).try_into().unwrap()
            )
        }
    }

    pub fn write_brick(&mut self, v: Voxel, signed_pos: ChunkPosition)
    {
        assert!(v != Voxel::Air);

        let signed_bound: i16 = (BRICK_MAP_EDGE_SIZE / 2) as i16;

        assert!(
            signed_pos.x >= -signed_bound && signed_pos.x < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.x
        );

        assert!(
            signed_pos.y >= -signed_bound && signed_pos.y < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.y
        );

        assert!(
            signed_pos.z >= -signed_bound && signed_pos.z < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.z
        );

        let unsigned_pos: glm::U16Vec3 = signed_pos.add_scalar(signed_bound).try_cast().unwrap();

        let current_brick_ptr = &mut self.cpu_brick_map[unsigned_pos.x as usize]
            [unsigned_pos.y as usize][unsigned_pos.z as usize];

        if let VoxelBrickPointerType::ValidBrickPointer(old_ptr) = current_brick_ptr.classify()
        {
            self.brick_allocator
                .free((old_ptr.into_integer() as usize).try_into().unwrap())
        }

        *current_brick_ptr = VoxelBrickPointer::new_voxel(v)
    }

    pub fn write_voxel(&mut self, v: Voxel, signed_pos: ChunkPosition)
    {
        let signed_bound: i16 = (BRICK_MAP_EDGE_SIZE * VOXEL_BRICK_SIZE / 2) as i16;

        assert!(
            signed_pos.x >= -signed_bound && signed_pos.x < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.x
        );

        assert!(
            signed_pos.y >= -signed_bound && signed_pos.y < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.y
        );

        assert!(
            signed_pos.z >= -signed_bound && signed_pos.z < signed_bound,
            "Bound {} is out of bounds",
            signed_pos.z
        );

        let unsigned_pos: glm::U16Vec3 = signed_pos.add_scalar(signed_bound).try_cast().unwrap();

        let voxel_pos = glm::U16Vec3::new(
            unsigned_pos.x % VOXEL_BRICK_SIZE as u16,
            unsigned_pos.y % VOXEL_BRICK_SIZE as u16,
            unsigned_pos.z % VOXEL_BRICK_SIZE as u16
        );

        let brick_pos = glm::U16Vec3::new(
            unsigned_pos.x / VOXEL_BRICK_SIZE as u16,
            unsigned_pos.y / VOXEL_BRICK_SIZE as u16,
            unsigned_pos.z / VOXEL_BRICK_SIZE as u16
        );

        let this_brick_head = (&self.cpu_brick_map[0][0][0]) as *const _;
        let this_brick = &mut self.cpu_brick_map[brick_pos.x as usize][brick_pos.y as usize]
            [brick_pos.z as usize];
        let this_brick_byte_offset =
            unsafe { (this_brick as *mut _ as *const u8).byte_offset_from(this_brick_head) };

        match this_brick.classify()
        {
            VoxelBrickPointerType::ValidBrickPointer(brick_ptr) =>
            {
                write_voxel_to_brick(*this_brick, &self.gpu_brick_buffer, voxel_pos, v);
            }
            VoxelBrickPointerType::Voxel(voxel) =>
            {
                allocate_new_brick(
                    &mut self.brick_allocator,
                    this_brick,
                    this_brick_byte_offset.try_into().unwrap(),
                    &self.gpu_brick_map
                );

                fill_brick_with_voxel(*this_brick, &self.gpu_brick_buffer, voxel);

                write_voxel_to_brick(*this_brick, &self.gpu_brick_buffer, voxel_pos, v);
            }
            VoxelBrickPointerType::Null =>
            {
                allocate_new_brick(
                    &mut self.brick_allocator,
                    this_brick,
                    this_brick_byte_offset.try_into().unwrap(),
                    &self.gpu_brick_map
                );

                write_voxel_to_brick(*this_brick, &self.gpu_brick_buffer, voxel_pos, v);
            }
        }
    }

    pub fn stop_writes(&mut self)
    {
        self.gpu_brick_buffer.unmap();
        self.gpu_brick_map.unmap();
    }
}

fn write_voxel_to_brick(
    brick_ptr: VoxelBrickPointer,
    brick_buffer: &wgpu::Buffer,
    local_voxel_pos: glm::U16Vec3,
    voxel: Voxel
)
{
    let VoxelBrickPointerType::ValidBrickPointer(brick_ptr_integer) = brick_ptr.classify()
    else
    {
        unreachable!()
    };

    let mapped_ptr = unsafe {
        (brick_buffer.slice(..).get_mapped_range_mut().as_mut_ptr() as *mut VoxelBrick)
            .add(brick_ptr_integer as usize)
    };

    VoxelBrick::write(unsafe { &mut *mapped_ptr })[local_voxel_pos.x as usize]
        [local_voxel_pos.y as usize][local_voxel_pos.z as usize] = voxel;
}

fn fill_brick_with_voxel(brick_ptr: VoxelBrickPointer, brick_buffer: &wgpu::Buffer, voxel: Voxel)
{
    let VoxelBrickPointerType::ValidBrickPointer(brick_ptr_integer) = brick_ptr.classify()
    else
    {
        unreachable!()
    };

    let mapped_ptr = unsafe {
        (brick_buffer.slice(..).get_mapped_range_mut().as_mut_ptr() as *mut VoxelBrick)
            .add(brick_ptr_integer as usize)
    };

    VoxelBrick::fill(unsafe { &mut *mapped_ptr }, voxel);
}

fn allocate_new_brick(
    allocator: &mut util::FreelistAllocator,
    cpu_location: &mut VoxelBrickPointer,
    cpu_offset: usize,
    gpu_buffer: &wgpu::Buffer
)
{
    let new_brick_ptr: NonZeroU32 = allocator.allocate().unwrap().try_into().unwrap();

    // update cpu side
    *cpu_location = VoxelBrickPointer::new_ptr(new_brick_ptr.into_integer());

    // update gpu side
    let mapped_ptr = gpu_buffer.slice(..).get_mapped_range_mut().as_mut_ptr();

    let brick_ptr_bytes = bytes_of(&new_brick_ptr);

    unsafe {
        mapped_ptr
            .add(cpu_offset) // this_brick_byte_offset.try_into().unwrap())
            .copy_from_nonoverlapping(brick_ptr_bytes.as_ptr(), brick_ptr_bytes.len())
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

    #[test]
    fn assert_values()
    {
        assert_matches!(
            VoxelBrickPointer::new_null().classify(),
            VoxelBrickPointerType::Null
        );

        for i in 1..4
        {
            let v = Voxel::try_from(i).unwrap();

            assert_matches!(
                VoxelBrickPointer::new_voxel(v).classify(),
                VoxelBrickPointerType::Voxel(v)
            );
        }

        for i in 1..100
        {
            assert_matches!(
                VoxelBrickPointer::new_ptr(i).classify(),
                VoxelBrickPointerType::ValidBrickPointer(i)
            );
        }

        for i in (VoxelBrickPointer::POINTER_TO_VOXEL_BOUND - 100)
            ..VoxelBrickPointer::POINTER_TO_VOXEL_BOUND
        {
            assert_matches!(
                VoxelBrickPointer::new_ptr(i).classify(),
                VoxelBrickPointerType::ValidBrickPointer(i)
            );
        }
    }
}
