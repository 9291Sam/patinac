use std::num::NonZeroU32;

use strum::{EnumIter, IntoEnumIterator};

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

pub type BrickStorageBuffer = [Brick; 131072];
pub type BrickPointer = NonZeroU32;
pub type MaybeBrickPointer = Option<NonZeroU32>;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chunk
{
    data: [[[MaybeBrickPointer; Self::SIDE_BRICKS]; Self::SIDE_BRICKS]; Self::SIDE_BRICKS]
}

impl Chunk
{
    pub const SIDE_BRICKS: usize = 64;
}

pub type ChunkStorageBuffer = [Chunk; 128];
pub type ChunkPointer = u32;

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
