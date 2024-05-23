use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};

use crate::material::Voxel;
use crate::{BRICK_EDGE_LEN_VOXELS, CHUNK_EDGE_LEN_BRICKS, VISIBILITY_BRICK_U32S_REQUIRED};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct MaybeBrickPtr(pub(crate) u32);
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct BrickPtr(pub(crate) u32);

impl MaybeBrickPtr
{
    const NULL: MaybeBrickPtr = MaybeBrickPtr(u32::MAX);

    pub fn to_option(self) -> Option<BrickPtr>
    {
        if self.0 == Self::NULL.0
        {
            None
        }
        else
        {
            Some(BrickPtr(self.0))
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, AnyBitPattern, NoUninit)]
pub(crate) struct BrickMap
{
    pub(crate) brick_map:
        [[[MaybeBrickPtr; CHUNK_EDGE_LEN_BRICKS]; CHUNK_EDGE_LEN_BRICKS]; CHUNK_EDGE_LEN_BRICKS]
}

impl BrickMap
{
    pub fn null_all_ptrs(&mut self)
    {
        self.brick_map
            .flatten_mut()
            .flatten_mut()
            .iter_mut()
            .for_each(|p| *p = MaybeBrickPtr::NULL);
    }
}

impl Clone for BrickMap
{
    #[track_caller]
    fn clone(&self) -> Self
    {
        log::warn!(
            "Calling BrickMap::clone()! {}",
            std::panic::Location::caller()
        );

        Self {
            brick_map: self.brick_map.clone()
        }
    }
}
