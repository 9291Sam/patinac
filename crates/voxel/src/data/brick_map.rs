use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};
use gfx::glm;
use itertools::iproduct;

use crate::{BrickCoordinate, CHUNK_EDGE_LEN_BRICKS};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct MaybeBrickPtr(pub(crate) u32);
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct BrickPtr(pub(crate) u32);

impl MaybeBrickPtr
{
    pub const NULL: MaybeBrickPtr = MaybeBrickPtr(u32::MAX);

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
    pub fn new() -> BrickMap
    {
        BrickMap {
            brick_map: [[[MaybeBrickPtr::NULL; CHUNK_EDGE_LEN_BRICKS]; CHUNK_EDGE_LEN_BRICKS];
                CHUNK_EDGE_LEN_BRICKS]
        }
    }

    #[allow(unused)]
    pub fn null_all_ptrs(&mut self)
    {
        self.brick_map
            .as_flattened_mut()
            .as_flattened_mut()
            .iter_mut()
            .for_each(|p| *p = MaybeBrickPtr::NULL);
    }

    pub fn access_mut_all(&mut self, mut func: impl FnMut(BrickCoordinate, &mut MaybeBrickPtr))
    {
        for (x, y, z) in iproduct!(
            0..CHUNK_EDGE_LEN_BRICKS as u8,
            0..CHUNK_EDGE_LEN_BRICKS as u8,
            0..CHUNK_EDGE_LEN_BRICKS as u8
        )
        {
            let coord = BrickCoordinate(glm::U8Vec3::new(x, y, z));

            func(coord, self.get_mut(coord))
        }
    }

    #[inline(always)]
    pub fn get(&self, coord: BrickCoordinate) -> MaybeBrickPtr
    {
        debug_assert!((coord.0.x as usize) < CHUNK_EDGE_LEN_BRICKS);
        debug_assert!((coord.0.y as usize) < CHUNK_EDGE_LEN_BRICKS);
        debug_assert!((coord.0.z as usize) < CHUNK_EDGE_LEN_BRICKS);

        unsafe {
            *self
                .brick_map
                .get_unchecked(coord.0.x as usize)
                .get_unchecked(coord.0.y as usize)
                .get_unchecked(coord.0.z as usize)
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, coord: BrickCoordinate) -> &mut MaybeBrickPtr
    {
        debug_assert!((coord.0.x as usize) < CHUNK_EDGE_LEN_BRICKS);
        debug_assert!((coord.0.y as usize) < CHUNK_EDGE_LEN_BRICKS);
        debug_assert!((coord.0.z as usize) < CHUNK_EDGE_LEN_BRICKS);

        unsafe {
            self.brick_map
                .get_unchecked_mut(coord.0.x as usize)
                .get_unchecked_mut(coord.0.y as usize)
                .get_unchecked_mut(coord.0.z as usize)
        }
    }
}

impl Clone for BrickMap
{
    #[allow(clippy::non_canonical_clone_impl)]
    #[track_caller]
    fn clone(&self) -> Self
    {
        log::warn!(
            "Calling BrickMap::clone()! {}",
            std::panic::Location::caller()
        );

        Self {
            brick_map: self.brick_map
        }
    }
}
