use bytemuck::{AnyBitPattern, NoUninit};

use crate::{BrickLocalPosition, BRICK_EDGE_LEN_VOXELS, VISIBILITY_BRICK_U32S_REQUIRED};

#[repr(C)]
#[derive(Clone, Copy, Debug, AnyBitPattern, NoUninit)]
pub(crate) struct VisibilityBrick
{
    data: [u32; VISIBILITY_BRICK_U32S_REQUIRED]
}

impl VisibilityBrick
{
    pub fn new_empty() -> Self
    {
        Self {
            data: [0; VISIBILITY_BRICK_U32S_REQUIRED]
        }
    }

    pub fn is_visible(&self, local_pos: BrickLocalPosition) -> bool
    {
        let (idx, bit) = Self::calculate_position(local_pos);

        (self.data[idx] & (1 << bit)) != 0
    }

    #[allow(unused_parens)]
    pub fn set_visibility(&mut self, local_pos: BrickLocalPosition, occupied: bool)
    {
        let (idx, bit) = Self::calculate_position(local_pos);

        if occupied
        {
            self.data[idx] |= (1 << bit);
        }
        else
        {
            self.data[idx] &= !(1 << bit);
        }
    }

    fn calculate_position(BrickLocalPosition(local_pos): BrickLocalPosition) -> (usize, u32)
    {
        let [x, y, z] = [
            local_pos.x as usize,
            local_pos.y as usize,
            local_pos.z as usize
        ];

        debug_assert!(x < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {x}");
        debug_assert!(y < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {y}");
        debug_assert!(z < BRICK_EDGE_LEN_VOXELS, "Out of range access @ {z}");

        //// !NOTE: the order is like this so that the cache lines are aligned
        //// ! vertically
        // i.e the bottom half is one cache line and the top is another
        let linear_index =
            x + z * BRICK_EDGE_LEN_VOXELS + y * BRICK_EDGE_LEN_VOXELS * BRICK_EDGE_LEN_VOXELS;

        // 32 is the number of bits in a u32
        let index = linear_index / u32::BITS as usize;
        let bit = linear_index % u32::BITS as usize;

        (index, bit as u32)
    }
}
