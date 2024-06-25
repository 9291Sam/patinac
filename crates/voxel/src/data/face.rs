use bytemuck::{Pod, Zeroable};
use gfx::glm;

use crate::ChunkLocalPosition;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoxelFace
{
    // [0,  7]  - x pos
    // [8, 15]  - y pos
    // [16, 23] - z pos
    // [24, 31] - unused
    data: u32
}

impl VoxelFace
{
    pub fn new(ChunkLocalPosition(pos): ChunkLocalPosition) -> Self
    {
        VoxelFace {
            data: pos.x as u32 | (pos.y as u32) << 8 | (pos.z as u32) << 16
        }
    }

    pub fn destructure(self) -> ChunkLocalPosition
    {
        ChunkLocalPosition(glm::U8Vec3::new(
            (self.data & 0b1111_1111) as u8,
            ((self.data >> 8) & 0b1111_1111) as u8,
            ((self.data >> 16) & 0b1111_1111) as u8
        ))
    }
}

#[cfg(test)]
mod test
{
    use gfx::glm;
    use itertools::iproduct;

    use super::VoxelFace;
    use crate::ChunkLocalPosition;

    #[test]
    pub fn test_construct_destroy()
    {
        for (x, y, z) in iproduct!(0..=255, 0..=255, 0..=255)
        {
            let ChunkLocalPosition(vec) =
                VoxelFace::new(crate::ChunkLocalPosition(glm::U8Vec3::new(x, y, z))).destructure();

            assert!(vec.x == x);
            assert!(vec.y == y);
            assert!(vec.z == z);
        }
    }
}
