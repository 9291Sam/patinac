use bytemuck::{Pod, Zeroable};
use gfx::glm;

use crate::ChunkLocalPosition;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VoxelFace
{
    // [0,  7]  - x pos
    // [8, 15]  - y pos
    // [16, 23] - z pos
    // [24, 27] - breadth width
    // [28, 31] - breadth height
    data: u32
}

impl VoxelFace
{
    pub fn new(ChunkLocalPosition(pos): ChunkLocalPosition, breadth: glm::U8Vec2) -> Self
    {
        debug_assert!(breadth.x < 16);
        debug_assert!(breadth.y < 16);

        VoxelFace {
            data: pos.x as u32
                | (pos.y as u32) << 8
                | (pos.z as u32) << 16
                | (breadth.x as u32) << 24
                | (breadth.y as u32) << 28
        }
    }

    pub fn destructure(self) -> (ChunkLocalPosition, glm::U8Vec2)
    {
        (
            ChunkLocalPosition(glm::U8Vec3::new(
                (self.data & 0b1111_1111) as u8,
                ((self.data >> 8) & 0b1111_1111) as u8,
                ((self.data >> 16) & 0b1111_1111) as u8
            )),
            glm::U8Vec2::new(
                ((self.data >> 24) & 0b1111) as u8,
                ((self.data >> 28) & 0b1111) as u8
            )
        )
    }
}

#[cfg(test)]
mod test
{
    use gfx::glm;
    use itertools::iproduct;

    use super::VoxelFace;

    #[test]
    pub fn test_construct_destroy()
    {
        for (x, y, z, w, h) in iproduct!(0..=255, 0..=255, 0..=255, 0..=15, 0..=15)
        {
            let (p, b) = VoxelFace::new(
                crate::ChunkLocalPosition(glm::U8Vec3::new(x, y, z)),
                glm::U8Vec2::new(w, h)
            )
            .destructure();

            assert!(p.0.x == x);
            assert!(p.0.y == y);
            assert!(p.0.z == z);
            assert!(b.x == w);
            assert!(b.y == h);
        }
    }
}
