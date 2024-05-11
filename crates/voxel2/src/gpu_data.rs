use bytemuck::{Pod, Zeroable};
use gfx::{glm, nal};

#[repr(C, align(16))]
/// The key raytracing primitive that this voxel scheme uses, the bit brick
/// this is an 8^3 array of data which is an occupancy map of all of the voxels
/// stored.
#[derive(Debug)]
struct BitBrick
{
    data: [u32; 16]
}

impl BitBrick
{
    const EDGE_LEN: usize = 8;

    pub fn new(filled: bool) -> Self
    {
        let fill_val = if filled
        {
            !0
        }
        else
        {
            0
        };

        BitBrick {
            data: [fill_val; 16]
        }
    }

    #[allow(unused_parens)]
    pub fn write(&mut self, pos: glm::U8Vec3, occupied: bool)
    {
        let (idx, bit) = Self::calculate_position(pos);

        if occupied
        {
            self.data[idx] |= (1 << bit);
        }
        else
        {
            self.data[idx] &= !(1 << bit);
        }
    }

    pub fn read(&self, pos: glm::U8Vec3) -> bool
    {
        let (idx, bit) = Self::calculate_position(pos);

        (self.data[idx] & (1 << bit)) != 0
    }

    // returns index and offset
    fn calculate_position(pos: glm::U8Vec3) -> (usize, u32)
    {
        let [x, y, z] = [pos.x as usize, pos.y as usize, pos.z as usize];

        debug_assert!(x < 8, "Out of range access @ {x}");
        debug_assert!(y < 8, "Out of range access @ {y}");
        debug_assert!(z < 8, "Out of range access @ {z}");

        // 8 is the dimension of each axis
        // !NOTE: the order is like this so that the cache lines
        // !are aligned vertically i.e the bottom half is one cache line and the top is
        // another
        let linear_index = x + z * 8 + y * 8 * 8;

        // 32 is the number of bits in a u32
        let index = linear_index / 32;
        let bit = linear_index % 32;

        (index, bit as u32)
    }
}

const _: () = const { assert!((std::mem::size_of::<BitBrick>() * 8) == 512) };

// As of now, each voxel face is 48 bytes
// by doing some bs with the vertex and index buffers, using the same trick dot
// taught you about, you can get this down to 30 bytes
// 0                         = 8 bytes
// 0, 0, 0, 0, 0, 0 = 4 * 6 = 24 bytes = 30 bytes
// also instancing

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct FaceId(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct VoxelFacePoint
{
    ///   x [0]    y [1]
    /// [0,   7] [      ] | 8 + 0 bits  | chunk_x_pos
    /// [8,  15] [      ] | 8 + 0 bits  | chunk_y_pos
    /// [16, 23] [      ] | 8 + 0 bits  | chunk_z_pos
    /// [24, 31] [      ] | 8 + 0 bits  | normal
    /// [      ] [0,  31] | 0 + 32 bits | face_id
    data: glm::U32Vec2
}

impl VoxelFacePoint
{
    pub fn new(pos: glm::U8Vec3, face_id: FaceId) -> Self
    {
        let FaceId(face_id) = face_id;

        let low: u32 = (pos.x as u32) | ((pos.y as u32) << 8) | ((pos.z as u32) << 16);

        let high: u32 = face_id;

        Self {
            data: glm::U32Vec2::new(low, high)
        }
    }

    pub fn destructure(self) -> (glm::U8Vec3, FaceId)
    {
        let low = self.data.x;
        let high = self.data.y;

        let pos = glm::U8Vec3::new(
            (low & 0xFF) as u8,
            ((low >> 8) & 0xFF) as u8,
            ((low >> 16) & 0xFF) as u8
        );

        (pos, FaceId(high))
    }
}

struct VoxelFace {}

#[repr(C, align(8))]
pub(crate) struct FaceInfo
{
    ///   x [0]    y [1]
    /// [0,  15] [      ] | chunk_id
    /// [16, 31] [      ] | material
    /// [      ] [0,   0] | is_visible
    /// [      ] [1,  31] | unused
    data: glm::U32Vec2
}

#[cfg(test)]
mod test
{
    use itertools::iproduct;
    use rand::Rng;

    use super::*;

    #[test]
    pub fn test_bit_brick_create()
    {
        {
            let filled_brick = BitBrick::new(true);

            for (x, y, z) in iproduct!(
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN
            )
            {
                assert!(
                    filled_brick.read(glm::U8Vec3::new(x as u8, y as u8, z as u8)),
                    "{x}{y}{z} {filled_brick:?}"
                );
            }
        }

        {
            let empty_brick = BitBrick::new(false);

            for (x, y, z) in iproduct!(
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN
            )
            {
                assert!(
                    !empty_brick.read(glm::U8Vec3::new(x as u8, y as u8, z as u8)),
                    "{x}{y}{z} {empty_brick:?}"
                );
            }
        }
    }

    #[test]
    pub fn test_bit_brick_read_write()
    {
        let mut working_brick = BitBrick::new(false);

        for (x1, y1, z1) in iproduct!(
            0..BitBrick::EDGE_LEN,
            0..BitBrick::EDGE_LEN,
            0..BitBrick::EDGE_LEN
        )
        {
            let vec1 = glm::U8Vec3::new(x1 as u8, y1 as u8, z1 as u8);

            assert!(!working_brick.read(vec1));

            working_brick.write(vec1, true);

            assert!(working_brick.read(vec1));

            println!("{working_brick:?}");

            for (x2, y2, z2) in iproduct!(
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN,
                0..BitBrick::EDGE_LEN
            )
            {
                let vec2 = glm::U8Vec3::new(x2 as u8, y2 as u8, z2 as u8);

                if vec1 != vec2
                {
                    assert!(
                        !working_brick.read(vec2),
                        "{:?} {:?} {:?}",
                        vec1,
                        vec2,
                        working_brick
                    );
                }
            }

            working_brick.write(vec1, false);
        }
    }

    #[test]
    pub fn test_voxel_face_point()
    {
        for _ in 0..32832487
        {
            let (x, y, z, f) = (
                rand::thread_rng().gen_range(0..=u8::MAX),
                rand::thread_rng().gen_range(0..=u8::MAX),
                rand::thread_rng().gen_range(0..=u8::MAX),
                rand::thread_rng().gen_range(0..=u32::MAX)
            );

            let (vec, ff) = VoxelFacePoint::new(glm::U8Vec3::new(x, y, z), FaceId(f)).destructure();

            assert!(vec.x == x);
            assert!(vec.y == y);
            assert!(vec.z == z);
            assert!(ff.0 == f);
        }
    }
}
