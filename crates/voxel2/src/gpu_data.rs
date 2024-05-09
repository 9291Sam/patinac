use gfx::{glm, wgpu};
use nonmax::NonMaxU32;

#[repr(C, align(16))]
/// The key raytracing primitive that this voxel scheme uses, the bit brick
/// this is an 8^3 array of data which is an occupancy map of all of the voxels
/// stored.
struct BitBrick
{
    data: [u32; 16]
}

impl BitBrick
{
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
    pub fn write(&mut self, pos: [u8; 3], occupied: bool)
    {
        let (idx, bit) = Self::calculate_position(pos);

        if occupied
        {
            self.data[idx] |= (1 << bit);
        }
        else
        {
            self.data[idx] &= (1 << bit);
        }
    }

    pub fn read(&self, pos: [u8; 3]) -> bool
    {
        let (idx, bit) = Self::calculate_position(pos);

        (self.data[idx] & (1 << bit)) != 0
    }

    // returns index and offset
    fn calculate_position(pos: [u8; 3]) -> (usize, u32)
    {
        let [x, y, z] = pos;

        debug_assert!(x < 8, "Out of range access @ {x}");
        debug_assert!(y < 8, "Out of range access @ {y}");
        debug_assert!(z < 8, "Out of range access @ {z}");

        let [x, y, z] = [x as usize, y as usize, z as usize];

        // 8 is the dimension of each axis
        let linear_index = x + y * 8 + z * 8 * 8;

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

struct ChunkFaceId {}

struct ChunkWorldId {}

struct VoxelFacePoint
{
    ///   x [0]    y [1]
    /// [0,   7] [      ] | 8 + 0 bits  | chunk_x_pos
    /// [8,  15] [      ] | 8 + 0 bits  | chunk_y_pos
    /// [16, 23] [      ] | 8 + 0 bits  | chunk_z_pos
    /// [24, 31] [0,  15] | 8 + 16 bits | chunk_face_id
    /// [      ] [16, 31] | 0 + 16 bits | chunk_world_id
    data: glm::U32Vec3
}

impl VoxelFacePoint
{
    pub fn new(pos: [u8; 3]) -> Self {}

    pub fn destructure(self) -> (glm::U8Vec3, ChunkFaceId, ChunkWorldId) {}
}

struct VoxelFace {}

#[repr(C, packed)]
struct FaceDataBuffer
{
    is_visible: u8,
    mat:        u16
}

#[cfg(test)]
mod test
{
    use super::*;

    // TODO: bit brick, bit brick index, chunk

    #[test]
    pub fn test_bit_brick_create()
    {
        let filled_brick = BitBrick::new(true);
    }

    pub fn test_bit_brick_read_write() {}
}
