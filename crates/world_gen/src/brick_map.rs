use std::array::from_fn;

#[repr(C)]
struct Brick
{
    bricks: [[[super::Voxel; Self::SIDE_LENGTH_VOXELS]; Self::SIDE_LENGTH_VOXELS];
        Self::SIDE_LENGTH_VOXELS]
}

impl Brick
{
    pub const SIDE_LENGTH_VOXELS: usize = 8;

    pub fn new_solid(voxel: super::Voxel) -> Brick
    {
        Brick {
            bricks: [[[voxel; 8]; 8]; 8]
        }
    }

    // function is called with args
    pub fn new_fill(mut fill_func: impl FnMut(usize, usize, usize) -> super::Voxel) -> Brick
    {
        Brick {
            bricks: from_fn(|x| from_fn(|y| from_fn(|z| fill_func(x, y, z))))
        }
    }
}

struct BrickMap
{
    tracking_array: Box<
        [[[Brick; Self::SIDE_LENGTH_BRICKS]; Self::SIDE_LENGTH_BRICKS]; Self::SIDE_LENGTH_BRICKS]
    >,
    brick_storage:  Vec<Brick>
}

impl BrickMap
{
    pub const SIDE_LENGTH_BRICKS: usize = 256;

    pub fn new() {}
}
