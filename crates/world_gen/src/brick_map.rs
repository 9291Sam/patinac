use std::array::from_fn;

#[repr(C)]
struct Brick
{
    bricks: [[[super::Voxel; 8]; 8]; 8]
}

impl Brick
{
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
