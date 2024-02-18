use super::Voxel;
struct Brick
{
    data: [[[Voxel; 8]; 8]; 8]
}

#[cfg(test)]
mod test
{
    pub use super::*;

    #[test]
    fn assert_sizes()
    {
        assert_eq!(1024, std::mem::size_of::<Brick>());
    }
}
