use gfx::glm;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoxelFaceDirection
{
    Top    = 0,
    Bottom = 1,
    Left   = 2,
    Right  = 3,
    Front  = 4,
    Back   = 5
}

impl TryFrom<u8> for VoxelFaceDirection
{
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error>
    {
        use VoxelFaceDirection::*;

        match value
        {
            0 => Ok(Top),
            1 => Ok(Bottom),
            2 => Ok(Left),
            3 => Ok(Right),
            4 => Ok(Front),
            5 => Ok(Back),
            _ => Err(value)
        }
    }
}

impl VoxelFaceDirection
{
    pub fn iterate() -> impl Iterator<Item = VoxelFaceDirection>
    {
        [
            VoxelFaceDirection::Top,
            VoxelFaceDirection::Bottom,
            VoxelFaceDirection::Left,
            VoxelFaceDirection::Right,
            VoxelFaceDirection::Front,
            VoxelFaceDirection::Back
        ]
        .into_iter()
    }

    pub fn get_axis(self) -> glm::I16Vec3
    {
        match self
        {
            VoxelFaceDirection::Top => glm::I16Vec3::new(0, 1, 0),
            VoxelFaceDirection::Bottom => glm::I16Vec3::new(0, -1, 0),
            VoxelFaceDirection::Left => glm::I16Vec3::new(-1, 0, 0),
            VoxelFaceDirection::Right => glm::I16Vec3::new(1, 0, 0),
            VoxelFaceDirection::Front => glm::I16Vec3::new(0, 0, -1),
            VoxelFaceDirection::Back => glm::I16Vec3::new(0, 0, 1)
        }
    }

    pub fn opposite(self) -> VoxelFaceDirection
    {
        match self
        {
            VoxelFaceDirection::Top => VoxelFaceDirection::Bottom,
            VoxelFaceDirection::Bottom => VoxelFaceDirection::Top,
            VoxelFaceDirection::Left => VoxelFaceDirection::Right,
            VoxelFaceDirection::Right => VoxelFaceDirection::Left,
            VoxelFaceDirection::Front => VoxelFaceDirection::Back,
            VoxelFaceDirection::Back => VoxelFaceDirection::Front
        }
    }
}
