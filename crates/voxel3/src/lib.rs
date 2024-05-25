#![feature(map_try_insert)]
mod chunk_manager;
mod face_manager;
mod material;
mod voxel_world;

use std::fmt::Debug;

use gfx::glm;
pub use voxel_world::VoxelWorld;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldPosition(pub glm::I32Vec3);

impl Debug for WorldPosition
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        write!(f, "[{}, {}, {}]", self.0.x, self.0.y, self.0.z)
    }
}

impl Ord for WorldPosition
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering
    {
        std::cmp::Ordering::Equal
            .then(self.0.x.cmp(&other.0.x))
            .then(self.0.y.cmp(&other.0.y))
            .then(self.0.z.cmp(&other.0.z))
    }
}

impl PartialOrd for WorldPosition
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering>
    {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct ChunkCoordinate(pub glm::I32Vec3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct ChunkLocalPosition(pub glm::U16Vec3);

pub fn get_chunk_position_from_world(
    WorldPosition(world_pos): WorldPosition
) -> (ChunkCoordinate, ChunkLocalPosition)
{
    (
        ChunkCoordinate(world_pos.map(|w| w.div_euclid(512))),
        ChunkLocalPosition(world_pos.map(|w| w.rem_euclid(512)).try_cast().unwrap())
    )
}

pub fn get_world_position_from_chunk(chunk_coord: ChunkCoordinate) -> WorldPosition
{
    WorldPosition(chunk_coord.0.map(|p| p * 512).cast())
}
