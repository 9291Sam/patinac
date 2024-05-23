mod chunk_manager;
mod face_manager;
mod material;
mod voxel_world;

use gfx::glm;
pub use voxel_world::VoxelWorld;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct WorldPosition(pub glm::I32Vec3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct ChunkCoordinate(pub glm::I32Vec3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct ChunkLocalPosition(pub glm::U16Vec3);

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
