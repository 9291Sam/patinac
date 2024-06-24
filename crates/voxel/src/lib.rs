use gfx::glm;

mod chunk;
mod material;

pub use chunk::{Chunk, ChunkPool};
pub use material::Voxel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct ChunkLocalPosition(pub glm::U8Vec3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoordinate(pub glm::I32Vec3);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct BrickCoordinate(pub glm::U8Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct BrickLocalPosition(pub glm::U8Vec3);
