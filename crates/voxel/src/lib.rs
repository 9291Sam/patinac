#![feature(exclusive_range_pattern)]
#![feature(assert_matches)]

mod brick_map_chunk;
mod gpu_data;

pub use brick_map_chunk::BrickMapChunk;
pub use gpu_data::{ChunkPosition, Voxel, VoxelBrick, VoxelChunkDataManager};
