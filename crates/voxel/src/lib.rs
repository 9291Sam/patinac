#![feature(exclusive_range_pattern)]
#![feature(assert_matches)]

mod gpu_data;

pub use gpu_data::{ChunkPosition, Voxel, VoxelBrick, VoxelChunkDataManager};
