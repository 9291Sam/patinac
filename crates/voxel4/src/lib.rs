#![feature(slice_flatten)]

use gfx::glm;

mod chunk_brick_manager;
mod face_manager;
mod gpu_data;
mod material;
mod voxel_rendering_dispatcher;
mod voxel_world;

const CHUNK_EDGE_LEN_VOXELS: usize = 512;
const BRICK_EDGE_LEN_VOXELS: usize = 8;
const CHUNK_EDGE_LEN_BRICKS: usize = CHUNK_EDGE_LEN_VOXELS / BRICK_EDGE_LEN_VOXELS;

const BRICK_TOTAL_VOXELS: usize = BRICK_EDGE_LEN_VOXELS.pow(3);
const VISIBILITY_BRICK_U32S_REQUIRED: usize = BRICK_TOTAL_VOXELS / u32::BITS as usize;

const _: () =
    const { assert!(CHUNK_EDGE_LEN_BRICKS * BRICK_EDGE_LEN_VOXELS == CHUNK_EDGE_LEN_VOXELS) };
const _: () =
    const { assert!(VISIBILITY_BRICK_U32S_REQUIRED * u32::BITS as usize == BRICK_TOTAL_VOXELS) };

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct WorldPosition(pub glm::I32Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct ChunkCoordinate(pub glm::I32Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct ChunkLocalPosition(pub glm::U16Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub struct BrickCoordinate(pub glm::U16Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct BrickLocalPosition(pub glm::U8Vec3);

pub fn world_position_to_chunk_position(
    WorldPosition(world_pos): WorldPosition
) -> (ChunkCoordinate, ChunkLocalPosition)
{
    (
        ChunkCoordinate(world_pos.map(|x| x.div_euclid(CHUNK_EDGE_LEN_VOXELS as i32))),
        ChunkLocalPosition(
            world_pos
                .map(|x| x.rem_euclid(CHUNK_EDGE_LEN_VOXELS as i32))
                .try_cast()
                .unwrap()
        )
    )
}

pub fn get_world_offset_of_chunk(
    ChunkCoordinate(chunk_coordinate): ChunkCoordinate
) -> WorldPosition
{
    WorldPosition(chunk_coordinate.map(|x| x * CHUNK_EDGE_LEN_VOXELS as i32))
}

pub fn chunk_local_position_to_brick_position(
    ChunkLocalPosition(local_chunk_pos): ChunkLocalPosition
) -> (BrickCoordinate, BrickLocalPosition)
{
    (
        BrickCoordinate(local_chunk_pos.map(|x| x.div_euclid(BRICK_EDGE_LEN_VOXELS as u16))),
        BrickLocalPosition(
            local_chunk_pos
                .map(|x| x.rem_euclid(BRICK_EDGE_LEN_VOXELS as u16))
                .try_cast()
                .unwrap()
        )
    )
}
