#![feature(map_entry_replace)]
#![feature(new_uninit)]
#![feature(iter_array_chunks)]

use bytemuck::{Pod, Zeroable};
use gfx::glm;

mod chunk_pool;
mod data;
mod suballocated_buffer;

pub use chunk_pool::{Chunk, ChunkPool};
pub use data::Voxel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkLocalPosition(pub glm::U8Vec3);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Zeroable, Pod)]
#[repr(C)]
pub struct ChunkCoordinate(pub glm::I32Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct BrickCoordinate(pub glm::U8Vec3);
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct BrickLocalPosition(pub glm::U8Vec3);

const CHUNK_EDGE_LEN_VOXELS: usize = 256;
const BRICK_EDGE_LEN_VOXELS: usize = 8;
const CHUNK_EDGE_LEN_BRICKS: usize = CHUNK_EDGE_LEN_VOXELS / BRICK_EDGE_LEN_VOXELS;

const BRICK_TOTAL_VOXELS: usize = BRICK_EDGE_LEN_VOXELS.pow(3);
const VISIBILITY_BRICK_U32S_REQUIRED: usize = BRICK_TOTAL_VOXELS / u32::BITS as usize;

#[allow(clippy::assertions_on_constants)]
const _: () =
    const { assert!(CHUNK_EDGE_LEN_BRICKS * BRICK_EDGE_LEN_VOXELS == CHUNK_EDGE_LEN_VOXELS) };
const _: () =
    const { assert!(VISIBILITY_BRICK_U32S_REQUIRED * u32::BITS as usize == BRICK_TOTAL_VOXELS) };

impl BrickLocalPosition
{
    #[inline(never)]
    pub fn iter() -> &'static [BrickLocalPosition]
    {
        const DATA: [BrickLocalPosition; BRICK_TOTAL_VOXELS] = const {
            let mut out = [BrickLocalPosition(glm::U8Vec3::new(0, 0, 0)); BRICK_TOTAL_VOXELS];
            let mut idx = 0;

            loop
            {
                if idx == BRICK_TOTAL_VOXELS
                {
                    break;
                }

                out[idx] = BrickLocalPosition(glm::U8Vec3::new(
                    (idx / (BRICK_EDGE_LEN_VOXELS * BRICK_EDGE_LEN_VOXELS)) as u8,
                    (idx / BRICK_EDGE_LEN_VOXELS) as u8,
                    (idx % BRICK_EDGE_LEN_VOXELS) as u8
                ));
                idx += 1;
            }

            out
        };

        &DATA
    }
}

// pub(crate) fn world_position_to_chunk_position(
//     WorldPosition(world_pos): WorldPosition
// ) -> (ChunkCoordinate, ChunkLocalPosition)
// {
//     (
//         ChunkCoordinate(world_pos.map(|x| x.div_euclid(CHUNK_EDGE_LEN_VOXELS
// as i32))),         ChunkLocalPosition(
//             world_pos
//                 .map(|x| x.rem_euclid(CHUNK_EDGE_LEN_VOXELS as i32))
//                 .try_cast()
//                 .unwrap()
//         )
//     )
// }

pub(crate) fn get_world_offset_of_chunk(
    ChunkCoordinate(chunk_coordinate): ChunkCoordinate
) -> glm::Vec3
{
    chunk_coordinate
        .map(|x| x * CHUNK_EDGE_LEN_VOXELS as i32)
        .cast()
}

pub(crate) fn chunk_local_position_from_brick_positions(
    coordinate: BrickCoordinate,
    position: BrickLocalPosition
) -> ChunkLocalPosition
{
    ChunkLocalPosition(position.0 + coordinate.0 * BRICK_EDGE_LEN_VOXELS as u8)
}

pub(crate) fn chunk_local_position_to_brick_position(
    ChunkLocalPosition(local_chunk_pos): ChunkLocalPosition
) -> (BrickCoordinate, BrickLocalPosition)
{
    (
        BrickCoordinate(local_chunk_pos.map(|x| x.div_euclid(BRICK_EDGE_LEN_VOXELS as u8))),
        BrickLocalPosition(
            local_chunk_pos
                .map(|x| x.rem_euclid(BRICK_EDGE_LEN_VOXELS as u8))
                .try_cast()
                .unwrap()
        )
    )
}
