use std::collections::{BTreeMap, HashMap};

use gfx::glm;
use smallvec::SmallVec;

use crate::chunk_brick_manager::{ChunkBrickManager, ChunkId};
use crate::face_manager::{FaceId, FaceManager};
use crate::material::Voxel;
use crate::{world_position_to_chunk_position, ChunkCoordinate, WorldPosition};

pub struct VoxelWorld
{
    rendering_dispatcher: (),
    face_manager:         FaceManager,
    chunk_brick_manager:  ChunkBrickManager,

    // TODO: there's a better way to do this...
    voxel_face_cache:   BTreeMap<WorldPosition, SmallVec<[FaceId; 6]>>,
    chunk_position_ids: HashMap<ChunkCoordinate, ChunkId>
}

impl VoxelWorld
{
    pub fn new() -> Self
    {
        todo!()
    }

    pub fn write_voxel(&self, pos: WorldPosition, voxel: Voxel) -> Voxel
    {
        // insert voxel into chunk, check adjacent points for other things, fix
        // up the faces as required

        todo!()
    }

    pub fn read_voxel(&self, pos: WorldPosition) -> Option<Voxel>
    {
        let (chunk_coord, chunk_local_pos) = world_position_to_chunk_position(pos);

        if let Some(chunk_id) = self.chunk_position_ids.get(&chunk_coord)
        {
            self.chunk_brick_manager
                .read_voxel(*chunk_id, chunk_local_pos)
        }
        else
        {
            Some(Voxel::Air)
        }
    }
}
// read write fns
// write face, lookup what faces to remove, remove them write in new ones, send
// writes to chunks
