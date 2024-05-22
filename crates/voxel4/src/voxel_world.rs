use std::collections::BTreeMap;

use gfx::glm;
use smallvec::SmallVec;

use crate::WorldPosition;

pub struct VoxelWorld
{
    rendering_dispatcher: (),
    face_manager:         (),
    chunk_brick_manager:  ChunkBrickManager,

    voxel_face_cache: BTreeMap<WorldPosition, SmallVec<[FaceId; 6]>>
}

// read write fns
// write face, lookup what faces to remove, remove them write in new ones, send
// writes to chunks
