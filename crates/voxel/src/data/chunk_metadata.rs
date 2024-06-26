use super::VoxelFace;
use crate::suballocated_buffer::SubAllocatedCpuTrackedDenseSet;
use crate::ChunkCoordinate;

pub(crate) struct ChunkMetaData
{
    pub(crate) coordinate: ChunkCoordinate,
    pub(crate) is_visible: bool,
    pub(crate) faces:      [SubAllocatedCpuTrackedDenseSet<VoxelFace>; 6]
}

impl Clone for ChunkMetaData
{
    fn clone(&self) -> Self
    {
        log::error!("Dont try to clone a ChunkMetaData");

        todo!()
    }
}
