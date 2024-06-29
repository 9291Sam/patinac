use super::VoxelFace;
use crate::suballocated_buffer::SubAllocatedCpuTrackedDenseSet;
use crate::ChunkCoordinate;

pub(crate) struct CpuChunkData
{
    pub(crate) coordinate: ChunkCoordinate,
    pub(crate) is_visible: bool,
    pub(crate) faces:      [SubAllocatedCpuTrackedDenseSet<VoxelFace>; 6]
}

impl Clone for CpuChunkData
{
    fn clone(&self) -> Self
    {
        log::error!("Dont try to clone a CpuChunkData");

        todo!()
    }
}
