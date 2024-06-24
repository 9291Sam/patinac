use bytemuck::{Pod, Zeroable};

use crate::ChunkCoordinate;

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub(crate) struct ChunkMetaData
{
    pub(crate) coordinate:      ChunkCoordinate,
    pub(crate) bool_is_visible: u32
}
