use bytemuck::{AnyBitPattern, NoUninit};
use gfx::glm;

#[repr(C)]
#[derive(Debug, Clone, Copy, AnyBitPattern, NoUninit)]
pub(crate) struct ChunkMetaData
{
    pub(crate) pos: glm::Vec4
}
