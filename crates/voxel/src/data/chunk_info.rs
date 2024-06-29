use bytemuck::{Pod, Zeroable};
use gfx::glm;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuChunkData
{
    pub position: glm::Vec4
}
