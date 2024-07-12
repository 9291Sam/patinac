use bytemuck::{Pod, Zeroable};
use gfx::glm;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RenderedFaceInfo
{
    pub data: glm::U32Vec3
}
