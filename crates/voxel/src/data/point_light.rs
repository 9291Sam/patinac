use bytemuck::{AnyBitPattern, NoUninit};
use gfx::glm;

#[repr(C)]
#[derive(Clone, Copy, Debug, AnyBitPattern, NoUninit)]
pub struct PointLight
{
    pub position:        glm::Vec4,
    pub color_and_power: glm::Vec4 /* x - constant
                                    * y - linear
                                    * z - quadratic
                                    * w - cubic
                                    * falloffs:        glm::Vec4 TODO */
}

impl PointLight {}
