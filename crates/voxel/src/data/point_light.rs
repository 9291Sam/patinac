use gfx::glm;

struct PointLight
{
    color_and_power: glm::Vec4,
    // x - constant
    // y - linear
    // z - quadratic
    // w - cubic
    falloffs:        glm::Vec4
}

impl PointLight {}
