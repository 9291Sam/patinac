use std::sync::atomic::AtomicBool;

use nalgebra::UnitQuaternion;
use nalgebra_glm as glm;

use crate::gfx::{self};
pub struct Game<'r>
{
    renderer: &'r gfx::Renderer
}

impl<'r> Game<'r>
{
    pub fn new(renderer: &'r gfx::Renderer) -> Self
    {
        Game {
            renderer
        }
    }

    pub fn enter_tick_loop(&self, should_stop: &AtomicBool)
    {
        let mut objs = Vec::new();

        for x in -5..=5
        {
            for z in -5..=5
            {
                let vec = glm::Vec3::new(x as f32, 0.0, z as f32);

                objs.push(gfx::flat_textured::FlatTextured::new(
                    self.renderer,
                    vec,
                    gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
                    gfx::flat_textured::FlatTextured::PENTAGON_INDICES
                ));
            }
        }

        let lit_textured = gfx::lit_textured::LitTextured::new_cube(
            self.renderer,
            gfx::Transform {
                translation: glm::Vec3::new(0.0, 5.0, 0.0),
                rotation:    UnitQuaternion::from_axis_angle(
                    &gfx::Transform::global_up_vector(),
                    0.0
                ),
                scale:       glm::Vec3::repeat(4.0)
            }
        );

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
