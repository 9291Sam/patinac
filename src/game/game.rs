use std::sync::atomic::AtomicBool;

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
            for y in -5..=5
            {
                for z in -5..=5
                {
                    let vec = glm::Vec3::new(x as f32, y as f32, z as f32);

                    objs.push(gfx::flat_textured::FlatTextured::new(
                        self.renderer,
                        vec,
                        gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
                        gfx::flat_textured::FlatTextured::PENTAGON_INDICES
                    ));
                }
            }
        }

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
