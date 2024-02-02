use std::sync::atomic::AtomicBool;

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
        let _obj = gfx::flat_textured::FlatTextured::new(
            self.renderer,
            gfx::flat_textured::FlatTextured::PENTAGON_VERTICES,
            gfx::flat_textured::FlatTextured::PENTAGON_INDICES
        );

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
