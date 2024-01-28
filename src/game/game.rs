use std::sync::atomic::AtomicBool;

use bevy_ecs::prelude::*;

use crate::gfx;
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
        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
