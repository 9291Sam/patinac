use std::sync::atomic::AtomicBool;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

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
        let all_u8s_rev: [u8; u8::MAX as usize + 1] =
            std::array::from_fn(|i| u8::wrapping_sub(1, i as u8).wrapping_sub(2));

        self.renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("wgpu test buffer"),
            contents: &all_u8s_rev,
            usage:    wgpu::BufferUsages::STORAGE
        });

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
