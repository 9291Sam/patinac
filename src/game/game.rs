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
        let all_u8s: [u8; u8::MAX as usize + 1] =
            std::array::from_fn(|i| u8::wrapping_add(255, i as u8));

        self.renderer.create_buffer_init(&BufferInitDescriptor {
            label:    Some("wgpu test buffer"),
            contents: &all_u8s,
            usage:    wgpu::BufferUsages::STORAGE
        });

        // self.renderer.create_texture_with_data(
        //     &self.renderer.queue,
        //     &wgpu::TextureDescriptor {
        //         label:           todo!(),
        //         size:            todo!(),
        //         mip_level_count: todo!(),
        //         sample_count:    todo!(),
        //         dimension:       todo!(),
        //         format:          todo!(),
        //         usage:           todo!(),
        //         view_formats:    todo!()
        //     },
        //     wgpu::util::TextureDataOrder::MipMajor,
        //     &[]
        // );

        while !should_stop.load(std::sync::atomic::Ordering::Acquire)
        {}
    }
}
