#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]

use std::ptr::{addr_of, addr_of_mut};
use std::sync::atomic::{AtomicBool, AtomicPtr};
use std::sync::OnceLock;

mod gfx;
mod util;

static LOGGER: OnceLock<util::AsyncLogger> = OnceLock::new();

fn main()
{
    // Initialize logger
    LOGGER.set(util::AsyncLogger::new()).unwrap();
    log::set_logger(LOGGER.get().unwrap()).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    let should_stop = AtomicBool::new(false);

    std::thread::scope(|s| {
        let renderer = gfx::Renderer::new();
        // let game = game::Game::new();

        // s.spawn(|| game.enter_tick_loop(&should_stop))

        renderer.enter_gfx_loop(&should_stop);

        renderer
            .get_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label:              Some("label"),
                size:               4096,
                usage:              wgpu::BufferUsages::INDEX,
                mapped_at_creation: false
            });
    });

    LOGGER.get().unwrap().stop_worker();
}
