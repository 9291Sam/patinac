#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]

use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

mod game;
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
    let renderer = gfx::Renderer::new();
    let game = game::Game::new(&renderer);

    std::thread::scope(|s| {
        s.spawn(|| game.enter_tick_loop(&should_stop));

        renderer.enter_gfx_loop(&should_stop);
    });

    LOGGER.get().unwrap().stop_worker();
}
