#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(effects)]

use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

mod entity;
mod game;

static LOGGER: OnceLock<util::AsyncLogger> = OnceLock::new();

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "1");

    // Initialize logger
    LOGGER.set(util::AsyncLogger::new()).unwrap();
    log::set_logger(LOGGER.get().unwrap()).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    let renderer = gfx::Renderer::new();
    let game = game::Game::new(&renderer);

    let should_stop = AtomicBool::new(false);

    std::thread::scope(|s| {
        s.spawn(|| game.enter_tick_loop(&should_stop));

        renderer.enter_gfx_loop(&should_stop);
    });

    LOGGER.get().unwrap().stop_worker();
}
