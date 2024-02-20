#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

mod recordables;
mod test_scene;

static LOGGER: OnceLock<util::AsyncLogger> = OnceLock::new();

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    // Initialize logger
    LOGGER.set(util::AsyncLogger::new()).unwrap();
    log::set_logger(LOGGER.get().unwrap()).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    // Safety: we try our best to drop the Renderer on this thread
    let renderer = Arc::new(unsafe { gfx::Renderer::new() });
    {
        let game = game::Game::new(renderer.clone());

        let should_stop = AtomicBool::new(false);

        std::thread::scope(|s| {
            let _scene_guard = test_scene::TestScene::new(&game);

            s.spawn(|| game.enter_tick_loop(&should_stop));

            renderer.enter_gfx_loop(&should_stop);
        });
    }

    if Arc::into_inner(renderer).is_none()
    {
        log::warn!("Renderer was retained via Arc cycle! Drop is not possible");
    }

    LOGGER.get().unwrap().stop_worker();
}
