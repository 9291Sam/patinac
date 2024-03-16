#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::{Arc, Mutex};

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    // Setup Logger
    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    // Setup threadpool
    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new());

    let crash_handler = util::CrashHandler::new();

    crash_handler.into_guarded_scope(|handle| {
        let renderer = handle.enter_constrained("Renderer Creation".to_string(), |_, _| {
            Arc::new(unsafe { gfx::Renderer::new() })
        });

        let game = handle.enter_constrained("Game Creation".to_string(), |_, _| {
            game::Game::new(renderer)
        });

        let _verdigris = handle.enter_constrained("Verdigris Creation".to_string(), |_, _| {
            verdigris::TestScene::new(game.clone())
        });

        handle.enter_constrained_thread("Game Tick Thread".to_string(), |continue_func, _| {
            game.enter_tick_loop(continue_func)
        });

        // handle.enter_constrained("Gfx Loop".to_string(), |_, _|
        // renderer.enter_gfx_loop());

        // TODO: check for retaining of renderer and game
        // TODO: remove arc now that the threads are scoped???
    });

    crash_handler.finish();

    util::access_global_thread_pool()
        .write()
        .unwrap()
        .take()
        .unwrap()
        .join_threads();

    logger.stop_worker();
}
