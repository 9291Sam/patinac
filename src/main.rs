#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
use std::sync::Arc;

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));

    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new());

    // Safety: we try our best to drop the Renderer on this thread
    let renderer = Arc::new(unsafe { gfx::Renderer::new() });

    let should_stop = AtomicBool::new(false);

    let run = || {
        let game = game::Game::new(renderer.clone());

        let _verdigris = verdigris::TestScene::new(game.clone());

        std::thread::scope(|s| {
            s.spawn(|| game.enter_tick_loop(&should_stop));

            renderer.enter_gfx_loop(&should_stop);
        });
    };

    if std::panic::catch_unwind(run).is_err()
    {
        should_stop.store(true, SeqCst);
    }

    if Arc::into_inner(renderer).is_none()
    {
        log::warn!("Renderer was retained via Arc cycle! Drop is not possible");
    }

    logger.stop_worker();
}
