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
        let renderer = handle.enter_oneshot("Renderer Creation".to_string(), |_| {
            Arc::new(unsafe { gfx::Renderer::new() })
        });

        let game = handle.enter_oneshot("Game Creation".to_string(), || game::Game::new(renderer));

        std::thread::scope(|s| {
            s.spawn(|| game.enter_tick_loop(&should_stop));

            renderer.enter_gfx_loop(&should_stop);
        });
        //     {
        //         let _verdigris = verdigris::TestScene::new(game.clone());

        //     }

        //     if Arc::into_inner(game).is_none()
        //     {
        //         log::warn!("Game was retained via Arc cycle! Drop is not
        // possible");     }

        //     if Arc::into_inner(renderer).is_none()
        //     {
        //         log::warn!("Renderer was retained via Arc cycle! Drop is not
        // possible");     }
        // };

        // if std::panic::catch_unwind(run).is_err()
        // {
        //     should_stop.store(true, SeqCst);
        // }
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
