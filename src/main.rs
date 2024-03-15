#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::*;
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
    // TODO: into scope
    {
        // this scope is what lets you spawn the handles and uses an unwinding
        // panic to transmit all of the info back
    }

    let renderer: Arc<gfx::Renderer> = {
        let maybe_renderer: Option<Arc<gfx::Renderer>> = crash_handler
            .create_handle("Renderer Creation".to_string())
            .enter_oneshot(|| Arc::new(unsafe { gfx::Renderer::new() }));

        crash_handler.handle_crash();

        maybe_renderer.unwrap()
    };

    let game: Arc<game::Game> = {
        let maybe_game: Option<Arc<game::Game>> = crash_handler
            .create_handle("Game Creation".to_string())
            .enter_oneshot(func);

        crash_handler.handle_crash();

        maybe_game.unwrap()
    };

    crash_handler
        .create_handle("Game Tick Thread".to_string())
        .enter_managed_thread(move || game.tick());

    // renderer.enter_gfx_loop(crash_handler.create_handle("Managed Render
    // Loop".to_string()));

    // TODO: make not a closure and properly deal with panics
    // let run = || {
    //     let renderer = Arc::new(unsafe { gfx::Renderer::new() });

    //     let game = game::Game::new(renderer.clone());

    //     {
    //         let _verdigris = verdigris::TestScene::new(game.clone());

    //         std::thread::scope(|s| {
    //             s.spawn(|| game.enter_tick_loop(&should_stop));

    //             renderer.enter_gfx_loop(&should_stop);
    //         });
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

    // util::access_global_thread_pool()
    //     .write()
    //     .unwrap()
    //     .take()
    //     .unwrap()
    //     .join_threads();

    // logger.stop_worker();
}

fn spawn_game(
    crash_handler: Arc<util::CrashHandler>,
    renderer: Arc<gfx::Renderer>
) -> Arc<game::Game>
{
    let game_lock: Mutex<Option<Arc<game::Game>>> = Mutex::new(None);

    crash_handler
        .create_handle("Game Creation".to_string())
        .enter_oneshot(|| {
            *game_lock.lock().unwrap() = Some(game::Game::new(renderer));
        });

    game_lock.into_inner().unwrap().unwrap()
}
