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

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));

    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new());

    // Safety: we try our best to drop the Renderer on this thread

    let crash_handler = util::CrashHandler::new();

    let renderer: Arc<gfx::Renderer> = spawn_renderer(crash_handler.clone());
    let game: Arc<game::Game> = spawn_game(crash_handler.clone());

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

fn spawn_renderer(crash_handler: Arc<util::CrashHandler>) -> Arc<gfx::Renderer>
{
    let renderer_lock: Mutex<Option<gfx::Renderer>> = Mutex::new(None);

    crash_handler
        .create_handle("Renderer Creation".to_string())
        .enter_managed_loop(|| {
            *renderer_lock.lock().unwrap() = Some(unsafe { gfx::Renderer::new() });
            util::TerminationResult::Terminate
        });

    Arc::new(renderer_lock.into_inner().unwrap().unwrap())
}

fn spawn_game(
    crash_handler: Arc<util::CrashHandler>,
    renderer: Arc<gfx::Renderer>
) -> Arc<game::Game>
{
    let game_lock: Mutex<Option<Arc<game::Game>>> = Mutex::new(None);

    crash_handler
        .create_handle("Game Creation".to_string())
        .enter_managed_loop(move || {
            *game_lock.lock().unwrap() = Some(game::Game::new(renderer));
            util::TerminationResult::Terminate
        });

    game_lock.into_inner().unwrap().unwrap()
}
