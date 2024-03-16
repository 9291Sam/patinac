#![feature(stmt_expr_attributes)]
#![feature(if_let_guard)]
#![feature(associated_type_defaults)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(effects)]

use std::sync::Arc;

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new());

    let crash_handler = util::CrashHandler::new();

    crash_handler.into_guarded_scope(|handle| {
        let renderer = handle.enter_constrained("Renderer Creation".to_string(), |_, _| {
            Arc::new(unsafe { gfx::Renderer::new() })
        });

        let game = handle.enter_constrained("Game Creation".to_string(), |_, _| {
            game::Game::new(renderer.clone())
        });

        {
            let _verdigris = handle.enter_constrained("Verdigris Creation".to_string(), |_, _| {
                verdigris::TestScene::new(game.clone())
            });

            let local_game = game.clone();
            handle.enter_constrained_thread(
                "Game Tick Thread".to_string(),
                move |continue_func, _| local_game.clone().enter_tick_loop(continue_func)
            );

            let local_renderer = renderer.clone();
            handle.enter_constrained(
                "Gfx Loop".to_string(),
                move |continue_func, terminate_func| {
                    local_renderer.enter_gfx_loop(continue_func, terminate_func)
                }
            );
        }

        util::access_global_thread_pool()
            .write()
            .unwrap()
            .take()
            .unwrap()
            .join_threads();

        if Arc::into_inner(game).is_none()
        {
            log::warn!("Game was retained!")
        }

        if Arc::into_inner(renderer).is_none()
        {
            log::warn!("Renderer was retained!")
        }
    });

    crash_handler.finish();

    logger.stop_worker();
}
