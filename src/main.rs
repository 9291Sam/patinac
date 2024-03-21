use std::sync::{Arc, Condvar, Mutex};

fn main()
{
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let logger: &'static mut util::AsyncLogger = Box::leak(Box::new(util::AsyncLogger::new()));
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Trace);

    *util::access_global_thread_pool().write().unwrap() = Some(util::ThreadPool::new(
        std::thread::available_parallelism()
            .unwrap()
            .get()
            .saturating_sub(2)
            .max(2)
    ));

    let crash_handler = util::CrashHandler::new();

    crash_handler.into_guarded_scope(|handle| {
        let renderer = handle.enter_constrained("Renderer Creation".to_string(), |_, _, _| {
            Arc::new(unsafe {
                gfx::Renderer::new(format!("Patinac {}", env!("CARGO_PKG_VERSION")))
            })
        });

        let game = handle.enter_constrained("Game Creation".to_string(), |_, _, _| {
            game::Game::new(renderer.clone())
        });

        {
            let _verdigris = handle
                .enter_constrained("Verdigris Creation".to_string(), |_, _, _| {
                    verdigris::TestScene::new(game.clone())
                });

            let _debug_menu = handle.enter_constrained("Game Creation".to_string(), |_, _, _| {
                gui::DebugMenu::new(&renderer)
            });

            let local_game = game.clone();
            handle.enter_constrained_thread(
                "Game Tick Thread".to_string(),
                move |continue_func, _, _| local_game.enter_tick_loop(continue_func)
            );

            let input_manager: Arc<(Mutex<Option<Arc<gfx::InputManager>>>, Condvar)> =
                Arc::new((Mutex::new(None), Condvar::new()));

            let local_game = game.clone();
            let local_input_manager_arc = input_manager.clone();
            handle.enter_constrained_thread(
                "Game Camera Thread".to_string(),
                move |continue_func, _, _| {
                    local_game.enter_camera_loop(&local_input_manager_arc, continue_func)
                }
            );

            // TODO: replace poll func with loop checking one for checking for long running
            // loops
            let local_renderer = renderer.clone();
            handle.enter_constrained(
                "Gfx Loop".to_string(),
                move |continue_func, terminate_func, crash_poll_func| {
                    let local_input_manager = input_manager.clone();

                    local_renderer.enter_gfx_loop(
                        continue_func,
                        terminate_func,
                        crash_poll_func,
                        &local_input_manager
                    )
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
